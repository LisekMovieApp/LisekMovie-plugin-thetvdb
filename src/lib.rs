//! TheTVDB (v4 API) metadata provider extension for LisekMovie.
//!
//! Implements `lisek_plugin_api::MetadataProvider` (see that crate's own
//! doc comment for why this is a normal Cargo dependency compiled into
//! `lisek-server`, not a `.dll`/`.so` loaded at runtime) on top of a plain
//! HTTP client for TVDB's v4 API: authenticates with an API key to get a
//! short-lived bearer token, then searches for a TV series by name.
//!
//! Unlike TMDB, TVDB's v4 API is a two-step flow: `POST /login` exchanges
//! the admin-configured API key for a JWT, which every subsequent request
//! carries as a `Bearer` token. [`TvdbClient`] caches that token in memory
//! (behind a [`tokio::sync::Mutex`], since `search_series` can be called
//! concurrently) and logs in lazily on first use, rather than eagerly at
//! construction — a client built for a provider the admin has since
//! disabled should never make a network call at all. On a `401` from a
//! search (an expired or revoked token — TVDB's own docs describe tokens
//! as valid for roughly a month, not tracked precisely here since v4 never
//! actually returns an expiry timestamp to parse), the cached token is
//! dropped and one fresh login + retry is attempted before giving up.
//!
//! `base_url` is injectable so this crate's tests run against a local
//! `wiremock` server rather than the real `api4.thetvdb.com` — the
//! request/response shapes here follow TVDB's published v4 API
//! documentation; a first real scan with a configured key is what gives
//! that its final live confirmation.
//!
//! Films (Faza 18) use the same flow with `type=movie` searches,
//! `/movies/{id}` records and `{ movie }` remote ids.

use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use lisek_plugin_api::{
    DirectId, MediaKind, MetadataProvider, MovieQuery, ProviderCandidate, ProviderError, ProviderMatch,
    SeriesQuery,
};
use regex::Regex;
use serde::Deserialize;
use thiserror::Error;
use tokio::sync::Mutex;

const DEFAULT_BASE_URL: &str = "https://api4.thetvdb.com/v4";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Error)]
pub enum TvdbError {
    #[error("request to TVDB failed: {0}")]
    Request(String),
    #[error("TVDB rejected the API key (unauthorized)")]
    Unauthorized,
    #[error("TVDB returned an unexpected response: {0}")]
    UnexpectedResponse(String),
}

impl From<reqwest::Error> for TvdbError {
    fn from(err: reqwest::Error) -> Self {
        TvdbError::Request(err.to_string())
    }
}

impl From<TvdbError> for ProviderError {
    fn from(err: TvdbError) -> Self {
        match err {
            TvdbError::Request(msg) => ProviderError::Request(msg),
            TvdbError::Unauthorized => ProviderError::Unauthorized,
            TvdbError::UnexpectedResponse(msg) => ProviderError::UnexpectedResponse(msg),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TvdbSeriesMatch {
    pub tvdb_id: String,
    pub title: String,
    pub overview: String,
    /// TVDB's search endpoint returns one ready-to-use thumbnail URL
    /// directly (unlike TMDB, which returns a bare path resolved against a
    /// separate image-size CDN) — there's no equivalent "poster vs backdrop"
    /// split to preserve here, just whatever TVDB considers this result's
    /// primary image.
    pub image_url: Option<String>,
    pub year: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct LoginResponse {
    data: LoginData,
}

#[derive(Debug, Deserialize)]
struct LoginData {
    token: String,
}

#[derive(Debug, Deserialize)]
struct TvdbSearchResponse {
    data: Vec<RawSearchResult>,
}

#[derive(Debug, Deserialize)]
struct RawSearchResult {
    tvdb_id: Option<String>,
    name: Option<String>,
    overview: Option<String>,
    image_url: Option<String>,
    year: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DataResponse<T> {
    data: T,
}

#[derive(Debug, Deserialize)]
struct RawSeriesRecord {
    id: u64,
    #[serde(default)]
    name: String,
    image: Option<String>,
    year: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawTranslation {
    overview: Option<String>,
}

/// `type=` of `/search` for an item kind.
fn search_type(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Series => "series",
        MediaKind::Movie => "movie",
    }
}

/// The record collection of an item kind: `/series/{id}`, `/movies/{id}`.
fn record_path(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Series => "series",
        MediaKind::Movie => "movies",
    }
}

/// Fields of a match: `tvdb_id`, the overview, and (lookups) the title and
/// year.
fn match_from(m: &TvdbSeriesMatch, with_title: bool) -> ProviderMatch {
    let mut fields = serde_json::json!({
        "tvdb_id": m.tvdb_id,
        "overview": m.overview,
    });
    if with_title {
        fields["title"] = serde_json::json!(m.title);
        fields["year"] = serde_json::json!(m.year);
    }
    ProviderMatch {
        poster_url: m.image_url.clone(),
        // TVDB v4 has wide-banner and clear-logo artwork, but its artwork
        // type ids come from `GET /artwork/types` rather than fixed
        // constants — see `lisek_plugin_api::ProviderMatch`. Left unset.
        backdrop_url: None,
        logo_url: None,
        banner_url: None,
        // TheTVDB credits are not read (yet).
        people: None,
        seasons: Vec::new(),
        fields,
    }
}

/// TVDB names languages with three letters.
fn tvdb_language(primary_language: &str) -> &'static str {
    match primary_language.split(['-', '_']).next().unwrap_or("") {
        "pl" => "pol",
        "de" => "deu",
        "fr" => "fra",
        "es" => "spa",
        "it" => "ita",
        "cs" => "ces",
        "ja" => "jpn",
        _ => "eng",
    }
}

static FOUR_DIGIT_YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d{4}$").unwrap());

impl TryFrom<RawSearchResult> for TvdbSeriesMatch {
    type Error = TvdbError;

    fn try_from(raw: RawSearchResult) -> Result<Self, Self::Error> {
        let tvdb_id = raw.tvdb_id.ok_or_else(|| {
            TvdbError::UnexpectedResponse("search result missing tvdb_id".to_string())
        })?;
        let title = raw.name.ok_or_else(|| {
            TvdbError::UnexpectedResponse("search result missing name".to_string())
        })?;
        let year = raw
            .year
            .filter(|y| FOUR_DIGIT_YEAR.is_match(y))
            .and_then(|y| y.parse::<i32>().ok());

        Ok(TvdbSeriesMatch {
            tvdb_id,
            title,
            overview: raw.overview.unwrap_or_default(),
            image_url: raw.image_url,
            year,
        })
    }
}

pub struct TvdbClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
    token: Mutex<Option<String>>,
}

impl TvdbClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Same as [`TvdbClient::new`] but pointed at a caller-chosen base URL —
    /// how this crate's own tests substitute a local mock server; real
    /// callers should always use [`TvdbClient::new`].
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            http: reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build().expect(
                "building the TVDB HTTP client failed — this only happens if the platform's TLS \
                 backend is broken, which is a startup-time configuration problem, not a normal \
                 runtime error worth threading a Result through every call site for",
            ),
            token: Mutex::new(None),
        }
    }

    async fn login(&self) -> Result<String, TvdbError> {
        let response = self
            .http
            .post(format!("{}/login", self.base_url))
            .json(&serde_json::json!({ "apikey": self.api_key }))
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TvdbError::Unauthorized);
        }
        let response = response.error_for_status()?;

        let body: LoginResponse = response
            .json()
            .await
            .map_err(|err| TvdbError::UnexpectedResponse(err.to_string()))?;
        Ok(body.data.token)
    }

    async fn ensure_token(&self) -> Result<String, TvdbError> {
        let mut cached = self.token.lock().await;
        if let Some(token) = cached.as_ref() {
            return Ok(token.clone());
        }
        let fresh = self.login().await?;
        *cached = Some(fresh.clone());
        Ok(fresh)
    }

    async fn clear_token(&self) {
        *self.token.lock().await = None;
    }

    /// Searches TVDB for a TV series by title, across an increasingly loose
    /// cascade of query candidates (see
    /// [`lisek_plugin_api::query_candidates`]) — the plain normalized title
    /// first, then a diacritics-stripped variant, then just its first two
    /// words — stopping at the first candidate that finds anything. Logs in
    /// for a bearer token first if this client hasn't already cached one
    /// (shared across every candidate attempt — see [`Self::ensure_token`]).
    pub async fn search_series(&self, title: &str) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
        for candidate in lisek_plugin_api::query_candidates(title) {
            if let Some(found) = self.search_series_once(&candidate).await? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// One query-candidate's worth of searching, including the
    /// token-refresh-on-401 retry — factored out of [`Self::search_series`]
    /// so the query-candidate cascade there can call this once per
    /// candidate.
    async fn search_series_once(&self, title: &str) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
        let token = self.ensure_token().await?;
        let mut response = self.search_request(&token, title).await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            // The cached token was likely expired or revoked server-side —
            // drop it and retry once with a freshly logged-in one before
            // surfacing an error.
            self.clear_token().await;
            let fresh_token = self.ensure_token().await?;
            response = self.search_request(&fresh_token, title).await?;
        }

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TvdbError::Unauthorized);
        }
        let response = response.error_for_status()?;

        let body: TvdbSearchResponse = response
            .json()
            .await
            .map_err(|err| TvdbError::UnexpectedResponse(err.to_string()))?;

        body.data
            .into_iter()
            .next()
            .map(TryInto::try_into)
            .transpose()
    }

    /// One authorized GET (with the same retry-once-after-401 as
    /// searching). `Ok(None)` for a 404.
    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<Option<T>, TvdbError> {
        let send = |token: String| {
            self.http
                .get(format!("{}{path}", self.base_url))
                .bearer_auth(token)
                .query(query)
                .send()
        };
        let mut response = send(self.ensure_token().await?).await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            self.clear_token().await;
            response = send(self.ensure_token().await?).await?;
        }
        match response.status() {
            reqwest::StatusCode::UNAUTHORIZED => return Err(TvdbError::Unauthorized),
            reqwest::StatusCode::NOT_FOUND => return Ok(None),
            _ => {}
        }
        let body = response
            .error_for_status()?
            .json()
            .await
            .map_err(|err| TvdbError::UnexpectedResponse(err.to_string()))?;
        Ok(Some(body))
    }

    /// Every series result for one exact query ("Fix match"), best first.
    pub async fn search_series_results(
        &self,
        title: &str,
        year: Option<i32>,
    ) -> Result<Vec<TvdbSeriesMatch>, TvdbError> {
        self.search_results(MediaKind::Series, title, year).await
    }

    /// Every result of one kind (`type=series` or `type=movie`) for one
    /// exact query, best first.
    pub async fn search_results(
        &self,
        kind: MediaKind,
        title: &str,
        year: Option<i32>,
    ) -> Result<Vec<TvdbSeriesMatch>, TvdbError> {
        let year = year.map(|y| y.to_string());
        let mut query = vec![("query", title), ("type", search_type(kind))];
        if let Some(year) = year.as_deref() {
            query.push(("year", year));
        }
        let body: Option<TvdbSearchResponse> = self.get_json("/search", &query).await?;
        body.map(|b| b.data)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|raw| TvdbSeriesMatch::try_from(raw).ok())
            .map(Ok)
            .collect()
    }

    /// One series by its TVDB id, with the overview in `language` (TVDB's
    /// three-letter code, e.g. `pol`) when TVDB has that translation, else
    /// English. `Ok(None)` for an unknown id.
    pub async fn series_by_id(
        &self,
        tvdb_id: &str,
        language: &str,
    ) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
        self.record_by_id(MediaKind::Series, tvdb_id, language).await
    }

    /// One series (`/series/{id}`) or film (`/movies/{id}`) by its TVDB id,
    /// with the overview in `language` when TVDB has that translation, else
    /// English. `Ok(None)` for an unknown id.
    pub async fn record_by_id(
        &self,
        kind: MediaKind,
        tvdb_id: &str,
        language: &str,
    ) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
        let collection = record_path(kind);
        let Some(body): Option<DataResponse<RawSeriesRecord>> =
            self.get_json(&format!("/{collection}/{tvdb_id}"), &[]).await?
        else {
            return Ok(None);
        };
        let record = body.data;
        let mut overview = String::new();
        for lang in [language, "eng"] {
            let translation: Option<DataResponse<RawTranslation>> = self
                .get_json(&format!("/{collection}/{tvdb_id}/translations/{lang}"), &[])
                .await
                .unwrap_or(None);
            if let Some(text) = translation.and_then(|t| t.data.overview).filter(|o| !o.is_empty()) {
                overview = text;
                break;
            }
        }
        Ok(Some(TvdbSeriesMatch {
            tvdb_id: record.id.to_string(),
            title: record.name,
            overview,
            image_url: record.image,
            year: record
                .year
                .filter(|y| FOUR_DIGIT_YEAR.is_match(y))
                .and_then(|y| y.parse().ok()),
        }))
    }

    /// The TVDB id of the series with this IMDb id
    /// (`/search/remoteid/{id}`). `Ok(None)`: TVDB has no series for it.
    pub async fn series_id_by_imdb_id(&self, imdb_id: &str) -> Result<Option<String>, TvdbError> {
        self.id_by_imdb_id(MediaKind::Series, imdb_id).await
    }

    /// The TVDB id of the series or film with this IMDb id
    /// (`/search/remoteid/{id}` answers `{ series }` or `{ movie }`
    /// records). `Ok(None)`: TVDB has none of that kind for it.
    pub async fn id_by_imdb_id(&self, kind: MediaKind, imdb_id: &str) -> Result<Option<String>, TvdbError> {
        #[derive(Deserialize)]
        struct Record {
            id: u64,
        }
        #[derive(Deserialize)]
        struct RemoteResult {
            series: Option<Record>,
            movie: Option<Record>,
        }
        let body: Option<DataResponse<Vec<RemoteResult>>> =
            self.get_json(&format!("/search/remoteid/{imdb_id}"), &[]).await?;
        Ok(body
            .and_then(|b| {
                b.data.into_iter().find_map(|r| match kind {
                    MediaKind::Series => r.series,
                    MediaKind::Movie => r.movie,
                })
            })
            .map(|record| record.id.to_string()))
    }

    /// The best film for a scanned title: the query-candidate cascade of
    /// [`Self::search_series`], with the year first and then without it.
    pub async fn search_movie(&self, title: &str, year: Option<i32>) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
        for candidate in lisek_plugin_api::query_candidates(title) {
            let years = if year.is_some() { vec![year, None] } else { vec![None] };
            for year in years {
                if let Some(found) = self
                    .search_results(MediaKind::Movie, &candidate, year)
                    .await?
                    .into_iter()
                    .next()
                {
                    return Ok(Some(found));
                }
            }
        }
        Ok(None)
    }

    async fn search_request(
        &self,
        token: &str,
        title: &str,
    ) -> Result<reqwest::Response, TvdbError> {
        Ok(self
            .http
            .get(format!("{}/search", self.base_url))
            .bearer_auth(token)
            .query(&[("query", title), ("type", "series")])
            .send()
            .await?)
    }
}

/// The [`lisek_plugin_api::MetadataProvider`] implementation `lisek-
/// server`'s provider registry constructs one of per scan, from the API
/// key `provider_config` has stored for `"tvdb"` — see that crate's own
/// doc comment for the compile-time-extension architecture this fits into.
pub struct TvdbProvider {
    api_key: String,
    base_url: String,
}

impl TvdbProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Tests point the provider at a mock server.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

#[async_trait]
impl MetadataProvider for TvdbProvider {
    fn provider_key(&self) -> &'static str {
        "tvdb"
    }

    /// Films through `/search?type=movie` (Faza 18; earlier versions
    /// answered series only).
    async fn find_movie(
        &self,
        query: &MovieQuery<'_>,
        _primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let client = self.client();
        Ok(client
            .search_movie(query.title, query.year)
            .await?
            .map(|m| match_from(&m, false)))
    }

    /// TVDB has no per-language `/images` endpoint equivalent to TMDB's, so
    /// `primary_language` goes unused here — the one thumbnail URL the
    /// search response carries is used as the poster; no backdrop is
    /// available from a TVDB-only match.
    async fn find_series(
        &self,
        query: &SeriesQuery<'_>,
        _primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let client = self.client();
        Ok(client
            .search_series(query.title)
            .await?
            .map(|m| match_from(&m, false)))
    }

    async fn search(
        &self,
        kind: MediaKind,
        title: &str,
        year: Option<i32>,
        _primary_language: &str,
    ) -> Result<Vec<ProviderCandidate>, ProviderError> {
        Ok(self
            .client()
            .search_results(kind, title, year)
            .await?
            .into_iter()
            .take(10)
            .map(|m| ProviderCandidate {
                provider_id: m.tvdb_id,
                title: m.title,
                year: m.year,
                overview: m.overview,
                poster_url: m.image_url,
            })
            .collect())
    }

    /// A series id at `/series/{id}`, a film id at `/movies/{id}`.
    async fn lookup(
        &self,
        kind: MediaKind,
        provider_id: &str,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let provider_id = provider_id.trim();
        if provider_id.is_empty() || !provider_id.chars().all(|c| c.is_ascii_digit()) {
            return Ok(None);
        }
        Ok(self
            .client()
            .record_by_id(kind, provider_id, tvdb_language(primary_language))
            .await?
            .map(|m| match_from(&m, true)))
    }

    /// TVDB ids as they are, IMDb ids through TVDB's remote ids — series
    /// or films, as asked.
    async fn lookup_by_id(
        &self,
        kind: MediaKind,
        id: &DirectId,
        primary_language: &str,
    ) -> Result<Option<ProviderCandidate>, ProviderError> {
        let tvdb_id = match id {
            DirectId::Numeric(own) => own.clone(),
            DirectId::Imdb(imdb) => match self.client().id_by_imdb_id(kind, imdb).await? {
                Some(found) => found,
                None => return Ok(None),
            },
        };
        Ok(self
            .lookup(kind, &tvdb_id, primary_language)
            .await?
            .and_then(|matched| ProviderCandidate::from_match(&tvdb_id, &matched)))
    }
}

impl TvdbProvider {
    fn client(&self) -> TvdbClient {
        TvdbClient::with_base_url(self.api_key.clone(), self.base_url.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn login_mock_success() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "success",
            "data": { "token": "test-jwt-token" }
        }))
    }

    #[tokio::test]
    async fn search_series_logs_in_then_searches_with_the_bearer_token() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            // Normalized (lowercased) by `query_candidates` before this
            // client ever sees it — see `lisek-plugin-api`'s own tests.
            .and(query_param("query", "breaking bad"))
            .and(query_param("type", "series"))
            .and(header("Authorization", "Bearer test-jwt-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "status": "success",
                "data": [{
                    "tvdb_id": "81189",
                    "name": "Breaking Bad",
                    "overview": "A high school chemistry teacher...",
                    "image_url": "https://artworks.thetvdb.com/banners/posters/81189-1.jpg",
                    "year": "2008"
                }]
            })))
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("test-key", server.uri());
        let result = client.search_series("Breaking Bad").await.unwrap();

        assert_eq!(
            result,
            Some(TvdbSeriesMatch {
                tvdb_id: "81189".to_string(),
                title: "Breaking Bad".to_string(),
                overview: "A high school chemistry teacher...".to_string(),
                image_url: Some(
                    "https://artworks.thetvdb.com/banners/posters/81189-1.jpg".to_string()
                ),
                year: Some(2008),
            })
        );
    }

    #[tokio::test]
    async fn search_series_reuses_a_cached_token_without_logging_in_again() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .expect(1) // exactly once — the second search must not log in again
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })),
            )
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("test-key", server.uri());
        client.search_series("Show One").await.unwrap();
        client.search_series("Show Two").await.unwrap();
        // wiremock's `.expect(1)` above is verified when `server` drops at
        // the end of the test; reaching this point without a panic from
        // that verification is the assertion.
    }

    #[tokio::test]
    async fn search_series_retries_once_after_a_401_with_a_fresh_login() {
        let server = MockServer::start().await;
        // First login hands out a token that the search endpoint will then
        // reject once (simulating an expired/stale cached token), forcing
        // exactly one re-login before the search succeeds.
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(header("Authorization", "Bearer test-jwt-token"))
            .respond_with(ResponseTemplate::new(401))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(header("Authorization", "Bearer test-jwt-token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })),
            )
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("test-key", server.uri());
        let result = client.search_series("Some Show").await;
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn search_series_falls_back_to_the_first_two_words_once_the_full_title_finds_nothing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("query", "a very obscure show title"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })),
            )
            .mount(&server)
            .await;
        // "a very" — the first two words — is what actually finds it.
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("query", "a very"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "tvdb_id": "12345",
                    "name": "A Very Obscure Show",
                    "overview": "It exists.",
                    "image_url": "https://artworks.thetvdb.com/banners/posters/12345-1.jpg",
                    "year": "2019"
                }]
            })))
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("test-key", server.uri());
        let result = client
            .search_series("A Very Obscure Show Title")
            .await
            .unwrap();

        assert_eq!(result.map(|m| m.tvdb_id), Some("12345".to_string()));
    }

    #[tokio::test]
    async fn login_failure_is_reported_as_unauthorized() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("bad-key", server.uri());
        let result = client.search_series("Anything").await;
        assert!(matches!(result, Err(TvdbError::Unauthorized)));
    }

    #[tokio::test]
    async fn search_series_returns_none_when_tvdb_finds_nothing() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })),
            )
            .mount(&server)
            .await;

        let client = TvdbClient::with_base_url("test-key", server.uri());
        let result = client.search_series("Nonexistent Show Xyz").await.unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn provider_key_is_tvdb() {
        assert_eq!(TvdbProvider::new("k").provider_key(), "tvdb");
    }

    #[tokio::test]
    async fn fix_match_search_and_lookup_by_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("query", "Inazuma Eleven"))
            .and(query_param("year", "2008"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    { "tvdb_id": "81797", "name": "Inazuma Eleven", "year": "2008", "image_url": "https://img/ie.jpg" },
                    { "name": "no id, skipped" }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81797"))
            .and(header("Authorization", "Bearer test-jwt-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": { "id": 81797, "name": "Inazuma Eleven", "image": "https://img/ie.jpg", "year": "2008" }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81797/translations/pol"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81797/translations/eng"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": { "overview": "Soccer!" }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/1"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let provider = TvdbProvider::with_base_url("k", server.uri());
        let found = provider.search(MediaKind::Series, "Inazuma Eleven", Some(2008), "pl").await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].provider_id.as_str(), found[0].year), ("81797", Some(2008)));
        assert!(provider.search(MediaKind::Movie, "x", None, "pl").await.unwrap().is_empty());

        let matched = provider.lookup(MediaKind::Series, "81797", "pl").await.unwrap().unwrap();
        assert_eq!(matched.fields["title"], "Inazuma Eleven");
        assert_eq!(matched.fields["overview"], "Soccer!");
        assert_eq!(matched.fields["year"], 2008);
        assert_eq!(matched.poster_url.as_deref(), Some("https://img/ie.jpg"));
        assert!(provider.lookup(MediaKind::Series, "1", "pl").await.unwrap().is_none());
        assert!(provider.lookup(MediaKind::Series, "../x", "pl").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn an_imdb_id_is_resolved_through_the_remote_ids() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/remoteid/tt0903747"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "series": { "id": 81189, "name": "Breaking Bad" } }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/remoteid/tt0000001"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81189"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": { "id": 81189, "name": "Breaking Bad", "year": "2008" }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81189/translations/pol"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/series/81189/translations/eng"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let provider = TvdbProvider::with_base_url("k", server.uri());
        let found = provider
            .lookup_by_id(MediaKind::Series, &DirectId::Imdb("tt0903747".into()), "pl")
            .await
            .unwrap()
            .unwrap();
        assert_eq!((found.provider_id.as_str(), found.title.as_str(), found.year), ("81189", "Breaking Bad", Some(2008)));
        let unknown = DirectId::Imdb("tt0000001".into());
        assert!(provider.lookup_by_id(MediaKind::Series, &unknown, "pl").await.unwrap().is_none());
        // Films are not looked up on TVDB.
        let movie = DirectId::Imdb("tt0903747".into());
        assert!(provider.lookup_by_id(MediaKind::Movie, &movie, "pl").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn films_use_the_movie_endpoints() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/login"))
            .respond_with(login_mock_success())
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("type", "movie"))
            .and(query_param("query", "dune"))
            .and(query_param("year", "2021"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [] })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("type", "movie"))
            .and(query_param("query", "dune"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "tvdb_id": "375507", "name": "Dune", "year": "2021", "image_url": "https://img/dune.jpg", "overview": "Arrakis." }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/movies/375507"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": { "id": 375507, "name": "Dune", "image": "https://img/dune.jpg", "year": "2021" }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/movies/375507/translations/pol"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": { "overview": "Diuna." }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/remoteid/tt1160419"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "movie": { "id": 375507 } }]
            })))
            .mount(&server)
            .await;
        let provider = TvdbProvider::with_base_url("k", server.uri());

        // The scanner: the year finds nothing, the search without it does.
        let matched = provider
            .find_movie(&MovieQuery { title: "Dune", year: Some(2021) }, "pl")
            .await
            .unwrap()
            .unwrap();
        assert_eq!((matched.fields["tvdb_id"].as_str(), matched.fields["overview"].as_str()), (Some("375507"), Some("Arrakis.")));

        let found = provider.search(MediaKind::Movie, "dune", None, "pl").await.unwrap();
        assert_eq!((found[0].provider_id.as_str(), found[0].year), ("375507", Some(2021)));

        // Fix match with the TVDB film id (and with its IMDb id).
        let by_id = provider
            .lookup_by_id(MediaKind::Movie, &DirectId::Numeric("375507".into()), "pl")
            .await
            .unwrap()
            .unwrap();
        assert_eq!((by_id.title.as_str(), by_id.year, by_id.overview.as_str()), ("Dune", Some(2021), "Diuna."));
        let by_imdb = provider
            .lookup_by_id(MediaKind::Movie, &DirectId::Imdb("tt1160419".into()), "pl")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_imdb.provider_id, "375507");
        // The same number is no series.
        assert!(provider.lookup(MediaKind::Series, "375507", "pl").await.unwrap().is_none());
    }

    #[test]
    fn languages_map_to_tvdb_codes() {
        assert_eq!(tvdb_language("pl"), "pol");
        assert_eq!(tvdb_language("pt-BR"), "eng");
    }
}

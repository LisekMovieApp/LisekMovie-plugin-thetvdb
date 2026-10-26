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
//! TVDB has no movie search wired up in this pass (its v4 movie endpoints
//! exist but aren't covered here — TMDB alone covers movies for now); see
//! [`TvdbProvider::find_movie`].

use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use lisek_plugin_api::{
    MediaKind, MetadataProvider, MovieQuery, ProviderCandidate, ProviderError, ProviderMatch,
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
        let year = year.map(|y| y.to_string());
        let mut query = vec![("query", title), ("type", "series")];
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
        let Some(body): Option<DataResponse<RawSeriesRecord>> =
            self.get_json(&format!("/series/{tvdb_id}"), &[]).await?
        else {
            return Ok(None);
        };
        let record = body.data;
        let mut overview = String::new();
        for lang in [language, "eng"] {
            let translation: Option<DataResponse<RawTranslation>> = self
                .get_json(&format!("/series/{tvdb_id}/translations/{lang}"), &[])
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

    /// TVDB's v4 movie search exists but isn't wired up in this pass — see
    /// this crate's own doc comment. `Ok(None)` here is indistinguishable
    /// to the caller from "TVDB covers movies but didn't recognize this
    /// title", which is exactly the point (see `MetadataProvider`'s doc
    /// comment in `lisek-plugin-api`).
    async fn find_movie(
        &self,
        _query: &MovieQuery<'_>,
        _primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        Ok(None)
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
        let client = TvdbClient::with_base_url(self.api_key.clone(), self.base_url.clone());
        let Some(m) = client.search_series(query.title).await? else {
            return Ok(None);
        };
        Ok(Some(ProviderMatch {
            poster_url: m.image_url.clone(),
            backdrop_url: None,
            // TVDB v4 does have wide-banner and clear-logo artwork, but
            // its artwork *type ids* are returned dynamically by
            // `GET /artwork/types` rather than being fixed, documented
            // constants the way TMDB's genre ids are — see
            // `lisek_plugin_api::ProviderMatch`'s own doc comment. Left
            // unset here rather than guessed at.
            logo_url: None,
            banner_url: None,
            // TheTVDB credits are not read (yet).
            people: None,
            fields: serde_json::json!({
                "tvdb_id": m.tvdb_id,
                "overview": m.overview,
            }),
        }))
    }

    async fn search(
        &self,
        kind: MediaKind,
        title: &str,
        year: Option<i32>,
        _primary_language: &str,
    ) -> Result<Vec<ProviderCandidate>, ProviderError> {
        if kind != MediaKind::Series {
            return Ok(Vec::new());
        }
        let client = TvdbClient::with_base_url(self.api_key.clone(), self.base_url.clone());
        Ok(client
            .search_series_results(title, year)
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

    async fn lookup(
        &self,
        kind: MediaKind,
        provider_id: &str,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        if kind != MediaKind::Series || !provider_id.chars().all(|c| c.is_ascii_digit()) || provider_id.is_empty() {
            return Ok(None);
        }
        let client = TvdbClient::with_base_url(self.api_key.clone(), self.base_url.clone());
        let Some(m) = client.series_by_id(provider_id, tvdb_language(primary_language)).await? else {
            return Ok(None);
        };
        Ok(Some(ProviderMatch {
            poster_url: m.image_url.clone(),
            backdrop_url: None,
            logo_url: None,
            banner_url: None,
            // TheTVDB credits are not read (yet).
            people: None,
            fields: serde_json::json!({
                "tvdb_id": m.tvdb_id,
                "overview": m.overview,
                "title": m.title,
                "year": m.year,
            }),
        }))
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

    #[tokio::test]
    async fn provider_movie_lookup_always_returns_none() {
        let provider = TvdbProvider::new("test-key");
        let result = provider
            .find_movie(
                &MovieQuery {
                    title: "Anything",
                    year: None,
                },
                "en",
            )
            .await
            .unwrap();
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

    #[test]
    fn languages_map_to_tvdb_codes() {
        assert_eq!(tvdb_language("pl"), "pol");
        assert_eq!(tvdb_language("pt-BR"), "eng");
    }
}

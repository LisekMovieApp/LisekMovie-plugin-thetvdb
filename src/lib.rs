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
use lisek_plugin_api::{MetadataProvider, MovieQuery, ProviderError, ProviderMatch, SeriesQuery};
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

    /// Searches TVDB for a TV series by title, logging in for a bearer
    /// token first if this client hasn't already cached one. Returns the
    /// best (first, TVDB's own relevance-ranked) match, or `None` if TVDB
    /// found nothing.
    pub async fn search_series(&self, title: &str) -> Result<Option<TvdbSeriesMatch>, TvdbError> {
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
}

impl TvdbProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
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
        let client = TvdbClient::new(self.api_key.clone());
        let Some(m) = client.search_series(query.title).await? else {
            return Ok(None);
        };
        Ok(Some(ProviderMatch {
            poster_url: m.image_url.clone(),
            backdrop_url: None,
            fields: serde_json::json!({
                "tvdb_id": m.tvdb_id,
                "overview": m.overview,
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
            .and(query_param("query", "Breaking Bad"))
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
}

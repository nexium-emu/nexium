use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const CLIENT_ID: &str = "nxc_E-rgPv2x_uY9";
pub const SCOPES: &str = "identity friends presence game.matchmaking";
const TIMEOUT: Duration = Duration::from_secs(15);
const BODY_LIMIT: u64 = 8 * 1024 * 1024;
const USER_AGENT: &str = concat!("NeXium/", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiError {
    Offline,
    Unauthorized,
    Forbidden(String),
    InvalidGrant,
    Rejected { status: u16, message: String },
    Malformed,
}

impl ApiError {
    pub fn describe(&self) -> String {
        match self {
            ApiError::Offline => "Nextendo can't be reached right now. Check your connection.".into(),
            ApiError::Unauthorized => "Your Nextendo session has expired. Sign in again.".into(),
            ApiError::Forbidden(message) if !message.is_empty() => message.clone(),
            ApiError::Forbidden(_) => "NeXium isn't allowed to do that for this account.".into(),
            ApiError::InvalidGrant => "Your Nextendo sign-in is no longer valid. Sign in again.".into(),
            ApiError::Rejected { message, .. } if !message.is_empty() => message.clone(),
            ApiError::Rejected { status, .. } => format!("Nextendo answered with an error (HTTP {status})."),
            ApiError::Malformed => "Nextendo sent a reply NeXium doesn't understand.".into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Identity {
    pub pid: u64,
    pub username: String,
    pub friend_code: String,
}

#[derive(Clone, Debug, Default)]
pub struct Tokens {
    pub access: String,
    pub refresh: String,
    pub expires_in: u64,
    pub scope: Option<String>,
    pub identity: Option<Identity>,
}

#[derive(Clone, Debug)]
pub struct GameToken {
    pub token: String,
    pub expires_in: u64,
    pub identity: Option<Identity>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RemoteFriend {
    pub pid: u64,
    pub name: String,
    pub friend_code: String,
    pub status: i32,
    pub app_field: Vec<u8>,
    pub app_id: u64,
    pub app_name: String,
    pub account_id: u64,
    pub console: bool,
    pub image: Vec<u8>,
    pub last_seen: Option<String>,
    pub staff: bool,
    pub favorite: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LobbyPlayer {
    pub pid: u64,
    pub name: String,
    pub known: bool,
    pub friend_code: String,
    pub host: bool,
    pub me: bool,
    pub title_id: u64,
    pub seen_at: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lobby {
    pub title_id: u64,
    pub state: String,
    pub count: u32,
    pub max: u32,
    pub players: Vec<LobbyPlayer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub title_id: u64,
    pub name: String,
    pub seconds: u64,
    pub last_played: String,
    pub icon: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct FriendList {
    pub friends: Vec<RemoteFriend>,
    pub requests: Vec<RemoteFriend>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Standing {
    pub allow: bool,
    pub reason: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PresenceUpdate {
    pub status: i32,
    pub app_field: Vec<u8>,
    pub app_id: Option<u64>,
    pub app_name: String,
}

pub struct Api {
    agent: ureq::Agent,
    base: String,
}

impl Api {
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .build()
            .into();
        Self {
            agent,
            base: base_url(),
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    pub fn exchange_code(&self, code: &str, verifier: &str, redirect_uri: &str) -> Result<Tokens, ApiError> {
        let response = self
            .agent
            .post(&self.url("/oauth/token"))
            .header("X-Nextendo-Client-Id", CLIENT_ID)
            .send_form([
                ("grant_type", "authorization_code"),
                ("client_id", CLIENT_ID),
                ("code", code),
                ("redirect_uri", redirect_uri),
                ("code_verifier", verifier),
            ]);
        parse_tokens(&token_reply(response)?)
    }

    pub fn refresh(&self, refresh_token: &str) -> Result<Tokens, ApiError> {
        let response = self
            .agent
            .post(&self.url("/oauth/token"))
            .header("X-Nextendo-Client-Id", CLIENT_ID)
            .send_form([
                ("grant_type", "refresh_token"),
                ("client_id", CLIENT_ID),
                ("refresh_token", refresh_token),
            ]);
        parse_tokens(&token_reply(response)?)
    }

    fn get(&self, path: &str, access: Option<&str>) -> Result<Vec<u8>, ApiError> {
        let mut request = self
            .agent
            .get(&self.url(path))
            .header("Accept", "application/json")
            .header("X-Nextendo-Client-Id", CLIENT_ID);
        if let Some(access) = access {
            request = request.header("Authorization", &format!("Bearer {access}"));
        }
        reply(request.call())
    }

    fn post(&self, path: &str, access: &str, body: Value) -> Result<Vec<u8>, ApiError> {
        let response = self
            .agent
            .post(&self.url(path))
            .header("Accept", "application/json")
            .header("X-Nextendo-Client-Id", CLIENT_ID)
            .header("Authorization", &format!("Bearer {access}"))
            .send_json(body);
        reply(response)
    }

    pub fn identity(&self, access: &str) -> Result<Identity, ApiError> {
        let value = parse_json(&self.get("/api/oauth/userinfo", Some(access))?)?;
        identity_from(&value).ok_or(ApiError::Malformed)
    }

    pub fn game_token(&self, access: &str) -> Result<GameToken, ApiError> {
        let value = parse_json(&self.get("/api/nex-token", Some(access))?)?;
        let token = text(&value, &["nex_token"]).filter(|token| !token.is_empty());
        let token = token.ok_or(ApiError::Malformed)?;
        Ok(GameToken {
            token,
            expires_in: number(&value, &["expires_in"]).unwrap_or(24 * 60 * 60),
            identity: identity_from(&value),
        })
    }

    pub fn friends(&self, access: &str) -> Result<FriendList, ApiError> {
        let value = parse_json(&self.get("/api/friends", Some(access))?)?;
        Ok(friend_list_from(&value))
    }

    pub fn add_friend(&self, access: &str, friend_code: &str) -> Result<(), ApiError> {
        self.post("/api/friends", access, json!({ "friend_code": friend_code }))
            .map(|_| ())
    }

    pub fn answer_request(&self, access: &str, pid: u64, accept: bool) -> Result<(), ApiError> {
        let path = if accept {
            "/api/friends/accept"
        } else {
            "/api/friends/decline"
        };
        self.post(path, access, json!({ "pid": pid })).map(|_| ())
    }

    pub fn remove_friend(&self, access: &str, pid: u64) -> Result<(), ApiError> {
        self.post("/api/friends/remove", access, json!({ "pid": pid }))
            .map(|_| ())
    }

    pub fn publish_presence(&self, access: &str, update: &PresenceUpdate) -> Result<(), ApiError> {
        let body = json!({
            "status": update.status,
            "app_field": STANDARD.encode(&update.app_field),
            "app_id": update.app_id.map(|id| format!("{id:016X}")).unwrap_or_default(),
            "app_name": update.app_name,
            "app_detail": "",
        });
        self.post("/api/presence", access, body).map(|_| ())
    }

    pub fn lobby(&self, access: &str) -> Result<Option<Lobby>, ApiError> {
        let value = parse_json(&self.get("/api/my-lobby", Some(access))?)?;
        Ok(lobby_from(&value))
    }

    pub fn recent_players(&self, access: &str) -> Result<Vec<LobbyPlayer>, ApiError> {
        let value = parse_json(&self.get("/api/recent-players", Some(access))?)?;
        Ok(players_from(&value))
    }

    pub fn sync_history(&self, access: &str, entries: &[HistoryEntry]) -> Result<(), ApiError> {
        let history: Vec<Value> = entries
            .iter()
            .map(|entry| {
                let mut item = json!({
                    "title_id": format!("{:016X}", entry.title_id),
                    "seconds": entry.seconds,
                    "last_played": entry.last_played,
                });
                if !entry.name.is_empty() {
                    item["name"] = json!(entry.name);
                }
                if !entry.icon.is_empty() {
                    item["icon"] = json!(STANDARD.encode(&entry.icon));
                }
                item
            })
            .collect();
        let response = self
            .agent
            .put(&self.url("/api/history"))
            .header("Accept", "application/json")
            .header("X-Nextendo-Client-Id", CLIENT_ID)
            .header("Authorization", &format!("Bearer {access}"))
            .send_json(json!({ "history": history }));
        reply(response).map(|_| ())
    }

    pub fn standing(&self, access: &str) -> Result<Standing, ApiError> {
        let value = parse_json(&self.get("/api/online-status", Some(access))?)?;
        Ok(Standing {
            allow: value.get("allow").and_then(Value::as_bool).unwrap_or(false),
            reason: text(&value, &["reason"]).unwrap_or_default(),
            message: text(&value, &["message"]).unwrap_or_default(),
        })
    }

    pub fn online_counts(&self) -> Result<HashMap<u64, u32>, ApiError> {
        let value = parse_json(&self.get("/api/online-counts", None)?)?;
        Ok(counts_from(&value))
    }

    pub fn avatar(&self, pid: u64) -> Result<Vec<u8>, ApiError> {
        let response = self
            .agent
            .get(&self.url(&format!("/api/avatar?pid={pid}")))
            .header("X-Nextendo-Client-Id", CLIENT_ID)
            .call();
        reply(response)
    }

    pub fn ping(&self) -> Result<Duration, ApiError> {
        let started = Instant::now();
        self.get("/api/health", None)?;
        Ok(started.elapsed())
    }
}

fn read_body(response: &mut ureq::http::Response<ureq::Body>) -> Result<Vec<u8>, ApiError> {
    response
        .body_mut()
        .with_config()
        .limit(BODY_LIMIT)
        .read_to_vec()
        .map_err(|_| ApiError::Offline)
}

fn reply(response: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<Vec<u8>, ApiError> {
    let mut response = response.map_err(|error| {
        log::debug!("nextendo: request failed: {error}");
        ApiError::Offline
    })?;
    let status = response.status().as_u16();
    let body = read_body(&mut response)?;
    match status {
        200..=299 => Ok(body),
        401 => Err(ApiError::Unauthorized),
        403 => Err(ApiError::Forbidden(error_message(&body))),
        status => Err(ApiError::Rejected {
            status,
            message: error_message(&body),
        }),
    }
}

fn token_reply(response: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<Value, ApiError> {
    match reply(response) {
        Ok(body) => parse_json(&body),
        Err(ApiError::Rejected { status: 400, message }) | Err(ApiError::Forbidden(message))
            if message == "invalid_grant" =>
        {
            Err(ApiError::InvalidGrant)
        }
        Err(ApiError::Unauthorized) => Err(ApiError::InvalidGrant),
        Err(error) => Err(error),
    }
}

fn parse_json(body: &[u8]) -> Result<Value, ApiError> {
    serde_json::from_slice(body).map_err(|_| ApiError::Malformed)
}

fn error_message(body: &[u8]) -> String {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return String::new();
    };
    text(&value, &["error_description", "message", "error"]).unwrap_or_default()
}

fn text(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match value.get(*key)? {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    })
}

fn number(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| match value.get(*key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

pub fn identity_from(value: &Value) -> Option<Identity> {
    let candidates = [
        Some(value),
        value.get("account"),
        value.get("user"),
        value.get("profile"),
    ];
    candidates.into_iter().flatten().find_map(|source| {
        let pid = number(source, &["pid", "principal_id"])?;
        if pid == 0 {
            return None;
        }
        Some(Identity {
            pid,
            username: text(source, &["username", "name", "preferred_username", "nickname"])
                .unwrap_or_default(),
            friend_code: text(source, &["friend_code", "friendCode"]).unwrap_or_default(),
        })
    })
}

pub fn parse_tokens(value: &Value) -> Result<Tokens, ApiError> {
    let access = text(value, &["access_token"]).filter(|token| !token.is_empty());
    let refresh = text(value, &["refresh_token"]).unwrap_or_default();
    let access = access.ok_or(ApiError::Malformed)?;
    Ok(Tokens {
        access,
        refresh,
        expires_in: number(value, &["expires_in"]).unwrap_or(3600),
        scope: text(value, &["scope"]),
        identity: value.get("account").and_then(identity_from),
    })
}

fn decode_b64(text: &str) -> Vec<u8> {
    let trimmed = text.trim();
    let trimmed = trimmed
        .split_once(";base64,")
        .map_or(trimmed, |(_, data)| data);
    STANDARD.decode(trimmed).unwrap_or_default()
}

fn friend_from(value: &Value) -> Option<RemoteFriend> {
    let pid = number(value, &["pid"])?;
    let name = text(value, &["name"])
        .filter(|name| !name.trim().is_empty())
        .or_else(|| text(value, &["username"]))
        .unwrap_or_default();
    let account_id = text(value, &["nsa", "account_hex"])
        .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok())
        .unwrap_or(0);
    let console = text(value, &["plateforme"]).as_deref() == Some("switch")
        || number(value, &["pf"]) == Some(1);
    let staff = value
        .get("badges")
        .and_then(Value::as_array)
        .is_some_and(|badges| {
            badges
                .iter()
                .filter_map(Value::as_str)
                .any(|badge| matches!(badge, "team" | "admin" | "staff"))
        });
    let mut friend = RemoteFriend {
        pid,
        name,
        friend_code: text(value, &["friend_code"]).unwrap_or_default(),
        account_id,
        console,
        image: text(value, &["image"]).map(|data| decode_b64(&data)).unwrap_or_default(),
        last_seen: text(value, &["vu", "last_seen"]).filter(|seen| !seen.is_empty()),
        staff,
        favorite: value.get("favorite").and_then(Value::as_bool).unwrap_or(false),
        ..RemoteFriend::default()
    };
    if let Some(presence) = value.get("presence").filter(|presence| presence.is_object()) {
        friend.status = number(presence, &["status"]).unwrap_or(0) as i32;
        friend.app_field = text(presence, &["app_field"])
            .map(|data| decode_b64(&data))
            .unwrap_or_default();
        friend.app_id = text(presence, &["app_id"])
            .and_then(|hex| nexium_common::nextendo::parse_title_id(&hex))
            .unwrap_or(0);
        friend.app_name = text(presence, &["app_name"]).unwrap_or_default();
    }
    Some(friend)
}

pub fn friend_list_from(value: &Value) -> FriendList {
    let list = |key: &str| -> Vec<RemoteFriend> {
        value
            .get(key)
            .and_then(Value::as_array)
            .map(|entries| entries.iter().filter_map(friend_from).collect())
            .unwrap_or_default()
    };
    FriendList {
        friends: list("friends"),
        requests: list("requests"),
    }
}

fn player_from(value: &Value) -> Option<LobbyPlayer> {
    let pid = number(value, &["pid"])?;
    Some(LobbyPlayer {
        pid,
        name: text(value, &["name", "username"]).unwrap_or_default(),
        known: value.get("known").and_then(Value::as_bool).unwrap_or(false),
        friend_code: text(value, &["friend_code"]).unwrap_or_default(),
        host: value.get("host").and_then(Value::as_bool).unwrap_or(false),
        me: value.get("is_me").and_then(Value::as_bool).unwrap_or(false),
        title_id: text(value, &["title_id"])
            .and_then(|hex| nexium_common::nextendo::parse_title_id(&hex))
            .unwrap_or(0),
        seen_at: text(value, &["seen_at"]).filter(|seen| !seen.is_empty()),
    })
}

pub fn players_from(value: &Value) -> Vec<LobbyPlayer> {
    value
        .get("players")
        .and_then(Value::as_array)
        .map(|players| players.iter().filter_map(player_from).collect())
        .unwrap_or_default()
}

pub fn lobby_from(value: &Value) -> Option<Lobby> {
    if !value.get("in_lobby").and_then(Value::as_bool).unwrap_or(false) {
        return None;
    }
    let details = value.get("lobby").cloned().unwrap_or(Value::Null);
    Some(Lobby {
        title_id: text(value, &["title_id"])
            .and_then(|hex| nexium_common::nextendo::parse_title_id(&hex))
            .unwrap_or(0),
        state: text(&details, &["state_code"]).unwrap_or_default(),
        count: number(&details, &["count"]).unwrap_or(0) as u32,
        max: number(&details, &["max"]).unwrap_or(0) as u32,
        players: players_from(value),
    })
}

pub fn counts_from(value: &Value) -> HashMap<u64, u32> {
    value
        .get("counts")
        .and_then(Value::as_object)
        .map(|counts| {
            counts
                .iter()
                .filter_map(|(title, count)| {
                    Some((
                        nexium_common::nextendo::parse_title_id(title)?,
                        count.as_u64()? as u32,
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn base_url() -> String {
    let canonical = nexium_common::nextendo::API_BASE.to_string();
    let Ok(raw) = std::env::var("NEXTENDO_API") else {
        return canonical;
    };
    match sanitize_base(&raw) {
        Some(base) => base,
        None => {
            log::warn!("nextendo: ignoring NEXTENDO_API, only https on nextendo.network or loopback may receive your account token");
            canonical
        }
    }
}

pub fn sanitize_base(raw: &str) -> Option<String> {
    let raw = raw.trim().trim_matches(|ch| matches!(ch, '"' | '\'')).trim_end_matches('/');
    let (scheme, authority) = raw.split_once("://")?;
    if authority.is_empty() || authority.contains('/') || authority.contains('@') {
        return None;
    }
    let host = match authority.strip_prefix('[') {
        Some(bracketed) => bracketed.split(']').next().unwrap_or_default(),
        None => authority.split(':').next().unwrap_or_default(),
    }
    .to_ascii_lowercase();
    if matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1") {
        return Some(raw.to_string());
    }
    if scheme != "https" {
        return None;
    }
    if host == "nextendo.network" || host.ends_with(".nextendo.network") {
        return Some(raw.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_parse_with_optional_account() {
        let value = json!({
            "access_token": "nxa_abc",
            "refresh_token": "nxr_def",
            "expires_in": 3600,
            "scope": "identity friends",
            "account": { "pid": 1800003542u64, "username": "Mythrax", "friend_code": "SW-1111-2222-3333" }
        });
        let tokens = parse_tokens(&value).unwrap();
        assert_eq!(tokens.access, "nxa_abc");
        assert_eq!(tokens.refresh, "nxr_def");
        assert_eq!(tokens.identity.unwrap().username, "Mythrax");
        assert!(matches!(
            parse_tokens(&json!({ "error": "invalid_grant" })),
            Err(ApiError::Malformed)
        ));
    }

    #[test]
    fn identity_is_found_at_the_root_or_nested() {
        let root = json!({ "pid": "1800000001", "username": "NextendoPlayer", "friend_code": "SW-0000-0000-0001" });
        assert_eq!(identity_from(&root).unwrap().pid, 1_800_000_001);
        let nested = json!({ "sub": "x", "account": { "pid": 7, "name": "Seven" } });
        assert_eq!(identity_from(&nested).unwrap().username, "Seven");
        assert!(identity_from(&json!({ "pid": 0 })).is_none());
    }

    #[test]
    fn friends_and_requests_are_parsed() {
        let value = json!({
            "friends": [{
                "pid": 11,
                "name": "",
                "username": "Peach",
                "friend_code": "SW-1234-5678-9012",
                "plateforme": "switch",
                "nsa": "00000000000000ab",
                "presence": { "status": 2, "app_field": "TW9kZQBjUHVibGljAA==", "app_id": "0100152000022000", "app_name": "Mario Kart 8 Deluxe" }
            }],
            "requests": [{ "pid": 12, "name": "Daisy" }]
        });
        let list = friend_list_from(&value);
        assert_eq!(list.friends.len(), 1);
        let friend = &list.friends[0];
        assert_eq!(friend.name, "Peach");
        assert_eq!(friend.status, 2);
        assert_eq!(friend.app_field, b"Mode\0cPublic\0");
        assert_eq!(friend.app_id, 0x0100152000022000);
        assert_eq!(friend.account_id, 0xAB);
        assert!(friend.console);
        assert_eq!(list.requests[0].name, "Daisy");
    }

    #[test]
    fn friend_extras_are_read() {
        let value = json!({
            "friends": [{ "pid": 3, "name": "Toad", "badges": ["admin", "team"], "favorite": true, "vu": "2026-10-06T03:27:35Z" }],
            "requests": []
        });
        let friend = &friend_list_from(&value).friends[0];
        assert!(friend.staff);
        assert!(friend.favorite);
        assert_eq!(friend.last_seen.as_deref(), Some("2026-10-06T03:27:35Z"));
    }

    #[test]
    fn lobbies_and_recent_players_parse() {
        assert!(lobby_from(&json!({ "in_lobby": false })).is_none());
        let lobby = lobby_from(&json!({
            "in_lobby": true,
            "title_id": "0100152000022000",
            "lobby": { "type": "public", "state": "x", "state_code": "matched", "id": 9, "count": 3, "max": 12 },
            "players": [
                { "pid": 1, "name": "Me", "known": true, "host": true, "is_me": true },
                { "pid": 2, "name": "Rival", "known": true, "friend_code": "SW-1111-2222-3333" }
            ]
        }))
        .unwrap();
        assert_eq!(lobby.title_id, 0x0100152000022000);
        assert_eq!((lobby.count, lobby.max), (3, 12));
        assert_eq!(lobby.state, "matched");
        assert!(lobby.players[0].me && lobby.players[0].host);
        assert_eq!(lobby.players[1].friend_code, "SW-1111-2222-3333");
        let recent = players_from(&json!({ "players": [{ "pid": 5, "name": "Ghost", "known": false, "title_id": "01006a800016e000", "seen_at": "2026-10-06T10:00:00Z" }] }));
        assert_eq!(recent[0].title_id, 0x01006A800016E000);
        assert!(!recent[0].known);
    }

    #[test]
    fn counts_use_lowercase_title_keys() {
        let value = json!({ "counts": { "0100152000022000": 16, "01006a800016e000": 71, "junk": 3 } });
        let counts = counts_from(&value);
        assert_eq!(counts.get(&0x0100152000022000), Some(&16));
        assert_eq!(counts.get(&0x01006A800016E000), Some(&71));
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn api_overrides_cannot_leak_the_token() {
        assert_eq!(sanitize_base("https://nextendo.network/"), Some("https://nextendo.network".into()));
        assert_eq!(sanitize_base("https://beta.nextendo.network"), Some("https://beta.nextendo.network".into()));
        assert_eq!(sanitize_base("http://127.0.0.1:8080"), Some("http://127.0.0.1:8080".into()));
        assert_eq!(sanitize_base("http://nextendo.network"), None);
        assert_eq!(sanitize_base("https://evilnextendo.network"), None);
        assert_eq!(sanitize_base("https://nextendo.network.evil.com"), None);
        assert_eq!(sanitize_base("https://user@nextendo.network"), None);
        assert_eq!(sanitize_base("https://nextendo.network/path"), None);
    }

    #[test]
    #[ignore = "needs network access"]
    fn live_public_endpoints_answer() {
        let api = Api::new();
        assert!(api.ping().is_ok());
        assert!(!api.online_counts().unwrap().is_empty());
        assert!(!api.avatar(1_800_003_803).unwrap().is_empty());
        assert_eq!(api.refresh("not-a-real-token").unwrap_err(), ApiError::InvalidGrant);
        assert_eq!(api.friends("not-a-real-token").unwrap_err(), ApiError::Unauthorized);
    }

    #[test]
    #[ignore = "needs a signed-in test profile"]
    fn live_session_endpoint_access() {
        let api = Api::new();
        let mut saved = crate::nextendo::vault::load().expect("signed-in test profile");
        let tokens = api.refresh(&saved.refresh_token).expect("refresh");
        if !tokens.refresh.is_empty() {
            saved.refresh_token = tokens.refresh.clone();
            crate::nextendo::vault::save(&saved).expect("save rotated token");
        }
        let game = api.game_token(&tokens.access).map(|game| game.token);
        for path in [
            "/api/oauth/userinfo",
            "/api/friends",
            "/api/online-status",
            "/api/history",
            "/api/my-lobby",
            "/api/recent-players",
            "/api/profile",
            "/api/me",
            "/api/game-invitations",
        ] {
            let with_access = api.get(path, Some(&tokens.access)).map(|body| body.len());
            let with_game = game
                .as_ref()
                .map(|token| api.get(path, Some(token)).map(|body| body.len()));
            eprintln!("{path:24} access={with_access:?} game={with_game:?}");
        }
    }

    #[test]
    fn data_uri_images_decode() {
        assert_eq!(decode_b64("data:image/jpeg;base64,AAEC"), vec![0, 1, 2]);
        assert_eq!(decode_b64("AAEC"), vec![0, 1, 2]);
    }
}

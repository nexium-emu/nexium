use super::signed::{self, text_field, SigningKey, Verified};
use chrono::Utc;
use nexium_common::nextendo::{self, TitleSupport};
use serde_json::Value;
use std::collections::HashSet;
use std::path::PathBuf;

pub const BASE: &str = "https://nextendo.network/service-catalog/v1";
const DOCUMENT_TYPE: &str = "game_catalog";
const KNOWN_CAPABILITIES: [&str; 2] = ["versions.exact.v1", "versions.any.v1"];
const RETAIN_EXPIRED_DAYS: i64 = 7;

pub struct Fetched {
    pub status: u16,
    pub etag: Option<String>,
    pub body: Vec<u8>,
}

pub fn fetch(agent: &ureq::Agent, url: &str, etag: Option<&str>) -> Result<Fetched, String> {
    let mut request = agent
        .get(url)
        .header("Accept", "application/json")
        .header("Cache-Control", "no-cache");
    if let Some(etag) = etag.filter(|etag| !etag.is_empty()) {
        request = request.header("If-None-Match", etag);
    }
    let mut response = request.call().map_err(|error| format!("request failed: {error}"))?;
    let status = response.status().as_u16();
    let etag = response
        .headers()
        .get("etag")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = response
        .body_mut()
        .with_config()
        .limit(signed::MAX_DOCUMENT as u64)
        .read_to_vec()
        .map_err(|error| format!("body unreadable: {error}"))?;
    Ok(Fetched { status, etag, body })
}

fn cache_path() -> PathBuf {
    nexium_common::paths::root()
        .join("nextendo")
        .join("service-catalog")
        .join("game-catalog.json")
}

pub fn capabilities(value: &Value, field: &str) -> Result<Vec<String>, String> {
    let array = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing {field}"))?;
    if array.len() > 64 {
        return Err("too many capabilities".into());
    }
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let text = item.as_str().ok_or("invalid capability")?;
        let valid = !text.is_empty()
            && text.len() <= 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-'));
        if !valid || !seen.insert(text.to_string()) {
            return Err("invalid capability".into());
        }
        out.push(text.to_string());
    }
    Ok(out)
}

struct Listing {
    enabled: bool,
    version: String,
    accepted: Vec<String>,
    any_version: bool,
}

fn parse_listing(game: &Value) -> Result<Listing, String> {
    let enabled = game.get("enabled").and_then(Value::as_bool).ok_or("missing enabled")?;
    let caps = capabilities(game, "required_capabilities")?;
    if game.get("protocol").is_some() {
        text_field(game, "protocol", 32, false)?;
    }
    if game.get("name").is_some() {
        text_field(game, "name", 256, false)?;
    }
    let mut policy = String::new();
    let mut version = String::new();
    let mut accepted = Vec::new();
    if enabled || game.get("versions").is_some() {
        let versions = game.get("versions").ok_or("missing versions")?;
        policy = text_field(versions, "policy", 64, true)?;
        if matches!(policy.as_str(), "exact" | "any") && !caps.iter().any(|cap| *cap == format!("versions.{policy}.v1")) {
            return Err("missing capability".into());
        }
        if policy == "exact" {
            let array = versions
                .get("accepted")
                .and_then(Value::as_array)
                .filter(|array| (1..=64).contains(&array.len()))
                .ok_or("invalid accepted")?;
            let mut seen = HashSet::new();
            for item in array {
                let text = item.as_str().ok_or("invalid accepted")?;
                if text.trim().is_empty() || text.len() > 32 || text.chars().any(char::is_control) || !seen.insert(text) {
                    return Err("invalid accepted".into());
                }
                accepted.push(text.to_string());
            }
            version = if versions.get("recommended").is_some() {
                text_field(versions, "recommended", 32, true)?
            } else {
                accepted[0].clone()
            };
            if !accepted.contains(&version) {
                return Err("recommended version is not accepted".into());
            }
        } else if policy == "any" && (versions.get("accepted").is_some() || versions.get("recommended").is_some()) {
            return Err("any-version policy with a version list".into());
        }
    }
    let evaluable = caps.iter().all(|cap| KNOWN_CAPABILITIES.contains(&cap.as_str()))
        && matches!(policy.as_str(), "exact" | "any");
    Ok(Listing {
        enabled: enabled && evaluable,
        version,
        accepted,
        any_version: policy == "any",
    })
}

pub fn parse_catalog(payload: &Value) -> Result<Vec<TitleSupport>, String> {
    let games = payload
        .get("games")
        .and_then(Value::as_array)
        .ok_or("missing games")?;
    if games.len() > 10_000 {
        return Err("too many games".into());
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for game in games {
        let ids = game
            .get("title_ids")
            .and_then(Value::as_array)
            .filter(|array| (1..=32).contains(&array.len()))
            .ok_or("invalid title IDs")?;
        let mut title_ids = Vec::with_capacity(ids.len());
        for id in ids {
            let text = id.as_str().ok_or("ambiguous title ID")?;
            let title_id = (text.len() == 16 && text.chars().all(|ch| ch.is_ascii_hexdigit()))
                .then(|| nextendo::parse_title_id(text))
                .flatten()
                .filter(|title_id| *title_id != 0)
                .ok_or("ambiguous title ID")?;
            if !seen.insert(title_id) {
                return Err("ambiguous title ID".into());
            }
            title_ids.push(title_id);
        }
        let name = game
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let listing = parse_listing(game);
        if let Err(error) = &listing {
            log::warn!("nextendo: catalog entry {:016X} is unusable: {error}", title_ids[0]);
        }
        for title_id in title_ids {
            let name = if name.is_empty() {
                format!("Title {}", nextendo::format_title_id(title_id))
            } else {
                name.clone()
            };
            out.push(match &listing {
                Ok(listing) => TitleSupport {
                    title_id,
                    name,
                    version: listing.version.clone(),
                    accepted: listing.accepted.clone(),
                    any_version: listing.any_version,
                    enabled: listing.enabled,
                    from_catalog: true,
                },
                Err(_) => TitleSupport {
                    title_id,
                    name,
                    enabled: false,
                    from_catalog: true,
                    ..TitleSupport::default()
                },
            });
        }
    }
    Ok(out)
}

fn apply(document: &Verified) -> Result<usize, String> {
    let titles = parse_catalog(&document.payload)?;
    let count = titles.len();
    nextendo::set_catalog_titles(titles);
    Ok(count)
}

pub fn load_cached() {
    let path = cache_path();
    let Ok(bytes) = std::fs::read(&path) else {
        return;
    };
    let now = Utc::now();
    match signed::verify(&bytes, SigningKey::Catalog, DOCUMENT_TYPE, now, true) {
        Ok(document) if document.expires_at + chrono::Duration::days(RETAIN_EXPIRED_DAYS) > now => {
            match apply(&document) {
                Ok(count) => log::info!(
                    "nextendo: cached game catalog revision {} lists {count} titles{}",
                    document.revision,
                    if document.is_valid_at(now) { "" } else { " (refresh due)" }
                ),
                Err(error) => log::warn!("nextendo: cached game catalog rejected: {error}"),
            }
        }
        Ok(_) => log::info!("nextendo: cached game catalog is too old to use"),
        Err(error) => log::warn!("nextendo: cached game catalog rejected: {error}"),
    }
}

pub fn refresh(agent: &ureq::Agent) -> Result<usize, String> {
    let fetched = fetch(agent, &format!("{BASE}/game-catalog"), None)?;
    if fetched.status != 200 {
        return Err(format!("HTTP {}", fetched.status));
    }
    let document = signed::verify(&fetched.body, SigningKey::Catalog, DOCUMENT_TYPE, Utc::now(), false)?;
    let count = apply(&document)?;
    let path = cache_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(error) = std::fs::write(&path, &fetched.body) {
        log::warn!("nextendo: game catalog cache not saved: {error}");
    }
    log::info!("nextendo: game catalog revision {} lists {count} titles", document.revision);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn catalog(games: Value) -> Vec<TitleSupport> {
        parse_catalog(&json!({ "games": games })).unwrap()
    }

    #[test]
    fn exact_and_any_policies_parse() {
        let titles = catalog(json!([
            {
                "title_ids": ["010040600c5ce000"],
                "name": "Tetris 99",
                "enabled": true,
                "required_capabilities": ["versions.exact.v1"],
                "protocol": "nex",
                "versions": { "policy": "exact", "accepted": ["2.6.0", "2.5.0"], "recommended": "2.6.0" }
            },
            {
                "title_ids": ["0100000000ABC000", "0100000000ABD000"],
                "enabled": true,
                "required_capabilities": ["versions.any.v1"],
                "versions": { "policy": "any" }
            }
        ]));
        assert_eq!(titles.len(), 3);
        assert_eq!(titles[0].name, "Tetris 99");
        assert_eq!(titles[0].version, "2.6.0");
        assert_eq!(titles[0].accepted, vec!["2.6.0", "2.5.0"]);
        assert!(titles[0].enabled && !titles[0].any_version && titles[0].from_catalog);
        assert!(titles[0].accepts("2.5.0"));
        assert!(!titles[0].accepts("2.4.0"));
        assert_eq!(titles[1].title_id, 0x0100000000ABC000);
        assert_eq!(titles[1].name, "Title 0100000000ABC000");
        assert!(titles[1].any_version && titles[1].enabled);
        assert!(titles[2].accepts("anything"));
    }

    #[test]
    fn unknown_capabilities_and_bad_entries_block_online_without_breaking_the_document() {
        let titles = catalog(json!([
            {
                "title_ids": ["01001b300b9be000"],
                "name": "Diablo III: Eternal Collection",
                "enabled": true,
                "required_capabilities": ["versions.exact.v1", "dlc.required.v1"],
                "versions": { "policy": "exact", "accepted": ["2.7.7.92380"] }
            },
            {
                "title_ids": ["0100000000AAA000"],
                "name": "Broken",
                "enabled": true,
                "required_capabilities": ["versions.exact.v1"],
                "versions": { "policy": "exact", "accepted": [] }
            },
            {
                "title_ids": ["0100000000BBB000"],
                "name": "Off",
                "enabled": false,
                "required_capabilities": []
            }
        ]));
        assert_eq!(titles.len(), 3);
        assert!(!titles[0].enabled);
        assert_eq!(titles[0].version, "2.7.7.92380");
        assert!(!titles[1].enabled);
        assert_eq!(titles[1].name, "Broken");
        assert!(!titles[2].enabled);
        assert!(!titles[2].accepts("1.0.0"));
    }

    #[test]
    fn identity_problems_invalidate_the_whole_catalog() {
        let duplicate = json!({ "games": [
            { "title_ids": ["0100000000AAA000"], "enabled": false, "required_capabilities": [] },
            { "title_ids": ["0100000000aaa000"], "enabled": false, "required_capabilities": [] }
        ] });
        assert!(parse_catalog(&duplicate).is_err());
        let short = json!({ "games": [ { "title_ids": ["0100"], "enabled": false, "required_capabilities": [] } ] });
        assert!(parse_catalog(&short).is_err());
        let zero = json!({ "games": [ { "title_ids": ["0000000000000000"], "enabled": false, "required_capabilities": [] } ] });
        assert!(parse_catalog(&zero).is_err());
        assert!(parse_catalog(&json!({})).is_err());
    }

    #[test]
    #[ignore = "needs network access and NEXIUM_DATA_ROOT"]
    fn live_game_catalog_lists_server_managed_titles() {
        let count = refresh(&super::super::launch::agent()).expect("game catalog");
        assert!(count > 0);
        let tetris = nextendo::supported_title(0x010040600C5CE000).expect("Tetris 99 listing");
        assert!(tetris.from_catalog && tetris.enabled);
        assert_eq!(tetris.version, "2.6.0");
        assert!(cache_path().is_file());
        nextendo::set_catalog_titles(Vec::new());
        load_cached();
        assert!(nextendo::supported_title(0x010040600C5CE000).is_some());
    }

    #[test]
    fn policy_capability_must_match() {
        let titles = catalog(json!([{
            "title_ids": ["0100000000AAA000"],
            "enabled": true,
            "required_capabilities": ["versions.any.v1"],
            "versions": { "policy": "exact", "accepted": ["1.0.0"] }
        }]));
        assert!(!titles[0].enabled);
        let titles = catalog(json!([{
            "title_ids": ["0100000000AAA000"],
            "enabled": true,
            "required_capabilities": ["versions.any.v1"],
            "versions": { "policy": "any", "accepted": ["1.0.0"] }
        }]));
        assert!(!titles[0].enabled);
    }
}

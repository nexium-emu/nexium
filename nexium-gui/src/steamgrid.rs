use std::process::Command;

pub struct IconOption {
    pub full_url: String,
    pub thumb_url: String,
}

fn curl_json(url: &str, key: &str) -> Option<serde_json::Value> {
    let out = Command::new("curl")
        .arg("-sSL")
        .arg("--max-time")
        .arg("15")
        .arg("-H")
        .arg(format!("Authorization: Bearer {}", key))
        .arg(url)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

pub fn curl_bytes(url: &str) -> Option<Vec<u8>> {
    let out = Command::new("curl")
        .arg("-sSL")
        .arg("--max-time")
        .arg("25")
        .arg(url)
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
}

fn search_game_id(key: &str, title: &str) -> Option<u64> {
    let enc = urlencode(title);
    let url = format!(
        "https://www.steamgriddb.com/api/v2/search/autocomplete/{}",
        enc
    );
    let json = curl_json(&url, key)?;
    let arr = json.get("data")?.as_array()?;
    arr.first()?.get("id")?.as_u64()
}

/// Search for a game by title and return up to `limit` icon options.
pub fn fetch_icons(key: &str, title: &str, limit: usize) -> Vec<IconOption> {
    let Some(game_id) = search_game_id(key, title) else {
        return Vec::new();
    };
    let url = format!(
        "https://www.steamgriddb.com/api/v2/icons/game/{}?types=static",
        game_id
    );
    let Some(json) = curl_json(&url, key) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(arr) = json.get("data").and_then(|d| d.as_array()) {
        for item in arr.iter().take(limit) {
            let full = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
            let thumb = item
                .get("thumb")
                .and_then(|u| u.as_str())
                .unwrap_or(full);
            if !full.is_empty() {
                out.push(IconOption {
                    full_url: full.to_string(),
                    thumb_url: thumb.to_string(),
                });
            }
        }
    }
    out
}

/// Verify an API key by hitting a lightweight authenticated endpoint.
pub fn verify_key(key: &str) -> bool {
    if key.trim().is_empty() {
        return false;
    }
    let url = "https://www.steamgriddb.com/api/v2/search/autocomplete/mario";
    match curl_json(url, key) {
        Some(json) => json.get("success").and_then(|s| s.as_bool()).unwrap_or(false),
        None => false,
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

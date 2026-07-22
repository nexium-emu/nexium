use std::process::Command;

const BASE: &str = "https://switch.cdn.fortheusers.org/";

#[derive(Clone)]
pub struct ShopApp {
    pub name: String,
    pub title: String,
    pub author: String,
    pub description: String,
    pub details: String,
    pub version: String,
    pub filesize: u64,
    pub extracted: u64,
    pub screens: u32,
    pub binary: String,
    pub license: String,
    pub updated: String,
}

impl ShopApp {
    pub fn icon_url(&self) -> String {
        format!("{BASE}packages/{}/icon.png", self.name)
    }
    pub fn screen_url(&self, i: u32) -> String {
        format!("{BASE}packages/{}/screen{}.png", self.name, i + 1)
    }
    pub fn zip_url(&self) -> String {
        format!("{BASE}zips/{}.zip", self.name)
    }
    pub fn page_url(&self) -> String {
        format!("https://hb-app.store/switch/{}", self.name)
    }
}

fn curl_bytes(url: &str, timeout: u32) -> Option<Vec<u8>> {
    let out = Command::new("curl")
        .arg("-sSL")
        .arg("--max-time")
        .arg(timeout.to_string())
        .arg(url)
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
}

pub fn icon_bytes(app: &ShopApp) -> Option<Vec<u8>> {
    curl_bytes(&app.icon_url(), 20)
}

pub fn screen_bytes(app: &ShopApp, i: u32) -> Option<Vec<u8>> {
    curl_bytes(&app.screen_url(i), 25)
}

pub fn fetch_games() -> Vec<ShopApp> {
    let Some(bytes) = curl_bytes(&format!("{BASE}repo.json"), 25) else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Vec::new();
    };
    let Some(pkgs) = json.get("packages").and_then(|p| p.as_array()) else {
        return Vec::new();
    };
    let str_of = |v: &serde_json::Value, k: &str| -> String {
        v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
    };
    let u64_of =
        |v: &serde_json::Value, k: &str| -> u64 { v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) };
    let mut out: Vec<ShopApp> = pkgs
        .iter()
        .filter(|p| p.get("category").and_then(|c| c.as_str()) == Some("game"))
        .map(|p| ShopApp {
            name: str_of(p, "name"),
            title: {
                let t = str_of(p, "title");
                if t.is_empty() {
                    str_of(p, "name")
                } else {
                    t
                }
            },
            author: str_of(p, "author"),
            description: str_of(p, "description"),
            details: str_of(p, "details").replace("\\n", "\n"),
            version: str_of(p, "version"),
            filesize: u64_of(p, "filesize"),
            extracted: u64_of(p, "extracted"),
            screens: p.get("screens").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            binary: str_of(p, "binary"),
            license: str_of(p, "license"),
            updated: str_of(p, "updated"),
        })
        .filter(|a| !a.name.is_empty())
        .collect();
    out.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    out
}

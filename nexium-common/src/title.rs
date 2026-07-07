use std::sync::RwLock;

static TITLE_KEY: RwLock<Option<String>> = RwLock::new(None);

pub fn set_title_key(name: &str) {
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.is_empty() {
        "default".to_string()
    } else {
        safe
    };
    if let Ok(mut g) = TITLE_KEY.write() {
        *g = Some(safe);
    }
}

pub fn title_key() -> Option<String> {
    TITLE_KEY.read().ok().and_then(|g| g.clone())
}

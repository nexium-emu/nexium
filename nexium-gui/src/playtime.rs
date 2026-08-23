use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct PlayTimes {
    pub secs: HashMap<PathBuf, f64>,
    dirty: bool,
}

impl PlayTimes {
    fn config_path() -> Option<PathBuf> {
        Some(nexium_common::paths::root().join("playtimes.json"))
    }

    pub fn load() -> Self {
        let secs = Self::config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<HashMap<String, u64>>(&s).ok())
            .map(|m| {
                m.into_iter()
                    .map(|(k, v)| (PathBuf::from(k), v as f64))
                    .collect()
            })
            .unwrap_or_default();
        Self { secs, dirty: false }
    }

    pub fn add(&mut self, path: &Path, dt: f64) {
        *self.secs.entry(path.to_path_buf()).or_insert(0.0) += dt;
        self.dirty = true;
    }

    pub fn get(&self, path: &Path) -> u64 {
        self.secs.get(path).copied().unwrap_or(0.0) as u64
    }

    pub fn save_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let Some(p) = Self::config_path() else {
            return;
        };
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let map: HashMap<String, u64> = self
            .secs
            .iter()
            .map(|(k, v)| (k.to_string_lossy().to_string(), *v as u64))
            .collect();
        if let Ok(s) = serde_json::to_string_pretty(&map) {
            let _ = std::fs::write(p, s);
        }
    }
}

pub fn format_playtime(secs: u64) -> String {
    if secs < 300 {
        return "Played for 5 minutes or less".to_string();
    }
    let mins = secs / 60;
    if mins < 60 {
        return format!("Played for {} minutes", mins);
    }
    let hours = mins / 60;
    let rem = mins % 60;
    if rem == 0 {
        format!("Played for {} hours", hours)
    } else {
        format!("Played for {} h {} min", hours, rem)
    }
}

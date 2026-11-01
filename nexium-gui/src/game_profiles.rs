use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::app_settings::{AppSettings, ResolutionScale};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GameProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution_scale: Option<ResolutionScale>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_resolution_scale: Option<f32>,
}

impl GameProfile {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct GameProfiles {
    #[serde(default)]
    titles: BTreeMap<String, GameProfile>,
}

fn key(title_id: u64) -> String {
    format!("{title_id:016X}")
}

impl GameProfiles {
    fn config_path() -> PathBuf {
        nexium_common::paths::root().join("game_profiles.json")
    }

    pub fn load() -> Self {
        std::fs::read_to_string(Self::config_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, s);
        }
    }

    pub fn get(&self, title_id: u64) -> GameProfile {
        self.titles.get(&key(title_id)).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, title_id: u64, profile: GameProfile) {
        if profile.is_empty() {
            self.titles.remove(&key(title_id));
        } else {
            self.titles.insert(key(title_id), profile);
        }
        self.save();
    }

    pub fn apply(&self, settings: &AppSettings) {
        let titles: HashMap<u64, nexium_common::render_scale::Setting> = self
            .titles
            .iter()
            .filter_map(|(id, profile)| {
                let id = u64::from_str_radix(id, 16).ok()?;
                let scale = profile.resolution_scale?;
                let custom = profile
                    .custom_resolution_scale
                    .unwrap_or(settings.custom_resolution_scale);
                Some((id, scale.setting(custom)))
            })
            .collect();
        nexium_common::render_scale::configure(
            settings
                .resolution_scale
                .setting(settings.custom_resolution_scale),
            titles,
        );
    }
}

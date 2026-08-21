use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn default_true() -> bool {
    true
}

fn default_volume() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Settings {
    #[serde(default)]
    pub rom_folders: Vec<PathBuf>,
    #[serde(default = "default_true")]
    pub touch_overlay: bool,
    #[serde(default)]
    pub docked: bool,
    #[serde(default = "default_volume")]
    pub audio_volume: f32,
    #[serde(default = "default_true")]
    pub multicore: bool,
    #[serde(default = "default_true")]
    pub show_hud: bool,
    #[serde(default = "default_true")]
    pub async_shaders: bool,
    #[serde(default = "default_true")]
    pub async_render: bool,
    #[serde(default = "default_true")]
    pub fastmem: bool,
    #[serde(skip)]
    path: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            rom_folders: Vec::new(),
            touch_overlay: true,
            docked: false,
            audio_volume: 1.0,
            multicore: true,
            show_hud: true,
            async_shaders: true,
            async_render: true,
            fastmem: true,
            path: PathBuf::new(),
        }
    }
}

impl Settings {
    pub fn load(config_dir: &Path) -> Self {
        let path = config_dir.join("android-settings.json");
        let mut settings = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Settings>(&text).ok())
            .unwrap_or_default();
        settings.path = path;
        settings
    }

    pub fn save(&self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(self) {
            Ok(text) => {
                if let Err(error) = std::fs::write(&self.path, text) {
                    log::warn!("settings save failed: {}", error);
                }
            }
            Err(error) => log::warn!("settings encode failed: {}", error),
        }
    }

    pub fn add_folder(&mut self, folder: PathBuf) -> bool {
        if self.rom_folders.iter().any(|existing| existing == &folder) {
            return false;
        }
        self.rom_folders.push(folder);
        self.save();
        true
    }

    pub fn remove_folder(&mut self, index: usize) {
        if index < self.rom_folders.len() {
            self.rom_folders.remove(index);
            self.save();
        }
    }

    pub fn apply_runtime(&self) {
        nexium_common::async_compile::set_enabled(self.async_shaders);
        if self.async_shaders {
            std::env::set_var("NEXIUM_ASYNC_SHADERS", "1");
        } else {
            std::env::remove_var("NEXIUM_ASYNC_SHADERS");
        }
        std::env::set_var(
            "NEXIUM_ASYNC_RENDER",
            if self.async_render { "1" } else { "0" },
        );
        if self.multicore {
            std::env::remove_var("NEXIUM_SINGLECORE");
        } else {
            std::env::set_var("NEXIUM_SINGLECORE", "1");
        }
        if self.fastmem {
            std::env::remove_var("NEXIUM_DYNARMIC_NO_FASTMEM");
            std::env::remove_var("NEXIUM_NO_FASTMEM_ARENA");
        } else {
            std::env::set_var("NEXIUM_DYNARMIC_NO_FASTMEM", "1");
            std::env::set_var("NEXIUM_NO_FASTMEM_ARENA", "1");
        }
        nexium_core::hid_state::set_docked(self.docked);
        nexium_runner::audio::set_master_volume(self.audio_volume);
    }

    pub fn scan_roots(&self, private_roms: &Path) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        roots.push(private_roms.to_path_buf());
        for folder in &self.rom_folders {
            if !roots.iter().any(|existing| existing == folder) {
                roots.push(folder.clone());
            }
        }
        roots
    }
}

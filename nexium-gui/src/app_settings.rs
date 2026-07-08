use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewMode {
    Grid,
    Carousel,
}

impl Default for ViewMode {
    fn default() -> Self {
        ViewMode::Carousel
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarouselTheme {
    Adaptive,
    Aqua,
    Azure,
    Violet,
    Magenta,
    Crimson,
    Amber,
    Emerald,
    Graphite,
}

impl Default for CarouselTheme {
    fn default() -> Self {
        CarouselTheme::Adaptive
    }
}

impl CarouselTheme {
    pub fn all() -> &'static [CarouselTheme] {
        &[
            CarouselTheme::Adaptive,
            CarouselTheme::Aqua,
            CarouselTheme::Azure,
            CarouselTheme::Violet,
            CarouselTheme::Magenta,
            CarouselTheme::Crimson,
            CarouselTheme::Amber,
            CarouselTheme::Emerald,
            CarouselTheme::Graphite,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            CarouselTheme::Adaptive => "Adaptive",
            CarouselTheme::Aqua => "Aqua",
            CarouselTheme::Azure => "Azure",
            CarouselTheme::Violet => "Violet",
            CarouselTheme::Magenta => "Magenta",
            CarouselTheme::Crimson => "Crimson",
            CarouselTheme::Amber => "Amber",
            CarouselTheme::Emerald => "Emerald",
            CarouselTheme::Graphite => "Graphite",
        }
    }

    pub fn color(&self) -> Option<(u8, u8, u8)> {
        Some(match self {
            CarouselTheme::Adaptive => return None,
            CarouselTheme::Aqua => (0x2F, 0xB4, 0xEF),
            CarouselTheme::Azure => (0x3B, 0x82, 0xF6),
            CarouselTheme::Violet => (0x8B, 0x5C, 0xF6),
            CarouselTheme::Magenta => (0xD9, 0x4F, 0xD0),
            CarouselTheme::Crimson => (0xE8, 0x33, 0x50),
            CarouselTheme::Amber => (0xF5, 0xA6, 0x23),
            CarouselTheme::Emerald => (0x22, 0xC5, 0x5E),
            CarouselTheme::Graphite => (0x5C, 0x6B, 0x7C),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuBackend {
    Dynarmic,
    Rustarmic,
}

impl Default for CpuBackend {
    fn default() -> Self {
        CpuBackend::Dynarmic
    }
}

impl CpuBackend {
    pub fn all() -> &'static [CpuBackend] {
        &[CpuBackend::Dynarmic, CpuBackend::Rustarmic]
    }
    pub fn label(&self) -> &'static str {
        match self {
            CpuBackend::Dynarmic => "Dynarmic (C++)",
            CpuBackend::Rustarmic => "Rustarmic (Rust JIT)",
        }
    }
    pub fn to_cpu_kind(&self) -> nexium_cpu::CpuBackendKind {
        match self {
            CpuBackend::Dynarmic => nexium_cpu::CpuBackendKind::Dynarmic,
            CpuBackend::Rustarmic => nexium_cpu::CpuBackendKind::Rustarmic,
        }
    }
    pub fn is_compiled_in(&self) -> bool {
        self.to_cpu_kind().is_compiled_in()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn all() -> &'static [LogLevel] {
        &[
            LogLevel::Error,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Debug,
            LogLevel::Trace,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            LogLevel::Error => "Error",
            LogLevel::Warn => "Warn",
            LogLevel::Info => "Info",
            LogLevel::Debug => "Debug",
            LogLevel::Trace => "Trace",
        }
    }

    pub fn to_filter(&self) -> log::LevelFilter {
        match self {
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Warn => log::LevelFilter::Warn,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectMode {
    Letterbox,
    Stretch,
    Integer,
}

impl Default for AspectMode {
    fn default() -> Self {
        AspectMode::Letterbox
    }
}

impl AspectMode {
    pub fn all() -> &'static [AspectMode] {
        &[
            AspectMode::Letterbox,
            AspectMode::Stretch,
            AspectMode::Integer,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            AspectMode::Letterbox => "Letterbox",
            AspectMode::Stretch => "Stretch",
            AspectMode::Integer => "Integer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterMode {
    Nearest,
    Linear,
}

impl Default for FilterMode {
    fn default() -> Self {
        FilterMode::Nearest
    }
}

impl FilterMode {
    pub fn all() -> &'static [FilterMode] {
        &[FilterMode::Nearest, FilterMode::Linear]
    }
    pub fn label(&self) -> &'static str {
        match self {
            FilterMode::Nearest => "Nearest",
            FilterMode::Linear => "Linear",
        }
    }
}

fn default_output_scale() -> u8 {
    1
}
fn default_vsync() -> bool {
    true
}
fn default_audio_volume() -> f32 {
    1.0
}
fn default_multicore() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppSettings {
    pub log_level: LogLevel,
    #[serde(default = "default_output_scale")]
    pub output_scale: u8,
    #[serde(default)]
    pub aspect: AspectMode,
    #[serde(default)]
    pub filter: FilterMode,
    #[serde(default)]
    pub dpi_aware: bool,
    #[serde(default = "default_vsync")]
    pub vsync: bool,
    #[serde(default)]
    pub cpu_backend: CpuBackend,
    #[serde(default)]
    pub audio_output_device: Option<String>,
    #[serde(default = "default_audio_volume")]
    pub audio_volume: f32,
    #[serde(default = "default_multicore")]
    pub multicore: bool,
    #[serde(default)]
    pub async_shaders: bool,
    #[serde(default)]
    pub library_folders: Vec<PathBuf>,
    #[serde(default)]
    pub view_mode: ViewMode,
    #[serde(default)]
    pub carousel_theme: CarouselTheme,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            output_scale: default_output_scale(),
            aspect: AspectMode::default(),
            filter: FilterMode::default(),
            dpi_aware: false,
            vsync: default_vsync(),
            cpu_backend: CpuBackend::default(),
            audio_output_device: None,
            audio_volume: default_audio_volume(),
            multicore: default_multicore(),
            async_shaders: false,
            library_folders: Vec::new(),
            view_mode: ViewMode::Carousel,
            carousel_theme: CarouselTheme::default(),
        }
    }
}

impl AppSettings {
    pub fn config_path() -> Option<PathBuf> {
        directories::BaseDirs::new().map(|d| d.config_dir().join("NeXium").join("app.json"))
    }

    pub fn load() -> Self {
        let mut cfg = if let Some(path) = Self::config_path() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(cfg) = serde_json::from_str::<AppSettings>(&s) {
                    cfg
                } else {
                    Self::default()
                }
            } else {
                Self::default()
            }
        } else {
            Self::default()
        };
        if let Ok(backend) = std::env::var("NEXIUM_CPU_BACKEND") {
            if backend.eq_ignore_ascii_case("rustarmic") || backend.eq_ignore_ascii_case("rust") {
                cfg.cpu_backend = CpuBackend::Rustarmic;
            } else if backend.eq_ignore_ascii_case("dynarmic")
                || backend.eq_ignore_ascii_case("dyn")
            {
                cfg.cpu_backend = CpuBackend::Dynarmic;
            }
        }
        cfg
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::config_path() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "no config dir",
            ));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let s = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, s)
    }
}

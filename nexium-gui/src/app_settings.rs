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
    Rgb,
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
            CarouselTheme::Rgb,
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
            CarouselTheme::Rgb => "RGB",
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
            CarouselTheme::Rgb => return None,
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
pub enum BackdropTheme {
    Waves,
    Gradient,
    Space,
    None,
}

impl Default for BackdropTheme {
    fn default() -> Self {
        BackdropTheme::Waves
    }
}

impl BackdropTheme {
    pub fn all() -> &'static [BackdropTheme] {
        &[BackdropTheme::Waves, BackdropTheme::Gradient, BackdropTheme::Space, BackdropTheme::None]
    }
    pub fn label(&self) -> &'static str {
        match self {
            BackdropTheme::Waves => "Waves",
            BackdropTheme::Gradient => "Gradient",
            BackdropTheme::Space => "Space",
            BackdropTheme::None => "None",
        }
    }
    pub fn next(&self) -> BackdropTheme {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> BackdropTheme {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
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
pub enum GpuBackend {
    Vulkan,
}

impl Default for GpuBackend {
    fn default() -> Self {
        GpuBackend::Vulkan
    }
}

impl GpuBackend {
    pub fn all() -> &'static [GpuBackend] {
        &[GpuBackend::Vulkan]
    }
    pub fn label(&self) -> &'static str {
        match self {
            GpuBackend::Vulkan => "Vulkan (ash)",
        }
    }
    pub fn next(&self) -> GpuBackend {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> GpuBackend {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionPreset {
    /// 1280 × 720 (native Switch docked)
    P720,
    /// 1920 × 1080
    P1080,
    /// 2560 × 1440
    P1440,
    /// 3840 × 2160
    P2160,
}

impl Default for ResolutionPreset {
    fn default() -> Self {
        ResolutionPreset::P720
    }
}

impl ResolutionPreset {
    pub fn all() -> &'static [ResolutionPreset] {
        &[
            ResolutionPreset::P720,
            ResolutionPreset::P1080,
            ResolutionPreset::P1440,
            ResolutionPreset::P2160,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            ResolutionPreset::P720  => "1280 \u{00D7} 720",
            ResolutionPreset::P1080 => "1920 \u{00D7} 1080",
            ResolutionPreset::P1440 => "2560 \u{00D7} 1440",
            ResolutionPreset::P2160 => "3840 \u{00D7} 2160",
        }
    }
    pub fn scale(&self) -> u32 {
        match self {
            ResolutionPreset::P720  => 1,
            ResolutionPreset::P1080 => 2,
            ResolutionPreset::P1440 => 2,
            ResolutionPreset::P2160 => 3,
        }
    }
    pub fn next(&self) -> ResolutionPreset {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> ResolutionPreset {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
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

fn default_sfx_volume() -> f32 {
    0.5
}

fn default_left_deadzone() -> f32 {
    0.12
}

fn default_right_deadzone() -> f32 {
    0.12
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
    pub gpu_backend: GpuBackend,
    #[serde(default)]
    pub resolution_preset: ResolutionPreset,
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
    #[serde(default)]
    pub backdrop_theme: BackdropTheme,
    #[serde(default)]
    pub light_mode: bool,
    #[serde(default = "default_music_volume")]
    pub music_volume: f32,
    #[serde(default = "default_sfx_volume")]
    pub sfx_volume: f32,
    #[serde(default)]
    pub favorites: Vec<PathBuf>,
    #[serde(default)]
    pub profile_avatar: Option<PathBuf>,
    #[serde(default = "default_profile_name")]
    pub profile_name: String,
    #[serde(default)]
    pub steamgriddb_key: String,
    #[serde(default)]
    pub eu_dates: bool,
    #[serde(default)]
    pub carousel_lists: Vec<GameList>,
    #[serde(default)]
    pub carousel_order: Vec<CarouselRef>,
    #[serde(default)]
    pub music_muted: bool,
    #[serde(default)]
    pub sfx_muted: bool,
    #[serde(default)]
    pub dockbar_theme: DockbarTheme,
    #[serde(default = "default_left_deadzone")]
    pub left_deadzone: f32,
    #[serde(default = "default_right_deadzone")]
    pub right_deadzone: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DockbarTheme {
    Metallic,
    Simple,
}

impl Default for DockbarTheme {
    fn default() -> Self {
        DockbarTheme::Metallic
    }
}

impl DockbarTheme {
    pub fn all() -> &'static [DockbarTheme] {
        &[DockbarTheme::Metallic, DockbarTheme::Simple]
    }
    pub fn label(&self) -> &'static str {
        match self {
            DockbarTheme::Metallic => "Metallic",
            DockbarTheme::Simple => "Simple",
        }
    }
    pub fn next(&self) -> DockbarTheme {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> DockbarTheme {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarouselRef {
    Game(PathBuf),
    List(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GameList {
    pub name: String,
    /// Game paths — duplicates are allowed here (lists are exempt from the no-dupe rule).
    pub games: Vec<PathBuf>,
    #[serde(default)]
    pub align: ListAlignment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListAlignment {
    Manual,
    Alphabetical,
    ReverseAlphabetical,
}

impl Default for ListAlignment {
    fn default() -> Self {
        ListAlignment::Manual
    }
}

impl ListAlignment {
    pub fn all() -> &'static [ListAlignment] {
        &[ListAlignment::Manual, ListAlignment::Alphabetical, ListAlignment::ReverseAlphabetical]
    }
    pub fn label(&self) -> &'static str {
        match self {
            ListAlignment::Manual => "Manual",
            ListAlignment::Alphabetical => "A-Z",
            ListAlignment::ReverseAlphabetical => "Z-A",
        }
    }
    pub fn next(&self) -> ListAlignment {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> ListAlignment {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

fn default_profile_name() -> String {
    "Player".to_string()
}

fn default_music_volume() -> f32 {
    0.5
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
            gpu_backend: GpuBackend::default(),
            resolution_preset: ResolutionPreset::default(),
            audio_output_device: None,
            audio_volume: default_audio_volume(),
            multicore: default_multicore(),
            async_shaders: false,
            library_folders: Vec::new(),
            view_mode: ViewMode::Carousel,
            carousel_theme: CarouselTheme::default(),
            backdrop_theme: BackdropTheme::default(),
            light_mode: false,
            music_volume: default_music_volume(),
            sfx_volume: default_sfx_volume(),
            favorites: Vec::new(),
            profile_avatar: None,
            profile_name: default_profile_name(),
            steamgriddb_key: String::new(),
            eu_dates: false,
            carousel_lists: Vec::new(),
            carousel_order: Vec::new(),
            music_muted: false,
            sfx_muted: false,
            dockbar_theme: DockbarTheme::default(),
            left_deadzone: default_left_deadzone(),
            right_deadzone: default_right_deadzone(),
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

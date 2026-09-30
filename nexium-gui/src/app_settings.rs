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
            CarouselTheme::Aqua => (0x14, 0xB8, 0xA6),
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
    CherryBlossom,
    None,
}

impl Default for BackdropTheme {
    fn default() -> Self {
        BackdropTheme::Waves
    }
}

impl BackdropTheme {
    pub fn all() -> &'static [BackdropTheme] {
        &[
            BackdropTheme::Waves,
            BackdropTheme::Gradient,
            BackdropTheme::Space,
            BackdropTheme::CherryBlossom,
            BackdropTheme::None,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            BackdropTheme::Waves => "Waves",
            BackdropTheme::Gradient => "Gradient",
            BackdropTheme::Space => "Space",
            BackdropTheme::CherryBlossom => "Cherry Blossom",
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
    P720,
    P1080,
    P1440,
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
            ResolutionPreset::P720 => "1280 \u{00D7} 720",
            ResolutionPreset::P1080 => "1920 \u{00D7} 1080",
            ResolutionPreset::P1440 => "2560 \u{00D7} 1440",
            ResolutionPreset::P2160 => "3840 \u{00D7} 2160",
        }
    }
    pub fn scale(&self) -> u32 {
        match self {
            ResolutionPreset::P720 => 1,
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
    Bicubic,
    ScaleForce,
    Fsr,
}

impl Default for FilterMode {
    fn default() -> Self {
        FilterMode::Nearest
    }
}

impl FilterMode {
    pub fn all() -> &'static [FilterMode] {
        &[
            FilterMode::Nearest,
            FilterMode::Linear,
            FilterMode::Bicubic,
            FilterMode::ScaleForce,
            FilterMode::Fsr,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            FilterMode::Nearest => "Nearest Neighbor",
            FilterMode::Linear => "Bilinear",
            FilterMode::Bicubic => "Bicubic",
            FilterMode::ScaleForce => "ScaleForce",
            FilterMode::Fsr => "AMD FSR 1",
        }
    }
    pub fn scaling_filter(&self) -> nexium_gpu::presentation::ScalingFilter {
        use nexium_gpu::presentation::ScalingFilter;
        match self {
            FilterMode::Nearest => ScalingFilter::Nearest,
            FilterMode::Linear => ScalingFilter::Linear,
            FilterMode::Bicubic => ScalingFilter::Bicubic,
            FilterMode::ScaleForce => ScalingFilter::ScaleForce,
            FilterMode::Fsr => ScalingFilter::Fsr,
        }
    }
}

fn default_output_scale() -> u8 {
    1
}
fn default_fsr_sharpness() -> u8 {
    87
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
fn default_docked() -> bool {
    true
}
fn default_game_bar() -> bool {
    true
}

fn default_menu_music_track() -> u8 {
    255
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceDebugSettings {
    pub cpu_backend_dynarmic: bool,
    pub cpu_cores_4: bool,
    pub dynarmic_code_page_cache: bool,
    pub dynarmic_jit_size_64: bool,
    pub async_gpu: bool,
    pub async_gpu_defer_smallrt: bool,
    pub gpu_pipeline: bool,
    pub fermi_lazy_drain: bool,
    pub input_mirror: bool,
    pub cbuf_writethrough: bool,
    pub resident_vb: bool,
    pub fermi_async_blit: bool,
    pub resident_cbuf: bool,
    pub async_smallrt_wb: bool,
}

impl PerformanceDebugSettings {
    fn entries(&self) -> [(bool, &'static str, &'static str); 14] {
        [
            (self.cpu_backend_dynarmic, "NEXIUM_CPU_BACKEND", "dynarmic"),
            (self.cpu_cores_4, "NEXIUM_CPU_CORES", "4"),
            (
                self.dynarmic_code_page_cache,
                "DYNARMIC_CODE_PAGE_CACHE",
                "1",
            ),
            (self.dynarmic_jit_size_64, "DYNARMIC_JIT_SIZE", "64"),
            (self.async_gpu, "NEXIUM_ASYNC_GPU", "1"),
            (
                self.async_gpu_defer_smallrt,
                "NEXIUM_ASYNC_GPU_DEFER_SMALLRT",
                "1",
            ),
            (self.gpu_pipeline, "NEXIUM_GPU_PIPELINE", "1"),
            (self.fermi_lazy_drain, "NEXIUM_FERMI_LAZY_DRAIN", "1"),
            (self.input_mirror, "NEXIUM_INPUT_MIRROR", "1"),
            (self.cbuf_writethrough, "NEXIUM_CBUF_WRITETHROUGH", "1"),
            (self.resident_vb, "NEXIUM_RESIDENT_VB", "1"),
            (self.fermi_async_blit, "NEXIUM_FERMI_ASYNC_BLIT", "1"),
            (self.resident_cbuf, "NEXIUM_RESIDENT_CBUF", "1"),
            (self.async_smallrt_wb, "NEXIUM_ASYNC_SMALLRT_WB", "1"),
        ]
    }

    pub fn enabled_overrides(&self) -> Vec<(&'static str, &'static str)> {
        self.entries()
            .into_iter()
            .filter_map(|(enabled, name, value)| enabled.then_some((name, value)))
            .collect()
    }

    pub fn apply_to_process(&self) {
        for (name, value) in self.enabled_overrides() {
            if std::env::var_os(name).is_none() {
                std::env::set_var(name, value);
            }
        }
    }

    pub fn enable_safe_preset(&mut self) {
        self.clear();
        self.cpu_backend_dynarmic = true;
        self.cpu_cores_4 = true;
        self.dynarmic_code_page_cache = true;
        self.dynarmic_jit_size_64 = true;
    }

    pub fn sanitize_unsafe_gpu_overrides(&mut self) -> bool {
        let unsafe_requested = self.async_gpu
            || self.async_gpu_defer_smallrt
            || self.gpu_pipeline
            || self.fermi_lazy_drain
            || self.cbuf_writethrough
            || self.fermi_async_blit
            || self.async_smallrt_wb;
        self.async_gpu = false;
        self.async_gpu_defer_smallrt = false;
        self.gpu_pipeline = false;
        self.fermi_lazy_drain = false;
        self.cbuf_writethrough = false;
        self.fermi_async_blit = false;
        self.async_smallrt_wb = false;
        unsafe_requested
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
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
    #[serde(default = "default_fsr_sharpness")]
    pub fsr_sharpness: u8,
    #[serde(default)]
    pub dpi_aware: bool,
    #[serde(default = "default_vsync")]
    pub vsync: bool,
    #[serde(default)]
    pub cpu_backend: CpuBackend,
    #[serde(default)]
    pub gpu_backend: GpuBackend,
    #[serde(default)]
    pub gpu_device: Option<String>,
    #[serde(default)]
    pub resolution_preset: ResolutionPreset,
    #[serde(default)]
    pub audio_output_device: Option<String>,
    #[serde(default = "default_audio_volume")]
    pub audio_volume: f32,
    #[serde(default = "default_multicore")]
    pub multicore: bool,
    #[serde(default = "default_docked")]
    pub docked: bool,
    #[serde(default)]
    pub async_shaders: bool,
    #[serde(default)]
    pub depth_share: bool,
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
    #[serde(default = "default_menu_music_track")]
    pub menu_music_track: u8,
    #[serde(default)]
    pub sfx_muted: bool,
    #[serde(default)]
    pub dockbar_theme: DockbarTheme,
    #[serde(default)]
    pub perf_overlay_corner: OverlayCorner,
    #[serde(default)]
    pub perf_overlay_hidden: bool,
    #[serde(default = "default_game_bar")]
    pub game_bar: bool,
    #[serde(default = "default_left_deadzone")]
    pub left_deadzone: f32,
    #[serde(default = "default_right_deadzone")]
    pub right_deadzone: f32,
    #[serde(default)]
    pub gamepad: Option<String>,
    #[serde(default = "default_emulated_device")]
    pub emulate_mouse: bool,
    #[serde(default = "default_emulated_device")]
    pub emulate_keyboard: bool,
    #[serde(default = "default_emulated_device")]
    pub emulate_touch: bool,
    #[serde(default = "default_motion_enabled")]
    pub motion_enabled: bool,
    #[serde(default = "default_motion_recenter_key")]
    pub motion_recenter_key: String,
    #[serde(default)]
    pub motion_recenter_button: MotionRecenterButton,
    #[serde(default = "default_vibration_enabled")]
    pub vibration_enabled: bool,
    #[serde(default = "default_vibration_strength")]
    pub vibration_strength: u8,
    #[serde(default)]
    pub performance_debug: PerformanceDebugSettings,
}

fn default_emulated_device() -> bool {
    true
}

fn default_motion_enabled() -> bool {
    true
}

fn default_motion_recenter_key() -> String {
    "F8".to_string()
}

fn default_vibration_enabled() -> bool {
    true
}

fn default_vibration_strength() -> u8 {
    100
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlayCorner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Default for OverlayCorner {
    fn default() -> Self {
        OverlayCorner::TopLeft
    }
}

impl OverlayCorner {
    pub fn all() -> &'static [OverlayCorner] {
        &[
            OverlayCorner::TopLeft,
            OverlayCorner::TopRight,
            OverlayCorner::BottomLeft,
            OverlayCorner::BottomRight,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            OverlayCorner::TopLeft => "Top Left",
            OverlayCorner::TopRight => "Top Right",
            OverlayCorner::BottomLeft => "Bottom Left",
            OverlayCorner::BottomRight => "Bottom Right",
        }
    }
    pub fn is_left(&self) -> bool {
        matches!(self, OverlayCorner::TopLeft | OverlayCorner::BottomLeft)
    }
    pub fn is_top(&self) -> bool {
        matches!(self, OverlayCorner::TopLeft | OverlayCorner::TopRight)
    }
    pub fn next(&self) -> OverlayCorner {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> OverlayCorner {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + all.len() - 1) % all.len()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionRecenterButton {
    Paddles,
    Capture,
    Touchpad,
    Unbound,
}

impl Default for MotionRecenterButton {
    fn default() -> Self {
        MotionRecenterButton::Paddles
    }
}

impl MotionRecenterButton {
    pub fn all() -> &'static [MotionRecenterButton] {
        &[
            MotionRecenterButton::Paddles,
            MotionRecenterButton::Capture,
            MotionRecenterButton::Touchpad,
            MotionRecenterButton::Unbound,
        ]
    }
    pub fn label(&self) -> &'static str {
        match self {
            MotionRecenterButton::Paddles => "SL / SR",
            MotionRecenterButton::Capture => "Capture",
            MotionRecenterButton::Touchpad => "Touchpad",
            MotionRecenterButton::Unbound => "None",
        }
    }
    pub fn next(&self) -> MotionRecenterButton {
        let all = Self::all();
        let i = all.iter().position(|x| x == self).unwrap_or(0);
        all[(i + 1) % all.len()]
    }
    pub fn prev(&self) -> MotionRecenterButton {
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
        &[
            ListAlignment::Manual,
            ListAlignment::Alphabetical,
            ListAlignment::ReverseAlphabetical,
        ]
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
            fsr_sharpness: default_fsr_sharpness(),
            dpi_aware: false,
            vsync: default_vsync(),
            cpu_backend: CpuBackend::default(),
            gpu_backend: GpuBackend::default(),
            gpu_device: None,
            resolution_preset: ResolutionPreset::default(),
            audio_output_device: None,
            audio_volume: default_audio_volume(),
            multicore: default_multicore(),
            docked: default_docked(),
            async_shaders: false,
            depth_share: false,
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
            menu_music_track: default_menu_music_track(),
            sfx_muted: false,
            dockbar_theme: DockbarTheme::default(),
            perf_overlay_corner: OverlayCorner::default(),
            perf_overlay_hidden: false,
            game_bar: default_game_bar(),
            left_deadzone: default_left_deadzone(),
            right_deadzone: default_right_deadzone(),
            gamepad: None,
            emulate_mouse: default_emulated_device(),
            emulate_keyboard: default_emulated_device(),
            emulate_touch: default_emulated_device(),
            motion_enabled: default_motion_enabled(),
            motion_recenter_key: default_motion_recenter_key(),
            motion_recenter_button: MotionRecenterButton::default(),
            vibration_enabled: default_vibration_enabled(),
            vibration_strength: default_vibration_strength(),
            performance_debug: PerformanceDebugSettings::default(),
        }
    }
}

impl AppSettings {
    pub fn gpu_device_label(&self) -> String {
        match self.gpu_device.as_deref() {
            None => "Auto".into(),
            Some(id) => nexium_gpu::adapter::available_devices()
                .iter()
                .find(|device| device.id == id)
                .map_or_else(|| "Unavailable GPU (Auto)".into(), |device| device.label()),
        }
    }

    pub fn cycle_gpu_device(&mut self, direction: i32) {
        let devices = nexium_gpu::adapter::available_devices();
        let index = self
            .gpu_device
            .as_ref()
            .and_then(|id| devices.iter().position(|device| device.id == *id))
            .map_or(0, |index| index + 1);
        let next = (index as i32 + direction).rem_euclid(devices.len() as i32 + 1) as usize;
        self.gpu_device = next.checked_sub(1).map(|index| devices[index].id.clone());
    }

    pub fn config_path() -> Option<PathBuf> {
        Some(nexium_common::paths::root().join("app.json"))
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
        cfg.fsr_sharpness = cfg.fsr_sharpness.min(100);
        cfg.vibration_strength = cfg.vibration_strength.clamp(10, 100);
        if cfg.performance_debug.sanitize_unsafe_gpu_overrides() {
            let _ = cfg.save();
        }
        cfg.performance_debug.apply_to_process();
        cfg
    }

    pub fn effective_cpu_backend(&self) -> CpuBackend {
        if let Ok(backend) = std::env::var("NEXIUM_CPU_BACKEND") {
            if backend.eq_ignore_ascii_case("rustarmic") || backend.eq_ignore_ascii_case("rust") {
                return CpuBackend::Rustarmic;
            } else if backend.eq_ignore_ascii_case("dynarmic")
                || backend.eq_ignore_ascii_case("dyn")
            {
                return CpuBackend::Dynarmic;
            }
        }
        self.cpu_backend
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAFE_PRESET_OVERRIDES: [(&str, &str); 4] = [
        ("NEXIUM_CPU_BACKEND", "dynarmic"),
        ("NEXIUM_CPU_CORES", "4"),
        ("DYNARMIC_CODE_PAGE_CACHE", "1"),
        ("DYNARMIC_JIT_SIZE", "64"),
    ];

    #[test]
    fn performance_debug_defaults_to_no_overrides() {
        assert!(PerformanceDebugSettings::default()
            .enabled_overrides()
            .is_empty());
    }

    #[test]
    fn performance_safe_preset_has_exact_environment() {
        let mut settings = PerformanceDebugSettings::default();
        settings.enable_safe_preset();
        assert_eq!(settings.enabled_overrides(), SAFE_PRESET_OVERRIDES);
    }

    #[test]
    fn unsafe_gpu_overrides_are_sanitized_without_disabling_safe_caches() {
        let mut settings = PerformanceDebugSettings {
            async_gpu: true,
            async_gpu_defer_smallrt: true,
            gpu_pipeline: true,
            fermi_lazy_drain: true,
            cbuf_writethrough: true,
            fermi_async_blit: true,
            async_smallrt_wb: true,
            input_mirror: true,
            resident_vb: true,
            resident_cbuf: true,
            ..PerformanceDebugSettings::default()
        };
        assert!(settings.sanitize_unsafe_gpu_overrides());
        assert!(!settings.async_gpu);
        assert!(!settings.async_gpu_defer_smallrt);
        assert!(!settings.gpu_pipeline);
        assert!(!settings.fermi_lazy_drain);
        assert!(!settings.cbuf_writethrough);
        assert!(!settings.fermi_async_blit);
        assert!(!settings.async_smallrt_wb);
        assert!(settings.input_mirror);
        assert!(settings.resident_vb);
        assert!(settings.resident_cbuf);
        assert!(!settings.sanitize_unsafe_gpu_overrides());
    }

    #[test]
    fn fsr_sharpness_defaults_and_persists() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        value.as_object_mut().unwrap().remove("fsr_sharpness");
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.fsr_sharpness, 87);
        for sharpness in [0, 37, 87, 100] {
            let settings = AppSettings {
                filter: FilterMode::Fsr,
                fsr_sharpness: sharpness,
                ..AppSettings::default()
            };
            let json = serde_json::to_string(&settings).unwrap();
            let restored: AppSettings = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.fsr_sharpness, sharpness);
            assert_eq!(restored.filter, FilterMode::Fsr);
        }
    }

    #[test]
    fn filter_settings_preserve_legacy_serialized_names() {
        for (name, filter) in [("Nearest", FilterMode::Nearest), ("Linear", FilterMode::Linear)] {
            let mut value = serde_json::to_value(AppSettings::default()).unwrap();
            value["filter"] = serde_json::json!(name);
            let settings: AppSettings = serde_json::from_value(value).unwrap();
            assert_eq!(settings.filter, filter);
            assert_eq!(serde_json::to_value(settings).unwrap()["filter"], name);
        }
    }

    #[test]
    fn legacy_settings_json_defaults_performance_debug() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        value.as_object_mut().unwrap().remove("performance_debug");
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert_eq!(
            settings.performance_debug,
            PerformanceDebugSettings::default()
        );
    }

    #[test]
    fn legacy_settings_json_defaults_overlay_corner() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        value.as_object_mut().unwrap().remove("perf_overlay_corner");
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.perf_overlay_corner, OverlayCorner::TopLeft);
        assert!(OverlayCorner::TopLeft.is_left() && OverlayCorner::TopLeft.is_top());
        assert!(!OverlayCorner::BottomRight.is_left() && !OverlayCorner::BottomRight.is_top());
        assert_eq!(OverlayCorner::BottomRight.next(), OverlayCorner::TopLeft);
        assert_eq!(OverlayCorner::TopLeft.prev(), OverlayCorner::BottomRight);
        let json = serde_json::to_string(&OverlayCorner::BottomLeft).unwrap();
        assert_eq!(serde_json::from_str::<OverlayCorner>(&json).unwrap(), OverlayCorner::BottomLeft);
    }

    #[test]
    fn partial_performance_debug_json_uses_field_defaults() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        value["performance_debug"] = serde_json::json!({ "async_gpu": true });
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert!(settings.performance_debug.async_gpu);
        assert_eq!(settings.performance_debug.enabled_overrides().len(), 1);
    }

    #[test]
    fn legacy_settings_json_defaults_motion() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in ["motion_enabled", "motion_recenter_key", "motion_recenter_button"] {
            assert!(object.remove(key).is_some());
        }
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert!(settings.motion_enabled);
        assert_eq!(settings.motion_recenter_key, "F8");
        assert_eq!(settings.motion_recenter_button, MotionRecenterButton::Paddles);
    }

    #[test]
    fn motion_settings_round_trip_and_cycle() {
        let settings = AppSettings {
            motion_enabled: false,
            motion_recenter_key: "Num5".to_string(),
            motion_recenter_button: MotionRecenterButton::Capture,
            ..AppSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let restored: AppSettings = serde_json::from_str(&json).unwrap();
        assert!(!restored.motion_enabled);
        assert_eq!(restored.motion_recenter_key, "Num5");
        assert_eq!(restored.motion_recenter_button, MotionRecenterButton::Capture);
        assert_eq!(serde_json::to_value(MotionRecenterButton::Paddles).unwrap(), "Paddles");
        assert_eq!(MotionRecenterButton::Unbound.next(), MotionRecenterButton::Paddles);
        assert_eq!(MotionRecenterButton::Paddles.prev(), MotionRecenterButton::Unbound);
    }

    #[test]
    fn chosen_gamepad_defaults_to_none_and_persists() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        assert!(value.as_object_mut().unwrap().remove("gamepad").is_some());
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert_eq!(settings.gamepad, None);
        let settings = AppSettings {
            gamepad: Some("Nintendo Switch Joy-Con (L/R)".to_string()),
            ..AppSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let restored: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.gamepad.as_deref(), Some("Nintendo Switch Joy-Con (L/R)"));
    }

    #[test]
    fn vibration_settings_default_and_persist() {
        let mut value = serde_json::to_value(AppSettings::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        for key in ["vibration_enabled", "vibration_strength"] {
            assert!(object.remove(key).is_some());
        }
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert!(settings.vibration_enabled);
        assert_eq!(settings.vibration_strength, 100);
        let settings = AppSettings {
            vibration_enabled: false,
            vibration_strength: 40,
            ..AppSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        let restored: AppSettings = serde_json::from_str(&json).unwrap();
        assert!(!restored.vibration_enabled);
        assert_eq!(restored.vibration_strength, 40);
    }
}

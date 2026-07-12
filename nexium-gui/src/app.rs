use crate::app_settings::{AppSettings, AspectMode, CpuBackend, FilterMode, LogLevel};
use crate::audio::{
    current_stream_info, list_output_devices, push_test_tone, set_master_volume, AudioStreamInfo,
};
use crate::boot::EmulationHandle;
use crate::controller_art::{PRO_BODY, PRO_LEFT_HANDLE};
use crate::controller_config::{ControllerConfig, SwitchButton};
use crate::debugger::DebuggerState;
use crate::input::{InputBackend, InputSnapshot};
use crate::performance::PerformanceMonitor;
use eframe::egui;
use eframe::egui::{Color32, FontId, Rounding, Sense, Stroke, Vec2};
use std::sync::Arc;

const BG: Color32 = Color32::from_rgb(0x0F, 0x0F, 0x11);
const BG_RAISED: Color32 = Color32::from_rgb(0x18, 0x18, 0x1C);
const BG_INPUT: Color32 = Color32::from_rgb(0x20, 0x20, 0x26);
const BORDER: Color32 = Color32::from_rgb(0x2A, 0x2A, 0x32);
const ACCENT: Color32 = Color32::from_rgb(0x2F, 0xB4, 0xEF);
const ACCENT_HV: Color32 = Color32::from_rgb(0x5C, 0xF2, 0xFF);
const ACCENT_DK: Color32 = Color32::from_rgb(0x0D, 0x6F, 0xBB);
const DANGER: Color32 = Color32::from_rgb(0xE0, 0x2A, 0x2A);
const DANGER_HV: Color32 = Color32::from_rgb(0xF0, 0x3C, 0x3C);
const TEXT: Color32 = Color32::from_rgb(0xEC, 0xEC, 0xF0);
const MUTED: Color32 = Color32::from_rgb(0x70, 0x70, 0x80);
const GREEN: Color32 = Color32::from_rgb(0x3C, 0xD4, 0x5C);
const AMBER: Color32 = Color32::from_rgb(0xF5, 0xA6, 0x23);

struct NativeGameTexture {
    texture: eframe::wgpu::Texture,
    id: egui::TextureId,
    width: u32,
    height: u32,
    filter: FilterMode,
}

fn legacy_gui_upload() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_LEGACY_GUI_UPLOAD").is_some())
}

enum StatusIcon {
    Play,
    Pause,
    Stop,
    Dock,
    Handheld,
}

fn status_icon(ui: &mut egui::Ui, icon: StatusIcon, color: Color32) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(13.0, 13.0), Sense::click());
    let p = ui.painter();
    let r = rect.shrink(1.5);
    match icon {
        StatusIcon::Play => {
            p.add(egui::Shape::convex_polygon(
                vec![
                    egui::pos2(r.left() + 1.5, r.top() + 0.5),
                    egui::pos2(r.right() - 0.5, r.center().y),
                    egui::pos2(r.left() + 1.5, r.bottom() - 0.5),
                ],
                color,
                Stroke::NONE,
            ));
        }
        StatusIcon::Pause => {
            let w = r.width() * 0.3;
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(r.left() + 0.5, r.top()),
                    Vec2::new(w, r.height()),
                ),
                1.0,
                color,
            );
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(r.right() - w - 0.5, r.top()),
                    Vec2::new(w, r.height()),
                ),
                1.0,
                color,
            );
        }
        StatusIcon::Stop => {
            p.rect_filled(r.shrink(1.0), 1.5, color);
        }
        StatusIcon::Dock => {
            let screen = egui::Rect::from_min_max(
                egui::pos2(r.left() + r.width() * 0.22, r.top()),
                egui::pos2(r.right() - r.width() * 0.22, r.top() + r.height() * 0.58),
            );
            p.rect_stroke(screen, 1.0, Stroke::new(1.2_f32, color));
            let dock = egui::Rect::from_min_max(
                egui::pos2(r.left(), r.top() + r.height() * 0.48),
                r.right_bottom(),
            );
            p.rect_filled(dock, 1.5, color);
        }
        StatusIcon::Handheld => {
            let body =
                egui::Rect::from_center_size(r.center(), Vec2::new(r.width(), r.height() * 0.64));
            let jc_w = body.width() * 0.26;
            p.rect_filled(
                egui::Rect::from_min_size(body.left_top(), Vec2::new(jc_w, body.height())),
                2.0,
                color,
            );
            p.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(body.right() - jc_w, body.top()),
                    Vec2::new(jc_w, body.height()),
                ),
                2.0,
                color,
            );
            p.rect_stroke(body, 2.0, Stroke::new(1.1_f32, color));
        }
    }
    resp
}

fn gui_rate_stats(kind: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_GUI_PROFILE").is_some()) {
        return;
    }
    static COUNTS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static LAST: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
        std::sync::OnceLock::new();
    COUNTS[kind.min(2)].fetch_add(1, Ordering::Relaxed);
    let last = LAST.get_or_init(|| std::sync::Mutex::new(std::time::Instant::now()));
    let Ok(mut guard) = last.try_lock() else {
        return;
    };
    let elapsed = guard.elapsed().as_secs_f64();
    if elapsed >= 2.0 {
        *guard = std::time::Instant::now();
        let u = COUNTS[0].swap(0, Ordering::Relaxed);
        let r = COUNTS[1].swap(0, Ordering::Relaxed);
        let d = COUNTS[2].swap(0, Ordering::Relaxed);
        log::info!(
            "[gui] updates/s={:.0} frames_new/s={:.0} frames_extra_dropped/s={:.0}",
            u as f64 / elapsed,
            r as f64 / elapsed,
            d as f64 / elapsed
        );
    }
}

pub struct HorizonApp {
    nro_path: String,
    emulation_handle: Option<EmulationHandle>,
    game_texture: Option<egui::TextureHandle>,
    wgpu_state: Option<eframe::egui_wgpu::RenderState>,
    game_texture_native: Option<NativeGameTexture>,
    frame_backlog: std::collections::VecDeque<crate::boot::Frame>,
    show_settings: bool,
    settings_tab: SettingsTab,
    input: Option<InputBackend>,
    last_input: InputSnapshot,
    debugger: DebuggerState,
    performance: PerformanceMonitor,
    log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    controller_config: ControllerConfig,
    rebinding: Option<SwitchButton>,
    rebinding_pad: Option<SwitchButton>,
    input_device: InputDevice,
    app_settings: AppSettings,
    last_buttons_logged: u64,
    last_sticks_logged: [i32; 4],
    audio_device_cache: Option<Vec<String>>,
    splash: crate::splash::Splash,
    library: crate::library::Library,
    stop_fade: Option<std::time::Instant>,
    pause_anim: Option<std::time::Instant>,
    resume_anim: Option<std::time::Instant>,
    stop_anim: Option<std::time::Instant>,
    pill_fade: Option<std::time::Instant>,
    playing_path: Option<std::path::PathBuf>,
    last_home: bool,
    profile_texture: Option<egui::TextureHandle>,
    profile_reload: bool,
    show_profile: bool,
    profile_anim: f32,
    profile: crate::profile::ProfileState,
    play_times: crate::playtime::PlayTimes,
    last_playtime_save: std::time::Instant,
    pub carousel: crate::carousel::CarouselState,
    icon_picker: Option<IconPicker>,
    shop: crate::shop::ShopState,
    active_downloads: Vec<(String, std::sync::Arc<std::sync::Mutex<crate::library::DownloadInfo>>)>,
    download_toast: Option<(String, std::time::Instant)>,
    carousel_settings_open: bool,
    cs_anim: f32,
    cs_tab: usize,
    cs_scroll: f32,
    cs_selected: usize,
    cs_focus_grid: bool,
    cs_nav_cd: f64,
    cs_ab_held: bool,
    cs_list_picks: Vec<std::path::PathBuf>,
    cs_pick_anim: std::collections::HashMap<std::path::PathBuf, (f32, f32)>,
    cs_creating: bool,
    cs_new_name: String,
    cs_new_align: crate::app_settings::ListAlignment,
    cs_dialog_row: usize,
    cs_name_editing: bool,
    cs_create_btn: f32,
    cs_plus_held: bool,
    cs_rename_idx: Option<usize>,
    cs_list_menu: Option<usize>,
    cs_list_menu_sel: usize,
    cs_x_held: bool,
    cs_build_grab: Option<usize>,
    cs_build_insert: usize,
    cs_row_scroll: f32,
    cs_grab_lift: f32,
    perf_pos: Option<egui::Pos2>,
    perf_drag: bool,
    perf_grab_off: egui::Vec2,
    perf_hist: Vec<f32>,
    perf_scale: f32,
    last_frame_res: (u32, u32),
    music_was_on: bool,
    music_fade_start: Option<std::time::Instant>,
    icon_reveal: Option<(usize, std::time::Instant, Option<egui::TextureHandle>)>,
    key_test: Option<std::sync::Arc<std::sync::Mutex<KeyTest>>>,
    key_test_result: Option<(bool, std::time::Instant)>,
    confirm: Option<ConfirmDialog>,
    teardown_at: Option<std::time::Instant>,
    pending_boot: Option<String>,
    modal_cd: f64,
    modal_hold: bool,
    modal_anim: f32,
    modal_snap: Option<ModalSnap>,
    pending_quick: Option<String>,
    modal_active_frame_start: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SettingsTab {
    General,
    Controller,
    Graphics,
    Audio,
    Emulation,
    Logging,
}

enum ConfirmKind {
    CloseGame,
    LaunchGame(String),
    QuickLaunch(String),
    DeleteList(usize),
}

struct ConfirmDialog {
    title: String,
    body: String,
    confirm_label: String,
    selected: usize,
    kind: ConfirmKind,
}

struct ModalSnap {
    teardown: bool,
    title: String,
    body: String,
    label: String,
}

#[derive(Default)]
struct IconFetch {
    done: bool,
    error: Option<String>,
    items: Vec<(String, Option<Vec<u8>>)>,
}

#[derive(Default)]
struct IconApply {
    done: bool,
    bytes: Option<Vec<u8>>,
}

#[derive(Default)]
struct KeyTest {
    done: bool,
    ok: bool,
}

struct IconPicker {
    game_idx: usize,
    game_path: std::path::PathBuf,
    search: String,
    editing: bool,
    anim: f32,
    selected: usize,
    scroll: f32,
    follow_sel: bool,
    squish_at: Option<f64>,
    nav_cd: f64,
    hold: bool,
    built: bool,
    full_urls: Vec<String>,
    thumbs: Vec<Option<egui::TextureHandle>>,
    fetch: std::sync::Arc<std::sync::Mutex<IconFetch>>,
    apply: Option<std::sync::Arc<std::sync::Mutex<IconApply>>>,
}

fn start_icon_fetch(key: String, query: String) -> std::sync::Arc<std::sync::Mutex<IconFetch>> {
    let shared = std::sync::Arc::new(std::sync::Mutex::new(IconFetch::default()));
    let s2 = shared.clone();
    std::thread::spawn(move || {
        let mut result = IconFetch { done: true, error: None, items: Vec::new() };
        if key.trim().is_empty() {
            result.error = Some("No SteamGridDB API key — set it in Preferences > General".into());
        } else {
            let icons = crate::steamgrid::fetch_icons(&key, &query, 15);
            if icons.is_empty() {
                result.error = Some("No icons found for that name".into());
            } else {
                for ic in icons {
                    let bytes = crate::steamgrid::curl_bytes(&ic.thumb_url);
                    result.items.push((ic.full_url, bytes));
                }
            }
        }
        if let Ok(mut g) = s2.lock() {
            *g = result;
        }
    });
    shared
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputDevice {
    Keyboard,
    Gamepad,
}

impl HorizonApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        log_buffer: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
        nro_arg: Option<String>,
    ) -> Self {
        Self::apply_theme(&cc.egui_ctx);
        crate::ui_audio::init();
        let input = InputBackend::new().or_else(|| {
            log::warn!("SDL3 gamepad init failed");
            None
        });
        let nro_path = nro_arg.unwrap_or_default();
        let mut app = Self {
            nro_path: nro_path.clone(),
            emulation_handle: None,
            game_texture: None,
            wgpu_state: cc.wgpu_render_state.clone(),
            game_texture_native: None,
            frame_backlog: std::collections::VecDeque::new(),
            show_settings: false,
            settings_tab: SettingsTab::General,
            input,
            last_input: InputSnapshot::default(),
            debugger: DebuggerState::new(),
            performance: PerformanceMonitor::new(),
            log_buffer,
            controller_config: ControllerConfig::load(),
            rebinding: None,
            rebinding_pad: None,
            input_device: InputDevice::Keyboard,
            app_settings: AppSettings::load(),
            last_buttons_logged: 0,
            last_sticks_logged: [0; 4],
            audio_device_cache: None,
            splash: crate::splash::Splash::new(),
            library: crate::library::Library::new(),
            stop_fade: None,
            pause_anim: None,
            resume_anim: None,
            stop_anim: None,
            pill_fade: None,
            playing_path: None,
            last_home: false,
            profile_texture: None,
            profile_reload: false,
            show_profile: false,
            profile_anim: 0.0,
            profile: crate::profile::ProfileState::new(),
            play_times: crate::playtime::PlayTimes::load(),
            last_playtime_save: std::time::Instant::now(),
            carousel: crate::carousel::CarouselState::new(),
            icon_picker: None,
            shop: crate::shop::ShopState::new(),
            active_downloads: Vec::new(),
            download_toast: None,
            carousel_settings_open: false,
            cs_anim: 0.0,
            cs_tab: 0,
            cs_scroll: 0.0,
            cs_selected: 0,
            cs_focus_grid: false,
            cs_nav_cd: 0.0,
            cs_ab_held: false,
            cs_list_picks: Vec::new(),
            cs_pick_anim: std::collections::HashMap::new(),
            cs_creating: false,
            cs_new_name: String::new(),
            cs_new_align: crate::app_settings::ListAlignment::Manual,
            cs_dialog_row: 0,
            cs_name_editing: false,
            cs_create_btn: 0.0,
            cs_plus_held: false,
            cs_rename_idx: None,
            cs_list_menu: None,
            cs_list_menu_sel: 0,
            cs_x_held: false,
            cs_build_grab: None,
            cs_build_insert: 0,
            cs_row_scroll: 0.0,
            cs_grab_lift: 0.0,
            perf_pos: None,
            perf_drag: false,
            perf_grab_off: egui::Vec2::ZERO,
            perf_hist: Vec::new(),
            perf_scale: 1.0,
            last_frame_res: (0, 0),
            music_was_on: false,
            music_fade_start: None,
            icon_reveal: None,
            key_test: None,
            key_test_result: None,
            confirm: None,
            teardown_at: None,
            pending_boot: None,
            modal_cd: 0.0,
            modal_hold: false,
            modal_anim: 0.0,
            modal_snap: None,
            pending_quick: None,
            modal_active_frame_start: false,
        };
        app.reload_profile_texture(&cc.egui_ctx);
        crate::ui_audio::set_sfx_volume(app.app_settings.sfx_volume);
        app.library
            .rescan(&cc.egui_ctx, &app.app_settings.library_folders);
        if !nro_path.is_empty() {
            let backend = app.app_settings.cpu_backend.to_cpu_kind();
            if let Ok(handle) = EmulationHandle::new(&nro_path, backend, Some(cc.egui_ctx.clone()))
            {
                app.emulation_handle = Some(handle);
                app.playing_path = Some(std::path::PathBuf::from(&nro_path));
                log::info!("Auto-loaded NRO: {} (CPU: {})", nro_path, backend.label());
            } else {
                log::error!("Failed to load NRO: {}", nro_path);
            }
        }
        app
    }

    fn apply_theme(ctx: &egui::Context) {
        let mut s = (*ctx.style()).clone();
        s.visuals.dark_mode = true;
        s.visuals.panel_fill = BG;
        s.visuals.window_fill = BG_RAISED;
        s.visuals.faint_bg_color = BG_RAISED;
        s.visuals.extreme_bg_color = BG;
        s.visuals.override_text_color = Some(TEXT);
        s.visuals.window_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.window_rounding = Rounding::same(8.0);
        s.visuals.menu_rounding = Rounding::same(6.0);
        s.visuals.popup_shadow = egui::epaint::Shadow {
            offset: Vec2::new(0.0, 6.0),
            blur: 16.0,
            spread: 0.0,
            color: Color32::from_black_alpha(100),
        };
        for w in [
            &mut s.visuals.widgets.noninteractive,
            &mut s.visuals.widgets.inactive,
            &mut s.visuals.widgets.hovered,
            &mut s.visuals.widgets.active,
            &mut s.visuals.widgets.open,
        ] {
            w.rounding = Rounding::same(4.0);
        }
        s.visuals.widgets.noninteractive.bg_fill = BG_RAISED;
        s.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, MUTED);
        s.visuals.widgets.inactive.bg_fill = BG_INPUT;
        s.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
        s.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
        s.visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x28, 0x28, 0x30);
        s.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(0x44, 0x44, 0x52));
        s.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
        s.visuals.widgets.active.bg_fill = ACCENT;
        s.visuals.widgets.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
        s.visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
        s.visuals.widgets.open.bg_fill = BG_INPUT;
        s.visuals.widgets.open.bg_stroke = Stroke::new(1.0_f32, ACCENT);
        s.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(0x2F, 0xB4, 0xEF, 0x50);
        s.spacing.item_spacing = Vec2::new(6.0, 4.0);
        s.spacing.button_padding = Vec2::new(10.0, 5.0);
        s.spacing.menu_margin = egui::Margin::same(6.0);
        s.spacing.window_margin = egui::Margin::same(12.0);
        ctx.set_style(s);
    }

    fn library_view(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        const MARGIN: f32 = 26.0;
        ui.add_space(16.0);
        ui.horizontal(|ui| {
            ui.add_space(MARGIN);
            ui.label(egui::RichText::new("Library").size(22.0).strong().color(TEXT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(MARGIN);
                if pill_button(ui, "Open file…", true).clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("Switch games", &["nro", "dxci", "dnsp"])
                        .pick_file()
                    {
                        self.nro_path = p.to_string_lossy().to_string();
                        self.boot_nro(ctx);
                    }
                }
                ui.add_space(8.0);
                if pill_button(ui, "Refresh", false).clicked() {
                    self.library.rescan(ctx, &self.app_settings.library_folders);
                }
            });
        });
        ui.add_space(12.0);

        if !self.library.loaded {
            ui.centered_and_justified(|ui| ui.add(egui::Spinner::new().size(28.0)));
            return;
        }
        {
            let w = &mut ui.style_mut().visuals.widgets;
            let dim = Color32::from_rgb(0x1E, 0x5D, 0x7A);
            w.inactive.bg_fill = dim;
            w.inactive.weak_bg_fill = dim;
            w.hovered.bg_fill = ACCENT;
            w.hovered.weak_bg_fill = ACCENT;
            w.active.bg_fill = ACCENT_HV;
            w.active.weak_bg_fill = ACCENT_HV;
        }

        let mut launch: Option<String> = None;
        let mut add_folder = false;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
            ui.add_space(8.0);
            let tw = 154.0;
            let gap = 20.0;
            let n_games = self.library.games.len();
            let total_tiles = n_games + 1;
            let avail = ui.clip_rect().width();
            let cols = (((avail - 2.0 * MARGIN + gap) / (tw + gap)).floor() as usize)
                .clamp(1, total_tiles);
            let total = cols as f32 * tw + (cols.saturating_sub(1)) as f32 * gap;
            let left = ((avail - total) * 0.5).max(MARGIN);
            let mut idx = 0;
            while idx < total_tiles {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.add_space(left);
                    for c in 0..cols {
                        if idx >= total_tiles {
                            break;
                        }
                        if idx < n_games {
                            let tex = self.library.texture(ctx, idx);
                            let selected = self.library.selected == Some(idx);
                            let resp =
                                game_tile(ui, &self.library.games[idx], tex.as_ref(), selected);
                            if resp.clicked() {
                                self.library.selected = Some(idx);
                            }
                            if resp.double_clicked() {
                                launch = Some(
                                    self.library.games[idx].path.to_string_lossy().to_string(),
                                );
                            }
                        } else if add_folder_tile(ui).clicked() {
                            add_folder = true;
                        }
                        if c + 1 < cols {
                            ui.add_space(gap);
                        }
                        idx += 1;
                    }
                });
                ui.add_space(20.0);
            }
            ui.add_space(4.0);
        });
        if let Some(path) = launch {
            self.nro_path = path;
            self.boot_nro(ctx);
        }
        if add_folder {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                if !self.app_settings.library_folders.contains(&dir) {
                    self.app_settings.library_folders.push(dir);
                    let _ = self.app_settings.save();
                }
                self.library.rescan(ctx, &self.app_settings.library_folders);
            }
        }
    }

    fn poll_frames(&mut self, ctx: &egui::Context) {
        let Some(handle) = &self.emulation_handle else {
            return;
        };
        let tex_opts = match self.app_settings.filter {
            crate::app_settings::FilterMode::Linear => egui::TextureOptions::LINEAR,
            crate::app_settings::FilterMode::Nearest => egui::TextureOptions::NEAREST,
        };
        while let Ok(frame) = handle.frame_rx.try_recv() {
            if frame.width == 0 || frame.height == 0 || frame.pixels.is_empty() {
                continue;
            }
            self.frame_backlog.push_back(frame);
        }
        while self.frame_backlog.len() > 3 {
            self.frame_backlog.pop_front();
            gui_rate_stats(2);
        }
        if let Some(frame) = self.frame_backlog.pop_front() {
            gui_rate_stats(1);
            self.performance.record_frame();
            self.last_frame_res = (frame.width, frame.height);
            log::trace!(
                "frame in: {}x{} ({} bytes)",
                frame.width,
                frame.height,
                frame.pixels.len()
            );
            if self.wgpu_state.is_some() && !legacy_gui_upload() {
                self.upload_frame_native(&frame);
                return;
            }
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [frame.width as usize, frame.height as usize],
                &frame.pixels,
            );
            match &mut self.game_texture {
                Some(t) => t.set(img, tex_opts),
                None => {
                    self.game_texture = Some(ctx.load_texture("game_frame", img, tex_opts));
                    self.carousel.boot_stage = crate::carousel::BootStage::None;
                }
            }
        }
    }

    fn upload_frame_native(&mut self, frame: &crate::boot::Frame) {
        let Some(rs) = self.wgpu_state.clone() else {
            return;
        };
        let need = frame.width as usize * frame.height as usize * 4;
        if frame.pixels.len() < need {
            return;
        }
        let filter = self.app_settings.filter;
        let recreate = self.game_texture_native.as_ref().map_or(true, |t| {
            t.width != frame.width || t.height != frame.height || t.filter != filter
        });
        if recreate {
            self.free_native_texture();
            let texture = rs.device.create_texture(&eframe::wgpu::TextureDescriptor {
                label: Some("nexium_game_frame"),
                size: eframe::wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: eframe::wgpu::TextureDimension::D2,
                format: eframe::wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING
                    | eframe::wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let wgpu_filter = match filter {
                FilterMode::Linear => eframe::wgpu::FilterMode::Linear,
                FilterMode::Nearest => eframe::wgpu::FilterMode::Nearest,
            };
            let id = rs
                .renderer
                .write()
                .register_native_texture(&rs.device, &view, wgpu_filter);
            self.game_texture_native = Some(NativeGameTexture {
                texture,
                id,
                width: frame.width,
                height: frame.height,
                filter,
            });
        }
        if self.carousel.boot_stage != crate::carousel::BootStage::None {
            self.carousel.boot_stage = crate::carousel::BootStage::None;
        }
        let Some(t) = self.game_texture_native.as_ref() else {
            return;
        };
        rs.queue.write_texture(
            eframe::wgpu::ImageCopyTexture {
                texture: &t.texture,
                mip_level: 0,
                origin: eframe::wgpu::Origin3d::ZERO,
                aspect: eframe::wgpu::TextureAspect::All,
            },
            &frame.pixels[..need],
            eframe::wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * 4),
                rows_per_image: Some(frame.height),
            },
            eframe::wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn free_native_texture(&mut self) {
        if let Some(t) = self.game_texture_native.take() {
            if let Some(rs) = &self.wgpu_state {
                rs.renderer.write().free_texture(&t.id);
            }
        }
    }

    fn modal_active(&self) -> bool {
        self.confirm.is_some() || self.teardown_at.is_some() || self.icon_picker.is_some() || self.shop.open || self.carousel_settings_open
    }

    fn update_icon_picker(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        const ICON_GRID_COLS: usize = 5;
        if self.icon_picker.is_none() {
            return;
        }
        let now = ctx.input(|i| i.time);
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let light = self.app_settings.light_mode;
        let accent = match self.app_settings.carousel_theme.color() {
            Some((r, g, b)) => Color32::from_rgb(r, g, b),
            None => self.carousel.ambient_color,
        };
        let panel = if light { Color32::from_rgb(0xF5, 0xF5, 0xF9) } else { Color32::from_rgb(0x16, 0x16, 0x20) };
        let text = if light { Color32::from_rgb(0x1E, 0x1E, 0x28) } else { Color32::from_rgb(0xEC, 0xEC, 0xF0) };
        let muted = if light { Color32::from_rgb(0x60, 0x60, 0x6A) } else { Color32::from_rgb(0x9A, 0x9A, 0xA6) };
        let border = if light { Color32::from_rgb(0xC6, 0xC6, 0xD0) } else { Color32::from_rgb(0x32, 0x32, 0x3E) };
        let field_bg = if light { Color32::from_rgb(0xE6, 0xE6, 0xEC) } else { Color32::from_rgb(0x24, 0x24, 0x2E) };

        // --- Build thumbnail textures once the fetch is done ---
        {
            let p = self.icon_picker.as_mut().unwrap();
            if !p.built {
                let mut ready: Option<Vec<(String, Option<Vec<u8>>)>> = None;
                if let Ok(g) = p.fetch.lock() {
                    if g.done {
                        ready = Some(g.items.clone());
                    }
                }
                if let Some(items) = ready {
                    for (i, (url, bytes)) in items.into_iter().enumerate() {
                        p.full_urls.push(url);
                        let tex = bytes
                            .as_deref()
                            .and_then(|b| image::load_from_memory(b).ok())
                            .map(|im| im.to_rgba8())
                            .map(|rgba| {
                                let (w, h) = rgba.dimensions();
                                let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw());
                                ctx.load_texture(format!("sgdb_thumb_{i}"), ci, egui::TextureOptions::LINEAR)
                            });
                        p.thumbs.push(tex);
                    }
                    p.built = true;
                }
            }
        }

        // --- Poll an in-flight apply download ---
        let mut apply_now: Option<Vec<u8>> = None;
        let mut apply_failed = false;
        if let Some(p) = self.icon_picker.as_ref() {
            if let Some(a) = &p.apply {
                if let Ok(g) = a.lock() {
                    if g.done {
                        match &g.bytes {
                            Some(b) => apply_now = Some(b.clone()),
                            None => apply_failed = true,
                        }
                    }
                }
            }
        }
        if let Some(bytes) = apply_now {
            if let Some(img) = image::load_from_memory(&bytes).ok().map(|im| {
                let rgba = im.to_rgba8();
                let (w, h) = rgba.dimensions();
                let (rgba, w, h) = if w.max(h) > 256 {
                    let r = image::imageops::resize(&rgba, 256, 256, image::imageops::FilterType::Lanczos3);
                    (r.into_raw(), 256u32, 256u32)
                } else {
                    (rgba.into_raw(), w, h)
                };
                egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba)
            }) {
                let (idx, path) = {
                    let p = self.icon_picker.as_ref().unwrap();
                    (p.game_idx, p.game_path.clone())
                };
                if let Some(cache) = crate::library::custom_icon_path(&path) {
                    if let Some(parent) = cache.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&cache, &bytes);
                }
                let old_tex = self.library.texture(ctx, idx);
                self.library.set_icon(idx, img);
                self.icon_reveal = Some((idx, std::time::Instant::now(), old_tex));
                self.carousel.pending_center = Some(idx);
                crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
            }
            self.icon_picker = None;
            return;
        }
        if apply_failed {
            crate::ui_audio::play(crate::ui_audio::Sfx::Error);
            self.icon_picker = None;
            return;
        }

        // --- Input ---
        let (kb_left, kb_right, kb_enter, kb_esc, kb_up, kb_down) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::ArrowUp),
                i.key_pressed(egui::Key::ArrowDown),
            )
        });
        let text_events: Vec<egui::Event> = ctx.input(|i| i.events.clone());
        let li = self.last_input;
        use crate::controller_config::SwitchButton;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let a_edge = kb_enter || (gp_a && !self.hold_pick());
        let b_edge = kb_esc || (gp_b && !self.hold_pick());

        let ready_nav = now - self.icon_picker.as_ref().unwrap().nav_cd > 0.16;
        let sel_before = self.icon_picker.as_ref().unwrap().selected;
        let n = self.icon_picker.as_ref().unwrap().thumbs.len();
        let dancing = self.icon_picker.as_ref().unwrap().squish_at.is_some();
        let editing = self.icon_picker.as_ref().unwrap().editing;

        let mut close = false;
        let mut research = false;
        if !dancing {
            let p = self.icon_picker.as_mut().unwrap();
            if editing {
                for ev in &text_events {
                    match ev {
                        egui::Event::Text(txt) => {
                            for c in txt.chars() {
                                if !c.is_control() && p.search.chars().count() < 48 {
                                    p.search.push(c);
                                }
                            }
                        }
                        egui::Event::Key { key: egui::Key::Backspace, pressed: true, .. } => {
                            p.search.pop();
                        }
                        _ => {}
                    }
                }
                if kb_enter {
                    p.editing = false;
                    p.built = false;
                    p.selected = 0;
                    p.scroll = 0.0;
                    p.full_urls.clear();
                    p.thumbs.clear();
                    research = true;
                } else if kb_esc || b_edge {
                    p.editing = false;
                }
            } else {
                let mut mv_left = kb_left;
                let mut mv_right = kb_right;
                let mut mv_up = kb_up;
                let mut mv_down = kb_down;
                if ready_nav && li.connected {
                    if li.is(SwitchButton::DLeft) || li.lx() < -0.5 {
                        mv_left = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DRight) || li.lx() > 0.5 {
                        mv_right = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DUp) || li.ly() > 0.5 {
                        mv_up = true;
                        p.nav_cd = now;
                    } else if li.is(SwitchButton::DDown) || li.ly() < -0.5 {
                        mv_down = true;
                        p.nav_cd = now;
                    }
                }
                if mv_left && p.selected > 0 {
                    p.selected -= 1;
                    crate::ui_audio::play_move();
                }
                if mv_right && n > 0 && p.selected + 1 < n {
                    p.selected += 1;
                    crate::ui_audio::play_move();
                }
                if mv_down && p.selected + ICON_GRID_COLS < n {
                    p.selected += ICON_GRID_COLS;
                    crate::ui_audio::play_move();
                }
                if mv_up && p.selected >= ICON_GRID_COLS {
                    p.selected -= ICON_GRID_COLS;
                    crate::ui_audio::play_move();
                } else if kb_up && p.selected < ICON_GRID_COLS {
                    p.editing = true;
                }
                if p.selected != sel_before {
                    p.follow_sel = true;
                }
                if a_edge && n > 0 && p.thumbs.get(p.selected).map_or(false, |t| t.is_some()) {
                    p.squish_at = Some(now);
                    crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
                }
                if b_edge {
                    close = true;
                }
            }
        }
        self.set_hold_pick(gp_a || gp_b);
        if close {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            self.icon_picker = None;
            return;
        }
        if research {
            let q = self.icon_picker.as_ref().map(|p| p.search.clone()).unwrap_or_default();
            let key = self.app_settings.steamgriddb_key.clone();
            if let Some(p) = self.icon_picker.as_mut() {
                p.fetch = start_icon_fetch(key, q);
            }
        }

        // --- Dance -> start apply ---
        {
            let p = self.icon_picker.as_mut().unwrap();
            p.anim += (1.0 - p.anim) * (dt * 12.0).min(1.0);
            if let Some(t0) = p.squish_at {
                if (now - t0) as f32 > 0.34 && p.apply.is_none() {
                    let url = p.full_urls.get(p.selected).cloned().unwrap_or_default();
                    let shared = std::sync::Arc::new(std::sync::Mutex::new(IconApply::default()));
                    let s2 = shared.clone();
                    std::thread::spawn(move || {
                        let bytes = crate::steamgrid::curl_bytes(&url);
                        if let Ok(mut g) = s2.lock() {
                            g.done = true;
                            g.bytes = bytes;
                        }
                    });
                    p.apply = Some(shared);
                }
            }
        }

        // --- Draw (snapshot state first to avoid borrow conflicts) ---
        ctx.request_repaint();
        let (anim, search, selected, scroll, squish_at, editing, built, thumbs) = {
            let p = self.icon_picker.as_ref().unwrap();
            (p.anim, p.search.clone(), p.selected, p.scroll, p.squish_at, p.editing, p.built, p.thumbs.clone())
        };
        let err = self
            .icon_picker
            .as_ref()
            .and_then(|p| p.fetch.lock().ok().and_then(|g| g.error.clone()));
        let ease = { let a = anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };
        let t = ctx.input(|i| i.time) as f32;
        let screen = ctx.screen_rect();
        let mut paint = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("icon_picker")));
        paint.set_opacity(ease);
        paint.rect_filled(screen, Rounding::ZERO, Color32::from_black_alpha(205));

        let pop = 0.92 + 0.08 * ease;
        let w = (screen.width() * 0.72).clamp(560.0, 1040.0) * pop;
        let h = (screen.height() * 0.6).clamp(360.0, 560.0) * pop;
        let box_rect = egui::Rect::from_center_size(screen.center(), Vec2::new(w, h));
        paint.rect_filled(box_rect.translate(Vec2::new(0.0, 12.0)), Rounding::same(20.0), Color32::from_black_alpha(90));
        paint.rect_filled(box_rect, Rounding::same(20.0), panel);
        paint.rect_stroke(box_rect, Rounding::same(20.0), Stroke::new(1.5_f32, border));
        paint.text(egui::pos2(box_rect.center().x, box_rect.min.y + 34.0), egui::Align2::CENTER_CENTER, "Choose an Icon", FontId::proportional(22.0), text);

        // Search field
        let field = egui::Rect::from_min_size(egui::pos2(box_rect.min.x + 40.0, box_rect.min.y + 58.0), Vec2::new(w - 80.0, 40.0));
        let field_resp = ui.allocate_rect(field, egui::Sense::click());
        paint.rect_filled(field, Rounding::same(10.0), field_bg);
        paint.rect_stroke(field, Rounding::same(10.0), Stroke::new(if editing { 2.0_f32 } else { 1.0_f32 }, if editing { accent } else { border }));
        let query_disp = if search.is_empty() { "Type a game name…".to_string() } else { search.clone() };
        let qcol = if search.is_empty() { muted } else { text };
        paint.text(egui::pos2(field.min.x + 14.0, field.center().y), egui::Align2::LEFT_CENTER, &query_disp, FontId::proportional(17.0), qcol);
        if editing && (now * 1.6).fract() < 0.5 {
            let tw = ui.fonts(|f| f.layout_no_wrap(search.clone(), FontId::proportional(17.0), text).size().x);
            let cx = (field.min.x + 14.0 + tw).min(field.max.x - 10.0);
            paint.line_segment([egui::pos2(cx, field.center().y - 11.0), egui::pos2(cx, field.center().y + 11.0)], Stroke::new(2.0_f32, accent));
        }

        // Content area
        let content = egui::Rect::from_min_max(egui::pos2(box_rect.min.x + 24.0, field.max.y + 20.0), egui::pos2(box_rect.max.x - 24.0, box_rect.max.y - 54.0));
        let mut new_selected: Option<usize> = None;
        let mut dbl_selected: Option<usize> = None;
        let mut scroll_target = scroll;
        let mut cell_px = 1.0f32;
        let mut max_scroll_rows = 0.0f32;
        let wheel_dy = ctx.input(|i| i.smooth_scroll_delta.y);
        if !built {
            paint.text(content.center(), egui::Align2::CENTER_CENTER, "Searching SteamGridDB…", FontId::proportional(18.0), muted);
        } else if let Some(e) = err {
            paint.text(content.center() - Vec2::new(0.0, 10.0), egui::Align2::CENTER_CENTER, &e, FontId::proportional(17.0), muted);
            paint.text(content.center() + Vec2::new(0.0, 22.0), egui::Align2::CENTER_CENTER, "[Up] search a different name  ·  [B] Cancel", FontId::proportional(13.0), muted);
        } else if thumbs.is_empty() {
            paint.text(content.center() - Vec2::new(0.0, 10.0), egui::Align2::CENTER_CENTER, "No icons found for this name.", FontId::proportional(17.0), muted);
            paint.text(content.center() + Vec2::new(0.0, 22.0), egui::Align2::CENTER_CENTER, "[Up] try a different name  ·  [B] Cancel", FontId::proportional(13.0), muted);
        } else {
            let cols = ICON_GRID_COLS;
            let gap = 18.0;
            let pad_top = 10.0;
            let tile = (((content.width() - gap * (cols as f32 - 1.0)) / cols as f32).min(118.0)).max(48.0);
            let cell = tile + gap;
            let n_items = thumbs.len();
            let rows = (n_items + cols - 1) / cols;
            let sel_row = (selected / cols) as f32;
            let rows_vis = ((content.height() - pad_top) / cell).max(1.0);
            let max_scroll = (rows as f32 - rows_vis).max(0.0);
            // keep-visible target (not centering) so mouse-wheel scroll persists
            let mut kv = scroll;
            if sel_row < scroll {
                kv = sel_row;
            } else if sel_row + 1.0 > scroll + rows_vis {
                kv = sel_row + 1.0 - rows_vis;
            }
            scroll_target = kv.clamp(0.0, max_scroll);
            cell_px = cell;
            max_scroll_rows = max_scroll;

            let grid_w = tile * cols as f32 + gap * (cols as f32 - 1.0);
            let x0 = content.center().x - grid_w * 0.5;
            let y0 = content.min.y + pad_top - scroll * cell;
            let clip = paint.with_clip_rect(content);
            for (i, thumb) in thumbs.iter().enumerate() {
                let row = i / cols;
                let col = i % cols;
                let cx = x0 + col as f32 * cell + tile * 0.5;
                let cy = y0 + row as f32 * cell + tile * 0.5;
                if cy + tile < content.min.y || cy - tile > content.max.y {
                    continue;
                }
                let sel = i == selected;
                let (mut sx, mut sy) = (1.0f32, 1.0f32);
                let mut sz = tile;
                if sel {
                    if let Some(t0) = squish_at {
                        let e = (now - t0) as f32;
                        if e < 0.34 {
                            let q = (e * 34.0).sin() * 0.16 * (1.0 - e / 0.34);
                            sx = 1.0 + q;
                            sy = 1.0 - q;
                        }
                    }
                    sz *= 1.06;
                }
                let r = egui::Rect::from_center_size(egui::pos2(cx, cy), Vec2::new(sz * sx, sz * sy));
                clip.rect_filled(r.translate(Vec2::new(0.0, 4.0)), Rounding::same(14.0), Color32::from_black_alpha(80));
                if sel {
                    crate::carousel::draw_gradient_rounded_rect(&clip, r.center(), r.expand(6.0), 18.0, t, 180);
                    crate::carousel::draw_gradient_rounded_rect(&clip, r.center(), r.expand(3.0), 16.0, t, 255);
                }
                clip.rect_filled(r, Rounding::same(14.0), field_bg);
                if let Some(tex) = thumb {
                    crate::carousel::draw_rounded_image(&clip, tex.id(), r, 14.0, Color32::WHITE);
                } else {
                    clip.text(r.center(), egui::Align2::CENTER_CENTER, "×", FontId::proportional(sz * 0.3), muted);
                }
                if content.contains(egui::pos2(cx, cy)) {
                    let resp = ui.allocate_rect(r, egui::Sense::click());
                    if resp.double_clicked() {
                        dbl_selected = Some(i);
                    } else if resp.clicked() {
                        new_selected = Some(i);
                    }
                }
            }
        }

        let hint = if editing {
            "Type a name  ·  [Enter] Search  ·  [Esc] Done"
        } else {
            "[D-Pad] Browse  ·  [A] Choose  ·  Click search to rename  ·  [B] Cancel"
        };
        paint.text(egui::pos2(box_rect.center().x, box_rect.max.y - 26.0), egui::Align2::CENTER_CENTER, hint, FontId::proportional(13.0), muted);

        let mut dbl_whistle = false;
        if let Some(pp) = self.icon_picker.as_mut() {
            if let Some(s) = new_selected {
                pp.selected = s;
            }
            if let Some(s) = dbl_selected {
                pp.selected = s;
                if pp.squish_at.is_none() && pp.thumbs.get(s).map_or(false, |t| t.is_some()) {
                    pp.squish_at = Some(now);
                    dbl_whistle = true;
                }
            }
            if wheel_dy.abs() > 0.1 && max_scroll_rows > 0.0 {
                pp.scroll = (pp.scroll - wheel_dy / cell_px).clamp(0.0, max_scroll_rows);
                pp.follow_sel = false;
            } else if pp.follow_sel {
                pp.scroll += (scroll_target - pp.scroll) * (dt * 16.0).min(1.0);
            }
            if field_resp.clicked() {
                pp.editing = true;
            }
        }
        if dbl_whistle {
            crate::ui_audio::play(crate::ui_audio::Sfx::Whistle);
        }
    }

    fn hold_pick(&self) -> bool {
        self.icon_picker.as_ref().map_or(false, |p| p.hold)
    }
    fn set_hold_pick(&mut self, v: bool) {
        if let Some(p) = self.icon_picker.as_mut() {
            p.hold = v;
        }
    }

    fn cs_working_order(&self) -> Vec<crate::app_settings::CarouselRef> {
        use crate::app_settings::CarouselRef;
        let list_names: Vec<String> = self.app_settings.carousel_lists.iter().map(|l| l.name.clone()).collect();
        let game_paths: Vec<std::path::PathBuf> = self.library.games.iter().filter(|g| g.download.is_none()).map(|g| g.path.clone()).collect();
        let mut out: Vec<CarouselRef> = Vec::new();
        for r in &self.app_settings.carousel_order {
            let keep = match r {
                CarouselRef::List(n) => list_names.iter().any(|x| x == n),
                CarouselRef::Game(p) => game_paths.iter().any(|x| x == p),
            };
            if keep && !out.contains(r) { out.push(r.clone()); }
        }
        for n in &list_names {
            let r = CarouselRef::List(n.clone());
            if !out.contains(&r) { out.push(r); }
        }
        for p in &game_paths {
            let r = CarouselRef::Game(p.clone());
            if !out.contains(&r) { out.push(r); }
        }
        out
    }

    fn update_carousel_settings(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let target = if self.carousel_settings_open { 1.0 } else { 0.0 };
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.cs_anim += (target - self.cs_anim) * (dt * 12.0).min(1.0);
        if self.cs_anim < 0.004 {
            return;
        }
        let ease = { let a = self.cs_anim.clamp(0.0, 1.0); a * a * (3.0 - 2.0 * a) };
        let t = ctx.input(|i| i.time) as f32;
        let full = ctx.screen_rect();
        let s = (full.height() / 820.0).clamp(1.0, 2.4);
        let backdrop_theme = self.app_settings.backdrop_theme;
        let space = backdrop_theme == crate::app_settings::BackdropTheme::Space;
        // Space is always dark, so force bright UI on it (no black-on-black).
        let lightish = self.app_settings.light_mode && !space;
        let pick = |dark: egui::Color32, lite: egui::Color32| if lightish { lite } else { dark };
        let text = pick(egui::Color32::from_rgb(0xEC, 0xEC, 0xF0), egui::Color32::from_rgb(0x1E, 0x1E, 0x28));
        let muted = pick(egui::Color32::from_rgb(0x8A, 0x8A, 0x98), egui::Color32::from_rgb(0x60, 0x60, 0x6A));
        let panel = pick(egui::Color32::from_rgb(0x16, 0x16, 0x1E), egui::Color32::from_rgb(0xFF, 0xFF, 0xFF));
        let border = pick(egui::Color32::from_rgb(0x30, 0x30, 0x3C), egui::Color32::from_rgb(0xC6, 0xC6, 0xD0));
        let sel = pick(egui::Color32::from_rgb(0x1E, 0x1E, 0x28), egui::Color32::from_rgb(0xDD, 0xDD, 0xE6));
        let hover = pick(egui::Color32::from_rgb(0x18, 0x18, 0x22), egui::Color32::from_rgb(0xEA, 0xEA, 0xF0));
        let accent = match self.app_settings.carousel_theme.color() {
            Some((r, g, b)) => egui::Color32::from_rgb(r, g, b),
            None => self.carousel.ambient_color,
        };

        // scale + fade pop, exactly like the Profile page
        let center = full.center();
        let sf = 0.965 + 0.035 * ease;
        let sp = |p: egui::Pos2| center + (p - center) * sf;
        let sr = |r: egui::Rect| egui::Rect::from_center_size(center + (r.center() - center) * sf, r.size() * sf);

        let mut p = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("carousel_settings")));
        p.set_opacity(ease);
        // themed backdrop + scrim (matches Profile)
        crate::carousel::draw_backdrop(&p, full, accent, t, backdrop_theme, 1.0, if lightish { 1.0 } else { 0.0 });
        let scrim = if lightish {
            egui::Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 55)
        } else {
            egui::Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 140)
        };
        p.rect_filled(full, egui::Rounding::ZERO, scrim);

        let mx = full.width() * 0.055;
        let header_font = 34.0 * s;
        p.text(sp(egui::pos2(full.min.x + mx, full.min.y + 40.0 * s)), egui::Align2::LEFT_TOP, "Carousel Settings", egui::FontId::proportional(header_font * sf), text);
        let header_y = full.min.y + 40.0 * s + header_font + 20.0 * s;
        p.line_segment([sp(egui::pos2(full.min.x + mx, header_y)), sp(egui::pos2(full.max.x - mx, header_y))], egui::Stroke::new(1.0_f32, border));

        // ---- input ----
        use crate::controller_config::SwitchButton;
        let now = ctx.input(|i| i.time);
        let li = self.last_input;
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let a_edge = ctx.input(|i| i.key_pressed(egui::Key::Enter)) || (gp_a && !self.cs_ab_held);
        let b_edge = ctx.input(|i| i.key_pressed(egui::Key::Escape)) || (gp_b && !self.cs_ab_held);
        self.cs_ab_held = gp_a || gp_b;
        let ready = now - self.cs_nav_cd > 0.16;
        let (mut nu, mut nd, mut nl, mut nr) = ctx.input(|i| (
            i.key_pressed(egui::Key::ArrowUp),
            i.key_pressed(egui::Key::ArrowDown),
            i.key_pressed(egui::Key::ArrowLeft),
            i.key_pressed(egui::Key::ArrowRight),
        ));
        let gp_plus = li.connected && li.is(SwitchButton::Plus);
        let plus_edge = (gp_plus && !self.cs_plus_held) || ctx.input(|i| i.key_pressed(egui::Key::N));
        self.cs_plus_held = gp_plus;
        let gp_x = li.connected && li.is(SwitchButton::X);
        let x_edge = (gp_x && !self.cs_x_held) || ctx.input(|i| i.key_pressed(egui::Key::E));
        self.cs_x_held = gp_x;
        if ready && li.connected {
            if li.is(SwitchButton::DUp) || li.ly() > 0.5 { nu = true; self.cs_nav_cd = now; }
            else if li.is(SwitchButton::DDown) || li.ly() < -0.5 { nd = true; self.cs_nav_cd = now; }
            else if li.is(SwitchButton::DLeft) || li.lx() < -0.5 { nl = true; self.cs_nav_cd = now; }
            else if li.is(SwitchButton::DRight) || li.lx() > 0.5 { nr = true; self.cs_nav_cd = now; }
        }

        let menu_open = self.cs_list_menu.is_some();
        let (raw_nu, raw_nd, raw_a, raw_b) = (nu, nd, a_edge, b_edge);
        let block = self.confirm.is_some() || menu_open;
        let (a_edge, b_edge, plus_edge, x_edge) = if block { (false, false, false, false) } else { (a_edge, b_edge, plus_edge, x_edge) };
        let (nu, nd, nl, nr) = if block { (false, false, false, false) } else { (nu, nd, nl, nr) };

        // ---- sidebar (Profile-style pills) ----
        let side_w = (full.width() * 0.22).max(300.0);
        let side_x = full.min.x + mx;
        let side_top = header_y + 34.0 * s;
        let item_h = 64.0 * s;
        let tabs = ["Create A List", "Manage Carousel"];
        for (i, label) in tabs.iter().enumerate() {
            let base = egui::Rect::from_min_size(egui::pos2(side_x, side_top + i as f32 * (item_h + 10.0 * s)), egui::Vec2::new(side_w, item_h));
            let r = sr(base);
            let selected = self.cs_tab == i;
            let ring = if !self.cs_focus_grid { accent } else { border };
            let rounding = egui::Rounding::same(12.0 * s);
            if selected {
                p.rect_filled(r, rounding, sel);
                p.rect_stroke(r, rounding, egui::Stroke::new(1.8_f32, ring));
                let bar = sr(egui::Rect::from_min_size(base.min + egui::Vec2::new(6.0 * s, 12.0 * s), egui::Vec2::new(4.0 * s, base.height() - 24.0 * s)));
                p.rect_filled(bar, egui::Rounding::same(2.0 * s), ring);
            } else if ui.rect_contains_pointer(r) {
                p.rect_filled(r, rounding, hover);
            }
            p.text(sp(egui::pos2(base.min.x + 26.0 * s, base.center().y)), egui::Align2::LEFT_CENTER, *label, egui::FontId::proportional(19.0 * s * sf), if selected { text } else { muted });
            if !block && ui.allocate_rect(r, egui::Sense::click()).clicked() {
                self.cs_tab = i;
                self.cs_focus_grid = false;
                self.cs_build_grab = None;
                self.cs_selected = 0;
                crate::ui_audio::play(crate::ui_audio::Sfx::Select);
            }
        }
        if !self.cs_focus_grid {
            if nu && self.cs_tab > 0 { self.cs_tab -= 1; self.cs_build_grab = None; self.cs_selected = 0; crate::ui_audio::play_move(); }
            if nd && self.cs_tab + 1 < tabs.len() { self.cs_tab += 1; self.cs_build_grab = None; self.cs_selected = 0; crate::ui_audio::play_move(); }
        }

        let footer_y = full.max.y - 60.0 * s;
        p.line_segment([sp(egui::pos2(full.min.x + mx, footer_y)), sp(egui::pos2(full.max.x - mx, footer_y))], egui::Stroke::new(1.0_f32, border));
        let pad = self.last_input.connected;
        let hint = if self.cs_creating {
            if pad { "Change    ·    [A] Edit    ·    [B] Cancel" } else { "[Arrows] Change    ·    [Enter] Edit    ·    [Esc] Cancel" }
        } else if self.cs_tab == 0 && self.cs_focus_grid {
            if pad { "Move    ·    [A] Pick    ·    [+] Create List    ·    [B] Back" } else { "[Arrows] Move    ·    [Enter] Pick    ·    [N] Create List    ·    [Esc] Back" }
        } else if self.cs_tab == 1 && self.cs_focus_grid {
            if pad { "Move    ·    [A] Grab / Drop    ·    [X] Edit list    ·    [B] Back" } else { "[Arrows] Move    ·    [Enter] Grab / Drop    ·    [E] Edit list    ·    [Esc] Back" }
        } else {
            if pad { "Move    ·    [B] Back" } else { "[Arrows] Move    ·    [Esc] Back" }
        };
        let hint_font = egui::FontId::proportional(14.0 * s * sf);
        let hy = full.max.y - 30.0 * s;
        let hint_w = ui.fonts(|f| f.layout_no_wrap(hint.to_string(), hint_font.clone(), muted).size().x);
        p.text(sp(egui::pos2(full.max.x - mx, hy)), egui::Align2::RIGHT_CENTER, hint, hint_font, muted);
        if pad {
            let dp_r = 8.0 * s * sf;
            draw_dpad(&p, sp(egui::pos2(full.max.x - mx - hint_w - dp_r - 8.0 * s, hy)), dp_r, muted);
        }

        let content = egui::Rect::from_min_max(egui::pos2(side_x + side_w + 48.0 * s, side_top), egui::pos2(full.max.x - mx, footer_y - 20.0 * s));

        let dlg = self.cs_creating;
        if self.cs_tab == 0 {
            let games: Vec<usize> = (0..self.library.games.len()).filter(|&i| self.library.games[i].download.is_none()).collect();
            let n = games.len();
            if n == 0 {
                p.text(sp(content.center()), egui::Align2::CENTER_CENTER, "No games in your library yet.", egui::FontId::proportional(18.0 * s * sf), muted);
            } else {
                let cols = 5usize;
                let pad = 16.0 * s;
                let grid = egui::Rect::from_min_max(content.min + egui::Vec2::splat(pad), egui::pos2(content.max.x - pad, content.max.y - 66.0 * s));
                let gap = 22.0 * s;
                let tile = ((grid.width() - gap * (cols as f32 - 1.0)) / cols as f32).min(186.0 * s);
                let cell_w = tile + gap;
                let cell_h = tile + 30.0 * s;
                let rows = (n + cols - 1) / cols;
                let max_scroll = (rows as f32 * cell_h - grid.height()).max(0.0);
                if !dlg {
                    if self.cs_focus_grid {
                        if nr && self.cs_selected + 1 < n { self.cs_selected += 1; crate::ui_audio::play_move(); }
                        if nl { if self.cs_selected % cols == 0 { self.cs_focus_grid = false; } else { self.cs_selected -= 1; crate::ui_audio::play_move(); } }
                        if nd && self.cs_selected + cols < n { self.cs_selected += cols; crate::ui_audio::play_move(); }
                        if nu && self.cs_selected >= cols { self.cs_selected -= cols; crate::ui_audio::play_move(); }
                    } else if nr {
                        self.cs_focus_grid = true;
                        crate::ui_audio::play_move();
                    }
                }
                self.cs_selected = self.cs_selected.min(n.saturating_sub(1));
                let wheel = ui.input(|i| i.smooth_scroll_delta.y);
                if !dlg && wheel.abs() > 0.1 { self.cs_scroll = (self.cs_scroll - wheel).clamp(0.0, max_scroll); }
                if self.cs_focus_grid {
                    let srow = (self.cs_selected / cols) as f32 * cell_h;
                    if srow < self.cs_scroll { self.cs_scroll = srow; }
                    else if srow + tile > self.cs_scroll + grid.height() { self.cs_scroll = srow + tile - grid.height(); }
                }
                self.cs_scroll = self.cs_scroll.clamp(0.0, max_scroll);

                let mut toggle: Option<usize> = None;
                if !dlg && self.cs_focus_grid && a_edge { toggle = Some(self.cs_selected); }

                let clip = p.with_clip_rect(sr(content));
                for (vi, &gi) in games.iter().enumerate() {
                    let rr = vi / cols;
                    let cc = vi % cols;
                    let tx = grid.min.x + cc as f32 * cell_w;
                    let ty = grid.min.y + rr as f32 * cell_h - self.cs_scroll;
                    if ty + tile < content.min.y || ty > grid.max.y { continue; }
                    let path = self.library.games[gi].path.clone();
                    let picked = self.cs_list_picks.iter().any(|p2| *p2 == path);
                    let (fade, pop) = {
                        let e = self.cs_pick_anim.entry(path.clone()).or_insert((if picked { 1.0 } else { 0.0 }, 0.0));
                        let target = if picked { 1.0 } else { 0.0 };
                        e.0 += (target - e.0) * (dt * 9.0).min(1.0);
                        e.1 *= 1.0 - (dt * 6.0).min(1.0);
                        (e.0, e.1)
                    };
                    let bounce = (pop.clamp(0.0, 1.0) * std::f32::consts::PI).sin() * 11.0 * s;
                    let base = egui::Rect::from_min_size(egui::pos2(tx, ty - bounce), egui::Vec2::splat(tile));
                    let rect = sr(base);
                    let selg = self.cs_focus_grid && vi == self.cs_selected;
                    if selg {
                        crate::carousel::draw_gradient_rounded_rect(&clip, rect.center(), rect.expand(5.0), 13.0, t, 110);
                        crate::carousel::draw_gradient_rounded_rect(&clip, rect.center(), rect.expand(2.5), 11.0, t, 255);
                    }
                    clip.rect_filled(rect, egui::Rounding::same(9.0), panel);
                    if let Some(tex) = self.library.texture(ctx, gi) {
                        let g = (66.0 + 189.0 * fade) as u8;
                        crate::carousel::draw_rounded_image(&clip, tex.id(), rect, 9.0, egui::Color32::from_rgb(g, g, g));
                    } else {
                        let g = (110.0 + 90.0 * fade) as u8;
                        clip.text(rect.center(), egui::Align2::CENTER_CENTER, &self.library.games[gi].title, egui::FontId::proportional(12.0 * s), egui::Color32::from_rgb(g, g, g));
                    }
                    if fade > 0.15 {
                        let br = 15.0 * s;
                        let bc = rect.right_top() + egui::Vec2::new(-br - 5.0 * s, br + 5.0 * s);
                        let a = (fade * 255.0) as u8;
                        clip.circle_filled(bc, br, egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a));
                        clip.circle_stroke(bc, br, egui::Stroke::new(1.5_f32, egui::Color32::from_rgba_unmultiplied(255, 255, 255, (a as f32 * 0.5) as u8)));
                        draw_check(&clip, bc, br * 1.15, egui::Color32::from_rgba_unmultiplied(255, 255, 255, a));
                    }
                    if !dlg && !block && ui.rect_contains_pointer(rect) && ui.allocate_rect(rect, egui::Sense::click()).clicked() {
                        toggle = Some(vi);
                        self.cs_focus_grid = true;
                        self.cs_selected = vi;
                    }
                }

                if let Some(vi) = toggle {
                    let gi = games[vi];
                    let path = self.library.games[gi].path.clone();
                    if let Some(pos) = self.cs_list_picks.iter().position(|p2| *p2 == path) {
                        self.cs_list_picks.remove(pos);
                    } else {
                        self.cs_list_picks.push(path.clone());
                    }
                    self.cs_pick_anim.entry(path).or_insert((0.0, 0.0)).1 = 1.0;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
            }

            let want_btn = !self.cs_list_picks.is_empty() && !dlg;
            self.cs_create_btn += ((if want_btn { 1.0 } else { 0.0 }) - self.cs_create_btn) * (dt * 10.0).min(1.0);
            if self.cs_create_btn > 0.01 {
                let bw = 224.0 * s;
                let bh = 50.0 * s;
                let base = egui::Rect::from_min_size(egui::pos2(content.max.x - bw, footer_y - 16.0 * s - bh), egui::Vec2::new(bw, bh));
                let r = sr(base);
                let a = (self.cs_create_btn * 255.0) as u8;
                p.rect_filled(r, egui::Rounding::same(12.0 * s), egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), a));
                p.text(r.center(), egui::Align2::CENTER_CENTER, &format!("Create List  ({})", self.cs_list_picks.len()), egui::FontId::proportional(18.0 * s * sf), egui::Color32::from_rgba_unmultiplied(255, 255, 255, a));
                let clicked = !block && ui.allocate_rect(r, egui::Sense::click()).clicked();
                if want_btn && (clicked || plus_edge) {
                    self.cs_creating = true;
                    self.cs_dialog_row = 0;
                    self.cs_name_editing = false;
                    if self.cs_new_name.trim().is_empty() { self.cs_new_name = format!("List {}", self.app_settings.carousel_lists.len() + 1); }
                    crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                }
            }
        } else {
            use crate::app_settings::CarouselRef;
            let order = self.cs_working_order();
            let mut slots: Vec<(bool, usize, usize)> = Vec::new();
            for r in &order {
                match r {
                    CarouselRef::Game(pth) => { if let Some(gi) = self.library.games.iter().position(|g| &g.path == pth) { slots.push((false, gi, 0)); } }
                    CarouselRef::List(nm) => { if let Some(li) = self.app_settings.carousel_lists.iter().position(|l| &l.name == nm) { let c = self.app_settings.carousel_lists[li].games.len(); slots.push((true, li, c)); } }
                }
            }
            let n = slots.len();
            if n == 0 {
                p.text(sp(content.center()), egui::Align2::CENTER_CENTER, "No games in your library yet.", egui::FontId::proportional(18.0 * s * sf), muted);
            } else {
                let grabbing = self.cs_build_grab.is_some();
                self.cs_grab_lift += ((if grabbing { 1.0 } else { 0.0 }) - self.cs_grab_lift) * (dt * 12.0).min(1.0);
                self.cs_selected = self.cs_selected.min(n - 1);

                let grab_g = self.cs_build_grab;
                let vis: Vec<usize> = (0..n).filter(|&i| Some(i) != grab_g).collect();
                let m = vis.len();

                if self.cs_focus_grid {
                    if grabbing {
                        if nl { self.cs_build_insert = self.cs_build_insert.saturating_sub(1); crate::ui_audio::play_move(); }
                        if nr && self.cs_build_insert < m { self.cs_build_insert += 1; crate::ui_audio::play_move(); }
                    } else {
                        if nr && self.cs_selected + 1 < n { self.cs_selected += 1; crate::ui_audio::play_move(); }
                        if nl { if self.cs_selected == 0 { self.cs_focus_grid = false; } else { self.cs_selected -= 1; crate::ui_audio::play_move(); } }
                    }
                } else if nr {
                    self.cs_focus_grid = true;
                    crate::ui_audio::play_move();
                }

                let mut commit = false;
                if self.cs_focus_grid && a_edge {
                    if grabbing { commit = true; }
                    else { self.cs_build_grab = Some(self.cs_selected); self.cs_build_insert = self.cs_selected; self.cs_grab_lift = 0.0; crate::ui_audio::play(crate::ui_audio::Sfx::Select); }
                }

                // layout
                let tile = (content.height() * 0.44).clamp(96.0 * s, 152.0 * s);
                let gap = 34.0 * s;
                let cell = tile + gap;
                let left_margin = 46.0 * s;
                let view_w = content.width() - left_margin - 30.0 * s;
                let cnt = if grabbing { m } else { n };
                let total = (cnt as f32 * cell - gap).max(tile);
                let max_scroll = (total - view_w).max(0.0);
                let focus_idx = if grabbing { self.cs_build_insert.min(cnt.saturating_sub(1)) } else { self.cs_selected };
                let wheel = ui.input(|i| { let d = i.smooth_scroll_delta; if d.x.abs() > d.y.abs() { d.x } else { d.y } });
                if wheel.abs() > 0.1 { self.cs_row_scroll = (self.cs_row_scroll - wheel).clamp(0.0, max_scroll); }
                if nl || nr || a_edge {
                    let il = focus_idx as f32 * cell;
                    let ir = il + tile;
                    if il < self.cs_row_scroll { self.cs_row_scroll = il; }
                    else if ir > self.cs_row_scroll + view_w { self.cs_row_scroll = ir - view_w; }
                }
                self.cs_row_scroll = self.cs_row_scroll.clamp(0.0, max_scroll);
                let row_x0 = content.min.x + left_margin - self.cs_row_scroll;
                let row_cy = content.center().y + 14.0 * s;
                let slot_x = |i: usize| row_x0 + i as f32 * cell;
                let clip_rect = egui::Rect::from_min_max(egui::pos2(content.min.x, content.min.y - 4.0 * s), content.max);
                let clip = p.with_clip_rect(sr(clip_rect));

                let draw_tile = |clip: &egui::Painter, lib: &mut crate::library::Library, lists: &[crate::app_settings::GameList], rect: egui::Rect, is_list: bool, idx: usize, count: usize, ss: f32| {
                    clip.rect_filled(rect, egui::Rounding::same(10.0), panel);
                    if is_list {
                        clip.rect_filled(rect.shrink(2.0), egui::Rounding::same(9.0), egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 220));
                        clip.text(rect.center() - egui::Vec2::new(0.0, 9.0 * ss), egui::Align2::CENTER_CENTER, &lists[idx].name, egui::FontId::proportional(15.0 * ss), egui::Color32::WHITE);
                        clip.text(rect.center() + egui::Vec2::new(0.0, 16.0 * ss), egui::Align2::CENTER_CENTER, &format!("{} games", count), egui::FontId::proportional(12.0 * ss), egui::Color32::from_rgba_unmultiplied(255, 255, 255, 210));
                    } else if let Some(tex) = lib.texture(ctx, idx) {
                        crate::carousel::draw_rounded_image(clip, tex.id(), rect, 10.0, egui::Color32::WHITE);
                    }
                };

                for (pos, &si) in vis.iter().enumerate() {
                    let (is_list, idx, count) = slots[si];
                    let tx = slot_x(pos);
                    if tx + tile < content.min.x || tx > content.max.x { continue; }
                    let base = egui::Rect::from_min_size(egui::pos2(tx, row_cy - tile / 2.0), egui::Vec2::splat(tile));
                    let rect = sr(base);
                    let focused = !grabbing && self.cs_focus_grid && si == self.cs_selected;
                    if focused {
                        crate::carousel::draw_gradient_rounded_rect(&clip, rect.center(), rect.expand(6.0), 14.0, t, 110);
                        crate::carousel::draw_gradient_rounded_rect(&clip, rect.center(), rect.expand(3.0), 12.0, t, 255);
                    }
                    draw_tile(&clip, &mut self.library, &self.app_settings.carousel_lists, rect, is_list, idx, count, s);
                }

                if grabbing {
                    for k in 0..=m {
                        let lx = slot_x(k) - gap / 2.0;
                        let active = k == self.cs_build_insert;
                        let col = if active { accent } else { egui::Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 70) };
                        let w = if active { 5.0_f32 } else { 2.5_f32 };
                        let top = row_cy - tile / 2.0 - 10.0 * s;
                        let bot = row_cy + tile / 2.0 + 10.0 * s;
                        clip.line_segment([sp(egui::pos2(lx, top)), sp(egui::pos2(lx, bot))], egui::Stroke::new(w, col));
                        if active {
                            clip.circle_filled(sp(egui::pos2(lx, top)), 5.0 * s, accent);
                            clip.circle_filled(sp(egui::pos2(lx, bot)), 5.0 * s, accent);
                        }
                    }
                    if let Some(g) = grab_g {
                        let (is_list, idx, count) = slots[g];
                        let gtile = tile * 0.74;
                        let gx = (slot_x(self.cs_build_insert.min(m)) - gap / 2.0)
                            .clamp(content.min.x + gtile / 2.0 + 8.0 * s, content.max.x - gtile / 2.0 - 8.0 * s);
                        let gy = row_cy - tile / 2.0 - 18.0 * s - self.cs_grab_lift * 24.0 * s - gtile / 2.0;
                        let grect = sr(egui::Rect::from_center_size(egui::pos2(gx, gy), egui::Vec2::splat(gtile)));
                        clip.rect_filled(grect.expand(4.0), egui::Rounding::same(11.0), egui::Color32::from_rgba_unmultiplied(0, 0, 0, 130));
                        draw_tile(&clip, &mut self.library, &self.app_settings.carousel_lists, grect, is_list, idx, count, s * 0.74);
                    }
                }

                // mouse
                let mut grab_req: Option<usize> = None;
                let mut menu_req: Option<usize> = None;
                if !grabbing && !block {
                    for (pos, &si) in vis.iter().enumerate() {
                        let (is_list, list_idx, _) = slots[si];
                        let rect = sr(egui::Rect::from_min_size(egui::pos2(slot_x(pos), row_cy - tile / 2.0), egui::Vec2::splat(tile)));
                        let resp = ui.allocate_rect(rect, egui::Sense::click());
                        if resp.clicked() && ui.rect_contains_pointer(rect) { grab_req = Some(si); }
                        if is_list && resp.secondary_clicked() { menu_req = Some(list_idx); self.cs_selected = si; }
                    }
                    if let Some(si) = grab_req {
                        self.cs_focus_grid = true;
                        self.cs_selected = si;
                        self.cs_build_grab = Some(si);
                        self.cs_build_insert = si;
                        self.cs_grab_lift = 0.0;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                    if self.cs_focus_grid && x_edge {
                        if let Some(&si) = vis.get(self.cs_selected.min(m.saturating_sub(1))) {
                            if slots[si].0 { menu_req = Some(slots[si].1); }
                        }
                    }
                    if let Some(li) = menu_req {
                        self.cs_list_menu = Some(li);
                        self.cs_list_menu_sel = 0;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                    }
                } else if grabbing && !block {
                    for pos in 0..m {
                        let rect = sr(egui::Rect::from_min_size(egui::pos2(slot_x(pos), row_cy - tile / 2.0), egui::Vec2::splat(tile)));
                        if ui.rect_contains_pointer(rect) && ui.allocate_rect(rect, egui::Sense::click()).clicked() {
                            self.cs_build_insert = pos;
                            commit = true;
                        }
                    }
                }

                if commit {
                    if let Some(g) = self.cs_build_grab.take() {
                        let ins = self.cs_build_insert.min(m);
                        let mut ord = order.clone();
                        let item = ord.remove(g);
                        ord.insert(ins.min(ord.len()), item);
                        self.app_settings.carousel_order = ord;
                        let _ = self.app_settings.save();
                        self.cs_selected = ins.min(n - 1);
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    }
                }

                p.text(sp(egui::pos2(content.min.x, content.max.y - 6.0 * s)), egui::Align2::LEFT_BOTTOM,
                    if grabbing { "Placing \u{2014} move to a slot, [A] to drop, [B] to cancel" } else { "[A] Grab \u{00B7} [X] Edit a list" },
                    egui::FontId::proportional(14.0 * s * sf), muted);
            }
        }

        if let Some(li) = self.cs_list_menu {
            if li >= self.app_settings.carousel_lists.len() {
                self.cs_list_menu = None;
            } else {
                p.rect_filled(full, egui::Rounding::ZERO, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120));
                let mw = 320.0 * s;
                let rowh = 54.0 * s;
                let mh = rowh * 2.0 + 66.0 * s;
                let mc = full.center();
                let mr = sr(egui::Rect::from_center_size(mc, egui::vec2(mw, mh)));
                p.rect_filled(mr, egui::Rounding::same(14.0 * s), panel);
                p.rect_stroke(mr, egui::Rounding::same(14.0 * s), egui::Stroke::new(1.5_f32, border));
                let name = self.app_settings.carousel_lists[li].name.clone();
                p.text(sp(egui::pos2(mc.x, mc.y - mh / 2.0 + 26.0 * s)), egui::Align2::CENTER_CENTER, &name, egui::FontId::proportional(17.0 * s * sf), text);
                let red = egui::Color32::from_rgb(0xE0, 0x5A, 0x5A);
                let items = ["Rename List", "Delete List"];
                let base_y = mc.y - mh / 2.0 + 52.0 * s;
                if raw_nu && self.cs_list_menu_sel > 0 { self.cs_list_menu_sel -= 1; crate::ui_audio::play_move(); }
                if raw_nd && self.cs_list_menu_sel < 1 { self.cs_list_menu_sel += 1; crate::ui_audio::play_move(); }
                let mut choose: Option<usize> = None;
                for (i, label) in items.iter().enumerate() {
                    let rb = egui::Rect::from_min_size(egui::pos2(mc.x - mw / 2.0 + 14.0 * s, base_y + i as f32 * rowh), egui::vec2(mw - 28.0 * s, rowh - 8.0 * s));
                    let r = sr(rb);
                    let selrow = self.cs_list_menu_sel == i;
                    let accent_row = if i == 1 { red } else { accent };
                    if selrow {
                        p.rect_filled(r, egui::Rounding::same(9.0 * s), sel);
                        p.rect_stroke(r, egui::Rounding::same(9.0 * s), egui::Stroke::new(2.0_f32, accent_row));
                    } else if ui.rect_contains_pointer(r) {
                        p.rect_filled(r, egui::Rounding::same(9.0 * s), hover);
                    }
                    p.text(sp(rb.center()), egui::Align2::CENTER_CENTER, *label, egui::FontId::proportional(17.0 * s * sf), if i == 1 { red } else { text });
                    let resp = ui.allocate_rect(r, egui::Sense::click());
                    if ui.rect_contains_pointer(r) { self.cs_list_menu_sel = i; }
                    if resp.clicked() { choose = Some(i); }
                }
                if raw_a { choose = Some(self.cs_list_menu_sel); }
                if let Some(c) = choose {
                    self.cs_list_menu = None;
                    if c == 0 {
                        self.cs_creating = true;
                        self.cs_rename_idx = Some(li);
                        self.cs_dialog_row = 0;
                        self.cs_name_editing = false;
                        self.cs_new_name = self.app_settings.carousel_lists[li].name.clone();
                        self.cs_new_align = self.app_settings.carousel_lists[li].align;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Select);
                    } else {
                        let nm = self.app_settings.carousel_lists[li].name.clone();
                        self.confirm = Some(ConfirmDialog {
                            title: "Delete List".into(),
                            body: format!("Delete the list \u{201C}{}\u{201D}? The games themselves are not removed.", nm),
                            confirm_label: "Delete".into(),
                            selected: 0,
                            kind: ConfirmKind::DeleteList(li),
                        });
                        self.modal_hold = true;
                        crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                    }
                } else if raw_b {
                    self.cs_list_menu = None;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                }
            }
        }

        if self.cs_creating {
            p.rect_filled(full, egui::Rounding::ZERO, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150));
            let dw = 580.0 * s;
            let dh = 384.0 * s;
            let cx = full.center().x;
            let top = full.center().y - dh / 2.0;
            let dr = sr(egui::Rect::from_center_size(full.center(), egui::vec2(dw, dh)));
            p.rect_filled(dr, egui::Rounding::same(16.0 * s), panel);
            p.rect_stroke(dr, egui::Rounding::same(16.0 * s), egui::Stroke::new(1.5_f32, border));
            p.text(sp(egui::pos2(cx, top + 34.0 * s)), egui::Align2::CENTER_CENTER, if self.cs_rename_idx.is_some() { "Rename List" } else { "Create List" }, egui::FontId::proportional(24.0 * s * sf), text);

            let field_w = dw - 88.0 * s;
            let field_h = 70.0 * s;
            let name_base = egui::Rect::from_min_size(egui::pos2(cx - field_w / 2.0, top + 78.0 * s), egui::vec2(field_w, field_h));
            let align_base = egui::Rect::from_min_size(egui::pos2(cx - field_w / 2.0, top + 78.0 * s + field_h + 18.0 * s), egui::vec2(field_w, field_h));
            let btn_w = (field_w - 16.0 * s) / 2.0;
            let create_base = egui::Rect::from_min_size(egui::pos2(cx - field_w / 2.0, top + dh - 74.0 * s), egui::vec2(btn_w, 50.0 * s));
            let cancel_base = egui::Rect::from_min_size(egui::pos2(cx - field_w / 2.0 + btn_w + 16.0 * s, top + dh - 74.0 * s), egui::vec2(btn_w, 50.0 * s));

            let mut do_create = false;
            let mut do_cancel = false;

            if self.cs_name_editing {
                let events = ui.input(|i| i.events.clone());
                for ev in events {
                    match ev {
                        egui::Event::Text(txt) => {
                            for ch in txt.chars() {
                                if !ch.is_control() && self.cs_new_name.chars().count() < 28 { self.cs_new_name.push(ch); }
                            }
                        }
                        egui::Event::Key { key: egui::Key::Backspace, pressed: true, .. } => { self.cs_new_name.pop(); }
                        egui::Event::Key { key: egui::Key::Enter, pressed: true, .. } => { self.cs_name_editing = false; }
                        egui::Event::Key { key: egui::Key::Escape, pressed: true, .. } => { self.cs_name_editing = false; }
                        _ => {}
                    }
                }
            } else {
                if nu && self.cs_dialog_row > 0 { self.cs_dialog_row -= 1; crate::ui_audio::play_move(); }
                if nd && self.cs_dialog_row < 3 { self.cs_dialog_row += 1; crate::ui_audio::play_move(); }
                if self.cs_dialog_row == 1 {
                    if nl { self.cs_new_align = self.cs_new_align.prev(); crate::ui_audio::play_move(); }
                    if nr { self.cs_new_align = self.cs_new_align.next(); crate::ui_audio::play_move(); }
                }
                if a_edge {
                    match self.cs_dialog_row {
                        0 => self.cs_name_editing = true,
                        1 => { self.cs_new_align = self.cs_new_align.next(); crate::ui_audio::play(crate::ui_audio::Sfx::Select); }
                        2 => do_create = true,
                        _ => do_cancel = true,
                    }
                }
                if b_edge { do_cancel = true; }
            }

            let value_font = egui::FontId::proportional(18.0 * s * sf);
            let name_val_w = ui.fonts(|f| f.layout_no_wrap(self.cs_new_name.clone(), value_font.clone(), text).size().x);
            let caret_on = self.cs_name_editing && (t * 1.6).fract() < 0.5;
            let draw_field = |p: &egui::Painter, base: egui::Rect, row: usize, label: &str, value: &str, is_align: bool, editing: bool| {
                let r = sr(base);
                let selected = self.cs_dialog_row == row;
                p.rect_filled(r, egui::Rounding::same(10.0 * s), sel);
                p.rect_stroke(r, egui::Rounding::same(10.0 * s), egui::Stroke::new(if selected { 2.0_f32 } else { 1.0_f32 }, if selected { accent } else { border }));
                p.text(sp(egui::pos2(base.min.x + 18.0 * s, base.min.y + 14.0 * s)), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(12.0 * s * sf), muted);
                let val_y = base.min.y + 38.0 * s;
                p.text(sp(egui::pos2(base.min.x + 18.0 * s, val_y)), egui::Align2::LEFT_TOP, value, egui::FontId::proportional(18.0 * s * sf), text);
                if editing && caret_on {
                    let cx0 = base.min.x + 18.0 * s + name_val_w + 2.0 * s;
                    let cr = egui::Rect::from_min_max(egui::pos2(cx0, val_y + 2.0 * s), egui::pos2(cx0 + 2.0 * s, val_y + 20.0 * s));
                    p.rect_filled(sr(cr), egui::Rounding::ZERO, text);
                }
                if is_align {
                    p.text(sp(egui::pos2(base.max.x - 40.0 * s, base.center().y)), egui::Align2::CENTER_CENTER, "\u{2039}", egui::FontId::proportional(22.0 * s * sf), muted);
                    p.text(sp(egui::pos2(base.max.x - 16.0 * s, base.center().y)), egui::Align2::CENTER_CENTER, "\u{203A}", egui::FontId::proportional(22.0 * s * sf), muted);
                }
            };
            draw_field(&p, name_base, 0, "List Name", &self.cs_new_name, false, self.cs_name_editing);
            draw_field(&p, align_base, 1, "List Alignment", self.cs_new_align.label(), true, false);

            let create_label = if self.cs_rename_idx.is_some() { "Save" } else { "Create" };
            for (row, base, lbl, fill) in [(2usize, create_base, create_label, true), (3usize, cancel_base, "Cancel", false)] {
                let r = sr(base);
                let selected = self.cs_dialog_row == row;
                if fill {
                    p.rect_filled(r, egui::Rounding::same(11.0 * s), accent);
                    p.text(r.center(), egui::Align2::CENTER_CENTER, lbl, egui::FontId::proportional(18.0 * s * sf), egui::Color32::WHITE);
                } else {
                    p.rect_filled(r, egui::Rounding::same(11.0 * s), sel);
                    p.text(r.center(), egui::Align2::CENTER_CENTER, lbl, egui::FontId::proportional(18.0 * s * sf), text);
                }
                if selected { p.rect_stroke(r, egui::Rounding::same(11.0 * s), egui::Stroke::new(2.0_f32, accent)); }
            }

            if ui.allocate_rect(sr(name_base), egui::Sense::click()).clicked() { self.cs_dialog_row = 0; self.cs_name_editing = true; }
            if ui.allocate_rect(sr(align_base), egui::Sense::click()).clicked() { self.cs_dialog_row = 1; self.cs_new_align = self.cs_new_align.next(); }
            if ui.allocate_rect(sr(create_base), egui::Sense::click()).clicked() { do_create = true; }
            if ui.allocate_rect(sr(cancel_base), egui::Sense::click()).clicked() { do_cancel = true; }

            if do_create {
                if let Some(li) = self.cs_rename_idx {
                    if li < self.app_settings.carousel_lists.len() {
                        let old = self.app_settings.carousel_lists[li].name.clone();
                        let name = { let nm = self.cs_new_name.trim().to_string(); if nm.is_empty() { old.clone() } else { nm } };
                        for e in self.app_settings.carousel_order.iter_mut() {
                            if let crate::app_settings::CarouselRef::List(n) = e {
                                if *n == old { *n = name.clone(); }
                            }
                        }
                        self.app_settings.carousel_lists[li].name = name;
                        self.app_settings.carousel_lists[li].align = self.cs_new_align;
                        let _ = self.app_settings.save();
                    }
                    self.cs_rename_idx = None;
                    self.cs_new_name.clear();
                    self.cs_creating = false;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
                } else {
                    let name = { let nm = self.cs_new_name.trim().to_string(); if nm.is_empty() { format!("List {}", self.app_settings.carousel_lists.len() + 1) } else { nm } };
                    let mut games = self.cs_list_picks.clone();
                    match self.cs_new_align {
                        crate::app_settings::ListAlignment::Manual => {}
                        crate::app_settings::ListAlignment::Alphabetical | crate::app_settings::ListAlignment::ReverseAlphabetical => {
                            let title_of = |pp: &std::path::PathBuf| self.library.games.iter().find(|g| &g.path == pp).map(|g| g.title.to_lowercase()).unwrap_or_default();
                            games.sort_by(|a, b| title_of(a).cmp(&title_of(b)));
                            if self.cs_new_align == crate::app_settings::ListAlignment::ReverseAlphabetical { games.reverse(); }
                        }
                    }
                    self.app_settings.carousel_lists.push(crate::app_settings::GameList { name, games, align: self.cs_new_align });
                    let _ = self.app_settings.save();
                    self.cs_list_picks.clear();
                    self.cs_pick_anim.clear();
                    self.cs_new_name.clear();
                    self.cs_creating = false;
                    self.cs_create_btn = 0.0;
                    self.cs_tab = 1;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Celebration);
                }
            } else if do_cancel {
                self.cs_creating = false;
                self.cs_rename_idx = None;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            }
        } else if b_edge {
            if self.cs_build_grab.is_some() {
                self.cs_build_grab = None;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else if self.cs_focus_grid {
                self.cs_focus_grid = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            } else {
                self.carousel_settings_open = false;
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            }
        }
        ctx.request_repaint();
    }

    fn theme_accent(&self) -> Color32 {
        match self.app_settings.carousel_theme.color() {
            Some((r, g, b)) => Color32::from_rgb(r, g, b),
            None => {
                let a = self.carousel.ambient_color;
                let mx = a.r().max(a.g()).max(a.b());
                let mn = a.r().min(a.g()).min(a.b());
                if (mx as i32 - mn as i32) < 28 || mx < 96 {
                    Color32::from_rgb(0x2F, 0xB4, 0xEF)
                } else {
                    a
                }
            }
        }
    }

    fn draw_perf_overlay(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let fps = self.performance.get_fps();
        self.perf_hist.push(fps);
        let cap = 140usize;
        if self.perf_hist.len() > cap {
            let n = self.perf_hist.len() - cap;
            self.perf_hist.drain(0..n);
        }
        let ft = self.performance.get_frame_time();
        let (svc, cyc) = self
            .emulation_handle
            .as_ref()
            .map(|h| { let s = h.stats.lock(); (s.svc_count, s.cycle_count) })
            .unwrap_or((0, 0));
        let (rw, rh) = self.last_frame_res;
        let (mut fmin, mut fmax) = (f32::MAX, 0.0f32);
        for &v in &self.perf_hist {
            if v > 0.0 { fmin = fmin.min(v); fmax = fmax.max(v); }
        }
        if fmin == f32::MAX { fmin = 0.0; }

        let fps_col = |v: f32| if v >= 30.0 { GREEN } else if v >= 20.0 { AMBER } else { DANGER };

        let full = ctx.screen_rect();
        let base_bw = 224.0;
        let base_bh = 138.0;
        let mut sc = self.perf_scale.clamp(0.7, 3.0);
        let margin = 14.0;
        let default_pos = egui::pos2(full.max.x - base_bw * sc - margin, full.max.y - base_bh * sc - margin);
        let mut pos = self.perf_pos.unwrap_or(default_pos);
        let bw = base_bw * sc;
        let bh = base_bh * sc;

        let hit = egui::Rect::from_min_size(pos, egui::vec2(bw, bh));
        let resp = ui.interact(hit, egui::Id::new("perf_overlay_drag"), egui::Sense::click_and_drag());
        // grip registered AFTER the body so it wins the shared top-right corner
        let grip = (20.0 * sc).clamp(16.0, 40.0);
        let grip_rect = egui::Rect::from_min_size(egui::pos2(pos.x + bw - grip, pos.y), egui::vec2(grip, grip));
        let gresp = ui.interact(grip_rect, egui::Id::new("perf_resize"), egui::Sense::click_and_drag());
        if gresp.hovered() || gresp.dragged() {
            ctx.set_cursor_icon(egui::CursorIcon::ResizeNeSw);
        }
        let resizing = gresp.dragged() || gresp.drag_started();
        if resizing {
            if let Some(mp) = ctx.pointer_interact_pos() {
                let new_bw = (mp.x - pos.x).clamp(base_bw * 0.7, base_bw * 3.0);
                self.perf_scale = new_bw / base_bw;
                self.perf_drag = false;
            }
        }
        sc = self.perf_scale.clamp(0.7, 3.0);
        let bw = base_bw * sc;
        let bh = base_bh * sc;

        let over_grip = ctx.pointer_interact_pos().map_or(false, |mp| grip_rect.contains(mp));
        if !resizing {
            if resp.drag_started() && !over_grip {
                self.perf_drag = true;
                if let Some(mp) = ctx.pointer_interact_pos() { self.perf_grab_off = mp - pos; }
            }
            if self.perf_drag {
                if let Some(mp) = ctx.pointer_interact_pos() { pos = mp - self.perf_grab_off; }
                if !ui.input(|i| i.pointer.any_down()) { self.perf_drag = false; }
            }
        }
        pos.x = pos.x.clamp(full.min.x + 4.0, (full.max.x - bw - 4.0).max(full.min.x + 4.0));
        pos.y = pos.y.clamp(full.min.y + 4.0, (full.max.y - bh - 4.0).max(full.min.y + 4.0));
        self.perf_pos = Some(pos);
        let rect = egui::Rect::from_min_size(pos, egui::vec2(bw, bh));

        let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("perf_overlay")));
        p.rect_filled(rect, egui::Rounding::same(6.0 * sc), egui::Color32::from_rgba_unmultiplied(0x0A, 0x0A, 0x0D, 236));
        p.rect_stroke(rect, egui::Rounding::same(6.0 * sc), egui::Stroke::new(1.0_f32, if resp.hovered() || self.perf_drag || resizing { egui::Color32::from_gray(110) } else { egui::Color32::from_gray(54) }));
        let mono = |sz: f32| egui::FontId::monospace(sz * sc);
        let white = egui::Color32::from_rgb(0xE8, 0xE8, 0xEC);
        let lx = rect.min.x + 9.0 * sc;
        let rxr = rect.max.x - 9.0 * sc;
        let mut y = rect.min.y + 7.0 * sc;
        p.text(egui::pos2(lx, y), egui::Align2::LEFT_TOP, "NeXium", mono(10.5), white);
        p.text(egui::pos2(rxr, y), egui::Align2::RIGHT_TOP, format!("[{}x{}]", rw, rh), mono(10.5), white);
        y += 15.0 * sc;
        p.text(egui::pos2(lx, y), egui::Align2::LEFT_TOP, format!("FPS: {:>6.1}", fps), mono(10.5), fps_col(fps));
        p.text(egui::pos2(rxr, y), egui::Align2::RIGHT_TOP, format!("[{:.0} {:.0}]", fmin, fmax), mono(10.0), MUTED);
        y += 14.0 * sc;
        p.text(egui::pos2(lx, y), egui::Align2::LEFT_TOP, format!("Frame:{:>5.1}ms", ft), mono(10.5), white);
        y += 14.0 * sc;
        p.text(egui::pos2(lx, y), egui::Align2::LEFT_TOP, format!("SVCs:{:>10}", svc), mono(10.5), egui::Color32::from_rgb(0x4C, 0xC2, 0xF0));
        y += 14.0 * sc;
        let cyc_str = if cyc >= 1_000_000_000 { format!("{:.2}B", cyc as f64 / 1e9) } else if cyc >= 1_000_000 { format!("{:.1}M", cyc as f64 / 1e6) } else { format!("{}", cyc) };
        p.text(egui::pos2(lx, y), egui::Align2::LEFT_TOP, format!("Cyc: {:>10}", cyc_str), mono(10.5), egui::Color32::from_rgb(0x86, 0xD0, 0x5A));
        y += 17.0 * sc;

        let graph = egui::Rect::from_min_max(egui::pos2(lx, y), egui::pos2(rxr, rect.max.y - 8.0 * sc));
        p.rect_filled(graph, egui::Rounding::same(2.0 * sc), egui::Color32::from_black_alpha(140));
        for k in 1..4 {
            let gx = graph.min.x + graph.width() * k as f32 / 4.0;
            p.line_segment([egui::pos2(gx, graph.min.y), egui::pos2(gx, graph.max.y)], egui::Stroke::new(1.0_f32, egui::Color32::from_gray(30)));
        }
        let n = self.perf_hist.len();
        if n > 1 {
            let scale = 70.0f32.max(fmax);
            let bwbar = graph.width() / cap as f32;
            for (i, &v) in self.perf_hist.iter().enumerate() {
                let hnorm = (v / scale).clamp(0.0, 1.0);
                let bx = graph.min.x + i as f32 * bwbar;
                let by = graph.max.y - hnorm * graph.height();
                p.line_segment([egui::pos2(bx, graph.max.y), egui::pos2(bx, by)], egui::Stroke::new(bwbar.max(1.0), fps_col(v)));
            }
        }
        // resize grip visual (two little corner ticks, top-right)
        let gc = egui::Color32::from_gray(if gresp.hovered() || resizing { 150 } else { 90 });
        for k in 0..2 {
            let off = 4.0 * sc + k as f32 * 4.0 * sc;
            p.line_segment([egui::pos2(rect.max.x - off, rect.min.y + 3.0 * sc), egui::pos2(rect.max.x - 3.0 * sc, rect.min.y + off)], egui::Stroke::new(1.5 * sc, gc));
        }
    }

    fn resolve_confirm(&mut self, confirmed: bool) {
        let Some(dlg) = self.confirm.take() else { return };
        if !confirmed {
            return;
        }
        match dlg.kind {
            ConfirmKind::CloseGame => {
                self.stop_emulation();
                self.teardown_at = Some(std::time::Instant::now());
                self.pending_boot = None;
            }
            ConfirmKind::LaunchGame(path) => {
                self.stop_emulation();
                self.teardown_at = Some(std::time::Instant::now());
                self.pending_boot = Some(path);
            }
            ConfirmKind::QuickLaunch(path) => {
                if let Some(idx) = self.library.index_of_path(&std::path::PathBuf::from(&path)) {
                    let n = self.library.move_to_front(idx);
                    self.carousel.selected = n + crate::carousel::CS_FRONT;
                    self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                }
                self.show_profile = false;
                if self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel {
                    self.app_settings.view_mode = crate::app_settings::ViewMode::Carousel;
                    let _ = self.app_settings.save();
                }
                self.pending_quick = Some(path);
            }
            ConfirmKind::DeleteList(li) => {
                if li < self.app_settings.carousel_lists.len() {
                    let name = self.app_settings.carousel_lists[li].name.clone();
                    self.app_settings.carousel_lists.remove(li);
                    self.app_settings.carousel_order.retain(|e| !matches!(e, crate::app_settings::CarouselRef::List(n) if *n == name));
                    let _ = self.app_settings.save();
                    self.cs_selected = 0;
                    self.cs_build_grab = None;
                }
            }
        }
    }

    fn quick_launch_tick(&mut self, ctx: &egui::Context) {
        if self.pending_quick.is_none() {
            return;
        }
        if self.show_profile || self.profile_anim > 0.02 {
            ctx.request_repaint();
            return;
        }
        let path = self.pending_quick.take().unwrap();
        let running = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_running());
        if running || crate::boot::emu_alive() {
            self.stop_emulation();
            self.teardown_at = Some(std::time::Instant::now());
            self.pending_boot = Some(path);
        } else {
            let now = ctx.input(|i| i.time) as f32;
            self.carousel.boot_stage = crate::carousel::BootStage::Transitioning {
                game_index: self.carousel.selected,
                start_time: now,
                launch_path: path,
            };
        }
    }

    fn handle_modal_input(&mut self, ctx: &egui::Context) {
        if self.confirm.is_none() {
            self.modal_hold = false;
            return;
        }
        use crate::controller_config::SwitchButton;
        let now = ctx.input(|i| i.time);
        let (kb_left, kb_right, kb_enter, kb_esc) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Escape),
            )
        });
        let li = self.last_input;
        let ready = now - self.modal_cd > 0.18;
        let mut mv = if kb_left { -1 } else if kb_right { 1 } else { 0 };
        if ready && li.connected {
            let lx = li.lx();
            if li.is(SwitchButton::DLeft) || lx < -0.5 {
                mv = -1;
                self.modal_cd = now;
            } else if li.is(SwitchButton::DRight) || lx > 0.5 {
                mv = 1;
                self.modal_cd = now;
            }
        }
        if mv != 0 {
            if let Some(c) = self.confirm.as_mut() {
                let ns = if mv < 0 { 0 } else { 1 };
                if ns != c.selected {
                    c.selected = ns;
                    crate::ui_audio::play(crate::ui_audio::Sfx::Move);
                }
            }
        }
        let gp_a = li.connected && li.is(SwitchButton::A);
        let gp_b = li.connected && li.is(SwitchButton::B);
        let a_edge = kb_enter || (gp_a && !self.modal_hold);
        let b_edge = kb_esc || (gp_b && !self.modal_hold);
        self.modal_hold = gp_a || gp_b;
        if b_edge {
            crate::ui_audio::play(crate::ui_audio::Sfx::Back);
            self.confirm = None;
            self.carousel.boot_stage = crate::carousel::BootStage::None;
        } else if a_edge {
            let confirmed = self.confirm.as_ref().map_or(false, |c| c.selected == 1);
            crate::ui_audio::play(if confirmed {
                crate::ui_audio::Sfx::WhistleOk
            } else {
                crate::ui_audio::Sfx::Back
            });
            self.resolve_confirm(confirmed);
        }
    }

    fn teardown_tick(&mut self, ctx: &egui::Context) {
        if let Some(start) = self.teardown_at {
            ctx.request_repaint();
            let elapsed = start.elapsed().as_secs_f32();
            let dead = !crate::boot::emu_alive();
            if elapsed >= 8.0 && !dead {
                self.teardown_at = None;
                self.pending_boot = None;
            } else if elapsed >= 0.5 && dead {
                self.teardown_at = None;
                if let Some(path) = self.pending_boot.take() {
                    let now = ctx.input(|i| i.time) as f32;
                    self.carousel.boot_stage = crate::carousel::BootStage::Transitioning {
                        game_index: self.carousel.selected,
                        start_time: now,
                        launch_path: path,
                    };
                }
            }
        }
    }

    fn draw_modal(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        let active = self.modal_active();
        let target = if active { 1.0 } else { 0.0 };
        let speed = if target > self.modal_anim { 12.0 } else { 10.0 };
        self.modal_anim += (target - self.modal_anim) * (dt * speed).min(1.0);
        if (self.modal_anim - target).abs() < 0.003 {
            self.modal_anim = target;
        }

        if active {
            let snap = if self.teardown_at.is_some() {
                ModalSnap { teardown: true, title: String::new(), body: String::new(), label: String::new() }
            } else if let Some(c) = &self.confirm {
                ModalSnap { teardown: false, title: c.title.clone(), body: c.body.clone(), label: c.confirm_label.clone() }
            } else {
                return;
            };
            self.modal_snap = Some(snap);
        }
        if self.modal_anim <= 0.004 {
            if !active {
                self.modal_snap = None;
            }
            return;
        }
        let Some(snap) = self.modal_snap.as_ref() else {
            return;
        };
        let (is_teardown, title, body, label) = (snap.teardown, snap.title.clone(), snap.body.clone(), snap.label.clone());

        let a = self.modal_anim.clamp(0.0, 1.0);
        let ease = a * a * (3.0 - 2.0 * a);

        let light = self.app_settings.light_mode;
        let panel = if light { Color32::from_rgb(0xF5, 0xF5, 0xF9) } else { Color32::from_rgb(0x18, 0x18, 0x22) };
        let text = if light { Color32::from_rgb(0x1E, 0x1E, 0x28) } else { Color32::from_rgb(0xEC, 0xEC, 0xF0) };
        let muted = if light { Color32::from_rgb(0x60, 0x60, 0x6A) } else { Color32::from_rgb(0x9A, 0x9A, 0xA6) };
        let border = if light { Color32::from_rgb(0xC6, 0xC6, 0xD0) } else { Color32::from_rgb(0x32, 0x32, 0x3E) };
        let btn_fill = if light { Color32::from_rgb(0xEA, 0xEA, 0xF0) } else { Color32::from_rgb(0x24, 0x24, 0x2E) };
        let accent = self.theme_accent();

        let screen = ctx.screen_rect();
        let mut p = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("modal_overlay")));
        p.set_opacity(ease);
        p.rect_filled(screen, Rounding::ZERO, Color32::from_black_alpha(195));

        let pop = 0.90 + 0.10 * ease;
        let body_font = FontId::proportional(19.0);
        let body_w = ui.fonts(|f| f.layout_no_wrap(body.clone(), body_font, text).size().x);
        let base_w = (body_w + 72.0).clamp(380.0, (screen.width() - 60.0).max(400.0));
        let w = base_w * pop;
        let h = 224.0 * pop;
        let box_rect = egui::Rect::from_center_size(screen.center(), Vec2::new(w, h));
        p.rect_filled(box_rect.translate(Vec2::new(0.0, 10.0)), Rounding::same(18.0), Color32::from_black_alpha(90));
        p.rect_filled(box_rect, Rounding::same(18.0), panel);
        p.rect_stroke(box_rect, Rounding::same(18.0), Stroke::new(1.5_f32, border));

        if is_teardown {
            p.text(box_rect.center() - Vec2::new(0.0, 14.0 * pop), egui::Align2::CENTER_CENTER, "Please Wait…", FontId::proportional(25.0 * pop), text);
            p.text(box_rect.center() + Vec2::new(0.0, 24.0 * pop), egui::Align2::CENTER_CENTER, "Shutting down the current game", FontId::proportional(14.0 * pop), muted);
            ctx.request_repaint();
            return;
        }

        p.text(egui::pos2(box_rect.center().x, box_rect.min.y + 40.0 * pop), egui::Align2::CENTER_CENTER, &title, FontId::proportional(15.0 * pop), muted);
        p.text(egui::pos2(box_rect.center().x, box_rect.center().y - 14.0 * pop), egui::Align2::CENTER_CENTER, &body, FontId::proportional(19.0 * pop), text);

        let btn_w = w * 0.42;
        let btn_h = 46.0 * pop;
        let by = box_rect.max.y - btn_h - 18.0 * pop;
        let gap = w * 0.05;
        let cancel_rect = egui::Rect::from_min_size(egui::pos2(box_rect.center().x - gap * 0.5 - btn_w, by), Vec2::new(btn_w, btn_h));
        let ok_rect = egui::Rect::from_min_size(egui::pos2(box_rect.center().x + gap * 0.5, by), Vec2::new(btn_w, btn_h));

        let selected = self.confirm.as_ref().map_or(0, |c| c.selected);
        let draw_btn = |rect: egui::Rect, label: &str, sel: bool| {
            let rounding = Rounding::same(10.0);
            p.rect_filled(rect, rounding, btn_fill);
            if sel {
                p.rect_filled(rect, rounding, Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), 42));
                p.rect_stroke(rect, rounding, Stroke::new(2.6_f32, accent));
            } else {
                p.rect_stroke(rect, rounding, Stroke::new(1.2_f32, border));
            }
            let col = if sel {
                let f = |x: u8| (x as f32 + (255.0 - x as f32) * 0.2) as u8;
                Color32::from_rgb(f(accent.r()), f(accent.g()), f(accent.b()))
            } else {
                text
            };
            p.text(rect.center(), egui::Align2::CENTER_CENTER, label, FontId::proportional(18.0 * pop), col);
        };
        draw_btn(cancel_rect, "Cancel", selected == 0);
        draw_btn(ok_rect, &label, selected == 1);

        if self.confirm.is_some() {
            let cancel_resp = ui.allocate_rect(cancel_rect, egui::Sense::click());
            let ok_resp = ui.allocate_rect(ok_rect, egui::Sense::click());
            if let Some(c) = self.confirm.as_mut() {
                if cancel_resp.hovered() {
                    c.selected = 0;
                }
                if ok_resp.hovered() {
                    c.selected = 1;
                }
            }
            if cancel_resp.clicked() {
                crate::ui_audio::play(crate::ui_audio::Sfx::Back);
                self.confirm = None;
                self.carousel.boot_stage = crate::carousel::BootStage::None;
            } else if ok_resp.clicked() {
                crate::ui_audio::play(crate::ui_audio::Sfx::WhistleOk);
                self.resolve_confirm(true);
            }
        }
        ctx.request_repaint();
    }

    fn game_display(&self) -> Option<(egui::TextureId, Vec2)> {
        if let Some(t) = &self.game_texture_native {
            Some((t.id, Vec2::new(t.width as f32, t.height as f32)))
        } else {
            self.game_texture.as_ref().map(|t| (t.id(), t.size_vec2()))
        }
    }

    fn boot_nro(&mut self, ctx: &egui::Context) {
        if self.nro_path.is_empty() {
            return;
        }
        let backend = self.app_settings.cpu_backend.to_cpu_kind();
        log::info!("Boot: NRO={} CPU={}", self.nro_path, backend.label());
        if let Some(mut old) = self.emulation_handle.take() {
            old.stop();
        }
        self.stop_fade = None;
        self.stop_anim = None;
        self.resume_anim = None;
        match EmulationHandle::new(&self.nro_path, backend, Some(ctx.clone())) {
            Ok(h) => {
                crate::ui_audio::play(crate::ui_audio::Sfx::GameBoot);
                self.emulation_handle = Some(h);
                self.game_texture = None;
                self.free_native_texture();
                self.frame_backlog.clear();
                self.pause_anim = None;
                self.pill_fade = None;
                let path = std::path::PathBuf::from(&self.nro_path);
                if let Some(idx) = self.library.index_of_path(&path) {
                    let n = self.library.move_to_front(idx);
                    self.carousel.selected = n + crate::carousel::CS_FRONT;
                    self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                }
                self.playing_path = Some(path);
            }
            Err(e) => log::error!("Boot: {}", e),
        }
    }

    fn stop_emulation(&mut self) {
        self.play_times.save_if_dirty();
        self.carousel.boot_stage = crate::carousel::BootStage::None;
        let was_paused = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_paused());
        let carousel = self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel;
        if let Some(mut h) = self.emulation_handle.take() {
            h.stop();
            self.frame_backlog.clear();
            self.pause_anim = None;
            self.resume_anim = None;
            let has_frame = self.game_texture.is_some() || self.game_texture_native.is_some();
            if carousel && self.playing_path.is_some() {
                if !was_paused && has_frame {
                    self.stop_anim = Some(std::time::Instant::now());
                } else {
                    self.pill_fade = Some(std::time::Instant::now());
                    self.game_texture = None;
                }
            } else if has_frame {
                self.stop_fade = Some(std::time::Instant::now());
                self.playing_path = None;
            } else {
                self.playing_path = None;
            }
        } else {
            self.playing_path = None;
        }
    }

    fn reload_profile_texture(&mut self, ctx: &egui::Context) {
        static DEFAULT_AVATAR: &[u8] = include_bytes!("../../branding/png/logo-256.png");
        self.profile_texture = None;

        let bytes: std::borrow::Cow<[u8]> = match self.app_settings.profile_avatar.clone() {
            Some(path) => match std::fs::read(&path) {
                Ok(b) => std::borrow::Cow::Owned(b),
                Err(_) => {
                    log::warn!("profile avatar unreadable: {}", path.display());
                    std::borrow::Cow::Borrowed(DEFAULT_AVATAR)
                }
            },
            None => std::borrow::Cow::Borrowed(DEFAULT_AVATAR),
        };

        let img = match image::load_from_memory(&bytes)
            .or_else(|_| image::load_from_memory(DEFAULT_AVATAR))
        {
            Ok(i) => i,
            Err(_) => return,
        };

        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let side = w.min(h);
        let ox = (w - side) / 2;
        let oy = (h - side) / 2;
        let cropped = image::imageops::crop_imm(&rgba, ox, oy, side, side).to_image();
        // Downscale large avatars (huge textures blow past GPU limits / egui rejects them).
        let cropped = if side > 256 {
            image::imageops::resize(&cropped, 256, 256, image::imageops::FilterType::Lanczos3)
        } else {
            cropped
        };
        let (fw, fh) = cropped.dimensions();
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [fw as usize, fh as usize],
            cropped.as_raw(),
        );
        self.profile_texture = Some(ctx.load_texture("profile_avatar", color, egui::TextureOptions::LINEAR));
    }

    fn pick_profile_avatar(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg"])
            .pick_file()
        {
            self.app_settings.profile_avatar = Some(p);
            let _ = self.app_settings.save();
            self.profile_reload = true;
        }
    }

    fn game_draw_rect(&self, panel: egui::Rect, tsz: Vec2) -> egui::Rect {
        let avail = panel.size();
        let user_scale = self.app_settings.output_scale.max(1) as f32;
        let draw_size = match self.app_settings.aspect {
            AspectMode::Stretch => avail,
            AspectMode::Letterbox => {
                let s = (avail.x / tsz.x).min(avail.y / tsz.y);
                tsz * s
            }
            AspectMode::Integer => {
                let max_s = (avail.x / tsz.x).min(avail.y / tsz.y).floor().max(1.0);
                tsz * user_scale.min(max_s)
            }
        };
        egui::Rect::from_center_size(panel.center(), draw_size)
    }

    fn is_running(&self) -> bool {
        self.emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_running())
    }
}

fn parse_u64_value(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

fn draw_check(p: &egui::Painter, c: egui::Pos2, sz: f32, col: Color32) {
    let a = c + egui::Vec2::new(-sz * 0.42, sz * 0.02);
    let b = c + egui::Vec2::new(-sz * 0.10, sz * 0.32);
    let d = c + egui::Vec2::new(sz * 0.46, -sz * 0.34);
    let stroke = egui::Stroke::new((sz * 0.20).max(1.6), col);
    p.line_segment([a, b], stroke);
    p.line_segment([b, d], stroke);
}

fn draw_dpad(p: &egui::Painter, c: egui::Pos2, r: f32, col: Color32) {
    let thick = r * 0.66;
    let round = egui::Rounding::same(thick * 0.28);
    p.rect_filled(egui::Rect::from_center_size(c, egui::Vec2::new(r * 2.0, thick)), round, col);
    p.rect_filled(egui::Rect::from_center_size(c, egui::Vec2::new(thick, r * 2.0)), round, col);
}

fn draw_dissolve(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let (cols, rows) = (44usize, 26usize);
    let cw = rect.width() / cols as f32;
    let ch = rect.height() / rows as f32;
    let mut mesh = egui::Mesh::with_texture(tex_id);
    for j in 0..rows {
        for i in 0..cols {
            let local = (t - cell_hash(i, j) * 0.55) / 0.35;
            if local >= 1.0 {
                continue;
            }
            let a = 1.0 - local.max(0.0);
            let drop = if local > 0.0 { local * local * ch * 12.0 } else { 0.0 };
            let jitter = if local > 0.0 {
                (cell_hash(i + 7, j + 3) - 0.5) * local * cw * 1.6
            } else {
                0.0
            };
            let min = egui::pos2(
                rect.min.x + i as f32 * cw + jitter,
                rect.min.y + j as f32 * ch + drop,
            );
            let cell = egui::Rect::from_min_size(min, Vec2::new(cw + 0.5, ch + 0.5));
            let uv = egui::Rect::from_min_max(
                egui::pos2(i as f32 / cols as f32, j as f32 / rows as f32),
                egui::pos2((i + 1) as f32 / cols as f32, (j + 1) as f32 / rows as f32),
            );
            mesh.add_rect_with_uv(cell, uv, Color32::from_white_alpha((a * 255.0) as u8));
        }
    }
    p.add(egui::Shape::mesh(mesh));
}

fn draw_zoom_fade_out(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let t = t.clamp(0.0, 1.0);
    let scale = 1.0 - 0.08 * t;
    let alpha = 1.0 - t;
    let draw_rect = egui::Rect::from_center_size(rect.center(), rect.size() * scale);
    p.image(tex_id, draw_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::from_white_alpha((alpha * 255.0) as u8));
}

fn draw_zoom_fade_in(p: &egui::Painter, rect: egui::Rect, tex_id: egui::TextureId, t: f32) {
    let t = t.clamp(0.0, 1.0);
    let scale = 1.06 - 0.06 * t;
    let alpha = t;
    let draw_rect = egui::Rect::from_center_size(rect.center(), rect.size() * scale);
    p.image(tex_id, draw_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::from_white_alpha((alpha * 255.0) as u8));
}

fn cell_hash(i: usize, j: usize) -> f32 {
    let mut n = (i as u32).wrapping_mul(374761393).wrapping_add((j as u32).wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0xffff) as f32 / 65535.0
}

fn game_tile(
    ui: &mut egui::Ui,
    game: &crate::library::GameEntry,
    tex: Option<&egui::TextureHandle>,
    selected: bool,
) -> egui::Response {
    let tile = Vec2::new(154.0, 202.0);
    let (rect, resp) = ui.allocate_exact_size(tile, Sense::click());
    let hovered = resp.hovered();
    let t = ui.input(|i| i.time) as f32;

    let bg = if selected {
        BG_INPUT
    } else if hovered {
        Color32::from_rgb(0x1C, 0x1C, 0x22)
    } else {
        BG_RAISED
    };
    let rounding = Rounding::same(10.0);
    ui.painter().rect_filled(rect, rounding, bg);

    if selected {
        animated_border(ui.painter(), rect, 10.0, t);
        ui.ctx().request_repaint();
    } else if hovered {
        for i in 1..=3 {
            let e = i as f32;
            ui.painter().rect_stroke(
                rect.expand(e * 1.5),
                Rounding::same(10.0 + e * 1.5),
                Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, (34 / i) as u8)),
            );
        }
        ui.painter()
            .rect_stroke(rect, rounding, Stroke::new(1.4_f32, ACCENT_HV));
    } else {
        ui.painter()
            .rect_stroke(rect, rounding, Stroke::new(1.0_f32, BORDER));
    }

    let pad = 11.0;
    let icon_sz = tile.x - pad * 2.0;
    let icon_rect = egui::Rect::from_min_size(rect.min + Vec2::splat(pad), Vec2::splat(icon_sz));
    if let Some(tex) = tex {
        egui::Image::new((tex.id(), icon_rect.size()))
            .rounding(Rounding::same(6.0))
            .paint_at(ui, icon_rect);
    } else {
        ui.painter()
            .rect_filled(icon_rect, Rounding::same(6.0), Color32::from_rgb(0x12, 0x12, 0x16));
        ui.painter().text(
            icon_rect.center(),
            egui::Align2::CENTER_CENTER,
            game.format,
            FontId::proportional(22.0),
            MUTED,
        );
    }

    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 14.0),
        egui::Align2::CENTER_CENTER,
        elide(&game.title, 20),
        FontId::proportional(12.5),
        if selected { TEXT } else { Color32::from_rgb(0xC8, 0xC8, 0xD2) },
    );

    let sub = if game.author.is_empty() {
        game.format.to_string()
    } else {
        elide(&game.author, 22)
    };
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 32.0),
        egui::Align2::CENTER_CENTER,
        sub,
        FontId::proportional(10.5),
        MUTED,
    );

    resp.on_hover_text(&game.title)
}

fn add_folder_tile(ui: &mut egui::Ui) -> egui::Response {
    let tile = Vec2::new(154.0, 202.0);
    let (rect, resp) = ui.allocate_exact_size(tile, Sense::click());
    let hovered = resp.hovered();

    let bg = if hovered {
        Color32::from_rgb(0x1C, 0x1C, 0x22)
    } else {
        BG_RAISED
    };
    let rounding = Rounding::same(10.0);
    ui.painter().rect_filled(rect, rounding, bg);
    if hovered {
        ui.painter()
            .rect_stroke(rect, rounding, Stroke::new(1.4_f32, ACCENT_HV));
    } else {
        ui.painter()
            .rect_stroke(rect, rounding, Stroke::new(1.0_f32, BORDER));
    }

    let pad = 11.0;
    let icon_sz = tile.x - pad * 2.0;
    let icon_rect = egui::Rect::from_min_size(rect.min + Vec2::splat(pad), Vec2::splat(icon_sz));
    ui.painter().rect_filled(
        icon_rect,
        Rounding::same(6.0),
        Color32::from_rgb(0x12, 0x12, 0x16),
    );
    ui.painter().text(
        icon_rect.center(),
        egui::Align2::CENTER_CENTER,
        "+",
        FontId::proportional(48.0),
        if hovered { ACCENT_HV } else { MUTED },
    );
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 14.0),
        egui::Align2::CENTER_CENTER,
        "Add Folder",
        FontId::proportional(12.5),
        if hovered {
            TEXT
        } else {
            Color32::from_rgb(0xC8, 0xC8, 0xD2)
        },
    );
    ui.painter().text(
        egui::pos2(rect.center().x, icon_rect.max.y + 32.0),
        egui::Align2::CENTER_CENTER,
        ".dnsp / .dxci",
        FontId::proportional(10.5),
        MUTED,
    );

    resp.on_hover_text("Add a folder of .dnsp / .dxci games")
}

fn animated_border(p: &egui::Painter, rect: egui::Rect, r: f32, t: f32) {
    let pulse = 0.5 + 0.5 * (t * 2.1).sin();
    for i in 1..=4 {
        let e = i as f32 * 2.4;
        let fade = 1.0 - (i as f32 - 1.0) / 4.0;
        let a = (52.0 * fade * (0.5 + 0.5 * pulse)) as u8;
        p.rect_stroke(
            rect.expand(e),
            Rounding::same(r + e),
            Stroke::new(2.2_f32, Color32::from_rgba_unmultiplied(0x2F, 0xB4, 0xEF, a)),
        );
    }
    p.rect_stroke(
        rect,
        Rounding::same(r),
        Stroke::new(1.8_f32, lerp_col(ACCENT, ACCENT_HV, pulse)),
    );
}

fn lerp_col(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

fn elide(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    } else {
        s.to_string()
    }
}

fn pill_button(ui: &mut egui::Ui, label: &str, filled: bool) -> egui::Response {
    let font = FontId::proportional(12.5);
    let text_w = ui.fonts(|f| {
        f.layout_no_wrap(label.to_string(), font.clone(), TEXT)
            .size()
            .x
    });
    let size = Vec2::new(text_w + 24.0, 26.0);
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    let (bg, text_col, stroke) = if filled {
        let c = if resp.is_pointer_button_down_on() {
            ACCENT_DK
        } else if resp.hovered() {
            ACCENT_HV
        } else {
            ACCENT
        };
        (c, Color32::WHITE, Stroke::NONE)
    } else {
        let c = if resp.hovered() {
            Color32::from_rgb(0x28, 0x28, 0x30)
        } else {
            Color32::TRANSPARENT
        };
        let bc = if resp.hovered() {
            Color32::from_rgb(0x50, 0x50, 0x60)
        } else {
            BORDER
        };
        (c, TEXT, Stroke::new(1.0_f32, bc))
    };

    ui.painter().rect(rect, Rounding::same(5.0), bg, stroke);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        font,
        text_col,
    );
    resp
}

impl eframe::App for HorizonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.viewport().close_requested()) {
            self.play_times.save_if_dirty();
            if let Some(mut h) = self.emulation_handle.take() {
                h.stop_blocking();
            }
            return;
        }
        gui_rate_stats(0);
        if self.splash.active() {
            if self.emulation_handle.is_some() {
                self.splash = crate::splash::Splash::finished();
            } else if let crate::splash::Step::Intro = self.splash.step(ctx) {
                return;
            }
        }

        self.library.poll();

        // Track shop downloads; on completion rescan + show a "Download Complete" toast.
        for pending in self.shop.new_installs.drain(..).collect::<Vec<_>>() {
            self.active_downloads.push((pending.title, pending.info));
        }
        let mut di = 0;
        while di < self.active_downloads.len() {
            let (done, ok) = self.active_downloads[di]
                .1
                .lock()
                .map(|g| (g.done, g.ok))
                .unwrap_or((false, false));
            if done {
                let (title, _) = self.active_downloads.remove(di);
                if ok {
                    self.download_toast = Some((title, std::time::Instant::now()));
                    self.library.rescan(ctx, &self.app_settings.library_folders);
                }
            } else {
                di += 1;
            }
        }

        if let Some((_, tm, _)) = &self.icon_reveal {
            if tm.elapsed().as_secs_f32() > 1.0 {
                self.icon_reveal = None;
            }
        }

        if self.profile_reload {
            self.profile_reload = false;
            self.reload_profile_texture(ctx);
        }

        if let Some(ref mut ib) = self.input {
            self.last_input = ib.poll(&self.controller_config);
            if self.last_input.connected {
                ctx.request_repaint_after(std::time::Duration::from_millis(8));
            }
            if let Some(btn) = self.rebinding_pad {
                if let Some(gp) = ib.first_pressed() {
                    self.controller_config.set_pad(btn, gp);
                    let _ = self.controller_config.save();
                    self.rebinding_pad = None;
                }
            }
        }

        {
            use crate::controller_config::SwitchButton;
            let a_down = ctx.input(|i| i.key_down(egui::Key::Enter))
                || (self.last_input.connected && self.last_input.is(SwitchButton::A));
            let x_down = ctx.input(|i| i.key_down(egui::Key::X))
                || (self.last_input.connected && self.last_input.is(SwitchButton::X));
            let b_down = self.last_input.connected && self.last_input.is(SwitchButton::B);
            let sel_down = ctx.input(|i| i.key_down(egui::Key::Tab))
                || (self.last_input.connected && self.last_input.is(SwitchButton::Minus));
            self.carousel.a_edge = a_down && !self.carousel.a_held;
            self.carousel.x_edge = x_down && !self.carousel.x_held;
            self.carousel.b_edge = b_down && !self.carousel.b_held;
            self.carousel.sel_edge = sel_down && !self.carousel.sel_held;
            self.carousel.a_held = a_down;
            self.carousel.x_held = x_down;
            self.carousel.b_held = b_down;
            self.carousel.sel_held = sel_down;
        }

        self.modal_active_frame_start = self.modal_active();
        self.handle_modal_input(ctx);
        self.teardown_tick(ctx);
        self.quick_launch_tick(ctx);

        {
            let (m_run, m_pause) = self
                .emulation_handle
                .as_ref()
                .map_or((false, false), |h| (h.is_running(), h.is_paused()));
            let carousel_view =
                self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel;
            let playing_fs = m_run && !m_pause;
            let base_on = carousel_view && !playing_fs && !self.splash.active();
            let shop_music = self.shop.visible();
            let music_mode = if shop_music {
                crate::ui_audio::MusicMode::Shop
            } else {
                crate::ui_audio::MusicMode::Carousel
            };
            // Slowly fade the music back in (~5s) after returning from a game.
            if base_on && !self.music_was_on {
                self.music_fade_start = Some(std::time::Instant::now());
            }
            self.music_was_on = base_on;
            let ramp = if base_on {
                self.music_fade_start
                    .map(|s| (s.elapsed().as_secs_f32() / 5.0).clamp(0.0, 1.0))
                    .unwrap_or(1.0)
            } else {
                1.0
            };
            let mut target = if base_on && !self.app_settings.music_muted { self.app_settings.music_volume * ramp } else { 0.0 };
            let mut lowpass = 0.0f32;
            let profiling = self.show_profile || self.profile_anim > 0.01;
            let on_music_slider = profiling
                && self.profile.tab == crate::profile::ProfileTab::Settings
                && self.profile.focus_content
                && self.profile.row_selected == 3;
            if profiling && !on_music_slider {
                target *= 0.34;
                lowpass = 0.75;
            }
            if self.modal_active() && !shop_music {
                lowpass = lowpass.max(0.6);
            }
            crate::ui_audio::set_music(music_mode, target, lowpass);
            if base_on && ramp < 1.0 {
                ctx.request_repaint();
            }

            let booting_now = m_run
                && !m_pause
                && carousel_view
                && self.game_display().is_none()
                && !self.splash.active();
            if booting_now {
                crate::ui_audio::play_looped(crate::ui_audio::Sfx::AwaitFrame);
            } else {
                crate::ui_audio::stop_loop(crate::ui_audio::Sfx::AwaitFrame);
            }
            if self.teardown_at.is_some() {
                crate::ui_audio::play_looped(crate::ui_audio::Sfx::PleaseWait);
            } else {
                crate::ui_audio::stop_loop(crate::ui_audio::Sfx::PleaseWait);
            }
        }

        {
            let kb_home = ctx.input(|i| i.key_pressed(egui::Key::Home) || i.key_pressed(egui::Key::Backtick));
            let gp_home = self.last_input.home;
            let home_edge = kb_home || (gp_home && !self.last_home);
            self.last_home = gp_home;

            let (running, paused) = self
                .emulation_handle
                .as_ref()
                .map_or((false, false), |h| (h.is_running(), h.is_paused()));

            if home_edge && running && !self.modal_active() && self.rebinding.is_none() && self.rebinding_pad.is_none() {
                if paused {
                    if let Some(h) = self.emulation_handle.as_ref() {
                        h.resume();
                    }
                    self.pause_anim = None;
                    self.resume_anim = Some(std::time::Instant::now());
                } else if self.game_display().is_some() {
                    if let Some(h) = self.emulation_handle.as_ref() {
                        h.pause();
                    }
                    if self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel {
                        self.app_settings.view_mode = crate::app_settings::ViewMode::Carousel;
                        let _ = self.app_settings.save();
                    }
                    if let Some(pp) = self.playing_path.clone() {
                        if let Some(idx) = self.library.index_of_path(&pp) {
                            let n = self.library.move_to_front(idx);
                            self.carousel.selected = n + crate::carousel::CS_FRONT;
                            self.carousel.scroll_offset = (n + crate::carousel::CS_FRONT) as f32;
                        }
                    }
                    self.carousel.active_dock = false;
                    self.pause_anim = Some(std::time::Instant::now());
                } else {
                    self.stop_emulation();
                    self.teardown_at = Some(std::time::Instant::now());
                    self.pending_boot = None;
                }
            }
        }

        if self.rebinding.is_none() && self.rebinding_pad.is_none() {
            let pressed: Vec<String> = ctx.input(|i| {
                let mut v = Vec::new();
                for ev in &i.events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = ev
                    {
                        v.push(format!("{:?}", key));
                    }
                }
                for k in [
                    egui::Key::A,
                    egui::Key::B,
                    egui::Key::C,
                    egui::Key::D,
                    egui::Key::E,
                    egui::Key::F,
                    egui::Key::G,
                    egui::Key::H,
                    egui::Key::I,
                    egui::Key::J,
                    egui::Key::K,
                    egui::Key::L,
                    egui::Key::M,
                    egui::Key::N,
                    egui::Key::O,
                    egui::Key::P,
                    egui::Key::Q,
                    egui::Key::R,
                    egui::Key::S,
                    egui::Key::T,
                    egui::Key::U,
                    egui::Key::V,
                    egui::Key::W,
                    egui::Key::X,
                    egui::Key::Y,
                    egui::Key::Z,
                    egui::Key::Num0,
                    egui::Key::Num1,
                    egui::Key::Num2,
                    egui::Key::Num3,
                    egui::Key::Num4,
                    egui::Key::Num5,
                    egui::Key::Num6,
                    egui::Key::Num7,
                    egui::Key::Num8,
                    egui::Key::Num9,
                    egui::Key::ArrowUp,
                    egui::Key::ArrowDown,
                    egui::Key::ArrowLeft,
                    egui::Key::ArrowRight,
                    egui::Key::Enter,
                    egui::Key::Tab,
                    egui::Key::Space,
                    egui::Key::Escape,
                ] {
                    if i.key_down(k) {
                        let s = format!("{:?}", k);
                        if !v.contains(&s) {
                            v.push(s);
                        }
                    }
                }
                v
            });

            let (kb_buttons, kb_sticks) = self.controller_config.buttons_pressed(&pressed);
            let (gp_buttons, gp_sticks) = if self.last_input.connected {
                self.last_input.to_npad()
            } else {
                (0u64, [0i32; 4])
            };
            let mut buttons = kb_buttons | gp_buttons;
            if std::env::var_os("NEXIUM_AUTO_PRESS_A_MS").is_some()
                || std::env::var_os("NEXIUM_AUTO_PRESS_SEQUENCE").is_some()
            {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed_ms = START
                    .get_or_init(std::time::Instant::now)
                    .elapsed()
                    .as_millis() as u64;
                let mut auto_buttons = 0u64;
                if let Ok(sequence) = std::env::var("NEXIUM_AUTO_PRESS_SEQUENCE") {
                    for step in sequence.split(',') {
                        let mut parts = step.split(':');
                        let Some(delay_ms) = parts.next().and_then(parse_u64_value) else {
                            continue;
                        };
                        let len_ms = parts.next().and_then(parse_u64_value).unwrap_or(2000);
                        let mask = parts
                            .next()
                            .and_then(parse_u64_value)
                            .unwrap_or(nexium_core::hid_state::NPAD_BUTTON_A);
                        if elapsed_ms >= delay_ms && elapsed_ms < delay_ms.saturating_add(len_ms) {
                            auto_buttons |= mask;
                        }
                    }
                } else if let Some(delay_ms) = std::env::var("NEXIUM_AUTO_PRESS_A_MS")
                    .ok()
                    .and_then(|s| parse_u64_value(&s))
                {
                    let len_ms = std::env::var("NEXIUM_AUTO_PRESS_A_LEN_MS")
                        .ok()
                        .and_then(|s| parse_u64_value(&s))
                        .unwrap_or(2000);
                    let mask = std::env::var("NEXIUM_AUTO_PRESS_MASK")
                        .ok()
                        .and_then(|s| parse_u64_value(&s))
                        .unwrap_or(nexium_core::hid_state::NPAD_BUTTON_A);
                    if elapsed_ms >= delay_ms && elapsed_ms < delay_ms.saturating_add(len_ms) {
                        auto_buttons |= mask;
                    }
                }
                buttons |= auto_buttons;
                static LAST_AUTO_BUTTONS: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(u64::MAX);
                let last =
                    LAST_AUTO_BUTTONS.swap(auto_buttons, std::sync::atomic::Ordering::Relaxed);
                if last != auto_buttons {
                    log::info!(
                        "auto input elapsed_ms={} buttons={:#x}",
                        elapsed_ms,
                        auto_buttons
                    );
                }
            }
            let mut sticks = kb_sticks;
            for i in 0..4 {
                if gp_sticks[i].abs() > sticks[i].abs() {
                    sticks[i] = gp_sticks[i];
                }
            }
            if buttons != self.last_buttons_logged || sticks != self.last_sticks_logged {
                self.last_buttons_logged = buttons;
                self.last_sticks_logged = sticks;
                log::info!(
                    "input change: pressed={:?} buttons={:#x} sticks={:?}",
                    pressed,
                    buttons,
                    sticks
                );
            }
            let state = nexium_core::hid_state::get_hid_state();
            let mut hid = state.lock();
            hid.update_input(nexium_core::hid_state::ControllerInput {
                buttons,
                stick_l_x: sticks[0],
                stick_l_y: sticks[1],
                stick_r_x: sticks[2],
                stick_r_y: sticks[3],
            });
        }

        self.poll_frames(ctx);

        if self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| !h.is_running())
        {
            self.emulation_handle = None;
            self.pause_anim = None;
            if self.stop_anim.is_none() && self.pill_fade.is_none() {
                self.playing_path = None;
            }
        }

        let fps = self.performance.get_fps();
        let running = self.is_running();
        let paused = self
            .emulation_handle
            .as_ref()
            .map_or(false, |h| h.is_paused());

        if running && !paused {
            let dt = ctx.input(|i| i.stable_dt).min(0.5) as f64;
            if let Some(p) = self.playing_path.clone() {
                self.play_times.add(&p, dt);
            }
        }
        if self.last_playtime_save.elapsed() >= std::time::Duration::from_secs(20) {
            self.play_times.save_if_dirty();
            self.last_playtime_save = std::time::Instant::now();
        }

        let playing_alpha = if let Some(start) = self.pill_fade {
            let f = start.elapsed().as_secs_f32() / 0.40;
            if f >= 1.0 {
                self.pill_fade = None;
                self.playing_path = None;
                0.0
            } else {
                1.0 - f * f
            }
        } else {
            1.0
        };
        let playing_index = self
            .playing_path
            .as_ref()
            .and_then(|p| self.library.index_of_path(p));

        {
            let ptarget = if self.show_profile { 1.0 } else { 0.0 };
            let pdt = ctx.input(|i| i.stable_dt).min(0.1);
            let speed = 3.6;
            if self.profile_anim < ptarget {
                self.profile_anim = (self.profile_anim + pdt * speed).min(ptarget);
            } else if self.profile_anim > ptarget {
                self.profile_anim = (self.profile_anim - pdt * speed).max(ptarget);
            }
        }
        let profile_showing = self.show_profile || self.profile_anim > 0.0;

        let carousel_mode = self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel
            && (!running
                || self.game_display().is_none()
                || paused
                || self.stop_anim.is_some()
                || self.pill_fade.is_some()
                || self.resume_anim.is_some());

        let show_chrome =
            self.app_settings.view_mode != crate::app_settings::ViewMode::Carousel;

        if show_chrome {
            egui::TopBottomPanel::top("topbar")
                .exact_height(36.0)
                .frame(
                    egui::Frame::none()
                        .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                        .stroke(Stroke::new(1.0_f32, BORDER)),
                )
                .show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(12.0);

                        ui.label(
                            egui::RichText::new("NeXium")
                                .size(13.5)
                                .strong()
                                .color(TEXT),
                        );

                        ui.add_space(8.0);
                        ui.painter().vline(
                            ui.cursor().left(),
                            ui.max_rect().y_range(),
                            Stroke::new(1.0_f32, BORDER),
                        );
                        ui.add_space(8.0);

                        ui.menu_button(egui::RichText::new("File").size(13.0).color(TEXT), |ui| {
                            if ui.button("Open game…").clicked() {
                                if let Some(p) = rfd::FileDialog::new()
                                    .add_filter("Switch games", &["nro", "dxci", "dnsp"])
                                    .pick_file()
                                {
                                    self.nro_path = p.to_string_lossy().to_string();
                                }
                                ui.close_menu();
                            }
                            ui.separator();
                            if ui.button("Exit").clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        });
                        ui.menu_button(
                            egui::RichText::new("Emulation").size(13.0).color(TEXT),
                            |ui| {
                                ui.set_min_width(110.0);
                                if running {
                                    let label = if paused { "Resume" } else { "Pause" };
                                    if ui.button(label).clicked() {
                                        if let Some(h) = self.emulation_handle.as_ref() {
                                            if paused {
                                                h.resume();
                                            } else {
                                                h.pause();
                                            }
                                        }
                                        self.pause_anim = None;
                                        self.resume_anim = None;
                                        ui.close_menu();
                                    }
                                    ui.separator();
                                }
                                if ui.button("Boot").clicked() {
                                    self.boot_nro(ctx);
                                    ui.close_menu();
                                }
                                if ui.button("Stop").clicked() {
                                    self.stop_emulation();
                                    ui.close_menu();
                                }
                            },
                        );
                        ui.menu_button(egui::RichText::new("Debug").size(13.0).color(TEXT), |ui| {
                            if ui.button("Memory").clicked() {
                                self.debugger.toggle_memory();
                                ui.close_menu();
                            }
                            if ui.button("Registers").clicked() {
                                self.debugger.toggle_registers();
                                ui.close_menu();
                            }
                            if ui.button("Disassembler").clicked() {
                                self.debugger.toggle_disasm();
                                ui.close_menu();
                            }
                            if ui.button("Logs").clicked() {
                                self.debugger.toggle_logs();
                                ui.close_menu();
                            }
                            ui.separator();
                            let mut dumps_on = nexium_common::dumps::enabled();
                            if ui.checkbox(&mut dumps_on, "Frame dumps (.bmp)").changed() {
                                nexium_common::dumps::set_enabled(dumps_on);
                            }
                        });
                        ui.menu_button(
                            egui::RichText::new("Settings").size(13.0).color(TEXT),
                            |ui| {
                                if ui.button("Preferences").clicked() {
                                    self.show_settings = !self.show_settings;
                                    ui.close_menu();
                                }
                            },
                        );

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(12.0);
                            let (fps_col, fps_str) = if fps >= 55.0 {
                                (GREEN, format!("{:.0} fps", fps))
                            } else if fps >= 28.0 {
                                (AMBER, format!("{:.0} fps", fps))
                            } else {
                                (DANGER, format!("{:.0} fps", fps))
                            };
                            ui.label(
                                egui::RichText::new(fps_str)
                                    .size(12.0)
                                    .color(fps_col)
                                    .monospace(),
                            );
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                            ui.add_space(6.0);
                            let paused = false;
                            let (run_icon, status, status_col) = if paused {
                                (StatusIcon::Pause, "Paused", AMBER)
                            } else if running {
                                (StatusIcon::Play, "Running", GREEN)
                            } else {
                                (StatusIcon::Stop, "Idle", MUTED)
                            };
                            ui.label(
                                egui::RichText::new(status)
                                    .size(12.0)
                                    .color(status_col),
                            );
                            ui.add_space(2.0);
                            status_icon(ui, run_icon, status_col);
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                            ui.add_space(6.0);
                            let docked = nexium_core::hid_state::is_docked();
                            let (mode_icon, mode_txt, mode_col) = if docked {
                                (StatusIcon::Dock, "Docked", GREEN)
                            } else {
                                (StatusIcon::Handheld, "Handheld", AMBER)
                            };
                            let label_resp = ui.add(
                                egui::Label::new(
                                    egui::RichText::new(mode_txt).size(12.0).color(mode_col),
                                )
                                .sense(egui::Sense::click()),
                            );
                            ui.add_space(2.0);
                            let icon_resp = status_icon(ui, mode_icon, mode_col);
                            let mode_resp = label_resp.union(icon_resp);
                            if mode_resp.hovered() {
                                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            if mode_resp
                                .on_hover_text("Toggle Docked / Handheld (Pro Controller vs Handheld)")
                                .clicked()
                            {
                                nexium_core::hid_state::set_docked(!docked);
                            }
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("·").color(MUTED).size(12.0));
                            ui.add_space(6.0);
                            if pill_button(ui, "⊞ Carousel", false).clicked() {
                                self.app_settings.view_mode = crate::app_settings::ViewMode::Carousel;
                                let _ = self.app_settings.save();
                            }
                        });
                    });
                });
        }

        if show_chrome {
            egui::TopBottomPanel::bottom("statusbar")
                .exact_height(22.0)
                .frame(
                    egui::Frame::none()
                        .fill(Color32::from_rgb(0x0C, 0x0C, 0x0E))
                        .stroke(Stroke::new(1.0_f32, BORDER)),
                )
                .show(ctx, |ui| {
                    ui.horizontal_centered(|ui| {
                        ui.add_space(12.0);
                        let name = std::path::Path::new(&self.nro_path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| "No file".into());
                        ui.label(egui::RichText::new(name).size(11.0).color(MUTED));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(12.0);
                            let stats = self
                                .emulation_handle
                                .as_ref()
                                .map(|h| h.stats.lock().clone())
                                .unwrap_or_default();
                            let building = nexium_common::shader_progress::in_flight();
                            let building_prefix = if building > 0 {
                                format!("Building Shader(s): {}  ·  ", building)
                            } else if nexium_common::shader_progress::recently_active() {
                                format!(
                                    "Built {} Shader(s)  ·  ",
                                    nexium_common::shader_progress::burst_built()
                                )
                            } else {
                                String::new()
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{}Frame {:.1}ms  ·  SVCs {}  ·  Cycles {}",
                                    building_prefix,
                                    self.performance.get_frame_time(),
                                    stats.svc_count,
                                    stats.cycle_count,
                                ))
                                .size(11.0)
                                .color(MUTED)
                                .monospace(),
                            );
                        });
                    });
                });
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(BG))
            .show(ctx, |ui| {
                let full_rect = ui.max_rect();
                if profile_showing {
                    let avatar_tex = self.profile_texture.as_ref().map(|t| t.id());
                    let ambient = self.carousel.ambient_color;
                    let accent = match self.app_settings.carousel_theme.color() {
                        Some((r, g, b)) => Color32::from_rgb(r, g, b),
                        None => self.carousel.ambient_color,
                    };

                    let smooth = |x: f32| {
                        let x = x.clamp(0.0, 1.0);
                        x * x * (3.0 - 2.0 * x)
                    };
                    let p = self.profile_anim.clamp(0.0, 1.0);
                    let backdrop_split = 0.4;
                    let backdrop_opacity = smooth(p / backdrop_split);
                    let content_opacity = smooth((p - backdrop_split) / (1.0 - backdrop_split));

                    let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                    let carousel_scale = 1.0;
                    let carousel_alpha = 1.0;
                    let _ = crate::carousel::carousel_view(
                        &mut self.carousel,
                        &mut self.library,
                        ctx,
                        ui,
                        &self.last_input,
                        &mut self.input,
                        running,
                        playing_index,
                        playing_alpha,
                        self.app_settings.carousel_theme,
                        profile_tex,
                        &self.app_settings.profile_name,
                        false,
                        carousel_scale,
                        carousel_alpha,
                        self.app_settings.backdrop_theme,
                        self.app_settings.light_mode,
                        &self.app_settings.favorites,
                        self.app_settings.eu_dates,
                        self.app_settings.dockbar_theme,
                        &self.app_settings.carousel_order,
                        &self.app_settings.carousel_lists,
                        self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                    );

                    let profile_scale = 0.92 + 0.08 * content_opacity;
                    let profile_active = !self.modal_active() && !self.modal_active_frame_start;
                    let action = crate::profile::profile_view(
                        &mut self.profile,
                        &mut self.library,
                        &self.play_times,
                        ctx,
                        ui,
                        full_rect,
                        &self.app_settings.profile_name,
                        avatar_tex,
                        ambient,
                        accent,
                        content_opacity,
                        backdrop_opacity,
                        profile_scale,
                        self.app_settings.backdrop_theme,
                        self.app_settings.dockbar_theme,
                        self.app_settings.light_mode,
                        self.app_settings.music_volume,
                        self.app_settings.sfx_volume,
                        self.app_settings.eu_dates,
                        self.app_settings.music_muted,
                        self.app_settings.sfx_muted,
                        profile_active,
                        &self.last_input,
                        &mut self.input,
                    );
                    if self.show_profile {
                        match action {
                            crate::profile::ProfileAction::Close => {
                                self.show_profile = false;
                                self.carousel.profile_focused = false;
                            }
                            crate::profile::ProfileAction::QuickLaunch(path) => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                self.confirm = Some(ConfirmDialog {
                                    title: "Quick Launch".into(),
                                    body: "Quick Launch this game? Any open game will be shut down first.".into(),
                                    confirm_label: "Launch".into(),
                                    selected: 0,
                                    kind: ConfirmKind::QuickLaunch(path),
                                });
                                self.modal_hold = true;
                            }
                            crate::profile::ProfileAction::PickIcon => {
                                self.pick_profile_avatar();
                            }
                            crate::profile::ProfileAction::SetName(name) => {
                                self.app_settings.profile_name = name;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetBackdropTheme(theme) => {
                                if self.app_settings.backdrop_theme != theme {
                                    self.app_settings.backdrop_theme = theme;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetDockbarTheme(theme) => {
                                if self.app_settings.dockbar_theme != theme {
                                    self.app_settings.dockbar_theme = theme;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetLightMode(on) => {
                                if self.app_settings.light_mode != on {
                                    self.app_settings.light_mode = on;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::profile::ProfileAction::SetMusicVolume(v) => {
                                self.app_settings.music_volume = v.clamp(0.0, 1.0);
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetSfxVolume(v) => {
                                self.app_settings.sfx_volume = v.clamp(0.0, 1.0);
                                if !self.app_settings.sfx_muted {
                                    crate::ui_audio::set_sfx_volume(self.app_settings.sfx_volume);
                                }
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetEuDates(on) => {
                                self.app_settings.eu_dates = on;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetMuteMusic(on) => {
                                self.app_settings.music_muted = on;
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::SetMuteSfx(on) => {
                                self.app_settings.sfx_muted = on;
                                crate::ui_audio::set_sfx_volume(if on { 0.0 } else { self.app_settings.sfx_volume });
                                let _ = self.app_settings.save();
                            }
                            crate::profile::ProfileAction::None => {}
                        }
                    }
                    ctx.request_repaint();
                } else if carousel_mode {
                    let booting = running && !paused && self.game_display().is_none();
                    if booting {
                        let bg_rect = ui.max_rect();
                        let painter = ui.painter();
                        let t = ui.input(|i| i.time) as f32;

                        painter.rect_filled(bg_rect, Rounding::ZERO, Color32::from_rgb(0x06, 0x06, 0x08));

                        let launched = std::path::PathBuf::from(&self.nro_path);
                        let game_index = self.library.index_of_path(&launched);
                        let game_info = game_index.and_then(|gi| self.library.games.get(gi)).map(|g| {
                            (g.title.clone(), g.format.to_string(), g.dominant_color)
                        });
                        let dom_color = game_info.as_ref().map(|(_, _, col)| *col).unwrap_or(Color32::from_rgb(0x2F, 0xB4, 0xEF));

                        let elapsed_awaiting = if let crate::carousel::BootStage::AwaitingFrame { start_time, .. } = self.carousel.boot_stage {
                            (t - start_time).max(0.0)
                        } else {
                            0.0
                        };
                        let fade_alpha = (elapsed_awaiting / 0.55).min(1.0); // 550ms fade-in

                        let glow_intensity = (((t * 2.2).sin() * 0.12 + 0.18).clamp(0.0, 1.0)) * fade_alpha;
                        let center = bg_rect.center();
                        for r_offset in (0..12).rev() {
                            let r = 140.0 + r_offset as f32 * 12.0;
                            let alpha = (4.0 * (1.0 - r_offset as f32 / 12.0) * glow_intensity * fade_alpha) as u8;
                            let col = Color32::from_rgba_unmultiplied(dom_color.r(), dom_color.g(), dom_color.b(), alpha);
                            painter.circle_filled(center - Vec2::new(0.0, 48.0), r, col);
                        }

                        let card_sz = 260.0f32;
                        let card_rect = egui::Rect::from_center_size(center - Vec2::new(0.0, 48.0), Vec2::splat(card_sz));

                        crate::carousel::draw_gradient_rounded_rect(&painter, card_rect.center(), card_rect.expand(6.0), 16.0, t, (120.0 * fade_alpha) as u8);
                        crate::carousel::draw_gradient_rounded_rect(&painter, card_rect.center(), card_rect.expand(2.5), 14.0, t, (255.0 * fade_alpha) as u8);

                        painter.rect_filled(card_rect, Rounding::same(14.0), Color32::from_rgba_unmultiplied(0x14, 0x14, 0x1A, (fade_alpha * 255.0) as u8));
                        let card_tint = Color32::from_white_alpha((fade_alpha * 255.0) as u8);
                        if let Some(tex) = game_index.and_then(|gi| self.library.texture(ctx, gi)) {
                            crate::carousel::draw_rounded_image(&painter, tex.id(), card_rect, 14.0, card_tint);
                        } else if let Some((_, format, _)) = &game_info {
                            painter.text(card_rect.center(), egui::Align2::CENTER_CENTER, format, FontId::proportional(card_sz * 0.13), Color32::from_rgba_unmultiplied(0x70, 0x70, 0x80, (fade_alpha * 255.0) as u8));
                        }

                        if let Some((title, _, _)) = &game_info {
                            let text_y = center.y + card_sz * 0.5 + 24.0;
                            let title_alpha = (fade_alpha * 255.0) as u8;
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y), egui::Align2::CENTER_CENTER, title, FontId::proportional(28.0), Color32::from_rgba_unmultiplied(255, 255, 255, title_alpha), true);

                            let dots = match (t as u64) % 4 {
                                0 => "",
                                1 => ".",
                                2 => "..",
                                3 => "...",
                                _ => "...",
                            };
                            let await_text = format!("Awaiting First Frame{}", dots);
                            let await_alpha = (fade_alpha * 255.0) as u8;
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y + 40.0), egui::Align2::CENTER_CENTER, &await_text, FontId::proportional(15.0), Color32::from_rgba_unmultiplied(0x00, 0xE5, 0xFF, await_alpha), false);
                            crate::carousel::shadowed_text(&painter, egui::pos2(center.x, text_y + 74.0), egui::Align2::CENTER_CENTER, "[Home / `] Cancel", FontId::proportional(13.0), Color32::from_rgba_unmultiplied(0xC0, 0xC0, 0xCC, await_alpha), false);
                        }

                        ctx.request_repaint();
                    } else {
                        let panel = ui.max_rect();
                        let scale_f = if let Some(start) = self.pause_anim.or(self.stop_anim) {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            let e = 1.0 - (1.0 - f) * (1.0 - f);
                            0.90 + 0.10 * e
                        } else {
                            1.0
                        };
                        let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                        let modal_active = self.modal_active();
                        let action = crate::carousel::carousel_view(
                            &mut self.carousel,
                            &mut self.library,
                            ctx,
                            ui,
                            &self.last_input,
                            &mut self.input,
                            running,
                            playing_index,
                            playing_alpha,
                            self.app_settings.carousel_theme,
                            profile_tex,
                            &self.app_settings.profile_name,
                            !self.show_settings && !modal_active,
                            scale_f,
                            1.0,
                            self.app_settings.backdrop_theme,
                            self.app_settings.light_mode,
                            &self.app_settings.favorites,
                            self.app_settings.eu_dates,
                            self.app_settings.dockbar_theme,
                            &self.app_settings.carousel_order,
                            &self.app_settings.carousel_lists,
                            self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                        );
                        match action {
                            crate::carousel::CarouselAction::Launch(path) => {
                                if running {
                                    crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                    self.confirm = Some(ConfirmDialog {
                                        title: "Launch Game".into(),
                                        body: "Close the current game and launch this one?".into(),
                                        confirm_label: "Launch".into(),
                                        selected: 0,
                                        kind: ConfirmKind::LaunchGame(path),
                                    });
                                    self.modal_hold = true;
                                } else if crate::boot::emu_alive() {
                                    self.teardown_at = Some(std::time::Instant::now());
                                    self.pending_boot = Some(path);
                                } else {
                                    self.nro_path = path;
                                    self.boot_nro(ctx);
                                }
                            }
                            crate::carousel::CarouselAction::SetTheme(th) => {
                                if self.app_settings.carousel_theme != th {
                                    self.app_settings.carousel_theme = th;
                                    let _ = self.app_settings.save();
                                }
                            }
                            crate::carousel::CarouselAction::Resume => {
                                if let Some(h) = self.emulation_handle.as_ref() {
                                    h.resume();
                                }
                                self.pause_anim = None;
                                self.resume_anim = Some(std::time::Instant::now());
                            }
                            crate::carousel::CarouselAction::OpenProfile => {
                                self.profile.tab = crate::profile::ProfileTab::Profile;
                                self.profile.name_buf = self.app_settings.profile_name.clone();
                                self.profile.focus_content = false;
                                self.profile.row_selected = 0;
                                self.show_profile = true;
                            }
                            crate::carousel::CarouselAction::SwitchToGrid => {
                                self.app_settings.view_mode = crate::app_settings::ViewMode::Grid;
                                let _ = self.app_settings.save();
                            }
                            crate::carousel::CarouselAction::StopEmulation => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                self.confirm = Some(ConfirmDialog {
                                    title: "Close Game".into(),
                                    body: "Close the current game?".into(),
                                    confirm_label: "Close".into(),
                                    selected: 0,
                                    kind: ConfirmKind::CloseGame,
                                });
                                self.modal_hold = true;
                            }
                            crate::carousel::CarouselAction::OpenSettings => {
                                self.settings_tab = SettingsTab::General;
                                self.show_settings = true;
                            }
                            crate::carousel::CarouselAction::OpenController => {
                                self.settings_tab = SettingsTab::Controller;
                                self.show_settings = true;
                            }
                            crate::carousel::CarouselAction::OpenDebug => {
                                self.debugger.toggle_logs();
                            }
                            crate::carousel::CarouselAction::Quit => {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                            crate::carousel::CarouselAction::Rescan => {
                                self.library.rescan(ctx, &self.app_settings.library_folders);
                            }
                            crate::carousel::CarouselAction::AddFolder => {
                                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                                    if !self.app_settings.library_folders.contains(&dir) {
                                        self.app_settings.library_folders.push(dir);
                                        let _ = self.app_settings.save();
                                    }
                                    self.library.rescan(ctx, &self.app_settings.library_folders);
                                }
                            }
                            crate::carousel::CarouselAction::ToggleFavorite(path) => {
                                crate::ui_audio::play(crate::ui_audio::Sfx::Favorite);
                                let p = std::path::PathBuf::from(path);
                                if let Some(pos) = self.app_settings.favorites.iter().position(|x| *x == p) {
                                    self.app_settings.favorites.remove(pos);
                                } else {
                                    self.app_settings.favorites.push(p);
                                }
                                let _ = self.app_settings.save();
                            }
                            crate::carousel::CarouselAction::DownloadIcon(path) => {
                                let pb = std::path::PathBuf::from(&path);
                                if let Some(idx) = self.library.index_of_path(&pb) {
                                    let title = self.library.games[idx].title.clone();
                                    crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                                    self.icon_picker = Some(IconPicker {
                                        game_idx: idx,
                                        game_path: pb,
                                        search: title.clone(),
                                        editing: false,
                                        anim: 0.0,
                                        selected: 0,
                                        scroll: 0.0,
                                        follow_sel: true,
                                        squish_at: None,
                                        nav_cd: 0.0,
                                        hold: true,
                                        built: false,
                                        full_urls: Vec::new(),
                                        thumbs: Vec::new(),
                                        fetch: start_icon_fetch(
                                            self.app_settings.steamgriddb_key.clone(),
                                            title,
                                        ),
                                        apply: None,
                                    });
                                }
                            }
                            crate::carousel::CarouselAction::OpenShop => {
                                self.shop.open();
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                            }
                            crate::carousel::CarouselAction::OpenCarouselSettings => {
                                self.carousel_settings_open = true;
                                crate::ui_audio::play(crate::ui_audio::Sfx::Open);
                            }
                            crate::carousel::CarouselAction::None => {}
                        }

                        if let Some(start) = self.pause_anim.or(self.stop_anim) {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            if f >= 1.0 {
                                self.pause_anim = None;
                                if self.stop_anim.take().is_some() {
                                    self.game_texture = None;
                                    if self.playing_path.is_some() {
                                        self.pill_fade = Some(std::time::Instant::now());
                                    }
                                }
                            } else if let Some((tid, tsz)) = self.game_display() {
                                let ef = 1.0 - (1.0 - f) * (1.0 - f);
                                let rect = self.game_draw_rect(panel, tsz);
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("home_zoom"),
                                ));
                                draw_zoom_fade_out(&p, rect, tid, ef);
                            }
                            ctx.request_repaint();
                        } else if let Some(start) = self.resume_anim {
                            let f = (start.elapsed().as_secs_f32() / 0.30).min(1.0);
                            if let Some((tid, tsz)) = self.game_display() {
                                let ef = 1.0 - (1.0 - f) * (1.0 - f);
                                let rect = self.game_draw_rect(panel, tsz);
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("home_zoom"),
                                ));
                                let cover = (ef * 1.5).min(1.0);
                                p.rect_filled(
                                    panel,
                                    Rounding::ZERO,
                                    Color32::from_rgba_unmultiplied(BG.r(), BG.g(), BG.b(), (cover * 255.0) as u8),
                                );
                                draw_zoom_fade_in(&p, rect, tid, ef);
                            }
                            if f >= 1.0 {
                                self.resume_anim = None;
                            }
                            ctx.request_repaint();
                        }
                    }
                } else if let Some(start) = self.stop_fade {
                    let t = start.elapsed().as_secs_f32() / 0.9;
                    let panel = ui.max_rect();
                    if self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel {
                        let scale_f = 0.92 + 0.08 * t.min(1.0);
                        let profile_tex = self.profile_texture.as_ref().map(|t| t.id());
                        let _ = crate::carousel::carousel_view(
                            &mut self.carousel,
                            &mut self.library,
                            ctx,
                            ui,
                            &self.last_input,
                            &mut self.input,
                            running,
                            playing_index,
                            playing_alpha,
                            self.app_settings.carousel_theme,
                            profile_tex,
                            &self.app_settings.profile_name,
                            true,
                            scale_f,
                            t.min(1.0),
                            self.app_settings.backdrop_theme,
                            self.app_settings.light_mode,
                            &self.app_settings.favorites,
                            self.app_settings.eu_dates,
                            self.app_settings.dockbar_theme,
                            &self.app_settings.carousel_order,
                            &self.app_settings.carousel_lists,
                            self.icon_reveal.as_ref().map(|(gi, tm, tx)| (*gi, tm.elapsed().as_secs_f32(), tx.clone())),
                        );
                        if t < 1.05 {
                            if let Some((tid, tsz)) = self.game_display() {
                                let rect = self.game_draw_rect(panel, tsz);
                                let p = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("stop_dissolve"),
                                ));
                                draw_zoom_fade_out(&p, rect, tid, t);
                            }
                        }
                    } else {
                        let display = self.game_display();
                        self.library_view(ui, ctx);
                        if let Some((tid, tsz)) = display {
                            let rect = self.game_draw_rect(panel, tsz);
                            let p = ctx.layer_painter(egui::LayerId::new(
                                egui::Order::Foreground,
                                egui::Id::new("stop_dissolve"),
                            ));
                            draw_dissolve(&p, rect, tid, t);
                        }
                    }
                    ctx.request_repaint();
                    if t >= 1.05 {
                        self.stop_fade = None;
                        self.game_texture = None;
                        self.free_native_texture();
                    }
                } else if running {
                    if let Some((tid, tsz)) = self.game_display() {
                        let draw_size = self.game_draw_rect(ui.max_rect(), tsz).size();
                        ui.centered_and_justified(|ui| {
                            ui.image((tid, draw_size));
                        });
                    } else {
                        self.library_view(ui, ctx);
                    }
                } else {
                    if self.game_texture.is_some() || self.game_texture_native.is_some() {
                        self.game_texture = None;
                        self.free_native_texture();
                    }
                    self.library_view(ui, ctx);
                }
                self.draw_modal(ctx, ui);
                self.update_icon_picker(ctx, ui);
                if running
                    && !carousel_mode
                    && self.game_display().is_some()
                    && self.app_settings.view_mode == crate::app_settings::ViewMode::Carousel
                {
                    self.draw_perf_overlay(ctx, ui);
                }
                {
                    let accent = self.theme_accent();
                    self.shop.update(ctx, ui, self.app_settings.light_mode, accent, &self.last_input);
                }
                if let Some((title, since)) = &self.download_toast {
                    let el = since.elapsed().as_secs_f32();
                    if el > 4.5 {
                        self.download_toast = None;
                    } else {
                        let fade = (el.min(0.3) / 0.3).min((4.5 - el) / 0.5).clamp(0.0, 1.0);
                        let screen = ctx.screen_rect();
                        let mut tp = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("dl_toast")));
                        tp.set_opacity(fade);
                        let w = 360.0;
                        let rect = egui::Rect::from_min_size(egui::pos2(screen.center().x - w * 0.5, screen.min.y + 24.0), egui::Vec2::new(w, 62.0));
                        tp.rect_filled(rect.translate(egui::Vec2::new(0.0, 4.0)), egui::Rounding::same(14.0), egui::Color32::from_black_alpha(90));
                        tp.rect_filled(rect, egui::Rounding::same(14.0), egui::Color32::from_rgb(0x1C, 0x24, 0x1E));
                        tp.rect_stroke(rect, egui::Rounding::same(14.0), egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(0x35, 0xD0, 0x6A)));
                        let cc = egui::pos2(rect.min.x + 30.0, rect.center().y);
                        tp.circle_filled(cc, 12.0, egui::Color32::from_rgb(0x35, 0xD0, 0x6A));
                        tp.add(egui::Shape::line(vec![cc + egui::Vec2::new(-5.0, 0.0), cc + egui::Vec2::new(-1.5, 4.0), cc + egui::Vec2::new(6.0, -5.0)], egui::Stroke::new(2.2_f32, egui::Color32::WHITE)));
                        tp.text(egui::pos2(rect.min.x + 54.0, rect.center().y - 9.0), egui::Align2::LEFT_CENTER, "Download Complete!", egui::FontId::proportional(16.0), egui::Color32::WHITE);
                        tp.text(egui::pos2(rect.min.x + 54.0, rect.center().y + 11.0), egui::Align2::LEFT_CENTER, title, egui::FontId::proportional(13.0), egui::Color32::from_gray(0xB0));
                        ctx.request_repaint();
                    }
                }
                self.update_carousel_settings(ctx, ui);
            });

        if self.show_settings {
            let mut open = self.show_settings;
            let mut tab = self.settings_tab;
            let mut cfg = self.controller_config.clone();
            let mut rebinding = self.rebinding;
            let mut save_needed = false;
            let mut app_cfg = self.app_settings.clone();
            let mut app_save_needed = false;
            let mut test_key: Option<String> = None;
            let last_input = self.last_input;
            let gp_name = self.input.as_ref().and_then(|ib| ib.name());
            let mut rebinding_pad = self.rebinding_pad;
            let mut input_device = self.input_device;

            let screen = ctx.screen_rect();
            let max_h = (screen.height() - 80.0).clamp(360.0, 760.0);
            let max_w = (screen.width() - 80.0).clamp(520.0, 980.0);
            egui::Window::new("Preferences")
                .open(&mut open)
                .resizable(true)
                .default_size([max_w.min(900.0), max_h.min(560.0)])
                .max_height(max_h)
                .max_width(max_w)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(tab == SettingsTab::General, "General")
                            .clicked()
                        {
                            tab = SettingsTab::General;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Controller, "Controller")
                            .clicked()
                        {
                            tab = SettingsTab::Controller;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Graphics, "Graphics")
                            .clicked()
                        {
                            tab = SettingsTab::Graphics;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Audio, "Audio")
                            .clicked()
                        {
                            tab = SettingsTab::Audio;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Emulation, "Emulation")
                            .clicked()
                        {
                            tab = SettingsTab::Emulation;
                        }
                        if ui
                            .selectable_label(tab == SettingsTab::Logging, "Logging")
                            .clicked()
                        {
                            tab = SettingsTab::Logging;
                        }
                    });
                    ui.separator();
                    ui.add_space(6.0);

                    match tab {
                        SettingsTab::General => {
                            settings_content(ui);
                            ui.add_space(10.0);
                            ui.separator();
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("SteamGridDB").strong());
                            ui.label(
                                egui::RichText::new("Paste your API key to download custom game icons.")
                                    .weak()
                                    .size(12.0),
                            );
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("API Key");
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut app_cfg.steamgriddb_key)
                                        .password(true)
                                        .hint_text("Paste key here")
                                        .desired_width(260.0),
                                );
                                if resp.changed() {
                                    app_save_needed = true;
                                }
                                if ui.button("Test").clicked() {
                                    test_key = Some(app_cfg.steamgriddb_key.clone());
                                }
                            });
                            ui.hyperlink_to(
                                "Get a key at steamgriddb.com/profile/preferences/api",
                                "https://www.steamgriddb.com/profile/preferences/api",
                            );
                        }
                        SettingsTab::Controller => {
                            controller_settings_content(
                                ui,
                                &mut cfg,
                                &mut rebinding,
                                &mut rebinding_pad,
                                &mut save_needed,
                                &last_input,
                                gp_name.as_deref(),
                                &mut input_device,
                            );
                        }
                        SettingsTab::Graphics => {
                            graphics_settings_content(ui, &mut app_cfg, &mut app_save_needed);
                        }
                        SettingsTab::Audio => {
                            audio_settings_content(
                                ui,
                                &mut app_cfg,
                                &mut app_save_needed,
                                &mut self.audio_device_cache,
                            );
                        }
                        SettingsTab::Emulation => {
                            emulation_settings_content(ui, &mut app_cfg, &mut app_save_needed);
                        }
                        SettingsTab::Logging => {
                            logging_settings_content(ui, &mut app_cfg, &mut app_save_needed);
                        }
                    }
                });

            self.show_settings = open;
            self.settings_tab = tab;
            self.controller_config = cfg;
            self.rebinding = rebinding;
            self.rebinding_pad = rebinding_pad;
            self.input_device = input_device;
            self.app_settings = app_cfg;
            if save_needed {
                if let Err(e) = self.controller_config.save() {
                    log::warn!("Failed to save controller config: {}", e);
                }
            }
            if app_save_needed {
                if let Err(e) = self.app_settings.save() {
                    log::warn!("Failed to save app settings: {}", e);
                }
            }
            if let Some(key) = test_key {
                let state = std::sync::Arc::new(std::sync::Mutex::new(KeyTest::default()));
                self.key_test = Some(state.clone());
                self.key_test_result = None;
                let ctx2 = ctx.clone();
                std::thread::spawn(move || {
                    let ok = crate::steamgrid::verify_key(&key);
                    if let Ok(mut g) = state.lock() {
                        g.done = true;
                        g.ok = ok;
                    }
                    ctx2.request_repaint();
                });
            }
        }

        // --- Poll SteamGridDB key test & draw its result dialog ---
        if let Some(state) = &self.key_test {
            let mut finished: Option<bool> = None;
            if let Ok(g) = state.lock() {
                if g.done {
                    finished = Some(g.ok);
                }
            }
            if let Some(ok) = finished {
                self.key_test = None;
                self.key_test_result = Some((ok, std::time::Instant::now()));
                crate::ui_audio::play(if ok {
                    crate::ui_audio::Sfx::Whistle
                } else {
                    crate::ui_audio::Sfx::Error
                });
            }
        }
        if self.key_test.is_some() {
            egui::Window::new("SteamGridDB")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Testing connection…");
                    });
                    ui.add_space(4.0);
                });
            ctx.request_repaint();
        } else if let Some((ok, _)) = self.key_test_result {
            let mut still_open = true;
            egui::Window::new("SteamGridDB")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(6.0);
                    ui.vertical_centered(|ui| {
                        if ok {
                            ui.label(
                                egui::RichText::new("✔  Connected")
                                    .size(20.0)
                                    .strong()
                                    .color(Color32::from_rgb(0x35, 0xD0, 0x6A)),
                            );
                            ui.add_space(4.0);
                            ui.label("Your API key is valid. You can now download icons.");
                        } else {
                            ui.label(
                                egui::RichText::new("✖  Failed")
                                    .size(20.0)
                                    .strong()
                                    .color(Color32::from_rgb(0xE8, 0x33, 0x50)),
                            );
                            ui.add_space(4.0);
                            ui.label("Couldn't verify the key. Check the key and your connection.");
                        }
                        ui.add_space(10.0);
                        if ui.button("OK").clicked() {
                            still_open = false;
                        }
                    });
                    ui.add_space(6.0);
                });
            if !still_open {
                self.key_test_result = None;
            }
        }

        let snapshot = self.emulation_handle.as_ref().map(|h| h.snapshot());
        let mem_req_handle = self
            .emulation_handle
            .as_ref()
            .map(|h| Arc::clone(&h.mem_request));
        let mem_req: Option<Box<dyn Fn(u64)>> = mem_req_handle.map(|m| {
            Box::new(move |addr: u64| {
                *m.lock() = addr;
            }) as Box<dyn Fn(u64)>
        });
        debug_windows(
            ctx,
            &mut self.debugger,
            snapshot.as_ref(),
            mem_req.as_deref(),
            &self.log_buffer,
        );

        if running {
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

fn idle_screen(ui: &mut egui::Ui, nro_path: &str, running: bool) -> u8 {
    let mut action = 255u8;
    let avail = ui.available_size();

    ui.vertical_centered(|ui| {
        ui.add_space((avail.y * 0.24).max(40.0));

        ui.label(
            egui::RichText::new("NeXium")
                .size(36.0)
                .strong()
                .color(TEXT),
        );
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new("Nintendo Switch Emulator")
                .size(13.0)
                .color(MUTED),
        );

        ui.add_space(36.0);

        let painter = ui.painter();
        let card_w = 380.0;
        let card_rect = egui::Rect::from_center_size(
            egui::pos2(ui.max_rect().center().x, ui.cursor().top() + 80.0),
            egui::vec2(card_w, 160.0),
        );
        painter.rect(
            card_rect,
            Rounding::same(10.0),
            BG_RAISED,
            Stroke::new(1.0_f32, BORDER),
        );

        ui.allocate_ui_with_layout(
            egui::vec2(card_w, 160.0),
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                ui.add_space(20.0);

                if nro_path.is_empty() {
                    ui.label(
                        egui::RichText::new("No file selected")
                            .size(13.0)
                            .color(MUTED),
                    );
                } else {
                    let fname = std::path::Path::new(nro_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| nro_path.to_string());
                    ui.label(egui::RichText::new(&fname).size(14.0).strong().color(TEXT));
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(nro_path).size(10.5).color(MUTED));
                }

                ui.add_space(16.0);

                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    if pill_button(ui, "Select NRO…", false).clicked() {
                        action = 0;
                    }
                    ui.add_space(8.0);
                    if !nro_path.is_empty() && !running {
                        if pill_button(ui, "Boot", true).clicked() {
                            action = 1;
                        }
                    }
                    if running {
                        if pill_button(ui, "Stop", false).clicked() {
                            action = 2;
                        }
                    }
                });

                if running {
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        ui.add_space(16.0);
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(8.0, 8.0), Sense::hover());
                        ui.painter().circle_filled(rect.center(), 3.5, GREEN);
                        ui.label(
                            egui::RichText::new("Running — awaiting first frame")
                                .size(11.5)
                                .color(MUTED),
                        );
                    });
                }
            },
        );
    });

    action
}

fn settings_content(ui: &mut egui::Ui) {
    egui::Grid::new("prefs")
        .num_columns(2)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            row(ui, "CPU Backend", "Dynarmic (JIT)");
            row(ui, "GPU Backend", "Vulkan (ash)");
            row(ui, "Audio", "Enabled · 100%");
            row(ui, "Resolution", "1280 × 720");
        });
}

#[allow(clippy::too_many_arguments)]
fn controller_settings_content(
    ui: &mut egui::Ui,
    cfg: &mut ControllerConfig,
    rebinding: &mut Option<SwitchButton>,
    rebinding_pad: &mut Option<SwitchButton>,
    save_needed: &mut bool,
    input: &InputSnapshot,
    gp_name: Option<&str>,
    input_device: &mut InputDevice,
) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Player 1")
                .size(15.0)
                .strong()
                .color(TEXT),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if input.connected {
                let name = gp_name.unwrap_or("Gamepad");
                let pro = gp_name
                    .map(crate::input::is_pro_controller)
                    .unwrap_or(false);
                let tag = if pro {
                    "  ·  auto-mapped 1:1"
                } else {
                    "  ·  standard mapping"
                };
                ui.label(
                    egui::RichText::new(format!("{}{}", name, tag))
                        .size(12.0)
                        .color(GREEN),
                );
                ui.label(egui::RichText::new("●").size(13.0).color(GREEN));
            } else {
                ui.label(
                    egui::RichText::new("No gamepad detected")
                        .size(12.0)
                        .color(MUTED),
                );
                ui.label(egui::RichText::new("○").size(13.0).color(MUTED));
            }
        });
    });
    ui.label(egui::RichText::new("Switch Pro Controllers map 1:1 automatically. Other gamepads use a standard layout. Keyboard and gamepad both work at once.")
        .size(11.0).color(MUTED));
    ui.add_space(10.0);

    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            ui.set_min_width(470.0);
            ui.set_max_width(470.0);
            draw_controller_diagram(ui, input);
        });
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Input device").size(12.0).color(MUTED));
                let sel = match *input_device {
                    InputDevice::Keyboard => "Keyboard".to_string(),
                    InputDevice::Gamepad => gp_name.unwrap_or("Gamepad").to_string(),
                };
                egui::ComboBox::from_id_salt("input_device_sel")
                    .selected_text(sel)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(input_device, InputDevice::Keyboard, "Keyboard");
                        ui.selectable_value(
                            input_device,
                            InputDevice::Gamepad,
                            gp_name.unwrap_or("Gamepad"),
                        );
                    });
            });
            ui.add_space(6.0);

            if let Some(btn) = *rebinding {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Press a key for {}…", btn.display_name()))
                            .size(12.0)
                            .color(AMBER),
                    );
                    if ui.small_button("Cancel").clicked() {
                        *rebinding = None;
                    }
                });
                let new_key = ui.input(|i| {
                    for ev in &i.events {
                        if let egui::Event::Key {
                            key, pressed: true, ..
                        } = ev
                        {
                            return Some(format!("{:?}", key));
                        }
                    }
                    None
                });
                if let Some(k) = new_key {
                    cfg.set_binding(btn, k);
                    *rebinding = None;
                    *save_needed = true;
                }
            } else if let Some(btn) = *rebinding_pad {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Press a button for {}…", btn.display_name()))
                            .size(12.0)
                            .color(AMBER),
                    );
                    if ui.small_button("Cancel").clicked() {
                        *rebinding_pad = None;
                    }
                });
            } else {
                ui.label(
                    egui::RichText::new("Click Rebind, then press the key or button.")
                        .size(11.0)
                        .color(MUTED),
                );
            }
            ui.add_space(4.0);

            let list_h = (ui.available_height() - 46.0).max(160.0);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .max_height(list_h)
                .show(ui, |ui| {
                    egui::Grid::new("binds")
                        .num_columns(3)
                        .spacing([10.0, 6.0])
                        .striped(true)
                        .show(ui, |ui| match *input_device {
                            InputDevice::Keyboard => {
                                for btn in SwitchButton::all() {
                                    ui.label(
                                        egui::RichText::new(btn.display_name())
                                            .size(12.0)
                                            .color(TEXT),
                                    );
                                    let cur = cfg.binding_for(*btn).unwrap_or("—").to_string();
                                    ui.label(
                                        egui::RichText::new(cur)
                                            .size(12.0)
                                            .monospace()
                                            .color(MUTED),
                                    );
                                    if ui.small_button("Rebind").clicked() {
                                        *rebinding = Some(*btn);
                                        *rebinding_pad = None;
                                    }
                                    ui.end_row();
                                }
                            }
                            InputDevice::Gamepad => {
                                for btn in ControllerConfig::pad_list() {
                                    ui.label(
                                        egui::RichText::new(btn.display_name())
                                            .size(12.0)
                                            .color(TEXT),
                                    );
                                    let cur =
                                        cfg.pad_for(*btn).map(|g| g.display_name()).unwrap_or("—");
                                    ui.label(egui::RichText::new(cur).size(12.0).color(MUTED));
                                    if ui.small_button("Rebind").clicked() {
                                        *rebinding_pad = Some(*btn);
                                        *rebinding = None;
                                    }
                                    ui.end_row();
                                }
                            }
                        });
                });
        });
    });

    ui.add_space(8.0);
    ui.horizontal(|ui| {
        if pill_button(ui, "Reset to Defaults", false).clicked() {
            *cfg = ControllerConfig::default();
            *save_needed = true;
        }
        if pill_button(ui, "Save", true).clicked() {
            *save_needed = true;
        }
    });
}

fn draw_controller_diagram(ui: &mut egui::Ui, input: &InputSnapshot) {
    let avail = ui.available_width();
    let s = (avail / 430.0).min(0.82);
    let h = 300.0 * s;
    let (rect, _resp) = ui.allocate_exact_size(Vec2::new(avail, h), Sense::hover());
    let painter = ui.painter_at(rect);
    let c = rect.center();
    let map = |x: f32, y: f32| c + Vec2::new(x * s, (y + 6.0) * s);

    painter.rect(rect, Rounding::same(10.0), BG, Stroke::NONE);

    let outline = Color32::from_rgb(0x4C, 0x4C, 0x58);
    let body_stroke = Stroke::new((2.0 * s).max(1.2), outline);

    let n = PRO_BODY.len() / 2;
    let mut body: Vec<egui::Pos2> = Vec::with_capacity(n * 2);
    for i in 0..n {
        body.push(map(PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    for i in (0..n).rev() {
        body.push(map(-PRO_BODY[i * 2], PRO_BODY[i * 2 + 1]));
    }
    let hn = PRO_LEFT_HANDLE.len() / 2;
    let lh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    let rh: Vec<egui::Pos2> = (0..hn)
        .map(|i| map(-PRO_LEFT_HANDLE[i * 2], PRO_LEFT_HANDLE[i * 2 + 1]))
        .collect();
    painter.add(egui::Shape::closed_line(lh, body_stroke));
    painter.add(egui::Shape::closed_line(rh, body_stroke));
    painter.add(egui::Shape::closed_line(body, body_stroke));

    let face = |center: egui::Pos2, r: f32, label: &str, on: bool| {
        let fill = if on { ACCENT } else { BG_INPUT };
        painter.circle(center, r, fill, Stroke::new(1.0_f32, BORDER));
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { MUTED };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional((r * 0.95).max(7.0)),
                tc,
            );
        }
    };
    let pad = |center: egui::Pos2, sz: Vec2, label: &str, on: bool| {
        let r = egui::Rect::from_center_size(center, sz);
        let fill = if on { ACCENT } else { BG_INPUT };
        painter.rect(r, Rounding::same(3.0), fill, Stroke::new(1.0_f32, BORDER));
        if !label.is_empty() {
            let tc = if on { Color32::WHITE } else { MUTED };
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                label,
                FontId::proportional(9.0 * s.max(0.8)),
                tc,
            );
        }
    };
    let stick = |base: egui::Pos2, sx: f32, sy: f32, clicked: bool| {
        let ring = 22.0 * s;
        let knob = 13.0 * s;
        painter.circle(base, ring, BG_INPUT, Stroke::new(1.5_f32, outline));
        let off = Vec2::new(sx, -sy) * (ring - knob - 1.0);
        let kc = if clicked {
            ACCENT
        } else {
            Color32::from_rgb(0x62, 0x62, 0x72)
        };
        painter.circle(base + off, knob, kc, Stroke::new(1.0_f32, BORDER));
    };

    pad(
        map(-120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZL",
        input.is(SwitchButton::ZL),
    );
    pad(
        map(120.0, -139.0),
        Vec2::new(52.0 * s, 13.0 * s),
        "ZR",
        input.is(SwitchButton::ZR),
    );
    pad(
        map(-120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "L",
        input.is(SwitchButton::L),
    );
    pad(
        map(120.0, -122.0),
        Vec2::new(62.0 * s, 14.0 * s),
        "R",
        input.is(SwitchButton::R),
    );

    stick(
        map(-111.0, -55.0),
        input.lx(),
        input.ly(),
        input.is(SwitchButton::StickL),
    );

    let dd = 19.0 * s;
    let dsz = Vec2::splat(16.0 * s);
    let dp = map(-61.0, 0.0);
    pad(
        dp + Vec2::new(0.0, -dd),
        dsz,
        "",
        input.is(SwitchButton::DUp),
    );
    pad(
        dp + Vec2::new(0.0, dd),
        dsz,
        "",
        input.is(SwitchButton::DDown),
    );
    pad(
        dp + Vec2::new(-dd, 0.0),
        dsz,
        "",
        input.is(SwitchButton::DLeft),
    );
    pad(
        dp + Vec2::new(dd, 0.0),
        dsz,
        "",
        input.is(SwitchButton::DRight),
    );

    let fr = 15.0 * s;
    face(map(136.0, -56.0), fr, "A", input.is(SwitchButton::A));
    face(map(105.0, -25.0), fr, "B", input.is(SwitchButton::B));
    face(map(105.0, -87.0), fr, "X", input.is(SwitchButton::X));
    face(map(74.0, -56.0), fr, "Y", input.is(SwitchButton::Y));

    stick(
        map(51.0, 0.0),
        input.rx(),
        input.ry(),
        input.is(SwitchButton::StickR),
    );

    face(
        map(-50.0, -86.0),
        9.0 * s,
        "-",
        input.is(SwitchButton::Minus),
    );
    face(map(50.0, -86.0), 9.0 * s, "+", input.is(SwitchButton::Plus));
}

fn row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(egui::RichText::new(label).size(12.0).color(MUTED));
    ui.label(egui::RichText::new(value).size(12.0).color(TEXT));
    ui.end_row();
}

fn graphics_settings_content(ui: &mut egui::Ui, cfg: &mut AppSettings, save_needed: &mut bool) {
    ui.label(
        egui::RichText::new("Aspect Mode")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(egui::RichText::new("Letterbox keeps the game's aspect ratio. Integer snaps to pixel-perfect 1x/2x/3x. Stretch fills the window.").size(11.0).color(MUTED));
    ui.add_space(6.0);
    for m in AspectMode::all() {
        if ui.radio(cfg.aspect == *m, m.label()).clicked() {
            cfg.aspect = *m;
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Output Scale")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("Used only by Integer mode (1x / 2x / 3x).")
            .size(11.0)
            .color(MUTED),
    );
    ui.add_space(6.0);
    let mut s = cfg.output_scale.max(1).min(3);
    ui.horizontal(|ui| {
        for v in 1u8..=3u8 {
            if ui.radio(s == v, format!("{}x", v)).clicked() {
                s = v;
            }
        }
    });
    if s != cfg.output_scale {
        cfg.output_scale = s;
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Texture Filter")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Nearest preserves pixel-art crispness. Linear smooths upscaled output.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);
    for f in FilterMode::all() {
        if ui.radio(cfg.filter == *f, f.label()).clicked() {
            cfg.filter = *f;
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Windowing")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    let mut dpi = cfg.dpi_aware;
    if ui
        .checkbox(&mut dpi, "High-DPI aware (Windows — requires restart)")
        .changed()
    {
        cfg.dpi_aware = dpi;
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Off lets Windows DPI virtualization upscale (blurry). On reports physical pixels — required for crisp output on 4K/2.5K displays.").size(10.5).color(MUTED));

    ui.add_space(8.0);
    let mut vsync = cfg.vsync;
    if ui
        .checkbox(&mut vsync, "V-Sync (requires restart)")
        .changed()
    {
        cfg.vsync = vsync;
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Shaders")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    let mut async_shaders = cfg.async_shaders;
    if ui
        .checkbox(&mut async_shaders, "Asynchronous shader compilation")
        .changed()
    {
        cfg.async_shaders = async_shaders;
        nexium_common::async_compile::set_enabled(async_shaders);
        *save_needed = true;
    }
    ui.label(egui::RichText::new("Builds new pipelines on a background thread so the game never stalls to compile. New effects pop in a frame or two the first time they appear. Off = compile on demand (brief hitch on first sight, no pop-in). Disk cache makes later launches stutter-free either way.").size(10.5).color(MUTED));
}

fn audio_settings_content(
    ui: &mut egui::Ui,
    cfg: &mut AppSettings,
    save_needed: &mut bool,
    device_cache: &mut Option<Vec<String>>,
) {
    ui.label(
        egui::RichText::new("Output Device")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Pick which audio output to use. Device changes apply on next emulator boot \
         (the stream is opened once at startup).",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);

    if device_cache.is_none() {
        *device_cache = Some(list_output_devices());
    }
    let devices: Vec<String> = device_cache.clone().unwrap_or_default();

    let current_label: String = match &cfg.audio_output_device {
        None => "System default".to_string(),
        Some(n) => n.clone(),
    };
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("audio_output_device")
            .selected_text(current_label.clone())
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(cfg.audio_output_device.is_none(), "System default")
                    .clicked()
                {
                    if cfg.audio_output_device.is_some() {
                        cfg.audio_output_device = None;
                        *save_needed = true;
                    }
                }
                for name in &devices {
                    let selected = cfg.audio_output_device.as_deref() == Some(name.as_str());
                    if ui.selectable_label(selected, name).clicked() && !selected {
                        cfg.audio_output_device = Some(name.clone());
                        *save_needed = true;
                    }
                }
            });
        if ui.button("Refresh").clicked() {
            *device_cache = Some(list_output_devices());
        }
    });

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Active Stream")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    match current_stream_info() {
        Some(AudioStreamInfo {
            device_name,
            sample_rate,
            channels,
            sample_format,
        }) => {
            ui.label(
                egui::RichText::new(format!(
                    "{}  ·  {} Hz  ·  {} ch  ·  {}",
                    device_name, sample_rate, channels, sample_format,
                ))
                .size(11.0)
                .color(MUTED),
            );
        }
        None => {
            ui.label(
                egui::RichText::new("No audio stream open (init failed or audio disabled).")
                    .size(11.0)
                    .color(AMBER),
            );
        }
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Master Volume")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Applied live in the audio callback. 1.0 = unity, 2.0 = +6 dB (may clip).",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(6.0);
    let mut vol = cfg.audio_volume;
    if ui
        .add(egui::Slider::new(&mut vol, 0.0..=2.0).text("vol"))
        .changed()
    {
        cfg.audio_volume = vol;
        set_master_volume(vol);
        *save_needed = true;
    }

    ui.add_space(12.0);
    ui.label(
        egui::RichText::new("Diagnostics")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(6.0);
    if ui.button("Play 1s 440 Hz test tone").clicked() {
        let pushed = push_test_tone(440.0, 1.0);
        log::info!("Audio test tone: pushed {} frames to sink", pushed);
    }
}

fn emulation_settings_content(ui: &mut egui::Ui, cfg: &mut AppSettings, save_needed: &mut bool) {
    ui.label(
        egui::RichText::new("CPU Backend")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(egui::RichText::new(
        "Selects which AArch64 JIT runs guest code. Applies on next boot — does not affect a running game.",
    ).size(11.0).color(MUTED));
    ui.add_space(8.0);

    for backend in CpuBackend::all() {
        let compiled = backend.is_compiled_in();
        let label = if compiled {
            backend.label().to_string()
        } else {
            format!("{} — not in build", backend.label())
        };
        ui.add_enabled_ui(compiled, |ui| {
            if ui.radio(cfg.cpu_backend == *backend, label).clicked() {
                cfg.cpu_backend = *backend;
                *save_needed = true;
            }
        });
    }

    ui.add_space(10.0);
    ui.label(
        egui::RichText::new(
            "Dynarmic: MerryMage's mature C++ JIT — the baseline used by yuzu/Ryujinx. \
        Rustarmic: our own Rust AArch64→x86_64 JIT — newer, useful when debugging guest \
        behaviour that dynarmic's opaque codegen makes hard to inspect.",
        )
        .size(10.5)
        .color(MUTED),
    );

    ui.add_space(16.0);
    ui.separator();
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Multi-Core")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new(
            "Runs each guest core on its own host thread (3 user cores). Applies on next boot.",
        )
        .size(11.0)
        .color(MUTED),
    );
    ui.add_space(8.0);
    let resp = ui.checkbox(&mut cfg.multicore, "Enable multi-core CPU emulation");
    if resp.changed() {
        *save_needed = true;
    }
    resp.on_hover_text("Keep this on — disables only for debugging");
}

fn logging_settings_content(ui: &mut egui::Ui, cfg: &mut AppSettings, save_needed: &mut bool) {
    ui.label(
        egui::RichText::new("Log Level")
            .size(13.0)
            .strong()
            .color(TEXT),
    );
    ui.add_space(4.0);
    ui.label(
        egui::RichText::new("Takes effect on restart.")
            .size(11.0)
            .color(MUTED),
    );
    ui.add_space(8.0);

    for level in LogLevel::all() {
        if ui.radio(cfg.log_level == *level, level.label()).clicked() {
            cfg.log_level = *level;
            *save_needed = true;
        }
    }

    ui.add_space(12.0);
    let path = AppSettings::config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "unknown".into());
    ui.label(
        egui::RichText::new(format!("Config: {}", path))
            .size(10.5)
            .color(MUTED),
    );
}

fn debug_windows(
    ctx: &egui::Context,
    dbg: &mut DebuggerState,
    snapshot: Option<&crate::boot::CpuSnapshot>,
    mem_request: Option<&dyn Fn(u64)>,
    log_buffer: &std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
) {
    if dbg.show_memory {
        egui::Window::new("Memory")
            .open(&mut dbg.show_memory)
            .default_size([520.0, 340.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Address").size(12.0).color(MUTED));
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut dbg.memory_address_input)
                            .desired_width(160.0)
                            .font(egui::TextStyle::Monospace),
                    );
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        let s = dbg.memory_address_input.trim().trim_start_matches("0x");
                        if let Ok(addr) = u64::from_str_radix(s, 16) {
                            dbg.memory_address = addr;
                            if let Some(req) = mem_request {
                                req(addr);
                            }
                        }
                    }
                    if ui.small_button("Go").clicked() {
                        let s = dbg.memory_address_input.trim().trim_start_matches("0x");
                        if let Ok(addr) = u64::from_str_radix(s, 16) {
                            dbg.memory_address = addr;
                            if let Some(req) = mem_request {
                                req(addr);
                            }
                        }
                    }
                });
                ui.add_space(4.0);

                if let Some(snap) = snapshot {
                    if snap.mem_data.is_empty() || snap.mem_address != dbg.memory_address {
                        if let Some(req) = mem_request {
                            req(dbg.memory_address);
                        }
                        ui.label(
                            egui::RichText::new("Reading memory…")
                                .size(11.0)
                                .color(MUTED),
                        );
                    } else {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut row = 0;
                                while row * 16 < snap.mem_data.len() {
                                    let off = row * 16;
                                    let addr = snap.mem_address + off as u64;
                                    let slice =
                                        &snap.mem_data[off..(off + 16).min(snap.mem_data.len())];
                                    let hex: Vec<String> =
                                        slice.iter().map(|b| format!("{:02x}", b)).collect();
                                    let ascii: String = slice
                                        .iter()
                                        .map(|&b| {
                                            if (0x20..0x7F).contains(&b) {
                                                b as char
                                            } else {
                                                '.'
                                            }
                                        })
                                        .collect();
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:016x}", addr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(hex.join(" "))
                                                .size(11.5)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.label(
                                            egui::RichText::new(ascii)
                                                .size(11.5)
                                                .monospace()
                                                .color(AMBER),
                                        );
                                    });
                                    row += 1;
                                }
                            });
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_registers {
        egui::Window::new("Registers")
            .open(&mut dbg.show_registers)
            .default_size([320.0, 480.0])
            .show(ctx, |ui| {
                if let Some(snap) = snapshot {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            egui::Grid::new("regs")
                                .num_columns(2)
                                .spacing([12.0, 2.0])
                                .show(ui, |ui| {
                                    for (n, v) in [
                                        ("PC", snap.pc),
                                        ("SP", snap.sp),
                                        ("TPIDRRO", snap.tpidrro_el0),
                                    ] {
                                        ui.label(
                                            egui::RichText::new(format!("{:<8}", n))
                                                .size(12.0)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:#018x}", v))
                                                .size(12.0)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.end_row();
                                    }
                                    for i in 0..31 {
                                        ui.label(
                                            egui::RichText::new(format!("X{:<7}", i))
                                                .size(12.0)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:#018x}", snap.x[i]))
                                                .size(12.0)
                                                .monospace()
                                                .color(TEXT),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_disasm {
        egui::Window::new("Disassembler")
            .open(&mut dbg.show_disasm)
            .default_size([520.0, 340.0])
            .show(ctx, |ui| {
                if let Some(snap) = snapshot {
                    if snap.instruction_bytes.is_empty() {
                        ui.label(
                            egui::RichText::new("No instruction data yet")
                                .size(11.0)
                                .color(MUTED),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!("PC {:#018x}", snap.pc))
                                .size(12.0)
                                .monospace()
                                .color(AMBER),
                        );
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .auto_shrink([false; 2])
                            .show(ui, |ui| {
                                let mut off = 0;
                                while off + 4 <= snap.instruction_bytes.len() {
                                    let instr = u32::from_le_bytes([
                                        snap.instruction_bytes[off],
                                        snap.instruction_bytes[off + 1],
                                        snap.instruction_bytes[off + 2],
                                        snap.instruction_bytes[off + 3],
                                    ]);
                                    let addr = snap.pc + off as u64;
                                    let mnemonic = disasm_arm64(instr);
                                    let color = if off == 0 { ACCENT } else { TEXT };
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("{:016x}", addr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(format!("{:08x}", instr))
                                                .size(11.5)
                                                .monospace()
                                                .color(MUTED),
                                        );
                                        ui.label(
                                            egui::RichText::new(mnemonic)
                                                .size(11.5)
                                                .monospace()
                                                .color(color),
                                        );
                                    });
                                    off += 4;
                                }
                            });
                    }
                } else {
                    ui.label(
                        egui::RichText::new("No emulation running")
                            .size(11.0)
                            .color(MUTED),
                    );
                }
            });
    }

    if dbg.show_logs {
        egui::Window::new("Logs")
            .open(&mut dbg.show_logs)
            .default_size([580.0, 300.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if pill_button(ui, "Clear", false).clicked() {
                        if let Ok(mut buf) = log_buffer.lock() {
                            buf.clear();
                        }
                    }
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if let Ok(buf) = log_buffer.lock() {
                            for entry in buf.iter() {
                                ui.label(
                                    egui::RichText::new(entry)
                                        .size(11.0)
                                        .monospace()
                                        .color(MUTED),
                                );
                            }
                        }
                    });
            });
    }
}

fn disasm_arm64(instr: u32) -> String {
    if instr == 0xD503201F {
        return "nop".into();
    }
    if instr == 0xD65F03C0 {
        return "ret".into();
    }
    if instr == 0xD4200000 {
        return "brk #0".into();
    }

    let op = instr >> 24;

    if (instr & 0xFFE0_0000) == 0xD400_0000 {
        let imm = (instr >> 5) & 0xFFFF;
        return format!("svc #{:#x}", imm);
    }
    if (instr & 0xFC00_0000) == 0x9400_0000 {
        let imm26 = instr & 0x03FF_FFFF;
        let off = if imm26 & 0x0200_0000 != 0 {
            ((imm26 | 0xFC00_0000) as i32) * 4
        } else {
            (imm26 as i32) * 4
        };
        return format!("bl pc{:+}", off);
    }
    if (instr & 0xFC00_0000) == 0x1400_0000 {
        let imm26 = instr & 0x03FF_FFFF;
        let off = if imm26 & 0x0200_0000 != 0 {
            ((imm26 | 0xFC00_0000) as i32) * 4
        } else {
            (imm26 as i32) * 4
        };
        return format!("b pc{:+}", off);
    }
    if (instr & 0xFFC0_0000) == 0x9100_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rd = instr & 0x1F;
        return format!("add x{}, x{}, #{:#x}", rd, rn, imm12);
    }
    if (instr & 0xFFC0_0000) == 0xD100_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rd = instr & 0x1F;
        return format!("sub x{}, x{}, #{:#x}", rd, rn, imm12);
    }
    if (instr & 0xFFC0_0000) == 0xF940_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rt = instr & 0x1F;
        return format!("ldr x{}, [x{}, #{:#x}]", rt, rn, imm12 * 8);
    }
    if (instr & 0xFFC0_0000) == 0xF900_0000 {
        let imm12 = (instr >> 10) & 0xFFF;
        let rn = (instr >> 5) & 0x1F;
        let rt = instr & 0x1F;
        return format!("str x{}, [x{}, #{:#x}]", rt, rn, imm12 * 8);
    }
    if (instr & 0xFFE0_0000) == 0xD280_0000 {
        let imm16 = (instr >> 5) & 0xFFFF;
        let rd = instr & 0x1F;
        return format!("mov x{}, #{:#x}", rd, imm16);
    }

    format!("? .word {:#010x}  (op={:#x})", instr, op)
}

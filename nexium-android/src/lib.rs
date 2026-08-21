#![cfg(target_os = "android")]

mod library;
mod overlay;
mod ui;

use android_activity::input::{
    Axis, InputEvent, KeyAction, Keycode, MotionAction, Source,
};
use android_activity::{AndroidApp, InputStatus, MainEvent, PollEvent};
use library::{GameEntry, LibraryScan, ICON_SIZE};
use nexium_core::hid_state::{
    ControllerInput, KeyboardInput, MouseInput, TouchInput,
};
use nexium_runner::boot::{EmulationHandle, Frame};
use overlay::{OverlayEvent, TouchOverlay};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ui::{Canvas, TextPainter, ACCENT, BG, DANGER, PANEL, PANEL_HI, TEXT, TEXT_DIM};

const STICK_RANGE: f32 = 30_000.0;
const STICK_DEADZONE: f32 = 0.15;
const TRIGGER_PRESS: f32 = 0.55;
const TRIGGER_RELEASE: f32 = 0.40;

const BTN_A: u64 = 1 << 0;
const BTN_B: u64 = 1 << 1;
const BTN_X: u64 = 1 << 2;
const BTN_Y: u64 = 1 << 3;
const BTN_STICK_L: u64 = 1 << 4;
const BTN_STICK_R: u64 = 1 << 5;
const BTN_L: u64 = 1 << 6;
const BTN_R: u64 = 1 << 7;
const BTN_ZL: u64 = 1 << 8;
const BTN_ZR: u64 = 1 << 9;
const BTN_PLUS: u64 = 1 << 10;
const BTN_MINUS: u64 = 1 << 11;
const BTN_DLEFT: u64 = 1 << 12;
const BTN_DUP: u64 = 1 << 13;
const BTN_DRIGHT: u64 = 1 << 14;
const BTN_DDOWN: u64 = 1 << 15;

#[derive(Default)]
struct PadState {
    buttons: u64,
    trigger_bits: u64,
    lx: f32,
    ly: f32,
    rx: f32,
    ry: f32,
    hat_buttons: u64,
}

fn radial_deadzone(x: f32, y: f32) -> (f32, f32) {
    let mag = (x * x + y * y).sqrt();
    if mag < STICK_DEADZONE {
        return (0.0, 0.0);
    }
    let scaled = ((mag - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)).clamp(0.0, 1.0);
    (x / mag * scaled, y / mag * scaled)
}

impl PadState {
    fn merged_buttons(&self) -> u64 {
        self.buttons | self.hat_buttons | self.trigger_bits
    }
}

fn keycode_to_button(keycode: Keycode) -> Option<u64> {
    Some(match keycode {
        Keycode::ButtonA => BTN_A,
        Keycode::ButtonB => BTN_B,
        Keycode::ButtonX => BTN_X,
        Keycode::ButtonY => BTN_Y,
        Keycode::ButtonL1 => BTN_L,
        Keycode::ButtonR1 => BTN_R,
        Keycode::ButtonL2 => BTN_ZL,
        Keycode::ButtonR2 => BTN_ZR,
        Keycode::ButtonStart => BTN_PLUS,
        Keycode::ButtonSelect => BTN_MINUS,
        Keycode::ButtonThumbl => BTN_STICK_L,
        Keycode::ButtonThumbr => BTN_STICK_R,
        Keycode::DpadUp => BTN_DUP,
        Keycode::DpadDown => BTN_DDOWN,
        Keycode::DpadLeft => BTN_DLEFT,
        Keycode::DpadRight => BTN_DRIGHT,
        Keycode::Enter => BTN_A,
        Keycode::Escape => BTN_B,
        _ => return None,
    })
}

fn is_joystick_source(source: Source) -> bool {
    u32::from(source) & 0x0000_0010 != 0
}

fn is_pointer_source(source: Source) -> bool {
    u32::from(source) & 0x0000_0002 != 0
}

fn find_roms_dir(app: &AndroidApp) -> PathBuf {
    let data_root = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
        .unwrap_or_else(|| PathBuf::from("/data/local/tmp/nexium"));
    data_root.join("roms")
}

fn init_environment(app: &AndroidApp) -> PathBuf {
    let data_root = app
        .external_data_path()
        .or_else(|| app.internal_data_path())
        .unwrap_or_else(|| PathBuf::from("/data/local/tmp/nexium"));
    let config_root = data_root.join("config");
    let appdata_root = data_root.join("appdata");
    let _ = std::fs::create_dir_all(&config_root);
    let _ = std::fs::create_dir_all(&appdata_root);
    let _ = std::fs::create_dir_all(data_root.join("roms"));
    std::env::set_var("HOME", &data_root);
    std::env::set_var("XDG_CONFIG_HOME", &config_root);
    std::env::set_var("APPDATA", &appdata_root);
    data_root
}

fn present(app: &AndroidApp, width: usize, height: usize, pixels: &[u8]) {
    if width == 0 || height == 0 || pixels.len() < width * height * 4 {
        return;
    }
    let Some(window) = app.native_window() else {
        return;
    };
    if window
        .set_buffers_geometry(
            width as i32,
            height as i32,
            Some(ndk::hardware_buffer_format::HardwareBufferFormat::R8G8B8A8_UNORM),
        )
        .is_err()
    {
        return;
    }
    let Ok(mut buffer) = window.lock(None) else {
        return;
    };
    let stride_px = buffer.stride() as usize;
    let buf_w = buffer.width() as usize;
    let buf_h = buffer.height() as usize;
    let out_w = buf_w.min(width);
    let out_h = buf_h.min(height);
    let src_row = width * 4;
    let dst = buffer.bits() as *mut u8;
    for row in 0..buf_h {
        let row_ptr = unsafe { dst.add(row * stride_px * 4) };
        if row < out_h {
            let src_off = row * src_row;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels[src_off..src_off + out_w * 4].as_ptr(),
                    row_ptr,
                    out_w * 4,
                );
            }
        }
        let filled = if row < out_h { out_w } else { 0 };
        if filled < buf_w {
            unsafe {
                std::ptr::write_bytes(row_ptr.add(filled * 4), 0, (buf_w - filled) * 4);
            }
        }
    }
}

const MENU_ITEMS: usize = 3;

enum Screen {
    Library {
        scan: LibraryScan,
        selected: usize,
        scroll: f32,
        drag: Option<(i32, f32, f32)>,
    },
    Booting {
        title: String,
        started: Instant,
    },
    Playing {
        menu: Option<usize>,
    },
    Stopping {
        done: Arc<AtomicBool>,
        since: Instant,
    },
    Error {
        message: String,
    },
}

struct App {
    screen: Screen,
    emulation: Option<EmulationHandle>,
    last_frame: Option<Frame>,
    pad: PadState,
    overlay: TouchOverlay,
    canvas: Canvas,
    text: TextPainter,
    logo: Option<(usize, usize, Vec<u8>)>,
    logo_cache: Vec<(usize, Vec<u8>)>,
    roms_dir: PathBuf,
    window_ready: bool,
    ui_dirty: bool,
    last_stick_nav: f32,
}

fn decode_logo() -> Option<(usize, usize, Vec<u8>)> {
    let bytes = include_bytes!("../../branding/png/logo-128.png");
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some((w, h, img.into_raw()))
}

impl App {
    fn window_size(&self, app: &AndroidApp) -> (usize, usize) {
        app.native_window()
            .map(|w| (w.width() as usize, w.height() as usize))
            .unwrap_or((1280, 720))
    }

    fn push_hid(&self) {
        let overlay_stick = self.overlay.stick();
        let overlay_active = overlay_stick.0 != 0.0 || overlay_stick.1 != 0.0;
        let (plx, ply) = radial_deadzone(self.pad.lx, self.pad.ly);
        let (prx, pry) = radial_deadzone(self.pad.rx, self.pad.ry);
        let (lx, ly) = if overlay_active {
            (overlay_stick.0, overlay_stick.1)
        } else {
            (plx, ply)
        };
        let guest_active = matches!(self.screen, Screen::Playing { menu: None });
        let overlay_buttons = if guest_active {
            self.overlay.buttons()
        } else {
            0
        };
        let touch = if guest_active {
            self.overlay.guest_touch()
        } else {
            TouchInput::default()
        };
        let (lx, ly, prx, pry) = if guest_active {
            (lx, ly, prx, pry)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let pad_buttons = if guest_active {
            self.pad.merged_buttons()
        } else {
            0
        };
        let state = nexium_core::hid_state::get_hid_state();
        let mut hid = state.lock();
        hid.update_input(ControllerInput {
            buttons: pad_buttons | overlay_buttons,
            stick_l_x: (lx * STICK_RANGE) as i32,
            stick_l_y: (-ly * STICK_RANGE) as i32,
            stick_r_x: (prx * STICK_RANGE) as i32,
            stick_r_y: (-pry * STICK_RANGE) as i32,
        });
        hid.update_devices(
            MouseInput {
                x: 0,
                y: 0,
                wheel_x: 0,
                wheel_y: 0,
                buttons: 0,
                connected: false,
            },
            KeyboardInput::default(),
            touch,
        );
    }

    fn start_game(&mut self, path: &std::path::Path, title: String) {
        let Some(path_str) = path.to_str() else {
            self.screen = Screen::Error {
                message: "unrepresentable path".into(),
            };
            return;
        };
        match EmulationHandle::new(path_str, nexium_cpu::CpuBackendKind::Dynarmic, None) {
            Ok(handle) => {
                log::info!("emulation started: {}", title);
                self.emulation = Some(handle);
                self.last_frame = None;
                self.screen = Screen::Booting {
                    title,
                    started: Instant::now(),
                };
            }
            Err(error) => {
                log::error!("emulation failed to start: {}", error);
                self.screen = Screen::Error {
                    message: error,
                };
            }
        }
        self.ui_dirty = true;
    }

    fn exit_to_library(&mut self) {
        self.overlay.cancel_all();
        self.pad = PadState::default();
        self.push_hid();
        let done = Arc::new(AtomicBool::new(false));
        if let Some(mut emu) = self.emulation.take() {
            let flag = Arc::clone(&done);
            let _ = std::thread::Builder::new()
                .name("nexium-stop".into())
                .spawn(move || {
                    emu.stop_blocking();
                    flag.store(true, Ordering::Release);
                });
        } else {
            done.store(true, Ordering::Release);
        }
        self.last_frame = None;
        self.screen = Screen::Stopping {
            done,
            since: Instant::now(),
        };
        self.ui_dirty = true;
    }

    fn take_emulation_end(&mut self) -> Option<Option<String>> {
        let finished = self
            .emulation
            .as_ref()
            .and_then(|emu| emu.thread_handle.as_ref())
            .map(|handle| handle.is_finished())
            .unwrap_or(false);
        if !finished {
            return None;
        }
        let mut emu = self.emulation.take()?;
        let outcome = match emu.thread_handle.take().map(|handle| handle.join()) {
            Some(Ok(Ok(()))) => None,
            Some(Ok(Err(error))) => Some(error),
            Some(Err(_)) => Some("emulation thread panicked".to_string()),
            None => Some("emulation exited unexpectedly".to_string()),
        };
        match &outcome {
            Some(error) => log::error!("emulation failed: {}", error),
            None => log::info!("emulation exited cleanly"),
        }
        self.last_frame = None;
        Some(outcome)
    }

    fn to_library(&mut self) {
        self.screen = Screen::Library {
            scan: LibraryScan::start(self.roms_dir.clone()),
            selected: 0,
            scroll: 0.0,
            drag: None,
        };
        self.ui_dirty = true;
    }

    fn nav_move(&mut self, delta: i8) {
        if let Screen::Library {
            scan, selected, ..
        } = &mut self.screen
        {
            if !scan.done || scan.games.is_empty() {
                return;
            }
            let len = scan.games.len();
            let cur = *selected as i64 + delta as i64;
            *selected = cur.rem_euclid(len as i64) as usize;
            self.ui_dirty = true;
        } else if let Screen::Playing { menu: Some(idx) } = &mut self.screen {
            let cur = *idx as i64 + delta as i64;
            *idx = cur.rem_euclid(MENU_ITEMS as i64) as usize;
            self.ui_dirty = true;
        }
    }

    fn nav_accept(&mut self) {
        match &mut self.screen {
            Screen::Library {
                scan, selected, ..
            } => {
                if scan.done {
                    if let Some(game) = scan.games.get(*selected) {
                        let path = game.path.clone();
                        let title = game.title.clone();
                        self.start_game(&path, title);
                    }
                }
            }
            Screen::Playing { menu } => {
                if let Some(idx) = *menu {
                    match idx {
                        0 => self.close_menu(),
                        1 => {
                            self.overlay.enabled = !self.overlay.enabled;
                            self.ui_dirty = true;
                        }
                        _ => self.exit_to_library(),
                    }
                }
            }
            Screen::Error { .. } => self.to_library(),
            _ => {}
        }
    }

    fn open_menu(&mut self) {
        if let Screen::Playing { menu } = &mut self.screen {
            if menu.is_none() {
                *menu = Some(0);
                if let Some(emu) = &self.emulation {
                    emu.pause();
                }
                self.overlay.cancel_all();
                self.push_hid();
                self.ui_dirty = true;
            }
        }
    }

    fn close_menu(&mut self) {
        if let Screen::Playing { menu } = &mut self.screen {
            if menu.is_some() {
                *menu = None;
                if let Some(emu) = &self.emulation {
                    emu.resume();
                }
                self.ui_dirty = true;
            }
        }
    }

    fn back_is_consumed(&self) -> bool {
        matches!(
            self.screen,
            Screen::Playing { .. } | Screen::Error { .. } | Screen::Booting { .. }
        )
    }

    fn on_back(&mut self) {
        match &self.screen {
            Screen::Booting { .. } => self.exit_to_library(),
            Screen::Playing { menu: None } => self.open_menu(),
            Screen::Playing { menu: Some(_) } => self.close_menu(),
            Screen::Error { .. } => self.nav_accept(),
            _ => {}
        }
    }

    fn render_library(&mut self, app: &AndroidApp) {
        let (win_w, win_h) = self.window_size(app);
        self.canvas.resize(win_w, win_h);
        self.canvas.clear(BG);
        let w = win_w as f32;
        let h = win_h as f32;
        let header_h = (h * 0.14).max(64.0);
        let list_top = header_h + h * 0.02;
        let row_h = (ICON_SIZE as f32 + h * 0.03).max(h * 0.16);
        let hint_px = h * 0.026;
        let hint_h = hint_px * 2.2;

        if let Screen::Library { scan, scroll, .. } = &mut self.screen {
            let content = scan.games.len() as f32 * row_h;
            let viewport = (h - list_top - hint_h).max(0.0);
            let max_scroll = (content - viewport).max(0.0);
            *scroll = scroll.clamp(0.0, max_scroll);
        }

        let mut empty_state = false;
        let mut scanning = false;
        if let Screen::Library {
            scan,
            selected,
            scroll,
            ..
        } = &self.screen
        {
            scanning = !scan.done;
            empty_state = scan.done && scan.games.is_empty();
            if scan.done && !scan.games.is_empty() {
                let selected = *selected;
                let scroll = *scroll;
                let row_x = (h * 0.02) as i32;
                let row_w = win_w as i32 - row_x * 2;
                for (i, game) in scan.games.iter().enumerate() {
                    let y = list_top + i as f32 * row_h - scroll;
                    if y + row_h < list_top || y > h - hint_h {
                        continue;
                    }
                    if i == selected {
                        self.canvas.fill_rounded(
                            row_x,
                            y as i32,
                            row_w,
                            (row_h * 0.94) as i32,
                            (h * 0.012) as i32,
                            PANEL_HI,
                        );
                        self.canvas.fill_rect(
                            row_x,
                            y as i32 + (row_h * 0.12) as i32,
                            (h * 0.006).max(3.0) as i32,
                            (row_h * 0.70) as i32,
                            ACCENT,
                        );
                    }
                    let icon_pad = (row_h * 0.94 - ICON_SIZE as f32) * 0.5;
                    let icon_x = row_x + (h * 0.02) as i32;
                    let icon_y = (y + icon_pad) as i32;
                    match &game.icon {
                        Some(pixels) => {
                            self.canvas
                                .blit_rgba(icon_x, icon_y, ICON_SIZE, ICON_SIZE, pixels);
                        }
                        None => {
                            self.canvas.fill_rounded(
                                icon_x,
                                icon_y,
                                ICON_SIZE as i32,
                                ICON_SIZE as i32,
                                (h * 0.012) as i32,
                                PANEL,
                            );
                            let initial = game.title.chars().next().unwrap_or('?');
                            let mut buf = [0u8; 4];
                            let initial = initial.encode_utf8(&mut buf);
                            let ipx = ICON_SIZE as f32 * 0.5;
                            let iw = self.text.measure(initial, ipx, false);
                            self.text.draw(
                                &mut self.canvas,
                                icon_x as f32 + (ICON_SIZE as f32 - iw) * 0.5,
                                icon_y as f32 + ICON_SIZE as f32 * 0.5 - ipx * 0.62,
                                ipx,
                                TEXT_DIM,
                                initial,
                            );
                        }
                    }
                    let text_x = icon_x as f32 + ICON_SIZE as f32 + h * 0.025;
                    let title_px = row_h * 0.26;
                    self.text.draw(
                        &mut self.canvas,
                        text_x,
                        y + row_h * 0.22,
                        title_px,
                        TEXT,
                        &game.title,
                    );
                    let sub = format_sub(game);
                    if !sub.is_empty() {
                        self.text.draw(
                            &mut self.canvas,
                            text_x,
                            y + row_h * 0.55,
                            row_h * 0.17,
                            TEXT_DIM,
                            &sub,
                        );
                    }
                }
            }
        }

        if scanning {
            let msg = "Scanning library...";
            let px = h * 0.045;
            let tw = self.text.measure(msg, px, false);
            self.text.draw(
                &mut self.canvas,
                (w - tw) * 0.5,
                h * 0.5 - px * 0.62,
                px,
                TEXT_DIM,
                msg,
            );
        } else if empty_state {
            let px = h * 0.05;
            let msg = "No games found";
            let tw = self.text.measure(msg, px, false);
            self.text
                .draw(&mut self.canvas, (w - tw) * 0.5, h * 0.36, px, TEXT, msg);
            let hint2 = h * 0.028;
            let path = self.roms_dir.display().to_string();
            let mut y = h * 0.36 + px * 2.0;
            let intro = "Copy games (.dnsp / .dxci / .nro) to:";
            let tw = self.text.measure(intro, hint2, false);
            self.text
                .draw(&mut self.canvas, (w - tw) * 0.5, y, hint2, TEXT_DIM, intro);
            y += hint2 * 1.8;
            let tw = self.text.measure(&path, hint2, true);
            self.text.draw_mono(
                &mut self.canvas,
                ((w - tw) * 0.5).max(h * 0.02),
                y,
                hint2,
                ACCENT,
                &path,
            );
        }

        self.canvas
            .fill_rect(0, 0, win_w as i32, header_h as i32, PANEL);
        let logo_size = (header_h * 0.62) as usize;
        self.draw_logo_scaled(
            (h * 0.02) as i32,
            ((header_h - logo_size as f32) * 0.5) as i32,
            logo_size,
        );
        let title_px = header_h * 0.42;
        self.text.draw(
            &mut self.canvas,
            h * 0.02 + logo_size as f32 + h * 0.02,
            header_h * 0.5 - title_px * 0.62,
            title_px,
            TEXT,
            "NeXium",
        );
        let (game_count, scan_done) = match &self.screen {
            Screen::Library { scan, .. } => (scan.games.len(), scan.done),
            _ => (0, true),
        };
        let count_text = if scan_done {
            format!(
                "{} game{}",
                game_count,
                if game_count == 1 { "" } else { "s" }
            )
        } else {
            "scanning...".to_string()
        };
        let count_px = header_h * 0.24;
        let count_w = self.text.measure(&count_text, count_px, false);
        self.text.draw(
            &mut self.canvas,
            w - count_w - h * 0.03,
            header_h * 0.5 - count_px * 0.62,
            count_px,
            TEXT_DIM,
            &count_text,
        );

        if scan_done && game_count > 0 {
            self.canvas.fill_rect(
                0,
                (h - hint_h) as i32,
                win_w as i32,
                hint_h as i32 + 1,
                PANEL,
            );
            let hint = "tap or press A to play";
            let tw = self.text.measure(hint, hint_px, false);
            self.text.draw(
                &mut self.canvas,
                (w - tw) * 0.5,
                h - hint_px * 1.8,
                hint_px,
                TEXT_DIM,
                hint,
            );
        }
        present(app, self.canvas.w, self.canvas.h, &self.canvas.pixels);
    }

    fn draw_logo_scaled(&mut self, x: i32, y: i32, target: usize) {
        if target == 0 {
            return;
        }
        if !self.logo_cache.iter().any(|(size, _)| *size == target) {
            let Some((w, h, pixels)) = &self.logo else {
                return;
            };
            let scaled = if *w == target && *h == target {
                pixels.clone()
            } else {
                let Some(img) =
                    image::RgbaImage::from_raw(*w as u32, *h as u32, pixels.clone())
                else {
                    return;
                };
                image::imageops::resize(
                    &img,
                    target as u32,
                    target as u32,
                    image::imageops::FilterType::Triangle,
                )
                .into_raw()
            };
            if self.logo_cache.len() >= 3 {
                self.logo_cache.remove(0);
            }
            self.logo_cache.push((target, scaled));
        }
        if let Some(idx) = self.logo_cache.iter().position(|(size, _)| *size == target) {
            let pixels = std::mem::take(&mut self.logo_cache[idx].1);
            self.canvas.blit_rgba(x, y, target, target, &pixels);
            self.logo_cache[idx].1 = pixels;
        }
    }

    fn render_center_screen(&mut self, app: &AndroidApp, lines: &[(String, [u8; 4], f32)]) {
        let (win_w, win_h) = self.window_size(app);
        self.canvas.resize(win_w, win_h);
        self.canvas.clear(BG);
        let w = win_w as f32;
        let h = win_h as f32;
        let logo_size = (h * 0.18) as usize;
        self.draw_logo_scaled(
            ((w - logo_size as f32) * 0.5) as i32,
            (h * 0.24) as i32,
            logo_size,
        );
        let mut y = h * 0.24 + logo_size as f32 + h * 0.06;
        let max_w = w * 0.86;
        for (line, color, scale) in lines {
            let px = h * scale;
            for part in wrap_text(&self.text, line, px, max_w) {
                let tw = self.text.measure(&part, px, false);
                self.text
                    .draw(&mut self.canvas, (w - tw) * 0.5, y, px, *color, &part);
                y += px * 1.7;
            }
        }
        present(app, self.canvas.w, self.canvas.h, &self.canvas.pixels);
    }

    fn compose_rect(&self, app: &AndroidApp) -> Option<(usize, usize, usize, usize, usize, usize)> {
        let frame = self.last_frame.as_ref()?;
        let (fw, fh) = (frame.width as usize, frame.height as usize);
        if fw == 0 || fh == 0 || frame.pixels.len() < fw * fh * 4 {
            return None;
        }
        let (win_w, win_h) = self.window_size(app);
        let win_aspect = win_w.max(1) as f32 / win_h.max(1) as f32;
        let frame_aspect = fw as f32 / fh as f32;
        let (cw, ch) = if win_aspect > frame_aspect {
            ((fh as f32 * win_aspect).round().max(fw as f32) as usize, fh)
        } else {
            (fw, (fw as f32 / win_aspect).round().max(fh as f32) as usize)
        };
        Some((cw, ch, (cw - fw) / 2, (ch - fh) / 2, fw, fh))
    }

    fn render_playing(&mut self, app: &AndroidApp) {
        let menu = match &self.screen {
            Screen::Playing { menu } => *menu,
            _ => None,
        };
        let Some((cw, ch, ox, oy, fw, fh)) = self.compose_rect(app) else {
            return;
        };
        self.canvas.resize(cw, ch);
        if ox > 0 || oy > 0 {
            self.canvas.clear([0, 0, 0, 0xFF]);
        }
        if let Some(frame) = &self.last_frame {
            for row in 0..fh {
                let src = row * fw * 4;
                let dst = ((row + oy) * cw + ox) * 4;
                self.canvas.pixels[dst..dst + fw * 4]
                    .copy_from_slice(&frame.pixels[src..src + fw * 4]);
            }
        }
        self.overlay.layout(cw as f32, ch as f32);
        self.overlay
            .set_frame_rect(ox as f32, oy as f32, fw as f32, fh as f32);
        if menu.is_none() {
            self.overlay.render(&mut self.canvas, &mut self.text);
        } else {
            self.canvas.dim(140);
            self.render_menu(menu.unwrap_or(0), cw as f32, ch as f32);
        }
        present(app, self.canvas.w, self.canvas.h, &self.canvas.pixels);
    }

    fn render_menu(&mut self, selected: usize, w: f32, h: f32) {
        let items = [
            "Resume".to_string(),
            format!(
                "Touch Controls: {}",
                if self.overlay.enabled { "On" } else { "Off" }
            ),
            "Exit to Library".to_string(),
        ];
        let panel_w = w * 0.34;
        let item_h = h * 0.085;
        let panel_h = item_h * items.len() as f32 + h * 0.10;
        let px0 = (w - panel_w) * 0.5;
        let py0 = (h - panel_h) * 0.5;
        self.canvas.fill_rounded(
            px0 as i32,
            py0 as i32,
            panel_w as i32,
            panel_h as i32,
            (h * 0.02) as i32,
            PANEL,
        );
        let title_px = h * 0.04;
        let tw = self.text.measure("Paused", title_px, false);
        self.text.draw(
            &mut self.canvas,
            px0 + (panel_w - tw) * 0.5,
            py0 + h * 0.02,
            title_px,
            ACCENT,
            "Paused",
        );
        let mut y = py0 + h * 0.08;
        for (i, item) in items.iter().enumerate() {
            if i == selected {
                self.canvas.fill_rounded(
                    (px0 + w * 0.012) as i32,
                    y as i32,
                    (panel_w - w * 0.024) as i32,
                    (item_h * 0.92) as i32,
                    (h * 0.012) as i32,
                    PANEL_HI,
                );
            }
            let ipx = item_h * 0.38;
            self.text.draw(
                &mut self.canvas,
                px0 + w * 0.03,
                y + item_h * 0.24,
                ipx,
                if i == selected { TEXT } else { TEXT_DIM },
                item,
            );
            y += item_h;
        }
    }

    fn menu_hit(&self, x: f32, y: f32, w: f32, h: f32) -> Option<usize> {
        let panel_w = w * 0.34;
        let item_h = h * 0.085;
        let panel_h = item_h * MENU_ITEMS as f32 + h * 0.10;
        let px0 = (w - panel_w) * 0.5;
        let py0 = (h - panel_h) * 0.5;
        if x < px0 || x > px0 + panel_w {
            return None;
        }
        let rel = y - (py0 + h * 0.08);
        if rel < 0.0 {
            return None;
        }
        let idx = (rel / item_h) as usize;
        if idx < MENU_ITEMS {
            Some(idx)
        } else {
            None
        }
    }
}

fn wrap_text(painter: &TextPainter, text: &str, px: f32, max_w: f32) -> Vec<String> {
    if painter.measure(text, px, false) <= max_w {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{} {}", current, word)
        };
        if painter.measure(&candidate, px, false) > max_w && !current.is_empty() {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        } else {
            current = candidate;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(text.to_string());
    }
    lines
}

fn format_sub(game: &GameEntry) -> String {
    match (game.author.trim(), game.version.trim()) {
        ("", "") => String::new(),
        (a, "") => a.to_string(),
        ("", v) => v.to_string(),
        (a, v) => format!("{}  ·  {}", a, v),
    }
}

#[no_mangle]
fn android_main(android_app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("nexium"),
    );
    log::info!("nexium-android starting");

    let data_root = init_environment(&android_app);
    log::info!("data root: {}", data_root.display());

    nexium_core::hid_state::set_docked(false);
    nexium_runner::audio::init_host_audio(None, 1.0);

    let Some(text) = TextPainter::new() else {
        log::error!("font init failed");
        return;
    };
    let roms_dir = find_roms_dir(&android_app);
    let mut state = App {
        screen: Screen::Library {
            scan: LibraryScan::start(roms_dir.clone()),
            selected: 0,
            scroll: 0.0,
            drag: None,
        },
        emulation: None,
        last_frame: None,
        pad: PadState::default(),
        overlay: TouchOverlay::new(),
        canvas: Canvas::new(4, 4),
        text,
        logo: decode_logo(),
        logo_cache: Vec::new(),
        roms_dir,
        window_ready: false,
        ui_dirty: true,
        last_stick_nav: 0.0,
    };
    let mut running = true;
    let mut last_ui_draw = Instant::now() - Duration::from_secs(1);

    while running {
        let poll_timeout = match &state.screen {
            Screen::Playing { menu: None } => Duration::from_millis(4),
            Screen::Booting { .. } | Screen::Stopping { .. } => {
                Duration::from_millis(16)
            }
            Screen::Library { scan, .. } if !scan.done => Duration::from_millis(16),
            _ => Duration::from_millis(100),
        };
        android_app.poll_events(Some(poll_timeout), |event| match event {
            PollEvent::Main(main_event) => match main_event {
                MainEvent::InitWindow { .. } => {
                    state.window_ready = true;
                    state.ui_dirty = true;
                    log::info!("window ready");
                }
                MainEvent::TerminateWindow { .. } => {
                    state.window_ready = false;
                    log::info!("window lost");
                }
                MainEvent::GainedFocus => {
                    if matches!(
                        state.screen,
                        Screen::Playing { menu: None } | Screen::Booting { .. }
                    ) {
                        if let Some(emu) = &state.emulation {
                            emu.resume();
                        }
                    }
                }
                MainEvent::LostFocus => {
                    if let Some(emu) = &state.emulation {
                        emu.pause();
                    }
                    state.overlay.cancel_all();
                    state.pad = PadState::default();
                    state.push_hid();
                }
                MainEvent::Destroy => {
                    running = false;
                }
                _ => {}
            },
            PollEvent::Timeout | PollEvent::Wake => {}
            _ => {}
        });

        if !running {
            break;
        }

        let mut input_dirty = false;
        let (win_w, win_h) = state.window_size(&android_app);
        if let Ok(mut iter) = android_app.input_events_iter() {
            loop {
                let handled = iter.next(|event| match event {
                    InputEvent::KeyEvent(key_event) => {
                        let keycode = key_event.key_code();
                        if keycode == Keycode::Back {
                            if !state.back_is_consumed() {
                                return InputStatus::Unhandled;
                            }
                            if matches!(key_event.action(), KeyAction::Up) {
                                state.on_back();
                            }
                            return InputStatus::Handled;
                        }
                        let Some(bit) = keycode_to_button(keycode) else {
                            return InputStatus::Unhandled;
                        };
                        let down = matches!(key_event.action(), KeyAction::Down);
                        match key_event.action() {
                            KeyAction::Down => state.pad.buttons |= bit,
                            KeyAction::Up => state.pad.buttons &= !bit,
                            _ => {}
                        }
                        if down {
                            match bit {
                                BTN_DUP => state.nav_move(-1),
                                BTN_DDOWN => state.nav_move(1),
                                BTN_A => {
                                    if !matches!(
                                        state.screen,
                                        Screen::Playing { menu: None }
                                    ) {
                                        state.nav_accept()
                                    }
                                }
                                BTN_B => {
                                    if matches!(
                                        state.screen,
                                        Screen::Playing { menu: Some(_) }
                                    ) {
                                        state.close_menu()
                                    }
                                }
                                _ => {}
                            }
                        }
                        input_dirty = true;
                        InputStatus::Handled
                    }
                    InputEvent::MotionEvent(motion_event) => {
                        let source = motion_event.source();
                        if is_joystick_source(source) {
                            if let Some(pointer) = motion_event.pointers().next() {
                                state.pad.lx = pointer.axis_value(Axis::X);
                                state.pad.ly = pointer.axis_value(Axis::Y);
                                let z = pointer.axis_value(Axis::Z);
                                let rz = pointer.axis_value(Axis::Rz);
                                let rx = pointer.axis_value(Axis::Rx);
                                let ry = pointer.axis_value(Axis::Ry);
                                if z == 0.0 && rz == 0.0 && (rx != 0.0 || ry != 0.0) {
                                    state.pad.rx = rx;
                                    state.pad.ry = ry;
                                } else {
                                    state.pad.rx = z;
                                    state.pad.ry = rz;
                                }
                                let lt = pointer
                                    .axis_value(Axis::Brake)
                                    .max(pointer.axis_value(Axis::Ltrigger));
                                let rt = pointer
                                    .axis_value(Axis::Gas)
                                    .max(pointer.axis_value(Axis::Rtrigger));
                                for (value, bit) in [(lt, BTN_ZL), (rt, BTN_ZR)] {
                                    if value >= TRIGGER_PRESS {
                                        state.pad.trigger_bits |= bit;
                                    } else if value <= TRIGGER_RELEASE {
                                        state.pad.trigger_bits &= !bit;
                                    }
                                }
                                let hat_x = pointer.axis_value(Axis::HatX);
                                let hat_y = pointer.axis_value(Axis::HatY);
                                let prev_hat = state.pad.hat_buttons;
                                state.pad.hat_buttons = 0;
                                if hat_x < -0.5 {
                                    state.pad.hat_buttons |= BTN_DLEFT;
                                } else if hat_x > 0.5 {
                                    state.pad.hat_buttons |= BTN_DRIGHT;
                                }
                                if hat_y < -0.5 {
                                    state.pad.hat_buttons |= BTN_DUP;
                                } else if hat_y > 0.5 {
                                    state.pad.hat_buttons |= BTN_DDOWN;
                                }
                                let new_hat = state.pad.hat_buttons & !prev_hat;
                                if new_hat & BTN_DUP != 0 {
                                    state.nav_move(-1);
                                }
                                if new_hat & BTN_DDOWN != 0 {
                                    state.nav_move(1);
                                }
                                let nav_y = pointer.axis_value(Axis::Y);
                                if state.last_stick_nav.abs() <= 0.5 && nav_y.abs() > 0.5 {
                                    state.nav_move(if nav_y < 0.0 { -1 } else { 1 });
                                }
                                state.last_stick_nav = nav_y;
                            }
                            input_dirty = true;
                            return InputStatus::Handled;
                        }
                        if !is_pointer_source(source) {
                            return InputStatus::Unhandled;
                        }
                        let action = motion_event.action();
                        let w = win_w as f32;
                        let h = win_h as f32;
                        let playing = matches!(state.screen, Screen::Playing { .. });
                        let menu_open =
                            matches!(state.screen, Screen::Playing { menu: Some(_) });
                        let compose = if playing {
                            state.compose_rect(&android_app)
                        } else {
                            None
                        };
                        let scale = compose
                            .map(|(cw, ch, ..)| {
                                (cw as f32 / w.max(1.0), ch as f32 / h.max(1.0))
                            })
                            .unwrap_or((1.0, 1.0));
                        match action {
                            MotionAction::Down | MotionAction::PointerDown => {
                                let idx = if matches!(action, MotionAction::Down) {
                                    0
                                } else {
                                    motion_event.pointer_index()
                                };
                                if let Some(p) = motion_event.pointers().nth(idx) {
                                    let (x, y) = (p.x(), p.y());
                                    if menu_open {
                                        let (cw, ch) = compose
                                            .map(|(cw, ch, ..)| (cw as f32, ch as f32))
                                            .unwrap_or((w, h));
                                        if let Some(idx) = state.menu_hit(
                                            x * scale.0,
                                            y * scale.1,
                                            cw,
                                            ch,
                                        ) {
                                            if let Screen::Playing { menu } =
                                                &mut state.screen
                                            {
                                                *menu = Some(idx);
                                            }
                                            state.nav_accept();
                                        } else {
                                            state.close_menu();
                                        }
                                    } else if playing {
                                        match state.overlay.pointer_down(
                                            p.pointer_id(),
                                            x * scale.0,
                                            y * scale.1,
                                        ) {
                                            OverlayEvent::MenuTap => {}
                                            OverlayEvent::None => {}
                                        }
                                    } else if let Screen::Library {
                                        drag, ..
                                    } = &mut state.screen
                                    {
                                        *drag = Some((p.pointer_id(), y, 0.0));
                                    } else if matches!(state.screen, Screen::Error { .. })
                                    {
                                        state.nav_accept();
                                    }
                                }
                                input_dirty = true;
                            }
                            MotionAction::Move => {
                                for p in motion_event.pointers() {
                                    let (x, y) = (p.x(), p.y());
                                    if playing && !menu_open {
                                        state.overlay.pointer_move(
                                            p.pointer_id(),
                                            x * scale.0,
                                            y * scale.1,
                                        );
                                    } else if let Screen::Library {
                                        drag,
                                        scroll,
                                        ..
                                    } = &mut state.screen
                                    {
                                        if let Some((id, last_y, total)) = drag {
                                            if *id == p.pointer_id() {
                                                let dy = y - *last_y;
                                                *scroll = (*scroll - dy).max(0.0);
                                                *total += dy.abs();
                                                *last_y = y;
                                                state.ui_dirty = true;
                                            }
                                        }
                                    }
                                }
                                input_dirty = true;
                            }
                            MotionAction::Up
                            | MotionAction::PointerUp
                            | MotionAction::Cancel => {
                                let cancelled = matches!(action, MotionAction::Cancel);
                                let idx = if matches!(action, MotionAction::PointerUp) {
                                    motion_event.pointer_index()
                                } else {
                                    0
                                };
                                if let Some(p) = motion_event.pointers().nth(idx) {
                                    let (x, y) = (p.x(), p.y());
                                    if playing && !menu_open {
                                        if cancelled {
                                            state.overlay.cancel_all();
                                        } else if let OverlayEvent::MenuTap =
                                            state.overlay.pointer_up(
                                                p.pointer_id(),
                                                x * scale.0,
                                                y * scale.1,
                                            )
                                        {
                                            state.open_menu();
                                        }
                                    } else if !playing {
                                        let mut launch: Option<usize> = None;
                                        if let Screen::Library {
                                            drag,
                                            scan,
                                            scroll,
                                            selected,
                                        } = &mut state.screen
                                        {
                                            if let Some((id, _, total)) = drag {
                                                if *id == p.pointer_id() {
                                                    let tap =
                                                        !cancelled && *total < h * 0.01;
                                                    *drag = None;
                                                    if tap && scan.done {
                                                        let header_h =
                                                            (h * 0.14).max(64.0);
                                                        let row_h = (ICON_SIZE as f32
                                                            + h * 0.03)
                                                            .max(h * 0.16);
                                                        let list_top =
                                                            header_h + h * 0.02;
                                                        let hint_h = h * 0.026 * 2.2;
                                                        let rel =
                                                            y - list_top + *scroll;
                                                        if rel >= 0.0
                                                            && y >= list_top
                                                            && y <= h - hint_h
                                                        {
                                                            let idx =
                                                                (rel / row_h) as usize;
                                                            if idx < scan.games.len() {
                                                                *selected = idx;
                                                                launch = Some(idx);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if launch.is_some() {
                                            state.ui_dirty = true;
                                            state.nav_accept();
                                        }
                                    }
                                }
                                input_dirty = true;
                            }
                            _ => {}
                        }
                        InputStatus::Handled
                    }
                    _ => InputStatus::Unhandled,
                });
                if !handled {
                    break;
                }
            }
        }
        if input_dirty {
            state.push_hid();
        }

        match &mut state.screen {
            Screen::Library { scan, .. } => {
                if scan.poll() {
                    state.ui_dirty = true;
                }
            }
            Screen::Booting { .. } => {
                let mut newest: Option<Frame> = None;
                if let Some(emu) = &state.emulation {
                    while let Ok(frame) = emu.frame_rx.try_recv() {
                        newest = Some(frame);
                    }
                }
                if let Some(frame) = newest {
                    state.last_frame = Some(frame);
                    state.screen = Screen::Playing { menu: None };
                    state.ui_dirty = true;
                } else if let Some(outcome) = state.take_emulation_end() {
                    match outcome {
                        Some(message) => {
                            state.screen = Screen::Error { message };
                            state.ui_dirty = true;
                        }
                        None => state.to_library(),
                    }
                }
            }
            Screen::Playing { menu } => {
                let menu_open = menu.is_some();
                let mut newest: Option<Frame> = None;
                if let Some(emu) = &state.emulation {
                    while let Ok(frame) = emu.frame_rx.try_recv() {
                        newest = Some(frame);
                    }
                }
                if let Some(frame) = newest {
                    state.last_frame = Some(frame);
                    if !menu_open && state.window_ready {
                        state.render_playing(&android_app);
                    }
                } else if let Some(outcome) = state.take_emulation_end() {
                    match outcome {
                        Some(message) => {
                            state.screen = Screen::Error { message };
                            state.ui_dirty = true;
                        }
                        None => state.to_library(),
                    }
                }
            }
            Screen::Stopping { done, since } => {
                if done.load(Ordering::Acquire)
                    || since.elapsed() >= Duration::from_secs(10)
                {
                    if !done.load(Ordering::Acquire) {
                        log::warn!("stop timed out; returning to library");
                    }
                    state.to_library();
                }
            }
            Screen::Error { .. } => {}
        }

        let animating = matches!(
            state.screen,
            Screen::Booting { .. } | Screen::Stopping { .. }
        ) || matches!(&state.screen, Screen::Library { scan, .. } if !scan.done);
        let ui_due = last_ui_draw.elapsed() >= Duration::from_millis(16);
        let anim_due = last_ui_draw.elapsed() >= Duration::from_millis(100);
        if state.window_ready
            && ((state.ui_dirty && ui_due) || (animating && anim_due))
        {
            {
                match &state.screen {
                    Screen::Library { .. } => state.render_library(&android_app),
                    Screen::Booting { title, started } => {
                        let dots = ".".repeat(1 + (started.elapsed().as_millis() / 400 % 3) as usize);
                        let lines = vec![
                            (format!("Booting {}{}", title, dots), TEXT, 0.045f32),
                            (
                                "first boot can take a little while".to_string(),
                                TEXT_DIM,
                                0.028,
                            ),
                        ];
                        state.render_center_screen(&android_app, &lines);
                    }
                    Screen::Playing { .. } => state.render_playing(&android_app),
                    Screen::Stopping { .. } => {
                        let lines =
                            vec![("Shutting down...".to_string(), TEXT_DIM, 0.04f32)];
                        state.render_center_screen(&android_app, &lines);
                    }
                    Screen::Error { message } => {
                        let lines = vec![
                            ("Could not start game".to_string(), DANGER, 0.045f32),
                            (message.clone(), TEXT_DIM, 0.026),
                            ("tap or press A to return".to_string(), TEXT_DIM, 0.028),
                        ];
                        state.render_center_screen(&android_app, &lines);
                    }
                }
                state.ui_dirty = false;
                last_ui_draw = Instant::now();
            }
        }
    }

    if let Some(mut emu) = state.emulation.take() {
        emu.stop_blocking();
    }
    log::info!("nexium-android exiting");
}

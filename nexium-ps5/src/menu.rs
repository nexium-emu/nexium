use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use nexium_core::hid_state;
use nexium_cpu::CpuBackendKind;
use nexium_runner::boot::{EmulationHandle, Frame};
use serde::{Deserialize, Serialize};

use crate::console::{ensure_shared_dir, share, DATA_ROOT};
use crate::display::Display;
use crate::library::LibraryScan;
use crate::menu_view::{scroll_for, Glyph, View, H, W};
use crate::pad::{self, Pad, PadState};

const DIR_UP: u32 = 1 << 28;
const DIR_DOWN: u32 = 1 << 29;
const DIR_LEFT: u32 = 1 << 30;
const DIR_RIGHT: u32 = 1 << 31;

#[derive(Serialize, Deserialize, Clone)]
struct Settings {
    #[serde(default = "yes")]
    docked: bool,
    #[serde(default = "full_volume")]
    volume: f32,
    #[serde(default)]
    show_fps: bool,
    #[serde(default = "yes")]
    stable_resolution: bool,
}

fn yes() -> bool {
    true
}

fn full_volume() -> f32 {
    1.0
}

impl Settings {
    fn path() -> String {
        format!("{DATA_ROOT}/ps5-settings.json")
    }

    fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Self { docked: true, volume: 1.0, show_fps: false, stable_resolution: true })
    }

    fn save(&self) {
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let path = Self::path();
            if std::fs::write(&path, text).is_ok() {
                share(&path, 0o666);
            }
        }
    }
}

enum Screen {
    Library,
    ConfirmQuit,
    Settings { cursor: usize },
    Loading { game: usize, started: Instant },
    Playing,
    Paused { cursor: usize, backdrop: Vec<u8> },
    Error { title: String, message: String },
}

struct Nav {
    held: u32,
    repeat_at: Option<Instant>,
}

impl Nav {
    fn update(&mut self, p: &PadState) -> u32 {
        let mut now = p.buttons & 0x00ff_ffff;
        if p.buttons & pad::UP != 0 || p.ly < 50 {
            now |= DIR_UP;
        }
        if p.buttons & pad::DOWN != 0 || p.ly > 206 {
            now |= DIR_DOWN;
        }
        if p.buttons & pad::LEFT != 0 || p.lx < 50 {
            now |= DIR_LEFT;
        }
        if p.buttons & pad::RIGHT != 0 || p.lx > 206 {
            now |= DIR_RIGHT;
        }
        let dirs = DIR_UP | DIR_DOWN | DIR_LEFT | DIR_RIGHT;
        let mut fired = now & !self.held;
        let held_dirs = now & self.held & dirs;
        if fired & dirs != 0 {
            self.repeat_at = Some(Instant::now() + Duration::from_millis(380));
        } else if held_dirs != 0 {
            if self.repeat_at.is_some_and(|t| Instant::now() >= t) {
                fired |= held_dirs;
                self.repeat_at = Some(Instant::now() + Duration::from_millis(85));
            }
        } else {
            self.repeat_at = None;
        }
        self.held = now;
        fired
    }
}

struct Session {
    emu: EmulationHandle,
    title: String,
    started: Instant,
    last_frame: Option<Frame>,
    guest_frames: u64,
    window_start: Instant,
    window_guest: u64,
    fps: f32,
}

pub struct App {
    display: Display,
    view: View,
    pad: Option<Pad>,
    settings: Settings,
    scan: LibraryScan,
    sizes: Vec<Option<u64>>,
    cursor: usize,
    scroll: usize,
    screen: Screen,
    session: Option<Session>,
    nav: Nav,
    dirty: bool,
    tick: u64,
    audio: Option<crate::audio::PullAudioOut>,
}

fn library_roots() -> Vec<PathBuf> {
    vec![PathBuf::from(format!("{DATA_ROOT}/games"))]
}

impl App {
    pub fn new() -> Result<Self, String> {
        let display = Display::new(true)?;
        let view = View::new().ok_or("font load failed")?;
        let pad = match Pad::open() {
            Ok(p) => Some(p),
            Err(e) => {
                crate::klog!("menu: no controller yet: {e}");
                None
            }
        };
        let settings = Settings::load();
        if settings.stable_resolution && std::env::var_os("NEXIUM_FAST_GPU_TIME").is_none() {
            std::env::set_var("NEXIUM_FAST_GPU_TIME", "1");
        }
        let _ = ensure_shared_dir(&format!("{DATA_ROOT}/games"));
        let audio = crate::frontend::open_audio(settings.volume);
        Ok(Self {
            display,
            view,
            pad,
            settings,
            scan: LibraryScan::start(library_roots()),
            sizes: Vec::new(),
            cursor: 0,
            scroll: 0,
            screen: Screen::Library,
            session: None,
            nav: Nav { held: 0, repeat_at: None },
            dirty: true,
            tick: 0,
            audio,
        })
    }

    pub fn run(mut self) -> u32 {
        let mut failures = 0;
        let mut pad_retry = Instant::now();
        loop {
            self.tick += 1;
            if self.pad.is_none() && pad_retry.elapsed() >= Duration::from_secs(2) {
                pad_retry = Instant::now();
                self.pad = Pad::open().ok();
            }
            let state = self.pad.as_mut().map(|p| p.poll()).unwrap_or_default();
            let fired = self.nav.update(&state);
            if self.scan.poll() {
                crate::klog!("menu: library has {} games", self.scan.games.len());
                self.sizes = self.scan.games.iter().map(|g| std::fs::metadata(&g.path).ok().map(|m| m.len())).collect();
                self.cursor = self.cursor.min(self.scan.games.len().saturating_sub(1));
                self.dirty = true;
            }
            if !self.step(&state, fired) {
                break;
            }
            let result = if matches!(self.screen, Screen::Playing) { self.present_game() } else { self.present_ui() };
            if let Err(e) = result {
                crate::klog!("menu: present failed: {e}");
                failures += 1;
                break;
            }
        }
        if let Some(s) = self.session.take() {
            if !crate::frontend::stop_emulation(s.emu) {
                failures += 1;
            }
        }
        drop(self.audio.take());
        failures
    }

    fn step(&mut self, state: &PadState, fired: u32) -> bool {
        let mut next: Option<Screen> = None;
        match &mut self.screen {
            Screen::Library => {
                let count = self.scan.games.len();
                if fired & DIR_UP != 0 && self.cursor > 0 {
                    self.cursor -= 1;
                    self.dirty = true;
                }
                if fired & DIR_DOWN != 0 && self.cursor + 1 < count {
                    self.cursor += 1;
                    self.dirty = true;
                }
                if fired & pad::CROSS != 0 && self.scan.done && count > 0 {
                    next = Some(self.launch(self.cursor));
                } else if fired & pad::TRIANGLE != 0 {
                    self.scan = LibraryScan::start(library_roots());
                    self.dirty = true;
                } else if fired & pad::OPTIONS != 0 {
                    next = Some(Screen::Settings { cursor: 0 });
                } else if fired & pad::CIRCLE != 0 {
                    next = Some(Screen::ConfirmQuit);
                }
            }
            Screen::ConfirmQuit => {
                if fired & pad::CROSS != 0 {
                    crate::klog!("menu: quit");
                    return false;
                }
                if fired & pad::CIRCLE != 0 {
                    next = Some(Screen::Library);
                }
            }
            Screen::Settings { cursor } => {
                if fired & DIR_UP != 0 && *cursor > 0 {
                    *cursor -= 1;
                    self.dirty = true;
                }
                if fired & DIR_DOWN != 0 && *cursor < 3 {
                    *cursor += 1;
                    self.dirty = true;
                }
                let delta = if fired & (DIR_RIGHT | pad::CROSS) != 0 {
                    1
                } else if fired & DIR_LEFT != 0 {
                    -1
                } else {
                    0
                };
                if delta != 0 {
                    match *cursor {
                        0 => self.settings.docked = !self.settings.docked,
                        1 => {
                            let v = (self.settings.volume * 10.0).round() as i32 + delta;
                            self.settings.volume = v.clamp(0, 20) as f32 / 10.0;
                            nexium_runner::audio::set_master_volume(self.settings.volume);
                        }
                        2 => self.settings.show_fps = !self.settings.show_fps,
                        _ => self.settings.stable_resolution = !self.settings.stable_resolution,
                    }
                    self.dirty = true;
                }
                if fired & (pad::CIRCLE | pad::OPTIONS) != 0 {
                    self.settings.save();
                    next = Some(Screen::Library);
                }
            }
            Screen::Loading { started, .. } => {
                let started = *started;
                if self.tick % 4 == 0 {
                    self.dirty = true;
                }
                crate::frontend::push_hid(state);
                if let Some(err) = self.pump_session() {
                    next = Some(err);
                } else if self.session.as_ref().is_some_and(|s| s.last_frame.is_some()) {
                    crate::klog!("menu: first frame after {:.1}s", started.elapsed().as_secs_f32());
                    next = Some(Screen::Playing);
                }
            }
            Screen::Playing => {
                let combo = pad::OPTIONS | pad::TOUCH_PAD;
                if state.buttons & combo == combo {
                    crate::frontend::push_hid(&PadState::default());
                    if let Some(s) = &self.session {
                        s.emu.pause();
                    }
                    next = Some(Screen::Paused { cursor: 0, backdrop: self.backdrop() });
                } else {
                    crate::frontend::push_hid(state);
                    if let Some(err) = self.pump_session() {
                        next = Some(err);
                    }
                }
            }
            Screen::Paused { cursor, .. } => {
                if fired & (DIR_UP | DIR_DOWN) != 0 {
                    *cursor ^= 1;
                    self.dirty = true;
                }
                let resume = fired & pad::CIRCLE != 0 || (fired & pad::CROSS != 0 && *cursor == 0);
                if resume {
                    if let Some(s) = &self.session {
                        s.emu.resume();
                    }
                    next = Some(Screen::Playing);
                } else if fired & pad::CROSS != 0 {
                    self.view.message("Closing game…", "");
                    let _ = self.display.present_rgba(W as u32, H as u32, Some(&self.view.canvas.pixels));
                    self.end_session();
                    next = Some(Screen::Library);
                }
            }
            Screen::Error { .. } => {
                if fired & (pad::CIRCLE | pad::CROSS) != 0 {
                    next = Some(Screen::Library);
                }
            }
        }
        if let Some(screen) = next {
            self.screen = screen;
            self.dirty = true;
        }
        true
    }

    fn launch(&mut self, index: usize) -> Screen {
        let Some(game) = self.scan.games.get(index) else { return Screen::Library };
        let path = game.path.to_string_lossy().into_owned();
        let title = game.title.clone();
        self.view.set_big_icon(game.icon.as_deref());
        hid_state::set_docked(self.settings.docked);
        nexium_runner::audio::set_master_volume(self.settings.volume);
        crate::klog!("menu: launching '{title}' {path} docked={}", self.settings.docked);
        match EmulationHandle::new(&path, CpuBackendKind::Dynarmic, None) {
            Ok(emu) => {
                if let Some(handle) = emu.thread_handle.as_ref() {
                    use std::os::unix::thread::JoinHandleExt;
                    crate::affinity::pin(handle.as_pthread_t() as usize, c"nexium-core0".as_ptr());
                }
                self.session = Some(Session {
                    emu,
                    title,
                    started: Instant::now(),
                    last_frame: None,
                    guest_frames: 0,
                    window_start: Instant::now(),
                    window_guest: 0,
                    fps: 0.0,
                });
                Screen::Loading { game: index, started: Instant::now() }
            }
            Err(e) => {
                crate::klog!("menu: launch failed: {e}");
                Screen::Error { title: format!("Couldn't start {title}"), message: e }
            }
        }
    }

    fn pump_session(&mut self) -> Option<Screen> {
        let session = self.session.as_mut()?;
        if session.emu.thread_handle.as_ref().is_some_and(|h| h.is_finished()) {
            let result = session.emu.thread_handle.take().map(|h| h.join());
            crate::klog!("menu: emulation thread ended: {result:?}");
            let title = session.title.clone();
            self.end_session();
            return match result {
                Some(Ok(Ok(()))) => Some(Screen::Library),
                Some(Ok(Err(e))) => Some(Screen::Error { title: format!("{title} stopped"), message: e }),
                _ => Some(Screen::Error { title: format!("{title} stopped"), message: "The emulation thread panicked.".into() }),
            };
        }
        while let Ok(frame) = session.emu.frame_rx.try_recv() {
            session.guest_frames += 1;
            session.window_guest += 1;
            session.last_frame = Some(frame);
            self.dirty = true;
        }
        let window = session.window_start.elapsed();
        if window >= Duration::from_secs(5) {
            session.fps = session.window_guest as f32 / window.as_secs_f32();
            session.window_start = Instant::now();
            session.window_guest = 0;
            let grains = self.audio.as_ref().map(|a| a.stats.grains.load(Ordering::Relaxed)).unwrap_or(0);
            crate::klog!(
                "menu: '{}' t={:.0}s {:.1} fps frames {} audio grains {grains}",
                session.title,
                session.started.elapsed().as_secs_f32(),
                session.fps,
                session.guest_frames
            );
        } else if window >= Duration::from_secs(1) && session.fps == 0.0 {
            session.fps = session.window_guest as f32 / window.as_secs_f32();
        }
        None
    }

    fn end_session(&mut self) {
        if let Some(s) = self.session.take() {
            s.emu.resume();
            crate::frontend::stop_emulation(s.emu);
        }
        crate::frontend::push_hid(&PadState::default());
    }

    fn backdrop(&self) -> Vec<u8> {
        let Some(frame) = self.session.as_ref().and_then(|s| s.last_frame.as_ref()) else {
            return Vec::new();
        };
        match image::RgbaImage::from_raw(frame.width, frame.height, frame.pixels.clone()) {
            Some(img) => image::imageops::resize(&img, W as u32, H as u32, image::imageops::FilterType::Triangle).into_raw(),
            None => Vec::new(),
        }
    }

    fn present_game(&mut self) -> Result<(), String> {
        let fresh = std::mem::take(&mut self.dirty);
        let show_fps = self.settings.show_fps;
        let Some(session) = self.session.as_mut() else { return Ok(()) };
        let fps = session.fps;
        let Some(frame) = session.last_frame.as_mut() else { return Ok(()) };
        if fresh && show_fps {
            self.view.fps_overlay(&mut frame.pixels, frame.width, frame.height, fps);
        }
        self.display.present_rgba(frame.width, frame.height, fresh.then_some(&frame.pixels[..]))
    }

    fn present_ui(&mut self) -> Result<(), String> {
        let fresh = std::mem::take(&mut self.dirty);
        if fresh {
            self.draw();
        }
        self.display.present_rgba(W as u32, H as u32, fresh.then_some(&self.view.canvas.pixels[..]))
    }

    fn draw(&mut self) {
        self.scroll = scroll_for(self.cursor, self.scroll);
        match &self.screen {
            Screen::Library => self.view.library(&self.scan.games, &self.sizes, self.scan.done, self.cursor, self.scroll),
            Screen::ConfirmQuit => {
                self.view.library(&self.scan.games, &self.sizes, self.scan.done, self.cursor, self.scroll);
                self.view.dialog("Quit NeXium?", "You'll return to the PS5 home screen.", &[(Glyph::Cross, "Quit"), (Glyph::Circle, "Cancel")]);
            }
            Screen::Settings { cursor } => self.view.settings(
                self.settings.docked,
                self.settings.volume,
                self.settings.show_fps,
                self.settings.stable_resolution,
                *cursor,
            ),
            Screen::Loading { game, started } => {
                let title = self.scan.games.get(*game).map(|g| g.title.as_str()).unwrap_or("");
                self.view.loading(title, started.elapsed());
            }
            Screen::Playing => {}
            Screen::Paused { cursor, backdrop } => {
                let (title, fps) = self.session.as_ref().map(|s| (s.title.as_str(), s.fps)).unwrap_or(("", 0.0));
                self.view.pause(backdrop, title, fps, *cursor);
            }
            Screen::Error { title, message } => self.view.error(title, message),
        }
    }
}

pub fn run() -> u32 {
    crate::frontend::setup_environment();
    match App::new() {
        Ok(app) => app.run(),
        Err(e) => {
            crate::klog!("menu: init failed: {e}");
            1
        }
    }
}

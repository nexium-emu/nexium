use crate::hd_rumble::{
    self, Frame, Pacer, Step, HID_PACE, NEUTRAL, SDL_HD_PACE, SDL_PACE, SDL_QUIET_PACE,
};
use nexium_core::hid_motion::{self, MotionSource};
use nexium_core::hid_vibration::{self, VibrationValue};
use parking_lot::{Condvar, Mutex};
use sdl3::gamepad::Gamepad;
use sdl3::sys::gamepad::{
    SDL_CloseGamepad, SDL_Gamepad, SDL_GetGamepadID, SDL_GetGamepadType, SDL_OpenGamepad,
    SDL_RumbleGamepad, SDL_SendGamepadEffect, SDL_UpdateGamepads,
    SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR, SDL_GAMEPAD_TYPE_PS5,
};
use sdl3::sys::hidapi::{
    SDL_hid_close, SDL_hid_device, SDL_hid_enumerate, SDL_hid_exit, SDL_hid_free_enumeration,
    SDL_hid_get_device_info, SDL_hid_init, SDL_hid_open_path, SDL_hid_write,
};
use std::ffi::{CStr, CString};
use std::panic::AssertUnwindSafe;
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const HEARTBEAT: Duration = Duration::from_millis(250);
const PREVIEW_TIME: Duration = Duration::from_millis(250);
const TICK: Duration = Duration::from_millis(5);
const IDLE_WAIT: Duration = Duration::from_millis(50);
const STOP_WAIT: Duration = Duration::from_millis(300);
const SDL_SETTLE: Duration = Duration::from_millis(150);
const SDL_PAUSE: Duration = Duration::from_secs(10);
const SDL_DURATION_MS: u32 = 250;
const SILENCE_BUDGET: Duration = Duration::from_millis(100);
const HEALTHY_RESET: Duration = Duration::from_secs(60);
const HAND_TIME: Duration = Duration::from_secs(3);
const HAND_SETTLE: Duration = Duration::from_secs(2);
const RETRY_DELAYS: [Duration; 4] = [
    Duration::from_secs(2),
    Duration::from_secs(10),
    Duration::from_secs(30),
    Duration::from_secs(60),
];
const NINTENDO: u16 = 0x057E;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadMeta {
    pub vendor: u16,
    pub product: u16,
    pub kind: i32,
    pub cap_rumble: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    None,
    HidPair,
    HidSingle { merge: bool, n64: bool },
    SdlHd { merge: bool, n64: bool },
    PairSplit,
    Band { curve: bool },
}

pub fn route(p: &PadMeta, hid: bool, raw: bool) -> Route {
    let nintendo = p.vendor == NINTENDO;
    let pair = p.kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR.0
        || (nintendo && matches!(p.product, 0x2008 | 0x2068));
    let n64 = nintendo && p.product == 0x2019;
    if !(p.cap_rumble || n64) {
        return Route::None;
    }
    if pair {
        return if hid && nintendo && p.product == 0x2008 {
            Route::HidPair
        } else {
            Route::PairSplit
        };
    }
    if nintendo && matches!(p.product, 0x2006 | 0x2007 | 0x2009 | 0x2019) {
        let merge = p.product != 0x2009;
        return if hid {
            Route::HidSingle { merge, n64 }
        } else if raw {
            Route::SdlHd { merge, n64 }
        } else {
            Route::Band { curve: false }
        };
    }
    Route::Band {
        curve: p.kind != SDL_GAMEPAD_TYPE_PS5.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RumbleMode {
    Hd,
    Standard,
    Unsupported,
}

impl RumbleMode {
    fn label(self) -> &'static str {
        match self {
            RumbleMode::Hd => "HD",
            RumbleMode::Standard => "Standard",
            RumbleMode::Unsupported => "No rumble",
        }
    }
}

fn shown_mode(route: Route, mode: RumbleMode) -> RumbleMode {
    match (route, mode) {
        (Route::HidSingle { n64: true, .. } | Route::SdlHd { n64: true, .. }, RumbleMode::Hd) => {
            RumbleMode::Standard
        }
        (_, mode) => mode,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputStatus {
    pub mode: RumbleMode,
    pub name: String,
}

pub fn status_text(enabled: bool, outputs: &[OutputStatus]) -> String {
    if !enabled {
        return "Off".to_string();
    }
    if outputs.is_empty() {
        return "No controller".to_string();
    }
    outputs
        .iter()
        .map(|output| {
            format!(
                "{} \u{b7} {}",
                output.mode.label(),
                output.name.trim_start_matches("Nintendo Switch ")
            )
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

pub struct GamepadRef(usize);

impl GamepadRef {
    fn raw(&self) -> *mut SDL_Gamepad {
        self.0 as *mut SDL_Gamepad
    }

    fn close(self) {
        if self.0 != 0 {
            unsafe { SDL_CloseGamepad(self.raw()) };
        }
    }
}

pub struct PadInfo {
    pub id: u32,
    pub name: String,
    pub vendor: u16,
    pub product: u16,
    pub kind: i32,
    pub cap_rumble: bool,
    pub path: Option<CString>,
    pub gamepad: GamepadRef,
    pub motion_only: bool,
}

impl PadInfo {
    fn from_pad(pad: &Gamepad, motion_only: bool) -> Option<PadInfo> {
        let id = unsafe { SDL_GetGamepadID(pad.raw()) };
        let gamepad = unsafe { SDL_OpenGamepad(id) };
        if gamepad.is_null() {
            return None;
        }
        Some(PadInfo {
            id: id.0,
            name: pad.name().unwrap_or_default(),
            vendor: pad.vendor_id().unwrap_or(0),
            product: pad.product_id().unwrap_or(0),
            kind: unsafe { SDL_GetGamepadType(pad.raw()) }.0,
            cap_rumble: unsafe { pad.has_rumble() },
            path: pad.path().and_then(|path| CString::new(path).ok()),
            gamepad: GamepadRef(gamepad as usize),
            motion_only,
        })
    }
}

struct Pulse {
    pad: u32,
    low: u16,
    high: u16,
    until: Instant,
}

#[derive(Default)]
struct Shared {
    pending: Option<Vec<PadInfo>>,
    live_until: Option<Instant>,
    strength: f32,
    preview: Option<(Instant, f32)>,
    pulses: Vec<Pulse>,
    wake: bool,
    quit: bool,
    status: Vec<OutputStatus>,
    release: Vec<GamepadRef>,
    dead: bool,
}

type SharedState = Arc<(Mutex<Shared>, Condvar)>;

pub struct RumbleOutput {
    shared: SharedState,
    thread: Option<JoinHandle<()>>,
    done: Option<mpsc::Receiver<()>>,
    hid_ref: bool,
    published: Vec<(u32, bool)>,
    was_live: bool,
    stopped: bool,
    died_reported: bool,
}

impl RumbleOutput {
    pub fn new() -> Self {
        let hid_ref = unsafe { SDL_hid_init() } == 0;
        let hid = match std::env::var("NEXIUM_RUMBLE_HID").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => cfg!(windows),
        } && hid_ref;
        let raw = std::env::var("NEXIUM_RUMBLE_RAW").as_deref() != Ok("0");
        let trace = std::env::var("NEXIUM_RUMBLE_TRACE").as_deref() == Ok("1");
        let shared: SharedState = Arc::new((Mutex::new(Shared::default()), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let worker_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("nexium-rumble".to_string())
            .spawn(move || {
                let _done = done_tx;
                Worker::new(worker_shared, hid, raw, trace).run();
            })
            .map_err(|error| log::warn!("rumble: worker thread failed to start: {}", error))
            .ok();
        log::info!("rumble: output started (direct HID {}, raw effects {})", hid, raw);
        Self {
            shared,
            thread,
            done: Some(done_rx),
            hid_ref,
            published: Vec::new(),
            was_live: false,
            stopped: false,
            died_reported: false,
        }
    }

    fn worker_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|thread| !thread.is_finished())
    }

    pub fn sync_pads(&mut self, pads: &[Option<&Gamepad>]) -> bool {
        let released = std::mem::take(&mut self.shared.0.lock().release);
        for gamepad in released {
            gamepad.close();
        }
        if self.stopped || self.died_reported {
            return false;
        }
        if !self.worker_running() {
            self.died_reported = true;
            self.published.clear();
            return true;
        }
        let mut ids = Vec::new();
        let mut connected = Vec::new();
        for (index, pad) in pads.iter().enumerate() {
            let Some(pad) = pad.filter(|pad| pad.connected()) else {
                continue;
            };
            let id = unsafe { SDL_GetGamepadID(pad.raw()) }.0;
            if id != 0 && !ids.iter().any(|(known, _)| *known == id) {
                ids.push((id, index > 0));
                connected.push((pad, index > 0));
            }
        }
        if ids == self.published {
            return false;
        }
        let infos: Vec<PadInfo> = connected
            .into_iter()
            .filter_map(|(pad, motion_only)| PadInfo::from_pad(pad, motion_only))
            .collect();
        let replaced = {
            let (lock, cv) = &*self.shared;
            let mut shared = lock.lock();
            let replaced = shared.pending.replace(infos);
            shared.wake = true;
            cv.notify_one();
            replaced
        };
        for info in replaced.into_iter().flatten() {
            info.gamepad.close();
        }
        self.published = ids;
        false
    }

    pub fn set_guest(&mut self, live: bool, strength_percent: u8) {
        let strength = f32::from(strength_percent.min(100)) / 100.0;
        let live = live && strength > 0.0;
        let now = Instant::now();
        let (lock, cv) = &*self.shared;
        let mut shared = lock.lock();
        shared.strength = strength;
        shared.live_until = live.then(|| now + HEARTBEAT);
        if live && !self.was_live {
            shared.wake = true;
            cv.notify_one();
        }
        drop(shared);
        self.was_live = live;
    }

    pub fn preview(&self, strength_percent: u8) {
        if strength_percent == 0 {
            return;
        }
        let (lock, cv) = &*self.shared;
        let mut shared = lock.lock();
        shared.preview = Some((
            Instant::now() + PREVIEW_TIME,
            f32::from(strength_percent.min(100)) / 100.0,
        ));
        shared.wake = true;
        cv.notify_one();
    }

    pub fn pulse(&self, pad: u32, low: u16, high: u16, ms: u32) {
        let now = Instant::now();
        let (lock, cv) = &*self.shared;
        let mut shared = lock.lock();
        shared.pulses.retain(|pulse| pulse.until > now);
        shared.pulses.push(Pulse {
            pad,
            low,
            high,
            until: now + Duration::from_millis(u64::from(ms)),
        });
        shared.wake = true;
        cv.notify_one();
    }

    pub fn status_text(&self, enabled: bool) -> String {
        let outputs = {
            let shared = self.shared.0.lock();
            if shared.dead || (!self.stopped && !self.worker_running()) {
                return "Unavailable".to_string();
            }
            shared.status.clone()
        };
        status_text(enabled, &outputs)
    }

    pub fn alive(&self) -> bool {
        !self.stopped && self.worker_running()
    }

    pub fn stop(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        {
            let (lock, cv) = &*self.shared;
            let mut shared = lock.lock();
            shared.quit = true;
            shared.wake = true;
            cv.notify_all();
        }
        let joined = match self.done.take().map(|done| done.recv_timeout(STOP_WAIT)) {
            Some(Err(mpsc::RecvTimeoutError::Timeout)) => {
                log::warn!("rumble: worker did not stop within 300 ms");
                false
            }
            _ => {
                if let Some(thread) = self.thread.take() {
                    let _ = thread.join();
                }
                true
            }
        };
        std::thread::sleep(Duration::from_millis(35));
        unsafe { SDL_UpdateGamepads() };
        std::thread::sleep(Duration::from_millis(15));
        let (released, pending) = {
            let mut shared = self.shared.0.lock();
            (std::mem::take(&mut shared.release), shared.pending.take())
        };
        for gamepad in released {
            gamepad.close();
        }
        for info in pending.into_iter().flatten() {
            info.gamepad.close();
        }
        if joined && self.hid_ref {
            self.hid_ref = false;
            unsafe { SDL_hid_exit() };
        }
    }
}

impl Drop for RumbleOutput {
    fn drop(&mut self) {
        self.stop();
    }
}

struct HidDev {
    raw: *mut SDL_hid_device,
    counter: u8,
}

unsafe impl Send for HidDev {}

impl HidDev {
    fn open(path: &CStr, vendor: u16, product: u16) -> Option<HidDev> {
        let raw = unsafe { SDL_hid_open_path(path.as_ptr()) };
        if raw.is_null() {
            return None;
        }
        let dev = HidDev { raw, counter: 0 };
        let info = unsafe { SDL_hid_get_device_info(raw) };
        if info.is_null() {
            return None;
        }
        let (found_vendor, found_product) = unsafe { ((*info).vendor_id, (*info).product_id) };
        (found_vendor == vendor && found_product == product).then_some(dev)
    }

    fn send(&mut self, left: [u8; 4], right: [u8; 4], stats: &mut Stats) -> bool {
        let bytes = hd_rumble::report(self.counter, left, right);
        self.counter = (self.counter + 1) & 0x0F;
        let started = Instant::now();
        let written = unsafe { SDL_hid_write(self.raw, bytes.as_ptr(), bytes.len()) };
        stats.hid(started.elapsed());
        stats.last = [left, right];
        written >= 0
    }
}

impl Drop for HidDev {
    fn drop(&mut self) {
        unsafe { SDL_hid_close(self.raw) };
    }
}

fn open_pair() -> Option<[HidDev; 2]> {
    let mut left: Vec<CString> = Vec::new();
    let mut right: Vec<CString> = Vec::new();
    unsafe {
        let list = SDL_hid_enumerate(NINTENDO, 0);
        let mut node = list;
        while !node.is_null() {
            let info = &*node;
            if !info.path.is_null() {
                let path = CStr::from_ptr(info.path).to_owned();
                let side = match info.product_id {
                    0x2006 => Some(&mut left),
                    0x2007 => Some(&mut right),
                    _ => None,
                };
                if let Some(side) = side {
                    if !side.contains(&path) {
                        side.push(path);
                    }
                }
            }
            node = info.next;
        }
        if !list.is_null() {
            SDL_hid_free_enumeration(list);
        }
    }
    if left.len() != 1 || right.len() != 1 {
        return None;
    }
    let left = HidDev::open(&left[0], NINTENDO, 0x2006)?;
    let right = HidDev::open(&right[0], NINTENDO, 0x2007)?;
    Some([left, right])
}

fn send_effect(gamepad: *mut SDL_Gamepad, left: [u8; 4], right: [u8; 4]) -> bool {
    let bytes = hd_rumble::report(0, left, right);
    unsafe { SDL_SendGamepadEffect(gamepad, bytes.as_ptr().cast(), bytes.len() as i32) }
}

fn encode(value: VibrationValue, neutral: [u8; 4], cap: f32) -> [u8; 4] {
    hd_rumble::encode_side(hd_rumble::limit(value, cap), neutral)
}

fn halves(frame: [VibrationValue; 2], merge: bool) -> [VibrationValue; 2] {
    if merge {
        let both = frame[0].louder_bands(frame[1]);
        [both, both]
    } else {
        frame
    }
}

fn sleep_until(at: Instant) {
    let now = Instant::now();
    if at > now {
        std::thread::sleep(at - now);
    }
}

fn in_hand(
    held_at: &mut Option<Instant>,
    now: Instant,
    settled: bool,
    moving: impl FnOnce() -> bool,
) -> bool {
    if settled && moving() {
        *held_at = Some(now);
    }
    held_at.is_some_and(|at| now.saturating_duration_since(at) < HAND_TIME)
}

fn pulse_for(pulses: &[(u32, u16, u16)], id: u32, strength: f32) -> (u16, u16) {
    if !(strength > 0.0) {
        return (0, 0);
    }
    let scale = |level: u16| (f32::from(level) * strength).round() as u16;
    pulses
        .iter()
        .filter(|pulse| pulse.0 == id)
        .fold((0, 0), |level, pulse| (level.0.max(scale(pulse.1)), level.1.max(scale(pulse.2))))
}

fn hex(bytes: [u8; 4]) -> String {
    format!("{:02X} {:02X} {:02X} {:02X}", bytes[0], bytes[1], bytes[2], bytes[3])
}

fn millis(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
}

#[derive(Default)]
struct Stats {
    writes: u32,
    stops: u32,
    fails: u32,
    hid_count: u32,
    hid_total: Duration,
    hid_min: Option<Duration>,
    hid_max: Duration,
    last: [[u8; 4]; 2],
    levels: (u16, u16),
}

impl Stats {
    fn hid(&mut self, elapsed: Duration) {
        self.hid_count += 1;
        self.hid_total += elapsed;
        self.hid_min = Some(self.hid_min.map_or(elapsed, |min| min.min(elapsed)));
        self.hid_max = self.hid_max.max(elapsed);
    }
}

enum Map {
    PairSplit,
    Band { curve: bool },
}

enum Sink {
    None,
    HidSingle {
        dev: HidDev,
        merge: bool,
        neutral: [u8; 4],
        pacer: Pacer<[VibrationValue; 2]>,
    },
    HidPair {
        dev: [HidDev; 2],
        pacer: [Pacer<VibrationValue>; 2],
    },
    SdlHd {
        merge: bool,
        neutral: [u8; 4],
        pacer: Pacer<[VibrationValue; 2]>,
    },
    Sdl {
        map: Map,
        pacer: Pacer<(u16, u16)>,
        fails: u8,
        paused_until: Option<Instant>,
    },
}

impl Sink {
    fn hid_single(dev: HidDev, merge: bool, neutral: [u8; 4]) -> Sink {
        Sink::HidSingle {
            dev,
            merge,
            neutral,
            pacer: Pacer::new(HID_PACE),
        }
    }

    fn hid_pair(dev: [HidDev; 2]) -> Sink {
        Sink::HidPair {
            dev,
            pacer: [Pacer::new(HID_PACE), Pacer::new(HID_PACE)],
        }
    }

    fn sdl_hd(merge: bool, neutral: [u8; 4]) -> Sink {
        Sink::SdlHd {
            merge,
            neutral,
            pacer: Pacer::new(SDL_HD_PACE),
        }
    }

    fn sdl(map: Map) -> Sink {
        Sink::Sdl {
            map,
            pacer: Pacer::new(SDL_PACE),
            fails: 0,
            paused_until: None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Sink::None => "None",
            Sink::HidSingle { .. } => "HidSingle",
            Sink::HidPair { .. } => "HidPair",
            Sink::SdlHd { .. } => "SdlHd",
            Sink::Sdl {
                map: Map::PairSplit,
                ..
            } => "PairSplit",
            Sink::Sdl { .. } => "Band",
        }
    }

    fn is_hid(&self) -> bool {
        matches!(self, Sink::HidSingle { .. } | Sink::HidPair { .. })
    }

    fn busy(&self) -> bool {
        match self {
            Sink::None => false,
            Sink::HidSingle { pacer, .. } | Sink::SdlHd { pacer, .. } => pacer.busy(),
            Sink::HidPair { pacer, .. } => pacer.iter().any(Pacer::busy),
            Sink::Sdl { pacer, .. } => pacer.busy(),
        }
    }

    fn last_write(&self) -> Option<Instant> {
        match self {
            Sink::None => None,
            Sink::HidSingle { pacer, .. } | Sink::SdlHd { pacer, .. } => pacer.last_at(),
            Sink::HidPair { pacer, .. } => pacer.iter().filter_map(Pacer::last_at).max(),
            Sink::Sdl { pacer, .. } => pacer.last_at(),
        }
    }

    fn owe_stop(&mut self) {
        match self {
            Sink::None => {}
            Sink::HidSingle { pacer, .. } | Sink::SdlHd { pacer, .. } => pacer.owe_stop(),
            Sink::HidPair { pacer, .. } => pacer.iter_mut().for_each(Pacer::owe_stop),
            Sink::Sdl { pacer, .. } => pacer.owe_stop(),
        }
    }

    fn mode(&self, now: Instant) -> RumbleMode {
        match self {
            Sink::HidSingle { .. } | Sink::HidPair { .. } | Sink::SdlHd { .. } => RumbleMode::Hd,
            Sink::Sdl { paused_until, .. } if paused_until.is_some_and(|until| now < until) => {
                RumbleMode::Unsupported
            }
            Sink::Sdl { .. } => RumbleMode::Standard,
            Sink::None => RumbleMode::Unsupported,
        }
    }
}

struct Output {
    id: u32,
    name: String,
    vendor: u16,
    product: u16,
    path: Option<CString>,
    route: Route,
    raw: bool,
    gamepad: GamepadRef,
    sink: Sink,
    motion_only: bool,
    held_at: Option<Instant>,
    retry_at: Option<Instant>,
    retry_step: usize,
    healthy_since: Option<Instant>,
    hold_until: Option<Instant>,
    pause_warned: bool,
    stats: Stats,
}

impl Output {
    fn build(info: PadInfo, hid: bool, raw: bool, now: Instant) -> Output {
        let meta = PadMeta {
            vendor: info.vendor,
            product: info.product,
            kind: info.kind,
            cap_rumble: info.cap_rumble,
        };
        let mut output = Output {
            id: info.id,
            name: info.name,
            vendor: info.vendor,
            product: info.product,
            path: info.path,
            route: route(&meta, hid, raw),
            raw,
            gamepad: info.gamepad,
            sink: Sink::None,
            motion_only: info.motion_only,
            held_at: None,
            retry_at: None,
            retry_step: 0,
            healthy_since: None,
            hold_until: None,
            pause_warned: false,
            stats: Stats::default(),
        };
        output.sink = match output.route {
            Route::None => Sink::None,
            Route::HidPair | Route::HidSingle { .. } => match output.open_hid() {
                Some(sink) => {
                    output.healthy_since = Some(now);
                    sink
                }
                None => {
                    output.schedule_retry(now);
                    output.fallback()
                }
            },
            Route::SdlHd { merge, .. } => Sink::sdl_hd(merge, hd_rumble::neutral_for(output.product)),
            Route::PairSplit => Sink::sdl(Map::PairSplit),
            Route::Band { curve } => Sink::sdl(Map::Band { curve }),
        };
        log::info!(
            "rumble: {} ({:04x}:{:04x}) -> {}{}",
            output.name,
            output.vendor,
            output.product,
            output.sink.label(),
            if output.retry_at.is_some() {
                " (direct HID unavailable, retrying)"
            } else {
                ""
            }
        );
        output
    }

    fn open_hid(&self) -> Option<Sink> {
        match self.route {
            Route::HidPair => open_pair().map(Sink::hid_pair),
            Route::HidSingle { merge, .. } => self
                .path
                .as_deref()
                .and_then(|path| HidDev::open(path, self.vendor, self.product))
                .map(|dev| Sink::hid_single(dev, merge, hd_rumble::neutral_for(self.product))),
            _ => None,
        }
    }

    fn fallback(&self) -> Sink {
        match self.route {
            Route::HidPair => Sink::sdl(Map::PairSplit),
            Route::HidSingle { merge, .. } if self.raw => {
                Sink::sdl_hd(merge, hd_rumble::neutral_for(self.product))
            }
            _ => Sink::sdl(Map::Band { curve: false }),
        }
    }

    fn schedule_retry(&mut self, now: Instant) {
        if self.retry_at.is_none() {
            self.retry_at = Some(now + RETRY_DELAYS[self.retry_step]);
            self.retry_step = (self.retry_step + 1).min(RETRY_DELAYS.len() - 1);
        }
    }

    fn hears_guest(&mut self, now: Instant) -> bool {
        if !self.motion_only {
            return true;
        }
        let settled = !self.sink.busy()
            && self
                .sink
                .last_write()
                .map_or(true, |at| now.saturating_duration_since(at) >= HAND_SETTLE);
        in_hand(&mut self.held_at, now, settled, || {
            !MotionSource::ALL.into_iter().all(hid_motion::source_at_rest)
        })
    }

    fn mode(&self, now: Instant) -> RumbleMode {
        shown_mode(self.route, self.sink.mode(now))
    }

    fn tick(
        &mut self,
        now: Instant,
        hd: &[VibrationValue; 2],
        sd: &[VibrationValue; 2],
        pulse: (u16, u16),
        cap: f32,
        quiet: bool,
    ) {
        if self.hold_until.is_some_and(|at| now < at) {
            return;
        }
        let gamepad = self.gamepad.raw();
        let name = &self.name;
        let stats = &mut self.stats;
        let pause_warned = &mut self.pause_warned;
        let failed = match &mut self.sink {
            Sink::None => None,
            Sink::HidSingle {
                dev,
                merge,
                neutral,
                pacer,
            } => {
                let extra = hd_rumble::pulse_value(pulse.0, pulse.1);
                let frame = [hd[0].louder_bands(extra), hd[1].louder_bands(extra)];
                match pacer.step(Instant::now(), frame) {
                    Step::Idle => None,
                    Step::Write(frame) => {
                        stats.writes += 1;
                        let [left, right] = halves(frame, *merge);
                        let (left, right) = (encode(left, *neutral, cap), encode(right, *neutral, cap));
                        (!dev.send(left, right, stats)).then_some(0)
                    }
                    Step::Stop => {
                        stats.stops += 1;
                        (!dev.send(*neutral, *neutral, stats)).then_some(0)
                    }
                }
            }
            Sink::HidPair { dev, pacer } => {
                let extra = hd_rumble::pulse_pair(pulse.0, pulse.1);
                let mut failed = None;
                for side in 0..2 {
                    let bytes = match pacer[side].step(Instant::now(), hd[side].louder_bands(extra[side])) {
                        Step::Idle => continue,
                        Step::Write(value) => {
                            stats.writes += 1;
                            encode(value, NEUTRAL, cap)
                        }
                        Step::Stop => {
                            stats.stops += 1;
                            NEUTRAL
                        }
                    };
                    if !dev[side].send(bytes, bytes, stats) {
                        failed = Some(side);
                        break;
                    }
                }
                failed
            }
            Sink::SdlHd {
                merge,
                neutral,
                pacer,
            } => {
                let extra = hd_rumble::pulse_value(pulse.0, pulse.1);
                let frame = [hd[0].louder_bands(extra), hd[1].louder_bands(extra)];
                let bytes = match pacer.step(Instant::now(), frame) {
                    Step::Idle => None,
                    Step::Write(frame) => {
                        stats.writes += 1;
                        let [left, right] = halves(frame, *merge);
                        Some((encode(left, *neutral, cap), encode(right, *neutral, cap)))
                    }
                    Step::Stop => {
                        stats.stops += 1;
                        Some((*neutral, *neutral))
                    }
                };
                bytes.and_then(|(left, right)| {
                    stats.last = [left, right];
                    (!send_effect(gamepad, left, right)).then_some(0)
                })
            }
            Sink::Sdl {
                map,
                pacer,
                fails,
                paused_until,
            } => {
                if paused_until.is_some_and(|until| now < until) {
                    return;
                }
                if paused_until.take().is_some() {
                    *fails = 0;
                }
                let mapped = match map {
                    Map::PairSplit => hd_rumble::pair_levels(hd[0], hd[1]),
                    Map::Band { curve } => hd_rumble::band_levels(sd[0], sd[1], *curve),
                };
                pacer.set_pace(if quiet { SDL_QUIET_PACE } else { SDL_PACE });
                match pacer.step(now, mapped.merge(pulse)) {
                    Step::Idle => {}
                    Step::Write((low, high)) => {
                        stats.writes += 1;
                        stats.levels = (low, high);
                        if unsafe { SDL_RumbleGamepad(gamepad, low, high, SDL_DURATION_MS) } {
                            *fails = 0;
                        } else {
                            stats.fails += 1;
                            *fails += 1;
                            if *fails >= 3 {
                                *paused_until = Some(now + SDL_PAUSE);
                                if !*pause_warned {
                                    *pause_warned = true;
                                    log::warn!("rumble: {} rejected rumble; pausing it for 10 s", name);
                                }
                            }
                        }
                    }
                    Step::Stop => {
                        stats.stops += 1;
                        stats.levels = (0, 0);
                        unsafe { SDL_RumbleGamepad(gamepad, 0, 0, 0) };
                    }
                }
                None
            }
        };
        if let Some(side) = failed {
            self.degrade(now, side);
        }
        if self.sink.is_hid()
            && self
                .healthy_since
                .is_some_and(|since| now.saturating_duration_since(since) >= HEALTHY_RESET)
        {
            self.retry_step = 0;
        }
    }

    fn degrade(&mut self, now: Instant, failed: usize) {
        let previous = std::mem::replace(&mut self.sink, Sink::None);
        self.sink = match previous {
            Sink::HidPair { mut dev, pacer } => {
                for side in 0..2 {
                    if side == failed || !pacer[side].busy() {
                        continue;
                    }
                    if let Some(last) = pacer[side].last_at() {
                        sleep_until(last + HID_PACE.spacing);
                    }
                    let _ = dev[side].send(NEUTRAL, NEUTRAL, &mut self.stats);
                }
                drop(dev);
                self.fallback()
            }
            Sink::HidSingle { dev, pacer, .. } => {
                drop(dev);
                let mut sink = self.fallback();
                if pacer.busy() {
                    sink.owe_stop();
                }
                sink
            }
            Sink::SdlHd { .. } => Sink::sdl(Map::Band { curve: false }),
            other => other,
        };
        self.stats.fails += 1;
        self.healthy_since = None;
        self.hold_until = Some(Instant::now() + HID_PACE.spacing);
        if matches!(self.route, Route::HidPair | Route::HidSingle { .. }) {
            self.schedule_retry(now);
        }
        log::warn!("rumble: {} write failed, switched to {}", self.name, self.sink.label());
    }

    fn stop_sdl(&mut self) {
        let gamepad = self.gamepad.raw();
        match &self.sink {
            Sink::Sdl { pacer, .. } if pacer.last_at().is_some() => {
                unsafe { SDL_RumbleGamepad(gamepad, 0, 0, 0) };
            }
            Sink::SdlHd { neutral, pacer, .. } => {
                if let Some(last) = pacer.last_at() {
                    sleep_until(last + SDL_HD_PACE.spacing);
                    send_effect(gamepad, *neutral, *neutral);
                }
            }
            _ => {}
        }
    }

    fn retry_if_due(&mut self, now: Instant) {
        if !self.retry_at.is_some_and(|at| now >= at) {
            return;
        }
        self.retry_at = None;
        match self.open_hid() {
            Some(mut sink) => {
                if self.sink.busy() {
                    sink.owe_stop();
                }
                self.stop_sdl();
                self.sink = sink;
                self.hold_until = Some(Instant::now() + SDL_SETTLE);
                self.healthy_since = Some(now);
                log::info!("rumble: {} is back on {}", self.name, self.sink.label());
            }
            None => self.schedule_retry(now),
        }
    }

    fn silence(&mut self, deadline: Instant) {
        let gamepad = self.gamepad.raw();
        match std::mem::replace(&mut self.sink, Sink::None) {
            Sink::HidSingle {
                mut dev,
                neutral,
                pacer,
                ..
            } => {
                if let Some(last) = pacer.last_at() {
                    sleep_until((last + HID_PACE.spacing).min(deadline));
                    let _ = dev.send(neutral, neutral, &mut self.stats);
                    sleep_until((Instant::now() + HID_PACE.spacing).min(deadline));
                    let _ = dev.send(neutral, neutral, &mut self.stats);
                }
            }
            Sink::HidPair { mut dev, pacer } => {
                if let Some(last) = pacer.iter().filter_map(Pacer::last_at).max() {
                    sleep_until((last + HID_PACE.spacing).min(deadline));
                    for side in dev.iter_mut() {
                        let _ = side.send(NEUTRAL, NEUTRAL, &mut self.stats);
                    }
                    sleep_until((Instant::now() + HID_PACE.spacing).min(deadline));
                    for side in dev.iter_mut() {
                        let _ = side.send(NEUTRAL, NEUTRAL, &mut self.stats);
                    }
                }
            }
            Sink::SdlHd { neutral, pacer, .. } => {
                if let Some(last) = pacer.last_at() {
                    sleep_until((last + SDL_HD_PACE.spacing).min(deadline));
                    send_effect(gamepad, neutral, neutral);
                }
            }
            Sink::Sdl { .. } => {
                unsafe { SDL_RumbleGamepad(gamepad, 0, 0, 0) };
            }
            Sink::None => {}
        }
    }

    fn trace(&self, index: usize, now: Instant, seconds: f32) -> String {
        let stats = &self.stats;
        let hid = if stats.hid_count > 0 {
            format!(
                " hid={:.1}/{:.1}/{:.1}ms",
                millis(stats.hid_min.unwrap_or_default()),
                millis(stats.hid_total) / stats.hid_count as f32,
                millis(stats.hid_max)
            )
        } else {
            String::new()
        };
        let last = match self.sink {
            Sink::Sdl { .. } => format!(" lvl={}/{}", stats.levels.0, stats.levels.1),
            _ => format!(" L[{}] R[{}]", hex(stats.last[0]), hex(stats.last[1])),
        };
        format!(
            "[{}] {} {} w={:.0}/s s={:.0}/s fail={}{}{} {}",
            index,
            self.sink.label(),
            self.name.trim_start_matches("Nintendo Switch "),
            stats.writes as f32 / seconds,
            stats.stops as f32 / seconds,
            stats.fails,
            hid,
            last,
            self.mode(now).label()
        )
    }
}

struct Worker {
    shared: SharedState,
    hid: bool,
    raw: bool,
    trace: bool,
    outputs: Vec<Output>,
    retiring: Vec<Output>,
    status: Vec<OutputStatus>,
    trace_at: Instant,
}

impl Worker {
    fn new(shared: SharedState, hid: bool, raw: bool, trace: bool) -> Worker {
        Worker {
            shared,
            hid,
            raw,
            trace,
            outputs: Vec::new(),
            retiring: Vec::new(),
            status: Vec::new(),
            trace_at: Instant::now(),
        }
    }

    fn run(mut self) {
        loop {
            match std::panic::catch_unwind(AssertUnwindSafe(|| self.tick())) {
                Ok(true) => {}
                Ok(false) => return,
                Err(_) => {
                    log::error!("rumble: output worker panicked; controller vibration is off");
                    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| self.final_silence()));
                    let released = self.take_refs();
                    let mut shared = self.shared.0.lock();
                    shared.dead = true;
                    shared.release.extend(released);
                    return;
                }
            }
        }
    }

    fn take_refs(&mut self) -> Vec<GamepadRef> {
        self.outputs
            .drain(..)
            .chain(self.retiring.drain(..))
            .map(|output| output.gamepad)
            .collect()
    }

    fn final_silence(&mut self) {
        let deadline = Instant::now() + SILENCE_BUDGET;
        for output in self.outputs.iter_mut().chain(self.retiring.iter_mut()) {
            output.silence(deadline);
        }
    }

    fn release(&self, refs: Vec<GamepadRef>) {
        if !refs.is_empty() {
            self.shared.0.lock().release.extend(refs);
        }
    }

    fn reconcile(&mut self, infos: Vec<PadInfo>, now: Instant) {
        let mut previous = std::mem::take(&mut self.outputs);
        let mut released = Vec::new();
        for info in infos {
            let existing = previous
                .iter()
                .position(|output| output.id == info.id)
                .map(|index| previous.remove(index))
                .or_else(|| {
                    self.retiring
                        .iter()
                        .position(|output| output.id == info.id)
                        .map(|index| self.retiring.remove(index))
                });
            match existing {
                Some(mut output) => {
                    output.motion_only = info.motion_only;
                    released.push(info.gamepad);
                    self.outputs.push(output);
                }
                None => self.outputs.push(Output::build(info, self.hid, self.raw, now)),
            }
        }
        self.retiring.extend(previous);
        self.release(released);
    }

    fn tick(&mut self) -> bool {
        let now = Instant::now();
        let (pending, live, strength, preview, pulses, quit) = {
            let mut shared = self.shared.0.lock();
            shared.wake = false;
            shared.pulses.retain(|pulse| pulse.until > now);
            (
                shared.pending.take(),
                shared.live_until.is_some_and(|until| until > now),
                shared.strength,
                shared.preview.filter(|(until, _)| *until > now).map(|(_, level)| level),
                shared
                    .pulses
                    .iter()
                    .map(|pulse| (pulse.pad, pulse.low, pulse.high))
                    .collect::<Vec<_>>(),
                shared.quit,
            )
        };
        let guest = hid_vibration::take_player1();
        if quit {
            self.final_silence();
            let mut released = self.take_refs();
            released.extend(pending.into_iter().flatten().map(|info| info.gamepad));
            self.release(released);
            return false;
        }
        if let Some(infos) = pending {
            self.reconcile(infos, now);
        }
        let mut hd = [VibrationValue::DEFAULT; 2];
        let mut sd = [VibrationValue::DEFAULT; 2];
        let mut alone = [VibrationValue::DEFAULT; 2];
        for side in 0..2 {
            if live {
                let value = guest[side].sanitized();
                hd[side] = hd_rumble::protect(value).scaled(strength);
                sd[side] = value.scaled(strength);
            }
            if let Some(level) = preview {
                let shaped = hd_rumble::PREVIEW.scaled(level);
                hd[side] = hd[side].louder_bands(shaped);
                sd[side] = sd[side].louder_bands(shaped);
                alone[side] = shaped;
            }
        }
        let cap = preview.map_or(strength, |level| level.max(strength));
        for output in &mut self.outputs {
            let pulse = pulse_for(&pulses, output.id, strength);
            if output.hears_guest(now) {
                output.tick(now, &hd, &sd, pulse, cap, !live);
            } else {
                output.tick(now, &alone, &alone, pulse, cap, true);
            }
            output.retry_if_due(now);
        }
        let silent = [VibrationValue::DEFAULT; 2];
        let mut released = Vec::new();
        let mut index = 0;
        while index < self.retiring.len() {
            let output = &mut self.retiring[index];
            output.tick(now, &silent, &silent, (0, 0), cap, true);
            let settled = !output.sink.busy()
                && match output.sink {
                    Sink::Sdl { .. } | Sink::SdlHd { .. } => output
                        .sink
                        .last_write()
                        .map_or(true, |at| now.saturating_duration_since(at) >= SDL_SETTLE),
                    _ => true,
                };
            if settled {
                released.push(self.retiring.remove(index).gamepad);
            } else {
                index += 1;
            }
        }
        self.release(released);
        self.publish_status(now);
        if self.trace {
            self.trace_line(now);
        }
        let busy = (!self.outputs.is_empty() && (live || preview.is_some() || !pulses.is_empty()))
            || !self.retiring.is_empty()
            || self.outputs.iter().any(|output| {
                output.sink.busy() || output.retry_at.is_some_and(|at| at <= now + IDLE_WAIT)
            });
        if busy {
            std::thread::sleep(TICK);
        } else {
            let (lock, cv) = &*self.shared;
            let mut shared = lock.lock();
            if !shared.wake && !shared.quit && shared.pending.is_none() {
                cv.wait_for(&mut shared, IDLE_WAIT);
            }
        }
        true
    }

    fn publish_status(&mut self, now: Instant) {
        let status: Vec<OutputStatus> = self
            .outputs
            .iter()
            .map(|output| OutputStatus {
                mode: output.mode(now),
                name: output.name.clone(),
            })
            .collect();
        if status != self.status {
            self.shared.0.lock().status = status.clone();
            self.status = status;
        }
    }

    fn trace_line(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.trace_at);
        if elapsed < Duration::from_secs(1) {
            return;
        }
        self.trace_at = now;
        if self.outputs.is_empty() && self.retiring.is_empty() {
            return;
        }
        let seconds = elapsed.as_secs_f32();
        let parts: Vec<String> = self
            .outputs
            .iter_mut()
            .chain(self.retiring.iter_mut())
            .enumerate()
            .map(|(index, output)| {
                let part = output.trace(index, now, seconds);
                output.stats = Stats::default();
                part
            })
            .collect();
        log::info!("rumble: {}", parts.join(" | "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdl3::sys::gamepad::{
        SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, SDL_GAMEPAD_TYPE_UNKNOWN, SDL_GAMEPAD_TYPE_XBOXONE,
    };

    fn pad(vendor: u16, product: u16, kind: i32, cap_rumble: bool) -> PadMeta {
        PadMeta {
            vendor,
            product,
            kind,
            cap_rumble,
        }
    }

    #[test]
    fn routes_follow_the_device_class() {
        let pair = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR.0;
        let pro = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO.0;
        let unknown = SDL_GAMEPAD_TYPE_UNKNOWN.0;
        assert_eq!(route(&pad(0x057E, 0x2008, pair, true), true, true), Route::HidPair);
        assert_eq!(route(&pad(0x057E, 0x2008, pair, true), false, true), Route::PairSplit);
        assert_eq!(route(&pad(0x057E, 0x2068, pair, true), true, true), Route::PairSplit);
        assert_eq!(route(&pad(0x057E, 0x2008, 0, true), true, true), Route::HidPair);
        assert_eq!(
            route(&pad(0x057E, 0x2009, pro, true), true, true),
            Route::HidSingle {
                merge: false,
                n64: false
            }
        );
        assert_eq!(
            route(&pad(0x057E, 0x2009, pro, true), false, true),
            Route::SdlHd {
                merge: false,
                n64: false
            }
        );
        assert_eq!(
            route(&pad(0x057E, 0x2009, pro, true), false, false),
            Route::Band { curve: false }
        );
        assert_eq!(
            route(&pad(0x057E, 0x2006, unknown, true), true, true),
            Route::HidSingle {
                merge: true,
                n64: false
            }
        );
        assert_eq!(
            route(&pad(0x057E, 0x2019, pro, false), true, true),
            Route::HidSingle {
                merge: true,
                n64: true
            }
        );
        for (hid, raw) in [(true, true), (false, false)] {
            assert_eq!(route(&pad(0x057E, 0x2017, pro, false), hid, raw), Route::None);
            assert_eq!(
                route(&pad(0x045E, 0x0B12, SDL_GAMEPAD_TYPE_XBOXONE.0, true), hid, raw),
                Route::Band { curve: true }
            );
            assert_eq!(
                route(&pad(0x054C, 0x0CE6, SDL_GAMEPAD_TYPE_PS5.0, true), hid, raw),
                Route::Band { curve: false }
            );
            assert_eq!(
                route(&pad(0x057E, 0x2069, pro, true), hid, raw),
                Route::Band { curve: true }
            );
            assert_eq!(route(&pad(0x045E, 0x0B12, unknown, false), hid, raw), Route::None);
        }
    }

    #[test]
    fn status_lists_every_output() {
        let output = |mode, name: &str| OutputStatus {
            mode,
            name: name.to_string(),
        };
        assert_eq!(status_text(false, &[output(RumbleMode::Hd, "Pad")]), "Off");
        assert_eq!(status_text(true, &[]), "No controller");
        assert_eq!(
            status_text(true, &[output(RumbleMode::Hd, "Nintendo Switch Joy-Con (L/R)")]),
            "HD \u{b7} Joy-Con (L/R)"
        );
        assert_eq!(
            status_text(
                true,
                &[
                    output(RumbleMode::Standard, "Xbox Series X Controller"),
                    output(RumbleMode::Hd, "Nintendo Switch Pro Controller"),
                ]
            ),
            "Standard \u{b7} Xbox Series X Controller + HD \u{b7} Pro Controller"
        );
        assert_eq!(
            status_text(true, &[output(RumbleMode::Unsupported, "SNES Controller")]),
            "No rumble \u{b7} SNES Controller"
        );
    }

    #[test]
    fn n64_rumble_shows_as_standard() {
        let n64 = Route::HidSingle {
            merge: true,
            n64: true,
        };
        let n64_sdl = Route::SdlHd {
            merge: true,
            n64: true,
        };
        let pro = Route::HidSingle {
            merge: false,
            n64: false,
        };
        assert_eq!(shown_mode(n64, RumbleMode::Hd), RumbleMode::Standard);
        assert_eq!(shown_mode(n64_sdl, RumbleMode::Hd), RumbleMode::Standard);
        assert_eq!(shown_mode(n64_sdl, RumbleMode::Unsupported), RumbleMode::Unsupported);
        assert_eq!(shown_mode(pro, RumbleMode::Hd), RumbleMode::Hd);
        assert_eq!(shown_mode(Route::HidPair, RumbleMode::Hd), RumbleMode::Hd);
    }

    #[test]
    fn motion_pad_hears_the_game_only_while_in_hand() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let mut held_at = None;
        assert!(!in_hand(&mut held_at, at(0), true, || false));
        assert!(!in_hand(&mut held_at, at(10), false, || unreachable!()));
        assert!(in_hand(&mut held_at, at(20), true, || true));
        assert_eq!(held_at, Some(at(20)));
        assert!(in_hand(&mut held_at, at(2500), false, || unreachable!()));
        assert!(in_hand(&mut held_at, at(3000), true, || false));
        assert!(!in_hand(&mut held_at, at(3020), false, || unreachable!()));
        assert_eq!(held_at, Some(at(20)));
        assert!(in_hand(&mut held_at, at(4000), true, || true));
        assert!(!in_hand(&mut held_at, at(7000), true, || false));
    }

    #[test]
    fn pulses_scale_with_strength_and_target_one_pad() {
        let pulses = [(7, 20000, 10000), (7, 5000, 28000), (9, 65535, 65535)];
        assert_eq!(pulse_for(&pulses, 7, 1.0), (20000, 28000));
        assert_eq!(pulse_for(&pulses, 7, 0.5), (10000, 14000));
        assert_eq!(pulse_for(&pulses, 7, 0.0), (0, 0));
        assert_eq!(pulse_for(&pulses, 8, 1.0), (0, 0));
    }
}

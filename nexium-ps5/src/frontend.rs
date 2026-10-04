use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use nexium_core::hid_state::{
    self, ControllerInput, KeyboardInput, MouseInput, TouchInput, NPAD_BUTTON_A, NPAD_BUTTON_B, NPAD_BUTTON_DOWN, NPAD_BUTTON_L,
    NPAD_BUTTON_LEFT, NPAD_BUTTON_MINUS, NPAD_BUTTON_PLUS, NPAD_BUTTON_R, NPAD_BUTTON_RIGHT, NPAD_BUTTON_STICK_L,
    NPAD_BUTTON_STICK_R, NPAD_BUTTON_UP, NPAD_BUTTON_X, NPAD_BUTTON_Y, NPAD_BUTTON_ZL, NPAD_BUTTON_ZR,
};
use nexium_cpu::CpuBackendKind;
use nexium_runner::boot::{EmulationHandle, Frame};

use crate::audio::PullAudioOut;
use crate::console::{ensure_shared_dir, DATA_ROOT};
use crate::display::{write_bmp, Display, PUSH_FLOATS};
use crate::pad::{self, Pad, PadState};

const STICK_RANGE: f32 = 32_767.0;
const EXIT_HOLD: Duration = Duration::from_secs(2);

pub struct LaunchConfig {
    pub game: Option<PathBuf>,
    pub docked: bool,
    pub volume: f32,
    pub exit_after: Option<Duration>,
    pub snapshot_every: Option<Duration>,
    pub snapshot_limit: u32,
    pub log_level: log::LevelFilter,
}

impl LaunchConfig {
    pub fn load() -> Self {
        let mut cfg = Self {
            game: None,
            docked: true,
            volume: 1.0,
            exit_after: None,
            snapshot_every: None,
            snapshot_limit: 120,
            log_level: log::LevelFilter::Info,
        };
        let path = format!("{DATA_ROOT}/launch.txt");
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let Some((key, value)) = line.split_once('=') else { continue };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "game" => cfg.game = Some(PathBuf::from(value)),
                "docked" => cfg.docked = value != "0",
                "volume" => cfg.volume = value.parse().unwrap_or(1.0),
                "exit_after" => cfg.exit_after = value.parse().ok().map(Duration::from_secs),
                "snapshot_every" => cfg.snapshot_every = value.parse().ok().map(Duration::from_secs_f32),
                "snapshot_limit" => cfg.snapshot_limit = value.parse().unwrap_or(120),
                "log" => cfg.log_level = value.parse().unwrap_or(log::LevelFilter::Info),
                _ => {
                    if let Some(var) = key.strip_prefix("env.") {
                        std::env::set_var(var, value);
                    }
                }
            }
        }
        if cfg.game.is_none() {
            cfg.game = find_game();
        }
        cfg
    }
}

pub fn find_game() -> Option<PathBuf> {
    let dir = format!("{DATA_ROOT}/games");
    let mut games: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "nro" | "nsp" | "xci" | "dnsp" | "dxci" | "nca"))
        })
        .collect();
    games.sort();
    games.into_iter().next()
}

pub fn setup_environment() {
    let root = format!("{DATA_ROOT}/NeXium");
    let home = format!("{DATA_ROOT}/home");
    let appdata = format!("{DATA_ROOT}/appdata");
    for dir in [&root, &home, &appdata, &format!("{home}/.config")] {
        if let Err(e) = ensure_shared_dir(dir) {
            crate::klog!("frontend: {dir}: {e}");
        }
    }
    std::env::set_var("NEXIUM_DATA_ROOT", &root);
    std::env::set_var("HOME", &home);
    std::env::set_var("XDG_CONFIG_HOME", format!("{home}/.config"));
    std::env::set_var("APPDATA", &appdata);
    if std::env::var_os("MESA_SHADER_CACHE_DIR").is_none() {
        std::env::set_var("MESA_SHADER_CACHE_DIR", format!("{DATA_ROOT}/mesa-cache"));
    }
    for memo in ["NEXIUM_CPU_ENV_MEMO", "NEXIUM_RENDER_ENV_MEMO"] {
        if std::env::var_os(memo).is_none() {
            std::env::set_var(memo, "1");
        }
    }
    nexium_common::paths::init();
}

fn switch_buttons(p: &PadState) -> u64 {
    let map = [
        (pad::CIRCLE, NPAD_BUTTON_A),
        (pad::CROSS, NPAD_BUTTON_B),
        (pad::TRIANGLE, NPAD_BUTTON_X),
        (pad::SQUARE, NPAD_BUTTON_Y),
        (pad::L1, NPAD_BUTTON_L),
        (pad::R1, NPAD_BUTTON_R),
        (pad::L2, NPAD_BUTTON_ZL),
        (pad::R2, NPAD_BUTTON_ZR),
        (pad::L3, NPAD_BUTTON_STICK_L),
        (pad::R3, NPAD_BUTTON_STICK_R),
        (pad::OPTIONS, NPAD_BUTTON_PLUS),
        (pad::TOUCH_PAD, NPAD_BUTTON_MINUS),
        (pad::UP, NPAD_BUTTON_UP),
        (pad::DOWN, NPAD_BUTTON_DOWN),
        (pad::LEFT, NPAD_BUTTON_LEFT),
        (pad::RIGHT, NPAD_BUTTON_RIGHT),
    ];
    let mut out = map.iter().filter(|(ps, _)| p.buttons & ps != 0).fold(0u64, |acc, (_, sw)| acc | sw);
    if p.l2 > 64 {
        out |= NPAD_BUTTON_ZL;
    }
    if p.r2 > 64 {
        out |= NPAD_BUTTON_ZR;
    }
    out
}

fn deadzone(v: f32) -> f32 {
    if v.abs() < 0.08 {
        0.0
    } else {
        v
    }
}

pub fn push_hid(p: &PadState) {
    let state = hid_state::get_hid_state();
    let mut hid = state.lock();
    hid.update_input(ControllerInput {
        buttons: switch_buttons(p),
        stick_l_x: (deadzone(PadState::stick(p.lx)) * STICK_RANGE) as i32,
        stick_l_y: (-deadzone(PadState::stick(p.ly)) * STICK_RANGE) as i32,
        stick_r_x: (deadzone(PadState::stick(p.rx)) * STICK_RANGE) as i32,
        stick_r_y: (-deadzone(PadState::stick(p.ry)) * STICK_RANGE) as i32,
    });
    hid.update_devices(
        MouseInput { x: 0, y: 0, wheel_x: 0, wheel_y: 0, buttons: 0, connected: false },
        KeyboardInput::default(),
        TouchInput::default(),
    );
}

pub fn open_audio(volume: f32) -> Option<PullAudioOut> {
    let mut pull = nexium_runner::audio::init_host_audio_pull("PS5 AudioOut", crate::audio::RATE, 2, volume);
    match PullAudioOut::open(Box::new(move |grain| pull.fill_i16(grain))) {
        Ok(a) => Some(a),
        Err(e) => {
            crate::klog!("frontend: audio disabled: {e}");
            None
        }
    }
}

pub fn stop_emulation(emu: EmulationHandle) -> bool {
    let stop_started = Instant::now();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let mut emu = emu;
    let _ = std::thread::Builder::new().name("nexium-stop".into()).spawn(move || {
        emu.stop_blocking();
        let _ = done_tx.send(());
    });
    match done_rx.recv_timeout(Duration::from_secs(8)) {
        Ok(()) => {
            crate::klog!("frontend: emulation stopped in {:.2}s", stop_started.elapsed().as_secs_f32());
            true
        }
        Err(_) => {
            crate::klog!("frontend: emulation did not stop within 8s");
            false
        }
    }
}

fn idle_screen(display: &mut Display, started: Instant, frame: u32) -> Result<(), String> {
    let push: [f32; PUSH_FLOATS] = [
        started.elapsed().as_secs_f32(),
        display.extent.width as f32,
        display.extent.height as f32,
        frame as f32,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    display.frame(push, false).map(|_| ())
}

pub fn run(cfg: &LaunchConfig) -> u32 {
    let started = Instant::now();
    let Some(game) = cfg.game.clone() else {
        crate::klog!("frontend: no game in {DATA_ROOT}/launch.txt (game=/app0/data/...)");
        return 1;
    };
    let mut display = match Display::new(true) {
        Ok(d) => d,
        Err(e) => {
            crate::klog!("frontend: display init failed: {e}");
            return 1;
        }
    };
    let mut pad = match Pad::open() {
        Ok(p) => Some(p),
        Err(e) => {
            crate::klog!("frontend: no controller: {e}");
            None
        }
    };
    let audio = open_audio(cfg.volume);
    hid_state::set_docked(cfg.docked);
    if let Some(ms) = std::env::var("NEXIUM_PS5_SAMPLE_MS").ok().and_then(|v| v.parse().ok()) {
        crate::sampler::start(ms);
    }
    let path = game.to_string_lossy().into_owned();
    let meta = std::fs::metadata(&game);
    crate::klog!(
        "frontend: launching {path} ({}) docked={} backend=Dynarmic",
        match &meta {
            Ok(m) => format!("{} bytes", m.len()),
            Err(e) => format!("stat failed: {e}"),
        },
        cfg.docked
    );
    let mut emu = match EmulationHandle::new(&path, CpuBackendKind::Dynarmic, None) {
        Ok(e) => e,
        Err(e) => {
            crate::klog!("frontend: emulation failed to start: {e}");
            return 1;
        }
    };
    if let Some(handle) = emu.thread_handle.as_ref() {
        use std::os::unix::thread::JoinHandleExt;
        crate::sampler::note_name(handle.as_pthread_t() as usize, c"nexium-core0".as_ptr());
        crate::affinity::pin(handle.as_pthread_t() as usize, c"nexium-core0".as_ptr());
    }
    let snapshot_dir = format!("{DATA_ROOT}/snapshots");
    if cfg.snapshot_every.is_some() {
        let _ = ensure_shared_dir(&snapshot_dir);
    }
    let mut last_frame: Option<Frame> = None;
    let mut frame_counter = 0u32;
    let mut guest_frames = 0u64;
    let mut presents = 0u64;
    let mut window_start = Instant::now();
    let mut window_guest = 0u64;
    let mut window_presents = 0u64;
    let mut last_snapshot = Instant::now();
    let mut snapshots = 0u32;
    let mut exit_hold: Option<Instant> = None;
    let mut last_pad = PadState::default();
    let mut first_frame_logged = false;
    let mut failures = 0u32;
    loop {
        if let Some(p) = pad.as_mut() {
            let state = p.poll();
            if state.buttons != last_pad.buttons {
                log::debug!("frontend: pad {:#x}", state.buttons);
            }
            last_pad = state;
            push_hid(&state);
            let combo = pad::OPTIONS | pad::TOUCH_PAD;
            if state.buttons & combo == combo {
                let since = *exit_hold.get_or_insert_with(Instant::now);
                if since.elapsed() >= EXIT_HOLD {
                    crate::klog!("frontend: Options+Touchpad held: exiting");
                    break;
                }
            } else {
                exit_hold = None;
            }
        }
        if cfg.exit_after.is_some_and(|d| started.elapsed() >= d) {
            crate::klog!("frontend: exit_after reached");
            break;
        }
        if emu.thread_handle.as_ref().is_some_and(|h| h.is_finished()) {
            let result = emu.thread_handle.take().map(|h| h.join());
            crate::klog!("frontend: emulation thread ended: {result:?}");
            failures += matches!(result, Some(Ok(Ok(())))).then_some(0).unwrap_or(1);
            break;
        }
        let mut fresh = false;
        while let Ok(frame) = emu.frame_rx.try_recv() {
            guest_frames += 1;
            window_guest += 1;
            last_frame = Some(frame);
            fresh = true;
        }
        let result = match &last_frame {
            Some(f) => {
                if !first_frame_logged {
                    crate::klog!("frontend: first guest frame {}x{} after {:.1}s", f.width, f.height, started.elapsed().as_secs_f32());
                    first_frame_logged = true;
                }
                display.present_rgba(f.width, f.height, fresh.then_some(&f.pixels[..]))
            }
            None => idle_screen(&mut display, started, frame_counter),
        };
        if let Err(e) = result {
            crate::klog!("frontend: present failed: {e}");
            failures += 1;
            break;
        }
        presents += 1;
        window_presents += 1;
        frame_counter += 1;
        if let (Some(every), Some(f)) = (cfg.snapshot_every, &last_frame) {
            if last_snapshot.elapsed() >= every && snapshots < cfg.snapshot_limit {
                last_snapshot = Instant::now();
                let path = format!("{snapshot_dir}/shot-{snapshots:04}.bmp");
                let cap = crate::display::Capture { bytes: f.pixels.clone(), width: f.width, height: f.height, bgra: false };
                let step = if f.width > 1280 { 2 } else { 1 };
                match write_bmp(&path, &cap, step) {
                    Ok(()) => snapshots += 1,
                    Err(e) => crate::klog!("frontend: snapshot {path}: {e}"),
                }
            }
        }
        if window_start.elapsed() >= Duration::from_secs(5) {
            let secs = window_start.elapsed().as_secs_f64();
            let stats = { let s = emu.stats.lock(); (s.svc_count, s.cycle_count) };
            let (grains, max_block, audible, peak) = audio
                .as_ref()
                .map(|a| {
                    (
                        a.stats.grains.load(Ordering::Relaxed),
                        a.stats.max_block_us.load(Ordering::Relaxed),
                        a.stats.audible_grains.swap(0, Ordering::Relaxed),
                        a.stats.peak.swap(0, Ordering::Relaxed),
                    )
                })
                .unwrap_or((0, 0, 0, 0));
            let fm = crate::filemap::stats();
            let mut heap = crate::sys::HeapStats::default();
            unsafe { crate::sys::ps5_heap_stats(&mut heap) };
            let watch = nexium_memory::soft_watch::stats();
            crate::klog!(
                "frontend: heap mapped {} MiB peak {} MiB arenas {} libc fallbacks {}, flexible free {} MiB, write-watch regions {} protects {} faults {} failures {}",
                heap.mapped_bytes >> 20,
                heap.peak_bytes >> 20,
                heap.arenas,
                heap.libc_fallbacks,
                crate::sys::flexible_available().unwrap_or(0) >> 20,
                watch.regions,
                watch.protects,
                watch.faults,
                watch.protect_failures
            );
            crate::klog!(
                "frontend: t={:.0}s guest {:.1} fps, present {:.1} fps, frames {guest_frames}, svc {} cycles {}, audio grains {grains} audible {audible} peak {peak} max block {max_block}us, pad {:#x}, filecache {}MiB loads {}",
                started.elapsed().as_secs_f64(),
                window_guest as f64 / secs,
                window_presents as f64 / secs,
                stats.0,
                stats.1,
                last_pad.buttons,
                fm.resident_mib,
                fm.loads
            );
            window_start = Instant::now();
            window_guest = 0;
            window_presents = 0;
        }
    }
    crate::klog!("frontend: stopping emulation ({guest_frames} guest frames, {presents} presents, {snapshots} snapshots)");
    if !stop_emulation(emu) {
        failures += 1;
    }
    let fm = crate::filemap::stats();
    crate::klog!(
        "frontend: filemap regions {} resident {} MiB (cap {} MiB) loads {} evictions {} avg load {:.0}us",
        fm.regions,
        fm.resident_mib,
        fm.cap_mib,
        fm.loads,
        fm.evictions,
        fm.avg_load_us
    );
    drop(audio);
    drop(display);
    failures
}

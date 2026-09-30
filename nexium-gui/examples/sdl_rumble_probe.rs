#![allow(dead_code)]

#[path = "../src/hd_rumble.rs"]
mod hd_rumble;

use nexium_core::hid_vibration::VibrationValue;
use sdl3::sys::events::*;
use sdl3::sys::gamepad::*;
use sdl3::sys::hidapi::*;
use sdl3::sys::properties::SDL_GetBooleanProperty;
use sdl3::sys::sensor::*;
use std::ffi::{CStr, CString};
use std::time::{Duration, Instant};

const BUDGET: Duration = Duration::from_secs(60);
const SWITCH_PRODUCTS: [u16; 4] = [0x2006, 0x2007, 0x2009, 0x2019];

fn product_label(product: u16) -> &'static str {
    match product {
        0x2006 => "Joy-Con L",
        0x2007 => "Joy-Con R",
        0x2009 => "Pro Controller",
        0x2019 => "N64 Controller",
        _ => "?",
    }
}

fn level(amplitude: f32, neutral: [u8; 4]) -> [u8; 4] {
    hd_rumble::encode_side(
        hd_rumble::protect(VibrationValue {
            amp_low: amplitude,
            freq_low: 160.0,
            amp_high: amplitude,
            freq_high: 320.0,
        }),
        neutral,
    )
}

#[derive(Default)]
struct Stats {
    writes: u64,
    fails: u64,
    total: Duration,
    min: Option<Duration>,
    max: Duration,
}

impl Stats {
    fn record(&mut self, elapsed: Duration, ok: bool) {
        self.writes += 1;
        if !ok {
            self.fails += 1;
        }
        self.total += elapsed;
        self.min = Some(self.min.map_or(elapsed, |min| min.min(elapsed)));
        self.max = self.max.max(elapsed);
    }

    fn line(&self) -> String {
        let ms = |duration: Duration| duration.as_secs_f64() * 1000.0;
        let avg = if self.writes > 0 {
            ms(self.total) / self.writes as f64
        } else {
            0.0
        };
        format!(
            "writes={} fails={} hid_write_ms={:.2}/{:.2}/{:.2}",
            self.writes,
            self.fails,
            ms(self.min.unwrap_or_default()),
            avg,
            ms(self.max)
        )
    }
}

struct Target {
    label: String,
    product: u16,
    dev: *mut SDL_hid_device,
    counter: u8,
    neutral: [u8; 4],
}

impl Target {
    fn send(&mut self, left: [u8; 4], right: [u8; 4], stats: &mut Stats) {
        let bytes = hd_rumble::report(self.counter, left, right);
        self.counter = (self.counter + 1) & 0x0F;
        let started = Instant::now();
        let written = unsafe { SDL_hid_write(self.dev, bytes.as_ptr(), bytes.len()) };
        stats.record(started.elapsed(), written >= 0);
    }

    fn silence(&mut self, stats: &mut Stats) {
        let neutral = self.neutral;
        self.send(neutral, neutral, stats);
    }
}

struct Probe {
    deadline: Instant,
    pads: Vec<*mut SDL_Gamepad>,
    targets: Vec<Target>,
    buffer: Vec<SDL_Event>,
    gamepad_events: u64,
    sensor_events: u64,
}

impl Probe {
    fn pump(&mut self) {
        unsafe {
            SDL_PumpEvents();
            loop {
                let count = SDL_PeepEvents(
                    self.buffer.as_mut_ptr(),
                    self.buffer.len() as i32,
                    SDL_GETEVENT,
                    SDL_EVENT_FIRST.0,
                    SDL_EVENT_LAST.0,
                );
                if count <= 0 {
                    break;
                }
                for event in &self.buffer[..count as usize] {
                    let kind = event.r#type;
                    if kind == SDL_EVENT_GAMEPAD_SENSOR_UPDATE.0 {
                        self.sensor_events += 1;
                    } else if (SDL_EVENT_GAMEPAD_AXIS_MOTION.0..=SDL_EVENT_GAMEPAD_STEAM_HANDLE_UPDATED.0)
                        .contains(&kind)
                    {
                        self.gamepad_events += 1;
                    }
                }
                if (count as usize) < self.buffer.len() {
                    break;
                }
            }
        }
    }

    fn wait(&mut self, duration: Duration) {
        let until = (Instant::now() + duration).min(self.deadline);
        while Instant::now() < until {
            self.pump();
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn connected(&self) -> String {
        self.pads
            .iter()
            .map(|pad| if unsafe { SDL_GamepadConnected(*pad) } { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn take_events(&mut self) -> (u64, u64) {
        let counts = (self.gamepad_events, self.sensor_events);
        self.gamepad_events = 0;
        self.sensor_events = 0;
        counts
    }

    fn open_pads(&mut self, sensors: bool) {
        unsafe {
            let mut count = 0;
            let list = SDL_GetGamepads(&mut count);
            for index in 0..count.max(0) as usize {
                let id = *list.add(index);
                let pad = SDL_OpenGamepad(id);
                if pad.is_null() {
                    println!("gamepad {:?} failed to open", id.0);
                    continue;
                }
                let name_ptr = SDL_GetGamepadName(pad);
                let name = if name_ptr.is_null() {
                    String::from("?")
                } else {
                    CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
                };
                let path_ptr = SDL_GetGamepadPath(pad);
                let path = if path_ptr.is_null() {
                    String::from("-")
                } else {
                    CStr::from_ptr(path_ptr).to_string_lossy().into_owned()
                };
                let rumble = SDL_GetBooleanProperty(
                    SDL_GetGamepadProperties(pad),
                    SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN,
                    false,
                );
                println!(
                    "gamepad id={} name={} type={} vid:pid={:04x}:{:04x} rumble={} path={}",
                    id.0,
                    name,
                    SDL_GetGamepadType(pad).0,
                    SDL_GetGamepadVendor(pad),
                    SDL_GetGamepadProduct(pad),
                    rumble,
                    path
                );
                if sensors {
                    for sensor in [
                        SDL_SENSOR_ACCEL,
                        SDL_SENSOR_GYRO,
                        SDL_SENSOR_ACCEL_L,
                        SDL_SENSOR_GYRO_L,
                        SDL_SENSOR_ACCEL_R,
                        SDL_SENSOR_GYRO_R,
                    ] {
                        if SDL_GamepadHasSensor(pad, sensor) {
                            SDL_SetGamepadSensorEnabled(pad, sensor, true);
                        }
                    }
                }
                self.pads.push(pad);
            }
            if !list.is_null() {
                sdl3::sys::stdinc::SDL_free(list.cast());
            }
        }
    }

    fn hid_entries(&self, print: bool) -> Vec<(u16, CString)> {
        let mut entries: Vec<(u16, CString)> = Vec::new();
        unsafe {
            let list = SDL_hid_enumerate(0x057E, 0);
            let mut node = list;
            while !node.is_null() {
                let info = &*node;
                let path = if info.path.is_null() {
                    CString::default()
                } else {
                    CStr::from_ptr(info.path).to_owned()
                };
                if print {
                    println!(
                        "hid 057e:{:04x} ({}) usage={:04x}/{:04x} bus={} path={}",
                        info.product_id,
                        product_label(info.product_id),
                        info.usage_page,
                        info.usage,
                        info.bus_type.0,
                        path.to_string_lossy()
                    );
                }
                if SWITCH_PRODUCTS.contains(&info.product_id)
                    && !path.as_bytes().is_empty()
                    && !entries.iter().any(|(_, known)| *known == path)
                {
                    entries.push((info.product_id, path));
                }
                node = info.next;
            }
            if !list.is_null() {
                SDL_hid_free_enumeration(list);
            }
        }
        entries
    }

    fn open_targets(&mut self) {
        for (product, path) in self.hid_entries(false) {
            let dev = unsafe { SDL_hid_open_path(path.as_ptr()) };
            let label = format!("{} {:04x}", product_label(product), product);
            if dev.is_null() {
                println!("open {} failed (second handle refused?) path={}", label, path.to_string_lossy());
                continue;
            }
            println!("open {} ok", label);
            self.targets.push(Target {
                label,
                product,
                dev,
                counter: 0,
                neutral: hd_rumble::neutral_for(product),
            });
        }
    }

    fn halves(&mut self) {
        for index in 0..self.targets.len() {
            if !matches!(self.targets[index].product, 0x2006 | 0x2007 | 0x2009) {
                continue;
            }
            let neutral = self.targets[index].neutral;
            let on = level(0.8, neutral);
            for (phase, left, right) in [("half 0", on, neutral), ("off", neutral, neutral), ("half 1", neutral, on)] {
                if Instant::now() >= self.deadline {
                    return;
                }
                let mut stats = Stats::default();
                let end = (Instant::now() + Duration::from_secs(1)).min(self.deadline);
                let mut next = Instant::now();
                while Instant::now() < end {
                    if Instant::now() >= next {
                        self.targets[index].send(left, right, &mut stats);
                        next += Duration::from_millis(30);
                    }
                    self.pump();
                    std::thread::sleep(Duration::from_millis(1));
                }
                println!("halves {} {}: {}", self.targets[index].label, phase, stats.line());
            }
            let mut stats = Stats::default();
            self.targets[index].silence(&mut stats);
            self.wait(Duration::from_millis(300));
        }
    }

    fn cadence(&mut self, spacing_ms: u64, seconds: u64) {
        let spacing = Duration::from_millis(spacing_ms);
        let start = Instant::now();
        let end = (start + Duration::from_secs(seconds)).min(self.deadline);
        let mut next = start;
        let mut report_at = start + Duration::from_secs(1);
        let mut stats = Stats::default();
        self.take_events();
        while Instant::now() < end {
            let now = Instant::now();
            if now >= next {
                let phase = (now - start).as_secs_f32().fract();
                let amplitude = 0.2 + 0.4 * (1.0 - (2.0 * phase - 1.0).abs());
                for target in &mut self.targets {
                    let value = level(amplitude, target.neutral);
                    target.send(value, value, &mut stats);
                }
                next += spacing;
            }
            if now >= report_at {
                let (gamepad, sensor) = self.take_events();
                println!(
                    "cadence {}ms t={:.0}s {} gamepad_ev/s={} sensor_ev/s={} connected=[{}]",
                    spacing_ms,
                    (now - start).as_secs_f32(),
                    stats.line(),
                    gamepad,
                    sensor,
                    self.connected()
                );
                stats = Stats::default();
                report_at += Duration::from_secs(1);
            }
            self.pump();
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn hold(&mut self, seconds: u64) {
        let mut stats = Stats::default();
        for target in &mut self.targets {
            let value = level(0.6, target.neutral);
            target.send(value, value, &mut stats);
        }
        println!("hold: one report sent, {}", stats.line());
        let start = Instant::now();
        let end = (start + Duration::from_secs(seconds)).min(self.deadline);
        self.take_events();
        while Instant::now() < end {
            self.wait(Duration::from_secs(1));
            let (gamepad, sensor) = self.take_events();
            println!(
                "hold t={:.0}s gamepad_ev/s={} sensor_ev/s={} connected=[{}]",
                (Instant::now() - start).as_secs_f32(),
                gamepad,
                sensor,
                self.connected()
            );
        }
    }

    fn finish(&mut self) {
        let mut stats = Stats::default();
        for target in &mut self.targets {
            target.silence(&mut stats);
        }
        std::thread::sleep(Duration::from_millis(30));
        for target in &mut self.targets {
            target.silence(&mut stats);
        }
        for target in self.targets.drain(..) {
            unsafe { SDL_hid_close(target.dev) };
        }
        for pad in self.pads.drain(..) {
            unsafe { SDL_CloseGamepad(pad) };
        }
        println!("finished: neutral sent twice, {}", stats.line());
    }
}

fn main() {
    let started = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let number = |index: usize, default: u64| {
        args.get(index)
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(default)
    };
    sdl3::hint::set("SDL_JOYSTICK_THREAD", "1");
    sdl3::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
    sdl3::hint::set_with_priority(
        "SDL_JOYSTICK_HIDAPI_COMBINE_JOY_CONS",
        "1",
        &sdl3::hint::Hint::Override,
    );
    let sdl = sdl3::init().expect("sdl init");
    let gamepad = sdl.gamepad().expect("gamepad subsystem");
    let mut probe = Probe {
        deadline: started + BUDGET,
        pads: Vec::new(),
        targets: Vec::new(),
        buffer: (0..256).map(|_| unsafe { std::mem::zeroed() }).collect(),
        gamepad_events: 0,
        sensor_events: 0,
    };
    probe.wait(Duration::from_millis(2000));
    probe.open_pads(mode != "list");
    let hid_ready = unsafe { SDL_hid_init() } == 0;
    probe.wait(Duration::from_millis(500));
    match mode.as_str() {
        "list" => {
            probe.hid_entries(true);
        }
        "halves" => {
            probe.open_targets();
            probe.halves();
        }
        "cadence" => {
            let spacing = match number(2, 30) {
                15 => 15,
                _ => 30,
            };
            probe.open_targets();
            probe.cadence(spacing, number(3, 10));
        }
        "hold" => {
            probe.open_targets();
            probe.hold(number(2, 10));
        }
        _ => println!("usage: sdl_rumble_probe list | halves | cadence <15|30> <secs> | hold <secs>"),
    }
    probe.finish();
    if hid_ready {
        unsafe { SDL_hid_exit() };
    }
    drop(gamepad);
    drop(sdl);
}

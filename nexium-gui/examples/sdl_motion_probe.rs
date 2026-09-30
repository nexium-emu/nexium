use sdl3::sys::events::*;
use sdl3::sys::gamepad::*;
use sdl3::sys::sensor::*;
use std::collections::BTreeMap;
use std::ffi::CStr;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Stat {
    count: u64,
    sum: [f64; 3],
    first_sensor_ts: u64,
    last_sensor_ts: u64,
    first_event_ts: u64,
    last_event_ts: u64,
    duplicate_ts: u64,
    min_dt: u64,
    max_dt: u64,
    head: Vec<(u64, u64, [f32; 3])>,
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let seconds: u64 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(4);
    sdl3::hint::set("SDL_JOYSTICK_THREAD", "1");
    sdl3::hint::set("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
    if mode.contains("separate") {
        sdl3::hint::set("SDL_JOYSTICK_HIDAPI_COMBINE_JOY_CONS", "0");
    }
    if mode.contains("vertical") {
        sdl3::hint::set("SDL_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS", "1");
    }
    let sdl = sdl3::init().expect("sdl init");
    let _gamepad = sdl.gamepad().expect("gamepad subsystem");
    let settle = Instant::now();
    while settle.elapsed() < Duration::from_millis(2500) {
        unsafe {
            SDL_PumpEvents();
            SDL_FlushEvents(0, u32::MAX);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let sensors = [
        (SDL_SENSOR_ACCEL, "ACCEL"),
        (SDL_SENSOR_GYRO, "GYRO"),
        (SDL_SENSOR_ACCEL_L, "ACCEL_L"),
        (SDL_SENSOR_GYRO_L, "GYRO_L"),
        (SDL_SENSOR_ACCEL_R, "ACCEL_R"),
        (SDL_SENSOR_GYRO_R, "GYRO_R"),
    ];
    let mut names = BTreeMap::new();
    let mut pads = Vec::new();
    unsafe {
        let mut count = 0;
        let list = SDL_GetGamepads(&mut count);
        for i in 0..count.max(0) as usize {
            let id = *list.add(i);
            let pad = SDL_OpenGamepad(id);
            if pad.is_null() {
                println!("open failed for {:?}", id);
                continue;
            }
            let name_ptr = SDL_GetGamepadName(pad);
            let name = if name_ptr.is_null() {
                String::from("?")
            } else {
                CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            };
            println!(
                "pad id={:?} name={} type={} vid={:04x} pid={:04x}",
                id,
                name,
                SDL_GetGamepadType(pad).0,
                SDL_GetGamepadVendor(pad),
                SDL_GetGamepadProduct(pad)
            );
            for (sensor, label) in sensors {
                if SDL_GamepadHasSensor(pad, sensor) {
                    let enabled = SDL_SetGamepadSensorEnabled(pad, sensor, true);
                    let rate = SDL_GetGamepadSensorDataRate(pad, sensor);
                    println!("    sensor {label} enabled={enabled} rate={rate}");
                }
            }
            names.insert(format!("{:?}", id), name);
            pads.push(pad);
        }
        sdl3::sys::stdinc::SDL_free(list as *mut _);
    }
    let mut stats: BTreeMap<(String, i32), Stat> = BTreeMap::new();
    let mut buffer: Vec<SDL_Event> = (0..256).map(|_| unsafe { std::mem::zeroed() }).collect();
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(seconds) {
        unsafe {
            SDL_PumpEvents();
            loop {
                let n = SDL_PeepEvents(
                    buffer.as_mut_ptr(),
                    buffer.len() as i32,
                    SDL_GETEVENT,
                    SDL_EVENT_GAMEPAD_SENSOR_UPDATE.0,
                    SDL_EVENT_GAMEPAD_SENSOR_UPDATE.0,
                );
                if n <= 0 {
                    break;
                }
                for event in &buffer[..n as usize] {
                    let g = event.gsensor;
                    let entry = stats.entry((format!("{:?}", g.which), g.sensor)).or_default();
                    if entry.count == 0 {
                        entry.first_sensor_ts = g.sensor_timestamp;
                        entry.first_event_ts = g.timestamp;
                        entry.min_dt = u64::MAX;
                    } else {
                        let dt = g.sensor_timestamp.wrapping_sub(entry.last_sensor_ts);
                        if dt == 0 {
                            entry.duplicate_ts += 1;
                        } else {
                            entry.min_dt = entry.min_dt.min(dt);
                            entry.max_dt = entry.max_dt.max(dt);
                        }
                    }
                    entry.last_sensor_ts = g.sensor_timestamp;
                    entry.last_event_ts = g.timestamp;
                    entry.count += 1;
                    for axis in 0..3 {
                        entry.sum[axis] += f64::from(g.data[axis]);
                    }
                    if entry.head.len() < 4 {
                        entry.head.push((g.sensor_timestamp, g.timestamp, g.data));
                    }
                }
            }
            SDL_FlushEvents(0, u32::MAX);
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    let label = |raw: i32| {
        sensors
            .iter()
            .find(|(sensor, _)| sensor.0 == raw)
            .map_or("?", |(_, name)| *name)
    };
    for ((which, sensor), stat) in &stats {
        let n = stat.count.max(1) as f64;
        let mean = [stat.sum[0] / n, stat.sum[1] / n, stat.sum[2] / n];
        let magnitude = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        let span = stat.last_sensor_ts.saturating_sub(stat.first_sensor_ts) as f64 / 1e9;
        println!(
            "{} [{}] {} samples={} rate={:.1}/s mean=({:.3},{:.3},{:.3}) |mean|={:.3} dt_ns[min={} max={}] dup_ts={} sensor_span={:.3}s event_span={:.3}s",
            names.get(which).map_or("?", |name| name.as_str()),
            which,
            label(*sensor),
            stat.count,
            stat.count as f64 / seconds as f64,
            mean[0],
            mean[1],
            mean[2],
            magnitude,
            stat.min_dt,
            stat.max_dt,
            stat.duplicate_ts,
            span,
            stat.last_event_ts.saturating_sub(stat.first_event_ts) as f64 / 1e9
        );
        for (sensor_ts, event_ts, data) in &stat.head {
            println!("      sensor_ts={sensor_ts} event_ts={event_ts} data=({:.3},{:.3},{:.3})", data[0], data[1], data[2]);
        }
    }
    unsafe {
        for pad in pads {
            SDL_CloseGamepad(pad);
        }
    }
}

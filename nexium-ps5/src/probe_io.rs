use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::audio::{AudioOut, RATE};
use crate::display::{write_bmp, Display};
use crate::pad::{self, Pad, PadState};

const BUTTONS: &[(u32, &str)] = &[
    (pad::CROSS, "cross"),
    (pad::CIRCLE, "circle"),
    (pad::SQUARE, "square"),
    (pad::TRIANGLE, "triangle"),
    (pad::UP, "up"),
    (pad::DOWN, "down"),
    (pad::LEFT, "left"),
    (pad::RIGHT, "right"),
    (pad::L1, "l1"),
    (pad::R1, "r1"),
    (pad::L2, "l2"),
    (pad::R2, "r2"),
    (pad::L3, "l3"),
    (pad::R3, "r3"),
    (pad::OPTIONS, "options"),
    (pad::TOUCH_PAD, "touchpad"),
];

fn names(mask: u32) -> String {
    let list: Vec<&str> = BUTTONS.iter().filter(|(b, _)| mask & b != 0).map(|(_, n)| *n).collect();
    if list.is_empty() {
        "-".into()
    } else {
        list.join("+")
    }
}

struct Tone {
    phase: f32,
}

impl Tone {
    fn fill(&mut self, out: &mut Vec<i16>, frames: usize, freq: f32, volume: f32) {
        let step = freq * std::f32::consts::TAU / RATE as f32;
        for _ in 0..frames {
            let v = if freq > 0.0 { (self.phase.sin() * volume * i16::MAX as f32) as i16 } else { 0 };
            self.phase = (self.phase + step) % std::f32::consts::TAU;
            out.push(v);
            out.push(v);
        }
    }
}

pub fn interactive(seconds: u64) -> Result<String, String> {
    let mut pad = Pad::open()?;
    crate::klog!("io: pad open for user {}", pad.user);
    let audio = AudioOut::open(RATE as usize / 8)?;
    crate::klog!("io: audio port open (48 kHz S16 stereo, 256-frame grains)");
    let mut display = Display::new(false)?;
    crate::klog!(
        "io: interactive test for {seconds}s: press every button, move both sticks, hold Cross for a tone; Options+Cross ends early"
    );
    let started = Instant::now();
    let mut tone = Tone { phase: 0.0 };
    let mut last = PadState::default();
    let mut seen = 0u32;
    let mut events = 0u32;
    let (mut lx_min, mut lx_max, mut ly_min, mut ly_max) = (255u8, 0u8, 255u8, 0u8);
    let (mut rx_min, mut rx_max) = (255u8, 0u8);
    let mut connected_seen = false;
    let mut scratch = Vec::with_capacity(RATE as usize / 4);
    let mut frame = 0u32;
    let mut tone_frames = 0u64;
    let mut capture_path = None;
    while started.elapsed() < Duration::from_secs(seconds) {
        let state = pad.poll();
        connected_seen |= state.connected;
        if state.buttons != last.buttons {
            events += 1;
            seen |= state.buttons;
            crate::klog!("io: pad buttons {:#08x} [{}] l2={} r2={}", state.buttons, names(state.buttons), state.l2, state.r2);
        }
        let moved = |a: u8, b: u8| (a as i32 - b as i32).abs() > 24;
        if moved(state.lx, last.lx) || moved(state.ly, last.ly) || moved(state.rx, last.rx) || moved(state.ry, last.ry) {
            crate::klog!("io: sticks L=({},{}) R=({},{})", state.lx, state.ly, state.rx, state.ry);
        } else if state.buttons == last.buttons {
            last.buttons = state.buttons;
        }
        lx_min = lx_min.min(state.lx);
        lx_max = lx_max.max(state.lx);
        ly_min = ly_min.min(state.ly);
        ly_max = ly_max.max(state.ly);
        rx_min = rx_min.min(state.rx);
        rx_max = rx_max.max(state.rx);
        last = state;
        if state.buttons & pad::OPTIONS != 0 && state.buttons & pad::CROSS != 0 {
            crate::klog!("io: Options+Cross: ending early");
            break;
        }

        let elapsed = started.elapsed().as_secs_f32();
        let freq = if elapsed < 0.4 {
            523.25
        } else if elapsed < 0.8 {
            659.25
        } else if state.buttons & pad::CROSS != 0 {
            440.0
        } else {
            0.0
        };
        {
            let mut ring = audio.ring.lock().unwrap();
            let want = ring.free_frames().min(RATE as usize / 20);
            scratch.clear();
            tone.fill(&mut scratch, want, freq, 0.25);
            ring.push(&scratch);
            if freq > 0.0 {
                tone_frames += want as u64;
            }
        }

        let push = [
            elapsed,
            display.extent.width as f32,
            display.extent.height as f32,
            frame as f32,
            (state.buttons & 0x00ff_ffff) as f32,
            PadState::stick(state.lx),
            PadState::stick(state.ly),
            1.0,
        ];
        let want_capture = state.buttons != 0 && capture_path.is_none() && seen.count_ones() >= 3;
        if let Some(cap) = display.frame(push, want_capture)? {
            let path = format!("{}/io-probe.bmp", crate::console::DATA_ROOT);
            if write_bmp(&path, &cap, 4).is_ok() {
                crate::klog!("io: captured {path} with buttons [{}]", names(state.buttons));
                capture_path = Some(path);
            }
        }
        frame += 1;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let grains = audio.stats.grains.load(Ordering::Relaxed);
    let underrun = audio.stats.underrun_frames.load(Ordering::Relaxed);
    let max_block = audio.stats.max_block_us.load(Ordering::Relaxed);
    drop(display);
    drop(audio);
    let expected_grains = elapsed * RATE as f64 / 256.0;
    let detail = format!(
        "{elapsed:.1}s {frame} frames; pad connected={connected_seen} events={events} buttons seen [{}] ({} of {}) L stick x {lx_min}..{lx_max} y {ly_min}..{ly_max} R stick x {rx_min}..{rx_max}; audio grains {grains} (expected ~{expected_grains:.0}) max block {max_block}us silent-fill frames {underrun} tone frames {tone_frames}",
        names(seen),
        BUTTONS.iter().filter(|(b, _)| seen & b != 0).count(),
        BUTTONS.len()
    );
    if !connected_seen {
        return Err(format!("no connected pad sample: {detail}"));
    }
    Ok(detail)
}

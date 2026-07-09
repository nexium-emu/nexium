use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

const SR: f32 = 48_000.0;
const MUSIC_SR: f32 = 32_000.0;

static MUSIC_WAV: &[u8] = include_bytes!("assets/carousel_music.wav");

// f32 bits: target music gain (post-volume) and lowpass amount (0=clean, 1=muffled)
static MUSIC_TARGET: AtomicU32 = AtomicU32::new(0);
static MUSIC_LOWPASS: AtomicU32 = AtomicU32::new(0);
static SFX_VOLUME: AtomicU32 = AtomicU32::new(1056964608); // 0.5f32.to_bits()

pub fn set_music(target_gain: f32, lowpass: f32) {
    MUSIC_TARGET.store(target_gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    MUSIC_LOWPASS.store(lowpass.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

pub fn set_sfx_volume(volume: f32) {
    SFX_VOLUME.store(volume.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

fn decode_wav_mono(bytes: &[u8]) -> Vec<f32> {
    // minimal PCM-s16 WAV parse: locate "data" chunk, read i16 samples
    let mut i = 12; // skip RIFF header
    let mut data: Option<(usize, usize)> = None;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let body = i + 8;
        if id == b"data" {
            data = Some((body, sz.min(bytes.len().saturating_sub(body))));
            break;
        }
        i = body + sz + (sz & 1);
    }
    let Some((off, len)) = data else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(len / 2);
    let mut j = off;
    while j + 1 < off + len {
        let s = i16::from_le_bytes([bytes[j], bytes[j + 1]]);
        out.push(s as f32 / 32768.0);
        j += 2;
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sfx {
    Move,
    Select,
    Back,
    Error,
    Favorite,
    Open,
    Boot,
    Whistle,
    GameBoot,
    AwaitFrame,
    PleaseWait,
    Celebration,
    WhistleOk,
    WhistleSquish,
}

struct Voice {
    key: u8,
    buf: std::sync::Arc<Vec<f32>>,
    pos: usize,
    looping: bool,
}

struct Engine {
    active: std::sync::Arc<Mutex<Vec<Voice>>>,
    banks: std::collections::HashMap<u8, std::sync::Arc<Vec<f32>>>,
    _stream: cpal::Stream,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

static ENGINE: OnceLock<Option<Engine>> = OnceLock::new();

fn key(s: Sfx) -> u8 {
    match s {
        Sfx::Move => 0,
        Sfx::Select => 1,
        Sfx::Back => 2,
        Sfx::Error => 3,
        Sfx::Favorite => 4,
        Sfx::Open => 5,
        Sfx::Boot => 6,
        Sfx::Whistle => 7,
        Sfx::GameBoot => 8,
        Sfx::AwaitFrame => 9,
        Sfx::PleaseWait => 10,
        Sfx::Celebration => 11,
        Sfx::WhistleOk => 12,
        Sfx::WhistleSquish => 13,
    }
}

pub fn init() {
    ENGINE.get_or_init(build);
}

pub fn play_move() {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    if let Ok(mut l) = LAST.lock() {
        let now = std::time::Instant::now();
        if l.map_or(true, |t| now.duration_since(t).as_millis() >= 55) {
            *l = Some(now);
            play(Sfx::Move);
        }
    }
}

pub fn play(s: Sfx) {
    if let Some(Some(eng)) = ENGINE.get() {
        if let Some(buf) = eng.banks.get(&key(s)) {
            if let Ok(mut act) = eng.active.lock() {
                if act.len() < 8 {
                    act.push(Voice {
                        key: key(s),
                        buf: buf.clone(),
                        pos: 0,
                        looping: false,
                    });
                }
            }
        }
    }
}

pub fn play_looped(s: Sfx) {
    if let Some(Some(eng)) = ENGINE.get() {
        if let Some(buf) = eng.banks.get(&key(s)) {
            if let Ok(mut act) = eng.active.lock() {
                let k = key(s);
                if !act.iter().any(|v| v.key == k && v.looping) {
                    if act.len() < 8 {
                        act.push(Voice {
                            key: k,
                            buf: buf.clone(),
                            pos: 0,
                            looping: true,
                        });
                    }
                }
            }
        }
    }
}

pub fn stop_loop(s: Sfx) {
    if let Some(Some(eng)) = ENGINE.get() {
        if let Ok(mut act) = eng.active.lock() {
            let k = key(s);
            act.retain(|v| v.key != k || !v.looping);
        }
    }
}

fn build() -> Option<Engine> {
    let mut banks = std::collections::HashMap::new();
    banks.insert(key(Sfx::Move), std::sync::Arc::new(render(Sfx::Move)));
    banks.insert(key(Sfx::Select), std::sync::Arc::new(render(Sfx::Select)));
    banks.insert(key(Sfx::Back), std::sync::Arc::new(render(Sfx::Back)));
    banks.insert(key(Sfx::Error), std::sync::Arc::new(render(Sfx::Error)));
    banks.insert(key(Sfx::Favorite), std::sync::Arc::new(render(Sfx::Favorite)));
    banks.insert(key(Sfx::Open), std::sync::Arc::new(render(Sfx::Open)));
    banks.insert(key(Sfx::Boot), std::sync::Arc::new(render(Sfx::Boot)));
    banks.insert(key(Sfx::Whistle), std::sync::Arc::new(render(Sfx::Whistle)));
    banks.insert(key(Sfx::GameBoot), std::sync::Arc::new(render(Sfx::GameBoot)));
    banks.insert(key(Sfx::AwaitFrame), std::sync::Arc::new(render(Sfx::AwaitFrame)));
    banks.insert(key(Sfx::PleaseWait), std::sync::Arc::new(render(Sfx::PleaseWait)));
    banks.insert(key(Sfx::Celebration), std::sync::Arc::new(render(Sfx::Celebration)));
    banks.insert(key(Sfx::WhistleOk), std::sync::Arc::new(render(Sfx::WhistleOk)));
    banks.insert(key(Sfx::WhistleSquish), std::sync::Arc::new(render(Sfx::WhistleSquish)));

    let music = std::sync::Arc::new(decode_wav_mono(MUSIC_WAV));

    let host = cpal::default_host();
    let device = host.default_output_device()?;

    // Prefer an F32 config so our mixer output maps directly.
    let cfg = device
        .supported_output_configs()
        .ok()?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .max_by_key(|c| c.max_sample_rate().0)
        .map(|c| c.with_max_sample_rate())
        .or_else(|| device.default_output_config().ok())?;
    if cfg.sample_format() != cpal::SampleFormat::F32 {
        return None;
    }
    let channels = cfg.channels() as usize;
    let dev_sr = cfg.sample_rate().0 as f32;
    let sfx_ratio = SR / dev_sr;
    let music_step = (MUSIC_SR / dev_sr) as f64;
    let gain_step = 1.0 / (0.45 * dev_sr);

    let active: std::sync::Arc<Mutex<Vec<Voice>>> = std::sync::Arc::new(Mutex::new(Vec::new()));
    let active_cb = active.clone();
    let music_cb = music.clone();

    let mut cur_gain = 0.0f32;
    let mut music_pos = 0.0f64;
    let mut lp_y = 0.0f32;

    let err_cb = |e| log::warn!("ui_audio stream error: {}", e);
    let stream = device
        .build_output_stream(
            &cfg.config(),
            move |out: &mut [f32], _| {
                let frames = out.len() / channels.max(1);
                let target = f32::from_bits(MUSIC_TARGET.load(Ordering::Relaxed));
                let lp = f32::from_bits(MUSIC_LOWPASS.load(Ordering::Relaxed));
                let alpha = 1.0 - lp * 0.9;
                let mlen = music_cb.len();
                let mut act = active_cb.lock().ok();
                for f in 0..frames {
                    let mut sfx = 0.0f32;
                    if let Some(a) = act.as_mut() {
                        for v in a.iter_mut() {
                            let mut idx = (v.pos as f32 * sfx_ratio) as usize;
                            if v.looping {
                                idx %= v.buf.len();
                            }
                            if idx < v.buf.len() {
                                sfx += v.buf[idx];
                            }
                            v.pos += 1;
                        }
                    }
                    cur_gain += (target - cur_gain) * gain_step;
                    let m = if mlen > 0 {
                        let idx = (music_pos as usize) % mlen;
                        music_pos += music_step;
                        if music_pos >= mlen as f64 {
                            music_pos -= mlen as f64;
                        }
                        music_cb[idx]
                    } else {
                        0.0
                    };
                    lp_y += alpha * (m - lp_y);
                    let sfx_vol = f32::from_bits(SFX_VOLUME.load(Ordering::Relaxed));
                    let s = ((sfx * sfx_vol) + lp_y * cur_gain).clamp(-1.0, 1.0);
                    for c in 0..channels {
                        out[f * channels + c] = s;
                    }
                }
                if let Some(a) = act.as_mut() {
                    a.retain(|v| v.looping || (v.pos as f32 * sfx_ratio) < v.buf.len() as f32 + 1.0);
                }
            },
            err_cb,
            None,
        )
        .ok()?;
    stream.play().ok()?;

    Some(Engine {
        active,
        banks,
        _stream: stream,
    })
}

fn env(i: usize, n: usize, attack: usize, release: usize) -> f32 {
    if i < attack {
        i as f32 / attack.max(1) as f32
    } else if i > n.saturating_sub(release) {
        ((n - i) as f32 / release.max(1) as f32).max(0.0)
    } else {
        1.0
    }
}

fn tri(phase: f32) -> f32 {
    let p = phase.rem_euclid(1.0);
    (2.0 * (2.0 * p - 1.0).abs()) - 1.0
}

fn tone(buf: &mut Vec<f32>, freq: f32, dur_ms: f32, gain: f32, attack_ms: f32, release_ms: f32) {
    let n = (SR * dur_ms / 1000.0) as usize;
    let attack = (SR * attack_ms / 1000.0) as usize;
    let release = (SR * release_ms / 1000.0) as usize;
    let start = buf.len();
    if buf.len() < start + n {
        buf.resize(start + n, 0.0);
    }
    for i in 0..n {
        let t = i as f32 / SR;
        let vib = 1.0 + 0.004 * (t * 38.0).sin();
        let s = tri(freq * vib * t) * 0.7 + (freq * vib * t * std::f32::consts::TAU).sin() * 0.3;
        buf[start + i] += s * gain * env(i, n, attack, release);
    }
}

fn chord_at(buf: &mut Vec<f32>, offset_ms: f32, freqs: &[f32], dur_ms: f32, gain: f32) {
    let off = (SR * offset_ms / 1000.0) as usize;
    let n = (SR * dur_ms / 1000.0) as usize;
    let end = off + n;
    if buf.len() < end {
        buf.resize(end, 0.0);
    }
    let attack = (SR * 0.008) as usize;
    let release = (SR * dur_ms / 1000.0 * 0.6) as usize;
    for i in 0..n {
        let t = i as f32 / SR;
        let mut s = 0.0;
        for f in freqs {
            s += tri(f * t) * 0.55 + (f * t * std::f32::consts::TAU).sin() * 0.45;
        }
        s /= freqs.len() as f32;
        buf[off + i] += s * gain * env(i, n, attack, release);
    }
}

fn render(s: Sfx) -> Vec<f32> {
    let mut b = Vec::new();
    match s {
        Sfx::Move => tone(&mut b, 880.0, 42.0, 0.22, 2.0, 34.0),
        Sfx::Select => {
            tone(&mut b, 659.25, 55.0, 0.26, 2.0, 30.0);
            let off = b.len();
            let _ = off;
            tone_after(&mut b, 40, 987.77, 90.0, 0.28, 2.0, 70.0);
        }
        Sfx::Back => {
            // downward whistle slide — a whistley "back out"
            whistle(&mut b, 1180.0, 540.0, 0.30, 70.0, 0.18);
        }
        Sfx::Open => {
            tone(&mut b, 523.25, 60.0, 0.24, 2.0, 30.0);
            tone_after(&mut b, 45, 659.25, 60.0, 0.24, 2.0, 30.0);
            tone_after(&mut b, 90, 783.99, 110.0, 0.26, 2.0, 90.0);
        }
        Sfx::Favorite => {
            tone(&mut b, 987.77, 45.0, 0.22, 2.0, 20.0);
            tone_after(&mut b, 40, 1318.51, 45.0, 0.22, 2.0, 20.0);
            tone_after(&mut b, 80, 1567.98, 120.0, 0.24, 2.0, 100.0);
        }
        Sfx::Error => {
            tone(&mut b, 220.0, 70.0, 0.28, 1.0, 10.0);
            tone_after(&mut b, 60, 174.61, 130.0, 0.28, 1.0, 90.0);
        }
        Sfx::Boot => {
            // Low warm swell chord (C3, G3, B3, D4, F#4) swelling up to 2.4 seconds
            tone_after_abs(&mut b, 0.0, 130.81, 2400.0, 0.12, 1800.0, 200.0);
            tone_after_abs(&mut b, 0.0, 196.00, 2400.0, 0.09, 1800.0, 200.0);
            tone_after_abs(&mut b, 400.0, 246.94, 2000.0, 0.09, 1500.0, 200.0);
            tone_after_abs(&mut b, 800.0, 293.66, 1600.0, 0.08, 1200.0, 200.0);
            tone_after_abs(&mut b, 1200.0, 369.99, 1200.0, 0.08, 900.0, 200.0);

            // Convergence point chime starting at 2.35s (2350 ms) with a long release
            tone_after_abs(&mut b, 2350.0, 523.25, 2650.0, 0.08, 15.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 659.25, 2650.0, 0.08, 15.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 783.99, 2650.0, 0.08, 20.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 987.77, 2650.0, 0.07, 25.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 1174.66, 2650.0, 0.06, 30.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 1760.00, 2650.0, 0.05, 40.0, 2200.0);

            // Shimmering arpeggio notes cascading upwards
            tone_after_abs(&mut b, 2450.0, 1318.51, 1500.0, 0.04, 10.0, 1200.0);
            tone_after_abs(&mut b, 2600.0, 1567.98, 1500.0, 0.04, 10.0, 1200.0);
            tone_after_abs(&mut b, 2750.0, 1975.53, 1500.0, 0.03, 10.0, 1200.0);
            tone_after_abs(&mut b, 2900.0, 2349.32, 1500.0, 0.03, 10.0, 1200.0);
        }
        Sfx::Whistle => {
            // cute upward whistle slide (pure sine, pitch bends up then a little flick)
            let dur = 0.34f32;
            let n = (SR * dur) as usize;
            b.resize(n, 0.0);
            let attack = (SR * 0.02) as usize;
            let release = (SR * 0.10) as usize;
            let mut phase = 0.0f32;
            for i in 0..n {
                let p = i as f32 / n as f32;
                let bend = 1.0 - (1.0 - p).powi(2);
                let freq = 780.0 + 620.0 * bend + 90.0 * (p * 22.0).sin() * (1.0 - p);
                phase += freq / SR;
                let s = (phase * std::f32::consts::TAU).sin();
                b[i] += s * 0.20 * env(i, n, attack, release);
            }
        }
        Sfx::WhistleOk => {
            // bright rising accept whistle with a little high flick at the end
            whistle(&mut b, 720.0, 1360.0, 0.24, 55.0, 0.19);
            tone_after_abs(&mut b, 180.0, 1720.0, 120.0, 0.12, 4.0, 100.0);
        }
        Sfx::WhistleSquish => {
            // playful "boing" whistle — quick up then a springy dip back down
            whistle(&mut b, 820.0, 1300.0, 0.13, 40.0, 0.2);
            let s = b.len();
            let _ = s;
            whistle_at(&mut b, 120.0, 1300.0, 880.0, 0.16, 120.0, 0.18);
        }
        Sfx::GameBoot => {
            // Chord 1: Dm9 (D3, F3, A3, C4, E4) from 0ms to 300ms
            tone_after_abs(&mut b, 0.0, 146.83, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 20.0, 174.61, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 40.0, 220.00, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 60.0, 261.63, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 80.0, 329.63, 300.0, 0.10, 10.0, 60.0);

            // Chord 2: G13 (G3, B3, F4, A4, E5) from 300ms to 600ms
            tone_after_abs(&mut b, 300.0, 196.00, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 320.0, 246.94, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 340.0, 349.23, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 360.0, 440.00, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 380.0, 659.25, 300.0, 0.08, 10.0, 60.0);

            // Chord 3: Cmaj9 (C3, G3, B3, D4, E4, G4) starting at 600ms, sustained with long release
            tone_after_abs(&mut b, 600.0, 130.81, 1400.0, 0.14, 15.0, 800.0);
            tone_after_abs(&mut b, 620.0, 196.00, 1400.0, 0.12, 15.0, 800.0);
            tone_after_abs(&mut b, 640.0, 246.94, 1400.0, 0.12, 15.0, 800.0);
            tone_after_abs(&mut b, 660.0, 293.66, 1400.0, 0.10, 15.0, 800.0);
            tone_after_abs(&mut b, 680.0, 329.63, 1400.0, 0.10, 15.0, 800.0);
            tone_after_abs(&mut b, 700.0, 392.00, 1400.0, 0.08, 15.0, 800.0);

            // Top melodic jazzy lead notes
            tone_after_abs(&mut b, 800.0, 493.88, 200.0, 0.09, 10.0, 80.0); // B4
            tone_after_abs(&mut b, 1000.0, 587.33, 200.0, 0.09, 10.0, 80.0); // D5
            tone_after_abs(&mut b, 1200.0, 783.99, 800.0, 0.08, 10.0, 400.0); // G5
        }
        Sfx::AwaitFrame => {
            b.resize((SR * 0.5) as usize, 0.0);
        }
        Sfx::PleaseWait => {
            let dur = 3.0f32;
            let n = (SR * dur) as usize;
            b.resize(n, 0.0);
            let freqs = [155.6667, 196.00, 233.00, 293.6667, 349.3333];
            for i in 0..n {
                let t = i as f32 / SR;
                let breathe = 0.5 + 0.5 * (t * std::f32::consts::TAU / 3.0).cos();
                let mut s = 0.0f32;
                for (idx, &f) in freqs.iter().enumerate() {
                    let phase = f * t;
                    let wave = tri(phase) * 0.6 + (phase * std::f32::consts::TAU).sin() * 0.4;
                    let note_breathe = 0.4 + 0.6 * (t * std::f32::consts::TAU / 3.0 + idx as f32 * 0.5).cos().abs();
                    s += wave * note_breathe;
                }
                b[i] = (s / freqs.len() as f32) * 0.16 * breathe;
            }
        }
        Sfx::Celebration => {
            // Cute little "ta-da!" — quick ascending C major arpeggio into a
            // sparkly high sprinkle, with a soft chord bloom underneath.
            let lead = [523.25f32, 659.25, 783.99, 1046.5];
            for (i, &f) in lead.iter().enumerate() {
                tone_after_abs(&mut b, i as f32 * 70.0, f, 260.0, 0.14, 4.0, 200.0);
            }
            // Sustained major chord bloom (C5/E5/G5) that lands with the top note.
            tone_after_abs(&mut b, 210.0, 523.25, 620.0, 0.10, 10.0, 480.0);
            tone_after_abs(&mut b, 220.0, 659.25, 620.0, 0.09, 10.0, 480.0);
            tone_after_abs(&mut b, 230.0, 783.99, 620.0, 0.09, 10.0, 480.0);
            // Twinkly high sprinkles (pure sine) fluttering above.
            let sparkle = [1567.98f32, 2093.0, 1760.0, 2349.32, 2093.0];
            for (i, &f) in sparkle.iter().enumerate() {
                let off = 300.0 + i as f32 * 55.0;
                let start = (SR * off / 1000.0) as usize;
                let n = (SR * 0.09) as usize;
                if b.len() < start + n {
                    b.resize(start + n, 0.0);
                }
                for j in 0..n {
                    let s = (f * j as f32 / SR * std::f32::consts::TAU).sin();
                    b[start + j] += s * 0.06 * env(j, n, (SR * 0.004) as usize, (SR * 0.07) as usize);
                }
            }
        }
    }
    b
}

fn tone_after(buf: &mut Vec<f32>, gap_ms: usize, freq: f32, dur_ms: f32, gain: f32, attack_ms: f32, release_ms: f32) {
    let target = buf.len() + (SR * gap_ms as f32 / 1000.0) as usize;
    if buf.len() < target {
        buf.resize(target, 0.0);
    }
    let n = (SR * dur_ms / 1000.0) as usize;
    let attack = (SR * attack_ms / 1000.0) as usize;
    let release = (SR * release_ms / 1000.0) as usize;
    let start = buf.len();
    buf.resize(start + n, 0.0);
    for i in 0..n {
        let t = i as f32 / SR;
        let vib = 1.0 + 0.004 * (t * 38.0).sin();
        let sv = tri(freq * vib * t) * 0.7 + (freq * vib * t * std::f32::consts::TAU).sin() * 0.3;
        buf[start + i] += sv * gain * env(i, n, attack, release);
    }
}

fn tone_after_abs(buf: &mut Vec<f32>, offset_ms: f32, freq: f32, dur_ms: f32, gain: f32, attack_ms: f32, release_ms: f32) {
    let off = (SR * offset_ms / 1000.0) as usize;
    let n = (SR * dur_ms / 1000.0) as usize;
    let end = off + n;
    if buf.len() < end {
        buf.resize(end, 0.0);
    }
    let attack = (SR * attack_ms / 1000.0) as usize;
    let release = (SR * release_ms / 1000.0) as usize;
    for i in 0..n {
        let t = i as f32 / SR;
        let sv = tri(freq * t) * 0.6 + (freq * t * std::f32::consts::TAU).sin() * 0.4;
        buf[off + i] += sv * gain * env(i, n, attack, release);
    }
}

fn whistle_at(buf: &mut Vec<f32>, offset_ms: f32, f0: f32, f1: f32, dur: f32, warble: f32, gain: f32) {
    let off = (SR * offset_ms / 1000.0) as usize;
    let n = (SR * dur) as usize;
    if buf.len() < off + n {
        buf.resize(off + n, 0.0);
    }
    let attack = (SR * 0.016) as usize;
    let release = (SR * dur * 0.42) as usize;
    let mut phase = 0.0f32;
    for i in 0..n {
        let p = i as f32 / n as f32;
        let bend = 1.0 - (1.0 - p).powi(2);
        let freq = f0 + (f1 - f0) * bend + warble * (p * 20.0).sin() * (1.0 - p);
        phase += freq / SR;
        buf[off + i] += (phase * std::f32::consts::TAU).sin() * gain * env(i, n, attack, release);
    }
}

fn whistle(buf: &mut Vec<f32>, f0: f32, f1: f32, dur: f32, warble: f32, gain: f32) {
    whistle_at(buf, 0.0, f0, f1, dur, warble, gain);
}

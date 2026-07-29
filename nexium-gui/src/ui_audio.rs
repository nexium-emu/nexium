use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::hash::{BuildHasher, Hasher};
use std::io::Cursor;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const SR: f32 = 48_000.0;

static CAROUSEL_MUSIC: [&[u8]; 4] = [
    include_bytes!("assets/carousel_bgm_01.mp3"),
    include_bytes!("assets/carousel_bgm_02.mp3"),
    include_bytes!("assets/carousel_bgm_03.mp3"),
    include_bytes!("assets/carousel_bgm_04.mp3"),
];
static SHOP_MUSIC: [&[u8]; 2] = [
    include_bytes!("assets/shop_bgm_01.mp3"),
    include_bytes!("assets/shop_bgm_02.mp3"),
];
static SETTINGS_MUSIC: &[u8] = include_bytes!("assets/settings_bgm_01.mp3");

static MUSIC_TARGET: AtomicU32 = AtomicU32::new(0);
static MUSIC_LOWPASS: AtomicU32 = AtomicU32::new(0);
static MUSIC_MODE: AtomicU8 = AtomicU8::new(MusicMode::Carousel as u8);
static SFX_VOLUME: AtomicU32 = AtomicU32::new(1056964608);
static CAROUSEL_SELECTION: AtomicU8 = AtomicU8::new(CAROUSEL_ALL);

pub const CAROUSEL_ALL: u8 = 255;
pub const CAROUSEL_TRACK_COUNT: usize = 4;

const CAROUSEL_TRACK_NAMES: [&str; CAROUSEL_TRACK_COUNT] =
    ["Drift", "Signal", "Voyage", "Afterglow"];

pub fn carousel_track_label(sel: u8) -> &'static str {
    if (sel as usize) < CAROUSEL_TRACK_COUNT {
        CAROUSEL_TRACK_NAMES[sel as usize]
    } else {
        "All"
    }
}

pub fn set_carousel_track(sel: u8) {
    let v = if (sel as usize) < CAROUSEL_TRACK_COUNT {
        sel
    } else {
        CAROUSEL_ALL
    };
    CAROUSEL_SELECTION.store(v, Ordering::Relaxed);
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MusicMode {
    Carousel,
    Shop,
    Settings,
}

pub fn set_music(mode: MusicMode, target_gain: f32, lowpass: f32) {
    MUSIC_MODE.store(mode as u8, Ordering::Relaxed);
    MUSIC_TARGET.store(target_gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    MUSIC_LOWPASS.store(lowpass.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

pub fn set_sfx_volume(volume: f32) {
    SFX_VOLUME.store(volume.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

struct MusicTrack {
    samples: Vec<[f32; 2]>,
    sample_rate: f64,
}

impl MusicTrack {
    fn silent() -> Self {
        Self {
            samples: Vec::new(),
            sample_rate: SR as f64,
        }
    }
}

fn decode_mp3_stereo(bytes: &'static [u8]) -> Option<MusicTrack> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let format_options = FormatOptions {
        enable_gapless: true,
        ..FormatOptions::default()
    };
    let probed = symphonia::default::get_probe()
        .format(&hint, source, &format_options, &MetadataOptions::default())
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(SR as u32) as f64;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(err) => {
                log::warn!("Failed to read UI music packet: {err}");
                break;
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) => break,
            Err(err) => {
                log::warn!("Failed to decode UI music packet: {err}");
                return None;
            }
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        if channels == 0 {
            continue;
        }
        let mut interleaved = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        interleaved.copy_interleaved_ref(decoded);
        for frame in interleaved.samples().chunks(channels) {
            let left = frame[0];
            let right = frame.get(1).copied().unwrap_or(left);
            samples.push([left, right]);
        }
    }

    (!samples.is_empty()).then_some(MusicTrack {
        samples,
        sample_rate,
    })
}

fn random_music_variants() -> (usize, usize) {
    let choice = getrandom::u64().unwrap_or_else(|err| {
        log::warn!("OS randomness unavailable for UI music selection: {err}");
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        );
        hasher.write_u32(std::process::id());
        hasher.finish()
    });
    (
        choice as usize % CAROUSEL_MUSIC.len(),
        (choice >> 32) as usize % SHOP_MUSIC.len(),
    )
}

fn next_music_sample(
    track: &MusicTrack,
    position: &mut f64,
    output_sample_rate: f64,
    wrapped: &mut bool,
) -> [f32; 2] {
    let len = track.samples.len();
    if len == 0 {
        return [0.0; 2];
    }
    let index = *position as usize;
    let next = (index + 1) % len;
    let fraction = (*position - index as f64) as f32;
    let sample = [
        track.samples[index][0] + (track.samples[next][0] - track.samples[index][0]) * fraction,
        track.samples[index][1] + (track.samples[next][1] - track.samples[index][1]) * fraction,
    ];
    *position += track.sample_rate / output_sample_rate;
    if *position >= len as f64 {
        *position %= len as f64;
        *wrapped = true;
    }
    sample
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
    banks.insert(
        key(Sfx::Favorite),
        std::sync::Arc::new(render(Sfx::Favorite)),
    );
    banks.insert(key(Sfx::Open), std::sync::Arc::new(render(Sfx::Open)));
    banks.insert(key(Sfx::Boot), std::sync::Arc::new(render(Sfx::Boot)));
    banks.insert(key(Sfx::Whistle), std::sync::Arc::new(render(Sfx::Whistle)));
    banks.insert(
        key(Sfx::GameBoot),
        std::sync::Arc::new(render(Sfx::GameBoot)),
    );
    banks.insert(
        key(Sfx::AwaitFrame),
        std::sync::Arc::new(render(Sfx::AwaitFrame)),
    );
    banks.insert(
        key(Sfx::PleaseWait),
        std::sync::Arc::new(render(Sfx::PleaseWait)),
    );
    banks.insert(
        key(Sfx::Celebration),
        std::sync::Arc::new(render(Sfx::Celebration)),
    );
    banks.insert(
        key(Sfx::WhistleOk),
        std::sync::Arc::new(render(Sfx::WhistleOk)),
    );
    banks.insert(
        key(Sfx::WhistleSquish),
        std::sync::Arc::new(render(Sfx::WhistleSquish)),
    );

    let (carousel_start, shop_variant) = random_music_variants();
    let carousel_tracks: Vec<MusicTrack> = CAROUSEL_MUSIC
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            decode_mp3_stereo(bytes).unwrap_or_else(|| {
                log::warn!("Failed to decode carousel music variant {}", i + 1);
                MusicTrack::silent()
            })
        })
        .collect();
    let shop_tracks: Vec<MusicTrack> = SHOP_MUSIC
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            decode_mp3_stereo(bytes).unwrap_or_else(|| {
                log::warn!("Failed to decode shop music variant {}", i + 1);
                MusicTrack::silent()
            })
        })
        .collect();
    let settings_track = decode_mp3_stereo(SETTINGS_MUSIC).unwrap_or_else(|| {
        log::warn!("Failed to decode settings music");
        MusicTrack::silent()
    });
    let mut carousel_idx = carousel_start % carousel_tracks.len().max(1);
    let mut shop_idx = shop_variant % shop_tracks.len().max(1);
    log::info!(
        "UI music: {} carousel tracks decoded, shuffle start {}, shop variant {}",
        carousel_tracks.len(),
        carousel_idx + 1,
        shop_variant + 1
    );

    let host = cpal::default_host();
    let device = host.default_output_device()?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "default output".to_string());

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
    let gain_step = 1.0 / (0.45 * dev_sr);
    let mode_step = 1.0 / (0.45 * dev_sr);

    let active: std::sync::Arc<Mutex<Vec<Voice>>> = std::sync::Arc::new(Mutex::new(Vec::new()));
    let active_cb = active.clone();

    let mut cur_gain = 0.0f32;
    let mut music_pos = [0.0f64; 3];
    let mut shop_mix = 0.0f32;
    let mut settings_mix = 0.0f32;
    let mut lp_y = [0.0f32; 2];

    let err_cb = |e| log::warn!("ui_audio stream error: {}", e);
    let stream = device
        .build_output_stream(
            &cfg.config(),
            move |out: &mut [f32], _| {
                let frames = out.len() / channels.max(1);
                let target = f32::from_bits(MUSIC_TARGET.load(Ordering::Relaxed));
                let lp = f32::from_bits(MUSIC_LOWPASS.load(Ordering::Relaxed));
                let alpha = 1.0 - lp * 0.9;
                let mode = MUSIC_MODE.load(Ordering::Relaxed);
                let shop_target = if mode == MusicMode::Shop as u8 {
                    1.0
                } else {
                    0.0
                };
                let settings_target = if mode == MusicMode::Settings as u8 {
                    1.0
                } else {
                    0.0
                };
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
                    if shop_mix < shop_target {
                        shop_mix = (shop_mix + mode_step).min(shop_target);
                    } else if shop_mix > shop_target {
                        shop_mix = (shop_mix - mode_step).max(shop_target);
                    }
                    if settings_mix < settings_target {
                        settings_mix = (settings_mix + mode_step).min(settings_target);
                    } else if settings_mix > settings_target {
                        settings_mix = (settings_mix - mode_step).max(settings_target);
                    }
                    let carousel_mix = (1.0 - shop_mix - settings_mix).max(0.0);
                    let m = if target > 0.0001 || cur_gain > 0.0001 {
                        let carousel = if carousel_mix > 0.0 && !carousel_tracks.is_empty() {
                            let sel = CAROUSEL_SELECTION.load(Ordering::Relaxed);
                            if (sel as usize) < carousel_tracks.len()
                                && carousel_idx != sel as usize
                            {
                                carousel_idx = sel as usize;
                                music_pos[0] = 0.0;
                            }
                            let mut wrapped = false;
                            let s = next_music_sample(
                                &carousel_tracks[carousel_idx],
                                &mut music_pos[0],
                                dev_sr as f64,
                                &mut wrapped,
                            );
                            if wrapped && (sel as usize) >= carousel_tracks.len() {
                                carousel_idx = (carousel_idx + 1) % carousel_tracks.len();
                                music_pos[0] = 0.0;
                            }
                            s
                        } else {
                            [0.0; 2]
                        };
                        let shop = if shop_mix > 0.0 && !shop_tracks.is_empty() {
                            let mut wrapped = false;
                            let s = next_music_sample(
                                &shop_tracks[shop_idx],
                                &mut music_pos[1],
                                dev_sr as f64,
                                &mut wrapped,
                            );
                            if wrapped {
                                shop_idx = (shop_idx + 1) % shop_tracks.len();
                                music_pos[1] = 0.0;
                            }
                            s
                        } else {
                            [0.0; 2]
                        };
                        let settings = if settings_mix > 0.0 {
                            let mut wrapped = false;
                            next_music_sample(
                                &settings_track,
                                &mut music_pos[2],
                                dev_sr as f64,
                                &mut wrapped,
                            )
                        } else {
                            [0.0; 2]
                        };
                        [
                            carousel[0] * carousel_mix
                                + shop[0] * shop_mix
                                + settings[0] * settings_mix,
                            carousel[1] * carousel_mix
                                + shop[1] * shop_mix
                                + settings[1] * settings_mix,
                        ]
                    } else {
                        [0.0; 2]
                    };
                    lp_y[0] += alpha * (m[0] - lp_y[0]);
                    lp_y[1] += alpha * (m[1] - lp_y[1]);
                    let sfx_vol = f32::from_bits(SFX_VOLUME.load(Ordering::Relaxed));
                    for c in 0..channels {
                        let music_sample = if channels == 1 {
                            (lp_y[0] + lp_y[1]) * 0.5
                        } else if c < 2 {
                            lp_y[c]
                        } else {
                            0.0
                        };
                        out[f * channels + c] =
                            ((sfx * sfx_vol) + music_sample * cur_gain).clamp(-1.0, 1.0);
                    }
                }
                if let Some(a) = act.as_mut() {
                    a.retain(|v| {
                        v.looping || (v.pos as f32 * sfx_ratio) < v.buf.len() as f32 + 1.0
                    });
                }
            },
            err_cb,
            None,
        )
        .ok()?;
    stream.play().ok()?;
    log::info!(
        "UI audio stream: '{}' @ {} Hz, {} channels",
        device_name,
        dev_sr as u32,
        channels
    );

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
            tone_after_abs(&mut b, 0.0, 130.81, 2400.0, 0.12, 1800.0, 200.0);
            tone_after_abs(&mut b, 0.0, 196.00, 2400.0, 0.09, 1800.0, 200.0);
            tone_after_abs(&mut b, 400.0, 246.94, 2000.0, 0.09, 1500.0, 200.0);
            tone_after_abs(&mut b, 800.0, 293.66, 1600.0, 0.08, 1200.0, 200.0);
            tone_after_abs(&mut b, 1200.0, 369.99, 1200.0, 0.08, 900.0, 200.0);

            tone_after_abs(&mut b, 2350.0, 523.25, 2650.0, 0.08, 15.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 659.25, 2650.0, 0.08, 15.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 783.99, 2650.0, 0.08, 20.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 987.77, 2650.0, 0.07, 25.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 1174.66, 2650.0, 0.06, 30.0, 2200.0);
            tone_after_abs(&mut b, 2350.0, 1760.00, 2650.0, 0.05, 40.0, 2200.0);

            tone_after_abs(&mut b, 2450.0, 1318.51, 1500.0, 0.04, 10.0, 1200.0);
            tone_after_abs(&mut b, 2600.0, 1567.98, 1500.0, 0.04, 10.0, 1200.0);
            tone_after_abs(&mut b, 2750.0, 1975.53, 1500.0, 0.03, 10.0, 1200.0);
            tone_after_abs(&mut b, 2900.0, 2349.32, 1500.0, 0.03, 10.0, 1200.0);
        }
        Sfx::Whistle => {
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
            whistle(&mut b, 720.0, 1360.0, 0.24, 55.0, 0.19);
            tone_after_abs(&mut b, 180.0, 1720.0, 120.0, 0.12, 4.0, 100.0);
        }
        Sfx::WhistleSquish => {
            whistle(&mut b, 820.0, 1300.0, 0.13, 40.0, 0.2);
            let s = b.len();
            let _ = s;
            whistle_at(&mut b, 120.0, 1300.0, 880.0, 0.16, 120.0, 0.18);
        }
        Sfx::GameBoot => {
            tone_after_abs(&mut b, 0.0, 146.83, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 20.0, 174.61, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 40.0, 220.00, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 60.0, 261.63, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 80.0, 329.63, 300.0, 0.10, 10.0, 60.0);

            tone_after_abs(&mut b, 300.0, 196.00, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 320.0, 246.94, 300.0, 0.12, 10.0, 60.0);
            tone_after_abs(&mut b, 340.0, 349.23, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 360.0, 440.00, 300.0, 0.10, 10.0, 60.0);
            tone_after_abs(&mut b, 380.0, 659.25, 300.0, 0.08, 10.0, 60.0);

            tone_after_abs(&mut b, 600.0, 130.81, 1400.0, 0.14, 15.0, 800.0);
            tone_after_abs(&mut b, 620.0, 196.00, 1400.0, 0.12, 15.0, 800.0);
            tone_after_abs(&mut b, 640.0, 246.94, 1400.0, 0.12, 15.0, 800.0);
            tone_after_abs(&mut b, 660.0, 293.66, 1400.0, 0.10, 15.0, 800.0);
            tone_after_abs(&mut b, 680.0, 329.63, 1400.0, 0.10, 15.0, 800.0);
            tone_after_abs(&mut b, 700.0, 392.00, 1400.0, 0.08, 15.0, 800.0);

            tone_after_abs(&mut b, 800.0, 493.88, 200.0, 0.09, 10.0, 80.0);
            tone_after_abs(&mut b, 1000.0, 587.33, 200.0, 0.09, 10.0, 80.0);
            tone_after_abs(&mut b, 1200.0, 783.99, 800.0, 0.08, 10.0, 400.0);
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
                    let note_breathe = 0.4
                        + 0.6
                            * (t * std::f32::consts::TAU / 3.0 + idx as f32 * 0.5)
                                .cos()
                                .abs();
                    s += wave * note_breathe;
                }
                b[i] = (s / freqs.len() as f32) * 0.16 * breathe;
            }
        }
        Sfx::Celebration => {
            let lead = [523.25f32, 659.25, 783.99, 1046.5];
            for (i, &f) in lead.iter().enumerate() {
                tone_after_abs(&mut b, i as f32 * 70.0, f, 260.0, 0.14, 4.0, 200.0);
            }
            tone_after_abs(&mut b, 210.0, 523.25, 620.0, 0.10, 10.0, 480.0);
            tone_after_abs(&mut b, 220.0, 659.25, 620.0, 0.09, 10.0, 480.0);
            tone_after_abs(&mut b, 230.0, 783.99, 620.0, 0.09, 10.0, 480.0);
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
                    b[start + j] +=
                        s * 0.06 * env(j, n, (SR * 0.004) as usize, (SR * 0.07) as usize);
                }
            }
        }
    }
    b
}

fn tone_after(
    buf: &mut Vec<f32>,
    gap_ms: usize,
    freq: f32,
    dur_ms: f32,
    gain: f32,
    attack_ms: f32,
    release_ms: f32,
) {
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

fn tone_after_abs(
    buf: &mut Vec<f32>,
    offset_ms: f32,
    freq: f32,
    dur_ms: f32,
    gain: f32,
    attack_ms: f32,
    release_ms: f32,
) {
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

fn whistle_at(
    buf: &mut Vec<f32>,
    offset_ms: f32,
    f0: f32,
    f1: f32,
    dur: f32,
    warble: f32,
    gain: f32,
) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_music_decodes() {
        for bytes in CAROUSEL_MUSIC.into_iter().chain(SHOP_MUSIC) {
            assert!(!bytes.starts_with(b"ID3"));
            let track = decode_mp3_stereo(bytes).expect("embedded MP3 should decode");
            assert_eq!(track.sample_rate, 48_000.0);
            assert!(track.samples.len() > 45 * 48_000);
            assert!(track
                .samples
                .iter()
                .any(|sample| (sample[0] - sample[1]).abs() > 0.000_001));
        }
    }
}

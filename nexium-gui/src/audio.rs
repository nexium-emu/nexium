
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, StreamConfig};
use nexium_kernel::audio_sink::{set_host_audio_sink, HostPcmSink};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const RENDER_SR: u32 = 48_000;
const RB_CAP_SAMPLES: usize = 48_000;
const PREBUF_TARGET_SAMPLES: usize = 9_600;

struct PrebufState {
    priming: bool,
    cur_l: f32,
    cur_r: f32,
    empty_run: u32,
}
impl PrebufState {
    fn new() -> Self {
        Self { priming: true, cur_l: 0.0, cur_r: 0.0, empty_run: 0 }
    }
}

type Prod = <HeapRb<f32> as Split>::Prod;

#[derive(Clone, Debug)]
pub struct AudioStreamInfo {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: &'static str,
}

static STREAM_INFO: OnceLock<AudioStreamInfo> = OnceLock::new();
static SINK_HANDLE: OnceLock<Arc<HostAudioSink>> = OnceLock::new();

pub static AUDIO_EVENTS_PENDING: AtomicU64 = AtomicU64::new(0);

pub fn drain_audio_events_pending() -> u64 {
    AUDIO_EVENTS_PENDING.swap(0, Ordering::Relaxed)
}

pub fn repost_pending_events(n: u64) {
    if n != 0 {
        AUDIO_EVENTS_PENDING.fetch_add(n, Ordering::Relaxed);
    }
}

pub fn current_stream_info() -> Option<AudioStreamInfo> {
    STREAM_INFO.get().cloned()
}

pub fn list_output_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.output_devices() {
        Ok(it) => it.filter_map(|d| d.name().ok()).collect(),
        Err(e) => {
            log::warn!("output_devices enumerate failed: {}", e);
            Vec::new()
        }
    }
}

pub fn set_master_volume(vol: f32) {
    if let Some(sink) = SINK_HANDLE.get() {
        let clamped = vol.clamp(0.0, 2.0);
        sink.volume.store(clamped.to_bits(), Ordering::Relaxed);
    }
}

pub fn push_test_tone(freq_hz: f32, seconds: f32) -> usize {
    let Some(sink) = SINK_HANDLE.get() else { return 0; };
    let frames = (RENDER_SR as f32 * seconds).max(0.0) as usize;
    let mut buf = Vec::with_capacity(frames * 2);
    let two_pi_over_sr = std::f32::consts::TAU * freq_hz / RENDER_SR as f32;
    for i in 0..frames {
        let s = 0.25 * (two_pi_over_sr * i as f32).sin();
        buf.push(s);
        buf.push(s);
    }
    sink.push_stereo_f32(&buf)
}

fn resolve_device(preferred: Option<&str>) -> Option<Device> {
    let host = cpal::default_host();
    if let Some(name) = preferred {
        match host.output_devices() {
            Ok(it) => {
                for d in it {
                    if d.name().ok().as_deref() == Some(name) {
                        return Some(d);
                    }
                }
                log::warn!("Audio device '{}' not found; falling back to default", name);
            }
            Err(e) => log::warn!("output_devices enumerate failed: {}; using default", e),
        }
    }
    host.default_output_device()
}

pub struct HostAudioSink {
    producer: Mutex<Prod>,
    consumed: Arc<AtomicU64>,
    volume: Arc<AtomicU32>,
}

impl HostPcmSink for HostAudioSink {
    fn push_stereo_f32(&self, samples: &[f32]) -> usize {
        let mut prod = match self.producer.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let pushed = prod.push_slice(samples);
        pushed / 2
    }

    fn queued_frames(&self) -> usize {
        let prod = match self.producer.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        prod.occupied_len() / 2
    }

    fn vacant_frames(&self) -> usize {
        let prod = match self.producer.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        prod.vacant_len() / 2
    }

    fn samples_consumed(&self) -> u64 {
        self.consumed.load(Ordering::Relaxed)
    }

    fn sample_rate(&self) -> u32 {
        RENDER_SR
    }

    fn drain_pending_events(&self) -> u64 {
        drain_audio_events_pending()
    }

    fn repost_pending_events(&self, n: u64) {
        repost_pending_events(n);
    }
}

fn pick_config(
    device: &cpal::Device,
) -> Result<(StreamConfig, SampleFormat), String> {
    let supported: Vec<_> = device
        .supported_output_configs()
        .map_err(|e| format!("supported_output_configs: {}", e))?
        .collect();

    let want_sr = SampleRate(RENDER_SR);
    for fmt in [SampleFormat::F32, SampleFormat::I16, SampleFormat::U16] {
        if let Some(range) = supported.iter().find(|r| {
            r.channels() == 2
                && r.sample_format() == fmt
                && r.min_sample_rate() <= want_sr
                && r.max_sample_rate() >= want_sr
        }) {
            return Ok((range.with_sample_rate(want_sr).config(), fmt));
        }
    }
    let def = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {}", e))?;
    Ok((def.config(), def.sample_format()))
}

pub fn init_host_audio(preferred_device: Option<&str>, initial_volume: f32) {
    let preferred = preferred_device.map(|s| s.to_string());
    let _ = std::thread::Builder::new()
        .name("nexium-audio".into())
        .spawn(move || {
            init_host_audio_on_thread(preferred.as_deref(), initial_volume);
            loop {
                std::thread::park();
            }
        });
}

fn init_host_audio_on_thread(preferred_device: Option<&str>, initial_volume: f32) {
    let Some(device) = resolve_device(preferred_device) else {
        log::warn!("Audio output disabled: no output device available");
        return;
    };

    let (config, sample_format) = match pick_config(&device) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("Audio output disabled: {}", e);
            return;
        }
    };
    let device_sr = config.sample_rate.0;
    let device_ch = config.channels;

    let rb = HeapRb::<f32>::new(RB_CAP_SAMPLES);
    let (producer, mut consumer) = rb.split();
    let consumed = Arc::new(AtomicU64::new(0));
    let consumed_cb = consumed.clone();
    let volume = Arc::new(AtomicU32::new(initial_volume.clamp(0.0, 2.0).to_bits()));
    let volume_cb = volume.clone();

    let resample_ratio = RENDER_SR as f32 / device_sr as f32;
    let _need_resample = (RENDER_SR != device_sr) || (device_ch != 2);

    let err_cb = |e| log::error!("cpal stream error: {}", e);

    let dev_ch = device_ch as usize;

    let stream_result = match sample_format {
        SampleFormat::F32 => {
            let vol = volume_cb.clone();
            let mut prebuf = PrebufState::new();
            device.build_output_stream(
                &config,
                move |out: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    drain_stereo_to(&mut consumer, out, dev_ch, resample_ratio, &consumed_cb, &vol, &mut prebuf);
                },
                err_cb,
                None,
            )
        }
        SampleFormat::I16 => {
            let consumed_cb = consumed_cb.clone();
            let vol = volume_cb.clone();
            let mut prebuf = PrebufState::new();
            device.build_output_stream(
                &config,
                move |out: &mut [i16], _info: &cpal::OutputCallbackInfo| {
                    drain_stereo_to_i16(&mut consumer, out, dev_ch, resample_ratio, &consumed_cb, &vol, &mut prebuf);
                },
                err_cb,
                None,
            )
        }
        other => {
            log::warn!("Audio output disabled: unsupported sample format {:?}", other);
            return;
        }
    };

    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Audio output disabled: build_output_stream: {}", e);
            return;
        }
    };

    if let Err(e) = stream.play() {
        log::warn!("Audio output disabled: stream.play: {}", e);
        return;
    }

    let device_name = device.name().unwrap_or_else(|_| "<unknown>".into());
    let fmt_str: &'static str = match sample_format {
        SampleFormat::F32 => "F32",
        SampleFormat::I16 => "I16",
        SampleFormat::U16 => "U16",
        _ => "other",
    };
    log::info!(
        "HostAudioSink: cpal device '{}' @ {} Hz {}ch {} (ring cap {} samples, vol {:.2})",
        device_name, device_sr, device_ch, fmt_str, RB_CAP_SAMPLES, initial_volume,
    );

    let _ = STREAM_INFO.set(AudioStreamInfo {
        device_name,
        sample_rate: device_sr,
        channels: device_ch,
        sample_format: fmt_str,
    });

    let sink = Arc::new(HostAudioSink {
        producer: Mutex::new(producer),
        consumed,
        volume,
    });
    let _ = SINK_HANDLE.set(sink.clone());
    let sink_dyn: Arc<dyn HostPcmSink> = sink;
    set_host_audio_sink(sink_dyn);

    std::thread_local! {
        static AUDIO_STREAM: std::cell::RefCell<Option<cpal::Stream>> =
            std::cell::RefCell::new(None);
    }
    AUDIO_STREAM.with(|s| *s.borrow_mut() = Some(stream));

    let frames = (RENDER_SR as usize) / 2;
    let mut beep = Vec::with_capacity(frames * 2);
    let phase_inc = std::f32::consts::TAU * 440.0 / RENDER_SR as f32;
    for i in 0..frames {
        let s = 0.25 * (phase_inc * i as f32).sin();
        beep.push(s);
        beep.push(s);
    }
    if let Some(h) = SINK_HANDLE.get() {
        let pushed = h.push_stereo_f32(&beep);
        log::info!("Audio startup beep: pushed {} frames @ 440 Hz", pushed);
    }
}

fn drain_stereo_to(
    consumer: &mut <HeapRb<f32> as Split>::Cons,
    out: &mut [f32],
    dev_ch: usize,
    resample_ratio: f32,
    consumed: &AtomicU64,
    volume: &AtomicU32,
    prebuf: &mut PrebufState,
) {
    use std::sync::atomic::{AtomicBool, AtomicU32 as AU32};
    static FIRED: AtomicBool = AtomicBool::new(false);
    static CALL_COUNT: AU32 = AU32::new(0);
    if !FIRED.swap(true, Ordering::Relaxed) {
        log::info!(
            "cpal callback FIRST FIRE: out_len={} dev_ch={} ratio={:.3}",
            out.len(), dev_ch, resample_ratio
        );
    }
    let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    let vol = f32::from_bits(volume.load(Ordering::Relaxed));

    let occ = consumer.occupied_len();
    if prebuf.priming {
        if occ >= PREBUF_TARGET_SAMPLES {
            prebuf.priming = false;
            prebuf.empty_run = 0;
        }
    } else if occ == 0 {
        prebuf.empty_run = prebuf.empty_run.saturating_add(1);
        if prebuf.empty_run >= 8 {
            prebuf.priming = true;
        }
    } else {
        prebuf.empty_run = 0;
    }
    let priming = prebuf.priming;

    let mut acc: f32 = 1.0;
    let mut frames_written = 0usize;
    let mut peak: f32 = 0.0;
    for chunk in out.chunks_mut(dev_ch) {
        if !priming {
            while acc >= 1.0 {
                if let (Some(l), Some(r)) = (consumer.try_pop(), consumer.try_pop()) {
                    prebuf.cur_l = l;
                    prebuf.cur_r = r;
                }
                acc -= 1.0;
            }
        }
        let (l, r) = if priming {
            (0.0, 0.0)
        } else {
            (prebuf.cur_l * vol, prebuf.cur_r * vol)
        };
        peak = peak.max(l.abs()).max(r.abs());
        if dev_ch == 1 {
            chunk[0] = 0.5 * (l + r);
        } else {
            chunk[0] = l;
            chunk[1] = r;
            for c in 2..dev_ch { chunk[c] = 0.0; }
        }
        acc += resample_ratio;
        frames_written += 1;
    }
    if n == 1 || (n > 0 && n % 100 == 0) {
        log::info!(
            "cpal callback #{}: frames_written={} peak_amp={:.4} priming={} occ={}",
            n, frames_written, peak, priming, occ
        );
    }
    let render_frames = ((frames_written as f32) * resample_ratio).round() as u64;
    let new_consumed = consumed.fetch_add(render_frames, Ordering::Relaxed) + render_frames;
    post_audio_events(new_consumed);
}

fn drain_stereo_to_i16(
    consumer: &mut <HeapRb<f32> as Split>::Cons,
    out: &mut [i16],
    dev_ch: usize,
    resample_ratio: f32,
    consumed: &AtomicU64,
    volume: &AtomicU32,
    prebuf: &mut PrebufState,
) {
    let vol = f32::from_bits(volume.load(Ordering::Relaxed));

    let occ = consumer.occupied_len();
    if prebuf.priming {
        if occ >= PREBUF_TARGET_SAMPLES {
            prebuf.priming = false;
            prebuf.empty_run = 0;
        }
    } else if occ == 0 {
        prebuf.empty_run = prebuf.empty_run.saturating_add(1);
        if prebuf.empty_run >= 8 {
            prebuf.priming = true;
        }
    } else {
        prebuf.empty_run = 0;
    }
    let priming = prebuf.priming;

    let mut acc: f32 = 1.0;
    let mut frames_written = 0usize;
    for chunk in out.chunks_mut(dev_ch) {
        if !priming {
            while acc >= 1.0 {
                if let (Some(l), Some(r)) = (consumer.try_pop(), consumer.try_pop()) {
                    prebuf.cur_l = l;
                    prebuf.cur_r = r;
                }
                acc -= 1.0;
            }
        }
        let (l, r) = if priming {
            (0.0, 0.0)
        } else {
            (prebuf.cur_l * vol, prebuf.cur_r * vol)
        };
        let li = (l.clamp(-1.0, 1.0) * 32767.0) as i16;
        let ri = (r.clamp(-1.0, 1.0) * 32767.0) as i16;
        if dev_ch == 1 {
            chunk[0] = ((li as i32 + ri as i32) / 2) as i16;
        } else {
            chunk[0] = li;
            chunk[1] = ri;
            for c in 2..dev_ch { chunk[c] = 0; }
        }
        acc += resample_ratio;
        frames_written += 1;
    }
    let render_frames = ((frames_written as f32) * resample_ratio).round() as u64;
    let new_consumed = consumed.fetch_add(render_frames, Ordering::Relaxed) + render_frames;
    post_audio_events(new_consumed);
}

fn post_audio_events(new_consumed: u64) {
    const FRAMES_PER_AUDIO_FRAME: u64 = 240;
    static LAST_SIGNALED: AtomicU64 = AtomicU64::new(0);
    let last = LAST_SIGNALED.load(Ordering::Relaxed);
    if new_consumed <= last {
        return;
    }
    let blocks = (new_consumed - last) / FRAMES_PER_AUDIO_FRAME;
    if blocks == 0 {
        return;
    }
    LAST_SIGNALED.store(last + blocks * FRAMES_PER_AUDIO_FRAME, Ordering::Relaxed);
    AUDIO_EVENTS_PENDING.fetch_add(blocks, Ordering::Relaxed);
}

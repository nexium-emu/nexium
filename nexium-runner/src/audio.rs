use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use nexium_kernel::audio_sink::{set_host_audio_sink, HostPcmSink};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const RENDER_SR: u32 = 48_000;
const RB_CAP_SAMPLES: usize = 96_000;
const PREBUF_TARGET_SAMPLES: usize = 5_760;

struct PrebufState {
    priming: bool,
    cur_l: f32,
    cur_r: f32,
    prev_l: f32,
    prev_r: f32,
    pos: f32,
    saw_emulated_audio: bool,
}
impl PrebufState {
    fn new(_nominal_ratio: f32) -> Self {
        Self {
            priming: true,
            cur_l: 0.0,
            cur_r: 0.0,
            prev_l: 0.0,
            prev_r: 0.0,
            pos: 1.0,
            saw_emulated_audio: false,
        }
    }

    fn callback_ratio(&self, _frames_written: usize, nominal_ratio: f32) -> f32 {
        nominal_ratio.clamp(0.05, 4.0)
    }
}

type Prod = <HeapRb<f32> as Split>::Prod;

const MAX_AUDIO_OUT_STREAMS: usize = 12;

struct AudioOutHostStream {
    id: u64,
    started: bool,
    samples: VecDeque<f32>,
    consumed_frames: u64,
    volume: f32,
}

#[derive(Default)]
struct AudioOutMixer {
    streams: Vec<AudioOutHostStream>,
}

impl AudioOutMixer {
    fn open(&mut self, id: u64) -> bool {
        if self.streams.iter().any(|stream| stream.id == id) {
            return true;
        }
        if self.streams.len() >= MAX_AUDIO_OUT_STREAMS {
            return false;
        }
        self.streams.push(AudioOutHostStream {
            id,
            started: false,
            samples: VecDeque::new(),
            consumed_frames: 0,
            volume: 1.0,
        });
        true
    }

    fn close(&mut self, id: u64) {
        self.streams.retain(|stream| stream.id != id);
    }

    fn start(&mut self, id: u64) {
        if let Some(stream) = self.streams.iter_mut().find(|stream| stream.id == id) {
            stream.started = true;
        }
    }

    fn stop(&mut self, id: u64) {
        if let Some(stream) = self.streams.iter_mut().find(|stream| stream.id == id) {
            stream.started = false;
            stream.samples.clear();
        }
    }

    fn push(&mut self, id: u64, samples: &[f32]) -> usize {
        let Some(stream) = self.streams.iter_mut().find(|stream| stream.id == id) else {
            return 0;
        };
        let sample_count = samples.len() & !1;
        stream
            .samples
            .extend(samples[..sample_count].iter().copied());
        sample_count / 2
    }

    fn consumed(&self, id: u64) -> u64 {
        self.streams
            .iter()
            .find(|stream| stream.id == id)
            .map_or(0, |stream| stream.consumed_frames)
    }

    fn set_volume(&mut self, id: u64, volume: f32) {
        if let Some(stream) = self.streams.iter_mut().find(|stream| stream.id == id) {
            stream.volume = volume;
        }
    }

    fn has_ready_frame(&self) -> bool {
        self.streams
            .iter()
            .any(|stream| stream.started && stream.samples.len() >= 2)
    }

    fn mix_next_frame(&mut self) -> (f32, f32) {
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for stream in &mut self.streams {
            if !stream.started || stream.samples.len() < 2 {
                continue;
            }
            let sample_l = stream.samples.pop_front().unwrap_or(0.0);
            let sample_r = stream.samples.pop_front().unwrap_or(0.0);
            left += sample_l * stream.volume;
            right += sample_r * stream.volume;
            stream.consumed_frames = stream.consumed_frames.saturating_add(1);
        }
        (left, right)
    }
}

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
        Ok(it) => it
            .filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
            .collect(),
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
    let Some(sink) = SINK_HANDLE.get() else {
        return 0;
    };
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
                    if d.description()
                        .ok()
                        .map(|desc| desc.name().to_string())
                        .as_deref()
                        == Some(name)
                    {
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
    audio_out: Arc<Mutex<AudioOutMixer>>,
    underrun_frames: Arc<AtomicU64>,
    dropped_frames: AtomicU64,
}

impl HostPcmSink for HostAudioSink {
    fn push_stereo_f32(&self, samples: &[f32]) -> usize {
        let mut prod = match self.producer.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let requested = samples.len() & !1;
        let pushed = prod.push_slice(&samples[..requested]);
        if pushed < requested {
            self.dropped_frames
                .fetch_add(((requested - pushed) / 2) as u64, Ordering::Relaxed);
        }
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

    fn open_audio_out_stream(&self, stream_id: u64) -> bool {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.open(stream_id)
    }

    fn close_audio_out_stream(&self, stream_id: u64) {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.close(stream_id);
    }

    fn start_audio_out_stream(&self, stream_id: u64) {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.start(stream_id);
    }

    fn stop_audio_out_stream(&self, stream_id: u64) {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.stop(stream_id);
    }

    fn push_audio_out_stereo_f32(&self, stream_id: u64, samples: &[f32]) -> usize {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.push(stream_id, samples)
    }

    fn audio_out_samples_consumed(&self, stream_id: u64) -> u64 {
        let mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.consumed(stream_id)
    }

    fn set_audio_out_volume(&self, stream_id: u64, volume: f32) {
        let mut mixer = self.audio_out.lock().unwrap_or_else(|p| p.into_inner());
        mixer.set_volume(stream_id, volume);
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

    fn drain_underrun_frames(&self) -> u64 {
        self.underrun_frames.swap(0, Ordering::Relaxed)
    }

    fn drain_dropped_frames(&self) -> u64 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }
}

#[cfg(target_os = "android")]
const DESIRED_BUFFER_FRAMES: Option<u32> = Some(1024);
#[cfg(not(target_os = "android"))]
const DESIRED_BUFFER_FRAMES: Option<u32> = None;

fn apply_buffer_size(config: &mut StreamConfig, supported: Option<&cpal::SupportedBufferSize>) {
    let Some(want) = DESIRED_BUFFER_FRAMES else {
        return;
    };
    let frames = match supported {
        Some(cpal::SupportedBufferSize::Range { min, max }) => want.clamp(*min, *max),
        _ => want,
    };
    config.buffer_size = cpal::BufferSize::Fixed(frames);
    log::info!("audio: requesting fixed buffer of {} frames", frames);
}

fn pick_config(device: &cpal::Device) -> Result<(StreamConfig, SampleFormat), String> {
    let supported: Vec<_> = device
        .supported_output_configs()
        .map_err(|e| format!("supported_output_configs: {}", e))?
        .collect();

    let want_sr = RENDER_SR;
    if let Some((range, format)) = supported
        .iter()
        .filter_map(|range| {
            output_config_preference(
                range.channels(),
                range.sample_format(),
                range.min_sample_rate(),
                range.max_sample_rate(),
                want_sr,
            )
            .map(|rank| (rank, range))
        })
        .min_by_key(|(rank, _)| *rank)
        .map(|(_, range)| (range, range.sample_format()))
    {
        let mut config = range.with_sample_rate(want_sr).config();
        apply_buffer_size(&mut config, Some(range.buffer_size()));
        return Ok((config, format));
    }
    let def = device
        .default_output_config()
        .map_err(|e| format!("default_output_config: {}", e))?;
    let format = def.sample_format();
    let buffer = def.buffer_size().clone();
    let mut config = def.config();
    apply_buffer_size(&mut config, Some(&buffer));
    Ok((config, format))
}

fn output_config_preference(
    channels: u16,
    format: SampleFormat,
    min_sample_rate: u32,
    max_sample_rate: u32,
    wanted_sample_rate: u32,
) -> Option<(u8, u8, u16)> {
    if channels == 0
        || min_sample_rate > wanted_sample_rate
        || max_sample_rate < wanted_sample_rate
    {
        return None;
    }
    let format_rank = match format {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 1,
        _ => return None,
    };
    Some((u8::from(channels != 2), format_rank, channels.abs_diff(2)))
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
    let device_sr = config.sample_rate;
    let device_ch = config.channels;

    let rb = HeapRb::<f32>::new(RB_CAP_SAMPLES);
    let (producer, mut consumer) = rb.split();
    let consumed = Arc::new(AtomicU64::new(0));
    let consumed_cb = consumed.clone();
    let volume = Arc::new(AtomicU32::new(initial_volume.clamp(0.0, 2.0).to_bits()));
    let volume_cb = volume.clone();
    let audio_out = Arc::new(Mutex::new(AudioOutMixer::default()));
    let underrun_frames = Arc::new(AtomicU64::new(0));
    let underrun_cb = underrun_frames.clone();

    let resample_ratio = RENDER_SR as f32 / device_sr as f32;
    let _need_resample = (RENDER_SR != device_sr) || (device_ch != 2);

    let err_cb = |e| log::error!("cpal stream error: {}", e);

    let dev_ch = device_ch as usize;

    let stream_result = match sample_format {
        SampleFormat::F32 => {
            let vol = volume_cb.clone();
            let audio_out_cb = audio_out.clone();
            let mut prebuf = PrebufState::new(resample_ratio);
            device.build_output_stream(
                config.clone(),
                move |out: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    drain_stereo_to(
                        &mut consumer,
                        out,
                        dev_ch,
                        resample_ratio,
                        &consumed_cb,
                        &vol,
                        &audio_out_cb,
                        &underrun_cb,
                        &mut prebuf,
                    );
                },
                err_cb,
                None,
            )
        }
        SampleFormat::I16 => {
            let consumed_cb = consumed_cb.clone();
            let vol = volume_cb.clone();
            let audio_out_cb = audio_out.clone();
            let underrun_cb = underrun_cb.clone();
            let mut prebuf = PrebufState::new(resample_ratio);
            device.build_output_stream(
                config.clone(),
                move |out: &mut [i16], _info: &cpal::OutputCallbackInfo| {
                    drain_stereo_to_i16(
                        &mut consumer,
                        out,
                        dev_ch,
                        resample_ratio,
                        &consumed_cb,
                        &vol,
                        &audio_out_cb,
                        &underrun_cb,
                        &mut prebuf,
                    );
                },
                err_cb,
                None,
            )
        }
        other => {
            log::warn!(
                "Audio output disabled: unsupported sample format {:?}",
                other
            );
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

    let device_name = device
        .description()
        .map(|desc| desc.name().to_string())
        .unwrap_or_else(|_| "<unknown>".into());
    let fmt_str: &'static str = match sample_format {
        SampleFormat::F32 => "F32",
        SampleFormat::I16 => "I16",
        SampleFormat::U16 => "U16",
        _ => "other",
    };
    log::info!(
        "HostAudioSink: cpal device '{}' @ {} Hz {}ch {} (ring cap {} samples, vol {:.2})",
        device_name,
        device_sr,
        device_ch,
        fmt_str,
        RB_CAP_SAMPLES,
        initial_volume,
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
        audio_out,
        underrun_frames,
        dropped_frames: AtomicU64::new(0),
    });
    let _ = SINK_HANDLE.set(sink.clone());
    let sink_dyn: Arc<dyn HostPcmSink> = sink;
    set_host_audio_sink(sink_dyn);

    std::thread_local! {
        static AUDIO_STREAM: std::cell::RefCell<Option<cpal::Stream>> =
            std::cell::RefCell::new(None);
    }
    AUDIO_STREAM.with(|s| *s.borrow_mut() = Some(stream));
}

fn drain_stereo_to(
    consumer: &mut <HeapRb<f32> as Split>::Cons,
    out: &mut [f32],
    dev_ch: usize,
    resample_ratio: f32,
    consumed: &AtomicU64,
    volume: &AtomicU32,
    audio_out: &Mutex<AudioOutMixer>,
    underrun_frames: &AtomicU64,
    prebuf: &mut PrebufState,
) {
    let vol = f32::from_bits(volume.load(Ordering::Relaxed));

    let occ = consumer.occupied_len();
    let mut audio_out = audio_out.lock().unwrap_or_else(|p| p.into_inner());
    if prebuf.priming && (occ >= PREBUF_TARGET_SAMPLES || audio_out.has_ready_frame()) {
        prebuf.priming = false;
    }
    let priming = prebuf.priming;
    let ratio = prebuf.callback_ratio(out.len() / dev_ch, resample_ratio);

    let mut frames_written = 0usize;
    for chunk in out.chunks_mut(dev_ch) {
        if !priming {
            while prebuf.pos >= 1.0 {
                prebuf.prev_l = prebuf.cur_l;
                prebuf.prev_r = prebuf.cur_r;
                (prebuf.cur_l, prebuf.cur_r) = pop_mixed_stereo_frame(
                    consumer,
                    &mut audio_out,
                    underrun_frames,
                    &mut prebuf.saw_emulated_audio,
                );
                prebuf.pos -= 1.0;
            }
        }
        let (l, r) = if priming {
            (0.0, 0.0)
        } else {
            let f = prebuf.pos;
            (
                (prebuf.prev_l + (prebuf.cur_l - prebuf.prev_l) * f) * vol,
                (prebuf.prev_r + (prebuf.cur_r - prebuf.prev_r) * f) * vol,
            )
        };
        let (l, r) = clamp_stereo_output(l, r);
        if dev_ch == 1 {
            chunk[0] = 0.5 * (l + r);
        } else {
            chunk[0] = l;
            chunk[1] = r;
            for c in 2..dev_ch {
                chunk[c] = 0.0;
            }
        }
        if !priming {
            prebuf.pos += ratio;
        }
        frames_written += 1;
    }
    let render_frames = ((frames_written as f32) * ratio).round() as u64;
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
    audio_out: &Mutex<AudioOutMixer>,
    underrun_frames: &AtomicU64,
    prebuf: &mut PrebufState,
) {
    let vol = f32::from_bits(volume.load(Ordering::Relaxed));

    let occ = consumer.occupied_len();
    let mut audio_out = audio_out.lock().unwrap_or_else(|p| p.into_inner());
    if prebuf.priming && (occ >= PREBUF_TARGET_SAMPLES || audio_out.has_ready_frame()) {
        prebuf.priming = false;
    }
    let priming = prebuf.priming;
    let ratio = prebuf.callback_ratio(out.len() / dev_ch, resample_ratio);

    let mut frames_written = 0usize;
    for chunk in out.chunks_mut(dev_ch) {
        if !priming {
            while prebuf.pos >= 1.0 {
                prebuf.prev_l = prebuf.cur_l;
                prebuf.prev_r = prebuf.cur_r;
                (prebuf.cur_l, prebuf.cur_r) = pop_mixed_stereo_frame(
                    consumer,
                    &mut audio_out,
                    underrun_frames,
                    &mut prebuf.saw_emulated_audio,
                );
                prebuf.pos -= 1.0;
            }
        }
        let (l, r) = if priming {
            (0.0, 0.0)
        } else {
            let f = prebuf.pos;
            (
                prebuf.prev_l + (prebuf.cur_l - prebuf.prev_l) * f,
                prebuf.prev_r + (prebuf.cur_r - prebuf.prev_r) * f,
            )
        };
        let li = ((l * vol).clamp(-1.0, 1.0) * 32767.0) as i16;
        let ri = ((r * vol).clamp(-1.0, 1.0) * 32767.0) as i16;
        if dev_ch == 1 {
            chunk[0] = ((li as i32 + ri as i32) / 2) as i16;
        } else {
            chunk[0] = li;
            chunk[1] = ri;
            for c in 2..dev_ch {
                chunk[c] = 0;
            }
        }
        if !priming {
            prebuf.pos += ratio;
        }
        frames_written += 1;
    }
    let render_frames = ((frames_written as f32) * ratio).round() as u64;
    let new_consumed = consumed.fetch_add(render_frames, Ordering::Relaxed) + render_frames;
    post_audio_events(new_consumed);
}

fn pop_mixed_stereo_frame(
    consumer: &mut <HeapRb<f32> as Split>::Cons,
    audio_out: &mut AudioOutMixer,
    underrun_frames: &AtomicU64,
    saw_emulated_audio: &mut bool,
) -> (f32, f32) {
    let (mut left, mut right) = if consumer.occupied_len() >= 2 {
        *saw_emulated_audio = true;
        (
            consumer.try_pop().unwrap_or(0.0),
            consumer.try_pop().unwrap_or(0.0),
        )
    } else {
        if *saw_emulated_audio {
            underrun_frames.fetch_add(1, Ordering::Relaxed);
        }
        (0.0, 0.0)
    };
    let (audio_out_left, audio_out_right) = audio_out.mix_next_frame();
    left += audio_out_left;
    right += audio_out_right;
    (left, right)
}

fn clamp_stereo_output(left: f32, right: f32) -> (f32, f32) {
    (left.clamp(-1.0, 1.0), right.clamp(-1.0, 1.0))
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

#[cfg(test)]
mod tests {
    use super::{
        clamp_stereo_output, output_config_preference, pop_mixed_stereo_frame, AudioOutMixer,
        PrebufState,
    };
    use cpal::SampleFormat;
    use ringbuf::traits::{Producer, Split};
    use ringbuf::HeapRb;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn audio_out_streams_mix_samplewise_with_independent_clocks() {
        let mut mixer = AudioOutMixer::default();
        assert!(mixer.open(1));
        assert!(mixer.open(2));
        mixer.start(1);
        mixer.start(2);
        assert_eq!(mixer.push(1, &[0.25, 0.5, 0.1, 0.2]), 2);
        assert_eq!(mixer.push(2, &[0.5, 0.25]), 1);

        assert_eq!(mixer.mix_next_frame(), (0.75, 0.75));
        assert_eq!(mixer.consumed(1), 1);
        assert_eq!(mixer.consumed(2), 1);

        assert_eq!(mixer.mix_next_frame(), (0.1, 0.2));
        assert_eq!(mixer.consumed(1), 2);
        assert_eq!(mixer.consumed(2), 1);
    }

    #[test]
    fn stopped_audio_out_stream_does_not_consume_queued_samples() {
        let mut mixer = AudioOutMixer::default();
        assert!(mixer.open(3));
        assert_eq!(mixer.push(3, &[0.5, -0.5]), 1);
        assert_eq!(mixer.mix_next_frame(), (0.0, 0.0));
        assert_eq!(mixer.consumed(3), 0);

        mixer.start(3);
        assert_eq!(mixer.mix_next_frame(), (0.5, -0.5));
        assert_eq!(mixer.consumed(3), 1);
    }

    #[test]
    fn audio_out_volume_is_per_stream_and_push_does_not_drop() {
        let mut mixer = AudioOutMixer::default();
        assert!(mixer.open(4));
        mixer.start(4);
        mixer.set_volume(4, 0.25);
        let samples = vec![1.0; 32_768];
        assert_eq!(mixer.push(4, &samples), samples.len() / 2);
        assert_eq!(mixer.mix_next_frame(), (0.25, 0.25));
        assert_eq!(mixer.consumed(4), 1);
    }

    #[test]
    fn mixed_f32_output_saturates_after_summing() {
        assert_eq!(clamp_stereo_output(1.75, -2.0), (1.0, -1.0));
        assert_eq!(clamp_stereo_output(0.25, -0.5), (0.25, -0.5));
    }

    #[test]
    fn emulated_ring_underrun_counts_only_after_pcm_started() {
        let ring = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = ring.split();
        let mut mixer = AudioOutMixer::default();
        let underruns = AtomicU64::new(0);
        let mut saw_emulated_audio = false;

        assert_eq!(
            pop_mixed_stereo_frame(
                &mut consumer,
                &mut mixer,
                &underruns,
                &mut saw_emulated_audio,
            ),
            (0.0, 0.0)
        );
        assert_eq!(underruns.load(Ordering::Relaxed), 0);

        assert_eq!(producer.push_slice(&[0.25, -0.5]), 2);
        assert_eq!(
            pop_mixed_stereo_frame(
                &mut consumer,
                &mut mixer,
                &underruns,
                &mut saw_emulated_audio,
            ),
            (0.25, -0.5)
        );
        assert!(saw_emulated_audio);

        assert_eq!(
            pop_mixed_stereo_frame(
                &mut consumer,
                &mut mixer,
                &underruns,
                &mut saw_emulated_audio,
            ),
            (0.0, 0.0)
        );
        assert_eq!(underruns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn resample_ratio_is_locked_to_the_stream_rates() {
        let prebuf = PrebufState::new(0.5);
        assert_eq!(prebuf.callback_ratio(240, 0.5), 0.5);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert_eq!(prebuf.callback_ratio(240, 0.5), 0.5);
    }

    #[test]
    fn native_rate_multichannel_output_precedes_rate_fallback() {
        let stereo_f32 = output_config_preference(2, SampleFormat::F32, 44_100, 48_000, 48_000);
        let surround_f32 =
            output_config_preference(8, SampleFormat::F32, 44_100, 96_000, 48_000);
        let stereo_i16 = output_config_preference(2, SampleFormat::I16, 48_000, 48_000, 48_000);

        assert!(stereo_f32 < surround_f32);
        assert!(stereo_f32 < stereo_i16);
        assert!(surround_f32.is_some());
        assert_eq!(
            output_config_preference(8, SampleFormat::F32, 96_000, 96_000, 48_000),
            None
        );
    }
}

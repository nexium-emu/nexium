use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const MAX_REGISTERED_BUFFERS: usize = 4;
const MAX_AUDIO_OUT_BUFFERS: usize = 32;

#[derive(Debug)]
struct AudioOutBufferState {
    tag: u64,
    samples: Vec<f32>,
    frames: u64,
}

#[derive(Debug)]
struct RegisteredAudioOutBuffer {
    buffer: AudioOutBufferState,
    submitted_samples: usize,
    consumed_target: u64,
    release_at: Instant,
}

#[derive(Debug)]
pub struct AudioOutSession {
    state: u8,
    volume_bits: u32,
    appended: VecDeque<AudioOutBufferState>,
    registered: VecDeque<RegisteredAudioOutBuffer>,
    released: VecDeque<u64>,
    next_consumed_target: u64,
    next_release_at: Option<Instant>,
    host_stream_open: bool,
}

impl Default for AudioOutSession {
    fn default() -> Self {
        Self {
            state: 1,
            volume_bits: 1.0f32.to_bits(),
            appended: VecDeque::new(),
            registered: VecDeque::new(),
            released: VecDeque::new(),
            next_consumed_target: 0,
            next_release_at: None,
            host_stream_open: false,
        }
    }
}

pub enum DueSignal {
    None,
    NeedMore,
    BufferDue,
}

pub fn open_audio_out_session(kernel: &mut Kernel, session: u32) {
    let host_stream_open = crate::audio_sink::host_audio_sink()
        .map(|sink| sink.open_audio_out_stream(session as u64))
        .unwrap_or(false);
    let state = kernel.audio_out_sessions.entry(session).or_default();
    state.host_stream_open |= host_stream_open;
}

pub fn close_audio_out_session(kernel: &mut Kernel, session: u32) {
    if let Some(sink) = crate::audio_sink::host_audio_sink() {
        sink.close_audio_out_stream(session as u64);
    }
    kernel.audio_out_sessions.remove(&session);
    if let Some(event) = kernel.audio_buffer_events.remove(&session) {
        kernel.event_signals.remove(&event);
    }
}

fn session_buffer_count(state: &AudioOutSession) -> usize {
    state.appended.len() + state.registered.len() + state.released.len()
}

fn can_append_buffer(state: &AudioOutSession) -> bool {
    session_buffer_count(state) < MAX_AUDIO_OUT_BUFFERS
}

fn signal_buffer_event(kernel: &mut Kernel, session: u32) {
    let Some(&event) = kernel.audio_buffer_events.get(&session) else {
        return;
    };
    let signaled = kernel.event_signals.entry(event).or_insert(false);
    if !*signaled {
        *signaled = true;
        kernel.threads.signal_handle(event);
    }
}

fn promote_buffers(state: &mut AudioOutSession, consumed: u64, now: Instant) {
    if state.state != 0 {
        return;
    }
    while state.registered.len() < MAX_REGISTERED_BUFFERS {
        let Some(buffer) = state.appended.pop_front() else {
            break;
        };
        let consumed_base = state.next_consumed_target.max(consumed);
        let consumed_target = consumed_base.saturating_add(buffer.frames);
        state.next_consumed_target = consumed_target;

        let release_base = state
            .next_release_at
            .filter(|release_at| *release_at > now)
            .unwrap_or(now);
        let duration = Duration::from_nanos(buffer.frames.saturating_mul(1_000_000_000) / 48_000);
        let release_at = release_base + duration;
        state.next_release_at = Some(release_at);
        state.registered.push_back(RegisteredAudioOutBuffer {
            buffer,
            submitted_samples: 0,
            consumed_target,
            release_at,
        });
    }
}

fn release_completed_buffers(
    state: &mut AudioOutSession,
    consumed: Option<u64>,
    now: Instant,
) -> usize {
    let mut released = 0;
    loop {
        let due = state.registered.front().is_some_and(|buffer| {
            consumed
                .map(|played| played >= buffer.consumed_target)
                .unwrap_or(buffer.release_at <= now)
        });
        if !due {
            break;
        }
        let buffer = state.registered.pop_front().unwrap();
        state.released.push_back(buffer.buffer.tag);
        released += 1;
    }
    released
}

fn pump_registered_buffers(
    state: &mut AudioOutSession,
    session: u32,
    sink: &dyn crate::audio_sink::HostPcmSink,
) -> usize {
    if !state.host_stream_open || state.state != 0 {
        return 0;
    }
    let mut pushed_frames = 0;
    for buffer in &mut state.registered {
        if buffer.submitted_samples >= buffer.buffer.samples.len() {
            continue;
        }
        let samples = &buffer.buffer.samples[buffer.submitted_samples..];
        let accepted_frames = sink.push_audio_out_stereo_f32(session as u64, samples);
        let accepted_samples = (accepted_frames.saturating_mul(2)).min(samples.len());
        buffer.submitted_samples += accepted_samples;
        pushed_frames += accepted_samples / 2;
        if accepted_samples < samples.len() {
            break;
        }
    }
    pushed_frames
}

fn update_audio_out_session(kernel: &mut Kernel, session: u32) -> usize {
    let now = Instant::now();
    let sink = crate::audio_sink::host_audio_sink().cloned();
    let (released, need_more, pushed) = {
        let Some(state) = kernel.audio_out_sessions.get_mut(&session) else {
            return 0;
        };
        if state.state != 0 {
            return 0;
        }
        if !state.host_stream_open {
            if let Some(sink) = sink
                .as_ref()
                .filter(|sink| sink.open_audio_out_stream(session as u64))
            {
                state.host_stream_open = true;
                let consumed = sink.audio_out_samples_consumed(session as u64);
                let mut target = consumed;
                for buffer in &mut state.registered {
                    target = target.saturating_add(buffer.buffer.frames);
                    buffer.consumed_target = target;
                    buffer.submitted_samples = 0;
                }
                state.next_consumed_target = target;
                sink.set_audio_out_volume(session as u64, f32::from_bits(state.volume_bits));
                sink.start_audio_out_stream(session as u64);
            }
        }
        let consumed = sink.as_ref().and_then(|sink| {
            state
                .host_stream_open
                .then(|| sink.audio_out_samples_consumed(session as u64))
        });
        let released = release_completed_buffers(state, consumed, now);
        promote_buffers(state, consumed.unwrap_or(0), now);
        let pushed = sink
            .as_ref()
            .map(|sink| pump_registered_buffers(state, session, sink.as_ref()))
            .unwrap_or(0);
        (released, state.registered.is_empty(), pushed)
    };
    if released != 0 || need_more {
        signal_buffer_event(kernel, session);
    }
    pushed
}

pub fn poll_audio_outs(kernel: &mut Kernel) {
    let sessions: Vec<u32> = kernel
        .audio_out_sessions
        .iter()
        .filter_map(|(session, state)| (state.state == 0).then_some(*session))
        .collect();
    for session in sessions {
        update_audio_out_session(kernel, session);
    }
}

pub fn drain_audio_spill(kernel: &mut Kernel) {
    poll_audio_outs(kernel);
}

pub fn check_due_signal(kernel: &Kernel, session: u32, _now: Instant) -> DueSignal {
    let Some(state) = kernel.audio_out_sessions.get(&session) else {
        return DueSignal::None;
    };
    if !state.released.is_empty() {
        return DueSignal::BufferDue;
    }
    if state.state == 0 && state.registered.is_empty() {
        return DueSignal::NeedMore;
    }
    DueSignal::None
}

pub fn mark_due_signaled(_kernel: &mut Kernel, _session: u32) {}

pub fn get_audio_out_state(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel
        .audio_out_sessions
        .get(&session)
        .map_or(1, |state| state.state as u32)
}

pub fn start_audio_out(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) {
    let _ = start_audio_out_result(kernel, ctx, session);
}

pub fn start_audio_out_result(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    const RESULT_OPERATION_FAILED: u32 = 153 | (2 << 9);
    open_audio_out_session(kernel, session);
    let sink = crate::audio_sink::host_audio_sink().cloned();
    let now = Instant::now();
    {
        let state = kernel.audio_out_sessions.entry(session).or_default();
        if state.state != 1 {
            return RESULT_OPERATION_FAILED;
        }
        state.state = 0;
        let consumed = sink
            .as_ref()
            .filter(|_| state.host_stream_open)
            .map(|sink| sink.audio_out_samples_consumed(session as u64))
            .unwrap_or(0);
        state.next_consumed_target = consumed;
        state.next_release_at = Some(now);
        if let Some(sink) = sink.as_ref().filter(|_| state.host_stream_open) {
            sink.set_audio_out_volume(session as u64, f32::from_bits(state.volume_bits));
            sink.start_audio_out_stream(session as u64);
        }
        promote_buffers(state, consumed, now);
        if let Some(sink) = sink.as_ref() {
            pump_registered_buffers(state, session, sink.as_ref());
        }
    }
    0
}

pub fn stop_audio_out(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) {
    let sink = crate::audio_sink::host_audio_sink().cloned();
    let mut signal = false;
    if let Some(state) = kernel.audio_out_sessions.get_mut(&session) {
        if state.state == 0 {
            let consumed = sink
                .as_ref()
                .filter(|_| state.host_stream_open)
                .map(|sink| sink.audio_out_samples_consumed(session as u64))
                .unwrap_or(state.next_consumed_target);
            if let Some(sink) = sink.as_ref().filter(|_| state.host_stream_open) {
                sink.stop_audio_out_stream(session as u64);
            }
            while let Some(buffer) = state.registered.pop_front() {
                state.released.push_back(buffer.buffer.tag);
            }
            state.next_consumed_target = consumed;
            state.next_release_at = None;
            state.state = 1;
            signal = true;
        }
    }
    if signal {
        signal_buffer_event(kernel, session);
    }
}

pub fn append_audio_out_buffer(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session: u32,
    client_ptr: u64,
) {
    if kernel
        .audio_out_sessions
        .get(&session)
        .is_some_and(|state| !can_append_buffer(state))
    {
        log::warn!(
            "AudioOut session {:#x} rejected buffer {:#x}: {} buffers are already active",
            session,
            client_ptr,
            MAX_AUDIO_OUT_BUFFERS
        );
        return;
    }
    let trace_id = if std::env::var_os("NEXIUM_AUDIO_TRACE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static APPENDS: AtomicU64 = AtomicU64::new(0);
        let append = APPENDS.fetch_add(1, Ordering::Relaxed) + 1;
        (append <= 128).then_some(append)
    } else {
        None
    };
    let selected_desc = ctx
        .send_buffers
        .iter()
        .find(|buffer| buffer.size >= 0x28 && buffer.addr != 0)
        .map(|buffer| ("send", *buffer))
        .or_else(|| {
            ctx.send_statics
                .iter()
                .find(|buffer| buffer.size >= 0x28 && buffer.addr != 0)
                .map(|buffer| ("static", *buffer))
        });
    let struct_src = selected_desc
        .map(|(_, buffer)| buffer.addr)
        .unwrap_or(client_ptr);
    let mut header = [0u8; 0x28];
    let header_read = kernel.address_space.read(struct_src, &mut header).is_ok();
    let next = u64::from_le_bytes(header[0x00..0x08].try_into().unwrap());
    let samples_ptr = u64::from_le_bytes(header[0x08..0x10].try_into().unwrap());
    let capacity = u64::from_le_bytes(header[0x10..0x18].try_into().unwrap());
    let data_size = u64::from_le_bytes(header[0x18..0x20].try_into().unwrap());
    let data_offset = u64::from_le_bytes(header[0x20..0x28].try_into().unwrap());
    let byte_count = if header_read && samples_ptr != 0 {
        (data_size.min(1 << 20) as usize) & !3
    } else {
        0
    };
    let mut pcm = vec![0u8; byte_count];
    let pcm_read = byte_count != 0 && kernel.address_space.read(samples_ptr, &mut pcm).is_ok();
    let mut samples = Vec::with_capacity(byte_count / 2);
    let mut pcm_nonzero = 0usize;
    let mut pcm_peak = 0u16;
    if pcm_read {
        for sample in pcm.chunks_exact(2) {
            let sample = i16::from_le_bytes([sample[0], sample[1]]);
            if trace_id.is_some() {
                pcm_nonzero += usize::from(sample != 0);
                pcm_peak = pcm_peak.max(sample.unsigned_abs());
            }
            samples.push(sample as f32 / 32768.0);
        }
    }
    let frames = (samples.len() / 2) as u64;
    let started = {
        let state = kernel.audio_out_sessions.entry(session).or_default();
        let started = state.state == 0;
        state.appended.push_back(AudioOutBufferState {
            tag: client_ptr,
            samples,
            frames,
        });
        started
    };
    let pushed_frames = if started {
        update_audio_out_session(kernel, session)
    } else {
        0
    };
    if let Some(trace_id) = trace_id {
        let (desc_kind, desc_addr, desc_size, desc_mode) = selected_desc
            .map(|(kind, buffer)| (kind, buffer.addr, buffer.size, buffer.mode))
            .unwrap_or(("client-tag", client_ptr, 0, 0));
        let queued = total_buffer_count(kernel, session);
        log::info!(
            "[aout-trace] n={} session={:#x} tag={:#x} selected={}:{:#x}:{:#x}:{} send={:?} statics={:?} struct_src={:#x} header_read={} header={:02x?} next={:#x} samples={:#x} capacity={:#x} size={:#x} offset={:#x} pcm_addr={:#x} pcm_read={} frames={} pcm_nonzero={} pcm_peak={} sink_present={} started={} pushed_frames={} queued={}",
            trace_id,
            session,
            client_ptr,
            desc_kind,
            desc_addr,
            desc_size,
            desc_mode,
            ctx.send_buffers,
            ctx.send_statics,
            struct_src,
            header_read,
            header,
            next,
            samples_ptr,
            capacity,
            data_size,
            data_offset,
            samples_ptr,
            pcm_read,
            frames,
            pcm_nonzero,
            pcm_peak,
            crate::audio_sink::host_audio_sink().is_some(),
            started,
            pushed_frames,
            queued,
        );
    }
    if std::env::var_os("NEXIUM_AUDIO_RATE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Mutex, OnceLock};
        static APPENDS: AtomicU64 = AtomicU64::new(0);
        static LAST: OnceLock<Mutex<(Instant, u64)>> = OnceLock::new();
        let append = APPENDS.fetch_add(1, Ordering::Relaxed) + 1;
        let cell = LAST.get_or_init(|| Mutex::new((Instant::now(), 0)));
        let mut last = cell.lock().unwrap();
        if last.0.elapsed() >= Duration::from_secs(1) {
            let elapsed = last.0.elapsed().as_secs_f64();
            let delta = append - last.1;
            *last = (Instant::now(), append);
            log::info!(
                "[aout-rate] appends/s={:.0} total={}",
                delta as f64 / elapsed,
                append
            );
        }
    }
}

pub fn append_audio_out_buffer_auto(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session: u32,
    client_ptr: u64,
) {
    append_audio_out_buffer(kernel, ctx, session, client_ptr);
}

pub fn register_buffer_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    if let Some(&event) = kernel.audio_buffer_events.get(&session) {
        return event;
    }
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    kernel.audio_buffer_events.insert(session, event);
    event
}

fn drain_released(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    update_audio_out_session(kernel, session);
    let recv = ctx
        .recv_buffers
        .iter()
        .find(|buffer| buffer.size > 0 && buffer.addr != 0)
        .or_else(|| {
            ctx.recv_statics
                .iter()
                .find(|buffer| buffer.size > 0 && buffer.addr != 0)
        })
        .copied();
    let max_count = recv.map(|buffer| buffer.size as usize / 8).unwrap_or(0);
    let mut tags = Vec::with_capacity(max_count);
    if let Some(state) = kernel.audio_out_sessions.get_mut(&session) {
        while tags.len() < max_count {
            let Some(tag) = state.released.pop_front() else {
                break;
            };
            tags.push(tag);
        }
    }
    if let Some(buffer) = recv {
        let bytes = released_tag_bytes(&tags, max_count != 0);
        let _ = kernel.address_space.write(buffer.addr, &bytes);
    }
    tags.len() as u32
}

fn released_tag_bytes(tags: &[u64], has_output_slot: bool) -> Vec<u8> {
    let slot_count = tags.len().max(usize::from(has_output_slot));
    let mut bytes = vec![0u8; slot_count * 8];
    for (slot, tag) in tags.iter().enumerate() {
        bytes[slot * 8..slot * 8 + 8].copy_from_slice(&tag.to_le_bytes());
    }
    bytes
}

pub fn get_released_audio_out_buffer(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    drain_released(kernel, ctx, session)
}

pub fn get_released_audio_out_buffer_auto(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session: u32,
) -> u32 {
    drain_released(kernel, ctx, session)
}

pub fn contains_audio_out_buffer(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    session: u32,
    client_ptr: u64,
) -> bool {
    kernel
        .audio_out_sessions
        .get(&session)
        .is_some_and(|state| {
            state.appended.iter().any(|buffer| buffer.tag == client_ptr)
                || state
                    .registered
                    .iter()
                    .any(|buffer| buffer.buffer.tag == client_ptr)
                || state.released.iter().any(|tag| *tag == client_ptr)
        })
}

pub fn get_audio_out_buffer_count(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel.audio_out_sessions.get(&session).map_or(0, |state| {
        (state.appended.len() + state.registered.len()) as u32
    })
}

pub fn total_buffer_count(kernel: &Kernel, session: u32) -> usize {
    kernel
        .audio_out_sessions
        .get(&session)
        .map_or(0, session_buffer_count)
}

pub fn get_audio_out_played_sample_count(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    session: u32,
) -> u64 {
    let host_stream_open = kernel
        .audio_out_sessions
        .get(&session)
        .is_some_and(|state| state.host_stream_open);
    crate::audio_sink::host_audio_sink()
        .filter(|_| host_stream_open)
        .map(|sink| sink.audio_out_samples_consumed(session as u64))
        .unwrap_or(0)
}

fn flush_started_buffers(state: &mut AudioOutSession) -> Option<bool> {
    if state.state != 0 {
        return None;
    }
    let mut released_any = false;
    while let Some(buffer) = state.registered.pop_front() {
        state.released.push_back(buffer.buffer.tag);
        released_any = true;
    }
    while let Some(buffer) = state.appended.pop_front() {
        state.released.push_back(buffer.tag);
        released_any = true;
    }
    Some(released_any)
}

pub fn flush_audio_out_buffers(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> bool {
    let Some(released_any) = kernel
        .audio_out_sessions
        .get_mut(&session)
        .and_then(flush_started_buffers)
    else {
        return false;
    };
    if released_any {
        signal_buffer_event(kernel, session);
    }
    true
}

pub fn set_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32, volume: u32) {
    let state = kernel.audio_out_sessions.entry(session).or_default();
    state.volume_bits = volume;
    if state.host_stream_open {
        if let Some(sink) = crate::audio_sink::host_audio_sink() {
            sink.set_audio_out_volume(session as u64, f32::from_bits(volume));
        }
    }
}

pub fn get_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel
        .audio_out_sessions
        .get(&session)
        .map_or(1.0f32.to_bits(), |state| state.volume_bits)
}

#[cfg(test)]
mod tests {
    use super::{
        can_append_buffer, flush_started_buffers, promote_buffers, release_completed_buffers,
        released_tag_bytes, AudioOutBufferState, AudioOutSession, MAX_AUDIO_OUT_BUFFERS,
    };
    use std::time::Instant;

    fn buffer(tag: u64, frames: usize) -> AudioOutBufferState {
        AudioOutBufferState {
            tag,
            samples: vec![0.0; frames * 2],
            frames: frames as u64,
        }
    }

    #[test]
    fn stopped_append_waits_for_start_and_start_registers_four() {
        let mut state = AudioOutSession::default();
        for tag in 1..=5 {
            state.appended.push_back(buffer(tag, 8));
        }
        promote_buffers(&mut state, 0, Instant::now());
        assert_eq!(state.appended.len(), 5);
        assert!(state.registered.is_empty());

        state.state = 0;
        promote_buffers(&mut state, 0, Instant::now());
        assert_eq!(state.appended.len(), 1);
        assert_eq!(state.registered.len(), 4);
        assert_eq!(state.registered.front().unwrap().buffer.tag, 1);
    }

    #[test]
    fn completion_releases_fifo_and_allows_next_promotion() {
        let mut state = AudioOutSession::default();
        state.state = 0;
        for tag in 1..=5 {
            state.appended.push_back(buffer(tag, 8));
        }
        let now = Instant::now();
        promote_buffers(&mut state, 0, now);
        assert_eq!(release_completed_buffers(&mut state, Some(8), now), 1);
        promote_buffers(&mut state, 8, now);
        assert_eq!(state.released.front(), Some(&1));
        assert_eq!(state.registered.len(), 4);
        assert!(state.appended.is_empty());
        assert_eq!(state.registered.back().unwrap().buffer.tag, 5);
    }

    #[test]
    fn consumed_clock_is_evaluated_per_session() {
        let now = Instant::now();
        let mut first = AudioOutSession::default();
        let mut second = AudioOutSession::default();
        first.state = 0;
        second.state = 0;
        first.appended.push_back(buffer(1, 8));
        second.appended.push_back(buffer(2, 8));
        promote_buffers(&mut first, 0, now);
        promote_buffers(&mut second, 0, now);
        assert_eq!(release_completed_buffers(&mut first, Some(8), now), 1);
        assert_eq!(release_completed_buffers(&mut second, Some(0), now), 0);
    }

    #[test]
    fn flush_is_started_only_and_releases_registered_and_appended() {
        let mut state = AudioOutSession::default();
        state.appended.push_back(buffer(1, 8));
        assert_eq!(flush_started_buffers(&mut state), None);
        assert_eq!(state.appended.len(), 1);

        state.state = 0;
        promote_buffers(&mut state, 0, Instant::now());
        state.appended.push_back(buffer(2, 8));
        assert_eq!(flush_started_buffers(&mut state), Some(true));
        assert_eq!(state.released.iter().copied().collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn total_capacity_counts_released_buffers() {
        let mut state = AudioOutSession::default();
        for tag in 0..MAX_AUDIO_OUT_BUFFERS as u64 {
            state.released.push_back(tag);
        }
        assert!(!can_append_buffer(&state));
        state.released.pop_front();
        assert!(can_append_buffer(&state));
    }

    #[test]
    fn released_tags_zero_first_output_when_none_are_ready() {
        assert_eq!(released_tag_bytes(&[], true), 0u64.to_le_bytes());
        assert!(released_tag_bytes(&[], false).is_empty());
    }

    #[test]
    fn released_tags_replace_zero_sentinel_with_returned_tags() {
        let tags = [0x1122_3344_5566_7788, 0x8877_6655_4433_2211];
        let bytes = released_tag_bytes(&tags, true);
        assert_eq!(&bytes[0..8], &tags[0].to_le_bytes());
        assert_eq!(&bytes[8..16], &tags[1].to_le_bytes());
    }
}

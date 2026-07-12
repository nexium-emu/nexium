use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

const SPILL_CAP_SAMPLES: usize = 48_000 * 2 * 4;
pub const RELEASE_OVERDUE: std::time::Duration = std::time::Duration::from_secs(2);

pub fn buffer_release_due(
    release_at: std::time::Instant,
    consumed_target: u64,
    now: std::time::Instant,
) -> bool {
    match crate::audio_sink::host_audio_sink() {
        Some(sink) => {
            sink.samples_consumed() >= consumed_target
                || now.saturating_duration_since(release_at) >= RELEASE_OVERDUE
        }
        None => release_at <= now,
    }
}

pub enum DueSignal {
    None,
    NeedMore,
    BufferDue,
}

pub fn check_due_signal(kernel: &Kernel, session: u32, now: std::time::Instant) -> DueSignal {
    let started = kernel.audio_out_state.get(&session).copied() == Some(0);
    let Some(q) = kernel.audio_out_buffers.get(&session) else {
        return if started {
            DueSignal::NeedMore
        } else {
            DueSignal::None
        };
    };
    if q.is_empty() {
        return if started {
            DueSignal::NeedMore
        } else {
            DueSignal::None
        };
    }
    let released = kernel
        .audio_out_released_count
        .get(&session)
        .copied()
        .unwrap_or(0);
    let signaled = kernel
        .audio_out_due_signaled
        .get(&session)
        .copied()
        .unwrap_or(0);
    if released > signaled {
        return DueSignal::BufferDue;
    }
    let need = signaled - released;
    let mut due = 0u64;
    for &(_, release_at, _, target) in q.iter() {
        if !buffer_release_due(release_at, target, now) {
            break;
        }
        due += 1;
        if due > need {
            return DueSignal::BufferDue;
        }
    }
    DueSignal::None
}

pub fn mark_due_signaled(kernel: &mut Kernel, session: u32) {
    *kernel.audio_out_due_signaled.entry(session).or_insert(0) += 1;
}

pub fn drain_audio_spill(kernel: &mut Kernel) {
    let Some(sink) = crate::audio_sink::host_audio_sink() else {
        return;
    };
    for spill in kernel.audio_out_spill.values_mut() {
        loop {
            let (head, _) = spill.as_slices();
            let take = head.len() & !1;
            if take == 0 {
                if spill.len() < 2 {
                    break;
                }
                spill.make_contiguous();
                continue;
            }
            let accepted = sink.push_stereo_f32(&head[..take]) * 2;
            spill.drain(..accepted);
            if accepted < take {
                break;
            }
        }
    }
}

pub fn get_audio_out_state(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel.audio_out_state.get(&session).copied().unwrap_or(1) as u32
}
pub fn start_audio_out(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) {
    kernel.audio_out_state.insert(session, 0);
    if let Some(sink) = crate::audio_sink::host_audio_sink() {
        kernel
            .audio_out_consumed_base
            .entry(session)
            .or_insert_with(|| sink.samples_consumed());
    }
}
pub fn stop_audio_out(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) {
    kernel.audio_out_state.insert(session, 1);
}

pub fn append_audio_out_buffer(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    session: u32,
    client_ptr: u64,
) {
    let struct_src = ctx
        .send_buffers
        .iter()
        .find(|b| b.size >= 0x28 && b.addr != 0)
        .or_else(|| ctx.send_statics.iter().find(|b| b.size >= 0x28 && b.addr != 0))
        .map(|b| b.addr)
        .unwrap_or(client_ptr);
    let mut frames: u64 = 0;
    let mut consumed_target: u64 = 0;
    let mut hdr = [0u8; 0x28];
    if struct_src != 0 && kernel.address_space.read(struct_src, &mut hdr).is_ok() {
        let buf_ptr = u64::from_le_bytes(hdr[0x08..0x10].try_into().unwrap());
        let data_size = u64::from_le_bytes(hdr[0x18..0x20].try_into().unwrap());
        let data_off = u64::from_le_bytes(hdr[0x20..0x28].try_into().unwrap());
        let n = (data_size.min(1 << 20) as usize) & !3;
        if buf_ptr != 0 && n >= 4 {
            let mut pcm = vec![0u8; n];
            if kernel
                .address_space
                .read(buf_ptr.wrapping_add(data_off), &mut pcm)
                .is_ok()
            {
                frames = (n / 4) as u64;
                if let Some(sink) = crate::audio_sink::host_audio_sink() {
                    let mut f32s = Vec::with_capacity(n / 2);
                    for ch in pcm.chunks_exact(2) {
                        f32s.push(i16::from_le_bytes([ch[0], ch[1]]) as f32 / 32768.0);
                    }
                    drain_audio_spill(kernel);
                    let consumed_now = sink.samples_consumed();
                    let appended_before = kernel
                        .audio_out_appended_frames
                        .get(&session)
                        .copied()
                        .unwrap_or(0);
                    {
                        let base = kernel
                            .audio_out_consumed_base
                            .entry(session)
                            .or_insert(consumed_now);
                        if consumed_now.saturating_sub(*base) > appended_before {
                            *base = consumed_now - appended_before;
                        }
                        consumed_target = *base + appended_before + frames;
                    }
                    let spill = kernel.audio_out_spill.entry(session).or_default();
                    if spill.is_empty() {
                        let accepted = sink.push_stereo_f32(&f32s) * 2;
                        if accepted < f32s.len() {
                            spill.extend(f32s[accepted..].iter().copied());
                        }
                    } else {
                        spill.extend(f32s.iter().copied());
                    }
                    if spill.len() > SPILL_CAP_SAMPLES {
                        let over = (spill.len() - SPILL_CAP_SAMPLES + 1) & !1;
                        spill.drain(..over);
                        use std::sync::atomic::{AtomicU64, Ordering};
                        static DROP_EVENTS: AtomicU64 = AtomicU64::new(0);
                        let n = DROP_EVENTS.fetch_add(1, Ordering::Relaxed);
                        if n % 200 == 0 {
                            log::warn!(
                                "audio_out spill overflow: dropped {} samples (event #{})",
                                over,
                                n
                            );
                        }
                    }
                }
            }
        }
    }
    let now = std::time::Instant::now();
    let dur = std::time::Duration::from_nanos(frames.saturating_mul(1_000_000_000) / 48_000);
    let start = kernel
        .audio_out_next_free
        .get(&session)
        .copied()
        .filter(|t| *t > now)
        .unwrap_or(now);
    let release_at = start + dur;
    kernel.audio_out_next_free.insert(session, release_at);
    let q = kernel.audio_out_buffers.entry(session).or_default();
    q.push_back((client_ptr, release_at, frames, consumed_target));
    *kernel.audio_out_appended_frames.entry(session).or_insert(0) += frames;
    if std::env::var_os("NEXIUM_AUDIO_RATE").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::{Mutex, OnceLock};
        static APPENDS: AtomicU64 = AtomicU64::new(0);
        static LAST: OnceLock<Mutex<(std::time::Instant, u64)>> = OnceLock::new();
        let n = APPENDS.fetch_add(1, Ordering::Relaxed) + 1;
        let cell = LAST.get_or_init(|| Mutex::new((std::time::Instant::now(), 0)));
        let mut last = cell.lock().unwrap();
        if last.0.elapsed() >= std::time::Duration::from_secs(1) {
            let dt = last.0.elapsed().as_secs_f64();
            let da = n - last.1;
            *last = (std::time::Instant::now(), n);
            log::info!("[aout-rate] appends/s={:.0} total={}", da as f64 / dt, n);
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
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    kernel.audio_buffer_events.insert(session, event);
    event
}

fn drain_released(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    let recv = ctx
        .recv_buffers
        .iter()
        .find(|b| b.size > 0 && b.addr != 0)
        .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))
        .copied();
    let max_count = recv.map(|b| (b.size as usize) / 8).unwrap_or(0);
    let now = std::time::Instant::now();
    let mut ptrs: Vec<u64> = Vec::new();
    let mut played: u64 = 0;
    if let Some(q) = kernel.audio_out_buffers.get_mut(&session) {
        while ptrs.len() < max_count {
            match q.front() {
                Some(&(_, release_at, _, target)) if buffer_release_due(release_at, target, now) => {
                    let (p, _, frames, _) = q.pop_front().unwrap();
                    ptrs.push(p);
                    played += frames;
                }
                _ => break,
            }
        }
    }
    if played > 0 {
        *kernel.audio_out_played_samples.entry(session).or_insert(0) += played;
    }
    if !ptrs.is_empty() {
        *kernel.audio_out_released_count.entry(session).or_insert(0) += ptrs.len() as u64;
    }
    if let Some(b) = recv {
        let mut bytes = Vec::with_capacity(ptrs.len() * 8);
        for p in &ptrs {
            bytes.extend_from_slice(&p.to_le_bytes());
        }
        let _ = kernel.address_space.write(b.addr, &bytes);
    }
    ptrs.len() as u32
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
        .audio_out_buffers
        .get(&session)
        .map_or(false, |q| q.iter().any(|&(p, _, _, _)| p == client_ptr))
}
pub fn get_audio_out_buffer_count(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel
        .audio_out_buffers
        .get(&session)
        .map_or(0, |q| q.len() as u32)
}
pub fn get_audio_out_played_sample_count(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    session: u32,
) -> u64 {
    let appended = kernel
        .audio_out_appended_frames
        .get(&session)
        .copied()
        .unwrap_or(0);
    if let (Some(sink), Some(&base)) = (
        crate::audio_sink::host_audio_sink(),
        kernel.audio_out_consumed_base.get(&session),
    ) {
        return sink.samples_consumed().saturating_sub(base).min(appended);
    }
    kernel
        .audio_out_played_samples
        .get(&session)
        .copied()
        .unwrap_or(0)
}
pub fn flush_audio_out_buffers(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    false
}

pub fn set_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32, volume: u32) {
    kernel.audio_out_volumes.insert(session, volume);
}

pub fn get_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel
        .audio_out_volumes
        .get(&session)
        .copied()
        .unwrap_or(0x3f800000)
}

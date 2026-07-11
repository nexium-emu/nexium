use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_audio_out_state(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel.audio_out_state.get(&session).copied().unwrap_or(1) as u32
}
pub fn start_audio_out(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) {
    kernel.audio_out_state.insert(session, 0);
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
                    let _ = sink.push_stereo_f32(&f32s);
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
    q.push_back((client_ptr, release_at, frames));
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
                Some(&(_, release_at, _)) if release_at <= now => {
                    let (p, _, frames) = q.pop_front().unwrap();
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
        .map_or(false, |q| q.iter().any(|&(p, _, _)| p == client_ptr))
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

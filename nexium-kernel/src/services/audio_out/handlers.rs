use crate::kernel::Kernel;
use crate::kernel::handles::HandleType;
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

pub fn append_audio_out_buffer(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32, client_ptr: u64) {
    let q = kernel.audio_out_buffers.entry(session).or_default();
    q.push_back(client_ptr);
    if let Some(&ev) = kernel.audio_buffer_events.get(&session) {
        kernel.event_signals.insert(ev, true);
    }
}

pub fn append_audio_out_buffer_auto(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32, client_ptr: u64) {
    append_audio_out_buffer(kernel, ctx, session, client_ptr);
}

pub fn register_buffer_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    kernel.audio_buffer_events.insert(session, event);
    event
}

fn drain_released(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    let recv = ctx.recv_buffers.iter().find(|b| b.size > 0 && b.addr != 0).copied();
    let max_count = recv.map(|b| (b.size as usize) / 8).unwrap_or(0);
    let mut ptrs: Vec<u64> = Vec::new();
    if let Some(q) = kernel.audio_out_buffers.get_mut(&session) {
        while ptrs.len() < max_count {
            match q.pop_front() {
                Some(p) => ptrs.push(p),
                None => break,
            }
        }
    }
    if let Some(b) = recv {
        let mut bytes = Vec::with_capacity(ptrs.len() * 8);
        for p in &ptrs { bytes.extend_from_slice(&p.to_le_bytes()); }
        let _ = kernel.address_space.write(b.addr, &bytes);
    }
    ptrs.len() as u32
}

pub fn get_released_audio_out_buffer(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    drain_released(kernel, ctx, session)
}

pub fn get_released_audio_out_buffer_auto(kernel: &mut Kernel, ctx: &mut IpcCtx, session: u32) -> u32 {
    drain_released(kernel, ctx, session)
}

pub fn contains_audio_out_buffer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _client_ptr: u64) -> bool { false }
pub fn get_audio_out_buffer_count(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 { 0 }
pub fn get_audio_out_played_sample_count(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 { 0 }
pub fn flush_audio_out_buffers(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool { false }

pub fn set_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32, volume: u32) {
    kernel.audio_out_volumes.insert(session, volume);
}

pub fn get_audio_out_volume(kernel: &mut Kernel, _ctx: &mut IpcCtx, session: u32) -> u32 {
    kernel.audio_out_volumes.get(&session).copied().unwrap_or(0x3f800000)
}

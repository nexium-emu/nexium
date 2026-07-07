use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_current_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.saturating_sub(946_684_800)
}

pub fn set_current_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _posix_time: u64) {}

pub fn get_system_clock_context(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn set_system_clock_context(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_operation_event_readable_handle(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    event
}

use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_current_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    crate::services::time::unix_time_seconds() as u64
}

pub fn set_current_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _posix_time: u64) {}

pub fn get_system_clock_context(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> Vec<u8> {
    crate::services::time::system_clock_context()
}

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

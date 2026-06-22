use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn start(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn finish(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_update_device_status(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_update_progress(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn get_update_device_status_change_event(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_update_progress2(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

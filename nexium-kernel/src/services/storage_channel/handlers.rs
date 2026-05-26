use crate::kernel::Kernel;
use crate::kernel::handles::HandleType;
use nexium_ipc::IpcCtx;

pub fn push(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn unpop(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn pop(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_pop_event_handle(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn clear(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

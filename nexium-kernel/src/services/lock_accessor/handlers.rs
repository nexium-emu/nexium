use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn try_lock(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32, _unk: bool) -> (bool, u32) {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    (true, h)
}

pub fn unlock(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    h
}

pub fn is_locked(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}

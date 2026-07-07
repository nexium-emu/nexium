use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_manual_gpu_error_info_size(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_manual_gpu_error_info(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _info: &mut Vec<u8>,
) -> u64 {
    0
}

pub fn get_manual_gpu_error_detection_system_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn finish_manual_gpu_error_handling(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

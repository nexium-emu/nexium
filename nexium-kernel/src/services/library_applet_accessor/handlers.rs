use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_applet_state_changed_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn is_completed(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn start(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn request_exit(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn terminate(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_result(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_out_of_focus_application_suspending_enabled(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _enabled: bool,
) {
}
pub fn preset_library_applet_gpu_time_slice_zero(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn request_for_library_applet_to_get_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cmd90(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u64, _b: u64, _cc: u64, _d: u64) {}
pub fn pop_out_data(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn pop_interactive_out_data(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_pop_out_data_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_pop_interactive_out_data_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn needs_to_exit_process(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn get_library_applet_info(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn request_for_applet_to_get_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_indirect_layer_consumer_handle(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) -> u64 {
    0
}

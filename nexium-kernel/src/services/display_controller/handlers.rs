use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_last_foreground_capture_image(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn update_last_foreground_capture_image(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn get_last_application_capture_image(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_caller_applet_capture_image(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn update_caller_applet_capture_image(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_last_foreground_capture_image_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u8 {
    0
}
pub fn get_last_application_capture_image_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u8 {
    0
}
pub fn get_caller_applet_capture_image_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u8 {
    0
}
pub fn take_screen_shot_of_own_layer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
) {
}
pub fn copy_between_capture_buffers(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _dst: u32,
    _src: u32,
) {
}

pub fn acquire_last_application_capture_buffer(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn release_last_application_capture_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn acquire_last_foreground_capture_buffer(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn release_last_foreground_capture_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn acquire_caller_applet_capture_buffer(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn release_caller_applet_capture_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn acquire_last_application_capture_buffer_ex(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u8, u32) {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    (0, h)
}
pub fn acquire_last_foreground_capture_buffer_ex(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u8, u32) {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    (0, h)
}
pub fn acquire_caller_applet_capture_buffer_ex(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u8, u32) {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    (0, h)
}

pub fn clear_capture_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _c: u32,
) {
}
pub fn clear_applet_transition_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
) {
}
pub fn acquire_last_application_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}
pub fn release_last_application_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn acquire_last_foreground_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}
pub fn release_last_foreground_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn acquire_caller_applet_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}
pub fn release_caller_applet_capture_shared_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn take_screen_shot_of_own_layer_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
) {
}

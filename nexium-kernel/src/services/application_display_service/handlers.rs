use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

const NATIVE_WINDOW_PARCEL_SIZE: u64 = 60;

pub fn get_relay_service(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_system_display_service(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_manager_display_service(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_indirect_display_transaction_service(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn list_displays(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}
pub fn open_display(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    1
}
pub fn open_default_display(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    1
}
pub fn close_display(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn set_display_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_display_resolution(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (1280, 720)
}

pub fn open_layer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    NATIVE_WINDOW_PARCEL_SIZE
}
pub fn close_layer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn create_stray_layer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) {
    (1, NATIVE_WINDOW_PARCEL_SIZE)
}
pub fn destroy_stray_layer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn set_layer_scaling_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn convert_scaling_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn cmd2103(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_indirect_layer_image_map(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_indirect_layer_image_crop_map(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_indirect_layer_image_required_memory_info(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}

pub fn list_display_resolution_ratios(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    60
}

pub fn get_display_vsync_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    kernel.vsync_handles.insert(h);
    log::info!(
        "IApplicationDisplayService.GetDisplayVsyncEvent → vsync_handle={:#x}",
        h
    );
    h
}
pub fn get_display_vsync_event_for_debug(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

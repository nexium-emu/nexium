use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_z_order_count_min(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_z_order_count_max(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    10
}
pub fn get_display_logical_resolution(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn set_display_magnification(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_position(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_size(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_layer_z(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn set_layer_z(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_visibility(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_alpha(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn open_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn close_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn flip_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn list_display_rgb_ranges(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn list_display_content_types(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_display_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    display_id: u64,
) -> (u32, u32, u32, u32) {
    let (width, height) = crate::services::am::default_display_resolution();
    if crate::services::am::mode_trace_enabled() {
        log::warn!(
            "[mode-trace] ISystemDisplayService.GetDisplayMode id={} -> {}x{}@60",
            display_id,
            width,
            height
        );
    }
    (width, height, 60.0f32.to_bits(), 0)
}
pub fn set_display_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_underscan(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn set_display_underscan(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_content_type(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_content_type(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_rgb_range(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_rgb_range(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_cmu_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_cmu_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_contrast_ratio(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_contrast_ratio(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_gamma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_gamma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_cmu_luma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_display_cmu_luma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_display_crc_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_layer_presentation_submission_timestamps(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_shared_buffer_memory_handle_id(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn open_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn close_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn connect_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn disconnect_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn acquire_shared_frame_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn present_shared_frame_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_shared_frame_buffer_acquirable_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    kernel.vsync_handles.insert(h);
    h
}

pub fn fill_shared_frame_buffer_color(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cancel_shared_frame_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_dp2hdmi_controller(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

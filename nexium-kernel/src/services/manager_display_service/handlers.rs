use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

const NATIVE_WINDOW_PARCEL_SIZE: u64 = 56;

pub fn allocate_process_heap_block(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn free_process_heap_block(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_resolution(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> (u64, u64) {
    (1280, 720)
}
pub fn create_managed_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    1
}
pub fn destroy_managed_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_stray_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> (u64, u64) {
    (1, NATIVE_WINDOW_PARCEL_SIZE)
}
pub fn create_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    1
}
pub fn destroy_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_indirect_producer_end_point(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    1
}
pub fn destroy_indirect_producer_end_point(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_indirect_consumer_end_point(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    1
}
pub fn destroy_indirect_consumer_end_point(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_watermark_compositor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_watermark_text(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_watermark_layer_stacks(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn acquire_layer_texture_presenting_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    kernel.vsync_handles.insert(h);
    h
}
pub fn release_layer_texture_presenting_event(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_display_hotplug_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_display_mode_changed_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_display_hotplug_state(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn get_compositor_error_info(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn get_display_error_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_display_fatal_error_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn set_display_alpha(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_display_layer_stack(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_display_power_state(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_default_display(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn reset_display_panel(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_display_fatal_error_enabled(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn is_display_panel_on(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    true
}
pub fn get_internal_panel_id(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn add_to_layer_stack(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn remove_from_layer_stack(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_visibility(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_config(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn attach_layer_presentation_tracer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn detach_layer_presentation_tracer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn start_layer_presentation_recording(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn stop_layer_presentation_recording(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn start_layer_presentation_fence_wait(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn stop_layer_presentation_fence_wait(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_layer_presentation_all_fences_expired_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn enable_layer_auto_clear_transition_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn disable_layer_auto_clear_transition_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_layer_opacity(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn attach_layer_watermark_compositor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn detach_layer_watermark_compositor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_content_visibility(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_conductor_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_timestamp_tracking(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_indirect_producer_flip_offset(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_shared_buffer_static_storage(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn create_shared_buffer_transfer_memory(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn destroy_shared_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn bind_shared_low_level_layer_to_managed_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn bind_shared_low_level_layer_to_indirect_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn unbind_shared_low_level_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn connect_shared_low_level_layer_to_shared_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn disconnect_shared_low_level_layer_from_shared_buffer(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) {
}
pub fn create_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn destroy_shared_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn attach_shared_layer_to_low_level_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn force_detach_shared_layer_from_low_level_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn start_detach_shared_layer_from_low_level_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn finish_detach_shared_layer_from_low_level_layer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_shared_layer_detach_ready_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_shared_low_level_layer_synchronized_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn check_shared_low_level_layer_synchronized(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    1
}
pub fn register_shared_buffer_importer_aruid(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn unregister_shared_buffer_importer_aruid(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_shared_buffer_process_heap(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_shared_layer_layer_stacks(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn set_shared_layer_layer_stacks(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn present_detached_shared_frame_buffer_to_low_level_layer(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
) {
}
pub fn fill_detached_shared_frame_buffer_color(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_detached_shared_frame_buffer_image(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn set_detached_shared_frame_buffer_image(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn copy_detached_shared_frame_buffer_image(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_detached_shared_frame_buffer_sub_image(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_shared_frame_buffer_content_parameter(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn expand_startup_logo_on_shared_frame_buffer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

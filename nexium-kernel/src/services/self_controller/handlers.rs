use crate::kernel::Kernel;
use crate::kernel::handles::HandleType;
use nexium_ipc::IpcCtx;

pub fn exit(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn lock_exit(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn unlock_exit(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn enter_fatal_section(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn leave_fatal_section(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_library_applet_launchable_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    h
}

pub fn set_screen_shot_permission(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _permission: u32) {}
pub fn set_operation_mode_changed_notification(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_performance_mode_changed_notification(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_focus_handling_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u8, _b: u8, _c: u8) {}
pub fn set_restart_message_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_screen_shot_applet_identity_info(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u64, _b: u64) {}
pub fn set_out_of_focus_suspending_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_controller_firmware_update_section(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_requires_capture_button_short_pressed_message(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_album_image_orientation(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _orientation: u32) {}
pub fn set_desirable_keyboard_layout(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _layout: u32) {}
pub fn get_screen_shot_program_id(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 { 0 }
pub fn get_screen_shot_acd_index(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u8 { 0 }
pub fn get_screen_shot_apparent_platform(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u8 { 0 }
pub fn get_screen_shot_application_property(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) { (0, 0) }

pub fn create_managed_display_layer(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    kernel.handles.create_handle(HandleType::Event) as u64
}

pub fn is_system_buffer_sharing_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool { true }
pub fn get_system_shared_layer_handle(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) { (0, 0) }
pub fn get_system_shared_buffer_handle(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 { 0 }
pub fn create_managed_display_separable_layer(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) {
    let a = kernel.handles.create_handle(HandleType::Event) as u64;
    let b = kernel.handles.create_handle(HandleType::Event) as u64;
    (a, b)
}
pub fn set_managed_display_layer_separation_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _mode: u32) {}
pub fn set_recording_layer_composition_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_handles_request_to_display(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn approve_to_display(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn override_auto_sleep_time_and_dimming_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u64, _b: u64) {}
pub fn set_media_playback_state(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_idle_time_detection_extension(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _ext: u32) {}
pub fn get_idle_time_detection_extension(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 { 0 }
pub fn set_input_detection_source_set(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _set: u32) {}
pub fn report_user_is_active(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_current_illuminance(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 { 0 }
pub fn is_illuminance_available(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool { false }
pub fn set_auto_sleep_disabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn is_auto_sleep_disabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool { false }
pub fn report_multimedia_error(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _err: u32) {}
pub fn get_current_illuminance_ex(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 { 0 }
pub fn set_input_detection_policy(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _policy: u32) {}
pub fn cmd73(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u32) {}
pub fn set_wireless_priority_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _mode: u32) {}
pub fn get_accumulated_suspended_tick_value(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 { 0 }

pub fn get_accumulated_suspended_tick_changed_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    h
}

pub fn set_album_image_taken_notification_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_application_album_user_data(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn save_current_screenshot(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _album: u32) {}
pub fn set_record_volume_muted(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn cmd200(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u64, _b: u64, _c: u64, _d: u64) {}
pub fn cmd210(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn cmd211(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn cmd220(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u8) {}
pub fn cmd221(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u8) {}
pub fn cmd230(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u32) -> u16 { 0 }
pub fn get_debug_storage_channel(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

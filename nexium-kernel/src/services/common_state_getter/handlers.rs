use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_event_handle(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    if let Some(h) = kernel.applet_message_event {
        return h;
    }
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    kernel.applet_message_event = Some(h);
    h
}
pub fn receive_message(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    kernel.applet_messages.pop_front().unwrap_or(0)
}
pub fn get_this_applet_kind(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}
pub fn allow_to_enter_sleep(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn disallow_to_enter_sleep(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_operation_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    crate::hid_state::is_docked()
}
pub fn get_performance_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    if crate::hid_state::is_docked() {
        1
    } else {
        0
    }
}
pub fn get_cradle_status(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u8 {
    if crate::hid_state::is_docked() {
        1
    } else {
        0
    }
}
pub fn get_boot_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    false
}
pub fn get_current_focus_state(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    true
}
pub fn request_to_acquire_sleep_lock(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn release_sleep_lock(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn release_sleep_lock_transiently(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_acquired_sleep_lock_event(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    h
}
pub fn get_wakeup_count(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}
pub fn cmd15(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn get_home_button_reader_lock_accessor(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn get_reader_lock_accessor_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
) {
}
pub fn get_writer_lock_accessor_ex(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
) {
}
pub fn get_cradle_fw_version(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) {
    (0, 0)
}
pub fn is_vr_mode_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    false
}
pub fn set_vr_mode_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _flag: u8) {}
pub fn set_lcd_backligh_off_enabled(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn begin_vr_mode_ex(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn end_vr_mode_ex(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn is_in_controller_firmware_update_section(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> bool {
    false
}
pub fn set_vr_position_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) {
}
pub fn get_default_display_resolution(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u32, u32) {
    crate::services::am::default_display_resolution()
}

pub fn get_default_display_resolution_change_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    if let Some(handle) = kernel.display_resolution_change_event {
        return handle;
    }
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    kernel.display_resolution_change_event = Some(h);
    h
}
pub fn get_hdcp_authentication_state(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    1
}
pub fn get_hdcp_authentication_state_change_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn set_tv_power_state_matching_mode(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _mode: u32,
) {
}
pub fn get_application_id_by_content_action_name(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}
pub fn set_cpu_boost_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _mode: u32) {}
pub fn cancel_cpu_boost_mode(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_built_in_display_type(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0
}
pub fn perform_system_button_pressing_if_in_focus(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _button: u32,
) {
}
pub fn set_performance_configuration_changed_notification(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn get_current_performance_configuration(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}
pub fn set_handling_home_button_short_pressed_enabled(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn open_my_gpu_error_handler(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_applet_launched_history(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0
}
pub fn cmd130(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_operation_mode_system_info(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}
pub fn get_settings_platform_region(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u8 {
    0
}
pub fn activate_migration_service(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn deactivate_migration_service(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn disable_sleep_till_shutdown(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn suppress_disabling_sleep_temporarily(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _ns: u64,
) {
}
pub fn is_sleep_enabled(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    true
}
pub fn is_disabling_sleep_suppressed(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> bool {
    false
}
pub fn set_hid_input_magnification_for_application(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _c: u32,
) {
}
pub fn cmd610(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u64) {}
pub fn cmd611(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u8) {}
pub fn set_request_exit_to_library_applet_at_execute_next_program_enabled(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn get_launch_required_tick(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}
pub fn begin_vr_mode3d(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn end_vr_mode3d(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn is_vr_mode_enabled3d(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    false
}
pub fn get_vr_labo_goggle_viewport(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_panel_physical_size_for_specific_title(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}
pub fn get_panel_resolution_for_specific_title(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    0
}

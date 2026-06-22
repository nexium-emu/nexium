use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn pop_launch_parameter(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _kind: u32,
) -> bool {
    false
}
pub fn create_application_and_push_and_request_to_start(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
) {
}
pub fn create_application_and_push_and_request_to_start_for_quest(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _app_id: u64,
) {
}
pub fn create_application_and_request_to_start(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
) {
}
pub fn create_application_and_request_to_start_for_quest(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _app_id: u64,
) {
}
pub fn create_application_with_attribute_and_push_and_request_to_start_for_quest(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
) {
}
pub fn create_application_with_attribute_and_request_to_start_for_quest(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
) {
}
pub fn ensure_save_data(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _user_id_lo: u64,
    _user_id_hi: u64,
) -> u64 {
    0
}
pub fn get_desired_language(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0x0000_0053_552D_6E65
}
pub fn set_terminate_result(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _result: u32) {}
pub fn get_display_version(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) {
    (u64::from_le_bytes(*b"1.0.0\0\0\0"), 0)
}
pub fn get_launch_storage_info_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u8, u8) {
    (0, 0)
}
pub fn extend_save_data(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u8,
    _user_id_lo: u64,
    _user_id_hi: u64,
    _b: u64,
    _c: u64,
) -> u64 {
    0
}
pub fn get_save_data_size(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u8,
    _user_id_lo: u64,
    _user_id_hi: u64,
) -> (u64, u64) {
    (0, 0)
}
pub fn create_cache_storage(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _idx: u16,
    _size: u64,
    _journal: u64,
) -> (u32, u64) {
    (0, 0)
}
pub fn get_save_data_size_max(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_cache_storage_max(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u32, u64) {
    (0, 0)
}
pub fn begin_blocking_home_button_short_and_long_pressed(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _ns: u64,
) {
}
pub fn end_blocking_home_button_short_and_long_pressed(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn begin_blocking_home_button(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _ns: u64,
) {
}
pub fn end_blocking_home_button(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn select_application_license(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0
}
pub fn get_device_save_data_size_max(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64) {
    (0, 0)
}
pub fn get_limited_application_license(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}

pub fn get_limited_application_license_upgradable_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn notify_running(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    1
}
pub fn get_pseudo_device_id(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> (u64, u64) {
    (0x4E65_5869_756D_5044, 0x0123_4567_89AB_CDEF)
}
pub fn set_media_playback_state_for_application(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn is_game_play_recording_supported(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> bool {
    false
}
pub fn initialize_game_play_recording(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _size: u64,
) {
}
pub fn set_game_play_recording_state(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _state: u32,
) {
}
pub fn request_flush_game_playing_movie_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn request_to_shutdown(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn request_to_reboot(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn request_to_sleep(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn exit_and_request_to_show_thanks_message(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn enable_application_crash_report(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn initialize_application_copyright_frame_buffer(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _width: u32,
    _height: u32,
    _tmem_size: u64,
) {
}
pub fn set_application_copyright_image(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _x: u32,
    _y: u32,
    _w: u32,
    _h: u32,
    _origin: u32,
) {
}
pub fn set_application_copyright_visibility(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn query_application_play_statistics(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}
pub fn query_application_play_statistics_by_uid(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _user_id_lo: u64,
    _user_id_hi: u64,
) -> u32 {
    0
}
pub fn execute_program(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _kind: u32,
    _value: u64,
) {
}
pub fn clear_user_channel(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn unpop_to_user_channel(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_previous_program_index(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0xFFFFFFFF
}
pub fn enable_application_all_thread_dump_on_crash(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}

pub fn get_gpu_error_detected_system_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn set_delay_time_to_abort_on_gpu_error(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _ns: u64,
) {
}

pub fn get_friend_invitation_storage_channel_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn try_pop_from_friend_invitation_storage_channel(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn get_notification_storage_channel_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn try_pop_from_notification_storage_channel(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}

pub fn get_health_warning_disappeared_system_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn set_hdcp_authentication_activated(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _flag: u8,
) {
}
pub fn get_launch_required_version(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
    _a: u64,
) -> u64 {
    0
}
pub fn upgrade_launch_required_version(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _app_id: u64,
    _a: u64,
) {
}
pub fn send_server_maintenance_overlay_notification(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn get_last_application_exit_reason(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}
pub fn start_continuous_recording_flush_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn create_movie_maker(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn prepare_for_jit(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn notify_message_to_home_menu_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _msg: u32,
) {
}
pub fn perform_system_button_pressing(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _button: u32,
) {
}
pub fn invalidate_transition_layer(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn request_launch_application_with_user_and_argument_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) {
}
pub fn request_launch_application_by_application_launch_info_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn get_applet_resource_usage_info(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u64, u64, u64, u64) {
    (0, 0, 0, 0)
}
pub fn add_system_program_id_and_applet_id_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) {
}
pub fn add_operation_confirmed_library_applet_id_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
) {
}
pub fn get_program_id_from_applet_id_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _applet_id: u32,
) -> u64 {
    0
}
pub fn get_program_id_from_applet_id_and_library_applet_mode_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _applet_id: u32,
    _mode: u32,
) -> u64 {
    0
}
pub fn set_cpu_boost_mode_for_applet(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _mode: u32,
) {
}
pub fn cancel_cpu_boost_mode_for_applet(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn try_pop_from_applet_bound_channel_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
) {
}
pub fn alarm_setting_notification_disable_app_event_reserve(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn friend_invitation_clear_application_parameter(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn restrict_power_operation_for_secure_launch_mode_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn cmd150(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn create_floating_library_applet_accepter_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
) {
}
pub fn terminate_all_running_applications_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn create_general_storage_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) {
}
pub fn read_general_storage_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) -> u64 {
    0
}
pub fn write_general_storage_for_debug(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u64,
    _b: u64,
) {
}
pub fn cmd430(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn cmd431(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _a: u32) {}

pub fn get_grc_process_launched_system_event(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}
pub fn cmd910(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0
}

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
pub fn request_for_application_to_get_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn terminate_all_library_applets(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn are_any_library_applets_left(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn get_application_id(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_application_launch_request_info(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_users(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _present: bool) {}
pub fn check_rights_environment_available(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    true
}
pub fn get_ns_rights_environment_handle(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn get_desirable_uids(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn has_save_data_access_permission(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _app_id: u64,
) -> bool {
    false
}
pub fn request_application_soft_reset(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn restart_application_timer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cmd300(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn cmd310(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

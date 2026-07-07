use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn request_to_enter_sleep(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn enter_sleep(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn start_sleep_sequence(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u8) {}
pub fn start_shutdown_sequence(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn start_reboot_sequence(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn is_auto_power_down_requested(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn load_and_apply_idle_policy_settings(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn notify_cec_settings_changed(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_default_home_button_long_press_time(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _ns: u64,
) {
}
pub fn update_default_display_resolution(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn should_sleep_on_boot(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}

pub fn get_hdcp_authentication_failed_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn open_cradle_firmware_updater(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

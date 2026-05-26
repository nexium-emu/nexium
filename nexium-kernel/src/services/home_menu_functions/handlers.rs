use crate::kernel::Kernel;
use crate::kernel::handles::HandleType;
use nexium_ipc::IpcCtx;

pub fn request_to_get_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn lock_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn unlock_foreground(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn pop_from_general_channel(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

pub fn get_pop_from_general_channel_event(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

pub fn get_home_button_writer_lock_accessor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_writer_lock_accessor_ex(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u32) {}
pub fn is_sleep_enabled(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool { true }
pub fn is_reboot_enabled(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool { true }
pub fn launch_system_applet(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn launch_starter(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cmd60(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cmd61(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn pop_request_launch_application_for_debug(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> (u64, u64) { (0, 0) }
pub fn is_force_terminate_application_disabled_for_debug(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool { false }
pub fn launch_dev_menu(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_last_application_exit_reason(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _reason: u32) {}

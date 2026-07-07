use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn create_window(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _unk: u32) {}
pub fn get_applet_resource_user_id(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    1
}
pub fn get_applet_resource_user_id_of_caller_applet(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u64 {
    1
}
pub fn acquire_foreground_rights(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn release_foreground_rights(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn reject_to_change_into_background(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn set_applet_window_visibility(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _visible: u8,
) {
}
pub fn set_applet_gpu_time_slice(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _slice: u64,
) {
}

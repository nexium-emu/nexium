use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_launch_reason(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}
pub fn open_calling_library_applet(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn pop_context(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn cancel_winding_reservation(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn wind_and_do_reserved(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

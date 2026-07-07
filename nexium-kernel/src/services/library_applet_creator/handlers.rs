use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn create_library_applet_old(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u64) {}
pub fn terminate_all_library_applets(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn are_any_library_applets_left(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn create_library_applet(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u64, _b: u64) {}
pub fn create_storage(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _size: u64) {}
pub fn create_transfer_memory_storage(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _a: u64, _b: u64) {
}
pub fn create_handle_storage(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _size: u64) {}

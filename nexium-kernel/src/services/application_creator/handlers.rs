use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn create_application(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _app_id: u64) {}
pub fn pop_launch_requested_application(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn create_system_application(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _app_id: u64) {}
pub fn pop_floating_application_for_development(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}

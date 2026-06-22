use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_display_service_u_from_root(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_service(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_service_m_from_root(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_display_service_with_proxy_name_exchange(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn prepare_fatal(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn show_fatal(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn draw_fatal_rectangle(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn draw_fatal_text32(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u32 {
    0
}

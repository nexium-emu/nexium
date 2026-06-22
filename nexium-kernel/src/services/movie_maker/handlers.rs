use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_grc_movie_maker(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_layer_handle(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}

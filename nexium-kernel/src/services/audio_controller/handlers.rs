use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn set_expected_master_volume(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _main: u32,
    _library: u32,
) {
}
pub fn get_main_applet_expected_master_volume(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0x3F800000
}
pub fn get_library_applet_expected_master_volume(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0x3F800000
}
pub fn change_main_applet_master_volume(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _ns: u64,
) {
}
pub fn set_transparent_volume_rate(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _rate: u32,
) {
}
pub fn cmd5(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _a: u32,
    _b: u32,
    _c: u32,
    _d: u32,
) {
}

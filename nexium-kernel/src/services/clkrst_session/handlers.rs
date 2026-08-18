use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

const DEFAULT_CLOCK_RATE_HZ: u32 = 1_020_000_000;

pub fn set_clock_rate(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_clock_rate(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    DEFAULT_CLOCK_RATE_HZ
}

pub fn get_possible_clock_rates(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> (u32, u32) {
    (1, 0)
}

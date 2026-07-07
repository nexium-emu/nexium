use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_current_time_point(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_test_offset(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}
pub fn set_test_offset(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _offset: u64) {}

pub fn get_rtc_value(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn is_rtc_reset_detected(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> bool {
    false
}
pub fn get_setup_result_value(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    0
}
pub fn get_internal_offset(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u64 {
    0
}

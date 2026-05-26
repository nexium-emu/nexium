use crate::kernel::Kernel;
use crate::kernel::handles::HandleType;
use nexium_ipc::IpcCtx;

pub fn get_device_location_name(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn to_calendar_time_with_my_rule(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _posix_time: u64) {}

pub fn set_device_location_name(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_total_location_name_count(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 { 0 }
pub fn load_location_name_list(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _index: u32) -> u32 { 0 }
pub fn load_time_zone_rule(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_time_zone_rule_version(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_device_location_name_and_updated_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn set_device_location_name_with_time_zone_rule(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn parse_time_zone_binary(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_device_location_name_operation_event_readable_handle(kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> u32 {
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    event
}

pub fn to_calendar_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _posix_time: u64) {}
pub fn to_posix_time(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _calendar_time: u64) -> u32 { 0 }
pub fn to_posix_time_with_my_rule(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32, _calendar_time: u64) -> u32 { 0 }

use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_device_location_name(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) -> Vec<u8> {
    let mut out = vec![0u8; 0x24];
    out[..3].copy_from_slice(b"UTC");
    out
}
pub fn to_calendar_time_with_my_rule(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    posix_time: i64,
) -> Vec<u8> {
    utc_calendar_time(posix_time)
}

pub fn set_device_location_name(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_total_location_name_count(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    0
}
pub fn load_location_name_list(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _index: u32,
) -> u32 {
    0
}
pub fn load_time_zone_rule(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_time_zone_rule_version(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}
pub fn get_device_location_name_and_updated_time(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn set_device_location_name_with_time_zone_rule(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) {
}
pub fn parse_time_zone_binary(_kernel: &mut Kernel, _ctx: &mut IpcCtx, _session: u32) {}

pub fn get_device_location_name_operation_event_readable_handle(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
) -> u32 {
    let event = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(event, false);
    event
}

pub fn to_calendar_time(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    posix_time: i64,
) -> Vec<u8> {
    utc_calendar_time(posix_time)
}

fn utc_calendar_time(posix_time: i64) -> Vec<u8> {
    let days = posix_time.div_euclid(86_400);
    let day_seconds = posix_time.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = day_seconds / 3_600;
    let minute = day_seconds % 3_600 / 60;
    let second = day_seconds % 60;
    let month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let leap_day = i32::from(month > 2 && is_leap_year(year));
    let year_day = month_starts[month as usize - 1] + day as i32 - 1 + leap_day;
    let day_of_week = (days + 4).rem_euclid(7) as i32;

    let mut out = vec![0u8; 0x20];
    out[0..2].copy_from_slice(&(year as i16).to_le_bytes());
    out[2] = month;
    out[3] = day;
    out[4] = hour as u8;
    out[5] = minute as u8;
    out[6] = second as u8;
    out[8..12].copy_from_slice(&day_of_week.to_le_bytes());
    out[12..16].copy_from_slice(&year_day.to_le_bytes());
    out[16..19].copy_from_slice(b"UTC");
    out
}

fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month as u8, day as u8)
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}
pub fn to_posix_time(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _calendar_time: u64,
) -> u32 {
    0
}
pub fn to_posix_time_with_my_rule(
    _kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _calendar_time: u64,
) -> u32 {
    0
}

#[cfg(test)]
mod tests {
    use super::utc_calendar_time;

    #[test]
    fn converts_unix_epoch_to_utc_calendar() {
        let out = utc_calendar_time(0);
        assert_eq!(&out[0..8], &[0xb2, 0x07, 1, 1, 0, 0, 0, 0]);
        assert_eq!(i32::from_le_bytes(out[8..12].try_into().unwrap()), 4);
        assert_eq!(i32::from_le_bytes(out[12..16].try_into().unwrap()), 0);
        assert_eq!(&out[16..24], b"UTC\0\0\0\0\0");
        assert_eq!(&out[24..32], &[0; 8]);
    }

    #[test]
    fn converts_leap_day_to_utc_calendar() {
        let out = utc_calendar_time(951_827_696);
        assert_eq!(&out[0..8], &[0xd0, 0x07, 2, 29, 12, 34, 56, 0]);
        assert_eq!(i32::from_le_bytes(out[8..12].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(out[12..16].try_into().unwrap()), 59);
    }
}

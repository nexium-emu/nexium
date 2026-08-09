#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum ThreadCpuSetTarget {
    GuestCore0 = 0,
    GpuPrep = 1,
    GpuSubmit = 2,
}

#[cfg(windows)]
impl ThreadCpuSetTarget {
    fn env_name(self) -> &'static str {
        match self {
            Self::GuestCore0 => "NEXIUM_WIN_CORE0_CPUSET_MASK",
            Self::GpuPrep => "NEXIUM_WIN_PREP_CPUSET_MASK",
            Self::GpuSubmit => "NEXIUM_WIN_GPU_SUBMIT_CPUSET_MASK",
        }
    }

    fn thread_name(self) -> &'static str {
        match self {
            Self::GuestCore0 => "guest core0",
            Self::GpuPrep => "nexium-gpu-prep",
            Self::GpuSubmit => "nexium-gpu-submit",
        }
    }
}

pub fn apply_current_thread_cpu_set(target: ThreadCpuSetTarget) {
    #[cfg(windows)]
    windows::apply(target);

    #[cfg(not(windows))]
    let _ = target;
}

#[cfg(any(windows, test))]
fn parse_cpu_set_mask(value: &str) -> Result<Option<u64>, ()> {
    let value = value.trim();
    let parsed = if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16)
    } else {
        value.parse::<u64>()
    }
    .map_err(|_| ())?;

    Ok((parsed != 0).then_some(parsed))
}

#[cfg(windows)]
mod windows {
    use super::{parse_cpu_set_mask, ThreadCpuSetTarget};
    use std::ffi::{c_char, c_void};
    use std::sync::atomic::{AtomicU8, Ordering};

    #[repr(C)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    type SetThreadSelectedCpuSetMasks =
        unsafe extern "system" fn(*mut c_void, *const GroupAffinity, u16) -> i32;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn GetModuleHandleA(module_name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
    }

    static LOGGED_TARGETS: AtomicU8 = AtomicU8::new(0);

    pub(super) fn apply(target: ThreadCpuSetTarget) {
        let Some(raw_value) = std::env::var_os(target.env_name()) else {
            return;
        };
        let Ok(value) = raw_value.into_string() else {
            warn_once(target, || {
                log::warn!(
                    "[cpu-set] {} is not valid Unicode; {} remains unpinned",
                    target.env_name(),
                    target.thread_name()
                );
            });
            return;
        };
        let mask = match parse_cpu_set_mask(&value) {
            Ok(Some(mask)) => mask,
            Ok(None) => return,
            Err(()) => {
                warn_once(target, || {
                    log::warn!(
                        "[cpu-set] invalid {}={value:?}; expected a decimal or 0x-prefixed u64 mask",
                        target.env_name()
                    );
                });
                return;
            }
        };

        let Ok(mask) = usize::try_from(mask) else {
            warn_once(target, || {
                log::warn!(
                    "[cpu-set] {} mask {mask:#x} does not fit this Windows target; {} remains unpinned",
                    target.env_name(),
                    target.thread_name()
                );
            });
            return;
        };

        let affinity = GroupAffinity {
            mask,
            group: 0,
            reserved: [0; 3],
        };
        let result = unsafe { set_current_thread_cpu_set(&affinity) };
        match result {
            Ok(()) => log_once(target, || {
                log::info!(
                    "[cpu-set] {} selected group 0 mask {mask:#x} via {}",
                    target.thread_name(),
                    target.env_name()
                );
            }),
            Err(error) => warn_once(target, || {
                log::warn!(
                    "[cpu-set] failed to select group 0 mask {mask:#x} for {} via {}: {error}",
                    target.thread_name(),
                    target.env_name()
                );
            }),
        }
    }

    unsafe fn set_current_thread_cpu_set(affinity: &GroupAffinity) -> std::io::Result<()> {
        let module = GetModuleHandleA(c"kernel32.dll".as_ptr());
        if module.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let proc = GetProcAddress(module, c"SetThreadSelectedCpuSetMasks".as_ptr());
        if proc.is_null() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "SetThreadSelectedCpuSetMasks is unavailable",
            ));
        }
        let set_masks: SetThreadSelectedCpuSetMasks = std::mem::transmute(proc);
        if set_masks(GetCurrentThread(), affinity, 1) == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn log_once(target: ThreadCpuSetTarget, log: impl FnOnce()) {
        let bit = 1u8 << target as u8;
        if LOGGED_TARGETS.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            log();
        }
    }

    fn warn_once(target: ThreadCpuSetTarget, log: impl FnOnce()) {
        log_once(target, log);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_set_mask;

    #[test]
    fn parses_decimal_and_hex_masks() {
        assert_eq!(parse_cpu_set_mask("48"), Ok(Some(0x30)));
        assert_eq!(parse_cpu_set_mask("0x30"), Ok(Some(0x30)));
        assert_eq!(parse_cpu_set_mask("  0X300  "), Ok(Some(0x300)));
        assert_eq!(
            parse_cpu_set_mask("18446744073709551615"),
            Ok(Some(u64::MAX))
        );
    }

    #[test]
    fn zero_is_an_inert_mask() {
        assert_eq!(parse_cpu_set_mask("0"), Ok(None));
        assert_eq!(parse_cpu_set_mask("0x0"), Ok(None));
    }

    #[test]
    fn rejects_malformed_or_out_of_range_masks() {
        for value in ["", "0x", "-1", "xyz", "0b10", "18446744073709551616"] {
            assert_eq!(parse_cpu_set_mask(value), Err(()), "value={value:?}");
        }
    }
}

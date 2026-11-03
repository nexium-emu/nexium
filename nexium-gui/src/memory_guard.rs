use std::time::Duration;
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};

pub fn install() {
    let spawned = std::thread::Builder::new()
        .name("nexium-memory-guard".into())
        .spawn(|| loop {
            std::thread::sleep(Duration::from_millis(250));
            if let Some((available, total, working_set)) = sample() {
                if exhausted(available, total, working_set) {
                    log::error!(
                        "memory guard: {} MiB of {} MiB RAM free with a {} MiB working set; stopping NeXium before the system starts thrashing",
                        available >> 20,
                        total >> 20,
                        working_set >> 20
                    );
                    let _ = nexium_common::FileLogger::flush_on_fault();
                    unsafe { TerminateProcess(GetCurrentProcess(), 3) };
                }
            }
        });
    if let Err(error) = spawned {
        log::warn!("memory guard: thread spawn failed: {error}");
    }
}

fn exhausted(available: u64, total: u64, working_set: u64) -> bool {
    available < total / 32 && working_set > total / 2
}

fn sample() -> Option<(u64, u64, u64)> {
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..unsafe { std::mem::zeroed() }
    };
    let ok = unsafe {
        GlobalMemoryStatusEx(&mut status) != 0
            && K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0
    };
    ok.then_some((status.ullAvailPhys, status.ullTotalPhys, counters.WorkingSetSize as u64))
}

#[cfg(test)]
mod tests {
    use super::exhausted;

    #[test]
    fn stops_only_when_ram_is_nearly_gone_and_nexium_holds_most_of_it() {
        let total = 64u64 << 30;
        assert!(exhausted(1 << 30, total, 40 << 30));
        assert!(!exhausted(8 << 30, total, 40 << 30));
        assert!(!exhausted(1 << 30, total, 16 << 30));
    }
}

use crossbeam::atomic::AtomicCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Phase {
    Idle = 0,
    Kick = 1,
    Entry = 2,
    Methods = 3,
    Flush = 4,
    CbufPages = 5,
    CbufResync = 6,
    RenderWait = 7,
    ComputeResolve = 8,
    Semaphore = 9,
    Present = 10,
    Barrier = 11,
    KickEnd = 12,
}

const PHASE_NAMES: [&str; 13] = [
    "idle",
    "kick",
    "entry",
    "methods",
    "flush",
    "cbuf-pages",
    "cbuf-resync",
    "render-wait",
    "compute-resolve",
    "semaphore",
    "present",
    "barrier",
    "kick-end",
];

static PHASE: AtomicUsize = AtomicUsize::new(0);
static DETAIL: AtomicU64 = AtomicU64::new(0);
static PROGRESS: AtomicU64 = AtomicU64::new(0);
static WORKER_TID: AtomicU32 = AtomicU32::new(0);
static RENDER_TID: AtomicU32 = AtomicU32::new(0);
static RENDER_LABEL: AtomicCell<&'static str> = AtomicCell::new("-");
static RENDER_DETAIL: AtomicU64 = AtomicU64::new(0);
static RENDER_PROGRESS: AtomicU64 = AtomicU64::new(0);
static PUSHER_OWNER: AtomicCell<&'static str> = AtomicCell::new("-");
static PUSHER_OWNER_TID: AtomicU32 = AtomicU32::new(0);

#[inline]
pub fn render_phase(label: &'static str, detail: u64) {
    RENDER_LABEL.store(label);
    RENDER_DETAIL.store(detail, Ordering::Relaxed);
    RENDER_PROGRESS.fetch_add(1, Ordering::Relaxed);
}

pub fn register_render_thread() {
    RENDER_TID.store(current_thread_id(), Ordering::Relaxed);
}

#[inline]
pub fn pusher_lock_acquired(site: &'static str) {
    PUSHER_OWNER.store(site);
    PUSHER_OWNER_TID.store(current_thread_id(), Ordering::Relaxed);
}

#[inline]
pub fn pusher_lock_released() {
    PUSHER_OWNER.store("-");
    PUSHER_OWNER_TID.store(0, Ordering::Relaxed);
}

pub fn pusher_lock_wait_report(site: &'static str, waited: Duration) {
    log::error!(
        "[pusher-lock] site={} waited_s={:.0} owner_site={} owner_tid={} caller_tid={}",
        site,
        waited.as_secs_f64(),
        PUSHER_OWNER.load(),
        PUSHER_OWNER_TID.load(Ordering::Relaxed),
        current_thread_id(),
    );
}

#[inline]
pub fn phase(phase: Phase, detail: u64) {
    PHASE.store(phase as usize, Ordering::Relaxed);
    DETAIL.store(detail, Ordering::Relaxed);
    PROGRESS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn progress_snapshot() -> (u64, u64) {
    (
        PROGRESS.load(Ordering::Relaxed),
        RENDER_PROGRESS.load(Ordering::Relaxed),
    )
}

fn stall_seconds() -> Option<u64> {
    static SECS: OnceLock<Option<u64>> = OnceLock::new();
    *SECS.get_or_init(|| {
        std::env::var("NEXIUM_GPU_WATCHDOG")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|secs| *secs > 0)
    })
}

#[cfg(windows)]
fn thread_cpu_ms(tid: u32) -> Option<u64> {
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> *mut std::ffi::c_void;
        fn GetThreadTimes(
            thread: *mut std::ffi::c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
        fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
    }
    const THREAD_QUERY_INFORMATION: u32 = 0x0040;
    unsafe {
        let handle = OpenThread(THREAD_QUERY_INFORMATION, 0, tid);
        if handle.is_null() {
            return None;
        }
        let (mut creation, mut exit, mut kernel, mut user) = (0u64, 0u64, 0u64, 0u64);
        let ok = GetThreadTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user);
        CloseHandle(handle);
        (ok != 0).then(|| (kernel + user) / 10_000)
    }
}

#[cfg(not(windows))]
fn thread_cpu_ms(_tid: u32) -> Option<u64> {
    None
}

#[cfg(windows)]
fn current_thread_id() -> u32 {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    unsafe { GetCurrentThreadId() }
}

#[cfg(not(windows))]
fn current_thread_id() -> u32 {
    0
}

pub fn armed() -> bool {
    stall_seconds().is_some()
}

fn sampler_config() -> Option<(u64, u64, u64)> {
    let value = std::env::var("NEXIUM_GPU_SAMPLER").ok()?;
    let mut parts = value.split(',').map(|part| part.trim().parse::<u64>().ok());
    let start = parts.next()??;
    let duration = parts.next()??;
    let interval_ms = parts.next().flatten().unwrap_or(1).max(1);
    Some((start, duration, interval_ms))
}

fn install_sampler() {
    let Some((start_s, duration_s, interval_ms)) = sampler_config() else {
        return;
    };
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("nexium-gpu-sampler".to_string())
        .spawn(move || {
            std::thread::sleep(Duration::from_secs(start_s));
            let sample_render = std::env::var("NEXIUM_GPU_SAMPLER_THREAD")
                .ok()
                .is_some_and(|value| value.eq_ignore_ascii_case("render"));
            let tid = if sample_render {
                RENDER_TID.load(Ordering::Relaxed)
            } else {
                WORKER_TID.load(Ordering::Relaxed)
            };
            let interval = Duration::from_millis(interval_ms);
            let samples = (duration_s * 1000 / interval_ms) as usize;
            log::warn!(
                "[gpu-sampler] sampling worker tid={} samples={} interval_ms={}",
                tid,
                samples,
                interval_ms
            );
            let stacks = super::stackdump::sample_thread(tid, samples, interval);
            let path = std::env::temp_dir().join("nexium-gpu-samples.txt");
            let mut text = format!("base {:x}\n", super::stackdump::exe_module_base());
            for stack in &stacks {
                for (index, pc) in stack.iter().enumerate() {
                    if index != 0 {
                        text.push(' ');
                    }
                    text.push_str(&format!("{:x}", pc));
                }
                text.push('\n');
            }
            match std::fs::write(&path, text) {
                Ok(()) => log::warn!(
                    "[gpu-sampler] wrote {} samples to {}",
                    stacks.len(),
                    path.display()
                ),
                Err(error) => log::error!("[gpu-sampler] write failed: {error}"),
            }
        });
    if let Err(error) = spawned {
        log::error!("[gpu-sampler] spawn failed: {error}");
    }
}

pub fn register_worker_thread() {
    WORKER_TID.store(current_thread_id(), Ordering::Relaxed);
    install_sampler();
}

pub fn install() {
    let Some(stall_secs) = stall_seconds() else {
        return;
    };
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("nexium-gpu-watchdog".to_string())
        .spawn(move || {
            let mut last_progress = PROGRESS.load(Ordering::Relaxed);
            let mut last_render_progress = RENDER_PROGRESS.load(Ordering::Relaxed);
            let mut stalled_for = 0u64;
            let mut last_cpu = None;
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let progress = PROGRESS.load(Ordering::Relaxed);
                let phase = PHASE.load(Ordering::Relaxed);
                let render_progress = RENDER_PROGRESS.load(Ordering::Relaxed);
                let worker_stalled = phase != Phase::Idle as usize && progress == last_progress;
                let render_label = RENDER_LABEL.load();
                let render_stalled = render_label != "idle-recv" && render_label != "-"
                    && render_progress == last_render_progress;
                last_progress = progress;
                last_render_progress = render_progress;
                if !worker_stalled && !render_stalled {
                    stalled_for = 0;
                    last_cpu = None;
                    continue;
                }
                stalled_for += 1;
                if stalled_for < stall_secs || stalled_for % stall_secs != 0 {
                    continue;
                }
                let tid = WORKER_TID.load(Ordering::Relaxed);
                let cpu = thread_cpu_ms(tid);
                let cpu_delta = match (cpu, last_cpu) {
                    (Some(now), Some(before)) => now.saturating_sub(before),
                    _ => 0,
                };
                last_cpu = cpu;
                let render_tid = RENDER_TID.load(Ordering::Relaxed);
                log::error!(
                    "[gpu-watchdog] worker stalled_s={} phase={} detail={:#x} progress={} worker_tid={} cpu_ms_total={:?} cpu_ms_since_last_report={} | render phase={} detail={:#x} progress={} render_tid={} render_cpu_ms={:?} | pusher_lock owner_site={} owner_tid={}",
                    stalled_for,
                    PHASE_NAMES.get(phase).copied().unwrap_or("?"),
                    DETAIL.load(Ordering::Relaxed),
                    progress,
                    tid,
                    cpu,
                    cpu_delta,
                    RENDER_LABEL.load(),
                    RENDER_DETAIL.load(Ordering::Relaxed),
                    RENDER_PROGRESS.load(Ordering::Relaxed),
                    render_tid,
                    thread_cpu_ms(render_tid),
                    PUSHER_OWNER.load(),
                    PUSHER_OWNER_TID.load(Ordering::Relaxed),
                );
                let reports = stalled_for / stall_secs;
                if reports == 1 || reports % 10 == 0 {
                    super::stackdump::dump_all_threads("gpu-watchdog stall");
                }
            }
        });
    if let Err(error) = spawned {
        log::error!("[gpu-watchdog] spawn failed: {error}");
    } else {
        log::info!("[gpu-watchdog] armed stall_s={stall_secs}");
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicCell;
    use std::sync::Barrier;

    #[test]
    fn watchdog_label_storage_preserves_whole_strings_under_concurrent_updates() {
        let labels = ["-", "x", "render:finish-marker-after-pending-draws", "λ"];
        let value: AtomicCell<&'static str> = AtomicCell::new(labels[0]);
        let started = Barrier::new(3);
        std::thread::scope(|scope| {
            for offset in [0, 1] {
                let value = &value;
                let started = &started;
                scope.spawn(move || {
                    started.wait();
                    for index in 0..20_000 {
                        value.store(labels[(index + offset) % labels.len()]);
                    }
                });
            }
            started.wait();
            for _ in 0..20_000 {
                assert!(labels.contains(&value.load()));
            }
        });
        value.store("-");
        assert_eq!(value.load(), "-");
    }
}

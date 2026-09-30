use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

use crossbeam::channel::{unbounded, Sender};

type CompletionJob = Box<dyn FnOnce() + Send + 'static>;

struct CompletionQueue {
    tx: Sender<CompletionJob>,
}

impl CompletionQueue {
    fn new_named(name: &str) -> Self {
        let (tx, rx) = unbounded::<CompletionJob>();
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_err() {
                        log::error!("[gpu-completion] job panicked; worker continuing");
                    }
                }
            })
            .expect("spawn GPU completion worker");
        Self { tx }
    }

    fn submit(&self, job: CompletionJob) -> Result<(), CompletionJob> {
        self.tx.send(job).map_err(|error| error.0)
    }
}

fn completion_queue() -> &'static CompletionQueue {
    static QUEUE: OnceLock<CompletionQueue> = OnceLock::new();
    QUEUE.get_or_init(|| CompletionQueue::new_named("nexium-gpu-completion"))
}

pub(crate) fn async_semaphore_completion_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_ASYNC_SEMREL").ok().as_deref(),
            Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF")
        )
    })
}

fn run_completion_after_wait(
    wait_for_predecessor: impl FnOnce() -> bool,
    completion: impl FnOnce(),
) -> bool {
    let started = std::time::Instant::now();
    if !wait_for_predecessor() {
        return false;
    }
    record_gpu_lag(started.elapsed());
    completion();
    true
}

fn record_gpu_lag(waited: Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static WAITS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    static MAX_NS: AtomicU64 = AtomicU64::new(0);
    if !crate::gpu::pusher::kickprof::rate_enabled() {
        return;
    }
    let ns = waited.as_nanos() as u64;
    TOTAL_NS.fetch_add(ns, Ordering::Relaxed);
    MAX_NS.fetch_max(ns, Ordering::Relaxed);
    let waits = WAITS.fetch_add(1, Ordering::Relaxed) + 1;
    if waits % 256 == 0 {
        let total = TOTAL_NS.swap(0, Ordering::Relaxed);
        let max = MAX_NS.swap(0, Ordering::Relaxed);
        log::warn!(
            "[gpu-lag] completions=256 mean_ms={:.3} max_ms={:.3}",
            total as f64 / 256.0 / 1_000_000.0,
            max as f64 / 1_000_000.0
        );
    }
}

static PENDING_GUEST_WRITES: AtomicUsize = AtomicUsize::new(0);

struct PendingGuestWriteGuard;

impl PendingGuestWriteGuard {
    fn new() -> Self {
        PENDING_GUEST_WRITES.fetch_add(1, Ordering::AcqRel);
        Self
    }
}

impl Drop for PendingGuestWriteGuard {
    fn drop(&mut self) {
        PENDING_GUEST_WRITES.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) fn wait_for_pending_guest_writes(timeout: Duration) -> bool {
    if PENDING_GUEST_WRITES.load(Ordering::Acquire) == 0 {
        return true;
    }
    let started = std::time::Instant::now();
    let mut spins = 0u32;
    while PENDING_GUEST_WRITES.load(Ordering::Acquire) != 0 {
        if started.elapsed() >= timeout {
            return false;
        }
        if spins < 256 {
            spins += 1;
            std::thread::yield_now();
        } else {
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    true
}

pub(crate) fn submit_renderer_completion(
    renderer: std::sync::Arc<nexium_gpu::Renderer>,
    completion: impl FnOnce() + Send + 'static,
) -> bool {
    let guard = PendingGuestWriteGuard::new();
    let marker = move || {
        let target = renderer.submitted_generation();
        let completion_job = Box::new(move || {
            let _guard = guard;
            let timeline_available = renderer.timeline_sync_available();
            let waited = run_completion_after_wait(
                || {
                    if timeline_available {
                        renderer.wait_submit_generation_patiently(target, "completion")
                    } else {
                        renderer.wait_idle_checked()
                    }
                },
                completion,
            );
            if !waited {
                if timeline_available {
                    log::error!(
                        "[gpu-completion] timeline wait failed target={}; completion dropped and device-idle fallback suppressed",
                        target
                    );
                } else {
                    log::error!(
                        "[gpu-completion] legacy device-idle wait failed target={}; completion dropped",
                        target
                    );
                }
            }
        }) as CompletionJob;
        if let Err(completion_job) = completion_queue().submit(completion_job) {
            completion_job();
        }
    };

    if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
        render_thread.submit_named("semaphore-completion-marker", Box::new(marker));
        true
    } else {
        marker();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{run_completion_after_wait, CompletionQueue};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn completion_worker_waits_for_predecessor_and_preserves_fifo_order() {
        let queue = CompletionQueue::new_named("nexium-completion-order-test");
        let (gate_tx, gate_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let order = Arc::new(Mutex::new(Vec::new()));

        let first_order = Arc::clone(&order);
        let first_done = done_tx.clone();
        assert!(queue
            .submit(Box::new(move || {
                gate_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                first_order.lock().unwrap().push(1);
                first_done.send(()).unwrap();
            }))
            .is_ok());

        let second_order = Arc::clone(&order);
        assert!(queue
            .submit(Box::new(move || {
                second_order.lock().unwrap().push(2);
                done_tx.send(()).unwrap();
            }))
            .is_ok());

        assert!(done_rx.recv_timeout(Duration::from_millis(25)).is_err());
        gate_tx.send(()).unwrap();
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn failed_predecessor_wait_drops_completion() {
        let calls = AtomicUsize::new(0);
        assert!(!run_completion_after_wait(
            || false,
            || {
                calls.fetch_add(1, Ordering::Relaxed);
            },
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn successful_predecessor_wait_runs_completion_once() {
        let completion_calls = AtomicUsize::new(0);
        assert!(run_completion_after_wait(
            || true,
            || {
                completion_calls.fetch_add(1, Ordering::Relaxed);
            },
        ));
        assert_eq!(completion_calls.load(Ordering::Relaxed), 1);
    }
}

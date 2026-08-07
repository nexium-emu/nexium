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

fn run_completion_after_waits(
    wait_timeline: impl FnOnce() -> bool,
    wait_idle: impl FnOnce() -> bool,
    completion: impl FnOnce(),
) -> bool {
    if !wait_timeline() && !wait_idle() {
        return false;
    }
    completion();
    true
}

pub(crate) fn submit_renderer_completion(
    renderer: std::sync::Arc<nexium_gpu::Renderer>,
    completion: impl FnOnce() + Send + 'static,
) -> bool {
    let marker = move || {
        let target = renderer.submitted_generation();
        let completion_job = Box::new(move || {
            if !run_completion_after_waits(
                || renderer.wait_submit_generation(target, Duration::from_secs(3)),
                || renderer.wait_idle_checked(),
                completion,
            ) {
                log::error!(
                    "[gpu-completion] timeline and device-idle waits failed target={}; completion dropped",
                    target
                );
            }
        }) as CompletionJob;
        if let Err(completion_job) = completion_queue().submit(completion_job) {
            completion_job();
        }
    };

    if let Some(render_thread) = crate::render_thread::maybe_render_thread() {
        render_thread.submit_timeout_named(
            "semaphore-completion-marker",
            Box::new(marker),
            Duration::from_secs(3),
        )
    } else {
        marker();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{run_completion_after_waits, CompletionQueue};
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
    fn failed_timeline_and_idle_waits_drop_completion() {
        let calls = AtomicUsize::new(0);
        assert!(!run_completion_after_waits(
            || false,
            || false,
            || {
                calls.fetch_add(1, Ordering::Relaxed);
            },
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn checked_idle_fallback_allows_completion() {
        let idle_calls = AtomicUsize::new(0);
        let completion_calls = AtomicUsize::new(0);
        assert!(run_completion_after_waits(
            || false,
            || {
                idle_calls.fetch_add(1, Ordering::Relaxed);
                true
            },
            || {
                completion_calls.fetch_add(1, Ordering::Relaxed);
            },
        ));
        assert_eq!(idle_calls.load(Ordering::Relaxed), 1);
        assert_eq!(completion_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn successful_timeline_wait_skips_idle_fallback() {
        let idle_calls = AtomicUsize::new(0);
        let completion_calls = AtomicUsize::new(0);
        assert!(run_completion_after_waits(
            || true,
            || {
                idle_calls.fetch_add(1, Ordering::Relaxed);
                false
            },
            || {
                completion_calls.fetch_add(1, Ordering::Relaxed);
            },
        ));
        assert_eq!(idle_calls.load(Ordering::Relaxed), 0);
        assert_eq!(completion_calls.load(Ordering::Relaxed), 1);
    }
}

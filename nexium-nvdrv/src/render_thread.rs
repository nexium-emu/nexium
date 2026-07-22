use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::OnceLock;

pub type RenderJob = Box<dyn FnOnce() + Send + 'static>;

pub struct RenderThread {
    tx: SyncSender<RenderJob>,
}

impl RenderThread {
    fn new() -> Self {
        Self::new_named("nexium-render")
    }

    fn new_named(name: &str) -> Self {
        let queue_depth = render_queue_depth();
        let (tx, rx) = sync_channel::<RenderJob>(queue_depth);
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let profile = std::env::var_os("NEXIUM_RENDER_PROFILE").is_some();
                    let started = profile.then(std::time::Instant::now);
                    job();
                    if let Some(started) = started {
                        let elapsed = started.elapsed();
                        if elapsed >= std::time::Duration::from_millis(10) {
                            log::warn!(
                                "[render-job] worker={} elapsed_ms={:.3}",
                                std::thread::current().name().unwrap_or("nexium-render"),
                                elapsed.as_secs_f64() * 1000.0,
                            );
                        }
                    }
                }
            })
            .expect("spawn render worker thread");
        RenderThread { tx }
    }

    pub fn submit(&self, job: RenderJob) {
        self.submit_named("unnamed", job);
    }

    pub fn submit_named(&self, label: &'static str, job: RenderJob) {
        let profile = std::env::var_os("NEXIUM_RENDER_PROFILE").is_some();
        let started = profile.then(std::time::Instant::now);
        let _ = self.tx.send(job);
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} blocked_ms={:.3}",
                    label,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
    }

    pub fn try_submit(&self, job: RenderJob) -> bool {
        match self.tx.try_send(job) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn submit_timeout(&self, job: RenderJob, timeout: std::time::Duration) -> bool {
        self.submit_timeout_named("unnamed", job, timeout)
    }

    pub fn submit_timeout_named(
        &self,
        label: &'static str,
        job: RenderJob,
        timeout: std::time::Duration,
    ) -> bool {
        let profile = std::env::var_os("NEXIUM_RENDER_PROFILE").is_some();
        let started = profile.then(std::time::Instant::now);
        let deadline = std::time::Instant::now() + timeout;
        let mut job = job;
        let submitted = loop {
            match self.tx.try_send(job) {
                Ok(()) => break true,
                Err(TrySendError::Full(j)) => {
                    if std::time::Instant::now() >= deadline {
                        break false;
                    }
                    job = j;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(TrySendError::Disconnected(_)) => break false,
            }
        };
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                log::warn!(
                    "[render-submit] label={} blocked_ms={:.3}",
                    label,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
        submitted
    }
}

fn render_queue_depth() -> usize {
    std::env::var("NEXIUM_RENDER_QUEUE_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|depth| *depth > 0)
        .unwrap_or(32)
}

fn async_render_enabled() -> bool {
    match std::env::var("NEXIUM_ASYNC_RENDER").ok().as_deref() {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") | Some("no")
        | Some("NO") => false,
        _ => true,
    }
}

pub fn maybe_render_thread() -> Option<&'static RenderThread> {
    static RT: OnceLock<Option<RenderThread>> = OnceLock::new();
    RT.get_or_init(|| {
        if async_render_enabled() {
            log::info!(
                "nexium-nvdrv: async render thread ENABLED (set NEXIUM_ASYNC_RENDER=0 to disable)"
            );
            Some(RenderThread::new())
        } else {
            log::info!("nexium-nvdrv: async render thread DISABLED (NEXIUM_ASYNC_RENDER=0)");
            None
        }
    })
    .as_ref()
}

pub fn present_thread() -> &'static RenderThread {
    if dedicated_present_thread() {
        static PT: OnceLock<RenderThread> = OnceLock::new();
        return PT.get_or_init(|| RenderThread::new_named("nexium-present"));
    }
    if let Some(rt) = maybe_render_thread() {
        return rt;
    }
    static PT: OnceLock<RenderThread> = OnceLock::new();
    PT.get_or_init(|| RenderThread::new_named("nexium-present"))
}

fn dedicated_present_thread() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_DEDICATED_PRESENT_THREAD").is_some())
}

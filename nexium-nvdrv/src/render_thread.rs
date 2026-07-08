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
        let (tx, rx) = sync_channel::<RenderJob>(32);
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("spawn render worker thread");
        RenderThread { tx }
    }

    pub fn submit(&self, job: RenderJob) {
        let _ = self.tx.send(job);
    }

    pub fn try_submit(&self, job: RenderJob) -> bool {
        match self.tx.try_send(job) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn submit_timeout(&self, job: RenderJob, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        let mut job = job;
        loop {
            match self.tx.try_send(job) {
                Ok(()) => return true,
                Err(TrySendError::Full(j)) => {
                    if std::time::Instant::now() >= deadline {
                        return false;
                    }
                    job = j;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(TrySendError::Disconnected(_)) => return false,
            }
        }
    }
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

use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::OnceLock;

pub type RenderJob = Box<dyn FnOnce() + Send + 'static>;

pub struct RenderThread {
    tx: SyncSender<RenderJob>,
}

impl RenderThread {
    fn new() -> Self {
        let (tx, rx) = sync_channel::<RenderJob>(32);
        std::thread::Builder::new()
            .name("nexium-render".to_string())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("spawn nexium-render thread");
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
    if let Some(rt) = maybe_render_thread() {
        return rt;
    }
    static PT: OnceLock<RenderThread> = OnceLock::new();
    PT.get_or_init(RenderThread::new)
}

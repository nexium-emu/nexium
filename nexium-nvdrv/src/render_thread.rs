use std::sync::mpsc::{sync_channel, SyncSender};
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
}

fn async_render_enabled() -> bool {
    match std::env::var("NEXIUM_ASYNC_RENDER").ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => true,
        _ => false,
    }
}

pub fn maybe_render_thread() -> Option<&'static RenderThread> {
    static RT: OnceLock<Option<RenderThread>> = OnceLock::new();
    RT.get_or_init(|| {
        if async_render_enabled() {
            log::info!("nexium-nvdrv: async render thread ENABLED (NEXIUM_ASYNC_RENDER=1)");
            Some(RenderThread::new())
        } else {
            log::info!("nexium-nvdrv: async render thread DISABLED (default)");
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

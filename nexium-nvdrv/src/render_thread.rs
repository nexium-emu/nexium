use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Mutex, OnceLock};

pub type RenderJob = Box<dyn FnOnce() + Send + 'static>;

pub struct RenderThread {
    tx: SyncSender<RenderJob>,
    sent_textures: Mutex<std::collections::HashSet<u64>>,
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
        RenderThread {
            tx,
            sent_textures: Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub fn submit(&self, job: RenderJob) {
        let _ = self.tx.send(job);
    }

    pub fn first_sight_texture(&self, gpu_va: u64) -> bool {
        self.sent_textures.lock().unwrap().insert(gpu_va)
    }
}

pub fn maybe_render_thread() -> Option<&'static RenderThread> {
    static RT: OnceLock<Option<RenderThread>> = OnceLock::new();
    RT.get_or_init(|| {
        if std::env::var("NEXIUM_ASYNC_RENDER").ok().as_deref() == Some("0") {
            log::info!("nexium-nvdrv: async render thread DISABLED (NEXIUM_ASYNC_RENDER=0)");
            None
        } else {
            log::info!("nexium-nvdrv: async render thread ENABLED (default)");
            Some(RenderThread::new())
        }
    })
    .as_ref()
}

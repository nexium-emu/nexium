use std::sync::Arc;
use parking_lot::Mutex;

pub struct Nvdrv {
    pub gpu_ctx: Option<Arc<Mutex<Box<dyn std::any::Any + Send>>>>,
}

impl Nvdrv {
    pub fn new() -> Self {
        Self {
            gpu_ctx: None,
        }
    }

    pub fn set_gpu_context(&mut self, ctx: Arc<Mutex<Box<dyn std::any::Any + Send>>>) {
        self.gpu_ctx = Some(ctx);
        log::debug!("GPU context registered with nvdrv");
    }

    pub fn process_ioctl(&mut self, cmd: u32, _data: &[u8]) -> Vec<u8> {
        log::trace!("nvdrv ioctl: {:#x}", cmd);
        vec![]
    }
}

impl Default for Nvdrv {
    fn default() -> Self {
        Self::new()
    }
}

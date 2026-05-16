pub struct Nvdrv {
}

impl Nvdrv {
    pub fn new() -> Self {
        Self {}
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

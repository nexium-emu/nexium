#[allow(unused_imports)]
use nexium_cmif::{service, command};

pub struct PlService;

impl PlService {
    pub fn new() -> Self { Self }
}

impl Default for PlService {
    fn default() -> Self { Self::new() }
}

#[service]
impl PlService {
    #[command(0)]
    fn request_load(&mut self, _font_type: u32) -> Result<(), u32> {
        Ok(())
    }

    #[command(1)]
    fn get_load_state(&mut self, _font_type: u32) -> Result<u32, u32> {
        Ok(1)
    }

    #[command(2)]
    fn get_size(&mut self, _font_type: u32) -> Result<u32, u32> {
        Ok(0x100_000)
    }

    #[command(3)]
    fn get_shared_memory_address_offset(&mut self, _font_type: u32) -> Result<u32, u32> {
        Ok(0)
    }
}

impl PlService {
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("pl cmd: {} (legacy stub fallback)", cmd_id);
        0
    }
}

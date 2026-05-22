#[allow(unused_imports)]
use nexium_cmif::{service, command, RecvBuffer};

const LANGUAGE_CODE_EN_US: u64 = 0x0000_0053_552D_6E65;

pub struct SettingsService;

impl SettingsService {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SettingsService {
    fn default() -> Self {
        Self::new()
    }
}

#[service]
impl SettingsService {
    #[command(0)]
    fn get_language_code(&mut self) -> Result<u64, u32> {
        Ok(LANGUAGE_CODE_EN_US)
    }

    #[command(1)]
    fn get_available_language_codes_legacy(&mut self, codes: RecvBuffer) -> Result<i32, u32> {
        codes.write_all(&LANGUAGE_CODE_EN_US.to_le_bytes());
        Ok(1)
    }

    #[command(2)]
    fn make_language_code(&mut self, _index: i32) -> Result<u64, u32> {
        Ok(LANGUAGE_CODE_EN_US)
    }

    #[command(3)]
    fn get_available_language_code_count_legacy(&mut self) -> Result<u32, u32> {
        Ok(1)
    }

    #[command(5)]
    fn get_available_language_codes(&mut self, codes: RecvBuffer) -> Result<i32, u32> {
        codes.write_all(&LANGUAGE_CODE_EN_US.to_le_bytes());
        Ok(1)
    }

    #[command(6)]
    fn get_available_language_code_count(&mut self) -> Result<u32, u32> {
        Ok(1)
    }
}

impl SettingsService {
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("set cmd: {} (legacy stub fallback)", cmd_id);
        0
    }
}

#[allow(unused_imports)]
use nexium_cmif::{command, service, RecvBuffer};

const LANGUAGE_CODE_EN_US: u64 = 0x0000_0053_552D_6E65;
const LEGACY_LANGUAGE_CODE_LIMIT: usize = 0xF;
const LANGUAGE_CODE_LIMIT: usize = 0x40;

const AVAILABLE_LANGUAGE_CODES: [u64; 18] = [
    0x0000_0000_0000_616A,
    LANGUAGE_CODE_EN_US,
    0x0000_0000_0000_7266,
    0x0000_0000_0000_6564,
    0x0000_0000_0000_7469,
    0x0000_0000_0000_7365,
    0x0000_004E_432D_687A,
    0x0000_0000_0000_6F6B,
    0x0000_0000_0000_6C6E,
    0x0000_0000_0000_7470,
    0x0000_0000_0000_7572,
    0x0000_0057_542D_687A,
    0x0000_0042_472D_6E65,
    0x0000_0041_432D_7266,
    0x0000_3931_342D_7365,
    0x0073_6E61_482D_687A,
    0x0074_6E61_482D_687A,
    0x0000_0052_422D_7470,
];

fn write_available_language_codes(codes: &RecvBuffer<'_>, limit: usize) -> i32 {
    let count = (codes.size as usize / 8)
        .min(limit)
        .min(AVAILABLE_LANGUAGE_CODES.len());
    let mut data = Vec::with_capacity(count * 8);

    for code in AVAILABLE_LANGUAGE_CODES.iter().take(count) {
        data.extend_from_slice(&code.to_le_bytes());
    }

    (codes.write_all(&data) / 8) as i32
}

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
        Ok(write_available_language_codes(
            &codes,
            LEGACY_LANGUAGE_CODE_LIMIT,
        ))
    }

    #[command(2)]
    fn make_language_code(&mut self, index: i32) -> Result<u64, u32> {
        Ok(AVAILABLE_LANGUAGE_CODES
            .get(index as usize)
            .copied()
            .unwrap_or(LANGUAGE_CODE_EN_US))
    }

    #[command(3)]
    fn get_available_language_code_count_legacy(&mut self) -> Result<u32, u32> {
        Ok(LEGACY_LANGUAGE_CODE_LIMIT as u32)
    }

    #[command(5)]
    fn get_available_language_codes(&mut self, codes: RecvBuffer) -> Result<i32, u32> {
        Ok(write_available_language_codes(&codes, LANGUAGE_CODE_LIMIT))
    }

    #[command(6)]
    fn get_available_language_code_count(&mut self) -> Result<u32, u32> {
        Ok(LANGUAGE_CODE_LIMIT as u32)
    }
}

impl SettingsService {
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("set cmd: {} (legacy stub fallback)", cmd_id);
        0
    }
}

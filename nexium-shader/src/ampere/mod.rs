mod lower;
mod maxwell;
mod sm70;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub use lower::Error;

pub const HEADER_SIZE: usize = 0x80;
pub const MAXWELL_HEADER_SIZE: usize = 0x50;
pub const MAX_PROGRAM_BYTES: usize = HEADER_SIZE + 64 * 1024;
pub const MAX_LARGE_PROGRAM_BYTES: usize = HEADER_SIZE + 256 * 1024;

const VERSION_SHIFT: u32 = 5;
const VERSION_MASK: u32 = 0x1F << VERSION_SHIFT;
const MAXWELL_VERSION: u32 = 3;
const CACHE_LIMIT: usize = 4096;

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub header: [u8; MAXWELL_HEADER_SIZE],
    pub code: Vec<u8>,
}

pub fn header(binary: &[u8]) -> Option<[u8; MAXWELL_HEADER_SIZE]> {
    let mut header: [u8; MAXWELL_HEADER_SIZE] = binary.get(..MAXWELL_HEADER_SIZE)?.try_into().ok()?;
    let word = u32::from_le_bytes(header[..4].try_into().unwrap());
    let word = (word & !VERSION_MASK) | (MAXWELL_VERSION << VERSION_SHIFT);
    header[..4].copy_from_slice(&word.to_le_bytes());
    Some(header)
}

pub fn translate(binary: &[u8]) -> Result<Program, Error> {
    if binary.len() < HEADER_SIZE {
        return Err(Error::Truncated);
    }
    let header = header(binary).ok_or(Error::Truncated)?;
    let code = lower::lower(&binary[HEADER_SIZE..])?;
    Ok(Program { header, code })
}

fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn dump(address: u64, binary: &[u8], program: Option<&Program>) {
    let Some(directory) = std::env::var_os("NEXIUM_AMPERE_DUMP") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let _ = std::fs::write(directory.join(format!("{address:010x}.sm86")), binary);
    if let Some(program) = program {
        let mut lowered = program.header.to_vec();
        lowered.extend_from_slice(&program.code);
        let _ = std::fs::write(directory.join(format!("{address:010x}.sm50")), lowered);
    }
}

pub fn translate_cached(address: u64, binary: &[u8]) -> Result<Arc<Program>, Error> {
    static CACHE: OnceLock<Mutex<HashMap<(u64, u64), Result<Arc<Program>, Error>>>> = OnceLock::new();
    let program = &binary[..binary.len().min(lower::end_of_program(binary))];
    let key = (address, fingerprint(program));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = cache.lock().unwrap().get(&key) {
        return found.clone();
    }
    let result = translate(program).map(Arc::new);
    dump(address, program, result.as_deref().ok());
    if let Err(error) = &result {
        log::warn!("[ampere] shader at {address:#x} not translated: {error}");
    }
    let mut cache = cache.lock().unwrap();
    if cache.len() >= CACHE_LIMIT {
        cache.clear();
    }
    cache.insert(key, result.clone());
    result
}

#[cfg(test)]
mod tests;

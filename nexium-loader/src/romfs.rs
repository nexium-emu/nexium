use crate::bin_read::{i64at, u32at};

pub const HASH_TYPE_NONE: u8 = 1;
pub const HASH_TYPE_SHA256: u8 = 2;
pub const HASH_TYPE_INTEGRITY: u8 = 3;

pub const IVFC_MAGIC: u32 = 0x43465649;

const HASH_DATA_OFFSET: usize = 0x08;

pub fn fs_data_extent(fs_header: &[u8], hash_type: u8, section_len: u64) -> Result<(u64, u64), String> {
    match hash_type {
        HASH_TYPE_SHA256 => sha256_target(fs_header),
        HASH_TYPE_INTEGRITY => ivfc_target(fs_header),
        HASH_TYPE_NONE => Ok((0, section_len)),
        other => Err(format!("unsupported NCA hash type {}", other)),
    }
}

fn sha256_target(fs_header: &[u8]) -> Result<(u64, u64), String> {
    let count = u32at(fs_header, HASH_DATA_OFFSET + 0x24)? as usize;
    if count == 0 || count > 5 {
        return Err(format!("bad sha256 hash_layer_count {}", count));
    }
    let region = HASH_DATA_OFFSET + 0x28 + (count - 1) * 0x10;
    let offset = i64at(fs_header, region)? as u64;
    let size = i64at(fs_header, region + 8)? as u64;
    Ok((offset, size))
}

fn ivfc_target(fs_header: &[u8]) -> Result<(u64, u64), String> {
    let magic = u32at(fs_header, HASH_DATA_OFFSET)?;
    if magic != IVFC_MAGIC {
        return Err(format!("IVFC magic {:#010x} mismatch", magic));
    }
    let max_layers = u32at(fs_header, HASH_DATA_OFFSET + 0x0C)? as usize;
    if max_layers < 2 || max_layers > 7 {
        return Err(format!("bad IVFC max_layers {}", max_layers));
    }
    let info = HASH_DATA_OFFSET + 0x10;
    let data = info + (max_layers - 2) * 0x18;
    let offset = i64at(fs_header, data)? as u64;
    let size = i64at(fs_header, data + 8)? as u64;
    Ok((offset, size))
}

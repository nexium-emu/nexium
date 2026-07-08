use crate::bin_read::{i64at, u32at};

pub const HASH_TYPE_NONE: u8 = 1;
pub const HASH_TYPE_SHA256: u8 = 2;
pub const HASH_TYPE_INTEGRITY: u8 = 3;

pub const IVFC_MAGIC: u32 = 0x43465649;

const HASH_DATA_OFFSET: usize = 0x08;

pub fn fs_data_extent(
    fs_header: &[u8],
    hash_type: u8,
    section_len: u64,
) -> Result<(u64, u64), String> {
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

fn u32le(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4)
        .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]))
}

fn u64le(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o + 8)
        .map(|v| u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]))
}

#[derive(Clone, Copy)]
pub struct RomfsHeader {
    pub dir_meta_off: usize,
    pub file_meta_off: usize,
    pub file_data_off: usize,
}

pub fn romfs_header(romfs: &[u8]) -> Option<RomfsHeader> {
    if u64le(romfs, 0)? != 0x50 {
        return None;
    }
    Some(RomfsHeader {
        dir_meta_off: u64le(romfs, 0x18)? as usize,
        file_meta_off: u64le(romfs, 0x38)? as usize,
        file_data_off: u64le(romfs, 0x48)? as usize,
    })
}

fn romfs_name(romfs: &[u8], entry_abs: usize, name_off: usize, name_len_off: usize) -> Option<&str> {
    let name_len = u32le(romfs, entry_abs + name_len_off)? as usize;
    let start = entry_abs + name_off;
    let end = start.checked_add(name_len)?;
    std::str::from_utf8(romfs.get(start..end)?).ok()
}

fn find_child_dir(romfs: &[u8], hdr: RomfsHeader, dir_off: u32, name: &str) -> Option<u32> {
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut child = u32le(romfs, dir_abs + 0x08)?;
    while child != 0xffff_ffff {
        let abs = hdr.dir_meta_off + child as usize;
        if romfs_name(romfs, abs, 0x18, 0x14)? == name {
            return Some(child);
        }
        child = u32le(romfs, abs + 0x04)?;
    }
    None
}

pub fn find_child_file(
    romfs: &[u8],
    hdr: RomfsHeader,
    dir_off: u32,
    name: &str,
) -> Option<(usize, usize)> {
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut child = u32le(romfs, dir_abs + 0x0c)?;
    while child != 0xffff_ffff {
        let abs = hdr.file_meta_off + child as usize;
        if romfs_name(romfs, abs, 0x20, 0x1c)? == name {
            let rel = u64le(romfs, abs + 0x08)? as usize;
            let size = u64le(romfs, abs + 0x10)? as usize;
            return Some((hdr.file_data_off + rel, size));
        }
        child = u32le(romfs, abs + 0x04)?;
    }
    None
}

pub fn romfs_file<'a>(romfs: &'a [u8], path: &str) -> Option<&'a [u8]> {
    let hdr = romfs_header(romfs)?;
    let parts = path
        .trim_start_matches('/')
        .split('/')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>();
    let (file, dirs) = parts.split_last()?;
    let mut dir = 0u32;
    for part in dirs {
        dir = find_child_dir(romfs, hdr, dir, part)?;
    }
    let (off, size) = find_child_file(romfs, hdr, dir, file)?;
    romfs.get(off..off.checked_add(size)?)
}

pub fn romfs_dir_files(romfs: &[u8], hdr: RomfsHeader, dir_off: u32) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let dir_abs = hdr.dir_meta_off + dir_off as usize;
    let mut file = match u32le(romfs, dir_abs + 0x0c) {
        Some(v) => v,
        None => return out,
    };
    while file != 0xffff_ffff {
        let abs = hdr.file_meta_off + file as usize;
        if let Some(name) = romfs_name(romfs, abs, 0x20, 0x1c) {
            let rel = u64le(romfs, abs + 0x08).unwrap_or(0) as usize;
            let size = u64le(romfs, abs + 0x10).unwrap_or(0) as usize;
            out.push((name.to_string(), hdr.file_data_off + rel, size));
        }
        file = match u32le(romfs, abs + 0x04) {
            Some(v) => v,
            None => break,
        };
    }
    out
}

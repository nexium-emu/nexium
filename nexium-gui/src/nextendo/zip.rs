use std::io::Read;

const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4B50;
const CENTRAL_FILE_HEADER: u32 = 0x0201_4B50;
const LOCAL_FILE_HEADER: u32 = 0x0403_4B50;
const METHOD_STORED: u16 = 0;
const METHOD_DEFLATE: u16 = 8;
const FLAG_ENCRYPTED: u16 = 1;
const UNIX_SYMLINK: u32 = 0xA000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub data: Vec<u8>,
}

fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, String> {
    bytes
        .get(offset..offset + 2)
        .map(|slice| u16::from_le_bytes([slice[0], slice[1]]))
        .ok_or_else(|| "truncated zip".to_string())
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, String> {
    bytes
        .get(offset..offset + 4)
        .map(|slice| u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
        .ok_or_else(|| "truncated zip".to_string())
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn find_end_record(bytes: &[u8]) -> Result<usize, String> {
    let minimum = 22;
    if bytes.len() < minimum {
        return Err("zip too small".into());
    }
    let start = bytes.len().saturating_sub(minimum + 0xFFFF);
    (start..=bytes.len() - minimum)
        .rev()
        .find(|&offset| u32_at(bytes, offset).ok() == Some(END_OF_CENTRAL_DIRECTORY))
        .ok_or_else(|| "zip end record not found".to_string())
}

pub fn extract(bytes: &[u8], max_entries: usize, max_total: usize) -> Result<Vec<Entry>, String> {
    let end = find_end_record(bytes)?;
    if u16_at(bytes, end + 4)? != 0 || u16_at(bytes, end + 6)? != 0 {
        return Err("multi-disk zip".into());
    }
    let count = usize::from(u16_at(bytes, end + 10)?);
    let directory_size = u32_at(bytes, end + 12)? as usize;
    let mut offset = u32_at(bytes, end + 16)? as usize;
    if count > max_entries {
        return Err(format!("zip has {count} entries, more than {max_entries}"));
    }
    if offset.checked_add(directory_size).is_none_or(|end_of_directory| end_of_directory > end) {
        return Err("zip central directory out of bounds".into());
    }
    let mut entries = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        if u32_at(bytes, offset)? != CENTRAL_FILE_HEADER {
            return Err("bad central directory entry".into());
        }
        let flags = u16_at(bytes, offset + 8)?;
        let method = u16_at(bytes, offset + 10)?;
        let crc = u32_at(bytes, offset + 16)?;
        let compressed = u32_at(bytes, offset + 20)? as usize;
        let uncompressed = u32_at(bytes, offset + 24)? as usize;
        let name_len = usize::from(u16_at(bytes, offset + 28)?);
        let extra_len = usize::from(u16_at(bytes, offset + 30)?);
        let comment_len = usize::from(u16_at(bytes, offset + 32)?);
        let external = u32_at(bytes, offset + 38)?;
        let local = u32_at(bytes, offset + 42)? as usize;
        let name = bytes
            .get(offset + 46..offset + 46 + name_len)
            .ok_or("truncated zip name")?;
        let name = std::str::from_utf8(name).map_err(|_| "zip name is not UTF-8")?.to_string();
        offset += 46 + name_len + extra_len + comment_len;
        if flags & FLAG_ENCRYPTED != 0 {
            return Err(format!("{name}: encrypted"));
        }
        if compressed == 0xFFFF_FFFF || uncompressed == 0xFFFF_FFFF {
            return Err(format!("{name}: zip64 is not supported"));
        }
        if (external >> 16) & 0xF000 == UNIX_SYMLINK {
            return Err(format!("{name}: symbolic link"));
        }
        if name.ends_with('/') {
            if uncompressed != 0 {
                return Err(format!("{name}: directory with data"));
            }
            continue;
        }
        total = total
            .checked_add(uncompressed)
            .filter(|total| *total <= max_total)
            .ok_or_else(|| format!("zip expands past {max_total} bytes"))?;
        if u32_at(bytes, local)? != LOCAL_FILE_HEADER {
            return Err(format!("{name}: bad local header"));
        }
        let local_name_len = usize::from(u16_at(bytes, local + 26)?);
        let local_extra_len = usize::from(u16_at(bytes, local + 28)?);
        let start = local + 30 + local_name_len + local_extra_len;
        let raw = bytes
            .get(start..start + compressed)
            .ok_or_else(|| format!("{name}: truncated data"))?;
        let data = match method {
            METHOD_STORED => raw.to_vec(),
            METHOD_DEFLATE => {
                let mut out = Vec::with_capacity(uncompressed);
                flate2::read::DeflateDecoder::new(raw)
                    .take(uncompressed as u64 + 1)
                    .read_to_end(&mut out)
                    .map_err(|error| format!("{name}: inflate failed: {error}"))?;
                out
            }
            other => return Err(format!("{name}: compression method {other} is not supported")),
        };
        if data.len() != uncompressed {
            return Err(format!("{name}: size mismatch"));
        }
        if crc32(&data) != crc {
            return Err(format!("{name}: checksum mismatch"));
        }
        entries.push(Entry { name, data });
    }
    Ok(entries)
}

#[cfg(test)]
pub fn build(entries: &[(&str, &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut out = Vec::new();
    let mut directory = Vec::new();
    for (index, (name, data)) in entries.iter().enumerate() {
        let deflate = index % 2 == 1;
        let (method, payload) = if deflate {
            let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(data).unwrap();
            (METHOD_DEFLATE, encoder.finish().unwrap())
        } else {
            (METHOD_STORED, data.to_vec())
        };
        let local = out.len() as u32;
        let crc = crc32(data);
        out.extend_from_slice(&LOCAL_FILE_HEADER.to_le_bytes());
        out.extend_from_slice(&[20, 0, 0, 0]);
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&payload);
        directory.extend_from_slice(&CENTRAL_FILE_HEADER.to_le_bytes());
        directory.extend_from_slice(&[20, 0, 20, 0, 0, 0]);
        directory.extend_from_slice(&method.to_le_bytes());
        directory.extend_from_slice(&[0u8; 4]);
        directory.extend_from_slice(&crc.to_le_bytes());
        directory.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(data.len() as u32).to_le_bytes());
        directory.extend_from_slice(&(name.len() as u16).to_le_bytes());
        directory.extend_from_slice(&[0u8; 8]);
        directory.extend_from_slice(&[0u8; 4]);
        directory.extend_from_slice(&local.to_le_bytes());
        directory.extend_from_slice(name.as_bytes());
    }
    let directory_offset = out.len() as u32;
    out.extend_from_slice(&directory);
    out.extend_from_slice(&END_OF_CENTRAL_DIRECTORY.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(directory.len() as u32).to_le_bytes());
    out.extend_from_slice(&directory_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_reference_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn stored_and_deflated_entries_round_trip() {
        let big: Vec<u8> = (0..20_000u32).map(|value| (value % 251) as u8).collect();
        let zip = build(&[("vsdata/VSSetting_0.byaml", b"hello"), ("coopdata/CoopSetting.byaml", &big), ("fesdata/", b"")]);
        let entries = extract(&zip, 512, 1 << 20).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "vsdata/VSSetting_0.byaml");
        assert_eq!(entries[0].data, b"hello");
        assert_eq!(entries[1].name, "coopdata/CoopSetting.byaml");
        assert_eq!(entries[1].data, big);
    }

    #[test]
    fn limits_and_corruption_are_rejected() {
        let zip = build(&[("a/b", b"one"), ("c/d", b"two")]);
        assert!(extract(&zip, 1, 1 << 20).unwrap_err().contains("entries"));
        assert!(extract(&zip, 8, 4).unwrap_err().contains("expands"));
        let mut damaged = zip.clone();
        let payload = damaged.windows(3).position(|window| window == b"one").unwrap();
        damaged[payload] = b'x';
        assert!(extract(&damaged, 8, 1 << 20).unwrap_err().contains("checksum"));
        assert!(extract(b"not a zip at all, definitely not one", 8, 1 << 20).is_err());
        assert!(extract(&zip[..zip.len() - 1], 8, 1 << 20).is_err());
    }
}

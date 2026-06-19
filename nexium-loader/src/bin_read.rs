use std::ops::Range;

pub fn u8at(b: &[u8], o: usize) -> Result<u8, String> {
    b.get(o).copied().ok_or_else(|| format!("u8 OOB at {:#x} (len {:#x})", o, b.len()))
}

pub fn u16at(b: &[u8], o: usize) -> Result<u16, String> {
    let s = b.get(o..o + 2).ok_or_else(|| format!("u16 OOB at {:#x} (len {:#x})", o, b.len()))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

pub fn u32at(b: &[u8], o: usize) -> Result<u32, String> {
    let s = b.get(o..o + 4).ok_or_else(|| format!("u32 OOB at {:#x} (len {:#x})", o, b.len()))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

pub fn u48at(b: &[u8], o: usize) -> Result<u64, String> {
    let s = b.get(o..o + 6).ok_or_else(|| format!("u48 OOB at {:#x} (len {:#x})", o, b.len()))?;
    Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], 0, 0]))
}

pub fn u64at(b: &[u8], o: usize) -> Result<u64, String> {
    let s = b.get(o..o + 8).ok_or_else(|| format!("u64 OOB at {:#x} (len {:#x})", o, b.len()))?;
    Ok(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
}

pub fn i64at(b: &[u8], o: usize) -> Result<i64, String> {
    Ok(u64at(b, o)? as i64)
}

pub fn slice<'a>(b: &'a [u8], range: Range<usize>) -> Result<&'a [u8], String> {
    if range.start > range.end || range.end > b.len() {
        return Err(format!("slice {:#x}..{:#x} out of range (len {:#x})", range.start, range.end, b.len()));
    }
    Ok(&b[range])
}

pub fn checked_range(start: u64, len: u64, limit: usize) -> Result<Range<usize>, String> {
    let end = start.checked_add(len).ok_or_else(|| format!("range overflow start={:#x} len={:#x}", start, len))?;
    if end > limit as u64 {
        return Err(format!("range {:#x}..{:#x} exceeds limit {:#x}", start, end, limit));
    }
    Ok(start as usize..end as usize)
}

use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

pub type Check = fn() -> Result<String, String>;

pub const CHECKS: &[(&str, Check)] = &[("filemap-demand-paging", filemap_demand_paging)];

const TEST_FILE: &str = "/app0/data/games/celeste.dxci";

fn filemap_demand_paging() -> Result<String, String> {
    let Ok(file) = std::fs::File::open(TEST_FILE) else {
        return Ok(format!("skipped: {TEST_FILE} absent"));
    };
    let len = file.metadata().map_err(|e| e.to_string())?.len() as usize;
    crate::filemap::set_cap_mib(256);
    let started = Instant::now();
    let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("Mmap::map: {e}"))?;
    let map_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut reader = std::fs::File::open(TEST_FILE).map_err(|e| e.to_string())?;
    let mut mismatches = 0;
    let mut checked = 0;
    let step = (len / 997).max(1);
    let mut buf = [0u8; 64];
    let mut offset = 7usize;
    while offset + 64 <= len {
        reader.seek(SeekFrom::Start(offset as u64)).map_err(|e| e.to_string())?;
        reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
        if map[offset..offset + 64] != buf {
            mismatches += 1;
        }
        checked += 1;
        offset += step;
    }
    let tail: Vec<u8> = map[len - 16..].to_vec();
    reader.seek(SeekFrom::Start((len - 16) as u64)).map_err(|e| e.to_string())?;
    let mut tail_buf = [0u8; 16];
    reader.read_exact(&mut tail_buf).map_err(|e| e.to_string())?;
    let random_ms = started.elapsed().as_secs_f64() * 1000.0 - map_ms;
    let seq_started = Instant::now();
    let mut sum = 0u64;
    let mut pos = 0usize;
    while pos < len {
        sum = sum.wrapping_add(map[pos] as u64);
        pos += 4096;
    }
    let seq_secs = seq_started.elapsed().as_secs_f64();
    let stats = crate::filemap::stats();
    drop(map);
    let after = crate::filemap::stats();
    crate::filemap::set_cap_mib(1536);
    let detail = format!(
        "{} MiB file mapped in {map_ms:.1}ms; {checked} random samples ({mismatches} mismatches) + tail ok={} in {random_ms:.0}ms; sequential touch {:.0} MiB/s (sum {sum}); resident {} MiB cap {} MiB loads {} evictions {} avg load {:.0}us; after unmap resident {} MiB regions {}",
        len >> 20,
        tail[..] == tail_buf[..],
        (len as f64 / (1 << 20) as f64) / seq_secs,
        stats.resident_mib,
        stats.cap_mib,
        stats.loads,
        stats.evictions,
        stats.avg_load_us,
        after.resident_mib,
        after.regions
    );
    if mismatches != 0 || tail[..] != tail_buf[..] || stats.resident_mib > stats.cap_mib + 2 || after.resident_mib != 0 {
        return Err(detail);
    }
    Ok(detail)
}

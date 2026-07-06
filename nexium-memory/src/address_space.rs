use bytemuck::{NoUninit, Pod};
use parking_lot::Mutex;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

use crate::perm::Perm;
use crate::region::{page_align_down, page_align_up, PAGE_MASK};

#[derive(Debug, Error)]
pub enum AddressSpaceError {
    #[error("range va={va:#x} len={len:#x} overlaps existing region '{name}' ({existing_base:#x}..{existing_end:#x})")]
    Overlap {
        va: u64,
        len: u64,
        name: String,
        existing_base: u64,
        existing_end: u64,
    },
    #[error("{what} {value:#x} is not page-aligned")]
    Unaligned { what: &'static str, value: u64 },
    #[error("zero-length mapping at {va:#x}")]
    ZeroLength { va: u64 },
    #[error("address range va={va:#x} len={len:#x} is unmapped")]
    Unmapped { va: u64, len: usize },
    #[error("permission denied at va={va:#x}: have {have} need {need}")]
    PermissionDenied { va: u64, have: Perm, need: Perm },
    #[error("integer overflow computing va={va:#x} + len={len:#x}")]
    Overflow { va: u64, len: u64 },
}

pub type Result<T> = core::result::Result<T, AddressSpaceError>;

struct Region {
    base: u64,
    buf: NonNull<u8>,
    len: usize,
    perm: Mutex<Perm>,
    name: String,
    arena: bool,
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    fn new(base: u64, len: usize, perm: Perm, name: String) -> Self {
        if let Some(ptr) = crate::fastmem::commit(base, len) {
            let buf = unsafe { NonNull::new_unchecked(ptr) };
            return Self {
                base,
                buf,
                len,
                perm: Mutex::new(perm),
                name,
                arena: true,
            };
        }
        let boxed: Box<[u8]> = vec![0u8; len].into_boxed_slice();
        let raw = Box::into_raw(boxed);
        let buf = unsafe { NonNull::new_unchecked(raw as *mut u8) };
        Self {
            base,
            buf,
            len,
            perm: Mutex::new(perm),
            name,
            arena: false,
        }
    }

    #[inline]
    fn end(&self) -> u64 {
        self.base + self.len as u64
    }

    #[inline]
    fn contains(&self, va: u64) -> bool {
        va >= self.base && va < self.end()
    }

    #[inline]
    fn perm(&self) -> Perm {
        *self.perm.lock()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if self.arena {
            crate::fastmem::decommit(self.buf.as_ptr(), self.len);
            return;
        }
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.buf.as_ptr(),
                self.len,
            ));
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegionInfo {
    pub base: u64,
    pub size: u64,
    pub perm: Perm,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct HostRegion {
    pub base: u64,
    pub size: u64,
    pub perm: Perm,
    pub host_ptr: *mut u8,
}

unsafe impl Send for HostRegion {}
unsafe impl Sync for HostRegion {}

pub struct AddressSpace {
    regions: Mutex<Vec<Arc<Region>>>,
    generation: AtomicU64,
}

impl AddressSpace {
    pub fn new() -> Self {
        Self {
            regions: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn map(&self, va: u64, len: u64, perm: Perm, name: impl Into<String>) -> Result<usize> {
        check_aligned("va", va)?;
        check_aligned("len", len)?;
        if len == 0 {
            return Err(AddressSpaceError::ZeroLength { va });
        }
        let end = va
            .checked_add(len)
            .ok_or(AddressSpaceError::Overflow { va, len })?;

        let mut regs = self.regions.lock();
        let insert_at = match regs.binary_search_by_key(&va, |r| r.base) {
            Ok(_) => {
                let r = &regs[regs.binary_search_by_key(&va, |r| r.base).unwrap()];
                return Err(AddressSpaceError::Overlap {
                    va,
                    len,
                    name: r.name.clone(),
                    existing_base: r.base,
                    existing_end: r.end(),
                });
            }
            Err(idx) => idx,
        };
        if let Some(prev) = insert_at.checked_sub(1).and_then(|i| regs.get(i)) {
            if prev.end() > va {
                return Err(AddressSpaceError::Overlap {
                    va,
                    len,
                    name: prev.name.clone(),
                    existing_base: prev.base,
                    existing_end: prev.end(),
                });
            }
        }
        if let Some(next) = regs.get(insert_at) {
            if next.base < end {
                return Err(AddressSpaceError::Overlap {
                    va,
                    len,
                    name: next.name.clone(),
                    existing_base: next.base,
                    existing_end: next.end(),
                });
            }
        }

        let region = Arc::new(Region::new(va, len as usize, perm, name.into()));
        log::debug!(
            "map va={va:#x} len={len:#x} perm={perm} name={}",
            region.name
        );
        regs.insert(insert_at, region);
        self.generation.fetch_add(1, Ordering::Release);
        Ok(insert_at)
    }

    pub fn protect(&self, va: u64, len: u64, perm: Perm) -> Result<()> {
        let regs = self.regions.lock();
        let r = regs
            .iter()
            .find(|r| r.base == va && r.len as u64 == len)
            .ok_or(AddressSpaceError::Unmapped {
                va,
                len: len as usize,
            })?;
        let mut p = r.perm.lock();
        log::debug!(
            "protect va={va:#x} len={len:#x} {} -> {} name={}",
            *p,
            perm,
            r.name
        );
        *p = perm;
        Ok(())
    }

    pub fn read(&self, va: u64, buf: &mut [u8]) -> Result<()> {
        let (region, off) = self.locate(va, buf.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                region.buf.as_ptr().add(off),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        Ok(())
    }

    pub fn write(&self, va: u64, buf: &[u8]) -> Result<()> {
        let (region, off) = self.locate(va, buf.len())?;
        trace_host_write(&region, va, off, buf);
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), region.buf.as_ptr().add(off), buf.len());
        }
        Ok(())
    }

    pub fn read_pod<T: Pod>(&self, va: u64) -> Result<T> {
        let mut value = T::zeroed();
        let bytes: &mut [u8] = bytemuck::bytes_of_mut(&mut value);
        self.read(va, bytes)?;
        Ok(value)
    }

    pub fn write_pod<T: NoUninit>(&self, va: u64, value: T) -> Result<()> {
        self.write(va, bytemuck::bytes_of(&value))
    }

    pub fn read_checked(&self, va: u64, buf: &mut [u8]) -> Result<()> {
        let (region, off) = self.locate(va, buf.len())?;
        if !region.perm().contains(Perm::R) {
            return Err(AddressSpaceError::PermissionDenied {
                va,
                have: region.perm(),
                need: Perm::R,
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                region.buf.as_ptr().add(off),
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        Ok(())
    }

    pub fn write_checked(&self, va: u64, buf: &[u8]) -> Result<()> {
        let (region, off) = self.locate(va, buf.len())?;
        if !region.perm().contains(Perm::W) {
            return Err(AddressSpaceError::PermissionDenied {
                va,
                have: region.perm(),
                need: Perm::W,
            });
        }
        trace_host_write(&region, va, off, buf);
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), region.buf.as_ptr().add(off), buf.len());
        }
        Ok(())
    }

    pub fn regions(&self) -> Vec<RegionInfo> {
        let regs = self.regions.lock();
        regs.iter()
            .map(|r| RegionInfo {
                base: r.base,
                size: r.len as u64,
                perm: r.perm(),
                name: r.name.clone(),
            })
            .collect()
    }

    pub fn host_regions(&self) -> Vec<HostRegion> {
        let regs = self.regions.lock();
        regs.iter()
            .map(|r| HostRegion {
                base: r.base,
                size: r.len as u64,
                perm: r.perm(),
                host_ptr: r.buf.as_ptr(),
            })
            .collect()
    }

    pub fn host_region_at(&self, va: u64) -> Option<HostRegion> {
        let regs = self.regions.lock();
        regs.iter().find(|r| r.base == va).map(|r| HostRegion {
            base: r.base,
            size: r.len as u64,
            perm: r.perm(),
            host_ptr: r.buf.as_ptr(),
        })
    }

    fn locate(&self, va: u64, len: usize) -> Result<(Arc<Region>, usize)> {
        let regs = self.regions.lock();
        let region = regs
            .iter()
            .find(|r| r.contains(va))
            .cloned()
            .ok_or(AddressSpaceError::Unmapped { va, len })?;
        let off = (va - region.base) as usize;
        if off
            .checked_add(len)
            .map(|end| end > region.len)
            .unwrap_or(true)
        {
            return Err(AddressSpaceError::Unmapped { va, len });
        }
        Ok((region, off))
    }
}

impl Default for AddressSpace {
    fn default() -> Self {
        Self::new()
    }
}

fn check_aligned(what: &'static str, value: u64) -> Result<()> {
    if value & PAGE_MASK != 0 {
        return Err(AddressSpaceError::Unaligned { what, value });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct HostWriteWatch {
    va: u64,
    len: u64,
}

fn trace_host_write(region: &Region, va: u64, off: usize, buf: &[u8]) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    static WATCH: OnceLock<Option<HostWriteWatch>> = OnceLock::new();
    static HITS: AtomicU64 = AtomicU64::new(0);

    let Some(watch) = *WATCH.get_or_init(parse_host_write_watch) else {
        return;
    };
    let write_end = va.saturating_add(buf.len() as u64);
    let watch_end = watch.va.saturating_add(watch.len);
    if va >= watch_end || write_end <= watch.va {
        return;
    }
    let hit = HITS.fetch_add(1, Ordering::Relaxed);
    let cap = std::env::var("NEXIUM_HOST_WRITE_WATCH_LIMIT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(64);
    if hit >= cap {
        return;
    }
    let overlap_start = va.max(watch.va);
    let overlap_end = write_end.min(watch_end);
    let src_off = (overlap_start - va) as usize;
    let n = (overlap_end - overlap_start).min(128) as usize;
    let incoming = &buf[src_off..src_off + n];
    let old_off = off + src_off;
    let old = unsafe { std::slice::from_raw_parts(region.buf.as_ptr().add(old_off), n) };
    log::warn!(
        "[host-write-watch] #{} va={:#x} len={:#x} overlap={:#x}..{:#x} region={} old={} new={} floats={}",
        hit + 1,
        va,
        buf.len(),
        overlap_start,
        overlap_end,
        region.name,
        hex_preview(old),
        hex_preview(incoming),
        float_preview(incoming)
    );
    if std::env::var("NEXIUM_HOST_WRITE_BACKTRACE").is_ok() {
        log::warn!(
            "[host-write-watch] backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
}

fn parse_host_write_watch() -> Option<HostWriteWatch> {
    let spec = std::env::var("NEXIUM_HOST_WRITE_WATCH")
        .ok()
        .or_else(|| std::env::var("NEXIUM_WATCH_WRITE_CPU").ok())?;
    let (va, len) = spec.trim().split_once(':')?;
    let va = parse_u64ish(va.trim())?;
    let len = parse_u64ish(len.trim()).unwrap_or(0x80);
    (va != 0 && len != 0).then_some(HostWriteWatch { va, len })
}

fn parse_u64ish(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

fn hex_preview(buf: &[u8]) -> String {
    buf.iter()
        .take(64)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ")
}

fn float_preview(buf: &[u8]) -> String {
    buf.chunks_exact(4)
        .take(16)
        .map(|c| {
            let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            format!("{:.3}", v)
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub fn align_request(base: u64, size: u64) -> (u64, u64) {
    let aligned_base = page_align_down(base);
    let end = page_align_up(base + size);
    (aligned_base, end - aligned_base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::PAGE_SIZE;

    fn fresh() -> AddressSpace {
        AddressSpace::new()
    }

    #[test]
    fn map_then_read_write_round_trip() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE, Perm::RW, "test").unwrap();
        a.write(0x1_0000, &[1, 2, 3, 4]).unwrap();
        let mut buf = [0u8; 4];
        a.read(0x1_0000, &mut buf).unwrap();
        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn write_spans_page_boundary_within_region() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE * 4, Perm::RW, "multi").unwrap();
        let payload: Vec<u8> = (0..=255u8).cycle().take(256).collect();
        let off = PAGE_SIZE - 16;
        a.write(0x1_0000 + off, &payload).unwrap();
        let mut buf = vec![0u8; 256];
        a.read(0x1_0000 + off, &mut buf).unwrap();
        assert_eq!(buf, payload);
    }

    #[test]
    fn pod_round_trip() {
        let a = fresh();
        a.map(0x2_0000, PAGE_SIZE, Perm::RW, "pod").unwrap();
        a.write_pod::<u64>(0x2_0040, 0xDEAD_BEEF_CAFE_BABE).unwrap();
        let v: u64 = a.read_pod(0x2_0040).unwrap();
        assert_eq!(v, 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn unmapped_read_errors() {
        let a = fresh();
        let mut buf = [0u8; 8];
        assert!(matches!(
            a.read(0xDEAD_0000, &mut buf),
            Err(AddressSpaceError::Unmapped { .. })
        ));
    }

    #[test]
    fn read_crossing_region_boundary_fails() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE, Perm::RW, "first").unwrap();
        a.map(0x1_0000 + PAGE_SIZE, PAGE_SIZE, Perm::RW, "second")
            .unwrap();
        let mut buf = [0u8; 8];
        let err = a.read(0x1_0000 + PAGE_SIZE - 4, &mut buf).unwrap_err();
        assert!(matches!(err, AddressSpaceError::Unmapped { .. }));
    }

    #[test]
    fn overlap_rejected() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE * 2, Perm::RW, "first").unwrap();
        let err = a
            .map(0x1_0000 + PAGE_SIZE, PAGE_SIZE, Perm::RW, "overlap")
            .unwrap_err();
        assert!(matches!(err, AddressSpaceError::Overlap { .. }));
    }

    #[test]
    fn unaligned_map_rejected() {
        let a = fresh();
        let err = a.map(0x1_0001, PAGE_SIZE, Perm::RW, "x").unwrap_err();
        assert!(matches!(
            err,
            AddressSpaceError::Unaligned { what: "va", .. }
        ));
        let err = a.map(0x1_0000, 0x800, Perm::RW, "x").unwrap_err();
        assert!(matches!(
            err,
            AddressSpaceError::Unaligned { what: "len", .. }
        ));
    }

    #[test]
    fn protect_changes_perm() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE, Perm::RW, "relro").unwrap();
        a.protect(0x1_0000, PAGE_SIZE, Perm::RO).unwrap();
        let regions = a.regions();
        assert_eq!(regions[0].perm, Perm::RO);
    }

    #[test]
    fn checked_read_respects_perm() {
        let a = fresh();
        a.map(0x1_0000, PAGE_SIZE, Perm::RW, "rw").unwrap();
        a.protect(0x1_0000, PAGE_SIZE, Perm::X).unwrap();
        let mut buf = [0u8; 4];
        let err = a.read_checked(0x1_0000, &mut buf).unwrap_err();
        assert!(matches!(err, AddressSpaceError::PermissionDenied { .. }));
    }

    #[test]
    fn three_disjoint_regions_coexist() {
        let a = fresh();
        a.map(crate::region::CODE_BASE, PAGE_SIZE * 16, Perm::RX, "code")
            .unwrap();
        a.map(crate::region::HEAP_BASE, PAGE_SIZE * 32, Perm::RW, "heap")
            .unwrap();
        a.map(crate::region::STACK_BASE, PAGE_SIZE * 8, Perm::RW, "stack")
            .unwrap();
        assert_eq!(a.regions().len(), 3);
    }

    #[test]
    fn shared_across_threads() {
        let a = Arc::new(fresh());
        a.map(0x1_0000, PAGE_SIZE * 4, Perm::RW, "shared").unwrap();
        let mut handles = Vec::new();
        for t in 0..4u8 {
            let a = a.clone();
            handles.push(std::thread::spawn(move || {
                let off = (t as u64) * 64;
                let payload = [t; 64];
                a.write(0x1_0000 + off, &payload).unwrap();
                let mut buf = [0u8; 64];
                a.read(0x1_0000 + off, &mut buf).unwrap();
                assert_eq!(buf, payload);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}

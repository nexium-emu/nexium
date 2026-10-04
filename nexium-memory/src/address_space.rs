use bytemuck::{NoUninit, Pod};
use parking_lot::Mutex;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
    #[error("failed to commit va={va:#x} len={len:#x}")]
    CommitFailed { va: u64, len: u64 },
    #[error("address range va={va:#x} len={len:#x} does not match a shared alias")]
    AliasMismatch { va: u64, len: u64 },
    #[error("address range va={va:#x} len={len:#x} has active shared aliases")]
    AliasInUse { va: u64, len: u64 },
}

pub type Result<T> = core::result::Result<T, AddressSpaceError>;

struct AliasBacking {
    source: Arc<Region>,
    offset: usize,
}

struct Region {
    base: u64,
    buf: NonNull<u8>,
    len: usize,
    committed_len: AtomicUsize,
    perm: Mutex<Perm>,
    name: String,
    arena: bool,
    alias_backing: Option<AliasBacking>,
    alias_users: AtomicUsize,
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
                committed_len: AtomicUsize::new(len),
                perm: Mutex::new(perm),
                name,
                arena: true,
                alias_backing: None,
                alias_users: AtomicUsize::new(0),
            };
        }
        let boxed: Box<[u8]> = vec![0u8; len].into_boxed_slice();
        let raw = Box::into_raw(boxed);
        let buf = unsafe { NonNull::new_unchecked(raw as *mut u8) };
        #[cfg(target_vendor = "sony")]
        crate::soft_watch::register(base, buf.as_ptr(), len, true);
        Self {
            base,
            buf,
            len,
            committed_len: AtomicUsize::new(len),
            perm: Mutex::new(perm),
            name,
            arena: false,
            alias_backing: None,
            alias_users: AtomicUsize::new(0),
        }
    }

    fn reserved(base: u64, len: usize, perm: Perm, name: String) -> Self {
        let (window_lo, window_hi) = crate::fastmem::va_window();
        if base >= window_lo
            && base
                .checked_add(len as u64)
                .is_some_and(|end| end <= window_hi)
        {
            if let Some(ptr) = crate::fastmem::host_ptr(base) {
                let buf = unsafe { NonNull::new_unchecked(ptr) };
                return Self {
                    base,
                    buf,
                    len,
                    committed_len: AtomicUsize::new(0),
                    perm: Mutex::new(perm),
                    name,
                    arena: true,
                alias_backing: None,
                alias_users: AtomicUsize::new(0),
                };
            }
        }
        let boxed: Box<[u8]> = vec![0u8; len].into_boxed_slice();
        let raw = Box::into_raw(boxed);
        let buf = unsafe { NonNull::new_unchecked(raw as *mut u8) };
        #[cfg(target_vendor = "sony")]
        crate::soft_watch::register(base, buf.as_ptr(), len, true);
        Self {
            base,
            buf,
            len,
            committed_len: AtomicUsize::new(0),
            perm: Mutex::new(perm),
            name,
            arena: false,
            alias_backing: None,
            alias_users: AtomicUsize::new(0),
        }
    }

    fn alias(base: u64, source: &Arc<Region>, offset: usize, len: usize, perm: Perm, name: String) -> Self {
        let (source, offset) = match &source.alias_backing {
            Some(backing) => (Arc::clone(&backing.source), backing.offset + offset),
            None => (Arc::clone(source), offset),
        };
        source.alias_users.fetch_add(1, Ordering::AcqRel);
        #[cfg(target_vendor = "sony")]
        crate::soft_watch::register(base, unsafe { source.buf.as_ptr().add(offset) }, len, false);
        Self {
            base,
            buf: unsafe { NonNull::new_unchecked(source.buf.as_ptr().add(offset)) },
            len,
            committed_len: AtomicUsize::new(len),
            perm: Mutex::new(perm),
            name,
            arena: false,
            alias_backing: Some(AliasBacking { source, offset }),
            alias_users: AtomicUsize::new(0),
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
    fn committed_len(&self) -> usize {
        self.committed_len.load(Ordering::Acquire)
    }

    #[inline]
    fn committed_end(&self) -> u64 {
        self.base + self.committed_len() as u64
    }

    #[inline]
    fn committed_contains(&self, va: u64) -> bool {
        va >= self.base && va < self.committed_end()
    }

    #[inline]
    fn perm(&self) -> Perm {
        *self.perm.lock()
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        #[cfg(target_vendor = "sony")]
        if !self.arena {
            crate::soft_watch::unregister(self.base, self.buf.as_ptr(), self.len);
        }
        if let Some(backing) = &self.alias_backing {
            backing.source.alias_users.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        if self.arena {
            let committed_len = self.committed_len();
            if committed_len != 0 {
                crate::fastmem::decommit(self.buf.as_ptr(), committed_len);
            }
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

#[derive(Clone)]
pub struct HostRegionLease {
    backing: Arc<Region>,
    generation: u64,
}

impl HostRegionLease {
    pub fn base(&self) -> u64 {
        self.backing.base
    }

    pub fn size(&self) -> u64 {
        self.backing.len as u64
    }

    pub fn perm(&self) -> Perm {
        self.backing.perm()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Debug)]
pub enum HostRegionChange {
    Upsert(HostRegion),
    Remove { base: u64, size: u64 },
    Invalidate { base: u64, size: u64 },
}

#[derive(Clone, Debug)]
pub struct HostRegionChanges {
    pub generation: u64,
    pub changes: Vec<HostRegionChange>,
}

#[derive(Clone, Debug)]
struct VersionedHostRegionChange {
    generation: u64,
    change: HostRegionChange,
}

pub struct AddressSpace {
    id: u64,
    regions: Mutex<Vec<Arc<Region>>>,
    host_changes: Mutex<Vec<VersionedHostRegionChange>>,
    generation: AtomicU64,
}

impl AddressSpace {
    pub fn new() -> Self {
        Self {
            id: ADDRESS_SPACE_IDS.fetch_add(1, Ordering::Relaxed),
            regions: Mutex::new(Vec::new()),
            host_changes: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn map(&self, va: u64, len: u64, perm: Perm, name: impl Into<String>) -> Result<usize> {
        self.map_inner(va, len, perm, name.into(), true)
    }

    pub fn map_reserved(
        &self,
        va: u64,
        len: u64,
        perm: Perm,
        name: impl Into<String>,
    ) -> Result<usize> {
        self.map_inner(va, len, perm, name.into(), false)
    }

    fn map_inner(
        &self,
        va: u64,
        len: u64,
        perm: Perm,
        name: String,
        commit: bool,
    ) -> Result<usize> {
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

        let region = Arc::new(if commit {
            Region::new(va, len as usize, perm, name)
        } else {
            Region::reserved(va, len as usize, perm, name)
        });
        log::trace!(
            "map va={va:#x} len={len:#x} perm={perm} name={}",
            region.name
        );
        let host_region = HostRegion {
            base: region.base,
            size: region.len as u64,
            perm: region.perm(),
            host_ptr: region.buf.as_ptr(),
        };
        regs.insert(insert_at, region);
        self.publish_host_change(HostRegionChange::Upsert(host_region));
        Ok(insert_at)
    }

    pub fn map_alias(&self, dst: u64, src: u64, len: u64, perm: Perm, name: impl Into<String>) -> Result<()> {
        for (what, value) in [("dst", dst), ("src", src), ("len", len)] {
            check_aligned(what, value)?;
        }
        if len == 0 { return Err(AddressSpaceError::ZeroLength { va: dst }); }
        let dst_end = dst.checked_add(len).ok_or(AddressSpaceError::Overflow { va: dst, len })?;
        let src_end = src.checked_add(len).ok_or(AddressSpaceError::Overflow { va: src, len })?;
        let mut regs = self.regions.lock();
        let (first, last) = validate_range(&regs, src, len as usize, None)?;
        if let Some(region) = regs.iter().find(|region| region.base < dst_end && dst < region.end()) {
            return Err(AddressSpaceError::Overlap {
                va: dst, len, name: region.name.clone(),
                existing_base: region.base, existing_end: region.end(),
            });
        }
        let name = name.into();
        let mut aliases = Vec::with_capacity(last - first + 1);
        for region in &regs[first..=last] {
            let start = src.max(region.base);
            let end = src_end.min(region.committed_end());
            aliases.push(Arc::new(Region::alias(
                dst + (start - src), region, (start - region.base) as usize,
                (end - start) as usize, perm, name.clone(),
            )));
        }
        let insert_at = regs.partition_point(|region| region.base < dst);
        for alias in &aliases {
            self.publish_host_change(HostRegionChange::Upsert(HostRegion {
                base: alias.base, size: alias.len as u64, perm: alias.perm(), host_ptr: alias.buf.as_ptr(),
            }));
        }
        regs.splice(insert_at..insert_at, aliases);
        Ok(())
    }

    pub fn unmap_alias(&self, dst: u64, src: u64, len: u64) -> Result<()> {
        for (what, value) in [("dst", dst), ("src", src), ("len", len)] {
            check_aligned(what, value)?;
        }
        if len == 0 { return Err(AddressSpaceError::ZeroLength { va: dst }); }
        let dst_end = dst.checked_add(len).ok_or(AddressSpaceError::Overflow { va: dst, len })?;
        src.checked_add(len).ok_or(AddressSpaceError::Overflow { va: src, len })?;
        let mut regs = self.regions.lock();
        let (first, last) = validate_range(&regs, dst, len as usize, None)?;
        let (mut source_index, _) = validate_range(&regs, src, len as usize, None)?;
        let mut destination_index = first;
        let mut offset = 0u64;
        while offset < len {
            let destination = &regs[destination_index];
            let source = &regs[source_index];
            let destination_offset = (dst + offset - destination.base) as usize;
            let source_offset = (src + offset - source.base) as usize;
            let same_backing = destination.alias_backing.is_some() && unsafe {
                destination.buf.as_ptr().add(destination_offset) == source.buf.as_ptr().add(source_offset)
            };
            if !same_backing {
                return Err(AddressSpaceError::AliasMismatch { va: dst, len });
            }
            offset += (destination.end() - dst - offset)
                .min(source.end() - src - offset).min(len - offset);
            if dst + offset == destination.end() { destination_index += 1; }
            if src + offset == source.end() { source_index += 1; }
        }
        let mut survivors = Vec::new();
        let mut changes = Vec::new();
        for region in &regs[first..=last] {
            if region.base < dst {
                survivors.push(Arc::new(Region::alias(region.base, region, 0,
                    (dst - region.base) as usize, region.perm(), region.name.clone())));
            }
            if region.end() > dst_end {
                survivors.push(Arc::new(Region::alias(dst_end, region,
                    (dst_end - region.base) as usize, (region.end() - dst_end) as usize,
                    region.perm(), region.name.clone())));
            }
            changes.push(HostRegionChange::Remove { base: region.base, size: region.len as u64 });
            let backing = region.alias_backing.as_ref().unwrap();
            let start = dst.max(region.base);
            let end = dst_end.min(region.end());
            changes.push(HostRegionChange::Invalidate {
                base: backing.source.base + backing.offset as u64 + (start - region.base),
                size: end - start,
            });
        }
        for survivor in &survivors {
            changes.push(HostRegionChange::Upsert(HostRegion {
                base: survivor.base, size: survivor.len as u64,
                perm: survivor.perm(), host_ptr: survivor.buf.as_ptr(),
            }));
        }
        regs.splice(first..=last, survivors);
        for change in changes { self.publish_host_change(change); }
        Ok(())
    }

    pub fn resize_committed(&self, va: u64, len: u64) -> Result<()> {
        check_aligned("va", va)?;
        check_aligned("len", len)?;

        let regs = self.regions.lock();
        let region = regs
            .binary_search_by_key(&va, |region| region.base)
            .ok()
            .and_then(|index| regs.get(index))
            .filter(|region| len <= region.len as u64)
            .ok_or(AddressSpaceError::Unmapped {
                va,
                len: len as usize,
            })?;
        let old_len = region.committed_len();
        let new_len = len as usize;
        if region.alias_backing.is_some()
            || (new_len < old_len && region.alias_users.load(Ordering::Acquire) != 0)
        {
            return Err(AddressSpaceError::AliasInUse { va, len });
        }

        if new_len > old_len {
            let delta = new_len - old_len;
            if region.arena {
                let commit_va = va + old_len as u64;
                let ptr = crate::fastmem::commit(commit_va, delta).ok_or(
                    AddressSpaceError::CommitFailed {
                        va: commit_va,
                        len: delta as u64,
                    },
                )?;
                debug_assert_eq!(ptr, unsafe { region.buf.as_ptr().add(old_len) });
            }
            region.committed_len.store(new_len, Ordering::Release);
        } else if new_len < old_len {
            let delta = old_len - new_len;
            region.committed_len.store(new_len, Ordering::Release);
            let ptr = unsafe { region.buf.as_ptr().add(new_len) };
            if region.arena {
                crate::fastmem::decommit(ptr, delta);
            } else {
                unsafe { std::ptr::write_bytes(ptr, 0, delta) };
            }
        }
        Ok(())
    }

    pub fn protect(&self, va: u64, len: u64, perm: Perm) -> Result<()> {
        let regs = self.regions.lock();
        let r = regs
            .binary_search_by_key(&va, |r| r.base)
            .ok()
            .and_then(|idx| regs.get(idx))
            .filter(|r| r.len as u64 == len)
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
        let host_region = HostRegion {
            base: r.base,
            size: r.len as u64,
            perm,
            host_ptr: r.buf.as_ptr(),
        };
        drop(p);
        self.publish_host_change(HostRegionChange::Upsert(host_region));
        Ok(())
    }

    #[inline]
    fn with_cached_region<R>(
        &self,
        generation: u64,
        va: u64,
        len: usize,
        visit: impl FnOnce(&Region, usize) -> R,
    ) -> Option<R> {
        LAST_REGION.with(|slot| {
            let slot = slot.borrow();
            let cached = slot.as_ref()?;
            if cached.space != self.id || cached.generation != generation {
                return None;
            }
            let region = &cached.region;
            let off = usize::try_from(va.checked_sub(region.base)?).ok()?;
            if off.checked_add(len)? > region.committed_len() {
                return None;
            }
            Some(visit(region, off))
        })
    }

    fn remember_region(&self, generation: u64, region: &Arc<Region>) {
        LAST_REGION.with(|slot| {
            *slot.borrow_mut() = Some(CachedRegion {
                space: self.id,
                generation,
                region: Arc::clone(region),
            });
        });
    }

    pub fn read(&self, va: u64, buf: &mut [u8]) -> Result<()> {
        let generation = self.generation();
        let len = buf.len();
        let hit = self.with_cached_region(generation, va, len, |region, off| unsafe {
            std::ptr::copy_nonoverlapping(region.buf.as_ptr().add(off), buf.as_mut_ptr(), len);
        });
        if hit.is_some() {
            return Ok(());
        }
        let plan = self.plan_range(va, len, None)?;
        if let RangePlan::Single(region) = &plan {
            self.remember_region(generation, region);
        }
        plan.for_each_chunk(va, buf.len(), |region, region_off, buf_off, len| unsafe {
            std::ptr::copy_nonoverlapping(
                region.buf.as_ptr().add(region_off),
                buf.as_mut_ptr().add(buf_off),
                len,
            );
        });
        Ok(())
    }

    pub fn write(&self, va: u64, buf: &[u8]) -> Result<()> {
        let generation = self.generation();
        let len = buf.len();
        let hit = self.with_cached_region(generation, va, len, |region, off| {
            trace_host_write(region, va, off, buf);
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), region.buf.as_ptr().add(off), len);
            }
        });
        if hit.is_some() {
            return Ok(());
        }
        let plan = self.plan_range(va, len, None)?;
        if let RangePlan::Single(region) = &plan {
            self.remember_region(generation, region);
        }
        plan.for_each_chunk(va, buf.len(), |region, region_off, buf_off, len| {
            let chunk = &buf[buf_off..buf_off + len];
            trace_host_write(region, va + buf_off as u64, region_off, chunk);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    region.buf.as_ptr().add(region_off),
                    len,
                );
            }
        });
        Ok(())
    }

    pub fn copy(&self, src_va: u64, dst_va: u64, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }

        let regs = self.regions.lock();
        let (src_first, src_last) = validate_range(&regs, src_va, len, None)?;
        let (dst_first, dst_last) = validate_range(&regs, dst_va, len, None)?;
        if src_va == dst_va {
            return Ok(());
        }

        if src_first == src_last && dst_first == dst_last {
            let src = &regs[src_first];
            let dst = &regs[dst_first];
            let src_off = (src_va - src.base) as usize;
            let dst_off = (dst_va - dst.base) as usize;
            unsafe {
                let src_ptr = src.buf.as_ptr().add(src_off);
                trace_host_write(
                    dst,
                    dst_va,
                    dst_off,
                    std::slice::from_raw_parts(src_ptr, len),
                );
                std::ptr::copy(src_ptr, dst.buf.as_ptr().add(dst_off), len);
            }
            return Ok(());
        }
        drop(regs);

        let mut staging = vec![0u8; len];
        self.read(src_va, &mut staging)?;
        self.write(dst_va, &staging)
    }

    pub fn atomic_load_u32(&self, va: u64) -> Result<u32> {
        if va & 3 != 0 {
            let mut b = [0u8; 4];
            self.read(va, &mut b)?;
            return Ok(u32::from_le_bytes(b));
        }
        let (region, off) = self.locate(va, 4)?;
        unsafe {
            let p = region.buf.as_ptr().add(off) as *const std::sync::atomic::AtomicU32;
            Ok((*p).load(Ordering::SeqCst))
        }
    }

    pub fn atomic_cas_u32(&self, va: u64, current: u32, new: u32) -> Result<bool> {
        if va & 3 != 0 {
            let mut b = [0u8; 4];
            self.read(va, &mut b)?;
            if u32::from_le_bytes(b) != current {
                return Ok(false);
            }
            self.write(va, &new.to_le_bytes())?;
            return Ok(true);
        }
        let (region, off) = self.locate(va, 4)?;
        unsafe {
            let p = region.buf.as_ptr().add(off) as *const std::sync::atomic::AtomicU32;
            Ok((*p)
                .compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok())
        }
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
        let plan = self.plan_range(va, buf.len(), Some(Perm::R))?;
        plan.for_each_chunk(va, buf.len(), |region, region_off, buf_off, len| unsafe {
            std::ptr::copy_nonoverlapping(
                region.buf.as_ptr().add(region_off),
                buf.as_mut_ptr().add(buf_off),
                len,
            );
        });
        Ok(())
    }

    pub fn write_checked(&self, va: u64, buf: &[u8]) -> Result<()> {
        let plan = self.plan_range(va, buf.len(), Some(Perm::W))?;
        plan.for_each_chunk(va, buf.len(), |region, region_off, buf_off, len| {
            let chunk = &buf[buf_off..buf_off + len];
            trace_host_write(region, va + buf_off as u64, region_off, chunk);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    region.buf.as_ptr().add(region_off),
                    len,
                );
            }
        });
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

    pub fn host_region_leases(&self) -> Vec<HostRegionLease> {
        let generation = self.generation();
        self.regions
            .lock()
            .iter()
            .map(|backing| HostRegionLease {
                backing: Arc::clone(backing),
                generation,
            })
            .collect()
    }

    pub fn host_region_lease_at(&self, va: u64) -> Option<HostRegionLease> {
        let generation = self.generation();
        self.regions
            .lock()
            .iter()
            .find(|region| region.contains(va))
            .map(|backing| HostRegionLease {
                backing: Arc::clone(backing),
                generation,
            })
    }

    pub fn host_region_changes_since(&self, generation: u64) -> HostRegionChanges {
        let changes = self.host_changes.lock();
        let current = self.generation.load(Ordering::Acquire);
        let first = if generation <= current {
            changes.partition_point(|change| change.generation <= generation)
        } else {
            0
        };
        HostRegionChanges {
            generation: current,
            changes: changes[first..]
                .iter()
                .map(|change| change.change.clone())
                .collect(),
        }
    }

    pub fn host_region_at(&self, va: u64) -> Option<HostRegion> {
        let regs = self.regions.lock();
        let idx = regs.partition_point(|r| r.base <= va).checked_sub(1)?;
        let r = &regs[idx];
        if !r.contains(va) {
            return None;
        }
        Some(HostRegion {
            base: r.base,
            size: r.len as u64,
            perm: r.perm(),
            host_ptr: r.buf.as_ptr(),
        })
    }

    pub fn unmapped_gaps(&self, va: u64, len: u64) -> Result<Vec<(u64, u64)>> {
        if len == 0 {
            return Err(AddressSpaceError::ZeroLength { va });
        }
        let end = va
            .checked_add(len)
            .ok_or(AddressSpaceError::Overflow { va, len })?;

        let regs = self.regions.lock();
        let first = regs.partition_point(|r| r.end() <= va);
        let mut gaps = Vec::new();
        let mut cursor = va;

        for r in &regs[first..] {
            if r.base >= end {
                break;
            }
            if r.base > cursor {
                gaps.push((cursor, r.base.min(end)));
            }
            cursor = cursor.max(r.end());
            if cursor >= end {
                break;
            }
        }
        if cursor < end {
            gaps.push((cursor, end));
        }
        Ok(gaps)
    }

    fn locate(&self, va: u64, len: usize) -> Result<(Arc<Region>, usize)> {
        let regs = self.regions.lock();
        let idx = regs.partition_point(|r| r.base <= va);
        let region = idx
            .checked_sub(1)
            .and_then(|idx| regs.get(idx))
            .filter(|r| r.contains(va))
            .cloned()
            .ok_or(AddressSpaceError::Unmapped { va, len })?;
        let off = (va - region.base) as usize;
        if off
            .checked_add(len)
            .map(|end| end > region.committed_len())
            .unwrap_or(true)
        {
            return Err(AddressSpaceError::Unmapped { va, len });
        }
        Ok((region, off))
    }

    fn plan_range(&self, va: u64, len: usize, required: Option<Perm>) -> Result<RangePlan> {
        let regs = self.regions.lock();
        let (first, last) = validate_range(&regs, va, len, required)?;
        if first == last {
            Ok(RangePlan::Single(regs[first].clone()))
        } else {
            Ok(RangePlan::Multiple(regs[first..=last].to_vec()))
        }
    }

    fn publish_host_change(&self, change: HostRegionChange) {
        let mut changes = self.host_changes.lock();
        let generation = self
            .generation
            .load(Ordering::Relaxed)
            .checked_add(1)
            .expect("address-space generation overflow");
        changes.push(VersionedHostRegionChange { generation, change });
        self.generation.store(generation, Ordering::Release);
    }
}

enum RangePlan {
    Single(Arc<Region>),
    Multiple(Vec<Arc<Region>>),
}

static ADDRESS_SPACE_IDS: AtomicU64 = AtomicU64::new(1);

struct CachedRegion {
    space: u64,
    generation: u64,
    region: Arc<Region>,
}

thread_local! {
    static LAST_REGION: std::cell::RefCell<Option<CachedRegion>> =
        const { std::cell::RefCell::new(None) };
}

impl RangePlan {
    fn for_each_chunk(&self, va: u64, len: usize, visit: impl FnMut(&Region, usize, usize, usize)) {
        match self {
            Self::Single(region) => {
                for_each_range_chunk(std::slice::from_ref(region), va, len, visit)
            }
            Self::Multiple(regions) => for_each_range_chunk(regions, va, len, visit),
        }
    }
}

fn validate_range(
    regs: &[Arc<Region>],
    va: u64,
    len: usize,
    required: Option<Perm>,
) -> Result<(usize, usize)> {
    let end = va
        .checked_add(len as u64)
        .ok_or(AddressSpaceError::Overflow {
            va,
            len: len as u64,
        })?;
    let first = regs
        .partition_point(|r| r.base <= va)
        .checked_sub(1)
        .filter(|idx| regs[*idx].committed_contains(va))
        .ok_or(AddressSpaceError::Unmapped { va, len })?;

    let mut index = first;
    let mut cursor = va;
    loop {
        let region = regs
            .get(index)
            .filter(|r| r.committed_contains(cursor))
            .ok_or(AddressSpaceError::Unmapped { va, len })?;
        if let Some(need) = required {
            let have = region.perm();
            if !have.contains(need) {
                return Err(AddressSpaceError::PermissionDenied {
                    va: cursor,
                    have,
                    need,
                });
            }
        }
        if cursor == end {
            return Ok((first, index));
        }

        cursor = region.committed_end().min(end);
        if cursor == end {
            return Ok((first, index));
        }
        index += 1;
        if regs.get(index).map(|r| r.base) != Some(cursor) {
            return Err(AddressSpaceError::Unmapped { va, len });
        }
    }
}

fn for_each_range_chunk(
    regs: &[Arc<Region>],
    va: u64,
    len: usize,
    mut visit: impl FnMut(&Region, usize, usize, usize),
) {
    let mut index = 0;
    let mut cursor = va;
    let mut buf_off = 0;
    while buf_off < len {
        let region = &regs[index];
        let region_off = (cursor - region.base) as usize;
        let chunk_len = (region.len - region_off).min(len - buf_off);
        visit(region, region_off, buf_off, chunk_len);
        cursor += chunk_len as u64;
        buf_off += chunk_len;
        index += 1;
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
    fn copy_handles_distinct_and_overlapping_ranges() {
        let a = fresh();
        let base = 0x2_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "copy").unwrap();
        let initial: Vec<u8> = (0..64u8).collect();
        a.write(base, &initial).unwrap();

        a.copy(base, base + 128, initial.len()).unwrap();
        let mut distinct = vec![0u8; initial.len()];
        a.read(base + 128, &mut distinct).unwrap();
        assert_eq!(distinct, initial);

        let mut forward_expected = initial.clone();
        forward_expected.copy_within(0..48, 8);
        a.write(base, &initial).unwrap();
        a.copy(base, base + 8, 48).unwrap();
        let mut forward = vec![0u8; initial.len()];
        a.read(base, &mut forward).unwrap();
        assert_eq!(forward, forward_expected);

        let mut backward_expected = initial.clone();
        backward_expected.copy_within(8..56, 0);
        a.write(base, &initial).unwrap();
        a.copy(base + 8, base, 48).unwrap();
        let mut backward = vec![0u8; initial.len()];
        a.read(base, &mut backward).unwrap();
        assert_eq!(backward, backward_expected);
    }

    #[test]
    fn copy_crosses_adjacent_backing_regions() {
        let a = fresh();
        let base = 0x30_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "src-a").unwrap();
        a.map(base + PAGE_SIZE, PAGE_SIZE, Perm::RW, "src-b")
            .unwrap();
        a.map(base + PAGE_SIZE * 3, PAGE_SIZE * 2, Perm::RW, "dst")
            .unwrap();
        let src = base + PAGE_SIZE - 32;
        let dst = base + PAGE_SIZE * 3 + 17;
        let payload: Vec<u8> = (0..96).map(|v| (v * 13) as u8).collect();
        a.write(src, &payload).unwrap();

        a.copy(src, dst, payload.len()).unwrap();
        let mut actual = vec![0u8; payload.len()];
        a.read(dst, &mut actual).unwrap();
        assert_eq!(actual, payload);
    }

    #[test]
    fn failed_copy_does_not_modify_destination() {
        let a = fresh();
        let base = 0x40_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "dst").unwrap();
        a.write(base, &[0xa5; 32]).unwrap();

        assert!(matches!(
            a.copy(base + PAGE_SIZE * 2, base, 32),
            Err(AddressSpaceError::Unmapped { .. })
        ));
        let mut actual = [0u8; 32];
        a.read(base, &mut actual).unwrap();
        assert_eq!(actual, [0xa5; 32]);
    }

    #[test]
    fn reserved_region_commits_on_demand() {
        let a = fresh();
        let base = 0x7E_0000_0000;
        a.map_reserved(base, PAGE_SIZE * 3, Perm::RW, "reserved")
            .unwrap();
        assert_eq!(a.regions()[0].size, PAGE_SIZE * 3);
        assert_eq!(
            a.host_region_at(base + 2 * PAGE_SIZE).unwrap().size,
            PAGE_SIZE * 3
        );

        let mut byte = [0u8; 1];
        assert!(matches!(
            a.read(base, &mut byte),
            Err(AddressSpaceError::Unmapped { .. })
        ));

        a.resize_committed(base, PAGE_SIZE).unwrap();
        a.write(base + 8, &[0x5a]).unwrap();
        a.resize_committed(base, PAGE_SIZE * 2).unwrap();
        a.read(base + 8, &mut byte).unwrap();
        assert_eq!(byte, [0x5a]);
        a.write(base + PAGE_SIZE + 8, &[0xa5]).unwrap();
        a.read(base + PAGE_SIZE + 8, &mut byte).unwrap();
        assert_eq!(byte, [0xa5]);
    }

    #[test]
    fn reserved_region_shrinks_and_regrows_zeroed() {
        let a = fresh();
        let base = 0x7E_0100_0000;
        a.map_reserved(base, PAGE_SIZE * 3, Perm::RW, "reserved")
            .unwrap();
        a.resize_committed(base, PAGE_SIZE * 3).unwrap();
        a.write(base + 2 * PAGE_SIZE + 8, &[0x7c]).unwrap();

        a.resize_committed(base, PAGE_SIZE).unwrap();
        let mut byte = [0u8; 1];
        assert!(matches!(
            a.read(base + PAGE_SIZE, &mut byte),
            Err(AddressSpaceError::Unmapped { .. })
        ));
        a.write(base + 8, &[0x3d]).unwrap();
        a.read(base + 8, &mut byte).unwrap();
        assert_eq!(byte, [0x3d]);

        a.resize_committed(base, PAGE_SIZE * 3).unwrap();
        a.read(base + 2 * PAGE_SIZE + 8, &mut byte).unwrap();
        assert_eq!(byte, [0]);
    }

    #[cfg(windows)]
    #[test]
    fn dropping_partial_reserved_overlap_preserves_live_fastmem_lease() {
        const BASE: u64 = 0xf2_0000_0000;
        const LEN: u64 = PAGE_SIZE * 3;
        let live = fresh();
        live.map(BASE, LEN, Perm::RW, "live").unwrap();

        let reserved = fresh();
        reserved
            .map_reserved(BASE, LEN, Perm::RW, "reserved")
            .unwrap();
        reserved.resize_committed(BASE, PAGE_SIZE).unwrap();
        drop(reserved);

        assert_ne!(
            crate::fastmem::take_write_watch(BASE, LEN as usize),
            crate::fastmem::WriteWatchResult::Unavailable
        );
        live.write(BASE + PAGE_SIZE * 2, &[0x5a]).unwrap();
        let mut actual = [0];
        live.read(BASE + PAGE_SIZE * 2, &mut actual).unwrap();
        assert_eq!(actual, [0x5a]);
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
    fn read_write_cross_adjacent_regions() {
        let a = fresh();
        let base = 0x40_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "first").unwrap();
        a.map(base + PAGE_SIZE, PAGE_SIZE, Perm::RW, "second")
            .unwrap();

        let payload = [1, 2, 3, 4, 5, 6, 7, 8];
        a.write(base + PAGE_SIZE - 4, &payload).unwrap();
        let mut buf = [0u8; 8];
        a.read(base + PAGE_SIZE - 4, &mut buf).unwrap();
        assert_eq!(buf, payload);
    }

    #[test]
    fn range_crossing_gap_fails_without_partial_write() {
        let a = fresh();
        let base = 0x50_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "first").unwrap();
        a.map(base + PAGE_SIZE * 2, PAGE_SIZE, Perm::RW, "second")
            .unwrap();
        a.write(base + PAGE_SIZE - 4, &[0xaa; 4]).unwrap();

        let err = a.write(base + PAGE_SIZE - 4, &[0x55; 8]).unwrap_err();
        assert!(matches!(err, AddressSpaceError::Unmapped { .. }));
        let mut tail = [0; 4];
        a.read(base + PAGE_SIZE - 4, &mut tail).unwrap();
        assert_eq!(tail, [0xaa; 4]);

        let mut crossing = [0; 8];
        let err = a.read(base + PAGE_SIZE - 4, &mut crossing).unwrap_err();
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
    fn shared_alias_writes_through_across_source_regions_without_new_storage() {
        let a = fresh();
        let src = 0x6910_0000;
        let dst = 0x6920_0000;
        a.map(src, PAGE_SIZE, Perm::RX, "text").unwrap();
        a.map(src + PAGE_SIZE, PAGE_SIZE, Perm::RO, "ro").unwrap();
        a.write(src, &[1, 2, 3, 4]).unwrap();
        a.map_alias(dst, src, PAGE_SIZE * 2, Perm::RW, "alias").unwrap();
        assert_eq!(a.host_region_at(dst).unwrap().host_ptr, a.host_region_at(src).unwrap().host_ptr);
        assert_eq!(a.host_region_at(dst + PAGE_SIZE).unwrap().host_ptr,
            a.host_region_at(src + PAGE_SIZE).unwrap().host_ptr);
        let payload = [9, 8, 7, 6, 5, 4, 3, 2];
        a.write_checked(dst + PAGE_SIZE - 4, &payload).unwrap();
        let mut read = [0; 8];
        a.read(src + PAGE_SIZE - 4, &mut read).unwrap();
        assert_eq!(read, payload);
        assert!(a.write_checked(src, &[0xff]).is_err());
        a.write(src, &[0xa5]).unwrap();
        a.read(dst, &mut read[..1]).unwrap();
        assert_eq!(read[0], 0xa5);
    }

    #[test]
    fn shared_alias_high_addresses_do_not_overflow_relative_offset_arithmetic() {
        let a = fresh();
        let src = 0xffff_ffff_fffb_0000;
        let dst = 0xffff_ffff_fffc_0000;
        a.map(src, PAGE_SIZE * 3, Perm::RO, "high_source").unwrap();
        a.map_alias(dst, src, PAGE_SIZE * 3, Perm::RW, "high_alias").unwrap();
        a.write_checked(dst + PAGE_SIZE, &[0x5a]).unwrap();
        let mut value = [0];
        a.read(src + PAGE_SIZE, &mut value).unwrap();
        assert_eq!(value, [0x5a]);
        let generation = a.generation();
        a.unmap_alias(dst + PAGE_SIZE, src + PAGE_SIZE, PAGE_SIZE).unwrap();
        assert!(a.host_region_at(dst + PAGE_SIZE).is_none());
        assert!(a.host_region_at(dst).is_some());
        assert!(a.host_region_at(dst + PAGE_SIZE * 2).is_some());
        assert!(a.host_region_changes_since(generation).changes.iter().any(|change| matches!(change,
            HostRegionChange::Invalidate { base, size } if *base == src + PAGE_SIZE && *size == PAGE_SIZE)));
        a.unmap_alias(dst, src, PAGE_SIZE).unwrap();
        a.unmap_alias(dst + PAGE_SIZE * 2, src + PAGE_SIZE * 2, PAGE_SIZE).unwrap();
        assert_eq!(a.regions().len(), 1);
    }

    #[test]
    fn shared_alias_failure_does_not_publish_or_partially_map() {
        let a = fresh();
        let src = 0x6930_0000;
        let dst = 0x6940_0000;
        a.map(src, PAGE_SIZE, Perm::RO, "source").unwrap();
        let generation = a.generation();
        assert!(a.map_alias(dst, src, PAGE_SIZE * 2, Perm::RW, "alias").is_err());
        assert_eq!(a.generation(), generation);
        assert!(a.host_region_at(dst).is_none());
        assert!(a.map_alias(src, src, PAGE_SIZE, Perm::RW, "alias").is_err());
        assert_eq!(a.generation(), generation);
        assert!(a.map_alias(dst + 1, src, PAGE_SIZE, Perm::RW, "alias").is_err());
        assert!(a.map_alias(dst, src, 0, Perm::RW, "alias").is_err());
        assert!(a.map_alias(u64::MAX - PAGE_SIZE + 1, src, PAGE_SIZE, Perm::RW, "alias").is_err());
        assert_eq!(a.regions().len(), 1);
    }

    #[test]
    fn shared_alias_partial_unmap_preserves_survivors_and_invalidates_source() {
        let a = fresh();
        let src = 0x6950_0000;
        let dst = 0x6960_0000;
        a.map(src, PAGE_SIZE * 3, Perm::RX, "source").unwrap();
        a.map_alias(dst, src, PAGE_SIZE * 3, Perm::RW, "alias").unwrap();
        a.write_checked(dst + PAGE_SIZE, &[0xa5]).unwrap();
        let generation = a.generation();
        a.unmap_alias(dst + PAGE_SIZE, src + PAGE_SIZE, PAGE_SIZE).unwrap();
        let updates = a.host_region_changes_since(generation);
        assert!(updates.changes.iter().any(|change| matches!(change,
            HostRegionChange::Invalidate { base, size } if *base == src + PAGE_SIZE && *size == PAGE_SIZE)));
        assert!(a.host_region_at(dst + PAGE_SIZE).is_none());
        assert!(a.host_region_at(dst).is_some());
        assert!(a.host_region_at(dst + PAGE_SIZE * 2).is_some());
        let mut value = [0];
        a.read(src + PAGE_SIZE, &mut value).unwrap();
        assert_eq!(value, [0xa5]);
        a.map_alias(dst + PAGE_SIZE, src + PAGE_SIZE, PAGE_SIZE, Perm::RW, "replacement").unwrap();
        a.unmap_alias(dst, src, PAGE_SIZE * 3).unwrap();
        assert!(a.host_region_at(dst).is_none());
        assert_eq!(a.regions().len(), 1);
        a.read(src + PAGE_SIZE, &mut value).unwrap();
        assert_eq!(value, [0xa5]);
    }

    #[test]
    fn shared_alias_unmap_requires_matching_source_and_preserves_source_storage() {
        let a = fresh();
        let src = 0x6970_0000;
        let other = 0x6980_0000;
        let dst = 0x6990_0000;
        a.map(src, PAGE_SIZE, Perm::RW, "source").unwrap();
        a.map(other, PAGE_SIZE, Perm::RW, "other").unwrap();
        a.map_alias(dst, src, PAGE_SIZE, Perm::RW, "alias").unwrap();
        let generation = a.generation();
        assert!(matches!(a.unmap_alias(dst, other, PAGE_SIZE), Err(AddressSpaceError::AliasMismatch { .. })));
        assert_eq!(a.generation(), generation);
        assert!(matches!(a.resize_committed(src, 0), Err(AddressSpaceError::AliasInUse { .. })));
        a.unmap_alias(dst, src, PAGE_SIZE).unwrap();
        assert!(a.unmap_alias(dst, src, PAGE_SIZE).is_err());
        a.write(src, &[0x3c]).unwrap();
        let mut value = [0];
        a.read(src, &mut value).unwrap();
        assert_eq!(value, [0x3c]);
        a.resize_committed(src, 0).unwrap();
    }

    #[test]
    fn shared_alias_lease_pins_original_backing_after_alias_and_space_drop() {
        let a = fresh();
        let src = 0x69a0_0000;
        let dst = 0x69b0_0000;
        a.map(src, PAGE_SIZE, Perm::RW, "source").unwrap();
        a.write(src, &[0x6d]).unwrap();
        a.map_alias(dst, src, PAGE_SIZE, Perm::RW, "alias").unwrap();
        let lease = a.host_region_lease_at(dst).unwrap();
        let source = Arc::downgrade(&lease.backing.alias_backing.as_ref().unwrap().source);
        a.unmap_alias(dst, src, PAGE_SIZE).unwrap();
        LAST_REGION.with(|slot| *slot.borrow_mut() = None);
        drop(a);
        assert!(source.upgrade().is_some());
        assert_eq!(unsafe { *lease.backing.buf.as_ptr() }, 0x6d);
        drop(lease);
        assert!(source.upgrade().is_none());
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
    fn checked_ranges_prevalidate_all_region_permissions() {
        let a = fresh();
        let base = 0x60_0000;
        a.map(base, PAGE_SIZE, Perm::RW, "rw").unwrap();
        a.map(base + PAGE_SIZE, PAGE_SIZE, Perm::RO, "ro").unwrap();
        a.map(base + PAGE_SIZE * 2, PAGE_SIZE, Perm::W, "write-only")
            .unwrap();
        a.write(base + PAGE_SIZE - 4, &[0xaa; 8]).unwrap();

        let err = a
            .write_checked(base + PAGE_SIZE - 4, &[0x55; 8])
            .unwrap_err();
        assert!(matches!(
            err,
            AddressSpaceError::PermissionDenied {
                va,
                need,
                ..
            } if va == base + PAGE_SIZE && need == Perm::W
        ));
        let mut unchanged = [0; 8];
        a.read(base + PAGE_SIZE - 4, &mut unchanged).unwrap();
        assert_eq!(unchanged, [0xaa; 8]);

        let mut readable = [0; 8];
        a.read_checked(base + PAGE_SIZE - 4, &mut readable).unwrap();
        assert_eq!(readable, [0xaa; 8]);

        let mut unreadable = [0; 8];
        let err = a
            .read_checked(base + PAGE_SIZE * 2 - 4, &mut unreadable)
            .unwrap_err();
        assert!(matches!(
            err,
            AddressSpaceError::PermissionDenied {
                va,
                need,
                ..
            } if va == base + PAGE_SIZE * 2 && need == Perm::R
        ));
    }

    #[test]
    fn binary_lookups_find_later_regions() {
        let a = fresh();
        a.map(0x10_0000, PAGE_SIZE, Perm::RW, "first").unwrap();
        a.map(0x12_0000, PAGE_SIZE, Perm::RO, "middle").unwrap();
        a.map(0x14_0000, PAGE_SIZE, Perm::RW, "last").unwrap();

        let middle = a.host_region_at(0x12_0000).unwrap();
        assert_eq!(middle.base, 0x12_0000);
        assert_eq!(middle.size, PAGE_SIZE);
        assert_eq!(middle.perm, Perm::RO);
        assert_eq!(a.host_region_at(0x12_0080).unwrap().base, 0x12_0000);
        assert!(a.host_region_at(0x13_0000).is_none());

        a.write(0x14_0080, &[0x5a]).unwrap();
        let mut byte = [0];
        a.read(0x14_0080, &mut byte).unwrap();
        assert_eq!(byte, [0x5a]);
    }

    #[test]
    fn host_region_changes_are_incremental_and_track_protection() {
        let a = fresh();
        assert_eq!(a.generation(), 0);
        assert!(a.host_region_changes_since(0).changes.is_empty());

        a.map(0x80_0000, PAGE_SIZE, Perm::RW, "later").unwrap();
        let first_generation = a.generation();
        a.map(0x70_0000, PAGE_SIZE, Perm::RO, "earlier").unwrap();
        let second_generation = a.generation();

        let all = a.host_region_changes_since(0);
        assert_eq!(all.generation, second_generation);
        assert_eq!(all.changes.len(), 2);
        assert!(matches!(
            &all.changes[0],
            HostRegionChange::Upsert(region)
                if region.base == 0x80_0000 && region.perm == Perm::RW
        ));
        assert!(matches!(
            &all.changes[1],
            HostRegionChange::Upsert(region)
                if region.base == 0x70_0000 && region.perm == Perm::RO
        ));

        let incremental = a.host_region_changes_since(first_generation);
        assert_eq!(incremental.generation, second_generation);
        assert_eq!(incremental.changes.len(), 1);
        assert!(matches!(
            &incremental.changes[0],
            HostRegionChange::Upsert(region) if region.base == 0x70_0000
        ));

        a.protect(0x70_0000, PAGE_SIZE, Perm::X).unwrap();
        let protected = a.host_region_changes_since(second_generation);
        assert_eq!(protected.generation, a.generation());
        assert_eq!(protected.changes.len(), 1);
        assert!(matches!(
            &protected.changes[0],
            HostRegionChange::Upsert(region)
                if region.base == 0x70_0000 && region.perm == Perm::X
        ));
        assert!(a
            .host_region_changes_since(protected.generation)
            .changes
            .is_empty());
    }

    #[test]
    fn host_region_change_cursors_are_independent() {
        let a = fresh();
        let mut cursor_a = 0;
        let mut cursor_b = 0;

        for page in 0..8u64 {
            a.map(
                0x90_0000 + page * PAGE_SIZE,
                PAGE_SIZE,
                Perm::RW,
                "incremental",
            )
            .unwrap();
            let update_a = a.host_region_changes_since(cursor_a);
            assert_eq!(update_a.changes.len(), 1);
            cursor_a = update_a.generation;
        }

        let update_b = a.host_region_changes_since(cursor_b);
        assert_eq!(update_b.changes.len(), 8);
        cursor_b = update_b.generation;
        assert_eq!(cursor_a, cursor_b);
        assert!(a.host_region_changes_since(cursor_a).changes.is_empty());
        assert!(a.host_region_changes_since(cursor_b).changes.is_empty());
    }

    #[test]
    fn unmapped_gaps_clip_overlapping_regions() {
        let a = fresh();
        a.map(0x20_e000, PAGE_SIZE * 4, Perm::RW, "head").unwrap();
        a.map(0x21_3000, PAGE_SIZE, Perm::RW, "middle").unwrap();
        a.map(0x21_6000, PAGE_SIZE * 3, Perm::RW, "tail").unwrap();

        assert_eq!(
            a.unmapped_gaps(0x21_0000, PAGE_SIZE * 8).unwrap(),
            vec![(0x21_2000, 0x21_3000), (0x21_4000, 0x21_6000)]
        );
    }

    #[test]
    fn unmapped_gaps_handle_empty_covered_and_overflowing_ranges() {
        let a = fresh();
        assert_eq!(
            a.unmapped_gaps(0x30_0000, PAGE_SIZE * 2).unwrap(),
            vec![(0x30_0000, 0x30_2000)]
        );

        a.map(0x30_0000, PAGE_SIZE * 2, Perm::RW, "covered")
            .unwrap();
        assert!(a
            .unmapped_gaps(0x30_0000, PAGE_SIZE * 2)
            .unwrap()
            .is_empty());
        assert!(matches!(
            a.unmapped_gaps(u64::MAX - 0x7ff, PAGE_SIZE),
            Err(AddressSpaceError::Overflow { .. })
        ));
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

    #[test]
    fn cached_region_reads_follow_the_space_and_new_mappings() {
        const BASE: u64 = 1 << 41;
        let a = fresh();
        let b = fresh();
        a.map(BASE, PAGE_SIZE, Perm::RW, "a").unwrap();
        b.map(BASE, PAGE_SIZE, Perm::RW, "b").unwrap();
        a.write(BASE + 8, &[0xaa]).unwrap();
        b.write(BASE + 8, &[0xbb]).unwrap();
        let mut byte = [0u8; 1];
        a.read(BASE + 8, &mut byte).unwrap();
        assert_eq!(byte, [0xaa]);
        b.read(BASE + 8, &mut byte).unwrap();
        assert_eq!(byte, [0xbb]);
        a.read(BASE + 8, &mut byte).unwrap();
        assert_eq!(byte, [0xaa]);

        let mut span = [0u8; 8];
        assert!(matches!(
            a.read(BASE + PAGE_SIZE - 4, &mut span),
            Err(AddressSpaceError::Unmapped { .. })
        ));
        a.map(BASE + PAGE_SIZE, PAGE_SIZE, Perm::RW, "a2").unwrap();
        a.write(BASE + PAGE_SIZE - 4, &[1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        a.read(BASE + PAGE_SIZE - 4, &mut span).unwrap();
        assert_eq!(span, [1, 2, 3, 4, 5, 6, 7, 8]);
        a.read(BASE + PAGE_SIZE + 2, &mut byte).unwrap();
        assert_eq!(byte, [7]);
    }
}

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::draw::PoolCbufDraw;

pub const CBUF_POOL_DIRECTORY_BYTES: u64 = 16 * 1024 * 1024;
pub const CBUF_POOL_DIRECTORY_STRIDE: u64 = 512;
const REGION_ALIGN: u64 = 16;

pub fn cbuf_pool_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_CBUF_POOL").map_or(true, |value| value != "0"))
}

pub fn cbuf_pool_bytes() -> u64 {
    static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("NEXIUM_CBUF_POOL_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(128)
            .clamp(16, 2048)
            .saturating_mul(1024 * 1024)
    })
}

const SIZE_CLASSES: usize = 13;

fn size_class(bytes: u64) -> Option<usize> {
    if bytes == 0 || bytes > (REGION_ALIGN << (SIZE_CLASSES - 1)) {
        return None;
    }
    let mut class = 0usize;
    while (REGION_ALIGN << class) < bytes {
        class += 1;
    }
    Some(class)
}

#[derive(Default)]
struct FreeList {
    classes: Vec<Vec<u64>>,
    head: u64,
    end: u64,
}

impl FreeList {
    fn new(start: u64, end: u64) -> Self {
        Self {
            classes: vec![Vec::new(); SIZE_CLASSES],
            head: start,
            end,
        }
    }

    fn take(&mut self, class: usize) -> Option<u64> {
        if let Some(offset) = self.classes.get_mut(class)?.pop() {
            return Some(offset);
        }
        let bytes = REGION_ALIGN << class;
        let offset = self.head;
        let next = offset.checked_add(bytes)?;
        if next > self.end {
            return None;
        }
        self.head = next;
        Some(offset)
    }

    fn insert(&mut self, class: usize, offset: u64) {
        if let Some(list) = self.classes.get_mut(class) {
            list.push(offset);
        }
    }
}

struct Region {
    offset: u64,
    class: usize,
    refs: u32,
}

#[derive(Default)]
struct RegionTable {
    free: FreeList,
    regions: Vec<Option<Region>>,
    free_ids: Vec<u32>,
    used_bytes: u64,
}

pub struct CbufPoolShared {
    mapped: usize,
    size: u64,
    data_start: u64,
    table: Mutex<RegionTable>,
    allocs: AtomicU64,
    alloc_failures: AtomicU64,
    frees: AtomicU64,
}

unsafe impl Send for CbufPoolShared {}
unsafe impl Sync for CbufPoolShared {}

impl CbufPoolShared {
    pub fn new(mapped: *mut u8, size: u64, data_start: u64) -> Self {
        let mut table = RegionTable::default();
        table.free = FreeList::new(data_start, size.max(data_start));
        Self {
            mapped: mapped as usize,
            size,
            data_start,
            table: Mutex::new(table),
            allocs: AtomicU64::new(0),
            alloc_failures: AtomicU64::new(0),
            frees: AtomicU64::new(0),
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn used_bytes(&self) -> u64 {
        self.table.lock().map(|table| table.used_bytes).unwrap_or(0)
    }

    pub fn counters(&self) -> (u64, u64, u64) {
        (
            self.allocs.load(Ordering::Relaxed),
            self.alloc_failures.load(Ordering::Relaxed),
            self.frees.load(Ordering::Relaxed),
        )
    }

    pub fn alloc(&self, len: usize) -> Option<(u32, u64)> {
        let class = size_class(len as u64)?;
        let bytes = REGION_ALIGN << class;
        let mut table = self.table.lock().ok()?;
        let Some(offset) = table.free.take(class) else {
            self.alloc_failures.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        table.used_bytes = table.used_bytes.saturating_add(bytes);
        let region = Region {
            offset,
            class,
            refs: 1,
        };
        let id = match table.free_ids.pop() {
            Some(id) => {
                table.regions[id as usize] = Some(region);
                id
            }
            None => {
                let id = u32::try_from(table.regions.len()).ok()?;
                table.regions.push(Some(region));
                id
            }
        };
        self.allocs.fetch_add(1, Ordering::Relaxed);
        Some((id, offset))
    }

    pub fn retain(&self, ids: &[u32]) {
        let Ok(mut table) = self.table.lock() else {
            return;
        };
        for &id in ids {
            if let Some(Some(region)) = table.regions.get_mut(id as usize) {
                region.refs = region.refs.saturating_add(1);
            }
        }
    }

    pub fn release(&self, ids: &[u32]) {
        let Ok(mut table) = self.table.lock() else {
            return;
        };
        for &id in ids {
            let Some(slot) = table.regions.get_mut(id as usize) else {
                continue;
            };
            let Some(region) = slot.as_mut() else {
                continue;
            };
            region.refs = region.refs.saturating_sub(1);
            if region.refs != 0 {
                continue;
            }
            let (offset, class) = (region.offset, region.class);
            *slot = None;
            table.free_ids.push(id);
            table.used_bytes = table.used_bytes.saturating_sub(REGION_ALIGN << class);
            table.free.insert(class, offset);
            self.frees.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn fill(&self, offset: u64, len: usize, write: impl FnOnce(&mut [u8]) -> bool) -> bool {
        if self.mapped == 0 || offset < self.data_start {
            return false;
        }
        let Some(end) = offset.checked_add(len as u64) else {
            return false;
        };
        if end > self.size {
            return false;
        }
        let dst = unsafe { std::slice::from_raw_parts_mut((self.mapped as *mut u8).add(offset as usize), len) };
        write(dst)
    }
}

pub struct CbufPool {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) mapped: *mut u8,
    pub(crate) size: u64,
    pub(crate) rebar_bytes: u64,
    pub(crate) dir_head: u64,
    pub(crate) dir_size: u64,
    pub(crate) dir_slot_span: Vec<Option<(u64, u64)>>,
    slot_regions: Vec<Vec<u32>>,
    shared: Arc<CbufPoolShared>,
    padded: bool,
}

unsafe impl Send for CbufPool {}
unsafe impl Sync for CbufPool {}

impl CbufPool {
    pub(crate) fn new(
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        mapped: *mut u8,
        size: u64,
        rebar_bytes: u64,
        frame_slots: usize,
        padded: bool,
    ) -> Option<Self> {
        let dir_size = CBUF_POOL_DIRECTORY_BYTES.min(size / 4);
        let data_end = if padded { size.checked_sub(dir_size)? } else { size };
        if dir_size < CBUF_POOL_DIRECTORY_STRIDE || data_end <= dir_size {
            return None;
        }
        Some(Self {
            buffer,
            memory,
            mapped,
            size: data_end,
            rebar_bytes,
            dir_head: 0,
            dir_size,
            dir_slot_span: vec![None; frame_slots.max(1)],
            slot_regions: vec![Vec::new(); frame_slots.max(1)],
            shared: Arc::new(CbufPoolShared::new(mapped, data_end, dir_size)),
            padded,
        })
    }

    pub(crate) fn dynamic_window(&self) -> Option<u64> {
        self.padded.then_some(self.size)
    }

    pub fn shared(&self) -> Arc<CbufPoolShared> {
        self.shared.clone()
    }

    pub(crate) fn directory_alloc(&mut self) -> Option<u64> {
        let end = self.dir_head.checked_add(CBUF_POOL_DIRECTORY_STRIDE)?;
        if end > self.dir_size {
            return None;
        }
        let offset = self.dir_head;
        self.dir_head = end;
        Some(offset)
    }

    pub(crate) fn directory_fits(&self, bytes: u64) -> bool {
        bytes <= self.dir_size
    }

    pub(crate) fn note_submitted(&mut self, frame_slot: usize, regions: &[u32]) {
        if self.slot_regions.len() <= frame_slot {
            self.slot_regions.resize_with(frame_slot + 1, Vec::new);
        }
        self.slot_regions[frame_slot].extend_from_slice(regions);
    }

    pub(crate) fn complete_frame_slot(&mut self, frame_slot: usize) {
        let Some(regions) = self.slot_regions.get_mut(frame_slot) else {
            return;
        };
        if regions.is_empty() {
            return;
        }
        let released = std::mem::take(regions);
        self.shared.release(&released);
        *regions = released;
        regions.clear();
    }
}

fn put_u32(dst: &mut [u8], byte_offset: usize, value: u32) {
    dst[byte_offset..byte_offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub fn write_pool_cbuf_directory(draw: &PoolCbufDraw, dir_offset: u64, dst: &mut [u8]) -> bool {
    let dir_len = nexium_spirv::GFX_CBUF_MIN_SIZE as usize;
    if dst.len() < dir_len {
        return false;
    }
    dst[..dir_len].fill(0);
    for slot in 0..nexium_spirv::GFX_CBUF_SLOTS as usize {
        put_u32(dst, slot * 8, nexium_spirv::GFX_CBUF_ZERO_WORD);
    }
    for slot in &draw.slots {
        if slot.logical_slot >= nexium_spirv::GFX_CBUF_SLOTS
            || slot.pool_offset < dir_offset
            || (slot.pool_offset - dir_offset) % 4 != 0
        {
            return false;
        }
        let Ok(base) = u32::try_from((slot.pool_offset - dir_offset) / 4) else {
            return false;
        };
        let directory = slot.logical_slot as usize * 8;
        put_u32(dst, directory, base);
        put_u32(dst, directory + 4, slot.word_count);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::PoolCbufSlot;

    #[test]
    fn regions_are_recycled_only_when_every_reference_is_gone() {
        let shared = CbufPoolShared::new(std::ptr::null_mut(), 4096, 1024);
        let (first, offset) = shared.alloc(100).unwrap();
        assert_eq!(offset, 1024);
        assert_eq!(shared.used_bytes(), 128);
        shared.retain(&[first]);
        shared.release(&[first]);
        assert_eq!(shared.used_bytes(), 128);
        shared.release(&[first]);
        assert_eq!(shared.used_bytes(), 0);
        let (second, second_offset) = shared.alloc(100).unwrap();
        assert_eq!(second, first);
        assert_eq!(second_offset, 1024);
        let (_, third_offset) = shared.alloc(16).unwrap();
        assert_eq!(third_offset, 1024 + 128);
    }

    #[test]
    fn allocation_fails_when_the_heap_is_exhausted() {
        let shared = CbufPoolShared::new(std::ptr::null_mut(), 1024 + 64, 1024);
        let (a, _) = shared.alloc(32).unwrap();
        let (b, _) = shared.alloc(32).unwrap();
        assert!(shared.alloc(16).is_none());
        assert_eq!(shared.counters().1, 1);
        shared.release(&[a, b]);
        let (_, offset) = shared.alloc(32).unwrap();
        assert_eq!(offset, 1024 + 32);
        assert!(shared.alloc(100_000).is_none());
    }

    #[test]
    fn fill_rejects_ranges_outside_the_heap() {
        let shared = CbufPoolShared::new(std::ptr::null_mut(), 4096, 1024);
        assert!(!shared.fill(1024, 16, |_| true));
        let mut backing = vec![0u8; 4096];
        let shared = CbufPoolShared::new(backing.as_mut_ptr(), 4096, 1024);
        assert!(!shared.fill(512, 16, |_| true));
        assert!(!shared.fill(4090, 16, |_| true));
        assert!(shared.fill(1024, 4, |dst| {
            dst.copy_from_slice(&[1, 2, 3, 4]);
            true
        }));
        assert_eq!(&backing[1024..1028], &[1, 2, 3, 4]);
    }

    #[test]
    fn directory_points_slots_at_pool_offsets_relative_to_itself() {
        let draw = PoolCbufDraw {
            slots: vec![
                PoolCbufSlot {
                    logical_slot: 1,
                    word_count: 16,
                    pool_offset: 1 << 20,
                    byte_len: 64,
                },
                PoolCbufSlot {
                    logical_slot: 20,
                    word_count: 4,
                    pool_offset: (1 << 20) + 64,
                    byte_len: 16,
                },
            ],
            regions: vec![0],
        };
        let mut dst = vec![0xffu8; nexium_spirv::GFX_CBUF_MIN_SIZE as usize];
        assert!(write_pool_cbuf_directory(&draw, 512, &mut dst));
        let word = |index: usize| u32::from_le_bytes(dst[index * 4..index * 4 + 4].try_into().unwrap());
        assert_eq!(word(2), ((1 << 20) - 512) / 4);
        assert_eq!(word(3), 16);
        assert_eq!(word(40), ((1 << 20) + 64 - 512) / 4);
        assert_eq!(word(41), 4);
        assert_eq!(word(0), nexium_spirv::GFX_CBUF_ZERO_WORD);
        assert_eq!(word(1), 0);
        assert_eq!(word(nexium_spirv::GFX_CBUF_ZERO_WORD as usize), 0);
        let behind = PoolCbufDraw {
            slots: vec![PoolCbufSlot {
                logical_slot: 0,
                word_count: 1,
                pool_offset: 256,
                byte_len: 4,
            }],
            regions: vec![],
        };
        assert!(!write_pool_cbuf_directory(&behind, 512, &mut dst));
    }
}

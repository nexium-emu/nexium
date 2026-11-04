use nexium_common::fast_hash::FastMap as HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const PAGE_SHIFT: u64 = 16;
const PAGE_MASK: u64 = !0xFFFFu64;

const BLOCK_SHIFT: u64 = 21;
const BLOCK_MASK: u64 = !((1u64 << BLOCK_SHIFT) - 1);

#[derive(Default)]
struct Generations {
    pages: std::collections::BTreeMap<u64, u64>,
    blocks: std::collections::BTreeMap<u64, u64>,
}

fn gens() -> &'static Mutex<Generations> {
    static S: OnceLock<Mutex<Generations>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Generations::default()))
}

const SMALL_PAGE_SHIFT: u64 = 12;
const SMALL_PAGE_MASK: u64 = !0xFFFu64;
static BUMP_SERIAL: AtomicU64 = AtomicU64::new(0);

fn bumped_pages() -> &'static Mutex<HashMap<u64, u64>> {
    static S: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::default()))
}

pub fn bump_serial() -> u64 {
    BUMP_SERIAL.load(Ordering::Acquire)
}

pub fn region_bumped_since(gpu_va: u64, size: u64, serial: u64) -> bool {
    if size == 0 || bump_serial() <= serial {
        return false;
    }
    let start = gpu_va & SMALL_PAGE_MASK;
    let end = gpu_va.saturating_add(size).saturating_add(0xFFF) & SMALL_PAGE_MASK;
    let pages = bumped_pages().lock().unwrap();
    let mut p = start;
    while p < end {
        if pages.get(&p).copied().unwrap_or(0) > serial {
            return true;
        }
        p = p.wrapping_add(1 << SMALL_PAGE_SHIFT);
    }
    false
}

pub fn bump_region(gpu_va: u64, size: u64) {
    if size == 0 {
        return;
    }
    let start = gpu_va & PAGE_MASK;
    let end = gpu_va.saturating_add(size).saturating_add(0xFFFF) & PAGE_MASK;
    let mut g = gens().lock().unwrap();
    let mut p = start;
    while p < end {
        let e = g.pages.entry(p).or_insert(0);
        *e = e.wrapping_add(1);
        let b = g.blocks.entry(p & BLOCK_MASK).or_insert(0);
        *b = b.wrapping_add(1);
        p = p.wrapping_add(1 << PAGE_SHIFT);
    }
    drop(g);
    let serial = BUMP_SERIAL.fetch_add(1, Ordering::AcqRel) + 1;
    let small_start = gpu_va & SMALL_PAGE_MASK;
    let small_end = gpu_va.saturating_add(size).saturating_add(0xFFF) & SMALL_PAGE_MASK;
    let mut pages = bumped_pages().lock().unwrap();
    let mut p = small_start;
    while p < small_end {
        pages.insert(p, serial);
        p = p.wrapping_add(1 << SMALL_PAGE_SHIFT);
    }
}

pub fn region_gen(gpu_va: u64) -> u64 {
    let g = gens().lock().unwrap();
    g.pages.get(&(gpu_va & PAGE_MASK)).copied().unwrap_or(0)
}

pub fn region_gen_range(gpu_va: u64, size: u64) -> u64 {
    if size == 0 {
        return region_gen(gpu_va);
    }
    let start = gpu_va & PAGE_MASK;
    let end = gpu_va.saturating_add(size).saturating_add(0xFFFF) & PAGE_MASK;
    let g = gens().lock().unwrap();
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |key: u64, generation: u64| {
        h ^= key;
        h = h.wrapping_mul(0x100000001b3);
        h ^= generation;
        h = h.wrapping_mul(0x100000001b3);
    };
    let block_start = start.saturating_add((1 << BLOCK_SHIFT) - 1) & BLOCK_MASK;
    let block_end = end & BLOCK_MASK;
    if block_start < block_end {
        for (&page, &generation) in g.pages.range(start..block_start) {
            mix(page, generation);
        }
        for (&block, &generation) in g.blocks.range(block_start..block_end) {
            mix(block, generation);
        }
        for (&page, &generation) in g.pages.range(block_end..end) {
            mix(page, generation);
        }
    } else {
        for (&page, &generation) in g.pages.range(start..end) {
            mix(page, generation);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    #[test]
    fn range_generation_changes_only_when_its_pages_are_bumped() {
        let base = 0x7a_0000_0000u64;
        let before = super::region_gen_range(base, 0x40_0000);
        super::bump_region(base + 0x80_0000, 0x1000);
        assert_eq!(super::region_gen_range(base, 0x40_0000), before);
        let serial = super::bump_serial();
        assert!(!super::region_bumped_since(base, 0x40_0000, serial));
        super::bump_region(base + 0x2_0000, 0x10);
        assert_ne!(super::region_gen_range(base, 0x40_0000), before);
        assert!(super::region_bumped_since(base, 0x40_0000, serial));
        assert!(!super::region_bumped_since(base + 0x10_0000, 0x1000, serial));
    }
}

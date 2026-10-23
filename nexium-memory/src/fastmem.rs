use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};

pub const ARENA_MAX_BITS: u32 = 40;
pub const ARENA_MIN_BITS: u32 = 36;

static ARENA_BITS_ACHIEVED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn arena_bits() -> u32 {
    let bits = ARENA_BITS_ACHIEVED.load(Ordering::Acquire);
    if bits != 0 {
        return bits;
    }
    arena();
    ARENA_BITS_ACHIEVED
        .load(Ordering::Acquire)
        .max(ARENA_MIN_BITS)
}

pub fn arena_size() -> u64 {
    1u64 << arena_bits()
}

static DIRECT_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn direct_mode_requested() -> bool {
    match DIRECT_MODE.load(Ordering::Acquire) {
        1 => true,
        2 => false,
        _ => {
            let requested = std::env::var("NEXIUM_NCE")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false);
            DIRECT_MODE.store(if requested { 1 } else { 2 }, Ordering::Release);
            requested
        }
    }
}

pub fn request_direct_mode(enabled: bool) -> bool {
    if ARENA.get().is_some() {
        return direct_mode_requested() == enabled;
    }
    DIRECT_MODE.store(if enabled { 1 } else { 2 }, Ordering::Release);
    true
}

pub fn direct_va_base() -> Option<u64> {
    let base = arena();
    if base.is_null() || !direct_mode_requested() {
        None
    } else {
        Some(base as u64)
    }
}

pub fn direct_va_range() -> Option<(u64, u64)> {
    let base = direct_va_base()?;
    Some((base, base + arena_size()))
}

pub fn va_offset(va: u64) -> Option<u64> {
    match direct_va_base() {
        Some(base) => va.checked_sub(base).filter(|offset| *offset < arena_size()),
        None => (va < arena_size()).then_some(va),
    }
}

pub fn va_window() -> (u64, u64) {
    match direct_va_base() {
        Some(base) => (base, base + arena_size()),
        None => (0, arena_size()),
    }
}

pub fn host_ptr(va: u64) -> Option<*mut u8> {
    let base = arena();
    if base.is_null() {
        return None;
    }
    let offset = va_offset(va)?;
    Some(unsafe { base.add(offset as usize) })
}

#[cfg(windows)]
mod sys {
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_DECOMMIT: u32 = 0x4000;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const WRITE_WATCH_FLAG_RESET: u32 = 0x1;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READONLY: u32 = 0x02;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
        fn VirtualProtect(addr: *mut u8, size: usize, protect: u32, old: *mut u32) -> i32;
        fn GetWriteWatch(
            flags: u32,
            base: *mut u8,
            size: usize,
            addresses: *mut *mut u8,
            count: *mut usize,
            granularity: *mut u32,
        ) -> u32;
    }

    pub fn reserve(size: usize) -> (*mut u8, bool) {
        if !crate::soft_watch::enabled() {
            let watched = unsafe {
                VirtualAlloc(
                    std::ptr::null_mut(),
                    size,
                    MEM_RESERVE | MEM_WRITE_WATCH,
                    PAGE_NOACCESS,
                )
            };
            if !watched.is_null() {
                return (watched, true);
            }
        }
        let plain = unsafe { VirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) };
        (plain, false)
    }

    pub fn commit(ptr: *mut u8, len: usize) -> bool {
        !unsafe { VirtualAlloc(ptr, len, MEM_COMMIT, PAGE_READWRITE) }.is_null()
    }

    pub fn decommit(ptr: *mut u8, len: usize) {
        unsafe {
            VirtualFree(ptr, len, MEM_DECOMMIT);
        }
    }

    pub fn protect(ptr: *mut u8, len: usize, trap: bool) -> bool {
        let mut old = 0u32;
        let p = if trap { PAGE_READONLY } else { PAGE_READWRITE };
        unsafe { VirtualProtect(ptr, len, p, &mut old) != 0 }
    }

    pub fn take_write_watch(ptr: *mut u8, len: usize, addresses: &mut [usize]) -> Option<usize> {
        let started = std::time::Instant::now();
        let mut count = addresses.len();
        let mut granularity = 0u32;
        let result = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                ptr,
                len,
                addresses.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        super::WRITE_WATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::WRITE_WATCH_PAGES.fetch_add((len >> 12) as u64, std::sync::atomic::Ordering::Relaxed);
        super::WRITE_WATCH_NS.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        (result == 0).then_some(count)
    }
}

#[cfg(unix)]
mod sys {
    pub fn protect(ptr: *mut u8, len: usize, trap: bool) -> bool {
        let p = if trap {
            libc::PROT_READ
        } else {
            libc::PROT_READ | libc::PROT_WRITE
        };
        unsafe { libc::mprotect(ptr as *mut libc::c_void, len, p) == 0 }
    }

    pub fn reserve(size: usize) -> (*mut u8, bool) {
        let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if host_page != 4096 {
            log::warn!(
                "fastmem: host page size {} != 4096; arena disabled (commit granularity mismatch)",
                host_page
            );
            return (std::ptr::null_mut(), false);
        }
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            (std::ptr::null_mut(), false)
        } else {
            (p as *mut u8, watch_reserved(p as usize, size))
        }
    }

    /// Registers the whole reservation for write tracking. Registration costs
    /// nothing per page: only pages a query has reset take a (kernel-resolved)
    /// fault on their next write.
    #[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
    fn watch_reserved(start: usize, size: usize) -> bool {
        let Some(watch) = crate::linux_watch::global() else {
            return false;
        };
        if watch.register(start, size) {
            return true;
        }
        log::warn!(
            "fastmem: write-watch registration of the arena failed ({}); disabled",
            std::io::Error::last_os_error()
        );
        false
    }

    #[cfg(not(all(target_os = "linux", not(target_vendor = "sony"))))]
    fn watch_reserved(_start: usize, _size: usize) -> bool {
        false
    }

    pub fn commit(ptr: *mut u8, len: usize) -> bool {
        unsafe {
            libc::mprotect(
                ptr as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
            ) == 0
        }
    }

    pub fn decommit(ptr: *mut u8, len: usize) {
        unsafe {
            libc::mmap(
                ptr as *mut libc::c_void,
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE | libc::MAP_FIXED,
                -1,
                0,
            );
        }
        // The replacement mapping is not write-tracked. Until this succeeds a
        // query of the range fails its registration check and reads as dirty.
        #[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
        if super::ARENA_WRITE_WATCH.load(std::sync::atomic::Ordering::Acquire) {
            let registered = crate::linux_watch::global()
                .is_some_and(|watch| watch.register(ptr as usize, len));
            if !registered {
                log::warn!(
                    "fastmem: write-watch re-registration failed at {:p}+{:#x}; range will always read dirty",
                    ptr,
                    len
                );
            }
        }
    }
}

static ARENA: OnceLock<AtomicPtr<u8>> = OnceLock::new();
static ARENA_WRITE_WATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static COMMITTED_RANGES: OnceLock<Mutex<Vec<CommittedRange>>> = OnceLock::new();
static COMMIT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WRITE_WATCH_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static WRITE_WATCH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WRITE_WATCH_PAGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WRITE_WATCH_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn take_write_watch_stats() -> (u64, u64, u64) {
    (
        WRITE_WATCH_CALLS.swap(0, Ordering::Relaxed),
        WRITE_WATCH_PAGES.swap(0, Ordering::Relaxed),
        WRITE_WATCH_NS.swap(0, Ordering::Relaxed),
    )
}

fn write_watch_memo_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_WRITE_WATCH_MEMO").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

thread_local! {
    static WRITE_WATCH_MEMO_THREAD_OPT_IN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn enable_write_watch_memo_for_thread() {
    WRITE_WATCH_MEMO_THREAD_OPT_IN.with(|opt_in| opt_in.set(true));
}

pub fn decode_write_watch_sweep_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_DECODE_WRITE_WATCH_SWEEP").map_or(true, |value| value != "0")
    })
}

#[derive(Default)]
struct WriteWatchMemo {
    active: bool,
    epoch: u64,
    chunks: nexium_common::fast_hash::FastMap<u64, u16>,
    used: nexium_common::fast_hash::FastMap<u64, u16>,
}

thread_local! {
    static WRITE_WATCH_MEMO: std::cell::RefCell<WriteWatchMemo> =
        std::cell::RefCell::new(WriteWatchMemo::default());
}

thread_local! {
    static WRITE_WATCH_EPOCH_SEEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn bump_write_watch_epoch() {
    WRITE_WATCH_EPOCH.fetch_add(1, Ordering::Release);
    WRITE_WATCH_EPOCH_SEEN.with(|seen| seen.set(true));
    if write_watch_memo_enabled() || WRITE_WATCH_MEMO_THREAD_OPT_IN.with(|opt_in| opt_in.get()) {
        WRITE_WATCH_MEMO.with(|memo| memo.borrow_mut().active = true);
    }
}

pub fn write_watch_epoch() -> u64 {
    WRITE_WATCH_EPOCH.load(Ordering::Acquire)
}

pub fn write_watch_epoch_active() -> bool {
    WRITE_WATCH_EPOCH_SEEN.with(|seen| seen.get())
}

impl WriteWatchMemo {
    fn sync(&mut self) -> bool {
        if !self.active {
            return false;
        }
        let epoch = WRITE_WATCH_EPOCH.load(Ordering::Acquire);
        if epoch != self.epoch {
            self.epoch = epoch;
            self.chunks.clear();
            self.used.clear();
        }
        true
    }

    fn for_each_chunk(lo: u64, hi: u64, mut visit: impl FnMut(u64, u16) -> bool) -> bool {
        if hi <= lo {
            return true;
        }
        let mut page = lo >> 12;
        let last = (hi - 1) >> 12;
        loop {
            let chunk = page >> 4;
            let first_slot = page & 15;
            let end_page = last.min((chunk << 4) | 15);
            let count = end_page - page + 1;
            let mask = (((1u32 << count) - 1) as u16) << first_slot;
            if !visit(chunk, mask) {
                return false;
            }
            if end_page == last {
                return true;
            }
            page = end_page + 1;
        }
    }

    fn covers(&mut self, lo: u64, hi: u64) -> bool {
        if !self.sync() {
            return false;
        }
        let chunks = &self.chunks;
        let covered = Self::for_each_chunk(lo, hi, |chunk, mask| {
            chunks.get(&chunk).is_some_and(|&seen| seen & mask == mask)
        });
        if covered {
            let used = &mut self.used;
            Self::for_each_chunk(lo, hi, |chunk, mask| {
                *used.entry(chunk).or_insert(0) |= mask;
                true
            });
        }
        covered
    }

    fn insert(&mut self, lo: u64, hi: u64) {
        if !self.sync() {
            return;
        }
        let chunks = &mut self.chunks;
        let used = &mut self.used;
        Self::for_each_chunk(lo, hi, |chunk, mask| {
            *chunks.entry(chunk).or_insert(0) |= mask;
            *used.entry(chunk).or_insert(0) |= mask;
            true
        });
    }

    fn adopt(&mut self, lo: u64, hi: u64) {
        if !self.sync() {
            return;
        }
        let chunks = &mut self.chunks;
        Self::for_each_chunk(lo, hi, |chunk, mask| {
            *chunks.entry(chunk).or_insert(0) |= mask;
            true
        });
    }

    fn used_ranges(&mut self) -> Vec<(u64, usize)> {
        if !self.sync() {
            return Vec::new();
        }
        let mut keys: Vec<(u64, u16)> = self.used.iter().map(|(&key, &mask)| (key, mask)).collect();
        keys.sort_unstable();
        let mut ranges: Vec<(u64, usize)> = Vec::with_capacity(keys.len());
        for (chunk, mask) in keys {
            let base = chunk << 16;
            let mut slot = 0u32;
            while slot < 16 {
                if mask & (1 << slot) == 0 {
                    slot += 1;
                    continue;
                }
                let start = slot;
                while slot < 16 && mask & (1 << slot) != 0 {
                    slot += 1;
                }
                let lo = base + u64::from(start) * 4096;
                let len = ((slot - start) as usize) * 4096;
                match ranges.last_mut() {
                    Some((last_lo, last_len)) if *last_lo + *last_len as u64 == lo => {
                        *last_len += len;
                    }
                    _ => ranges.push((lo, len)),
                }
            }
        }
        ranges
    }
}

pub fn write_watch_memo_adopt(ranges: &[(u64, usize)]) {
    WRITE_WATCH_MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        for &(va, len) in ranges {
            if let Some((lo, query_len)) = write_watch_query_range(va, len) {
                memo.adopt(lo, lo + query_len as u64);
            }
        }
    });
}

pub fn write_watch_memo_used_ranges() -> Vec<(u64, usize)> {
    WRITE_WATCH_MEMO.with(|memo| memo.borrow_mut().used_ranges())
}

fn write_watch_memo_covers(lo: u64, query_len: usize) -> bool {
    WRITE_WATCH_MEMO.with(|memo| memo.borrow_mut().covers(lo, lo + query_len as u64))
}

fn write_watch_memo_insert(lo: u64, query_len: usize) {
    WRITE_WATCH_MEMO.with(|memo| memo.borrow_mut().insert(lo, lo + query_len as u64));
}

const OBSERVED_WRITE_PAGE_SHIFT: u64 = 16;
const OBSERVED_WRITE_PAGE_MASK: u64 = !((1u64 << OBSERVED_WRITE_PAGE_SHIFT) - 1);
static OBSERVED_WRITE_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OBSERVED_WRITE_PAGE_4K_MASK: u64 = !0xfffu64;

#[derive(Default)]
struct ObservedWriteChunk {
    generation: u64,
    pages: [u64; 16],
}

type ObservedWriteChunks = nexium_common::fast_hash::FastMap<u64, ObservedWriteChunk>;
static OBSERVED_WRITE_CHUNKS: OnceLock<Mutex<ObservedWriteChunks>> = OnceLock::new();

fn observed_write_chunks() -> &'static Mutex<ObservedWriteChunks> {
    OBSERVED_WRITE_CHUNKS.get_or_init(|| Mutex::new(ObservedWriteChunks::default()))
}

fn visit_observed_pages(
    chunks: &ObservedWriteChunks,
    va: u64,
    end: u64,
    generation: u64,
    mut visit: impl FnMut(u64, u64),
) {
    let mut page = va & OBSERVED_WRITE_PAGE_4K_MASK;
    let last = (end - 1) & OBSERVED_WRITE_PAGE_4K_MASK;
    loop {
        let base = page & OBSERVED_WRITE_PAGE_MASK;
        let chunk_last = (base | 0xf000).min(last);
        if let Some(chunk) = chunks.get(&base).filter(|chunk| chunk.generation > generation) {
            let first_slot = ((page - base) >> 12) as usize;
            let last_slot = ((chunk_last - base) >> 12) as usize;
            for slot in first_slot..=last_slot {
                let written = chunk.pages[slot];
                if written > generation {
                    visit(base + (slot as u64) * 4096, written);
                }
            }
        }
        if chunk_last == last {
            break;
        }
        page = chunk_last + 4096;
    }
}

pub fn observed_write_pages_changed_since(va: u64, len: usize, generation: u64) -> usize {
    if len == 0 {
        return 0;
    }
    let Some(end) = va.checked_add(len as u64) else {
        return usize::MAX;
    };
    let chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut changed = 0;
    visit_observed_pages(&chunks, va, end, generation, |_, _| changed += 1);
    changed
}

pub fn observed_write_spans_since(
    va: u64,
    len: usize,
    generation: u64,
    spans: &mut Vec<(u64, usize)>,
) {
    if len == 0 {
        return;
    }
    let Some(end) = va.checked_add(len as u64) else {
        spans.push((va, len));
        return;
    };
    let chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    visit_observed_pages(&chunks, va, end, generation, |page, _| {
        match spans.last_mut() {
            Some((last_va, last_len)) if last_va.checked_add(*last_len as u64) == Some(page) => {
                *last_len += 4096;
            }
            _ => spans.push((page, 4096)),
        }
    });
}

pub fn observed_write_generation_pages(va: u64, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let Some(end) = va.checked_add(len as u64) else {
        return u64::MAX;
    };
    let chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut generation = 0;
    visit_observed_pages(&chunks, va, end, 0, |_, written| generation = generation.max(written));
    generation
}

#[cfg(windows)]
fn record_observed_write_pages(base: *mut u8, addresses: &[usize]) {
    if addresses.is_empty() {
        return;
    }
    let base_addr = base as usize as u64;
    let mut chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let serial = OBSERVED_WRITE_SERIAL
        .load(Ordering::Relaxed)
        .wrapping_add(1);
    for &address in addresses {
        let guest_va = (address as u64).saturating_sub(base_addr);
        let chunk = chunks.entry(guest_va & OBSERVED_WRITE_PAGE_MASK).or_default();
        chunk.generation = chunk.generation.max(serial);
        let page = &mut chunk.pages[((guest_va >> 12) & 15) as usize];
        *page = (*page).max(serial);
    }
    OBSERVED_WRITE_SERIAL.store(serial, Ordering::Release);
}

pub fn observed_write_serial() -> u64 {
    OBSERVED_WRITE_SERIAL.load(Ordering::Acquire)
}

pub fn observed_write_snapshot_range(va: u64, len: usize) -> (u64, u64) {
    let chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let serial = observed_write_serial();
    let generation = if len == 0 {
        0
    } else if let Some(end) = va.checked_add(len as u64) {
        let mut page = va & OBSERVED_WRITE_PAGE_MASK;
        let last = (end - 1) & OBSERVED_WRITE_PAGE_MASK;
        let mut generation = 0;
        loop {
            generation = generation.max(chunks.get(&page).map_or(0, |chunk| chunk.generation));
            if page == last {
                break;
            }
            page += 1u64 << OBSERVED_WRITE_PAGE_SHIFT;
        }
        generation
    } else {
        u64::MAX
    };
    (serial, generation)
}

pub fn observed_write_generation_range(va: u64, len: usize) -> u64 {
    observed_write_snapshot_range(va, len).1
}

pub fn commit_generation() -> u64 {
    COMMIT_GENERATION.load(Ordering::Relaxed)
}

type CommitChanges = std::collections::BTreeMap<u64, (u64, u64)>;
static COMMIT_CHANGES: OnceLock<Mutex<CommitChanges>> = OnceLock::new();

fn commit_changes() -> &'static Mutex<CommitChanges> {
    COMMIT_CHANGES.get_or_init(|| Mutex::new(CommitChanges::new()))
}

fn record_commit_change(changes: &mut CommitChanges, lo: u64, hi: u64, generation: u64) {
    if hi <= lo {
        return;
    }
    let overlapping: Vec<(u64, u64, u64)> = changes
        .range(..hi)
        .rev()
        .take_while(|&(_, &(end, _))| end > lo)
        .map(|(&start, &(end, previous))| (start, end, previous))
        .collect();
    for (start, end, previous) in overlapping {
        changes.remove(&start);
        if start < lo {
            changes.insert(start, (lo, previous));
        }
        if end > hi {
            changes.insert(hi, (end, previous));
        }
    }
    changes.insert(lo, (hi, generation));
}

fn commit_change_generation(changes: &CommitChanges, lo: u64, hi: u64) -> u64 {
    changes
        .range(..hi)
        .rev()
        .take_while(|&(_, &(end, _))| end > lo)
        .map(|(_, &(_, generation))| generation)
        .max()
        .unwrap_or(0)
}

fn note_commit_changes(ranges: &[(u64, u64)]) {
    let mut changes = commit_changes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let generation = COMMIT_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
    for &(lo, hi) in ranges {
        record_commit_change(&mut changes, lo, hi, generation);
    }
}

pub fn commit_generation_range(va: u64, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let Some(lo) = va_offset(va) else {
        return commit_generation();
    };
    let Some(hi) = lo.checked_add(len as u64) else {
        return commit_generation();
    };
    let changes = commit_changes()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    commit_change_generation(&changes, lo, hi)
}

pub fn write_watch_available() -> bool {
    ARENA_WRITE_WATCH.load(Ordering::Acquire) || soft_write_watch_available()
}

#[cfg(any(target_vendor = "sony", windows))]
fn soft_write_watch_available() -> bool {
    crate::soft_watch::enabled()
}

#[cfg(not(any(target_vendor = "sony", windows)))]
fn soft_write_watch_available() -> bool {
    false
}

#[cfg(any(target_vendor = "sony", windows, target_os = "linux"))]
fn record_observed_write_vas(pages: &[u64]) {
    if pages.is_empty() {
        return;
    }
    let mut chunks = observed_write_chunks()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let serial = OBSERVED_WRITE_SERIAL
        .load(Ordering::Relaxed)
        .wrapping_add(1);
    for &guest_va in pages {
        let chunk = chunks.entry(guest_va & OBSERVED_WRITE_PAGE_MASK).or_default();
        chunk.generation = chunk.generation.max(serial);
        let page = &mut chunk.pages[((guest_va >> 12) & 15) as usize];
        *page = (*page).max(serial);
    }
    OBSERVED_WRITE_SERIAL.store(serial, Ordering::Release);
}

#[cfg(any(target_vendor = "sony", windows))]
fn soft_take_write_watch(
    va: u64,
    len: usize,
    spans: Option<&mut Vec<(u64, usize)>>,
) -> WriteWatchResult {
    thread_local! {
        static SOFT_PAGES: std::cell::RefCell<Vec<u64>> = const {
            std::cell::RefCell::new(Vec::new())
        };
    }
    SOFT_PAGES.with(|pages| {
        let mut pages = pages.borrow_mut();
        pages.clear();
        match crate::soft_watch::take(va, len, &mut pages) {
            None => WriteWatchResult::Unavailable,
            Some(false) => WriteWatchResult::Clean,
            Some(true) => {
                record_observed_write_vas(&pages);
                if let Some(spans) = spans {
                    for &page_va in pages.iter() {
                        match spans.last_mut() {
                            Some((last_va, last_len)) if *last_va + *last_len as u64 == page_va => {
                                *last_len += 4096;
                            }
                            _ => spans.push((page_va, 4096)),
                        }
                    }
                }
                WriteWatchResult::Dirty
            }
        }
    })
}

#[cfg(any(target_vendor = "sony", windows))]
fn soft_take_write_watch_observed(
    va: u64,
    len: usize,
    spans: &mut Vec<(u64, usize)>,
) -> WriteWatchObservation {
    if len == 0 {
        return WriteWatchObservation::unavailable();
    }
    let lo = va & !0xfffu64;
    let query_len = (va + len as u64 - lo) as usize;
    let (_, generation_before) = observed_write_snapshot_range(lo, query_len);
    let result = soft_take_write_watch(va, len, Some(spans));
    let (serial_after, generation_after) = observed_write_snapshot_range(lo, query_len);
    WriteWatchObservation {
        result,
        serial_after,
        generation_before,
        generation_after,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CommittedRange {
    lo: u64,
    hi: u64,
    refs: u32,
}

fn arena() -> *mut u8 {
    ARENA
        .get_or_init(|| {
            let mut chosen = std::ptr::null_mut();
            let mut chosen_bits = 0u32;
            let mut write_watch = false;
            let disabled = std::env::var("NEXIUM_NO_FASTMEM_ARENA")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false);
            let mut bits = if disabled { 0 } else { ARENA_MAX_BITS };
            if disabled {
                log::info!("fastmem: arena disabled by NEXIUM_NO_FASTMEM_ARENA");
            }
            while bits >= ARENA_MIN_BITS {
                let (base, ww) = sys::reserve(1usize << bits);
                if !base.is_null() {
                    chosen = base;
                    chosen_bits = bits;
                    write_watch = ww;
                    break;
                }
                log::info!(
                    "fastmem: {}GB arena unavailable; trying {}GB",
                    1u64 << (bits - 30),
                    1u64 << (bits - 31)
                );
                bits -= 1;
            }
            ARENA_WRITE_WATCH.store(write_watch, Ordering::Release);
            ARENA_BITS_ACHIEVED.store(chosen_bits, Ordering::Release);
            if chosen.is_null() {
                log::warn!(
                    "fastmem: could not reserve an arena down to {}GB; falling back to heap regions",
                    1u64 << (ARENA_MIN_BITS - 30)
                );
            } else {
                log::info!(
                    "fastmem: reserved {}GB arena at {:p} (bits={}, write-watch={})",
                    1u64 << (chosen_bits - 30),
                    chosen,
                    chosen_bits,
                    write_watch,
                );
            }
            AtomicPtr::new(chosen)
        })
        .load(Ordering::Relaxed)
}

pub fn base() -> Option<*mut u8> {
    let p = arena();
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

pub fn commit(va: u64, len: usize) -> Option<*mut u8> {
    let base = arena();
    if base.is_null() {
        return None;
    }
    let Some(offset) = va_offset(va) else {
        log::warn!(
            "fastmem: region va={:#x} len={:#x} outside arena; using heap",
            va,
            len
        );
        return None;
    };
    let va = offset;
    let end = va.checked_add(len as u64)?;
    if end > arena_size() {
        log::warn!(
            "fastmem: region va={:#x} len={:#x} outside arena; using heap",
            va,
            len
        );
        return None;
    }
    let ptr = unsafe { base.add(va as usize) };
    let mut ranges = committed_ranges()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !sys::commit(ptr, len) {
        log::warn!(
            "fastmem: commit failed va={:#x} len={:#x}; using heap",
            va,
            len
        );
        return None;
    }
    #[cfg(windows)]
    crate::soft_watch::mark_dirty_host(ptr, len);
    let fresh = !is_committed(&ranges, va, end);
    update_commit_refs(&mut ranges, va, end, true);
    mark_committed_pages(va, end, true);
    if fresh {
        note_commit_changes(&[(va, end)]);
    }
    Some(ptr)
}

pub fn committed_any(va: u64, len: usize) -> bool {
    if arena().is_null() || len == 0 {
        return false;
    }
    let Some(lo) = va_offset(va) else {
        return false;
    };
    let Some(hi) = lo.checked_add(len as u64) else {
        return false;
    };
    let ranges = committed_ranges()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    ranges
        .get(ranges.partition_point(|range| range.hi <= lo))
        .is_some_and(|range| range.lo < hi)
}

pub fn decommit(ptr: *mut u8, len: usize) {
    let base = arena();
    let base_addr = base as usize;
    let ptr_addr = ptr as usize;
    let Some(lo) = ptr_addr.checked_sub(base_addr).map(|offset| offset as u64) else {
        return;
    };
    let Some(hi) = lo.checked_add(len as u64) else {
        return;
    };
    if hi > arena_size() || hi <= lo {
        return;
    }
    let mut ranges = committed_ranges()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let released = update_commit_refs(&mut ranges, lo, hi, false);
    if released.is_empty() {
        return;
    }
    for &(released_lo, released_hi) in &released {
        mark_committed_pages(released_lo, released_hi, false);
    }
    wait_for_direct_readers();
    note_commit_changes(&released);
    for &(released_lo, released_hi) in &released {
        let released_ptr = unsafe { base.add(released_lo as usize) };
        sys::decommit(released_ptr, (released_hi - released_lo) as usize);
        #[cfg(windows)]
        crate::soft_watch::mark_dirty_host(released_ptr, (released_hi - released_lo) as usize);
    }
    note_commit_changes(&released);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteWatchResult {
    Clean,
    Dirty,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WriteWatchObservation {
    pub result: WriteWatchResult,
    pub serial_after: u64,
    pub generation_before: u64,
    pub generation_after: u64,
}

impl WriteWatchObservation {
    fn unavailable() -> Self {
        Self {
            result: WriteWatchResult::Unavailable,
            serial_after: observed_write_serial(),
            generation_before: 0,
            generation_after: 0,
        }
    }
}

pub fn write_watch_query_range(va: u64, len: usize) -> Option<(u64, usize)> {
    const PAGE_MASK: u64 = 0xfff;
    if len == 0 {
        return None;
    }
    let lo = va & !PAGE_MASK;
    let end = va.checked_add(len as u64)?;
    let hi = end.checked_add(PAGE_MASK)? & !PAGE_MASK;
    if hi <= lo || hi > arena_size() {
        return None;
    }
    Some((lo, usize::try_from(hi - lo).ok()?))
}

/// Page-aligned query range for the Linux backend, which also accepts VAs in
/// the direct-mode window. Kept separate from `write_watch_query_range`:
/// callers use that one to decide whether a VA may be read as an arena
/// offset, which must not start succeeding for direct-mode host addresses.
#[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
fn linux_watch_query_range(va: u64, len: usize) -> Option<(u64, usize)> {
    const PAGE_MASK: u64 = 0xfff;
    if len == 0 {
        return None;
    }
    let lo = va & !PAGE_MASK;
    let end = va.checked_add(len as u64)?;
    let hi = end.checked_add(PAGE_MASK)? & !PAGE_MASK;
    let (window_lo, window_hi) = va_window();
    if hi <= lo || lo < window_lo || hi > window_hi {
        return None;
    }
    Some((lo, usize::try_from(hi - lo).ok()?))
}

/// Query-and-reset over the arena for Linux. Holds the commit lock for the
/// whole operation so a concurrent decommit cannot swap the mapping mid-scan,
/// and publishes the reset pages to the observed-write log before releasing
/// it, exactly like the Windows path.
#[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
fn linux_take_write_watch(
    va: u64,
    len: usize,
    spans: Option<&mut Vec<(u64, usize)>>,
    observe: bool,
) -> WriteWatchObservation {
    if !ARENA_WRITE_WATCH.load(Ordering::Acquire) {
        return WriteWatchObservation::unavailable();
    }
    let Some(watch) = crate::linux_watch::global() else {
        return WriteWatchObservation::unavailable();
    };
    let Some((lo, query_len)) = linux_watch_query_range(va, len) else {
        return WriteWatchObservation::unavailable();
    };
    let base = arena();
    let Some(offset) = va_offset(lo) else {
        return WriteWatchObservation::unavailable();
    };
    let offset_hi = offset + query_len as u64;
    if base.is_null() || offset_hi > arena_size() {
        return WriteWatchObservation::unavailable();
    }
    thread_local! {
        static DIRTY_PAGES: std::cell::RefCell<Vec<u64>> = const {
            std::cell::RefCell::new(Vec::new())
        };
    }
    let ranges = committed_ranges()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !is_committed(&ranges, offset, offset_hi) {
        return WriteWatchObservation::unavailable();
    }
    let generation_before = if observe {
        observed_write_snapshot_range(lo, query_len).1
    } else {
        0
    };
    let host_lo = base as usize + offset as usize;
    let result = DIRTY_PAGES.with(|pages| {
        let mut pages = pages.borrow_mut();
        pages.clear();
        let scan = watch.take(host_lo, query_len, |run_lo, run_hi| {
            let mut page = lo + (run_lo - host_lo) as u64;
            let end = lo + (run_hi - host_lo) as u64;
            while page < end {
                pages.push(page);
                page += 4096;
            }
        });
        if let Err(errno) = scan {
            // Some pages may already have been reset without being reported.
            static LOGGED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "fastmem: write-watch scan failed (errno {errno}) at va={lo:#x} len={query_len:#x}; treating range as dirty"
                );
            }
            pages.clear();
            pages.extend((0..query_len as u64 >> 12).map(|index| lo + index * 4096));
        }
        if pages.is_empty() {
            return WriteWatchResult::Clean;
        }
        record_observed_write_vas(&pages);
        if let Some(spans) = spans {
            for &page_va in pages.iter() {
                match spans.last_mut() {
                    Some((last_va, last_len)) if *last_va + *last_len as u64 == page_va => {
                        *last_len += 4096;
                    }
                    _ => spans.push((page_va, 4096)),
                }
            }
        }
        WriteWatchResult::Dirty
    });
    let (serial_after, generation_after) = if observe {
        observed_write_snapshot_range(lo, query_len)
    } else {
        (0, 0)
    };
    drop(ranges);
    WriteWatchObservation {
        result,
        serial_after,
        generation_before,
        generation_after,
    }
}

pub fn take_write_watch(va: u64, len: usize) -> WriteWatchResult {
    #[cfg(target_vendor = "sony")]
    {
        return soft_take_write_watch(va, len, None);
    }

    #[cfg(not(any(windows, target_vendor = "sony", target_os = "linux")))]
    {
        let _ = (va, len);
        return WriteWatchResult::Unavailable;
    }

    #[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
    {
        return linux_take_write_watch(va, len, None, false).result;
    }

    #[cfg(windows)]
    {
        if crate::soft_watch::enabled() {
            return soft_take_write_watch(va, len, None);
        }
        if !ARENA_WRITE_WATCH.load(Ordering::Acquire) {
            return WriteWatchResult::Unavailable;
        }
        let Some((lo, query_len)) = write_watch_query_range(va, len) else {
            return WriteWatchResult::Unavailable;
        };
        if write_watch_memo_covers(lo, query_len) {
            return WriteWatchResult::Clean;
        }
        let hi = lo + query_len as u64;
        let base = arena();
        if base.is_null() {
            return WriteWatchResult::Unavailable;
        }
        let ranges = committed_ranges()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !is_committed(&ranges, lo, hi) {
            return WriteWatchResult::Unavailable;
        }
        let page_count = query_len >> 12;
        thread_local! {
            static ADDRESSES: std::cell::RefCell<Vec<usize>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }
        let result = ADDRESSES.with(|addresses| {
            let mut addresses = addresses.borrow_mut();
            if addresses.len() < page_count {
                addresses.resize(page_count, 0);
            }
            let addresses = &mut addresses[..page_count];
            let ptr = unsafe { base.add(lo as usize) };
            match sys::take_write_watch(ptr, query_len, addresses) {
                Some(0) => WriteWatchResult::Clean,
                Some(count) => {
                    record_observed_write_pages(base, &addresses[..count.min(page_count)]);
                    WriteWatchResult::Dirty
                }
                None => WriteWatchResult::Unavailable,
            }
        });
        drop(ranges);
        if result != WriteWatchResult::Unavailable {
            write_watch_memo_insert(lo, query_len);
        }
        result
    }
}

pub fn take_write_watch_spans(
    va: u64,
    len: usize,
    spans: &mut Vec<(u64, usize)>,
) -> WriteWatchResult {
    #[cfg(target_vendor = "sony")]
    {
        return soft_take_write_watch(va, len, Some(spans));
    }

    #[cfg(not(any(windows, target_vendor = "sony", target_os = "linux")))]
    {
        let _ = (va, len, spans);
        return WriteWatchResult::Unavailable;
    }

    #[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
    {
        return linux_take_write_watch(va, len, Some(spans), false).result;
    }

    #[cfg(windows)]
    {
        if crate::soft_watch::enabled() {
            return soft_take_write_watch(va, len, Some(spans));
        }
        if !ARENA_WRITE_WATCH.load(Ordering::Acquire) {
            return WriteWatchResult::Unavailable;
        }
        let Some((lo, query_len)) = write_watch_query_range(va, len) else {
            return WriteWatchResult::Unavailable;
        };
        if write_watch_memo_covers(lo, query_len) {
            return WriteWatchResult::Clean;
        }
        let hi = lo + query_len as u64;
        let base = arena();
        if base.is_null() {
            return WriteWatchResult::Unavailable;
        }
        let ranges = committed_ranges()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !is_committed(&ranges, lo, hi) {
            return WriteWatchResult::Unavailable;
        }
        let page_count = query_len >> 12;
        thread_local! {
            static SPAN_ADDRESSES: std::cell::RefCell<Vec<usize>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }
        let result = SPAN_ADDRESSES.with(|addresses| {
            let mut addresses = addresses.borrow_mut();
            if addresses.len() < page_count {
                addresses.resize(page_count, 0);
            }
            let addresses = &mut addresses[..page_count];
            let ptr = unsafe { base.add(lo as usize) };
            match sys::take_write_watch(ptr, query_len, addresses) {
                Some(0) => WriteWatchResult::Clean,
                Some(count) => {
                    let base_addr = base as usize as u64;
                    let dirty_addresses = &addresses[..count.min(page_count)];
                    record_observed_write_pages(base, dirty_addresses);
                    for &addr in dirty_addresses.iter() {
                        let page_va = (addr as u64).saturating_sub(base_addr) & !0xfffu64;
                        match spans.last_mut() {
                            Some((last_va, last_len)) if *last_va + *last_len as u64 == page_va => {
                                *last_len += 4096;
                            }
                            _ => spans.push((page_va, 4096)),
                        }
                    }
                    WriteWatchResult::Dirty
                }
                None => WriteWatchResult::Unavailable,
            }
        });
        drop(ranges);
        if result != WriteWatchResult::Unavailable {
            write_watch_memo_insert(lo, query_len);
        }
        result
    }
}

pub fn take_write_watch_spans_observed(
    va: u64,
    len: usize,
    spans: &mut Vec<(u64, usize)>,
) -> WriteWatchObservation {
    #[cfg(target_vendor = "sony")]
    {
        return soft_take_write_watch_observed(va, len, spans);
    }

    #[cfg(not(any(windows, target_vendor = "sony", target_os = "linux")))]
    {
        let _ = (va, len, spans);
        return WriteWatchObservation::unavailable();
    }

    #[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
    {
        return linux_take_write_watch(va, len, Some(spans), true);
    }

    #[cfg(windows)]
    {
        if crate::soft_watch::enabled() {
            return soft_take_write_watch_observed(va, len, spans);
        }
        if !ARENA_WRITE_WATCH.load(Ordering::Acquire) {
            return WriteWatchObservation::unavailable();
        }
        let Some((lo, query_len)) = write_watch_query_range(va, len) else {
            return WriteWatchObservation::unavailable();
        };
        if write_watch_memo_covers(lo, query_len) {
            let (serial_after, generation) = observed_write_snapshot_range(lo, query_len);
            return WriteWatchObservation {
                result: WriteWatchResult::Clean,
                serial_after,
                generation_before: generation,
                generation_after: generation,
            };
        }
        let hi = lo + query_len as u64;
        let base = arena();
        if base.is_null() {
            return WriteWatchObservation::unavailable();
        }
        let ranges = committed_ranges()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !is_committed(&ranges, lo, hi) {
            return WriteWatchObservation::unavailable();
        }
        let (_, generation_before) = observed_write_snapshot_range(lo, query_len);
        let page_count = query_len >> 12;
        thread_local! {
            static SPAN_ADDRESSES: std::cell::RefCell<Vec<usize>> = const {
                std::cell::RefCell::new(Vec::new())
            };
        }
        let result = SPAN_ADDRESSES.with(|addresses| {
            let mut addresses = addresses.borrow_mut();
            if addresses.len() < page_count {
                addresses.resize(page_count, 0);
            }
            let addresses = &mut addresses[..page_count];
            let ptr = unsafe { base.add(lo as usize) };
            match sys::take_write_watch(ptr, query_len, addresses) {
                Some(0) => WriteWatchResult::Clean,
                Some(count) => {
                    let base_addr = base as usize as u64;
                    let dirty_addresses = &addresses[..count.min(page_count)];
                    record_observed_write_pages(base, dirty_addresses);
                    for &addr in dirty_addresses.iter() {
                        let page_va = (addr as u64).saturating_sub(base_addr) & !0xfffu64;
                        match spans.last_mut() {
                            Some((last_va, last_len)) if *last_va + *last_len as u64 == page_va => {
                                *last_len += 4096;
                            }
                            _ => spans.push((page_va, 4096)),
                        }
                    }
                    WriteWatchResult::Dirty
                }
                None => WriteWatchResult::Unavailable,
            }
        });
        let (serial_after, generation_after) = observed_write_snapshot_range(lo, query_len);
        drop(ranges);
        if result != WriteWatchResult::Unavailable {
            write_watch_memo_insert(lo, query_len);
        }
        WriteWatchObservation {
            result,
            serial_after,
            generation_before,
            generation_after,
        }
    }
}

fn committed_ranges() -> &'static Mutex<Vec<CommittedRange>> {
    COMMITTED_RANGES.get_or_init(|| Mutex::new(Vec::new()))
}

const COMMIT_PAGE_SHIFT: u32 = 12;
const DIRECT_READER_SLOTS: usize = 64;

static COMMIT_BITMAP: OnceLock<Box<[std::sync::atomic::AtomicU64]>> = OnceLock::new();
#[repr(align(128))]
struct DirectReaderSlot(std::sync::atomic::AtomicU32);

static DIRECT_READERS: [DirectReaderSlot; DIRECT_READER_SLOTS] =
    [const { DirectReaderSlot(std::sync::atomic::AtomicU32::new(0)) }; DIRECT_READER_SLOTS];
static NEXT_DIRECT_SLOT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static DIRECT_SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

fn commit_bitmap() -> &'static [std::sync::atomic::AtomicU64] {
    COMMIT_BITMAP.get_or_init(|| {
        let pages = (arena_size() >> COMMIT_PAGE_SHIFT) as usize;
        let words = pages.div_ceil(64).max(1);
        std::iter::repeat_with(|| std::sync::atomic::AtomicU64::new(0))
            .take(words)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

fn mark_committed_pages(lo: u64, hi: u64, committed: bool) {
    if hi <= lo {
        return;
    }
    let bitmap = commit_bitmap();
    let first = lo >> COMMIT_PAGE_SHIFT;
    let last = (hi - 1) >> COMMIT_PAGE_SHIFT;
    for page in first..=last {
        let Some(word) = bitmap.get((page / 64) as usize) else {
            break;
        };
        let bit = 1u64 << (page % 64);
        if committed {
            word.fetch_or(bit, Ordering::SeqCst);
        } else {
            word.fetch_and(!bit, Ordering::SeqCst);
        }
    }
}

fn pages_committed(lo: u64, hi: u64) -> bool {
    if hi <= lo {
        return false;
    }
    let bitmap = commit_bitmap();
    let first = lo >> COMMIT_PAGE_SHIFT;
    let last = (hi - 1) >> COMMIT_PAGE_SHIFT;
    for page in first..=last {
        let Some(word) = bitmap.get((page / 64) as usize) else {
            return false;
        };
        if word.load(Ordering::SeqCst) & (1u64 << (page % 64)) == 0 {
            return false;
        }
    }
    true
}

fn direct_reader_slot() -> &'static std::sync::atomic::AtomicU32 {
    let index = DIRECT_SLOT.with(|slot| {
        let current = slot.get();
        if current != usize::MAX {
            return current;
        }
        let assigned = NEXT_DIRECT_SLOT.fetch_add(1, Ordering::Relaxed) % DIRECT_READER_SLOTS;
        slot.set(assigned);
        assigned
    });
    &DIRECT_READERS[index].0
}

fn wait_for_direct_readers() {
    let mut spins = 0u32;
    while DIRECT_READERS
        .iter()
        .any(|slot| slot.0.load(Ordering::SeqCst) != 0)
    {
        spins = spins.wrapping_add(1);
        if spins % 64 == 0 {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

pub fn read_committed(va: u64, buf: &mut [u8]) -> bool {
    let base = arena();
    if base.is_null() || buf.is_empty() {
        return false;
    }
    let Some(lo) = va_offset(va) else {
        return false;
    };
    let Some(hi) = lo.checked_add(buf.len() as u64) else {
        return false;
    };
    if hi > arena_size() {
        return false;
    }
    let slot = direct_reader_slot();
    slot.fetch_add(1, Ordering::SeqCst);
    let committed = pages_committed(lo, hi);
    if committed {
        unsafe {
            std::ptr::copy_nonoverlapping(base.add(lo as usize), buf.as_mut_ptr(), buf.len());
        }
    }
    slot.fetch_sub(1, Ordering::SeqCst);
    committed
}

fn push_commit_segment(segments: &mut Vec<CommittedRange>, lo: u64, hi: u64, refs: u32) {
    if hi <= lo || refs == 0 {
        return;
    }
    if let Some(last) = segments.last_mut() {
        if last.hi == lo && last.refs == refs {
            last.hi = hi;
            return;
        }
    }
    segments.push(CommittedRange { lo, hi, refs });
}

fn note_released(released: &mut Vec<(u64, u64)>, lo: u64, hi: u64) {
    if hi <= lo {
        return;
    }
    if let Some((_, released_hi)) = released.last_mut() {
        if *released_hi == lo {
            *released_hi = hi;
            return;
        }
    }
    released.push((lo, hi));
}

fn update_commit_refs(
    ranges: &mut Vec<CommittedRange>,
    lo: u64,
    hi: u64,
    increment: bool,
) -> Vec<(u64, u64)> {
    let mut released = Vec::new();
    if hi <= lo {
        return released;
    }
    let adjust = |refs: u32| {
        if increment {
            refs.saturating_add(1)
        } else {
            refs.saturating_sub(1)
        }
    };
    let start = ranges.partition_point(|range| range.hi <= lo);
    let end = ranges.partition_point(|range| range.lo < hi);
    let mut segments = Vec::with_capacity(end.saturating_sub(start).saturating_mul(2).saturating_add(3));
    let mut cursor = lo;
    for range in &ranges[start..end] {
        if range.lo < lo {
            push_commit_segment(&mut segments, range.lo, lo, range.refs);
        }
        let covered_lo = range.lo.max(lo);
        let covered_hi = range.hi.min(hi);
        push_commit_segment(&mut segments, cursor, covered_lo, adjust(0));
        let refs = adjust(range.refs);
        if refs == 0 {
            note_released(&mut released, covered_lo, covered_hi);
        }
        push_commit_segment(&mut segments, covered_lo, covered_hi, refs);
        if range.hi > hi {
            push_commit_segment(&mut segments, hi, range.hi, range.refs);
        }
        cursor = covered_hi;
    }
    push_commit_segment(&mut segments, cursor, hi, adjust(0));
    let mut splice_start = start;
    let mut splice_end = end;
    if let (Some(previous), Some(first)) = (
        splice_start.checked_sub(1).map(|index| ranges[index]),
        segments.first_mut(),
    ) {
        if previous.hi == first.lo && previous.refs == first.refs {
            first.lo = previous.lo;
            splice_start -= 1;
        }
    }
    if let (Some(next), Some(last)) = (ranges.get(splice_end).copied(), segments.last_mut()) {
        if last.hi == next.lo && last.refs == next.refs {
            last.hi = next.hi;
            splice_end += 1;
        }
    }
    ranges.splice(splice_start..splice_end, segments);
    released
}

fn is_committed(ranges: &[CommittedRange], lo: u64, hi: u64) -> bool {
    let mut cursor = lo;
    for range in &ranges[ranges.partition_point(|range| range.hi <= lo)..] {
        if range.hi <= cursor {
            continue;
        }
        if range.lo > cursor {
            return false;
        }
        cursor = cursor.max(range.hi);
        if cursor >= hi {
            return true;
        }
    }
    false
}

use std::sync::atomic::AtomicU64;

static WATCH_LO: AtomicU64 = AtomicU64::new(0);
static WATCH_HI: AtomicU64 = AtomicU64::new(0);
static WATCH_EXACT_LO: AtomicU64 = AtomicU64::new(0);
static WATCH_EXACT_HI: AtomicU64 = AtomicU64::new(0);
static GUEST_PROBE_EVENT: AtomicU64 = AtomicU64::new(0);

pub fn mark_guest_probe_event(va: u64) {
    if va != 0 {
        GUEST_PROBE_EVENT.store(va, Ordering::SeqCst);
    }
}

pub fn take_guest_probe_event() -> Option<u64> {
    let va = GUEST_PROBE_EVENT.swap(0, Ordering::SeqCst);
    (va != 0).then_some(va)
}

pub fn watch_arm(va: u64, len: u64) -> bool {
    let lo = va & !0xFFF;
    let hi = (va + len + 0xFFF) & !0xFFF;
    let (window_lo, window_hi) = va_window();
    if lo < window_lo || hi > window_hi {
        return false;
    }
    let Some(ptr) = host_ptr(lo) else {
        return false;
    };
    if !sys::protect(ptr, (hi - lo) as usize, true) {
        return false;
    }
    WATCH_LO.store(lo, Ordering::SeqCst);
    WATCH_HI.store(hi, Ordering::SeqCst);
    WATCH_EXACT_LO.store(va, Ordering::SeqCst);
    WATCH_EXACT_HI.store(va.saturating_add(len), Ordering::SeqCst);
    true
}

pub fn watch_mark(va: u64, len: u64) -> bool {
    let lo = va & !0xFFF;
    let hi = (va + len + 0xFFF) & !0xFFF;
    let (window_lo, window_hi) = va_window();
    if hi <= lo || lo < window_lo || hi > window_hi {
        return false;
    }
    WATCH_LO.store(lo, Ordering::SeqCst);
    WATCH_HI.store(hi, Ordering::SeqCst);
    WATCH_EXACT_LO.store(va, Ordering::SeqCst);
    WATCH_EXACT_HI.store(va.saturating_add(len), Ordering::SeqCst);
    true
}

pub fn watch_range() -> Option<(u64, u64)> {
    let lo = WATCH_LO.load(Ordering::SeqCst);
    let hi = WATCH_HI.load(Ordering::SeqCst);
    if hi > lo {
        Some((lo, hi))
    } else {
        None
    }
}

pub fn watch_exact_range() -> Option<(u64, u64)> {
    let lo = WATCH_EXACT_LO.load(Ordering::SeqCst);
    let hi = WATCH_EXACT_HI.load(Ordering::SeqCst);
    if hi > lo {
        Some((lo, hi))
    } else {
        watch_range()
    }
}

pub fn watch_reprotect() -> bool {
    if let Some((lo, hi)) = watch_range() {
        let Some(ptr) = host_ptr(lo) else {
            return false;
        };
        return sys::protect(ptr, (hi - lo) as usize, true);
    }
    false
}

pub fn watch_disarm() {
    if let Some((lo, hi)) = watch_range() {
        if let Some(ptr) = host_ptr(lo) {
            sys::protect(ptr, (hi - lo) as usize, false);
        }
    }
    WATCH_LO.store(0, Ordering::SeqCst);
    WATCH_HI.store(0, Ordering::SeqCst);
    WATCH_EXACT_LO.store(0, Ordering::SeqCst);
    WATCH_EXACT_HI.store(0, Ordering::SeqCst);
}

pub fn watch_write_through(addr: u64, size: usize, value: u64) -> bool {
    let Some((lo, hi)) = watch_range() else {
        return false;
    };
    if addr < lo || addr + size as u64 > hi {
        return false;
    }
    let (Some(ptr), Some(dst)) = (host_ptr(lo), host_ptr(addr)) else {
        return false;
    };
    if !sys::protect(ptr, (hi - lo) as usize, false) {
        return false;
    }
    unsafe {
        let bytes = value.to_le_bytes();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, size.min(8));
    }
    let _ = sys::protect(ptr, (hi - lo) as usize, true);
    true
}

#[cfg(all(test, any(windows, all(target_os = "linux", not(target_vendor = "sony")))))]
mod tests {
    use super::*;

    #[test]
    fn direct_reads_follow_commit_state() {
        let va = 0xD_EAD0_0000u64;
        let len = 0x3000usize;
        let Some(ptr) = commit(va, len) else {
            return;
        };
        let pattern: Vec<u8> = (0..len).map(|index| (index * 7 % 251) as u8).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), ptr, len);
        }
        let mut out = vec![0u8; 0x1800];
        assert!(read_committed(va + 0x800, &mut out));
        assert_eq!(out, pattern[0x800..0x2000]);
        let mut past_end = vec![0u8; 0x1000];
        assert!(!read_committed(va + len as u64 - 0x800, &mut past_end));
        decommit(ptr, len);
        let mut after = vec![0u8; 16];
        assert!(!read_committed(va, &mut after));
    }

    #[test]
    fn write_watch_memo_covers_only_queried_pages_within_an_epoch() {
        let mut memo = WriteWatchMemo {
            active: true,
            ..WriteWatchMemo::default()
        };
        memo.insert(0x1_0000, 0x1_4000);
        assert!(memo.covers(0x1_0000, 0x1_4000));
        assert!(memo.covers(0x1_1000, 0x1_2000));
        assert!(!memo.covers(0x1_0000, 0x1_5000));
        assert!(!memo.covers(0x0_f000, 0x1_1000));
        memo.insert(0x1_f000, 0x3_1000);
        assert!(memo.covers(0x2_0000, 0x3_0000));
        assert!(memo.covers(0x1_f000, 0x3_1000));
        assert!(!memo.covers(0x3_1000, 0x3_2000));
        let epoch = memo.epoch;
        bump_write_watch_epoch();
        assert!(!memo.covers(0x1_0000, 0x1_4000));
        assert_ne!(memo.epoch, epoch);
        let mut inactive = WriteWatchMemo::default();
        inactive.insert(0x1_0000, 0x1_4000);
        assert!(!inactive.covers(0x1_0000, 0x1_4000));
    }

    /// True when the arena is write-watched. A host that cannot provide it
    /// fails the test loudly unless the skip was requested explicitly.
    fn require_watch() -> bool {
        let _ = arena();
        if write_watch_available() {
            return true;
        }
        if std::env::var_os("NEXIUM_TEST_ALLOW_NO_WRITE_WATCH").is_some() {
            eprintln!("SKIPPED: fastmem write-watch unavailable on this host");
            return false;
        }
        #[cfg(target_os = "linux")]
        if crate::linux_watch::disabled_by_env() {
            eprintln!("SKIPPED: fastmem write-watch disabled by NEXIUM_NO_WRITE_WATCH");
            return false;
        }
        panic!(
            "fastmem write-watch unavailable on this host (see the fastmem/write-watch log line). \
             Set NEXIUM_TEST_ALLOW_NO_WRITE_WATCH=1 to skip the tests that need it."
        );
    }

    fn reference_update_commit_refs(
        ranges: &mut Vec<CommittedRange>,
        lo: u64,
        hi: u64,
        increment: bool,
    ) -> Vec<(u64, u64)> {
        let mut boundaries: Vec<u64> = ranges.iter().flat_map(|range| [range.lo, range.hi]).collect();
        boundaries.extend([lo, hi]);
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut rebuilt: Vec<CommittedRange> = Vec::new();
        let mut released = Vec::new();
        for segment in boundaries.windows(2) {
            let (segment_lo, segment_hi) = (segment[0], segment[1]);
            let old_refs = ranges
                .iter()
                .find(|range| range.lo <= segment_lo && range.hi >= segment_hi)
                .map_or(0, |range| range.refs);
            let new_refs = if segment_lo >= lo && segment_hi <= hi {
                if increment {
                    old_refs.saturating_add(1)
                } else {
                    old_refs.saturating_sub(1)
                }
            } else {
                old_refs
            };
            if old_refs != 0 && new_refs == 0 {
                note_released(&mut released, segment_lo, segment_hi);
            }
            push_commit_segment(&mut rebuilt, segment_lo, segment_hi, new_refs);
        }
        *ranges = rebuilt;
        released
    }

    fn reference_is_committed(ranges: &[CommittedRange], lo: u64, hi: u64) -> bool {
        let mut cursor = lo;
        for range in ranges {
            if range.hi <= cursor {
                continue;
            }
            if range.lo > cursor {
                return false;
            }
            cursor = cursor.max(range.hi);
            if cursor >= hi {
                return true;
            }
        }
        false
    }

    #[test]
    fn commit_changes_report_the_newest_overlapping_generation() {
        let mut changes = CommitChanges::new();
        record_commit_change(&mut changes, 0x1000, 0x5000, 1);
        record_commit_change(&mut changes, 0x3000, 0x4000, 2);
        record_commit_change(&mut changes, 0x8000, 0x9000, 3);
        assert_eq!(commit_change_generation(&changes, 0x1000, 0x2000), 1);
        assert_eq!(commit_change_generation(&changes, 0x2000, 0x3800), 2);
        assert_eq!(commit_change_generation(&changes, 0x4000, 0x5000), 1);
        assert_eq!(commit_change_generation(&changes, 0x5000, 0x8000), 0);
        assert_eq!(commit_change_generation(&changes, 0x0, 0x10000), 3);
        record_commit_change(&mut changes, 0x0, 0x10000, 4);
        assert_eq!(changes.len(), 1);
        assert_eq!(commit_change_generation(&changes, 0x5000, 0x6000), 4);
        record_commit_change(&mut changes, 0x2000, 0x3000, 5);
        assert_eq!(changes.len(), 3);
        assert_eq!(commit_change_generation(&changes, 0x1000, 0x2000), 4);
        assert_eq!(commit_change_generation(&changes, 0x2fff, 0x3001), 5);
    }

    #[test]
    fn commit_refs_match_the_reference_merge() {
        let mut state = 0x853c_49e6_748f_ea9bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..40 {
            let mut fast = Vec::new();
            let mut reference = Vec::new();
            for _ in 0..400 {
                let lo = next() % 96;
                let hi = lo + next() % 24;
                let increment = next() % 3 != 0;
                let fast_released = update_commit_refs(&mut fast, lo, hi, increment);
                let reference_released = reference_update_commit_refs(&mut reference, lo, hi, increment);
                assert_eq!(fast, reference, "lo={lo} hi={hi} increment={increment}");
                assert_eq!(fast_released, reference_released, "lo={lo} hi={hi} increment={increment}");
                let query_lo = next() % 100;
                let query_hi = query_lo + next() % 16;
                assert_eq!(
                    is_committed(&fast, query_lo, query_hi),
                    reference_is_committed(&reference, query_lo, query_hi),
                    "query {query_lo}..{query_hi}"
                );
                assert_eq!(
                    fast.get(fast.partition_point(|range| range.hi <= query_lo))
                        .is_some_and(|range| range.lo < query_hi),
                    reference.iter().any(|range| range.lo < query_hi && query_lo < range.hi),
                    "any {query_lo}..{query_hi}"
                );
            }
        }
    }

    #[test]
    fn write_watch_detects_direct_fastmem_stores_and_resets_atomically() {
        const VA: u64 = 0xf0_0000_0000;
        const LEN: usize = 0x2000;
        if !require_watch() {
            return;
        }
        let ptr = commit(VA, LEN).expect("write-watched fastmem allocation");

        let _ = take_write_watch(VA, LEN);
        let (serial_before, generation_before) = observed_write_snapshot_range(VA, LEN);
        assert!(generation_before <= serial_before);
        unsafe { ptr.add(0x123).write_volatile(0x5a) };
        assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Dirty);
        let published_serial = observed_write_serial();
        let (serial_after, generation_after) = observed_write_snapshot_range(VA, LEN);
        assert!(serial_after >= published_serial);
        assert!(generation_after <= serial_after);
        assert!(generation_after > generation_before);
        assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
        assert_eq!(observed_write_generation_range(VA, LEN), generation_after);

        decommit(ptr, LEN);
        assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);
    }

    #[test]
    fn observed_span_take_brackets_scoped_generation() {
        const VA: u64 = 0xf3_0000_0000;
        const LEN: usize = 0x1_0000;
        if !require_watch() {
            return;
        }
        let ptr = commit(VA, LEN).expect("write-watched fastmem allocation");
        let _ = take_write_watch(VA, LEN);
        unsafe { ptr.add(0x100).write_volatile(0x31) };
        assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Dirty);
        let consumed_generation = observed_write_generation_range(VA, LEN);
        unsafe { ptr.add(0x2100).write_volatile(0x72) };
        let mut spans = Vec::new();

        let observation = take_write_watch_spans_observed(VA, LEN, &mut spans);

        assert_eq!(observation.result, WriteWatchResult::Dirty);
        assert_eq!(observation.generation_before, consumed_generation);
        assert!(observation.generation_after > observation.generation_before);
        assert!(observation.serial_after >= observation.generation_after);
        assert_eq!(spans, vec![(VA + 0x2000, 0x1000)]);
        assert_eq!(observed_write_pages_changed_since(VA, LEN, consumed_generation), 1);
        assert_eq!(observed_write_generation_pages(VA, 0x1000), consumed_generation);
        assert_eq!(observed_write_generation_pages(VA + 0x2000, 1), observation.generation_after);
        let mut replay = Vec::new();
        observed_write_spans_since(VA + 1, LEN - 1, consumed_generation, &mut replay);
        assert_eq!(replay, spans);
        decommit(ptr, LEN);
    }

    #[test]
    fn observed_write_chunks_match_page_scan_across_sparse_and_partial_ranges() {
        let mut chunks = ObservedWriteChunks::default();
        let mut expected_pages = std::collections::BTreeMap::new();
        for (page, generation) in [(0x0000, 1), (0xf000, 2), (0x10000, 3), (0x1f000, 4), (0x40000, 7)] {
            let chunk = chunks.entry(page & OBSERVED_WRITE_PAGE_MASK).or_default();
            chunk.generation = chunk.generation.max(generation);
            chunk.pages[((page >> 12) & 15) as usize] = generation;
            expected_pages.insert(page, generation);
        }
        for (va, len) in [(0, 0x50000), (1, 0xf000), (0xf123, 0x11000), (0x10001, 1), (0x20000, 0x10000), (0x40001, 0xfff)] {
            for generation in 0..=8 {
                let mut actual = Vec::new();
                visit_observed_pages(&chunks, va, va + len, generation, |page, written| actual.push((page, written)));
                let expected: Vec<_> = expected_pages.iter().filter_map(|(&page, &written)| {
                    (page >= (va & OBSERVED_WRITE_PAGE_4K_MASK) && page < va + len && written > generation)
                        .then_some((page, written))
                }).collect();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn overlapping_fastmem_leases_do_not_decommit_each_other() {
        const VA: u64 = 0xf1_0000_0000;
        const LEN: usize = 0x1000;
        if !require_watch() {
            return;
        }
        let first = commit(VA, LEN).expect("first fastmem lease");
        let second = commit(VA, LEN).expect("overlapping fastmem lease");
        assert_eq!(first, second);
        unsafe { first.write_volatile(0xa5) };

        decommit(first, LEN);
        assert_eq!(unsafe { second.read_volatile() }, 0xa5);
        assert_ne!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);

        decommit(second, LEN);
        assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);
    }

    #[cfg(target_os = "linux")]
    mod linux {
        use super::*;

        fn spans_of(va: u64, len: usize) -> (WriteWatchResult, Vec<(u64, usize)>) {
            let mut spans = Vec::new();
            let result = take_write_watch_spans(va, len, &mut spans);
            (result, spans)
        }

        #[test]
        fn probe_reports_available_and_untouched_range_reads_clean() {
            const VA: u64 = 0xe3_0000_0000;
            const LEN: usize = 0x8000;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            // Fresh pages have no tracking state yet: dirty, never clean.
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Dirty);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            // Reads do not dirty.
            assert_eq!(unsafe { ptr.add(0x3000).read_volatile() }, 0);
            assert_eq!(spans_of(VA, LEN), (WriteWatchResult::Clean, vec![]));
            decommit(ptr, LEN);
        }

        #[test]
        fn write_dirties_exactly_the_containing_pages_and_take_resets() {
            const VA: u64 = 0xe3_1000_0000;
            const LEN: usize = 0x10_000;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            let _ = take_write_watch(VA, LEN);
            unsafe {
                ptr.add(0x2ffc).cast::<u64>().write_unaligned(u64::MAX);
                ptr.add(0x9000).write_volatile(1);
                ptr.add(0xffff).write_volatile(1);
            }
            assert_eq!(
                spans_of(VA, LEN),
                (
                    WriteWatchResult::Dirty,
                    vec![(VA + 0x2000, 0x2000), (VA + 0x9000, 0x1000), (VA + 0xf000, 0x1000)]
                )
            );
            assert_eq!(spans_of(VA, LEN), (WriteWatchResult::Clean, vec![]));
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            unsafe { ptr.add(0x9001).write_volatile(2) };
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Dirty);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            decommit(ptr, LEN);
        }

        #[test]
        fn unaligned_sub_range_queries_reset_only_their_own_pages() {
            const VA: u64 = 0xe3_2000_0000;
            const LEN: usize = 0x8000;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            let _ = take_write_watch(VA, LEN);
            unsafe {
                ptr.add(0x1010).write_volatile(1);
                ptr.add(0x5010).write_volatile(1);
            }
            // [0x1800, 0x2800) covers pages 1 and 2 only.
            assert_eq!(
                spans_of(VA + 0x1800, 0x1000),
                (WriteWatchResult::Dirty, vec![(VA + 0x1000, 0x1000)])
            );
            assert_eq!(take_write_watch(VA + 0x1800, 0x1000), WriteWatchResult::Clean);
            assert_eq!(take_write_watch(VA + 0x6000, 0x2000), WriteWatchResult::Clean);
            // Page 5 is still pending for whoever asks about it; spans append.
            let mut spans = vec![(1, 2)];
            assert_eq!(take_write_watch_spans(VA, LEN, &mut spans), WriteWatchResult::Dirty);
            assert_eq!(spans, vec![(1, 2), (VA + 0x5000, 0x1000)]);
            assert_eq!(take_write_watch(VA, 0), WriteWatchResult::Unavailable);
            decommit(ptr, LEN);
        }

        #[test]
        fn writes_from_other_threads_are_seen_and_never_lost_under_contention() {
            const VA: u64 = 0xe3_3000_0000;
            const PAGES: usize = 64;
            const LEN: usize = PAGES * 0x1000;
            const ROUNDS: usize = 200;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            let _ = take_write_watch(VA, LEN);
            let addr = ptr as usize;
            std::thread::spawn(move || unsafe { ((addr + 0x7123) as *mut u8).write_volatile(9) })
                .join()
                .unwrap();
            assert_eq!(
                spans_of(VA, LEN),
                (WriteWatchResult::Dirty, vec![(VA + 0x7000, 0x1000)])
            );

            // Several writers each bump a per-page counter while this thread
            // keeps taking. Every write must be covered by a later take: after
            // the writers stop, the sum of what takes reported is the truth.
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let writers: Vec<_> = (0..4usize)
                .map(|writer| {
                    std::thread::spawn(move || {
                        for round in 0..ROUNDS {
                            for page in (writer..PAGES).step_by(4) {
                                let slot = (addr + page * 0x1000 + 8) as *mut u32;
                                unsafe { slot.write_volatile(round as u32 + 1) };
                            }
                        }
                    })
                })
                .collect();
            let mut snapshot = vec![0u32; PAGES];
            let refresh = |snapshot: &mut Vec<u32>| {
                let (result, spans) = spans_of(VA, LEN);
                assert_ne!(result, WriteWatchResult::Unavailable);
                for (span_va, span_len) in spans {
                    let first = ((span_va - VA) >> 12) as usize;
                    for page in first..first + (span_len >> 12) {
                        let slot = (addr + page * 0x1000 + 8) as *const u32;
                        snapshot[page] = unsafe { slot.read_volatile() };
                    }
                }
            };
            while !stop.load(Ordering::Relaxed) {
                refresh(&mut snapshot);
                if writers.iter().all(|writer| writer.is_finished()) {
                    stop.store(true, Ordering::Relaxed);
                }
            }
            for writer in writers {
                writer.join().unwrap();
            }
            refresh(&mut snapshot);
            assert_eq!(snapshot, vec![ROUNDS as u32; PAGES]);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            decommit(ptr, LEN);
        }

        #[test]
        fn decommit_and_recommit_never_read_clean_without_a_fresh_baseline() {
            const VA: u64 = 0xe3_4000_0000;
            const LEN: usize = 0x4000;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            unsafe { ptr.add(0x1000).write_volatile(0x77) };
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Dirty);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            let generation = commit_generation();
            decommit(ptr, LEN);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);
            assert_ne!(commit_generation(), generation);

            // The contents changed (zeroed) without a guest write: the first
            // query after recommit must say dirty for every page.
            let again = commit(VA, LEN).expect("fastmem recommit");
            assert_eq!(again, ptr);
            assert_eq!(unsafe { again.add(0x1000).read_volatile() }, 0);
            assert_eq!(spans_of(VA, LEN), (WriteWatchResult::Dirty, vec![(VA, LEN)]));
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Clean);
            unsafe { again.add(0x3fff).write_volatile(1) };
            assert_eq!(
                spans_of(VA, LEN),
                (WriteWatchResult::Dirty, vec![(VA + 0x3000, 0x1000)])
            );
            decommit(again, LEN);
        }

        #[test]
        fn partial_decommit_keeps_tracking_the_live_part_and_regrowth_reads_dirty() {
            const VA: u64 = 0xe3_5000_0000;
            const LEN: usize = 0x6000;
            if !require_watch() {
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit");
            let _ = take_write_watch(VA, LEN);
            unsafe { ptr.add(0x1000).write_volatile(1) };
            // Shrink: drop the top four pages (what resize_committed does).
            decommit(unsafe { ptr.add(0x2000) }, 0x4000);
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);
            assert_eq!(
                spans_of(VA, 0x2000),
                (WriteWatchResult::Dirty, vec![(VA + 0x1000, 0x1000)])
            );
            assert_eq!(take_write_watch(VA, 0x2000), WriteWatchResult::Clean);
            // Regrow: the new pages are dirty, the surviving ones stay clean.
            commit(VA + 0x2000, 0x4000).expect("fastmem regrow");
            assert_eq!(
                spans_of(VA, LEN),
                (WriteWatchResult::Dirty, vec![(VA + 0x2000, 0x4000)])
            );
            unsafe { ptr.add(0x5000).write_volatile(1) };
            assert_eq!(
                spans_of(VA, LEN),
                (WriteWatchResult::Dirty, vec![(VA + 0x5000, 0x1000)])
            );
            decommit(ptr, LEN);
        }

        #[test]
        fn uncommitted_and_out_of_arena_ranges_are_unavailable() {
            if !require_watch() {
                return;
            }
            assert_eq!(take_write_watch(0xe3_6000_0000, 0x1000), WriteWatchResult::Unavailable);
            assert_eq!(take_write_watch(arena_size(), 0x1000), WriteWatchResult::Unavailable);
            assert_eq!(take_write_watch(u64::MAX - 0x10, 0x1000), WriteWatchResult::Unavailable);
            let mut spans = Vec::new();
            let observation = take_write_watch_spans_observed(0xe3_6000_0000, 0x1000, &mut spans);
            assert_eq!(observation.result, WriteWatchResult::Unavailable);
            assert!(spans.is_empty());
        }

        /// Per-query cost on this host. Run with
        /// `cargo test -p nexium-memory -- --ignored --nocapture query_cost`.
        #[test]
        #[ignore]
        fn query_cost() {
            const VA: u64 = 0xe3_8000_0000;
            if !require_watch() {
                return;
            }
            for len in [0x1000usize, 0x1_0000, 0x10_0000, 0x100_0000, 0x1000_0000] {
                let ptr = commit(VA, len).expect("fastmem commit");
                unsafe { std::ptr::write_bytes(ptr, 1, len) };
                let _ = take_write_watch(VA, len);
                let rounds = (0x4000_0000 / len).clamp(16, 20_000);
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    assert_eq!(take_write_watch(VA, len), WriteWatchResult::Clean);
                }
                let clean = start.elapsed() / rounds as u32;
                let mut spans = Vec::new();
                let start = std::time::Instant::now();
                for round in 0..rounds {
                    unsafe { ptr.add((round * 0x1000) % len).write_volatile(2) };
                    spans.clear();
                    assert_eq!(take_write_watch_spans(VA, len, &mut spans), WriteWatchResult::Dirty);
                }
                let one_dirty = start.elapsed() / rounds as u32;
                let rounds = rounds.min(64);
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    unsafe { std::ptr::write_bytes(ptr, 3, len) };
                }
                let fill = start.elapsed() / rounds as u32;
                let start = std::time::Instant::now();
                for _ in 0..rounds {
                    unsafe { std::ptr::write_bytes(ptr, 3, len) };
                    spans.clear();
                    assert_eq!(take_write_watch_spans(VA, len, &mut spans), WriteWatchResult::Dirty);
                }
                let all_dirty = (start.elapsed() / rounds as u32).saturating_sub(fill);
                eprintln!(
                    "len {len:#10x}: clean {clean:?}, one page dirty {one_dirty:?}, \
                     all dirty {all_dirty:?} (+ refault cost inside a {fill:?} fill)"
                );
                decommit(ptr, len);
            }
        }

        fn run_self_in_child(test: &str, marker: &str, envs: &[(&str, &str)]) {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args(["--exact", test, "--nocapture", "--test-threads=1"]).env(marker, "1");
            for (key, value) in envs {
                command.env(key, value);
            }
            let output = command.output().expect("spawn test child");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                "child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        /// The arena is process-global, so the disabled configuration runs in
        /// a child process: it must behave exactly as Linux did before the
        /// backend existed.
        #[test]
        fn fallback_without_backend_is_unavailable_everywhere() {
            const MARKER: &str = "NEXIUM_TEST_FALLBACK_CHILD";
            const VA: u64 = 0xe3_7000_0000;
            const LEN: usize = 0x2000;
            if std::env::var_os(MARKER).is_none() {
                run_self_in_child(
                    "fastmem::tests::linux::fallback_without_backend_is_unavailable_everywhere",
                    MARKER,
                    &[("NEXIUM_NO_WRITE_WATCH", "1")],
                );
                return;
            }
            let ptr = commit(VA, LEN).expect("fastmem commit without write-watch");
            assert!(!write_watch_available());
            unsafe { ptr.write_volatile(1) };
            assert_eq!(take_write_watch(VA, LEN), WriteWatchResult::Unavailable);
            let mut spans = Vec::new();
            assert_eq!(take_write_watch_spans(VA, LEN, &mut spans), WriteWatchResult::Unavailable);
            let observation = take_write_watch_spans_observed(VA, LEN, &mut spans);
            assert_eq!(observation.result, WriteWatchResult::Unavailable);
            assert!(spans.is_empty());
            decommit(ptr, LEN);
            let again = commit(VA, LEN).expect("recommit without write-watch");
            assert_eq!(unsafe { again.read_volatile() }, 0);
            decommit(again, LEN);
        }

        /// Direct (NCE) mode addresses the arena by host address. Runs in a
        /// child because the mode is latched per process.
        #[test]
        fn direct_mode_queries_use_host_addresses() {
            const MARKER: &str = "NEXIUM_TEST_DIRECT_CHILD";
            const LEN: usize = 0x4000;
            if std::env::var_os(MARKER).is_none() {
                if !require_watch() {
                    return;
                }
                run_self_in_child(
                    "fastmem::tests::linux::direct_mode_queries_use_host_addresses",
                    MARKER,
                    &[("NEXIUM_NCE", "1")],
                );
                return;
            }
            let (window_lo, _) = direct_va_range().expect("direct mode");
            let va = window_lo + 0x3_0000_0000;
            let ptr = commit(va, LEN).expect("fastmem commit");
            assert_eq!(ptr as u64, va);
            assert!(write_watch_available());
            assert_eq!(take_write_watch(va, LEN), WriteWatchResult::Dirty);
            assert_eq!(take_write_watch(va, LEN), WriteWatchResult::Clean);
            unsafe { ptr.add(0x2000).write_volatile(1) };
            assert_eq!(spans_of(va, LEN), (WriteWatchResult::Dirty, vec![(va + 0x2000, 0x1000)]));
            assert_eq!(observed_write_pages_changed_since(va, LEN, 0), 4);
            assert_eq!(take_write_watch(0x3_0000_0000, LEN), WriteWatchResult::Unavailable);
            decommit(ptr, LEN);
            assert_eq!(take_write_watch(va, LEN), WriteWatchResult::Unavailable);
        }
    }
}

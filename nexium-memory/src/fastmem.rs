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
            (p as *mut u8, false)
        }
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
    }
}

static ARENA: OnceLock<AtomicPtr<u8>> = OnceLock::new();
static ARENA_WRITE_WATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static COMMITTED_RANGES: OnceLock<Mutex<Vec<CommittedRange>>> = OnceLock::new();
static COMMIT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

#[cfg(any(target_vendor = "sony", windows))]
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

#[derive(Clone, Copy, Debug)]
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
    if fresh {
        COMMIT_GENERATION.fetch_add(1, Ordering::Relaxed);
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
    ranges.iter().any(|range| range.lo < hi && lo < range.hi)
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
    if !released.is_empty() {
        COMMIT_GENERATION.fetch_add(1, Ordering::Relaxed);
    }
    for (released_lo, released_hi) in released {
        let released_ptr = unsafe { base.add(released_lo as usize) };
        sys::decommit(released_ptr, (released_hi - released_lo) as usize);
        #[cfg(windows)]
        crate::soft_watch::mark_dirty_host(released_ptr, (released_hi - released_lo) as usize);
    }
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

pub fn take_write_watch(va: u64, len: usize) -> WriteWatchResult {
    #[cfg(target_vendor = "sony")]
    {
        return soft_take_write_watch(va, len, None);
    }

    #[cfg(not(any(windows, target_vendor = "sony")))]
    {
        let _ = (va, len);
        return WriteWatchResult::Unavailable;
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

    #[cfg(not(any(windows, target_vendor = "sony")))]
    {
        let _ = (va, len, spans);
        return WriteWatchResult::Unavailable;
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

    #[cfg(not(any(windows, target_vendor = "sony")))]
    {
        let _ = (va, len, spans);
        return WriteWatchObservation::unavailable();
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

fn update_commit_refs(
    ranges: &mut Vec<CommittedRange>,
    lo: u64,
    hi: u64,
    increment: bool,
) -> Vec<(u64, u64)> {
    let mut boundaries = Vec::with_capacity(ranges.len().saturating_mul(2).saturating_add(2));
    boundaries.extend([lo, hi]);
    for range in ranges.iter() {
        boundaries.extend([range.lo, range.hi]);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut rebuilt: Vec<CommittedRange> = Vec::with_capacity(ranges.len().saturating_add(2));
    let mut released = Vec::new();
    for segment in boundaries.windows(2) {
        let segment_lo = segment[0];
        let segment_hi = segment[1];
        if segment_hi <= segment_lo {
            continue;
        }
        let old_refs = ranges
            .iter()
            .find(|range| range.lo <= segment_lo && range.hi >= segment_hi)
            .map_or(0, |range| range.refs);
        let affected = segment_lo >= lo && segment_hi <= hi;
        let new_refs = if affected {
            if increment {
                old_refs.saturating_add(1)
            } else {
                old_refs.saturating_sub(1)
            }
        } else {
            old_refs
        };
        if old_refs != 0 && new_refs == 0 {
            if let Some((_, released_hi)) = released.last_mut() {
                if *released_hi == segment_lo {
                    *released_hi = segment_hi;
                } else {
                    released.push((segment_lo, segment_hi));
                }
            } else {
                released.push((segment_lo, segment_hi));
            }
        }
        if new_refs == 0 {
            continue;
        }
        if let Some(last) = rebuilt.last_mut() {
            if last.hi == segment_lo && last.refs == new_refs {
                last.hi = segment_hi;
                continue;
            }
        }
        rebuilt.push(CommittedRange {
            lo: segment_lo,
            hi: segment_hi,
            refs: new_refs,
        });
    }
    *ranges = rebuilt;
    released
}

fn is_committed(ranges: &[CommittedRange], lo: u64, hi: u64) -> bool {
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

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn write_watch_detects_direct_fastmem_stores_and_resets_atomically() {
        const VA: u64 = 0xf0_0000_0000;
        const LEN: usize = 0x2000;
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
}

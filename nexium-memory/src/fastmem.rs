use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

pub const ARENA_BITS: u32 = 40;
pub const ARENA_SIZE: u64 = 1u64 << ARENA_BITS;

#[cfg(windows)]
mod sys {
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_DECOMMIT: u32 = 0x4000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READONLY: u32 = 0x02;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
        fn VirtualProtect(addr: *mut u8, size: usize, protect: u32, old: *mut u32) -> i32;
    }

    pub fn reserve(size: usize) -> *mut u8 {
        unsafe { VirtualAlloc(std::ptr::null_mut(), size, MEM_RESERVE, PAGE_NOACCESS) }
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
        let p = if trap { PAGE_NOACCESS } else { PAGE_READWRITE };
        let _ = PAGE_READONLY;
        unsafe { VirtualProtect(ptr, len, p, &mut old) != 0 }
    }
}

#[cfg(unix)]
mod sys {
    pub fn protect(ptr: *mut u8, len: usize, trap: bool) -> bool {
        let p = if trap {
            libc::PROT_NONE
        } else {
            libc::PROT_READ | libc::PROT_WRITE
        };
        unsafe { libc::mprotect(ptr as *mut libc::c_void, len, p) == 0 }
    }

    pub fn reserve(size: usize) -> *mut u8 {
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
            std::ptr::null_mut()
        } else {
            p as *mut u8
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

fn arena() -> *mut u8 {
    ARENA
        .get_or_init(|| {
            let base = sys::reserve(ARENA_SIZE as usize);
            if base.is_null() {
                log::warn!(
                    "fastmem: failed to reserve {}GB arena; falling back to heap regions",
                    ARENA_SIZE >> 30
                );
            } else {
                log::info!(
                    "fastmem: reserved {}GB arena at {:p}",
                    ARENA_SIZE >> 30,
                    base
                );
            }
            AtomicPtr::new(base)
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
    let end = va.checked_add(len as u64)?;
    if end > ARENA_SIZE {
        log::warn!(
            "fastmem: region va={:#x} len={:#x} outside arena; using heap",
            va,
            len
        );
        return None;
    }
    let ptr = unsafe { base.add(va as usize) };
    if !sys::commit(ptr, len) {
        log::warn!(
            "fastmem: commit failed va={:#x} len={:#x}; using heap",
            va,
            len
        );
        return None;
    }
    Some(ptr)
}

pub fn decommit(ptr: *mut u8, len: usize) {
    sys::decommit(ptr, len);
}

use std::sync::atomic::AtomicU64;

static WATCH_LO: AtomicU64 = AtomicU64::new(0);
static WATCH_HI: AtomicU64 = AtomicU64::new(0);

pub fn watch_arm(va: u64, len: u64) -> bool {
    let base = arena();
    if base.is_null() {
        return false;
    }
    let lo = va & !0xFFF;
    let hi = (va + len + 0xFFF) & !0xFFF;
    if hi > ARENA_SIZE {
        return false;
    }
    let ptr = unsafe { base.add(lo as usize) };
    if !sys::protect(ptr, (hi - lo) as usize, true) {
        return false;
    }
    WATCH_LO.store(lo, Ordering::SeqCst);
    WATCH_HI.store(hi, Ordering::SeqCst);
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

pub fn watch_reprotect() -> bool {
    if let Some((lo, hi)) = watch_range() {
        let base = arena();
        if base.is_null() {
            return false;
        }
        let ptr = unsafe { base.add(lo as usize) };
        return sys::protect(ptr, (hi - lo) as usize, true);
    }
    false
}

pub fn watch_disarm() {
    if let Some((lo, hi)) = watch_range() {
        let base = arena();
        if !base.is_null() {
            let ptr = unsafe { base.add(lo as usize) };
            sys::protect(ptr, (hi - lo) as usize, false);
        }
    }
    WATCH_LO.store(0, Ordering::SeqCst);
    WATCH_HI.store(0, Ordering::SeqCst);
}

pub fn watch_write_through(addr: u64, size: usize, value: u64) -> bool {
    let Some((lo, hi)) = watch_range() else {
        return false;
    };
    if addr < lo || addr + size as u64 > hi {
        return false;
    }
    let base = arena();
    if base.is_null() {
        return false;
    }
    let ptr = unsafe { base.add(lo as usize) };
    if !sys::protect(ptr, (hi - lo) as usize, false) {
        return false;
    }
    unsafe {
        let dst = base.add(addr as usize);
        let bytes = value.to_le_bytes();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, size.min(8));
    }
    let _ = sys::protect(ptr, (hi - lo) as usize, true);
    true
}

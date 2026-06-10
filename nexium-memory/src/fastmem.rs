use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, Ordering};

pub const ARENA_BITS: u32 = 40;
pub const ARENA_SIZE: u64 = 1u64 << ARENA_BITS;

#[cfg(windows)]
mod sys {
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_DECOMMIT: u32 = 0x4000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
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
}

#[cfg(unix)]
mod sys {
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
        unsafe { libc::mprotect(ptr as *mut libc::c_void, len, libc::PROT_READ | libc::PROT_WRITE) == 0 }
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
                log::warn!("fastmem: failed to reserve {}GB arena; falling back to heap regions", ARENA_SIZE >> 30);
            } else {
                log::info!("fastmem: reserved {}GB arena at {:p}", ARENA_SIZE >> 30, base);
            }
            AtomicPtr::new(base)
        })
        .load(Ordering::Relaxed)
}

pub fn base() -> Option<*mut u8> {
    let p = arena();
    if p.is_null() { None } else { Some(p) }
}

pub fn commit(va: u64, len: usize) -> Option<*mut u8> {
    let base = arena();
    if base.is_null() {
        return None;
    }
    let end = va.checked_add(len as u64)?;
    if end > ARENA_SIZE {
        log::warn!("fastmem: region va={:#x} len={:#x} outside arena; using heap", va, len);
        return None;
    }
    let ptr = unsafe { base.add(va as usize) };
    if !sys::commit(ptr, len) {
        log::warn!("fastmem: commit failed va={:#x} len={:#x}; using heap", va, len);
        return None;
    }
    Some(ptr)
}

pub fn decommit(ptr: *mut u8, len: usize) {
    sys::decommit(ptr, len);
}

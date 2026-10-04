use std::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    pub fn sceKernelDebugOutText(channel: c_int, text: *const c_char) -> c_int;
    pub fn sceKernelUsleep(micros: u32) -> c_int;
    pub fn sceKernelGetDirectMemorySize() -> usize;
    pub fn sceKernelAvailableFlexibleMemorySize(size: *mut usize) -> c_int;
    pub fn sceKernelGetProcessTime() -> u64;
}

#[cfg(feature = "title")]
unsafe extern "C" {
    pub fn sceSystemServiceLoadExec(path: *const c_char, argv: *const *const c_char) -> c_int;
    pub fn sceSystemServiceHideSplashScreen() -> c_int;
    pub fn ps5_klog_capture_stderr(prefix: *const c_char) -> c_int;
}

pub const EXEC_NEAR: u32 = 0x1;
pub const SHM_READ: c_int = 0x1;
pub const SHM_WRITE: c_int = 0x2;
pub const SHM_FIXED: u32 = 0x1;
pub const SHM_KEEP_RESERVED: u32 = 0x2;

#[repr(C)]
#[derive(Default)]
pub struct ExecRequest {
    pub bytes: usize,
    pub address: usize,
    pub anchor: usize,
    pub flags: u32,
}

#[repr(C)]
pub struct ExecRegion {
    pub base: *mut c_void,
    pub write_view: *mut c_void,
    pub bytes: usize,
    pub direct_start: i64,
    pub flags: u32,
}

impl Default for ExecRegion {
    fn default() -> Self {
        Self { base: std::ptr::null_mut(), write_view: std::ptr::null_mut(), bytes: 0, direct_start: 0, flags: 0 }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Shm {
    pub direct_start: i64,
    pub bytes: usize,
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct HeapStats {
    pub range_base: usize,
    pub range_bytes: usize,
    pub mapped_bytes: usize,
    pub peak_bytes: usize,
    pub segments: u32,
    pub libc_fallbacks: u64,
    pub arenas: u32,
}

#[cfg(feature = "title")]
unsafe extern "C" {
    pub fn ps5_exec_alloc(request: *const ExecRequest, region: *mut ExecRegion) -> c_int;
    pub fn ps5_exec_free(region: *mut ExecRegion);
    pub fn ps5_exec_live(regions: *mut u64, bytes: *mut u64);
    pub fn ps5_shm_create(bytes: usize, shm: *mut Shm) -> c_int;
    pub fn ps5_shm_destroy(shm: *mut Shm);
    pub fn ps5_shm_map(
        shm: *const Shm,
        offset: usize,
        bytes: usize,
        address: *mut c_void,
        protection: c_int,
        flags: u32,
        view: *mut *mut c_void,
    ) -> c_int;
    pub fn ps5_shm_unmap(view: *mut c_void, bytes: usize, flags: u32) -> c_int;
    pub fn ps5_vrange_reserve(bytes: usize, hint: *mut c_void, alignment: usize, base: *mut *mut c_void) -> c_int;
    pub fn ps5_vrange_release(base: *mut c_void, bytes: usize) -> c_int;
    pub fn ps5_heap_stats(stats: *mut HeapStats);
}

pub const UC_MCONTEXT: usize = 64;
pub const MC_RAX: usize = 56;
pub const MC_RBP: usize = 72;
pub const MC_ADDR: usize = 136;
pub const MC_RIP: usize = 160;
pub const MC_RSP: usize = 184;

pub unsafe fn context_reg(ucontext: *mut c_void, mc_offset: usize) -> *mut u64 {
    unsafe { (ucontext as *mut u8).add(UC_MCONTEXT + mc_offset) as *mut u64 }
}

pub fn mxcsr() -> u32 {
    let mut value: u32 = 0;
    unsafe {
        std::arch::asm!("stmxcsr [{}]", in(reg) &mut value as *mut u32, options(nostack));
    }
    value
}

pub fn set_mxcsr(value: u32) {
    unsafe {
        std::arch::asm!("ldmxcsr [{}]", in(reg) &value as *const u32, options(nostack, readonly));
    }
}

pub fn flexible_available() -> Option<usize> {
    let mut size = 0usize;
    let rc = unsafe { sceKernelAvailableFlexibleMemorySize(&mut size) };
    (rc == 0).then_some(size)
}

pub fn direct_size() -> usize {
    unsafe { sceKernelGetDirectMemorySize() }
}

pub fn opaque<T>(value: T) -> T {
    std::hint::black_box(value)
}

pub type RawPtr = *mut c_void;

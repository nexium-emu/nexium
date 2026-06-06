pub const VSYNC_PERIOD_NS: u64 = 16_666_667;

pub const PAGE_SIZE: u64 = 0x1000;
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;
pub const TLS_PAGE_SIZE: u64 = 0x200;
pub const TLS_BUFFER_SIZE: u64 = 0x200;

pub const CODE_BASE: u64 = 0x0800_0000_0000;
pub const HEAP_BASE: u64 = 0x0801_0000_0000;
pub const STACK_BASE: u64 = 0x0820_0000_0000;

pub const HEAP_DEFAULT_SIZE: u64 = 256 * 1024 * 1024;
pub const STACK_DEFAULT_SIZE: u64 = 1 * 1024 * 1024;

pub const TLS_REQUEST_OFFSET: u64 = 0x100;
pub const TLS_RESPONSE_OFFSET: u64 = 0x100;

pub const NUM_CORES: u32 = 4;

pub const NUM_GUEST_CORES: usize = 3;

pub const DEFAULT_JIT_SIZE_MB: u64 = 128;
pub const MIN_JIT_SIZE_MB: u64 = 8;
pub const MAX_JIT_SIZE_MB: u64 = 512;

pub const DEFAULT_RENDER_WIDTH: u32 = 1280;
pub const DEFAULT_RENDER_HEIGHT: u32 = 720;

pub const MEMORY_REGION_HEAP: u32 = 0;
pub const MEMORY_REGION_ALIAS: u32 = 1;
pub const MEMORY_REGION_STACK: u32 = 2;
pub const MEMORY_REGION_TLS: u32 = 3;

pub const THREAD_PRIORITY_HIGHEST: u32 = 0;
pub const THREAD_PRIORITY_NORMAL: u32 = 16;
pub const THREAD_PRIORITY_LOWEST: u32 = 31;

pub const INFO_CORE_MASK: u32 = 0;
pub const INFO_PRIORITY_MASK: u32 = 1;
pub const INFO_STATE_MASK: u32 = 2;

pub const HANDLE_INVALID: u32 = 0;

pub const PORT_NAME_MAX: usize = 11;

pub struct MemoryAttribute;
impl MemoryAttribute {
    pub const READ: u32 = 0x1;
    pub const WRITE: u32 = 0x2;
    pub const EXECUTE: u32 = 0x4;
}

pub struct ThreadAttribute;
impl ThreadAttribute {
    pub const DETACH_ON_EXIT: u32 = 0x1;
}

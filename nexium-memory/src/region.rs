pub const PAGE_SHIFT: u32 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

pub const CODE_BASE: u64 = 0x80_0000_0000;

pub const HEAP_BASE: u64 = 0x90_0000_0000;
pub const HEAP_DEFAULT_SIZE: u64 = 0x1000_0000;

pub const STACK_BASE: u64 = 0xA0_0000_0000;
pub const STACK_DEFAULT_SIZE: u64 = 0x10_0000;

pub const TLS_PAGE_SIZE: u64 = PAGE_SIZE;

#[inline]
pub fn page_align_down(addr: u64) -> u64 {
    addr & !PAGE_MASK
}

#[inline]
pub fn page_align_up(addr: u64) -> u64 {
    (addr.saturating_add(PAGE_MASK)) & !PAGE_MASK
}

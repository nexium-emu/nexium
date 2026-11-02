pub mod address_space;
pub mod fastmem;
#[cfg(all(target_os = "linux", not(target_vendor = "sony")))]
pub mod linux_watch;
pub mod perm;
pub mod region;
#[cfg(any(target_vendor = "sony", windows))]
pub mod soft_watch;

pub use address_space::{
    align_request, AddressSpace, AddressSpaceError, HostRegion, HostRegionChange,
    HostRegionChanges, HostRegionLease, RegionInfo,
};
pub use perm::Perm;
pub use region::{page_align_down, page_align_up};
pub use region::{CODE_BASE, HEAP_BASE, STACK_BASE};
pub use region::{HEAP_DEFAULT_SIZE, STACK_DEFAULT_SIZE};
pub use region::{PAGE_MASK, PAGE_SHIFT, PAGE_SIZE, TLS_PAGE_SIZE};

pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (pages > 0 && page_size > 0).then(|| pages as u64 * page_size as u64)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[cfg(test)]
mod physical_memory_tests {
    #[cfg(unix)]
    #[test]
    fn physical_memory_is_reported_on_unix() {
        assert!(super::physical_memory_bytes().is_some_and(|bytes| bytes >= 1 << 30));
    }
}

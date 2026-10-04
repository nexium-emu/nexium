pub mod address_space;
pub mod fastmem;
pub mod perm;
pub mod region;
#[cfg(target_vendor = "sony")]
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

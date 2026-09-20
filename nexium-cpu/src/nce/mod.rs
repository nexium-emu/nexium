pub mod asm;
pub mod backend;
pub mod context;
pub mod signal;

pub use backend::{flush_icache, NceCpu};
pub use context::NceThreadContext;

use std::collections::HashMap;
use std::sync::OnceLock;

fn post_handlers() -> &'static parking_lot::RwLock<HashMap<u64, u64>> {
    static REGISTRY: OnceLock<parking_lot::RwLock<HashMap<u64, u64>>> = OnceLock::new();
    REGISTRY.get_or_init(|| parking_lot::RwLock::new(HashMap::new()))
}

pub fn register_post_handlers(entries: &[(u64, u64)]) {
    let mut registry = post_handlers().write();
    for &(resume_pc, trampoline) in entries {
        registry.insert(resume_pc, trampoline);
    }
}

pub fn clear_post_handlers() {
    post_handlers().write().clear();
}

pub fn lookup_post_handler(pc: u64) -> Option<u64> {
    post_handlers().read().get(&pc).copied()
}

pub fn supported() -> bool {
    nexium_memory::fastmem::direct_va_range().is_some()
}

pub fn direct_window_contains(va: u64, len: u64) -> bool {
    let Some((lo, hi)) = nexium_memory::fastmem::direct_va_range() else {
        return false;
    };
    va >= lo && va.checked_add(len).is_some_and(|end| end <= hi)
}

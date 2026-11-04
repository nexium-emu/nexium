use super::{build_ipc_response, build_ipc_response_copy, long, new_event, open_session, reply_longs, reply_words, signal, word, Kernel, GPU};
use nexium_ipc::IpcCtx;
use std::collections::HashMap;

const EVENT_IDS: u32 = 32;
const VA_START: u64 = 0x800_0000;
const VA_END: u64 = 0xC0_0000_0000;
const ADDRESS_SPACE_KIND: u64 = 1;
const IDENTITY_FLOOR: u64 = 0x80_0000_0000;
const POOL_BASE: u64 = 0xC0_0000_0000;
const POOL_END: u64 = 0x100_0000_0000;
const PAGE: u64 = 0x1000;

#[derive(Default)]
pub(super) struct GpuState {
    events: HashMap<u32, u32>,
    memories: HashMap<u32, GpuMemory>,
    next_memory: u32,
    private: HashMap<u64, GpuMemory>,
    pool_next: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GpuMemory {
    pub(super) address: u64,
    pub(super) size: u64,
}

impl GpuState {
    pub(super) fn memory(&self, handle: u32) -> Option<GpuMemory> {
        self.memories.get(&handle).copied()
    }
}

fn event(kernel: &mut Kernel, id: u32) -> u32 {
    let id = id % EVENT_IDS;
    if let Some(&handle) = kernel.nx2_graphics.gpu.events.get(&id) {
        return handle;
    }
    let handle = new_event(kernel, "nx2_gpu");
    kernel.nx2_graphics.gpu.events.insert(id, handle);
    handle
}

fn complete(kernel: &mut Kernel, slot: u32) {
    let handle = event(kernel, slot);
    signal(kernel, handle);
}

fn back_private_region(kernel: &mut Kernel, first: u64, size: u64) -> Option<u64> {
    let size = size.next_multiple_of(PAGE);
    if let Some(existing) = kernel.nx2_graphics.gpu.private.get(&first).filter(|memory| memory.size >= size) {
        return Some(existing.address);
    }
    let identity = first >= IDENTITY_FLOOR && first.checked_add(size).is_some_and(|end| end <= POOL_BASE);
    let mut candidates = Vec::with_capacity(2);
    if identity {
        candidates.push(first);
    }
    let state = &mut kernel.nx2_graphics.gpu;
    let pooled = state.pool_next.max(POOL_BASE);
    if pooled.checked_add(size).is_some_and(|end| end <= POOL_END) {
        candidates.push(pooled);
    }
    for backing in candidates {
        match kernel
            .address_space
            .map(backing, size, nexium_memory::Perm::RW, "nx2 gpu private")
        {
            Ok(_) => {
                if backing == pooled {
                    kernel.nx2_graphics.gpu.pool_next = pooled + size;
                }
                kernel.nx2_graphics.gpu.private.insert(first, GpuMemory { address: backing, size });
                return Some(backing);
            }
            Err(error) => log::warn!("[nx2] gpu private backing {backing:#x}+{size:#x} failed: {error}"),
        }
    }
    None
}

fn range(ctx: &IpcCtx) -> (u64, u64) {
    let (first, last) = (long(ctx, 1), long(ctx, 3));
    (first, last.saturating_sub(first).saturating_add(1))
}

fn update_mappings(kernel: &mut Kernel, update: impl Fn(&mut nexium_nvdrv::Nvdrv, bool) -> bool) -> bool {
    if update(&mut kernel.nvdrv, false) {
        return true;
    }
    if crate::kernel::svc::rewind_svc_for_retry(kernel) {
        kernel.ipc_retry_pending = true;
        return false;
    }
    update(&mut kernel.nvdrv, true)
}

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    match cmd_id {
        11 => Some(reply_words(ctx, &[0, EVENT_IDS - 1])),
        12 => {
            let handle = event(kernel, word(ctx, 0));
            Some(build_ipc_response_copy(ctx, 0, &[], &[handle]))
        }
        21 => {
            complete(kernel, word(ctx, 0));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        22 => Some(open_session(kernel, ctx, GPU)),
        33 => Some(reply_words(ctx, &[1])),
        201 => Some(reply_longs(ctx, &[VA_START, VA_END - VA_START])),
        202 => Some(reply_longs(ctx, &[0, 0])),
        203 => Some(reply_longs(ctx, &[ADDRESS_SPACE_KIND, VA_START, VA_END - VA_START])),
        101 => {
            let (start, last) = (long(ctx, 0), long(ctx, 2));
            let size = last.saturating_sub(start).saturating_add(1);
            if !update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_map(start, size, start, wait)) {
                return Some(Vec::new());
            }
            let state = &mut kernel.nx2_graphics.gpu;
            state.next_memory += 1;
            let handle = state.next_memory;
            state.memories.insert(handle, GpuMemory { address: start, size });
            log::debug!("[nx2] gpu memory {handle} = {start:#x}+{size:#x}");
            Some(reply_words(ctx, &[handle]))
        }
        102 => {
            let (handle, slot) = (word(ctx, 0), word(ctx, 1));
            if let Some(memory) = kernel.nx2_graphics.gpu.memory(handle) {
                if !update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_unmap(memory.address, memory.size, wait)) {
                    return Some(Vec::new());
                }
                kernel.nx2_graphics.gpu.memories.remove(&handle);
                log::debug!("[nx2] gpu memory {handle} destroyed");
            }
            complete(kernel, slot);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        211 => {
            let ((first, size), handle, offset, slot) = (range(ctx), word(ctx, 5), long(ctx, 6), word(ctx, 15));
            match kernel.nx2_graphics.gpu.memory(handle) {
                Some(memory) => {
                    let cpu_addr = memory.address + offset;
                    if !update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_map(first, size, cpu_addr, wait)) {
                        return Some(Vec::new());
                    }
                }
                None => log::warn!("[nx2] gpu map {first:#x}+{size:#x} of unknown memory {handle}"),
            }
            complete(kernel, slot);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        212 => {
            let (kind, slot) = (word(ctx, 5) & 0xFF, word(ctx, 8));
            complete(kernel, slot);
            Some(reply_words(ctx, &[kind]))
        }
        213 | 215 => {
            let ((first, size), slot) = (range(ctx), word(ctx, 6));
            if !update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_unmap(first, size, wait)) {
                return Some(Vec::new());
            }
            if cmd_id == 215 {
                if let Some(backing) = kernel.nx2_graphics.gpu.private.remove(&first) {
                    let _ = kernel.address_space.unmap(backing.address, backing.size);
                }
            }
            complete(kernel, slot);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        214 => {
            let ((first, size), flags, slot) = (range(ctx), word(ctx, 5), word(ctx, 9));
            let backing = if std::env::var_os("NEXIUM_NX2_SPARSE_PRIVATE").is_some() {
                None
            } else {
                back_private_region(kernel, first, size)
            };
            let mapped = match backing {
                Some(backing) => update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_map(first, size, backing, wait)),
                None => update_mappings(kernel, |nvdrv, wait| nvdrv.nx2_map_sparse(first, size, wait)),
            };
            if !mapped {
                return Some(Vec::new());
            }
            log::debug!("[nx2] gpu private region {first:#x}+{size:#x} flags={flags:#x}");
            complete(kernel, slot);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        _ => None,
    }
}

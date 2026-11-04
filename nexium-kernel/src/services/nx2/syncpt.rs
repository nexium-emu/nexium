use super::{build_ipc_response, build_ipc_response_copy, long, new_event, reply_words, signal, word, Kernel};
use nexium_common::result::KERNEL_INVALID_HANDLE;
use nexium_ipc::IpcCtx;
use std::collections::HashMap;

const SYNCPOINTS: u32 = 192;
const COUNTERS: u32 = 32;
const PAGE_VERSION: u32 = 1;
const PAGE_CAPACITY: u32 = (0x1000 - 0x40) / 8;
const VALUES: u64 = 0x40;
const RW_SUMMARY: u64 = 0;
const RO_SUMMARY: u64 = 0x1040;
const RW_ARMED: u64 = 0x100;
const RO_ACKNOWLEDGED: u64 = 0x1100;
const BIT_WORDS: usize = 0x40;
const RO_SIZE: u64 = 0x2000;

#[derive(Default)]
pub(super) struct SyncptState {
    slots: HashMap<u32, Slot>,
    page: Option<SharedPage>,
}

struct Slot {
    event: u32,
    wait: Option<(u32, u32)>,
}

struct SharedPage {
    rw: u64,
    ro: u64,
    count: u32,
    reply: u32,
}

fn syncpoint_of(counter: u32) -> u32 {
    !counter
}

fn slot_event(kernel: &mut Kernel, slot: u32) -> u32 {
    if let Some(existing) = kernel.nx2_graphics.syncpt.slots.get(&slot) {
        return existing.event;
    }
    let event = new_event(kernel, "nx2_syncpt_slot");
    kernel.nx2_graphics.syncpt.slots.insert(slot, Slot { event, wait: None });
    event
}

fn reached(kernel: &Kernel, syncpoint: u32, threshold: u32) -> bool {
    syncpoint >= SYNCPOINTS
        || kernel.nvdrv.nx2_syncpoint_value(syncpoint).wrapping_sub(threshold) < 0x8000_0000
}

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    match cmd_id {
        0 => {
            let token = long(ctx, 0);
            log::info!("[nx2] syncpt handshake token={token:#x}");
            Some(build_ipc_response(ctx, 0, &token.to_le_bytes(), &[]))
        }
        1 => Some(reply_words(ctx, &[COUNTERS])),
        21 => Some(build_ipc_response(ctx, 0, &[], &[])),
        22 => {
            let event = slot_event(kernel, word(ctx, 0));
            Some(build_ipc_response_copy(ctx, 0, &[], &[event]))
        }
        103 => {
            let syncpoint = syncpoint_of(word(ctx, 0));
            let value = if syncpoint < SYNCPOINTS {
                kernel.nvdrv.nx2_syncpoint_value(syncpoint)
            } else {
                0
            };
            Some(reply_words(ctx, &[value, 0, 0]))
        }
        201 => {
            let (slot, syncpoint, threshold) = (word(ctx, 0), syncpoint_of(word(ctx, 1)), word(ctx, 2));
            let event = slot_event(kernel, slot);
            kernel.event_signals.insert(event, false);
            let done = reached(kernel, syncpoint, threshold);
            log::debug!(
                "[nx2] syncpt wait slot={slot} syncpoint={syncpoint} threshold={threshold} value={} done={done}",
                if syncpoint < SYNCPOINTS { kernel.nvdrv.nx2_syncpoint_value(syncpoint) } else { 0 }
            );
            if let Some(entry) = kernel.nx2_graphics.syncpt.slots.get_mut(&slot) {
                entry.wait = (!done).then_some((syncpoint, threshold));
            }
            if done {
                signal(kernel, event);
            }
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        202 => {
            let event = slot_event(kernel, word(ctx, 0));
            if let Some(entry) = kernel.nx2_graphics.syncpt.slots.get_mut(&word(ctx, 0)) {
                entry.wait = None;
            }
            signal(kernel, event);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        _ => None,
    }
}

pub(super) fn attach_page(kernel: &mut Kernel, ctx: &mut IpcCtx) -> Vec<u8> {
    let regions: Vec<Option<(u64, u64)>> = ctx
        .copy_handles
        .iter()
        .map(|handle| kernel.transfer_memories.get(handle).copied())
        .collect();
    let (Some(Some((rw, _))), Some(Some((ro, ro_size)))) = (regions.first().copied(), regions.get(1).copied()) else {
        log::warn!("[nx2] gsv 204 without its two transfer memories: {:x?}", ctx.copy_handles);
        return build_ipc_response(ctx, KERNEL_INVALID_HANDLE, &[], &[]);
    };
    if ro_size < RO_SIZE {
        log::warn!("[nx2] gsv 204 read-only page too small: {ro_size:#x}");
        return build_ipc_response(ctx, KERNEL_INVALID_HANDLE, &[], &[]);
    }
    let count = word(ctx, 1).min(PAGE_CAPACITY);
    let header = [PAGE_VERSION, 0, count, VALUES as u32, RW_SUMMARY as u32, RO_SUMMARY as u32, RW_ARMED as u32, RO_ACKNOWLEDGED as u32];
    let bytes: Vec<u8> = header.iter().flat_map(|value| value.to_le_bytes()).collect();
    let _ = kernel.address_space.write(ro, &bytes);
    let request = new_event(kernel, "nx2_syncpt_request");
    let reply = new_event(kernel, "nx2_syncpt_reply");
    kernel.nx2_graphics.syncpt.page = Some(SharedPage { rw, ro, count, reply });
    log::info!("[nx2] syncpoint page rw={rw:#x} ro={ro:#x} count={count} version={}", word(ctx, 0));
    build_ipc_response_copy(ctx, 0, &[], &[request, reply])
}

pub(super) fn poll(kernel: &mut Kernel) {
    let waits: Vec<(u32, u32, u32)> = kernel
        .nx2_graphics
        .syncpt
        .slots
        .iter()
        .filter_map(|(&slot, entry)| entry.wait.map(|(syncpoint, threshold)| (slot, syncpoint, threshold)))
        .collect();
    for (slot, syncpoint, threshold) in waits {
        if reached(kernel, syncpoint, threshold) {
            let event = kernel.nx2_graphics.syncpt.slots.get_mut(&slot).map(|entry| {
                entry.wait = None;
                entry.event
            });
            if let Some(event) = event {
                signal(kernel, event);
            }
        }
    }
    let Some((rw, ro, count, reply)) = kernel
        .nx2_graphics
        .syncpt
        .page
        .as_ref()
        .map(|page| (page.rw, page.ro, page.count, page.reply))
    else {
        return;
    };
    let mut values = vec![0u32; count.min(SYNCPOINTS) as usize];
    kernel.nvdrv.nx2_syncpoint_values(&mut values);
    let bytes: Vec<u8> = values.iter().flat_map(|value| u64::from(*value).to_le_bytes()).collect();
    let _ = kernel.address_space.write(ro + VALUES, &bytes);
    let mut armed = vec![0u8; BIT_WORDS * 8];
    let mut acknowledged = vec![0u8; BIT_WORDS * 8];
    if kernel.address_space.read(rw + RW_ARMED, &mut armed).is_err()
        || kernel.address_space.read(ro + RO_ACKNOWLEDGED, &mut acknowledged).is_err()
    {
        return;
    }
    if armed == acknowledged {
        return;
    }
    let _ = kernel.address_space.write(ro + RO_ACKNOWLEDGED, &armed);
    let mut summary = [0u8; 8];
    if kernel.address_space.read(rw + RW_SUMMARY, &mut summary).is_ok() {
        let _ = kernel.address_space.write(ro + RO_SUMMARY, &summary);
    }
    signal(kernel, reply);
}

use super::{build_ipc_response, long, reply_longs, reply_words, word, words, Kernel};
use nexium_ipc::IpcCtx;
use std::collections::HashMap;

const PREFERENCE_WORDS: usize = 16;
const BYTES_PER_PIXEL: u64 = 4;
const DEFAULT_ROWS_PER_BLOCK: u64 = 128;
const MAX_ROWS_PER_BLOCK_LOG2: u32 = 8;
const MAX_PITCH_UNITS: u64 = 4096;
const MIN_ALIGNMENT: u64 = 0x1000;

#[derive(Default)]
pub(super) struct NativeState {
    memories: HashMap<u32, NativeMemory>,
    next_memory: u32,
    images: HashMap<u32, NativeImage>,
    next_image: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct NativeMemory {
    pub(super) address: u64,
    pub(super) size: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct NativeImage {
    pub(super) memory: u32,
    pub(super) offset: u64,
    pub(super) layout: ImageLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImageLayout {
    pub(super) kind: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: u32,
    pub(super) pitch: u64,
    pub(super) rows_per_block: u64,
    pub(super) size: u64,
    pub(super) alignment: u64,
}

impl ImageLayout {
    pub(super) fn from_preference(preference: &[u32]) -> Self {
        let field = |index: usize| preference.get(index).copied().unwrap_or(0);
        let width = (field(2) & 0xFFFF) + 1;
        let height = (field(2) >> 16) + 1;
        let packed_units = (u64::from(width) * BYTES_PER_PIXEL).div_ceil(64);
        let units = u64::from(field(5));
        let (pitch, rows_per_block) =
            if field(4) <= MAX_ROWS_PER_BLOCK_LOG2 && (packed_units..=MAX_PITCH_UNITS).contains(&units) {
                (units * 64, 1u64 << field(4))
            } else {
                (packed_units * 64, DEFAULT_ROWS_PER_BLOCK)
            };
        Self {
            kind: field(1),
            width,
            height,
            format: field(6),
            pitch,
            rows_per_block,
            size: pitch * u64::from(height).next_multiple_of(rows_per_block),
            alignment: (64 * rows_per_block).max(MIN_ALIGNMENT),
        }
    }
}

impl NativeState {
    pub(super) fn image_address(&self, image: u32) -> Option<(u64, ImageLayout)> {
        let image = self.images.get(&image)?;
        let memory = self.memories.get(&image.memory)?;
        Some((memory.address + image.offset, image.layout))
    }
}

fn preference(ctx: &IpcCtx) -> (ImageLayout, u32) {
    let request = words(ctx);
    let preference = &request[..request.len().min(PREFERENCE_WORDS)];
    let layout = ImageLayout::from_preference(preference);
    (layout, word(ctx, PREFERENCE_WORDS))
}

fn bind(kernel: &mut Kernel, id: u32, address: u64, size: u64) {
    let memory = kernel.nx2_graphics.native.memories.entry(id).or_default();
    memory.address = address;
    if memory.size == 0 {
        memory.size = size;
    }
    log::debug!("[nx2] native memory {id} bound at {address:#x} size={:#x}", memory.size);
}

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    if matches!(cmd_id, 429 | 435 | 1510) {
        let (layout, _) = preference(ctx);
        nexium_common::render_scale::set_reference(layout.width, layout.height);
    }
    let state = &mut kernel.nx2_graphics.native;
    match cmd_id {
        111 => {
            state.next_memory += 1;
            let id = state.next_memory;
            state.memories.insert(id, NativeMemory { address: 0, size: long(ctx, 1) });
            Some(reply_words(ctx, &[id]))
        }
        112 => {
            state.memories.remove(&word(ctx, 0));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        114 => {
            bind(kernel, word(ctx, 0), long(ctx, 1) + long(ctx, 7), long(ctx, 3));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        423 => {
            bind(kernel, word(ctx, 0), long(ctx, 1) + long(ctx, 6), long(ctx, 3));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        429 | 1211 => {
            let (layout, count) = preference(ctx);
            let memory = word(ctx, PREFERENCE_WORDS + 1);
            let first = state.next_image + 1;
            for index in 0..count.max(1) {
                let offset = u64::from(index) * layout.size;
                state.images.insert(first + index, NativeImage { memory, offset, layout });
            }
            state.next_image += count.max(1);
            log::info!(
                "[nx2] native images {first}..{} in memory {memory}: {}x{} format={} pitch={} block_rows={} size={:#x}",
                first + count.max(1) - 1,
                layout.width,
                layout.height,
                layout.format,
                layout.pitch,
                layout.rows_per_block,
                layout.size
            );
            Some(reply_words(ctx, &[first]))
        }
        430 => {
            state.images.remove(&word(ctx, 0));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        435 => {
            let (layout, count) = preference(ctx);
            Some(reply_longs(ctx, &[layout.size * u64::from(count.max(1))]))
        }
        1291 => {
            let (layout, count) = preference(ctx);
            Some(reply_longs(ctx, &[layout.size * u64::from(count.max(1)), layout.alignment]))
        }
        426 | 428 => {
            let image = word(ctx, 0);
            let Some(found) = state.images.get(&image).copied() else {
                log::warn!("[nx2] gsv {cmd_id} unknown image {image}");
                return Some(build_ipc_response(ctx, 0, &[0; 0x9C], &[]));
            };
            let base = state.memories.get(&found.memory).map_or(0, |memory| memory.address);
            let address = if cmd_id == 426 { base + found.offset } else { 0 };
            let mut out = Vec::with_capacity(0x9C);
            out.extend_from_slice(&address.to_le_bytes());
            out.extend_from_slice(&found.offset.to_le_bytes());
            out.extend_from_slice(&image_descriptor(&found.layout));
            out.extend_from_slice(&0u32.to_le_bytes());
            Some(build_ipc_response(ctx, 0, &out, &[]))
        }
        1510 => {
            let (layout, _) = preference(ctx);
            Some(reply_longs(ctx, &[layout.size, layout.alignment]))
        }
        314 | 315 => Some(build_ipc_response(ctx, 0, &[], &[])),
        _ => None,
    }
}

fn image_descriptor(layout: &ImageLayout) -> [u8; 128] {
    let mut descriptor = [0u8; 128];
    let mut put = |offset: usize, value: u32| descriptor[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    put(0x00, 1);
    put(0x04, layout.kind);
    put(0x08, (layout.width - 1) | ((layout.height - 1) << 16));
    put(0x10, layout.rows_per_block.trailing_zeros());
    put(0x14, (layout.pitch / 64) as u32);
    put(0x20, 1);
    put(0x24, if (0x99..=0x9C).contains(&layout.format) { 2 } else { 1 });
    put(0x2C, layout.size as u32);
    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_image_layout_matches_the_title_offsets() {
        let preference = [0, 1, 0x059f_09ff, 0, 7, 0xa0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let layout = ImageLayout::from_preference(&preference);
        assert_eq!((layout.width, layout.height), (2560, 1440));
        assert_eq!((layout.pitch, layout.rows_per_block), (10240, 128));
        assert_eq!(layout.size, 0xF0_0000);
        assert_eq!(layout.alignment, 0x2000);
    }

    #[test]
    fn unusable_pitch_falls_back_to_packed_rows() {
        let preference = [0, 1, (215 << 16) | 383, 0, 9, 0, 0x38];
        let layout = ImageLayout::from_preference(&preference);
        assert_eq!((layout.pitch, layout.rows_per_block), (1536, 128));
        assert_eq!(layout.size, 1536 * 256);
    }

    #[test]
    fn camera_image_descriptor_passes_the_sdk_checks() {
        let preference = [0, 2, (215 << 16) | 383, 0, 9, 0, 0x38];
        let descriptor = image_descriptor(&ImageLayout::from_preference(&preference));
        let read = |offset: usize| u32::from_le_bytes(descriptor[offset..offset + 4].try_into().unwrap());
        assert_eq!((read(0x00), read(0x04), read(0x08)), (1, 2, (215 << 16) | 383));
        assert_eq!((read(0x10), read(0x14), read(0x20), read(0x24)), (7, 24, 1, 1));
        assert_eq!(read(0x2C), 1536 * 256);
    }
}

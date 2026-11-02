use super::{build_ipc_response, build_ipc_response_copy, new_event, open_session, reply_longs, reply_words, signal, word, Kernel, VDEC};
use nexium_ipc::IpcCtx;
use std::collections::VecDeque;

const NO_OUTPUT: u32 = 0x1c286f;
const BLOCK_ROWS: u64 = 128;
const PREFERENCE_WORDS: usize = 16;

fn image_layout(code: u32, extent: u32) -> (u64, u64, u64) {
    let (width, height) = (u64::from(extent & 0xFFFF) + 1, u64::from(extent >> 16) + 1);
    let nv12 = matches!(code, 1 | 4);
    let pitch = (width * if nv12 { 1 } else { 4 }).next_multiple_of(64);
    let chroma = pitch * height.next_multiple_of(BLOCK_ROWS);
    let size = chroma + if nv12 { pitch * height.div_ceil(2).next_multiple_of(BLOCK_ROWS) } else { 0 };
    (pitch, chroma, size)
}

pub(super) fn image_pool(ctx: &IpcCtx) -> Vec<u8> {
    let (_, _, size) = image_layout(word(ctx, 1), word(ctx, 2));
    reply_longs(ctx, &[size * u64::from(word(ctx, PREFERENCE_WORDS).max(1)), 64 * 128])
}

pub(super) fn create_images(kernel: &mut Kernel, ctx: &IpcCtx) -> Vec<u8> {
    let (code, extent, count) = (word(ctx, 1), word(ctx, 2), word(ctx, PREFERENCE_WORDS).max(1));
    let (_, _, size) = image_layout(code, extent);
    let first = kernel.nx2_graphics.native.reserve_images(count);
    for index in 0..count {
        kernel.nx2_graphics.decoder_images.insert(first + index, (code, extent, size * u64::from(index)));
    }
    reply_words(ctx, &[first])
}

pub(super) fn describe_image(kernel: &mut Kernel, ctx: &IpcCtx) -> Vec<u8> {
    let mut words = [0u32; 32];
    if let Some(&(code, extent, offset)) = kernel.nx2_graphics.decoder_images.get(&word(ctx, 0)) {
        let (pitch, chroma, _) = image_layout(code, extent);
        let nv12 = matches!(code, 1 | 4);
        words[0] = if nv12 { 0xA } else if matches!(code, 2 | 5) { 0x9 } else { 0x8 };
        words[2] = offset as u32;
        words[4] = pitch as u32;
        if nv12 {
            words[6] = (offset + chroma) as u32;
            words[8] = pitch as u32;
        }
    }
    reply_words(ctx, &words)
}

#[derive(Default)]
pub(super) struct Decoder {
    syncpoint: Option<(u32, u32)>,
    handed: u32,
    width: u32,
    height: u32,
    event: Option<u32>,
    free: VecDeque<u32>,
    inputs: VecDeque<(u64, bool)>,
    outputs: VecDeque<(u32, u64, bool)>,
    submitted: u64,
}

pub(super) fn open(kernel: &mut Kernel, ctx: &IpcCtx) -> Vec<u8> {
    kernel.nx2_graphics.video_sessions += 1;
    let name = format!("{VDEC}#{}", kernel.nx2_graphics.video_sessions);
    open_session(kernel, ctx, &name)
}

fn decoder<'a>(kernel: &'a mut Kernel, name: &str) -> &'a mut Decoder {
    kernel.nx2_graphics.video.entry(name.to_string()).or_default()
}

fn ready_event(kernel: &mut Kernel, name: &str) -> u32 {
    let event = match decoder(kernel, name).event {
        Some(event) => event,
        None => {
            let event = new_event(kernel);
            decoder(kernel, name).event = Some(event);
            event
        }
    };
    signal(kernel, event);
    event
}

fn hand_out(kernel: &mut Kernel, name: &str, image: u32) -> [u32; 4] {
    let (syncpoint, base) = match decoder(kernel, name).syncpoint {
        Some(syncpoint) => syncpoint,
        None => {
            let id = kernel.nvdrv.nx2_allocate_syncpoint();
            let syncpoint = (id, kernel.nvdrv.nx2_syncpoint_value(id));
            decoder(kernel, name).syncpoint = Some(syncpoint);
            syncpoint
        }
    };
    let decoder = decoder(kernel, name);
    decoder.free.push_back(image);
    decoder.handed += 1;
    [!syncpoint, 0, base.wrapping_add(decoder.handed), 0]
}

impl Decoder {
    fn pair(&mut self) -> usize {
        let mut filled = 0;
        while let (Some(&(pts, end)), Some(&image)) = (self.inputs.front(), self.free.front()) {
            self.inputs.pop_front();
            self.free.pop_front();
            self.outputs.push_back((image, pts, end));
            filled += 1;
        }
        filled
    }
}

fn fill(kernel: &mut Kernel, name: &str) {
    let decoder = decoder(kernel, name);
    let Some((syncpoint, _)) = decoder.syncpoint else {
        return;
    };
    for _ in 0..decoder.pair() {
        kernel.nvdrv.nx2_increment_syncpoint(syncpoint);
    }
}

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, name: &str, cmd_id: u32) -> Option<Vec<u8>> {
    match cmd_id {
        1 => Some(open(kernel, ctx)),
        3 => {
            let event = ready_event(kernel, name);
            Some(build_ipc_response_copy(ctx, 0, &[], &[event]))
        }
        6 | 115 | 200 | 201 => Some(build_ipc_response(ctx, 0, &[], &[])),
        8 | 207 => {
            decoder(kernel, name).inputs.clear();
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        101 => {
            let decoder = decoder(kernel, name);
            (decoder.width, decoder.height) = (word(ctx, 0), word(ctx, 1));
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        109 => {
            let pts = u64::from(word(ctx, 0)) | (u64::from(word(ctx, 1)) << 32);
            let decoder = decoder(kernel, name);
            decoder.inputs.push_back((pts, word(ctx, 2) & 0xFF != 0));
            decoder.submitted += 1;
            let submitted = decoder.submitted;
            fill(kernel, name);
            ready_event(kernel, name);
            Some(reply_longs(ctx, &[submitted]))
        }
        212 => {
            let image = word(ctx, 0);
            decoder(kernel, name).free.retain(|&free| free != image);
            Some(build_ipc_response(ctx, 0, &[], &[]))
        }
        213 => {
            let decoder = decoder(kernel, name);
            let (width, height) = (decoder.width, decoder.height);
            Some(match decoder.outputs.pop_front() {
                Some((image, pts, end)) => {
                    let ns = pts.saturating_mul(1000);
                    reply_words(ctx, &[image, width, height, width, height, 0, ns as u32, (ns >> 32) as u32, u32::from(end), 0, 0, 0, 0, 0, 0, 0])
                }
                None => build_ipc_response(ctx, NO_OUTPUT, &[], &[]),
            })
        }
        311 | 314 => {
            let handle = hand_out(kernel, name, word(ctx, 0));
            fill(kernel, name);
            Some(reply_words(ctx, &handle))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{image_layout, Decoder};

    #[test]
    fn decoder_images_are_block_row_padded() {
        let extent = 639 | (359 << 16);
        assert_eq!(image_layout(1, extent), (640, 640 * 384, 640 * 384 + 640 * 256));
        assert_eq!(image_layout(3, extent).2, 2560 * 384);
    }

    #[test]
    fn frames_fill_images_in_hand_out_order() {
        let mut decoder = Decoder::default();
        decoder.inputs.extend([(10, false), (20, true)]);
        decoder.free.push_back(7);
        assert_eq!(decoder.pair(), 1);
        decoder.free.push_back(8);
        assert_eq!(decoder.pair(), 1);
        assert_eq!(decoder.outputs, [(7, 10, false), (8, 20, true)]);
        assert!(decoder.inputs.is_empty() && decoder.free.is_empty());
    }
}

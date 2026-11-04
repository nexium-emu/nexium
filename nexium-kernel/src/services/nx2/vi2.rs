use super::{build_ipc_response, build_ipc_response_copy, new_event, reply_longs, reply_words, word, Kernel};
use nexium_ipc::IpcCtx;

const DISPLAY_COUNT: u32 = 1;
const TIMESTAMP_RECORDERS: u32 = 7;
const FREE_SLOTS: u64 = 0x7FFF;

pub(super) fn dispatch(kernel: &mut Kernel, ctx: &mut IpcCtx, cmd_id: u32) -> Option<Vec<u8>> {
    match cmd_id {
        1 | 3 => Some(reply_words(ctx, &[DISPLAY_COUNT])),
        5 => Some(reply_words(ctx, &[TIMESTAMP_RECORDERS])),
        11 => {
            let handle = new_event(kernel, "vi2_cmd11");
            Some(build_ipc_response_copy(ctx, 0, &[], &[handle]))
        }
        31 | 32 => {
            let handle = new_event(kernel, "vi2_vsync");
            kernel.vsync_handles.insert(handle);
            log::info!("[nx2] vi2 cmd={cmd_id} arg={:#x} vsync event {handle:#x}", word(ctx, 0));
            Some(build_ipc_response_copy(ctx, 0, &[], &[handle]))
        }
        104 | 105 => Some(build_ipc_response(ctx, 0, &[], &[])),
        212 => Some(reply_longs(ctx, &[FREE_SLOTS])),
        228 => Some(reply_longs(ctx, &[FREE_SLOTS, 0, 0])),
        229 => {
            present(kernel, word(ctx, 3), word(ctx, 5), word(ctx, 20), word(ctx, 22));
            Some(reply_longs(ctx, &[FREE_SLOTS, 0, 0]))
        }
        _ => None,
    }
}

fn present(kernel: &mut Kernel, image: u32, crop: u32, counter: u32, value: u32) {
    let Some((address, layout)) = kernel.nx2_graphics.native.image_address(image) else {
        log::warn!("[nx2] vi2 present of unknown image {image}");
        return;
    };
    kernel.nx2_graphics.presents += 1;
    let presents = kernel.nx2_graphics.presents;
    if presents <= 4 || presents % 600 == 0 {
        log::info!(
            "[nx2] vi2 present #{presents} image={image} at {address:#x} {}x{} fence=({:#x}, {value})",
            layout.width,
            layout.height,
            !counter
        );
    }
    super::peek_memory(kernel, presents);
    let renderer = kernel.nvdrv.renderer().cloned();
    let renderer_for_dark = renderer.clone();
    let address_space = std::sync::Arc::clone(&kernel.address_space);
    kernel.nvdrv.submit_present_job("nx2-present", move |submit| {
        let drawn = crop_extent(crop, (layout.width, layout.height), (layout.width, layout.height));
        if super::dump_directory(presents).is_none()
            && renderer.as_ref().is_some_and(|renderer| present_on_gpu(renderer, address, drawn))
        {
            return;
        }
        if let Some(renderer) = renderer.as_ref() {
            super::dump_render_targets(presents, renderer);
        }
        let (mut width, mut height) = (layout.width, layout.height);
        let rendered = renderer.and_then(|renderer| renderer.readback_target_scaled_at(0, width, height, address));
        let mut pixels = match rendered {
            Some((read_width, read_height, pixels)) => {
                (width, height) = (read_width, read_height);
                pixels
            }
            None => {
                let mut tiled = vec![0u8; layout.size as usize];
                let _ = address_space.read(address, &mut tiled);
                super::swizzle::unswizzle(&tiled, width as usize * 4, height as usize, layout.pitch as usize, layout.rows_per_block as usize)
            }
        };
        let (crop_width, crop_height) = crop_extent(crop, (layout.width, layout.height), (width, height));
        if (crop_width, crop_height) != (width, height) {
            pixels = crop_top_left(&pixels, width, crop_width, crop_height);
            (width, height) = (crop_width, crop_height);
        }
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = 0xFF;
        }
        if let Some(renderer) = renderer_for_dark.as_ref() {
            if super::dark_dump_due(presents, width, height, &pixels) {
                super::dump_dark_frame(presents, renderer, width, height, &pixels);
            }
        }
        super::dump_frame(presents, width, height, &pixels);
        submit(nexium_nvdrv::QueuedFrame { width, height, pixels, present_at: None, depth: None });
    });
}

fn present_on_gpu(renderer: &nexium_gpu::Renderer, address: u64, (width, height): (u32, u32)) -> bool {
    let Some((key, stamp)) = renderer.render_target_at_va(0, address) else {
        return false;
    };
    let parameters = nexium_gpu::presentation::PresentParameters {
        read_rect: Some([0, 0, width, height]),
        crop: None,
        flip_y: false,
        transform: 0,
        present_at: std::time::Instant::now(),
    };
    matches!(renderer.present_image(key, stamp, true, parameters, None), Ok(true))
}

fn crop_extent(crop: u32, image: (u32, u32), frame: (u32, u32)) -> (u32, u32) {
    let (crop_width, crop_height) = ((crop & 0xFFFF) + 1, (crop >> 16) + 1);
    if crop == 0 || crop_width > image.0 || crop_height > image.1 {
        return frame;
    }
    let scale = |length: u32, frame: u32, image: u32| ((u64::from(length) * u64::from(frame) / u64::from(image)) as u32).max(1);
    (scale(crop_width, frame.0, image.0), scale(crop_height, frame.1, image.1))
}

fn crop_top_left(pixels: &[u8], width: u32, crop_width: u32, crop_height: u32) -> Vec<u8> {
    pixels
        .chunks_exact(width as usize * 4)
        .take(crop_height as usize)
        .flat_map(|row| &row[..crop_width as usize * 4])
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_crop_keeps_the_drawn_top_left_region() {
        assert_eq!(crop_extent(0x059f_09ff, (3840, 2160), (1920, 1080)), (1280, 720));
        assert_eq!(crop_extent(0x059f_09ff, (3840, 2160), (3840, 2160)), (2560, 1440));
        assert_eq!(crop_extent(0x0437_077f, (3840, 2160), (3840, 2160)), (1920, 1080));
        assert_eq!(crop_extent(0, (3840, 2160), (1920, 1080)), (1920, 1080));
        assert_eq!(crop_extent(0x086f_0eff, (2560, 1440), (2560, 1440)), (2560, 1440));
        let pixels: Vec<u8> = (0..4 * 2 * 4).map(|value| value as u8).collect();
        assert_eq!(crop_top_left(&pixels, 4, 2, 1), (0..8).collect::<Vec<u8>>());
    }
}

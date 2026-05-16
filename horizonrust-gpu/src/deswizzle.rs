const GOB_WIDTH_BYTES: usize = 64;
const GOB_HEIGHT: usize = 8;
const GOB_SIZE_BYTES: usize = 512;

pub fn unswizzle_block_linear(
    src: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let stride = width * bytes_per_pixel;
    let mut dst = vec![0u8; stride * height];

    let gobs_per_row = (width * bytes_per_pixel + GOB_WIDTH_BYTES - 1) / GOB_WIDTH_BYTES;

    for y in 0..height {
        for x in 0..width {
            let pixel_x = x * bytes_per_pixel;
            let gob_x = pixel_x / GOB_WIDTH_BYTES;
            let gob_y = y / GOB_HEIGHT;
            let offset_x = pixel_x % GOB_WIDTH_BYTES;
            let offset_y = y % GOB_HEIGHT;

            let gob_index = gob_y * gobs_per_row + gob_x;
            let src_offset = gob_index * GOB_SIZE_BYTES + offset_y * GOB_WIDTH_BYTES + offset_x;

            if src_offset + bytes_per_pixel <= src.len() {
                let dst_offset = y * stride + pixel_x;
                if dst_offset + bytes_per_pixel <= dst.len() {
                    dst[dst_offset..dst_offset + bytes_per_pixel]
                        .copy_from_slice(&src[src_offset..src_offset + bytes_per_pixel]);
                }
            }
        }
    }

    dst
}

pub fn swizzle_block_linear(
    src: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let stride = width * bytes_per_pixel;

    let gobs_per_row = (width * bytes_per_pixel + GOB_WIDTH_BYTES - 1) / GOB_WIDTH_BYTES;
    let total_gobs = gobs_per_row * ((height + GOB_HEIGHT - 1) / GOB_HEIGHT);
    let mut dst = vec![0u8; total_gobs * GOB_SIZE_BYTES];

    for y in 0..height {
        for x in 0..width {
            let pixel_x = x * bytes_per_pixel;
            let gob_x = pixel_x / GOB_WIDTH_BYTES;
            let gob_y = y / GOB_HEIGHT;
            let offset_x = pixel_x % GOB_WIDTH_BYTES;
            let offset_y = y % GOB_HEIGHT;

            let gob_index = gob_y * gobs_per_row + gob_x;
            let dst_offset = gob_index * GOB_SIZE_BYTES + offset_y * GOB_WIDTH_BYTES + offset_x;

            let src_offset = y * stride + pixel_x;
            if src_offset + bytes_per_pixel <= src.len() && dst_offset + bytes_per_pixel <= dst.len() {
                dst[dst_offset..dst_offset + bytes_per_pixel]
                    .copy_from_slice(&src[src_offset..src_offset + bytes_per_pixel]);
            }
        }
    }

    dst
}

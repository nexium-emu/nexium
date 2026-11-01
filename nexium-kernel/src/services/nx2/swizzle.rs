const GOB_WIDTH: usize = 64;
const GOB_HEIGHT: usize = 8;
const GOB_BYTES: usize = GOB_WIDTH * GOB_HEIGHT;
const SECTOR: usize = 16;

fn gob_offset(x: usize, y: usize) -> usize {
    let tegra = (x / 32) * 256 + (y / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + (x % 16);
    let bit5 = (tegra >> 5) & 1;
    let bit6 = (tegra >> 6) & 1;
    (tegra & !0x60) | (bit5 << 6) | (bit6 << 5)
}

pub(super) fn unswizzle(source: &[u8], width_bytes: usize, height: usize, pitch: usize, block_rows: usize) -> Vec<u8> {
    let gobs_x = pitch.max(width_bytes).div_ceil(GOB_WIDTH);
    let block_gobs = (block_rows / GOB_HEIGHT).max(1);
    let mut linear = vec![0u8; width_bytes * height];
    for y in 0..height {
        let gob_y = y / GOB_HEIGHT;
        let block_base = (gob_y / block_gobs) * gobs_x * block_gobs + gob_y % block_gobs;
        let row = &mut linear[y * width_bytes..(y + 1) * width_bytes];
        for x in (0..width_bytes).step_by(SECTOR) {
            let gob = block_base + (x / GOB_WIDTH) * block_gobs;
            let offset = gob * GOB_BYTES + gob_offset(x % GOB_WIDTH, y % GOB_HEIGHT);
            let length = SECTOR.min(width_bytes - x);
            if let Some(chunk) = source.get(offset..offset + length) {
                row[x..x + length].copy_from_slice(chunk);
            }
        }
    }
    linear
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gob_offsets_swap_bits_five_and_six_of_the_tegra_layout() {
        assert_eq!(gob_offset(0, 0), 0);
        assert_eq!(gob_offset(16, 0), 64);
        assert_eq!(gob_offset(0, 2), 32);
        assert_eq!(gob_offset(32, 0), 256);
        assert_eq!(gob_offset(0, 1), 16);
        let mut seen = vec![false; GOB_BYTES];
        for y in 0..GOB_HEIGHT {
            for x in 0..GOB_WIDTH {
                let offset = gob_offset(x, y);
                assert!(!seen[offset]);
                seen[offset] = true;
            }
        }
    }

    #[test]
    fn blocks_stack_gobs_vertically_then_advance_across_the_row() {
        let (width, height, pitch, rows) = (128usize, 32usize, 128usize, 16usize);
        let mut tiled = vec![0u8; pitch * height];
        for y in 0..height {
            for x in (0..width).step_by(SECTOR) {
                let gob_y = y / GOB_HEIGHT;
                let gob = (gob_y / 2) * 2 * 2 + (x / GOB_WIDTH) * 2 + gob_y % 2;
                let offset = gob * GOB_BYTES + gob_offset(x % GOB_WIDTH, y % GOB_HEIGHT);
                tiled[offset] = (y * 7 + x / SECTOR) as u8;
            }
        }
        let linear = unswizzle(&tiled, width, height, pitch, rows);
        for y in 0..height {
            for x in (0..width).step_by(SECTOR) {
                assert_eq!(linear[y * width + x], (y * 7 + x / SECTOR) as u8, "x={x} y={y}");
            }
        }
    }
}

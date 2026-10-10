use nexium_gpu::bcn_encode::{self, BcTarget};
use nexium_gpu::Renderer;

const BLOCK_SIZES: [(u32, u32); 14] = [
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
    (10, 5),
    (10, 6),
    (10, 8),
    (10, 10),
    (12, 10),
    (12, 12),
];

const WEIGHT_A: [u32; 16] = [0, 0, 0, 3, 0, 5, 3, 0, 0, 0, 5, 3, 0, 5, 3, 0];
const WEIGHT_B: [u32; 16] = [0, 0, 1, 0, 2, 0, 1, 3, 0, 0, 1, 2, 4, 2, 3, 5];

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn spec_valid(block: &[u8; 16]) -> bool {
    let b0 = u32::from(block[0]);
    let b1 = u32::from(block[1]);
    if b0 == 0xfc && b1 & 1 == 1 {
        return true;
    }
    if ((b0 & 0xc3) == 0xc0 && (b1 & 1) == 1) || (b0 & 0xf) == 0 {
        return false;
    }
    let w16 = b0 | (b1 << 8);
    let mut dual = b1 & 4 != 0;
    let mut range = ((b0 >> 4) & 1) | ((b1 << 2) & 8);
    let (width, height);
    if b0 & 3 != 0 {
        range |= (b0 << 1) & 6;
        (width, height) = match b0 & 0xc {
            0 => (((w16 >> 7) & 3) + 4, ((b0 >> 5) & 3) + 2),
            4 => (((w16 >> 7) & 3) + 8, ((b0 >> 5) & 3) + 2),
            8 => (((b0 >> 5) & 3) + 2, ((w16 >> 7) & 3) + 8),
            _ if b1 & 1 != 0 => (((b0 >> 7) & 1) + 2, ((b0 >> 5) & 3) + 2),
            _ => (((b0 >> 5) & 3) + 2, ((b0 >> 7) & 1) + 6),
        };
    } else {
        range |= (b0 >> 1) & 6;
        (width, height) = match w16 & 0x180 {
            0 => (12, ((b0 >> 5) & 3) + 2),
            0x80 => (((b0 >> 5) & 3) + 2, 12),
            0x100 => {
                dual = false;
                range &= 7;
                (((b0 >> 5) & 3) + 6, ((b1 >> 1) & 3) + 6)
            }
            _ => {
                if b0 & 0x20 != 0 {
                    (10, 6)
                } else {
                    (6, 10)
                }
            }
        };
    }
    let partitions = ((b1 >> 3) & 3) + 1;
    if partitions == 4 && dual {
        return false;
    }
    let count = width * height * if dual { 2 } else { 1 };
    let (a, b) = (WEIGHT_A[range as usize], WEIGHT_B[range as usize]);
    if count > 64 || (a == 0 && b == 0) {
        return false;
    }
    let bits = count * b
        + match a {
            3 => (count * 8 + 4) / 5,
            5 => (count * 7 + 2) / 3,
            _ => 0,
        };
    let config = if partitions == 1 { 17 } else { 29 } + if dual { 2 } else { 0 } + if partitions > 1 { 3 * partitions } else { 0 };
    (24..=96).contains(&bits) && bits + config <= 128
}

fn cpu_block(block: &[u8; 16], width: u32, height: u32) -> Option<Vec<u32>> {
    let block = *block;
    std::panic::catch_unwind(move || {
        let mut out = vec![0u32; (width * height) as usize];
        texture2ddecoder::decode_astc(&block, width as usize, height as usize, width as usize, height as usize, &mut out)
            .ok()
            .map(|_| out)
    })
    .ok()
    .flatten()
}

fn rgba(pixel: u32) -> [u8; 4] {
    [(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8, (pixel >> 24) as u8]
}

fn random_astc(state: &mut u64, blocks_x: u32, blocks_y: u32, block_width: u32, block_height: u32) -> (Vec<u8>, Vec<Vec<u32>>) {
    let mut blocks = Vec::with_capacity((blocks_x * blocks_y * 16) as usize);
    let mut expected = Vec::new();
    for index in 0..blocks_x * blocks_y {
        let block = loop {
            let mut candidate = [0u8; 16];
            for chunk in candidate.chunks_exact_mut(8) {
                chunk.copy_from_slice(&next(state).to_le_bytes());
            }
            if index % 97 == 0 {
                candidate[0] = 0xfc;
                candidate[1] |= 1;
            }
            if !spec_valid(&candidate) {
                continue;
            }
            if let Some(decoded) = cpu_block(&candidate, block_width, block_height) {
                break (candidate, decoded);
            }
        };
        blocks.extend_from_slice(&block.0);
        expected.push(block.1);
    }
    (blocks, expected)
}

fn assemble_rgba(expected: &[Vec<u32>], blocks_x: u32, block_width: u32, block_height: u32, width: u32, height: u32) -> Vec<u8> {
    let mut image = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let block = &expected[((y / block_height) * blocks_x + x / block_width) as usize];
            let texel = block[((y % block_height) * block_width + x % block_width) as usize];
            let offset = ((y * width + x) * 4) as usize;
            image[offset..offset + 4].copy_from_slice(&rgba(texel));
        }
    }
    image
}

#[test]
#[ignore = "requires a Vulkan device"]
fn gpu_astc_decode_matches_cpu_for_random_blocks() {
    let renderer = Renderer::new().unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut compared = 0usize;
    let mut colored = 0usize;
    for (block_width, block_height) in BLOCK_SIZES {
        let blocks_x = 48u32;
        let blocks_y = 48u32;
        let width = blocks_x * block_width - 1;
        let height = blocks_y * block_height - 2;
        let (blocks, expected) = random_astc(&mut state, blocks_x, blocks_y, block_width, block_height);
        let gpu = renderer
            .decode_astc_gpu(&blocks, width, height, block_width, block_height, None)
            .unwrap();
        assert_eq!(gpu.len(), (width * height * 4) as usize);
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let reference = &expected[(by * blocks_x + bx) as usize];
                for t in 0..block_height {
                    for s in 0..block_width {
                        let (x, y) = (bx * block_width + s, by * block_height + t);
                        if x >= width || y >= height {
                            continue;
                        }
                        let offset = ((y * width + x) * 4) as usize;
                        let want = rgba(reference[(t * block_width + s) as usize]);
                        assert_eq!(
                            &gpu[offset..offset + 4],
                            &want,
                            "{block_width}x{block_height} block ({bx},{by}) texel ({s},{t}) data {:02x?}",
                            &blocks[((by * blocks_x + bx) * 16) as usize..((by * blocks_x + bx) * 16 + 16) as usize],
                        );
                        compared += 1;
                        colored += usize::from(want != [0xff, 0x00, 0xff, 0xff]);
                    }
                }
            }
        }
    }
    std::panic::set_hook(previous_hook);
    assert!(compared > 1_000_000);
    assert!(colored * 10 > compared * 9, "{colored} of {compared} texels were real colors");
}

#[test]
#[ignore = "requires a Vulkan device"]
fn gpu_bc_recompression_matches_the_cpu_encoder() {
    let renderer = Renderer::new().unwrap();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut total = 0usize;
    let mut equal = 0usize;
    for (block_width, block_height) in [(4u32, 4u32), (5, 5), (6, 6), (8, 8), (10, 5), (12, 12)] {
        let blocks_x = 32u32;
        let blocks_y = 24u32;
        let width = blocks_x * block_width - 3;
        let height = blocks_y * block_height - 1;
        let (blocks, expected) = random_astc(&mut state, blocks_x, blocks_y, block_width, block_height);
        let image = assemble_rgba(&expected, blocks_x, block_width, block_height, width, height);
        for target in [BcTarget::Bc1, BcTarget::Bc3] {
            let gpu = renderer
                .decode_astc_gpu(&blocks, width, height, block_width, block_height, Some(target))
                .unwrap();
            let cpu = bcn_encode::encode(&image, width, height, target);
            assert_eq!(gpu.len(), cpu.len(), "{block_width}x{block_height} {target:?}");
            let mut level_equal = 0usize;
            let mut level_total = 0usize;
            for (gpu_block, cpu_block) in gpu.chunks_exact(target.block_bytes()).zip(cpu.chunks_exact(target.block_bytes())) {
                level_total += 1;
                level_equal += usize::from(gpu_block == cpu_block);
            }
            println!("{block_width}x{block_height} {target:?}: {level_equal} of {level_total} blocks identical");
            total += level_total;
            equal += level_equal;
        }
    }
    std::panic::set_hook(previous_hook);
    assert!(total > 10_000);
    assert!(equal * 1000 >= total * 995, "{equal} of {total} BC blocks matched the CPU encoder");
}

#[test]
#[ignore = "requires a Vulkan device"]
fn gpu_astc_batches_wrap_the_output_arena_and_grow_input_chunks() {
    let renderer = Renderer::new().unwrap();
    let mut state = 0x6a09_e667_f3bc_c908u64;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut textures = Vec::new();
    for index in 0..12u32 {
        let (block_width, block_height) = BLOCK_SIZES[index as usize % BLOCK_SIZES.len()];
        let blocks_x = 6 + index;
        let blocks_y = 5 + index % 3;
        let width = blocks_x * block_width - index % 3;
        let height = blocks_y * block_height - index % 2;
        let (blocks, expected) = random_astc(&mut state, blocks_x, blocks_y, block_width, block_height);
        let image = assemble_rgba(&expected, blocks_x, block_width, block_height, width, height);
        textures.push((blocks, image, width, height, block_width, block_height));
    }
    std::panic::set_hook(previous_hook);
    let views: Vec<(&[u8], u32, u32, u32, u32)> = textures
        .iter()
        .map(|texture| (texture.0.as_slice(), texture.2, texture.3, texture.4, texture.5))
        .collect();
    let largest = textures
        .iter()
        .map(|texture| u64::from(texture.2) * u64::from(texture.3) * 5)
        .max()
        .unwrap();
    for target in [None, Some(BcTarget::Bc3), Some(BcTarget::Bc1)] {
        let outputs = renderer
            .decode_astc_gpu_batch(&views, target, 256, (largest * 2).next_multiple_of(256) + 1024)
            .unwrap();
        assert_eq!(outputs.len(), textures.len());
        let mut blocks = 0usize;
        let mut equal = 0usize;
        for (texture, output) in textures.iter().zip(&outputs) {
            match target {
                None => assert!(output == &texture.1, "{}x{} decode differs", texture.2, texture.3),
                Some(target) => {
                    let cpu = bcn_encode::encode(&texture.1, texture.2, texture.3, target);
                    assert_eq!(output.len(), cpu.len());
                    for (gpu_block, cpu_block) in output
                        .chunks_exact(target.block_bytes())
                        .zip(cpu.chunks_exact(target.block_bytes()))
                    {
                        blocks += 1;
                        equal += usize::from(gpu_block == cpu_block);
                    }
                }
            }
        }
        assert!(equal * 100 >= blocks * 99, "{target:?}: {equal} of {blocks} blocks matched");
    }
}

//! Runs the GPU ASTC decoder (`Renderer::decode_astc_gpu`) on the real Vulkan
//! device and compares every decoded byte with the CPU decoder
//! (texture2ddecoder, through the same `decode_to_rgba8` fallback the renderer
//! uses), including void-extent (LDR and HDR), error and HDR-endpoint blocks.
//!
//! The test prints "SKIPPED" and passes when no Vulkan device is available.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::Arc;

use nexium_gpu::texture::{decode_to_rgba8, TicFormat};
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

const RANDOM_BLOCKS_PER_SIZE: usize = 6000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn block(&mut self) -> [u8; 16] {
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&self.next().to_le_bytes());
        block[8..].copy_from_slice(&self.next().to_le_bytes());
        block
    }
}

/// What a block exercises, parsed the way texture2ddecoder parses it.
#[derive(Clone, Copy, Debug)]
enum BlockKind {
    VoidLdr,
    VoidHdr,
    Error,
    Normal {
        part_num: u32,
        dual_plane: bool,
        weight_range: u32,
        cem: [u32; 4],
        cem_range: u32,
    },
}

const WEIGHT_PREC_A: [u32; 16] = [0, 0, 0, 3, 0, 5, 3, 0, 0, 0, 5, 3, 0, 5, 3, 0];
const WEIGHT_PREC_B: [u32; 16] = [0, 0, 1, 0, 2, 0, 1, 3, 0, 0, 1, 2, 4, 2, 3, 5];
const CEM_A: [u32; 19] = [0, 3, 5, 0, 3, 5, 0, 3, 5, 0, 3, 5, 0, 3, 5, 0, 3, 0, 0];
const CEM_B: [u32; 19] = [8, 6, 5, 7, 5, 4, 6, 4, 3, 5, 3, 2, 4, 2, 1, 3, 1, 2, 1];

fn bits(block: &[u8; 16], offset: u32, len: u32) -> u32 {
    let value = u128::from_le_bytes(*block);
    ((value >> offset) & ((1u128 << len) - 1)) as u32
}

/// Classifies a block, returning None for blocks the ASTC specification calls
/// illegal (texture2ddecoder does not check these and may panic on them).
fn classify(block: &[u8; 16], block_w: u32, block_h: u32) -> Option<BlockKind> {
    let b0 = u32::from(block[0]);
    let b1 = u32::from(block[1]);
    if b0 == 0xfc && b1 & 1 == 1 {
        return Some(if b1 & 2 != 0 { BlockKind::VoidHdr } else { BlockKind::VoidLdr });
    }
    if ((b0 & 0xc3) == 0xc0 && b1 & 1 == 1) || b0 & 0xf == 0 {
        return Some(BlockKind::Error);
    }
    let mode = bits(block, 0, 16);
    let mut dual_plane = b1 & 4 != 0;
    let mut weight_range = ((b0 >> 4) & 1) | ((b1 << 2) & 8);
    let (gw, gh);
    if b0 & 3 != 0 {
        weight_range |= (b0 << 1) & 6;
        (gw, gh) = match b0 & 0xc {
            0 => (((mode >> 7) & 3) + 4, ((b0 >> 5) & 3) + 2),
            4 => (((mode >> 7) & 3) + 8, ((b0 >> 5) & 3) + 2),
            8 => (((b0 >> 5) & 3) + 2, ((mode >> 7) & 3) + 8),
            _ if b1 & 1 != 0 => (((b0 >> 7) & 1) + 2, ((b0 >> 5) & 3) + 2),
            _ => (((b0 >> 5) & 3) + 2, ((b0 >> 7) & 1) + 6),
        };
    } else {
        weight_range |= (b0 >> 1) & 6;
        (gw, gh) = match mode & 0x180 {
            0 => (12, ((b0 >> 5) & 3) + 2),
            0x80 => (((b0 >> 5) & 3) + 2, 12),
            0x100 => {
                dual_plane = false;
                weight_range &= 7;
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
    let part_num = ((b1 >> 3) & 3) + 1;
    let weight_num = gw * gh * if dual_plane { 2 } else { 1 };
    let (a, b) = (WEIGHT_PREC_A[weight_range as usize], WEIGHT_PREC_B[weight_range as usize]);
    if a == 0 && b == 0 {
        return None;
    }
    let weight_bits = weight_num * b
        + match a {
            3 => (weight_num * 8 + 4) / 5,
            5 => (weight_num * 7 + 2) / 3,
            _ => 0,
        };
    if gw > block_w
        || gh > block_h
        || weight_num > 64
        || !(24..=96).contains(&weight_bits)
        || (part_num == 4 && dual_plane)
    {
        return None;
    }
    let mut cem = [0u32; 4];
    let mut config_bits;
    let mut cem_base = 0;
    if part_num == 1 {
        cem[0] = bits(block, 13, 4);
        config_bits = 17;
    } else {
        cem_base = bits(block, 23, 2);
        if cem_base == 0 {
            let shared = bits(block, 25, 4);
            cem[..part_num as usize].fill(shared);
            config_bits = 29;
        } else {
            let b3 = u32::from(block[3]);
            for (index, value) in cem.iter_mut().enumerate().take(part_num as usize) {
                *value = (((b3 >> (index + 1)) & 1) + cem_base - 1) << 2;
            }
            match part_num {
                2 => {
                    cem[0] |= (b3 >> 3) & 3;
                    cem[1] |= bits(block, 126 - weight_bits, 2);
                }
                3 => {
                    cem[0] |= (b3 >> 4) & 1;
                    cem[0] |= bits(block, 122 - weight_bits, 2) & 2;
                    cem[1] |= bits(block, 124 - weight_bits, 2);
                    cem[2] |= bits(block, 126 - weight_bits, 2);
                }
                _ => {
                    for (index, value) in cem.iter_mut().enumerate() {
                        *value |= bits(block, 120 + index as u32 * 2 - weight_bits, 2);
                    }
                }
            }
            config_bits = 25 + part_num * 3;
        }
    }
    let _ = cem_base;
    if dual_plane {
        config_bits += 2;
    }
    let remain = 128u32.checked_sub(config_bits + weight_bits)?;
    let values: u32 = cem[..part_num as usize].iter().map(|cem| ((cem >> 1) & 6) + 2).sum();
    if values > 18 {
        return None;
    }
    let cem_range = (0..19u32).find(|&index| {
        let a = CEM_A[index as usize];
        let b = CEM_B[index as usize];
        let needed = values * b
            + match a {
                3 => (values * 8).div_ceil(5),
                5 => (values * 7).div_ceil(3),
                _ => 0,
            };
        needed <= remain
    })?;
    Some(BlockKind::Normal {
        part_num,
        dual_plane,
        weight_range,
        cem,
        cem_range,
    })
}

thread_local! {
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// texture2ddecoder panics on some unusual blocks (mostly debug-build integer
/// overflow). Those blocks are left out, without spamming the test output.
/// Its `floor` also wraps `i32::MIN - 1` in release builds, so a negative
/// infinity half in an HDR void-extent block comes out as 255 instead of 0;
/// those blocks are left out too, since the panic only catches them in debug.
fn reference_accepts(block: &[u8; 16], block_w: u32, block_h: u32) -> bool {
    let negative_infinity = |at: usize| u16::from_le_bytes([block[at], block[at + 1]]) == 0xfc00;
    if block[0] == 0xfc && block[1] & 0x02 != 0 && (8..16).step_by(2).any(negative_infinity) {
        return false;
    }
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(Cell::get) {
                previous(info);
            }
        }));
    });
    QUIET.with(|quiet| quiet.set(true));
    let result = std::panic::catch_unwind(|| {
        let mut out = [0u32; 144];
        texture2ddecoder::decode_astc_block(block, block_w as usize, block_h as usize, &mut out);
    });
    QUIET.with(|quiet| quiet.set(false));
    result.is_ok()
}

fn void_extent(rgba: [u16; 4], hdr: bool) -> [u8; 16] {
    let mut block = [0xffu8; 16];
    block[0] = 0xfc;
    block[1] = if hdr { 0xff } else { 0xfd };
    for (index, value) in rgba.into_iter().enumerate() {
        block[8 + index * 2..10 + index * 2].copy_from_slice(&value.to_le_bytes());
    }
    block
}

fn edge_case_blocks(rng: &mut Rng) -> Vec<[u8; 16]> {
    let mut blocks = vec![[0u8; 16]];
    let mut reserved = rng.block();
    reserved[0] = 0xc0;
    reserved[1] |= 1;
    blocks.push(reserved);
    let mut low_nibble_zero = rng.block();
    low_nibble_zero[0] &= 0xf0;
    blocks.push(low_nibble_zero);
    for rgba in [
        [0, 0, 0, 0],
        [0xffff; 4],
        [0x1234, 0x8000, 0xff00, 0x00ff],
        [0x7f80, 0x0180, 0xfe7f, 0x8080],
    ] {
        blocks.push(void_extent(rgba, false));
    }
    let halves = [
        0x0000u16, 0x0001, 0x03ff, 0x0400, 0x1c00, 0x3800, 0x3bff, 0x3c00, 0x3c01, 0x4000,
        0x5bf8, 0x7bff, 0x7c00, 0x7e00, 0x8000, 0x8001, 0xbc00, 0xfc00, 0xfe00,
    ];
    for chunk in halves.chunks(4) {
        let mut rgba = [0x3c00u16; 4];
        rgba[..chunk.len()].copy_from_slice(chunk);
        blocks.push(void_extent(rgba, true));
    }
    blocks
}

struct Coverage {
    weight_ranges: BTreeSet<u32>,
    partitions: BTreeSet<u32>,
    dual_plane: BTreeSet<bool>,
    cems: BTreeSet<u32>,
    cem_ranges: BTreeSet<u32>,
    void_ldr: usize,
    void_hdr: usize,
    error: usize,
    hdr_blocks: usize,
}

impl Coverage {
    fn new() -> Self {
        Self {
            weight_ranges: BTreeSet::new(),
            partitions: BTreeSet::new(),
            dual_plane: BTreeSet::new(),
            cems: BTreeSet::new(),
            cem_ranges: BTreeSet::new(),
            void_ldr: 0,
            void_hdr: 0,
            error: 0,
            hdr_blocks: 0,
        }
    }

    fn add(&mut self, kind: BlockKind) {
        match kind {
            BlockKind::VoidLdr => self.void_ldr += 1,
            BlockKind::VoidHdr => self.void_hdr += 1,
            BlockKind::Error => self.error += 1,
            BlockKind::Normal {
                part_num,
                dual_plane,
                weight_range,
                cem,
                cem_range,
            } => {
                self.weight_ranges.insert(weight_range);
                self.partitions.insert(part_num);
                self.dual_plane.insert(dual_plane);
                self.cem_ranges.insert(cem_range);
                let used = &cem[..part_num as usize];
                self.cems.extend(used.iter().copied());
                if used.iter().any(|cem| matches!(cem, 2 | 3 | 7 | 11 | 14 | 15)) {
                    self.hdr_blocks += 1;
                }
            }
        }
    }
}

/// Valid random blocks plus edge cases for one block size, all accepted by
/// the reference decoder.
fn test_blocks(rng: &mut Rng, block_w: u32, block_h: u32, count: usize, coverage: &mut Coverage) -> Vec<[u8; 16]> {
    let mut blocks = Vec::with_capacity(count + 32);
    for block in edge_case_blocks(rng) {
        if let Some(kind) = classify(&block, block_w, block_h) {
            if reference_accepts(&block, block_w, block_h) {
                coverage.add(kind);
                blocks.push(block);
            }
        }
    }
    let mut attempts = 0usize;
    while blocks.len() < count && attempts < count * 400 {
        attempts += 1;
        let mut block = rng.block();
        // Most random block modes are illegal; bias towards legal ones by
        // sometimes clearing the high weight-grid bits.
        if attempts % 2 == 0 {
            block[0] &= 0x9f;
            block[1] &= 0xfe;
        }
        let Some(kind) = classify(&block, block_w, block_h) else {
            continue;
        };
        if matches!(kind, BlockKind::Error | BlockKind::VoidLdr | BlockKind::VoidHdr) {
            continue;
        }
        if reference_accepts(&block, block_w, block_h) {
            coverage.add(kind);
            blocks.push(block);
        }
    }
    blocks
}

fn renderer_or_skip(test: &str) -> Option<Arc<Renderer>> {
    match Renderer::new() {
        Ok(renderer) => Some(renderer),
        Err(error) => {
            println!("SKIPPED {test}: no Vulkan device ({error})");
            eprintln!("SKIPPED {test}: no Vulkan device ({error})");
            None
        }
    }
}

fn first_mismatch(expected: &[u8], actual: &[u8]) -> Option<usize> {
    expected.iter().zip(actual).position(|(a, b)| a != b)
}

#[test]
fn gpu_astc_decoder_matches_cpu_for_every_block_size() {
    let Some(renderer) = renderer_or_skip("gpu_astc_decoder_matches_cpu_for_every_block_size") else {
        return;
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut coverage = Coverage::new();
    let mut total_blocks = 0usize;
    for &(block_w, block_h) in BLOCK_SIZES.iter() {
        let mut blocks = test_blocks(&mut rng, block_w, block_h, RANDOM_BLOCKS_PER_SIZE, &mut coverage);
        let compared = blocks.len();
        assert!(compared >= RANDOM_BLOCKS_PER_SIZE, "{block_w}x{block_h}: only {compared} blocks");
        let blocks_x = 64u32;
        let blocks_y = (compared as u32).div_ceil(blocks_x);
        while blocks.len() < (blocks_x * blocks_y) as usize {
            blocks.push(void_extent([0x4000, 0x8000, 0xc000, 0xffff], false));
        }
        let width = blocks_x * block_w;
        let height = blocks_y * block_h;
        let raw: Vec<u8> = blocks.iter().flatten().copied().collect();
        let actual = renderer
            .decode_astc_gpu(&raw, width, height, block_w, block_h, None)
            .expect("GPU ASTC decode");
        let format = TicFormat::Astc(block_w as u8, block_h as u8);
        let expected = decode_to_rgba8(&raw, width, height, format);
        if let Some(byte) = first_mismatch(&expected, &actual) {
            let texel = byte / 4;
            let (x, y) = (texel as u32 % width, texel as u32 / width);
            let block = (y / block_h * blocks_x + x / block_w) as usize;
            let mismatched_blocks = (0..blocks.len())
                .filter(|&candidate| {
                    let bx = candidate as u32 % blocks_x;
                    let by = candidate as u32 / blocks_x;
                    (0..block_h).any(|ty| {
                        let row = ((by * block_h + ty) * width + bx * block_w) as usize * 4;
                        let len = block_w as usize * 4;
                        expected[row..row + len] != actual[row..row + len]
                    })
                })
                .count();
            panic!(
                "{block_w}x{block_h}: {mismatched_blocks} of {} blocks differ; first at texel ({x},{y}) block {block} {:02x?} {:?}: expected {:?} got {:?}",
                blocks.len(),
                blocks[block],
                classify(&blocks[block], block_w, block_h),
                &expected[texel * 4..texel * 4 + 4],
                &actual[texel * 4..texel * 4 + 4],
            );
        }
        println!("{block_w}x{block_h}: {compared} blocks ({width}x{height}) match");
        total_blocks += compared;
    }
    println!(
        "compared {total_blocks} blocks; weight ranges {:?}, partitions {:?}, dual plane {:?}, CEMs {:?}, endpoint ranges {:?}, void-extent LDR {} HDR {}, error {}, HDR-endpoint blocks {}",
        coverage.weight_ranges,
        coverage.partitions,
        coverage.dual_plane,
        coverage.cems,
        coverage.cem_ranges,
        coverage.void_ldr,
        coverage.void_hdr,
        coverage.error,
        coverage.hdr_blocks,
    );
    let legal_weight_ranges: BTreeSet<u32> = [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15].into();
    assert_eq!(coverage.weight_ranges, legal_weight_ranges);
    assert_eq!(coverage.partitions, [1, 2, 3, 4].into());
    assert_eq!(coverage.dual_plane, [false, true].into());
    assert_eq!(coverage.cems, (0..16).collect());
    assert!(coverage.cem_ranges.len() >= 15, "{:?}", coverage.cem_ranges);
    assert!(coverage.void_ldr > 0 && coverage.void_hdr > 0 && coverage.error > 0);
}

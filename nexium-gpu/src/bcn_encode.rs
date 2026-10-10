use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BcTarget {
    Bc1,
    Bc3,
}

impl BcTarget {
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc1 => 8,
            Self::Bc3 => 16,
        }
    }

    pub fn encoded_size(self, width: u32, height: u32) -> usize {
        width.div_ceil(4) as usize * height.div_ceil(4) as usize * self.block_bytes()
    }

    pub fn shader_mode(self) -> u32 {
        match self {
            Self::Bc1 => 1,
            Self::Bc3 => 3,
        }
    }

    pub fn format(self, srgb: bool) -> vk::Format {
        match (self, srgb) {
            (Self::Bc1, false) => vk::Format::BC1_RGBA_UNORM_BLOCK,
            (Self::Bc1, true) => vk::Format::BC1_RGBA_SRGB_BLOCK,
            (Self::Bc3, false) => vk::Format::BC3_UNORM_BLOCK,
            (Self::Bc3, true) => vk::Format::BC3_SRGB_BLOCK,
        }
    }

    pub fn for_format(format: vk::Format) -> Option<Self> {
        match format {
            vk::Format::BC1_RGBA_UNORM_BLOCK | vk::Format::BC1_RGBA_SRGB_BLOCK => Some(Self::Bc1),
            vk::Format::BC3_UNORM_BLOCK | vk::Format::BC3_SRGB_BLOCK => Some(Self::Bc3),
            _ => None,
        }
    }
}

type Texels = [[i32; 4]; 16];

const MIN_BLOCKS_PER_WORKER: usize = 1024;

pub fn encode(rgba: &[u8], width: u32, height: u32, target: BcTarget) -> Vec<u8> {
    let mut out = vec![0u8; target.encoded_size(width, height)];
    let (width, height) = (width as usize, height as usize);
    if width == 0 || height == 0 || rgba.len() < width * height * 4 {
        return out;
    }
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let row_bytes = blocks_x * target.block_bytes();
    let workers = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min((blocks_x * blocks_y) / MIN_BLOCKS_PER_WORKER)
        .clamp(1, 16);
    let rows_per_band = blocks_y.div_ceil(workers);
    let encode_rows = |first_row: usize, rows: &mut [u8]| {
        for (offset, row) in rows.chunks_exact_mut(row_bytes).enumerate() {
            let by = first_row + offset;
            for (bx, block) in row.chunks_exact_mut(target.block_bytes()).enumerate() {
                let (texels, inside) = load_block(rgba, width, height, bx, by);
                encode_block(&texels, inside, target, block);
            }
        }
    };
    if workers == 1 {
        encode_rows(0, &mut out);
        return out;
    }
    std::thread::scope(|scope| {
        let mut bands = out.chunks_mut(rows_per_band * row_bytes).enumerate();
        let first = bands.next();
        for (band, rows) in bands {
            scope.spawn(move || encode_rows(band * rows_per_band, rows));
        }
        if let Some((_, rows)) = first {
            encode_rows(0, rows);
        }
    });
    out
}

fn load_block(rgba: &[u8], width: usize, height: usize, bx: usize, by: usize) -> (Texels, u32) {
    let mut texels = [[0; 4]; 16];
    let mut inside = 0u32;
    for (index, texel) in texels.iter_mut().enumerate() {
        let x = bx * 4 + (index & 3);
        let y = by * 4 + (index >> 2);
        let offset = (y.min(height - 1) * width + x.min(width - 1)) * 4;
        *texel = [0, 1, 2, 3].map(|channel| i32::from(rgba[offset + channel]));
        if x < width && y < height {
            inside |= 1 << index;
        }
    }
    (texels, inside)
}

fn encode_block(texels: &Texels, inside: u32, target: BcTarget, out: &mut [u8]) {
    match target {
        BcTarget::Bc1 => {
            let transparent = (0..16)
                .filter(|&index| inside & (1 << index) != 0 && texels[index][3] < 128)
                .fold(0u32, |mask, index| mask | (1 << index));
            let (colors, indices) = if transparent == 0 {
                encode_color(texels, inside, false)
            } else {
                encode_punch_through(texels, inside & !transparent, transparent)
            };
            out[..4].copy_from_slice(&colors.to_le_bytes());
            out[4..8].copy_from_slice(&indices.to_le_bytes());
        }
        BcTarget::Bc3 => {
            let (alpha_low, alpha_high) = encode_alpha(texels, inside);
            let (colors, indices) = encode_color(texels, inside, false);
            out[..4].copy_from_slice(&alpha_low.to_le_bytes());
            out[4..8].copy_from_slice(&alpha_high.to_le_bytes());
            out[8..12].copy_from_slice(&colors.to_le_bytes());
            out[12..16].copy_from_slice(&indices.to_le_bytes());
        }
    }
}

fn encode_punch_through(texels: &Texels, opaque: u32, transparent: u32) -> (u32, u32) {
    let (colors, mut indices) = if opaque == 0 {
        (0, 0)
    } else {
        encode_color(texels, opaque, true)
    };
    for index in 0..16 {
        if transparent & (1 << index) != 0 {
            indices |= 3 << (2 * index);
        }
    }
    (colors, indices)
}

fn encode_color(texels: &Texels, mask: u32, three_color: bool) -> (u32, u32) {
    let (high, low) = principal_endpoints(texels, mask);
    let mut endpoints = (quantize_color(high), quantize_color(low));
    let choices = if three_color { 3 } else { 4 };
    let (mut indices, mut best_error) =
        select_indices(texels, mask, &palette(endpoints.0, endpoints.1, three_color), choices);
    let mut best = (endpoints, indices);
    for _ in 0..2 {
        let Some(refined) = refit(texels, mask, indices, three_color) else {
            break;
        };
        if refined == endpoints {
            break;
        }
        endpoints = refined;
        let (next_indices, error) =
            select_indices(texels, mask, &palette(endpoints.0, endpoints.1, three_color), choices);
        indices = next_indices;
        if error < best_error {
            best_error = error;
            best = (endpoints, indices);
        }
    }
    let ((first, second), mut indices) = best;
    let (mut color0, mut color1) = (pack565(first), pack565(second));
    if three_color {
        if color0 > color1 {
            std::mem::swap(&mut color0, &mut color1);
            indices ^= !((indices & 0xaaaa_aaaa) >> 1) & 0x5555_5555;
        }
    } else if color0 < color1 {
        std::mem::swap(&mut color0, &mut color1);
        indices ^= 0x5555_5555;
    } else if color0 == color1 {
        indices = 0;
    }
    (color0 | (color1 << 16), indices & mask_bits(mask))
}

fn mask_bits(mask: u32) -> u32 {
    (0..16)
        .filter(|&index| mask & (1 << index) != 0)
        .fold(0u32, |bits, index| bits | (3 << (2 * index)))
}

fn principal_endpoints(texels: &Texels, mask: u32) -> ([i32; 3], [i32; 3]) {
    let count = mask.count_ones() as i32;
    let mut sum = [0i32; 3];
    for index in (0..16).filter(|&index| mask & (1 << index) != 0) {
        for channel in 0..3 {
            sum[channel] += texels[index][channel];
        }
    }
    let mut covariance = [0i32; 6];
    for index in (0..16).filter(|&index| mask & (1 << index) != 0) {
        let [r, g, b] = [0, 1, 2].map(|channel| count * texels[index][channel] - sum[channel]);
        covariance[0] += r * r;
        covariance[1] += r * g;
        covariance[2] += r * b;
        covariance[3] += g * g;
        covariance[4] += g * b;
        covariance[5] += b * b;
    }
    let c = covariance.map(|value| value as f32);
    let mut axis = if c[0] >= c[3] && c[0] >= c[5] {
        [c[0], c[1], c[2]]
    } else if c[3] >= c[5] {
        [c[1], c[3], c[4]]
    } else {
        [c[2], c[4], c[5]]
    };
    for _ in 0..8 {
        let next = [
            c[0] * axis[0] + c[1] * axis[1] + c[2] * axis[2],
            c[1] * axis[0] + c[3] * axis[1] + c[4] * axis[2],
            c[2] * axis[0] + c[4] * axis[1] + c[5] * axis[2],
        ];
        let scale = next[0].abs().max(next[1].abs()).max(next[2].abs());
        if scale == 0.0 {
            break;
        }
        axis = next.map(|value| value / scale);
    }
    let direction = axis.map(|value| (value * 256.0 + 0.5).floor() as i32);
    let mut lowest = (i32::MAX, 0usize);
    let mut highest = (i32::MIN, 0usize);
    for index in (0..16).filter(|&index| mask & (1 << index) != 0) {
        let projection = (0..3).map(|channel| direction[channel] * texels[index][channel]).sum::<i32>();
        if projection < lowest.0 {
            lowest = (projection, index);
        }
        if projection > highest.0 {
            highest = (projection, index);
        }
    }
    let rgb = |index: usize| [texels[index][0], texels[index][1], texels[index][2]];
    (rgb(highest.1), rgb(lowest.1))
}

fn quantize_color(color: [i32; 3]) -> [i32; 3] {
    [
        (color[0] * 31 + 127) / 255,
        (color[1] * 63 + 127) / 255,
        (color[2] * 31 + 127) / 255,
    ]
}

fn expand_color(color: [i32; 3]) -> [i32; 3] {
    [
        (color[0] << 3) | (color[0] >> 2),
        (color[1] << 2) | (color[1] >> 4),
        (color[2] << 3) | (color[2] >> 2),
    ]
}

fn pack565(color: [i32; 3]) -> u32 {
    ((color[0] << 11) | (color[1] << 5) | color[2]) as u32
}

fn palette(first: [i32; 3], second: [i32; 3], three_color: bool) -> [[i32; 3]; 4] {
    let first = expand_color(first);
    let second = expand_color(second);
    let mut colors = [first, second, [0; 3], [0; 3]];
    for channel in 0..3 {
        if three_color {
            colors[2][channel] = (first[channel] + second[channel] + 1) / 2;
        } else {
            colors[2][channel] = (2 * first[channel] + second[channel] + 1) / 3;
            colors[3][channel] = (first[channel] + 2 * second[channel] + 1) / 3;
        }
    }
    colors
}

fn select_indices(texels: &Texels, mask: u32, colors: &[[i32; 3]; 4], choices: usize) -> (u32, i32) {
    let mut indices = 0u32;
    let mut total = 0;
    for index in (0..16).filter(|&index| mask & (1 << index) != 0) {
        let mut best = (i32::MAX, 0u32);
        for (choice, color) in colors.iter().enumerate().take(choices) {
            let error = (0..3)
                .map(|channel| {
                    let delta = texels[index][channel] - color[channel];
                    delta * delta
                })
                .sum::<i32>();
            if error < best.0 {
                best = (error, choice as u32);
            }
        }
        indices |= best.1 << (2 * index);
        total += best.0;
    }
    (indices, total)
}

fn refit(texels: &Texels, mask: u32, indices: u32, three_color: bool) -> Option<([i32; 3], [i32; 3])> {
    let (scale, weights) = if three_color {
        (2, [2, 0, 1, 0])
    } else {
        (3, [3, 0, 2, 1])
    };
    let (mut aa, mut ab, mut bb) = (0i32, 0i32, 0i32);
    let mut first = [0i32; 3];
    let mut second = [0i32; 3];
    for index in (0..16).filter(|&index| mask & (1 << index) != 0) {
        let weight = weights[((indices >> (2 * index)) & 3) as usize];
        let other = scale - weight;
        aa += weight * weight;
        ab += weight * other;
        bb += other * other;
        for channel in 0..3 {
            first[channel] += weight * texels[index][channel];
            second[channel] += other * texels[index][channel];
        }
    }
    let determinant = aa * bb - ab * ab;
    if determinant == 0 {
        return None;
    }
    let solve = |numerator: i32, maximum: i32| -> i32 {
        if numerator <= 0 {
            return 0;
        }
        ((2 * numerator * maximum + determinant * 255) / (2 * determinant * 255)).min(maximum)
    };
    let maxima = [31, 63, 31];
    Some((
        [0, 1, 2].map(|channel| solve(scale * (bb * first[channel] - ab * second[channel]), maxima[channel])),
        [0, 1, 2].map(|channel| solve(scale * (aa * second[channel] - ab * first[channel]), maxima[channel])),
    ))
}

fn encode_alpha(texels: &Texels, inside: u32) -> (u32, u32) {
    let alphas = (0..16).filter(|&index| inside & (1 << index) != 0).map(|index| texels[index][3]);
    let lowest = alphas.clone().min().unwrap_or(0);
    let highest = alphas.max().unwrap_or(0);
    if lowest == highest {
        return (highest as u32 | ((highest as u32) << 8), 0);
    }
    let mut wide = [highest, lowest, 0, 0, 0, 0, 0, 0];
    for step in 1..7i32 {
        wide[step as usize + 1] = ((7 - step) * highest + step * lowest + 3) / 7;
    }
    let (wide_indices, wide_error) = select_alpha(texels, inside, &wide);
    let mid = (0..16)
        .filter(|&index| inside & (1 << index) != 0)
        .map(|index| texels[index][3])
        .filter(|&alpha| alpha > 0 && alpha < 255);
    let narrow_low = mid.clone().min();
    let narrow_high = mid.max();
    if let (Some(narrow_low), Some(narrow_high)) = (narrow_low, narrow_high) {
        let mut narrow = [narrow_low, narrow_high, 0, 0, 0, 0, 0, 255];
        for step in 1..5i32 {
            narrow[step as usize + 1] = ((5 - step) * narrow_low + step * narrow_high + 2) / 5;
        }
        let (narrow_indices, narrow_error) = select_alpha(texels, inside, &narrow);
        if narrow_error < wide_error {
            return pack_alpha(narrow_low, narrow_high, narrow_indices);
        }
    }
    pack_alpha(highest, lowest, wide_indices)
}

fn select_alpha(texels: &Texels, inside: u32, levels: &[i32; 8]) -> ([u32; 16], i32) {
    let mut indices = [0u32; 16];
    let mut total = 0;
    for index in (0..16).filter(|&index| inside & (1 << index) != 0) {
        let mut best = (i32::MAX, 0u32);
        for (choice, level) in levels.iter().enumerate() {
            let delta = texels[index][3] - level;
            if delta * delta < best.0 {
                best = (delta * delta, choice as u32);
            }
        }
        indices[index] = best.1;
        total += best.0;
    }
    (indices, total)
}

fn pack_alpha(first: i32, second: i32, indices: [u32; 16]) -> (u32, u32) {
    let bits = indices
        .iter()
        .enumerate()
        .fold(0u64, |bits, (index, &value)| bits | (u64::from(value) << (3 * index)));
    let packed = first as u64 | ((second as u64) << 8) | (bits << 16);
    (packed as u32, (packed >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::{encode, BcTarget};

    fn decode_bc1_block(block: &[u8], bc3: bool) -> [[u8; 4]; 16] {
        let color0 = u16::from_le_bytes([block[0], block[1]]);
        let color1 = u16::from_le_bytes([block[2], block[3]]);
        let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let expand = |color: u16| {
            let r = i32::from((color >> 11) & 31);
            let g = i32::from((color >> 5) & 63);
            let b = i32::from(color & 31);
            [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
        };
        let (first, second) = (expand(color0), expand(color1));
        let four_color = bc3 || color0 > color1;
        let mut colors = [[0u8; 4]; 4];
        for channel in 0..3 {
            colors[0][channel] = first[channel] as u8;
            colors[1][channel] = second[channel] as u8;
            if four_color {
                colors[2][channel] = ((2 * first[channel] + second[channel]) / 3) as u8;
                colors[3][channel] = ((first[channel] + 2 * second[channel]) / 3) as u8;
            } else {
                colors[2][channel] = ((first[channel] + second[channel]) / 2) as u8;
            }
        }
        colors[0][3] = 255;
        colors[1][3] = 255;
        colors[2][3] = 255;
        colors[3][3] = if four_color { 255 } else { 0 };
        std::array::from_fn(|index| colors[((indices >> (2 * index)) & 3) as usize])
    }

    fn decode_alpha(block: &[u8]) -> [u8; 16] {
        let first = i32::from(block[0]);
        let second = i32::from(block[1]);
        let mut levels = [first, second, 0, 0, 0, 0, 0, 0];
        if first > second {
            for step in 1..7i32 {
                levels[step as usize + 1] = ((7 - step) * first + step * second) / 7;
            }
        } else {
            for step in 1..5i32 {
                levels[step as usize + 1] = ((5 - step) * first + step * second) / 5;
            }
            levels[6] = 0;
            levels[7] = 255;
        }
        let bits = u64::from_le_bytes([block[2], block[3], block[4], block[5], block[6], block[7], 0, 0]);
        std::array::from_fn(|index| levels[((bits >> (3 * index)) & 7) as usize] as u8)
    }

    pub(crate) fn decode(encoded: &[u8], width: u32, height: u32, target: BcTarget) -> Vec<u8> {
        let mut out = vec![0u8; width as usize * height as usize * 4];
        let blocks_x = width.div_ceil(4) as usize;
        for (block_index, block) in encoded.chunks_exact(target.block_bytes()).enumerate() {
            let (bx, by) = (block_index % blocks_x, block_index / blocks_x);
            let mut texels = match target {
                BcTarget::Bc1 => decode_bc1_block(block, false),
                BcTarget::Bc3 => decode_bc1_block(&block[8..], true),
            };
            if target == BcTarget::Bc3 {
                for (texel, alpha) in texels.iter_mut().zip(decode_alpha(block)) {
                    texel[3] = alpha;
                }
            }
            for (index, texel) in texels.iter().enumerate() {
                let (x, y) = (bx * 4 + (index & 3), by * 4 + (index >> 2));
                if x < width as usize && y < height as usize {
                    let offset = (y * width as usize + x) * 4;
                    out[offset..offset + 4].copy_from_slice(texel);
                }
            }
        }
        out
    }

    fn psnr(original: &[u8], decoded: &[u8], channels: std::ops::Range<usize>) -> f64 {
        let mut error = 0f64;
        let mut samples = 0f64;
        for (a, b) in original.chunks_exact(4).zip(decoded.chunks_exact(4)) {
            for channel in channels.clone() {
                let delta = f64::from(a[channel]) - f64::from(b[channel]);
                error += delta * delta;
                samples += 1.0;
            }
        }
        if error == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0 * 255.0 / (error / samples)).log10()
    }

    fn gradient(width: u32, height: u32) -> Vec<u8> {
        let mut image = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let r = (x * 4).min(255) as u8;
                let g = (y * 4).min(255) as u8;
                let b = ((x + y) * 2).min(255) as u8;
                let a = (x * 3 + y * 2).min(255) as u8;
                image.extend_from_slice(&[r, g, b, a]);
            }
        }
        image
    }

    #[test]
    fn two_color_blocks_round_trip_exactly() {
        let colors = [[0u8, 0, 0, 255], [255, 255, 255, 255]];
        let image: Vec<u8> = (0..16).flat_map(|index| colors[(index * 7 / 5) % 2]).collect();
        for target in [BcTarget::Bc1, BcTarget::Bc3] {
            let encoded = encode(&image, 4, 4, target);
            assert_eq!(encoded.len(), target.block_bytes());
            assert_eq!(super::tests::decode(&encoded, 4, 4, target), image, "{target:?}");
        }
    }

    #[test]
    fn solid_blocks_decode_to_the_nearest_565_color() {
        let image: Vec<u8> = (0..16).flat_map(|_| [200u8, 100, 50, 255]).collect();
        let decoded = decode(&encode(&image, 4, 4, BcTarget::Bc1), 4, 4, BcTarget::Bc1);
        for texel in decoded.chunks_exact(4) {
            assert!(texel.iter().zip([200u8, 100, 50, 255]).all(|(&a, b)| a.abs_diff(b) <= 4), "{texel:?}");
        }
    }

    #[test]
    fn bc1_keeps_punch_through_alpha() {
        let image: Vec<u8> = (0..64)
            .flat_map(|index: u32| {
                let opaque = (index % 8) < 5;
                [(index * 4) as u8, 180, 40, if opaque { 230 } else { 20 }]
            })
            .collect();
        let decoded = decode(&encode(&image, 8, 8, BcTarget::Bc1), 8, 8, BcTarget::Bc1);
        for (original, texel) in image.chunks_exact(4).zip(decoded.chunks_exact(4)) {
            assert_eq!(texel[3] == 255, original[3] >= 128, "{original:?} -> {texel:?}");
        }
    }

    #[test]
    fn gradients_keep_high_quality() {
        for (width, height) in [(64u32, 64u32), (61, 37), (3, 2), (130, 7)] {
            let image = gradient(width, height);
            let bc1 = decode(&encode(&image, width, height, BcTarget::Bc1), width, height, BcTarget::Bc1);
            let bc3 = decode(&encode(&image, width, height, BcTarget::Bc3), width, height, BcTarget::Bc3);
            let opaque: Vec<u8> = image
                .chunks_exact(4)
                .flat_map(|texel| [texel[0], texel[1], texel[2], 255])
                .collect();
            let bc1_opaque =
                decode(&encode(&opaque, width, height, BcTarget::Bc1), width, height, BcTarget::Bc1);
            assert!(psnr(&opaque, &bc1_opaque, 0..3) > 30.0, "{width}x{height} bc1 opaque");
            assert!(psnr(&image, &bc3, 0..3) > 30.0, "{width}x{height} bc3 color");
            assert!(psnr(&image, &bc3, 3..4) > 34.0, "{width}x{height} bc3 alpha");
            assert_eq!(bc1.len(), image.len());
        }
    }

    #[test]
    fn banded_encode_matches_a_single_block_walk() {
        let (width, height) = (300u32, 257u32);
        let image = gradient(width, height);
        for target in [BcTarget::Bc1, BcTarget::Bc3] {
            let banded = encode(&image, width, height, target);
            let mut single = vec![0u8; target.encoded_size(width, height)];
            let blocks_x = width.div_ceil(4) as usize;
            for (index, block) in single.chunks_exact_mut(target.block_bytes()).enumerate() {
                let (texels, inside) =
                    super::load_block(&image, width as usize, height as usize, index % blocks_x, index / blocks_x);
                super::encode_block(&texels, inside, target, block);
            }
            assert!(banded == single, "{target:?}");
        }
    }

    #[test]
    fn bc3_alpha_uses_the_six_level_mode_for_extremes() {
        let alphas = [0u8, 255, 120, 130, 125, 0, 255, 128, 0, 255, 122, 127, 255, 0, 124, 129];
        let image: Vec<u8> = alphas.iter().flat_map(|&alpha| [10u8, 20, 30, alpha]).collect();
        let encoded = encode(&image, 4, 4, BcTarget::Bc3);
        assert!(encoded[0] <= encoded[1]);
        let decoded = decode(&encoded, 4, 4, BcTarget::Bc3);
        for (texel, alpha) in decoded.chunks_exact(4).zip(alphas) {
            assert!(texel[3].abs_diff(alpha) <= 2, "{} vs {alpha}", texel[3]);
        }
    }
}

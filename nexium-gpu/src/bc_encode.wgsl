struct Params {
    src_offset: u32,
    dst_offset: u32,
    width: u32,
    height: u32,
    src_layer_texels: u32,
    dst_layer_words: u32,
    mode: u32,
    reserved: u32,
}

struct Endpoints {
    high: vec3<i32>,
    low: vec3<i32>,
}

struct Selection {
    indices: u32,
    error: i32,
}

struct Refit {
    valid: bool,
    first: vec3<i32>,
    second: vec3<i32>,
}

struct AlphaSelection {
    low_bits: u32,
    high_bits: u32,
    error: i32,
}

var<immediate> params: Params;

@group(0) @binding(1) var<storage, read_write> data: array<u32>;

var<private> texels: array<vec4<i32>, 16>;
var<private> colors: array<vec3<i32>, 4>;
var<private> alpha_levels: array<i32, 8>;

fn in_mask(mask: u32, index: u32) -> bool {
    return (mask & (1u << index)) != 0u;
}

fn mask_bits(mask: u32) -> u32 {
    var bits = 0u;
    for (var i = 0u; i < 16u; i++) {
        if (in_mask(mask, i)) {
            bits |= 3u << (2u * i);
        }
    }
    return bits;
}

fn quantize_color(color: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        (color.x * 31 + 127) / 255,
        (color.y * 63 + 127) / 255,
        (color.z * 31 + 127) / 255,
    );
}

fn expand_color(color: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        (color.x << 3u) | (color.x >> 2u),
        (color.y << 2u) | (color.y >> 4u),
        (color.z << 3u) | (color.z >> 2u),
    );
}

fn pack565(color: vec3<i32>) -> u32 {
    return u32((color.x << 11u) | (color.y << 5u) | color.z);
}

fn build_palette(first: vec3<i32>, second: vec3<i32>, three_color: bool) {
    let a = expand_color(first);
    let b = expand_color(second);
    colors[0] = a;
    colors[1] = b;
    if (three_color) {
        colors[2] = (a + b + vec3<i32>(1)) / 2;
        colors[3] = vec3<i32>(0);
    } else {
        colors[2] = (2 * a + b + vec3<i32>(1)) / 3;
        colors[3] = (a + 2 * b + vec3<i32>(1)) / 3;
    }
}

fn select_indices(mask: u32, choices: u32) -> Selection {
    var indices = 0u;
    var total = 0;
    for (var i = 0u; i < 16u; i++) {
        if (!in_mask(mask, i)) {
            continue;
        }
        var best_error = 0x7fffffff;
        var best = 0u;
        for (var k = 0u; k < choices; k++) {
            let delta = texels[i].xyz - colors[k];
            let error = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            if (error < best_error) {
                best_error = error;
                best = k;
            }
        }
        indices |= best << (2u * i);
        total += best_error;
    }
    return Selection(indices, total);
}

fn principal_endpoints(mask: u32) -> Endpoints {
    let count = i32(countOneBits(mask));
    var sum = vec3<i32>(0);
    for (var i = 0u; i < 16u; i++) {
        if (in_mask(mask, i)) {
            sum += texels[i].xyz;
        }
    }
    var rr = 0;
    var rg = 0;
    var rb = 0;
    var gg = 0;
    var gb = 0;
    var bb = 0;
    for (var i = 0u; i < 16u; i++) {
        if (in_mask(mask, i)) {
            let d = count * texels[i].xyz - sum;
            rr += d.x * d.x;
            rg += d.x * d.y;
            rb += d.x * d.z;
            gg += d.y * d.y;
            gb += d.y * d.z;
            bb += d.z * d.z;
        }
    }
    let c0 = f32(rr);
    let c1 = f32(rg);
    let c2 = f32(rb);
    let c3 = f32(gg);
    let c4 = f32(gb);
    let c5 = f32(bb);
    var axis: vec3<f32>;
    if (c0 >= c3 && c0 >= c5) {
        axis = vec3<f32>(c0, c1, c2);
    } else if (c3 >= c5) {
        axis = vec3<f32>(c1, c3, c4);
    } else {
        axis = vec3<f32>(c2, c4, c5);
    }
    for (var iteration = 0; iteration < 8; iteration++) {
        let next = vec3<f32>(
            c0 * axis.x + c1 * axis.y + c2 * axis.z,
            c1 * axis.x + c3 * axis.y + c4 * axis.z,
            c2 * axis.x + c4 * axis.y + c5 * axis.z,
        );
        let scale = max(abs(next.x), max(abs(next.y), abs(next.z)));
        if (scale == 0.0) {
            break;
        }
        axis = next / scale;
    }
    let direction = vec3<i32>(floor(axis * 256.0 + 0.5));
    var lowest = 0x7fffffff;
    var highest = -0x7fffffff - 1;
    var low_index = 0u;
    var high_index = 0u;
    for (var i = 0u; i < 16u; i++) {
        if (!in_mask(mask, i)) {
            continue;
        }
        let texel = texels[i].xyz;
        let projection = direction.x * texel.x + direction.y * texel.y + direction.z * texel.z;
        if (projection < lowest) {
            lowest = projection;
            low_index = i;
        }
        if (projection > highest) {
            highest = projection;
            high_index = i;
        }
    }
    return Endpoints(texels[high_index].xyz, texels[low_index].xyz);
}

fn solve(numerator: i32, determinant: i32, maximum: i32) -> i32 {
    if (numerator <= 0) {
        return 0;
    }
    return min((2 * numerator * maximum + determinant * 255) / (2 * determinant * 255), maximum);
}

fn refit(mask: u32, indices: u32, three_color: bool) -> Refit {
    var scale = 3;
    if (three_color) {
        scale = 2;
    }
    var aa = 0;
    var ab = 0;
    var bb = 0;
    var first = vec3<i32>(0);
    var second = vec3<i32>(0);
    for (var i = 0u; i < 16u; i++) {
        if (!in_mask(mask, i)) {
            continue;
        }
        let index = (indices >> (2u * i)) & 3u;
        var weight = 0;
        if (three_color) {
            if (index == 0u) {
                weight = 2;
            } else if (index == 2u) {
                weight = 1;
            }
        } else {
            if (index == 0u) {
                weight = 3;
            } else if (index == 2u) {
                weight = 2;
            } else if (index == 3u) {
                weight = 1;
            }
        }
        let other = scale - weight;
        aa += weight * weight;
        ab += weight * other;
        bb += other * other;
        first += weight * texels[i].xyz;
        second += other * texels[i].xyz;
    }
    let determinant = aa * bb - ab * ab;
    if (determinant == 0) {
        return Refit(false, vec3<i32>(0), vec3<i32>(0));
    }
    let a = scale * (bb * first - ab * second);
    let b = scale * (aa * second - ab * first);
    return Refit(
        true,
        vec3<i32>(solve(a.x, determinant, 31), solve(a.y, determinant, 63), solve(a.z, determinant, 31)),
        vec3<i32>(solve(b.x, determinant, 31), solve(b.y, determinant, 63), solve(b.z, determinant, 31)),
    );
}

fn encode_color(mask: u32, three_color: bool) -> vec2<u32> {
    let ends = principal_endpoints(mask);
    var first = quantize_color(ends.high);
    var second = quantize_color(ends.low);
    var choices = 4u;
    if (three_color) {
        choices = 3u;
    }
    build_palette(first, second, three_color);
    var selection = select_indices(mask, choices);
    var indices = selection.indices;
    var best_error = selection.error;
    var best_first = first;
    var best_second = second;
    var best_indices = indices;
    for (var round = 0; round < 2; round++) {
        let refined = refit(mask, indices, three_color);
        if (!refined.valid) {
            break;
        }
        if (all(refined.first == first) && all(refined.second == second)) {
            break;
        }
        first = refined.first;
        second = refined.second;
        build_palette(first, second, three_color);
        selection = select_indices(mask, choices);
        indices = selection.indices;
        if (selection.error < best_error) {
            best_error = selection.error;
            best_first = first;
            best_second = second;
            best_indices = indices;
        }
    }
    var color0 = pack565(best_first);
    var color1 = pack565(best_second);
    var result = best_indices;
    if (three_color) {
        if (color0 > color1) {
            let swapped = color0;
            color0 = color1;
            color1 = swapped;
            result ^= (~((result & 0xaaaaaaaau) >> 1u)) & 0x55555555u;
        }
    } else if (color0 < color1) {
        let swapped = color0;
        color0 = color1;
        color1 = swapped;
        result ^= 0x55555555u;
    } else if (color0 == color1) {
        result = 0u;
    }
    return vec2<u32>(color0 | (color1 << 16u), result & mask_bits(mask));
}

fn select_alpha(inside: u32) -> AlphaSelection {
    var low_bits = 0u;
    var high_bits = 0u;
    var total = 0;
    for (var i = 0u; i < 16u; i++) {
        if (!in_mask(inside, i)) {
            continue;
        }
        var best_error = 0x7fffffff;
        var best = 0u;
        for (var k = 0u; k < 8u; k++) {
            let delta = texels[i].w - alpha_levels[k];
            if (delta * delta < best_error) {
                best_error = delta * delta;
                best = k;
            }
        }
        let bit = 3u * i;
        if (bit < 32u) {
            low_bits |= best << bit;
            if (bit > 29u) {
                high_bits |= best >> (32u - bit);
            }
        } else {
            high_bits |= best << (bit - 32u);
        }
        total += best_error;
    }
    return AlphaSelection(low_bits, high_bits, total);
}

fn pack_alpha(first: i32, second: i32, selection: AlphaSelection) -> vec2<u32> {
    return vec2<u32>(
        u32(first) | (u32(second) << 8u) | (selection.low_bits << 16u),
        (selection.low_bits >> 16u) | (selection.high_bits << 16u),
    );
}

fn encode_alpha(inside: u32) -> vec2<u32> {
    var lowest = 255;
    var highest = 0;
    var narrow_low = 256;
    var narrow_high = -1;
    for (var i = 0u; i < 16u; i++) {
        if (!in_mask(inside, i)) {
            continue;
        }
        let alpha = texels[i].w;
        lowest = min(lowest, alpha);
        highest = max(highest, alpha);
        if (alpha > 0 && alpha < 255) {
            narrow_low = min(narrow_low, alpha);
            narrow_high = max(narrow_high, alpha);
        }
    }
    if (lowest == highest) {
        return vec2<u32>(u32(highest) | (u32(highest) << 8u), 0u);
    }
    alpha_levels[0] = highest;
    alpha_levels[1] = lowest;
    for (var step = 1; step < 7; step++) {
        alpha_levels[step + 1] = ((7 - step) * highest + step * lowest + 3) / 7;
    }
    let wide = select_alpha(inside);
    if (narrow_high >= 0) {
        alpha_levels[0] = narrow_low;
        alpha_levels[1] = narrow_high;
        for (var step = 1; step < 5; step++) {
            alpha_levels[step + 1] = ((5 - step) * narrow_low + step * narrow_high + 2) / 5;
        }
        alpha_levels[6] = 0;
        alpha_levels[7] = 255;
        let narrow = select_alpha(inside);
        if (narrow.error < wide.error) {
            return pack_alpha(narrow_low, narrow_high, narrow);
        }
    }
    return pack_alpha(highest, lowest, wide);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let blocks_x = (params.width + 3u) / 4u;
    let blocks_y = (params.height + 3u) / 4u;
    if (id.x >= blocks_x || id.y >= blocks_y) {
        return;
    }
    let base = params.src_offset + id.z * params.src_layer_texels;
    var inside = 0u;
    for (var i = 0u; i < 16u; i++) {
        let x = id.x * 4u + (i & 3u);
        let y = id.y * 4u + (i >> 2u);
        let texel = data[base + min(y, params.height - 1u) * params.width + min(x, params.width - 1u)];
        texels[i] = vec4<i32>(
            i32(texel & 0xffu),
            i32((texel >> 8u) & 0xffu),
            i32((texel >> 16u) & 0xffu),
            i32(texel >> 24u),
        );
        if (x < params.width && y < params.height) {
            inside |= 1u << i;
        }
    }
    if (params.mode == 1u) {
        var transparent = 0u;
        for (var i = 0u; i < 16u; i++) {
            if (in_mask(inside, i) && texels[i].w < 128) {
                transparent |= 1u << i;
            }
        }
        var block = vec2<u32>(0u, 0u);
        if (transparent == 0u) {
            block = encode_color(inside, false);
        } else {
            let opaque = inside & ~transparent;
            if (opaque != 0u) {
                block = encode_color(opaque, true);
            }
            for (var i = 0u; i < 16u; i++) {
                if (in_mask(transparent, i)) {
                    block.y |= 3u << (2u * i);
                }
            }
        }
        let out = params.dst_offset + id.z * params.dst_layer_words + (id.y * blocks_x + id.x) * 2u;
        data[out] = block.x;
        data[out + 1u] = block.y;
    } else {
        let alpha = encode_alpha(inside);
        let color = encode_color(inside, false);
        let out = params.dst_offset + id.z * params.dst_layer_words + (id.y * blocks_x + id.x) * 4u;
        data[out] = alpha.x;
        data[out + 1u] = alpha.y;
        data[out + 2u] = color.x;
        data[out + 3u] = color.y;
    }
}

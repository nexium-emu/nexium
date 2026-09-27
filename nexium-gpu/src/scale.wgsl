fn scale_texel(position: vec2<i32>, source_size: vec2<f32>) -> vec3<f32> {
    let limit = vec2<i32>(max(source_size, vec2<f32>(1.0))) - vec2<i32>(1);
    return textureLoad(frame, clamp(position, vec2<i32>(0), limit), 0).rgb;
}

fn scale_linear(uv: vec2<f32>, source_size: vec2<f32>) -> vec3<f32> {
    let border = vec2<f32>(0.5) / max(source_size, vec2<f32>(1.0));
    return textureSampleLevel(frame, filtering, clamp(uv, border, vec2<f32>(1.0) - border), 0.0).rgb;
}

fn scale_cubic_weight(distance: f32) -> f32 {
    let x = abs(distance);
    if x < 1.0 {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if x < 2.0 {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

fn scale_bicubic(uv: vec2<f32>, source_size: vec2<f32>) -> vec3<f32> {
    let position = uv * source_size - vec2<f32>(0.5);
    let origin = vec2<i32>(floor(position));
    let phase = fract(position);
    var total = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var y = -1; y <= 2; y += 1) {
        let vertical = scale_cubic_weight(f32(y) - phase.y);
        for (var x = -1; x <= 2; x += 1) {
            let weight = vertical * scale_cubic_weight(f32(x) - phase.x);
            total += scale_texel(origin + vec2<i32>(x, y), source_size) * weight;
            weight_sum += weight;
        }
    }
    return clamp(total / max(weight_sum, 0.000001), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn scale_color_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let difference = a - b;
    let luminance = dot(difference, vec3<f32>(0.2627, 0.6780, 0.0593));
    let chroma = (difference.br - vec2<f32>(luminance)) / vec2<f32>(1.8814, 1.4746);
    return length(vec3<f32>(luminance, chroma));
}

fn scale_force(uv: vec2<f32>, source_size: vec2<f32>) -> vec3<f32> {
    let center = scale_linear(uv, source_size);
    var direction = vec2<f32>(0.0);
    var contrast = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            if x != 0 || y != 0 {
                let offset = vec2<f32>(f32(x), f32(y));
                let neighbor = scale_linear(uv + offset / source_size, source_size);
                let distance = scale_color_distance(center, neighbor);
                direction += offset * distance;
                contrast += distance;
            }
        }
    }
    if contrast <= 0.000001 {
        return center;
    }
    let reach = length(direction) / contrast;
    let displacement = clamp(direction, vec2<f32>(-reach), vec2<f32>(reach));
    return scale_linear(uv - displacement / source_size, source_size);
}

fn scale_luma(color: vec3<f32>) -> f32 {
    return color.g + 0.5 * (color.r + color.b);
}

fn scale_edge_cross(up: f32, left: f32, center: f32, right: f32, down: f32) -> vec3<f32> {
    let gradient = vec2<f32>(right - left, down - up);
    let span = vec2<f32>(max(abs(right - center), abs(center - left)), max(abs(down - center), abs(center - up)));
    let coherence = clamp(abs(gradient) / max(span, vec2<f32>(0.000001)), vec2<f32>(0.0), vec2<f32>(1.0));
    return vec3<f32>(gradient, dot(coherence, coherence));
}

fn scale_easu_weight(offset: vec2<f32>, direction: vec2<f32>, anisotropy: vec2<f32>, lobe: f32) -> f32 {
    let rotated = vec2<f32>(dot(offset, direction), dot(offset, vec2<f32>(-direction.y, direction.x))) * anisotropy;
    let radius_squared = min(dot(rotated, rotated), 1.0 / lobe);
    let window = lobe * radius_squared - 1.0;
    let wave = 0.4 * radius_squared - 1.0;
    return (1.5625 * wave * wave - 0.5625) * window * window;
}

fn scale_easu_tap(color: vec3<f32>, offset: vec2<f32>, direction: vec2<f32>, anisotropy: vec2<f32>, lobe: f32) -> vec4<f32> {
    let weight = scale_easu_weight(offset, direction, anisotropy, lobe);
    return vec4<f32>(color * weight, weight);
}

fn scale_easu(uv: vec2<f32>, source_size: vec2<f32>) -> vec3<f32> {
    let position = uv * source_size - vec2<f32>(0.5);
    let origin = vec2<i32>(floor(position));
    let phase = fract(position);
    let b = clamp(scale_texel(origin + vec2<i32>(0, -1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let c = clamp(scale_texel(origin + vec2<i32>(1, -1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let e = clamp(scale_texel(origin + vec2<i32>(-1, 0), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let f = clamp(scale_texel(origin, source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let g = clamp(scale_texel(origin + vec2<i32>(1, 0), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let h = clamp(scale_texel(origin + vec2<i32>(2, 0), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let i = clamp(scale_texel(origin + vec2<i32>(-1, 1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let j = clamp(scale_texel(origin + vec2<i32>(0, 1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let k = clamp(scale_texel(origin + vec2<i32>(1, 1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let l = clamp(scale_texel(origin + vec2<i32>(2, 1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let n = clamp(scale_texel(origin + vec2<i32>(0, 2), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let o = clamp(scale_texel(origin + vec2<i32>(1, 2), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let upper_left = scale_edge_cross(scale_luma(b), scale_luma(e), scale_luma(f), scale_luma(g), scale_luma(j));
    let upper_right = scale_edge_cross(scale_luma(c), scale_luma(f), scale_luma(g), scale_luma(h), scale_luma(k));
    let lower_left = scale_edge_cross(scale_luma(f), scale_luma(i), scale_luma(j), scale_luma(k), scale_luma(n));
    let lower_right = scale_edge_cross(scale_luma(g), scale_luma(j), scale_luma(k), scale_luma(l), scale_luma(o));
    let edge = mix(mix(upper_left, upper_right, phase.x), mix(lower_left, lower_right, phase.x), phase.y);
    var direction = vec2<f32>(1.0, 0.0);
    let magnitude_squared = dot(edge.xy, edge.xy);
    if magnitude_squared >= 0.000030517578125 {
        direction = edge.xy * inverseSqrt(magnitude_squared);
    }
    let strength = 0.25 * edge.z * edge.z;
    let stretch = dot(direction, direction) / max(abs(direction.x), abs(direction.y));
    let anisotropy = vec2<f32>(mix(1.0, stretch, strength), 1.0 - 0.5 * strength);
    let lobe = mix(0.5, 0.21, strength);
    var total = vec4<f32>(0.0);
    total += scale_easu_tap(b, vec2<f32>(0.0, -1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(c, vec2<f32>(1.0, -1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(e, vec2<f32>(-1.0, 0.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(f, -phase, direction, anisotropy, lobe);
    total += scale_easu_tap(g, vec2<f32>(1.0, 0.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(h, vec2<f32>(2.0, 0.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(i, vec2<f32>(-1.0, 1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(j, vec2<f32>(0.0, 1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(k, vec2<f32>(1.0, 1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(l, vec2<f32>(2.0, 1.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(n, vec2<f32>(0.0, 2.0) - phase, direction, anisotropy, lobe);
    total += scale_easu_tap(o, vec2<f32>(1.0, 2.0) - phase, direction, anisotropy, lobe);
    let low = min(min(f, g), min(j, k));
    let high = max(max(f, g), max(j, k));
    return clamp(total.rgb / max(total.a, 0.000001), low, high);
}

fn scale_rcas(uv: vec2<f32>, source_size: vec2<f32>, sharpness: f32) -> vec3<f32> {
    let position = vec2<i32>(floor(uv * source_size));
    let center = clamp(scale_texel(position, source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let north = clamp(scale_texel(position + vec2<i32>(0, -1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let west = clamp(scale_texel(position + vec2<i32>(-1, 0), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let east = clamp(scale_texel(position + vec2<i32>(1, 0), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let south = clamp(scale_texel(position + vec2<i32>(0, 1), source_size), vec3<f32>(0.0), vec3<f32>(1.0));
    let ring_low = min(min(north, west), min(east, south));
    let ring_high = max(max(north, west), max(east, south));
    if all(ring_low == center) && all(ring_high == center) {
        return center;
    }
    let black_bound = -min(ring_low, center) / max(4.0 * ring_high, vec3<f32>(0.000001));
    let white_bound = (max(ring_high, center) - vec3<f32>(1.0)) / max(4.0 * (vec3<f32>(1.0) - ring_low), vec3<f32>(0.000001));
    let bounds = max(black_bound, white_bound);
    let lobe = clamp(max(max(bounds.r, bounds.g), bounds.b), -0.1875, 0.0) * clamp(sharpness, 0.0, 1.0);
    let sharpened = (center + lobe * (north + west + east + south)) / (1.0 + 4.0 * lobe);
    return clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_scaled(uv: vec2<f32>, source_size: vec2<f32>, output_size: vec2<f32>, mode: u32, sharpness: f32) -> vec3<f32> {
    switch mode {
        case 0u: { return scale_texel(vec2<i32>(floor(uv * source_size)), source_size); }
        case 2u: { return scale_bicubic(uv, source_size); }
        case 3u: { return scale_force(uv, source_size); }
        case 4u: { return scale_easu(uv, source_size); }
        case 5u: { return scale_rcas(uv, source_size, sharpness); }
        default: { return scale_linear(uv, source_size); }
    }
}

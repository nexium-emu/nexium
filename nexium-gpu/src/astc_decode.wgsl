struct Params {
    src_offset: u32,
    dst_offset: u32,
    width: u32,
    height: u32,
    block_width: u32,
    block_height: u32,
    src_layer_stride: u32,
    dst_layer_stride: u32,
}

var<immediate> params: Params;

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> texels: array<u32>;

var<private> block: array<u32, 4>;
var<private> reversed: array<u32, 4>;
var<private> seq: array<u32, 160>;
var<private> ev: array<i32, 32>;
var<private> ep: array<i32, 32>;
var<private> wv: array<i32, 160>;
var<private> cem: array<u32, 4>;

var<private> WEIGHT_A: array<u32, 16> = array<u32, 16>(0u, 0u, 0u, 3u, 0u, 5u, 3u, 0u, 0u, 0u, 5u, 3u, 0u, 5u, 3u, 0u);
var<private> WEIGHT_B: array<u32, 16> = array<u32, 16>(0u, 0u, 1u, 0u, 2u, 0u, 1u, 3u, 0u, 0u, 1u, 2u, 4u, 2u, 3u, 5u);
var<private> CEM_A: array<u32, 19> = array<u32, 19>(0u, 3u, 5u, 0u, 3u, 5u, 0u, 3u, 5u, 0u, 3u, 5u, 0u, 3u, 5u, 0u, 3u, 0u, 0u);
var<private> CEM_B: array<u32, 19> = array<u32, 19>(8u, 6u, 5u, 7u, 5u, 4u, 6u, 4u, 3u, 5u, 3u, 2u, 4u, 2u, 1u, 3u, 1u, 2u, 1u);
var<private> TRIT_M: array<u32, 5> = array<u32, 5>(0u, 2u, 4u, 5u, 7u);
var<private> QUINT_M: array<u32, 3> = array<u32, 3>(0u, 3u, 5u);
var<private> TRIT_C: array<u32, 7> = array<u32, 7>(0u, 204u, 93u, 44u, 22u, 11u, 5u);
var<private> QUINT_C: array<u32, 6> = array<u32, 6>(0u, 113u, 54u, 26u, 13u, 6u);

const MAGENTA: u32 = 0xffff00ffu;

fn stream_word(rev: bool, index: u32) -> u32 {
    if index >= 4u {
        return 0u;
    }
    if rev {
        return reversed[index];
    }
    return block[index];
}

fn read_bits(rev: bool, start: u32, count: u32) -> u32 {
    if count == 0u || start >= 128u {
        return 0u;
    }
    let word = start >> 5u;
    let shift = start & 31u;
    var value = stream_word(rev, word) >> shift;
    if shift != 0u {
        value = value | (stream_word(rev, word + 1u) << (32u - shift));
    }
    if count < 32u {
        value = value & ((1u << count) - 1u);
    }
    return value;
}

fn read_bits_limited(rev: bool, start: u32, count: u32, limit: u32) -> u32 {
    if start >= limit {
        return 0u;
    }
    return read_bits(rev, start, min(count, limit - start));
}

fn block_byte(index: u32) -> u32 {
    return (block[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu;
}

fn decode_trits(t: u32) -> u32 {
    var c: u32;
    var t3: u32;
    var t4: u32;
    if ((t >> 2u) & 7u) == 7u {
        c = (((t >> 5u) & 7u) << 2u) | (t & 3u);
        t4 = 2u;
        t3 = 2u;
    } else {
        c = t & 0x1fu;
        if ((t >> 5u) & 3u) == 3u {
            t4 = 2u;
            t3 = (t >> 7u) & 1u;
        } else {
            t4 = (t >> 7u) & 1u;
            t3 = (t >> 5u) & 3u;
        }
    }
    var t0: u32;
    var t1: u32;
    var t2: u32;
    if (c & 3u) == 3u {
        t2 = 2u;
        t1 = (c >> 4u) & 1u;
        t0 = (((c >> 3u) & 1u) << 1u) | ((c >> 2u) & ~(c >> 3u) & 1u);
    } else if ((c >> 2u) & 3u) == 3u {
        t2 = 2u;
        t1 = 2u;
        t0 = c & 3u;
    } else {
        t2 = (c >> 4u) & 1u;
        t1 = (c >> 2u) & 3u;
        t0 = (((c >> 1u) & 1u) << 1u) | (c & ~(c >> 1u) & 1u);
    }
    return t0 | (t1 << 2u) | (t2 << 4u) | (t3 << 6u) | (t4 << 8u);
}

fn decode_quints(q: u32) -> u32 {
    var q0: u32;
    var q1: u32;
    var q2: u32;
    if ((q >> 1u) & 3u) == 3u && ((q >> 5u) & 3u) == 0u {
        let low = q & 1u;
        q2 = (low << 2u) | ((((q >> 4u) & ~low) & 1u) << 1u) | ((q >> 3u) & ~low & 1u);
        q1 = 4u;
        q0 = 4u;
    } else {
        var c: u32;
        if ((q >> 1u) & 3u) == 3u {
            q2 = 4u;
            c = (((q >> 3u) & 3u) << 3u) | ((~(q >> 5u) & 3u) << 1u) | (q & 1u);
        } else {
            q2 = (q >> 5u) & 3u;
            c = q & 0x1fu;
        }
        if (c & 7u) == 5u {
            q1 = 4u;
            q0 = (c >> 3u) & 3u;
        } else {
            q1 = (c >> 3u) & 3u;
            q0 = c & 7u;
        }
    }
    return q0 | (q1 << 3u) | (q2 << 6u);
}

fn decode_ise(rev: bool, offset: u32, a: u32, b: u32, count: u32) {
    let mask = (1u << b) - 1u;
    var n = 0u;
    var p = offset;
    if a == 3u {
        let block_count = (count + 4u) / 5u;
        let last_block_count = (count + 4u) % 5u + 1u;
        let block_size = 8u + 5u * b;
        let last_block_size = (block_size * last_block_count + 4u) / 5u;
        for (var i = 0u; i < block_count; i++) {
            let now_size = select(last_block_size, block_size, i + 1u < block_count);
            let limit = p + now_size;
            let t = read_bits_limited(rev, p + b, 2u, limit)
                | (read_bits_limited(rev, p + 2u * b + 2u, 2u, limit) << 2u)
                | (read_bits_limited(rev, p + 3u * b + 4u, 1u, limit) << 4u)
                | (read_bits_limited(rev, p + 4u * b + 5u, 2u, limit) << 5u)
                | (read_bits_limited(rev, p + 5u * b + 7u, 1u, limit) << 7u);
            let trits = decode_trits(t);
            for (var j = 0u; j < 5u; j++) {
                if n < count {
                    let bits = read_bits_limited(rev, p + TRIT_M[j] + b * j, b, limit) & mask;
                    seq[n] = bits | (((trits >> (2u * j)) & 3u) << 16u);
                    n++;
                }
            }
            p += block_size;
        }
    } else if a == 5u {
        let block_count = (count + 2u) / 3u;
        let last_block_count = (count + 2u) % 3u + 1u;
        let block_size = 7u + 3u * b;
        let last_block_size = (block_size * last_block_count + 2u) / 3u;
        for (var i = 0u; i < block_count; i++) {
            let now_size = select(last_block_size, block_size, i + 1u < block_count);
            let limit = p + now_size;
            let q = read_bits_limited(rev, p + b, 3u, limit)
                | (read_bits_limited(rev, p + 2u * b + 3u, 2u, limit) << 3u)
                | (read_bits_limited(rev, p + 3u * b + 5u, 2u, limit) << 5u);
            let quints = decode_quints(q);
            for (var j = 0u; j < 3u; j++) {
                if n < count {
                    let bits = read_bits_limited(rev, p + QUINT_M[j] + b * j, b, limit) & mask;
                    seq[n] = bits | (((quints >> (3u * j)) & 7u) << 16u);
                    n++;
                }
            }
            p += block_size;
        }
    } else {
        for (var i = 0u; i < count; i++) {
            seq[i] = read_bits(rev, p, b);
            p += b;
        }
    }
}

fn seq_bits(index: u32) -> u32 {
    return seq[index] & 0xffffu;
}

fn seq_nonbits(index: u32) -> u32 {
    return seq[index] >> 16u;
}

fn set_endpoint(base: u32, r1: i32, g1: i32, b1: i32, a1: i32, r2: i32, g2: i32, b2: i32, a2: i32) {
    ep[base] = r1;
    ep[base + 1u] = g1;
    ep[base + 2u] = b1;
    ep[base + 3u] = a1;
    ep[base + 4u] = r2;
    ep[base + 5u] = g2;
    ep[base + 6u] = b2;
    ep[base + 7u] = a2;
}

fn set_endpoint_clamp(base: u32, r1: i32, g1: i32, b1: i32, a1: i32, r2: i32, g2: i32, b2: i32, a2: i32) {
    set_endpoint(
        base,
        clamp(r1, 0, 255), clamp(g1, 0, 255), clamp(b1, 0, 255), clamp(a1, 0, 255),
        clamp(r2, 0, 255), clamp(g2, 0, 255), clamp(b2, 0, 255), clamp(a2, 0, 255),
    );
}

fn set_endpoint_blue(base: u32, r1: i32, g1: i32, b1: i32, a1: i32, r2: i32, g2: i32, b2: i32, a2: i32) {
    set_endpoint(base, (r1 + b1) >> 1u, (g1 + b1) >> 1u, b1, a1, (r2 + b2) >> 1u, (g2 + b2) >> 1u, b2, a2);
}

fn set_endpoint_blue_clamp(base: u32, r1: i32, g1: i32, b1: i32, a1: i32, r2: i32, g2: i32, b2: i32, a2: i32) {
    set_endpoint(
        base,
        clamp((r1 + b1) >> 1u, 0, 255), clamp((g1 + b1) >> 1u, 0, 255), clamp(b1, 0, 255), clamp(a1, 0, 255),
        clamp((r2 + b2) >> 1u, 0, 255), clamp((g2 + b2) >> 1u, 0, 255), clamp(b2, 0, 255), clamp(a2, 0, 255),
    );
}

fn set_endpoint_hdr_clamp(base: u32, r1: i32, g1: i32, b1: i32, a1: i32, r2: i32, g2: i32, b2: i32, a2: i32) {
    set_endpoint(
        base,
        clamp(r1, 0, 0xfff), clamp(g1, 0, 0xfff), clamp(b1, 0, 0xfff), clamp(a1, 0, 0xfff),
        clamp(r2, 0, 0xfff), clamp(g2, 0, 0xfff), clamp(b2, 0, 0xfff), clamp(a2, 0, 0xfff),
    );
}

fn bit_transfer_signed(a: u32, b: u32) {
    ev[b] = (ev[b] >> 1u) | (ev[a] & 0x80);
    ev[a] = (ev[a] >> 1u) & 0x3f;
    if (ev[a] & 0x20) != 0 {
        ev[a] = ev[a] - 0x40;
    }
}

fn decode_endpoints_hdr7(base: u32, v: u32) {
    let modeval = ((ev[v + 2u] >> 4u) & 0x8) | ((ev[v + 1u] >> 5u) & 0x4) | (ev[v] >> 6u);
    var major_component: i32;
    var mode: i32;
    if (modeval & 0xc) != 0xc {
        major_component = modeval >> 2u;
        mode = modeval & 3;
    } else if modeval != 0xf {
        major_component = modeval & 3;
        mode = 4;
    } else {
        major_component = 0;
        mode = 5;
    }
    var c0 = ev[v] & 0x3f;
    var c1 = ev[v + 1u] & 0x1f;
    var c2 = ev[v + 2u] & 0x1f;
    var c3 = ev[v + 3u] & 0x1f;
    let v1 = ev[v + 1u];
    let v2 = ev[v + 2u];
    let v3 = ev[v + 3u];
    switch mode {
        case 0: {
            c3 |= v3 & 0x60;
            c0 |= (v3 >> 1u) & 0x40;
            c0 |= (v2 << 1u) & 0x80;
            c0 |= (v1 << 3u) & 0x300;
            c0 |= (v2 << 5u) & 0x400;
            c0 <<= 1u;
            c1 <<= 1u;
            c2 <<= 1u;
            c3 <<= 1u;
        }
        case 1: {
            c1 |= v1 & 0x20;
            c2 |= v2 & 0x20;
            c0 |= (v3 >> 1u) & 0x40;
            c0 |= (v2 << 1u) & 0x80;
            c0 |= (v1 << 2u) & 0x100;
            c0 |= (v3 << 4u) & 0x600;
            c0 <<= 1u;
            c1 <<= 1u;
            c2 <<= 1u;
            c3 <<= 1u;
        }
        case 2: {
            c3 |= v3 & 0xe0;
            c0 |= (v2 << 1u) & 0xc0;
            c0 |= (v1 << 3u) & 0x300;
            c0 <<= 2u;
            c1 <<= 2u;
            c2 <<= 2u;
            c3 <<= 2u;
        }
        case 3: {
            c1 |= v1 & 0x20;
            c2 |= v2 & 0x20;
            c3 |= v3 & 0x60;
            c0 |= (v3 >> 1u) & 0x40;
            c0 |= (v2 << 1u) & 0x80;
            c0 |= (v1 << 2u) & 0x100;
            c0 <<= 3u;
            c1 <<= 3u;
            c2 <<= 3u;
            c3 <<= 3u;
        }
        case 4: {
            c1 |= v1 & 0x60;
            c2 |= v2 & 0x60;
            c3 |= v3 & 0x20;
            c0 |= (v3 >> 1u) & 0x40;
            c0 |= (v3 << 1u) & 0x80;
            c0 <<= 4u;
            c1 <<= 4u;
            c2 <<= 4u;
            c3 <<= 4u;
        }
        default: {
            c1 |= v1 & 0x60;
            c2 |= v2 & 0x60;
            c3 |= v3 & 0x60;
            c0 |= (v3 >> 1u) & 0x40;
            c0 <<= 5u;
            c1 <<= 5u;
            c2 <<= 5u;
            c3 <<= 5u;
        }
    }
    if mode != 5 {
        c1 = c0 - c1;
        c2 = c0 - c2;
    }
    switch major_component {
        case 1: {
            set_endpoint_hdr_clamp(base, c1 - c3, c0 - c3, c2 - c3, 0x780, c1, c0, c2, 0x780);
        }
        case 2: {
            set_endpoint_hdr_clamp(base, c2 - c3, c1 - c3, c0 - c3, 0x780, c2, c1, c0, 0x780);
        }
        default: {
            set_endpoint_hdr_clamp(base, c0 - c3, c1 - c3, c2 - c3, 0x780, c0, c1, c2, 0x780);
        }
    }
}

fn sign_extend(value: i32, bits: u32) -> i32 {
    let sign = 1 << (bits - 1u);
    let masked = value & ((1 << bits) - 1);
    if (masked & sign) != 0 {
        return masked | (0xffff & ~((1 << bits) - 1));
    }
    return masked;
}

fn decode_endpoints_hdr11(base: u32, v: u32, alpha1: i32, alpha2: i32) {
    let v0 = ev[v];
    let v1 = ev[v + 1u];
    let v2 = ev[v + 2u];
    let v3 = ev[v + 3u];
    let v4 = ev[v + 4u];
    let v5 = ev[v + 5u];
    let major_component = (v4 >> 7u) | ((v5 >> 6u) & 2);
    if major_component == 3 {
        set_endpoint(base, v0 << 4u, v2 << 4u, (v4 << 5u) & 0xfe0, alpha1, v1 << 4u, v3 << 4u, (v5 << 5u) & 0xfe0, alpha2);
        return;
    }
    let mode = (v1 >> 7u) | ((v2 >> 6u) & 2) | ((v3 >> 5u) & 4);
    var va = v0 | ((v1 << 2u) & 0x100);
    var vb0 = v2 & 0x3f;
    var vb1 = v3 & 0x3f;
    var vc = v1 & 0x3f;
    var vd0: i32;
    var vd1: i32;
    switch mode {
        case 0, 2: {
            vd0 = sign_extend(v4, 7u);
            vd1 = sign_extend(v5, 7u);
        }
        case 1, 3, 5, 7: {
            vd0 = sign_extend(v4, 6u);
            vd1 = sign_extend(v5, 6u);
        }
        default: {
            vd0 = sign_extend(v4, 5u);
            vd1 = sign_extend(v5, 5u);
        }
    }
    switch mode {
        case 0: {
            vb0 |= v2 & 0x40;
            vb1 |= v3 & 0x40;
        }
        case 1: {
            vb0 |= v2 & 0x40;
            vb1 |= v3 & 0x40;
            vb0 |= (v4 << 1u) & 0x80;
            vb1 |= (v5 << 1u) & 0x80;
        }
        case 2: {
            va |= (v2 << 3u) & 0x200;
            vc |= v3 & 0x40;
        }
        case 3: {
            va |= (v4 << 3u) & 0x200;
            vc |= v5 & 0x40;
            vb0 |= v2 & 0x40;
            vb1 |= v3 & 0x40;
        }
        case 4: {
            va |= (v4 << 4u) & 0x200;
            va |= (v5 << 5u) & 0x400;
            vb0 |= v2 & 0x40;
            vb1 |= v3 & 0x40;
            vb0 |= (v4 << 1u) & 0x80;
            vb1 |= (v5 << 1u) & 0x80;
        }
        case 5: {
            va |= (v2 << 3u) & 0x200;
            va |= (v3 << 4u) & 0x400;
            vc |= v5 & 0x40;
            vc |= (v4 << 1u) & 0x80;
        }
        case 6: {
            va |= (v4 << 4u) & 0x200;
            va |= (v5 << 5u) & 0x400;
            va |= (v4 << 5u) & 0x800;
            vc |= v5 & 0x40;
            vb0 |= v2 & 0x40;
            vb1 |= v3 & 0x40;
        }
        default: {
            va |= (v2 << 3u) & 0x200;
            va |= (v3 << 4u) & 0x400;
            va |= (v4 << 5u) & 0x800;
            vc |= v5 & 0x40;
        }
    }
    let shamt = u32((mode >> 1u) ^ 3);
    va <<= shamt;
    vb0 <<= shamt;
    vb1 <<= shamt;
    vc <<= shamt;
    let mult = 1 << shamt;
    vd0 *= mult;
    vd1 *= mult;
    switch major_component {
        case 1: {
            set_endpoint_hdr_clamp(base, va - vb0 - vc - vd0, va - vc, va - vb1 - vc - vd1, alpha1, va - vb0, va, va - vb1, alpha2);
        }
        case 2: {
            set_endpoint_hdr_clamp(base, va - vb1 - vc - vd1, va - vb0 - vc - vd0, va - vc, alpha1, va - vb1, va - vb0, va, alpha2);
        }
        default: {
            set_endpoint_hdr_clamp(base, va - vc, va - vb0 - vc - vd0, va - vb1 - vc - vd1, alpha1, va, va - vb0, va - vb1, alpha2);
        }
    }
}

fn unquantize_endpoint(index: u32, a: u32, b: u32) -> i32 {
    let bits = seq_bits(index);
    if a == 3u || a == 5u {
        let flip = (bits & 1u) * 0x1ffu;
        let x = bits >> 1u;
        var add = 0u;
        var c: u32;
        if a == 3u {
            c = TRIT_C[b];
            switch b {
                case 2u: { add = 0x116u * x; }
                case 3u: { add = (x << 7u) | (x << 2u) | x; }
                case 4u: { add = (x << 6u) | x; }
                case 5u: { add = (x << 5u) | (x >> 2u); }
                case 6u: { add = (x << 4u) | (x >> 4u); }
                default: { add = 0u; }
            }
        } else {
            c = QUINT_C[b];
            switch b {
                case 2u: { add = 0x10cu * x; }
                case 3u: { add = (x << 7u) | (x << 1u) | (x >> 1u); }
                case 4u: { add = (x << 6u) | (x >> 1u); }
                case 5u: { add = (x << 5u) | (x >> 3u); }
                default: { add = 0u; }
            }
        }
        return i32((flip & 0x80u) | (((seq_nonbits(index) * c + add) ^ flip) >> 2u));
    }
    switch b {
        case 1u: { return i32(bits * 0xffu); }
        case 2u: { return i32(bits * 0x55u); }
        case 3u: { return i32((bits << 5u) | (bits << 2u) | (bits >> 1u)); }
        case 4u: { return i32((bits << 4u) | bits); }
        case 5u: { return i32((bits << 3u) | (bits >> 2u)); }
        case 6u: { return i32((bits << 2u) | (bits >> 4u)); }
        case 7u: { return i32((bits << 1u) | (bits >> 6u)); }
        case 8u: { return i32(bits); }
        default: { return 0; }
    }
}

fn decode_partition_endpoints(part: u32, mode: u32, v: u32) {
    let base = part * 8u;
    switch mode {
        case 0u: {
            set_endpoint(base, ev[v], ev[v], ev[v], 255, ev[v + 1u], ev[v + 1u], ev[v + 1u], 255);
        }
        case 1u: {
            let l0 = (ev[v] >> 2u) | (ev[v + 1u] & 0xc0);
            let l1 = clamp(l0 + (ev[v + 1u] & 0x3f), 0, 255);
            set_endpoint(base, l0, l0, l0, 255, l1, l1, l1, 255);
        }
        case 2u: {
            var y0: i32;
            var y1: i32;
            if ev[v] <= ev[v + 1u] {
                y0 = ev[v] << 4u;
                y1 = ev[v + 1u] << 4u;
            } else {
                y0 = (ev[v + 1u] << 4u) + 8;
                y1 = (ev[v] << 4u) - 8;
            }
            set_endpoint(base, y0, y0, y0, 0x780, y1, y1, y1, 0x780);
        }
        case 3u: {
            var y0: i32;
            var d: i32;
            if (ev[v] & 0x80) != 0 {
                y0 = ((ev[v + 1u] & 0xe0) << 4u) | ((ev[v] & 0x7f) << 2u);
                d = (ev[v + 1u] & 0x1f) << 2u;
            } else {
                y0 = ((ev[v + 1u] & 0xf0) << 4u) | ((ev[v] & 0x7f) << 1u);
                d = (ev[v + 1u] & 0x0f) << 1u;
            }
            let y1 = clamp(y0 + d, 0, 0xfff);
            set_endpoint(base, y0, y0, y0, 0x780, y1, y1, y1, 0x780);
        }
        case 4u: {
            set_endpoint(base, ev[v], ev[v], ev[v], ev[v + 2u], ev[v + 1u], ev[v + 1u], ev[v + 1u], ev[v + 3u]);
        }
        case 5u: {
            bit_transfer_signed(v + 1u, v);
            bit_transfer_signed(v + 3u, v + 2u);
            ev[v + 1u] = ev[v + 1u] + ev[v];
            set_endpoint_clamp(base, ev[v], ev[v], ev[v], ev[v + 2u], ev[v + 1u], ev[v + 1u], ev[v + 1u], ev[v + 2u] + ev[v + 3u]);
        }
        case 6u: {
            set_endpoint(
                base,
                (ev[v] * ev[v + 3u]) >> 8u, (ev[v + 1u] * ev[v + 3u]) >> 8u, (ev[v + 2u] * ev[v + 3u]) >> 8u, 255,
                ev[v], ev[v + 1u], ev[v + 2u], 255,
            );
        }
        case 7u: {
            decode_endpoints_hdr7(base, v);
        }
        case 8u: {
            if ev[v] + ev[v + 2u] + ev[v + 4u] <= ev[v + 1u] + ev[v + 3u] + ev[v + 5u] {
                set_endpoint(base, ev[v], ev[v + 2u], ev[v + 4u], 255, ev[v + 1u], ev[v + 3u], ev[v + 5u], 255);
            } else {
                set_endpoint_blue(base, ev[v + 1u], ev[v + 3u], ev[v + 5u], 255, ev[v], ev[v + 2u], ev[v + 4u], 255);
            }
        }
        case 9u: {
            bit_transfer_signed(v + 1u, v);
            bit_transfer_signed(v + 3u, v + 2u);
            bit_transfer_signed(v + 5u, v + 4u);
            if ev[v + 1u] + ev[v + 3u] + ev[v + 5u] >= 0 {
                set_endpoint_clamp(
                    base,
                    ev[v], ev[v + 2u], ev[v + 4u], 255,
                    ev[v] + ev[v + 1u], ev[v + 2u] + ev[v + 3u], ev[v + 4u] + ev[v + 5u], 255,
                );
            } else {
                set_endpoint_blue_clamp(
                    base,
                    ev[v] + ev[v + 1u], ev[v + 2u] + ev[v + 3u], ev[v + 4u] + ev[v + 5u], 255,
                    ev[v], ev[v + 2u], ev[v + 4u], 255,
                );
            }
        }
        case 10u: {
            set_endpoint(
                base,
                (ev[v] * ev[v + 3u]) >> 8u, (ev[v + 1u] * ev[v + 3u]) >> 8u, (ev[v + 2u] * ev[v + 3u]) >> 8u, ev[v + 4u],
                ev[v], ev[v + 1u], ev[v + 2u], ev[v + 5u],
            );
        }
        case 11u: {
            decode_endpoints_hdr11(base, v, 0x780, 0x780);
        }
        case 12u: {
            if ev[v] + ev[v + 2u] + ev[v + 4u] <= ev[v + 1u] + ev[v + 3u] + ev[v + 5u] {
                set_endpoint(base, ev[v], ev[v + 2u], ev[v + 4u], ev[v + 6u], ev[v + 1u], ev[v + 3u], ev[v + 5u], ev[v + 7u]);
            } else {
                set_endpoint_blue(base, ev[v + 1u], ev[v + 3u], ev[v + 5u], ev[v + 7u], ev[v], ev[v + 2u], ev[v + 4u], ev[v + 6u]);
            }
        }
        case 13u: {
            bit_transfer_signed(v + 1u, v);
            bit_transfer_signed(v + 3u, v + 2u);
            bit_transfer_signed(v + 5u, v + 4u);
            bit_transfer_signed(v + 7u, v + 6u);
            if ev[v + 1u] + ev[v + 3u] + ev[v + 5u] >= 0 {
                set_endpoint_clamp(
                    base,
                    ev[v], ev[v + 2u], ev[v + 4u], ev[v + 6u],
                    ev[v] + ev[v + 1u], ev[v + 2u] + ev[v + 3u], ev[v + 4u] + ev[v + 5u], ev[v + 6u] + ev[v + 7u],
                );
            } else {
                set_endpoint_blue_clamp(
                    base,
                    ev[v] + ev[v + 1u], ev[v + 2u] + ev[v + 3u], ev[v + 4u] + ev[v + 5u], ev[v + 6u] + ev[v + 7u],
                    ev[v], ev[v + 2u], ev[v + 4u], ev[v + 6u],
                );
            }
        }
        case 14u: {
            decode_endpoints_hdr11(base, v, ev[v + 6u], ev[v + 7u]);
        }
        default: {
            let mode15 = ((ev[v + 6u] >> 7u) & 1) | ((ev[v + 7u] >> 6u) & 2);
            ev[v + 6u] = ev[v + 6u] & 0x7f;
            ev[v + 7u] = ev[v + 7u] & 0x7f;
            if mode15 == 3 {
                decode_endpoints_hdr11(base, v, ev[v + 6u] << 5u, ev[v + 7u] << 5u);
            } else {
                let shift = u32(mode15);
                ev[v + 6u] = ev[v + 6u] | ((ev[v + 7u] << (shift + 1u)) & 0x780);
                ev[v + 7u] = ((ev[v + 7u] & (0x3f >> shift)) ^ (0x20 >> shift)) - (0x20 >> shift);
                ev[v + 6u] = ev[v + 6u] << (4u - shift);
                ev[v + 7u] = ev[v + 7u] << (4u - shift);
                decode_endpoints_hdr11(base, v, ev[v + 6u], clamp(ev[v + 6u] + ev[v + 7u], 0, 0xfff));
            }
        }
    }
}

fn unquantize_weight(index: u32, a: u32, b: u32) -> i32 {
    let bits = seq_bits(index);
    let nonbits = seq_nonbits(index);
    if a == 0u {
        var w: u32;
        switch b {
            case 1u: { w = select(0u, 63u, bits != 0u); }
            case 2u: { w = (bits << 4u) | (bits << 2u) | bits; }
            case 3u: { w = (bits << 3u) | bits; }
            case 4u: { w = (bits << 2u) | (bits >> 2u); }
            default: { w = (bits << 1u) | (bits >> 4u); }
        }
        if w > 32u {
            w += 1u;
        }
        return i32(w);
    }
    if b == 0u {
        return i32(nonbits * select(16u, 32u, a == 3u));
    }
    var w: u32;
    if a == 3u {
        switch b {
            case 1u: { w = nonbits * 50u; }
            case 2u: {
                w = nonbits * 23u;
                if (bits & 2u) != 0u {
                    w += 0x45u;
                }
            }
            default: { w = nonbits * 11u + (((bits << 4u) | (bits >> 1u)) & 0x63u); }
        }
    } else {
        switch b {
            case 1u: { w = nonbits * 28u; }
            default: {
                w = nonbits * 13u;
                if (bits & 2u) != 0u {
                    w += 0x42u;
                }
            }
        }
    }
    let flip = (bits & 1u) * 0x7fu;
    w = (flip & 0x20u) | ((w ^ flip) >> 2u);
    if w > 32u {
        w += 1u;
    }
    return i32(w);
}

fn select_color(v0: i32, v1: i32, weight: i32) -> u32 {
    return u32((((((v0 << 8u) | v0) * (64 - weight) + ((v1 << 8u) | v1) * weight + 32) >> 6u) * 255 + 32768) / 65536);
}

fn half_unorm8(bits: u32) -> u32 {
    if (bits & 0x7c00u) == 0x7c00u {
        if (bits & 0x3ffu) != 0u {
            return 0u;
        }
        return select(255u, 0u, (bits & 0x8000u) != 0u);
    }
    let value = unpack2x16float(bits).x;
    return u32(clamp(floor(value * 255.0), 0.0, 255.0));
}

fn select_color_hdr(v0: i32, v1: i32, weight: i32) -> u32 {
    let c = u32((((v0 << 4u) * (64 - weight) + (v1 << 4u) * weight + 32) >> 6u)) & 0xffffu;
    var m = c & 0x7ffu;
    if m < 512u {
        m = m * 3u;
    } else if m < 1536u {
        m = 4u * m - 512u;
    } else {
        m = 5u * m - 2048u;
    }
    let half = ((c >> 1u) & 0x7c00u) | ((m & 0xffffu) >> 3u);
    if (half & 0x7c00u) == 0x7c00u {
        return 255u;
    }
    let value = unpack2x16float(half).x;
    return u32(clamp(floor(value * 255.0), 0.0, 255.0));
}

fn hdr_rgb(mode: u32) -> bool {
    return mode == 2u || mode == 3u || mode == 7u || mode == 11u || mode == 14u || mode == 15u;
}

fn hdr_alpha(mode: u32) -> bool {
    return mode == 2u || mode == 3u || mode == 7u || mode == 11u || mode == 15u;
}

fn component(mode: u32, alpha: bool, v0: i32, v1: i32, weight: i32) -> u32 {
    let hdr = select(hdr_rgb(mode), hdr_alpha(mode), alpha);
    if hdr {
        return select_color_hdr(v0, v1, weight);
    }
    return select_color(v0, v1, weight);
}

fn pack(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 0xffu) | ((g & 0xffu) << 8u) | ((b & 0xffu) << 16u) | ((a & 0xffu) << 24u);
}

fn store(layer: u32, bx: u32, by: u32, s: u32, t: u32, color: u32) {
    let x = bx * params.block_width + s;
    let y = by * params.block_height + t;
    if x < params.width && y < params.height {
        texels[params.dst_offset + layer * params.dst_layer_stride + y * params.width + x] = color;
    }
}

fn fill(layer: u32, bx: u32, by: u32, color: u32) {
    for (var t = 0u; t < params.block_height; t++) {
        for (var s = 0u; s < params.block_width; s++) {
            store(layer, bx, by, s, t, color);
        }
    }
}

fn partition_of(seeds: array<u32, 8>, rnum: u32, part_num: u32, x: u32, y: u32) -> u32 {
    let a = (seeds[0] * x + seeds[1] * y + (rnum >> 14u)) & 0x3fu;
    let b = (seeds[2] * x + seeds[3] * y + (rnum >> 10u)) & 0x3fu;
    var c = 0u;
    if part_num >= 3u {
        c = (seeds[4] * x + seeds[5] * y + (rnum >> 6u)) & 0x3fu;
    }
    var d = 0u;
    if part_num >= 4u {
        d = (seeds[6] * x + seeds[7] * y + (rnum >> 2u)) & 0x3fu;
    }
    if a >= b && a >= c && a >= d {
        return 0u;
    }
    if b >= c && b >= d {
        return 1u;
    }
    if c >= d {
        return 2u;
    }
    return 3u;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let bw = params.block_width;
    let bh = params.block_height;
    let blocks_x = (params.width + bw - 1u) / bw;
    let blocks_y = (params.height + bh - 1u) / bh;
    if id.x >= blocks_x || id.y >= blocks_y {
        return;
    }
    let layer = id.z;
    let word = params.src_offset + layer * params.src_layer_stride + (id.y * blocks_x + id.x) * 4u;
    block[0] = source[word];
    block[1] = source[word + 1u];
    block[2] = source[word + 2u];
    block[3] = source[word + 3u];
    reversed[0] = reverseBits(block[3]);
    reversed[1] = reverseBits(block[2]);
    reversed[2] = reverseBits(block[1]);
    reversed[3] = reverseBits(block[0]);

    let b0 = block_byte(0u);
    let b1 = block_byte(1u);
    let b2 = block_byte(2u);
    let b3 = block_byte(3u);

    if b0 == 0xfcu && (b1 & 1u) == 1u {
        var color: u32;
        if (b1 & 2u) != 0u {
            color = pack(
                half_unorm8(block[2] & 0xffffu),
                half_unorm8(block[2] >> 16u),
                half_unorm8(block[3] & 0xffffu),
                half_unorm8(block[3] >> 16u),
            );
        } else {
            color = pack(block_byte(9u), block_byte(11u), block_byte(13u), block_byte(15u));
        }
        fill(layer, id.x, id.y, color);
        return;
    }
    if ((b0 & 0xc3u) == 0xc0u && (b1 & 1u) == 1u) || (b0 & 0xfu) == 0u {
        fill(layer, id.x, id.y, MAGENTA);
        return;
    }

    let w16 = b0 | (b1 << 8u);
    var dual_plane = (b1 & 4u) != 0u;
    var weight_range = ((b0 >> 4u) & 1u) | ((b1 << 2u) & 8u);
    var grid_w = 0u;
    var grid_h = 0u;
    if (b0 & 3u) != 0u {
        weight_range |= (b0 << 1u) & 6u;
        switch b0 & 0xcu {
            case 0u: {
                grid_w = ((w16 >> 7u) & 3u) + 4u;
                grid_h = ((b0 >> 5u) & 3u) + 2u;
            }
            case 4u: {
                grid_w = ((w16 >> 7u) & 3u) + 8u;
                grid_h = ((b0 >> 5u) & 3u) + 2u;
            }
            case 8u: {
                grid_w = ((b0 >> 5u) & 3u) + 2u;
                grid_h = ((w16 >> 7u) & 3u) + 8u;
            }
            default: {
                if (b1 & 1u) != 0u {
                    grid_w = ((b0 >> 7u) & 1u) + 2u;
                    grid_h = ((b0 >> 5u) & 3u) + 2u;
                } else {
                    grid_w = ((b0 >> 5u) & 3u) + 2u;
                    grid_h = ((b0 >> 7u) & 1u) + 6u;
                }
            }
        }
    } else {
        weight_range |= (b0 >> 1u) & 6u;
        switch w16 & 0x180u {
            case 0u: {
                grid_w = 12u;
                grid_h = ((b0 >> 5u) & 3u) + 2u;
            }
            case 0x80u: {
                grid_w = ((b0 >> 5u) & 3u) + 2u;
                grid_h = 12u;
            }
            case 0x100u: {
                grid_w = ((b0 >> 5u) & 3u) + 6u;
                grid_h = ((b1 >> 1u) & 3u) + 6u;
                dual_plane = false;
                weight_range &= 7u;
            }
            default: {
                if (b0 & 0x20u) != 0u {
                    grid_w = 10u;
                    grid_h = 6u;
                } else {
                    grid_w = 6u;
                    grid_h = 10u;
                }
            }
        }
    }

    let part_num = ((b1 >> 3u) & 3u) + 1u;
    let planes = select(1u, 2u, dual_plane);
    let weight_num = grid_w * grid_h * planes;
    let wa = WEIGHT_A[weight_range];
    let wb = WEIGHT_B[weight_range];
    if weight_num > 128u || (wa == 0u && (wb == 0u || wb > 5u)) {
        fill(layer, id.x, id.y, MAGENTA);
        return;
    }
    var weight_bits = weight_num * wb;
    if wa == 3u {
        weight_bits += (weight_num * 8u + 4u) / 5u;
    } else if wa == 5u {
        weight_bits += (weight_num * 7u + 2u) / 3u;
    }
    if weight_bits > 120u {
        fill(layer, id.x, id.y, MAGENTA);
        return;
    }

    var config_bits: u32;
    var cem_base = 0u;
    if part_num == 1u {
        cem[0] = ((b1 | (b2 << 8u)) >> 5u) & 0xfu;
        config_bits = 17u;
    } else {
        cem_base = ((b2 | (b3 << 8u)) >> 7u) & 3u;
        if cem_base == 0u {
            let shared_mode = (b3 >> 1u) & 0xfu;
            for (var i = 0u; i < part_num; i++) {
                cem[i] = shared_mode;
            }
            config_bits = 29u;
        } else {
            for (var i = 0u; i < part_num; i++) {
                cem[i] = (((b3 >> (i + 1u)) & 1u) + cem_base - 1u) << 2u;
            }
            switch part_num {
                case 2u: {
                    cem[0] |= (b3 >> 3u) & 3u;
                    cem[1] |= read_bits(false, 126u - weight_bits, 2u);
                }
                case 3u: {
                    cem[0] |= (b3 >> 4u) & 1u;
                    cem[0] |= read_bits(false, 122u - weight_bits, 2u) & 2u;
                    cem[1] |= read_bits(false, 124u - weight_bits, 2u);
                    cem[2] |= read_bits(false, 126u - weight_bits, 2u);
                }
                default: {
                    for (var i = 0u; i < 4u; i++) {
                        cem[i] |= read_bits(false, 120u + i * 2u - weight_bits, 2u);
                    }
                }
            }
            config_bits = 25u + part_num * 3u;
        }
    }
    var plane_selector = 0u;
    if dual_plane {
        config_bits += 2u;
        var position = 126u - weight_bits;
        if cem_base != 0u {
            position = 130u - weight_bits - part_num * 3u;
        }
        plane_selector = read_bits(false, position, 2u);
    }
    if config_bits + weight_bits > 128u {
        fill(layer, id.x, id.y, MAGENTA);
        return;
    }
    let remain_bits = 128u - config_bits - weight_bits;

    var endpoint_value_num = 0u;
    for (var i = 0u; i < part_num; i++) {
        endpoint_value_num += ((cem[i] >> 1u) & 6u) + 2u;
    }
    var cem_range = 0u;
    for (var i = 0u; i < 19u; i++) {
        var endpoint_bits = endpoint_value_num * CEM_B[i];
        if CEM_A[i] == 3u {
            endpoint_bits += (endpoint_value_num * 8u + 4u) / 5u;
        } else if CEM_A[i] == 5u {
            endpoint_bits += (endpoint_value_num * 7u + 2u) / 3u;
        }
        if endpoint_bits <= remain_bits {
            cem_range = i;
            break;
        }
    }

    let ca = CEM_A[cem_range];
    let cb = CEM_B[cem_range];
    decode_ise(false, select(29u, 17u, part_num == 1u), ca, cb, endpoint_value_num);
    for (var i = 0u; i < endpoint_value_num; i++) {
        ev[i] = unquantize_endpoint(i, ca, cb);
    }
    var v = 0u;
    for (var part = 0u; part < part_num; part++) {
        decode_partition_endpoints(part, cem[part], v);
        v += ((cem[part] >> 2u) + 1u) * 2u;
    }

    decode_ise(true, 0u, wa, wb, weight_num);
    for (var i = 0u; i < weight_num; i++) {
        wv[i] = unquantize_weight(i, wa, wb);
    }

    var seeds: array<u32, 8>;
    var rnum = 0u;
    let small_block = bw * bh < 31u;
    if part_num > 1u {
        let seed = ((block[0] >> 13u) & 0x3ffu) | ((part_num - 1u) << 10u);
        rnum = seed;
        rnum ^= rnum >> 15u;
        rnum = rnum - (rnum << 17u);
        rnum = rnum + (rnum << 7u);
        rnum = rnum + (rnum << 4u);
        rnum ^= rnum >> 5u;
        rnum = rnum + (rnum << 16u);
        rnum ^= rnum >> 7u;
        rnum ^= rnum >> 3u;
        rnum ^= rnum << 6u;
        rnum ^= rnum >> 17u;
        let even_shift = select(5u, 4u, (seed & 2u) != 0u);
        let odd_shift = select(5u, 6u, part_num == 3u);
        for (var i = 0u; i < 8u; i++) {
            let value = (rnum >> (i * 4u)) & 0xfu;
            var shift: u32;
            if (seed & 1u) != 0u {
                shift = select(odd_shift, even_shift, (i & 1u) == 0u);
            } else {
                shift = select(even_shift, odd_shift, (i & 1u) == 0u);
            }
            seeds[i] = (value * value) >> shift;
        }
    }

    let ds = (1024u + bw / 2u) / (bw - 1u);
    let dt = (1024u + bh / 2u) / (bh - 1u);
    var plane_of: array<u32, 4> = array<u32, 4>(0u, 0u, 0u, 0u);
    if dual_plane {
        plane_of[plane_selector] = 1u;
    }
    for (var t = 0u; t < bh; t++) {
        for (var s = 0u; s < bw; s++) {
            var part = 0u;
            if part_num > 1u {
                if small_block {
                    part = partition_of(seeds, rnum, part_num, s << 1u, t << 1u);
                } else {
                    part = partition_of(seeds, rnum, part_num, s, t);
                }
            }
            let gs = (ds * s * (grid_w - 1u) + 32u) >> 6u;
            let gt = (dt * t * (grid_h - 1u) + 32u) >> 6u;
            let fs = gs & 0xfu;
            let ft = gt & 0xfu;
            let vi = (gs >> 4u) + (gt >> 4u) * grid_w;
            let w11 = i32((fs * ft + 8u) >> 4u);
            let w10 = i32(ft) - w11;
            let w01 = i32(fs) - w11;
            let w00 = 16 - i32(fs) - i32(ft) + w11;
            var weights: array<i32, 2>;
            for (var p = 0u; p < planes; p++) {
                let p00 = wv[min(vi * planes + p, 159u)];
                let p01 = wv[min((vi + 1u) * planes + p, 159u)];
                let p10 = wv[min((vi + grid_w) * planes + p, 159u)];
                let p11 = wv[min((vi + grid_w + 1u) * planes + p, 159u)];
                weights[p] = (p00 * w00 + p01 * w01 + p10 * w10 + p11 * w11 + 8) >> 4u;
            }
            let mode = cem[part];
            let base = part * 8u;
            let r = component(mode, false, ep[base], ep[base + 4u], weights[plane_of[0]]);
            let g = component(mode, false, ep[base + 1u], ep[base + 5u], weights[plane_of[1]]);
            let b = component(mode, false, ep[base + 2u], ep[base + 6u], weights[plane_of[2]]);
            let a = component(mode, true, ep[base + 3u], ep[base + 7u], weights[plane_of[3]]);
            store(layer, id.x, id.y, s, t, pack(r, g, b, a));
        }
    }
}

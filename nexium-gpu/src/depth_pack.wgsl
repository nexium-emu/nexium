@group(0) @binding(0) var<storage, read_write> texels: array<u32>;

fn pack_d24(value: f32) -> u32 {
    if !(value > 0.0) {
        return 0u;
    }
    let bits = bitcast<u32>(clamp(value, 0.0, 1.0));
    if bits >= 0x3f800000u {
        return 0xffffffu;
    }
    let exponent = bits >> 23u;
    if exponent < 102u {
        return 0u;
    }
    let significand = (bits & 0x7fffffu) | 0x800000u;
    if exponent == 126u {
        return significand - select(1u, 0u, significand == 0x800000u);
    }
    let shift = 126u - exponent;
    return (significand + (1u << (shift - 1u)) - 1u) >> shift;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let index = (group.y * groups.x + group.x) * 64u + local;
    if index < arrayLength(&texels) {
        texels[index] = pack_d24(bitcast<f32>(texels[index]));
    }
}

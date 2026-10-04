struct Push {
    time: f32,
    width: f32,
    height: f32,
    frame: f32,
    buttons: f32,
    lx: f32,
    ly: f32,
    mode: f32,
};

var<immediate> push: Push;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var out: VsOut;
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn interactive(uv: vec2<f32>) -> vec3<f32> {
    let b = u32(push.buttons);
    var c = vec3<f32>(0.08, 0.08, 0.1);
    if ((b & 0x4000u) != 0u) { c = c + vec3<f32>(0.1, 0.3, 1.0); }
    if ((b & 0x2000u) != 0u) { c = c + vec3<f32>(1.0, 0.1, 0.1); }
    if ((b & 0x8000u) != 0u) { c = c + vec3<f32>(1.0, 0.3, 0.8); }
    if ((b & 0x1000u) != 0u) { c = c + vec3<f32>(0.1, 1.0, 0.3); }
    let aspect = vec2<f32>(push.width / push.height, 1.0);
    let dot_pos = vec2<f32>(0.5 + push.lx * 0.4, 0.5 + push.ly * 0.4);
    if (distance(uv * aspect, dot_pos * aspect) < 0.03) { c = vec3<f32>(1.0, 1.0, 1.0); }
    if (uv.y > 0.92) {
        let cell = min(u32(uv.x * 24.0), 23u);
        if (((b >> cell) & 1u) != 0u) { c = vec3<f32>(1.0, 0.9, 0.2); } else { c = vec3<f32>(0.2, 0.2, 0.2); }
        if (fract(uv.x * 24.0) < 0.05) { c = vec3<f32>(0.0, 0.0, 0.0); }
    }
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (push.mode > 0.5) {
        return vec4<f32>(interactive(in.uv), 1.0);
    }
    let t = push.time;
    let c = vec3<f32>(
        0.5 + 0.5 * sin(in.uv.x * 6.2831 + t),
        0.5 + 0.5 * sin(in.uv.y * 6.2831 + t * 1.3),
        0.5 + 0.5 * sin((in.uv.x + in.uv.y) * 3.1415 + t * 0.7)
    );
    let bar = step(0.48, in.uv.x) * step(in.uv.x, 0.52);
    return vec4<f32>(mix(c, vec3<f32>(1.0, 1.0, 1.0), bar), 1.0);
}

@group(0) @binding(0)
var<storage, read_write> values: array<u32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    values[id.x] = id.x * 2u + 7u;
}

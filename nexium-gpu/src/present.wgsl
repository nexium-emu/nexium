struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Mapping {
    origin: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
}

var<immediate> mapping: Mapping;
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var filtering: sampler;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = mapping.origin + mapping.dx * uv.x + mapping.dy * uv.y;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(frame, filtering, in.uv).rgb, 1.0);
}

struct Mapping {
    origin: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
    mode: u32,
    sharpness: f32,
    output_size: vec2<f32>,
}

var<immediate> mapping: Mapping;
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var filtering: sampler;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let destination = position.xy / mapping.output_size;
    let uv = mapping.origin + mapping.dx * destination.x + mapping.dy * destination.y;
    return vec4<f32>(sample_scaled(uv, vec2<f32>(textureDimensions(frame)), mapping.output_size, mapping.mode, mapping.sharpness), 1.0);
}

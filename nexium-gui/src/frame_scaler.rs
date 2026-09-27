use eframe::{egui, egui_wgpu::RenderState, wgpu};
use nexium_gpu::presentation::ScalingFilter;

#[cfg(test)]
#[path = "frame_scaler_tests.rs"]
mod tests;

const SHADER: &str = concat!(
    include_str!("../../nexium-gpu/src/scale.wgsl"),
    r#"
struct Params {
    output_size: vec2<f32>,
    mode: u32,
    sharpness: f32,
}
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var filtering: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}
@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy / params.output_size;
    return vec4<f32>(sample_scaled(uv, vec2<f32>(textureDimensions(frame)), params.output_size, params.mode, params.sharpness), 1.0);
}
"#
);

struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nexium_scaled_frame"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

pub struct FrameScaler {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: [wgpu::Buffer; 2],
    output: Option<Target>,
    intermediate: Option<Target>,
    bindings: Option<[wgpu::BindGroup; 2]>,
    size: [u32; 2],
    filter: ScalingFilter,
    sharpness: u8,
    pub id: Option<egui::TextureId>,
}

impl FrameScaler {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nexium_frame_scaler"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nexium_frame_scaler"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nexium_frame_scaler"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nexium_frame_scaler"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nexium_frame_scaler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nexium_frame_scaler_params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Self {
            pipeline,
            layout,
            sampler,
            uniforms,
            output: None,
            intermediate: None,
            bindings: None,
            size: [0; 2],
            filter: ScalingFilter::Nearest,
            sharpness: 87,
            id: None,
        }
    }

    pub fn render(
        &mut self,
        rs: &RenderState,
        source: &wgpu::Texture,
        size: [u32; 2],
        filter: ScalingFilter,
        sharpness: u8,
        dirty: bool,
    ) -> egui::TextureId {
        let limit = rs.device.limits().max_texture_dimension_2d;
        let size = size.map(|value| value.clamp(1, limit));
        let reconfigure = self.size != size || self.filter != filter;
        let sharpness = sharpness.min(100);
        let sharpness_changed = filter == ScalingFilter::Fsr && self.sharpness != sharpness;
        self.sharpness = sharpness;
        if reconfigure {
            self.size = size;
            self.filter = filter;
            self.output = Some(Target::new(&rs.device, size));
            self.intermediate =
                (filter == ScalingFilter::Fsr).then(|| Target::new(&rs.device, size));
            let output = self.output.as_ref().unwrap();
            let mut renderer = rs.renderer.write();
            if let Some(id) = self.id {
                renderer.update_egui_texture_from_wgpu_texture(
                    &rs.device,
                    &output.view,
                    wgpu::FilterMode::Nearest,
                    id,
                );
            } else {
                self.id = Some(renderer.register_native_texture(
                    &rs.device,
                    &output.view,
                    wgpu::FilterMode::Nearest,
                ));
            }
            let source_view = source.create_view(&Default::default());
            self.bindings = Some(std::array::from_fn(|index| {
                let view = if index == 1 {
                    self.intermediate
                        .as_ref()
                        .map_or(&source_view, |target| &target.view)
                } else {
                    &source_view
                };
                rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("nexium_frame_scaler"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.uniforms[index].as_entire_binding(),
                        },
                    ],
                })
            }));
        }
        if reconfigure || sharpness_changed {
            for (index, mode) in [filter as u32, 5].into_iter().enumerate() {
                let mut bytes = [0; 16];
                bytes[0..4].copy_from_slice(&(size[0] as f32).to_le_bytes());
                bytes[4..8].copy_from_slice(&(size[1] as f32).to_le_bytes());
                bytes[8..12].copy_from_slice(&mode.to_le_bytes());
                bytes[12..16].copy_from_slice(&(f32::from(sharpness) / 100.0).to_le_bytes());
                rs.queue.write_buffer(&self.uniforms[index], 0, &bytes);
            }
        }
        if dirty || reconfigure || sharpness_changed {
            let mut encoder = rs
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nexium_frame_scaler"),
                });
            let bindings = self.bindings.as_ref().unwrap();
            let output = self.output.as_ref().unwrap();
            let targets = [self.intermediate.as_ref().unwrap_or(output), output];
            let passes = if filter == ScalingFilter::Fsr { 2 } else { 1 };
            for index in 0..passes {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("nexium_frame_scaler"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &targets[index].view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bindings[index], &[]);
                pass.draw(0..3, 0..1);
            }
            rs.queue.submit([encoder.finish()]);
        }
        self.id.unwrap()
    }
}

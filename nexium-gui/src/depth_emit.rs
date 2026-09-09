use std::sync::Arc;

use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;
use nexium_gpu::PresentDepth;

pub struct DepthEmit {
    pub depth: Arc<PresentDepth>,
    pub seq: u64,
    pub rect: egui::Rect,
}

struct SourceTexture {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

struct TargetTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: [u32; 2],
}

struct Resources {
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    source: Option<SourceTexture>,
    target: Option<TargetTexture>,
    bind_group: Option<wgpu::BindGroup>,
    uploaded_seq: Option<u64>,
}

const SHADER: &str = r#"
struct Params {
    rect: vec4<f32>,
    src_size: vec2<u32>,
    mode: u32,
    pad: u32,
};

@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index & 1u) * 4 - 1);
    let y = f32(i32(index >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> FragOut {
    let u = (pos.x - params.rect.x) / params.rect.z;
    let v = (pos.y - params.rect.y) / params.rect.w;
    let sx = clamp(i32(u * f32(params.src_size.x)), 0, i32(params.src_size.x) - 1);
    let sy = clamp(i32(v * f32(params.src_size.y)), 0, i32(params.src_size.y) - 1);
    let raw = textureLoad(src, vec2<i32>(sx, sy), 0).x;
    var d: f32;
    if (params.mode == 1u) {
        d = f32(raw & 0xFFFFFFu) / 16777215.0;
    } else if (params.mode == 2u) {
        d = f32(raw & 0xFFFFu) / 65535.0;
    } else {
        d = bitcast<f32>(raw);
    }
    var out: FragOut;
    out.depth = clamp(d, 0.0, 1.0);
    return out;
}
"#;

impl Resources {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nexium_depth_emit"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nexium_depth_emit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nexium_depth_emit"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nexium_depth_emit"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nexium_depth_emit_params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bind_layout,
            uniforms,
            source: None,
            target: None,
            bind_group: None,
            uploaded_seq: None,
        }
    }
}

impl CallbackTrait for DepthEmit {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let [sw, sh] = screen.size_in_pixels;
        if sw == 0 || sh == 0 || self.depth.width == 0 || self.depth.height == 0 {
            return Vec::new();
        }
        if resources.get::<Resources>().is_none() {
            resources.insert(Resources::new(device));
        }
        let Some(res) = resources.get_mut::<Resources>() else {
            return Vec::new();
        };
        if res.target.as_ref().map_or(true, |t| t.size != [sw, sh]) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nexium_depth_emit_target"),
                size: wgpu::Extent3d {
                    width: sw,
                    height: sh,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            res.target = Some(TargetTexture {
                _texture: texture,
                view,
                size: [sw, sh],
            });
        }
        let dims_changed = res
            .source
            .as_ref()
            .map_or(true, |s| s.width != self.depth.width || s.height != self.depth.height);
        if dims_changed {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nexium_depth_emit_source"),
                size: wgpu::Extent3d {
                    width: self.depth.width,
                    height: self.depth.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            res.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nexium_depth_emit"),
                layout: &res.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: res.uniforms.as_entire_binding(),
                    },
                ],
            }));
            res.source = Some(SourceTexture {
                texture,
                width: self.depth.width,
                height: self.depth.height,
            });
            res.uploaded_seq = None;
        }
        let Some(source) = res.source.as_ref() else {
            return Vec::new();
        };
        if res.uploaded_seq != Some(self.seq) {
            let need = source.width as usize * source.height as usize * 4;
            if self.depth.texels.len() < need {
                return Vec::new();
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &source.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.depth.texels[..need],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(source.width * 4),
                    rows_per_image: Some(source.height),
                },
                wgpu::Extent3d {
                    width: source.width,
                    height: source.height,
                    depth_or_array_layers: 1,
                },
            );
            res.uploaded_seq = Some(self.seq);
        }
        let ppp = screen.pixels_per_point;
        let x0 = (self.rect.min.x * ppp).round().clamp(0.0, sw as f32);
        let y0 = (self.rect.min.y * ppp).round().clamp(0.0, sh as f32);
        let x1 = (self.rect.max.x * ppp).round().clamp(0.0, sw as f32);
        let y1 = (self.rect.max.y * ppp).round().clamp(0.0, sh as f32);
        let (w, h) = (x1 - x0, y1 - y0);
        if w < 1.0 || h < 1.0 {
            return Vec::new();
        }
        let mut params = [0u8; 32];
        params[0..4].copy_from_slice(&x0.to_le_bytes());
        params[4..8].copy_from_slice(&y0.to_le_bytes());
        params[8..12].copy_from_slice(&w.to_le_bytes());
        params[12..16].copy_from_slice(&h.to_le_bytes());
        params[16..20].copy_from_slice(&source.width.to_le_bytes());
        params[20..24].copy_from_slice(&source.height.to_le_bytes());
        params[24..28].copy_from_slice(&self.depth.encoding.to_le_bytes());
        queue.write_buffer(&res.uniforms, 0, &params);
        let (Some(target), Some(bind_group)) = (res.target.as_ref(), res.bind_group.as_ref())
        else {
            return Vec::new();
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("nexium_depth_emit"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&res.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.set_viewport(x0, y0, w, h, 0.0, 1.0);
        pass.set_scissor_rect(x0 as u32, y0 as u32, w as u32, h as u32);
        pass.draw(0..3, 0..1);
        drop(pass);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_pass: &mut wgpu::RenderPass<'static>,
        _resources: &CallbackResources,
    ) {
    }
}

use crate::rt_cache::RtKey;
use ash::vk;

#[derive(Clone, Copy, Debug)]
pub struct VertexAttr {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBinding {
    pub binding: u32,
    pub stride: u32,
    pub divisor: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBufferBinding {
    pub binding: u32,
    pub addr: u64,
    pub stride: u32,
    pub divisor: u32,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct VertexLayout {
    pub bindings: Vec<VertexBinding>,
    pub attrs: Vec<VertexAttr>,
}

impl VertexLayout {
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in &self.bindings {
            h ^= b.binding as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= b.stride as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= b.divisor as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for a in &self.attrs {
            h ^= a.location as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= a.binding as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= a.format.as_raw() as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= a.offset as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DrawState {
    pub topology: vk::PrimitiveTopology,
    pub vertex_count: u32,
    pub index_count: u32,
    pub indexed: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendAttachmentState {
    pub enabled: bool,
    pub src_factor: vk::BlendFactor,
    pub dst_factor: vk::BlendFactor,
    pub op: vk::BlendOp,
    pub src_alpha_factor: vk::BlendFactor,
    pub dst_alpha_factor: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendState {
    pub enabled: bool,
    pub src_factor: vk::BlendFactor,
    pub dst_factor: vk::BlendFactor,
    pub op: vk::BlendOp,
    pub src_alpha_factor: vk::BlendFactor,
    pub dst_alpha_factor: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    pub color_write_mask: vk::ColorComponentFlags,
    pub attachments: [BlendAttachmentState; 8],
}

#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
    pub compare_op: vk::CompareOp,
}

#[derive(Clone, Debug)]
pub struct Maxwell3dDrawCall {
    pub vs_spirv: std::sync::Arc<Vec<u32>>,
    pub fs_spirv: std::sync::Arc<Vec<u32>>,
    pub vs_gpu_va: u64,
    pub fs_gpu_va: u64,
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub fs_tex_ids: Vec<u32>,
    pub vs_tex_base: u32,
    pub vs_tex_count: u32,
    pub vertex_layout: VertexLayout,
    pub cbuf_addr: u64,
    pub cbuf_size: u32,
    pub cbuf_data: Option<Vec<u8>>,
    pub vertex_addr: u64,
    pub vertex_bindings: Vec<VertexBufferBinding>,
    pub vertex_count: u32,
    pub first_vertex: u32,
    pub instance_count: u32,
    pub first_instance: u32,
    pub index_addr: Option<u64>,
    pub index_count: Option<u32>,
    pub index_type: vk::IndexType,
    pub index_data: Option<Vec<u8>>,
    pub quad_expand: bool,
    pub rt_key: RtKey,
    pub color_rt_keys: Vec<RtKey>,
    pub color_rt_formats: Vec<vk::Format>,
    pub rt_format: vk::Format,
    pub vp_rect: Option<[f32; 4]>,
    pub scissor: Option<[i32; 4]>,
    pub state: DrawState,
    pub blend: BlendState,
    pub depth: DepthState,
    pub depth_clamp_enabled: bool,
    pub depth_key: Option<RtKey>,
    pub clear_depth_hint: f32,
    pub sampled_rt_key: Option<RtKey>,
    pub sampled_rt_keys: Vec<RtKey>,
    pub sampled_rt_slots: Vec<Option<RtKey>>,
    pub sampled_rt_fuzzy: bool,
    pub clear: bool,
    pub clear_color: [f32; 4],
    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_pool_limit: u32,
    pub fs_sampler_ids: Vec<u32>,
    pub fs_sampler_arrayed: bool,

    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub ssbo_data: Vec<(u32, Vec<u8>)>,
    pub flip_y: bool,
}

pub fn vertex_binding_read_range(
    call: &Maxwell3dDrawCall,
    binding: &VertexBufferBinding,
) -> Option<(u64, usize)> {
    if binding.stride == 0 {
        return None;
    }
    let stride = binding.stride as u64;
    let instanced = binding.divisor != 0;
    let start_vertex = if instanced || call.state.indexed {
        0
    } else {
        call.first_vertex
    };
    let vertex_span = if instanced {
        call.first_instance
            .saturating_add(call.instance_count.max(1))
    } else if call.state.indexed {
        call.first_vertex.saturating_add(call.vertex_count)
    } else {
        call.vertex_count
    };
    let start_byte = stride.saturating_mul(start_vertex as u64);
    let mut bytes = stride.saturating_mul(vertex_span as u64);
    if binding.size > 0 {
        if start_byte >= binding.size {
            return None;
        }
        bytes = bytes.min(binding.size - start_byte);
    }
    let bytes = bytes as usize;
    if bytes == 0 {
        return None;
    }
    Some((binding.addr.wrapping_add(start_byte), bytes))
}

pub fn expand_quad_vertices(src: &[u8], stride: usize) -> Vec<u8> {
    if stride == 0 || src.len() < stride * 4 {
        return src.to_vec();
    }
    let quads = (src.len() / stride) / 4;
    let mut out = Vec::with_capacity(quads * 6 * stride);
    for q in 0..quads {
        let base = q * 4;
        for &i in &[base, base + 1, base + 2, base, base + 2, base + 3] {
            let off = i * stride;
            out.extend_from_slice(&src[off..off + stride]);
        }
    }
    out
}

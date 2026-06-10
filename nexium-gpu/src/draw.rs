use ash::vk;
use crate::rt_cache::RtKey;

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
pub struct BlendState {
    pub enabled: bool,
    pub src_factor: vk::BlendFactor,
    pub dst_factor: vk::BlendFactor,
    pub op: vk::BlendOp,
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
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub fs_tex_ids: Vec<u32>,
    pub vertex_layout: VertexLayout,
    pub cbuf_addr: u64,
    pub cbuf_size: u32,
    pub cbuf_data: Option<Vec<u8>>,
    pub vertex_addr: u64,
    pub vertex_count: u32,
    pub index_addr: Option<u64>,
    pub index_count: Option<u32>,
    pub index_type: vk::IndexType,
    pub rt_key: RtKey,
    pub rt_format: vk::Format,
    pub vp_rect: Option<[f32; 4]>,
    pub state: DrawState,
    pub blend: BlendState,
    pub depth: DepthState,
    pub depth_key: Option<RtKey>,
    pub sampled_rt_key: Option<RtKey>,
    pub sampled_rt_fuzzy: bool,
    pub clear: bool,
    pub clear_color: [f32; 4],
    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_pool_limit: u32,
    pub fs_sampler_ids: Vec<u32>,

    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
}

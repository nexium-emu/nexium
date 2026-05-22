use ash::vk;
use crate::rt_cache::RtKey;

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

#[derive(Clone, Copy, Debug)]
pub struct Maxwell3dDrawCall {
    pub vs_bytes: u64,
    pub fs_bytes: u64,
    pub cbuf_addr: u64,
    pub cbuf_size: u32,
    pub vertex_addr: u64,
    pub vertex_count: u32,
    pub index_addr: Option<u64>,
    pub index_count: Option<u32>,
    pub index_type: vk::IndexType,
    pub rt_key: RtKey,
    pub state: DrawState,
    pub blend: BlendState,
    pub depth: DepthState,
    pub clear: bool,
    pub clear_color: [f32; 4],
}

pub struct DrawDispatcher;

impl DrawDispatcher {
    pub fn record_draw(
        _cmd_buf: vk::CommandBuffer,
        _draw: &Maxwell3dDrawCall,
    ) -> Result<(), String> {
        log::debug!("Recording draw call: {} vertices", _draw.vertex_count);
        Ok(())
    }
}

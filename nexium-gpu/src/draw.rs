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

pub fn primitive_restart_fixed_index(index_type: vk::IndexType) -> Option<u32> {
    match index_type {
        vk::IndexType::UINT16 => Some(u16::MAX as u32),
        vk::IndexType::UINT32 => Some(u32::MAX),
        _ => None,
    }
}

pub fn primitive_restart_topology_supported(topology: vk::PrimitiveTopology) -> bool {
    matches!(
        topology,
        vk::PrimitiveTopology::LINE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_STRIP
            | vk::PrimitiveTopology::TRIANGLE_FAN
    )
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
    pub constants: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct DepthState {
    pub test_enabled: bool,
    pub write_enabled: bool,
    pub compare_op: vk::CompareOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FermiExactRtToken {
    pub key: RtKey,
    pub stamp: u64,
    pub guest_va: u64,
    pub guest_size: u64,
    pub guest_generation: u64,
    pub bytes_per_pixel: usize,
    pub fermi_block_size: u32,
    pub block_width_log2: u32,
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
    pub tile_width_spacing: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct StencilFaceState {
    pub fail_op: vk::StencilOp,
    pub pass_op: vk::StencilOp,
    pub depth_fail_op: vk::StencilOp,
    pub compare_op: vk::CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
            reference: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StencilState {
    pub enabled: bool,
    pub front: StencilFaceState,
    pub back: StencilFaceState,
}

#[derive(Clone, Debug)]
pub struct StorageBufferSnapshot {
    pub binding: u32,
    pub guest_addr: u64,
    pub logical_size: usize,
    pub data_offset: usize,
    pub data: std::sync::Arc<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct GraphicsCbufSlotSnapshot {
    logical_slot: u8,
    read_len: u32,
    packed_offset: u32,
    word_count: u32,
    source_offset: u32,
    data: std::sync::Arc<Vec<u8>>,
}

impl GraphicsCbufSlotSnapshot {
    pub fn new(
        logical_slot: usize,
        read_len: usize,
        packed_offset: usize,
        data: std::sync::Arc<Vec<u8>>,
    ) -> Option<Self> {
        Self::new_with_offset(logical_slot, read_len, packed_offset, 0, data)
    }

    pub fn new_with_offset(
        logical_slot: usize,
        read_len: usize,
        packed_offset: usize,
        source_offset: usize,
        data: std::sync::Arc<Vec<u8>>,
    ) -> Option<Self> {
        let word_count = read_len.div_ceil(4);
        if logical_slot >= nexium_spirv::GFX_CBUF_SLOTS as usize
            || source_offset.checked_add(read_len)? > data.len()
            || packed_offset < nexium_spirv::GFX_CBUF_MIN_SIZE as usize
            || packed_offset % 16 != 0
        {
            return None;
        }
        Some(Self {
            logical_slot: logical_slot.try_into().ok()?,
            read_len: read_len.try_into().ok()?,
            packed_offset: packed_offset.try_into().ok()?,
            word_count: word_count.try_into().ok()?,
            source_offset: source_offset.try_into().ok()?,
            data,
        })
    }

    pub fn logical_slot(&self) -> usize {
        self.logical_slot as usize
    }

    pub fn read_len(&self) -> usize {
        self.read_len as usize
    }

    pub fn packed_offset(&self) -> usize {
        self.packed_offset as usize
    }

    pub fn word_count(&self) -> usize {
        self.word_count as usize
    }

    pub fn source_offset(&self) -> usize {
        self.source_offset as usize
    }

    pub fn data(&self) -> &std::sync::Arc<Vec<u8>> {
        &self.data
    }
}

#[derive(Clone, Debug)]
pub enum GraphicsCbufPayload {
    Owned(std::sync::Arc<Vec<u8>>),
    Slots {
        packed_size: u32,
        slots: Vec<GraphicsCbufSlotSnapshot>,
    },
}

impl GraphicsCbufPayload {
    pub fn owned(data: std::sync::Arc<Vec<u8>>) -> Self {
        Self::Owned(data)
    }

    pub fn from_slots(packed_size: usize, slots: Vec<GraphicsCbufSlotSnapshot>) -> Option<Self> {
        if packed_size < nexium_spirv::GFX_CBUF_MIN_SIZE as usize {
            return None;
        }
        let mut occupied = [false; nexium_spirv::GFX_CBUF_SLOTS as usize];
        for slot in &slots {
            let logical_slot = slot.logical_slot();
            let payload_len = slot.word_count().checked_mul(4)?;
            let end = slot.packed_offset().checked_add(payload_len)?;
            if occupied[logical_slot]
                || end > packed_size
                || slot.read_len() > payload_len
                || slot.source_offset().saturating_add(slot.read_len()) > slot.data().len()
            {
                return None;
            }
            occupied[logical_slot] = true;
        }
        Some(Self::Slots {
            packed_size: packed_size.try_into().ok()?,
            slots,
        })
    }

    pub fn packed_len(&self) -> usize {
        match self {
            Self::Owned(data) if data.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize => {
                data.len()
            }
            Self::Owned(_) => nexium_spirv::GFX_CBUF_MIN_SIZE as usize,
            Self::Slots { packed_size, .. } => *packed_size as usize,
        }
    }

    pub fn slots(&self) -> Option<&[GraphicsCbufSlotSnapshot]> {
        match self {
            Self::Owned(_) => None,
            Self::Slots { slots, .. } => Some(slots),
        }
    }

    pub fn word(&self, logical_slot: usize, byte_offset: usize) -> Option<u32> {
        if logical_slot >= nexium_spirv::GFX_CBUF_SLOTS as usize {
            return None;
        }
        match self {
            Self::Owned(data) => {
                let directory = logical_slot.checked_mul(8)?;
                let base_word =
                    u32::from_le_bytes(data.get(directory..directory + 4)?.try_into().ok()?);
                let word_count =
                    u32::from_le_bytes(data.get(directory + 4..directory + 8)?.try_into().ok()?);
                let start = usize::try_from(base_word)
                    .ok()?
                    .checked_mul(4)?
                    .checked_add(byte_offset)?;
                let slot_end = usize::try_from(base_word)
                    .ok()?
                    .checked_add(usize::try_from(word_count).ok()?)?
                    .checked_mul(4)?;
                let end = start.checked_add(4)?;
                (end <= slot_end)
                    .then(|| data.get(start..end))
                    .flatten()
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u32::from_le_bytes)
            }
            Self::Slots { slots, .. } => {
                let slot = slots
                    .iter()
                    .find(|slot| slot.logical_slot() == logical_slot)?;
                let end = byte_offset.checked_add(4)?;
                if end > slot.word_count().checked_mul(4)? {
                    return None;
                }
                let mut bytes = [0u8; 4];
                if byte_offset < slot.read_len() {
                    let copy_len = (slot.read_len() - byte_offset).min(4);
                    let source_offset = slot.source_offset().checked_add(byte_offset)?;
                    bytes[..copy_len]
                        .copy_from_slice(slot.data().get(source_offset..source_offset + copy_len)?);
                }
                Some(u32::from_le_bytes(bytes))
            }
        }
    }

    pub fn write_packed_to(&self, dst: &mut [u8]) -> bool {
        let packed_len = self.packed_len();
        let Some(dst) = dst.get_mut(..packed_len) else {
            return false;
        };
        match self {
            Self::Owned(data) if data.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize => {
                dst.copy_from_slice(data);
                true
            }
            Self::Owned(_) => write_empty_graphics_cbuf(dst),
            Self::Slots { slots, .. } => {
                if !write_empty_graphics_cbuf(dst) {
                    return false;
                }
                for slot in slots {
                    let directory = slot.logical_slot() * 8;
                    let base_word = slot.packed_offset() / 4;
                    dst[directory..directory + 4]
                        .copy_from_slice(&(base_word as u32).to_le_bytes());
                    dst[directory + 4..directory + 8]
                        .copy_from_slice(&(slot.word_count() as u32).to_le_bytes());
                    let start = slot.packed_offset();
                    let read_end = start + slot.read_len();
                    let source_start = slot.source_offset();
                    dst[start..read_end].copy_from_slice(
                        &slot.data()[source_start..source_start + slot.read_len()],
                    );
                }
                true
            }
        }
    }

    pub fn materialize(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Self::Owned(data) if data.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize => {
                std::borrow::Cow::Borrowed(data.as_slice())
            }
            _ => {
                let mut data = vec![0u8; self.packed_len()];
                let written = self.write_packed_to(&mut data);
                debug_assert!(written);
                std::borrow::Cow::Owned(data)
            }
        }
    }
}

fn write_empty_graphics_cbuf(dst: &mut [u8]) -> bool {
    if dst.len() < nexium_spirv::GFX_CBUF_MIN_SIZE as usize {
        return false;
    }
    dst.fill(0);
    for logical_slot in 0..nexium_spirv::GFX_CBUF_SLOTS as usize {
        let directory = logical_slot * 8;
        dst[directory..directory + 4]
            .copy_from_slice(&nexium_spirv::GFX_CBUF_ZERO_WORD.to_le_bytes());
    }
    true
}

pub const RESIDENT_CHUNK_SHIFT: u32 = 16;
pub const RESIDENT_CHUNK_SIZE: usize = 1 << RESIDENT_CHUNK_SHIFT;

#[derive(Clone, Debug)]
pub struct ResidentVertexChunk {
    pub chunk_key: u64,
    pub generation: u64,
    pub serial: u64,
    pub data: std::sync::Arc<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct ResidentVertexRange {
    pub binding: u32,
    pub stride: u64,
    pub cpu_va: u64,
    pub len: usize,
    pub chunks: Vec<ResidentVertexChunk>,
}

impl ResidentVertexRange {
    pub fn has_exact_coverage(&self) -> bool {
        let Ok(range_len) = u64::try_from(self.len) else {
            return false;
        };
        if range_len == 0 {
            return false;
        }
        let Some(range_end) = self.cpu_va.checked_add(range_len) else {
            return false;
        };
        let first_key = self.cpu_va >> RESIDENT_CHUNK_SHIFT;
        let last_key = (range_end - 1) >> RESIDENT_CHUNK_SHIFT;
        let Some(expected_count) = last_key
            .checked_sub(first_key)
            .and_then(|span| span.checked_add(1))
            .and_then(|count| usize::try_from(count).ok())
        else {
            return false;
        };
        expected_count == self.chunks.len()
            && self.chunks.iter().enumerate().all(|(index, chunk)| {
                chunk.data.len() == RESIDENT_CHUNK_SIZE
                    && u64::try_from(index)
                        .ok()
                        .and_then(|index| first_key.checked_add(index))
                        == Some(chunk.chunk_key)
            })
    }

    pub fn assemble(&self) -> Option<Vec<u8>> {
        if !self.has_exact_coverage() {
            return None;
        }
        let range_end = self.cpu_va.checked_add(u64::try_from(self.len).ok()?)?;
        let mut out = vec![0u8; self.len];
        let mut copied = 0usize;
        for chunk in &self.chunks {
            let chunk_base = chunk.chunk_key << RESIDENT_CHUNK_SHIFT;
            let lo = self.cpu_va.max(chunk_base);
            let hi = range_end.min(chunk_base.saturating_add(RESIDENT_CHUNK_SIZE as u64));
            let src = (lo - chunk_base) as usize;
            let dst = (lo - self.cpu_va) as usize;
            let len = (hi - lo) as usize;
            out[dst..dst + len].copy_from_slice(&chunk.data[src..src + len]);
            copied = copied.checked_add(len)?;
        }
        (copied == self.len).then_some(out)
    }

    pub fn matches_bytes(&self, bytes: &[u8]) -> bool {
        if bytes.len() != self.len || !self.has_exact_coverage() {
            return false;
        }
        let Some(range_end) = self.cpu_va.checked_add(self.len as u64) else {
            return false;
        };
        let mut compared = 0usize;
        for chunk in &self.chunks {
            let chunk_base = chunk.chunk_key << RESIDENT_CHUNK_SHIFT;
            let lo = self.cpu_va.max(chunk_base);
            let hi = range_end.min(chunk_base.saturating_add(RESIDENT_CHUNK_SIZE as u64));
            let src = (lo - chunk_base) as usize;
            let dst = (lo - self.cpu_va) as usize;
            let len = (hi - lo) as usize;
            if chunk.data[src..src + len] != bytes[dst..dst + len] {
                return false;
            }
            let Some(total) = compared.checked_add(len) else {
                return false;
            };
            compared = total;
        }
        compared == self.len
    }
}

#[derive(Clone, Debug)]
pub struct ResidentCbufSlot {
    pub logical_slot: u32,
    pub word_count: u32,
    pub chunk_index: u16,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub packed_offset: u32,
}

#[derive(Clone, Debug)]
pub struct ResidentCbufSource {
    pub page_key: u64,
    pub generation: u64,
    pub serial: u64,
    pub data: std::sync::Arc<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentCbufSegment {
    pub source_index: u32,
    pub byte_offset: u32,
    pub byte_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentCbufArenaSlot {
    pub logical_slot: u32,
    pub word_count: u32,
    pub segment_start: u32,
    pub segment_count: u16,
    pub byte_len: u32,
    pub packed_offset: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ResidentCbufArena {
    pub sources: Box<[ResidentCbufSource]>,
    pub segments: Box<[ResidentCbufSegment]>,
    pub slots: Box<[ResidentCbufArenaSlot]>,
}

#[derive(Clone, Debug, Default)]
pub struct ResidentCbufDraw {
    pub chunks: Vec<ResidentVertexChunk>,
    pub slots: Vec<ResidentCbufSlot>,
    pub packed_size: usize,
    pub arena: Option<std::sync::Arc<ResidentCbufArena>>,
    pub arena_slot_start: u32,
    pub arena_slot_count: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfiguredColorRt {
    pub raw_slot: u8,
    pub key: RtKey,
    pub format: vk::Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmallColorRtWriteback {
    pub key: RtKey,
    pub tile_mode: u32,
}

#[derive(Clone, Debug)]
pub struct Maxwell3dDrawCall {
    pub vs_spirv: std::sync::Arc<Vec<u32>>,
    pub fs_spirv: std::sync::Arc<Vec<u32>>,
    pub vs_gpu_va: u64,
    pub fs_gpu_va: u64,
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub vs_cbuf_mask: u64,
    pub fs_cbuf_mask: u64,
    pub fs_tex_ids: Vec<u32>,
    pub sprite_batch_mirror: bool,
    pub texture_numeric_manifest: Vec<crate::texture_manifest::TextureNumericBinding>,
    pub texture_sampled_only_mask: u32,
    pub texel_buffer_mask: u32,
    pub vs_tex_base: u32,
    pub vs_tex_count: u32,
    pub vertex_layout: VertexLayout,
    pub cbuf_addr: u64,
    pub cbuf_size: u32,
    pub cbuf_data: Option<GraphicsCbufPayload>,
    pub resident_cbuf: Option<ResidentCbufDraw>,
    pub vertex_addr: u64,
    pub vertex_bindings: Vec<VertexBufferBinding>,
    pub resident_vertex: Vec<ResidentVertexRange>,
    pub vertex_count: u32,
    pub first_vertex: u32,
    pub instance_count: u32,
    pub first_instance: u32,
    pub index_addr: Option<u64>,
    pub index_count: Option<u32>,
    pub index_type: vk::IndexType,
    pub index_data: Option<std::sync::Arc<Vec<u8>>>,
    pub resident_index: Option<ResidentVertexRange>,
    pub primitive_restart_enabled: bool,
    pub primitive_restart_index: u32,
    pub quad_expand: bool,
    pub rt_key: RtKey,
    pub small_color_rt_writebacks: Vec<SmallColorRtWriteback>,
    pub configured_color_rts: Vec<ConfiguredColorRt>,
    pub color_rt_keys: Vec<RtKey>,
    pub color_rt_formats: Vec<vk::Format>,
    pub rt_format: vk::Format,
    pub vp_rect: Option<[f32; 4]>,
    pub scissor: Option<[i32; 4]>,
    pub state: DrawState,
    pub blend: BlendState,
    pub depth: DepthState,
    pub depth_mode: u32,
    pub depth_format: vk::Format,
    pub depth_aspects: vk::ImageAspectFlags,
    pub stencil: StencilState,
    pub depth_clamp_enabled: bool,
    pub depth_key: Option<RtKey>,
    pub clear_depth_hint: f32,
    pub clear_stencil_hint: u32,
    pub sampled_rt_key: Option<RtKey>,
    pub sampled_rt_keys: Vec<RtKey>,
    pub sampled_rt_slots: Vec<Option<RtKey>>,
    pub sampled_rt_copy_sources: Vec<Option<RtKey>>,
    pub sampled_rt_fermi_exact_slots: Vec<Option<FermiExactRtToken>>,
    pub sampled_rt_fermi_snapshot_slots: Vec<Option<u64>>,
    pub sampled_rt_snapshot_slots: Vec<bool>,
    pub fragment_barrier_after: bool,
    pub texture_cache_invalidate_after: bool,
    pub clear: bool,
    pub clear_color: [f32; 4],
    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_pool_limit: u32,
    pub fs_sampler_ids: Vec<u32>,
    pub fs_sampler_arrayed: bool,
    pub vs_sampler_arrayed: bool,
    pub depth_compare_2d_mask: u32,
    pub depth_compare_cube_mask: u32,
    pub depth_compare_cube_array_mask: u32,

    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub ssbo_data: Vec<StorageBufferSnapshot>,
    pub present_flip_y: bool,
}

impl Maxwell3dDrawCall {
    pub fn host_primitive_restart_enabled(&self) -> bool {
        self.state.indexed
            && self.primitive_restart_enabled
            && primitive_restart_topology_supported(self.state.topology)
            && primitive_restart_fixed_index(self.index_type) == Some(self.primitive_restart_index)
    }
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
        u64::from(
            call.first_instance
                .saturating_add(call.instance_count.max(1)),
        )
    } else if call.state.indexed {
        indexed_vertex_span(call.first_vertex, call.vertex_count)
    } else {
        u64::from(call.vertex_count)
    };
    let start_byte = stride.saturating_mul(start_vertex as u64);
    let mut bytes = stride.saturating_mul(vertex_span);
    let attr_end_max = call
        .vertex_layout
        .attrs
        .iter()
        .filter(|a| a.binding == binding.binding)
        .map(|a| a.offset as u64 + attr_format_byte_size(a.format))
        .max()
        .unwrap_or(0);
    if attr_end_max > stride {
        bytes = bytes.saturating_add(attr_end_max - stride);
    }
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

fn indexed_vertex_span(first_vertex: u32, vertex_count: u32) -> u64 {
    let base_vertex = i64::from(first_vertex as i32);
    base_vertex.saturating_add(i64::from(vertex_count)).max(0) as u64
}

fn attr_format_byte_size(format: vk::Format) -> u64 {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_UINT | vk::Format::R8_SINT => {
            1
        }
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT => 2,
        vk::Format::R8G8B8_UNORM | vk::Format::R8G8B8_SNORM => 3,
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2B10G10R10_SNORM_PACK32
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R32_SFLOAT
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT => 4,
        vk::Format::R16G16B16_SFLOAT => 6,
        vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT => 8,
        vk::Format::R32G32B32_SFLOAT | vk::Format::R32G32B32_UINT | vk::Format::R32G32B32_SINT => {
            12
        }
        vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT => 16,
        _ => 16,
    }
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

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::{
        indexed_vertex_span, primitive_restart_fixed_index, primitive_restart_topology_supported,
        GraphicsCbufPayload, GraphicsCbufSlotSnapshot, ResidentVertexChunk, ResidentVertexRange,
        StorageBufferSnapshot, RESIDENT_CHUNK_SIZE,
    };

    fn resident_chunk(chunk_key: u64, fill: u8) -> ResidentVertexChunk {
        ResidentVertexChunk {
            chunk_key,
            generation: 1,
            serial: 1,
            data: std::sync::Arc::new(vec![fill; RESIDENT_CHUNK_SIZE]),
        }
    }

    #[test]
    fn resident_vertex_range_assembles_only_exact_full_coverage() {
        let chunk = RESIDENT_CHUNK_SIZE as u64;
        let range = ResidentVertexRange {
            binding: 3,
            stride: 4,
            cpu_va: chunk - 2,
            len: 4,
            chunks: vec![resident_chunk(0, 0x11), resident_chunk(1, 0x22)],
        };
        assert!(range.has_exact_coverage());
        assert_eq!(
            range.assemble().as_deref(),
            Some(&[0x11, 0x11, 0x22, 0x22][..])
        );
        assert!(range.matches_bytes(&[0x11, 0x11, 0x22, 0x22]));
        assert!(!range.matches_bytes(&[0x11, 0x11, 0x22, 0x23]));
        assert!(!range.matches_bytes(&[0x11, 0x11, 0x22]));

        for invalid in [
            ResidentVertexRange {
                chunks: range.chunks[..1].to_vec(),
                ..range.clone()
            },
            ResidentVertexRange {
                chunks: vec![resident_chunk(0, 0x11), resident_chunk(2, 0x22)],
                ..range.clone()
            },
            ResidentVertexRange {
                chunks: vec![
                    resident_chunk(0, 0x11),
                    resident_chunk(1, 0x22),
                    resident_chunk(2, 0x33),
                ],
                ..range.clone()
            },
            ResidentVertexRange {
                len: 0,
                ..range.clone()
            },
        ] {
            assert!(!invalid.has_exact_coverage());
            assert!(invalid.assemble().is_none());
            assert!(!invalid.matches_bytes(&[0x11, 0x11, 0x22, 0x22]));
        }

        let mut short = range;
        short.chunks[1].data = std::sync::Arc::new(vec![0; RESIDENT_CHUNK_SIZE - 1]);
        assert!(!short.has_exact_coverage());
        assert!(short.assemble().is_none());
    }

    #[test]
    fn primitive_restart_helpers_match_core_vulkan_rules() {
        assert_eq!(
            primitive_restart_fixed_index(vk::IndexType::UINT16),
            Some(u16::MAX as u32)
        );
        assert_eq!(
            primitive_restart_fixed_index(vk::IndexType::UINT32),
            Some(u32::MAX)
        );
        assert!(primitive_restart_topology_supported(
            vk::PrimitiveTopology::LINE_STRIP
        ));
        assert!(primitive_restart_topology_supported(
            vk::PrimitiveTopology::TRIANGLE_STRIP
        ));
        assert!(primitive_restart_topology_supported(
            vk::PrimitiveTopology::TRIANGLE_FAN
        ));
        assert!(!primitive_restart_topology_supported(
            vk::PrimitiveTopology::TRIANGLE_LIST
        ));
    }

    #[test]
    fn indexed_vertex_span_treats_base_vertex_as_signed() {
        assert_eq!(indexed_vertex_span((-1024i32) as u32, 1227), 203);
        assert_eq!(indexed_vertex_span(32, 1227), 1259);
        assert_eq!(indexed_vertex_span((-2048i32) as u32, 1227), 0);
    }

    #[test]
    fn storage_snapshot_clone_shares_immutable_payload() {
        let snapshot = StorageBufferSnapshot {
            binding: 1,
            guest_addr: 0x1234,
            logical_size: 4,
            data_offset: 0,
            data: std::sync::Arc::new(vec![1, 2, 3, 4]),
        };
        let cloned = snapshot.clone();

        assert!(std::sync::Arc::ptr_eq(&snapshot.data, &cloned.data));
        assert_eq!(cloned.data.as_slice(), [1, 2, 3, 4]);
    }

    #[test]
    fn graphics_cbuf_slot_payload_writes_packed_abi() {
        let first = std::sync::Arc::new(vec![1, 2, 3, 4, 5]);
        let second = std::sync::Arc::new(vec![9, 8, 7, 6]);
        let slots = vec![
            GraphicsCbufSlotSnapshot::new(0, 5, 304, first.clone()).unwrap(),
            GraphicsCbufSlotSnapshot::new(24, 4, 320, second.clone()).unwrap(),
        ];
        let payload = GraphicsCbufPayload::from_slots(324, slots).unwrap();
        let mut packed = vec![0xff; payload.packed_len()];

        assert!(payload.write_packed_to(&mut packed));
        assert_eq!(u32::from_le_bytes(packed[0..4].try_into().unwrap()), 76);
        assert_eq!(u32::from_le_bytes(packed[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(packed[8..12].try_into().unwrap()),
            nexium_spirv::GFX_CBUF_ZERO_WORD
        );
        assert_eq!(
            u32::from_le_bytes(packed[24 * 8..24 * 8 + 4].try_into().unwrap()),
            80
        );
        assert_eq!(&packed[304..309], first.as_slice());
        assert_eq!(&packed[309..320], &[0; 11]);
        assert_eq!(&packed[320..324], second.as_slice());
        assert_eq!(payload.word(0, 4), Some(5));
        assert!(std::sync::Arc::ptr_eq(
            payload.slots().unwrap()[0].data(),
            &first
        ));
    }
}

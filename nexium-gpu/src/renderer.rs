use ash::vk;
use parking_lot::Mutex;
use std::collections::{hash_map::DefaultHasher, hash_map::Entry, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::compute::ComputeBackend;
use crate::descriptor::{DescriptorPool, DescriptorSetLayout};
use crate::pipeline::PipelineCache;
use crate::rt_cache::{
    find_memory_type, rt_formats_compatible, same_physical_backing, RtCache, RtKey,
};
use crate::shader::ShaderCompiler;
use crate::texture_manifest::{
    texture_image_kind_for_slot, texture_numeric_type_for_slot, GraphicsTextureImageKind,
    TextureNumericBinding,
};

pub struct Renderer {
    inner: Mutex<RendererInner>,
}

static SUBMIT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_IDLE_GENERATION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

fn note_queue_submission() {
    SUBMIT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

fn note_queue_drained() {
    LAST_IDLE_GENERATION.store(
        SUBMIT_GENERATION.load(std::sync::atomic::Ordering::Acquire),
        std::sync::atomic::Ordering::Release,
    );
}

fn idle_skip_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_NO_IDLE_SKIP").is_some())
}

pub const GRAPHICS_RING_CAPACITY_BYTES: u64 = 16 * 1024 * 1024;
pub const GRAPHICS_RING_SAFE_BATCH_BYTES: u64 = GRAPHICS_RING_CAPACITY_BYTES / 2;
const MAX_STORAGE_BUFFER_OFFSET_ALIGNMENT: u64 = 256;

fn ring_request_upper_bound(size: u64, alignment: u64) -> u64 {
    align_up(size, alignment).saturating_add(alignment.saturating_sub(1))
}

pub fn graphics_draw_ring_bytes_upper_bound(call: &crate::draw::Maxwell3dDrawCall) -> u64 {
    let mut bytes = 0u64;
    for binding in &call.vertex_bindings {
        let Some((_, read_len)) = crate::draw::vertex_binding_read_range(call, binding) else {
            continue;
        };
        let stride = u64::from(binding.stride);
        let mut upload_len = read_len as u64;
        if call.quad_expand && stride != 0 && upload_len >= stride.saturating_mul(4) {
            let quads = (upload_len / stride) / 4;
            upload_len = quads.saturating_mul(6).saturating_mul(stride);
        }
        let alignment = stride.max(16);
        bytes = bytes.saturating_add(ring_request_upper_bound(upload_len, alignment));
    }
    if call.vertex_layout.bindings.iter().any(|binding| binding.stride == 0) {
        bytes = bytes.saturating_add(ring_request_upper_bound(16, 16));
    }
    if call.index_count.is_some_and(|count| count != 0)
        && call.index_data.as_ref().is_some_and(|data| !data.is_empty())
    {
        bytes = bytes.saturating_add(ring_request_upper_bound(
            call.index_data.as_ref().map_or(0, |data| data.len() as u64),
            4,
        ));
    }
    let cbuf_len = call
        .cbuf_data
        .as_ref()
        .filter(|data| data.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize)
        .map_or(nexium_spirv::GFX_CBUF_MIN_SIZE as u64, |data| data.len() as u64);
    bytes = bytes.saturating_add(ring_request_upper_bound(
        cbuf_len,
        MAX_STORAGE_BUFFER_OFFSET_ALIGNMENT,
    ));

    let mut ssbo_provided = [false; crate::descriptor::MAX_SSBO as usize];
    for (index, data) in &call.ssbo_data {
        if *index >= crate::descriptor::MAX_SSBO || data.is_empty() {
            continue;
        }
        ssbo_provided[*index as usize] = true;
        bytes = bytes.saturating_add(ring_request_upper_bound(data.len() as u64, 16));
    }
    if ssbo_provided.iter().any(|provided| !provided) {
        bytes = bytes.saturating_add(ring_request_upper_bound(16, 16));
    }
    bytes
}

struct RendererInner {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    cmd_pool: vk::CommandPool,
    rt_cache: RtCache,
    staging: HashMap<(u32, u32), StagingBuffer>,
    descriptor_layout: DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    shader_compiler: ShaderCompiler,
    pipeline_cache: PipelineCache,
    compute_backend: Option<ComputeBackend>,
    compute_unavailable_reason: String,
    dummy_images: HashMap<(u8, DummyImageKind), DummyImage>,
    dummy_texel_buffers: [Option<TexelBufferResource>; 3],
    default_sampler: Option<vk::Sampler>,
    sampler_cache: HashMap<crate::texture::TscEntry, vk::Sampler>,
    integer_sampler_cache: HashMap<crate::texture::TscEntry, vk::Sampler>,
    tex_cache: HashMap<TexCacheKey, CachedTexture>,
    texel_buffer_cache: HashMap<TexelBufferCacheKey, CachedTexelBuffer>,
    rt_reinterpret_cache: HashMap<RtReinterpretKey, RtReinterpretTexture>,
    frame_slots: [FrameSlot; 2],
    frame_index: usize,
    utility_slot: FrameSlot,
    pending_computes: Vec<PendingCompute>,
    compute_slot_pool: Vec<(vk::CommandBuffer, vk::Fence)>,
    next_pending_compute_id: u64,
    clear_slots: Vec<ClearSlot>,
    clear_slot_index: usize,
    ubo_ring: UboRing,
    min_storage_buffer_offset_alignment: u64,
    max_storage_buffer_range: u64,
    max_texel_buffer_elements: u32,
    pending_readbacks: HashMap<RtKey, VecDeque<PendingReadback>>,
    readback_slots: Vec<ReadbackSlot>,
    tele_last_emit_ns: u64,
    tele_ring_wraps: u64,
    tele_ring_waits: u64,
    tele_in_flight_mask: u32,
    depth_clamp_supported: bool,
    depth_clip_control_enabled: bool,
    vertex_attribute_divisor_supported: bool,
    sampler_filter_minmax_supported: bool,
    sampler_anisotropy_supported: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TexCacheKey {
    gpu_va: u64,
    width: u32,
    height: u32,
    layers: u32,
    base_layer: u32,
    view_layers: u32,
    mip_levels: u32,
    base_mip: u32,
    view_mips: u32,
    arrayed: bool,
    cube: bool,
    cube_array: bool,
    volume: bool,
    format: crate::texture::TicFormat,
    component_types: [crate::texture::ComponentType; 4],
    swizzle: [crate::texture::SwizzleSource; 4],
    is_srgb: bool,
    is_block_linear: bool,
    block_width_log2: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    tile_width_spacing: u32,
    numeric_type: u8,
}

type PendingTexture = (TexCacheKey, crate::texture::TicEntry, usize, usize);

#[derive(Clone, Copy)]
struct TextureMipCopy {
    buffer_offset: u64,
    mip_level: u32,
    width: u32,
    height: u32,
}

struct TextureUploadData {
    bytes: Vec<u8>,
    copies: Vec<TextureMipCopy>,
}

impl TextureUploadData {
    fn base(bytes: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            bytes,
            copies: vec![TextureMipCopy {
                buffer_offset: 0,
                mip_level: 0,
                width,
                height,
            }],
        }
    }
}

const TEXTURE_IDENTITY_SWIZZLE: [crate::texture::SwizzleSource; 4] = [
    crate::texture::SwizzleSource::R,
    crate::texture::SwizzleSource::G,
    crate::texture::SwizzleSource::B,
    crate::texture::SwizzleSource::A,
];

fn texture_numeric_cache_key(numeric_type: nexium_spirv::TextureNumericType) -> u8 {
    match numeric_type {
        nexium_spirv::TextureNumericType::Float => 0,
        nexium_spirv::TextureNumericType::Uint => 1,
        nexium_spirv::TextureNumericType::Sint => 2,
    }
}

const GRAPHICS_TEXTURE_NUMERIC_TYPES: [nexium_spirv::TextureNumericType; 3] = [
    nexium_spirv::TextureNumericType::Float,
    nexium_spirv::TextureNumericType::Uint,
    nexium_spirv::TextureNumericType::Sint,
];

fn texture_numeric_index(numeric_type: nexium_spirv::TextureNumericType) -> usize {
    texture_numeric_cache_key(numeric_type) as usize
}

fn texture_shader_id_for_slot(manifest: &[TextureNumericBinding], slot: usize) -> Option<u32> {
    manifest
        .binary_search_by_key(&(slot as u32), |binding| binding.descriptor_slot)
        .ok()
        .map(|index| manifest[index].shader_id)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphicsTextureBindOutcome {
    Success,
    Dummy,
    RtAlias,
    TexelBuffer,
    Rejection,
}

impl GraphicsTextureBindOutcome {
    fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Dummy => "dummy",
            Self::RtAlias => "rt-alias",
            Self::TexelBuffer => "texel-buffer",
            Self::Rejection => "rejection",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct GraphicsTextureTraceResource {
    resource_va: Option<u64>,
    raw_texture_type: Option<u32>,
    format: Option<String>,
    component_types: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    depth: Option<u32>,
    computed_layers: Option<u32>,
    base_layer: Option<u32>,
    min_mip: Option<u32>,
    max_mip: Option<u32>,
}

impl GraphicsTextureTraceResource {
    fn from_pending(pending: Option<PendingTexture>) -> Self {
        let Some((key, tic, _, _)) = pending else {
            return Self::default();
        };
        Self {
            resource_va: Some(tic.gpu_va),
            raw_texture_type: Some(tic.texture_type),
            format: Some(format!("{:?}", tic.format)),
            component_types: Some(format!("{:?}", tic.component_types)),
            width: Some(tic.width),
            height: Some(tic.height),
            depth: Some(tic.depth),
            computed_layers: Some(key.layers),
            base_layer: Some(tic.base_layer),
            min_mip: Some(tic.res_min_mip_level),
            max_mip: Some(tic.res_max_mip_level),
        }
    }

    fn rt_alias(alias: RtAlias) -> Self {
        Self {
            resource_va: Some(alias.key.gpu_va),
            format: Some(format!("vk:{:?}", alias.format)),
            width: Some(alias.key.width),
            height: Some(alias.key.height),
            depth: Some(alias.key.depth),
            computed_layers: Some(alias.key.render_layer_count()),
            base_layer: Some(0),
            min_mip: Some(0),
            max_mip: Some(0),
            ..Self::default()
        }
    }

    fn fill_missing_from(mut self, fallback: Self) -> Self {
        if self.resource_va.is_none() {
            self.resource_va = fallback.resource_va;
        }
        if self.raw_texture_type.is_none() {
            self.raw_texture_type = fallback.raw_texture_type;
        }
        if self.format.is_none() {
            self.format = fallback.format;
        }
        if self.component_types.is_none() {
            self.component_types = fallback.component_types;
        }
        if self.width.is_none() {
            self.width = fallback.width;
        }
        if self.height.is_none() {
            self.height = fallback.height;
        }
        if self.depth.is_none() {
            self.depth = fallback.depth;
        }
        if self.computed_layers.is_none() {
            self.computed_layers = fallback.computed_layers;
        }
        if self.base_layer.is_none() {
            self.base_layer = fallback.base_layer;
        }
        if self.min_mip.is_none() {
            self.min_mip = fallback.min_mip;
        }
        if self.max_mip.is_none() {
            self.max_mip = fallback.max_mip;
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphicsTextureBindTraceRecord {
    fs_gpu_va: u64,
    fs_hash: u64,
    shader_id: Option<u32>,
    descriptor_slot: usize,
    tic_id: Option<u32>,
    tic_address: Option<u64>,
    resource: GraphicsTextureTraceResource,
    numeric_family: &'static str,
    image_kind: GraphicsTextureImageKind,
    selected_binding: u32,
    outcome: GraphicsTextureBindOutcome,
    source: String,
    selected_view: String,
    selected_format: Option<String>,
    reason: Option<String>,
}

fn trace_optional_decimal<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn trace_optional_hex(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:#x}"))
}

fn trace_optional_text(value: Option<&str>) -> String {
    value.unwrap_or("-").to_string()
}

fn format_graphics_texture_bind_trace(record: &GraphicsTextureBindTraceRecord) -> String {
    let dimensions = match (
        record.resource.width,
        record.resource.height,
        record.resource.depth,
    ) {
        (Some(width), Some(height), Some(depth)) => format!("{width}x{height}x{depth}"),
        _ => "-".to_string(),
    };
    let mip_range = match (record.resource.min_mip, record.resource.max_mip) {
        (Some(min), Some(max)) => format!("{min}..{max}"),
        _ => "-".to_string(),
    };
    format!(
        "[bind-trace] fs_hash={:016x} fs_va={:#x} shader_id={} slot={} tic_id={} tic_addr={} resource_va={} raw_type={} format={:?} components={:?} dimensions={} layers={} base_layer={} mips={} numeric={} image_kind={:?} binding={} outcome={} source={:?} view={} view_format={:?} reason={:?}",
        record.fs_hash,
        record.fs_gpu_va,
        trace_optional_decimal(record.shader_id),
        record.descriptor_slot,
        trace_optional_decimal(record.tic_id),
        trace_optional_hex(record.tic_address),
        trace_optional_hex(record.resource.resource_va),
        trace_optional_decimal(record.resource.raw_texture_type),
        trace_optional_text(record.resource.format.as_deref()),
        trace_optional_text(record.resource.component_types.as_deref()),
        dimensions,
        trace_optional_decimal(record.resource.computed_layers),
        trace_optional_decimal(record.resource.base_layer),
        mip_range,
        record.numeric_family,
        record.image_kind,
        record.selected_binding,
        record.outcome.name(),
        record.source,
        record.selected_view,
        trace_optional_text(record.selected_format.as_deref()),
        record.reason.as_deref().unwrap_or("-"),
    )
}

fn texture_numeric_family_name(numeric_type: nexium_spirv::TextureNumericType) -> &'static str {
    match numeric_type {
        nexium_spirv::TextureNumericType::Float => "float",
        nexium_spirv::TextureNumericType::Uint => "uint",
        nexium_spirv::TextureNumericType::Sint => "sint",
    }
}

fn emit_graphics_texture_binding(
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    pending: Option<PendingTexture>,
    fallback_resource: Option<GraphicsTextureTraceResource>,
    numeric_type: nexium_spirv::TextureNumericType,
    outcome: GraphicsTextureBindOutcome,
    source: impl Into<String>,
    selected_view: impl Into<String>,
    selected_format: Option<String>,
    reason: Option<String>,
) {
    let tic_id = call.fs_tex_ids.get(slot).copied();
    let tic_address = tic_id.and_then(|tic_id| {
        (tic_id != u32::MAX && call.tic_pool_gpu_va != 0)
            .then(|| call.tic_pool_gpu_va.wrapping_add(u64::from(tic_id) * 32))
    });
    let mut resource = GraphicsTextureTraceResource::from_pending(pending);
    if let Some(fallback) = fallback_resource {
        resource = resource.fill_missing_from(fallback);
    }
    let image_kind = texture_image_kind_for_slot(&call.texture_numeric_manifest, slot);
    let record = GraphicsTextureBindTraceRecord {
        fs_gpu_va: call.fs_gpu_va,
        fs_hash: call.fs_hash,
        shader_id: texture_shader_id_for_slot(&call.texture_numeric_manifest, slot),
        descriptor_slot: slot,
        tic_id,
        tic_address,
        resource,
        numeric_family: texture_numeric_family_name(numeric_type),
        image_kind,
        selected_binding: nexium_spirv::graphics_image_binding(
            numeric_type,
            image_kind.spirv_kind(),
        ),
        outcome,
        source: source.into(),
        selected_view: selected_view.into(),
        selected_format,
        reason,
    };
    log::warn!("{}", format_graphics_texture_bind_trace(&record));
}

macro_rules! trace_graphics_texture_binding {
    ($call:expr, $($arg:expr),* $(,)?) => {{
        let call = $call;
        if bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
            emit_graphics_texture_binding(call, $($arg),*);
        }
    }};
}

fn log_graphics_texture_rejection(
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    pending: Option<PendingTexture>,
    numeric_type: nexium_spirv::TextureNumericType,
    phase: &str,
    error: &str,
) {
    if bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
        trace_graphics_texture_binding!(
            call,
            slot,
            pending,
            None,
            numeric_type,
            GraphicsTextureBindOutcome::Rejection,
            phase,
            "-",
            None,
            Some(error.to_string()),
        );
        return;
    }
    let tic = pending.map(|(_, tic, _, _)| tic);
    log::warn!(
        "texture upload rejected: fs={:#x} fs_hash={:#016x} slot={} shader_id={:?} tic_id={:?} tic_va={:?} raw_type={:?} fmt={:?} ctypes={:?} numeric={:?} dims={:?} base_layer={:?} mips={:?} phase={} reason={}",
        call.fs_gpu_va,
        call.fs_hash,
        slot,
        texture_shader_id_for_slot(&call.texture_numeric_manifest, slot),
        call.fs_tex_ids.get(slot),
        tic.map(|tic| tic.gpu_va),
        tic.map(|tic| tic.texture_type),
        tic.map(|tic| tic.format),
        tic.map(|tic| tic.component_types),
        numeric_type,
        tic.map(|tic| (tic.width, tic.height, tic.depth)),
        tic.map(|tic| tic.base_layer),
        tic.map(|tic| (tic.res_min_mip_level, tic.res_max_mip_level)),
        phase,
        error,
    );
}

#[derive(Clone, Copy)]
struct GraphicsDummyViews {
    image_2d: [vk::ImageView; 3],
    image_2d_array: [vk::ImageView; 3],
    image_3d: [vk::ImageView; 3],
    image_cube: [vk::ImageView; 3],
    image_cube_array: [vk::ImageView; 3],
    depth_image_2d: vk::ImageView,
    depth_image_2d_array: vk::ImageView,
    depth_image_cube: vk::ImageView,
    depth_image_cube_array: vk::ImageView,
    texel_buffer: [vk::BufferView; 3],
}

fn descriptor_slot_uses_arrayed_2d(
    slot: usize,
    vs_tex_base: u32,
    vs_tex_count: u32,
    fs_sampler_arrayed: bool,
    vs_sampler_arrayed: bool,
) -> bool {
    let slot = slot as u32;
    if slot >= vs_tex_base && slot < vs_tex_base.saturating_add(vs_tex_count) {
        vs_sampler_arrayed
    } else {
        fs_sampler_arrayed
    }
}

fn descriptor_slot_masked(mask: u32, slot: usize) -> bool {
    slot < u32::BITS as usize && mask & (1u32 << slot) != 0
}

impl GraphicsDummyViews {
    fn image_2d_for_slot(
        &self,
        call: &crate::draw::Maxwell3dDrawCall,
        slot: usize,
    ) -> [vk::ImageView; 3] {
        let arrayed = descriptor_slot_uses_arrayed_2d(
            slot,
            call.vs_tex_base,
            call.vs_tex_count,
            call.fs_sampler_arrayed,
            call.vs_sampler_arrayed,
        );
        let mut views = if arrayed {
            self.image_2d_array
        } else {
            self.image_2d
        };
        if descriptor_slot_masked(call.depth_compare_2d_mask, slot) {
            views[texture_numeric_index(
                nexium_spirv::TextureNumericType::Float,
            )] = if arrayed {
                self.depth_image_2d_array
            } else {
                self.depth_image_2d
            };
        }
        views
    }

    fn image_cube_for_slot(
        &self,
        call: &crate::draw::Maxwell3dDrawCall,
        slot: usize,
    ) -> [vk::ImageView; 3] {
        let mut views = self.image_cube;
        if descriptor_slot_masked(call.depth_compare_cube_mask, slot) {
            views[texture_numeric_index(
                nexium_spirv::TextureNumericType::Float,
            )] = self.depth_image_cube;
        }
        views
    }

    fn image_cube_array_for_slot(
        &self,
        call: &crate::draw::Maxwell3dDrawCall,
        slot: usize,
    ) -> [vk::ImageView; 3] {
        let mut views = self.image_cube_array;
        if descriptor_slot_masked(call.depth_compare_cube_array_mask, slot) {
            views[texture_numeric_index(
                nexium_spirv::TextureNumericType::Float,
            )] = self.depth_image_cube_array;
        }
        views
    }
}

fn typed_sampled_image_infos(
    selected_views: &[vk::ImageView],
    selected_layouts: Option<&[vk::ImageLayout]>,
    manifest: &[TextureNumericBinding],
    dummy_views: &[[vk::ImageView; 3]],
) -> [Vec<vk::DescriptorImageInfo>; 3] {
    std::array::from_fn(|family| {
        selected_views
            .iter()
            .enumerate()
            .map(|(slot, selected_view)| {
                let selected_family =
                    texture_numeric_index(texture_numeric_type_for_slot(manifest, slot));
                let selected = selected_family == family;
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: if selected {
                        *selected_view
                    } else {
                        dummy_views[slot][family]
                    },
                    image_layout: if selected {
                        selected_layouts
                            .and_then(|layouts| layouts.get(slot).copied())
                            .unwrap_or(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    },
                }
            })
            .collect()
    })
}

fn typed_texel_buffer_views(
    selected_views: &[vk::BufferView],
    manifest: &[TextureNumericBinding],
    dummy_views: [vk::BufferView; 3],
) -> [Vec<vk::BufferView>; 3] {
    std::array::from_fn(|family| {
        selected_views
            .iter()
            .enumerate()
            .map(|(slot, selected_view)| {
                if texture_numeric_index(texture_numeric_type_for_slot(manifest, slot)) == family {
                    *selected_view
                } else {
                    dummy_views[family]
                }
            })
            .collect()
    })
}

fn texture_key_has_special_view(key: TexCacheKey) -> bool {
    key.arrayed || key.volume || key.cube || key.cube_array
}

fn route_texture_key_to_shader_image_kind(
    mut key: TexCacheKey,
    tic: &crate::texture::TicEntry,
    image_kind: GraphicsTextureImageKind,
) -> Result<TexCacheKey, String> {
    if image_kind == GraphicsTextureImageKind::Buffer {
        if !tic.is_buffer() {
            return Err(format!(
                "shader requires a Buffer view but TIC target type {} is not a buffer",
                tic.texture_type
            ));
        }
        key.arrayed = false;
        key.cube = false;
        key.cube_array = false;
        key.volume = false;
        return Ok(key);
    }
    if tic.is_buffer() {
        return Err(format!(
            "shader requires a {image_kind:?} image view but TIC target type {} is a buffer",
            tic.texture_type
        ));
    }

    key.arrayed = false;
    key.cube = false;
    key.cube_array = false;
    key.volume = false;
    match image_kind {
        GraphicsTextureImageKind::D2 => {
            if tic_is_volume(tic) {
                return Err("a 3D TIC cannot back a shader 2D image view".to_string());
            }
            key.view_layers = 1;
        }
        GraphicsTextureImageKind::D2Array => {
            if tic_is_volume(tic) {
                return Err("a 3D TIC cannot back a shader 2D-array image view".to_string());
            }
            key.arrayed = true;
        }
        GraphicsTextureImageKind::D3 => {
            if !tic_is_volume(tic) {
                return Err(format!(
                    "shader requires a 3D image view but TIC target type {} is not 3D",
                    tic.texture_type
                ));
            }
            key.volume = true;
        }
        GraphicsTextureImageKind::Cube | GraphicsTextureImageKind::CubeArray => {
            if tic_is_volume(tic) || key.width != key.height {
                return Err(format!(
                    "shader requires a {image_kind:?} view but TIC shape is {}x{} target_type={}",
                    key.width, key.height, tic.texture_type
                ));
            }
            if key.base_layer % 6 != 0 {
                return Err(format!(
                    "shader {image_kind:?} view base layer {} is not aligned to six faces",
                    key.base_layer
                ));
            }
            if image_kind == GraphicsTextureImageKind::Cube {
                key.cube = true;
                key.view_layers = 6;
            } else {
                key.cube_array = true;
            }
            if key.view_layers == 0 || key.view_layers % 6 != 0 {
                return Err(format!(
                    "shader {image_kind:?} view requires a nonzero multiple of six layers, got {}",
                    key.view_layers
                ));
            }
        }
        GraphicsTextureImageKind::Buffer => unreachable!(),
    }

    if key.base_layer.saturating_add(key.view_layers) > key.layers {
        return Err(format!(
            "shader {image_kind:?} view layers {}..{} exceed TIC storage layer count {}",
            key.base_layer,
            key.base_layer.saturating_add(key.view_layers),
            key.layers
        ));
    }
    Ok(key)
}

fn texture_requires_integer_sampler(
    numeric_type: nexium_spirv::TextureNumericType,
    stencil_alias: bool,
) -> bool {
    numeric_type != nexium_spirv::TextureNumericType::Float || stencil_alias
}

struct PreparedVertexBinding {
    binding: u32,
    stride: u64,
    data: Vec<u8>,
}

struct CachedTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    hash: u64,
    gen: u64,
    verified: std::time::Instant,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TexelBufferCacheKey {
    gpu_va: u64,
    elements: u32,
    format: crate::texture::TicFormat,
    view_format: i32,
}

struct TexelBufferResource {
    buffer: vk::Buffer,
    view: vk::BufferView,
    memory: vk::DeviceMemory,
}

struct CachedTexelBuffer {
    resource: TexelBufferResource,
    hash: u64,
    gen: u64,
    verified: std::time::Instant,
}

const TEX_VERIFY_PERIOD: std::time::Duration = std::time::Duration::from_millis(250);

#[derive(Clone, Copy)]
struct VolumeRtSlice {
    layer: u32,
    key: RtKey,
    image: vk::Image,
    layout: vk::ImageLayout,
    format: vk::Format,
    stamp: u64,
    src_x: u32,
    src_y: u32,
}

#[derive(Clone, Copy)]
struct RtAlias {
    key: RtKey,
    image: vk::Image,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    depth: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RtReinterpretKey {
    source: RtKey,
    texture: TexCacheKey,
}

struct RtReinterpretTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    source_stamp: u64,
}

#[derive(Clone, Copy)]
struct PendingRtReinterpret {
    key: RtReinterpretKey,
    source: RtAlias,
    source_layout: vk::ImageLayout,
    source_stamp: u64,
    track_source_layout: bool,
}

#[derive(Clone, Copy)]
struct PostSubmitTextureProbe {
    source: &'static str,
    key: RtKey,
    image: vk::Image,
    layout: vk::ImageLayout,
    format: vk::Format,
}

#[derive(Clone, Copy)]
struct ColorAliasSync {
    src_key: RtKey,
    src_image: vk::Image,
    src_layout: vk::ImageLayout,
    src_format: vk::Format,
    src_stamp: u64,
    dst_key: RtKey,
    dst_image: vk::Image,
    dst_layout: vk::ImageLayout,
    dst_format: vk::Format,
    dst_stamp: u64,
    src_width: u32,
    height: u32,
    bytes: u64,
}

#[derive(Clone, Copy)]
struct ColorRegionSync {
    src_key: RtKey,
    src_image: vk::Image,
    src_layout: vk::ImageLayout,
    src_format: vk::Format,
    src_stamp: u64,
    dst_format: vk::Format,
    dst_stamp: u64,
    src_x: u32,
    src_y: u32,
}

struct StagingBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct DummyImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum DummyImageKind {
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
    DepthD2,
    DepthD2Array,
    DepthCube,
    DepthCubeArray,
}

struct FrameSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    in_flight: bool,
    retired_dsets: Vec<vk::DescriptorSet>,
    retired_dset_pools: Vec<vk::DescriptorPool>,
    retired_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    retired_textures: Vec<CachedTexture>,
    retired_texel_buffers: Vec<CachedTexelBuffer>,
    retired_rt_reinterprets: Vec<RtReinterpretTexture>,
    retired_views: Vec<vk::ImageView>,
}

const BATCH_DESCRIPTOR_POOL_SETS: u32 = 4096;

struct DescriptorPoolBatch<'a> {
    device: &'a ash::Device,
    pools: Vec<vk::DescriptorPool>,
}

impl<'a> DescriptorPoolBatch<'a> {
    fn new(device: &'a ash::Device) -> Result<Self, String> {
        Ok(Self {
            device,
            pools: vec![DescriptorPool::new(device, BATCH_DESCRIPTOR_POOL_SETS)?.into_raw()],
        })
    }

    fn current(&self) -> vk::DescriptorPool {
        *self.pools.last().unwrap()
    }

    fn grow(&mut self) -> Result<vk::DescriptorPool, String> {
        let pool = DescriptorPool::new(self.device, BATCH_DESCRIPTOR_POOL_SETS)?.into_raw();
        self.pools.push(pool);
        Ok(pool)
    }

    fn into_raw(mut self) -> Vec<vk::DescriptorPool> {
        std::mem::take(&mut self.pools)
    }
}

impl Drop for DescriptorPoolBatch<'_> {
    fn drop(&mut self) {
        destroy_descriptor_pools(self.device, &mut self.pools);
    }
}

struct ClearSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    in_flight: bool,
}

struct PendingReadback {
    slot: usize,
    width: u32,
    height: u32,
    format: vk::Format,
    flip_y: Option<bool>,
}

struct ReadbackSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    stage: Option<StagingBuffer>,
    in_flight: bool,
}

struct PendingComputeOutput {
    binding: u32,
    width: u32,
    height: u32,
    depth: u32,
    format: crate::compute::ComputeStorageFormat,
    byte_len: usize,
}

struct PendingComputeTexel {
    resource_index: usize,
    byte_len: usize,
}

struct PendingCompute {
    id: u64,
    program_key: u64,
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    resources: Option<PreparedComputeResources>,
    outputs: Vec<PendingComputeOutput>,
    texels: Vec<PendingComputeTexel>,
    result: Option<Result<crate::compute::ComputeDispatchResult, String>>,
}

struct UboRing {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    head: u64,
    slot_head: [u64; 2],
}

unsafe impl Send for UboRing {}
unsafe impl Sync for UboRing {}

impl Renderer {
    pub fn dispatch_compute_sync(
        &self,
        dispatch: crate::compute::ComputeDispatch,
    ) -> crate::compute::ComputeDispatchOutcome {
        let mut inner = self.inner.lock();
        execute_compute_dispatch(&mut inner, dispatch, false)
    }

    pub fn dispatch_compute_lazy(
        &self,
        dispatch: crate::compute::ComputeDispatch,
    ) -> crate::compute::ComputeDispatchOutcome {
        let mut inner = self.inner.lock();
        execute_compute_dispatch(&mut inner, dispatch, true)
    }

    pub fn take_pending_compute(
        &self,
        id: u64,
    ) -> Result<crate::compute::ComputeDispatchResult, String> {
        let mut inner = self.inner.lock();
        let index = inner
            .pending_computes
            .iter()
            .position(|pending| pending.id == id)
            .ok_or_else(|| format!("pending compute dispatch {id} is unknown"))?;
        settle_pending_compute(&mut inner, index);
        let pending = inner.pending_computes.remove(index);
        pending
            .result
            .unwrap_or_else(|| Err(format!("pending compute dispatch {id} was never settled")))
    }

    pub fn settle_pending_computes(&self) {
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
    }

    pub fn readback_compute_storage_seed(
        &self,
        key: RtKey,
        format: crate::compute::ComputeStorageFormat,
    ) -> Result<Option<Vec<u8>>, String> {
        let (alias_key, alias_layout, alias_format) = {
            let inner = self.inner.lock();
            let gpu_alias = (key.gpu_va != 0)
                .then(|| {
                    inner
                        .rt_cache
                        .find_content_bearing_color_at(key.width, key.height, key.gpu_va)
                })
                .flatten();
            let cpu_alias = (key.cpu_addr != 0)
                .then(|| {
                    inner.rt_cache.find_drawn_color_at_cpu(
                        key.width,
                        key.height,
                        key.nvmap_id,
                        key.cpu_addr,
                    )
                })
                .flatten();
            let any_alias = match (gpu_alias, cpu_alias) {
                (Some(gpu), Some(cpu)) => Some(if gpu.4 >= cpu.4 { gpu } else { cpu }),
                (Some(alias), None) | (None, Some(alias)) => Some(alias),
                (None, None) => None,
            };
            let Some((alias_key, _, alias_layout, alias_format, _)) = any_alias else {
                return Ok(None);
            };
            if alias_key.width != key.width
                || alias_key.height != key.height
                || alias_key.depth != key.depth
                || alias_key.is_3d != key.is_3d
            {
                return Err(format!(
                    "live storage alias {} does not exactly match compute target {}",
                    alias_key.label(),
                    key.label()
                ));
            }
            (alias_key, alias_layout, alias_format)
        };

        if alias_layout == vk::ImageLayout::UNDEFINED {
            return Err(format!(
                "live storage alias {} has undefined contents",
                alias_key.label()
            ));
        }
        if alias_key.is_3d {
            return Err(format!(
                "live 3D storage alias {} cannot be seeded safely yet",
                alias_key.label()
            ));
        }
        let storage_format = format.vk_format();
        if !rt_formats_compatible(alias_format, storage_format) {
            return Err(format!(
                "live storage alias {} format {:?} is incompatible with {:?}",
                alias_key.label(),
                alias_format,
                storage_format
            ));
        }
        let expected_len = (key.width as usize)
            .checked_mul(key.height as usize)
            .and_then(|pixels| pixels.checked_mul(format.bytes_per_pixel()))
            .ok_or_else(|| format!("live storage alias {} size overflows", alias_key.label()))?;
        let (width, height, bpp, bytes) = self
            .readback_key_raw(alias_key)
            .ok_or_else(|| format!("could not read live storage alias {}", alias_key.label()))?;
        if width != key.width
            || height != key.height
            || bpp != format.bytes_per_pixel()
            || bytes.len() != expected_len
        {
            return Err(format!(
                "live storage alias {} readback is {}x{}x{} ({} bytes), expected {}x{}x{} ({} bytes)",
                alias_key.label(),
                width,
                height,
                bpp,
                bytes.len(),
                key.width,
                key.height,
                format.bytes_per_pixel(),
                expected_len
            ));
        }
        Ok(Some(bytes))
    }

    pub fn wait_idle(&self) {
        let generation = SUBMIT_GENERATION.load(std::sync::atomic::Ordering::Acquire);
        let inner = self.inner.lock();
        unsafe {
            let _ = inner.device.device_wait_idle();
        }
        LAST_IDLE_GENERATION.store(generation, std::sync::atomic::Ordering::Release);
    }

    pub fn wait_idle_if_dirty(&self) -> bool {
        if !idle_skip_disabled()
            && LAST_IDLE_GENERATION.load(std::sync::atomic::Ordering::Acquire)
                == SUBMIT_GENERATION.load(std::sync::atomic::Ordering::Acquire)
        {
            return false;
        }
        self.wait_idle();
        true
    }

    pub fn clear_texture_cache(&self) {
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let drained: Vec<_> = inner.tex_cache.drain().map(|(_, t)| t).collect();
        let texel_drained: Vec<_> = inner
            .texel_buffer_cache
            .drain()
            .map(|(_, buffer)| buffer)
            .collect();
        let reinterpret_drained: Vec<_> =
            inner.rt_reinterpret_cache.drain().map(|(_, t)| t).collect();
        let cleared = drained.len() + texel_drained.len() + reinterpret_drained.len();
        let retire_idx = (inner.frame_index + 1) % inner.frame_slots.len();
        if inner.frame_slots[retire_idx].in_flight {
            inner.frame_slots[retire_idx]
                .retired_textures
                .extend(drained);
            inner.frame_slots[retire_idx]
                .retired_texel_buffers
                .extend(texel_drained);
            inner.frame_slots[retire_idx]
                .retired_rt_reinterprets
                .extend(reinterpret_drained);
        } else {
            for t in drained {
                unsafe {
                    inner.device.destroy_image_view(t.view, None);
                    inner.device.destroy_image(t.image, None);
                    inner.device.free_memory(t.memory, None);
                }
            }
            for buffer in texel_drained {
                destroy_texel_buffer(&inner.device, buffer.resource);
            }
            for t in reinterpret_drained {
                unsafe {
                    inner.device.destroy_image_view(t.view, None);
                    inner.device.destroy_image(t.image, None);
                    inner.device.free_memory(t.memory, None);
                }
            }
        }
        if cleared != 0 && std::env::var_os("NEXIUM_TEX_CACHE_DBG").is_some() {
            log::warn!("[tex-cache] cleared {} cached textures", cleared);
        }
    }

    pub fn invalidate_texture_address(&self, gpu_va: u64) {
        let mut inner = self.inner.lock();
        let keys: Vec<_> = inner
            .tex_cache
            .keys()
            .filter(|key| key.gpu_va == gpu_va)
            .copied()
            .collect();
        if keys.is_empty() {
            return;
        }
        let drained: Vec<_> = keys
            .into_iter()
            .filter_map(|key| inner.tex_cache.remove(&key))
            .collect();
        let retire_idx = (inner.frame_index + 1) % inner.frame_slots.len();
        if inner.frame_slots[retire_idx].in_flight {
            inner.frame_slots[retire_idx]
                .retired_textures
                .extend(drained);
        } else {
            for texture in drained {
                unsafe {
                    inner.device.destroy_image_view(texture.view, None);
                    inner.device.destroy_image(texture.image, None);
                    inner.device.free_memory(texture.memory, None);
                }
            }
        }
    }

    pub fn invalidate_render_target_content(&self, key: RtKey) {
        self.inner.lock().rt_cache.mark_guest_written(key);
    }

    pub fn new() -> Result<Arc<Self>, String> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("Vulkan entry load failed: {:?}", e))?;

        let want_validation = std::env::var("NEXIUM_VK_VALIDATION").ok().as_deref() == Some("1");
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let validation_available = want_validation
            && unsafe { entry.enumerate_instance_layer_properties() }
                .map(|layers| {
                    layers.iter().any(|l| {
                        let name = unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) };
                        name == validation_layer
                    })
                })
                .unwrap_or(false);
        if want_validation && !validation_available {
            log::warn!(
                "NEXIUM_VK_VALIDATION=1 but VK_LAYER_KHRONOS_validation unavailable; continuing without"
            );
        }
        let mut layer_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();
        let mut ext_ptrs: Vec<*const std::os::raw::c_char> = Vec::new();
        let want_syncval = std::env::var("NEXIUM_VK_SYNCVAL").ok().as_deref() == Some("1");
        if validation_available {
            layer_ptrs.push(validation_layer.as_ptr());
            ext_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());
            if want_syncval {
                ext_ptrs.push(ash::ext::validation_features::NAME.as_ptr());
            }
            log::info!("Vulkan validation layers ENABLED (guest ash instance)");
        }
        let syncval_enables = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let validation_features = vk::ValidationFeaturesEXT {
            s_type: vk::StructureType::VALIDATION_FEATURES_EXT,
            enabled_validation_feature_count: syncval_enables.len() as u32,
            p_enabled_validation_features: syncval_enables.as_ptr(),
            disabled_validation_feature_count: 0,
            p_disabled_validation_features: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let inst_pnext: *const std::ffi::c_void = if validation_available && want_syncval {
            &validation_features as *const _ as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };

        let app = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_application_name: c"NeXium".as_ptr(),
            application_version: 1,
            p_engine_name: c"NeXium".as_ptr(),
            engine_version: 1,
            api_version: vk::API_VERSION_1_3,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let inst_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app,
            enabled_extension_count: ext_ptrs.len() as u32,
            pp_enabled_extension_names: ext_ptrs.as_ptr(),
            enabled_layer_count: layer_ptrs.len() as u32,
            pp_enabled_layer_names: layer_ptrs.as_ptr(),
            p_next: inst_pnext,
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let instance = unsafe {
            entry
                .create_instance(&inst_info, None)
                .map_err(|e| format!("create_instance: {:?}", e))?
        };

        if validation_available {
            let dbg = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT {
                s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vk_validation_callback),
                p_user_data: std::ptr::null_mut(),
                p_next: std::ptr::null(),
                flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
                _marker: std::marker::PhantomData,
            };
            match unsafe { dbg.create_debug_utils_messenger(&info, None) } {
                Ok(_) => {}
                Err(e) => log::warn!("debug_utils messenger create failed: {:?}", e),
            }
        }

        let phys_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("enumerate_physical_devices: {:?}", e))?;
        if phys_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err("no Vulkan physical devices".to_string());
        }
        let physical_device = phys_devices
            .iter()
            .copied()
            .find(|d| {
                let p = unsafe { instance.get_physical_device_properties(*d) };
                p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap_or(phys_devices[0]);

        let qf = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family =
            qf.iter()
                .position(|q| {
                    q.queue_flags
                        .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .or_else(|| {
                    qf.iter()
                        .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                })
                .ok_or_else(|| "no graphics queue family".to_string())? as u32;
        let queue_supports_compute = qf[queue_family as usize]
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE);

        let prio = 1.0f32;
        let queue_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            queue_family_index: queue_family,
            queue_count: 1,
            p_queue_priorities: &prio,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let dcc_ext_supported = device_extensions.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::EXT_DEPTH_CLIP_CONTROL_NAME
        });
        let sampler_filter_minmax_supported = device_extensions.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::EXT_SAMPLER_FILTER_MINMAX_NAME
        });
        let vertex_attribute_divisor_khr_supported = device_extensions.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME
        });
        let vertex_attribute_divisor_ext_supported = device_extensions.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::EXT_VERTEX_ATTRIBUTE_DIVISOR_NAME
        });
        let vertex_attribute_divisor_ext_present =
            vertex_attribute_divisor_khr_supported || vertex_attribute_divisor_ext_supported;
        let workgroup_explicit_layout_ext_supported = device_extensions.iter().any(|e| {
            let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME
        });
        let workgroup_explicit_layout_supported = if workgroup_explicit_layout_ext_supported {
            let mut feature = vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
            let mut features = vk::PhysicalDeviceFeatures2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
                p_next: &mut feature as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
            feature.workgroup_memory_explicit_layout == vk::TRUE
        } else {
            false
        };
        let vertex_attribute_divisor_supported = if vertex_attribute_divisor_ext_present {
            let mut vad_feat = vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default();
            let mut feats2 = vk::PhysicalDeviceFeatures2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
                p_next: &mut vad_feat as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe { instance.get_physical_device_features2(physical_device, &mut feats2) };
            vad_feat.vertex_attribute_instance_rate_divisor == vk::TRUE
        } else {
            false
        };
        let dcc_opt_in = std::env::var("NEXIUM_DEPTH_CLIP_CTL")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let enable_depth_clip_control = if dcc_opt_in && dcc_ext_supported {
            let mut dcc_feat = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default();
            let mut feats2 = vk::PhysicalDeviceFeatures2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
                p_next: &mut dcc_feat as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe { instance.get_physical_device_features2(physical_device, &mut feats2) };
            dcc_feat.depth_clip_control == vk::TRUE
        } else {
            false
        };

        let mut supported_features_11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut supported_features_12 = vk::PhysicalDeviceVulkan12Features {
            p_next: &mut supported_features_11 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut supported_features_13 = vk::PhysicalDeviceVulkan13Features {
            p_next: &mut supported_features_12 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut supported_features = vk::PhysicalDeviceFeatures2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
            p_next: &mut supported_features_13 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe { instance.get_physical_device_features2(physical_device, &mut supported_features) };
        let shader_draw_parameters_supported =
            supported_features_11.shader_draw_parameters == vk::TRUE;
        let shader_output_layer_supported = supported_features_12.shader_output_layer == vk::TRUE;
        let uniform_buffer_standard_layout_supported =
            supported_features_12.uniform_buffer_standard_layout == vk::TRUE;
        let subgroup_size_control_supported =
            supported_features_13.subgroup_size_control == vk::TRUE;
        let mut subgroup_size_properties =
            vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut float_controls_properties = vk::PhysicalDeviceFloatControlsProperties::default();
        subgroup_size_properties.p_next =
            &mut subgroup_properties as *mut _ as *mut std::ffi::c_void;
        subgroup_properties.p_next =
            &mut float_controls_properties as *mut _ as *mut std::ffi::c_void;
        let mut properties2 = vk::PhysicalDeviceProperties2 {
            s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
            p_next: &mut subgroup_size_properties as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        let signed_zero_inf_nan_preserve_float32_supported =
            float_controls_properties.shader_signed_zero_inf_nan_preserve_float32 == vk::TRUE;
        let mut features_11 = vk::PhysicalDeviceVulkan11Features {
            shader_draw_parameters: if shader_draw_parameters_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        };
        let mut features_12 = vk::PhysicalDeviceVulkan12Features {
            shader_output_layer: if shader_output_layer_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            uniform_buffer_standard_layout: if uniform_buffer_standard_layout_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            p_next: &mut features_11 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut features_13 = vk::PhysicalDeviceVulkan13Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            subgroup_size_control: if subgroup_size_control_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            p_next: &mut features_12 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        let mut workgroup_explicit_layout_feature =
            vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR::default();
        workgroup_explicit_layout_feature.workgroup_memory_explicit_layout = vk::TRUE;
        if workgroup_explicit_layout_supported {
            workgroup_explicit_layout_feature.p_next =
                &mut features_13 as *mut _ as *mut std::ffi::c_void;
        }
        let compute_feature_chain: *mut std::ffi::c_void = if workgroup_explicit_layout_supported {
            &mut workgroup_explicit_layout_feature as *mut _ as *mut std::ffi::c_void
        } else {
            &mut features_13 as *mut _ as *mut std::ffi::c_void
        };
        let mut dcc_feature = vk::PhysicalDeviceDepthClipControlFeaturesEXT {
            s_type: vk::StructureType::PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
            depth_clip_control: vk::TRUE,
            p_next: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
        };
        let mut vertex_attribute_divisor_feature =
            vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default();
        vertex_attribute_divisor_feature.vertex_attribute_instance_rate_divisor = vk::TRUE;
        if vertex_attribute_divisor_supported {
            vertex_attribute_divisor_feature.p_next = compute_feature_chain;
        }
        if enable_depth_clip_control {
            dcc_feature.p_next = if vertex_attribute_divisor_supported {
                &mut vertex_attribute_divisor_feature as *mut _ as *mut std::ffi::c_void
            } else {
                compute_feature_chain
            };
            log::info!(
                "VK_EXT_depth_clip_control enabled via NEXIUM_DEPTH_CLIP_CTL \
                 (Maxwell -1..+1 clip-Z honored)"
            );
        } else if dcc_opt_in && !dcc_ext_supported {
            log::info!(
                "VK_EXT_depth_clip_control opt-in requested but NOT supported; \
                 falling back to 0..1 clip-Z"
            );
        } else {
            log::info!(
                "VK_EXT_depth_clip_control disabled (default); \
                 set NEXIUM_DEPTH_CLIP_CTL=1 to opt in"
            );
        }

        let mut enabled_ext_names: Vec<*const std::os::raw::c_char> = Vec::new();
        if enable_depth_clip_control {
            enabled_ext_names.push(vk::EXT_DEPTH_CLIP_CONTROL_NAME.as_ptr());
        }
        if workgroup_explicit_layout_supported {
            enabled_ext_names.push(vk::KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME.as_ptr());
            log::info!("VK_KHR_workgroup_memory_explicit_layout enabled for guest compute");
        } else {
            log::info!(
                "VK_KHR_workgroup_memory_explicit_layout unavailable; kernels requiring it will fall back"
            );
        }
        if sampler_filter_minmax_supported {
            enabled_ext_names.push(vk::EXT_SAMPLER_FILTER_MINMAX_NAME.as_ptr());
            log::info!("VK_EXT_sampler_filter_minmax enabled");
        } else {
            log::info!(
                "VK_EXT_sampler_filter_minmax unavailable; min/max samplers use weighted average"
            );
        }
        if vertex_attribute_divisor_supported && vertex_attribute_divisor_khr_supported {
            enabled_ext_names.push(vk::KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME.as_ptr());
            log::info!("VK_KHR_vertex_attribute_divisor enabled (feature confirmed)");
        } else if vertex_attribute_divisor_supported && vertex_attribute_divisor_ext_supported {
            enabled_ext_names.push(vk::EXT_VERTEX_ATTRIBUTE_DIVISOR_NAME.as_ptr());
            log::info!("VK_EXT_vertex_attribute_divisor enabled (feature confirmed)");
        } else if vertex_attribute_divisor_ext_present {
            log::info!("vertex attribute divisor extension present but feature unsupported; divisors >1 collapse to instance rate");
        } else {
            log::info!("vertex attribute divisor extension unavailable; divisors >1 collapse to instance rate");
        }
        let p_next_chain: *mut std::ffi::c_void = if enable_depth_clip_control {
            &mut dcc_feature as *mut _ as *mut std::ffi::c_void
        } else if vertex_attribute_divisor_supported {
            &mut vertex_attribute_divisor_feature as *mut _ as *mut std::ffi::c_void
        } else {
            compute_feature_chain
        };
        let core_features = unsafe { instance.get_physical_device_features(physical_device) };
        let depth_clamp_supported = core_features.depth_clamp == vk::TRUE;
        let independent_blend_supported = core_features.independent_blend == vk::TRUE;
        let sampler_anisotropy_supported = core_features.sampler_anisotropy == vk::TRUE;
        let storage_image_write_without_format_supported =
            core_features.shader_storage_image_write_without_format == vk::TRUE;
        let storage_image_extended_formats_supported =
            core_features.shader_storage_image_extended_formats == vk::TRUE;
        let image_gather_extended_supported =
            core_features.shader_image_gather_extended == vk::TRUE;
        if !depth_clamp_supported {
            log::info!("Vulkan depthClamp feature unavailable; Maxwell depth clamp disabled");
        }
        if !independent_blend_supported {
            log::info!("Vulkan independentBlend feature unavailable; per-target masks collapsed");
        }
        if !sampler_anisotropy_supported {
            log::info!("Vulkan samplerAnisotropy feature unavailable; TSC anisotropy disabled");
        }
        if shader_draw_parameters_supported {
            log::info!("Vulkan shaderDrawParameters enabled");
        } else {
            log::warn!(
                "Vulkan shaderDrawParameters unavailable; guest BaseInstance reads unsupported"
            );
        }
        if shader_output_layer_supported {
            log::info!("Vulkan shaderOutputLayer enabled");
        } else {
            log::warn!("Vulkan shaderOutputLayer unavailable; layered guest draws are unsupported");
        }
        let enabled_core_features = vk::PhysicalDeviceFeatures {
            robust_buffer_access: vk::TRUE,
            depth_clamp: if depth_clamp_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            independent_blend: if independent_blend_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            sampler_anisotropy: if sampler_anisotropy_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            shader_storage_image_write_without_format:
                if storage_image_write_without_format_supported {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
            shader_storage_image_extended_formats: if storage_image_extended_formats_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            shader_image_gather_extended: if image_gather_extended_supported {
                vk::TRUE
            } else {
                vk::FALSE
            },
            ..Default::default()
        };
        let dev_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: enabled_ext_names.len() as u32,
            pp_enabled_extension_names: if enabled_ext_names.is_empty() {
                std::ptr::null()
            } else {
                enabled_ext_names.as_ptr()
            },
            p_enabled_features: &enabled_core_features,
            p_next: p_next_chain,
            ..Default::default()
        };
        let device = unsafe {
            instance
                .create_device(physical_device, &dev_info, None)
                .map_err(|e| format!("create_device: {:?}", e))?
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let cmd_pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            queue_family_index: queue_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let cmd_pool = unsafe {
            device
                .create_command_pool(&cmd_pool_info, None)
                .map_err(|e| format!("create_command_pool: {:?}", e))?
        };

        let mut rt_cache = RtCache::new();
        rt_cache.set_mem_properties(mem_props);

        let descriptor_layout = DescriptorSetLayout::new(&device)?;
        let descriptor_pool = DescriptorPool::new(&device, 8192)?;
        let shader_compiler = ShaderCompiler::new();
        let cache_uuid =
            unsafe { instance.get_physical_device_properties(physical_device) }.pipeline_cache_uuid;
        let device_tag: String = cache_uuid.iter().map(|b| format!("{:02x}", b)).collect();
        let pipeline_cache = PipelineCache::new(&device, descriptor_layout.layout, &device_tag)?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        let min_storage_buffer_offset_alignment =
            props.limits.min_storage_buffer_offset_alignment.max(1);
        let max_storage_buffer_range = u64::from(props.limits.max_storage_buffer_range);
        let max_texel_buffer_elements = props.limits.max_texel_buffer_elements;
        let compute_feature_reason = if !queue_supports_compute {
            Some("selected graphics queue has no compute capability".to_string())
        } else if !storage_image_write_without_format_supported {
            Some("shaderStorageImageWriteWithoutFormat is unavailable".to_string())
        } else if !image_gather_extended_supported {
            Some("shaderImageGatherExtended is unavailable".to_string())
        } else if !signed_zero_inf_nan_preserve_float32_supported {
            Some("shaderSignedZeroInfNanPreserveFloat32 is unavailable".to_string())
        } else if !uniform_buffer_standard_layout_supported {
            Some("uniformBufferStandardLayout is unavailable".to_string())
        } else if !subgroup_size_control_supported
            || !subgroup_size_properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            || subgroup_size_properties.min_subgroup_size > 32
            || subgroup_size_properties.max_subgroup_size < 32
        {
            Some("required compute subgroup size 32 is unavailable".to_string())
        } else if !subgroup_properties
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            || !subgroup_properties.supported_operations.contains(
                vk::SubgroupFeatureFlags::BASIC
                    | vk::SubgroupFeatureFlags::BALLOT
                    | vk::SubgroupFeatureFlags::VOTE
                    | vk::SubgroupFeatureFlags::SHUFFLE,
            )
        {
            Some(format!(
                "required compute subgroup operations are unavailable (stages={:?}, operations={:?})",
                subgroup_properties.supported_stages,
                subgroup_properties.supported_operations
            ))
        } else {
            None
        };
        let mut compute_unavailable_reason = compute_feature_reason.unwrap_or_default();
        let compute_backend = if compute_unavailable_reason.is_empty() {
            match ComputeBackend::new(
                &device,
                subgroup_size_properties.min_subgroup_size,
                subgroup_size_properties.max_subgroup_size,
                subgroup_size_properties.required_subgroup_size_stages,
                workgroup_explicit_layout_supported,
                storage_image_extended_formats_supported,
                &props.limits,
            ) {
                Ok(backend) => {
                    log::info!(
                        "generic Vulkan compute enabled (required subgroup range {}..={})",
                        subgroup_size_properties.min_subgroup_size,
                        subgroup_size_properties.max_subgroup_size
                    );
                    Some(backend)
                }
                Err(error) => {
                    compute_unavailable_reason = error;
                    None
                }
            }
        } else {
            None
        };
        if compute_backend.is_none() {
            log::warn!(
                "generic Vulkan compute unavailable: {}",
                compute_unavailable_reason
            );
        }

        let cb_alloc = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: cmd_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 3,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let frame_cmds = unsafe {
            device
                .allocate_command_buffers(&cb_alloc)
                .map_err(|e| format!("allocate_command_buffers(frame_slots): {:?}", e))?
        };

        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let fence_a = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|e| format!("create_fence(0): {:?}", e))?
        };
        let fence_b = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|e| format!("create_fence(1): {:?}", e))?
        };
        let fence_util = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|e| format!("create_fence(utility): {:?}", e))?
        };
        let frame_slots = [
            FrameSlot {
                fence: fence_a,
                cmd: frame_cmds[0],
                in_flight: false,
                retired_dsets: Vec::new(),
                retired_dset_pools: Vec::new(),
                retired_buffers: Vec::new(),
                retired_textures: Vec::new(),
                retired_texel_buffers: Vec::new(),
                retired_rt_reinterprets: Vec::new(),
                retired_views: Vec::new(),
            },
            FrameSlot {
                fence: fence_b,
                cmd: frame_cmds[1],
                in_flight: false,
                retired_dsets: Vec::new(),
                retired_dset_pools: Vec::new(),
                retired_buffers: Vec::new(),
                retired_textures: Vec::new(),
                retired_texel_buffers: Vec::new(),
                retired_rt_reinterprets: Vec::new(),
                retired_views: Vec::new(),
            },
        ];
        let utility_slot = FrameSlot {
            fence: fence_util,
            cmd: frame_cmds[2],
            in_flight: false,
            retired_dsets: Vec::new(),
            retired_dset_pools: Vec::new(),
            retired_buffers: Vec::new(),
            retired_textures: Vec::new(),
            retired_texel_buffers: Vec::new(),
            retired_rt_reinterprets: Vec::new(),
            retired_views: Vec::new(),
        };

        let mut clear_slots = Vec::with_capacity(4);
        for i in 0..4 {
            let fence = unsafe {
                device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("create_fence(clear {}): {:?}", i, e))?
            };
            let cmd = alloc_one_time_cmd(&device, cmd_pool)?;
            clear_slots.push(ClearSlot {
                fence,
                cmd,
                in_flight: false,
            });
        }

        let mut readback_slots = Vec::with_capacity(4);
        for i in 0..4 {
            let fence = unsafe {
                device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("create_fence(readback {}): {:?}", i, e))?
            };
            let cmd = alloc_one_time_cmd(&device, cmd_pool)?;
            readback_slots.push(ReadbackSlot {
                fence,
                cmd,
                stage: None,
                in_flight: false,
            });
        }

        let ubo_ring = create_ubo_ring(&device, &mem_props, GRAPHICS_RING_CAPACITY_BYTES)?;

        log::info!("nexium-gpu Renderer init OK: {} (Vulkan via Ash)", name);

        let renderer = Arc::new(Self {
            inner: Mutex::new(RendererInner {
                entry,
                instance,
                device,
                physical_device,
                queue,
                queue_family,
                mem_props,
                cmd_pool,
                rt_cache,
                staging: HashMap::new(),
                descriptor_layout,
                descriptor_pool,
                shader_compiler,
                pipeline_cache,
                compute_backend,
                compute_unavailable_reason,
                dummy_images: HashMap::new(),
                dummy_texel_buffers: [None, None, None],
                default_sampler: None,
                sampler_cache: HashMap::new(),
                integer_sampler_cache: HashMap::new(),
                tex_cache: HashMap::new(),
                texel_buffer_cache: HashMap::new(),
                rt_reinterpret_cache: HashMap::new(),
                frame_slots,
                frame_index: 0,
                utility_slot,
                pending_computes: Vec::new(),
                compute_slot_pool: Vec::new(),
                next_pending_compute_id: 1,
                clear_slots,
                clear_slot_index: 0,
                ubo_ring,
                min_storage_buffer_offset_alignment,
                max_storage_buffer_range,
                max_texel_buffer_elements,
                pending_readbacks: HashMap::new(),
                readback_slots,
                tele_last_emit_ns: 0,
                tele_ring_wraps: 0,
                tele_ring_waits: 0,
                tele_in_flight_mask: 0,
                depth_clamp_supported,
                depth_clip_control_enabled: enable_depth_clip_control,
                vertex_attribute_divisor_supported,
                sampler_filter_minmax_supported,
                sampler_anisotropy_supported,
            }),
        });
        renderer.prewarm();
        Ok(renderer)
    }

    fn prewarm(&self) {
        if !nexium_common::async_compile::enabled() {
            return;
        }
        let mut inner = self.inner.lock();
        let specs = inner.pipeline_cache.prewarm_specs();
        if specs.is_empty() {
            return;
        }
        let RendererInner {
            device,
            shader_compiler,
            pipeline_cache,
            vertex_attribute_divisor_supported,
            ..
        } = &mut *inner;
        let mut queued = 0usize;
        for spec in &specs {
            if pipeline_cache.get(&spec.key).is_some() {
                continue;
            }
            let vs_label = format!("prewarm-vs key={:?}", spec.key);
            let vs_mod =
                match shader_compiler.compile_or_get_labeled(&spec.vs_spirv, device, &vs_label) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
            let fs_label = format!("prewarm-fs key={:?}", spec.key);
            let fs_mod =
                match shader_compiler.compile_or_get_labeled(&spec.fs_spirv, device, &fs_label) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
            let req = crate::pipeline::spec_to_request(
                spec,
                vs_mod,
                fs_mod,
                *vertex_attribute_divisor_supported,
            );
            pipeline_cache.queue_build(req);
            queued += 1;
        }
        log::info!(
            "prewarm: queued {} pipelines from {} cached specs",
            queued,
            specs.len()
        );
    }

    pub fn clear_target(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        rgba: [f32; 4],
    ) -> Result<(), String> {
        self.clear_target_with_format(
            nvmap_id,
            width,
            height,
            gpu_va,
            rgba,
            vk::Format::R8G8B8A8_UNORM,
        )
    }

    pub fn clear_target_with_format(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        rgba: [f32; 4],
        format: vk::Format,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let RendererInner {
            device,
            queue,
            rt_cache,
            clear_slots,
            clear_slot_index,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create_with_format(key, device, format)?;

        let clear_slot = acquire_clear_slot(device, clear_slots, clear_slot_index)?;
        reset_command_buffer(device, clear_slot.cmd)?;
        let cmd = clear_slot.cmd;

        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(utility): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: rgba },
        };
        let attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: img.view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachments: &attachment,
            p_depth_attachment: std::ptr::null(),
            p_stencil_attachment: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device.cmd_begin_rendering(cmd, &render_info);
            device.cmd_end_rendering(cmd);
        }
        img.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(utility): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, clear_slot.fence)?;
        clear_slot.in_flight = true;
        rt_cache.mark_cleared(key, true);
        Ok(())
    }

    pub fn clear_target_rect(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        rgba: [f32; 4],
        rect: [i32; 4],
    ) -> Result<(), String> {
        self.clear_target_rect_with_format(
            nvmap_id,
            width,
            height,
            gpu_va,
            rgba,
            rect,
            vk::Format::R8G8B8A8_UNORM,
        )
    }

    pub fn clear_target_rect_with_format(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        rgba: [f32; 4],
        rect: [i32; 4],
        format: vk::Format,
    ) -> Result<(), String> {
        let x = rect[0].max(0) as u32;
        let y = rect[1].max(0) as u32;
        let w = rect[2].max(0) as u32;
        let h = rect[3].max(0) as u32;
        if w == 0 || h == 0 || x >= width || y >= height {
            return Ok(());
        }
        let w = w.min(width - x);
        let h = h.min(height - y);
        if x == 0 && y == 0 && w == width && h == height {
            return self.clear_target_with_format(nvmap_id, width, height, gpu_va, rgba, format);
        }
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let RendererInner {
            device,
            queue,
            rt_cache,
            clear_slots,
            clear_slot_index,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create_with_format(key, device, format)?;

        let clear_slot = acquire_clear_slot(device, clear_slots, clear_slot_index)?;
        reset_command_buffer(device, clear_slot.cmd)?;
        let cmd = clear_slot.cmd;

        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(rect clear): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: rgba },
        };
        let attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: img.view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachments: &attachment,
            p_depth_attachment: std::ptr::null(),
            p_stencil_attachment: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let clear_attachment = vk::ClearAttachment {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            color_attachment: 0,
            clear_value,
        };
        let clear_rect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D {
                    x: x as i32,
                    y: y as i32,
                },
                extent: vk::Extent2D {
                    width: w,
                    height: h,
                },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            device.cmd_begin_rendering(cmd, &render_info);
            device.cmd_clear_attachments(cmd, &[clear_attachment], &[clear_rect]);
            device.cmd_end_rendering(cmd);
        }
        img.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(rect clear): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, clear_slot.fence)?;
        clear_slot.in_flight = true;
        rt_cache.mark_cleared(key, false);
        Ok(())
    }

    pub fn upload_target_rgba(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        rgba8: &[u8],
    ) -> Result<(), String> {
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        if rgba8.len() < expected {
            return Err(format!(
                "upload_target_rgba short buffer: got {} need {}",
                rgba8.len(),
                expected
            ));
        }
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            mem_props,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let (image, old_layout) = {
            let img = rt_cache.get_or_create(key, device)?;
            (img.image, img.layout)
        };
        let stage = create_host_buffer(
            device,
            mem_props,
            &rgba8[..expected],
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        reset_command_buffer(device, utility_slot.cmd)?;
        let cmd = utility_slot.cmd;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(upload target): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            image,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                stage.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
        }
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(upload target): {:?}", e))?;
        }
        let result = submit_with_fence(device, *queue, cmd, utility_slot.fence)
            .and_then(|_| wait_fence(device, utility_slot.fence));
        unsafe {
            device.destroy_buffer(stage.buffer, None);
            device.free_memory(stage.memory, None);
        }
        result?;
        rt_cache.set_color_layout(key, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        rt_cache.mark_drawn(key);
        Ok(())
    }

    pub fn resolve_rt_copy(
        &self,
        src_nvmap: u32,
        src_width: u32,
        src_height: u32,
        src_va: u64,
        dst_nvmap: u32,
        dst_width: u32,
        dst_height: u32,
        dst_va: u64,
        src_rect: [i32; 4],
        dst_rect: [i32; 4],
    ) -> Result<bool, String> {
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let Some((src_key, src_image, _src_view, src_layout)) =
            rt_cache.find_color(RtKey::new(src_nvmap, src_width, src_height, src_va))
        else {
            return Ok(false);
        };
        let dst_key = RtKey::new(dst_nvmap, dst_width, dst_height, dst_va);
        let (dst_image, dst_layout) = {
            let img = rt_cache.get_or_create(dst_key, device)?;
            (img.image, img.layout)
        };
        if src_image == dst_image {
            return Ok(false);
        }
        let clampi = |v: i32, hi: u32| v.clamp(0, hi as i32);
        let s0 = [
            clampi(src_rect[0], src_key.width),
            clampi(src_rect[1], src_key.height),
        ];
        let s1 = [
            clampi(src_rect[2], src_key.width),
            clampi(src_rect[3], src_key.height),
        ];
        let d0 = [
            clampi(dst_rect[0], dst_key.width),
            clampi(dst_rect[1], dst_key.height),
        ];
        let d1 = [
            clampi(dst_rect[2], dst_key.width),
            clampi(dst_rect[3], dst_key.height),
        ];
        if s1[0] <= s0[0] || s1[1] <= s0[1] || d1[0] <= d0[0] || d1[1] <= d0[1] {
            return Ok(false);
        }
        reset_command_buffer(device, utility_slot.cmd)?;
        let cmd = utility_slot.cmd;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(rt resolve): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            src_image,
            src_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        transition_image(
            device,
            cmd,
            dst_image,
            dst_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D {
                    x: s0[0],
                    y: s0[1],
                    z: 0,
                },
                vk::Offset3D {
                    x: s1[0],
                    y: s1[1],
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D {
                    x: d0[0],
                    y: d0[1],
                    z: 0,
                },
                vk::Offset3D {
                    x: d1[0],
                    y: d1[1],
                    z: 1,
                },
            ],
        };
        unsafe {
            device.cmd_blit_image(
                cmd,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }
        transition_image(
            device,
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        transition_image(
            device,
            cmd,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(rt resolve): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, utility_slot.fence)
            .and_then(|_| wait_fence(device, utility_slot.fence))?;
        rt_cache.set_color_layout(src_key, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        rt_cache.set_color_layout(dst_key, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        rt_cache.mark_drawn(dst_key);
        Ok(true)
    }

    pub fn clear_depth(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        depth: f32,
    ) -> Result<(), String> {
        self.clear_depth_stencil(
            RtKey::new(nvmap_id, width, height, gpu_va),
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageAspectFlags::DEPTH,
            depth,
            0,
        )
    }

    pub fn clear_depth_stencil(
        &self,
        key: RtKey,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        clear_aspects: vk::ImageAspectFlags,
        depth: f32,
        stencil: u32,
    ) -> Result<(), String> {
        if clear_aspects.is_empty() || !aspects.contains(clear_aspects) {
            return Err(format!(
                "invalid depth/stencil clear aspects {:?} for image aspects {:?}",
                clear_aspects, aspects
            ));
        }
        let mut inner = self.inner.lock();
        settle_all_pending_computes(&mut inner);
        let RendererInner {
            device,
            queue,
            rt_cache,
            clear_slots,
            clear_slot_index,
            ..
        } = &mut *inner;
        let (img, _) = rt_cache.get_or_create_depth(key, device, format, aspects)?;

        let clear_slot = acquire_clear_slot(device, clear_slots, clear_slot_index)?;
        reset_command_buffer(device, clear_slot.cmd)?;
        let cmd = clear_slot.cmd;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(depth clear): {:?}", e))?;
        }
        transition_image_aspect(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            aspects,
        );
        let clear = vk::ClearDepthStencilValue { depth, stencil };
        let range = vk::ImageSubresourceRange {
            aspect_mask: clear_aspects,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            device.cmd_clear_depth_stencil_image(
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
        }
        img.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(depth clear): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, clear_slot.fence)?;
        clear_slot.in_flight = true;
        rt_cache.invalidate_depth_pass(key);
        rt_cache.mark_depth_written(key);
        Ok(())
    }

    pub fn rt_key_for_nvmap(&self, nvmap_id: u32, width: u32, height: u32) -> Option<(u32, u32)> {
        let inner = self.inner.lock();
        inner
            .rt_cache
            .find_color(RtKey::request(nvmap_id, width, height))
            .map(|(k, _, _, _)| (k.width, k.height))
    }

    pub fn rt_key_at_va(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(u32, u32)> {
        let inner = self.inner.lock();
        inner
            .rt_cache
            .find_color(RtKey::new(nvmap_id, width, height, gpu_va))
            .map(|(k, _, _, _)| (k.width, k.height))
    }

    pub fn render_target_at_va(&self, nvmap_id: u32, gpu_va: u64) -> Option<(RtKey, u64)> {
        let inner = self.inner.lock();
        let key = inner.rt_cache.find_color_key_at_va(nvmap_id, gpu_va)?;
        let stamp = inner.rt_cache.drawn_stamp(key)?;
        Some((key, stamp))
    }

    pub fn render_target_stamp(&self, key: RtKey) -> Option<u64> {
        self.inner.lock().rt_cache.drawn_stamp(key)
    }

    pub fn readback_target(&self, nvmap_id: u32, width: u32, height: u32) -> Option<Vec<u8>> {
        self.readback_target_at(nvmap_id, width, height, 0)
    }

    pub fn readback_target_at(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            cmd_pool,
            queue,
            rt_cache,
            mem_props,
            pending_readbacks,
            readback_slots,
            ..
        } = &mut *inner;
        let requested_key = RtKey::new(nvmap_id, width, height, gpu_va);
        let key = rt_cache.resolve_present_key(requested_key, false)?;
        trace_present_key(rt_cache, requested_key, key);
        for (_, mut pending) in pending_readbacks.drain() {
            while let Some(prev) = pending.pop_front() {
                if let Some(slot) = readback_slots.get_mut(prev.slot) {
                    unsafe {
                        let _ = device.wait_for_fences(&[slot.fence], true, 2_000_000_000);
                    }
                    slot.in_flight = false;
                }
            }
        }

        let total = (width as u64) * 4 * (height as u64);
        let stage = create_staging_owned(device, mem_props, total).ok()?;
        let cleanup = |device: &ash::Device,
                       cmd_pool: vk::CommandPool,
                       fence: Option<vk::Fence>,
                       cmd: Option<vk::CommandBuffer>,
                       stage: &StagingBuffer| unsafe {
            if let Some(c) = cmd {
                device.free_command_buffers(cmd_pool, &[c]);
            }
            if let Some(f) = fence {
                device.destroy_fence(f, None);
            }
            device.destroy_buffer(stage.buffer, None);
            device.free_memory(stage.memory, None);
        };
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::empty(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(_) => {
                cleanup(device, *cmd_pool, None, None, &stage);
                return None;
            }
        };
        let cmd = match alloc_one_time_cmd(device, *cmd_pool) {
            Ok(c) => c,
            Err(_) => {
                cleanup(device, *cmd_pool, Some(fence), None, &stage);
                return None;
            }
        };
        if begin_one_time(device, cmd).is_err() {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let img = match rt_cache.get_existing(key) {
            Some(i) => i,
            None => {
                unsafe {
                    let _ = device.end_command_buffer(cmd);
                }
                cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        };
        transition_image(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage.buffer,
                &[copy],
            );
        }
        img.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        if end_one_time(device, cmd).is_err()
            || submit_with_fence(device, *queue, cmd, fence).is_err()
        {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let mut raw = vec![0u8; total as usize];
        if unsafe { device.wait_for_fences(&[fence], true, 2_000_000_000) }.is_err() {
            log::warn!("readback_target_at fence wait failed/timed out");
            unsafe {
                let _ = device.wait_for_fences(&[fence], true, 8_000_000_000);
            }
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        unsafe {
            if let Ok(ptr) =
                device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
            {
                std::ptr::copy_nonoverlapping(ptr as *const u8, raw.as_mut_ptr(), total as usize);
                device.unmap_memory(stage.memory);
            }
        }
        let out = readback_to_rgba8(&raw, img.format, width, height);
        cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
        Some(out)
    }

    pub fn readback_target_raw(
        &self,
        nvmap_id: u32,
        gpu_va: u64,
    ) -> Option<(u32, u32, usize, Vec<u8>)> {
        let key = {
            self.inner
                .lock()
                .rt_cache
                .find_color_key_at_va(nvmap_id, gpu_va)
        }?;
        self.readback_key_raw(key)
    }

    pub fn readback_target_raw_key(
        &self,
        key: RtKey,
    ) -> Option<(u32, u32, usize, Vec<u8>)> {
        self.readback_key_raw(key)
    }

    pub fn readback_target_raw_content(
        &self,
        nvmap_id: u32,
        gpu_va: u64,
        want_bpp: usize,
    ) -> Option<(u32, u32, usize, Vec<u8>)> {
        let key = {
            let inner = self.inner.lock();
            content_readback_key(&inner.rt_cache, nvmap_id, gpu_va, want_bpp)
        }?;
        self.readback_key_raw(key)
    }

    fn readback_key_raw(&self, key: RtKey) -> Option<(u32, u32, usize, Vec<u8>)> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            cmd_pool,
            queue,
            rt_cache,
            mem_props,
            ..
        } = &mut *inner;
        let format = rt_cache.get_existing(key)?.format;
        let bpp = readback_format_bpp(format);
        let total = (key.width as u64) * (key.height as u64) * bpp as u64;
        let stage = create_staging_owned(device, mem_props, total).ok()?;
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::empty(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let cleanup = |device: &ash::Device,
                       cmd_pool: vk::CommandPool,
                       fence: Option<vk::Fence>,
                       cmd: Option<vk::CommandBuffer>,
                       stage: &StagingBuffer| unsafe {
            if let Some(c) = cmd {
                device.free_command_buffers(cmd_pool, &[c]);
            }
            if let Some(f) = fence {
                device.destroy_fence(f, None);
            }
            device.destroy_buffer(stage.buffer, None);
            device.free_memory(stage.memory, None);
        };
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(_) => {
                cleanup(device, *cmd_pool, None, None, &stage);
                return None;
            }
        };
        let cmd = match alloc_one_time_cmd(device, *cmd_pool) {
            Ok(c) => c,
            Err(_) => {
                cleanup(device, *cmd_pool, Some(fence), None, &stage);
                return None;
            }
        };
        if begin_one_time(device, cmd).is_err() {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let img = match rt_cache.get_existing(key) {
            Some(i) => i,
            None => {
                unsafe {
                    let _ = device.end_command_buffer(cmd);
                }
                cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        };
        let prev_layout = img.layout;
        transition_image(
            device,
            cmd,
            img.image,
            prev_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage.buffer,
                &[copy],
            );
        }
        if prev_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            && prev_layout != vk::ImageLayout::UNDEFINED
        {
            transition_image(
                device,
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                prev_layout,
            );
        } else {
            img.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
        if end_one_time(device, cmd).is_err()
            || submit_with_fence(device, *queue, cmd, fence).is_err()
        {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let mut raw = vec![0u8; total as usize];
        let waited = unsafe { device.wait_for_fences(&[fence], true, 1_000_000_000) };
        match waited {
            Ok(()) => unsafe {
                note_queue_drained();
                let Ok(ptr) =
                    device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
                else {
                    cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                    return None;
                };
                std::ptr::copy_nonoverlapping(ptr as *const u8, raw.as_mut_ptr(), total as usize);
                device.unmap_memory(stage.memory);
            },
            Err(e) => {
                log::warn!("readback_target_raw fence wait failed: {:?}", e);
                unsafe {
                    let _ = device.wait_for_fences(&[fence], true, 5_000_000_000);
                }
                cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        }
        cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
        Some((key.width, key.height, bpp, raw))
    }

    pub fn readback_depth_target_raw(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
    ) -> Option<(u32, u32, usize, Vec<u8>)> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            cmd_pool,
            queue,
            rt_cache,
            mem_props,
            ..
        } = &mut *inner;
        let requested = RtKey::new(nvmap_id, width, height, gpu_va);
        let (key, _, _, layout, format, aspects) = rt_cache
            .find_depth(requested)
            .or_else(|| rt_cache.find_d24_depth_covering(requested))?;
        if format != vk::Format::D24_UNORM_S8_UINT
            || !aspects.contains(vk::ImageAspectFlags::DEPTH)
            || layout == vk::ImageLayout::UNDEFINED
            || width == 0
            || height == 0
        {
            return None;
        }
        let bpp = 4usize;
        let total = (width as u64) * (height as u64) * bpp as u64;
        let stage = create_staging_owned(device, mem_props, total).ok()?;
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::empty(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let cleanup = |device: &ash::Device,
                       cmd_pool: vk::CommandPool,
                       fence: Option<vk::Fence>,
                       cmd: Option<vk::CommandBuffer>,
                       stage: &StagingBuffer| unsafe {
            if let Some(c) = cmd {
                device.free_command_buffers(cmd_pool, &[c]);
            }
            if let Some(f) = fence {
                device.destroy_fence(f, None);
            }
            device.destroy_buffer(stage.buffer, None);
            device.free_memory(stage.memory, None);
        };
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(_) => {
                cleanup(device, *cmd_pool, None, None, &stage);
                return None;
            }
        };
        let cmd = match alloc_one_time_cmd(device, *cmd_pool) {
            Ok(c) => c,
            Err(_) => {
                cleanup(device, *cmd_pool, Some(fence), None, &stage);
                return None;
            }
        };
        if begin_one_time(device, cmd).is_err() {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let img = match rt_cache.get_existing_depth(key) {
            Some(i) => i,
            None => {
                unsafe {
                    let _ = device.end_command_buffer(cmd);
                }
                cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        };
        let prev_layout = img.layout;
        transition_image_aspect(
            device,
            cmd,
            img.image,
            prev_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            aspects,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage.buffer,
                &[copy],
            );
        }
        if prev_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            && prev_layout != vk::ImageLayout::UNDEFINED
        {
            transition_image_aspect(
                device,
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                prev_layout,
                aspects,
            );
        } else {
            img.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
        if end_one_time(device, cmd).is_err()
            || submit_with_fence(device, *queue, cmd, fence).is_err()
        {
            cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let mut raw = vec![0u8; total as usize];
        let waited = unsafe { device.wait_for_fences(&[fence], true, 1_000_000_000) };
        match waited {
            Ok(()) => unsafe {
                if let Ok(ptr) =
                    device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
                {
                    std::ptr::copy_nonoverlapping(
                        ptr as *const u8,
                        raw.as_mut_ptr(),
                        total as usize,
                    );
                    device.unmap_memory(stage.memory);
                }
            },
            Err(e) => {
                log::warn!("readback_depth_target_raw fence wait failed: {:?}", e);
                unsafe {
                    let _ = device.wait_for_fences(&[fence], true, 5_000_000_000);
                }
                cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        }
        cleanup(device, *cmd_pool, Some(fence), Some(cmd), &stage);
        Some((width, height, bpp, raw))
    }

    pub fn readback_target_pipelined(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        cpu_addr: u64,
        copy_rect: Option<[u32; 4]>,
    ) -> Option<(u32, u32, Vec<u8>, Option<bool>)> {
        let pp_t0 = std::time::Instant::now();
        let raw_result = self
            .readback_target_pipelined_raw(nvmap_id, width, height, gpu_va, cpu_addr, copy_rect);
        let pp_raw = pp_t0.elapsed();
        let (w, h, raw, format, flip_y) = raw_result?;
        let pp_t1 = std::time::Instant::now();
        let out = if legacy_present_enabled() {
            readback_to_rgba8(&raw, format, w, h)
        } else {
            let vflip = flip_y.unwrap_or(!(w == 1600 && h == 900));
            readout_present_rgba8(raw, format, w, h, vflip)
        };
        pprof_record(w, h, pp_raw, pp_t1.elapsed());
        Some((w, h, out, flip_y))
    }

    fn readback_target_pipelined_raw(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        cpu_addr: u64,
        copy_rect: Option<[u32; 4]>,
    ) -> Option<(u32, u32, Vec<u8>, vk::Format, Option<bool>)> {
        let pr_t0 = std::time::Instant::now();
        let mut inner = self.inner.lock();
        pprof_raw_lock(pr_t0.elapsed());
        let RendererInner {
            device,
            cmd_pool,
            queue,
            rt_cache,
            mem_props,
            pending_readbacks,
            readback_slots,
            ..
        } = &mut *inner;
        let requested_key = if gpu_va != 0 {
            RtKey::with_cpu(nvmap_id, width, height, gpu_va, cpu_addr)
        } else if cpu_addr != 0 {
            RtKey::with_cpu(nvmap_id, width, height, 0, cpu_addr)
        } else {
            RtKey::request(nvmap_id, width, height)
        };
        let key = match rt_cache.resolve_present_key(requested_key, true) {
            Some(k) => k,
            None => rt_cache.present_fallback_key(requested_key)?,
        };
        let resolved_flip_y = rt_cache.present_flip_y(key);
        trace_present_key(rt_cache, requested_key, key);
        if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static FCT: AtomicU64 = AtomicU64::new(0);
            let n = FCT.fetch_add(1, Ordering::Relaxed);
            if n % 60 == 0 {
                log::warn!(
                    "[present-flip #{}] resolved={} flip_y={:?}",
                    n,
                    key.label(),
                    resolved_flip_y
                );
            }
        }
        trace_rt_stats(
            device,
            *cmd_pool,
            *queue,
            rt_cache,
            mem_props,
            requested_key,
            key,
        );
        rt_cache.reset_frame_draws();

        let mut ready_frame = None;
        let mut latest_ready = None;
        let mut pending_for_key = pending_readbacks.remove(&key).unwrap_or_default();
        let mut keep_pending = VecDeque::with_capacity(pending_for_key.len());
        while let Some(prev) = pending_for_key.pop_front() {
            let ready = readback_slots
                .get(prev.slot)
                .map(|slot| unsafe { device.get_fence_status(slot.fence).unwrap_or(true) })
                .unwrap_or(true);
            if ready {
                if let Some(old) = latest_ready.replace(prev) {
                    if let Some(slot) = readback_slots.get_mut(old.slot) {
                        slot.in_flight = false;
                    }
                }
            } else {
                keep_pending.push_back(prev);
            }
        }
        if let Some(prev) = latest_ready {
            let total = (prev.width as u64) * 4 * (prev.height as u64);
            let mut raw = vec![0u8; total as usize];
            if let Some(slot) = readback_slots.get_mut(prev.slot) {
                if let Some(stage) = slot.stage.as_ref() {
                    unsafe {
                        if let Ok(ptr) = device.map_memory(
                            stage.memory,
                            0,
                            stage.size,
                            vk::MemoryMapFlags::empty(),
                        ) {
                            std::ptr::copy_nonoverlapping(
                                ptr as *const u8,
                                raw.as_mut_ptr(),
                                total as usize,
                            );
                            device.unmap_memory(stage.memory);
                        }
                    }
                }
                slot.in_flight = false;
            }
            ready_frame = Some((prev.width, prev.height, raw, prev.format, prev.flip_y));
        }
        let Some(slot_idx) = readback_slots.iter().position(|slot| !slot.in_flight) else {
            if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                use std::sync::atomic::{AtomicU64, Ordering};
                static CT: AtomicU64 = AtomicU64::new(0);
                let n = CT.fetch_add(1, Ordering::Relaxed);
                if n % 120 == 0 {
                    let statuses: Vec<String> = readback_slots
                        .iter()
                        .map(|s| match unsafe { device.get_fence_status(s.fence) } {
                            Ok(true) => "sig".to_string(),
                            Ok(false) => "unsig".to_string(),
                            Err(e) => format!("err:{:?}", e),
                        })
                        .collect();
                    log::warn!(
                        "[readback-noslot #{}] all 4 in_flight, fences=[{}] key={}",
                        n,
                        statuses.join(","),
                        key.label()
                    );
                }
            }
            pending_readbacks.insert(key, keep_pending);
            return ready_frame;
        };

        let copy_rect = if key.width != width || key.height != height {
            None
        } else {
            copy_rect
        };
        let (copy_x, copy_y, copy_w, copy_h) = copy_rect
            .map(|r| (r[0], r[1], r[2], r[3]))
            .unwrap_or((0, 0, key.width, key.height));
        let copy_w = copy_w.min(key.width.saturating_sub(copy_x));
        let copy_h = copy_h.min(key.height.saturating_sub(copy_y));
        let total = (copy_w as u64) * 4 * (copy_h as u64);
        {
            let slot = &mut readback_slots[slot_idx];
            let needs_stage = slot
                .stage
                .as_ref()
                .map(|stage| stage.size < total)
                .unwrap_or(true);
            if needs_stage {
                if let Some(old) = slot.stage.take() {
                    unsafe {
                        device.destroy_buffer(old.buffer, None);
                        device.free_memory(old.memory, None);
                    }
                }
                match create_staging_owned(device, mem_props, total) {
                    Ok(stage) => {
                        slot.stage = Some(stage);
                    }
                    Err(_) => {
                        if !keep_pending.is_empty() {
                            pending_readbacks.insert(key, keep_pending);
                        }
                        return ready_frame;
                    }
                }
            }
        }
        let slot = &mut readback_slots[slot_idx];
        if reset_command_buffer(device, slot.cmd).is_err() {
            if !keep_pending.is_empty() {
                pending_readbacks.insert(key, keep_pending);
            }
            return ready_frame;
        }
        if begin_one_time(device, slot.cmd).is_err() {
            if !keep_pending.is_empty() {
                pending_readbacks.insert(key, keep_pending);
            }
            return ready_frame;
        }
        let cmd = slot.cmd;
        let fence = slot.fence;
        let stage_buffer = match slot.stage.as_ref() {
            Some(stage) => stage.buffer,
            None => {
                if !keep_pending.is_empty() {
                    pending_readbacks.insert(key, keep_pending);
                }
                return ready_frame;
            }
        };
        let img = match rt_cache.get_existing(key) {
            Some(i) => i,
            None => {
                unsafe {
                    let _ = device.end_command_buffer(cmd);
                }
                if !keep_pending.is_empty() {
                    pending_readbacks.insert(key, keep_pending);
                }
                return ready_frame;
            }
        };
        transition_image(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: copy_x as i32,
                y: copy_y as i32,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width: copy_w,
                height: copy_h,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                img.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                stage_buffer,
                &[copy],
            );
        }
        img.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        let end_res = end_one_time(device, cmd);
        let sub_res = if end_res.is_ok() {
            submit_with_fence(device, *queue, cmd, fence)
        } else {
            Ok(())
        };
        if end_res.is_err() || sub_res.is_err() {
            if std::env::var_os("NEXIUM_PRESENT_KEYS").is_some() {
                use std::sync::atomic::{AtomicU64, Ordering};
                static CT: AtomicU64 = AtomicU64::new(0);
                let n = CT.fetch_add(1, Ordering::Relaxed);
                if n % 120 == 0 {
                    log::warn!(
                        "[readback-submitfail #{}] end={:?} submit={:?} key={}",
                        n,
                        end_res,
                        sub_res,
                        key.label()
                    );
                }
            }
            if !keep_pending.is_empty() {
                pending_readbacks.insert(key, keep_pending);
            }
            return ready_frame;
        }
        readback_slots[slot_idx].in_flight = true;
        keep_pending.push_back(PendingReadback {
            slot: slot_idx,
            width: copy_w,
            height: copy_h,
            format: img.format,
            flip_y: resolved_flip_y,
        });
        pending_readbacks.insert(key, keep_pending);
        ready_frame
    }

    pub fn compile_pipeline(
        &self,
        vs_spirv: &[u32],
        fs_spirv: &[u32],
        vs_hash: u64,
        fs_hash: u64,
        vs_cbuf_mask: u32,
        fs_cbuf_mask: u32,
        layout: &crate::draw::VertexLayout,
        topology: vk::PrimitiveTopology,
        color_formats: &[vk::Format],
        blend: crate::draw::BlendState,
        cull_test_enable: bool,
        cull_face: u32,
        front_face: u32,
        depth_clamp_enabled: bool,
        poly_offset_enable: bool,
        poly_offset_units: f32,
        poly_offset_factor: f32,
        depth: crate::draw::DepthState,
        depth_format: vk::Format,
        depth_aspects: vk::ImageAspectFlags,
        stencil: crate::draw::StencilState,
        _vertex_count: u32,
    ) -> Result<Option<vk::Pipeline>, String> {
        if crate::pipeline::known_driver_hostile_pipeline(vs_hash, fs_hash) {
            use std::sync::atomic::{AtomicBool, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "rejected driver-hostile shader pair vs_hash={:016x} fs_hash={:016x}",
                    vs_hash,
                    fs_hash
                );
            }
            return Ok(None);
        }
        let mut inner = self.inner.lock();
        let mut blend_signature: u64 = 0xcbf29ce484222325;
        for att in &blend.attachments {
            let packed = (att.enabled as u64)
                | ((att.src_factor.as_raw() as u64 & 0xFF) << 8)
                | ((att.dst_factor.as_raw() as u64 & 0xFF) << 16)
                | ((att.op.as_raw() as u64 & 0xFF) << 24)
                | ((att.src_alpha_factor.as_raw() as u64 & 0xFF) << 32)
                | ((att.dst_alpha_factor.as_raw() as u64 & 0xFF) << 40)
                | ((att.alpha_op.as_raw() as u64 & 0xFF) << 48)
                | ((att.color_write_mask.as_raw() as u64 & 0xF) << 56);
            blend_signature ^= packed;
            blend_signature = blend_signature.wrapping_mul(0x100000001b3);
        }
        let raster_state_packed: u32 =
            (cull_test_enable as u32) | ((cull_face & 0xFF) << 8) | ((front_face & 0xFF) << 16);
        let has_depth = depth_format != vk::Format::UNDEFINED && !depth_aspects.is_empty();
        let depth_state_packed: u32 = (depth.test_enabled as u32)
            | ((depth.write_enabled as u32) << 1)
            | ((has_depth as u32) << 2)
            | ((depth.compare_op.as_raw() as u32 & 0xFF) << 8);
        let stencil_face_key = |face: crate::draw::StencilFaceState| {
            [
                face.fail_op.as_raw() as u32,
                face.pass_op.as_raw() as u32,
                face.depth_fail_op.as_raw() as u32,
                face.compare_op.as_raw() as u32,
                face.compare_mask,
                face.write_mask,
                face.reference,
            ]
        };
        let depth_clamp_enabled = depth_clamp_enabled && inner.depth_clamp_supported;
        let poly_offset_packed: u64 = (poly_offset_enable as u64)
            | ((poly_offset_units.to_bits() as u64) << 1)
            | ((poly_offset_factor.to_bits() as u64) << 33);
        let color_formats = crate::pipeline::normalized_color_formats(color_formats);
        let (color_format, color_format_key, color_attachment_count) =
            crate::pipeline::color_format_key(&color_formats);
        let key = crate::pipeline::PipelineKey {
            vs_hash,
            fs_hash,
            topology: topology.as_raw() as u32,
            color_format,
            color_formats: color_format_key,
            color_attachment_count,
            vs_cbuf_mask,
            fs_cbuf_mask,
            vertex_layout_hash: layout.hash(),
            blend_signature,
            raster_state_packed,
            depth_state_packed,
            depth_format: depth_format.as_raw(),
            depth_aspects: depth_aspects.as_raw(),
            stencil_enabled: stencil.enabled,
            stencil_front: stencil_face_key(stencil.front),
            stencil_back: stencil_face_key(stencil.back),
            depth_clamp_enabled,
            poly_offset_packed,
            color_write_mask: blend.color_write_mask.as_raw(),
        };
        let depth_clip_control_enabled = inner.depth_clip_control_enabled;
        let vertex_attribute_divisor_supported = inner.vertex_attribute_divisor_supported;
        let RendererInner {
            device,
            shader_compiler,
            pipeline_cache,
            ..
        } = &mut *inner;

        pipeline_cache.drain_completed(device);
        if let Some(p) = pipeline_cache.get(&key) {
            return Ok(Some(p));
        }

        let vs_label = format!("runtime-vs hash={:016x}", vs_hash);
        let fs_label = format!("runtime-fs hash={:016x}", fs_hash);
        let vs_mod = shader_compiler.compile_or_get_labeled(vs_spirv, device, &vs_label)?;
        let fs_mod = shader_compiler.compile_or_get_labeled(fs_spirv, device, &fs_label)?;

        let bindings: Vec<vk::VertexInputBindingDescription> = layout
            .bindings
            .iter()
            .map(|b| vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                input_rate: if b.divisor != 0 {
                    vk::VertexInputRate::INSTANCE
                } else {
                    vk::VertexInputRate::VERTEX
                },
            })
            .collect();
        let binding_divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR> = layout
            .bindings
            .iter()
            .filter_map(|b| {
                (vertex_attribute_divisor_supported && b.divisor > 1).then_some(
                    vk::VertexInputBindingDivisorDescriptionKHR {
                        binding: b.binding,
                        divisor: b.divisor,
                    },
                )
            })
            .collect();
        let attrs: Vec<vk::VertexInputAttributeDescription> = layout
            .attrs
            .iter()
            .map(|a| vk::VertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: a.format,
                offset: a.offset,
            })
            .collect();

        let req = crate::pipeline::PipelineBuildRequest {
            key,
            vs_mod,
            fs_mod,
            bindings,
            binding_divisors,
            attrs,
            topology,
            color_formats: color_formats.clone(),
            depth_format,
            has_depth,
            depth_aspects,
            blend,
            depth,
            stencil,
            depth_clamp_enabled,
            cull_test_enable,
            cull_face,
            front_face,
            poly_offset_enable,
            poly_offset_units,
            poly_offset_factor,
            depth_clip_control_enabled,
        };

        pipeline_cache.register_spec(crate::pipeline::PipelineSpec {
            key,
            vs_spirv: vs_spirv.to_vec(),
            fs_spirv: fs_spirv.to_vec(),
            bindings: layout
                .bindings
                .iter()
                .map(|b| (b.binding, b.stride, b.divisor))
                .collect(),
            attrs: layout
                .attrs
                .iter()
                .map(|a| (a.location, a.binding, a.format.as_raw(), a.offset))
                .collect(),
            topology: topology.as_raw(),
            color_format: color_format as i32,
            color_formats: color_formats.iter().map(|f| f.as_raw()).collect(),
            color_attachment_count,
            depth_format: depth_format.as_raw(),
            has_depth,
            depth_aspects: depth_aspects.as_raw(),
            blend: (
                blend.enabled,
                blend.src_factor.as_raw(),
                blend.dst_factor.as_raw(),
                blend.op.as_raw(),
                blend.src_alpha_factor.as_raw(),
                blend.dst_alpha_factor.as_raw(),
                blend.alpha_op.as_raw(),
            ),
            blend_attachments: blend
                .attachments
                .iter()
                .map(|att| {
                    (
                        att.enabled,
                        att.src_factor.as_raw(),
                        att.dst_factor.as_raw(),
                        att.op.as_raw(),
                        att.src_alpha_factor.as_raw(),
                        att.dst_alpha_factor.as_raw(),
                        att.alpha_op.as_raw(),
                        att.color_write_mask.as_raw(),
                    )
                })
                .collect(),
            color_write_mask: blend.color_write_mask.as_raw(),
            depth: (
                depth.test_enabled,
                depth.write_enabled,
                depth.compare_op.as_raw(),
            ),
            stencil_enabled: stencil.enabled,
            stencil_front: (
                stencil.front.fail_op.as_raw(),
                stencil.front.pass_op.as_raw(),
                stencil.front.depth_fail_op.as_raw(),
                stencil.front.compare_op.as_raw(),
                stencil.front.compare_mask,
                stencil.front.write_mask,
                stencil.front.reference,
            ),
            stencil_back: (
                stencil.back.fail_op.as_raw(),
                stencil.back.pass_op.as_raw(),
                stencil.back.depth_fail_op.as_raw(),
                stencil.back.compare_op.as_raw(),
                stencil.back.compare_mask,
                stencil.back.write_mask,
                stencil.back.reference,
            ),
            depth_clamp_enabled,
            cull_test_enable,
            cull_face,
            front_face,
            poly_offset_enable,
            poly_offset_units,
            poly_offset_factor,
            depth_clip_control_enabled,
        });

        if async_shaders_enabled() && has_depth {
            match pipeline_cache.try_async_skip(req) {
                None => return Ok(None),
                Some(req) => {
                    let pipeline = {
                        let _g = nexium_common::shader_progress::guard();
                        pipeline_cache.build(device, &req)?
                    };
                    pipeline_cache.insert(key, pipeline);
                    return Ok(Some(pipeline));
                }
            }
        }
        let pipeline = {
            let _g = nexium_common::shader_progress::guard();
            pipeline_cache.build(device, &req)?
        };
        pipeline_cache.insert(key, pipeline);
        Ok(Some(pipeline))
    }

    pub fn execute_draw<F>(
        &self,
        call: &crate::draw::Maxwell3dDrawCall,
        read_guest: F,
    ) -> Result<(), String>
    where
        F: Fn(u64, usize) -> Option<Vec<u8>>,
    {
        self.settle_pending_computes();
        let use_depth = call.depth_key.is_some();
        let depth_format = if use_depth {
            call.depth_format
        } else {
            vk::Format::UNDEFINED
        };
        let depth_aspects = if use_depth {
            call.depth_aspects
        } else {
            vk::ImageAspectFlags::empty()
        };
        if !use_depth && !call_writes_any_color(call) {
            return Ok(());
        }
        let color_keys = active_color_keys_for_call(call);
        let color_formats = color_formats_for_call(call, color_keys.len());
        let pipeline = match self.compile_pipeline(
            &call.vs_spirv,
            &call.fs_spirv,
            call.vs_hash,
            call.fs_hash,
            call.vs_cbuf_mask,
            call.fs_cbuf_mask,
            &call.vertex_layout,
            call.state.topology,
            &color_formats,
            call.blend,
            call.cull_test_enable,
            call.cull_face,
            call.front_face,
            call.depth_clamp_enabled,
            call.poly_offset_enable,
            call.poly_offset_units,
            call.poly_offset_factor,
            call.depth,
            depth_format,
            depth_aspects,
            call.stencil,
            call.vertex_count,
        )? {
            Some(p) => p,
            None => return Ok(()),
        };

        let (vertex_bindings, draw_vertex_count) = prepare_vertex_bindings(call, &read_guest)?;

        let cbuf_data = if let Some(d) = &call.cbuf_data {
            if d.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize {
                d.clone()
            } else {
                empty_graphics_cbuf_data()
            }
        } else {
            empty_graphics_cbuf_data()
        };

        let (index_data, index_count, index_type) = match (&call.index_data, call.index_count) {
            (Some(d), Some(c)) if c > 0 && !d.is_empty() => (d.clone(), c, call.index_type),
            _ => (Vec::new(), 0u32, call.index_type),
        };

        log::debug!(
            "cbuf_data: addr={:#x} size={} all_zero={}",
            call.cbuf_addr,
            call.cbuf_size,
            cbuf_data.iter().all(|&b| b == 0),
        );

        let tex_pendings = collect_tex_pendings(call, &read_guest)?;

        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            queue,
            mem_props,
            cmd_pool,
            rt_cache,
            descriptor_layout,
            descriptor_pool,
            pipeline_cache,
            dummy_images,
            dummy_texel_buffers,
            default_sampler,
            sampler_cache,
            integer_sampler_cache,
            sampler_filter_minmax_supported,
            sampler_anisotropy_supported,
            tex_cache,
            texel_buffer_cache,
            rt_reinterpret_cache,
            frame_slots,
            frame_index,
            ubo_ring,
            min_storage_buffer_offset_alignment,
            max_storage_buffer_range,
            max_texel_buffer_elements,
            tele_last_emit_ns,
            tele_ring_wraps,
            tele_ring_waits,
            tele_in_flight_mask,
            ..
        } = &mut *inner;

        let cur_idx = *frame_index;
        let other_idx = (cur_idx + 1) % 2;
        let cbuf_alignment = *min_storage_buffer_offset_alignment;
        {
            let slot = &mut frame_slots[cur_idx];
            if slot.in_flight {
                wait_fence(device, slot.fence)?;
                *tele_ring_waits += 1;
                if !slot.retired_dsets.is_empty() {
                    unsafe {
                        let _ =
                            device.free_descriptor_sets(descriptor_pool.pool, &slot.retired_dsets);
                    }
                    slot.retired_dsets.clear();
                }
                destroy_descriptor_pools(device, &mut slot.retired_dset_pools);
                for (b, m) in slot.retired_buffers.drain(..) {
                    unsafe {
                        device.destroy_buffer(b, None);
                        device.free_memory(m, None);
                    }
                }
                for view in slot.retired_views.drain(..) {
                    unsafe {
                        device.destroy_image_view(view, None);
                    }
                }
                for t in slot.retired_textures.drain(..) {
                    unsafe {
                        device.destroy_image_view(t.view, None);
                        device.destroy_image(t.image, None);
                        device.free_memory(t.memory, None);
                    }
                }
                for buffer in slot.retired_texel_buffers.drain(..) {
                    destroy_texel_buffer(device, buffer.resource);
                }
                for t in slot.retired_rt_reinterprets.drain(..) {
                    unsafe {
                        device.destroy_image_view(t.view, None);
                        device.destroy_image(t.image, None);
                        device.free_memory(t.memory, None);
                    }
                }
                reset_command_buffer(device, slot.cmd)?;
                slot.in_flight = false;
                ubo_ring.head = ubo_ring.slot_head[cur_idx];
            }
        }

        let graphics_dummies = ensure_graphics_dummy_views(
            dummy_images,
            dummy_texel_buffers,
            device,
            *queue,
            *cmd_pool,
            mem_props,
        )?;
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let default_samp = default_sampler.unwrap();
        let tsc_entries = collect_tsc_entries(call, &read_guest);
        let rt_aliases: Vec<_> = (0..tex_pendings.len())
            .map(|slot| {
                let tic = tex_pendings
                    .get(slot)
                    .and_then(|pending| pending.as_ref().map(|(_, tic, _, _)| tic));
                rt_alias_for_slot(rt_cache, call, slot, call.rt_key, false, tic)
            })
            .map(|alias| alias.filter(|alias| alias.depth || !color_keys.contains(&alias.key)))
            .collect();
        let dummy_views_2d = (0..max_texture_descriptors())
            .map(|slot| graphics_dummies.image_2d_for_slot(call, slot))
            .collect::<Vec<_>>();
        let dummy_views_3d = vec![graphics_dummies.image_3d; max_texture_descriptors()];
        let dummy_views_cube = (0..max_texture_descriptors())
            .map(|slot| graphics_dummies.image_cube_for_slot(call, slot))
            .collect::<Vec<_>>();
        let dummy_views_cube_array = (0..max_texture_descriptors())
            .map(|slot| graphics_dummies.image_cube_array_for_slot(call, slot))
            .collect::<Vec<_>>();
        let mut bound_tex_views = (0..max_texture_descriptors())
            .map(|slot| {
                let family = texture_numeric_index(texture_numeric_type_for_slot(
                    &call.texture_numeric_manifest,
                    slot,
                ));
                dummy_views_2d[slot][family]
            })
            .collect::<Vec<_>>();
        let mut bound_tex_views_3d = (0..max_texture_descriptors())
            .map(|slot| {
                dummy_views_3d[slot][texture_numeric_index(texture_numeric_type_for_slot(
                    &call.texture_numeric_manifest,
                    slot,
                ))]
            })
            .collect::<Vec<_>>();
        let mut bound_tex_views_cube = (0..max_texture_descriptors())
            .map(|slot| {
                dummy_views_cube[slot][texture_numeric_index(texture_numeric_type_for_slot(
                    &call.texture_numeric_manifest,
                    slot,
                ))]
            })
            .collect::<Vec<_>>();
        let mut bound_tex_views_cube_array = (0..max_texture_descriptors())
            .map(|slot| {
                dummy_views_cube_array[slot][texture_numeric_index(
                    texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                )]
            })
            .collect::<Vec<_>>();
        let mut bound_texel_views = (0..max_texture_descriptors())
            .map(|slot| {
                graphics_dummies.texel_buffer[texture_numeric_index(texture_numeric_type_for_slot(
                    &call.texture_numeric_manifest,
                    slot,
                ))]
            })
            .collect::<Vec<_>>();
        let mut stencil_alias_bound = vec![false; max_texture_descriptors()];
        let mut pending_rt_reinterprets = Vec::new();
        for (slot, pending) in tex_pendings.iter().enumerate() {
            if let Some((_, tic, _, read_size)) = *pending {
                if tic.is_buffer() && slot < 32 && call.texel_buffer_mask & (1u32 << slot) != 0 {
                    let numeric_type =
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot);
                    match texel_buffer_view_for_tic(
                        device,
                        mem_props,
                        *max_texel_buffer_elements,
                        texel_buffer_cache,
                        &mut frame_slots[cur_idx].retired_texel_buffers,
                        &tic,
                        numeric_type,
                        read_size,
                        crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64),
                        &read_guest,
                    ) {
                        Ok(Some(view)) => {
                            bound_texel_views[slot] = view;
                            trace_graphics_texture_binding!(
                                call,
                                slot,
                                *pending,
                                None,
                                numeric_type,
                                GraphicsTextureBindOutcome::TexelBuffer,
                                "texel-buffer",
                                format!("{view:?}"),
                                texel_buffer_format(&tic, numeric_type)
                                    .map(|(format, _)| format!("{format:?}")),
                                None,
                            );
                            continue;
                        }
                        Ok(None) => {
                            trace_graphics_texture_binding!(
                                call,
                                slot,
                                *pending,
                                None,
                                numeric_type,
                                GraphicsTextureBindOutcome::Rejection,
                                "texel-buffer",
                                "-",
                                None,
                                Some("view creation produced no view".to_string()),
                            );
                            return Err(format!(
                                "texel buffer slot {} va={:#x} produced no view",
                                slot, tic.gpu_va
                            ));
                        }
                        Err(error) => {
                            trace_graphics_texture_binding!(
                                call,
                                slot,
                                *pending,
                                None,
                                numeric_type,
                                GraphicsTextureBindOutcome::Rejection,
                                "texel-buffer",
                                "-",
                                None,
                                Some(error.clone()),
                            );
                            return Err(format!(
                                "texel buffer slot {} va={:#x}: {}",
                                slot, tic.gpu_va, error
                            ));
                        }
                    }
                }
            }
            let direct_volume_rt = pending.map_or(false, |(key, _, _, _)| key.volume)
                && sampled_rt_key_for_slot(call, slot).is_some_and(|key| key.is_3d);
            let pending_special = pending
                .map_or(false, |(key, _, _, _)| texture_key_has_special_view(key))
                && !direct_volume_rt;
            if !pending_special {
                if let Some(alias) = rt_aliases.get(slot).copied().flatten().filter(|alias| {
                    let view_format = pending
                        .map(|(_, tic, _, _)| rt_alias_view_format(alias.key, tic, alias.format))
                        .unwrap_or(alias.format);
                    rt_alias_numeric_type_matches(
                        *alias,
                        pending.map(|(_, tic, _, _)| tic),
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                        view_format,
                    )
                }) {
                    let view_format = pending
                        .map(|(_, tic, _, _)| rt_alias_view_format(alias.key, tic, alias.format))
                        .unwrap_or(alias.format);
                    let reinterpret_supported =
                        pending.map_or(false, |(texture_key, tic, _, _)| {
                            rt_alias_reinterpret_supported(texture_key, tic, alias, view_format)
                        });
                    if !reinterpret_supported {
                        if let Some(alias_view) = rt_alias_sample_view(
                            device,
                            rt_cache,
                            &mut frame_slots[cur_idx],
                            alias,
                            pending.map(|(_, tic, _, _)| tic),
                            view_format,
                        ) {
                            if direct_volume_rt {
                                bound_tex_views_3d[slot] = alias_view;
                            } else {
                                bound_tex_views[slot] = alias_view;
                            }
                            stencil_alias_bound[slot] = pending
                                .map(|(_, tic, _, _)| {
                                    rt_alias_sample_aspect(alias, tic)
                                        == Some(vk::ImageAspectFlags::STENCIL)
                                })
                                .unwrap_or(false);
                            let selected_view = if direct_volume_rt {
                                bound_tex_views_3d[slot]
                            } else {
                                bound_tex_views[slot]
                            };
                            trace_graphics_texture_binding!(
                                call,
                                slot,
                                *pending,
                                Some(GraphicsTextureTraceResource::rt_alias(alias)),
                                texture_numeric_type_for_slot(
                                    &call.texture_numeric_manifest,
                                    slot,
                                ),
                                GraphicsTextureBindOutcome::RtAlias,
                                format!("rt-alias:{}", alias.key.label()),
                                format!("{selected_view:?}"),
                                Some(format!("{view_format:?}")),
                                None,
                            );
                            trace_vs_tex_bind_alias(
                                device,
                                *cmd_pool,
                                *queue,
                                rt_cache,
                                mem_props,
                                call,
                                slot,
                                alias,
                                view_format,
                                bound_tex_views[slot],
                            );
                            continue;
                        }
                    }
                    if let Some((texture_key, tic, _, _)) = *pending {
                        if let Some((reinterpret_view, reinterpret_copy)) =
                            prepare_rt_alias_reinterpret(
                                device,
                                mem_props,
                                rt_cache,
                                rt_reinterpret_cache,
                                texture_key,
                                tic,
                                alias,
                                view_format,
                                false,
                            )?
                        {
                            bound_tex_views[slot] = reinterpret_view;
                            if let Some(reinterpret_copy) = reinterpret_copy {
                                pending_rt_reinterprets.push(reinterpret_copy);
                            }
                            trace_graphics_texture_binding!(
                                call,
                                slot,
                                *pending,
                                Some(GraphicsTextureTraceResource::rt_alias(alias)),
                                texture_numeric_type_for_slot(
                                    &call.texture_numeric_manifest,
                                    slot,
                                ),
                                GraphicsTextureBindOutcome::RtAlias,
                                format!("rt-reinterpret:{}", alias.key.label()),
                                format!("{:?}", bound_tex_views[slot]),
                                Some(format!("{view_format:?}")),
                                None,
                            );
                            trace_vs_tex_bind_alias(
                                device,
                                *cmd_pool,
                                *queue,
                                rt_cache,
                                mem_props,
                                call,
                                slot,
                                alias,
                                view_format,
                                bound_tex_views[slot],
                            );
                            continue;
                        }
                    }
                }
            }
            let Some((key, tic, pitch_size, read_size)) = *pending else {
                trace_graphics_texture_binding!(
                    call,
                    slot,
                    None,
                    None,
                    texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    GraphicsTextureBindOutcome::Dummy,
                    "dummy-descriptor",
                    format!("{:?}", bound_tex_views[slot]),
                    None,
                    Some(
                        call.fs_tex_ids
                            .get(slot)
                            .map(|tex_id| vs_tex_dummy_reason(call, *tex_id, &read_guest))
                            .unwrap_or_else(|| "tex-slot-out-of-range".to_string()),
                    ),
                );
                trace_vs_tex_bind_dummy(call, slot, &read_guest);
                continue;
            };
            let cur_gen = crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64);
            let numeric_type = texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot);
            let numeric_family = texture_numeric_index(numeric_type);
            let mut binding_failure_reason = None;
            let identity_volume =
                key.volume && std::env::var_os("NEXIUM_VOLUME_IDENTITY").is_some();
            let volume_slices = if key.volume
                && !identity_volume
                && numeric_type == nexium_spirv::TextureNumericType::Float
            {
                let sampled_key = sampled_rt_key_for_slot(call, slot);
                find_volume_rt_slices(rt_cache, &tic, pitch_size, key.layers, sampled_key)
            } else {
                None
            };
            if let Some(slices) = volume_slices.as_ref() {
                trace_volume_rt_pixels(
                    device, *cmd_pool, *queue, rt_cache, mem_props, &tic, slices,
                );
            }
            let force_refresh_early = force_refresh_texture(tic.gpu_va);
            let cache_fresh = tex_gen_gating_enabled()
                && !force_refresh_early
                && volume_slices.is_none()
                && !identity_volume
                && tex_cache.get(&key).map_or(false, |t| t.gen == cur_gen);
            let mut refresh_verified = false;
            let sampled_fresh = !cache_fresh
                && !force_refresh_early
                && volume_slices.is_none()
                && !identity_volume
                && !key.volume
                && tex_cache.get(&key).map_or(false, |t| {
                    if t.gen != cur_gen {
                        return false;
                    }
                    if t.verified.elapsed() < TEX_VERIFY_PERIOD {
                        return true;
                    }
                    if hash_sampled_guest(&read_guest, tic.gpu_va, read_size) == Some(t.hash) {
                        refresh_verified = true;
                        true
                    } else {
                        false
                    }
                });
            if refresh_verified {
                if let Some(t) = tex_cache.get_mut(&key) {
                    t.verified = std::time::Instant::now();
                }
            }
            let raw = if cache_fresh || sampled_fresh {
                None
            } else {
                read_guest(tic.gpu_va, read_size)
            };
            if raw.is_some() || volume_slices.is_some() || identity_volume {
                let raw_hash = raw.as_ref().map(|raw| hash_sampled(raw));
                let mut tex_hash = raw_hash.unwrap_or_else(|| texture_seed_hash(&key));
                if let Some(slices) = volume_slices.as_ref() {
                    tex_hash = volume_rt_slice_hash(tex_hash, slices);
                }
                let force_refresh = force_refresh_texture(tic.gpu_va);
                let need_upload = force_refresh
                    || match tex_cache.get(&key) {
                        Some(t) => {
                            if key.volume && volume_slices.is_none() {
                                raw_hash.map_or(false, |raw_hash| {
                                    if t.hash != raw_hash {
                                        t.gen != cur_gen
                                    } else {
                                        t.gen != cur_gen || t.hash != tex_hash
                                    }
                                })
                            } else {
                                t.gen != cur_gen || t.hash != tex_hash
                            }
                        }
                        None => true,
                    };
                if need_upload {
                    let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH")
                        .map(|v| v == "1")
                        .unwrap_or(false);
                    let image_format = if identity_volume {
                        Ok(texture_image_format(
                            crate::texture::TicFormat::R8G8B8A8,
                            false,
                            false,
                            numeric_type,
                        ))
                    } else if let Some(slice) =
                        volume_slices.as_ref().and_then(|slices| slices.first())
                    {
                        Ok(slice.format)
                    } else {
                        texture_image_format_for_tic(&tic, numeric_type)
                    };
                    let image_format = match image_format {
                        Ok(format) => format,
                        Err(error) => {
                            log_graphics_texture_rejection(
                                call,
                                slot,
                                *pending,
                                numeric_type,
                                "format",
                                &error,
                            );
                            continue;
                        }
                    };
                    let upload = if volume_slices.is_some() {
                        Ok(TextureUploadData::base(Vec::new(), key.width, key.height))
                    } else if identity_volume {
                        Ok(TextureUploadData::base(
                            identity_volume_rgba8(key.width, key.height, key.layers),
                            key.width,
                            key.height,
                        ))
                    } else if let Some(raw) = raw.as_ref() {
                        texture_upload_data(
                            raw,
                            &tic,
                            key.layers,
                            pitch_size,
                            force_pitch,
                            image_format,
                        )
                    } else {
                        Ok(TextureUploadData::base(Vec::new(), key.width, key.height))
                    };
                    let upload = match upload {
                        Ok(upload) => upload,
                        Err(error) => {
                            log_graphics_texture_rejection(
                                call,
                                slot,
                                *pending,
                                numeric_type,
                                "decode",
                                &error,
                            );
                            continue;
                        }
                    };
                    let texels = &upload.bytes;
                    if std::env::var_os("NEXIUM_TEX_AVG").is_some() && texels.len() >= 4 {
                        let n = (texels.len() / 4).max(1) as u64;
                        let (mut ar, mut ag, mut ab) = (0u64, 0u64, 0u64);
                        for px in texels.chunks_exact(4) {
                            ar += px[0] as u64;
                            ag += px[1] as u64;
                            ab += px[2] as u64;
                        }
                        log::warn!(
                            "[tex-avg] va={:#x} {}x{} {:?} srgb={} avg=({},{},{})",
                            tic.gpu_va,
                            tic.width,
                            tic.height,
                            tic.format,
                            tic.is_srgb,
                            ar / n,
                            ag / n,
                            ab / n
                        );
                    }
                    log::debug!(
                        "TIC gpu_va={:#x} {}x{}x{} fmt={:?} bl={} bh={} bd={} src_bytes={} upload_bytes={} vkfmt={:?} (cache miss -> upload)",
                        tic.gpu_va, tic.width, tic.height, key.layers, tic.format,
                        tic.is_block_linear, tic.block_height_log2, tic.block_depth_log2, read_size, texels.len(), image_format
                    );
                    match upload_texture_oneshot(
                        device,
                        *queue,
                        *cmd_pool,
                        mem_props,
                        key.width,
                        key.height,
                        key.layers,
                        key.base_layer,
                        key.view_layers,
                        key.arrayed,
                        key.cube,
                        key.cube_array,
                        key.volume,
                        key.mip_levels,
                        key.base_mip,
                        key.view_mips,
                        texels,
                        &upload.copies,
                        volume_slices.as_deref(),
                        texture_view_swizzle(tic.format, numeric_type, tic.swizzle),
                        image_format,
                        tex_hash,
                        cur_gen,
                    ) {
                        Ok(tex) => {
                            if let Some(old) = tex_cache.insert(key, tex) {
                                frame_slots[cur_idx].retired_textures.push(old);
                            }
                        }
                        Err(error) => {
                            if !bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
                                log::warn!("texture upload failed: {}", error);
                            }
                            binding_failure_reason = Some(error);
                        }
                    }
                } else if raw.is_some() {
                    if let Some(t) = tex_cache.get_mut(&key) {
                        t.verified = std::time::Instant::now();
                    }
                }
            }
            if key.cube_array {
                bound_tex_views_cube_array[slot] = tex_cache
                    .get(&key)
                    .map(|t| t.view)
                    .unwrap_or(dummy_views_cube_array[slot][numeric_family]);
            } else if key.cube {
                bound_tex_views_cube[slot] = tex_cache
                    .get(&key)
                    .map(|t| t.view)
                    .unwrap_or(dummy_views_cube[slot][numeric_family]);
            } else if key.volume {
                bound_tex_views_3d[slot] = tex_cache
                    .get(&key)
                    .map(|t| t.view)
                    .unwrap_or(dummy_views_3d[slot][numeric_family]);
            } else {
                let fallback_view = dummy_views_2d[slot][numeric_family];
                bound_tex_views[slot] =
                    tex_cache.get(&key).map(|t| t.view).unwrap_or(fallback_view);
            }
            let cache_hit = tex_cache.get(&key).is_some();
            let selected_view = if key.cube_array {
                bound_tex_views_cube_array[slot]
            } else if key.cube {
                bound_tex_views_cube[slot]
            } else if key.volume {
                bound_tex_views_3d[slot]
            } else {
                bound_tex_views[slot]
            };
            let source = if key.cube_array {
                "texture-cube-array"
            } else if key.cube {
                "texture-cube"
            } else if key.volume {
                "texture-3d"
            } else if key.arrayed {
                "texture-2d-array"
            } else {
                "texture-2d"
            };
            let (outcome, reason) = if cache_hit {
                (GraphicsTextureBindOutcome::Success, None)
            } else if let Some(reason) = binding_failure_reason {
                (GraphicsTextureBindOutcome::Rejection, Some(reason))
            } else {
                (
                    GraphicsTextureBindOutcome::Dummy,
                    Some("resource data unavailable and no cached view".to_string()),
                )
            };
            trace_graphics_texture_binding!(
                call,
                slot,
                *pending,
                None,
                numeric_type,
                outcome,
                source,
                format!("{selected_view:?}"),
                texture_image_format_for_tic(&tic, numeric_type)
                    .ok()
                    .map(|format| format!("{format:?}")),
                reason,
            );
            trace_vs_tex_bind_texture(
                call,
                slot,
                key,
                tic,
                tex_cache.get(&key).is_some(),
                if key.cube_array {
                    bound_tex_views_cube_array[slot]
                } else if key.cube {
                    bound_tex_views_cube[slot]
                } else if key.volume {
                    bound_tex_views_3d[slot]
                } else {
                    bound_tex_views[slot]
                },
            );
        }
        if bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
            for binding in &call.texture_numeric_manifest {
                let slot = binding.descriptor_slot as usize;
                if slot < tex_pendings.len() || slot >= max_texture_descriptors() {
                    continue;
                }
                let texel_buffer = descriptor_slot_masked(call.texel_buffer_mask, slot);
                trace_graphics_texture_binding!(
                    call,
                    slot,
                    None,
                    None,
                    texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    GraphicsTextureBindOutcome::Dummy,
                    if texel_buffer {
                        "dummy-texel-buffer"
                    } else {
                        "dummy-descriptor"
                    },
                    if texel_buffer {
                        format!("{:?}", bound_texel_views[slot])
                    } else {
                        format!("{:?}", bound_tex_views[slot])
                    },
                    None,
                    Some("manifest slot has no resolved TIC descriptor".to_string()),
                );
            }
        }
        let mut bound_samplers = vec![default_samp; max_texture_descriptors()];
        for (slot, tsc) in tsc_entries.iter().enumerate() {
            let Some(t) = *tsc else {
                continue;
            };
            let integer_sample = texture_requires_integer_sampler(
                texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                stencil_alias_bound.get(slot).copied().unwrap_or(false),
            );
            let cache = if integer_sample {
                &mut *integer_sampler_cache
            } else {
                &mut *sampler_cache
            };
            bound_samplers[slot] = match cached_sampler_for_tsc(
                device,
                cache,
                t,
                integer_sample,
                *sampler_filter_minmax_supported,
                *sampler_anisotropy_supported,
            ) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("tsc sampler create failed: {}", e);
                    default_samp
                }
            };
        }

        let vertex_binds = upload_vertex_bindings(
            device,
            frame_slots,
            other_idx,
            descriptor_pool.pool,
            ubo_ring,
            &vertex_bindings,
            true,
        )?;

        let white_bind: Option<(u32, vk::Buffer, u64)> =
            if let Some(wb) = call.vertex_layout.bindings.iter().find(|b| b.stride == 0) {
                let (wbuf, woff, wptr) = ring_alloc(ubo_ring, 16, 16)
                    .map_err(|e| format!("ring_alloc(const_attr): {}", e))?;
                unsafe {
                    let const_default = [0.0f32, 0.0, 0.0, 1.0];
                    std::ptr::copy_nonoverlapping(const_default.as_ptr() as *const u8, wptr, 16);
                }
                Some((wb.binding, wbuf, woff))
            } else {
                None
            };

        let index_bind: Option<(vk::Buffer, u64)> = if index_count > 0 && !index_data.is_empty() {
            let isz = align_up(index_data.len() as u64, 4);
            if ubo_ring.head + isz > ubo_ring.size {
                ubo_ring.head = 0;
                ubo_ring.slot_head[other_idx] = 0;
            }
            let (ibuf, ioff, iptr) =
                ring_alloc(ubo_ring, isz, 4).map_err(|e| format!("ring_alloc(index): {}", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(index_data.as_ptr(), iptr, index_data.len());
            }
            Some((ibuf, ioff))
        } else {
            None
        };

        let (_cbuf_range, cbuf_size_aligned) = graphics_cbuf_allocation_size(
            cbuf_data.len(),
            cbuf_alignment,
            *max_storage_buffer_range,
        )?;
        {
            let v_size = vertex_bindings_size(&vertex_bindings);
            debug_assert!(
                v_size + cbuf_size_aligned <= ubo_ring.size,
                "execute_draw: per-draw ring payload ({} vertex + {} ubo) exceeds ring capacity ({})",
                v_size, cbuf_size_aligned, ubo_ring.size,
            );
        }
        if ubo_ring.head + cbuf_size_aligned > ubo_ring.size {
            *tele_ring_wraps += 1;
            let other = &mut frame_slots[other_idx];
            if other.in_flight {
                wait_fence(device, other.fence)?;
                *tele_ring_waits += 1;
                if !other.retired_dsets.is_empty() {
                    unsafe {
                        let _ =
                            device.free_descriptor_sets(descriptor_pool.pool, &other.retired_dsets);
                    }
                    other.retired_dsets.clear();
                }
                destroy_descriptor_pools(device, &mut other.retired_dset_pools);
                for view in other.retired_views.drain(..) {
                    unsafe {
                        device.destroy_image_view(view, None);
                    }
                }
                reset_command_buffer(device, other.cmd)?;
                other.in_flight = false;
            }
            ubo_ring.head = 0;
            ubo_ring.slot_head[other_idx] = 0;
        }
        let (ubo_buffer, ubo_offset, ubo_ptr) =
            ring_alloc(ubo_ring, cbuf_size_aligned, cbuf_alignment)
                .map_err(|e| format!("ring_alloc(graphics-cbuf): {}", e))?;
        unsafe {
            std::ptr::copy_nonoverlapping(cbuf_data.as_ptr(), ubo_ptr, cbuf_data.len());
        }

        let set_layouts = [descriptor_layout.layout];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: descriptor_pool.pool,
            descriptor_set_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let dsets = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("allocate_descriptor_sets: {:?}", e))?
        };
        let dset = dsets[0];

        let ubo_info = vk::DescriptorBufferInfo {
            buffer: ubo_buffer,
            offset: ubo_offset,
            range: cbuf_data.len() as u64,
        };
        let image_infos = typed_sampled_image_infos(
            &bound_tex_views,
            None,
            &call.texture_numeric_manifest,
            &dummy_views_2d,
        );
        let image_infos_3d = typed_sampled_image_infos(
            &bound_tex_views_3d,
            None,
            &call.texture_numeric_manifest,
            &dummy_views_3d,
        );
        let image_infos_cube = typed_sampled_image_infos(
            &bound_tex_views_cube,
            None,
            &call.texture_numeric_manifest,
            &dummy_views_cube,
        );
        let image_infos_cube_array = typed_sampled_image_infos(
            &bound_tex_views_cube_array,
            None,
            &call.texture_numeric_manifest,
            &dummy_views_cube_array,
        );
        let typed_texel_views = typed_texel_buffer_views(
            &bound_texel_views,
            &call.texture_numeric_manifest,
            graphics_dummies.texel_buffer,
        );
        let sampler_infos: Vec<vk::DescriptorImageInfo> = bound_samplers
            .iter()
            .map(|sampler| vk::DescriptorImageInfo {
                sampler: *sampler,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            })
            .collect();
        let mut ssbo_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut ssbo_bindings: Vec<u32> = Vec::new();
        let mut ssbo_provided = [false; crate::descriptor::MAX_SSBO as usize];
        for (idx, data) in &call.ssbo_data {
            if *idx >= crate::descriptor::MAX_SSBO || data.is_empty() {
                continue;
            }
            let sz = data.len() as u64;
            let sz_al = align_up(sz, 16);
            if ubo_ring.head + sz_al > ubo_ring.size {
                ubo_ring.head = 0;
                ubo_ring.slot_head[other_idx] = 0;
            }
            let (sbuf, soff, sptr) =
                ring_alloc(ubo_ring, sz_al, 16).map_err(|e| format!("ring_alloc(ssbo): {}", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), sptr, data.len());
            }
            ssbo_infos.push(vk::DescriptorBufferInfo {
                buffer: sbuf,
                offset: soff,
                range: sz,
            });
            ssbo_bindings.push(*idx);
            ssbo_provided[*idx as usize] = true;
        }
        if ssbo_provided.iter().any(|p| !p) {
            if ubo_ring.head + 16 > ubo_ring.size {
                ubo_ring.head = 0;
                ubo_ring.slot_head[other_idx] = 0;
            }
            let (dbuf, doff, dptr) = ring_alloc(ubo_ring, 16, 16)
                .map_err(|e| format!("ring_alloc(ssbo-dummy): {}", e))?;
            unsafe {
                std::ptr::write_bytes(dptr, 0, 16);
            }
            for i in 0..crate::descriptor::MAX_SSBO {
                if !ssbo_provided[i as usize] {
                    ssbo_infos.push(vk::DescriptorBufferInfo {
                        buffer: dbuf,
                        offset: doff,
                        range: 16,
                    });
                    ssbo_bindings.push(i);
                }
            }
        }
        let mut writes = vec![
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: crate::descriptor::CBUF_BINDING,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &ubo_info,
                p_image_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: crate::descriptor::SAMPLER_BINDING,
                dst_array_element: 0,
                descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                descriptor_type: vk::DescriptorType::SAMPLER,
                p_image_info: sampler_infos.as_ptr(),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
        ];
        for (i, binding) in ssbo_bindings.iter().enumerate() {
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: crate::descriptor::SSBO_BINDING_BASE + *binding,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &ssbo_infos[i],
                p_image_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }
        for (binding, infos) in crate::descriptor::SAMPLED_IMAGE_BINDINGS.into_iter().zip([
            &image_infos[0],
            &image_infos_3d[0],
            &image_infos_cube[0],
            &image_infos_cube_array[0],
            &image_infos[1],
            &image_infos_3d[1],
            &image_infos_cube[1],
            &image_infos_cube_array[1],
            &image_infos[2],
            &image_infos_3d[2],
            &image_infos_cube[2],
            &image_infos_cube_array[2],
        ]) {
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: binding,
                dst_array_element: 0,
                descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: infos.as_ptr(),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }
        for (binding, views) in crate::descriptor::TEXEL_BUFFER_BINDINGS
            .into_iter()
            .zip([&typed_texel_views[0], &typed_texel_views[1], &typed_texel_views[2]])
        {
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: binding,
                dst_array_element: 0,
                descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                descriptor_type: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: views.as_ptr(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut color_bind = Vec::with_capacity(color_keys.len());
        for (idx, key) in color_keys.iter().enumerate() {
            let format = color_formats.get(idx).copied().unwrap_or(call.rt_format);
            let rt = rt_cache.get_or_create_with_format(*key, device, format)?;
            color_bind.push((*key, rt.image, rt.view, rt.extent, rt.layout));
        }
        let mut depth_fresh = false;
        let depth_bind: Option<(vk::Image, vk::ImageView, vk::ImageLayout, vk::Extent2D)> =
            if use_depth {
                let (d, fresh) = rt_cache.get_or_create_depth(
                    call.depth_key.unwrap(),
                    device,
                    call.depth_format,
                    call.depth_aspects,
                )?;
                depth_fresh = fresh;
                Some((d.image, d.view, d.layout, d.extent))
            } else {
                None
            };
        let mut rt_extent = vk::Extent2D {
            width: call.rt_key.width,
            height: call.rt_key.height,
        };
        for (_, _, _, extent, _) in &color_bind {
            rt_extent.width = rt_extent.width.min(extent.width);
            rt_extent.height = rt_extent.height.min(extent.height);
        }
        if let Some((_, _, _, extent)) = depth_bind {
            rt_extent.width = rt_extent.width.min(extent.width);
            rt_extent.height = rt_extent.height.min(extent.height);
        }

        let cmd = frame_slots[cur_idx].cmd;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(slot): {:?}", e))?;
        }
        for pending in pending_rt_reinterprets {
            record_rt_alias_reinterpret(
                device,
                cmd,
                mem_props,
                rt_cache,
                &mut frame_slots[cur_idx],
                rt_reinterpret_cache,
                pending,
            )?;
        }
        for (key, image, _, _, prev_layout) in &color_bind {
            let prev_layout = rt_cache.color_layout(*key).unwrap_or(*prev_layout);
            transition_image(
                device,
                cmd,
                *image,
                prev_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
        if let Some((d_image, _, d_prev, _)) = depth_bind {
            transition_image_aspect(
                device,
                cmd,
                d_image,
                d_prev,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                call.depth_aspects,
            );
        }
        for alias in &rt_aliases {
            if let Some(alias) = *alias {
                if alias.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                    if alias.depth {
                        transition_image_aspect(
                            device,
                            cmd,
                            alias.image,
                            alias.layout,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            alias.aspects,
                        );
                        rt_cache
                            .set_depth_layout(alias.key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    } else {
                        transition_image(
                            device,
                            cmd,
                            alias.image,
                            alias.layout,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        );
                        rt_cache
                            .set_color_layout(alias.key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                    }
                }
            }
        }

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: call.clear_color,
            },
        };
        let depth_clear_far = if call.depth.compare_op == vk::CompareOp::GREATER
            || call.depth.compare_op == vk::CompareOp::GREATER_OR_EQUAL
        {
            0.0
        } else {
            call.clear_depth_hint
        };
        let depth_attachment = depth_bind.map(|(_, d_view, _, _)| vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: d_view,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            load_op: if depth_fresh {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            },
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: depth_clear_far,
                    stencil: call.clear_stencil_hint,
                },
            },
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
        let p_depth_attachment = if call.depth_aspects.contains(vk::ImageAspectFlags::DEPTH) {
            depth_attachment
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _)
        } else {
            std::ptr::null()
        };
        let p_stencil_attachment = if call.depth_aspects.contains(vk::ImageAspectFlags::STENCIL) {
            depth_attachment
                .as_ref()
                .map_or(std::ptr::null(), |a| a as *const _)
        } else {
            std::ptr::null()
        };
        let attachments = color_bind
            .iter()
            .map(|(_, _, view, _, _)| vk::RenderingAttachmentInfo {
                s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                image_view: *view,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(),
                resolve_image_layout: vk::ImageLayout::UNDEFINED,
                load_op: if call.clear {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::LOAD
                },
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value,
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            })
            .collect::<Vec<_>>();
        let render_layer_count = attachment_render_layer_count(
            &color_keys,
            if use_depth { call.depth_key } else { None },
        )?;
        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: rt_extent,
            },
            layer_count: render_layer_count,
            view_mask: 0,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment,
            p_stencil_attachment,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        unsafe { device.cmd_begin_rendering(cmd, &render_info) };

        let viewport = match call.vp_rect {
            Some([x, y, w, h]) => vk::Viewport {
                x,
                y,
                width: w,
                height: h,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            None => vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: rt_extent.width as f32,
                height: rt_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        };
        let scissor = draw_scissor(call.scissor, rt_extent);
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            set_dynamic_stencil_state(device, cmd, call.stencil);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_cache.layout,
                0,
                &[dset],
                &[],
            );
            for (binding, vbuf, voff) in &vertex_binds {
                device.cmd_bind_vertex_buffers(cmd, *binding, &[*vbuf], &[*voff]);
            }
            if let Some((wbinding, wbuf, woff)) = white_bind {
                device.cmd_bind_vertex_buffers(cmd, wbinding, &[wbuf], &[woff]);
            }
            let cmd_first_vertex = if !vertex_binds.is_empty() {
                0
            } else {
                call.first_vertex
            };
            if let Some((ibuf, ioff)) = index_bind {
                device.cmd_bind_index_buffer(cmd, ibuf, ioff, index_type);
                let vertex_offset = if !vertex_binds.is_empty() {
                    call.first_vertex as i32
                } else {
                    0
                };
                device.cmd_draw_indexed(
                    cmd,
                    index_count,
                    call.instance_count.max(1),
                    0,
                    vertex_offset,
                    call.first_instance,
                );
            } else {
                device.cmd_draw(
                    cmd,
                    draw_vertex_count,
                    call.instance_count.max(1),
                    cmd_first_vertex,
                    call.first_instance,
                );
            }
            device.cmd_end_rendering(cmd);
        }
        if use_depth && (depth_fresh || call_writes_depth_stencil(call)) {
            rt_cache.mark_depth_written(call.depth_key.unwrap());
        }
        for (_, image, _, _, _) in &color_bind {
            barrier_color_attachment_after_pass(
                device,
                cmd,
                *image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }

        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(slot): {:?}", e))?;
        }

        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx].retired_dsets.push(dset);

        for (key, _, _, _, _) in &color_bind {
            rt_cache.set_color_layout(*key, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
            if call
                .blend
                .attachments
                .get(idx)
                .is_some_and(|att| !att.color_write_mask.is_empty())
            {
                let stamp = rt_cache.mark_drawn(*key);
                rt_cache.record_present_flip(*key, call.present_flip_y);
                trace_rt_stamp(stamp, *key, &[call]);
            }
        }
        if use_depth {
            if let Ok((d, _)) = rt_cache.get_or_create_depth(
                call.depth_key.unwrap(),
                device,
                call.depth_format,
                call.depth_aspects,
            ) {
                d.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
        }
        for alias in rt_aliases {
            if let Some(alias) = alias {
                if alias.depth {
                    rt_cache.set_depth_layout(alias.key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                } else if !color_keys.contains(&alias.key) {
                    rt_cache.set_color_layout(alias.key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                }
            }
        }

        let next_idx = other_idx;
        ubo_ring.slot_head[next_idx] = ubo_ring.head;
        *frame_index = next_idx;
        pipeline_cache.maybe_save(device);

        *tele_in_flight_mask =
            (frame_slots[0].in_flight as u32) | ((frame_slots[1].in_flight as u32) << 1);
        let now_ns = monotonic_nanos();
        if now_ns.saturating_sub(*tele_last_emit_ns) >= 1_000_000_000 {
            log::info!(
                "[ring] head={} slot_head=[{},{}] wraps={} waits={} in_flight=0b{:02b}",
                ubo_ring.head,
                ubo_ring.slot_head[0],
                ubo_ring.slot_head[1],
                *tele_ring_wraps,
                *tele_ring_waits,
                *tele_in_flight_mask,
            );
            log::info!(
                "[frameslot] cur_idx={} dsets=[{},{}]",
                *frame_index,
                frame_slots[0].retired_dsets.len(),
                frame_slots[1].retired_dsets.len(),
            );
            *tele_ring_wraps = 0;
            *tele_ring_waits = 0;
            *tele_last_emit_ns = now_ns;
        }
        Ok(())
    }

    pub fn execute_draws<F>(
        &self,
        calls: &[crate::draw::Maxwell3dDrawCall],
        read_guest: F,
    ) -> Result<(), String>
    where
        F: Fn(u64, usize) -> Option<Vec<u8>>,
    {
        self.execute_draws_with_texture_generations(calls, read_guest, |gpu_va, len| {
            crate::tex_invalidate::region_gen_range(gpu_va, len as u64)
        })
    }

    pub fn execute_draws_with_texture_generations<F, G>(
        &self,
        calls: &[crate::draw::Maxwell3dDrawCall],
        read_guest: F,
        texture_generation: G,
    ) -> Result<(), String>
    where
        F: Fn(u64, usize) -> Option<Vec<u8>>,
        G: Fn(u64, usize) -> u64,
    {
        if calls.is_empty() {
            return Ok(());
        }
        self.settle_pending_computes();
        let rp_t0 = std::time::Instant::now();

        struct Prep {
            pipeline: vk::Pipeline,
            vertex_bindings: Vec<PreparedVertexBinding>,
            cbuf_data: Vec<u8>,
            tex_pendings: Vec<Option<PendingTexture>>,
            use_depth: bool,
            tsc_entries: Vec<Option<crate::texture::TscEntry>>,
            index_data: Vec<u8>,
            index_count: u32,
            index_type: vk::IndexType,
            draw_vertex_count: u32,
        }
        let mut preps: Vec<(&crate::draw::Maxwell3dDrawCall, Prep)> =
            Vec::with_capacity(calls.len());
        let batch_color_formats = calls
            .iter()
            .find(|call| call_writes_any_color(call))
            .map(|call| {
                let color_keys = active_color_keys_for_call(call);
                color_formats_for_call(call, color_keys.len())
            })
            .unwrap_or_default();
        for (call_index, call) in calls.iter().enumerate() {
            let use_depth = call.depth_key.is_some();
            let depth_format = if use_depth {
                call.depth_format
            } else {
                vk::Format::UNDEFINED
            };
            let depth_aspects = if use_depth {
                call.depth_aspects
            } else {
                vk::ImageAspectFlags::empty()
            };
            if !use_depth && !call_writes_any_color(call) {
                continue;
            }
            let pipeline = match self
                .compile_pipeline(
                    &call.vs_spirv,
                    &call.fs_spirv,
                    call.vs_hash,
                    call.fs_hash,
                    call.vs_cbuf_mask,
                    call.fs_cbuf_mask,
                    &call.vertex_layout,
                    call.state.topology,
                    &batch_color_formats,
                    call.blend,
                    call.cull_test_enable,
                    call.cull_face,
                    call.front_face,
                    call.depth_clamp_enabled,
                    call.poly_offset_enable,
                    call.poly_offset_units,
                    call.poly_offset_factor,
                    call.depth,
                    depth_format,
                    depth_aspects,
                    call.stencil,
                    call.vertex_count,
                )
                .map_err(|error| {
                    graphics_draw_call_error(call_index, call, "pipeline", error)
                })?
            {
                Some(p) => p,
                None => continue,
            };
            let (vertex_bindings, draw_vertex_count) =
                match prepare_vertex_bindings(call, &read_guest) {
                    Ok(v) => v,
                    Err(e) => {
                        log_vertex_bindings_skip(call, &e);
                        continue;
                    }
                };
            let cbuf_data = if let Some(d) = &call.cbuf_data {
                if d.len() >= nexium_spirv::GFX_CBUF_MIN_SIZE as usize {
                    d.clone()
                } else {
                    empty_graphics_cbuf_data()
                }
            } else {
                empty_graphics_cbuf_data()
            };
            let tex_pendings = collect_tex_pendings(call, &read_guest).map_err(|error| {
                graphics_draw_call_error(call_index, call, "texture-prepare", error)
            })?;
            let tsc_entries = collect_tsc_entries(call, &read_guest);
            let (index_data, index_count, index_type) = match (&call.index_data, call.index_count) {
                (Some(d), Some(c)) if c > 0 && !d.is_empty() => (d.clone(), c, call.index_type),
                _ => (Vec::new(), 0u32, call.index_type),
            };
            if let Ok(want) = std::env::var("NEXIUM_VTX_DBG") {
                let all_mode = want.trim().eq_ignore_ascii_case("all");
                let is_3d = call.vertex_layout.attrs.iter().any(|a| {
                    a.location == 0
                        && vertex_debug_float_components(a.format).is_some_and(|c| c >= 3)
                });
                let under_cap = if all_mode {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    static N3D: AtomicU64 = AtomicU64::new(0);
                    if is_3d {
                        N3D.fetch_add(1, Ordering::Relaxed) < 60
                    } else {
                        N.fetch_add(1, Ordering::Relaxed) < 20
                    }
                } else {
                    true
                };
                if under_cap && (all_mode || parse_u64_value(&want) == Some(call.vs_gpu_va)) {
                    let vertex_base_addr = call
                        .vertex_bindings
                        .first()
                        .map(|b| {
                            b.addr.wrapping_add(
                                (b.stride as u64).saturating_mul(call.first_vertex as u64),
                            )
                        })
                        .unwrap_or(call.vertex_addr);
                    if let Ok(addr) = std::env::var("NEXIUM_VTX_DBG_ADDR") {
                        if parse_u64_value(&addr) != Some(vertex_base_addr) {
                            continue;
                        }
                    }
                    if let Ok(min) = std::env::var("NEXIUM_VTX_DBG_MIN_VERTS") {
                        if min
                            .parse::<u32>()
                            .ok()
                            .is_some_and(|min| call.vertex_count < min)
                        {
                            continue;
                        }
                    }
                    let first_binding = vertex_bindings.first();
                    let floats: Vec<f32> = first_binding
                        .map(|b| {
                            b.data
                                .chunks_exact(4)
                                .take(24)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect()
                        })
                        .unwrap_or_default();
                    let idx: Vec<u16> = index_data
                        .chunks_exact(2)
                        .take(8)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let attrs = call
                        .vertex_layout
                        .attrs
                        .iter()
                        .map(|a| {
                            format!(
                                "loc{}:b{}:{:?}:off{}",
                                a.location, a.binding, a.format, a.offset
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    let vertex_stride = first_binding.map(|b| b.stride).unwrap_or(0);
                    let vertex_len: usize = vertex_bindings.iter().map(|b| b.data.len()).sum();
                    let attr_summary = vertex_debug_attr_summary(call, &vertex_bindings);
                    let index_summary =
                        vertex_debug_index_summary(&index_data, index_count, index_type);
                    log::warn!(
                        "[vtx-dbg] vs={:#x} fs={:#x} tex={:?} rt={} addr={:#x} stride={} vlen={} vcount={} icount={} itype={:?} attrs=[{}] attr_summary={} index_summary={} floats={:?} idx={:?}",
                        call.vs_gpu_va,
                        call.fs_gpu_va,
                        call.fs_tex_ids,
                        call.rt_key.label(),
                        vertex_base_addr,
                        vertex_stride,
                        vertex_len,
                        call.vertex_count,
                        index_count,
                        index_type,
                        attrs,
                        attr_summary,
                        index_summary,
                        floats,
                        idx
                    );
                }
            }
            preps.push((
                call,
                Prep {
                    pipeline,
                    vertex_bindings,
                    cbuf_data,
                    tex_pendings,
                    use_depth,
                    tsc_entries,
                    index_data,
                    index_count,
                    index_type,
                    draw_vertex_count,
                },
            ));
        }

        if preps.is_empty() {
            return Ok(());
        }
        let rp_draws = preps.len() as u64;
        let rp_prep = rp_t0.elapsed();

        let rp_t1 = std::time::Instant::now();
        let mut inner = self.inner.lock();
        let rp_lock = rp_t1.elapsed();
        let RendererInner {
            device,
            queue,
            mem_props,
            cmd_pool,
            rt_cache,
            descriptor_layout,
            descriptor_pool,
            pipeline_cache,
            dummy_images,
            dummy_texel_buffers,
            default_sampler,
            sampler_cache,
            integer_sampler_cache,
            sampler_filter_minmax_supported,
            sampler_anisotropy_supported,
            tex_cache,
            texel_buffer_cache,
            rt_reinterpret_cache,
            frame_slots,
            frame_index,
            ubo_ring,
            min_storage_buffer_offset_alignment,
            max_storage_buffer_range,
            max_texel_buffer_elements,
            ..
        } = &mut *inner;

        let cur_idx = *frame_index;
        let other_idx = (cur_idx + 1) % 2;
        let cbuf_alignment = *min_storage_buffer_offset_alignment;

        let rp_t2 = std::time::Instant::now();
        {
            let slot = &mut frame_slots[cur_idx];
            if slot.in_flight {
                wait_fence(device, slot.fence)?;
                if !slot.retired_dsets.is_empty() {
                    unsafe {
                        let _ =
                            device.free_descriptor_sets(descriptor_pool.pool, &slot.retired_dsets);
                    }
                    slot.retired_dsets.clear();
                }
                destroy_descriptor_pools(device, &mut slot.retired_dset_pools);
                for (b, m) in slot.retired_buffers.drain(..) {
                    unsafe {
                        device.destroy_buffer(b, None);
                        device.free_memory(m, None);
                    }
                }
                for view in slot.retired_views.drain(..) {
                    unsafe {
                        device.destroy_image_view(view, None);
                    }
                }
                for t in slot.retired_textures.drain(..) {
                    unsafe {
                        device.destroy_image_view(t.view, None);
                        device.destroy_image(t.image, None);
                        device.free_memory(t.memory, None);
                    }
                }
                for buffer in slot.retired_texel_buffers.drain(..) {
                    destroy_texel_buffer(device, buffer.resource);
                }
                for t in slot.retired_rt_reinterprets.drain(..) {
                    unsafe {
                        device.destroy_image_view(t.view, None);
                        device.destroy_image(t.image, None);
                        device.free_memory(t.memory, None);
                    }
                }
                reset_command_buffer(device, slot.cmd)?;
                slot.in_flight = false;
                ubo_ring.head = ubo_ring.slot_head[cur_idx];
            }
        }
        let rp_fence = rp_t2.elapsed();
        if cbuf_alignment > MAX_STORAGE_BUFFER_OFFSET_ALIGNMENT {
            return Err(format!(
                "device minStorageBufferOffsetAlignment {} exceeds Vulkan's supported graphics-ring bound {}",
                cbuf_alignment, MAX_STORAGE_BUFFER_OFFSET_ALIGNMENT
            ));
        }
        let batch_ring_upper = preps.iter().fold(0u64, |total, (call, _)| {
            total.saturating_add(graphics_draw_ring_bytes_upper_bound(call))
        });
        if batch_ring_upper > ubo_ring.size {
            return Err(format!(
                "graphics batch upload footprint {:#x} exceeds ring capacity {:#x}",
                batch_ring_upper, ubo_ring.size
            ));
        }
        if ubo_ring.head.saturating_add(batch_ring_upper) > ubo_ring.size {
            ring_wrap_other(
                device,
                frame_slots,
                other_idx,
                descriptor_pool.pool,
                ubo_ring,
            )?;
        }
        let rp_t3 = std::time::Instant::now();
        let mut rp_alias = std::time::Duration::ZERO;
        let mut rp_tex = std::time::Duration::ZERO;
        let mut rp_vtx = std::time::Duration::ZERO;
        let mut rp_dset = std::time::Duration::ZERO;
        let graphics_dummies = ensure_graphics_dummy_views(
            dummy_images,
            dummy_texel_buffers,
            device,
            *queue,
            *cmd_pool,
            mem_props,
        )?;
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let default_samp = default_sampler.unwrap();

        let color_source = preps
            .iter()
            .map(|(call, _)| *call)
            .find(|call| call_writes_any_color(call))
            .unwrap_or(preps[0].0);
        let rt_key = color_source.rt_key;
        let color_keys = active_color_keys_for_call(color_source);
        let color_formats = color_formats_for_call(color_source, color_keys.len());
        let mut color_bind = Vec::with_capacity(color_keys.len());
        for (idx, key) in color_keys.iter().enumerate() {
            let format = color_formats
                .get(idx)
                .copied()
                .unwrap_or(calls[0].rt_format);
            if std::env::var_os("NEXIUM_RT_FORMAT_BIND_DBG").is_some() && key.nvmap_id == 16 {
                let sampled = calls[0]
                    .sampled_rt_slots
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, sampled)| {
                        sampled.map(|sampled| format!("s{}={}", slot, sampled.label()))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                log::warn!(
                    "[rt-format-bind] fs={:#x} rt={} idx={} fmt={:?} sampled=[{}] tex={:?}",
                    calls[0].fs_gpu_va,
                    key.label(),
                    idx,
                    format,
                    sampled,
                    calls[0].fs_tex_ids
                );
            }
            let rt = rt_cache.get_or_create_with_format(*key, device, format)?;
            color_bind.push((*key, rt.image, rt.view, rt.extent, rt.layout));
        }
        let mut rt_extent = vk::Extent2D {
            width: rt_key.width,
            height: rt_key.height,
        };
        for (_, _, _, extent, _) in &color_bind {
            rt_extent.width = rt_extent.width.min(extent.width);
            rt_extent.height = rt_extent.height.min(extent.height);
        }
        let rt_prev_layout = color_bind
            .first()
            .map(|(_, _, _, _, layout)| *layout)
            .unwrap_or(vk::ImageLayout::UNDEFINED);
        let any_depth = preps.iter().any(|p| p.1.use_depth);
        let depth_key = if any_depth { calls[0].depth_key } else { None };
        let logical_depth_pass = if let Some(depth_key) = depth_key {
            rt_cache.begin_depth_pass(depth_key, &color_keys)
        } else {
            false
        };
        let (depth_image, depth_view, depth_prev, depth_fresh, depth_extent) = if any_depth {
            let (d, fresh) = rt_cache.get_or_create_depth(
                depth_key.unwrap(),
                device,
                calls[0].depth_format,
                calls[0].depth_aspects,
            )?;
            (Some(d.image), Some(d.view), d.layout, fresh, Some(d.extent))
        } else {
            (None, None, vk::ImageLayout::UNDEFINED, false, None)
        };
        if let Some(extent) = depth_extent {
            rt_extent.width = rt_extent.width.min(extent.width);
            rt_extent.height = rt_extent.height.min(extent.height);
        }

        let mut dset_pools = DescriptorPoolBatch::new(device)?;
        let cmd = frame_slots[cur_idx].cmd;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(batch): {:?}", e))?;
        }
        let mut color_layouts = color_bind
            .iter()
            .map(|(_, _, _, _, layout)| *layout)
            .collect::<Vec<_>>();
        if let Some(di) = depth_image {
            transition_image_aspect(
                device,
                cmd,
                di,
                depth_prev,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                calls[0].depth_aspects,
            );
        }

        let clear_rt = !color_bind.is_empty() && rt_prev_layout == vk::ImageLayout::UNDEFINED;

        let mut alias_used: Vec<(RtKey, bool)> = Vec::new();
        let mut post_submit_texture_probe: [Option<PostSubmitTextureProbe>; 2] = [None, None];
        let mut tex_raw_cache: HashMap<(u64, usize), Option<(u64, Vec<u8>)>> = HashMap::new();
        let mut tex_sam_cache: HashMap<(u64, usize), Option<u64>> = HashMap::new();
        let mut pass_open = false;
        let mut pass_depth = false;
        let mut depth_needs_clear = depth_fresh || logical_depth_pass;
        let mut pass_rt_layout = vk::ImageLayout::UNDEFINED;
        let mut pass_dirty = vec![false; color_bind.len()];
        let mut pass_trace_calls: Vec<&crate::draw::Maxwell3dDrawCall> = Vec::new();
        let mut had_pass = false;
        for (_i, (call, prep)) in preps.iter().enumerate() {
            let call = *call;
            let rp_a0 = std::time::Instant::now();
            let mut rt_aliases: Vec<_> = (0..prep.tex_pendings.len())
                .map(|slot| {
                    let tic = prep
                        .tex_pendings
                        .get(slot)
                        .and_then(|pending| pending.as_ref().map(|(_, tic, _, _)| tic));
                    rt_alias_for_slot(rt_cache, call, slot, rt_key, true, tic)
                })
                .collect();
            let feedback_loop = color_keys
                .iter()
                .copied()
                .any(|key| call_samples_rt(call, key))
                || rt_aliases
                    .iter()
                    .flatten()
                    .any(|alias| !alias.depth && color_keys.contains(&alias.key));
            let required_rt_layout = if feedback_loop {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            };
            if feedback_loop && pass_open {
                unsafe {
                    device.cmd_end_rendering(cmd);
                }
                finish_color_pass(
                    device,
                    cmd,
                    rt_cache,
                    &color_bind,
                    &mut color_layouts,
                    pass_rt_layout,
                    &mut pass_dirty,
                    &pass_trace_calls,
                );
                pass_open = false;
                pass_trace_calls.clear();
            }
            let mut alias_snapshotted = vec![false; rt_aliases.len()];
            if feedback_loop {
                for (slot, alias_opt) in rt_aliases.iter_mut().enumerate() {
                    let Some(alias) = alias_opt else {
                        continue;
                    };
                    if alias.depth || !color_keys.contains(&alias.key) {
                        continue;
                    }
                    match snapshot_feedback_alias(device, cmd, rt_cache, alias.key) {
                        Ok((snap_image, snap_view, snap_format)) => {
                            alias.image = snap_image;
                            alias.view = snap_view;
                            alias.format = snap_format;
                            alias.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                            alias_snapshotted[slot] = true;
                        }
                        Err(e) => {
                            log::debug!("feedback snapshot failed: {}", e);
                        }
                    }
                }
            }
            rp_alias += rp_a0.elapsed();

            for alias in &rt_aliases {
                if let Some(alias) = *alias {
                    if !alias.depth && color_keys.contains(&alias.key) {
                        continue;
                    }
                    let used_key = (alias.key, alias.depth);
                    if !alias_used.contains(&used_key) {
                        let alias_prev = if alias.depth {
                            rt_cache.depth_layout(alias.key).unwrap_or(alias.layout)
                        } else {
                            rt_cache.color_layout(alias.key).unwrap_or(alias.layout)
                        };
                        if alias_prev != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
                            if pass_open {
                                unsafe {
                                    device.cmd_end_rendering(cmd);
                                }
                                finish_color_pass(
                                    device,
                                    cmd,
                                    rt_cache,
                                    &color_bind,
                                    &mut color_layouts,
                                    pass_rt_layout,
                                    &mut pass_dirty,
                                    &pass_trace_calls,
                                );
                                pass_open = false;
                                pass_trace_calls.clear();
                            }
                            if alias.depth {
                                transition_image_aspect(
                                    device,
                                    cmd,
                                    alias.image,
                                    alias_prev,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    alias.aspects,
                                );
                                rt_cache.set_depth_layout(
                                    alias.key,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                );
                            } else {
                                if alias_prev == vk::ImageLayout::UNDEFINED {
                                    transition_image(
                                        device,
                                        cmd,
                                        alias.image,
                                        vk::ImageLayout::UNDEFINED,
                                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    );
                                    let clear = vk::ClearColorValue {
                                        float32: [0.0, 0.0, 0.0, 0.0],
                                    };
                                    let range = vk::ImageSubresourceRange {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        base_mip_level: 0,
                                        level_count: 1,
                                        base_array_layer: 0,
                                        layer_count: 1,
                                    };
                                    unsafe {
                                        device.cmd_clear_color_image(
                                            cmd,
                                            alias.image,
                                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                            &clear,
                                            &[range],
                                        );
                                    }
                                    transition_image(
                                        device,
                                        cmd,
                                        alias.image,
                                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    );
                                } else {
                                    transition_image(
                                        device,
                                        cmd,
                                        alias.image,
                                        alias_prev,
                                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    );
                                }
                                rt_cache.set_color_layout(
                                    alias.key,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                );
                            }
                        }
                        alias_used.push(used_key);
                    }
                }
            }

            let dummy_views_2d = (0..max_texture_descriptors())
                .map(|slot| graphics_dummies.image_2d_for_slot(call, slot))
                .collect::<Vec<_>>();
            let dummy_views_3d = vec![graphics_dummies.image_3d; max_texture_descriptors()];
            let dummy_views_cube = (0..max_texture_descriptors())
                .map(|slot| graphics_dummies.image_cube_for_slot(call, slot))
                .collect::<Vec<_>>();
            let dummy_views_cube_array = (0..max_texture_descriptors())
                .map(|slot| graphics_dummies.image_cube_array_for_slot(call, slot))
                .collect::<Vec<_>>();
            let mut bound_tex_views = (0..max_texture_descriptors())
                .map(|slot| {
                    let family = texture_numeric_index(texture_numeric_type_for_slot(
                        &call.texture_numeric_manifest,
                        slot,
                    ));
                    dummy_views_2d[slot][family]
                })
                .collect::<Vec<_>>();
            let mut bound_tex_views_3d = (0..max_texture_descriptors())
                .map(|slot| {
                    dummy_views_3d[slot][texture_numeric_index(texture_numeric_type_for_slot(
                        &call.texture_numeric_manifest,
                        slot,
                    ))]
                })
                .collect::<Vec<_>>();
            let mut bound_tex_views_cube = (0..max_texture_descriptors())
                .map(|slot| {
                    dummy_views_cube[slot][texture_numeric_index(
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    )]
                })
                .collect::<Vec<_>>();
            let mut bound_tex_views_cube_array = (0..max_texture_descriptors())
                .map(|slot| {
                    dummy_views_cube_array[slot][texture_numeric_index(
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    )]
                })
                .collect::<Vec<_>>();
            let mut bound_texel_views = (0..max_texture_descriptors())
                .map(|slot| {
                    graphics_dummies.texel_buffer[texture_numeric_index(
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    )]
                })
                .collect::<Vec<_>>();
            let mut stencil_alias_bound = vec![false; max_texture_descriptors()];
            let mut bound_tex_layouts =
                vec![vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL; max_texture_descriptors()];
            let rp_tx0 = std::time::Instant::now();
            for (slot, pending) in prep.tex_pendings.iter().enumerate() {
                if let Some((_, tic, _, read_size)) = *pending {
                    if tic.is_buffer() && slot < 32 && call.texel_buffer_mask & (1u32 << slot) != 0
                    {
                        let numeric_type =
                            texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot);
                        match texel_buffer_view_for_tic(
                            device,
                            mem_props,
                            *max_texel_buffer_elements,
                            texel_buffer_cache,
                            &mut frame_slots[cur_idx].retired_texel_buffers,
                            &tic,
                            numeric_type,
                            read_size,
                            texture_generation(tic.gpu_va, read_size),
                            &read_guest,
                        ) {
                            Ok(Some(view)) => {
                                bound_texel_views[slot] = view;
                                trace_graphics_texture_binding!(
                                    call,
                                    slot,
                                    *pending,
                                    None,
                                    numeric_type,
                                    GraphicsTextureBindOutcome::TexelBuffer,
                                    "texel-buffer",
                                    format!("{view:?}"),
                                    texel_buffer_format(&tic, numeric_type)
                                        .map(|(format, _)| format!("{format:?}")),
                                    None,
                                );
                                continue;
                            }
                            Ok(None) => {
                                trace_graphics_texture_binding!(
                                    call,
                                    slot,
                                    *pending,
                                    None,
                                    numeric_type,
                                    GraphicsTextureBindOutcome::Rejection,
                                    "texel-buffer",
                                    "-",
                                    None,
                                    Some("view creation produced no view".to_string()),
                                );
                                return Err(format!(
                                    "texel buffer slot {} va={:#x} produced no view",
                                    slot, tic.gpu_va
                                ));
                            }
                            Err(error) => {
                                trace_graphics_texture_binding!(
                                    call,
                                    slot,
                                    *pending,
                                    None,
                                    numeric_type,
                                    GraphicsTextureBindOutcome::Rejection,
                                    "texel-buffer",
                                    "-",
                                    None,
                                    Some(error.clone()),
                                );
                                return Err(format!(
                                    "texel buffer slot {} va={:#x}: {}",
                                    slot, tic.gpu_va, error
                                ));
                            }
                        }
                    }
                }
                let direct_volume_rt = pending.map_or(false, |(key, _, _, _)| key.volume)
                    && sampled_rt_key_for_slot(call, slot).is_some_and(|key| key.is_3d);
                let pending_special = pending
                    .map_or(false, |(key, _, _, _)| texture_key_has_special_view(key))
                    && !direct_volume_rt;
                if !pending_special {
                    if let Some(sk) = sampled_rt_key_for_slot(call, slot) {
                        let depth_self = pending
                            .map(|(_, tic, _, _)| tic_format_prefers_depth_alias(tic.format))
                            .unwrap_or(false)
                            && sampled_active_depth_key(rt_cache, call, sk).is_some();
                        let depth_self_needs_sync =
                            depth_self && depth_self_shadow_needs_sync(rt_cache, call, sk);
                        let depth_as_color = pending
                            .map(|(_, tic, _, _)| tic_reads_depth_as_color(tic.format))
                            .unwrap_or(false)
                            && rt_cache.find_depth(sk).is_some();
                        if !depth_as_color
                            && pending.map_or(false, |(_, tic, _, _)| {
                                !tic_format_prefers_depth_alias(tic.format)
                            })
                            && rt_cache.find_depth(sk).is_some()
                        {
                            use std::sync::atomic::{AtomicU64, Ordering};
                            static N: AtomicU64 = AtomicU64::new(0);
                            if N.fetch_add(1, Ordering::Relaxed) < 12 {
                                log::warn!(
                                    "[depth-as-color-miss] slot={} tic_fmt={:?} sk={} fs={:#x}",
                                    slot,
                                    pending.map(|(_, tic, _, _)| tic.format),
                                    sk.label(),
                                    call.fs_gpu_va
                                );
                            }
                        }
                        if pass_open
                            && (sampled_color_needs_sync(rt_cache, sk)
                                || depth_self_needs_sync
                                || depth_as_color)
                        {
                            unsafe {
                                device.cmd_end_rendering(cmd);
                            }
                            finish_color_pass(
                                device,
                                cmd,
                                rt_cache,
                                &color_bind,
                                &mut color_layouts,
                                pass_rt_layout,
                                &mut pass_dirty,
                                &pass_trace_calls,
                            );
                            pass_open = false;
                            pass_trace_calls.clear();
                        }
                        let alias_synced = sync_sampled_color_alias(
                            device,
                            cmd,
                            rt_cache,
                            mem_props,
                            &mut frame_slots[cur_idx],
                            sk,
                        )?;
                        let region_synced = sync_sampled_color_region(device, cmd, rt_cache, sk)?;
                        if alias_synced || region_synced {
                            if let Some(alias_slot) = rt_aliases.get_mut(slot) {
                                let tic = prep
                                    .tex_pendings
                                    .get(slot)
                                    .and_then(|pending| pending.as_ref().map(|(_, tic, _, _)| tic));
                                *alias_slot =
                                    rt_alias_for_slot(rt_cache, call, slot, rt_key, true, tic);
                            }
                        }
                        if depth_self {
                            if let Some(alias) =
                                sync_sampled_depth_self(device, cmd, rt_cache, call, sk)?
                            {
                                if let Some(alias_slot) = rt_aliases.get_mut(slot) {
                                    *alias_slot = Some(alias);
                                }
                            }
                        }
                        if depth_as_color && rt_aliases.get(slot).copied().flatten().is_none() {
                            if let Some(alias) = sync_sampled_depth_as_color(
                                device,
                                cmd,
                                rt_cache,
                                mem_props,
                                &mut frame_slots[cur_idx],
                                sk,
                            )? {
                                if let Some(alias_slot) = rt_aliases.get_mut(slot) {
                                    *alias_slot = Some(alias);
                                }
                            }
                        }
                    }
                }
                if !pending_special {
                    if let Some(alias) = rt_aliases.get(slot).copied().flatten().filter(|alias| {
                        let view_format = pending
                            .map(|(_, tic, _, _)| {
                                rt_alias_view_format(alias.key, tic, alias.format)
                            })
                            .unwrap_or(alias.format);
                        rt_alias_numeric_type_matches(
                            *alias,
                            pending.map(|(_, tic, _, _)| tic),
                            texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                            view_format,
                        )
                    }) {
                        let view_format = pending
                            .map(|(_, tic, _, _)| {
                                rt_alias_view_format(alias.key, tic, alias.format)
                            })
                            .unwrap_or(alias.format);
                        let reinterpret_supported =
                            pending.map_or(false, |(texture_key, tic, _, _)| {
                                rt_alias_reinterpret_supported(texture_key, tic, alias, view_format)
                            });
                        if !reinterpret_supported {
                            if let Some(alias_view) = rt_alias_sample_view(
                                device,
                                rt_cache,
                                &mut frame_slots[cur_idx],
                                alias,
                                pending.map(|(_, tic, _, _)| tic),
                                view_format,
                            ) {
                                if direct_volume_rt {
                                    bound_tex_views_3d[slot] = alias_view;
                                } else {
                                    bound_tex_views[slot] = alias_view;
                                }
                                stencil_alias_bound[slot] = pending
                                    .map(|(_, tic, _, _)| {
                                        rt_alias_sample_aspect(alias, tic)
                                            == Some(vk::ImageAspectFlags::STENCIL)
                                    })
                                    .unwrap_or(false);
                                if slot < post_submit_texture_probe.len()
                                    && !alias.depth
                                    && post_submit_texture_probe_enabled(call.fs_gpu_va)
                                    && post_submit_texture_probe[slot].is_none()
                                {
                                    post_submit_texture_probe[slot] =
                                        Some(PostSubmitTextureProbe {
                                            source: "ALIAS",
                                            key: alias.key,
                                            image: alias.image,
                                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            format: alias.format,
                                        });
                                }
                                let selected_view = if direct_volume_rt {
                                    bound_tex_views_3d[slot]
                                } else {
                                    bound_tex_views[slot]
                                };
                                let snapshotted =
                                    alias_snapshotted.get(slot).copied().unwrap_or(false);
                                trace_graphics_texture_binding!(
                                    call,
                                    slot,
                                    *pending,
                                    Some(GraphicsTextureTraceResource::rt_alias(alias)),
                                    texture_numeric_type_for_slot(
                                        &call.texture_numeric_manifest,
                                        slot,
                                    ),
                                    GraphicsTextureBindOutcome::RtAlias,
                                    format!(
                                        "rt-alias:{}:{}",
                                        alias.key.label(),
                                        if snapshotted { "snapshot" } else { "live" }
                                    ),
                                    format!("{selected_view:?}"),
                                    Some(format!("{view_format:?}")),
                                    None,
                                );
                                trace_vs_tex_bind_alias(
                                    device,
                                    *cmd_pool,
                                    *queue,
                                    rt_cache,
                                    mem_props,
                                    call,
                                    slot,
                                    alias,
                                    view_format,
                                    bound_tex_views[slot],
                                );
                                if !alias.depth
                                    && color_keys.contains(&alias.key)
                                    && !alias_snapshotted.get(slot).copied().unwrap_or(false)
                                {
                                    bound_tex_layouts[slot] = required_rt_layout;
                                }
                                continue;
                            }
                        }
                        if let Some((texture_key, tic, _, _)) = *pending {
                            if rt_alias_reinterpret_supported(texture_key, tic, alias, view_format)
                            {
                                let source_is_snapshot =
                                    alias_snapshotted.get(slot).copied().unwrap_or(false);
                                if pass_open
                                    && !source_is_snapshot
                                    && color_keys.contains(&alias.key)
                                {
                                    unsafe {
                                        device.cmd_end_rendering(cmd);
                                    }
                                    finish_color_pass(
                                        device,
                                        cmd,
                                        rt_cache,
                                        &color_bind,
                                        &mut color_layouts,
                                        pass_rt_layout,
                                        &mut pass_dirty,
                                        &pass_trace_calls,
                                    );
                                    pass_open = false;
                                    pass_trace_calls.clear();
                                }
                                if let Some((reinterpret_view, reinterpret_copy)) =
                                    prepare_rt_alias_reinterpret(
                                        device,
                                        mem_props,
                                        rt_cache,
                                        rt_reinterpret_cache,
                                        texture_key,
                                        tic,
                                        alias,
                                        view_format,
                                        source_is_snapshot,
                                    )?
                                {
                                    if reinterpret_copy.is_some() && pass_open {
                                        unsafe {
                                            device.cmd_end_rendering(cmd);
                                        }
                                        finish_color_pass(
                                            device,
                                            cmd,
                                            rt_cache,
                                            &color_bind,
                                            &mut color_layouts,
                                            pass_rt_layout,
                                            &mut pass_dirty,
                                            &pass_trace_calls,
                                        );
                                        pass_open = false;
                                        pass_trace_calls.clear();
                                    }
                                    if let Some(reinterpret_copy) = reinterpret_copy {
                                        record_rt_alias_reinterpret(
                                            device,
                                            cmd,
                                            mem_props,
                                            rt_cache,
                                            &mut frame_slots[cur_idx],
                                            rt_reinterpret_cache,
                                            reinterpret_copy,
                                        )?;
                                    }
                                    bound_tex_views[slot] = reinterpret_view;
                                    trace_graphics_texture_binding!(
                                        call,
                                        slot,
                                        *pending,
                                        Some(GraphicsTextureTraceResource::rt_alias(alias)),
                                        texture_numeric_type_for_slot(
                                            &call.texture_numeric_manifest,
                                            slot,
                                        ),
                                        GraphicsTextureBindOutcome::RtAlias,
                                        format!(
                                            "rt-reinterpret:{}:{}",
                                            alias.key.label(),
                                            if source_is_snapshot { "snapshot" } else { "live" }
                                        ),
                                        format!("{:?}", bound_tex_views[slot]),
                                        Some(format!("{view_format:?}")),
                                        None,
                                    );
                                    trace_vs_tex_bind_alias(
                                        device,
                                        *cmd_pool,
                                        *queue,
                                        rt_cache,
                                        mem_props,
                                        call,
                                        slot,
                                        alias,
                                        view_format,
                                        bound_tex_views[slot],
                                    );
                                    continue;
                                }
                            }
                        }
                    }
                }
                let Some((key, tic, pitch_size, read_size)) = *pending else {
                    trace_graphics_texture_binding!(
                        call,
                        slot,
                        None,
                        None,
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                        GraphicsTextureBindOutcome::Dummy,
                        "dummy-descriptor",
                        format!("{:?}", bound_tex_views[slot]),
                        None,
                        Some(
                            call.fs_tex_ids
                                .get(slot)
                                .map(|tex_id| vs_tex_dummy_reason(call, *tex_id, &read_guest))
                                .unwrap_or_else(|| "tex-slot-out-of-range".to_string()),
                        ),
                    );
                    trace_vs_tex_bind_dummy(call, slot, &read_guest);
                    continue;
                };
                let cur_gen = texture_generation(tic.gpu_va, read_size);
                let numeric_type =
                    texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot);
                let numeric_family = texture_numeric_index(numeric_type);
                let mut binding_failure_reason = None;
                let identity_volume =
                    key.volume && std::env::var_os("NEXIUM_VOLUME_IDENTITY").is_some();
                let volume_slices = if key.volume
                    && !identity_volume
                    && numeric_type == nexium_spirv::TextureNumericType::Float
                {
                    let sampled_key = sampled_rt_key_for_slot(call, slot);
                    find_volume_rt_slices(rt_cache, &tic, pitch_size, key.layers, sampled_key)
                } else {
                    None
                };
                if let Some(slices) = volume_slices.as_ref() {
                    trace_volume_rt_pixels(
                        device, *cmd_pool, *queue, rt_cache, mem_props, &tic, slices,
                    );
                }
                let force_refresh_early = force_refresh_texture(tic.gpu_va);
                let cache_fresh = tex_gen_gating_enabled()
                    && !force_refresh_early
                    && volume_slices.is_none()
                    && !identity_volume
                    && tex_cache.get(&key).map_or(false, |t| t.gen == cur_gen);
                let mut refresh_verified = false;
                let sampled_fresh =
                    !cache_fresh
                        && !force_refresh_early
                        && volume_slices.is_none()
                        && !identity_volume
                        && !key.volume
                        && tex_cache.get(&key).map_or(false, |t| {
                            if t.gen != cur_gen {
                                texstat_event(5, 0);
                                return false;
                            }
                            if t.verified.elapsed() < TEX_VERIFY_PERIOD {
                                return true;
                            }
                            let sam = *tex_sam_cache.entry((tic.gpu_va, read_size)).or_insert_with(
                                || hash_sampled_guest(&read_guest, tic.gpu_va, read_size),
                            );
                            match sam {
                                None => {
                                    texstat_event(3, 0);
                                    false
                                }
                                Some(h) if h != t.hash => {
                                    texstat_event(4, 0);
                                    false
                                }
                                Some(_) => {
                                    refresh_verified = true;
                                    true
                                }
                            }
                        });
                if refresh_verified {
                    if let Some(t) = tex_cache.get_mut(&key) {
                        t.verified = std::time::Instant::now();
                    }
                }
                let (raw, raw_hash): (Option<&[u8]>, Option<u64>) = if cache_fresh || sampled_fresh
                {
                    texstat_event(0, 0);
                    (None, None)
                } else {
                    let fresh_read = !tex_raw_cache.contains_key(&(tic.gpu_va, read_size));
                    let raw_entry = match tex_raw_cache.entry((tic.gpu_va, read_size)) {
                        Entry::Occupied(entry) => entry.into_mut(),
                        Entry::Vacant(entry) => {
                            entry.insert(read_guest(tic.gpu_va, read_size).map(|raw| {
                                let tex_hash = hash_sampled(&raw);
                                (tex_hash, raw)
                            }))
                        }
                    };
                    if fresh_read {
                        texstat_event(1, raw_entry.as_ref().map_or(0, |(_, r)| r.len()));
                    }
                    (
                        raw_entry.as_ref().map(|(_, raw)| raw.as_slice()),
                        raw_entry.as_ref().map(|(tex_hash, _)| *tex_hash),
                    )
                };
                if raw.is_some() || volume_slices.is_some() || identity_volume {
                    let mut tex_hash = raw_hash.unwrap_or_else(|| texture_seed_hash(&key));
                    if let Some(slices) = volume_slices.as_ref() {
                        tex_hash = volume_rt_slice_hash(tex_hash, slices);
                    }
                    let force_refresh = force_refresh_texture(tic.gpu_va);
                    let need_upload = force_refresh
                        || match tex_cache.get(&key) {
                            Some(t) => {
                                if key.volume && volume_slices.is_none() {
                                    raw_hash.map_or(false, |raw_hash| {
                                        if t.hash != raw_hash {
                                            t.gen != cur_gen
                                        } else {
                                            t.gen != cur_gen || t.hash != tex_hash
                                        }
                                    })
                                } else {
                                    t.gen != cur_gen || t.hash != tex_hash
                                }
                            }
                            None => true,
                        };
                    if need_upload {
                        if tic.is_srgb && std::env::var_os("NEXIUM_TIC_SRGB_LOG").is_some() {
                            log::warn!(
                                "[tic-srgb] batch {}x{} fmt={:?} bl={} va={:#x}",
                                tic.width,
                                tic.height,
                                tic.format,
                                tic.is_block_linear,
                                tic.gpu_va
                            );
                        }
                        let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                        let image_format = if identity_volume {
                            Ok(texture_image_format(
                                crate::texture::TicFormat::R8G8B8A8,
                                false,
                                false,
                                numeric_type,
                            ))
                        } else if let Some(slice) =
                            volume_slices.as_ref().and_then(|slices| slices.first())
                        {
                            Ok(slice.format)
                        } else {
                            texture_image_format_for_tic(&tic, numeric_type)
                        };
                        let image_format = match image_format {
                            Ok(format) => format,
                            Err(error) => {
                                log_graphics_texture_rejection(
                                    call,
                                    slot,
                                    *pending,
                                    numeric_type,
                                    "format",
                                    &error,
                                );
                                continue;
                            }
                        };
                        let upload = if volume_slices.is_some() {
                            Ok(TextureUploadData::base(Vec::new(), key.width, key.height))
                        } else if identity_volume {
                            Ok(TextureUploadData::base(
                                identity_volume_rgba8(key.width, key.height, key.layers),
                                key.width,
                                key.height,
                            ))
                        } else if let Some(raw) = raw {
                            texture_upload_data(
                                raw,
                                &tic,
                                key.layers,
                                pitch_size,
                                force_pitch,
                                image_format,
                            )
                        } else {
                            Ok(TextureUploadData::base(Vec::new(), key.width, key.height))
                        };
                        let upload = match upload {
                            Ok(upload) => upload,
                            Err(error) => {
                                log_graphics_texture_rejection(
                                    call,
                                    slot,
                                    *pending,
                                    numeric_type,
                                    "decode",
                                    &error,
                                );
                                continue;
                            }
                        };
                        let texels = &upload.bytes;
                        texstat_event(2, texels.len());
                        let dump_stats = std::env::var_os("NEXIUM_TEXDUMP")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                        let dump_img = std::env::var_os("NEXIUM_TEXDUMP_IMG")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                        let dump_rgba8 = if dump_stats || dump_img {
                            Some(
                                if image_format == vk::Format::R8G8B8A8_UNORM
                                    && volume_slices.is_none()
                                {
                                    upload.bytes.clone()
                                } else if identity_volume {
                                    identity_volume_rgba8(key.width, key.height, key.layers)
                                } else if let Some(raw) = raw {
                                    decode_texture_rgba8_layers(
                                        raw,
                                        &tic,
                                        key.layers,
                                        pitch_size,
                                        force_pitch,
                                    )
                                } else {
                                    Vec::new()
                                },
                            )
                        } else {
                            None
                        };
                        if dump_stats {
                            use std::sync::{Mutex, OnceLock};
                            static SEEN: OnceLock<Mutex<std::collections::HashSet<u64>>> =
                                OnceLock::new();
                            let s =
                                SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                            if s.lock().unwrap().insert(tic.gpu_va) {
                                let rgba8 = dump_rgba8.as_deref().unwrap_or(&[]);
                                let (mut sr, mut sg, mut sb, mut sa) = (0u64, 0u64, 0u64, 0u64);
                                let (mut amin, mut amax) = (255u8, 0u8);
                                for c in rgba8.chunks_exact(4) {
                                    sr += c[0] as u64;
                                    sg += c[1] as u64;
                                    sb += c[2] as u64;
                                    sa += c[3] as u64;
                                    amin = amin.min(c[3]);
                                    amax = amax.max(c[3]);
                                }
                                let n = (rgba8.len() / 4).max(1) as u64;
                                log::warn!(
                                    "TEXDUMP va={:#x} {}x{} fmt={:?} bl={} bh_log2={} read_size={} pitch_size={} pitchdst={} avg=({},{},{},{}) a=[{}..{}] raw16={:02x?}",
                                    tic.gpu_va, tic.width, tic.height, tic.format,
                                    tic.is_block_linear, tic.block_height_log2, read_size, pitch_size,
                                    crate::pitch_oracle::is_pitch_dst(tic.gpu_va),
                                    sr / n, sg / n, sb / n, sa / n, amin, amax,
                                    raw.map(|raw| &raw[..16.min(raw.len())]).unwrap_or(&[]),
                                );
                            }
                        }
                        if dump_img {
                            dump_texture_bmp_once(
                                tic.gpu_va,
                                tic.width,
                                tic.height,
                                key.layers,
                                dump_rgba8.as_deref().unwrap_or(&[]),
                                tic.swizzle,
                            );
                        }
                        if pass_open {
                            unsafe {
                                device.cmd_end_rendering(cmd);
                            }
                            finish_color_pass(
                                device,
                                cmd,
                                rt_cache,
                                &color_bind,
                                &mut color_layouts,
                                pass_rt_layout,
                                &mut pass_dirty,
                                &pass_trace_calls,
                            );
                            pass_open = false;
                            pass_trace_calls.clear();
                        }
                        match create_texture_image(
                            device,
                            cmd,
                            mem_props,
                            key.width,
                            key.height,
                            key.layers,
                            key.base_layer,
                            key.view_layers,
                            key.arrayed,
                            key.cube,
                            key.cube_array,
                            key.volume,
                            key.mip_levels,
                            key.base_mip,
                            key.view_mips,
                            texels,
                            &upload.copies,
                            volume_slices.as_deref(),
                            texture_view_swizzle(tic.format, numeric_type, tic.swizzle),
                            image_format,
                            tex_hash,
                            cur_gen,
                        ) {
                            Ok((tex, stage)) => {
                                if let Some(old) = tex_cache.insert(key, tex) {
                                    frame_slots[cur_idx].retired_textures.push(old);
                                }
                                if let Some((sbuf, smem)) = stage {
                                    frame_slots[cur_idx].retired_buffers.push((sbuf, smem));
                                }
                            }
                            Err(error) => {
                                if !bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
                                    log::warn!("texture upload failed: {}", error);
                                }
                                binding_failure_reason = Some(error);
                            }
                        }
                    } else if raw_hash.is_some() {
                        if let Some(t) = tex_cache.get_mut(&key) {
                            t.verified = std::time::Instant::now();
                        }
                    }
                }
                if key.cube_array {
                    bound_tex_views_cube_array[slot] = tex_cache
                        .get(&key)
                        .map(|t| t.view)
                        .unwrap_or(dummy_views_cube_array[slot][numeric_family]);
                } else if key.cube {
                    bound_tex_views_cube[slot] = tex_cache
                        .get(&key)
                        .map(|t| t.view)
                        .unwrap_or(dummy_views_cube[slot][numeric_family]);
                } else if key.volume {
                    bound_tex_views_3d[slot] = tex_cache
                        .get(&key)
                        .map(|t| t.view)
                        .unwrap_or(dummy_views_3d[slot][numeric_family]);
                } else {
                    let fallback_view = dummy_views_2d[slot][numeric_family];
                    bound_tex_views[slot] =
                        tex_cache.get(&key).map(|t| t.view).unwrap_or(fallback_view);
                }
                if slot < post_submit_texture_probe.len()
                    && !key.volume
                    && !key.cube
                    && !key.cube_array
                    && post_submit_texture_probe_enabled(call.fs_gpu_va)
                    && post_submit_texture_probe[slot].is_none()
                {
                    if let Some(texture) = tex_cache.get(&key) {
                        post_submit_texture_probe[slot] = Some(PostSubmitTextureProbe {
                            source: "TEX",
                            key: RtKey::new(0, key.width, key.height, key.gpu_va),
                            image: texture.image,
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            format: texture_image_format_for_tic(&tic, numeric_type)
                                .unwrap_or(vk::Format::UNDEFINED),
                        });
                    }
                }
                let cache_hit = tex_cache.get(&key).is_some();
                let selected_view = if key.cube_array {
                    bound_tex_views_cube_array[slot]
                } else if key.cube {
                    bound_tex_views_cube[slot]
                } else if key.volume {
                    bound_tex_views_3d[slot]
                } else {
                    bound_tex_views[slot]
                };
                let source = if key.cube_array {
                    "texture-cube-array"
                } else if key.cube {
                    "texture-cube"
                } else if key.volume {
                    "texture-3d"
                } else if key.arrayed {
                    "texture-2d-array"
                } else {
                    "texture-2d"
                };
                let (outcome, reason) = if cache_hit {
                    (GraphicsTextureBindOutcome::Success, None)
                } else if let Some(reason) = binding_failure_reason {
                    (GraphicsTextureBindOutcome::Rejection, Some(reason))
                } else {
                    (
                        GraphicsTextureBindOutcome::Dummy,
                        Some("resource data unavailable and no cached view".to_string()),
                    )
                };
                trace_graphics_texture_binding!(
                    call,
                    slot,
                    *pending,
                    None,
                    numeric_type,
                    outcome,
                    source,
                    format!("{selected_view:?}"),
                    texture_image_format_for_tic(&tic, numeric_type)
                        .ok()
                        .map(|format| format!("{format:?}")),
                    reason,
                );
                if bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
                    if key.volume {
                        if let Some(t) = tex_cache.get(&key) {
                            verify_volume_image(
                                device, *cmd_pool, *queue, mem_props, t.image, key.width,
                                key.height, key.layers, tic.gpu_va,
                            );
                        }
                    }
                }
                trace_vs_tex_bind_texture(
                    call,
                    slot,
                    key,
                    tic,
                    tex_cache.get(&key).is_some(),
                    if key.cube_array {
                        bound_tex_views_cube_array[slot]
                    } else if key.cube {
                        bound_tex_views_cube[slot]
                    } else if key.volume {
                        bound_tex_views_3d[slot]
                    } else {
                        bound_tex_views[slot]
                    },
                );
            }

            rp_tex += rp_tx0.elapsed();
            let rp_vx0 = std::time::Instant::now();
            let vertex_binds = upload_vertex_bindings(
                device,
                frame_slots,
                other_idx,
                descriptor_pool.pool,
                ubo_ring,
                &prep.vertex_bindings,
                false,
            )?;
            rp_vtx += rp_vx0.elapsed();

            let white_bind: Option<(u32, vk::Buffer, u64)> =
                if let Some(wb) = call.vertex_layout.bindings.iter().find(|b| b.stride == 0) {
                    if !ring_allocation_fits(ubo_ring, 16, 16) {
                        return Err(
                            "batched graphics ring preflight underestimated constant-attribute upload"
                                .to_string(),
                        );
                    }
                    let (wbuf, woff, wptr) = ring_alloc(ubo_ring, 16, 16)
                        .map_err(|e| format!("ring_alloc(white): {}", e))?;
                    unsafe {
                        let default = [0.0f32, 0.0, 0.0, 1.0];
                        std::ptr::copy_nonoverlapping(default.as_ptr() as *const u8, wptr, 16);
                    }
                    Some((wb.binding, wbuf, woff))
                } else {
                    None
                };

            let index_bind: Option<(vk::Buffer, u64)> =
                if prep.index_count > 0 && !prep.index_data.is_empty() {
                    let isz = align_up(prep.index_data.len() as u64, 4);
                    if !ring_allocation_fits(ubo_ring, isz, 4) {
                        return Err(format!(
                            "batched graphics ring preflight underestimated index upload ({isz:#x} bytes)"
                        ));
                    }
                    let (ibuf, ioff, iptr) = ring_alloc(ubo_ring, isz, 4)
                        .map_err(|e| format!("ring_alloc(index): {}", e))?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            prep.index_data.as_ptr(),
                            iptr,
                            prep.index_data.len(),
                        );
                    }
                    Some((ibuf, ioff))
                } else {
                    None
                };

            let (_cbuf_range, cbuf_size_aligned) = graphics_cbuf_allocation_size(
                prep.cbuf_data.len(),
                cbuf_alignment,
                *max_storage_buffer_range,
            )?;
            if !ring_allocation_fits(ubo_ring, cbuf_size_aligned, cbuf_alignment) {
                return Err(format!(
                    "batched graphics ring preflight underestimated graphics cbuf upload ({cbuf_size_aligned:#x} bytes, alignment {cbuf_alignment})"
                ));
            }
            let (ubo_buffer, ubo_offset, ubo_ptr) =
                ring_alloc(ubo_ring, cbuf_size_aligned, cbuf_alignment)
                    .map_err(|e| format!("ring_alloc(graphics-cbuf): {}", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    prep.cbuf_data.as_ptr(),
                    ubo_ptr,
                    prep.cbuf_data.len(),
                );
            }

            if bind_trace_fs(call.fs_gpu_va, call.fs_hash) {
                for binding in &call.texture_numeric_manifest {
                    let slot = binding.descriptor_slot as usize;
                    if slot < prep.tex_pendings.len() || slot >= max_texture_descriptors() {
                        continue;
                    }
                    let texel_buffer = descriptor_slot_masked(call.texel_buffer_mask, slot);
                    trace_graphics_texture_binding!(
                        call,
                        slot,
                        None,
                        None,
                        texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                        GraphicsTextureBindOutcome::Dummy,
                        if texel_buffer {
                            "dummy-texel-buffer"
                        } else {
                            "dummy-descriptor"
                        },
                        if texel_buffer {
                            format!("{:?}", bound_texel_views[slot])
                        } else {
                            format!("{:?}", bound_tex_views[slot])
                        },
                        None,
                        Some("manifest slot has no resolved TIC descriptor".to_string()),
                    );
                }
            }

            let rp_ds0 = std::time::Instant::now();
            let set_layouts = [descriptor_layout.layout];
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: dset_pools.current(),
                descriptor_set_count: 1,
                p_set_layouts: set_layouts.as_ptr(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            };
            let mut alloc_info = alloc_info;
            let dset = loop {
                match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
                    Ok(dsets) => break dsets[0],
                    Err(
                        vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL,
                    ) => {
                        let pool = match dset_pools.grow() {
                            Ok(pool) => pool,
                            Err(error) => {
                                unsafe {
                                    let _ = device.end_command_buffer(cmd);
                                }
                                let _ = reset_command_buffer(device, cmd);
                                return Err(error);
                            }
                        };
                        alloc_info.descriptor_pool = pool;
                    }
                    Err(e) => {
                        unsafe {
                            let _ = device.end_command_buffer(cmd);
                        }
                        let _ = reset_command_buffer(device, cmd);
                        return Err(format!("allocate_descriptor_sets: {:?}", e));
                    }
                }
            };
            let ubo_info = vk::DescriptorBufferInfo {
                buffer: ubo_buffer,
                offset: ubo_offset,
                range: prep.cbuf_data.len() as u64,
            };
            let image_infos = typed_sampled_image_infos(
                &bound_tex_views,
                Some(&bound_tex_layouts),
                &call.texture_numeric_manifest,
                &dummy_views_2d,
            );
            let image_infos_3d = typed_sampled_image_infos(
                &bound_tex_views_3d,
                None,
                &call.texture_numeric_manifest,
                &dummy_views_3d,
            );
            let image_infos_cube = typed_sampled_image_infos(
                &bound_tex_views_cube,
                None,
                &call.texture_numeric_manifest,
                &dummy_views_cube,
            );
            let image_infos_cube_array = typed_sampled_image_infos(
                &bound_tex_views_cube_array,
                None,
                &call.texture_numeric_manifest,
                &dummy_views_cube_array,
            );
            let typed_texel_views = typed_texel_buffer_views(
                &bound_texel_views,
                &call.texture_numeric_manifest,
                graphics_dummies.texel_buffer,
            );
            let mut bound_samplers = vec![default_samp; max_texture_descriptors()];
            for (slot, tsc) in prep.tsc_entries.iter().enumerate() {
                let Some(t) = *tsc else {
                    continue;
                };
                let integer_sample = texture_requires_integer_sampler(
                    texture_numeric_type_for_slot(&call.texture_numeric_manifest, slot),
                    stencil_alias_bound.get(slot).copied().unwrap_or(false),
                );
                let cache = if integer_sample {
                    &mut *integer_sampler_cache
                } else {
                    &mut *sampler_cache
                };
                bound_samplers[slot] = match cached_sampler_for_tsc(
                    device,
                    cache,
                    t,
                    integer_sample,
                    *sampler_filter_minmax_supported,
                    *sampler_anisotropy_supported,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!("tsc sampler create failed: {}", e);
                        default_samp
                    }
                };
            }
            let sampler_infos: Vec<vk::DescriptorImageInfo> = bound_samplers
                .iter()
                .map(|sampler| vk::DescriptorImageInfo {
                    sampler: *sampler,
                    image_view: vk::ImageView::null(),
                    image_layout: vk::ImageLayout::UNDEFINED,
                })
                .collect();
            let mut ssbo_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut ssbo_bindings: Vec<u32> = Vec::new();
            let mut ssbo_provided = [false; crate::descriptor::MAX_SSBO as usize];
            for (idx, data) in &call.ssbo_data {
                if *idx >= crate::descriptor::MAX_SSBO || data.is_empty() {
                    continue;
                }
                let sz = data.len() as u64;
                let sz_al = align_up(sz, 16);
                if !ring_allocation_fits(ubo_ring, sz_al, 16) {
                    return Err(format!(
                        "batched graphics ring preflight underestimated storage-buffer {} upload ({sz_al:#x} bytes)",
                        idx
                    ));
                }
                let (sbuf, soff, sptr) = ring_alloc(ubo_ring, sz_al, 16)
                    .map_err(|e| format!("ring_alloc(ssbo): {}", e))?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), sptr, data.len());
                }
                ssbo_infos.push(vk::DescriptorBufferInfo {
                    buffer: sbuf,
                    offset: soff,
                    range: sz,
                });
                ssbo_bindings.push(*idx);
                ssbo_provided[*idx as usize] = true;
            }
            if ssbo_provided.iter().any(|p| !p) {
                if !ring_allocation_fits(ubo_ring, 16, 16) {
                    return Err(
                        "batched graphics ring preflight underestimated dummy storage-buffer upload"
                            .to_string(),
                    );
                }
                let (dbuf, doff, dptr) = ring_alloc(ubo_ring, 16, 16)
                    .map_err(|e| format!("ring_alloc(ssbo-dummy): {}", e))?;
                unsafe {
                    std::ptr::write_bytes(dptr, 0, 16);
                }
                for i in 0..crate::descriptor::MAX_SSBO {
                    if !ssbo_provided[i as usize] {
                        ssbo_infos.push(vk::DescriptorBufferInfo {
                            buffer: dbuf,
                            offset: doff,
                            range: 16,
                        });
                        ssbo_bindings.push(i);
                    }
                }
            }
            let mut writes = vec![
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: dset,
                    dst_binding: crate::descriptor::CBUF_BINDING,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &ubo_info,
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    p_next: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: dset,
                    dst_binding: crate::descriptor::SAMPLER_BINDING,
                    dst_array_element: 0,
                    descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    p_image_info: sampler_infos.as_ptr(),
                    p_buffer_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    p_next: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                },
            ];
            for (i, binding) in ssbo_bindings.iter().enumerate() {
                writes.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: dset,
                    dst_binding: crate::descriptor::SSBO_BINDING_BASE + *binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: &ssbo_infos[i],
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    p_next: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                });
            }
            for (binding, infos) in crate::descriptor::SAMPLED_IMAGE_BINDINGS.into_iter().zip([
                &image_infos[0],
                &image_infos_3d[0],
                &image_infos_cube[0],
                &image_infos_cube_array[0],
                &image_infos[1],
                &image_infos_3d[1],
                &image_infos_cube[1],
                &image_infos_cube_array[1],
                &image_infos[2],
                &image_infos_3d[2],
                &image_infos_cube[2],
                &image_infos_cube_array[2],
            ]) {
                writes.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: dset,
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: infos.as_ptr(),
                    p_buffer_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    p_next: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                });
            }
            for (binding, views) in crate::descriptor::TEXEL_BUFFER_BINDINGS
                .into_iter()
                .zip([&typed_texel_views[0], &typed_texel_views[1], &typed_texel_views[2]])
            {
                writes.push(vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: dset,
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                    descriptor_type: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                    p_image_info: std::ptr::null(),
                    p_buffer_info: std::ptr::null(),
                    p_texel_buffer_view: views.as_ptr(),
                    p_next: std::ptr::null(),
                    _marker: std::marker::PhantomData,
                });
            }
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
            rp_dset += rp_ds0.elapsed();

            let need_depth = prep.use_depth;
            if !pass_open {
                for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
                    if let Some(layout) = rt_cache.color_layout(*key) {
                        color_layouts[idx] = layout;
                    }
                }
            }
            if !pass_open || pass_depth != need_depth || pass_rt_layout != required_rt_layout {
                if pass_open {
                    unsafe {
                        device.cmd_end_rendering(cmd);
                    }
                    finish_color_pass(
                        device,
                        cmd,
                        rt_cache,
                        &color_bind,
                        &mut color_layouts,
                        pass_rt_layout,
                        &mut pass_dirty,
                        &pass_trace_calls,
                    );
                    pass_trace_calls.clear();
                }
                let needs_color_transition = color_layouts
                    .iter()
                    .any(|layout| *layout != required_rt_layout);
                if needs_color_transition {
                    for (idx, (key, image, _, _, _)) in color_bind.iter().enumerate() {
                        transition_image(
                            device,
                            cmd,
                            *image,
                            color_layouts[idx],
                            required_rt_layout,
                        );
                        color_layouts[idx] = required_rt_layout;
                        rt_cache.set_color_layout(*key, required_rt_layout);
                    }
                }
                if had_pass {
                    if let Some(di) = depth_image {
                        if need_depth {
                            transition_image_aspect(
                                device,
                                cmd,
                                di,
                                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                call.depth_aspects,
                            );
                        }
                    }
                }
                let first = !had_pass;
                let clear_value = if first && clear_rt {
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                } else {
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: call.clear_color,
                        },
                    }
                };
                let depth_attachment = if need_depth {
                    let clear_now = depth_needs_clear;
                    depth_needs_clear = false;
                    if clear_now {
                        rt_cache.mark_depth_written(depth_key.unwrap());
                    }
                    let depth_clear_far = if call.depth.compare_op == vk::CompareOp::GREATER
                        || call.depth.compare_op == vk::CompareOp::GREATER_OR_EQUAL
                    {
                        0.0
                    } else {
                        call.clear_depth_hint
                    };
                    depth_view.map(|dv| vk::RenderingAttachmentInfo {
                        s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                        image_view: dv,
                        image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        resolve_mode: vk::ResolveModeFlags::NONE,
                        resolve_image_view: vk::ImageView::null(),
                        resolve_image_layout: vk::ImageLayout::UNDEFINED,
                        load_op: if clear_now {
                            vk::AttachmentLoadOp::CLEAR
                        } else {
                            vk::AttachmentLoadOp::LOAD
                        },
                        store_op: vk::AttachmentStoreOp::STORE,
                        clear_value: vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: depth_clear_far,
                                stencil: call.clear_stencil_hint,
                            },
                        },
                        p_next: std::ptr::null(),
                        _marker: std::marker::PhantomData,
                    })
                } else {
                    None
                };
                let p_depth_attachment = if call.depth_aspects.contains(vk::ImageAspectFlags::DEPTH)
                {
                    depth_attachment
                        .as_ref()
                        .map_or(std::ptr::null(), |a| a as *const _)
                } else {
                    std::ptr::null()
                };
                let p_stencil_attachment =
                    if call.depth_aspects.contains(vk::ImageAspectFlags::STENCIL) {
                        depth_attachment
                            .as_ref()
                            .map_or(std::ptr::null(), |a| a as *const _)
                    } else {
                        std::ptr::null()
                    };
                let attachments = color_bind
                    .iter()
                    .map(|(_, _, view, _, _)| vk::RenderingAttachmentInfo {
                        s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                        image_view: *view,
                        image_layout: required_rt_layout,
                        resolve_mode: vk::ResolveModeFlags::NONE,
                        resolve_image_view: vk::ImageView::null(),
                        resolve_image_layout: vk::ImageLayout::UNDEFINED,
                        load_op: if first && clear_rt {
                            vk::AttachmentLoadOp::CLEAR
                        } else {
                            vk::AttachmentLoadOp::LOAD
                        },
                        store_op: vk::AttachmentStoreOp::STORE,
                        clear_value,
                        p_next: std::ptr::null(),
                        _marker: std::marker::PhantomData,
                    })
                    .collect::<Vec<_>>();
                let render_layer_count = attachment_render_layer_count(
                    &color_keys,
                    if prep.use_depth { call.depth_key } else { None },
                )?;
                let render_info = vk::RenderingInfo {
                    s_type: vk::StructureType::RENDERING_INFO,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: rt_extent,
                    },
                    layer_count: render_layer_count,
                    view_mask: 0,
                    color_attachment_count: attachments.len() as u32,
                    p_color_attachments: attachments.as_ptr(),
                    p_depth_attachment,
                    p_stencil_attachment,
                    p_next: std::ptr::null(),
                    flags: Default::default(),
                    _marker: std::marker::PhantomData,
                };
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: rt_extent.width as f32,
                    height: rt_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor = draw_scissor(call.scissor, rt_extent);
                unsafe {
                    device.cmd_begin_rendering(cmd, &render_info);
                    device.cmd_set_viewport(cmd, 0, &[viewport]);
                    device.cmd_set_scissor(cmd, 0, &[scissor]);
                }
                pass_open = true;
                pass_depth = need_depth;
                pass_rt_layout = required_rt_layout;
                pass_dirty.fill(false);
                pass_trace_calls.clear();
                had_pass = true;
            }
            unsafe {
                let vp = match call.vp_rect {
                    Some([x, y, w, h]) => vk::Viewport {
                        x,
                        y,
                        width: w,
                        height: h,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                    None => vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: rt_extent.width as f32,
                        height: rt_extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                };
                device.cmd_set_viewport(cmd, 0, &[vp]);
                let scissor = draw_scissor(call.scissor, rt_extent);
                device.cmd_set_scissor(cmd, 0, &[scissor]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, prep.pipeline);
                set_dynamic_stencil_state(device, cmd, call.stencil);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_cache.layout,
                    0,
                    &[dset],
                    &[],
                );
                for (binding, vbuf, voff) in &vertex_binds {
                    device.cmd_bind_vertex_buffers(cmd, *binding, &[*vbuf], &[*voff]);
                }
                if let Some((wbinding, wbuf, woff)) = white_bind {
                    device.cmd_bind_vertex_buffers(cmd, wbinding, &[wbuf], &[woff]);
                }
                let cmd_first_vertex = if !vertex_binds.is_empty() {
                    0
                } else {
                    call.first_vertex
                };
                if let Some((ibuf, ioff)) = index_bind {
                    device.cmd_bind_index_buffer(cmd, ibuf, ioff, prep.index_type);
                    let vertex_offset = if !vertex_binds.is_empty() {
                        call.first_vertex as i32
                    } else {
                        0
                    };
                    device.cmd_draw_indexed(
                        cmd,
                        prep.index_count,
                        call.instance_count.max(1),
                        0,
                        vertex_offset,
                        call.first_instance,
                    );
                } else {
                    device.cmd_draw(
                        cmd,
                        prep.draw_vertex_count,
                        call.instance_count.max(1),
                        cmd_first_vertex,
                        call.first_instance,
                    );
                }
            }
            if prep.use_depth && call_writes_depth_stencil(call) {
                rt_cache.mark_depth_written(call.depth_key.unwrap());
            }
            for (idx, dirty) in pass_dirty.iter_mut().enumerate() {
                if call_writes_color(call, idx) {
                    *dirty = true;
                }
            }
            pass_trace_calls.push(call);
        }
        if pass_open {
            unsafe {
                device.cmd_end_rendering(cmd);
            }
            finish_color_pass(
                device,
                cmd,
                rt_cache,
                &color_bind,
                &mut color_layouts,
                pass_rt_layout,
                &mut pass_dirty,
                &pass_trace_calls,
            );
        }

        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(batch): {:?}", e))?;
        }
        let rp_record = rp_t3.elapsed();
        let rp_t4 = std::time::Instant::now();
        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        let rp_submit = rp_t4.elapsed();
        run_post_submit_texture_probe(
            device,
            *cmd_pool,
            *queue,
            mem_props,
            slot_fence,
            post_submit_texture_probe,
        );
        rprof_record(
            rp_draws,
            rp_prep,
            rp_lock,
            rp_fence,
            rp_record,
            rp_submit,
            rp_t0.elapsed(),
        );
        rprof_record_detail(rp_alias, rp_tex, rp_vtx, rp_dset);
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx]
            .retired_dset_pools
            .extend(dset_pools.into_raw());
        for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
            rt_cache.set_color_layout(*key, color_layouts[idx]);
        }
        if any_depth {
            if let Ok((d, _)) = rt_cache.get_or_create_depth(
                depth_key.unwrap(),
                device,
                calls[0].depth_format,
                calls[0].depth_aspects,
            ) {
                d.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
        }
        for (ak, is_depth) in alias_used {
            if is_depth {
                rt_cache.set_depth_layout(ak, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            } else if !color_keys.contains(&ak) {
                rt_cache.set_color_layout(ak, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        }
        let next_idx = other_idx;
        ubo_ring.slot_head[next_idx] = ubo_ring.head;
        *frame_index = next_idx;
        pipeline_cache.maybe_save(device);
        Ok(())
    }
}

fn pprof_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NEXIUM_RENDER_PROFILE").is_some())
}

fn pprof_record(w: u32, h: u32, raw: std::time::Duration, convert: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    if !pprof_enabled() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    static RAW: AtomicU64 = AtomicU64::new(0);
    static CONVERT: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    RAW.fetch_add(raw.as_nanos() as u64, Ordering::Relaxed);
    CONVERT.fetch_add(convert.as_nanos() as u64, Ordering::Relaxed);
    if n % 64 == 0 {
        log::warn!(
            "[pprof] presents={} dims={}x{} avg_ms raw={:.2} convert={:.2}",
            n,
            w,
            h,
            RAW.load(Ordering::Relaxed) as f64 / n as f64 / 1_000_000.0,
            CONVERT.load(Ordering::Relaxed) as f64 / n as f64 / 1_000_000.0
        );
    }
}

fn pprof_raw_lock(lock: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    if !pprof_enabled() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    static LOCK: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    LOCK.fetch_add(lock.as_nanos() as u64, Ordering::Relaxed);
    if n % 64 == 0 {
        log::warn!(
            "[pprof] raw_lock avg_ms={:.2}",
            LOCK.load(Ordering::Relaxed) as f64 / n as f64 / 1_000_000.0
        );
    }
}

fn rprof_record(
    draws: u64,
    prep: std::time::Duration,
    lock: std::time::Duration,
    fence: std::time::Duration,
    record: std::time::Duration,
    submit: std::time::Duration,
    total: std::time::Duration,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_RENDER_PROFILE").is_some()) {
        return;
    }
    static BATCHES: AtomicU64 = AtomicU64::new(0);
    static DRAWS: AtomicU64 = AtomicU64::new(0);
    static PREP: AtomicU64 = AtomicU64::new(0);
    static LOCK: AtomicU64 = AtomicU64::new(0);
    static FENCE: AtomicU64 = AtomicU64::new(0);
    static RECORD: AtomicU64 = AtomicU64::new(0);
    static SUBMIT: AtomicU64 = AtomicU64::new(0);
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    let n = BATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    DRAWS.fetch_add(draws, Ordering::Relaxed);
    PREP.fetch_add(prep.as_nanos() as u64, Ordering::Relaxed);
    LOCK.fetch_add(lock.as_nanos() as u64, Ordering::Relaxed);
    FENCE.fetch_add(fence.as_nanos() as u64, Ordering::Relaxed);
    RECORD.fetch_add(record.as_nanos() as u64, Ordering::Relaxed);
    SUBMIT.fetch_add(submit.as_nanos() as u64, Ordering::Relaxed);
    TOTAL.fetch_add(total.as_nanos() as u64, Ordering::Relaxed);
    if n % 64 == 0 {
        let ms = |v: &AtomicU64| v.load(Ordering::Relaxed) as f64 / n as f64 / 1_000_000.0;
        log::warn!(
            "[rprof] batches={} draws={} avg_ms prep={:.2} lock={:.2} fence={:.2} record={:.2} submit={:.2} total={:.2}",
            n,
            DRAWS.load(Ordering::Relaxed),
            ms(&PREP),
            ms(&LOCK),
            ms(&FENCE),
            ms(&RECORD),
            ms(&SUBMIT),
            ms(&TOTAL)
        );
    }
}

fn rprof_record_detail(
    alias: std::time::Duration,
    tex: std::time::Duration,
    vtx: std::time::Duration,
    dset: std::time::Duration,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_RENDER_PROFILE").is_some()) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    static ALIAS: AtomicU64 = AtomicU64::new(0);
    static TEX: AtomicU64 = AtomicU64::new(0);
    static VTX: AtomicU64 = AtomicU64::new(0);
    static DSET: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    ALIAS.fetch_add(alias.as_nanos() as u64, Ordering::Relaxed);
    TEX.fetch_add(tex.as_nanos() as u64, Ordering::Relaxed);
    VTX.fetch_add(vtx.as_nanos() as u64, Ordering::Relaxed);
    DSET.fetch_add(dset.as_nanos() as u64, Ordering::Relaxed);
    if n % 64 == 0 {
        let ms = |v: &AtomicU64| v.load(Ordering::Relaxed) as f64 / n as f64 / 1_000_000.0;
        log::warn!(
            "[rprof2] batches={} avg_ms alias={:.2} tex={:.2} vtx={:.2} dset={:.2}",
            n,
            ms(&ALIAS),
            ms(&TEX),
            ms(&VTX),
            ms(&DSET)
        );
    }
}

fn movie_trace_present_key_ring() -> &'static Mutex<VecDeque<RtKey>> {
    static KEYS: std::sync::OnceLock<Mutex<VecDeque<RtKey>>> = std::sync::OnceLock::new();
    KEYS.get_or_init(|| Mutex::new(VecDeque::with_capacity(8)))
}

fn record_movie_trace_present_key(key: RtKey) {
    if std::env::var_os("NEXIUM_MOVIE_DRAW_TRACE_ALL").is_none() {
        return;
    }
    let mut keys = movie_trace_present_key_ring().lock();
    keys.retain(|candidate| *candidate != key);
    keys.push_back(key);
    while keys.len() > 8 {
        keys.pop_front();
    }
}

pub fn movie_trace_present_keys() -> Vec<RtKey> {
    if std::env::var_os("NEXIUM_MOVIE_DRAW_TRACE_ALL").is_none() {
        return Vec::new();
    }
    movie_trace_present_key_ring()
        .lock()
        .iter()
        .copied()
        .collect()
}

fn trace_present_key(rt_cache: &RtCache, requested_key: RtKey, key: RtKey) {
    record_movie_trace_present_key(key);
    if std::env::var_os("NEXIUM_PRESENT_KEYS").is_none() {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    if seq % 60 != 0 {
        return;
    }
    let candidates: Vec<String> = rt_cache
        .present_candidates(requested_key)
        .into_iter()
        .map(|(k, stamp)| format!("{}#{}{}", k.label(), stamp, if k == key { "*" } else { "" }))
        .collect();
    let all: Vec<String> = rt_cache
        .debug_all()
        .into_iter()
        .map(|(k, stamp)| format!("{}#{}", k.label(), stamp))
        .collect();
    log::warn!(
        "present key seq={} requested={} resolved={} candidates=[{}] ALL=[{}]",
        seq,
        requested_key.label(),
        key.label(),
        candidates.join(", "),
        all.join(", ")
    );
}

#[derive(Default)]
struct RtImageStats {
    pixels: u64,
    raw_nonzero_bytes: u64,
    raw_nonzero_words: u64,
    raw_first_word: Option<(u32, u32, u32)>,
    raw_mid_word: Option<(u32, u32, u32)>,
    rgb_nonzero: u64,
    alpha_nonzero: u64,
    rgb_sum: u64,
    alpha_sum: u64,
    rgb_max: u8,
    bbox: Option<(u32, u32, u32, u32)>,
    first: Option<(u32, u32, [u8; 4])>,
    pixel_rows: Vec<String>,
    format: vk::Format,
}

fn dump_rt_bmp(key: RtKey, rgba: &[u8]) {
    use std::io::Write;
    if key.width == 0 || key.height == 0 {
        return;
    }
    {
        let (mut amin, mut amax, mut asum) = (255u8, 0u8, 0u64);
        let (mut rsum, mut gsum, mut bsum, mut nonblack, mut n) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for px in rgba.chunks_exact(4) {
            amin = amin.min(px[3]);
            amax = amax.max(px[3]);
            asum += px[3] as u64;
            rsum += px[0] as u64;
            gsum += px[1] as u64;
            bsum += px[2] as u64;
            if px[0] | px[1] | px[2] != 0 {
                nonblack += 1;
            }
            n += 1;
        }
        let d = n.max(1);
        log::warn!(
            "[rt-content] {} rgb_avg=({},{},{}) a=[{}..{}] a_avg={} nonblack={}/{} ({}%)",
            key.label(),
            rsum / d,
            gsum / d,
            bsum / d,
            amin,
            amax,
            asum / d,
            nonblack,
            n,
            nonblack * 100 / d
        );
    }
    let dir = nexium_common::paths::log_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!(
        "rt-{}-{}x{}-{:x}.bmp",
        key.nvmap_id, key.width, key.height, key.gpu_va
    ));
    let row_stride = ((key.width as usize * 3 + 3) / 4) * 4;
    let image_size = row_stride * key.height as usize;
    let file_size = 54 + image_size;
    let Ok(mut file) = std::fs::File::create(path) else {
        return;
    };
    let mut header = Vec::with_capacity(54);
    header.extend_from_slice(b"BM");
    header.extend_from_slice(&(file_size as u32).to_le_bytes());
    header.extend_from_slice(&[0u8; 4]);
    header.extend_from_slice(&54u32.to_le_bytes());
    header.extend_from_slice(&40u32.to_le_bytes());
    header.extend_from_slice(&(key.width as i32).to_le_bytes());
    header.extend_from_slice(&(key.height as i32).to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&24u16.to_le_bytes());
    header.extend_from_slice(&[0u8; 24]);
    if file.write_all(&header).is_err() {
        return;
    }
    let mut row = vec![0u8; row_stride];
    for y in (0..key.height as usize).rev() {
        row.fill(0);
        for x in 0..key.width as usize {
            let idx = (y * key.width as usize + x) * 4;
            if idx + 4 > rgba.len() {
                continue;
            }
            row[x * 3] = rgba[idx + 2];
            row[x * 3 + 1] = rgba[idx + 1];
            row[x * 3 + 2] = rgba[idx];
        }
        if file.write_all(&row).is_err() {
            return;
        }
    }
}

fn trace_rt_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    requested_key: RtKey,
    resolved_key: RtKey,
) {
    let Some(seq) = rt_stats_seq() else {
        return;
    };
    let mut stamps: HashMap<RtKey, u64> = HashMap::new();
    for (k, stamp) in rt_cache.debug_all() {
        stamps.insert(k, stamp);
    }
    for key in rt_stats_keys(rt_cache, requested_key, resolved_key) {
        let stamp = stamps.get(&key).copied().unwrap_or(0);
        match read_rt_image_stats(device, cmd_pool, queue, rt_cache, mem_props, key) {
            Some(stats) => {
                let pct = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_nonzero as f64 * 100.0 / stats.pixels as f64
                };
                let avg_rgb = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_sum as f64 / (stats.pixels as f64 * 3.0)
                };
                let avg_alpha = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.alpha_sum as f64 / stats.pixels as f64
                };
                let bbox = stats
                    .bbox
                    .map(|(x0, y0, x1, y1)| format!("{},{}-{},{}", x0, y0, x1, y1))
                    .unwrap_or_else(|| "-".to_string());
                let first = stats
                    .first
                    .map(|(x, y, rgba)| {
                        format!(
                            "{},{}:{:02x}{:02x}{:02x}{:02x}",
                            x, y, rgba[0], rgba[1], rgba[2], rgba[3]
                        )
                    })
                    .unwrap_or_else(|| "-".to_string());
                let raw_first = stats
                    .raw_first_word
                    .map(|(x, y, word)| format!("{},{}:{:08x}", x, y, word))
                    .unwrap_or_else(|| "-".to_string());
                let raw_mid = stats
                    .raw_mid_word
                    .map(|(x, y, word)| format!("{},{}:{:08x}", x, y, word))
                    .unwrap_or_else(|| "-".to_string());
                log::warn!(
                    "[rt-stats] seq={} key={} fmt={:?} stamp={} rawbnz={} rawwnz={} rawfirst={} rawmid={} rgbnz={}/{} ({:.2}%) anz={} avg_rgb={:.2} avg_a={:.2} max={} bbox={} first={}",
                    seq,
                    key.label(),
                    stats.format,
                    stamp,
                    stats.raw_nonzero_bytes,
                    stats.raw_nonzero_words,
                    raw_first,
                    raw_mid,
                    stats.rgb_nonzero,
                    stats.pixels,
                    pct,
                    stats.alpha_nonzero,
                    avg_rgb,
                    avg_alpha,
                    stats.rgb_max,
                    bbox,
                    first
                );
                for row in &stats.pixel_rows {
                    log::warn!(
                        "[rt-pixels] seq={} key={} stamp={} {}",
                        seq,
                        key.label(),
                        stamp,
                        row
                    );
                }
            }
            None => {
                log::warn!(
                    "[rt-stats] seq={} key={} stamp={} readback=failed",
                    seq,
                    key.label(),
                    stamp
                );
            }
        }
    }
    if std::env::var_os("NEXIUM_RT_STATS_DEPTH").is_some() {
        for (key, image, layout, format, aspects) in rt_cache.debug_depth_all() {
            if key.width < 256 || layout == vk::ImageLayout::UNDEFINED {
                continue;
            }
            match read_depth_image_stats(
                device, cmd_pool, queue, mem_props, key, image, layout, aspects,
            ) {
                Some((nonzero, min, max, total)) => {
                    log::warn!(
                        "[rt-depth-stats] seq={} key={} fmt={:?} nonzero={}/{} min={:#x} max={:#x}",
                        seq,
                        key.label(),
                        format,
                        nonzero,
                        total,
                        min,
                        max
                    );
                }
                None => {
                    log::warn!(
                        "[rt-depth-stats] seq={} key={} fmt={:?} readback=failed",
                        seq,
                        key.label(),
                        format
                    );
                }
            }
        }
    }
}

fn read_depth_image_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    key: RtKey,
    image: vk::Image,
    prev_layout: vk::ImageLayout,
    aspects: vk::ImageAspectFlags,
) -> Option<(usize, u32, u32, usize)> {
    if !aspects.contains(vk::ImageAspectFlags::DEPTH) {
        return None;
    }
    let total = (key.width as u64)
        .checked_mul(key.height as u64)?
        .checked_mul(4)?;
    let stage = create_staging_owned(device, mem_props, total).ok()?;
    let cleanup = |device: &ash::Device,
                   fence: Option<vk::Fence>,
                   cmd: Option<vk::CommandBuffer>,
                   stage: &StagingBuffer| unsafe {
        if let Some(c) = cmd {
            device.free_command_buffers(cmd_pool, &[c]);
        }
        if let Some(f) = fence {
            device.destroy_fence(f, None);
        }
        device.destroy_buffer(stage.buffer, None);
        device.free_memory(stage.memory, None);
    };
    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::empty(),
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(f) => f,
        Err(_) => {
            cleanup(device, None, None, &stage);
            return None;
        }
    };
    let cmd = match alloc_one_time_cmd(device, cmd_pool) {
        Ok(c) => c,
        Err(_) => {
            cleanup(device, Some(fence), None, &stage);
            return None;
        }
    };
    if begin_one_time(device, cmd).is_err() {
        cleanup(device, Some(fence), Some(cmd), &stage);
        return None;
    }
    transition_image_aspect(
        device,
        cmd,
        image,
        prev_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        aspects,
    );
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stage.buffer,
            &[copy],
        );
    }
    transition_image_aspect(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        prev_layout,
        aspects,
    );
    if end_one_time(device, cmd).is_err() || submit_with_fence(device, queue, cmd, fence).is_err() {
        cleanup(device, Some(fence), Some(cmd), &stage);
        return None;
    }
    let waited = unsafe { device.wait_for_fences(&[fence], true, 1_000_000_000) };
    if waited.is_err() {
        cleanup(device, Some(fence), Some(cmd), &stage);
        return None;
    }
    let mut nonzero = 0usize;
    let mut min = u32::MAX;
    let mut max = 0u32;
    let mut count = 0usize;
    unsafe {
        if let Ok(ptr) = device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
        {
            let words = std::slice::from_raw_parts(ptr as *const u32, total as usize / 4);
            for word in words {
                let depth = *word & 0x00ff_ffff;
                nonzero += usize::from(depth != 0);
                min = min.min(depth);
                max = max.max(depth);
            }
            count = words.len();
            device.unmap_memory(stage.memory);
        }
    }
    cleanup(device, Some(fence), Some(cmd), &stage);
    if count == 0 {
        return None;
    }
    Some((nonzero, min, max, count))
}

fn rt_stats_seq() -> Option<u64> {
    let enabled = std::env::var_os("NEXIUM_RT_STATS")?;
    if enabled.to_string_lossy().trim() == "0" {
        return None;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let start = std::env::var("NEXIUM_RT_STATS_START")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(0);
    if seq < start {
        return None;
    }
    let period = std::env::var("NEXIUM_RT_STATS_PERIOD")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(60)
        .max(1);
    if seq % period == 0 {
        Some(seq)
    } else {
        None
    }
}

fn rt_stats_keys(rt_cache: &RtCache, requested_key: RtKey, resolved_key: RtKey) -> Vec<RtKey> {
    let mut keys = Vec::new();
    push_unique_rt_key(&mut keys, resolved_key);
    for (k, _) in rt_cache.present_candidates(requested_key) {
        push_unique_rt_key(&mut keys, k);
    }
    if let Ok(list) = std::env::var("NEXIUM_RT_STATS_KEYS") {
        for item in list.split(',') {
            if let Some(k) = parse_rt_key(item.trim()) {
                push_unique_rt_key(&mut keys, k);
            }
        }
    }
    if let Ok(list) = std::env::var("NEXIUM_RT_STATS_NVMAPS") {
        let ids: Vec<u32> = list
            .split(',')
            .filter_map(|item| parse_u64_value(item.trim()).map(|v| v as u32))
            .collect();
        if !ids.is_empty() {
            let mut all = rt_cache.debug_all();
            all.retain(|(key, stamp)| *stamp != 0 && ids.contains(&key.nvmap_id));
            all.sort_by_key(|(_, stamp)| std::cmp::Reverse(*stamp));
            for (key, _) in all {
                push_unique_rt_key(&mut keys, key);
            }
        }
    }
    if let Ok(list) = std::env::var("NEXIUM_RT_STATS_DIMS") {
        let dims: Vec<(u32, u32)> = list
            .split(',')
            .filter_map(|item| {
                let (width, height) = item.trim().split_once('x')?;
                Some((
                    parse_u64_value(width)? as u32,
                    parse_u64_value(height)? as u32,
                ))
            })
            .collect();
        if !dims.is_empty() {
            let mut all = rt_cache.debug_all();
            all.retain(|(key, stamp)| *stamp != 0 && dims.contains(&(key.width, key.height)));
            all.sort_by_key(|(_, stamp)| std::cmp::Reverse(*stamp));
            for (key, _) in all {
                push_unique_rt_key(&mut keys, key);
            }
        }
    }
    let max_recent = std::env::var("NEXIUM_RT_STATS_MAX")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(12) as usize;
    let mut recent = rt_cache.debug_all();
    recent.retain(|(_, stamp)| *stamp != 0);
    recent.sort_by_key(|(_, stamp)| std::cmp::Reverse(*stamp));
    for (k, _) in recent.into_iter().take(max_recent) {
        push_unique_rt_key(&mut keys, k);
    }
    keys
}

fn push_unique_rt_key(keys: &mut Vec<RtKey>, key: RtKey) {
    if !keys.contains(&key) {
        keys.push(key);
    }
}

fn parse_rt_key(s: &str) -> Option<RtKey> {
    let (nvmap, dims) = s.split_once(':')?;
    let (width, height) = dims.split_once('x')?;
    let (height, gpu_va) = if let Some((height, addr)) = height.split_once('@') {
        (height, parse_u64_value(addr)?)
    } else {
        (height, 0)
    };
    Some(RtKey::new(
        parse_u64_value(nvmap)? as u32,
        parse_u64_value(width)? as u32,
        parse_u64_value(height)? as u32,
        gpu_va,
    ))
}

fn parse_u64_value(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        t.parse().ok()
    }
}

fn vertex_debug_index_summary(
    index_data: &[u8],
    index_count: u32,
    index_type: vk::IndexType,
) -> String {
    if index_count == 0 || index_data.is_empty() {
        return "empty".to_string();
    }
    let mut min_idx = u32::MAX;
    let mut max_idx = 0u32;
    let mut restarts = 0u32;
    let mut seen = 0u32;
    match index_type {
        vk::IndexType::UINT32 => {
            for c in index_data.chunks_exact(4).take(index_count as usize) {
                let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if v == u32::MAX {
                    restarts += 1;
                    continue;
                }
                seen += 1;
                min_idx = min_idx.min(v);
                max_idx = max_idx.max(v);
            }
        }
        _ => {
            for c in index_data.chunks_exact(2).take(index_count as usize) {
                let v = u16::from_le_bytes([c[0], c[1]]) as u32;
                if v == u16::MAX as u32 {
                    restarts += 1;
                    continue;
                }
                seen += 1;
                min_idx = min_idx.min(v);
                max_idx = max_idx.max(v);
            }
        }
    }
    if seen == 0 {
        format!("seen=0 restarts={}", restarts)
    } else {
        format!(
            "seen={} min={} max={} restarts={}",
            seen, min_idx, max_idx, restarts
        )
    }
}

fn vertex_debug_attr_summary(
    call: &crate::draw::Maxwell3dDrawCall,
    vertex_bindings: &[PreparedVertexBinding],
) -> String {
    let mut parts = Vec::new();
    for attr in call.vertex_layout.attrs.iter().take(16) {
        let Some(binding) = vertex_bindings.iter().find(|b| b.binding == attr.binding) else {
            continue;
        };
        if binding.stride == 0 {
            continue;
        }
        let Some(comps) = vertex_debug_float_components(attr.format) else {
            continue;
        };
        let stride = binding.stride as usize;
        let offset = attr.offset as usize;
        if offset + comps * 4 > stride {
            continue;
        }
        let vertices = (binding.data.len() / stride).min(call.vertex_count as usize);
        if vertices == 0 {
            continue;
        }
        let mut min_v = vec![f32::INFINITY; comps];
        let mut max_v = vec![f32::NEG_INFINITY; comps];
        let mut seen = 0usize;
        for vi in 0..vertices {
            let base = vi * stride + offset;
            let mut ok = true;
            for c in 0..comps {
                let off = base + c * 4;
                let v = f32::from_le_bytes([
                    binding.data[off],
                    binding.data[off + 1],
                    binding.data[off + 2],
                    binding.data[off + 3],
                ]);
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                min_v[c] = min_v[c].min(v);
                max_v[c] = max_v[c].max(v);
            }
            if ok {
                seen += 1;
            }
        }
        if seen == 0 {
            continue;
        }
        let ranges = (0..comps)
            .map(|i| format!("{:.3}..{:.3}", min_v[i], max_v[i]))
            .collect::<Vec<_>>()
            .join("/");
        parts.push(format!(
            "l{} b{}+{} {:?} n{} [{}]",
            attr.location, attr.binding, attr.offset, attr.format, seen, ranges
        ));
    }
    if parts.is_empty() {
        "[]".to_string()
    } else {
        parts.join(" ")
    }
}

fn vertex_debug_float_components(format: vk::Format) -> Option<usize> {
    match format {
        vk::Format::R32_SFLOAT => Some(1),
        vk::Format::R32G32_SFLOAT => Some(2),
        vk::Format::R32G32B32_SFLOAT => Some(3),
        vk::Format::R32G32B32A32_SFLOAT => Some(4),
        _ => None,
    }
}

fn rt_pixels_enabled() -> bool {
    std::env::var_os("NEXIUM_RT_PIXELS")
        .map(|v| v.to_string_lossy().trim() != "0")
        .unwrap_or(false)
}

fn volume_pixels_enabled() -> bool {
    std::env::var_os("NEXIUM_VOLUME_PIXELS")
        .map(|v| v.to_string_lossy().trim() != "0")
        .unwrap_or(false)
}

fn rt_pixel_limit(name: &str, default: u32, max: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .map(|v| v as u32)
        .unwrap_or(default)
        .clamp(1, max)
}

fn trace_rt_stamp(stamp: u64, rt_key: RtKey, calls: &[&crate::draw::Maxwell3dDrawCall]) {
    if std::env::var_os("NEXIUM_RT_STAMP_DBG").is_none() {
        return;
    }
    static NVMAPS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    let nvmaps = NVMAPS.get_or_init(|| {
        std::env::var("NEXIUM_RT_STAMP_NVMAPS")
            .ok()
            .map(|list| {
                list.split(',')
                    .filter_map(|item| parse_u64_value(item.trim()).map(|value| value as u32))
                    .collect()
            })
            .unwrap_or_default()
    });
    if !nvmaps.is_empty() && !nvmaps.contains(&rt_key.nvmap_id) {
        return;
    }
    let start = std::env::var("NEXIUM_RT_STAMP_START")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(0);
    let end = std::env::var("NEXIUM_RT_STAMP_END")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(u64::MAX);
    if stamp < start || stamp > end {
        return;
    }
    let mut parts = Vec::new();
    for (i, call) in calls.iter().take(24).enumerate() {
        let sampled = call
            .sampled_rt_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, key)| key.map(|k| format!("s{}={}", slot, k.label())))
            .collect::<Vec<_>>()
            .join(",");
        let vp = match call.vp_rect {
            Some([x, y, w, h]) => format!("{},{},{}x{}", x, y, w, h),
            None => "full".to_string(),
        };
        let sc = match call.scissor {
            Some([x, y, w, h]) => format!("{},{},{}x{}", x, y, w, h),
            None => "full".to_string(),
        };
        let formats = call
            .color_rt_formats
            .iter()
            .map(|format| format!("{:?}", format))
            .collect::<Vec<_>>()
            .join(",");
        parts.push(format!(
            "{}:vs={:#x} fs={:#x} topo={} v={} i={} tex={:?} sampled=[{}] fmts=[{}] blend={} cw={:#x} depth={}/{} vp={} sc={}",
            i,
            call.vs_gpu_va,
            call.fs_gpu_va,
            call.state.topology.as_raw(),
            call.state.vertex_count,
            call.state.index_count,
            call.fs_tex_ids,
            sampled,
            formats,
            call.blend.enabled,
            call.blend.color_write_mask.as_raw(),
            call.depth.test_enabled,
            call.depth.write_enabled,
            vp,
            sc
        ));
    }
    log::warn!(
        "[rt-stamp] stamp={} rt={} calls={} {}",
        stamp,
        rt_key.label(),
        calls.len(),
        parts.join(" | ")
    );
}

fn call_writes_color(call: &crate::draw::Maxwell3dDrawCall, idx: usize) -> bool {
    call.blend
        .attachments
        .get(idx)
        .is_some_and(|att| !att.color_write_mask.is_empty())
}

fn call_writes_any_color(call: &crate::draw::Maxwell3dDrawCall) -> bool {
    if legacy_color_attachments() {
        return true;
    }
    if call.clear {
        return true;
    }
    let count = call.color_rt_keys.len().max(1).min(8);
    (0..count).any(|idx| call_writes_color(call, idx))
}

fn call_writes_depth_stencil(call: &crate::draw::Maxwell3dDrawCall) -> bool {
    let writes_depth = call.depth_aspects.contains(vk::ImageAspectFlags::DEPTH)
        && call.depth.test_enabled
        && call.depth.write_enabled;
    let face_writes = |face: crate::draw::StencilFaceState| {
        face.write_mask != 0
            && (face.fail_op != vk::StencilOp::KEEP
                || face.pass_op != vk::StencilOp::KEEP
                || face.depth_fail_op != vk::StencilOp::KEEP)
    };
    let writes_stencil = call.depth_aspects.contains(vk::ImageAspectFlags::STENCIL)
        && call.stencil.enabled
        && (face_writes(call.stencil.front) || face_writes(call.stencil.back));
    writes_depth || writes_stencil
}

fn active_color_keys_for_call(call: &crate::draw::Maxwell3dDrawCall) -> Vec<RtKey> {
    if !call_writes_any_color(call) {
        Vec::new()
    } else if call.color_rt_keys.is_empty() {
        vec![call.rt_key]
    } else {
        call.color_rt_keys.clone()
    }
}

fn attachment_render_layer_count(
    color_keys: &[RtKey],
    depth_key: Option<RtKey>,
) -> Result<u32, String> {
    let mut layer_count = color_keys
        .first()
        .copied()
        .map(RtKey::render_layer_count)
        .or_else(|| depth_key.map(RtKey::render_layer_count))
        .unwrap_or(1);
    layer_count = layer_count.max(1);
    if color_keys
        .iter()
        .copied()
        .any(|key| key.render_layer_count() != layer_count)
        || depth_key.is_some_and(|key| key.render_layer_count() != layer_count)
    {
        return Err("mixed 2D/layered render-target attachments are unsupported".to_string());
    }
    Ok(layer_count)
}

fn legacy_color_attachments() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_LEGACY_COLOR_ATTACH").is_some())
}

fn finish_color_pass(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    color_bind: &[(
        RtKey,
        vk::Image,
        vk::ImageView,
        vk::Extent2D,
        vk::ImageLayout,
    )],
    color_layouts: &mut [vk::ImageLayout],
    pass_rt_layout: vk::ImageLayout,
    pass_dirty: &mut [bool],
    trace_calls: &[&crate::draw::Maxwell3dDrawCall],
) {
    for (idx, (key, image, _, _, _)) in color_bind.iter().enumerate() {
        barrier_color_attachment_after_pass(device, cmd, *image, pass_rt_layout);
        color_layouts[idx] = pass_rt_layout;
        rt_cache.set_color_layout(*key, pass_rt_layout);
        if pass_dirty.get(idx).copied().unwrap_or(false) {
            let stamp = rt_cache.mark_drawn(*key);
            if let Some(c) = trace_calls.last() {
                rt_cache.record_present_flip(*key, c.present_flip_y);
            }
            trace_rt_stamp(stamp, *key, trace_calls);
        }
    }
    pass_dirty.fill(false);
}

fn trace_volume_rt_pixels(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    tic: &crate::texture::TicEntry,
    slices: &[VolumeRtSlice],
) {
    if !volume_pixels_enabled() {
        return;
    }
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if !seen.lock().unwrap().insert(tic.gpu_va) {
        return;
    }
    let max_layers = std::env::var("NEXIUM_VOLUME_PIXELS_LAYERS")
        .ok()
        .and_then(|s| parse_u64_value(&s))
        .unwrap_or(8)
        .clamp(1, 64) as usize;
    for slice in slices.iter().take(max_layers) {
        let stamp = slice.stamp;
        match read_rt_image_stats(device, cmd_pool, queue, rt_cache, mem_props, slice.key) {
            Some(stats) => {
                let pct = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_nonzero as f64 * 100.0 / stats.pixels as f64
                };
                let avg_rgb = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_sum as f64 / (stats.pixels as f64 * 3.0)
                };
                let avg_alpha = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.alpha_sum as f64 / stats.pixels as f64
                };
                let bbox = stats
                    .bbox
                    .map(|(x0, y0, x1, y1)| format!("{},{}-{},{}", x0, y0, x1, y1))
                    .unwrap_or_else(|| "-".to_string());
                let first = stats
                    .first
                    .map(|(x, y, rgba)| {
                        format!(
                            "{},{}:{:02x}{:02x}{:02x}{:02x}",
                            x, y, rgba[0], rgba[1], rgba[2], rgba[3]
                        )
                    })
                    .unwrap_or_else(|| "-".to_string());
                log::warn!(
                    "[volume-pixels] va={:#x} slice={} key={} stamp={} rgbnz={}/{} ({:.2}%) anz={} avg_rgb={:.2} avg_a={:.2} max={} bbox={} first={} fmt={:?}",
                    tic.gpu_va,
                    slice.layer,
                    slice.key.label(),
                    stamp,
                    stats.rgb_nonzero,
                    stats.pixels,
                    pct,
                    stats.alpha_nonzero,
                    avg_rgb,
                    avg_alpha,
                    stats.rgb_max,
                    bbox,
                    first,
                    slice.format
                );
                for row in &stats.pixel_rows {
                    log::warn!(
                        "[volume-pixel-row] va={:#x} slice={} key={} {}",
                        tic.gpu_va,
                        slice.layer,
                        slice.key.label(),
                        row
                    );
                }
            }
            None => {
                log::warn!(
                    "[volume-pixels] va={:#x} slice={} key={} readback=miss fmt={:?}",
                    tic.gpu_va,
                    slice.layer,
                    slice.key.label(),
                    slice.format
                );
            }
        }
    }
}

struct PreparedComputeSample {
    binding: u32,
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
    descriptor_layout: vk::ImageLayout,
    old_layout: vk::ImageLayout,
    aspects: vk::ImageAspectFlags,
}

enum ComputeSamplePlan {
    Live(PreparedComputeSample),
    CrossAccess {
        binding: u32,
        sampler: vk::Sampler,
        output_index: usize,
        components: vk::ComponentMapping,
    },
    Guest {
        binding: u32,
        sampler: vk::Sampler,
        width: u32,
        height: u32,
        depth: u32,
        is_3d: bool,
        format: vk::Format,
        components: vk::ComponentMapping,
        mip_levels: u32,
        view_base_mip: u32,
        view_mip_levels: u32,
        upload_bytes: Vec<u8>,
        upload_copies: Vec<TextureMipCopy>,
    },
}

struct PreparedComputeImage {
    binding: u32,
    image: vk::Image,
    view: vk::ImageView,
    descriptor_layout: vk::ImageLayout,
    old_layout: vk::ImageLayout,
    aspects: vk::ImageAspectFlags,
}

fn compute_sampled_image_alias_extent_matches(
    alias: RtAlias,
    tic: crate::texture::TicEntry,
    is_3d: bool,
    depth: u32,
) -> bool {
    alias.key.is_3d == is_3d
        && alias.key.width == tic.width
        && alias.key.height == tic.height
        && (!is_3d || alias.key.depth == depth)
}

fn prepare_compute_live_sampled_image(
    inner: &mut RendererInner,
    binding: u32,
    sample_type: crate::compute::ComputeSampleType,
    tic: crate::texture::TicEntry,
    is_3d: bool,
    depth: u32,
    numeric_type: nexium_spirv::TextureNumericType,
    alias: RtAlias,
) -> Result<PreparedComputeImage, String> {
    if alias.layout == vk::ImageLayout::UNDEFINED {
        return Err(format!(
            "live render target at sampled-image binding {binding} has undefined contents"
        ));
    }
    if !compute_sampled_image_alias_extent_matches(alias, tic, is_3d, depth) {
        return Err(format!(
            "live render target at sampled-image binding {binding} has incompatible extent: requested={}x{}x{} 3d={} TIC-va={:#x}, candidate={} 3d={}",
            tic.width,
            tic.height,
            depth,
            is_3d,
            tic.gpu_va,
            alias.key.label(),
            alias.key.is_3d,
        ));
    }
    let view_format = if alias.depth {
        alias.format
    } else {
        rt_alias_view_format(alias.key, tic, alias.format)
    };
    if !rt_alias_numeric_type_matches(alias, Some(tic), numeric_type, view_format) {
        return Err(format!(
            "live render target at sampled-image binding {binding} has incompatible numeric type: requested={sample_type:?} TIC={:?}/{:?}, candidate={} format={:?} view={view_format:?}",
            tic.format,
            tic.component_types,
            alias.key.label(),
            alias.format,
        ));
    }
    let view = {
        let RendererInner {
            device,
            rt_cache,
            utility_slot,
            ..
        } = inner;
        rt_alias_sample_view(
            device,
            rt_cache,
            utility_slot,
            alias,
            Some(tic),
            view_format,
        )
    }
    .ok_or_else(|| {
        format!(
            "could not create live sampled view for compute binding {binding}: TIC-va={:#x} candidate={} format={view_format:?}",
            tic.gpu_va,
            alias.key.label(),
        )
    })?;
    Ok(PreparedComputeImage {
        binding,
        image: alias.image,
        view,
        descriptor_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        old_layout: alias.layout,
        aspects: alias.aspects,
    })
}

enum ComputeImagePlan {
    Live(PreparedComputeImage),
    CrossAccess {
        binding: u32,
        output_index: usize,
        components: vk::ComponentMapping,
    },
    Guest {
        binding: u32,
        width: u32,
        height: u32,
        depth: u32,
        is_3d: bool,
        format: vk::Format,
        components: vk::ComponentMapping,
        mip_levels: u32,
        view_base_mip: u32,
        view_mip_levels: u32,
        upload_bytes: Vec<u8>,
        upload_copies: Vec<TextureMipCopy>,
    },
}

fn compute_cross_access_output_index(
    dispatch: &crate::compute::ComputeDispatch,
    sampled_binding: u32,
) -> Option<usize> {
    let storage_binding = dispatch
        .image_aliases
        .iter()
        .find(|alias| alias.sampled_binding == sampled_binding)?
        .storage_binding;
    dispatch
        .outputs
        .iter()
        .position(|output| output.binding == storage_binding)
}

fn compute_cross_access_view_components(
    sampled_binding: u32,
    tic: &crate::texture::TicEntry,
    sample_type: crate::compute::ComputeSampleType,
    output: &crate::compute::ComputeStorageImage,
) -> Result<vk::ComponentMapping, String> {
    let is_3d = tic_is_volume(tic);
    let depth = if is_3d { tic.depth.max(1) } else { 1 };
    let view_mip = tic.view_base_mip();
    let width = (tic.width >> view_mip).max(1);
    let height = (tic.height >> view_mip).max(1);
    let view_depth = if is_3d { (depth >> view_mip).max(1) } else { 1 };
    if tic.is_buffer()
        || !matches!(tic.texture_type, 1 | 2)
        || tic.base_layer != 0
        || tic.view_mip_levels() != 1
        || (is_3d && (tic.mip_levels() != 1 || view_mip != 0))
        || output.width != width
        || output.height != height
        || output.depth != view_depth
        || output.is_3d != is_3d
    {
        return Err(format!(
            "compute sampled/storage alias {} has incompatible view/output geometry: view={}x{}x{} 3d={} mip={}/{} output={}x{}x{} 3d={}",
            sampled_binding,
            width,
            height,
            view_depth,
            is_3d,
            view_mip,
            tic.view_mip_levels(),
            output.width,
            output.height,
            output.depth,
            output.is_3d
        ));
    }
    if output.initial_bytes.is_none() {
        return Err(format!(
            "compute sampled/storage alias {} has no defined initial contents",
            sampled_binding
        ));
    }
    let numeric_type = sample_type.spirv_type();
    let sampled_format = texture_image_format_for_tic(tic, numeric_type)?;
    let storage_format = output.format.vk_format();
    if !compute_cross_access_formats_compatible(sampled_format, storage_format) {
        return Err(format!(
            "compute sampled/storage alias {} has incompatible formats {:?}/{:?}",
            sampled_binding, sampled_format, storage_format
        ));
    }
    let swizzle = texture_view_swizzle(tic.format, numeric_type, tic.swizzle);
    Ok(texture_component_mapping(swizzle))
}

fn compute_cross_access_formats_compatible(sampled: vk::Format, storage: vk::Format) -> bool {
    if sampled == storage {
        return true;
    }
    matches!(
        (sampled, storage),
        (
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::A8B8G8R8_UNORM_PACK32
        ) | (
            vk::Format::A8B8G8R8_UNORM_PACK32,
            vk::Format::R8G8B8A8_UNORM
        ) | (
            vk::Format::R8G8B8A8_SNORM,
            vk::Format::A8B8G8R8_SNORM_PACK32
        ) | (
            vk::Format::A8B8G8R8_SNORM_PACK32,
            vk::Format::R8G8B8A8_SNORM
        ) | (vk::Format::R8G8B8A8_UINT, vk::Format::A8B8G8R8_UINT_PACK32)
            | (vk::Format::A8B8G8R8_UINT_PACK32, vk::Format::R8G8B8A8_UINT)
            | (vk::Format::R8G8B8A8_SINT, vk::Format::A8B8G8R8_SINT_PACK32)
            | (vk::Format::A8B8G8R8_SINT_PACK32, vk::Format::R8G8B8A8_SINT)
    )
}

fn validate_compute_cross_access_aliases(
    dispatch: &crate::compute::ComputeDispatch,
) -> Result<(), String> {
    for (index, alias) in dispatch.image_aliases.iter().enumerate() {
        if dispatch.image_aliases[..index]
            .iter()
            .any(|previous| previous.sampled_binding == alias.sampled_binding)
        {
            return Err(format!(
                "compute sampled binding {} declares more than one storage alias",
                alias.sampled_binding
            ));
        }
        let sampled_rt = dispatch
            .sampled_rts
            .iter()
            .find(|sampled| sampled.binding == alias.sampled_binding);
        let sampled_image = dispatch
            .sampled_images
            .iter()
            .find(|sampled| sampled.binding == alias.sampled_binding);
        let sampled_count =
            usize::from(sampled_rt.is_some()) + usize::from(sampled_image.is_some());
        if sampled_count != 1 {
            return Err(format!(
                "compute image alias references unknown or ambiguous sampled binding {}",
                alias.sampled_binding
            ));
        }
        let mut outputs = dispatch
            .outputs
            .iter()
            .filter(|output| output.binding == alias.storage_binding);
        let output = outputs.next().ok_or_else(|| {
            format!(
                "compute image alias references unknown storage binding {}",
                alias.storage_binding
            )
        })?;
        if outputs.next().is_some() {
            return Err(format!(
                "compute image alias references ambiguous storage binding {}",
                alias.storage_binding
            ));
        }
        if let Some(sampled) = sampled_rt {
            let _ = compute_cross_access_view_components(
                sampled.binding,
                &sampled.tic,
                sampled.sample_type,
                output,
            )?;
        } else if let Some(sampled) = sampled_image {
            let _ = compute_cross_access_view_components(
                sampled.binding,
                &sampled.tic,
                sampled.sample_type,
                output,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ComputeSampleTransition {
    image: vk::Image,
    old_layout: vk::ImageLayout,
    aspects: vk::ImageAspectFlags,
}

#[derive(Default)]
struct PreparedComputeResources {
    uniforms: Vec<crate::compute::ComputeBufferResource>,
    texels: Vec<crate::compute::ComputeBufferResource>,
    uniform_texels: Vec<crate::compute::ComputeBufferResource>,
    guest_images: Vec<crate::compute::ComputeGuestImageResource>,
    sampled_alias_views: Vec<vk::ImageView>,
    outputs: Vec<crate::compute::ComputeImageResource>,
    output_uploads: Vec<Option<crate::compute::ComputeBufferResource>>,
    readbacks: Vec<crate::compute::ComputeBufferResource>,
}

impl PreparedComputeResources {
    fn destroy(mut self, device: &ash::Device) {
        for buffer in self.uniforms.drain(..) {
            buffer.destroy(device);
        }
        for buffer in self.texels.drain(..) {
            buffer.destroy(device);
        }
        for buffer in self.uniform_texels.drain(..) {
            buffer.destroy(device);
        }
        for image in self.guest_images.drain(..) {
            image.destroy(device);
        }
        for view in self.sampled_alias_views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }
        for output in self.outputs.drain(..) {
            output.destroy(device);
        }
        for upload in self.output_uploads.drain(..).flatten() {
            upload.destroy(device);
        }
        for readback in self.readbacks.drain(..) {
            readback.destroy(device);
        }
    }
}

fn compute_readback_trace_enabled() -> bool {
    std::env::var_os("NEXIUM_COMPUTE_READBACK_TRACE").is_some_and(|value| {
        let value = value.to_string_lossy();
        let value = value.trim();
        !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
    })
}

fn acquire_compute_slot(
    inner: &mut RendererInner,
) -> Result<(vk::CommandBuffer, vk::Fence), String> {
    if let Some(slot) = inner.compute_slot_pool.pop() {
        return Ok(slot);
    }
    let cmd = alloc_one_time_cmd(&inner.device, inner.cmd_pool)?;
    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    match unsafe { inner.device.create_fence(&fence_info, None) } {
        Ok(fence) => Ok((cmd, fence)),
        Err(error) => {
            unsafe { inner.device.free_command_buffers(inner.cmd_pool, &[cmd]) };
            Err(format!("create_fence(compute slot): {:?}", error))
        }
    }
}

fn free_compute_descriptor_set(
    inner: &RendererInner,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
) {
    if set != vk::DescriptorSet::null() {
        unsafe {
            let _ = inner.device.free_descriptor_sets(pool, &[set]);
        }
    }
}

fn settle_all_pending_computes(inner: &mut RendererInner) {
    for index in 0..inner.pending_computes.len() {
        settle_pending_compute(inner, index);
    }
}

fn settle_pending_compute(inner: &mut RendererInner, index: usize) {
    if inner.pending_computes[index].result.is_some() {
        return;
    }
    let mut pending = inner.pending_computes.remove(index);
    let result = settle_pending_compute_record(inner, &mut pending);
    pending.result = Some(result);
    inner.pending_computes.insert(index, pending);
}

fn settle_pending_compute_record(
    inner: &mut RendererInner,
    pending: &mut PendingCompute,
) -> Result<crate::compute::ComputeDispatchResult, String> {
    use crate::compute::{ComputeDispatchResult, ComputeImageReadback, ComputeTexelReadback};

    let Some(mut resources) = pending.resources.take() else {
        return Err(format!(
            "pending compute dispatch {} has no transient resources",
            pending.id
        ));
    };
    if let Err(error) = wait_fence(&inner.device, pending.fence) {
        match unsafe { inner.device.device_wait_idle() } {
            Ok(()) => {
                resources.destroy(&inner.device);
                free_compute_descriptor_set(inner, pending.descriptor_pool, pending.descriptor_set);
                inner.compute_slot_pool.push((pending.cmd, pending.fence));
                return Err(error);
            }
            Err(idle_error) => {
                std::mem::forget(resources);
                if let Some(backend) = inner.compute_backend.take() {
                    std::mem::forget(backend);
                }
                let failure = format!(
                    "{error}; device_wait_idle after deferred compute wait failure also failed: \
                     {idle_error:?}; generic compute backend poisoned and in-flight resources retained"
                );
                inner.compute_unavailable_reason = failure.clone();
                return Err(failure);
            }
        }
    }
    let trace = compute_readback_trace_enabled();
    let mut image_readbacks = Vec::with_capacity(pending.outputs.len());
    for (resource_index, (meta, readback)) in
        pending.outputs.iter().zip(&resources.readbacks).enumerate()
    {
        let bytes = match readback.read(&inner.device, meta.byte_len) {
            Ok(bytes) => bytes,
            Err(error) => {
                resources.destroy(&inner.device);
                free_compute_descriptor_set(inner, pending.descriptor_pool, pending.descriptor_set);
                inner.compute_slot_pool.push((pending.cmd, pending.fence));
                return Err(error);
            }
        };
        if trace {
            let nonzero = bytes.iter().filter(|byte| **byte != 0).count();
            let max = bytes.iter().copied().max().unwrap_or(0);
            let min = bytes.iter().copied().min().unwrap_or(0);
            log::warn!(
                "[compute-readback] program={:#x} binding={} resource={} bytes={} nonzero={} min={:#x} max={:#x}",
                pending.program_key,
                meta.binding,
                resource_index,
                bytes.len(),
                nonzero,
                min,
                max,
            );
        }
        image_readbacks.push(ComputeImageReadback {
            resource_index,
            binding: meta.binding,
            bytes,
            width: meta.width,
            height: meta.height,
            depth: meta.depth,
            format: meta.format,
        });
    }
    let mut texel_readbacks = Vec::with_capacity(pending.texels.len());
    for meta in &pending.texels {
        match resources.texels[meta.resource_index].read(&inner.device, meta.byte_len) {
            Ok(bytes) => texel_readbacks.push(ComputeTexelReadback {
                resource_index: meta.resource_index,
                bytes,
            }),
            Err(error) => {
                resources.destroy(&inner.device);
                free_compute_descriptor_set(inner, pending.descriptor_pool, pending.descriptor_set);
                inner.compute_slot_pool.push((pending.cmd, pending.fence));
                return Err(error);
            }
        }
    }
    let uniforms = std::mem::take(&mut resources.uniforms);
    let outputs = std::mem::take(&mut resources.outputs);
    let readbacks = std::mem::take(&mut resources.readbacks);
    resources.destroy(&inner.device);
    {
        let RendererInner {
            device,
            compute_backend,
            ..
        } = &mut *inner;
        match compute_backend.as_mut() {
            Some(backend) => backend.recycle_dispatch_resources(device, uniforms, outputs, readbacks),
            None => {
                for resource in uniforms {
                    resource.destroy(device);
                }
                for resource in outputs {
                    resource.destroy(device);
                }
                for resource in readbacks {
                    resource.destroy(device);
                }
            }
        }
    }
    free_compute_descriptor_set(inner, pending.descriptor_pool, pending.descriptor_set);
    inner.compute_slot_pool.push((pending.cmd, pending.fence));
    Ok(ComputeDispatchResult {
        image_readbacks,
        texel_readbacks,
    })
}

fn compute_guest_sampler_format_features(
    tsc: &crate::texture::TscEntry,
    integer_sample: bool,
    sampler_filter_minmax_supported: bool,
    sampler_anisotropy_supported: bool,
    force_linear: bool,
) -> vk::FormatFeatureFlags2 {
    let mut required =
        vk::FormatFeatureFlags2::SAMPLED_IMAGE | vk::FormatFeatureFlags2::TRANSFER_DST;
    if integer_sample {
        return required;
    }

    let anisotropy = sampler_anisotropy_supported && tsc.max_anisotropy() > 1.0;
    if force_linear
        || matches!(tsc.mag_filter, crate::texture::TexFilter::Linear)
        || matches!(tsc.min_filter, crate::texture::TexFilter::Linear)
        || matches!(tsc.mip_filter, crate::texture::TexFilter::Linear)
        || anisotropy
    {
        required |= vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR;
    }
    if sampler_filter_minmax_supported
        && !matches!(
            tsc.reduction,
            crate::texture::SamplerReduction::WeightedAverage
        )
    {
        required |= vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX;
    }
    if tsc.depth_compare_enabled {
        required |= vk::FormatFeatureFlags2::SAMPLED_IMAGE_DEPTH_COMPARISON;
    }
    required
}

fn compute_texel_buffer_format_features(requires_atomics: bool) -> vk::FormatFeatureFlags {
    vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER
        | if requires_atomics {
            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC
        } else {
            vk::FormatFeatureFlags::empty()
        }
}

fn execute_compute_dispatch(
    inner: &mut RendererInner,
    dispatch: crate::compute::ComputeDispatch,
    lazy: bool,
) -> crate::compute::ComputeDispatchOutcome {
    use crate::compute::{
        ComputeDispatchOutcome, ComputeDispatchResult, ComputeImageReadback, ComputeTexelReadback,
    };

    if let Err(reason) = validate_compute_dispatch(inner, &dispatch) {
        return ComputeDispatchOutcome::Unsupported(reason);
    }

    let mut sample_plans = Vec::with_capacity(dispatch.sampled_rts.len());
    for sampled in &dispatch.sampled_rts {
        let numeric_type = sampled.sample_type.spirv_type();
        let cross_access_output = compute_cross_access_output_index(&dispatch, sampled.binding);
        let live_view_compatible = sampled.tic.mip_levels() == 1
            && sampled.tic.view_base_mip() == 0
            && sampled.tic.view_mip_levels() == 1;
        let alias = if cross_access_output.is_none()
            && live_view_compatible
            && (!sampled.guest_bytes_authoritative || sampled.guest_bytes.is_none())
        {
            compute_rt_alias(&inner.rt_cache, sampled)
        } else {
            None
        };
        if std::env::var_os("NEXIUM_COMPUTE_ALIAS_TRACE").is_some_and(|value| {
            let value = value.to_string_lossy();
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        }) {
            match alias {
                Some(alias) => log::warn!(
                    "[compute-alias] program={:#x} binding={} request={} tic={:?} source=live key={} fmt={:?} layout={:?} depth={} guest={} authoritative={} require_live={}",
                    dispatch.program_key,
                    sampled.binding,
                    sampled.key.label(),
                    sampled.tic.format,
                    alias.key.label(),
                    alias.format,
                    alias.layout,
                    alias.depth,
                    sampled.guest_bytes.is_some(),
                    sampled.guest_bytes_authoritative,
                    sampled.require_live,
                ),
                None => log::warn!(
                    "[compute-alias] program={:#x} binding={} request={} tic={:?} source=guest guest={} authoritative={} require_live={}",
                    dispatch.program_key,
                    sampled.binding,
                    sampled.key.label(),
                    sampled.tic.format,
                    sampled.guest_bytes.is_some(),
                    sampled.guest_bytes_authoritative,
                    sampled.require_live,
                ),
            }
        }
        let (plan, integer_sample) = if let Some(output_index) = cross_access_output {
            let output = &dispatch.outputs[output_index];
            let components = match compute_cross_access_view_components(
                sampled.binding,
                &sampled.tic,
                sampled.sample_type,
                output,
            ) {
                Ok(components) => components,
                Err(error) => return ComputeDispatchOutcome::Unsupported(error),
            };
            let integer_sample = texture_requires_integer_sampler(numeric_type, false);
            let required_features = compute_guest_sampler_format_features(
                &sampled.tsc,
                integer_sample,
                inner.sampler_filter_minmax_supported,
                inner.sampler_anisotropy_supported,
                std::env::var_os("NEXIUM_TEX_FORCE_LINEAR").is_some(),
            );
            let format = output.format.vk_format();
            let mut format_features3 = vk::FormatProperties3::default();
            let mut format_features2 = vk::FormatProperties2 {
                s_type: vk::StructureType::FORMAT_PROPERTIES_2,
                p_next: &mut format_features3 as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe {
                inner.instance.get_physical_device_format_properties2(
                    inner.physical_device,
                    format,
                    &mut format_features2,
                );
            }
            if !format_features3
                .optimal_tiling_features
                .contains(required_features)
            {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute sampled/storage alias binding {} format {:?} lacks sampler features {:?} (has {:?})",
                    sampled.binding,
                    format,
                    required_features,
                    format_features3.optimal_tiling_features
                ));
            }
            (
                ComputeSamplePlan::CrossAccess {
                    binding: sampled.binding,
                    sampler: vk::Sampler::null(),
                    output_index,
                    components,
                },
                integer_sample,
            )
        } else if let Some(alias) = alias {
            if alias.layout == vk::ImageLayout::UNDEFINED {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "live render target at compute binding {} has undefined contents",
                    sampled.binding
                ));
            }
            let view_format = if alias.depth {
                alias.format
            } else {
                rt_alias_view_format(alias.key, sampled.tic, alias.format)
            };
            if !rt_alias_numeric_type_matches(alias, Some(sampled.tic), numeric_type, view_format) {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute binding {} numeric type {:?} is incompatible with {:?}",
                    sampled.binding, sampled.sample_type, view_format
                ));
            }
            let view = {
                let RendererInner {
                    device,
                    rt_cache,
                    utility_slot,
                    ..
                } = &mut *inner;
                rt_alias_sample_view(
                    device,
                    rt_cache,
                    utility_slot,
                    alias,
                    Some(sampled.tic),
                    view_format,
                )
            };
            let Some(view) = view else {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "could not create sampled view for compute binding {}",
                    sampled.binding
                ));
            };
            let stencil_alias = alias.depth
                && rt_alias_sample_aspect(alias, sampled.tic)
                    == Some(vk::ImageAspectFlags::STENCIL);
            (
                ComputeSamplePlan::Live(PreparedComputeSample {
                    binding: sampled.binding,
                    image: alias.image,
                    view,
                    sampler: vk::Sampler::null(),
                    descriptor_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    old_layout: alias.layout,
                    aspects: alias.aspects,
                }),
                texture_requires_integer_sampler(numeric_type, stencil_alias),
            )
        } else {
            if sampled.require_live {
                log::warn!(
                    "[compute-alias-miss] binding={} tic_fmt={:?} :: {}",
                    sampled.binding,
                    sampled.tic.format,
                    inner.rt_cache.debug_depth_resolution(sampled.key)
                );
                return ComputeDispatchOutcome::Unsupported(format!(
                    "no content-bearing live render target for required compute binding {} ({})",
                    sampled.binding,
                    sampled.key.label()
                ));
            }
            let Some(raw) = sampled.guest_bytes.as_deref() else {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute binding {} has neither a live target nor guest bytes",
                    sampled.binding
                ));
            };
            if raw.is_empty()
                || sampled.tic.is_buffer()
                || tic_requires_dedicated_sampled_view(&sampled.tic)
            {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute guest fallback binding {} is not a supported 2D image",
                    sampled.binding
                ));
            }
            let format = match texture_image_format_for_tic(&sampled.tic, numeric_type) {
                Ok(format) => format,
                Err(error) => return ComputeDispatchOutcome::Unsupported(error),
            };
            let integer_sample = texture_requires_integer_sampler(numeric_type, false);
            let required_features = compute_guest_sampler_format_features(
                &sampled.tsc,
                integer_sample,
                inner.sampler_filter_minmax_supported,
                inner.sampler_anisotropy_supported,
                std::env::var_os("NEXIUM_TEX_FORCE_LINEAR").is_some(),
            );
            let mut format_features3 = vk::FormatProperties3::default();
            let mut format_features2 = vk::FormatProperties2 {
                s_type: vk::StructureType::FORMAT_PROPERTIES_2,
                p_next: &mut format_features3 as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };
            unsafe {
                inner.instance.get_physical_device_format_properties2(
                    inner.physical_device,
                    format,
                    &mut format_features2,
                );
            }
            if !format_features3
                .optimal_tiling_features
                .contains(required_features)
            {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute guest sample binding {} format {:?} lacks sampler features {:?} (has {:?})",
                    sampled.binding,
                    format,
                    required_features,
                    format_features3.optimal_tiling_features
                ));
            }
            let upload = match texture_upload_data(raw, &sampled.tic, 1, raw.len(), false, format) {
                Ok(upload) => upload,
                Err(error) => return ComputeDispatchOutcome::Unsupported(error),
            };
            let swizzle =
                texture_view_swizzle(sampled.tic.format, numeric_type, sampled.tic.swizzle);
            (
                ComputeSamplePlan::Guest {
                    binding: sampled.binding,
                    sampler: vk::Sampler::null(),
                    width: sampled.tic.width,
                    height: sampled.tic.height,
                    depth: 1,
                    is_3d: false,
                    format,
                    components: texture_component_mapping(swizzle),
                    mip_levels: sampled.tic.mip_levels(),
                    view_base_mip: sampled.tic.view_base_mip(),
                    view_mip_levels: sampled.tic.view_mip_levels(),
                    upload_bytes: upload.bytes,
                    upload_copies: upload.copies,
                },
                integer_sample,
            )
        };
        let sampler = {
            let RendererInner {
                device,
                sampler_cache,
                integer_sampler_cache,
                sampler_filter_minmax_supported,
                sampler_anisotropy_supported,
                ..
            } = &mut *inner;
            let cache = if integer_sample {
                integer_sampler_cache
            } else {
                sampler_cache
            };
            match cached_sampler_for_tsc(
                device,
                cache,
                sampled.tsc,
                integer_sample,
                *sampler_filter_minmax_supported,
                *sampler_anisotropy_supported,
            ) {
                Ok(sampler) => sampler,
                Err(error) => return ComputeDispatchOutcome::FailedBeforeSubmit(error),
            }
        };
        let plan = match plan {
            ComputeSamplePlan::Live(mut live) => {
                live.sampler = sampler;
                ComputeSamplePlan::Live(live)
            }
            ComputeSamplePlan::CrossAccess {
                binding,
                output_index,
                components,
                ..
            } => ComputeSamplePlan::CrossAccess {
                binding,
                sampler,
                output_index,
                components,
            },
            ComputeSamplePlan::Guest {
                binding,
                width,
                height,
                depth,
                is_3d,
                format,
                components,
                mip_levels,
                view_base_mip,
                view_mip_levels,
                upload_bytes,
                upload_copies,
                ..
            } => ComputeSamplePlan::Guest {
                binding,
                sampler,
                width,
                height,
                depth,
                is_3d,
                format,
                components,
                mip_levels,
                view_base_mip,
                view_mip_levels,
                upload_bytes,
                upload_copies,
            },
        };
        sample_plans.push(plan);
    }

    let mut image_plans = Vec::with_capacity(dispatch.sampled_images.len());
    for sampled in &dispatch.sampled_images {
        let tic = &sampled.tic;
        let is_3d = tic_is_volume(tic);
        let depth = if is_3d { tic.depth.max(1) } else { 1 };
        if tic.is_buffer()
            || !matches!(tic.texture_type, 1 | 2)
            || tic.base_layer != 0
            || (is_3d && tic.mip_levels() != 1)
        {
            return ComputeDispatchOutcome::Unsupported(format!(
                "compute sampled-image binding {} has an unsupported 2D/3D view: {:?}",
                sampled.binding, tic
            ));
        }
        let numeric_type = sampled.sample_type.spirv_type();
        let format = match texture_image_format_for_tic(tic, numeric_type) {
            Ok(format) => format,
            Err(error) => return ComputeDispatchOutcome::Unsupported(error),
        };
        if let Some(output_index) = compute_cross_access_output_index(&dispatch, sampled.binding) {
            let output = &dispatch.outputs[output_index];
            let components = match compute_cross_access_view_components(
                sampled.binding,
                tic,
                sampled.sample_type,
                output,
            ) {
                Ok(components) => components,
                Err(error) => return ComputeDispatchOutcome::Unsupported(error),
            };
            image_plans.push(ComputeImagePlan::CrossAccess {
                binding: sampled.binding,
                output_index,
                components,
            });
            continue;
        }
        let live_view_compatible =
            tic.mip_levels() == 1 && tic.view_base_mip() == 0 && tic.view_mip_levels() == 1;
        let alias = if sampled.guest_bytes_authoritative || !live_view_compatible {
            None
        } else {
            sampled
                .key
                .and_then(|key| compute_rt_alias_for_key(&inner.rt_cache, key, tic.format))
        };
        if std::env::var_os("NEXIUM_COMPUTE_ALIAS_TRACE").is_some_and(|value| {
            let value = value.to_string_lossy();
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        }) {
            match alias {
                Some(alias) => log::warn!(
                    "[compute-image-alias] program={:#x} binding={} request={} tic={:?} source=live key={} fmt={:?} layout={:?} depth={} guest={} authoritative={} require_live={}",
                    dispatch.program_key,
                    sampled.binding,
                    sampled
                        .key
                        .map(|key| key.label())
                        .unwrap_or_else(|| "none".to_string()),
                    sampled.tic.format,
                    alias.key.label(),
                    alias.format,
                    alias.layout,
                    alias.depth,
                    !sampled.guest_bytes.is_empty(),
                    sampled.guest_bytes_authoritative,
                    sampled.require_live,
                ),
                None => log::warn!(
                    "[compute-image-alias] program={:#x} binding={} request={} tic={:?} source=guest guest={} authoritative={} require_live={}",
                    dispatch.program_key,
                    sampled.binding,
                    sampled
                        .key
                        .map(|key| key.label())
                        .unwrap_or_else(|| "none".to_string()),
                    sampled.tic.format,
                    !sampled.guest_bytes.is_empty(),
                    sampled.guest_bytes_authoritative,
                    sampled.require_live,
                ),
            }
        }
        if let Some(alias) = alias {
            match prepare_compute_live_sampled_image(
                inner,
                sampled.binding,
                sampled.sample_type,
                *tic,
                is_3d,
                depth,
                numeric_type,
                alias,
            ) {
                Ok(image) => {
                    image_plans.push(ComputeImagePlan::Live(image));
                    continue;
                }
                Err(reason) if sampled.require_live => {
                    return ComputeDispatchOutcome::Unsupported(reason);
                }
                Err(reason) => {
                    log::debug!(
                        "compute sampled-image binding {} discarded optional live alias: {}; using guest fallback",
                        sampled.binding,
                        reason
                    );
                }
            }
        }
        if sampled.require_live {
            return ComputeDispatchOutcome::Unsupported(format!(
                "no content-bearing live render target for sampled-image binding {}",
                sampled.binding
            ));
        }
        if sampled.guest_bytes.is_empty() {
            return ComputeDispatchOutcome::Unsupported(format!(
                "sampled-image binding {} has neither a live target nor guest bytes",
                sampled.binding
            ));
        }
        let mut format_features3 = vk::FormatProperties3::default();
        let mut format_features2 = vk::FormatProperties2 {
            s_type: vk::StructureType::FORMAT_PROPERTIES_2,
            p_next: &mut format_features3 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe {
            inner.instance.get_physical_device_format_properties2(
                inner.physical_device,
                format,
                &mut format_features2,
            );
        }
        let required =
            vk::FormatFeatureFlags2::SAMPLED_IMAGE | vk::FormatFeatureFlags2::TRANSFER_DST;
        if !format_features3.optimal_tiling_features.contains(required) {
            return ComputeDispatchOutcome::Unsupported(format!(
                "compute sampled-image binding {} format {:?} lacks {:?}",
                sampled.binding, format, required
            ));
        }
        let slice_pitch =
            if is_3d && (!tic.is_block_linear || crate::pitch_oracle::is_pitch_dst(tic.gpu_va)) {
                tic.format.linear_size(tic.width, tic.height)
            } else {
                sampled.guest_bytes.len()
            };
        let upload =
            match texture_upload_data(&sampled.guest_bytes, tic, depth, slice_pitch, false, format)
            {
                Ok(upload) => upload,
                Err(error) => return ComputeDispatchOutcome::Unsupported(error),
            };
        let swizzle = texture_view_swizzle(tic.format, numeric_type, tic.swizzle);
        image_plans.push(ComputeImagePlan::Guest {
            binding: sampled.binding,
            width: tic.width,
            height: tic.height,
            depth,
            is_3d,
            format,
            components: texture_component_mapping(swizzle),
            mip_levels: tic.mip_levels(),
            view_base_mip: tic.view_base_mip(),
            view_mip_levels: tic.view_mip_levels(),
            upload_bytes: upload.bytes,
            upload_copies: upload.copies,
        });
    }

    let mut sample_transitions: Vec<ComputeSampleTransition> = Vec::new();
    for sample in sample_plans.iter().filter_map(|plan| match plan {
        ComputeSamplePlan::Live(sample) => Some(sample),
        ComputeSamplePlan::CrossAccess { .. } | ComputeSamplePlan::Guest { .. } => None,
    }) {
        if let Some(existing) = sample_transitions
            .iter_mut()
            .find(|existing| existing.image == sample.image)
        {
            if existing.old_layout != sample.old_layout {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute sampled image has conflicting layouts {:?} and {:?}",
                    existing.old_layout, sample.old_layout
                ));
            }
            existing.aspects |= sample.aspects;
        } else {
            sample_transitions.push(ComputeSampleTransition {
                image: sample.image,
                old_layout: sample.old_layout,
                aspects: sample.aspects,
            });
        }
    }
    for image in image_plans.iter().filter_map(|plan| match plan {
        ComputeImagePlan::Live(image) => Some(image),
        ComputeImagePlan::CrossAccess { .. } | ComputeImagePlan::Guest { .. } => None,
    }) {
        if let Some(existing) = sample_transitions
            .iter_mut()
            .find(|existing| existing.image == image.image)
        {
            if existing.old_layout != image.old_layout {
                return ComputeDispatchOutcome::Unsupported(format!(
                    "compute sampled image has conflicting layouts {:?} and {:?}",
                    existing.old_layout, image.old_layout
                ));
            }
            existing.aspects |= image.aspects;
        } else {
            sample_transitions.push(ComputeSampleTransition {
                image: image.image,
                old_layout: image.old_layout,
                aspects: image.aspects,
            });
        }
    }

    let descriptor_spec = crate::compute::descriptor_spec(&dispatch);
    let reuse_sets = lazy && !inner.pending_computes.is_empty();
    let prepared = {
        let device = &inner.device;
        inner.compute_backend.as_mut().unwrap().prepare_program(
            device,
            dispatch.program_key,
            &dispatch.spirv,
            dispatch.local_size,
            dispatch.required_subgroup_size,
            descriptor_spec.clone(),
            reuse_sets,
        )
    };
    let program = match prepared {
        Ok(program) => program,
        Err(_) if reuse_sets => {
            settle_all_pending_computes(inner);
            let Some(backend) = inner.compute_backend.as_mut() else {
                return ComputeDispatchOutcome::FailedBeforeSubmit(
                    inner.compute_unavailable_reason.clone(),
                );
            };
            match backend.prepare_program(
                &inner.device,
                dispatch.program_key,
                &dispatch.spirv,
                dispatch.local_size,
                dispatch.required_subgroup_size,
                descriptor_spec,
                false,
            ) {
                Ok(program) => program,
                Err(error) => return ComputeDispatchOutcome::FailedBeforeSubmit(error),
            }
        }
        Err(error) => return ComputeDispatchOutcome::FailedBeforeSubmit(error),
    };

    let output_lengths: Vec<usize> = dispatch
        .outputs
        .iter()
        .map(|output| {
            output.width as usize
                * output.height as usize
                * output.depth as usize
                * output.format.bytes_per_pixel()
        })
        .collect();
    let mut resources = PreparedComputeResources::default();
    let prepare_result = (|| -> Result<(), String> {
        for uniform in &dispatch.uniform_buffers {
            let RendererInner {
                device,
                mem_props,
                compute_backend,
                ..
            } = &mut *inner;
            resources
                .uniforms
                .push(compute_backend.as_mut().unwrap().acquire_uniform_buffer(
                    device,
                    mem_props,
                    &uniform.bytes,
                )?);
        }
        for texel in &dispatch.texel_buffers {
            resources
                .texels
                .push(crate::compute::create_compute_texel_buffer(
                    &inner.device,
                    &inner.mem_props,
                    &texel.bytes,
                    vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                    texel.format.vk_format(),
                )?);
        }
        for texel in &dispatch.uniform_texel_buffers {
            resources
                .uniform_texels
                .push(crate::compute::create_compute_texel_buffer(
                    &inner.device,
                    &inner.mem_props,
                    &texel.bytes,
                    vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                    texel.format.vk_format(),
                )?);
        }
        for plan in &sample_plans {
            let ComputeSamplePlan::Guest {
                width,
                height,
                depth,
                is_3d,
                format,
                components,
                mip_levels,
                view_base_mip,
                view_mip_levels,
                upload_bytes,
                upload_copies,
                ..
            } = plan
            else {
                continue;
            };
            let copies: Vec<_> = upload_copies
                .iter()
                .map(|copy| crate::compute::ComputeGuestImageCopy {
                    buffer_offset: copy.buffer_offset,
                    mip_level: copy.mip_level,
                    width: copy.width,
                    height: copy.height,
                })
                .collect();
            resources
                .guest_images
                .push(crate::compute::create_compute_guest_image(
                    &inner.device,
                    &inner.mem_props,
                    *width,
                    *height,
                    *depth,
                    *is_3d,
                    *format,
                    *components,
                    *mip_levels,
                    *view_base_mip,
                    *view_mip_levels,
                    upload_bytes,
                    &copies,
                )?);
        }
        for plan in &image_plans {
            let ComputeImagePlan::Guest {
                width,
                height,
                depth,
                is_3d,
                format,
                components,
                mip_levels,
                view_base_mip,
                view_mip_levels,
                upload_bytes,
                upload_copies,
                ..
            } = plan
            else {
                continue;
            };
            let copies: Vec<_> = upload_copies
                .iter()
                .map(|copy| crate::compute::ComputeGuestImageCopy {
                    buffer_offset: copy.buffer_offset,
                    mip_level: copy.mip_level,
                    width: copy.width,
                    height: copy.height,
                })
                .collect();
            resources
                .guest_images
                .push(crate::compute::create_compute_guest_image(
                    &inner.device,
                    &inner.mem_props,
                    *width,
                    *height,
                    *depth,
                    *is_3d,
                    *format,
                    *components,
                    *mip_levels,
                    *view_base_mip,
                    *view_mip_levels,
                    upload_bytes,
                    &copies,
                )?);
        }
        for (output, output_len) in dispatch.outputs.iter().zip(&output_lengths) {
            let RendererInner {
                device,
                mem_props,
                compute_backend,
                ..
            } = &mut *inner;
            let backend = compute_backend.as_mut().unwrap();
            let sampled = dispatch
                .image_aliases
                .iter()
                .any(|alias| alias.storage_binding == output.binding);
            resources.outputs.push(backend.acquire_output_image(
                device,
                mem_props,
                output.width,
                output.height,
                output.depth,
                output.is_3d,
                output.format.vk_format(),
                sampled,
            )?);
            resources.output_uploads.push(
                output
                    .initial_bytes
                    .as_deref()
                    .map(|bytes| {
                        crate::compute::create_compute_buffer(
                            device,
                            mem_props,
                            bytes,
                            vk::BufferUsageFlags::TRANSFER_SRC,
                            false,
                        )
                    })
                    .transpose()?,
            );
            resources.readbacks.push(backend.acquire_readback_buffer(
                device,
                mem_props,
                *output_len as u64,
            )?);
        }
        for plan in &sample_plans {
            let ComputeSamplePlan::CrossAccess {
                output_index,
                components,
                ..
            } = plan
            else {
                continue;
            };
            resources
                .sampled_alias_views
                .push(crate::compute::create_compute_sampled_alias_view(
                    &inner.device,
                    &resources.outputs[*output_index],
                    *components,
                )?);
        }
        for plan in &image_plans {
            let ComputeImagePlan::CrossAccess {
                output_index,
                components,
                ..
            } = plan
            else {
                continue;
            };
            resources
                .sampled_alias_views
                .push(crate::compute::create_compute_sampled_alias_view(
                    &inner.device,
                    &resources.outputs[*output_index],
                    *components,
                )?);
        }
        Ok(())
    })();
    if let Err(error) = prepare_result {
        resources.destroy(&inner.device);
        if lazy {
            free_compute_descriptor_set(inner, program.descriptor_pool, program.descriptor_set);
        }
        return ComputeDispatchOutcome::FailedBeforeSubmit(error);
    }

    let mut next_guest = 0usize;
    let mut next_alias_view = 0usize;
    let samples: Vec<PreparedComputeSample> = sample_plans
        .into_iter()
        .map(|plan| match plan {
            ComputeSamplePlan::Live(sample) => sample,
            ComputeSamplePlan::CrossAccess {
                binding,
                sampler,
                output_index,
                ..
            } => {
                let view = resources.sampled_alias_views[next_alias_view];
                next_alias_view += 1;
                PreparedComputeSample {
                    binding,
                    image: resources.outputs[output_index].image,
                    view,
                    sampler,
                    descriptor_layout: vk::ImageLayout::GENERAL,
                    old_layout: vk::ImageLayout::GENERAL,
                    aspects: vk::ImageAspectFlags::COLOR,
                }
            }
            ComputeSamplePlan::Guest {
                binding, sampler, ..
            } => {
                let guest_index = next_guest;
                next_guest += 1;
                let image = &resources.guest_images[guest_index].image;
                PreparedComputeSample {
                    binding,
                    image: image.image,
                    view: image.view,
                    sampler,
                    descriptor_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    aspects: vk::ImageAspectFlags::COLOR,
                }
            }
        })
        .collect();
    let images: Vec<PreparedComputeImage> = image_plans
        .into_iter()
        .map(|plan| match plan {
            ComputeImagePlan::Live(image) => image,
            ComputeImagePlan::CrossAccess {
                binding,
                output_index,
                ..
            } => {
                let view = resources.sampled_alias_views[next_alias_view];
                next_alias_view += 1;
                PreparedComputeImage {
                    binding,
                    image: resources.outputs[output_index].image,
                    view,
                    descriptor_layout: vk::ImageLayout::GENERAL,
                    old_layout: vk::ImageLayout::GENERAL,
                    aspects: vk::ImageAspectFlags::COLOR,
                }
            }
            ComputeImagePlan::Guest { binding, .. } => {
                let guest_index = next_guest;
                next_guest += 1;
                let image = &resources.guest_images[guest_index].image;
                PreparedComputeImage {
                    binding,
                    image: image.image,
                    view: image.view,
                    descriptor_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    aspects: vk::ImageAspectFlags::COLOR,
                }
            }
        })
        .collect();

    let descriptor_set = program.descriptor_set;
    let uniform_infos: Vec<_> = dispatch
        .uniform_buffers
        .iter()
        .zip(&resources.uniforms)
        .map(|(uniform, resource)| vk::DescriptorBufferInfo {
            buffer: resource.buffer,
            offset: 0,
            range: uniform.bytes.len() as u64,
        })
        .collect();
    let sampled_infos: Vec<_> = samples
        .iter()
        .map(|sample| vk::DescriptorImageInfo {
            sampler: sample.sampler,
            image_view: sample.view,
            image_layout: sample.descriptor_layout,
        })
        .collect();
    let image_infos: Vec<_> = images
        .iter()
        .map(|image| vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: image.view,
            image_layout: image.descriptor_layout,
        })
        .collect();
    let output_infos: Vec<_> = resources
        .outputs
        .iter()
        .map(|output| vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: output.view,
            image_layout: vk::ImageLayout::GENERAL,
        })
        .collect();
    let mut writes = Vec::with_capacity(
        dispatch.uniform_buffers.len()
            + dispatch
                .texel_buffers
                .iter()
                .map(|buffer| buffer.bindings.len())
                .sum::<usize>()
            + dispatch.uniform_texel_buffers.len()
            + samples.len()
            + images.len()
            + dispatch.outputs.len(),
    );
    for (index, uniform) in dispatch.uniform_buffers.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: uniform.binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: &uniform_infos[index],
            ..Default::default()
        });
    }
    for (resource_index, texel) in dispatch.texel_buffers.iter().enumerate() {
        for binding in &texel.bindings {
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: descriptor_set,
                dst_binding: *binding,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                p_texel_buffer_view: &resources.texels[resource_index].view,
                ..Default::default()
            });
        }
    }
    for (index, texel) in dispatch.uniform_texel_buffers.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: texel.binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            p_texel_buffer_view: &resources.uniform_texels[index].view,
            ..Default::default()
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: sample.binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &sampled_infos[index],
            ..Default::default()
        });
    }
    for (index, image) in images.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: image.binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: &image_infos[index],
            ..Default::default()
        });
    }
    for (index, output) in dispatch.outputs.iter().enumerate() {
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: descriptor_set,
            dst_binding: output.binding,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &output_infos[index],
            ..Default::default()
        });
    }
    unsafe { inner.device.update_descriptor_sets(&writes, &[]) };

    let mut buffer_barriers = Vec::with_capacity(
        dispatch.uniform_buffers.len()
            + dispatch.texel_buffers.len()
            + dispatch.uniform_texel_buffers.len(),
    );
    for (uniform, resource) in dispatch.uniform_buffers.iter().zip(&resources.uniforms) {
        buffer_barriers.push(vk::BufferMemoryBarrier {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
            src_access_mask: vk::AccessFlags::HOST_WRITE,
            dst_access_mask: vk::AccessFlags::UNIFORM_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: resource.buffer,
            offset: 0,
            size: uniform.bytes.len() as u64,
            ..Default::default()
        });
    }
    for (texel, resource) in dispatch.texel_buffers.iter().zip(&resources.texels) {
        buffer_barriers.push(vk::BufferMemoryBarrier {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
            src_access_mask: vk::AccessFlags::HOST_WRITE,
            dst_access_mask: if texel.writable {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
            } else {
                vk::AccessFlags::SHADER_READ
            },
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: resource.buffer,
            offset: 0,
            size: texel.bytes.len() as u64,
            ..Default::default()
        });
    }
    for (texel, resource) in dispatch
        .uniform_texel_buffers
        .iter()
        .zip(&resources.uniform_texels)
    {
        buffer_barriers.push(vk::BufferMemoryBarrier {
            s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
            src_access_mask: vk::AccessFlags::HOST_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: resource.buffer,
            offset: 0,
            size: texel.bytes.len() as u64,
            ..Default::default()
        });
    }
    let mut image_barriers = Vec::with_capacity(sample_transitions.len() + resources.outputs.len());
    for transition in &sample_transitions {
        image_barriers.push(vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            src_access_mask: compute_layout_access(transition.old_layout),
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: transition.old_layout,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: transition.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: transition.aspects,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        });
    }
    for (request, output) in dispatch.outputs.iter().zip(&resources.outputs) {
        if request.initial_bytes.is_some() {
            continue;
        }
        image_barriers.push(vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: output.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        });
    }

    let (cmd, fence) = if lazy {
        match acquire_compute_slot(inner) {
            Ok(slot) => slot,
            Err(error) => {
                resources.destroy(&inner.device);
                free_compute_descriptor_set(inner, program.descriptor_pool, program.descriptor_set);
                return ComputeDispatchOutcome::FailedBeforeSubmit(error);
            }
        }
    } else {
        (inner.utility_slot.cmd, inner.utility_slot.fence)
    };
    let command_result = (|| -> Result<(), String> {
        reset_command_buffer(&inner.device, cmd)?;
        begin_one_time(&inner.device, cmd)?;
        if !resources.guest_images.is_empty() {
            let guest_buffer_barriers: Vec<_> = resources
                .guest_images
                .iter()
                .map(|guest| vk::BufferMemoryBarrier {
                    s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                    src_access_mask: vk::AccessFlags::HOST_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: guest.upload.buffer,
                    offset: 0,
                    size: guest.upload.size,
                    ..Default::default()
                })
                .collect();
            let guest_upload_barriers: Vec<_> = resources
                .guest_images
                .iter()
                .map(|guest| vk::ImageMemoryBarrier {
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: guest.image.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: guest.mip_levels,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                })
                .collect();
            unsafe {
                inner.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &guest_buffer_barriers,
                    &guest_upload_barriers,
                );
                for guest in &resources.guest_images {
                    inner.device.cmd_copy_buffer_to_image(
                        cmd,
                        guest.upload.buffer,
                        guest.image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &guest.copies,
                    );
                }
                let guest_sample_barriers: Vec<_> = resources
                    .guest_images
                    .iter()
                    .map(|guest| vk::ImageMemoryBarrier {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: guest.image.image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: guest.mip_levels,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    })
                    .collect();
                inner.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &guest_sample_barriers,
                );
            }
        }
        if resources.output_uploads.iter().any(Option::is_some) {
            let upload_buffer_barriers: Vec<_> = resources
                .output_uploads
                .iter()
                .flatten()
                .map(|upload| vk::BufferMemoryBarrier {
                    s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                    src_access_mask: vk::AccessFlags::HOST_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: upload.buffer,
                    offset: 0,
                    size: upload.size,
                    ..Default::default()
                })
                .collect();
            let upload_image_barriers: Vec<_> = resources
                .outputs
                .iter()
                .zip(&resources.output_uploads)
                .filter_map(|(output, upload)| {
                    upload.as_ref().map(|_| vk::ImageMemoryBarrier {
                        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                        src_access_mask: vk::AccessFlags::empty(),
                        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: output.image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    })
                })
                .collect();
            unsafe {
                inner.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &upload_buffer_barriers,
                    &upload_image_barriers,
                );
                for ((request, output), upload) in dispatch
                    .outputs
                    .iter()
                    .zip(&resources.outputs)
                    .zip(&resources.output_uploads)
                {
                    let Some(upload) = upload else {
                        continue;
                    };
                    let copy = vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: request.width,
                            height: request.height,
                            depth: request.depth,
                        },
                    };
                    inner.device.cmd_copy_buffer_to_image(
                        cmd,
                        upload.buffer,
                        output.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
                let ready_barriers: Vec<_> = resources
                    .outputs
                    .iter()
                    .zip(&resources.output_uploads)
                    .filter_map(|(output, upload)| {
                        upload.as_ref().map(|_| vk::ImageMemoryBarrier {
                            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                            dst_access_mask: vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE,
                            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            new_layout: vk::ImageLayout::GENERAL,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: output.image,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            },
                            ..Default::default()
                        })
                    })
                    .collect();
                inner.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &ready_barriers,
                );
            }
        }
        unsafe {
            inner.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &buffer_barriers,
                &image_barriers,
            );
            inner
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, program.pipeline);
            inner.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                program.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            inner.device.cmd_dispatch(
                cmd,
                dispatch.group_count[0],
                dispatch.group_count[1],
                dispatch.group_count[2],
            );
        }

        let mut after_buffer_barriers = Vec::new();
        for (texel, resource) in dispatch.texel_buffers.iter().zip(&resources.texels) {
            if !texel.writable {
                continue;
            }
            after_buffer_barriers.push(vk::BufferMemoryBarrier {
                s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::HOST_READ,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: resource.buffer,
                offset: 0,
                size: texel.bytes.len() as u64,
                ..Default::default()
            });
        }
        let mut after_image_barriers =
            Vec::with_capacity(sample_transitions.len() + resources.outputs.len());
        for transition in &sample_transitions {
            after_image_barriers.push(vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: compute_layout_access(transition.old_layout),
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: transition.old_layout,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: transition.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: transition.aspects,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });
        }
        for (request, output) in dispatch.outputs.iter().zip(&resources.outputs) {
            after_image_barriers.push(vk::ImageMemoryBarrier {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                src_access_mask: vk::AccessFlags::SHADER_WRITE
                    | if request.initial_bytes.is_some() {
                        vk::AccessFlags::TRANSFER_WRITE
                    } else {
                        vk::AccessFlags::empty()
                    },
                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: output.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });
        }
        unsafe {
            inner.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &after_buffer_barriers,
                &after_image_barriers,
            );
            for ((request, output), readback) in dispatch
                .outputs
                .iter()
                .zip(&resources.outputs)
                .zip(&resources.readbacks)
            {
                let copy = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: request.width,
                        height: request.height,
                        depth: request.depth,
                    },
                };
                inner.device.cmd_copy_image_to_buffer(
                    cmd,
                    output.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.buffer,
                    &[copy],
                );
            }
            let readback_barriers: Vec<_> = resources
                .readbacks
                .iter()
                .zip(&output_lengths)
                .map(|(readback, output_len)| vk::BufferMemoryBarrier {
                    s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::HOST_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: readback.buffer,
                    offset: 0,
                    size: *output_len as u64,
                    ..Default::default()
                })
                .collect();
            inner.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &readback_barriers,
                &[],
            );
        }
        end_one_time(&inner.device, cmd)
    })();
    if let Err(error) = command_result {
        let _ = reset_command_buffer(&inner.device, cmd);
        resources.destroy(&inner.device);
        if lazy {
            free_compute_descriptor_set(inner, program.descriptor_pool, program.descriptor_set);
            inner.compute_slot_pool.push((cmd, fence));
        }
        return ComputeDispatchOutcome::FailedBeforeSubmit(error);
    }

    let submit_result = if lazy {
        submit_with_fence_untracked(&inner.device, inner.queue, cmd, fence)
    } else {
        submit_with_fence(&inner.device, inner.queue, cmd, fence)
            .and_then(|_| wait_fence(&inner.device, fence))
    };
    if let Err(error) = submit_result {
        match unsafe { inner.device.device_wait_idle() } {
            Ok(()) => {
                resources.destroy(&inner.device);
                if lazy {
                    free_compute_descriptor_set(
                        inner,
                        program.descriptor_pool,
                        program.descriptor_set,
                    );
                    inner.compute_slot_pool.push((cmd, fence));
                }
                return ComputeDispatchOutcome::SubmittedFailure(error);
            }
            Err(idle_error) => {
                std::mem::forget(resources);
                if let Some(backend) = inner.compute_backend.take() {
                    std::mem::forget(backend);
                }
                let failure = format!(
                    "{error}; device_wait_idle after compute submission failure also failed: \
                     {idle_error:?}; generic compute backend poisoned and in-flight resources retained"
                );
                inner.compute_unavailable_reason = failure.clone();
                return ComputeDispatchOutcome::SubmittedFailure(failure);
            }
        }
    }

    if lazy {
        let outputs = dispatch
            .outputs
            .iter()
            .zip(&output_lengths)
            .map(|(request, output_len)| PendingComputeOutput {
                binding: request.binding,
                width: request.width,
                height: request.height,
                depth: request.depth,
                format: request.format,
                byte_len: *output_len,
            })
            .collect();
        let texels = dispatch
            .texel_buffers
            .iter()
            .enumerate()
            .filter(|(_, request)| request.writable)
            .map(|(resource_index, request)| PendingComputeTexel {
                resource_index,
                byte_len: request.bytes.len(),
            })
            .collect();
        let id = inner.next_pending_compute_id;
        inner.next_pending_compute_id += 1;
        inner.pending_computes.push(PendingCompute {
            id,
            program_key: dispatch.program_key,
            fence,
            cmd,
            descriptor_pool: program.descriptor_pool,
            descriptor_set: program.descriptor_set,
            resources: Some(resources),
            outputs,
            texels,
            result: None,
        });
        return ComputeDispatchOutcome::Submitted(id);
    }

    let mut image_readbacks = Vec::with_capacity(dispatch.outputs.len());
    for (resource_index, ((request, readback), output_len)) in dispatch
        .outputs
        .iter()
        .zip(&resources.readbacks)
        .zip(&output_lengths)
        .enumerate()
    {
        let bytes = match readback.read(&inner.device, *output_len) {
            Ok(bytes) => bytes,
            Err(error) => {
                resources.destroy(&inner.device);
                return ComputeDispatchOutcome::SubmittedFailure(error);
            }
        };
        if std::env::var_os("NEXIUM_COMPUTE_READBACK_TRACE").is_some_and(|value| {
            let value = value.to_string_lossy();
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        }) {
            let nonzero = bytes.iter().filter(|byte| **byte != 0).count();
            let max = bytes.iter().copied().max().unwrap_or(0);
            let min = bytes.iter().copied().min().unwrap_or(0);
            log::warn!(
                "[compute-readback] program={:#x} binding={} resource={} bytes={} nonzero={} min={:#x} max={:#x}",
                dispatch.program_key,
                request.binding,
                resource_index,
                bytes.len(),
                nonzero,
                min,
                max,
            );
        }
        image_readbacks.push(ComputeImageReadback {
            resource_index,
            binding: request.binding,
            bytes,
            width: request.width,
            height: request.height,
            depth: request.depth,
            format: request.format,
        });
    }
    let mut texel_readbacks = Vec::new();
    for (resource_index, (request, resource)) in dispatch
        .texel_buffers
        .iter()
        .zip(&resources.texels)
        .enumerate()
    {
        if !request.writable {
            continue;
        }
        match resource.read(&inner.device, request.bytes.len()) {
            Ok(bytes) => texel_readbacks.push(ComputeTexelReadback {
                resource_index,
                bytes,
            }),
            Err(error) => {
                resources.destroy(&inner.device);
                return ComputeDispatchOutcome::SubmittedFailure(error);
            }
        }
    }
    let result = ComputeDispatchResult {
        image_readbacks,
        texel_readbacks,
    };
    let uniforms = std::mem::take(&mut resources.uniforms);
    let outputs = std::mem::take(&mut resources.outputs);
    let readbacks = std::mem::take(&mut resources.readbacks);
    resources.destroy(&inner.device);
    {
        let RendererInner {
            device,
            compute_backend,
            ..
        } = &mut *inner;
        compute_backend
            .as_mut()
            .unwrap()
            .recycle_dispatch_resources(device, uniforms, outputs, readbacks);
    }
    ComputeDispatchOutcome::Executed(result)
}

fn validate_compute_dispatch(
    inner: &RendererInner,
    dispatch: &crate::compute::ComputeDispatch,
) -> Result<(), String> {
    let Some(backend) = inner.compute_backend.as_ref() else {
        return Err(if inner.compute_unavailable_reason.is_empty() {
            "generic Vulkan compute backend is unavailable".to_string()
        } else {
            inner.compute_unavailable_reason.clone()
        });
    };
    validate_compute_cross_access_aliases(dispatch)?;
    if dispatch.spirv.len() < 5 || dispatch.spirv[0] != 0x0723_0203 {
        return Err("compute program is not a valid SPIR-V word stream".to_string());
    }
    if dispatch.requires_workgroup_explicit_layout && !backend.workgroup_explicit_layout_enabled {
        return Err(
            "compute program requires unavailable workgroup-memory explicit layout".to_string(),
        );
    }
    crate::compute::validate_compute_local_size(
        dispatch.local_size,
        backend.max_compute_work_group_size,
        backend.max_compute_work_group_invocations,
    )?;
    if dispatch.shared_memory_size > backend.max_compute_shared_memory_size {
        return Err(format!(
            "compute shared memory size {:#x} exceeds device limit {:#x}",
            dispatch.shared_memory_size, backend.max_compute_shared_memory_size
        ));
    }
    for axis in 0..3 {
        if dispatch.group_count[axis] == 0
            || dispatch.group_count[axis] > backend.max_group_count[axis]
        {
            return Err(format!(
                "compute group count {:?} exceeds device limit {:?}",
                dispatch.group_count, backend.max_group_count
            ));
        }
    }
    if let Some(size) = dispatch.required_subgroup_size {
        if !backend
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            || size < backend.min_subgroup_size
            || size > backend.max_subgroup_size
            || !size.is_power_of_two()
        {
            return Err(format!(
                "required compute subgroup size {} is unsupported ({}..={})",
                size, backend.min_subgroup_size, backend.max_subgroup_size
            ));
        }
    }
    if dispatch.uniform_buffers.len() > backend.max_uniform_buffers as usize
        || dispatch.uniform_buffers.iter().any(|buffer| {
            buffer.bytes.is_empty()
                || buffer.bytes.len() > backend.max_uniform_buffer_range as usize
        })
    {
        return Err("invalid or excessive compute uniform buffers".to_string());
    }
    let texel_binding_count: usize = dispatch
        .texel_buffers
        .iter()
        .map(|buffer| buffer.bindings.len())
        .sum();
    if texel_binding_count > backend.max_storage_texel_buffers as usize {
        return Err("too many compute storage texel-buffer bindings".to_string());
    }
    for buffer in &dispatch.texel_buffers {
        if buffer.bytes.is_empty()
            || buffer.bytes.len() % buffer.format.bytes_per_element() != 0
            || buffer.bytes.len() / buffer.format.bytes_per_element()
                > inner.max_texel_buffer_elements as usize
            || buffer.bindings.is_empty()
        {
            return Err("invalid typed compute storage texel buffer".to_string());
        }
        if buffer.requires_atomics && !buffer.format.supports_storage_atomics() {
            return Err(format!(
                "compute storage texel format {:?} cannot be used for image atomics",
                buffer.format
            ));
        }
        if buffer.format.requires_storage_image_extended_formats()
            && !backend.storage_image_extended_formats_enabled
        {
            return Err(format!(
                "compute storage texel format {:?} requires unavailable shaderStorageImageExtendedFormats",
                buffer.format
            ));
        }
        let features = unsafe {
            inner.instance.get_physical_device_format_properties(
                inner.physical_device,
                buffer.format.vk_format(),
            )
        };
        let required = compute_texel_buffer_format_features(buffer.requires_atomics);
        if !features.buffer_features.contains(required) {
            return Err(format!(
                "compute storage texel format {:?} lacks required features {required:?}",
                buffer.format
            ));
        }
    }
    if dispatch.uniform_texel_buffers.len() > backend.max_sampled_images as usize {
        return Err("too many compute uniform texel-buffer bindings".to_string());
    }
    for buffer in &dispatch.uniform_texel_buffers {
        if buffer.bytes.is_empty()
            || buffer.bytes.len() % buffer.format.bytes_per_element() != 0
            || buffer.bytes.len() / buffer.format.bytes_per_element()
                > inner.max_texel_buffer_elements as usize
        {
            return Err("invalid compute uniform texel buffer".to_string());
        }
        let features = unsafe {
            inner.instance.get_physical_device_format_properties(
                inner.physical_device,
                buffer.format.vk_format(),
            )
        };
        if !features
            .buffer_features
            .contains(vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER)
        {
            return Err(format!(
                "compute uniform texel format {:?} is unsupported",
                buffer.format
            ));
        }
    }
    if dispatch.sampled_rts.len() > backend.max_sampled_images.min(backend.max_samplers) as usize {
        return Err("too many compute sampled-image bindings".to_string());
    }
    let sampled_descriptor_count = dispatch
        .sampled_rts
        .len()
        .saturating_add(dispatch.sampled_images.len())
        .saturating_add(dispatch.uniform_texel_buffers.len());
    if sampled_descriptor_count > backend.max_sampled_images as usize {
        return Err("too many compute sampled image/texel descriptors".to_string());
    }
    if (dispatch.outputs.is_empty() && !dispatch.texel_buffers.iter().any(|buffer| buffer.writable))
        || dispatch.outputs.len() > backend.max_storage_images as usize
        || dispatch.outputs.len().saturating_add(texel_binding_count)
            > backend.max_storage_images as usize
    {
        return Err("invalid compute storage-image count".to_string());
    }
    for output in &dispatch.outputs {
        let sampled_cross_access = dispatch
            .image_aliases
            .iter()
            .any(|alias| alias.storage_binding == output.binding);
        let dimension_limit = if output.is_3d {
            backend.max_image_dimension_3d
        } else {
            backend.max_image_dimension_2d
        };
        if output.width == 0
            || output.height == 0
            || output.depth == 0
            || (!output.is_3d && output.depth != 1)
            || output.width > dimension_limit
            || output.height > dimension_limit
            || output.depth > dimension_limit
        {
            return Err(format!(
                "invalid compute storage image at binding {}: {}x{}x{}",
                output.binding, output.width, output.height, output.depth
            ));
        }
        let byte_len = output
            .width
            .checked_mul(output.height)
            .and_then(|pixels| pixels.checked_mul(output.depth))
            .and_then(|pixels| pixels.checked_mul(output.format.bytes_per_pixel() as u32))
            .ok_or_else(|| {
                format!(
                    "compute output byte size overflows at binding {}",
                    output.binding
                )
            })? as usize;
        if output
            .initial_bytes
            .as_ref()
            .is_some_and(|bytes| bytes.len() != byte_len)
        {
            return Err(format!(
                "compute output binding {} initial content is not tightly packed (got {}, expected {})",
                output.binding,
                output.initial_bytes.as_ref().map_or(0, Vec::len),
                byte_len
            ));
        }
        let mut format_features3 = vk::FormatProperties3::default();
        let mut format_features2 = vk::FormatProperties2 {
            s_type: vk::StructureType::FORMAT_PROPERTIES_2,
            p_next: &mut format_features3 as *mut _ as *mut std::ffi::c_void,
            ..Default::default()
        };
        unsafe {
            inner.instance.get_physical_device_format_properties2(
                inner.physical_device,
                output.format.vk_format(),
                &mut format_features2,
            );
        }
        let mut required_features = vk::FormatFeatureFlags2::STORAGE_IMAGE
            | vk::FormatFeatureFlags2::TRANSFER_SRC
            | vk::FormatFeatureFlags2::TRANSFER_DST
            | vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT;
        if sampled_cross_access {
            required_features |= vk::FormatFeatureFlags2::SAMPLED_IMAGE;
        }
        if !format_features3
            .optimal_tiling_features
            .contains(required_features)
        {
            return Err(format!(
                "compute output binding {} format {:?} lacks required features {:?}",
                output.binding, output.format, required_features
            ));
        }
    }
    let descriptors = crate::compute::descriptor_spec(dispatch);
    if descriptors.is_empty() || descriptors.len() > backend.max_resources as usize {
        return Err("invalid or excessive compute descriptor count".to_string());
    }
    for duplicate in descriptors.windows(2) {
        if duplicate[0].binding == duplicate[1].binding {
            return Err(format!(
                "compute descriptor binding {} is declared more than once",
                duplicate[0].binding
            ));
        }
    }
    Ok(())
}

fn compute_rt_alias(
    rt_cache: &RtCache,
    sampled: &crate::compute::ComputeSampledRt,
) -> Option<RtAlias> {
    compute_rt_alias_for_key(rt_cache, sampled.key, sampled.tic.format)
}

fn compute_rt_alias_for_key(
    rt_cache: &RtCache,
    key: RtKey,
    tic_format: crate::texture::TicFormat,
) -> Option<RtAlias> {
    let color = drawn_color_content_alias_for_key(rt_cache, key).map(
        |(key, image, view, layout, format)| RtAlias {
            key,
            image,
            view,
            layout,
            format,
            aspects: vk::ImageAspectFlags::COLOR,
            depth: false,
        },
    );
    let depth = rt_cache
        .find_depth(key)
        .filter(|(_, _, _, layout, _, _)| *layout != vk::ImageLayout::UNDEFINED)
        .or_else(|| rt_cache.find_d24_depth_covering(key))
        .filter(|(_, _, _, layout, _, _)| *layout != vk::ImageLayout::UNDEFINED)
        .or_else(|| rt_cache.find_current_depth_shadow(key))
        .map(|(key, image, view, layout, format, aspects)| RtAlias {
            key,
            image,
            view,
            layout,
            format,
            aspects,
            depth: true,
        });
    if tic_format_prefers_depth_alias(tic_format) {
        depth
    } else {
        color.or(depth)
    }
}

fn compute_layout_access(layout: vk::ImageLayout) -> vk::AccessFlags {
    match layout {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        }
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL => {
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        | vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
        | vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::GENERAL => vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        _ => vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
    }
}

fn post_submit_texture_probe_enabled(fs_gpu_va: u64) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    fs_gpu_va == 0x400028d30
        && *ENABLED.get_or_init(|| {
            std::env::var_os("NEXIUM_ATLAS_BIND_STATS")
                .is_some_and(|value| value.to_string_lossy().trim() != "0")
        })
}

fn claim_post_submit_texture_probe() -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static CLAIMED: AtomicBool = AtomicBool::new(false);
    CLAIMED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

fn wait_fence_no_reset(device: &ash::Device, fence: vk::Fence) -> Result<(), String> {
    unsafe {
        match device.wait_for_fences(&[fence], true, 2_000_000_000) {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => device
                .wait_for_fences(&[fence], true, 8_000_000_000)
                .map_err(|e| format!("post-submit wait_for_fences: {:?}", e)),
            Err(e) => Err(format!("post-submit wait_for_fences: {:?}", e)),
        }
    }
}

fn run_post_submit_texture_probe(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    batch_fence: vk::Fence,
    probes: [Option<PostSubmitTextureProbe>; 2],
) {
    if probes.iter().any(Option::is_none) || !claim_post_submit_texture_probe() {
        return;
    }
    if let Err(e) = wait_fence_no_reset(device, batch_fence) {
        log::warn!(
            "[atlas-bind-stats] fs=0x400028d30 batch_wait=failed error={}",
            e
        );
        return;
    }
    for (slot, probe) in probes.into_iter().enumerate() {
        let probe = probe.unwrap();
        match read_image_stats(
            device,
            cmd_pool,
            queue,
            mem_props,
            probe.key,
            probe.image,
            probe.layout,
            probe.format,
        ) {
            Some(stats) => {
                let pct = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_nonzero as f64 * 100.0 / stats.pixels as f64
                };
                let avg_rgb = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.rgb_sum as f64 / (stats.pixels as f64 * 3.0)
                };
                let avg_alpha = if stats.pixels == 0 {
                    0.0
                } else {
                    stats.alpha_sum as f64 / stats.pixels as f64
                };
                let first = stats
                    .first
                    .map(|(x, y, rgba)| {
                        format!(
                            "{},{}:{:02x}{:02x}{:02x}{:02x}",
                            x, y, rgba[0], rgba[1], rgba[2], rgba[3]
                        )
                    })
                    .unwrap_or_else(|| "-".to_string());
                log::warn!(
                    "[atlas-bind-stats] fs=0x400028d30 slot={} source={} key={} fmt={:?} layout={:?} rawbnz={} rawwnz={} rgbnz={}/{} ({:.2}%) avg_rgb={:.2} avg_a={:.2} max={} first={}",
                    slot,
                    probe.source,
                    probe.key.label(),
                    stats.format,
                    probe.layout,
                    stats.raw_nonzero_bytes,
                    stats.raw_nonzero_words,
                    stats.rgb_nonzero,
                    stats.pixels,
                    pct,
                    avg_rgb,
                    avg_alpha,
                    stats.rgb_max,
                    first
                );
            }
            None => log::warn!(
                "[atlas-bind-stats] fs=0x400028d30 slot={} source={} key={} fmt={:?} layout={:?} readback=failed",
                slot,
                probe.source,
                probe.key.label(),
                probe.format,
                probe.layout
            ),
        }
    }
}

fn read_rt_image_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    key: RtKey,
) -> Option<RtImageStats> {
    let (image, layout, format) = {
        let existing = rt_cache.get_existing(key)?;
        (existing.image, existing.layout, existing.format)
    };
    read_image_stats(
        device, cmd_pool, queue, mem_props, key, image, layout, format,
    )
}

fn read_image_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    key: RtKey,
    image: vk::Image,
    prev_layout: vk::ImageLayout,
    format: vk::Format,
) -> Option<RtImageStats> {
    let total = (key.width as u64)
        .checked_mul(key.height as u64)?
        .checked_mul(readback_format_bpp(format) as u64)?;
    let stage = create_staging_owned(device, mem_props, total).ok()?;
    let cleanup = |device: &ash::Device,
                   cmd_pool: vk::CommandPool,
                   fence: Option<vk::Fence>,
                   cmd: Option<vk::CommandBuffer>,
                   stage: &StagingBuffer| unsafe {
        if let Some(c) = cmd {
            device.free_command_buffers(cmd_pool, &[c]);
        }
        if let Some(f) = fence {
            device.destroy_fence(f, None);
        }
        device.destroy_buffer(stage.buffer, None);
        device.free_memory(stage.memory, None);
    };
    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::empty(),
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let fence = match unsafe { device.create_fence(&fence_info, None) } {
        Ok(f) => f,
        Err(_) => {
            cleanup(device, cmd_pool, None, None, &stage);
            return None;
        }
    };
    let cmd = match alloc_one_time_cmd(device, cmd_pool) {
        Ok(c) => c,
        Err(_) => {
            cleanup(device, cmd_pool, Some(fence), None, &stage);
            return None;
        }
    };
    if begin_one_time(device, cmd).is_err() {
        cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
        return None;
    }
    transition_image(
        device,
        cmd,
        image,
        prev_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stage.buffer,
            &[copy],
        );
    }
    if prev_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            prev_layout,
        );
    }
    if end_one_time(device, cmd).is_err() || submit_with_fence(device, queue, cmd, fence).is_err() {
        cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
        return None;
    }
    let mut stats = RtImageStats {
        pixels: (key.width as u64) * (key.height as u64),
        format,
        ..RtImageStats::default()
    };
    unsafe {
        if wait_fence(device, fence).is_err() {
            cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
            return None;
        }
        let ptr = match device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => ptr as *const u8,
            Err(_) => {
                cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        };
        let data = std::slice::from_raw_parts(ptr, total as usize);
        for (idx, byte) in data.iter().enumerate() {
            if *byte != 0 {
                stats.raw_nonzero_bytes += 1;
                if stats.raw_first_word.is_none() {
                    let pixel_size = readback_format_bpp(format).max(1);
                    let pixel = idx / pixel_size;
                    let word_start = (idx / 4) * 4;
                    let mut raw = [0u8; 4];
                    let available = data.len().saturating_sub(word_start).min(4);
                    raw[..available].copy_from_slice(&data[word_start..word_start + available]);
                    stats.raw_first_word = Some((
                        (pixel as u32) % key.width,
                        (pixel as u32) / key.width,
                        u32::from_le_bytes(raw),
                    ));
                }
            }
        }
        for word in data.chunks(4) {
            if word.iter().any(|byte| *byte != 0) {
                stats.raw_nonzero_words += 1;
            }
        }
        {
            let pixel_size = readback_format_bpp(format).max(1);
            let cx = key.width / 2;
            let cy = key.height * 5 / 8;
            let idx = (cy as usize * key.width as usize + cx as usize) * pixel_size;
            if idx + 4 <= data.len() {
                let raw = [data[idx], data[idx + 1], data[idx + 2], data[idx + 3]];
                stats.raw_mid_word = Some((cx, cy, u32::from_le_bytes(raw)));
            }
        }
        let rgba = readback_to_rgba8(data, format, key.width, key.height);
        if std::env::var_os("NEXIUM_RT_DUMP").is_some() {
            dump_rt_bmp(key, &rgba);
        }
        for (i, px) in rgba.chunks_exact(4).enumerate() {
            let r = px[0];
            let g = px[1];
            let b = px[2];
            let a = px[3];
            stats.rgb_sum += r as u64 + g as u64 + b as u64;
            stats.alpha_sum += a as u64;
            stats.rgb_max = stats.rgb_max.max(r).max(g).max(b);
            if a != 0 {
                stats.alpha_nonzero += 1;
            }
            if r != 0 || g != 0 || b != 0 {
                let x = (i as u32) % key.width;
                let y = (i as u32) / key.width;
                stats.rgb_nonzero += 1;
                stats.first.get_or_insert((x, y, [r, g, b, a]));
                stats.bbox = Some(match stats.bbox {
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    None => (x, y, x, y),
                });
            }
        }
        if rt_pixels_enabled() || volume_pixels_enabled() {
            let w = key.width.min(rt_pixel_limit("NEXIUM_RT_PIXELS_W", 8, 64));
            let h = key.height.min(rt_pixel_limit("NEXIUM_RT_PIXELS_H", 8, 64));
            for y in 0..h {
                let mut cells = Vec::with_capacity(w as usize);
                for x in 0..w {
                    let off = ((y as usize * key.width as usize) + x as usize) * 4;
                    cells.push(format!(
                        "{:02x}{:02x}{:02x}{:02x}",
                        rgba[off],
                        rgba[off + 1],
                        rgba[off + 2],
                        rgba[off + 3]
                    ));
                }
                stats.pixel_rows.push(format!("y{}={}", y, cells.join(" ")));
            }
        }
        device.unmap_memory(stage.memory);
    }
    cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
    Some(stats)
}

fn readback_format_bpp(format: vk::Format) -> usize {
    match format {
        vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_UINT => 16,
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_UINT => 8,
        vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_SINT
        | vk::Format::R16_UINT
        | vk::Format::R16_SFLOAT
        | vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_UINT
        | vk::Format::R5G6B5_UNORM_PACK16 => 2,
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_SINT | vk::Format::R8_UINT => {
            1
        }
        _ => 4,
    }
}

fn content_readback_key(
    rt_cache: &RtCache,
    nvmap_id: u32,
    gpu_va: u64,
    want_bpp: usize,
) -> Option<RtKey> {
    let cands = rt_cache.color_keys_for_nvmap(nvmap_id);
    let dump = std::env::var_os("NEXIUM_ALIAS_DUMP").is_some();
    if dump {
        for (k, fmt, stamp, real) in cands.iter().filter(|(k, ..)| k.gpu_va == gpu_va) {
            log::warn!(
                "[alias-dump] nvmap={} va={:#x} key={} fmt={:?} bpp={} stamp={} real_draws={}",
                nvmap_id,
                gpu_va,
                k.label(),
                fmt,
                readback_format_bpp(*fmt),
                stamp,
                real
            );
        }
    }
    let baseline = cands
        .iter()
        .filter(|(k, ..)| k.gpu_va == gpu_va)
        .max_by_key(|(_, _, stamp, _)| *stamp)?;
    let baseline_key = baseline.0;
    let baseline_real = baseline.3;
    if baseline_real > 0 || std::env::var_os("NEXIUM_NO_ALIAS_CONTENT_TIEBREAK").is_some() {
        return Some(baseline_key);
    }
    let bw = baseline_key.width;
    let bh = baseline_key.height;
    let pick = cands
        .iter()
        .filter(|(k, fmt, _, real)| {
            k.gpu_va == gpu_va && *real > 0 && readback_format_bpp(*fmt) == want_bpp
        })
        .max_by_key(|(_, _, stamp, _)| *stamp)
        .or_else(|| {
            cands
                .iter()
                .filter(|(k, fmt, _, real)| {
                    k.width == bw
                        && k.height == bh
                        && *real > 0
                        && readback_format_bpp(*fmt) == want_bpp
                })
                .max_by_key(|(_, _, stamp, _)| *stamp)
        })
        .map(|(k, ..)| *k);
    match pick {
        Some(k) => {
            if dump {
                log::warn!(
                    "[alias-dump] nvmap={} va={:#x} content-tiebreak baseline={} empty -> key={}",
                    nvmap_id,
                    gpu_va,
                    baseline_key.label(),
                    k.label()
                );
            }
            Some(k)
        }
        None => Some(baseline_key),
    }
}

fn readback_to_rgba8(src: &[u8], format: vk::Format, width: u32, height: u32) -> Vec<u8> {
    let pixels = width as usize * height as usize;
    let mut out = vec![0u8; pixels.saturating_mul(4)];
    match format {
        vk::Format::A2B10G10R10_UNORM_PACK32 => {
            for i in 0..pixels.min(src.len() / 4) {
                let off = i * 4;
                let v = u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]]);
                let r = v & 0x3ff;
                let g = (v >> 10) & 0x3ff;
                let b = (v >> 20) & 0x3ff;
                let a = (v >> 30) & 0x3;
                out[off] = ((r * 255 + 511) / 1023) as u8;
                out[off + 1] = ((g * 255 + 511) / 1023) as u8;
                out[off + 2] = ((b * 255 + 511) / 1023) as u8;
                out[off + 3] = ((a * 255 + 1) / 3) as u8;
            }
        }
        vk::Format::B10G11R11_UFLOAT_PACK32 => {
            return crate::texture::decode_to_rgba8(
                src,
                width,
                height,
                crate::texture::TicFormat::B10G11R11,
            );
        }
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
            for i in 0..pixels.min(src.len() / 4) {
                let off = i * 4;
                out[off] = src[off + 2];
                out[off + 1] = src[off + 1];
                out[off + 2] = src[off];
                out[off + 3] = src[off + 3];
            }
        }
        _ => {
            let n = out.len().min(src.len());
            out[..n].copy_from_slice(&src[..n]);
        }
    }
    out
}

fn legacy_present_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_LEGACY_PRESENT").is_some())
}

fn readout_present_rgba8(
    mut raw: Vec<u8>,
    format: vk::Format,
    width: u32,
    height: u32,
    vflip: bool,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let row = w * 4;
    let need = row.saturating_mul(h);
    if raw.len() < need {
        raw.resize(need, 0);
    }
    match format {
        vk::Format::A2B10G10R10_UNORM_PACK32 => {
            let mut out = vec![0u8; need];
            for y in 0..h {
                let sy = if vflip { h - 1 - y } else { y };
                let src_row = &raw[sy * row..sy * row + row];
                let dst_row = &mut out[y * row..y * row + row];
                for x in 0..w {
                    let off = x * 4;
                    let v = u32::from_le_bytes([
                        src_row[off],
                        src_row[off + 1],
                        src_row[off + 2],
                        src_row[off + 3],
                    ]);
                    let r = v & 0x3ff;
                    let g = (v >> 10) & 0x3ff;
                    let b = (v >> 20) & 0x3ff;
                    dst_row[off] = ((r * 255 + 511) / 1023) as u8;
                    dst_row[off + 1] = ((g * 255 + 511) / 1023) as u8;
                    dst_row[off + 2] = ((b * 255 + 511) / 1023) as u8;
                    dst_row[off + 3] = 0xFF;
                }
            }
            out
        }
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
            let mut out = vec![0u8; need];
            for y in 0..h {
                let sy = if vflip { h - 1 - y } else { y };
                let src_row = &raw[sy * row..sy * row + row];
                let dst_row = &mut out[y * row..y * row + row];
                for x in 0..w {
                    let off = x * 4;
                    dst_row[off] = src_row[off + 2];
                    dst_row[off + 1] = src_row[off + 1];
                    dst_row[off + 2] = src_row[off];
                    dst_row[off + 3] = 0xFF;
                }
            }
            out
        }
        vk::Format::B10G11R11_UFLOAT_PACK32 => {
            let mut out = crate::texture::decode_to_rgba8(
                &raw,
                width,
                height,
                crate::texture::TicFormat::B10G11R11,
            );
            if vflip {
                flip_rows_v(&mut out, width, height);
            }
            for px in out.chunks_exact_mut(4) {
                px[3] = 0xFF;
            }
            out
        }
        _ => {
            if vflip {
                flip_rows_v(&mut raw, width, height);
            }
            for px in raw.chunks_exact_mut(4) {
                px[3] = 0xFF;
            }
            raw
        }
    }
}

fn flip_rows_v(bytes: &mut [u8], width: u32, height: u32) {
    let row = width as usize * 4;
    let h = height as usize;
    if row == 0 || h == 0 || bytes.len() < row * h {
        return;
    }
    for y in 0..h / 2 {
        let top = y * row;
        let bot = (h - 1 - y) * row;
        let (a, b) = bytes.split_at_mut(bot);
        a[top..top + row].swap_with_slice(&mut b[..row]);
    }
}

fn max_texture_descriptors() -> usize {
    crate::descriptor::MAX_TEXTURE_DESCRIPTORS as usize
}

#[derive(Debug, Default, PartialEq, Eq)]
struct BindTraceFilter {
    all: bool,
    addresses: Vec<u64>,
    hashes: Vec<u64>,
}

fn parse_bind_trace_filter(value: &str) -> BindTraceFilter {
    let mut filter = BindTraceFilter::default();
    for raw in value.split(',') {
        let token = raw.trim();
        if token.eq_ignore_ascii_case("all") {
            filter.all = true;
            continue;
        }
        if let Some((prefix, value)) = token.split_once(':') {
            if prefix.eq_ignore_ascii_case("hash") {
                if let Ok(hash) = u64::from_str_radix(
                    value
                        .trim()
                        .trim_start_matches("0x")
                        .trim_start_matches("0X"),
                    16,
                ) {
                    filter.hashes.push(hash);
                }
                continue;
            }
        }
        if let Ok(address) =
            u64::from_str_radix(token.trim_start_matches("0x").trim_start_matches("0X"), 16)
        {
            filter.addresses.push(address);
        }
    }
    filter
}

fn bind_trace_fs(fs_gpu_va: u64, fs_hash: u64) -> bool {
    static FILTER: std::sync::OnceLock<BindTraceFilter> = std::sync::OnceLock::new();
    let filter = FILTER.get_or_init(|| {
        std::env::var("NEXIUM_BIND_TRACE_FS")
            .map(|value| parse_bind_trace_filter(&value))
            .unwrap_or_default()
    });
    filter.all || filter.addresses.contains(&fs_gpu_va) || filter.hashes.contains(&fs_hash)
}

fn vs_tex_slot(call: &crate::draw::Maxwell3dDrawCall, slot: usize) -> Option<(usize, u32)> {
    let base = call.vs_tex_base as usize;
    let count = call.vs_tex_count as usize;
    if count == 0 || slot < base || slot >= base.saturating_add(count) {
        return None;
    }
    Some((
        slot - base,
        call.fs_tex_ids.get(slot).copied().unwrap_or(u32::MAX),
    ))
}

fn vs_tex_bind_trace(call: &crate::draw::Maxwell3dDrawCall) -> bool {
    if call.vs_tex_count == 0 {
        return false;
    }
    if let Ok(list) = std::env::var("NEXIUM_VS_TEX_BIND_FS") {
        return list
            .split(',')
            .filter_map(|part| parse_u64_value(part.trim()))
            .any(|addr| addr == call.fs_gpu_va);
    }
    std::env::var_os("NEXIUM_VS_TEX_BIND").is_some()
}

fn rt_stamp(rt_cache: &RtCache, key: RtKey) -> u64 {
    rt_cache
        .debug_all()
        .into_iter()
        .find_map(|(k, stamp)| if k == key { Some(stamp) } else { None })
        .unwrap_or(0)
}

fn trace_vs_tex_bind_alias(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    alias: RtAlias,
    view_format: vk::Format,
    bound: vk::ImageView,
) {
    let Some((vs_slot, tex_id)) = vs_tex_slot(call, slot) else {
        return;
    };
    if !vs_tex_bind_trace(call) {
        return;
    }
    log::warn!(
        "[vs-tex-bind] vs={:#x} fs={:#x} vs_slot={} slot={} tex_id={} base={} count={} source=ALIAS key={} fmt={:?}->{:?} stamp={} depth={} bound={:?}",
        call.vs_gpu_va,
        call.fs_gpu_va,
        vs_slot,
        slot,
        tex_id,
        call.vs_tex_base,
        call.vs_tex_count,
        alias.key.label(),
        alias.format,
        view_format,
        rt_stamp(rt_cache, alias.key),
        alias.depth,
        bound
    );
    trace_vs_tex_bind_alias_stats(
        device, cmd_pool, queue, rt_cache, mem_props, call, vs_slot, slot, tex_id, alias,
    );
}

fn trace_vs_tex_bind_dummy<F>(call: &crate::draw::Maxwell3dDrawCall, slot: usize, read_guest: &F)
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    let Some((vs_slot, tex_id)) = vs_tex_slot(call, slot) else {
        return;
    };
    if !vs_tex_bind_trace(call) {
        return;
    }
    let reason = vs_tex_dummy_reason(call, tex_id, read_guest);
    log::warn!(
        "[vs-tex-bind] vs={:#x} fs={:#x} vs_slot={} slot={} tex_id={} base={} count={} source=DUMMY tic_pool={:#x} limit={} reason={}",
        call.vs_gpu_va,
        call.fs_gpu_va,
        vs_slot,
        slot,
        tex_id,
        call.vs_tex_base,
        call.vs_tex_count,
        call.tic_pool_gpu_va,
        call.tic_pool_limit,
        reason
    );
}

fn vs_tex_dummy_reason<F>(
    call: &crate::draw::Maxwell3dDrawCall,
    tex_id: u32,
    read_guest: &F,
) -> String
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    if tex_id == u32::MAX {
        return "tex-id-invalid".to_string();
    }
    if call.tic_pool_gpu_va == 0 {
        return "tic-pool-zero".to_string();
    }
    if tex_id > call.tic_pool_limit {
        return format!(
            "tic-out-of-range id={} limit={}",
            tex_id, call.tic_pool_limit
        );
    }
    let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
    let Some(raw) = read_guest(tic_addr, 32) else {
        return format!("tic-read-fail addr={:#x}", tic_addr);
    };
    let raw_hex = raw
        .iter()
        .take(32)
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join("");
    match crate::texture::TicEntry::parse(&raw) {
        Some(tic) => format!(
            "tic-parse-ok-unexpected addr={:#x} va={:#x} {}x{} fmt={:?} raw={}",
            tic_addr, tic.gpu_va, tic.width, tic.height, tic.format, raw_hex
        ),
        None => format!("tic-parse-fail addr={:#x} raw={}", tic_addr, raw_hex),
    }
}

fn trace_vs_tex_bind_texture(
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    key: TexCacheKey,
    tic: crate::texture::TicEntry,
    cache_hit: bool,
    bound: vk::ImageView,
) {
    let Some((vs_slot, tex_id)) = vs_tex_slot(call, slot) else {
        return;
    };
    if !vs_tex_bind_trace(call) {
        return;
    }
    log::warn!(
        "[vs-tex-bind] vs={:#x} fs={:#x} vs_slot={} slot={} tex_id={} base={} count={} source=TEX va={:#x} {}x{}x{} fmt={:?} vol={} cache_hit={} bound={:?}",
        call.vs_gpu_va,
        call.fs_gpu_va,
        vs_slot,
        slot,
        tex_id,
        call.vs_tex_base,
        call.vs_tex_count,
        tic.gpu_va,
        tic.width,
        tic.height,
        key.layers,
        tic.format,
        key.volume,
        cache_hit,
        bound
    );
}

fn trace_vs_tex_bind_alias_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    call: &crate::draw::Maxwell3dDrawCall,
    vs_slot: usize,
    slot: usize,
    tex_id: u32,
    alias: RtAlias,
) {
    if alias.depth || std::env::var_os("NEXIUM_VS_TEX_BIND_STATS").is_none() {
        return;
    }
    {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<std::collections::HashSet<(u64, u64, usize, RtKey)>>> =
            OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
        if !seen
            .lock()
            .unwrap()
            .insert((call.vs_gpu_va, call.fs_gpu_va, slot, alias.key))
        {
            return;
        }
    }
    let stamp = rt_stamp(rt_cache, alias.key);
    match read_rt_image_stats(device, cmd_pool, queue, rt_cache, mem_props, alias.key) {
        Some(stats) => {
            let avg_rgb = if stats.pixels == 0 {
                0.0
            } else {
                stats.rgb_sum as f64 / (stats.pixels as f64 * 3.0)
            };
            let avg_alpha = if stats.pixels == 0 {
                0.0
            } else {
                stats.alpha_sum as f64 / stats.pixels as f64
            };
            let first = stats
                .first
                .map(|(x, y, rgba)| {
                    format!(
                        "{},{}:{:02x}{:02x}{:02x}{:02x}",
                        x, y, rgba[0], rgba[1], rgba[2], rgba[3]
                    )
                })
                .unwrap_or_else(|| "-".to_string());
            log::warn!(
                "[vs-tex-bind-stats] vs={:#x} fs={:#x} vs_slot={} slot={} tex_id={} key={} fmt={:?} stamp={} rawbnz={} rawwnz={} rgbnz={}/{} avg_rgb={:.2} avg_a={:.2} max={} first={}",
                call.vs_gpu_va,
                call.fs_gpu_va,
                vs_slot,
                slot,
                tex_id,
                alias.key.label(),
                stats.format,
                stamp,
                stats.raw_nonzero_bytes,
                stats.raw_nonzero_words,
                stats.rgb_nonzero,
                stats.pixels,
                avg_rgb,
                avg_alpha,
                stats.rgb_max,
                first
            );
        }
        None => {
            log::warn!(
                "[vs-tex-bind-stats] vs={:#x} fs={:#x} vs_slot={} slot={} tex_id={} key={} stamp={} readback=failed",
                call.vs_gpu_va,
                call.fs_gpu_va,
                vs_slot,
                slot,
                tex_id,
                alias.key.label(),
                stamp
            );
        }
    }
}

fn verify_volume_image(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
    width: u32,
    height: u32,
    layers: u32,
    va: u64,
) {
    use ash::vk::Handle;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashMap<u64, u32>>> = OnceLock::new();
    {
        let mut seen = SEEN
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        let count = seen.entry(image.as_raw()).or_insert(0);
        *count += 1;
        if *count != 3 {
            return;
        }
    }
    let size = (width as usize) * (height as usize) * (layers as usize) * 4;
    let Ok(buf) = create_staging_owned(device, mem_props, size as u64) else {
        return;
    };
    let result = (|| -> Result<Vec<u8>, String> {
        let cmd = alloc_one_time_cmd(device, cmd_pool)?;
        begin_one_time(device, cmd)?;
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: layers,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buf.buffer,
                &[copy],
            );
        }
        transition_image(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        end_one_time(device, cmd)?;
        submit_and_wait(device, queue, cmd)?;
        let mut out = vec![0u8; size];
        unsafe {
            let ptr = device
                .map_memory(buf.memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map_memory: {:?}", e))?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), size);
            device.unmap_memory(buf.memory);
            device.free_command_buffers(cmd_pool, &[cmd]);
        }
        Ok(out)
    })();
    unsafe {
        device.destroy_buffer(buf.buffer, None);
        device.free_memory(buf.memory, None);
    }
    match result {
        Ok(data) => {
            let slice_bytes = (width as usize) * (height as usize) * 4;
            for z in 0..layers as usize {
                let base = z * slice_bytes;
                let w00 = u32::from_le_bytes(data[base..base + 4].try_into().unwrap_or_default());
                let mid = base + ((height as usize / 2) * width as usize + width as usize / 2) * 4;
                let wmid = u32::from_le_bytes(data[mid..mid + 4].try_into().unwrap_or_default());
                log::warn!(
                    "[volume-verify] va={:#x} z={} w00={:08x} wmid={:08x}",
                    va,
                    z,
                    w00,
                    wmid
                );
            }
        }
        Err(e) => log::warn!("[volume-verify] va={:#x} failed: {}", va, e),
    }
}

fn async_shaders_enabled() -> bool {
    if std::env::var_os("NEXIUM_SYNC_SHADERS").is_some() {
        return false;
    }
    nexium_common::async_compile::enabled() || std::env::var_os("NEXIUM_ASYNC_SHADERS").is_some()
}

fn tex_gen_gating_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var_os("NEXIUM_TEX_GEN_GATING").map_or(true, |value| {
            !matches!(
                value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
    })
}

fn force_refresh_texture(gpu_va: u64) -> bool {
    if std::env::var_os("NEXIUM_TEX_FORCE_REFRESH")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }
    let Some(target) = std::env::var_os("NEXIUM_TEX_FORCE_REFRESH_VA") else {
        return false;
    };
    let s = target.to_string_lossy();
    let s = s.trim().trim_start_matches("0x");
    u64::from_str_radix(s, 16)
        .map(|target| target == gpu_va)
        .unwrap_or(false)
}

fn texture_seed_hash(key: &TexCacheKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn tic_is_arrayed(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 5
}

fn tic_is_volume(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 2
}

fn tic_is_cube(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 3
}

fn tic_is_cube_array(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 8
}

fn tic_requires_dedicated_sampled_view(tic: &crate::texture::TicEntry) -> bool {
    tic.is_buffer()
        || tic_is_arrayed(tic)
        || tic_is_volume(tic)
        || tic_is_cube(tic)
        || tic_is_cube_array(tic)
}

fn tic_layer_count(tic: &crate::texture::TicEntry) -> u32 {
    if tic_is_arrayed(tic) {
        tic.depth.max(1)
    } else if tic_is_cube(tic) {
        6
    } else if tic_is_cube_array(tic) {
        tic.depth.saturating_mul(6).max(6)
    } else if tic_is_volume(tic) {
        tic.depth.max(1)
    } else {
        1
    }
}

fn tic_view_base_layer(_tic: &crate::texture::TicEntry) -> u32 {
    0
}

fn tic_view_layer_count(tic: &crate::texture::TicEntry) -> u32 {
    if tic_is_arrayed(tic) {
        tic.depth.max(1)
    } else if tic_is_cube(tic) {
        6
    } else if tic_is_cube_array(tic) {
        tic.depth.saturating_mul(6).max(6)
    } else {
        1
    }
}

fn tic_layer_read_size(tic: &crate::texture::TicEntry, pitch_size: usize) -> usize {
    if tic.is_block_linear {
        tic.format
            .block_linear_size(tic.width, tic.height, tic.block_height_log2)
            .max(pitch_size)
    } else {
        pitch_size
    }
}

fn tic_read_size(tic: &crate::texture::TicEntry, pitch_size: usize, layers: u32) -> usize {
    if let Some(size) = crate::texture::texture_guest_size_bytes(tic, layers) {
        size
    } else if tic.is_block_linear && tic_is_volume(tic) {
        block_linear_volume_byte_size(tic, layers).max(
            tic.format
                .linear_size(tic.width, tic.height)
                .saturating_mul(layers as usize),
        )
    } else {
        tic_layer_read_size(tic, pitch_size).saturating_mul(layers as usize)
    }
}

fn linear_texture_layers(
    raw: &[u8],
    tic: &crate::texture::TicEntry,
    layer_count: u32,
    pitch_size: usize,
    force_pitch: bool,
) -> Vec<u8> {
    let layers = layer_count.max(1) as usize;
    let layer_linear_size = tic.format.linear_size(tic.width, tic.height);
    let effective_block_linear =
        tic.is_block_linear && !crate::pitch_oracle::is_pitch_dst(tic.gpu_va);
    if effective_block_linear && !force_pitch && tic_is_volume(tic) {
        let (storage_width, storage_height, bpp) = tic.format.storage_extent(tic.width, tic.height);
        let read_size = block_linear_volume_byte_size(tic, layers as u32).min(raw.len());
        let mut out = crate::texture::unswizzle_block_linear_3d(
            &raw[..read_size],
            storage_width,
            storage_height,
            layers as u32,
            bpp,
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
        );
        out.resize(layer_linear_size.saturating_mul(layers), 0);
        return out;
    }

    let mip_layout = if effective_block_linear && !force_pitch {
        crate::texture::block_linear_mip_layout(tic)
    } else {
        None
    };
    let layer_read_size = mip_layout
        .as_ref()
        .map(|layout| layout.layer_stride)
        .unwrap_or_else(|| tic_layer_read_size(tic, pitch_size));
    let mut out = Vec::with_capacity(layer_linear_size.saturating_mul(layers));
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_read_size);
        if start >= raw.len() {
            out.resize(out.len() + layer_linear_size, 0);
            break;
        }
        let base_guest_size = mip_layout
            .as_ref()
            .and_then(|layout| layout.levels.first())
            .map(|level| level.guest_size)
            .unwrap_or(layer_read_size);
        let end = start.saturating_add(base_guest_size).min(raw.len());
        let layer_raw = &raw[start..end];
        let mut linear: Vec<u8> = if effective_block_linear && !force_pitch {
            let (storage_width, storage_height, bpp) =
                tic.format.storage_extent(tic.width, tic.height);
            let block_height_log2 = mip_layout
                .as_ref()
                .and_then(|layout| layout.levels.first())
                .map(|level| level.block_height_log2)
                .unwrap_or(tic.block_height_log2);
            crate::texture::unswizzle_block_linear(
                layer_raw,
                storage_width,
                storage_height,
                bpp,
                block_height_log2,
            )
        } else if layer_raw.len() >= pitch_size {
            layer_raw[..pitch_size].to_vec()
        } else {
            layer_raw.to_vec()
        };
        linear.resize(layer_linear_size, 0);
        out.extend(linear);
    }
    out.resize(layer_linear_size.saturating_mul(layers), 0);
    out
}

fn decode_texture_rgba8_layers(
    raw: &[u8],
    tic: &crate::texture::TicEntry,
    layer_count: u32,
    pitch_size: usize,
    force_pitch: bool,
) -> Vec<u8> {
    let layers = layer_count.max(1) as usize;
    let linear = linear_texture_layers(raw, tic, layer_count, pitch_size, force_pitch);
    let mut out = Vec::new();
    let layer_rgba_size = tic.width as usize * tic.height as usize * 4;
    let layer_linear_size = tic.format.linear_size(tic.width, tic.height);
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_linear_size);
        if start >= linear.len() {
            out.resize(out.len() + layer_rgba_size, 0);
            break;
        }
        let end = (start + layer_linear_size).min(linear.len());
        let mut decoded =
            crate::texture::decode_to_rgba8(&linear[start..end], tic.width, tic.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        out.extend(decoded);
    }
    out.resize(layer_rgba_size.saturating_mul(layers), 0);
    out
}

fn texture_upload_data(
    raw: &[u8],
    tic: &crate::texture::TicEntry,
    layer_count: u32,
    pitch_size: usize,
    force_pitch: bool,
    format: vk::Format,
) -> Result<TextureUploadData, String> {
    let layers = layer_count.max(1);
    let effective_block_linear =
        tic.is_block_linear && !crate::pitch_oracle::is_pitch_dst(tic.gpu_va);
    if !tic_is_volume(tic) && tic.mip_levels() > 1 && (!effective_block_linear || force_pitch) {
        return Err("multi-mip pitch-linear upload is not implemented".to_string());
    }
    if effective_block_linear && !force_pitch && !tic_is_volume(tic) {
        if tic.mip_levels() > 1 && (tic.block_width_log2 != 0 || tic.block_depth_log2 != 0) {
            return Err(format!(
                "unsupported multi-mip 2D block geometry bw={} bd={}",
                tic.block_width_log2, tic.block_depth_log2
            ));
        }
        let layout = crate::texture::block_linear_mip_layout(tic)
            .ok_or_else(|| "invalid block-linear mip layout".to_string())?;
        let required = layout.guest_size_bytes(layers);
        if raw.len() < required {
            return Err(format!(
                "short block-linear mip read: got {} need {}",
                raw.len(),
                required
            ));
        }
        let mut bytes = Vec::new();
        let mut copies = Vec::with_capacity(layout.levels.len());
        for level in &layout.levels {
            let buffer_offset = bytes.len() as u64;
            for layer in 0..layers as usize {
                let start = layer
                    .saturating_mul(layout.layer_stride)
                    .saturating_add(level.guest_offset);
                let end = start.saturating_add(level.guest_size);
                let mut linear = crate::texture::unswizzle_block_linear_strided(
                    &raw[start..end],
                    level.storage_width,
                    level.storage_height,
                    tic.format.src_bpp(),
                    level.block_height_log2,
                    level.stride_alignment_log2,
                );
                linear.resize(level.linear_size, 0);
                bytes.extend(texture_level_upload(
                    &linear,
                    tic.format,
                    level.width,
                    level.height,
                    1,
                    format,
                ));
            }
            copies.push(TextureMipCopy {
                buffer_offset,
                mip_level: level.level,
                width: level.width,
                height: level.height,
            });
        }
        return Ok(TextureUploadData { bytes, copies });
    }

    let linear = linear_texture_layers(raw, tic, layers, pitch_size, force_pitch);
    let bytes = texture_level_upload(&linear, tic.format, tic.width, tic.height, layers, format);
    Ok(TextureUploadData::base(bytes, tic.width, tic.height))
}

fn texture_level_upload(
    linear: &[u8],
    tic_format: crate::texture::TicFormat,
    width: u32,
    height: u32,
    layers: u32,
    format: vk::Format,
) -> Vec<u8> {
    let native_layout = matches!(
        (tic_format, format),
        (
            crate::texture::TicFormat::R8,
            vk::Format::R8_UNORM
                | vk::Format::R8_SNORM
                | vk::Format::R8_UINT
                | vk::Format::R8_SINT
                | vk::Format::R8_SRGB
        ) | (
            crate::texture::TicFormat::R8G8,
            vk::Format::R8G8_UNORM
                | vk::Format::R8G8_SNORM
                | vk::Format::R8G8_UINT
                | vk::Format::R8G8_SINT
                | vk::Format::R8G8_SRGB
        ) | (
            crate::texture::TicFormat::R8G8B8A8,
            vk::Format::R8G8B8A8_UNORM
                | vk::Format::R8G8B8A8_SNORM
                | vk::Format::R8G8B8A8_UINT
                | vk::Format::R8G8B8A8_SINT
                | vk::Format::R8G8B8A8_SRGB
        ) | (
            crate::texture::TicFormat::R16,
            vk::Format::R16_SFLOAT
                | vk::Format::R16_UNORM
                | vk::Format::R16_SNORM
                | vk::Format::R16_UINT
                | vk::Format::R16_SINT
        ) | (
            crate::texture::TicFormat::R16G16,
            vk::Format::R16G16_SFLOAT
                | vk::Format::R16G16_UNORM
                | vk::Format::R16G16_SNORM
                | vk::Format::R16G16_UINT
                | vk::Format::R16G16_SINT
        ) | (
            crate::texture::TicFormat::R16G16B16A16,
            vk::Format::R16G16B16A16_SFLOAT
                | vk::Format::R16G16B16A16_UNORM
                | vk::Format::R16G16B16A16_SNORM
                | vk::Format::R16G16B16A16_UINT
                | vk::Format::R16G16B16A16_SINT
        ) | (
            crate::texture::TicFormat::R32,
            vk::Format::R32_SFLOAT | vk::Format::R32_UINT | vk::Format::R32_SINT
        ) | (
            crate::texture::TicFormat::R32G32,
            vk::Format::R32G32_SFLOAT | vk::Format::R32G32_UINT | vk::Format::R32G32_SINT
        ) | (
            crate::texture::TicFormat::R32G32B32A32,
            vk::Format::R32G32B32A32_SFLOAT
                | vk::Format::R32G32B32A32_UINT
                | vk::Format::R32G32B32A32_SINT
        )
    );
    if native_layout
        || tic_format == crate::texture::TicFormat::R16G16B16A16
        || (tic_format == crate::texture::TicFormat::R16 && format == vk::Format::R16_SFLOAT)
    {
        linear.to_vec()
    } else if tic_format == crate::texture::TicFormat::G24R8 {
        let numeric_type = if format == vk::Format::R32_UINT {
            nexium_spirv::TextureNumericType::Uint
        } else if format == vk::Format::R32_SINT {
            nexium_spirv::TextureNumericType::Sint
        } else {
            nexium_spirv::TextureNumericType::Float
        };
        g24r8_scalar_upload(linear, numeric_type)
    } else if format == vk::Format::B10G11R11_UFLOAT_PACK32 {
        linear.to_vec()
    } else {
        let layer_size = tic_format.linear_size(width, height);
        let decoded_layer_size = width as usize * height as usize * 4;
        let mut out = Vec::with_capacity(decoded_layer_size.saturating_mul(layers as usize));
        for layer in 0..layers as usize {
            let start = layer.saturating_mul(layer_size);
            let end = start.saturating_add(layer_size).min(linear.len());
            if start >= linear.len() {
                out.resize(out.len().saturating_add(decoded_layer_size), 0);
                continue;
            }
            let mut decoded =
                crate::texture::decode_to_rgba8(&linear[start..end], width, height, tic_format);
            decoded.resize(decoded_layer_size, 0);
            out.extend(decoded);
        }
        out
    }
}

fn g24r8_scalar_upload(linear: &[u8], numeric_type: nexium_spirv::TextureNumericType) -> Vec<u8> {
    let mut out = Vec::with_capacity(linear.len() / 4 * 4);
    for texel in linear.chunks_exact(4) {
        let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
        match numeric_type {
            nexium_spirv::TextureNumericType::Float => {
                let depth = ((packed >> 8) as f32) / 16_777_215.0;
                out.extend_from_slice(&depth.to_le_bytes());
            }
            nexium_spirv::TextureNumericType::Uint => {
                out.extend_from_slice(&(packed & 0xff).to_le_bytes());
            }
            nexium_spirv::TextureNumericType::Sint => {
                out.extend_from_slice(&((packed & 0xff) as i32).to_le_bytes());
            }
        }
    }
    out
}

fn volume_from_guest(gpu_va: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("NEXIUM_VOLUME_FROM_GUEST")
            .map(|v| {
                v.split(',')
                    .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_default()
    });
    list.contains(&gpu_va)
}

fn trace_volume_rt_skip(
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    layers: u32,
    reason: &'static str,
) {
    if std::env::var_os("NEXIUM_VOLUME_DBG").is_none() {
        return;
    }
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32, &'static str)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().insert((tic.gpu_va, layers, reason)) {
        log::warn!(
            "[volume-rt-skip] va={:#x} reason={} {}x{}x{} pitch={} bl={} bw={} bh={} bd={} tw={}",
            tic.gpu_va,
            reason,
            tic.width,
            tic.height,
            layers,
            pitch_size,
            tic.is_block_linear,
            tic.block_width_log2,
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing
        );
    }
}

fn find_volume_rt_slices(
    rt_cache: &RtCache,
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    layers: u32,
    base_key: Option<RtKey>,
) -> Option<Vec<VolumeRtSlice>> {
    if layers == 0 {
        trace_volume_rt_skip(tic, pitch_size, layers, "no-layers");
        return None;
    }
    if volume_from_guest(tic.gpu_va) {
        trace_volume_rt_skip(tic, pitch_size, layers, "forced-guest");
        return None;
    }
    let Some(offsets) = volume_slice_offsets(tic, pitch_size, layers) else {
        trace_volume_rt_skip(tic, pitch_size, layers, "no-offsets");
        return None;
    };
    if offsets.is_empty() {
        trace_volume_rt_skip(tic, pitch_size, layers, "empty-offsets");
        return None;
    }
    if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static PROBED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let probed = PROBED.get_or_init(|| Mutex::new(HashSet::new()));
        if probed.lock().unwrap().insert(tic.gpu_va) {
            log::warn!(
                "[volume-rt-probe] va={:#x} {}x{}x{} pitch={} offs0={:#x} offslast={:#x} bl={} bw={} bh={} bd={} tw={} base_key={}",
                tic.gpu_va,
                tic.width,
                tic.height,
                layers,
                pitch_size,
                offsets.first().copied().unwrap_or(0),
                offsets.last().copied().unwrap_or(0),
                tic.is_block_linear,
                tic.block_width_log2,
                tic.block_height_log2,
                tic.block_depth_log2,
                tic.tile_width_spacing,
                base_key.map(|key| key.label()).unwrap_or_default()
            );
        }
    }
    let allow_partial = std::env::var_os("NEXIUM_VOLUME_PARTIAL").is_some();
    let mut out = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        let offset = offsets.get(layer as usize).copied()?;
        let va = tic.gpu_va.checked_add(offset)?;
        let cpu_addr = base_key
            .and_then(|key| key.cpu_addr.checked_add(offset))
            .unwrap_or(0);
        let Some(region) = rt_cache
            .find_drawn_color_region_at(tic.width, tic.height, va)
            .or_else(|| {
                base_key.and_then(|key| {
                    rt_cache.find_drawn_color_region_at_cpu(
                        tic.width,
                        tic.height,
                        key.nvmap_id,
                        cpu_addr,
                    )
                })
            })
        else {
            if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
                use std::collections::HashSet;
                use std::sync::{Mutex, OnceLock};
                static MISSING: OnceLock<Mutex<HashSet<(u64, u32)>>> = OnceLock::new();
                let missing = MISSING.get_or_init(|| Mutex::new(HashSet::new()));
                if missing.lock().unwrap().insert((tic.gpu_va, layer)) {
                    log::warn!(
                        "[volume-rt-miss] va={:#x} layer={} slice_va={:#x} cpu={:#x} off={:#x} {}x{}x{} bl={} bw={} bh={} bd={} tw={}",
                        tic.gpu_va,
                        layer,
                        va,
                        cpu_addr,
                        offset,
                        tic.width,
                        tic.height,
                        layers,
                        tic.is_block_linear,
                        tic.block_width_log2,
                        tic.block_height_log2,
                        tic.block_depth_log2,
                        tic.tile_width_spacing
                    );
                }
            }
            if allow_partial {
                continue;
            }
            return None;
        };
        let mut src_x = region.src_x;
        let src_y = region.src_y;
        if std::env::var_os("NEXIUM_VOLUME_SRC_RIGHT").is_some()
            && region.key.width >= tic.width.saturating_mul(2)
            && src_x == 0
            && tic.width <= region.key.width.saturating_sub(src_x)
            && tic.height <= region.key.height.saturating_sub(src_y)
        {
            src_x = tic.width;
        }
        if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
            use std::collections::HashSet;
            use std::sync::{Mutex, OnceLock};
            static SEEN_SLICE: OnceLock<Mutex<HashSet<(u64, u32, RtKey, u32, u32)>>> =
                OnceLock::new();
            let seen = SEEN_SLICE.get_or_init(|| Mutex::new(HashSet::new()));
            if seen
                .lock()
                .unwrap()
                .insert((tic.gpu_va, layer, region.key, src_x, src_y))
            {
                log::warn!(
                    "[volume-rt-slice] va={:#x} layer={} slice_va={:#x} off={:#x} src={} src_xy=({}, {}) copy_xy=({}, {}) fmt={:?} stamp={}",
                    tic.gpu_va,
                    layer,
                    va,
                    offset,
                    region.key.label(),
                    region.src_x,
                    region.src_y,
                    src_x,
                    src_y,
                    region.format,
                    region.stamp
                );
            }
        }
        out.push(VolumeRtSlice {
            layer,
            key: region.key,
            image: region.image,
            layout: region.layout,
            format: region.format,
            stamp: region.stamp,
            src_x,
            src_y,
        });
    }
    if out.is_empty() {
        return None;
    }
    if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if seen.lock().unwrap().insert(tic.gpu_va) {
            let first = out.first().map(|s| s.key.label()).unwrap_or_default();
            let last = out.last().map(|s| s.key.label()).unwrap_or_default();
            let mut formats = out
                .iter()
                .map(|s| format!("{:?}", s.format))
                .collect::<Vec<_>>();
            formats.sort();
            formats.dedup();
            log::warn!(
                "[volume-rt] va={:#x} {}x{}x{} found={}/{} last_off={:#x} bl={} bw={} bh={} bd={} tw={} first={} last={} formats={}",
                tic.gpu_va,
                tic.width,
                tic.height,
                layers,
                out.len(),
                layers,
                offsets.last().copied().unwrap_or(0),
                tic.is_block_linear,
                tic.block_width_log2,
                tic.block_height_log2,
                tic.block_depth_log2,
                tic.tile_width_spacing,
                first,
                last,
                formats.join("|")
            );
        }
    }
    Some(out)
}

fn volume_slice_offsets(
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    layers: u32,
) -> Option<Vec<u64>> {
    if layers == 0 {
        return None;
    }
    if tic.is_block_linear && tic_is_volume(tic) {
        return Some(block_linear_volume_slice_offsets(tic, layers));
    }
    let slice_size = tic_layer_read_size(tic, pitch_size) as u64;
    if slice_size == 0 {
        return None;
    }
    Some(
        (0..layers)
            .map(|layer| slice_size.saturating_mul(layer as u64))
            .collect(),
    )
}

fn block_linear_volume_slice_offsets(tic: &crate::texture::TicEntry, layers: u32) -> Vec<u64> {
    let (storage_width, storage_height, bpp) = tic.format.storage_extent(tic.width, tic.height);
    let bpp_log2 = bytes_per_block_log2(bpp);
    let width_bytes = (storage_width as u64) << bpp_log2;
    let height_blocks = storage_height as u64;
    let depth = layers.max(1) as u64;
    let gobs_width = ceil_div_pow2(width_bytes, 6);
    let gobs_height = ceil_div_pow2(height_blocks, 3);
    let block_width = tic.block_width_log2;
    let block_height = tic.block_height_log2;
    let block_depth = tic.block_depth_log2;
    let gob_width = 6u32
        .saturating_sub(bpp_log2)
        .saturating_add(tic.tile_width_spacing);
    let gob_height = 3u32.saturating_add(block_height);
    let small = width_bytes <= (1u64 << gob_width)
        || height_blocks <= (1u64 << gob_height)
        || depth < (1u64 << block_depth);
    let aligned_gobs_width = if small {
        gobs_width
    } else {
        align_up_pow2(gobs_width, tic.tile_width_spacing)
    };
    let tiles_width = ceil_div_pow2(aligned_gobs_width, block_width);
    let tiles_height = ceil_div_pow2(gobs_height, block_height);
    let gob_size_shift = 9u32.saturating_add(block_height);
    let slice_size = (tiles_width.saturating_mul(tiles_height)) << gob_size_shift;
    let z_mask = (1u64 << block_depth).saturating_sub(1);
    (0..layers as u64)
        .map(|z| {
            ((z & !z_mask).saturating_mul(slice_size))
                .saturating_add((z & z_mask) << gob_size_shift)
        })
        .collect()
}

fn block_linear_volume_byte_size(tic: &crate::texture::TicEntry, layers: u32) -> usize {
    let (storage_width, storage_height, bpp) = tic.format.storage_extent(tic.width, tic.height);
    crate::texture::block_linear_byte_size_3d(
        storage_width,
        storage_height,
        layers.max(1),
        bpp,
        tic.block_height_log2,
        tic.block_depth_log2,
        tic.tile_width_spacing,
    )
}

fn bytes_per_block_log2(bpp: usize) -> u32 {
    bpp.next_power_of_two().trailing_zeros()
}

fn ceil_div_pow2(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        value
    } else {
        (value + (1u64 << shift) - 1) >> shift
    }
}

fn align_up_pow2(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        value
    } else {
        let mask = (1u64 << shift) - 1;
        (value + mask) & !mask
    }
}

fn volume_rt_slice_hash(mut hash: u64, slices: &[VolumeRtSlice]) -> u64 {
    for slice in slices {
        hash ^= slice.layer as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= slice.key.gpu_va;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= ((slice.key.width as u64) << 32) | slice.key.height as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= slice.stamp;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= slice.format.as_raw() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        hash ^= ((slice.src_x as u64) << 32) | slice.src_y as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn identity_volume_rgba8(width: u32, height: u32, layers: u32) -> Vec<u8> {
    let width = width.max(1);
    let height = height.max(1);
    let layers = layers.max(1);
    let mut out = vec![0; width as usize * height as usize * layers as usize * 4];
    for z in 0..layers {
        for y in 0..height {
            for x in 0..width {
                let off = (((z as usize * height as usize + y as usize) * width as usize)
                    + x as usize)
                    * 4;
                out[off] = scale_to_u8(x, width);
                out[off + 1] = scale_to_u8(y, height);
                out[off + 2] = scale_to_u8(z, layers);
                out[off + 3] = 255;
            }
        }
    }
    out
}

fn scale_to_u8(v: u32, max: u32) -> u8 {
    if max <= 1 {
        0
    } else {
        ((v as u64 * 255 + (max as u64 - 1) / 2) / (max as u64 - 1)) as u8
    }
}

fn graphics_draw_call_error(
    call_index: usize,
    call: &crate::draw::Maxwell3dDrawCall,
    phase: &str,
    error: impl std::fmt::Display,
) -> String {
    format!(
        "draw call {call_index} phase={phase} vs_va={:#x} vs_hash={:016x} fs_va={:#x} fs_hash={:016x}: {error}",
        call.vs_gpu_va, call.vs_hash, call.fs_gpu_va, call.fs_hash
    )
}

fn collect_tex_pendings<F>(
    call: &crate::draw::Maxwell3dDrawCall,
    read_guest: &F,
) -> Result<Vec<Option<PendingTexture>>, String>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    call.fs_tex_ids
        .iter()
        .take(max_texture_descriptors())
        .enumerate()
        .map(|(slot, tex_id)| -> Result<Option<PendingTexture>, String> {
            if *tex_id == u32::MAX || *tex_id > call.tic_pool_limit || call.tic_pool_gpu_va == 0 {
                return Ok(None);
            }
            let tic_addr = call.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
            let Some(tic_raw) = read_guest(tic_addr, 32) else {
                return Ok(None);
            };
            let Some(tic) = crate::texture::TicEntry::parse(&tic_raw) else {
                return Ok(None);
            };
            if tic.is_srgb && std::env::var_os("NEXIUM_TIC_SRGB_LOG").is_some() {
                use std::sync::atomic::{AtomicU64, Ordering};
                static N: AtomicU64 = AtomicU64::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 400 {
                    log::warn!(
                        "[tic-srgb] fs={:#x} id={} {}x{} fmt={:?} bl={} va={:#x}",
                        call.fs_gpu_va,
                        tex_id,
                        tic.width,
                        tic.height,
                        tic.format,
                        tic.is_block_linear,
                        tic.gpu_va
                    );
                }
            }
            let pitch_size = tic.format.linear_size(tic.width, tic.height);
            let volume = tic_is_volume(&tic);
            let cube = tic_is_cube(&tic);
            let cube_array = tic_is_cube_array(&tic);
            let arrayed = descriptor_slot_uses_arrayed_2d(
                slot,
                call.vs_tex_base,
                call.vs_tex_count,
                call.fs_sampler_arrayed,
                call.vs_sampler_arrayed,
            ) && !volume
                && !cube
                && !cube_array;
            let layers = if arrayed || volume || cube || cube_array {
                tic_layer_count(&tic)
            } else {
                1
            };
            let mip_levels = if tic.is_block_linear && !volume {
                tic.mip_levels()
            } else {
                1
            };
            let base_mip = if mip_levels > 1 {
                tic.view_base_mip()
            } else {
                0
            };
            let view_mips = if mip_levels > 1 {
                tic.view_mip_levels()
            } else {
                1
            };
            let read_size = tic_read_size(&tic, pitch_size, layers);
            let key = TexCacheKey {
                gpu_va: tic.gpu_va,
                width: tic.width,
                height: tic.height,
                layers,
                base_layer: if arrayed || cube || cube_array {
                    tic_view_base_layer(&tic)
                } else {
                    0
                },
                view_layers: if arrayed || cube || cube_array {
                    tic_view_layer_count(&tic)
                } else {
                    1
                },
                mip_levels,
                base_mip,
                view_mips,
                arrayed,
                cube,
                cube_array,
                volume,
                format: tic.format,
                component_types: tic.component_types,
                swizzle: tic.swizzle,
                is_srgb: tic.is_srgb,
                is_block_linear: tic.is_block_linear,
                block_width_log2: tic.block_width_log2,
                block_height_log2: tic.block_height_log2,
                block_depth_log2: tic.block_depth_log2,
                tile_width_spacing: tic.tile_width_spacing,
                numeric_type: texture_numeric_cache_key(texture_numeric_type_for_slot(
                    &call.texture_numeric_manifest,
                    slot,
                )),
            };
            let image_kind = texture_image_kind_for_slot(&call.texture_numeric_manifest, slot);
            let key = route_texture_key_to_shader_image_kind(key, &tic, image_kind).map_err(
                |error| {
                    format!(
                        "texture slot {slot} shader image family {image_kind:?} is incompatible with TIC {} at {tic_addr:#x}: {error}",
                        tic.texture_type
                    )
                },
            )?;
            Ok(Some((key, tic, pitch_size, read_size)))
        })
        .collect()
}

fn collect_tsc_entries<F>(
    call: &crate::draw::Maxwell3dDrawCall,
    read_guest: &F,
) -> Vec<Option<crate::texture::TscEntry>>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    call.fs_sampler_ids
        .iter()
        .take(max_texture_descriptors())
        .map(|tsc_id| {
            if *tsc_id > call.tsc_pool_limit || call.tsc_pool_gpu_va == 0 {
                return None;
            }
            let tsc_addr = call.tsc_pool_gpu_va.wrapping_add((*tsc_id as u64) * 32);
            read_guest(tsc_addr, 32).and_then(|r| crate::texture::TscEntry::parse(&r))
        })
        .collect()
}

fn sampled_rt_key_for_slot(call: &crate::draw::Maxwell3dDrawCall, slot: usize) -> Option<RtKey> {
    sampled_rt_key_from_lists(
        &call.sampled_rt_slots,
        &call.sampled_rt_keys,
        call.sampled_rt_key,
        slot,
    )
}

fn sampled_rt_key_from_lists(
    slots: &[Option<RtKey>],
    compressed: &[RtKey],
    legacy_first: Option<RtKey>,
    slot: usize,
) -> Option<RtKey> {
    if !slots.is_empty() {
        return slots.get(slot).copied().flatten();
    }
    compressed
        .get(slot)
        .copied()
        .or_else(|| if slot == 0 { legacy_first } else { None })
}

fn call_samples_rt(call: &crate::draw::Maxwell3dDrawCall, rt_key: RtKey) -> bool {
    call.sampled_rt_key == Some(rt_key)
        || call.sampled_rt_keys.contains(&rt_key)
        || call
            .sampled_rt_slots
            .iter()
            .any(|slot| *slot == Some(rt_key))
        || call
            .sampled_rt_copy_sources
            .iter()
            .any(|slot| *slot == Some(rt_key))
}

fn rt_alias_for_slot(
    rt_cache: &RtCache,
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    rt_key: RtKey,
    allow_self: bool,
    tic: Option<&crate::texture::TicEntry>,
) -> Option<RtAlias> {
    let sk = sampled_rt_key_for_slot(call, slot)?;
    let direct_volume = sk.is_3d && tic.is_some_and(|tic| tic_is_volume(&tic));
    if tic.is_some_and(tic_requires_dedicated_sampled_view) && !direct_volume {
        return None;
    }
    let mut used_copy_color = false;
    let found_color = (if direct_volume {
        rt_cache
            .color_exact_with_format(sk)
            .map(|(key, image, view, layout, format, _)| (key, image, view, layout, format))
    } else {
        drawn_color_alias_for_key(rt_cache, sk).or_else(|| rt_cache.find_color_with_format(sk))
    })
    .or_else(|| {
        let source = call.sampled_rt_copy_sources.get(slot).copied().flatten()?;
        let found = drawn_color_alias_for_key(rt_cache, source)
            .or_else(|| rt_cache.find_color_with_format(source));
        used_copy_color = found.is_some();
        found
    })
    .filter(|(candidate, _, _, _, _)| {
        if !used_copy_color || call.fs_gpu_va != 0x4000b3030 || slot != 0 {
            return true;
        }
        let duplicates_exact_input = call
            .sampled_rt_slots
            .iter()
            .enumerate()
            .any(|(other_slot, other)| other_slot != slot && *other == Some(*candidate));
        if duplicates_exact_input {
            trace_fuzzy_duplicate_reject(call, slot, sk, *candidate);
            false
        } else {
            true
        }
    });
    let color_found_key = found_color.as_ref().map(|(k, _, _, _, _)| *k);
    let color_alias = found_color
        .filter(|(k, _, _, _, _)| allow_self || *k != rt_key)
        .map(|(key, image, view, layout, format)| RtAlias {
            key,
            image,
            view,
            layout,
            format,
            aspects: vk::ImageAspectFlags::COLOR,
            depth: false,
        });
    let depth_source = rt_cache.find_depth(sk);
    let depth_alias = depth_source.and_then(|(key, image, view, layout, format, aspects)| {
        let active = call
            .depth_key
            .is_some_and(|active| key == active || same_physical_backing(key, active));
        if active {
            return None;
        }
        Some(RtAlias {
            key,
            image,
            view,
            layout,
            format,
            aspects,
            depth: true,
        })
    });
    let filtered = if tic.is_some_and(|tic| tic_format_prefers_depth_alias(tic.format)) {
        depth_alias
    } else {
        color_alias.or(depth_alias)
    };
    let found_key = filtered.map(|alias| alias.key).or(color_found_key);
    trace_rt_alias(
        slot,
        call,
        rt_key,
        sk,
        found_key,
        filtered.is_some(),
        used_copy_color,
    );
    filtered
}

fn trace_fuzzy_duplicate_reject(
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    requested: RtKey,
    candidate: RtKey,
) {
    if std::env::var_os("NEXIUM_RT_ALIAS_DBG").is_none()
        && std::env::var_os("NEXIUM_RT_ALIAS_UNIQUE").is_none()
    {
        return;
    }
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, usize, RtKey, RtKey)>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap();
    if seen.insert((call.fs_gpu_va, slot, requested, candidate)) {
        log::warn!(
            "[rt-fuzzy-reject] fs={:#x} slot={} requested={} candidate={} reason=duplicates-exact-input",
            call.fs_gpu_va,
            slot,
            requested.label(),
            candidate.label(),
        );
    }
}

fn tic_format_prefers_depth_alias(format: crate::texture::TicFormat) -> bool {
    matches!(
        format,
        crate::texture::TicFormat::G24R8
            | crate::texture::TicFormat::Z24S8
            | crate::texture::TicFormat::X8Z24
            | crate::texture::TicFormat::S8Z24
            | crate::texture::TicFormat::Z32
    )
}

fn drawn_color_alias_for_key(
    rt_cache: &RtCache,
    key: RtKey,
) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
    if key.is_3d {
        return rt_cache.find_color_with_format(key);
    }
    let found = if key.gpu_va != 0 {
        rt_cache.find_drawn_color_at(key.width, key.height, key.gpu_va)
    } else if key.cpu_addr != 0 {
        rt_cache.find_drawn_color_at_cpu(key.width, key.height, key.nvmap_id, key.cpu_addr)
    } else {
        None
    }?;
    rt_cache.find_color_with_format(found.0)
}

fn drawn_color_content_alias_for_key(
    rt_cache: &RtCache,
    key: RtKey,
) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
    if key.is_3d {
        return rt_cache.find_color_with_format(key);
    }
    let found = if key.gpu_va != 0 {
        rt_cache.find_content_bearing_color_at(key.width, key.height, key.gpu_va)
    } else if key.cpu_addr != 0 {
        rt_cache.find_drawn_color_at_cpu(key.width, key.height, key.nvmap_id, key.cpu_addr)
    } else {
        None
    }?;
    rt_cache.find_color_with_format(found.0)
}

fn snapshot_feedback_alias(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    key: RtKey,
) -> Result<(vk::Image, vk::ImageView, vk::Format), String> {
    let (_, live_image, _, live_layout, _, _) = rt_cache
        .color_exact_with_format(key)
        .ok_or_else(|| format!("feedback snapshot: no live entry {}", key.label()))?;
    let live_prev = rt_cache.color_layout(key).unwrap_or(live_layout);
    let (snap_image, snap_view, snap_format) =
        rt_cache.get_or_create_feedback_snapshot(device, key)?;
    transition_image(
        device,
        cmd,
        live_image,
        live_prev,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    transition_image(
        device,
        cmd,
        snap_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    let region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image(
            cmd,
            live_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            snap_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    transition_image(
        device,
        cmd,
        live_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        live_prev,
    );
    transition_image(
        device,
        cmd,
        snap_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    Ok((snap_image, snap_view, snap_format))
}

fn sampled_color_alias_needs_sync(rt_cache: &RtCache, key: RtKey) -> bool {
    if std::env::var_os("NEXIUM_NO_RT_ALIAS_SYNC").is_some() {
        return false;
    }
    color_alias_sync_pair(rt_cache, key).is_some()
}

fn sampled_color_region_needs_sync(rt_cache: &RtCache, key: RtKey) -> bool {
    if std::env::var_os("NEXIUM_NO_RT_ALIAS_SYNC").is_some() {
        return false;
    }
    color_region_sync_pair(rt_cache, key).is_some()
}

fn sampled_color_needs_sync(rt_cache: &RtCache, key: RtKey) -> bool {
    sampled_color_alias_needs_sync(rt_cache, key) || sampled_color_region_needs_sync(rt_cache, key)
}

fn color_alias_sync_pair(rt_cache: &RtCache, key: RtKey) -> Option<ColorAliasSync> {
    let (dst_key, dst_image, _, dst_layout, dst_format, dst_stamp) =
        rt_cache.color_exact_with_format(key)?;
    let allow_resize = std::env::var_os("NEXIUM_RT_ALIAS_SCALE_BLIT").is_some()
        || std::env::var_os("NEXIUM_RT_ALIAS_RESIZE_SYNC").is_some();
    let mut best = None;
    for (src_key, src_image, src_layout, src_format, src_stamp) in
        rt_cache.drawn_color_aliases(dst_key)
    {
        if src_stamp <= dst_stamp || src_layout == vk::ImageLayout::UNDEFINED {
            continue;
        }
        if !allow_resize && (src_key.width != dst_key.width || src_key.height != dst_key.height) {
            continue;
        }
        if !rt_alias_formats_syncable(src_format, dst_format) {
            continue;
        }
        let Some((src_width, height, bytes)) =
            rt_alias_copy_geometry(src_key, src_format, dst_key, dst_format)
        else {
            continue;
        };
        let sync = ColorAliasSync {
            src_key,
            src_image,
            src_layout,
            src_format,
            src_stamp,
            dst_key,
            dst_image,
            dst_layout,
            dst_format,
            dst_stamp,
            src_width,
            height,
            bytes,
        };
        if best
            .as_ref()
            .map_or(true, |old: &ColorAliasSync| src_stamp > old.src_stamp)
        {
            best = Some(sync);
        }
    }
    best
}

fn color_region_sync_pair(rt_cache: &RtCache, key: RtKey) -> Option<ColorRegionSync> {
    if std::env::var_os("NEXIUM_NO_RT_REGION_SYNC").is_some() {
        return None;
    }
    let allow_offset = std::env::var_os("NEXIUM_RT_REGION_OFFSET_SYNC").is_some();
    let region = if key.gpu_va != 0 {
        rt_cache
            .find_drawn_color_region_at(key.width, key.height, key.gpu_va)
            .filter(|region| {
                region.key != key && (allow_offset || (region.src_x == 0 && region.src_y == 0))
            })
    } else {
        None
    }?;
    if region.layout == vk::ImageLayout::UNDEFINED {
        return None;
    }
    let (dst_format, dst_stamp) = rt_cache
        .color_exact_with_format(key)
        .map(|(_, _, _, _, format, stamp)| (format, stamp))
        .unwrap_or((region.format, 0));
    if region.stamp <= dst_stamp || !rt_alias_formats_syncable(region.format, dst_format) {
        return None;
    }
    Some(ColorRegionSync {
        src_key: region.key,
        src_image: region.image,
        src_layout: region.layout,
        src_format: region.format,
        src_stamp: region.stamp,
        dst_format,
        dst_stamp,
        src_x: region.src_x,
        src_y: region.src_y,
    })
}

fn rt_alias_formats_syncable(src: vk::Format, dst: vk::Format) -> bool {
    src == dst
        || rt_alias_format_family(src)
            .is_some_and(|family| Some(family) == rt_alias_format_family(dst))
}

fn rt_alias_format_family(format: vk::Format) -> Option<u8> {
    match format {
        vk::Format::A8B8G8R8_UNORM_PACK32
        | vk::Format::A8B8G8R8_SNORM_PACK32
        | vk::Format::A8B8G8R8_UINT_PACK32
        | vk::Format::A8B8G8R8_SINT_PACK32
        | vk::Format::A8B8G8R8_SRGB_PACK32 => Some(1),
        vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::A2B10G10R10_UINT_PACK32
        | vk::Format::A2B10G10R10_SINT_PACK32 => Some(2),
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => Some(3),
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => Some(4),
        vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT => Some(5),
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT => Some(6),
        vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT => Some(7),
        vk::Format::R8_UNORM | vk::Format::R8_SNORM | vk::Format::R8_UINT | vk::Format::R8_SINT => {
            Some(8)
        }
        vk::Format::B10G11R11_UFLOAT_PACK32 => Some(9),
        _ => None,
    }
}

fn rt_alias_copy_geometry(
    src_key: RtKey,
    src_format: vk::Format,
    dst_key: RtKey,
    dst_format: vk::Format,
) -> Option<(u32, u32, u64)> {
    let src_bpp = readback_format_bpp(src_format) as u64;
    let dst_bpp = readback_format_bpp(dst_format) as u64;
    if src_bpp == 0 || dst_bpp == 0 || dst_key.width == 0 || dst_key.height == 0 {
        return None;
    }
    let dst_row = (dst_key.width as u64).checked_mul(dst_bpp)?;
    if dst_row % src_bpp != 0 {
        return None;
    }
    let src_width = dst_row / src_bpp;
    if src_width == 0 || src_width > src_key.width as u64 || dst_key.height > src_key.height {
        return None;
    }
    let bytes = dst_row.checked_mul(dst_key.height as u64)?;
    Some((src_width as u32, dst_key.height, bytes))
}

fn sync_sampled_color_alias(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    frame_slot: &mut FrameSlot,
    key: RtKey,
) -> Result<bool, String> {
    if std::env::var_os("NEXIUM_NO_RT_ALIAS_SYNC").is_some() {
        return Ok(false);
    }
    let Some(sync) = color_alias_sync_pair(rt_cache, key) else {
        return Ok(false);
    };
    let use_blit = std::env::var_os("NEXIUM_RT_ALIAS_SCALE_BLIT").is_some()
        && (sync.src_key.width != sync.dst_key.width || sync.src_key.height != sync.dst_key.height);
    let transfer = if use_blit {
        None
    } else {
        Some(create_transfer_buffer_owned(device, mem_props, sync.bytes)?)
    };
    let src_prev = rt_cache
        .color_layout(sync.src_key)
        .unwrap_or(sync.src_layout);
    let dst_prev = rt_cache
        .color_layout(sync.dst_key)
        .unwrap_or(sync.dst_layout);
    transition_image(
        device,
        cmd,
        sync.src_image,
        src_prev,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    transition_image(
        device,
        cmd,
        sync.dst_image,
        dst_prev,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    if use_blit {
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: sync.src_key.width as i32,
                    y: sync.src_key.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: sync.dst_key.width as i32,
                    y: sync.dst_key.height as i32,
                    z: 1,
                },
            ],
        };
        let filter = if std::env::var_os("NEXIUM_RT_ALIAS_SCALE_BLIT_LINEAR").is_some() {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        unsafe {
            device.cmd_blit_image(
                cmd,
                sync.src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                sync.dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                filter,
            );
        }
    } else {
        let transfer = transfer.as_ref().unwrap();
        let src_copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: sync.src_width,
                height: sync.height,
                depth: 1,
            },
        };
        let dst_copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: sync.dst_key.width,
                height: sync.dst_key.height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                sync.src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                transfer.buffer,
                &[src_copy],
            );
            device.cmd_copy_buffer_to_image(
                cmd,
                transfer.buffer,
                sync.dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[dst_copy],
            );
        }
    }
    transition_image(
        device,
        cmd,
        sync.dst_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    if src_prev != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        transition_image(
            device,
            cmd,
            sync.src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_prev,
        );
    }
    rt_cache.set_color_layout(sync.src_key, src_prev);
    rt_cache.set_color_layout(sync.dst_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let stamp = rt_cache.mark_synced_sample(sync.dst_key);
    if std::env::var_os("NEXIUM_RT_ALIAS_SYNC_DBG").is_some() {
        log::warn!(
            "[rt-alias-sync] src={}#{}/{} {:?} dst={}#{}/{} {:?} src_width={} h={} bytes={} blit={}",
            sync.src_key.label(),
            sync.src_stamp,
            rt_cache.color_layout(sync.src_key).is_some() as u8,
            sync.src_format,
            sync.dst_key.label(),
            sync.dst_stamp,
            stamp,
            sync.dst_format,
            sync.src_width,
            sync.height,
            sync.bytes,
            use_blit
        );
    }
    if let Some(transfer) = transfer {
        frame_slot
            .retired_buffers
            .push((transfer.buffer, transfer.memory));
    }
    Ok(true)
}

fn sync_sampled_color_region(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    key: RtKey,
) -> Result<bool, String> {
    if std::env::var_os("NEXIUM_NO_RT_ALIAS_SYNC").is_some() {
        return Ok(false);
    }
    let Some(sync) = color_region_sync_pair(rt_cache, key) else {
        return Ok(false);
    };
    let (dst_image, dst_prev, dst_format) = {
        let dst = rt_cache.get_or_create_with_format(key, device, sync.dst_format)?;
        (dst.image, dst.layout, dst.format)
    };
    if !rt_alias_formats_syncable(sync.src_format, dst_format) {
        return Ok(false);
    }
    let src_prev = rt_cache
        .color_layout(sync.src_key)
        .unwrap_or(sync.src_layout);
    if src_prev == vk::ImageLayout::UNDEFINED {
        return Ok(false);
    }
    transition_image(
        device,
        cmd,
        sync.src_image,
        src_prev,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    );
    transition_image(
        device,
        cmd,
        dst_image,
        dst_prev,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    let region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D {
            x: sync.src_x as i32,
            y: sync.src_y as i32,
            z: 0,
        },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: key.width,
            height: key.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image(
            cmd,
            sync.src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    transition_image(
        device,
        cmd,
        dst_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    if src_prev != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        transition_image(
            device,
            cmd,
            sync.src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_prev,
        );
    }
    rt_cache.set_color_layout(sync.src_key, src_prev);
    rt_cache.set_color_layout(key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let stamp = rt_cache.mark_synced_sample(key);
    if std::env::var_os("NEXIUM_RT_ALIAS_SYNC_DBG").is_some() {
        log::warn!(
            "[rt-region-sync] src={}#{}/{} {:?} dst={}#{}/{} {:?} xy={},{}",
            sync.src_key.label(),
            sync.src_stamp,
            rt_cache.color_layout(sync.src_key).is_some() as u8,
            sync.src_format,
            key.label(),
            sync.dst_stamp,
            stamp,
            dst_format,
            sync.src_x,
            sync.src_y
        );
    }
    Ok(true)
}

fn sampled_active_depth_key(
    rt_cache: &RtCache,
    call: &crate::draw::Maxwell3dDrawCall,
    sk: RtKey,
) -> Option<RtKey> {
    let (key, _, _, _, _, _) = rt_cache.find_depth(sk)?;
    call.depth_key
        .is_some_and(|active| key == active || same_physical_backing(key, active))
        .then_some(key)
}

fn depth_self_shadow_key(key: RtKey) -> RtKey {
    RtKey {
        nvmap_id: key.nvmap_id ^ 0x4000_0000,
        width: key.width,
        height: key.height,
        depth: 1,
        is_3d: false,
        gpu_va: 0,
        cpu_addr: 0,
    }
}

fn depth_self_shadow_needs_sync(
    rt_cache: &RtCache,
    call: &crate::draw::Maxwell3dDrawCall,
    sk: RtKey,
) -> bool {
    if std::env::var_os("NEXIUM_NO_DEPTH_SELF_SAMPLE").is_some() {
        return false;
    }
    let Some(src_key) = sampled_active_depth_key(rt_cache, call, sk) else {
        return false;
    };
    let Some((_, _, _, src_layout, _, _)) = rt_cache.find_depth(src_key) else {
        return false;
    };
    if src_layout == vk::ImageLayout::UNDEFINED {
        return false;
    }
    let shadow_key = depth_self_shadow_key(src_key);
    !rt_cache.depth_shadow_is_current(src_key, shadow_key)
        || rt_cache.depth_layout(shadow_key) != Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

fn sync_sampled_depth_self(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    call: &crate::draw::Maxwell3dDrawCall,
    sk: RtKey,
) -> Result<Option<RtAlias>, String> {
    if std::env::var_os("NEXIUM_NO_DEPTH_SELF_SAMPLE").is_some() {
        return Ok(None);
    }
    let Some(src_key) = sampled_active_depth_key(rt_cache, call, sk) else {
        return Ok(None);
    };
    let Some((_, src_image, _, src_layout, src_format, src_aspects)) = rt_cache.find_depth(src_key)
    else {
        return Ok(None);
    };
    if src_layout == vk::ImageLayout::UNDEFINED {
        return Ok(None);
    }
    let shadow_key = depth_self_shadow_key(src_key);
    let (shadow_image, shadow_view, shadow_layout) = {
        let (img, _) = rt_cache.get_or_create_depth(shadow_key, device, src_format, src_aspects)?;
        (img.image, img.view, img.layout)
    };
    if rt_cache.depth_shadow_is_current(src_key, shadow_key)
        && shadow_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    {
        return Ok(Some(RtAlias {
            key: shadow_key,
            image: shadow_image,
            view: shadow_view,
            layout: shadow_layout,
            format: src_format,
            aspects: src_aspects,
            depth: true,
        }));
    }
    transition_image_aspect(
        device,
        cmd,
        src_image,
        src_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_aspects,
    );
    transition_image_aspect(
        device,
        cmd,
        shadow_image,
        shadow_layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_aspects,
    );
    let subresource = vk::ImageSubresourceLayers {
        aspect_mask: src_aspects,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let region = vk::ImageCopy {
        src_subresource: subresource,
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: subresource,
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: src_key.width,
            height: src_key.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            shadow_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    transition_image_aspect(
        device,
        cmd,
        shadow_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_aspects,
    );
    transition_image_aspect(
        device,
        cmd,
        src_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_layout,
        src_aspects,
    );
    rt_cache.set_depth_layout(shadow_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    rt_cache.mark_depth_shadow_synced(src_key, shadow_key);
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || n % 512 == 0 {
            log::warn!(
                "[depth-self-sync] #{} src={} fmt={:?} fs={:#x} -> shadow={}",
                n,
                src_key.label(),
                src_format,
                call.fs_gpu_va,
                shadow_key.label()
            );
        }
    }
    Ok(Some(RtAlias {
        key: shadow_key,
        image: shadow_image,
        view: shadow_view,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        format: src_format,
        aspects: src_aspects,
        depth: true,
    }))
}

fn tic_reads_depth_as_color(format: crate::texture::TicFormat) -> bool {
    matches!(
        format,
        crate::texture::TicFormat::A8B8G8R8 | crate::texture::TicFormat::R8G8B8A8
    )
}

fn sync_sampled_depth_as_color(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    frame_slot: &mut FrameSlot,
    sk: RtKey,
) -> Result<Option<RtAlias>, String> {
    if std::env::var_os("NEXIUM_NO_DEPTH_AS_COLOR").is_some() {
        return Ok(None);
    }
    let Some((src_key, src_image, _, src_layout, src_format, src_aspects)) =
        rt_cache.find_depth(sk)
    else {
        return Ok(None);
    };
    if src_layout == vk::ImageLayout::UNDEFINED {
        return Ok(None);
    }
    if src_format != vk::Format::D24_UNORM_S8_UINT && src_format != vk::Format::X8_D24_UNORM_PACK32
    {
        return Ok(None);
    }
    let shadow_key = RtKey {
        nvmap_id: src_key.nvmap_id ^ 0x2000_0000,
        width: src_key.width,
        height: src_key.height,
        depth: 1,
        is_3d: false,
        gpu_va: 0,
        cpu_addr: 0,
    };
    let (shadow_image, shadow_view, shadow_prev) = {
        let img =
            rt_cache.get_or_create_with_format(shadow_key, device, vk::Format::R8G8B8A8_UNORM)?;
        (img.image, img.view, img.layout)
    };
    let bytes = src_key.width as u64 * src_key.height as u64 * 4;
    let transfer = create_transfer_buffer_owned(device, mem_props, bytes)?;
    transition_image_aspect(
        device,
        cmd,
        src_image,
        src_layout,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_aspects,
    );
    transition_image(
        device,
        cmd,
        shadow_image,
        shadow_prev,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    let extent = vk::Extent3D {
        width: src_key.width,
        height: src_key.height,
        depth: 1,
    };
    let depth_copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: extent,
    };
    let color_copy = vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            ..depth_copy.image_subresource
        },
        ..depth_copy
    };
    let buffer_barrier = vk::BufferMemoryBarrier {
        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
        p_next: std::ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::TRANSFER_READ,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer: transfer.buffer,
        offset: 0,
        size: vk::WHOLE_SIZE,
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            transfer.buffer,
            &[depth_copy],
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier],
            &[],
        );
        device.cmd_copy_buffer_to_image(
            cmd,
            transfer.buffer,
            shadow_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[color_copy],
        );
    }
    transition_image(
        device,
        cmd,
        shadow_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
    transition_image_aspect(
        device,
        cmd,
        src_image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_layout,
        src_aspects,
    );
    rt_cache.set_color_layout(shadow_key, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    frame_slot
        .retired_buffers
        .push((transfer.buffer, transfer.memory));
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || n % 512 == 0 {
            log::warn!(
                "[depth-as-color] #{} src={} {:?} -> shadow={}",
                n,
                src_key.label(),
                src_format,
                shadow_key.label()
            );
        }
    }
    Ok(Some(RtAlias {
        key: shadow_key,
        image: shadow_image,
        view: shadow_view,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        format: vk::Format::R8G8B8A8_UNORM,
        aspects: vk::ImageAspectFlags::COLOR,
        depth: false,
    }))
}

fn trace_rt_alias(
    slot: usize,
    call: &crate::draw::Maxwell3dDrawCall,
    dst: RtKey,
    src: RtKey,
    found: Option<RtKey>,
    used: bool,
    fuzzy: bool,
) {
    let unique = std::env::var_os("NEXIUM_RT_ALIAS_UNIQUE").is_some();
    if std::env::var_os("NEXIUM_RT_ALIAS_DBG").is_none() && !unique {
        return;
    }
    if let Ok(list) = std::env::var("NEXIUM_RT_ALIAS_FS") {
        let matched = list
            .split(',')
            .filter_map(|part| parse_u64_value(part.trim()))
            .any(|addr| addr == call.fs_gpu_va);
        if !matched {
            return;
        }
    }
    if !unique && (src.width < 512 || src.height < 256) {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    if unique {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<
            Mutex<HashSet<(u64, u64, usize, RtKey, RtKey, Option<RtKey>, bool, bool)>>,
        > = OnceLock::new();
        let key = (
            call.vs_gpu_va,
            call.fs_gpu_va,
            slot,
            dst,
            src,
            found,
            used,
            fuzzy,
        );
        let mut seen = SEEN
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap();
        if !seen.insert(key) {
            return;
        }
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    let limit = std::env::var("NEXIUM_RT_ALIAS_LIMIT")
        .ok()
        .and_then(|v| parse_u64_value(&v))
        .unwrap_or(300);
    if !unique && n >= limit {
        return;
    }
    let found = found.map(|k| k.label()).unwrap_or_else(|| "-".to_string());
    log::warn!(
        "[rt-alias] #{} slot={} vs={:#x} fs={:#x} tex={:?} dst={} src={} found={} used={} fuzzy={}",
        n,
        slot,
        call.vs_gpu_va,
        call.fs_gpu_va,
        call.fs_tex_ids,
        dst.label(),
        src.label(),
        found,
        used,
        fuzzy
    );
}

fn log_vertex_bindings_skip(call: &crate::draw::Maxwell3dDrawCall, err: &str) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 8 {
        log::warn!(
            "[vtx-bind-skip #{}] err={} vs={:#x} fs={:#x} rt={} v={} inst={} first_inst={}",
            n,
            err,
            call.vs_gpu_va,
            call.fs_gpu_va,
            call.rt_key.label(),
            call.vertex_count,
            call.instance_count,
            call.first_instance
        );
    }
}

fn prepare_vertex_bindings<F>(
    call: &crate::draw::Maxwell3dDrawCall,
    read_guest: &F,
) -> Result<(Vec<PreparedVertexBinding>, u32), String>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    let mut out = Vec::new();
    let mut draw_vertex_count = call.vertex_count;
    for binding in &call.vertex_bindings {
        let Some((base, bytes)) = crate::draw::vertex_binding_read_range(call, binding) else {
            continue;
        };
        let stride = binding.stride as u64;
        let mut data = read_guest(base, bytes).ok_or_else(|| {
            format!(
                "vertex read failed binding={} va={:#x} bytes={:#x} stride={} size={:#x} indexed={} first={:#x}/{} vertices={}",
                binding.binding,
                base,
                bytes,
                binding.stride,
                binding.size,
                call.state.indexed,
                call.first_vertex,
                call.first_vertex as i32,
                call.vertex_count
            )
        })?;
        if call.quad_expand && !data.is_empty() {
            data = crate::draw::expand_quad_vertices(&data, stride as usize);
            if out.is_empty() {
                draw_vertex_count = (data.len() / stride as usize) as u32;
            }
        }
        if !data.is_empty() {
            out.push(PreparedVertexBinding {
                binding: binding.binding,
                stride,
                data,
            });
        }
    }
    Ok((out, draw_vertex_count))
}

fn upload_vertex_bindings(
    device: &ash::Device,
    frame_slots: &mut [FrameSlot; 2],
    other_idx: usize,
    pool: vk::DescriptorPool,
    ubo_ring: &mut UboRing,
    vertex_bindings: &[PreparedVertexBinding],
    allow_wrap: bool,
) -> Result<Vec<(u32, vk::Buffer, u64)>, String> {
    let mut out = Vec::with_capacity(vertex_bindings.len());
    for binding in vertex_bindings {
        if binding.data.is_empty() {
            continue;
        }
        let align = binding.stride.max(16);
        let size = align_up(binding.data.len() as u64, align);
        if !ring_allocation_fits(ubo_ring, size, align) {
            if !allow_wrap {
                return Err(format!(
                    "batched graphics ring preflight underestimated vertex binding {} upload ({size:#x} bytes, alignment {align})",
                    binding.binding
                ));
            }
            ring_wrap_other(device, frame_slots, other_idx, pool, ubo_ring)?;
        }
        let (buf, off, ptr) =
            ring_alloc(ubo_ring, size, align).map_err(|e| format!("ring_alloc(vertex): {}", e))?;
        unsafe {
            std::ptr::copy_nonoverlapping(binding.data.as_ptr(), ptr, binding.data.len());
        }
        out.push((binding.binding, buf, off));
    }
    Ok(out)
}

fn vertex_bindings_size(vertex_bindings: &[PreparedVertexBinding]) -> u64 {
    vertex_bindings
        .iter()
        .map(|b| align_up(b.data.len() as u64, b.stride.max(16)))
        .sum()
}

fn destroy_descriptor_pools(device: &ash::Device, pools: &mut Vec<vk::DescriptorPool>) {
    for pool in pools.drain(..) {
        unsafe {
            device.destroy_descriptor_pool(pool, None);
        }
    }
}

fn ring_wrap_other(
    device: &ash::Device,
    frame_slots: &mut [FrameSlot; 2],
    other_idx: usize,
    pool: vk::DescriptorPool,
    ubo_ring: &mut UboRing,
) -> Result<(), String> {
    let other = &mut frame_slots[other_idx];
    if other.in_flight {
        wait_fence(device, other.fence)?;
        if !other.retired_dsets.is_empty() {
            unsafe {
                let _ = device.free_descriptor_sets(pool, &other.retired_dsets);
            }
            other.retired_dsets.clear();
        }
        destroy_descriptor_pools(device, &mut other.retired_dset_pools);
        for (b, m) in other.retired_buffers.drain(..) {
            unsafe {
                device.destroy_buffer(b, None);
                device.free_memory(m, None);
            }
        }
        for view in other.retired_views.drain(..) {
            unsafe {
                device.destroy_image_view(view, None);
            }
        }
        for t in other.retired_textures.drain(..) {
            unsafe {
                device.destroy_image_view(t.view, None);
                device.destroy_image(t.image, None);
                device.free_memory(t.memory, None);
            }
        }
        for buffer in other.retired_texel_buffers.drain(..) {
            destroy_texel_buffer(device, buffer.resource);
        }
        for t in other.retired_rt_reinterprets.drain(..) {
            unsafe {
                device.destroy_image_view(t.view, None);
                device.destroy_image(t.image, None);
                device.free_memory(t.memory, None);
            }
        }
        reset_command_buffer(device, other.cmd)?;
        other.in_flight = false;
    }
    ubo_ring.head = 0;
    ubo_ring.slot_head[other_idx] = 0;
    Ok(())
}

fn monotonic_nanos() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_nanos() as u64
}

pub fn texel_buffer_format(
    tic: &crate::texture::TicEntry,
    numeric_type: nexium_spirv::TextureNumericType,
) -> Option<(vk::Format, usize)> {
    use crate::texture::{ComponentType, SwizzleSource, TicFormat};

    if !tic.is_buffer() || tic.is_block_linear || tic.height != 1 || tic.depth != 1 || tic.is_srgb {
        return None;
    }
    let rgba = [
        SwizzleSource::R,
        SwizzleSource::G,
        SwizzleSource::B,
        SwizzleSource::A,
    ];
    let rg01 = [
        SwizzleSource::R,
        SwizzleSource::G,
        SwizzleSource::Zero,
        SwizzleSource::One,
    ];
    let r001 = [
        SwizzleSource::R,
        SwizzleSource::Zero,
        SwizzleSource::Zero,
        SwizzleSource::One,
    ];
    match (numeric_type, tic.format) {
        (nexium_spirv::TextureNumericType::Float, TicFormat::R32G32B32A32)
            if tic.component_types == [ComponentType::Float; 4] && tic.swizzle == rgba =>
        {
            Some((vk::Format::R32G32B32A32_SFLOAT, 16))
        }
        (nexium_spirv::TextureNumericType::Float, TicFormat::R32G32)
            if tic.component_types == [ComponentType::Float; 4] && tic.swizzle == rg01 =>
        {
            Some((vk::Format::R32G32_SFLOAT, 8))
        }
        (nexium_spirv::TextureNumericType::Float, TicFormat::R16G16)
            if tic.component_types == [ComponentType::Float; 4] && tic.swizzle == rg01 =>
        {
            Some((vk::Format::R16G16_SFLOAT, 4))
        }
        (nexium_spirv::TextureNumericType::Uint, TicFormat::R16G16)
            if tic.component_types == [ComponentType::Float; 4] && tic.swizzle == rg01 =>
        {
            Some((vk::Format::R16G16_UINT, 4))
        }
        (nexium_spirv::TextureNumericType::Sint, TicFormat::R16G16)
            if tic.component_types == [ComponentType::Float; 4] && tic.swizzle == rg01 =>
        {
            Some((vk::Format::R16G16_SINT, 4))
        }
        (nexium_spirv::TextureNumericType::Float, TicFormat::A8B8G8R8)
            if tic.component_types == [ComponentType::Unorm; 4] && tic.swizzle == rgba =>
        {
            Some((vk::Format::A8B8G8R8_UNORM_PACK32, 4))
        }
        (nexium_spirv::TextureNumericType::Float, TicFormat::A8B8G8R8)
            if tic.component_types == [ComponentType::Snorm; 4] && tic.swizzle == rgba =>
        {
            Some((vk::Format::A8B8G8R8_SNORM_PACK32, 4))
        }
        (nexium_spirv::TextureNumericType::Uint, TicFormat::A8B8G8R8)
            if tic.component_types == [ComponentType::Snorm; 4] && tic.swizzle == rgba =>
        {
            Some((vk::Format::A8B8G8R8_UINT_PACK32, 4))
        }
        (nexium_spirv::TextureNumericType::Sint, TicFormat::A8B8G8R8)
            if tic.component_types == [ComponentType::Snorm; 4] && tic.swizzle == rgba =>
        {
            Some((vk::Format::A8B8G8R8_SINT_PACK32, 4))
        }
        (nexium_spirv::TextureNumericType::Uint, TicFormat::R16)
            if tic.component_types == [ComponentType::Uint; 4] && tic.swizzle == r001 =>
        {
            Some((vk::Format::R16_UINT, 2))
        }
        (nexium_spirv::TextureNumericType::Uint, TicFormat::R32)
            if tic.component_types == [ComponentType::Uint; 4] && tic.swizzle == r001 =>
        {
            Some((vk::Format::R32_UINT, 4))
        }
        (nexium_spirv::TextureNumericType::Sint, TicFormat::R16)
            if tic.component_types == [ComponentType::Sint; 4] && tic.swizzle == r001 =>
        {
            Some((vk::Format::R16_SINT, 2))
        }
        (nexium_spirv::TextureNumericType::Sint, TicFormat::R32)
            if tic.component_types == [ComponentType::Sint; 4] && tic.swizzle == r001 =>
        {
            Some((vk::Format::R32_SINT, 4))
        }
        _ => None,
    }
}

fn destroy_texel_buffer(device: &ash::Device, resource: TexelBufferResource) {
    unsafe {
        device.destroy_buffer_view(resource.view, None);
        device.destroy_buffer(resource.buffer, None);
        device.free_memory(resource.memory, None);
    }
}

fn create_texel_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    bytes: &[u8],
    format: vk::Format,
) -> Result<TexelBufferResource, String> {
    let host = create_host_buffer(
        device,
        mem_props,
        bytes,
        vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
    )?;
    let view_info = vk::BufferViewCreateInfo {
        s_type: vk::StructureType::BUFFER_VIEW_CREATE_INFO,
        buffer: host.buffer,
        format,
        offset: 0,
        range: bytes.len() as u64,
        p_next: std::ptr::null(),
        flags: vk::BufferViewCreateFlags::empty(),
        _marker: std::marker::PhantomData,
    };
    let view = match unsafe { device.create_buffer_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            unsafe {
                device.destroy_buffer(host.buffer, None);
                device.free_memory(host.memory, None);
            }
            return Err(format!("create_buffer_view(texel): {:?}", error));
        }
    };
    Ok(TexelBufferResource {
        buffer: host.buffer,
        view,
        memory: host.memory,
    })
}

fn ensure_dummy_texel_buffer(
    dummy: &mut [Option<TexelBufferResource>; 3],
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    numeric_type: nexium_spirv::TextureNumericType,
) -> Result<vk::BufferView, String> {
    let (index, format) = match numeric_type {
        nexium_spirv::TextureNumericType::Float => (0, vk::Format::R32_SFLOAT),
        nexium_spirv::TextureNumericType::Uint => (1, vk::Format::R32_UINT),
        nexium_spirv::TextureNumericType::Sint => (2, vk::Format::R32_SINT),
    };
    if dummy[index].is_none() {
        dummy[index] = Some(create_texel_buffer(
            device,
            mem_props,
            &0u32.to_le_bytes(),
            format,
        )?);
    }
    Ok(dummy[index].as_ref().unwrap().view)
}

fn texel_buffer_view_for_tic<F>(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    max_texel_buffer_elements: u32,
    cache: &mut HashMap<TexelBufferCacheKey, CachedTexelBuffer>,
    retired: &mut Vec<CachedTexelBuffer>,
    tic: &crate::texture::TicEntry,
    numeric_type: nexium_spirv::TextureNumericType,
    read_size: usize,
    generation: u64,
    read_guest: &F,
) -> Result<Option<vk::BufferView>, String>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    let Some((format, element_size)) = texel_buffer_format(tic, numeric_type) else {
        return Err(format!(
            "unsupported descriptor fmt={:?} ctypes={:?} swz={:?} bl={} {}x{}x{} srgb={}",
            tic.format,
            tic.component_types,
            tic.swizzle,
            tic.is_block_linear,
            tic.width,
            tic.height,
            tic.depth,
            tic.is_srgb,
        ));
    };
    if tic.width > max_texel_buffer_elements {
        return Err(format!(
            "{} elements exceed Vulkan maxTexelBufferElements {}",
            tic.width, max_texel_buffer_elements
        ));
    }
    let expected_size = (tic.width as usize)
        .checked_mul(element_size)
        .ok_or_else(|| "texel buffer size overflow".to_string())?;
    if read_size != expected_size {
        return Err(format!(
            "descriptor span {} does not match {} elements x {} bytes",
            read_size, tic.width, element_size
        ));
    }

    let key = TexelBufferCacheKey {
        gpu_va: tic.gpu_va,
        elements: tic.width,
        format: tic.format,
        view_format: format.as_raw(),
    };
    let force_refresh = force_refresh_texture(tic.gpu_va);
    if !force_refresh {
        if let Some(cached) = cache.get(&key) {
            if cached.gen == generation {
                if tex_gen_gating_enabled() || cached.verified.elapsed() < TEX_VERIFY_PERIOD {
                    return Ok(Some(cached.resource.view));
                }
                let cached_hash = cached.hash;
                if hash_sampled_guest(read_guest, tic.gpu_va, read_size) == Some(cached_hash) {
                    let cached = cache.get_mut(&key).unwrap();
                    cached.verified = std::time::Instant::now();
                    return Ok(Some(cached.resource.view));
                }
            }
        }
    }

    let bytes = read_guest(tic.gpu_va, read_size)
        .ok_or_else(|| format!("guest read failed for {} bytes", read_size))?;
    if bytes.len() != read_size {
        return Err(format!(
            "guest read returned {} bytes, expected {}",
            bytes.len(),
            read_size
        ));
    }
    let hash = hash_sampled(&bytes);
    if !force_refresh && cache.get(&key).is_some_and(|cached| cached.hash == hash) {
        let cached = cache.get_mut(&key).unwrap();
        cached.gen = generation;
        cached.verified = std::time::Instant::now();
        return Ok(Some(cached.resource.view));
    }

    let resource = create_texel_buffer(device, mem_props, &bytes, format)?;
    let view = resource.view;
    let cached = CachedTexelBuffer {
        resource,
        hash,
        gen: generation,
        verified: std::time::Instant::now(),
    };
    if let Some(old) = cache.insert(key, cached) {
        retired.push(old);
    }
    log::debug!(
        "TBUF gpu_va={:#x} elems={} fmt={:?} bytes={} (cache miss -> upload)",
        tic.gpu_va,
        tic.width,
        format,
        read_size,
    );
    Ok(Some(view))
}

fn create_host_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<HostBuffer, String> {
    let size = data.len().max(16) as u64;
    let info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let buffer = unsafe {
        device
            .create_buffer(&info, None)
            .map_err(|e| format!("create_buffer: {:?}", e))?
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| "no HOST_VISIBLE memory type".to_string())?;
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc, None)
            .map_err(|e| format!("allocate_memory(host buffer): {:?}", e))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind_buffer_memory: {:?}", e))?;
    }
    if !data.is_empty() {
        unsafe {
            let ptr = device
                .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map_memory: {:?}", e))? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            device.unmap_memory(memory);
        }
    }
    Ok(HostBuffer { buffer, memory })
}

fn create_ubo_ring(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<UboRing, String> {
    let info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let buffer = unsafe {
        device
            .create_buffer(&info, None)
            .map_err(|e| format!("create_buffer(ubo_ring): {:?}", e))?
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| "no HOST_VISIBLE|HOST_COHERENT for ubo_ring".to_string())?;
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc, None)
            .map_err(|e| format!("allocate_memory(ubo_ring): {:?}", e))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind_buffer_memory(ubo_ring): {:?}", e))?;
    }
    let mapped = unsafe {
        device
            .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map_memory(ubo_ring): {:?}", e))? as *mut u8
    };
    Ok(UboRing {
        buffer,
        memory,
        mapped,
        size: req.size,
        head: 0,
        slot_head: [0, 0],
    })
}

fn create_default_sampler(device: &ash::Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        mip_lod_bias: 0.0,
        anisotropy_enable: vk::FALSE,
        max_anisotropy: 1.0,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::NEVER,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
        unnormalized_coordinates: vk::FALSE,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .create_sampler(&info, None)
            .map_err(|e| format!("create_sampler: {:?}", e))
    }
}

fn map_wrap(w: crate::texture::WrapMode, mag: vk::Filter) -> vk::SamplerAddressMode {
    use crate::texture::WrapMode as W;
    match w {
        W::Wrap => vk::SamplerAddressMode::REPEAT,
        W::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
        W::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        W::Border => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        W::Clamp => {
            if mag == vk::Filter::LINEAR {
                vk::SamplerAddressMode::CLAMP_TO_BORDER
            } else {
                vk::SamplerAddressMode::CLAMP_TO_EDGE
            }
        }
        W::MirrorOnceClampToEdge | W::MirrorOnceBorder | W::MirrorOnceClampOgl => {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        }
        W::Unknown => vk::SamplerAddressMode::REPEAT,
    }
}

fn vk_filter(f: crate::texture::TexFilter) -> vk::Filter {
    match f {
        crate::texture::TexFilter::Linear => vk::Filter::LINEAR,
        _ => vk::Filter::NEAREST,
    }
}

fn vk_mipmap_mode(f: crate::texture::TexFilter) -> vk::SamplerMipmapMode {
    match f {
        crate::texture::TexFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        _ => vk::SamplerMipmapMode::NEAREST,
    }
}

fn vk_compare_func(f: crate::texture::DepthCompareFunc) -> vk::CompareOp {
    match f {
        crate::texture::DepthCompareFunc::Less => vk::CompareOp::LESS,
        crate::texture::DepthCompareFunc::Equal => vk::CompareOp::EQUAL,
        crate::texture::DepthCompareFunc::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        crate::texture::DepthCompareFunc::Greater => vk::CompareOp::GREATER,
        crate::texture::DepthCompareFunc::NotEqual => vk::CompareOp::NOT_EQUAL,
        crate::texture::DepthCompareFunc::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        crate::texture::DepthCompareFunc::Always => vk::CompareOp::ALWAYS,
        crate::texture::DepthCompareFunc::Never => vk::CompareOp::NEVER,
    }
}

fn vk_sampler_reduction(r: crate::texture::SamplerReduction) -> vk::SamplerReductionMode {
    match r {
        crate::texture::SamplerReduction::Min => vk::SamplerReductionMode::MIN,
        crate::texture::SamplerReduction::Max => vk::SamplerReductionMode::MAX,
        crate::texture::SamplerReduction::WeightedAverage => {
            vk::SamplerReductionMode::WEIGHTED_AVERAGE
        }
    }
}

fn vk_border_color(bits: [u32; 4]) -> vk::BorderColor {
    if bits == [0, 0, 0, 0] {
        vk::BorderColor::FLOAT_TRANSPARENT_BLACK
    } else if bits == [0x3f80_0000, 0x3f80_0000, 0x3f80_0000, 0x3f80_0000] {
        vk::BorderColor::FLOAT_OPAQUE_WHITE
    } else {
        vk::BorderColor::FLOAT_OPAQUE_BLACK
    }
}

fn vk_integer_border_color(bits: [u32; 4]) -> vk::BorderColor {
    if bits == [0; 4] {
        vk::BorderColor::INT_TRANSPARENT_BLACK
    } else if bits == [1; 4] || bits == [u32::MAX; 4] {
        vk::BorderColor::INT_OPAQUE_WHITE
    } else {
        vk::BorderColor::INT_OPAQUE_BLACK
    }
}

fn create_integer_sampler_for_tsc(
    device: &ash::Device,
    tsc: &crate::texture::TscEntry,
) -> Result<vk::Sampler, String> {
    let mip_none = matches!(tsc.mip_filter, crate::texture::TexFilter::None);
    let info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: vk::Filter::NEAREST,
        min_filter: vk::Filter::NEAREST,
        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
        address_mode_u: map_wrap(tsc.wrap_u, vk::Filter::NEAREST),
        address_mode_v: map_wrap(tsc.wrap_v, vk::Filter::NEAREST),
        address_mode_w: map_wrap(tsc.wrap_p, vk::Filter::NEAREST),
        mip_lod_bias: tsc.lod_bias(),
        anisotropy_enable: vk::FALSE,
        max_anisotropy: 1.0,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::NEVER,
        min_lod: if mip_none { 0.0 } else { tsc.min_lod() },
        max_lod: if mip_none { 0.25 } else { tsc.max_lod() },
        border_color: vk_integer_border_color(tsc.border_color_bits),
        unnormalized_coordinates: vk::FALSE,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .create_sampler(&info, None)
            .map_err(|e| format!("create_sampler(integer tsc): {:?}", e))
    }
}

fn create_sampler_for_tsc(
    device: &ash::Device,
    tsc: &crate::texture::TscEntry,
    sampler_filter_minmax_supported: bool,
    sampler_anisotropy_supported: bool,
) -> Result<vk::Sampler, String> {
    let force_linear = std::env::var_os("NEXIUM_TEX_FORCE_LINEAR").is_some();
    let mag = if force_linear {
        vk::Filter::LINEAR
    } else {
        vk_filter(tsc.mag_filter)
    };
    let min = if force_linear {
        vk::Filter::LINEAR
    } else {
        vk_filter(tsc.min_filter)
    };
    let mip = vk_mipmap_mode(tsc.mip_filter);
    let reduction_mode = vk_sampler_reduction(tsc.reduction);
    let use_reduction = sampler_filter_minmax_supported;
    let reduction_info = vk::SamplerReductionModeCreateInfoEXT {
        s_type: vk::StructureType::SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
        reduction_mode: if sampler_filter_minmax_supported {
            reduction_mode
        } else {
            vk::SamplerReductionMode::WEIGHTED_AVERAGE
        },
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let anisotropy = tsc.max_anisotropy().clamp(1.0, 16.0);
    let mip_none = matches!(tsc.mip_filter, crate::texture::TexFilter::None);
    let info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: mag,
        min_filter: min,
        mipmap_mode: mip,
        address_mode_u: map_wrap(tsc.wrap_u, mag),
        address_mode_v: map_wrap(tsc.wrap_v, mag),
        address_mode_w: map_wrap(tsc.wrap_p, mag),
        mip_lod_bias: tsc.lod_bias(),
        anisotropy_enable: if sampler_anisotropy_supported && anisotropy > 1.0 {
            vk::TRUE
        } else {
            vk::FALSE
        },
        max_anisotropy: if sampler_anisotropy_supported {
            anisotropy
        } else {
            1.0
        },
        compare_enable: if tsc.depth_compare_enabled {
            vk::TRUE
        } else {
            vk::FALSE
        },
        compare_op: vk_compare_func(tsc.depth_compare_func),
        min_lod: if mip_none { 0.0 } else { tsc.min_lod() },
        max_lod: if mip_none { 0.25 } else { tsc.max_lod() },
        border_color: vk_border_color(tsc.border_color_bits),
        unnormalized_coordinates: vk::FALSE,
        p_next: if use_reduction {
            &reduction_info as *const _ as *const std::ffi::c_void
        } else {
            std::ptr::null()
        },
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .create_sampler(&info, None)
            .map_err(|e| format!("create_sampler(tsc): {:?}", e))
    }
}

fn cached_sampler_for_tsc(
    device: &ash::Device,
    cache: &mut HashMap<crate::texture::TscEntry, vk::Sampler>,
    tsc: crate::texture::TscEntry,
    integer_sample: bool,
    sampler_filter_minmax_supported: bool,
    sampler_anisotropy_supported: bool,
) -> Result<vk::Sampler, String> {
    if let Some(sampler) = cache.get(&tsc) {
        return Ok(*sampler);
    }
    let sampler = if integer_sample {
        create_integer_sampler_for_tsc(device, &tsc)?
    } else {
        create_sampler_for_tsc(
            device,
            &tsc,
            sampler_filter_minmax_supported,
            sampler_anisotropy_supported,
        )?
    };
    cache.insert(tsc, sampler);
    Ok(sampler)
}

fn upload_texture_oneshot(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    layers: u32,
    base_layer: u32,
    view_layers: u32,
    arrayed: bool,
    cube: bool,
    cube_array: bool,
    volume: bool,
    mip_levels: u32,
    base_mip: u32,
    view_mips: u32,
    rgba8: &[u8],
    mip_copies: &[TextureMipCopy],
    volume_slices: Option<&[VolumeRtSlice]>,
    swizzle: [crate::texture::SwizzleSource; 4],
    format: vk::Format,
    hash: u64,
    gen: u64,
) -> Result<CachedTexture, String> {
    let cmd = alloc_one_time_cmd(device, cmd_pool)?;
    begin_one_time(device, cmd)?;
    let (tex, stage) = create_texture_image(
        device,
        cmd,
        mem_props,
        width,
        height,
        layers,
        base_layer,
        view_layers,
        arrayed,
        cube,
        cube_array,
        volume,
        mip_levels,
        base_mip,
        view_mips,
        rgba8,
        mip_copies,
        volume_slices,
        swizzle,
        format,
        hash,
        gen,
    )?;
    end_one_time(device, cmd)?;
    submit_and_wait(device, queue, cmd)?;
    unsafe {
        device.free_command_buffers(cmd_pool, &[cmd]);
        if let Some((sbuf, smem)) = stage {
            device.destroy_buffer(sbuf, None);
            device.free_memory(smem, None);
        }
    }
    Ok(tex)
}

fn dump_texture_bmp_once(
    gpu_va: u64,
    width: u32,
    height: u32,
    layers: u32,
    rgba8: &[u8],
    swizzle: [crate::texture::SwizzleSource; 4],
) {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if !seen.lock().map(|mut s| s.insert(gpu_va)).unwrap_or(false) {
        return;
    }
    let Some(base) = std::env::var_os("APPDATA") else {
        return;
    };
    let layers = layers.max(1);
    let out_h = height.saturating_mul(layers);
    if width == 0 || out_h == 0 {
        return;
    }
    let row_stride = ((width as usize * 3 + 3) / 4) * 4;
    let image_size = row_stride.saturating_mul(out_h as usize);
    let file_size = 14usize.saturating_add(40).saturating_add(image_size);
    let dir = std::path::PathBuf::from(base).join("NeXium").join("logs");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("tex-{gpu_va:010x}-{width}x{height}x{layers}.bmp"));
    let mut file = match std::fs::File::create(path) {
        Ok(file) => file,
        Err(_) => return,
    };
    let mut header = Vec::with_capacity(54);
    header.extend_from_slice(b"BM");
    header.extend_from_slice(&(file_size as u32).to_le_bytes());
    header.extend_from_slice(&[0u8; 4]);
    header.extend_from_slice(&(54u32).to_le_bytes());
    header.extend_from_slice(&(40u32).to_le_bytes());
    header.extend_from_slice(&(width as i32).to_le_bytes());
    header.extend_from_slice(&(out_h as i32).to_le_bytes());
    header.extend_from_slice(&(1u16).to_le_bytes());
    header.extend_from_slice(&(24u16).to_le_bytes());
    header.extend_from_slice(&(0u32).to_le_bytes());
    header.extend_from_slice(&(image_size as u32).to_le_bytes());
    header.extend_from_slice(&(2835u32).to_le_bytes());
    header.extend_from_slice(&(2835u32).to_le_bytes());
    header.extend_from_slice(&(0u32).to_le_bytes());
    header.extend_from_slice(&(0u32).to_le_bytes());
    if file.write_all(&header).is_err() {
        return;
    }
    let layer_size = width as usize * height as usize * 4;
    let mut row = vec![0u8; row_stride];
    for y_out_rev in 0..out_h {
        let y_out = out_h - 1 - y_out_rev;
        let layer = (y_out / height) as usize;
        let y = (y_out % height) as usize;
        row.fill(0);
        for x in 0..width as usize {
            let idx = layer
                .saturating_mul(layer_size)
                .saturating_add((y * width as usize + x) * 4);
            if idx + 4 > rgba8.len() {
                continue;
            }
            let src = [rgba8[idx], rgba8[idx + 1], rgba8[idx + 2], rgba8[idx + 3]];
            let mapped = swizzle_rgba_for_dump(src, swizzle);
            let checker = if ((x / 8) + (y / 8) + layer) & 1 == 0 {
                [224u8, 224u8, 224u8]
            } else {
                [96u8, 96u8, 96u8]
            };
            let a = mapped[3] as u32;
            let r = (mapped[0] as u32 * a + checker[0] as u32 * (255 - a)) / 255;
            let g = (mapped[1] as u32 * a + checker[1] as u32 * (255 - a)) / 255;
            let b = (mapped[2] as u32 * a + checker[2] as u32 * (255 - a)) / 255;
            let dst = x * 3;
            row[dst] = b as u8;
            row[dst + 1] = g as u8;
            row[dst + 2] = r as u8;
        }
        if file.write_all(&row).is_err() {
            return;
        }
    }
}

fn swizzle_rgba_for_dump(src: [u8; 4], swizzle: [crate::texture::SwizzleSource; 4]) -> [u8; 4] {
    fn one(src: [u8; 4], s: crate::texture::SwizzleSource) -> u8 {
        match s {
            crate::texture::SwizzleSource::Zero => 0,
            crate::texture::SwizzleSource::R => src[0],
            crate::texture::SwizzleSource::G => src[1],
            crate::texture::SwizzleSource::B => src[2],
            crate::texture::SwizzleSource::A => src[3],
            crate::texture::SwizzleSource::One => 255,
            crate::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    [
        one(src, swizzle[0]),
        one(src, swizzle[1]),
        one(src, swizzle[2]),
        one(src, swizzle[3]),
    ]
}

fn texture_view_layer_range(
    layers: u32,
    base_layer: u32,
    view_layers: u32,
    arrayed: bool,
    cube: bool,
    cube_array: bool,
    volume: bool,
) -> Result<(u32, u32), String> {
    if layers == 0 {
        return Err("texture storage has zero layers".to_string());
    }
    if base_layer != 0 {
        return Err(format!(
            "per-view texture storage requires normalized base layer 0, got {}",
            base_layer
        ));
    }
    if u32::from(arrayed) + u32::from(cube) + u32::from(cube_array) + u32::from(volume) > 1 {
        return Err("texture view kinds overlap".to_string());
    }
    if cube {
        if layers != 6 || view_layers != 6 {
            return Err(format!(
                "cube texture requires exactly 6 storage/view layers, got {}/{}",
                layers, view_layers
            ));
        }
        return Ok((0, 6));
    }
    if cube_array {
        if layers < 6 || layers % 6 != 0 || view_layers != layers {
            return Err(format!(
                "cube-array texture requires matching nonzero multiples of 6, got storage={} view={}",
                layers, view_layers
            ));
        }
        return Ok((0, layers));
    }
    if arrayed {
        if view_layers != layers {
            return Err(format!(
                "2D-array texture requires matching storage/view layers, got {}/{}",
                layers, view_layers
            ));
        }
        return Ok((0, layers));
    }
    if volume {
        if view_layers != 1 {
            return Err(format!(
                "3D texture requires one view layer, got {}",
                view_layers
            ));
        }
        return Ok((0, 1));
    }
    if layers != 1 || view_layers != 1 {
        return Err(format!(
            "2D texture requires one storage/view layer, got {}/{}",
            layers, view_layers
        ));
    }
    Ok((0, 1))
}

fn create_texture_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    layers: u32,
    base_layer: u32,
    view_layers: u32,
    arrayed: bool,
    cube: bool,
    cube_array: bool,
    volume: bool,
    mip_levels: u32,
    base_mip: u32,
    view_mips: u32,
    rgba8: &[u8],
    mip_copies: &[TextureMipCopy],
    volume_slices: Option<&[VolumeRtSlice]>,
    swizzle: [crate::texture::SwizzleSource; 4],
    format: vk::Format,
    hash: u64,
    gen: u64,
) -> Result<(CachedTexture, Option<(vk::Buffer, vk::DeviceMemory)>), String> {
    if (cube || cube_array) && width != height {
        return Err(format!(
            "cube texture must be square, got {}x{}",
            width, height
        ));
    }
    let (view_base_layer, view_layer_count) = texture_view_layer_range(
        layers,
        base_layer,
        view_layers,
        arrayed,
        cube,
        cube_array,
        volume,
    )?;
    let mip_levels = if volume { 1 } else { mip_levels.max(1) };
    let max_image_mip_levels = 32u32.saturating_sub(width.max(height).max(1).leading_zeros());
    if mip_levels > max_image_mip_levels {
        return Err(format!(
            "texture mip count {} exceeds {}x{} image limit {}",
            mip_levels, width, height, max_image_mip_levels
        ));
    }
    if base_mip >= mip_levels || view_mips == 0 || base_mip.saturating_add(view_mips) > mip_levels {
        return Err(format!(
            "texture mip view exceeds storage: base={} view={} levels={}",
            base_mip, view_mips, mip_levels
        ));
    }
    let img_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: if volume {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        },
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: if volume { layers } else { 1 },
        },
        mip_levels,
        array_layers: if volume { 1 } else { layers },
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        p_next: std::ptr::null(),
        flags: if cube || cube_array {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        },
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let image = unsafe {
        device
            .create_image(&img_info, None)
            .map_err(|e| format!("create_image(tex {}x{}): {:?}", width, height, e))?
    };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "no DEVICE_LOCAL for texture image".to_string())?;
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc, None)
            .map_err(|e| format!("allocate_memory(tex): {:?}", e))?
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("bind_image_memory(tex): {:?}", e))?;
    }

    let stage = if volume_slices.is_none() {
        Some(create_host_buffer(
            device,
            mem_props,
            rgba8,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?)
    } else {
        None
    };

    transition_image_range(
        device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
        0,
        mip_levels,
        0,
        if volume { 1 } else { layers },
    );
    if let Some(slices) = volume_slices {
        unsafe {
            let clear = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            };
            let range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
        }
        for slice in slices {
            if slice.layer >= layers || slice.format != format {
                continue;
            }
            let yflip = std::env::var_os("NEXIUM_VOLUME_YFLIP").is_some();
            let zflip = std::env::var_os("NEXIUM_VOLUME_ZFLIP").is_some();
            let dst_layer = if zflip {
                layers.saturating_sub(1).saturating_sub(slice.layer)
            } else {
                slice.layer
            };
            let restore_layout = slice.layout;
            if restore_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                transition_image(
                    device,
                    cmd,
                    slice.image,
                    restore_layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
            }
            let regions: Vec<vk::ImageCopy> = (0..height)
                .map(|y| vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offset: vk::Offset3D {
                        x: slice.src_x as i32,
                        y: slice.src_y.saturating_add(y) as i32,
                        z: 0,
                    },
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offset: vk::Offset3D {
                        x: 0,
                        y: if yflip { height - 1 - y } else { y } as i32,
                        z: dst_layer as i32,
                    },
                    extent: vk::Extent3D {
                        width,
                        height: 1,
                        depth: 1,
                    },
                })
                .collect();
            unsafe {
                device.cmd_copy_image(
                    cmd,
                    slice.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
            }
            if restore_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                transition_image(
                    device,
                    cmd,
                    slice.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    restore_layout,
                );
            }
        }
    } else if let Some(stage) = stage.as_ref() {
        let copies: Vec<vk::BufferImageCopy> = mip_copies
            .iter()
            .map(|copy| vk::BufferImageCopy {
                buffer_offset: copy.buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: copy.mip_level,
                    base_array_layer: 0,
                    layer_count: if volume { 1 } else { layers },
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: copy.width,
                    height: copy.height,
                    depth: if volume { layers } else { 1 },
                },
            })
            .collect();
        if copies.is_empty() {
            return Err("texture upload has no mip copy regions".to_string());
        }
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                stage.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
        }
    }
    transition_image_range(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageAspectFlags::COLOR,
        0,
        mip_levels,
        0,
        if volume { 1 } else { layers },
    );

    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if volume {
            vk::ImageViewType::TYPE_3D
        } else if cube_array {
            vk::ImageViewType::CUBE_ARRAY
        } else if cube {
            vk::ImageViewType::CUBE
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: base_mip,
            level_count: view_mips,
            base_array_layer: if volume { 0 } else { view_base_layer },
            layer_count: if volume { 1 } else { view_layer_count },
        },
        components: texture_component_mapping(swizzle),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("create_image_view(tex): {:?}", e))?
    };
    Ok((
        CachedTexture {
            image,
            view,
            memory,
            hash,
            gen,
            verified: std::time::Instant::now(),
        },
        stage.map(|stage| (stage.buffer, stage.memory)),
    ))
}

fn texture_image_format(
    format: crate::texture::TicFormat,
    from_rt_slices: bool,
    is_srgb: bool,
    numeric_type: nexium_spirv::TextureNumericType,
) -> vk::Format {
    if format == crate::texture::TicFormat::G24R8 {
        match numeric_type {
            nexium_spirv::TextureNumericType::Float => vk::Format::R32_SFLOAT,
            nexium_spirv::TextureNumericType::Uint => vk::Format::R32_UINT,
            nexium_spirv::TextureNumericType::Sint => vk::Format::R32_SINT,
        }
    } else if numeric_type == nexium_spirv::TextureNumericType::Uint {
        vk::Format::R8G8B8A8_UINT
    } else if numeric_type == nexium_spirv::TextureNumericType::Sint {
        vk::Format::R8G8B8A8_SINT
    } else if from_rt_slices && format == crate::texture::TicFormat::B10G11R11 {
        vk::Format::B10G11R11_UFLOAT_PACK32
    } else if is_srgb && std::env::var_os("NEXIUM_NO_TEX_SRGB").is_none() {
        vk::Format::R8G8B8A8_SRGB
    } else {
        vk::Format::R8G8B8A8_UNORM
    }
}

fn texture_image_format_for_tic(
    tic: &crate::texture::TicEntry,
    numeric_type: nexium_spirv::TextureNumericType,
) -> Result<vk::Format, String> {
    use crate::texture::{ComponentType, TicFormat};

    if tic.format == TicFormat::G24R8 {
        if tic.is_srgb {
            return Err("G24R8 sRGB view is not representable".to_string());
        }
        return Ok(texture_image_format(
            tic.format,
            false,
            tic.is_srgb,
            numeric_type,
        ));
    }

    let component_type = tic.component_types[0];
    let native_color_format = matches!(
        tic.format,
        TicFormat::R8
            | TicFormat::R8G8
            | TicFormat::R8G8B8A8
            | TicFormat::R16
            | TicFormat::R16G16
            | TicFormat::R16G16B16A16
            | TicFormat::R32
            | TicFormat::R32G32
            | TicFormat::R32G32B32A32
    );
    if tic.is_srgb
        && native_color_format
        && !matches!(
            (tic.format, component_type),
            (
                TicFormat::R8 | TicFormat::R8G8 | TicFormat::R8G8B8A8,
                ComponentType::Unorm | ComponentType::UnormForceFp16
            )
        )
    {
        return Err(format!(
            "sRGB image format {:?}/{:?} is not representable",
            tic.format, component_type
        ));
    }
    let represented_components = match tic.format {
        TicFormat::R8 | TicFormat::R16 | TicFormat::R32 => 1,
        TicFormat::R8G8 | TicFormat::R16G16 | TicFormat::R32G32 => 2,
        TicFormat::R8G8B8A8 | TicFormat::R16G16B16A16 | TicFormat::R32G32B32A32 => 4,
        _ => 0,
    };
    if represented_components > 1
        && !tic.component_types[..represented_components]
            .iter()
            .all(|component| *component == component_type)
    {
        return Err(format!(
            "mixed represented image component types are not representable: {:?}",
            tic.component_types
        ));
    }
    let native = match (tic.format, component_type) {
        (TicFormat::R8, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
            Some(if tic.is_srgb {
                vk::Format::R8_SRGB
            } else {
                vk::Format::R8_UNORM
            })
        }
        (TicFormat::R8, ComponentType::Snorm | ComponentType::SnormForceFp16) => {
            Some(vk::Format::R8_SNORM)
        }
        (TicFormat::R8, ComponentType::Uint) => Some(vk::Format::R8_UINT),
        (TicFormat::R8, ComponentType::Sint) => Some(vk::Format::R8_SINT),
        (TicFormat::R8G8, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
            Some(if tic.is_srgb {
                vk::Format::R8G8_SRGB
            } else {
                vk::Format::R8G8_UNORM
            })
        }
        (TicFormat::R8G8, ComponentType::Snorm | ComponentType::SnormForceFp16) => {
            Some(vk::Format::R8G8_SNORM)
        }
        (TicFormat::R8G8, ComponentType::Uint) => Some(vk::Format::R8G8_UINT),
        (TicFormat::R8G8, ComponentType::Sint) => Some(vk::Format::R8G8_SINT),
        (TicFormat::R8G8B8A8, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
            Some(if tic.is_srgb {
                vk::Format::R8G8B8A8_SRGB
            } else {
                vk::Format::R8G8B8A8_UNORM
            })
        }
        (TicFormat::R8G8B8A8, ComponentType::Snorm | ComponentType::SnormForceFp16) => {
            Some(vk::Format::R8G8B8A8_SNORM)
        }
        (TicFormat::R8G8B8A8, ComponentType::Uint) => Some(vk::Format::R8G8B8A8_UINT),
        (TicFormat::R8G8B8A8, ComponentType::Sint) => Some(vk::Format::R8G8B8A8_SINT),
        (TicFormat::R16, ComponentType::Float) => Some(vk::Format::R16_SFLOAT),
        (TicFormat::R16, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
            Some(vk::Format::R16_UNORM)
        }
        (TicFormat::R16, ComponentType::Snorm | ComponentType::SnormForceFp16) => {
            Some(vk::Format::R16_SNORM)
        }
        (TicFormat::R16, ComponentType::Uint) => Some(vk::Format::R16_UINT),
        (TicFormat::R16, ComponentType::Sint) => Some(vk::Format::R16_SINT),
        (TicFormat::R16G16, ComponentType::Float) => Some(vk::Format::R16G16_SFLOAT),
        (TicFormat::R16G16, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
            Some(vk::Format::R16G16_UNORM)
        }
        (TicFormat::R16G16, ComponentType::Snorm | ComponentType::SnormForceFp16) => {
            Some(vk::Format::R16G16_SNORM)
        }
        (TicFormat::R16G16, ComponentType::Uint) => Some(vk::Format::R16G16_UINT),
        (TicFormat::R16G16, ComponentType::Sint) => Some(vk::Format::R16G16_SINT),
        (TicFormat::R32, ComponentType::Float) => Some(vk::Format::R32_SFLOAT),
        (TicFormat::R32, ComponentType::Uint) => Some(vk::Format::R32_UINT),
        (TicFormat::R32, ComponentType::Sint) => Some(vk::Format::R32_SINT),
        (TicFormat::R32G32, ComponentType::Float) => Some(vk::Format::R32G32_SFLOAT),
        (TicFormat::R32G32, ComponentType::Uint) => Some(vk::Format::R32G32_UINT),
        (TicFormat::R32G32, ComponentType::Sint) => Some(vk::Format::R32G32_SINT),
        (TicFormat::R32G32B32A32, ComponentType::Float) => Some(vk::Format::R32G32B32A32_SFLOAT),
        (TicFormat::R32G32B32A32, ComponentType::Uint) => Some(vk::Format::R32G32B32A32_UINT),
        (TicFormat::R32G32B32A32, ComponentType::Sint) => Some(vk::Format::R32G32B32A32_SINT),
        _ => None,
    };
    if let Some(format) = native {
        if !texture_numeric_type_matches_format(numeric_type, format) {
            return Err(format!(
                "image format {:?}/{:?} cannot back {:?} shader sampling",
                tic.format, component_type, numeric_type
            ));
        }
        return Ok(format);
    }
    if tic.format == crate::texture::TicFormat::B10G11R11
        && !tic.is_srgb
        && numeric_type == nexium_spirv::TextureNumericType::Float
        && tic
            .component_types
            .iter()
            .all(|component| *component == crate::texture::ComponentType::Float)
    {
        return Ok(vk::Format::B10G11R11_UFLOAT_PACK32);
    }
    if tic.format == crate::texture::TicFormat::R16
        && !tic.is_srgb
        && numeric_type == nexium_spirv::TextureNumericType::Float
        && tic
            .component_types
            .iter()
            .all(|component| *component == crate::texture::ComponentType::Float)
    {
        return Ok(vk::Format::R16_SFLOAT);
    }
    if tic.format != crate::texture::TicFormat::R16G16B16A16 {
        return Ok(texture_image_format(
            tic.format,
            false,
            tic.is_srgb,
            numeric_type,
        ));
    }
    if tic.is_srgb {
        return Err("R16G16B16A16 sRGB view is not representable".to_string());
    }
    let component_type = tic.component_types[0];
    if !tic
        .component_types
        .iter()
        .all(|component| *component == component_type)
    {
        return Err(format!(
            "mixed R16G16B16A16 component types are not representable: {:?}",
            tic.component_types
        ));
    }
    let (format, expected_numeric) = match component_type {
        ComponentType::Float => (
            vk::Format::R16G16B16A16_SFLOAT,
            nexium_spirv::TextureNumericType::Float,
        ),
        ComponentType::Unorm => (
            vk::Format::R16G16B16A16_UNORM,
            nexium_spirv::TextureNumericType::Float,
        ),
        ComponentType::Snorm => (
            vk::Format::R16G16B16A16_SNORM,
            nexium_spirv::TextureNumericType::Float,
        ),
        ComponentType::Uint => (
            vk::Format::R16G16B16A16_UINT,
            nexium_spirv::TextureNumericType::Uint,
        ),
        ComponentType::Sint => (
            vk::Format::R16G16B16A16_SINT,
            nexium_spirv::TextureNumericType::Sint,
        ),
        ComponentType::SnormForceFp16 | ComponentType::UnormForceFp16 => {
            return Err(format!(
                "forced-FP16 R16G16B16A16 component type is not representable: {:?}",
                component_type
            ));
        }
        ComponentType::Unknown(raw) => {
            return Err(format!("unknown R16G16B16A16 component type {}", raw));
        }
    };
    if numeric_type != expected_numeric {
        return Err(format!(
            "R16G16B16A16 {:?} cannot back {:?} shader sampling",
            component_type, numeric_type
        ));
    }
    Ok(format)
}

fn texture_numeric_type_matches_format(
    numeric_type: nexium_spirv::TextureNumericType,
    format: vk::Format,
) -> bool {
    let uint = matches!(
        format,
        vk::Format::R8_UINT
            | vk::Format::R8G8_UINT
            | vk::Format::R8G8B8_UINT
            | vk::Format::R8G8B8A8_UINT
            | vk::Format::R16_UINT
            | vk::Format::R16G16_UINT
            | vk::Format::R16G16B16_UINT
            | vk::Format::R16G16B16A16_UINT
            | vk::Format::R32_UINT
            | vk::Format::R32G32_UINT
            | vk::Format::R32G32B32_UINT
            | vk::Format::R32G32B32A32_UINT
            | vk::Format::A2B10G10R10_UINT_PACK32
            | vk::Format::A8B8G8R8_UINT_PACK32
    );
    let sint = matches!(
        format,
        vk::Format::R8_SINT
            | vk::Format::R8G8_SINT
            | vk::Format::R8G8B8_SINT
            | vk::Format::R8G8B8A8_SINT
            | vk::Format::R16_SINT
            | vk::Format::R16G16_SINT
            | vk::Format::R16G16B16_SINT
            | vk::Format::R16G16B16A16_SINT
            | vk::Format::R32_SINT
            | vk::Format::R32G32_SINT
            | vk::Format::R32G32B32_SINT
            | vk::Format::R32G32B32A32_SINT
            | vk::Format::A2B10G10R10_SINT_PACK32
            | vk::Format::A8B8G8R8_SINT_PACK32
    );
    match numeric_type {
        nexium_spirv::TextureNumericType::Float => !uint && !sint,
        nexium_spirv::TextureNumericType::Uint => uint,
        nexium_spirv::TextureNumericType::Sint => sint,
    }
}

fn rt_alias_sample_aspect(
    alias: RtAlias,
    tic: crate::texture::TicEntry,
) -> Option<vk::ImageAspectFlags> {
    if !alias.depth {
        return Some(vk::ImageAspectFlags::COLOR);
    }
    depth_stencil_sample_aspect(tic.format, tic.swizzle, alias.aspects)
}

fn depth_stencil_sample_aspect(
    format: crate::texture::TicFormat,
    swizzle: [crate::texture::SwizzleSource; 4],
    available: vk::ImageAspectFlags,
) -> Option<vk::ImageAspectFlags> {
    use crate::texture::{SwizzleSource, TicFormat};
    let any_r = swizzle.iter().any(|source| *source == SwizzleSource::R);
    let aspect = match format {
        TicFormat::G24R8 | TicFormat::Z24S8 => {
            if any_r {
                vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        }
        TicFormat::S8Z24 => {
            if any_r {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::STENCIL
            }
        }
        TicFormat::X8Z24 | TicFormat::Z32 => vk::ImageAspectFlags::DEPTH,
        _ => return None,
    };
    available.contains(aspect).then_some(aspect)
}

fn rt_alias_numeric_type_matches(
    alias: RtAlias,
    tic: Option<crate::texture::TicEntry>,
    numeric_type: nexium_spirv::TextureNumericType,
    view_format: vk::Format,
) -> bool {
    if !alias.depth {
        return texture_numeric_type_matches_format(numeric_type, view_format);
    }
    match tic.and_then(|tic| rt_alias_sample_aspect(alias, tic)) {
        Some(vk::ImageAspectFlags::STENCIL) => {
            numeric_type == nexium_spirv::TextureNumericType::Uint
        }
        Some(vk::ImageAspectFlags::DEPTH) => {
            numeric_type == nexium_spirv::TextureNumericType::Float
        }
        _ => false,
    }
}

fn color_formats_for_call(
    call: &crate::draw::Maxwell3dDrawCall,
    attachment_count: usize,
) -> Vec<vk::Format> {
    if attachment_count == 0 {
        return Vec::new();
    }
    let mut formats = if call.color_rt_formats.is_empty() {
        vec![call.rt_format]
    } else {
        call.color_rt_formats.clone()
    };
    let count = attachment_count.min(8);
    if formats.len() < count {
        formats.resize(count, call.rt_format);
    }
    formats.truncate(count);
    formats
}

fn rt_alias_view_format(
    key: RtKey,
    tic: crate::texture::TicEntry,
    fallback: vk::Format,
) -> vk::Format {
    let _ = key;
    use crate::texture::{ComponentType, TicFormat};
    let ty = tic.component_types[0];
    match tic.format {
        TicFormat::A2B10G10R10 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => {
                vk::Format::A2B10G10R10_UNORM_PACK32
            }
            ComponentType::Uint => vk::Format::A2B10G10R10_UINT_PACK32,
            ComponentType::Sint => vk::Format::A2B10G10R10_SINT_PACK32,
            _ => fallback,
        },
        TicFormat::A8B8G8R8 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => {
                vk::Format::A8B8G8R8_UNORM_PACK32
            }
            ComponentType::Snorm | ComponentType::SnormForceFp16 => {
                vk::Format::A8B8G8R8_SNORM_PACK32
            }
            ComponentType::Uint => vk::Format::A8B8G8R8_UINT_PACK32,
            ComponentType::Sint => vk::Format::A8B8G8R8_SINT_PACK32,
            _ => fallback,
        },
        TicFormat::R8G8B8A8 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R8G8B8A8_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R8G8B8A8_SNORM,
            ComponentType::Uint => vk::Format::R8G8B8A8_UINT,
            ComponentType::Sint => vk::Format::R8G8B8A8_SINT,
            _ => fallback,
        },
        TicFormat::R16G16B16A16 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R16G16B16A16_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R16G16B16A16_SNORM,
            ComponentType::Uint => vk::Format::R16G16B16A16_UINT,
            ComponentType::Sint => vk::Format::R16G16B16A16_SINT,
            ComponentType::Float => vk::Format::R16G16B16A16_SFLOAT,
            _ => fallback,
        },
        TicFormat::R16G16 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R16G16_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R16G16_SNORM,
            ComponentType::Uint => vk::Format::R16G16_UINT,
            ComponentType::Sint => vk::Format::R16G16_SINT,
            ComponentType::Float => vk::Format::R16G16_SFLOAT,
            _ => fallback,
        },
        TicFormat::R16 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R16_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R16_SNORM,
            ComponentType::Uint => vk::Format::R16_UINT,
            ComponentType::Sint => vk::Format::R16_SINT,
            ComponentType::Float => vk::Format::R16_SFLOAT,
            _ => fallback,
        },
        TicFormat::R8G8 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R8G8_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R8G8_SNORM,
            ComponentType::Uint => vk::Format::R8G8_UINT,
            ComponentType::Sint => vk::Format::R8G8_SINT,
            _ => fallback,
        },
        TicFormat::R8 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R8_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R8_SNORM,
            ComponentType::Uint => vk::Format::R8_UINT,
            ComponentType::Sint => vk::Format::R8_SINT,
            _ => fallback,
        },
        TicFormat::R32G32B32A32 => match ty {
            ComponentType::Uint => vk::Format::R32G32B32A32_UINT,
            ComponentType::Sint => vk::Format::R32G32B32A32_SINT,
            ComponentType::Float => vk::Format::R32G32B32A32_SFLOAT,
            _ => fallback,
        },
        TicFormat::R32G32 => match ty {
            ComponentType::Uint => vk::Format::R32G32_UINT,
            ComponentType::Sint => vk::Format::R32G32_SINT,
            ComponentType::Float => vk::Format::R32G32_SFLOAT,
            _ => fallback,
        },
        TicFormat::R32 => match ty {
            ComponentType::Uint => vk::Format::R32_UINT,
            ComponentType::Sint => vk::Format::R32_SINT,
            ComponentType::Float => vk::Format::R32_SFLOAT,
            _ => fallback,
        },
        TicFormat::B10G11R11 => vk::Format::B10G11R11_UFLOAT_PACK32,
        _ => fallback,
    }
}

fn rt_alias_reinterpret_supported(
    texture_key: TexCacheKey,
    tic: crate::texture::TicEntry,
    alias: RtAlias,
    view_format: vk::Format,
) -> bool {
    !alias.depth
        && alias.key.gpu_va != 0
        && alias.key.gpu_va == tic.gpu_va
        && texture_key.gpu_va == tic.gpu_va
        && alias.key.width == 1
        && alias.key.height == 1
        && tic.width == 1
        && tic.height == 1
        && tic.depth == 1
        && tic.base_layer == 0
        && texture_key.width == 1
        && texture_key.height == 1
        && texture_key.layers == 1
        && texture_key.base_layer == 0
        && texture_key.view_layers == 1
        && !texture_key.arrayed
        && !texture_key.cube
        && !texture_key.cube_array
        && !texture_key.volume
        && alias.format == vk::Format::R32G32B32A32_SFLOAT
        && tic.format == crate::texture::TicFormat::A8B8G8R8
        && view_format == vk::Format::A8B8G8R8_UNORM_PACK32
}

fn prepare_rt_alias_reinterpret(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    rt_cache: &RtCache,
    cache: &mut HashMap<RtReinterpretKey, RtReinterpretTexture>,
    texture_key: TexCacheKey,
    tic: crate::texture::TicEntry,
    alias: RtAlias,
    view_format: vk::Format,
    source_is_snapshot: bool,
) -> Result<Option<(vk::ImageView, Option<PendingRtReinterpret>)>, String> {
    if !rt_alias_reinterpret_supported(texture_key, tic, alias, view_format) {
        return Ok(None);
    }
    let Some((_, live_image, _, live_layout, live_format, source_stamp)) =
        rt_cache.color_exact_with_format(alias.key)
    else {
        return Ok(None);
    };
    if source_stamp == 0
        || live_format != vk::Format::R32G32B32A32_SFLOAT
        || (!source_is_snapshot && live_image != alias.image)
    {
        return Ok(None);
    }
    let source_layout = if source_is_snapshot {
        alias.layout
    } else {
        rt_cache.color_layout(alias.key).unwrap_or(live_layout)
    };
    if source_layout == vk::ImageLayout::UNDEFINED {
        return Ok(None);
    }

    let key = RtReinterpretKey {
        source: alias.key,
        texture: texture_key,
    };
    let target = match cache.entry(key) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(create_rt_reinterpret_texture(
            device,
            mem_props,
            tic.swizzle,
        )?),
    };
    let pending = if target.source_stamp == source_stamp {
        None
    } else {
        Some(PendingRtReinterpret {
            key,
            source: alias,
            source_layout,
            source_stamp,
            track_source_layout: !source_is_snapshot,
        })
    };
    Ok(Some((target.view, pending)))
}

fn create_rt_reinterpret_texture(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    swizzle: [crate::texture::SwizzleSource; 4],
) -> Result<RtReinterpretTexture, String> {
    let format = vk::Format::A8B8G8R8_UNORM_PACK32;
    let image_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        p_next: std::ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("create_image(rt reinterpret): {:?}", e))?
    };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) else {
        unsafe { device.destroy_image(image, None) };
        return Err("no DEVICE_LOCAL memory for RT reinterpret image".to_string());
    };
    let allocation = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_image(image, None) };
            return Err(format!("allocate_memory(rt reinterpret): {:?}", e));
        }
    };
    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        return Err(format!("bind_image_memory(rt reinterpret): {:?}", e));
    }
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        components: texture_component_mapping(swizzle),
        p_next: std::ptr::null(),
        flags: vk::ImageViewCreateFlags::empty(),
        _marker: std::marker::PhantomData,
    };
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(e) => {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(format!("create_image_view(rt reinterpret): {:?}", e));
        }
    };
    Ok(RtReinterpretTexture {
        image,
        view,
        memory,
        source_stamp: 0,
    })
}

fn rt_reinterpret_layout_scope(
    layout: vk::ImageLayout,
) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

fn rt_reinterpret_image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn record_rt_alias_reinterpret(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    rt_cache: &mut RtCache,
    frame_slot: &mut FrameSlot,
    cache: &mut HashMap<RtReinterpretKey, RtReinterpretTexture>,
    pending: PendingRtReinterpret,
) -> Result<(), String> {
    let Some(target) = cache.get_mut(&pending.key) else {
        return Err("RT reinterpret cache entry disappeared".to_string());
    };
    if target.source_stamp == pending.source_stamp {
        return Ok(());
    }
    let transfer = create_transfer_buffer_owned(device, mem_props, 16)?;
    let (source_stage, source_access) = rt_reinterpret_layout_scope(pending.source_layout);
    if pending.source_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        rt_reinterpret_image_barrier(
            device,
            cmd,
            pending.source.image,
            pending.source_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            source_stage,
            vk::PipelineStageFlags::TRANSFER,
            source_access,
            vk::AccessFlags::TRANSFER_READ,
        );
    }
    let target_old_layout = if target.source_stamp == 0 {
        vk::ImageLayout::UNDEFINED
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    };
    let (target_stage, target_access) = rt_reinterpret_layout_scope(target_old_layout);
    rt_reinterpret_image_barrier(
        device,
        cmd,
        target.image,
        target_old_layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        target_stage,
        vk::PipelineStageFlags::TRANSFER,
        target_access,
        vk::AccessFlags::TRANSFER_WRITE,
    );

    let source_copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    let target_copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_image_to_buffer(
            cmd,
            pending.source.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            transfer.buffer,
            &[source_copy],
        );
    }

    let buffer_barrier = vk::BufferMemoryBarrier {
        s_type: vk::StructureType::BUFFER_MEMORY_BARRIER,
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::TRANSFER_READ,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        buffer: transfer.buffer,
        offset: 0,
        size: 16,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let mut source_restore = Vec::new();
    if pending.source_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        source_restore.push(vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: pending.source_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: pending.source.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_access_mask: vk::AccessFlags::TRANSFER_READ,
            dst_access_mask: source_access,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
    }
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER | source_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier],
            &source_restore,
        );
        device.cmd_copy_buffer_to_image(
            cmd,
            transfer.buffer,
            target.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[target_copy],
        );
    }
    rt_reinterpret_image_barrier(
        device,
        cmd,
        target.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::ALL_GRAPHICS,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
    );
    if pending.track_source_layout {
        rt_cache.set_color_layout(pending.source.key, pending.source_layout);
    }
    target.source_stamp = pending.source_stamp;
    frame_slot
        .retired_buffers
        .push((transfer.buffer, transfer.memory));
    log::debug!(
        "rt alias raw reinterpret key={} stamp={} {:?}->{:?}",
        pending.source.key.label(),
        pending.source_stamp,
        pending.source.format,
        vk::Format::A8B8G8R8_UNORM_PACK32
    );
    Ok(())
}

fn rt_alias_sample_view(
    device: &ash::Device,
    rt_cache: &mut RtCache,
    slot: &mut FrameSlot,
    alias: RtAlias,
    tic: Option<crate::texture::TicEntry>,
    view_format: vk::Format,
) -> Option<vk::ImageView> {
    let swizzle = tic
        .map(|tic| tic.swizzle)
        .unwrap_or(TEXTURE_IDENTITY_SWIZZLE);
    if !alias.depth
        && !alias.key.is_3d
        && swizzle == TEXTURE_IDENTITY_SWIZZLE
        && view_format == alias.format
    {
        return Some(alias.view);
    }
    if !alias.depth && !rt_formats_compatible(alias.format, view_format) {
        log::debug!(
            "rt alias sample view incompatible key={} fmt={:?}->{:?}",
            alias.key.label(),
            alias.format,
            view_format
        );
        return None;
    }

    let aspect = if alias.depth {
        rt_alias_sample_aspect(alias, tic?)?
    } else {
        vk::ImageAspectFlags::COLOR
    };
    let components = if alias.depth {
        depth_stencil_component_mapping(swizzle)
    } else {
        texture_component_mapping(swizzle)
    };

    let sample_format = if alias.depth {
        alias.format
    } else {
        view_format
    };
    let view_type = if alias.key.is_3d {
        vk::ImageViewType::TYPE_3D
    } else {
        vk::ImageViewType::TYPE_2D
    };
    match rt_cache.get_or_create_sample_view(
        device,
        alias.key,
        alias.image,
        sample_format,
        aspect,
        view_type,
        components,
    ) {
        Ok(Some(view)) => Some(view),
        Ok(None) => {
            let view_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                image: alias.image,
                view_type,
                format: sample_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                components,
                p_next: std::ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                _marker: std::marker::PhantomData,
            };
            match unsafe { device.create_image_view(&view_info, None) } {
                Ok(view) => {
                    slot.retired_views.push(view);
                    Some(view)
                }
                Err(error) => {
                    log::warn!(
                        "rt alias sample view failed key={} fmt={:?}->{:?} aspect={:?} swz={:?}: {:?}",
                        alias.key.label(),
                        alias.format,
                        view_format,
                        aspect,
                        swizzle,
                        error
                    );
                    None
                }
            }
        }
        Err(e) => {
            log::warn!(
                "rt alias sample view failed key={} fmt={:?}->{:?} aspect={:?} swz={:?}: {:?}",
                alias.key.label(),
                alias.format,
                view_format,
                aspect,
                swizzle,
                e
            );
            None
        }
    }
}

fn depth_stencil_component_mapping(
    mut swizzle: [crate::texture::SwizzleSource; 4],
) -> vk::ComponentMapping {
    for source in &mut swizzle {
        if *source == crate::texture::SwizzleSource::G {
            *source = crate::texture::SwizzleSource::R;
        }
    }
    texture_component_mapping(swizzle)
}

fn texture_component_mapping(swizzle: [crate::texture::SwizzleSource; 4]) -> vk::ComponentMapping {
    fn one(src: crate::texture::SwizzleSource) -> vk::ComponentSwizzle {
        match src {
            crate::texture::SwizzleSource::Zero => vk::ComponentSwizzle::ZERO,
            crate::texture::SwizzleSource::R => vk::ComponentSwizzle::R,
            crate::texture::SwizzleSource::G => vk::ComponentSwizzle::G,
            crate::texture::SwizzleSource::B => vk::ComponentSwizzle::B,
            crate::texture::SwizzleSource::A => vk::ComponentSwizzle::A,
            crate::texture::SwizzleSource::One => vk::ComponentSwizzle::ONE,
            crate::texture::SwizzleSource::Unknown(_) => vk::ComponentSwizzle::ZERO,
        }
    }

    if std::env::var_os("NEXIUM_TEX_FORCE_RB_SWAP").is_some() {
        vk::ComponentMapping {
            r: one(swizzle[2]),
            g: one(swizzle[1]),
            b: one(swizzle[0]),
            a: one(swizzle[3]),
        }
    } else {
        vk::ComponentMapping {
            r: one(swizzle[0]),
            g: one(swizzle[1]),
            b: one(swizzle[2]),
            a: one(swizzle[3]),
        }
    }
}

fn texture_view_swizzle(
    format: crate::texture::TicFormat,
    numeric_type: nexium_spirv::TextureNumericType,
    mut swizzle: [crate::texture::SwizzleSource; 4],
) -> [crate::texture::SwizzleSource; 4] {
    if format == crate::texture::TicFormat::G24R8
        && numeric_type == nexium_spirv::TextureNumericType::Float
    {
        for component in &mut swizzle {
            if *component == crate::texture::SwizzleSource::G {
                *component = crate::texture::SwizzleSource::R;
            }
        }
    }
    swizzle
}

fn ensure_graphics_dummy_views(
    images: &mut HashMap<(u8, DummyImageKind), DummyImage>,
    texel_buffers: &mut [Option<TexelBufferResource>; 3],
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<GraphicsDummyViews, String> {
    let mut image_2d = [vk::ImageView::null(); 3];
    let mut image_2d_array = [vk::ImageView::null(); 3];
    let mut image_3d = [vk::ImageView::null(); 3];
    let mut image_cube = [vk::ImageView::null(); 3];
    let mut image_cube_array = [vk::ImageView::null(); 3];
    let mut texel_buffer = [vk::BufferView::null(); 3];

    for (family, numeric_type) in GRAPHICS_TEXTURE_NUMERIC_TYPES.into_iter().enumerate() {
        image_2d[family] = ensure_dummy_image(
            images,
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            DummyImageKind::D2,
        )?;
        image_2d_array[family] = ensure_dummy_image(
            images,
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            DummyImageKind::D2Array,
        )?;
        image_3d[family] = ensure_dummy_image(
            images,
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            DummyImageKind::D3,
        )?;
        image_cube[family] = ensure_dummy_image(
            images,
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            DummyImageKind::Cube,
        )?;
        image_cube_array[family] = ensure_dummy_image(
            images,
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            DummyImageKind::CubeArray,
        )?;
        texel_buffer[family] =
            ensure_dummy_texel_buffer(texel_buffers, device, mem_props, numeric_type)?;
    }
    let depth_image_2d = ensure_dummy_image(
        images,
        device,
        queue,
        cmd_pool,
        mem_props,
        nexium_spirv::TextureNumericType::Float,
        DummyImageKind::DepthD2,
    )?;
    let depth_image_2d_array = ensure_dummy_image(
        images,
        device,
        queue,
        cmd_pool,
        mem_props,
        nexium_spirv::TextureNumericType::Float,
        DummyImageKind::DepthD2Array,
    )?;
    let depth_image_cube = ensure_dummy_image(
        images,
        device,
        queue,
        cmd_pool,
        mem_props,
        nexium_spirv::TextureNumericType::Float,
        DummyImageKind::DepthCube,
    )?;
    let depth_image_cube_array = ensure_dummy_image(
        images,
        device,
        queue,
        cmd_pool,
        mem_props,
        nexium_spirv::TextureNumericType::Float,
        DummyImageKind::DepthCubeArray,
    )?;

    Ok(GraphicsDummyViews {
        image_2d,
        image_2d_array,
        image_3d,
        image_cube,
        image_cube_array,
        depth_image_2d,
        depth_image_2d_array,
        depth_image_cube,
        depth_image_cube_array,
        texel_buffer,
    })
}

fn ensure_dummy_image(
    images: &mut HashMap<(u8, DummyImageKind), DummyImage>,
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    numeric_type: nexium_spirv::TextureNumericType,
    kind: DummyImageKind,
) -> Result<vk::ImageView, String> {
    let key = (texture_numeric_cache_key(numeric_type), kind);
    if let Entry::Vacant(entry) = images.entry(key) {
        entry.insert(create_dummy_image(
            device,
            queue,
            cmd_pool,
            mem_props,
            numeric_type,
            kind,
        )?);
    }
    Ok(images[&key].view)
}

fn dummy_image_format_and_aspect(
    numeric_type: nexium_spirv::TextureNumericType,
    kind: DummyImageKind,
) -> Result<(vk::Format, vk::ImageAspectFlags), String> {
    let depth = matches!(
        kind,
        DummyImageKind::DepthD2
            | DummyImageKind::DepthD2Array
            | DummyImageKind::DepthCube
            | DummyImageKind::DepthCubeArray
    );
    if depth {
        if numeric_type != nexium_spirv::TextureNumericType::Float {
            return Err("depth dummy images require the Float numeric family".to_string());
        }
        return Ok((vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH));
    }
    let format = match numeric_type {
        nexium_spirv::TextureNumericType::Float => vk::Format::R8G8B8A8_UNORM,
        nexium_spirv::TextureNumericType::Uint => vk::Format::R32_UINT,
        nexium_spirv::TextureNumericType::Sint => vk::Format::R32_SINT,
    };
    Ok((format, vk::ImageAspectFlags::COLOR))
}

fn create_dummy_image(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    numeric_type: nexium_spirv::TextureNumericType,
    kind: DummyImageKind,
) -> Result<DummyImage, String> {
    let arrayed = matches!(kind, DummyImageKind::D2Array | DummyImageKind::DepthD2Array);
    let cube = matches!(kind, DummyImageKind::Cube | DummyImageKind::DepthCube);
    let cube_array = matches!(
        kind,
        DummyImageKind::CubeArray | DummyImageKind::DepthCubeArray
    );
    let volume = kind == DummyImageKind::D3;
    let layer_count = if cube || cube_array { 6 } else { 1 };
    let (format, aspect) = dummy_image_format_and_aspect(numeric_type, kind)?;
    let img_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: if volume {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        },
        format,
        extent: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: layer_count,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        p_next: std::ptr::null(),
        flags: if cube || cube_array {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        },
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let image = unsafe {
        device
            .create_image(&img_info, None)
            .map_err(|e| format!("create_image(dummy): {:?}", e))?
    };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| "no DEVICE_LOCAL for dummy image".to_string())?;
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc, None)
            .map_err(|e| format!("allocate_memory(dummy image): {:?}", e))?
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| format!("bind_image_memory(dummy): {:?}", e))?;
    }

    let pixel: [u8; 4] = match numeric_type {
        nexium_spirv::TextureNumericType::Float => [0, 0, 0, 0],
        nexium_spirv::TextureNumericType::Uint | nexium_spirv::TextureNumericType::Sint => {
            0u32.to_le_bytes()
        }
    };
    let pixels = pixel.repeat(layer_count as usize);
    let stage = create_host_buffer(
        device,
        mem_props,
        &pixels,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    let cmd = alloc_one_time_cmd(device, cmd_pool)?;
    begin_one_time(device, cmd)?;
    transition_image_aspect(
        device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        aspect,
    );
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            stage.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
    }
    transition_image_aspect(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        aspect,
    );
    end_one_time(device, cmd)?;
    submit_and_wait(device, queue, cmd)?;
    unsafe {
        device.free_command_buffers(cmd_pool, &[cmd]);
        device.destroy_buffer(stage.buffer, None);
        device.free_memory(stage.memory, None);
    }

    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if volume {
            vk::ImageViewType::TYPE_3D
        } else if cube_array {
            vk::ImageViewType::CUBE_ARRAY
        } else if cube {
            vk::ImageViewType::CUBE
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count,
        },
        components: vk::ComponentMapping::default(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("create_image_view(dummy): {:?}", e))?
    };
    Ok(DummyImage {
        image,
        view,
        memory,
    })
}

pub fn hash_spirv(spirv: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in spirv {
        h ^= *w as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn texstat_event(kind: u8, bytes: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NEXIUM_RENDER_PROFILE").is_some()) {
        return;
    }
    static FRESH: AtomicU64 = AtomicU64::new(0);
    static READS: AtomicU64 = AtomicU64::new(0);
    static READ_BYTES: AtomicU64 = AtomicU64::new(0);
    static UPLOADS: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    static SAM_NONE: AtomicU64 = AtomicU64::new(0);
    static SAM_MISS: AtomicU64 = AtomicU64::new(0);
    static GEN_MISS: AtomicU64 = AtomicU64::new(0);
    match kind {
        0 => {
            FRESH.fetch_add(1, Ordering::Relaxed);
        }
        1 => {
            READS.fetch_add(1, Ordering::Relaxed);
            READ_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        }
        2 => {
            UPLOADS.fetch_add(1, Ordering::Relaxed);
            UPLOAD_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        }
        3 => {
            SAM_NONE.fetch_add(1, Ordering::Relaxed);
        }
        4 => {
            SAM_MISS.fetch_add(1, Ordering::Relaxed);
        }
        _ => {
            GEN_MISS.fetch_add(1, Ordering::Relaxed);
        }
    }
    static LAST: OnceLock<Mutex<(std::time::Instant, [u64; 8])>> = OnceLock::new();
    let cell = LAST.get_or_init(|| Mutex::new((std::time::Instant::now(), [0; 8])));
    let mut last = cell.lock().unwrap();
    if last.0.elapsed() >= std::time::Duration::from_secs(1) {
        let dt = last.0.elapsed().as_secs_f64();
        let cur = [
            FRESH.load(Ordering::Relaxed),
            READS.load(Ordering::Relaxed),
            READ_BYTES.load(Ordering::Relaxed),
            UPLOADS.load(Ordering::Relaxed),
            UPLOAD_BYTES.load(Ordering::Relaxed),
            SAM_NONE.load(Ordering::Relaxed),
            SAM_MISS.load(Ordering::Relaxed),
            GEN_MISS.load(Ordering::Relaxed),
        ];
        let d: Vec<u64> = cur.iter().zip(last.1.iter()).map(|(c, p)| c - p).collect();
        *last = (std::time::Instant::now(), cur);
        log::warn!(
            "[texstat] fresh/s={:.0} reads/s={:.0} read_mb/s={:.1} uploads/s={:.0} upload_mb/s={:.1} sam_none/s={:.0} sam_miss/s={:.0} gen_miss/s={:.0}",
            d[0] as f64 / dt,
            d[1] as f64 / dt,
            d[2] as f64 / dt / 1e6,
            d[3] as f64 / dt,
            d[4] as f64 / dt / 1e6,
            d[5] as f64 / dt,
            d[6] as f64 / dt,
            d[7] as f64 / dt
        );
    }
}

const TEX_SAMPLE_CHUNKS: usize = 64;
const TEX_SAMPLE_CHUNK_LEN: usize = 1024;

fn tex_sample_spans(len: usize) -> Vec<(usize, usize)> {
    if len <= TEX_SAMPLE_CHUNKS * TEX_SAMPLE_CHUNK_LEN {
        return vec![(0, len)];
    }
    (0..TEX_SAMPLE_CHUNKS)
        .map(|i| {
            (
                i * (len - TEX_SAMPLE_CHUNK_LEN) / (TEX_SAMPLE_CHUNKS - 1),
                TEX_SAMPLE_CHUNK_LEN,
            )
        })
        .collect()
}

fn fnv_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn hash_sampled(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    h ^= bytes.len() as u64;
    h = h.wrapping_mul(0x100000001b3);
    for (off, len) in tex_sample_spans(bytes.len()) {
        h = fnv_bytes(h, &bytes[off..off + len]);
    }
    h
}

fn hash_sampled_guest<F>(read_guest: &F, gpu_va: u64, len: usize) -> Option<u64>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    let mut h: u64 = 0xcbf29ce484222325;
    h ^= len as u64;
    h = h.wrapping_mul(0x100000001b3);
    for (off, chunk_len) in tex_sample_spans(len) {
        let chunk = read_guest(gpu_va + off as u64, chunk_len)?;
        if chunk.len() != chunk_len {
            return None;
        }
        h = fnv_bytes(h, &chunk);
    }
    Some(h)
}

#[allow(dead_code)]
fn hash_src_prefix(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;

    h ^= bytes.len() as u64;
    h = h.wrapping_mul(0x100000001b3);

    if bytes.len() <= 65_536 {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        return h;
    }

    const SAMPLES: usize = 65_536;
    let last = bytes.len() - 1;
    for i in 0..SAMPLES {
        let idx = i * last / (SAMPLES - 1);
        h ^= bytes[idx] as u64;
        h = h.wrapping_mul(0x100000001b3);
    }

    h
}

fn alloc_one_time_cmd(
    device: &ash::Device,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, String> {
    let info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let v = unsafe {
        device
            .allocate_command_buffers(&info)
            .map_err(|e| format!("allocate_command_buffers: {:?}", e))?
    };
    Ok(v[0])
}

fn acquire_clear_slot<'a>(
    device: &ash::Device,
    clear_slots: &'a mut [ClearSlot],
    clear_slot_index: &mut usize,
) -> Result<&'a mut ClearSlot, String> {
    let idx = *clear_slot_index;
    *clear_slot_index = (idx + 1) % clear_slots.len();
    let slot = &mut clear_slots[idx];
    if slot.in_flight {
        wait_fence(device, slot.fence)?;
        slot.in_flight = false;
    }
    Ok(slot)
}

fn set_dynamic_stencil_state(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    state: crate::draw::StencilState,
) {
    unsafe {
        if state.front.reference == state.back.reference {
            device.cmd_set_stencil_reference(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.front.reference,
            );
        } else {
            device.cmd_set_stencil_reference(
                cmd,
                vk::StencilFaceFlags::FRONT,
                state.front.reference,
            );
            device.cmd_set_stencil_reference(cmd, vk::StencilFaceFlags::BACK, state.back.reference);
        }
        if state.front.compare_mask == state.back.compare_mask {
            device.cmd_set_stencil_compare_mask(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.front.compare_mask,
            );
        } else {
            device.cmd_set_stencil_compare_mask(
                cmd,
                vk::StencilFaceFlags::FRONT,
                state.front.compare_mask,
            );
            device.cmd_set_stencil_compare_mask(
                cmd,
                vk::StencilFaceFlags::BACK,
                state.back.compare_mask,
            );
        }
        if state.front.write_mask == state.back.write_mask {
            device.cmd_set_stencil_write_mask(
                cmd,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                state.front.write_mask,
            );
        } else {
            device.cmd_set_stencil_write_mask(
                cmd,
                vk::StencilFaceFlags::FRONT,
                state.front.write_mask,
            );
            device.cmd_set_stencil_write_mask(
                cmd,
                vk::StencilFaceFlags::BACK,
                state.back.write_mask,
            );
        }
    }
}

fn draw_scissor(rect: Option<[i32; 4]>, extent: vk::Extent2D) -> vk::Rect2D {
    let Some([x, y, w, h]) = rect else {
        return vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
    };
    let x = x.max(0) as u32;
    let y = y.max(0) as u32;
    if x >= extent.width || y >= extent.height {
        return vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
        };
    }
    let w = (w.max(0) as u32).min(extent.width - x);
    let h = (h.max(0) as u32).min(extent.height - y);
    vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D {
            width: w,
            height: h,
        },
    }
}

fn begin_one_time(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), String> {
    let begin = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: std::ptr::null(),
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| format!("begin_command_buffer: {:?}", e))
    }
}

fn end_one_time(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), String> {
    unsafe {
        device
            .end_command_buffer(cmd)
            .map_err(|e| format!("end_command_buffer: {:?}", e))
    }
}

fn submit_and_wait(
    device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> Result<(), String> {
    let submit = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        wait_semaphore_count: 0,
        p_wait_semaphores: std::ptr::null(),
        p_wait_dst_stage_mask: std::ptr::null(),
        signal_semaphore_count: 0,
        p_signal_semaphores: std::ptr::null(),
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .queue_submit(queue, &[submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit: {:?}", e))?;
        note_queue_submission();
        device
            .queue_wait_idle(queue)
            .map_err(|e| format!("queue_wait_idle: {:?}", e))?;
    }
    note_queue_drained();
    Ok(())
}

fn submit_with_fence(
    device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<(), String> {
    submit_with_fence_untracked(device, queue, cmd, fence)?;
    note_queue_submission();
    Ok(())
}

fn submit_with_fence_untracked(
    device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<(), String> {
    let submit = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        wait_semaphore_count: 0,
        p_wait_semaphores: std::ptr::null(),
        p_wait_dst_stage_mask: std::ptr::null(),
        signal_semaphore_count: 0,
        p_signal_semaphores: std::ptr::null(),
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device
            .reset_fences(&[fence])
            .map_err(|e| format!("reset_fences(submit): {:?}", e))?;
        device
            .queue_submit(queue, &[submit], fence)
            .map_err(|e| format!("queue_submit(fence): {:?}", e))?;
    }
    Ok(())
}

fn wait_fence(device: &ash::Device, fence: vk::Fence) -> Result<(), String> {
    unsafe {
        match device.wait_for_fences(&[fence], true, 2_000_000_000) {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => {
                log::warn!("wait_fence: 2s timeout, extended wait");
                device
                    .wait_for_fences(&[fence], true, 8_000_000_000)
                    .map_err(|e| format!("wait_for_fences(hung 10s): {:?}", e))?;
            }
            Err(e) => return Err(format!("wait_for_fences: {:?}", e)),
        }
        device
            .reset_fences(&[fence])
            .map_err(|e| format!("reset_fences: {:?}", e))?;
    }
    Ok(())
}

fn align_up(x: u64, align: u64) -> u64 {
    debug_assert!(align > 0);
    let remainder = x % align;
    if remainder == 0 {
        x
    } else {
        x.saturating_add(align - remainder)
    }
}

fn graphics_cbuf_allocation_size(
    byte_len: usize,
    min_storage_alignment: u64,
    max_storage_range: u64,
) -> Result<(u64, u64), String> {
    let range = u64::try_from(byte_len)
        .map_err(|_| "graphics cbuf storage range does not fit in u64".to_string())?;
    if range > max_storage_range {
        return Err(format!(
            "graphics cbuf storage range {:#x} exceeds device maxStorageBufferRange {:#x}",
            range, max_storage_range
        ));
    }
    let alignment = min_storage_alignment.max(1);
    let allocation = range
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| "graphics cbuf aligned allocation size overflowed".to_string())?;
    Ok((range, allocation))
}

fn empty_graphics_cbuf_data() -> Vec<u8> {
    let mut data = vec![0u8; nexium_spirv::GFX_CBUF_MIN_SIZE as usize];
    for slot in 0..nexium_spirv::GFX_CBUF_SLOTS as usize {
        let directory = slot * 8;
        data[directory..directory + 4]
            .copy_from_slice(&nexium_spirv::GFX_CBUF_ZERO_WORD.to_le_bytes());
    }
    data
}

fn reset_command_buffer(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), String> {
    unsafe {
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| format!("reset_command_buffer: {:?}", e))
    }
}

fn ring_alloc(
    ring: &mut UboRing,
    size: u64,
    align: u64,
) -> Result<(vk::Buffer, u64, *mut u8), &'static str> {
    if size == 0 {
        return Err("ring_alloc: zero-size request");
    }
    if align == 0 {
        return Err("ring_alloc: zero alignment");
    }
    let remainder = ring.head % align;
    let padding = if remainder == 0 { 0 } else { align - remainder };
    let aligned_head = ring
        .head
        .checked_add(padding)
        .ok_or("ring_alloc: alignment overflow")?;
    let end = aligned_head
        .checked_add(size)
        .ok_or("ring_alloc: overflow")?;
    if end > ring.size {
        return Err("ring_alloc: out of space (wrap required)");
    }
    let ptr = unsafe { ring.mapped.add(aligned_head as usize) };
    ring.head = end;
    Ok((ring.buffer, aligned_head, ptr))
}

fn ring_allocation_fits(ring: &UboRing, size: u64, align: u64) -> bool {
    if size == 0 || align == 0 {
        return false;
    }
    let remainder = ring.head % align;
    let padding = if remainder == 0 { 0 } else { align - remainder };
    ring.head
        .checked_add(padding)
        .and_then(|head| head.checked_add(size))
        .is_some_and(|end| end <= ring.size)
}

unsafe extern "system" fn vk_validation_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let d = &*data;
        let msg = if d.p_message.is_null() {
            std::borrow::Cow::Borrowed("<null>")
        } else {
            std::ffi::CStr::from_ptr(d.p_message).to_string_lossy()
        };
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            log::error!("[vk-validation] {}", msg);
        } else {
            log::warn!("[vk-validation] {}", msg);
        }
    }
    vk::FALSE
}

fn transition_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    transition_image_aspect(device, cmd, image, old, new, vk::ImageAspectFlags::COLOR);
}

fn barrier_color_attachment_after_pass(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
) {
    let (dst_stage, dst_access) = if layout == vk::ImageLayout::GENERAL {
        (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
    } else {
        (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
    };
    let barrier = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        old_layout: layout,
        new_layout: layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        },
        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_access_mask: dst_access,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn transition_image_aspect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) {
    transition_image_range(
        device,
        cmd,
        image,
        old,
        new,
        aspect,
        0,
        1,
        0,
        vk::REMAINING_ARRAY_LAYERS,
    );
}

fn transition_image_range(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
) {
    let (src_stage, dst_stage, src_access, dst_access) = match (old, new) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        | (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        ),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        (
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        ),
        (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),
        (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    };

    let barrier = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        old_layout: old,
        new_layout: new,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level,
            level_count,
            base_array_layer,
            layer_count,
        },
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn ensure_staging<'a>(
    staging: &'a mut HashMap<(u32, u32), StagingBuffer>,
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    key: (u32, u32),
    size: u64,
) -> Result<&'a StagingBuffer, String> {
    if !staging.contains_key(&key) {
        let buf_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            size,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let buffer = unsafe {
            device
                .create_buffer(&buf_info, None)
                .map_err(|e| format!("create_buffer: {:?}", e))?
        };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mt = find_memory_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| "no HOST_VISIBLE memory type".to_string())?;
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: req.size,
            memory_type_index: mt,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| format!("allocate_memory(staging): {:?}", e))?
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| format!("bind_buffer_memory: {:?}", e))?;
        }
        staging.insert(
            key,
            StagingBuffer {
                buffer,
                memory,
                size: req.size,
            },
        );
    }
    Ok(staging.get(&key).unwrap())
}

fn create_staging_owned(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<StagingBuffer, String> {
    let buf_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let buffer = unsafe {
        device
            .create_buffer(&buf_info, None)
            .map_err(|e| format!("create_buffer(readback): {:?}", e))?
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(mt) = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED,
    )
    .or_else(|| {
        find_memory_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }) else {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err("no HOST_VISIBLE memory type".to_string());
    };
    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(format!("allocate_memory(readback staging): {:?}", error));
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(format!("bind_buffer_memory(readback): {:?}", error));
    }
    Ok(StagingBuffer {
        buffer,
        memory,
        size: req.size,
    })
}

fn create_transfer_buffer_owned(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<StagingBuffer, String> {
    let buf_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size: size.max(16),
        usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: std::ptr::null(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let buffer = unsafe {
        device
            .create_buffer(&buf_info, None)
            .map_err(|e| format!("create_buffer(alias transfer): {:?}", e))?
    };
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| {
        find_memory_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    })
    .ok_or_else(|| "no memory type for alias transfer buffer".to_string())?;
    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| format!("allocate_memory(alias transfer): {:?}", e))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind_buffer_memory(alias transfer): {:?}", e))?;
    }
    Ok(StagingBuffer {
        buffer,
        memory,
        size: req.size,
    })
}

impl Drop for RendererInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        for pending in std::mem::take(&mut self.pending_computes) {
            if let Some(resources) = pending.resources {
                resources.destroy(&self.device);
            }
            unsafe {
                self.device.destroy_fence(pending.fence, None);
                self.device.free_command_buffers(self.cmd_pool, &[pending.cmd]);
            }
        }
        for (cmd, fence) in std::mem::take(&mut self.compute_slot_pool) {
            unsafe {
                self.device.destroy_fence(fence, None);
                self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            }
        }
        for (_, d) in self.dummy_images.drain() {
            unsafe {
                self.device.destroy_image_view(d.view, None);
                self.device.destroy_image(d.image, None);
                self.device.free_memory(d.memory, None);
            }
        }
        for dummy in &mut self.dummy_texel_buffers {
            if let Some(dummy) = dummy.take() {
                destroy_texel_buffer(&self.device, dummy);
            }
        }
        for (_, t) in self.tex_cache.drain() {
            unsafe {
                self.device.destroy_image_view(t.view, None);
                self.device.destroy_image(t.image, None);
                self.device.free_memory(t.memory, None);
            }
        }
        for (_, buffer) in self.texel_buffer_cache.drain() {
            destroy_texel_buffer(&self.device, buffer.resource);
        }
        for (_, t) in self.rt_reinterpret_cache.drain() {
            unsafe {
                self.device.destroy_image_view(t.view, None);
                self.device.destroy_image(t.image, None);
                self.device.free_memory(t.memory, None);
            }
        }
        if let Some(s) = self.default_sampler.take() {
            unsafe { self.device.destroy_sampler(s, None) };
        }
        for (_, s) in self.sampler_cache.drain() {
            unsafe { self.device.destroy_sampler(s, None) };
        }
        for (_, s) in self.integer_sampler_cache.drain() {
            unsafe { self.device.destroy_sampler(s, None) };
        }
        if let Some(mut backend) = self.compute_backend.take() {
            backend.destroy(&self.device);
        }
        self.pipeline_cache.clear(&self.device);
        self.shader_compiler.clear(&self.device);
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool.pool, None);
            self.descriptor_pool.pool = vk::DescriptorPool::null();
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout.layout, None);
            self.descriptor_layout.layout = vk::DescriptorSetLayout::null();
        }
        for (_, s) in self.staging.drain() {
            unsafe {
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
        }
        for slot in self.frame_slots.iter_mut() {
            slot.retired_dsets.clear();
            destroy_descriptor_pools(&self.device, &mut slot.retired_dset_pools);
            for view in slot.retired_views.drain(..) {
                unsafe {
                    self.device.destroy_image_view(view, None);
                }
            }
            for t in slot.retired_textures.drain(..) {
                unsafe {
                    self.device.destroy_image_view(t.view, None);
                    self.device.destroy_image(t.image, None);
                    self.device.free_memory(t.memory, None);
                }
            }
            for buffer in slot.retired_texel_buffers.drain(..) {
                destroy_texel_buffer(&self.device, buffer.resource);
            }
            for t in slot.retired_rt_reinterprets.drain(..) {
                unsafe {
                    self.device.destroy_image_view(t.view, None);
                    self.device.destroy_image(t.image, None);
                    self.device.free_memory(t.memory, None);
                }
            }
            unsafe {
                self.device.destroy_fence(slot.fence, None);
            }
            slot.fence = vk::Fence::null();
        }
        self.utility_slot.retired_dsets.clear();
        destroy_descriptor_pools(&self.device, &mut self.utility_slot.retired_dset_pools);
        for view in self.utility_slot.retired_views.drain(..) {
            unsafe {
                self.device.destroy_image_view(view, None);
            }
        }
        for buffer in self.utility_slot.retired_texel_buffers.drain(..) {
            destroy_texel_buffer(&self.device, buffer.resource);
        }
        for t in self.utility_slot.retired_rt_reinterprets.drain(..) {
            unsafe {
                self.device.destroy_image_view(t.view, None);
                self.device.destroy_image(t.image, None);
                self.device.free_memory(t.memory, None);
            }
        }
        unsafe {
            self.device.destroy_fence(self.utility_slot.fence, None);
        }
        self.utility_slot.fence = vk::Fence::null();
        for slot in self.clear_slots.drain(..) {
            unsafe {
                self.device.destroy_fence(slot.fence, None);
                self.device.free_command_buffers(self.cmd_pool, &[slot.cmd]);
            }
        }
        for (_, mut pending) in self.pending_readbacks.drain() {
            while let Some(pr) = pending.pop_front() {
                if let Some(slot) = self.readback_slots.get_mut(pr.slot) {
                    unsafe {
                        let _ = self
                            .device
                            .wait_for_fences(&[slot.fence], true, 2_000_000_000);
                    }
                    slot.in_flight = false;
                }
            }
        }
        for slot in self.readback_slots.drain(..) {
            unsafe {
                let _ = self
                    .device
                    .wait_for_fences(&[slot.fence], true, 2_000_000_000);
                self.device.destroy_fence(slot.fence, None);
                self.device.free_command_buffers(self.cmd_pool, &[slot.cmd]);
                if let Some(stage) = slot.stage {
                    self.device.destroy_buffer(stage.buffer, None);
                    self.device.free_memory(stage.memory, None);
                }
            }
        }
        unsafe {
            self.device.unmap_memory(self.ubo_ring.memory);
            self.device.destroy_buffer(self.ubo_ring.buffer, None);
            self.device.free_memory(self.ubo_ring.memory, None);
        }
        self.ubo_ring.mapped = std::ptr::null_mut();
        self.rt_cache.clear(&self.device);
        unsafe {
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
        let _ = &self.physical_device;
        let _ = &self.queue_family;
    }
}

unsafe impl Send for Renderer {}
unsafe impl Sync for Renderer {}

#[cfg(test)]
mod tests {
    use super::{
        align_up,
        compute_cross_access_view_components, compute_guest_sampler_format_features,
        compute_sampled_image_alias_extent_matches, compute_texel_buffer_format_features,
        depth_stencil_component_mapping, depth_stencil_sample_aspect,
        descriptor_slot_uses_arrayed_2d, dummy_image_format_and_aspect, g24r8_scalar_upload,
        format_graphics_texture_bind_trace, graphics_cbuf_allocation_size,
        parse_bind_trace_filter, ring_request_upper_bound,
        route_texture_key_to_shader_image_kind, sampled_rt_key_from_lists,
        texel_buffer_format,
        texture_image_format_for_tic, texture_key_has_special_view, texture_level_upload,
        texture_numeric_cache_key, texture_numeric_type_matches_format, texture_upload_data,
        texture_requires_integer_sampler, texture_view_layer_range, texture_view_swizzle,
        tic_format_prefers_depth_alias, tic_is_cube, tic_layer_count, tic_read_size,
        tic_requires_dedicated_sampled_view, tic_view_base_layer, tic_view_layer_count,
        typed_sampled_image_infos, typed_texel_buffer_views, vk_integer_border_color,
        DummyImageKind, GraphicsTextureBindOutcome, GraphicsTextureBindTraceRecord,
        GraphicsTextureTraceResource, RtAlias, TexCacheKey,
    };
    use ash::vk;

    #[test]
    fn graphics_cbuf_storage_range_and_alignment_follow_device_limits() {
        assert_eq!(
            graphics_cbuf_allocation_size(nexium_spirv::GFX_CBUF_MIN_SIZE as usize, 256, 0x4000)
                .unwrap(),
            (nexium_spirv::GFX_CBUF_MIN_SIZE as u64, 512)
        );
        assert_eq!(
            graphics_cbuf_allocation_size(0x4000, 256, 0x4000).unwrap(),
            (0x4000, 0x4000)
        );
        let error = graphics_cbuf_allocation_size(0x4001, 256, 0x4000).unwrap_err();
        assert!(error.contains("maxStorageBufferRange"), "{error}");
    }

    #[test]
    fn graphics_ring_alignment_handles_non_power_of_two_vertex_strides() {
        assert_eq!(align_up(0, 20), 0);
        assert_eq!(align_up(17, 20), 20);
        assert_eq!(align_up(20, 20), 20);
        assert_eq!(align_up(21, 20), 40);
        assert_eq!(ring_request_upper_bound(21, 20), 59);
    }

    #[test]
    fn binding_trace_filter_accepts_stable_fragment_hashes() {
        let filter = parse_bind_trace_filter("hash:1234abcd,0x4000,all,bad");
        assert!(filter.all);
        assert_eq!(filter.hashes, [0x1234_abcd]);
        assert_eq!(filter.addresses, [0x4000, 0xbad]);
    }

    #[test]
    fn graphics_texture_binding_trace_format_has_complete_stable_fields() {
        let record = GraphicsTextureBindTraceRecord {
            fs_gpu_va: 0x2200,
            fs_hash: 0x0123_4567_89ab_cdef,
            shader_id: Some(17),
            descriptor_slot: 4,
            tic_id: Some(9),
            tic_address: Some(0x4120),
            resource: GraphicsTextureTraceResource {
                resource_va: Some(0x1234_5000),
                raw_texture_type: Some(3),
                format: Some("R16".to_string()),
                component_types: Some("[Uint, Uint, Uint, Uint]".to_string()),
                width: Some(128),
                height: Some(64),
                depth: Some(6),
                computed_layers: Some(210),
                base_layer: Some(12),
                min_mip: Some(1),
                max_mip: Some(7),
            },
            numeric_family: "uint",
            image_kind: crate::texture_manifest::GraphicsTextureImageKind::Buffer,
            selected_binding: nexium_spirv::GFX_BINDING_UINT_TEXEL_BUFFER,
            outcome: GraphicsTextureBindOutcome::TexelBuffer,
            source: "texel-buffer".to_string(),
            selected_view: "0xfeed".to_string(),
            selected_format: Some("R16_UINT".to_string()),
            reason: Some("preserved integer bits".to_string()),
        };

        assert_eq!(
            format_graphics_texture_bind_trace(&record),
            "[bind-trace] fs_hash=0123456789abcdef fs_va=0x2200 shader_id=17 slot=4 tic_id=9 tic_addr=0x4120 resource_va=0x12345000 raw_type=3 format=\"R16\" components=\"[Uint, Uint, Uint, Uint]\" dimensions=128x64x6 layers=210 base_layer=12 mips=1..7 numeric=uint image_kind=Buffer binding=19 outcome=texel-buffer source=\"texel-buffer\" view=0xfeed view_format=\"R16_UINT\" reason=\"preserved integer bits\""
        );
    }

    #[test]
    fn graphics_texture_binding_trace_names_every_outcome_and_missing_field() {
        let mut record = GraphicsTextureBindTraceRecord {
            fs_gpu_va: 0,
            fs_hash: 1,
            shader_id: None,
            descriptor_slot: 0,
            tic_id: None,
            tic_address: None,
            resource: GraphicsTextureTraceResource::default(),
            numeric_family: "float",
            image_kind: crate::texture_manifest::GraphicsTextureImageKind::D2,
            selected_binding: nexium_spirv::GFX_BINDING_FLOAT_2D,
            outcome: GraphicsTextureBindOutcome::Success,
            source: "texture-2d".to_string(),
            selected_view: "0x0".to_string(),
            selected_format: None,
            reason: None,
        };
        for (outcome, expected) in [
            (GraphicsTextureBindOutcome::Success, "outcome=success"),
            (GraphicsTextureBindOutcome::Dummy, "outcome=dummy"),
            (GraphicsTextureBindOutcome::RtAlias, "outcome=rt-alias"),
            (
                GraphicsTextureBindOutcome::TexelBuffer,
                "outcome=texel-buffer",
            ),
            (
                GraphicsTextureBindOutcome::Rejection,
                "outcome=rejection",
            ),
        ] {
            record.outcome = outcome;
            assert!(format_graphics_texture_bind_trace(&record).contains(expected));
        }
        let formatted = format_graphics_texture_bind_trace(&record);
        assert!(formatted.contains("shader_id=-"));
        assert!(formatted.contains("tic_addr=-"));
        assert!(formatted.contains("dimensions=-"));
        assert!(formatted.contains("reason=\"-\""));
    }

    #[test]
    fn compute_guest_format_features_follow_actual_sampler_mode() {
        use crate::texture::{SamplerReduction, TexFilter, TscEntry};

        let mut tsc = TscEntry::parse(&[0; 32]).unwrap();
        let base = vk::FormatFeatureFlags2::SAMPLED_IMAGE | vk::FormatFeatureFlags2::TRANSFER_DST;
        assert_eq!(
            compute_guest_sampler_format_features(&tsc, false, false, false, false),
            base
        );

        tsc.mag_filter = TexFilter::Linear;
        assert!(
            compute_guest_sampler_format_features(&tsc, false, false, false, false)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
        );
        tsc.mag_filter = TexFilter::Nearest;
        assert!(
            compute_guest_sampler_format_features(&tsc, false, false, false, true)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
        );
        tsc.max_anisotropy = 1;
        assert!(
            compute_guest_sampler_format_features(&tsc, false, false, true, false)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
        );
        tsc.max_anisotropy = 0;

        tsc.reduction = SamplerReduction::Min;
        assert!(
            !compute_guest_sampler_format_features(&tsc, false, false, false, false)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX)
        );
        assert!(
            compute_guest_sampler_format_features(&tsc, false, true, false, false)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX)
        );

        tsc.depth_compare_enabled = true;
        assert!(
            compute_guest_sampler_format_features(&tsc, false, true, false, false)
                .contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_DEPTH_COMPARISON)
        );

        assert_eq!(
            compute_guest_sampler_format_features(&tsc, true, true, true, true),
            base
        );
    }

    #[test]
    fn compute_atomic_texel_buffers_require_atomic_format_support() {
        assert_eq!(
            compute_texel_buffer_format_features(false),
            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER
        );
        assert_eq!(
            compute_texel_buffer_format_features(true),
            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER
                | vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER_ATOMIC
        );
    }

    #[test]
    fn compute_live_sampled_alias_requires_exact_descriptor_extent() {
        use crate::rt_cache::RtKey;
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};

        let tic = TicEntry {
            format: TicFormat::R16G16B16A16,
            component_types: [ComponentType::Float; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x5280_7000_0,
            width: 1067,
            height: 600,
            block_width_log2: 0,
            block_height_log2: 4,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let alias = |width, height| RtAlias {
            key: RtKey::new(68, width, height, tic.gpu_va),
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            format: vk::Format::R16G16B16A16_SFLOAT,
            aspects: vk::ImageAspectFlags::COLOR,
            depth: false,
        };

        assert!(compute_sampled_image_alias_extent_matches(
            alias(1067, 600),
            tic,
            false,
            1
        ));
        assert!(!compute_sampled_image_alias_extent_matches(
            alias(1072, 600),
            tic,
            false,
            1
        ));
    }

    #[test]
    fn compute_cross_access_alias_requires_exact_16x16x16_view_and_format() {
        use crate::compute::{ComputeSampleType, ComputeStorageFormat, ComputeStorageImage};
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};

        let tic = TicEntry {
            format: TicFormat::R32,
            component_types: [ComponentType::Float; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x55dc_2180_0,
            width: 16,
            height: 16,
            block_width_log2: 0,
            block_height_log2: 1,
            block_depth_log2: 1,
            tile_width_spacing: 0,
            is_block_linear: true,
            texture_type: 2,
            depth: 16,
            base_layer: 0,
            normalized_coords: false,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let output = ComputeStorageImage {
            binding: 3,
            width: 16,
            height: 16,
            depth: 16,
            is_3d: true,
            format: ComputeStorageFormat::R32Float,
            initial_bytes: Some(vec![0; 16 * 16 * 16 * 4]),
        };

        assert!(
            compute_cross_access_view_components(1, &tic, ComputeSampleType::Float, &output)
                .is_ok()
        );
        assert!(compute_cross_access_view_components(
            1,
            &tic,
            ComputeSampleType::Float,
            &ComputeStorageImage {
                width: 8,
                ..output.clone()
            }
        )
        .is_err());
        assert!(compute_cross_access_view_components(
            1,
            &tic,
            ComputeSampleType::Float,
            &ComputeStorageImage {
                format: ComputeStorageFormat::R32Uint,
                ..output
            }
        )
        .is_err());
    }

    #[test]
    fn compute_texel_formats_keep_exact_view_types_and_capabilities() {
        use crate::compute::ComputeTexelFormat;

        assert_eq!(
            ComputeTexelFormat::R32Float.vk_format(),
            vk::Format::R32_SFLOAT
        );
        assert_eq!(
            ComputeTexelFormat::R32Uint.vk_format(),
            vk::Format::R32_UINT
        );
        assert_eq!(
            ComputeTexelFormat::R32Sint.vk_format(),
            vk::Format::R32_SINT
        );
        assert_eq!(
            ComputeTexelFormat::R16Uint.vk_format(),
            vk::Format::R16_UINT
        );
        assert_eq!(ComputeTexelFormat::R16Uint.bytes_per_element(), 2);
        assert!(ComputeTexelFormat::R16Uint.requires_storage_image_extended_formats());
        assert_eq!(
            ComputeTexelFormat::R16Uint.spirv_format(),
            nexium_spirv::ComputeTexelFormat::R16Uint
        );
        assert_eq!(
            ComputeTexelFormat::Rgba32Float.vk_format(),
            vk::Format::R32G32B32A32_SFLOAT
        );
        assert_eq!(ComputeTexelFormat::Rgba32Float.bytes_per_element(), 16);
        assert_eq!(
            ComputeTexelFormat::Rgba32Float.spirv_format(),
            nexium_spirv::ComputeTexelFormat::Rgba32Float
        );
        assert!(!ComputeTexelFormat::R32Float.supports_storage_atomics());
        assert!(ComputeTexelFormat::R32Uint.supports_storage_atomics());
        assert!(!ComputeTexelFormat::R32Sint.supports_storage_atomics());
        assert!(!ComputeTexelFormat::R16Uint.supports_storage_atomics());
        assert!(!ComputeTexelFormat::Rgba32Float.supports_storage_atomics());
    }

    #[test]
    fn pps_buffer_tics_select_exact_uint_views_and_spans() {
        use crate::texture::{ComponentType, TicEntry};

        let slot8 = [
            0x1b, 0x92, 0x14, 0x60, 0x00, 0x00, 0x77, 0x03, 0x04, 0x00, 0x00, 0x00, 0x0b, 0x00,
            0x00, 0x00, 0xff, 0xb7, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let slot9 = [
            0x0f, 0x92, 0x14, 0x60, 0x00, 0x00, 0x6d, 0x05, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x7f, 0xbb, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let mut r16 = TicEntry::parse(&slot8).unwrap();
        let r32 = TicEntry::parse(&slot9).unwrap();

        assert_eq!(
            texel_buffer_format(&r16, nexium_spirv::TextureNumericType::Uint),
            Some((vk::Format::R16_UINT, 2))
        );
        assert_eq!(r16.width as usize * 2, 0x177000);
        assert_eq!(
            texel_buffer_format(&r32, nexium_spirv::TextureNumericType::Uint),
            Some((vk::Format::R32_UINT, 4))
        );
        assert_eq!(r32.width as usize * 4, 0x2ee00);

        r16.component_types[0] = ComponentType::Unorm;
        assert_eq!(
            texel_buffer_format(&r16, nexium_spirv::TextureNumericType::Uint),
            None
        );
    }

    #[test]
    fn pps_float_buffer_tics_select_native_views() {
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
        use nexium_spirv::TextureNumericType;

        let mut tic = TicEntry {
            format: TicFormat::R32G32B32A32,
            component_types: [ComponentType::Float; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 1,
            width: 25_966,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: false,
            texture_type: 6,
            depth: 1,
            base_layer: 0,
            normalized_coords: false,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Float),
            Some((vk::Format::R32G32B32A32_SFLOAT, 16))
        );

        tic.format = TicFormat::A8B8G8R8;
        tic.component_types = [ComponentType::Snorm; 4];
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Float),
            Some((vk::Format::A8B8G8R8_SNORM_PACK32, 4))
        );
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Uint),
            Some((vk::Format::A8B8G8R8_UINT_PACK32, 4))
        );
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Sint),
            Some((vk::Format::A8B8G8R8_SINT_PACK32, 4))
        );
        tic.component_types = [ComponentType::Unorm; 4];
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Float),
            Some((vk::Format::A8B8G8R8_UNORM_PACK32, 4))
        );

        tic.format = TicFormat::R16G16;
        tic.component_types = [ComponentType::Float; 4];
        tic.swizzle = [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::Zero,
            SwizzleSource::One,
        ];
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Float),
            Some((vk::Format::R16G16_SFLOAT, 4))
        );
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Uint),
            Some((vk::Format::R16G16_UINT, 4))
        );
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Sint),
            Some((vk::Format::R16G16_SINT, 4))
        );

        tic.format = TicFormat::R32G32;
        assert_eq!(tic.format.src_bpp(), 8);
        assert_eq!(
            texel_buffer_format(&tic, TextureNumericType::Float),
            Some((vk::Format::R32G32_SFLOAT, 8))
        );
    }

    #[test]
    fn mixed_fragment_uint_and_vertex_float_texel_slots_keep_manifest_types() {
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
        use crate::texture_manifest::{
            normalize_texture_numeric_manifest, texture_numeric_type_for_slot,
            TextureNumericBinding,
        };
        use nexium_spirv::TextureNumericType::{Float, Uint};

        let manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(10, 10, Uint),
            TextureNumericBinding::new(11, 11, Uint),
            TextureNumericBinding::new(16, 16, Uint),
        ])
        .unwrap();

        assert_eq!(texture_numeric_type_for_slot(&manifest, 10), Uint);
        assert_eq!(texture_numeric_type_for_slot(&manifest, 11), Uint);
        for slot in 12..16 {
            assert_eq!(texture_numeric_type_for_slot(&manifest, slot), Float);
        }
        assert_eq!(texture_numeric_type_for_slot(&manifest, 16), Uint);

        let tic = TicEntry {
            format: TicFormat::A8B8G8R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x40001ce00,
            width: 4,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: false,
            texture_type: 6,
            depth: 1,
            base_layer: 0,
            normalized_coords: false,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let numeric_type = texture_numeric_type_for_slot(&manifest, 12);
        assert_eq!(numeric_type, Float);
        assert_eq!(
            texel_buffer_format(&tic, numeric_type),
            Some((vk::Format::A8B8G8R8_UNORM_PACK32, 4))
        );
    }

    #[test]
    fn typed_graphics_descriptors_route_each_slot_to_one_numeric_family() {
        use crate::texture_manifest::{normalize_texture_numeric_manifest, TextureNumericBinding};
        use ash::vk::Handle;
        use nexium_spirv::TextureNumericType::{Sint, Uint};

        let manifest = normalize_texture_numeric_manifest(vec![
            TextureNumericBinding::new(0x21, 1, Uint),
            TextureNumericBinding::new(0x22, 2, Sint),
        ])
        .unwrap();
        let selected_images = (0..crate::descriptor::MAX_TEXTURE_DESCRIPTORS)
            .map(|slot| vk::ImageView::from_raw(0x100 + u64::from(slot)))
            .collect::<Vec<_>>();
        let selected_layouts =
            vec![vk::ImageLayout::GENERAL; crate::descriptor::MAX_TEXTURE_DESCRIPTORS as usize];
        let image_dummies_by_slot = (0..crate::descriptor::MAX_TEXTURE_DESCRIPTORS)
            .map(|slot| {
                [
                    vk::ImageView::from_raw(0x200 + u64::from(slot) * 3),
                    vk::ImageView::from_raw(0x201 + u64::from(slot) * 3),
                    vk::ImageView::from_raw(0x202 + u64::from(slot) * 3),
                ]
            })
            .collect::<Vec<_>>();
        let image_infos = typed_sampled_image_infos(
            &selected_images,
            Some(&selected_layouts),
            &manifest,
            &image_dummies_by_slot,
        );

        assert_eq!(image_infos[0][0].image_view, selected_images[0]);
        assert_eq!(image_infos[0][0].image_layout, vk::ImageLayout::GENERAL);
        assert_eq!(image_infos[1][1].image_view, selected_images[1]);
        assert_eq!(image_infos[1][1].image_layout, vk::ImageLayout::GENERAL);
        assert_eq!(image_infos[2][2].image_view, selected_images[2]);
        assert_eq!(image_infos[2][2].image_layout, vk::ImageLayout::GENERAL);
        assert_eq!(image_infos[0][1].image_view, image_dummies_by_slot[1][0]);
        assert_eq!(image_infos[1][2].image_view, image_dummies_by_slot[2][1]);
        assert_eq!(image_infos[2][0].image_view, image_dummies_by_slot[0][2]);
        assert_eq!(
            image_infos[0][1].image_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );

        let selected_texels = (0..crate::descriptor::MAX_TEXTURE_DESCRIPTORS)
            .map(|slot| vk::BufferView::from_raw(0x300 + u64::from(slot)))
            .collect::<Vec<_>>();
        let texel_dummies = [
            vk::BufferView::from_raw(0x401),
            vk::BufferView::from_raw(0x402),
            vk::BufferView::from_raw(0x403),
        ];
        let texel_views = typed_texel_buffer_views(&selected_texels, &manifest, texel_dummies);
        assert_eq!(texel_views[0][0], selected_texels[0]);
        assert_eq!(texel_views[1][1], selected_texels[1]);
        assert_eq!(texel_views[2][2], selected_texels[2]);
        assert_eq!(texel_views[0][1], texel_dummies[0]);
        assert_eq!(texel_views[1][2], texel_dummies[1]);
        assert_eq!(texel_views[2][0], texel_dummies[2]);
    }

    #[test]
    fn graphics_2d_view_shape_is_selected_per_fragment_or_vertex_slot() {
        assert!(!descriptor_slot_uses_arrayed_2d(0, 2, 2, false, true));
        assert!(!descriptor_slot_uses_arrayed_2d(1, 2, 2, false, true));
        assert!(descriptor_slot_uses_arrayed_2d(2, 2, 2, false, true));
        assert!(descriptor_slot_uses_arrayed_2d(3, 2, 2, false, true));
        assert!(!descriptor_slot_uses_arrayed_2d(4, 2, 2, false, true));

        assert!(descriptor_slot_uses_arrayed_2d(0, 2, 2, true, false));
        assert!(!descriptor_slot_uses_arrayed_2d(2, 2, 2, true, false));
    }

    #[test]
    fn depth_compare_dummy_kinds_use_d32_depth_views() {
        use nexium_spirv::TextureNumericType::{Float, Uint};

        for kind in [
            DummyImageKind::DepthD2,
            DummyImageKind::DepthD2Array,
            DummyImageKind::DepthCube,
            DummyImageKind::DepthCubeArray,
        ] {
            assert_eq!(
                dummy_image_format_and_aspect(Float, kind).unwrap(),
                (vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH)
            );
            assert!(dummy_image_format_and_aspect(Uint, kind).is_err());
        }
        assert_eq!(
            dummy_image_format_and_aspect(Float, DummyImageKind::D2).unwrap(),
            (vk::Format::R8G8B8A8_UNORM, vk::ImageAspectFlags::COLOR)
        );
    }

    #[test]
    fn r16_uint_slot_keeps_native_format_and_distinct_cache_family() {
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
        use crate::texture_manifest::{
            normalize_texture_numeric_manifest, texture_numeric_type_for_slot,
            TextureNumericBinding,
        };
        use nexium_spirv::TextureNumericType::{Float, Uint};

        let tic = TicEntry {
            format: TicFormat::R16,
            component_types: [ComponentType::Uint; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::Zero,
                SwizzleSource::Zero,
                SwizzleSource::One,
            ],
            gpu_va: 0x6000_0000,
            width: 64,
            height: 64,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: false,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: false,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        let manifest =
            normalize_texture_numeric_manifest(vec![TextureNumericBinding::new(0x44, 7, Uint)])
                .unwrap();
        let numeric_type = texture_numeric_type_for_slot(&manifest, 7);

        assert_eq!(numeric_type, Uint);
        assert_eq!(
            texture_image_format_for_tic(&tic, numeric_type).unwrap(),
            vk::Format::R16_UINT
        );
        assert!(texture_image_format_for_tic(&tic, Float).is_err());
        assert_ne!(
            texture_numeric_cache_key(Float),
            texture_numeric_cache_key(numeric_type)
        );
    }

    #[test]
    fn positional_rt_slots_do_not_shift_compressed_aliases() {
        let depth = crate::rt_cache::RtKey::new(317, 1600, 900, 0x5a0ab0000);
        let color = crate::rt_cache::RtKey::new(329, 1600, 900, 0x5a9390000);
        assert_eq!(
            sampled_rt_key_from_lists(&[None, Some(color)], &[color], Some(color), 0),
            None
        );
        assert_eq!(
            sampled_rt_key_from_lists(&[Some(depth), Some(color)], &[depth, color], None, 1),
            Some(color)
        );
        assert_eq!(
            sampled_rt_key_from_lists(&[], &[depth, color], None, 1),
            Some(color)
        );
    }

    #[test]
    fn depth_tic_formats_prefer_native_depth_aliases() {
        use crate::texture::TicFormat;

        for format in [
            TicFormat::G24R8,
            TicFormat::Z24S8,
            TicFormat::X8Z24,
            TicFormat::S8Z24,
            TicFormat::Z32,
        ] {
            assert!(tic_format_prefers_depth_alias(format));
        }
        assert!(!tic_format_prefers_depth_alias(TicFormat::A8B8G8R8));
    }

    #[test]
    fn cube_tic_uses_six_face_view_and_distinct_cache_identity() {
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
        use std::collections::HashSet;

        let tic = TicEntry {
            format: TicFormat::R8G8B8A8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x1000,
            width: 64,
            height: 64,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: false,
            texture_type: 3,
            depth: 1,
            base_layer: 4,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        assert!(tic_is_cube(&tic));
        assert_eq!(tic_layer_count(&tic), 6);
        assert_eq!(tic_view_base_layer(&tic), 0);
        assert_eq!(tic_view_layer_count(&tic), 6);
        let pitch_size = tic.format.linear_size(tic.width, tic.height);
        assert_eq!(
            tic_read_size(&tic, pitch_size, tic_layer_count(&tic)),
            pitch_size * 6
        );

        let plain = TexCacheKey {
            gpu_va: tic.gpu_va,
            width: tic.width,
            height: tic.height,
            layers: 6,
            base_layer: 0,
            view_layers: 6,
            mip_levels: 1,
            base_mip: 0,
            view_mips: 1,
            arrayed: false,
            cube: false,
            cube_array: false,
            volume: false,
            format: tic.format,
            component_types: tic.component_types,
            swizzle: tic.swizzle,
            is_srgb: tic.is_srgb,
            is_block_linear: tic.is_block_linear,
            block_width_log2: tic.block_width_log2,
            block_height_log2: tic.block_height_log2,
            block_depth_log2: tic.block_depth_log2,
            tile_width_spacing: tic.tile_width_spacing,
            numeric_type: 0,
        };
        let cube = TexCacheKey {
            cube: true,
            ..plain
        };
        assert_eq!(HashSet::from([plain, cube]).len(), 2);

        let cube_array = route_texture_key_to_shader_image_kind(
            cube,
            &tic,
            crate::texture_manifest::GraphicsTextureImageKind::CubeArray,
        )
        .unwrap();
        assert!(!cube_array.cube);
        assert!(cube_array.cube_array);
        assert_eq!(cube_array.layers, 6);
        assert_eq!(cube_array.base_layer, 0);
        assert_eq!(cube_array.view_layers, 6);
        assert_eq!(HashSet::from([cube, cube_array]).len(), 2);
        assert_eq!(
            crate::texture_manifest::GraphicsTextureImageKind::CubeArray.spirv_kind(),
            nexium_spirv::GraphicsImageKind::CubeArray
        );
        assert_eq!(nexium_spirv::GFX_BINDING_FLOAT_CUBE_ARRAY, 13);

        let float_components = TexCacheKey {
            component_types: [ComponentType::Float; 4],
            ..plain
        };
        let different_tiling = TexCacheKey {
            is_block_linear: true,
            block_height_log2: 4,
            ..plain
        };
        assert_eq!(
            HashSet::from([plain, float_components, different_tiling]).len(),
            3
        );

        let arrayed = TexCacheKey {
            arrayed: true,
            ..plain
        };
        assert!(texture_key_has_special_view(arrayed));
        let mut array_tic = tic;
        array_tic.texture_type = 5;
        assert!(tic_requires_dedicated_sampled_view(&array_tic));
    }

    #[test]
    fn cube_array_upload_preserves_210_layers_across_eight_mips() {
        use crate::texture::{
            block_linear_mip_layout, swizzle_block_linear_strided, ComponentType, SwizzleSource,
            TicEntry, TicFormat,
        };

        let tic = TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::R,
                SwizzleSource::R,
                SwizzleSource::One,
            ],
            gpu_va: 0x7fff_f000_0000,
            width: 32,
            height: 32,
            block_width_log2: 0,
            block_height_log2: 2,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: true,
            texture_type: 8,
            depth: 35,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 7,
            res_min_mip_level: 0,
            res_max_mip_level: 7,
        };
        crate::pitch_oracle::clear_pitch_range(tic.gpu_va, 0x10_0000);
        let layers = tic_layer_count(&tic);
        assert_eq!(layers, 210);
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!(layout.levels.len(), 8);

        let mut guest = vec![0u8; layout.guest_size_bytes(layers)];
        for layer in 0..layers as usize {
            for level in &layout.levels {
                let marker = (layer as u8)
                    .wrapping_mul(17)
                    .wrapping_add((level.level as u8).wrapping_mul(29));
                let linear = vec![marker; level.linear_size];
                let swizzled = swizzle_block_linear_strided(
                    &linear,
                    level.storage_width,
                    level.storage_height,
                    tic.format.src_bpp(),
                    level.block_height_log2,
                    level.stride_alignment_log2,
                );
                assert!(swizzled.len() <= level.guest_size);
                let start = layer * layout.layer_stride + level.guest_offset;
                guest[start..start + swizzled.len()].copy_from_slice(&swizzled);
            }
        }

        let upload = texture_upload_data(
            &guest,
            &tic,
            layers,
            layout.layer_stride,
            false,
            vk::Format::R8_UNORM,
        )
        .unwrap();
        assert_eq!(upload.copies.len(), 8);

        let mut expected_offset = 0usize;
        for (copy, level) in upload.copies.iter().zip(&layout.levels) {
            assert_eq!(copy.buffer_offset as usize, expected_offset);
            assert_eq!(copy.mip_level, level.level);
            assert_eq!((copy.width, copy.height), (level.width, level.height));
            for layer in 0..layers as usize {
                let marker = (layer as u8)
                    .wrapping_mul(17)
                    .wrapping_add((level.level as u8).wrapping_mul(29));
                let start = expected_offset + layer * level.linear_size;
                let end = start + level.linear_size;
                assert!(upload.bytes[start..end].iter().all(|byte| *byte == marker));
            }
            expected_offset += level.linear_size * layers as usize;
        }
        assert_eq!(upload.bytes.len(), expected_offset);
    }

    #[test]
    fn texture_view_layers_reject_non_normalized_or_invalid_cube_storage() {
        assert_eq!(
            texture_view_layer_range(6, 0, 6, false, true, false, false).unwrap(),
            (0, 6)
        );
        assert!(texture_view_layer_range(6, 2, 6, false, true, false, false).is_err());
        assert!(texture_view_layer_range(10, 0, 6, false, true, false, false).is_err());
        assert!(texture_view_layer_range(7, 0, 7, false, false, true, false).is_err());
        assert_eq!(
            texture_view_layer_range(12, 0, 12, false, false, true, false).unwrap(),
            (0, 12)
        );
    }

    #[test]
    fn integer_textures_always_use_integer_safe_samplers() {
        use nexium_spirv::TextureNumericType;

        assert!(!texture_requires_integer_sampler(
            TextureNumericType::Float,
            false
        ));
        assert!(texture_requires_integer_sampler(
            TextureNumericType::Float,
            true
        ));
        assert!(texture_requires_integer_sampler(
            TextureNumericType::Uint,
            false
        ));
        assert!(texture_requires_integer_sampler(
            TextureNumericType::Sint,
            false
        ));
    }

    #[test]
    fn r16g16b16a16_format_selection_is_typed_and_fail_closed() {
        use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
        use nexium_spirv::TextureNumericType;

        let mut tic = TicEntry {
            format: TicFormat::R16G16B16A16,
            component_types: [ComponentType::Float; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 1,
            width: 1,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::R16G16B16A16_SFLOAT
        );
        tic.component_types = [ComponentType::Unorm; 4];
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::R16G16B16A16_UNORM
        );
        tic.component_types[3] = ComponentType::Float;
        assert!(texture_image_format_for_tic(&tic, TextureNumericType::Float).is_err());

        tic.format = TicFormat::R16;
        tic.component_types = [ComponentType::Float; 4];
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::R16_SFLOAT
        );
        let half = [0x00, 0x3c, 0x00, 0x38];
        assert_eq!(
            texture_level_upload(&half, TicFormat::R16, 2, 1, 1, vk::Format::R16_SFLOAT),
            half
        );

        tic.format = TicFormat::B10G11R11;
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::B10G11R11_UFLOAT_PACK32
        );
        let packed = 0x8abc_def0u32.to_le_bytes();
        assert_eq!(
            texture_level_upload(
                &packed,
                TicFormat::B10G11R11,
                1,
                1,
                1,
                vk::Format::B10G11R11_UFLOAT_PACK32,
            ),
            packed
        );

        tic.component_types = [ComponentType::Unorm; 4];
        tic.is_srgb = true;
        tic.format = TicFormat::R8;
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::R8_SRGB
        );
        assert_eq!(
            texture_level_upload(&[0x80], TicFormat::R8, 1, 1, 1, vk::Format::R8_SRGB),
            [0x80]
        );
        tic.format = TicFormat::R8G8;
        assert_eq!(
            texture_image_format_for_tic(&tic, TextureNumericType::Float).unwrap(),
            vk::Format::R8G8_SRGB
        );
        tic.format = TicFormat::R16;
        assert!(texture_image_format_for_tic(&tic, TextureNumericType::Float).is_err());
    }

    #[test]
    fn g24r8_scalar_upload_preserves_stencil_and_depth() {
        let packed = 0x1234_56abu32.to_le_bytes();
        let uint = g24r8_scalar_upload(&packed, nexium_spirv::TextureNumericType::Uint);
        assert_eq!(u32::from_le_bytes(uint.try_into().unwrap()), 0xab);

        let float = g24r8_scalar_upload(&packed, nexium_spirv::TextureNumericType::Float);
        let depth = f32::from_le_bytes(float.try_into().unwrap());
        let expected = 0x12_3456 as f32 / 0x00ff_ffff as f32;
        assert!((depth - expected).abs() <= f32::EPSILON);
    }

    #[test]
    fn g24r8_scalar_view_maps_logical_green_to_red() {
        use crate::texture::{SwizzleSource, TicFormat};

        assert_eq!(
            texture_view_swizzle(
                TicFormat::G24R8,
                nexium_spirv::TextureNumericType::Float,
                [
                    SwizzleSource::G,
                    SwizzleSource::R,
                    SwizzleSource::One,
                    SwizzleSource::Zero,
                ],
            ),
            [
                SwizzleSource::R,
                SwizzleSource::R,
                SwizzleSource::One,
                SwizzleSource::Zero,
            ]
        );

        let stencil_swizzle = [
            SwizzleSource::G,
            SwizzleSource::R,
            SwizzleSource::One,
            SwizzleSource::Zero,
        ];
        assert_eq!(
            texture_view_swizzle(
                TicFormat::G24R8,
                nexium_spirv::TextureNumericType::Uint,
                stencil_swizzle,
            ),
            stencil_swizzle
        );
    }

    #[test]
    fn typed_texture_bindings_reject_numeric_mismatches() {
        use nexium_spirv::TextureNumericType;

        assert!(texture_numeric_type_matches_format(
            TextureNumericType::Uint,
            vk::Format::R32_UINT,
        ));
        assert!(!texture_numeric_type_matches_format(
            TextureNumericType::Uint,
            vk::Format::R8G8B8A8_UNORM,
        ));
        assert!(!texture_numeric_type_matches_format(
            TextureNumericType::Uint,
            vk::Format::D32_SFLOAT,
        ));
        assert!(texture_numeric_type_matches_format(
            TextureNumericType::Float,
            vk::Format::D32_SFLOAT,
        ));
    }

    #[test]
    fn g24r8_rt_alias_selects_only_the_stencil_aspect() {
        use crate::texture::{SwizzleSource, TicFormat};

        let all_r = [SwizzleSource::R; 4];
        let combined = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        assert_eq!(
            depth_stencil_sample_aspect(TicFormat::G24R8, all_r, combined),
            Some(vk::ImageAspectFlags::STENCIL),
        );
        let mapping = depth_stencil_component_mapping([
            SwizzleSource::G,
            SwizzleSource::R,
            SwizzleSource::One,
            SwizzleSource::Zero,
        ]);
        assert_eq!(mapping.r, vk::ComponentSwizzle::R);
        assert_eq!(mapping.g, vk::ComponentSwizzle::R);
        assert_eq!(mapping.b, vk::ComponentSwizzle::ONE);
        assert_eq!(mapping.a, vk::ComponentSwizzle::ZERO);
        assert_eq!(
            depth_stencil_sample_aspect(TicFormat::G24R8, all_r, vk::ImageAspectFlags::DEPTH,),
            None,
        );
    }

    #[test]
    fn integer_alias_border_colors_never_use_float_variants() {
        assert_eq!(
            vk_integer_border_color([0; 4]),
            vk::BorderColor::INT_TRANSPARENT_BLACK,
        );
        assert_eq!(
            vk_integer_border_color([1; 4]),
            vk::BorderColor::INT_OPAQUE_WHITE,
        );
        assert_eq!(
            vk_integer_border_color([2, 3, 4, 5]),
            vk::BorderColor::INT_OPAQUE_BLACK,
        );
    }
}

use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::rt_cache::{find_memory_type, RtKey};
use crate::texture::{TicEntry, TscEntry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeSampleType {
    Float,
    Uint,
    Sint,
}

impl ComputeSampleType {
    pub(crate) fn spirv_type(self) -> nexium_spirv::TextureNumericType {
        match self {
            Self::Float => nexium_spirv::TextureNumericType::Float,
            Self::Uint => nexium_spirv::TextureNumericType::Uint,
            Self::Sint => nexium_spirv::TextureNumericType::Sint,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeStorageFormat {
    R8Unorm,
    R16Float,
    B10G11R11Float,
    Rgba16Float,
    Abgr8Unorm,
    Abgr8Uint,
    Rgba8Uint,
    R32Uint,
    R32Float,
}

impl ComputeStorageFormat {
    pub fn from_tic(tic: &TicEntry) -> Option<Self> {
        use crate::texture::{ComponentType, TicFormat};

        let component = tic.component_types[0];
        if !tic
            .component_types
            .iter()
            .all(|candidate| *candidate == component)
        {
            return None;
        }
        match (tic.format, component) {
            (TicFormat::R8, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
                Some(Self::R8Unorm)
            }
            (TicFormat::R16, ComponentType::Float) => Some(Self::R16Float),
            (TicFormat::B10G11R11, ComponentType::Float) => Some(Self::B10G11R11Float),
            (TicFormat::R16G16B16A16, ComponentType::Float) => Some(Self::Rgba16Float),
            (TicFormat::A8B8G8R8, ComponentType::Unorm | ComponentType::UnormForceFp16) => {
                Some(Self::Abgr8Unorm)
            }
            (TicFormat::A8B8G8R8, ComponentType::Uint) => Some(Self::Abgr8Uint),
            (TicFormat::R8G8B8A8, ComponentType::Uint) => Some(Self::Rgba8Uint),
            (TicFormat::R32, ComponentType::Uint) => Some(Self::R32Uint),
            (TicFormat::R32, ComponentType::Float) => Some(Self::R32Float),
            _ => None,
        }
    }

    pub(crate) fn vk_format(self) -> vk::Format {
        match self {
            Self::R8Unorm => vk::Format::R8_UNORM,
            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::B10G11R11Float => vk::Format::B10G11R11_UFLOAT_PACK32,
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::Abgr8Unorm => vk::Format::A8B8G8R8_UNORM_PACK32,
            Self::Abgr8Uint => vk::Format::A8B8G8R8_UINT_PACK32,
            Self::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            Self::R32Uint => vk::Format::R32_UINT,
            Self::R32Float => vk::Format::R32_SFLOAT,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8Unorm => 1,
            Self::R16Float => 2,
            Self::B10G11R11Float
            | Self::Abgr8Unorm
            | Self::Abgr8Uint
            | Self::Rgba8Uint
            | Self::R32Uint
            | Self::R32Float => 4,
            Self::Rgba16Float => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeTexelFormat {
    R16Float,
    R16Uint,
    R16Sint,
    R32Float,
    R32Uint,
    R32Sint,
    Rgba32Float,
    Rgba32Uint,
    Rgba32Sint,
}

impl ComputeTexelFormat {
    pub fn from_tic(tic: &TicEntry, sample_type: ComputeSampleType) -> Option<Self> {
        use crate::texture::{ComponentType, TicFormat};

        if !tic.is_buffer() {
            return None;
        }
        let component = tic.component_types[0];
        if tic.format == TicFormat::R32G32B32A32
            && tic
                .component_types
                .iter()
                .any(|candidate| *candidate != component)
        {
            return None;
        }
        match (tic.format, sample_type, component) {
            (TicFormat::R16, ComputeSampleType::Float, ComponentType::Float) => {
                Some(Self::R16Float)
            }
            (TicFormat::R16, ComputeSampleType::Uint, ComponentType::Uint) => Some(Self::R16Uint),
            (TicFormat::R16, ComputeSampleType::Sint, ComponentType::Sint) => Some(Self::R16Sint),
            (TicFormat::R32, ComputeSampleType::Float, ComponentType::Float) => {
                Some(Self::R32Float)
            }
            (TicFormat::R32, ComputeSampleType::Uint, ComponentType::Uint) => Some(Self::R32Uint),
            (TicFormat::R32, ComputeSampleType::Sint, ComponentType::Sint) => Some(Self::R32Sint),
            (TicFormat::R32G32B32A32, ComputeSampleType::Float, ComponentType::Float) => {
                Some(Self::Rgba32Float)
            }
            (TicFormat::R32G32B32A32, ComputeSampleType::Uint, ComponentType::Uint) => {
                Some(Self::Rgba32Uint)
            }
            (TicFormat::R32G32B32A32, ComputeSampleType::Sint, ComponentType::Sint) => {
                Some(Self::Rgba32Sint)
            }
            _ => None,
        }
    }

    pub(crate) fn vk_format(self) -> vk::Format {
        match self {
            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::R16Uint => vk::Format::R16_UINT,
            Self::R16Sint => vk::Format::R16_SINT,
            Self::R32Float => vk::Format::R32_SFLOAT,
            Self::R32Uint => vk::Format::R32_UINT,
            Self::R32Sint => vk::Format::R32_SINT,
            Self::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::Rgba32Uint => vk::Format::R32G32B32A32_UINT,
            Self::Rgba32Sint => vk::Format::R32G32B32A32_SINT,
        }
    }

    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::R16Float | Self::R16Uint | Self::R16Sint => 2,
            Self::R32Float | Self::R32Uint | Self::R32Sint => 4,
            Self::Rgba32Float | Self::Rgba32Uint | Self::Rgba32Sint => 16,
        }
    }

    pub(crate) const fn supports_storage_atomics(self) -> bool {
        matches!(self, Self::R32Uint)
    }

    pub(crate) const fn requires_storage_image_extended_formats(self) -> bool {
        matches!(self, Self::R16Float | Self::R16Uint | Self::R16Sint)
    }

    pub fn spirv_format(self) -> nexium_spirv::ComputeTexelFormat {
        match self {
            Self::R16Float => nexium_spirv::ComputeTexelFormat::R16Float,
            Self::R16Uint => nexium_spirv::ComputeTexelFormat::R16Uint,
            Self::R16Sint => nexium_spirv::ComputeTexelFormat::R16Sint,
            Self::R32Float => nexium_spirv::ComputeTexelFormat::R32Float,
            Self::R32Uint => nexium_spirv::ComputeTexelFormat::R32Uint,
            Self::R32Sint => nexium_spirv::ComputeTexelFormat::R32Sint,
            Self::Rgba32Float => nexium_spirv::ComputeTexelFormat::Rgba32Float,
            Self::Rgba32Uint => nexium_spirv::ComputeTexelFormat::Rgba32Uint,
            Self::Rgba32Sint => nexium_spirv::ComputeTexelFormat::Rgba32Sint,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ComputeUniformBuffer {
    pub binding: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeRawStorageKey {
    pub mapping_epoch: u64,
    pub nvmap_id: u32,
    pub gpu_va: u64,
    pub cpu_addr: u64,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct ComputeTexelBuffer {
    pub bindings: Vec<u32>,
    pub bytes: Vec<u8>,
    pub byte_len: usize,
    pub format: ComputeTexelFormat,
    pub raw: bool,
    pub raw_storage_key: Option<ComputeRawStorageKey>,
    pub writable: bool,
    pub requires_atomics: bool,
}

#[derive(Clone, Debug)]
pub struct ComputeUniformTexelBuffer {
    pub binding: u32,
    pub bytes: Vec<u8>,
    pub format: ComputeTexelFormat,
}

#[derive(Clone, Debug)]
pub struct ComputeSampledRt {
    pub binding: u32,
    pub key: RtKey,
    pub tic: TicEntry,
    pub tsc: TscEntry,
    pub sample_type: ComputeSampleType,
    pub guest_bytes: Option<Vec<u8>>,
    pub guest_bytes_authoritative: bool,
    pub require_live: bool,
    pub content_key: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ComputeSampledImage {
    pub binding: u32,
    pub key: Option<RtKey>,
    pub tic: TicEntry,
    pub sample_type: ComputeSampleType,
    pub guest_bytes: Vec<u8>,
    pub guest_bytes_authoritative: bool,
    pub require_live: bool,
    pub content_key: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ComputeStorageImage {
    pub binding: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub is_3d: bool,
    pub format: ComputeStorageFormat,
    pub initial_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeImageAlias {
    pub sampled_binding: u32,
    pub storage_binding: u32,
}

#[derive(Clone, Debug)]
pub struct ComputeDispatch {
    pub program_key: u64,
    pub spirv: Arc<[u32]>,
    pub spirv_hash: u64,
    pub group_count: [u32; 3],
    pub local_size: [u32; 3],
    pub shared_memory_size: u32,
    pub required_subgroup_size: Option<u32>,
    pub requires_workgroup_explicit_layout: bool,
    pub uniform_buffers: Vec<ComputeUniformBuffer>,
    pub texel_buffers: Vec<ComputeTexelBuffer>,
    pub uniform_texel_buffers: Vec<ComputeUniformTexelBuffer>,
    pub sampled_rts: Vec<ComputeSampledRt>,
    pub sampled_images: Vec<ComputeSampledImage>,
    pub outputs: Vec<ComputeStorageImage>,
    pub image_aliases: Vec<ComputeImageAlias>,
}

#[derive(Clone, Debug)]
pub struct ComputeTexelReadback {
    pub resource_index: usize,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct ComputeImageReadback {
    pub resource_index: usize,
    pub binding: u32,
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: ComputeStorageFormat,
}

#[derive(Clone, Debug)]
pub struct ComputeDispatchResult {
    pub image_readbacks: Vec<ComputeImageReadback>,
    pub texel_readbacks: Vec<ComputeTexelReadback>,
}

#[derive(Clone, Debug)]
pub enum ComputeDispatchOutcome {
    Executed(ComputeDispatchResult),
    Submitted(u64),
    Unsupported(String),
    FailedBeforeSubmit(String),
    SubmittedFailure(String),
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum ComputeDescriptorKind {
    UniformBuffer,
    StorageBuffer,
    StorageTexelBuffer,
    UniformTexelBuffer,
    CombinedSampledImage,
    SampledImage,
    StorageImage,
}

impl ComputeDescriptorKind {
    fn vk_type(self) -> vk::DescriptorType {
        match self {
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            Self::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            Self::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            Self::CombinedSampledImage => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            Self::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ComputeDescriptorSpec {
    pub binding: u32,
    pub kind: ComputeDescriptorKind,
}

pub(crate) fn descriptor_spec(dispatch: &ComputeDispatch) -> Vec<ComputeDescriptorSpec> {
    let mut descriptors = Vec::new();
    descriptors.extend(
        dispatch
            .uniform_buffers
            .iter()
            .map(|buffer| ComputeDescriptorSpec {
                binding: buffer.binding,
                kind: ComputeDescriptorKind::UniformBuffer,
            }),
    );
    for buffer in &dispatch.texel_buffers {
        descriptors.extend(buffer.bindings.iter().map(|binding| ComputeDescriptorSpec {
            binding: *binding,
            kind: if buffer.raw {
                ComputeDescriptorKind::StorageBuffer
            } else {
                ComputeDescriptorKind::StorageTexelBuffer
            },
        }));
    }
    descriptors.extend(
        dispatch
            .uniform_texel_buffers
            .iter()
            .map(|buffer| ComputeDescriptorSpec {
                binding: buffer.binding,
                kind: ComputeDescriptorKind::UniformTexelBuffer,
            }),
    );
    descriptors.extend(
        dispatch
            .sampled_rts
            .iter()
            .map(|sample| ComputeDescriptorSpec {
                binding: sample.binding,
                kind: ComputeDescriptorKind::CombinedSampledImage,
            }),
    );
    descriptors.extend(
        dispatch
            .sampled_images
            .iter()
            .map(|sample| ComputeDescriptorSpec {
                binding: sample.binding,
                kind: ComputeDescriptorKind::SampledImage,
            }),
    );
    descriptors.extend(dispatch.outputs.iter().map(|output| ComputeDescriptorSpec {
        binding: output.binding,
        kind: ComputeDescriptorKind::StorageImage,
    }));
    descriptors.sort_unstable();
    descriptors
}

pub(crate) fn compute_buffer_binding_counts(buffers: &[ComputeTexelBuffer]) -> (usize, usize) {
    buffers
        .iter()
        .fold((0, 0), |(raw_storage, storage_texel), buffer| {
            if buffer.raw {
                (
                    raw_storage.saturating_add(buffer.bindings.len()),
                    storage_texel,
                )
            } else {
                (
                    raw_storage,
                    storage_texel.saturating_add(buffer.bindings.len()),
                )
            }
        })
}

pub(crate) fn validate_compute_local_size(
    local_size: [u32; 3],
    max_size: [u32; 3],
    max_invocations: u32,
) -> Result<u32, String> {
    for axis in 0..3 {
        if local_size[axis] == 0 {
            return Err(format!(
                "compute local size {local_size:?} contains a zero component"
            ));
        }
        if local_size[axis] > max_size[axis] {
            return Err(format!(
                "compute local size {local_size:?} exceeds per-axis device limit {max_size:?}"
            ));
        }
    }
    let invocations = local_size.into_iter().try_fold(1u32, |product, size| {
        product
            .checked_mul(size)
            .ok_or_else(|| format!("compute local size product overflows u32 for {local_size:?}"))
    })?;
    if invocations > max_invocations {
        return Err(format!(
            "compute local size {local_size:?} has {invocations} invocations, exceeding device limit {max_invocations}"
        ));
    }
    Ok(invocations)
}

pub fn compute_spirv_hash(words: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in words.iter().flat_map(|word| word.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ComputePipelineKey {
    program_key: u64,
    spirv_hash: u64,
    local_size: [u32; 3],
    required_subgroup_size: u32,
    descriptors: Vec<ComputeDescriptorSpec>,
}

struct ComputeProgram {
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline: vk::Pipeline,
}

const COMPUTE_UNIFORM_POOL_MAX_ITEMS: usize = 32;
const COMPUTE_UNIFORM_POOL_MAX_BYTES: u64 = 1024 * 1024;
const COMPUTE_RAW_STORAGE_POOL_MAX_ITEMS: usize = 512;
const COMPUTE_RAW_STORAGE_POOL_MAX_BYTES: u64 = 128 * 1024 * 1024;
const COMPUTE_OUTPUT_POOL_MAX_ITEMS: usize = 8;
const COMPUTE_OUTPUT_POOL_MAX_BYTES: u64 = 64 * 1024 * 1024;
const COMPUTE_READBACK_POOL_MAX_ITEMS: usize = 8;
const COMPUTE_READBACK_POOL_MAX_BYTES: u64 = 64 * 1024 * 1024;

struct PooledResource<K, V> {
    key: K,
    value: V,
    bytes: u64,
}

struct BoundedResourcePool<K, V> {
    entries: VecDeque<PooledResource<K, V>>,
    retained_bytes: u64,
    max_items: usize,
    max_bytes: u64,
}

impl<K: PartialEq, V> BoundedResourcePool<K, V> {
    fn new(max_items: usize, max_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            retained_bytes: 0,
            max_items,
            max_bytes,
        }
    }

    fn take(&mut self, key: &K) -> Option<V> {
        let index = self.entries.iter().position(|entry| &entry.key == key)?;
        let entry = self.entries.remove(index)?;
        self.retained_bytes = self.retained_bytes.saturating_sub(entry.bytes);
        Some(entry.value)
    }

    fn insert(&mut self, key: K, value: V, bytes: u64) -> Vec<V> {
        let mut evicted = Vec::new();
        if self.max_items == 0 || bytes > self.max_bytes {
            evicted.push(value);
            return evicted;
        }
        while self.entries.len() >= self.max_items
            || self.retained_bytes.saturating_add(bytes) > self.max_bytes
        {
            let Some(entry) = self.entries.pop_front() else {
                break;
            };
            self.retained_bytes = self.retained_bytes.saturating_sub(entry.bytes);
            evicted.push(entry.value);
        }
        self.retained_bytes = self.retained_bytes.saturating_add(bytes);
        self.entries.push_back(PooledResource { key, value, bytes });
        evicted
    }

    fn drain_values(&mut self) -> impl Iterator<Item = V> + '_ {
        self.retained_bytes = 0;
        self.entries.drain(..).map(|entry| entry.value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeOutputPoolKey {
    width: u32,
    height: u32,
    depth: u32,
    is_3d: bool,
    format: vk::Format,
    sampled: bool,
}

struct ComputeResourcePool {
    uniforms: BoundedResourcePool<u64, ComputeBufferResource>,
    raw_storage: BoundedResourcePool<u64, ComputeBufferResource>,
    outputs: BoundedResourcePool<ComputeOutputPoolKey, ComputeImageResource>,
    readbacks: BoundedResourcePool<u64, ComputeBufferResource>,
    stats: ComputeResourcePoolStats,
}

#[derive(Default)]
struct ComputeResourcePoolStats {
    completed_dispatches: u64,
    uniform_hits: u64,
    uniform_misses: u64,
    raw_storage_hits: u64,
    raw_storage_misses: u64,
    output_hits: u64,
    output_misses: u64,
    readback_hits: u64,
    readback_misses: u64,
}

impl ComputeResourcePool {
    fn new() -> Self {
        Self {
            uniforms: BoundedResourcePool::new(
                COMPUTE_UNIFORM_POOL_MAX_ITEMS,
                COMPUTE_UNIFORM_POOL_MAX_BYTES,
            ),
            raw_storage: BoundedResourcePool::new(
                COMPUTE_RAW_STORAGE_POOL_MAX_ITEMS,
                COMPUTE_RAW_STORAGE_POOL_MAX_BYTES,
            ),
            outputs: BoundedResourcePool::new(
                COMPUTE_OUTPUT_POOL_MAX_ITEMS,
                COMPUTE_OUTPUT_POOL_MAX_BYTES,
            ),
            readbacks: BoundedResourcePool::new(
                COMPUTE_READBACK_POOL_MAX_ITEMS,
                COMPUTE_READBACK_POOL_MAX_BYTES,
            ),
            stats: ComputeResourcePoolStats::default(),
        }
    }

    fn log_profile_after_recycle(&mut self) {
        self.stats.completed_dispatches = self.stats.completed_dispatches.saturating_add(1);
        let dispatches = self.stats.completed_dispatches;
        if !compute_pool_profile_enabled() || (dispatches > 8 && dispatches % 128 != 0) {
            return;
        }
        log::warn!(
            "[compute-pool] dispatches={} uniform_hit_miss={}/{} raw_storage_hit_miss={}/{} \
             output_hit_miss={}/{} readback_hit_miss={}/{} retained_items={}/{}/{}/{} \
             retained_bytes={}/{}/{}/{}",
            dispatches,
            self.stats.uniform_hits,
            self.stats.uniform_misses,
            self.stats.raw_storage_hits,
            self.stats.raw_storage_misses,
            self.stats.output_hits,
            self.stats.output_misses,
            self.stats.readback_hits,
            self.stats.readback_misses,
            self.uniforms.entries.len(),
            self.raw_storage.entries.len(),
            self.outputs.entries.len(),
            self.readbacks.entries.len(),
            self.uniforms.retained_bytes,
            self.raw_storage.retained_bytes,
            self.outputs.retained_bytes,
            self.readbacks.retained_bytes,
        );
    }

    fn destroy(&mut self, device: &ash::Device) {
        for buffer in self.uniforms.drain_values() {
            buffer.destroy(device);
        }
        for buffer in self.raw_storage.drain_values() {
            buffer.destroy(device);
        }
        for image in self.outputs.drain_values() {
            image.destroy(device);
        }
        for buffer in self.readbacks.drain_values() {
            buffer.destroy(device);
        }
    }
}

fn compute_pool_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        ["NEXIUM_COMPUTE_POOL_PROFILE", "NEXIUM_NVDRV_PROFILE"]
            .into_iter()
            .any(|name| {
                std::env::var_os(name).is_some_and(|value| {
                    let value = value.to_string_lossy();
                    let value = value.trim();
                    !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
                })
            })
    })
}

impl ComputeProgram {
    fn new(
        device: &ash::Device,
        spirv: &[u32],
        required_subgroup_size: Option<u32>,
        descriptors: &[ComputeDescriptorSpec],
    ) -> Result<Self, String> {
        let bindings: Vec<_> = descriptors
            .iter()
            .map(|descriptor| vk::DescriptorSetLayoutBinding {
                binding: descriptor.binding,
                descriptor_type: descriptor.kind.vk_type(),
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                p_immutable_samplers: std::ptr::null(),
                _marker: std::marker::PhantomData,
            })
            .collect();
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("create compute descriptor layout: {e:?}"))?
        };
        let set_layouts = [descriptor_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        let pipeline_layout =
            match unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) } {
                Ok(layout) => layout,
                Err(error) => {
                    unsafe { device.destroy_descriptor_set_layout(descriptor_layout, None) };
                    return Err(format!("create compute pipeline layout: {error:?}"));
                }
            };

        let mut descriptor_counts = [0u32; 7];
        for descriptor in descriptors {
            descriptor_counts[descriptor.kind as usize] += 1;
        }
        let descriptor_kinds = [
            ComputeDescriptorKind::UniformBuffer,
            ComputeDescriptorKind::StorageBuffer,
            ComputeDescriptorKind::StorageTexelBuffer,
            ComputeDescriptorKind::UniformTexelBuffer,
            ComputeDescriptorKind::CombinedSampledImage,
            ComputeDescriptorKind::SampledImage,
            ComputeDescriptorKind::StorageImage,
        ];
        let pool_sizes: Vec<_> = descriptor_kinds
            .into_iter()
            .zip(descriptor_counts)
            .filter(|(_, count)| *count != 0)
            .map(|(kind, descriptor_count)| vk::DescriptorPoolSize {
                ty: kind.vk_type(),
                descriptor_count: descriptor_count * COMPUTE_PROGRAM_SET_CAPACITY,
            })
            .collect();
        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            max_sets: COMPUTE_PROGRAM_SET_CAPACITY,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            ..Default::default()
        };
        let descriptor_pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(format!("create compute descriptor pool: {error:?}"));
            }
        };

        let module_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: std::mem::size_of_val(spirv),
            p_code: spirv.as_ptr(),
            ..Default::default()
        };
        let module = match unsafe { device.create_shader_module(&module_info, None) } {
            Ok(module) => module,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(format!("create compute shader module: {error:?}"));
            }
        };
        let subgroup_info = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            required_subgroup_size: required_subgroup_size.unwrap_or(0),
            ..Default::default()
        };
        let stage = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::COMPUTE,
            module,
            p_name: c"main".as_ptr(),
            p_next: if required_subgroup_size.is_some() {
                &subgroup_info as *const _ as *const std::ffi::c_void
            } else {
                std::ptr::null()
            },
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            ..Default::default()
        };
        let created = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        unsafe { device.destroy_shader_module(module, None) };
        let pipeline = match created {
            Ok(pipelines) => match pipelines.into_iter().next() {
                Some(pipeline) => pipeline,
                None => {
                    unsafe {
                        device.destroy_descriptor_pool(descriptor_pool, None);
                        device.destroy_pipeline_layout(pipeline_layout, None);
                        device.destroy_descriptor_set_layout(descriptor_layout, None);
                    }
                    return Err("create compute pipeline returned no pipeline".to_string());
                }
            },
            Err((pipelines, error)) => {
                unsafe {
                    for pipeline in pipelines {
                        if pipeline != vk::Pipeline::null() {
                            device.destroy_pipeline(pipeline, None);
                        }
                    }
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_layout, None);
                }
                return Err(format!("create compute pipeline: {error:?}"));
            }
        };
        Ok(Self {
            descriptor_layout,
            pipeline_layout,
            descriptor_pool,
            pipeline,
        })
    }

    fn reset_and_allocate_set(&self, device: &ash::Device) -> Result<vk::DescriptorSet, String> {
        unsafe {
            device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| format!("reset compute descriptor pool: {e:?}"))?;
        }
        self.allocate_set(device)
    }

    fn allocate_set(&self, device: &ash::Device) -> Result<vk::DescriptorSet, String> {
        let layouts = [self.descriptor_layout];
        let info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };
        unsafe {
            device
                .allocate_descriptor_sets(&info)
                .map(|sets| sets[0])
                .map_err(|e| format!("allocate compute descriptor set: {e:?}"))
        }
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

pub(crate) struct PreparedComputeProgram {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
}

pub(crate) const COMPUTE_PROGRAM_SET_CAPACITY: u32 = 64;

pub(crate) struct ComputeBackend {
    programs: HashMap<ComputePipelineKey, ComputeProgram>,
    resource_pool: ComputeResourcePool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub required_subgroup_size_stages: vk::ShaderStageFlags,
    pub max_group_count: [u32; 3],
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_uniform_buffer_range: u32,
    pub max_uniform_buffers: u32,
    pub max_storage_buffer_range: u32,
    pub max_storage_buffers: u32,
    pub max_storage_texel_buffers: u32,
    pub max_samplers: u32,
    pub max_sampled_images: u32,
    pub max_storage_images: u32,
    pub max_resources: u32,
    pub workgroup_explicit_layout_enabled: bool,
    pub storage_image_extended_formats_enabled: bool,
}

impl ComputeBackend {
    pub fn new(
        _device: &ash::Device,
        min_subgroup_size: u32,
        max_subgroup_size: u32,
        required_subgroup_size_stages: vk::ShaderStageFlags,
        workgroup_explicit_layout_enabled: bool,
        storage_image_extended_formats_enabled: bool,
        limits: &vk::PhysicalDeviceLimits,
    ) -> Result<Self, String> {
        Ok(Self {
            programs: HashMap::new(),
            resource_pool: ComputeResourcePool::new(),
            min_subgroup_size,
            max_subgroup_size,
            required_subgroup_size_stages,
            max_group_count: limits.max_compute_work_group_count,
            max_compute_work_group_size: limits.max_compute_work_group_size,
            max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
            max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
            max_image_dimension_2d: limits.max_image_dimension2_d,
            max_image_dimension_3d: limits.max_image_dimension3_d,
            max_uniform_buffer_range: limits.max_uniform_buffer_range,
            max_uniform_buffers: limits.max_per_stage_descriptor_uniform_buffers,
            max_storage_buffer_range: limits.max_storage_buffer_range,
            max_storage_buffers: limits
                .max_per_stage_descriptor_storage_buffers
                .min(limits.max_descriptor_set_storage_buffers),
            max_storage_texel_buffers: limits.max_per_stage_descriptor_storage_images,
            max_samplers: limits.max_per_stage_descriptor_samplers,
            max_sampled_images: limits.max_per_stage_descriptor_sampled_images,
            max_storage_images: limits.max_per_stage_descriptor_storage_images,
            max_resources: limits.max_per_stage_resources,
            workgroup_explicit_layout_enabled,
            storage_image_extended_formats_enabled,
        })
    }

    pub fn prepare_program(
        &mut self,
        device: &ash::Device,
        program_key: u64,
        spirv: &[u32],
        spirv_hash: u64,
        local_size: [u32; 3],
        required_subgroup_size: Option<u32>,
        descriptors: Vec<ComputeDescriptorSpec>,
        reuse_sets: bool,
    ) -> Result<PreparedComputeProgram, String> {
        validate_compute_local_size(
            local_size,
            self.max_compute_work_group_size,
            self.max_compute_work_group_invocations,
        )?;
        let key = ComputePipelineKey {
            program_key,
            spirv_hash,
            local_size,
            required_subgroup_size: required_subgroup_size.unwrap_or(0),
            descriptors,
        };
        if !self.programs.contains_key(&key) {
            let program =
                ComputeProgram::new(device, spirv, required_subgroup_size, &key.descriptors)?;
            self.programs.insert(key.clone(), program);
        }
        let program = self.programs.get(&key).unwrap();
        let descriptor_set = if reuse_sets {
            program.allocate_set(device)?
        } else {
            program.reset_and_allocate_set(device)?
        };
        Ok(PreparedComputeProgram {
            pipeline: program.pipeline,
            pipeline_layout: program.pipeline_layout,
            descriptor_pool: program.descriptor_pool,
            descriptor_set,
        })
    }

    pub fn acquire_uniform_buffer(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        data: &[u8],
    ) -> Result<ComputeBufferResource, String> {
        let key = (data.len().max(16) as u64).next_power_of_two().max(256);
        if let Some(resource) = self.resource_pool.uniforms.take(&key) {
            self.resource_pool.stats.uniform_hits =
                self.resource_pool.stats.uniform_hits.saturating_add(1);
            if let Err(error) = resource.write(device, data) {
                resource.destroy(device);
                return Err(error);
            }
            return Ok(resource);
        }
        self.resource_pool.stats.uniform_misses =
            self.resource_pool.stats.uniform_misses.saturating_add(1);
        create_compute_buffer_allocation(
            device,
            mem_props,
            key,
            Some(data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            None,
            false,
            true,
        )
    }

    pub fn acquire_raw_storage_buffer(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        data: &[u8],
    ) -> Result<ComputeBufferResource, String> {
        let key = data.len().max(16) as u64;
        if let Some(resource) = self.resource_pool.raw_storage.take(&key) {
            self.resource_pool.stats.raw_storage_hits =
                self.resource_pool.stats.raw_storage_hits.saturating_add(1);
            if let Err(error) = resource.write(device, data) {
                resource.destroy(device);
                return Err(error);
            }
            return Ok(resource);
        }
        self.resource_pool.stats.raw_storage_misses = self
            .resource_pool
            .stats
            .raw_storage_misses
            .saturating_add(1);
        create_compute_raw_storage_buffer(device, mem_props, data)
    }

    pub fn acquire_output_image(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        depth: u32,
        is_3d: bool,
        format: vk::Format,
        sampled: bool,
    ) -> Result<ComputeImageResource, String> {
        let key = ComputeOutputPoolKey {
            width,
            height,
            depth,
            is_3d,
            format,
            sampled,
        };
        if let Some(resource) = self.resource_pool.outputs.take(&key) {
            self.resource_pool.stats.output_hits =
                self.resource_pool.stats.output_hits.saturating_add(1);
            return Ok(resource);
        }
        self.resource_pool.stats.output_misses =
            self.resource_pool.stats.output_misses.saturating_add(1);
        create_compute_output_image(
            device, mem_props, width, height, depth, is_3d, format, sampled,
        )
    }

    pub fn acquire_readback_buffer(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: u64,
    ) -> Result<ComputeBufferResource, String> {
        let key = size.max(16);
        if let Some(resource) = self.resource_pool.readbacks.take(&key) {
            self.resource_pool.stats.readback_hits =
                self.resource_pool.stats.readback_hits.saturating_add(1);
            return Ok(resource);
        }
        self.resource_pool.stats.readback_misses =
            self.resource_pool.stats.readback_misses.saturating_add(1);
        create_compute_readback_buffer(device, mem_props, size)
    }

    pub fn recycle_dispatch_resources(
        &mut self,
        device: &ash::Device,
        uniforms: Vec<ComputeBufferResource>,
        raw_storage: Vec<ComputeBufferResource>,
        outputs: Vec<ComputeImageResource>,
        readbacks: Vec<ComputeBufferResource>,
    ) {
        for resource in uniforms {
            let key = resource.size;
            let bytes = resource.allocation_size;
            for evicted in self.resource_pool.uniforms.insert(key, resource, bytes) {
                evicted.destroy(device);
            }
        }
        for resource in raw_storage {
            let key = resource.size;
            let bytes = resource.allocation_size;
            for evicted in self.resource_pool.raw_storage.insert(key, resource, bytes) {
                evicted.destroy(device);
            }
        }
        for resource in outputs {
            let key = ComputeOutputPoolKey {
                width: resource.width,
                height: resource.height,
                depth: resource.depth,
                is_3d: resource.is_3d,
                format: resource.format,
                sampled: resource.sampled,
            };
            let bytes = resource.allocation_size;
            for evicted in self.resource_pool.outputs.insert(key, resource, bytes) {
                evicted.destroy(device);
            }
        }
        for resource in readbacks {
            let key = resource.size;
            let bytes = resource.allocation_size;
            for evicted in self.resource_pool.readbacks.insert(key, resource, bytes) {
                evicted.destroy(device);
            }
        }
        self.resource_pool.log_profile_after_recycle();
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.resource_pool.destroy(device);
        for (_, program) in self.programs.drain() {
            program.destroy(device);
        }
    }
}

pub(crate) struct ComputeBufferResource {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub allocation_size: u64,
    pub size: u64,
    pub view: vk::BufferView,
    mapped: *mut u8,
}

impl ComputeBufferResource {
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            if self.view != vk::BufferView::null() {
                device.destroy_buffer_view(self.view, None);
            }
            if !self.mapped.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }

    pub fn read(&self, device: &ash::Device, byte_len: usize) -> Result<Vec<u8>, String> {
        if byte_len as u64 > self.size {
            return Err(format!(
                "compute buffer read exceeds allocation: {} > {}",
                byte_len, self.size
            ));
        }
        let mut bytes = vec![0u8; byte_len];
        if bytes.is_empty() {
            return Ok(bytes);
        }
        unsafe {
            let transient = self.mapped.is_null();
            let ptr = if transient {
                device
                    .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map compute readback buffer: {e:?}"))?
                    as *const u8
            } else {
                self.mapped.cast_const()
            };
            std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), byte_len);
            if transient {
                device.unmap_memory(self.memory);
            }
        }
        Ok(bytes)
    }

    pub(crate) fn write(&self, device: &ash::Device, data: &[u8]) -> Result<(), String> {
        if data.len() as u64 > self.size {
            return Err(format!(
                "compute buffer write exceeds allocation: {} > {}",
                data.len(),
                self.size
            ));
        }
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            let transient = self.mapped.is_null();
            let ptr = if transient {
                device
                    .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                    .map_err(|e| format!("map compute upload buffer: {e:?}"))?
                    as *mut u8
            } else {
                self.mapped
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            if transient {
                device.unmap_memory(self.memory);
            }
        }
        Ok(())
    }
}

fn create_compute_raw_storage_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    data: &[u8],
) -> Result<ComputeBufferResource, String> {
    create_compute_buffer_allocation(
        device,
        mem_props,
        data.len().max(16) as u64,
        Some(data),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        None,
        false,
        true,
    )
}

pub(crate) fn create_compute_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    with_r32_uint_view: bool,
) -> Result<ComputeBufferResource, String> {
    create_compute_buffer_allocation(
        device,
        mem_props,
        data.len().max(16) as u64,
        Some(data),
        usage,
        with_r32_uint_view.then_some(vk::Format::R32_UINT),
        false,
        false,
    )
}

pub(crate) fn create_compute_texel_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    format: vk::Format,
) -> Result<ComputeBufferResource, String> {
    create_compute_buffer_allocation(
        device,
        mem_props,
        data.len().max(16) as u64,
        Some(data),
        usage,
        Some(format),
        false,
        false,
    )
}

fn create_compute_readback_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
) -> Result<ComputeBufferResource, String> {
    create_compute_buffer_allocation(
        device,
        mem_props,
        size.max(16),
        None,
        vk::BufferUsageFlags::TRANSFER_DST,
        None,
        true,
        false,
    )
}

fn create_compute_buffer_allocation(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    initial_data: Option<&[u8]>,
    usage: vk::BufferUsageFlags,
    view_format: Option<vk::Format>,
    prefer_host_cached: bool,
    persistently_mapped: bool,
) -> Result<ComputeBufferResource, String> {
    let info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buffer = unsafe {
        device
            .create_buffer(&info, None)
            .map_err(|e| format!("create compute buffer: {e:?}"))?
    };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let required_properties =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let memory_type_index = if prefer_host_cached {
        find_memory_type(
            mem_props,
            requirements.memory_type_bits,
            required_properties | vk::MemoryPropertyFlags::HOST_CACHED,
        )
        .or_else(|| {
            find_memory_type(
                mem_props,
                requirements.memory_type_bits,
                required_properties,
            )
        })
    } else {
        find_memory_type(
            mem_props,
            requirements.memory_type_bits,
            required_properties,
        )
    };
    let Some(memory_type_index) = memory_type_index else {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err("no HOST_VISIBLE|HOST_COHERENT memory for compute buffer".to_string());
    };
    let allocation = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(format!("allocate compute buffer memory: {e:?}"));
        }
    };
    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(format!("bind compute buffer memory: {e:?}"));
    }
    let should_map = persistently_mapped || initial_data.is_some_and(|data| !data.is_empty());
    let mut mapped = std::ptr::null_mut();
    if should_map {
        mapped = match unsafe {
            device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
        } {
            Ok(mapped) => mapped.cast(),
            Err(e) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(format!("map compute upload buffer: {e:?}"));
            }
        };
    }
    if let Some(data) = initial_data.filter(|data| !data.is_empty()) {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len());
        }
    }
    if !persistently_mapped && !mapped.is_null() {
        unsafe {
            device.unmap_memory(memory);
        }
        mapped = std::ptr::null_mut();
    }
    let view = if let Some(format) = view_format {
        let view_range = initial_data.map_or(size, |data| data.len() as u64);
        let view_info = vk::BufferViewCreateInfo {
            s_type: vk::StructureType::BUFFER_VIEW_CREATE_INFO,
            buffer,
            format,
            offset: 0,
            range: view_range,
            ..Default::default()
        };
        match unsafe { device.create_buffer_view(&view_info, None) } {
            Ok(view) => view,
            Err(e) => {
                unsafe {
                    if !mapped.is_null() {
                        device.unmap_memory(memory);
                    }
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(format!("create {:?} compute buffer view: {e:?}", format));
            }
        }
    } else {
        vk::BufferView::null()
    };
    Ok(ComputeBufferResource {
        buffer,
        memory,
        allocation_size: requirements.size,
        size,
        view,
        mapped,
    })
}

pub(crate) struct ComputeImageResource {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub allocation_size: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub is_3d: bool,
    pub format: vk::Format,
    pub sampled: bool,
}

impl ComputeImageResource {
    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub(crate) fn create_compute_output_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    depth: u32,
    is_3d: bool,
    format: vk::Format,
    sampled: bool,
) -> Result<ComputeImageResource, String> {
    let info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: if is_3d {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        },
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | if sampled {
                vk::ImageUsageFlags::SAMPLED
            } else {
                vk::ImageUsageFlags::empty()
            },
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let image = unsafe {
        device
            .create_image(&info, None)
            .map_err(|e| format!("create transient compute image: {e:?}"))?
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = find_memory_type(
        mem_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) else {
        unsafe { device.destroy_image(image, None) };
        return Err("no DEVICE_LOCAL memory for transient compute image".to_string());
    };
    let allocation = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(e) => {
            unsafe { device.destroy_image(image, None) };
            return Err(format!("allocate transient compute image: {e:?}"));
        }
    };
    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        return Err(format!("bind transient compute image: {e:?}"));
    }
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if is_3d {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        components: vk::ComponentMapping::default(),
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..Default::default()
    };
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(e) => {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(format!("create transient compute image view: {e:?}"));
        }
    };
    Ok(ComputeImageResource {
        image,
        view,
        memory,
        allocation_size: requirements.size,
        width,
        height,
        depth,
        is_3d,
        format,
        sampled,
    })
}

pub(crate) fn create_compute_sampled_alias_view(
    device: &ash::Device,
    image: &ComputeImageResource,
    components: vk::ComponentMapping,
) -> Result<vk::ImageView, String> {
    if !image.sampled {
        return Err("compute output image was not created for sampled cross-access".to_string());
    }
    let info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image: image.image,
        view_type: if image.is_3d {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format: image.format,
        components,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..Default::default()
    };
    unsafe { device.create_image_view(&info, None) }
        .map_err(|error| format!("create compute sampled/storage alias view: {error:?}"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_compute_sampled_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    depth: u32,
    is_3d: bool,
    format: vk::Format,
    components: vk::ComponentMapping,
    mip_levels: u32,
    view_base_mip: u32,
    view_mip_levels: u32,
) -> Result<ComputeImageResource, String> {
    let mip_levels = mip_levels.max(1);
    let max_mip_levels = u32::BITS - width.max(height).max(depth).max(1).leading_zeros();
    if width == 0
        || height == 0
        || depth == 0
        || (is_3d && mip_levels != 1)
        || mip_levels > max_mip_levels
        || view_mip_levels == 0
        || view_base_mip >= mip_levels
        || view_base_mip.saturating_add(view_mip_levels) > mip_levels
    {
        return Err(format!(
            "invalid cached compute sampled image/view: levels={mip_levels} view={view_base_mip}/{view_mip_levels} extent={width}x{height}x{depth} 3d={is_3d}"
        ));
    }
    let info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: if is_3d {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        },
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
        mip_levels,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let image = unsafe {
        device
            .create_image(&info, None)
            .map_err(|error| format!("create cached compute sampled image: {error:?}"))?
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = find_memory_type(
        mem_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) else {
        unsafe { device.destroy_image(image, None) };
        return Err("no DEVICE_LOCAL memory for cached compute sampled image".to_string());
    };
    let allocation = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_image(image, None) };
            return Err(format!("allocate cached compute sampled image: {error:?}"));
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        return Err(format!("bind cached compute sampled image: {error:?}"));
    }
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if is_3d {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        components,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: view_base_mip,
            level_count: view_mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..Default::default()
    };
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            return Err(format!(
                "create cached compute sampled image view: {error:?}"
            ));
        }
    };
    Ok(ComputeImageResource {
        image,
        view,
        memory,
        allocation_size: requirements.size,
        width,
        height,
        depth,
        is_3d,
        format,
        sampled: true,
    })
}

pub(crate) struct ComputeGuestImageResource {
    pub image: ComputeImageResource,
    pub upload: ComputeBufferResource,
    pub needs_upload: bool,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub copies: Vec<vk::BufferImageCopy>,
    pub layout: vk::ImageLayout,
}

impl ComputeGuestImageResource {
    pub fn destroy(self, device: &ash::Device) {
        self.image.destroy(device);
        self.upload.destroy(device);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ComputeGuestImageCopy {
    pub buffer_offset: u64,
    pub mip_level: u32,
    pub width: u32,
    pub height: u32,
}

pub(crate) fn create_compute_guest_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    depth: u32,
    is_3d: bool,
    is_cube: bool,
    format: vk::Format,
    components: vk::ComponentMapping,
    mip_levels: u32,
    view_base_mip: u32,
    view_mip_levels: u32,
    upload_bytes: &[u8],
    upload_copies: &[ComputeGuestImageCopy],
) -> Result<ComputeGuestImageResource, String> {
    if upload_bytes.is_empty() {
        return Err("compute guest sampled image has no upload bytes".to_string());
    }
    if is_cube && (is_3d || depth != 1 || width != height) {
        return Err(format!(
            "invalid compute guest cube image: extent={width}x{height}x{depth} 3d={is_3d} levels={mip_levels}"
        ));
    }
    let array_layers = if is_cube { 6 } else { 1 };
    let mip_levels = mip_levels.max(1);
    let max_mip_levels = u32::BITS - width.max(height).max(depth).max(1).leading_zeros();
    if (is_3d && mip_levels != 1)
        || mip_levels > max_mip_levels
        || view_mip_levels == 0
        || view_base_mip >= mip_levels
        || view_base_mip.saturating_add(view_mip_levels) > mip_levels
    {
        return Err(format!(
            "invalid compute guest mip storage/view: levels={mip_levels} view={view_base_mip}/{view_mip_levels} extent={width}x{height}x{depth} 3d={is_3d}"
        ));
    }
    if upload_copies.is_empty()
        || upload_copies.iter().any(|copy| {
            copy.mip_level >= mip_levels
                || copy.width != (width >> copy.mip_level).max(1)
                || copy.height != (height >> copy.mip_level).max(1)
                || copy.buffer_offset >= upload_bytes.len() as u64
        })
    {
        return Err("compute guest sampled image has invalid mip copy regions".to_string());
    }
    let upload = create_compute_buffer_allocation(
        device,
        mem_props,
        upload_bytes.len().max(16) as u64,
        Some(upload_bytes),
        vk::BufferUsageFlags::TRANSFER_SRC,
        None,
        false,
        true,
    )?;
    let info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: if is_3d {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        },
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth,
        },
        mip_levels,
        array_layers,
        flags: if is_cube {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        },
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let image = match unsafe { device.create_image(&info, None) } {
        Ok(image) => image,
        Err(error) => {
            upload.destroy(device);
            return Err(format!("create compute guest sampled image: {error:?}"));
        }
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let Some(memory_type_index) = find_memory_type(
        mem_props,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) else {
        unsafe { device.destroy_image(image, None) };
        upload.destroy(device);
        return Err("no DEVICE_LOCAL memory for compute guest sampled image".to_string());
    };
    let allocation = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: requirements.size,
        memory_type_index,
        ..Default::default()
    };
    let memory = match unsafe { device.allocate_memory(&allocation, None) } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_image(image, None) };
            upload.destroy(device);
            return Err(format!("allocate compute guest sampled image: {error:?}"));
        }
    };
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        upload.destroy(device);
        return Err(format!("bind compute guest sampled image: {error:?}"));
    }
    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if is_3d {
            vk::ImageViewType::TYPE_3D
        } else if is_cube {
            vk::ImageViewType::CUBE
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        components,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: view_base_mip,
            level_count: view_mip_levels,
            base_array_layer: 0,
            layer_count: array_layers,
        },
        ..Default::default()
    };
    let view = match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => view,
        Err(error) => {
            unsafe {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
            upload.destroy(device);
            return Err(format!(
                "create compute guest sampled image view: {error:?}"
            ));
        }
    };
    Ok(ComputeGuestImageResource {
        image: ComputeImageResource {
            image,
            view,
            memory,
            allocation_size: requirements.size,
            width,
            height,
            depth,
            is_3d,
            format,
            sampled: true,
        },
        upload,
        needs_upload: true,
        width,
        height,
        depth,
        mip_levels,
        copies: upload_copies
            .iter()
            .map(|copy| vk::BufferImageCopy {
                buffer_offset: copy.buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: copy.mip_level,
                    base_array_layer: 0,
                    layer_count: array_layers,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width: copy.width,
                    height: copy.height,
                    depth: if is_3d { depth } else { 1 },
                },
            })
            .collect(),
        layout: vk::ImageLayout::UNDEFINED,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        compute_buffer_binding_counts, compute_spirv_hash, validate_compute_local_size,
        BoundedResourcePool, ComputeOutputPoolKey, ComputePipelineKey, ComputeStorageFormat,
        ComputeTexelBuffer, ComputeTexelFormat,
    };
    use crate::texture::{ComponentType, SwizzleSource, TicEntry, TicFormat};
    use ash::vk;

    fn abgr8_storage_tic(component: ComponentType) -> TicEntry {
        TicEntry {
            format: TicFormat::A8B8G8R8,
            component_types: [component; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x1000,
            width: 8,
            height: 8,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        }
    }

    fn r8_storage_tic(component: ComponentType) -> TicEntry {
        let mut tic = abgr8_storage_tic(component);
        tic.format = TicFormat::R8;
        tic.component_types = [component; 4];
        tic
    }

    #[test]
    fn abgr8_unorm_storage_format_keeps_exact_vulkan_layout() {
        for component in [ComponentType::Unorm, ComponentType::UnormForceFp16] {
            let format = ComputeStorageFormat::from_tic(&abgr8_storage_tic(component));
            assert_eq!(format, Some(ComputeStorageFormat::Abgr8Unorm));
        }
        assert_eq!(
            ComputeStorageFormat::Abgr8Unorm.vk_format(),
            vk::Format::A8B8G8R8_UNORM_PACK32
        );
        assert_eq!(ComputeStorageFormat::Abgr8Unorm.bytes_per_pixel(), 4);
        assert_eq!(
            ComputeStorageFormat::from_tic(&abgr8_storage_tic(ComponentType::Uint)),
            Some(ComputeStorageFormat::Abgr8Uint)
        );
        assert_eq!(
            ComputeStorageFormat::Abgr8Uint.vk_format(),
            vk::Format::A8B8G8R8_UINT_PACK32
        );
        assert_eq!(ComputeStorageFormat::Abgr8Uint.bytes_per_pixel(), 4);
        for component in [ComponentType::Unorm, ComponentType::UnormForceFp16] {
            assert_eq!(
                ComputeStorageFormat::from_tic(&r8_storage_tic(component)),
                Some(ComputeStorageFormat::R8Unorm)
            );
        }
        assert_eq!(
            ComputeStorageFormat::R8Unorm.vk_format(),
            vk::Format::R8_UNORM
        );
        assert_eq!(ComputeStorageFormat::R8Unorm.bytes_per_pixel(), 1);
    }

    #[test]
    fn bounded_pool_reuses_only_exact_keys() {
        let mut pool = BoundedResourcePool::new(4, 64);
        assert!(pool.insert(16u64, "sixteen", 16).is_empty());
        assert_eq!(pool.take(&8), None);
        assert_eq!(pool.take(&16), Some("sixteen"));
        assert_eq!(pool.retained_bytes, 0);
    }

    #[test]
    fn output_pool_key_requires_exact_extent_and_format() {
        let base = ComputeOutputPoolKey {
            width: 1600,
            height: 900,
            depth: 1,
            is_3d: false,
            format: vk::Format::R16_SFLOAT,
            sampled: false,
        };
        assert_eq!(base, base);
        assert_ne!(base, ComputeOutputPoolKey { width: 800, ..base });
        assert_ne!(
            base,
            ComputeOutputPoolKey {
                height: 450,
                ..base
            }
        );
        assert_ne!(base, ComputeOutputPoolKey { depth: 4, ..base });
        assert_ne!(
            base,
            ComputeOutputPoolKey {
                is_3d: true,
                ..base
            }
        );
        assert_ne!(
            base,
            ComputeOutputPoolKey {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..base
            }
        );
        assert_ne!(
            base,
            ComputeOutputPoolKey {
                sampled: true,
                ..base
            }
        );
    }

    #[test]
    fn bounded_pool_evicts_fifo_for_item_and_byte_limits() {
        let mut pool = BoundedResourcePool::new(2, 20);
        assert!(pool.insert(1u8, "one", 8).is_empty());
        assert!(pool.insert(2u8, "two", 8).is_empty());
        assert_eq!(pool.insert(3u8, "three", 16), vec!["one", "two"]);
        assert_eq!(pool.retained_bytes, 16);
        assert_eq!(pool.take(&1), None);
        assert_eq!(pool.take(&3), Some("three"));
    }

    #[test]
    fn bounded_pool_rejects_an_oversized_item_without_disturbing_cache() {
        let mut pool = BoundedResourcePool::new(2, 16);
        assert!(pool.insert(1u8, "kept", 8).is_empty());
        assert_eq!(pool.insert(2u8, "oversized", 17), vec!["oversized"]);
        assert_eq!(pool.retained_bytes, 8);
        assert_eq!(pool.take(&1), Some("kept"));
    }

    #[test]
    fn compute_local_size_enforces_axis_and_invocation_limits() {
        assert_eq!(
            validate_compute_local_size([64, 2, 1], [1024, 1024, 64], 1024),
            Ok(128)
        );
        assert!(
            validate_compute_local_size([0, 1, 1], [1024, 1024, 64], 1024)
                .unwrap_err()
                .contains("zero component")
        );
        assert!(
            validate_compute_local_size([1025, 1, 1], [1024, 1024, 64], 1024)
                .unwrap_err()
                .contains("per-axis")
        );
        assert!(
            validate_compute_local_size([64, 64, 1], [1024, 1024, 64], 1024)
                .unwrap_err()
                .contains("4096 invocations")
        );
    }

    #[test]
    fn compute_local_size_product_is_checked() {
        assert!(validate_compute_local_size(
            [u32::MAX, 2, 1],
            [u32::MAX, u32::MAX, u32::MAX],
            u32::MAX,
        )
        .unwrap_err()
        .contains("overflows"));
    }

    #[test]
    fn compute_buffer_limits_count_raw_and_typed_descriptors_separately() {
        let buffers = vec![
            ComputeTexelBuffer {
                bindings: (0..11).collect(),
                bytes: vec![0; 4],
                byte_len: 4,
                format: ComputeTexelFormat::R32Uint,
                raw: true,
                raw_storage_key: None,
                writable: true,
                requires_atomics: false,
            },
            ComputeTexelBuffer {
                bindings: vec![32, 33, 34],
                bytes: vec![0; 4],
                byte_len: 4,
                format: ComputeTexelFormat::R32Uint,
                raw: false,
                raw_storage_key: None,
                writable: true,
                requires_atomics: false,
            },
        ];
        assert_eq!(compute_buffer_binding_counts(&buffers), (11, 3));
    }

    #[test]
    fn compute_pipeline_key_includes_local_size_and_spirv() {
        let first_spirv = compute_spirv_hash(&[0x0723_0203, 1]);
        let second_spirv = compute_spirv_hash(&[0x0723_0203, 2]);
        assert_ne!(first_spirv, second_spirv);
        let base = ComputePipelineKey {
            program_key: 7,
            spirv_hash: first_spirv,
            local_size: [32, 1, 1],
            required_subgroup_size: 0,
            descriptors: Vec::new(),
        };
        assert_ne!(
            base,
            ComputePipelineKey {
                local_size: [64, 1, 1],
                ..base.clone()
            }
        );
        assert_ne!(
            base,
            ComputePipelineKey {
                spirv_hash: second_spirv,
                ..base.clone()
            }
        );
    }
}

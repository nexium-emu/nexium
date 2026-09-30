use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use nexium_gpu::compute::{
    ComputeDispatch, ComputeDispatchOutcome, ComputeDispatchResult, ComputeImageAlias,
    ComputeRawStorageKey, ComputeSampleType, ComputeSampledImage, ComputeSampledRt,
    ComputeStorageFormat, ComputeStorageImage, ComputeTexelBuffer, ComputeTexelFormat,
    ComputeUniformBuffer, ComputeUniformTexelBuffer,
};
use nexium_gpu::rt_cache::RtKey;
use nexium_gpu::texture::{
    block_linear_byte_size_3d, block_linear_mip_layout, texture_guest_size_bytes, ComponentType,
    SwizzleSource, TicEntry, TicFormat, TscEntry,
};
use nexium_shader::{ImageDimension, IrOp, TextureHandleOrigin};
use nexium_spirv::{
    ComputeDescriptor, ComputeDescriptorKind, ComputeImageResource, ComputeModule, ComputeOptions,
    ComputeResourceKind, TextureNumericType,
};
use sha2::{Digest, Sha256};

use super::kepler_compute::ComputeTextureState;
use crate::gpu::{vk_dispatch, GpuMappings};

const MAX_CODE_BYTES: usize = 0x1_0000;
const MAX_RESOURCE_BYTES: usize = 512 * 1024 * 1024;

static FRONTEND_CACHE_PROFILE_LOOKUPS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FRONTEND_CACHE_PROFILE_DIRECT_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FRONTEND_CACHE_PROFILE_INDIRECT_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FRONTEND_CACHE_PROFILE_MISSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FRONTEND_CACHE_PROFILE_CFG_BUILDS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static FRONTEND_CACHE_PROFILE_INDIRECT_VARIANT_INSERTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[derive(Debug)]
pub(crate) enum MaxwellComputeOutcome {
    Executed,
    Unsupported(String),
    SubmittedFailure(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ResourceAccess {
    Sampled,
    FilteredSample,
    Storage,
    Atomic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ResourceNeed {
    handle: TextureHandleOrigin,
    access: ResourceAccess,
    instruction_dimension: Option<ImageDimension>,
    referenced_components: u8,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedResource {
    metadata: ComputeImageResource,
    access: ResourceAccess,
    tic: TicEntry,
    tsc: Option<TscEntry>,
    null: bool,
}

#[derive(Clone, Copy, Debug)]
struct OutputTarget {
    resource_index: usize,
    binding: u32,
    tic: TicEntry,
    subresource: ImageSubresourceLayout,
    gpu_va: u64,
    cpu_addr: u64,
    guest_size: usize,
    format: ComputeStorageFormat,
}

struct PreparedWrite {
    target: OutputTarget,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct StorageWritebackPool {
    buffers: Vec<Vec<u8>>,
}

impl StorageWritebackPool {
    fn take(&mut self, size: usize) -> Vec<u8> {
        self.buffers.iter().position(|buffer| buffer.len() == size)
            .map(|index| self.buffers.swap_remove(index))
            .unwrap_or_default()
    }

    fn recycle(&mut self, buffer: Vec<u8>) {
        const MAX_BYTES: usize = 16 * 1024 * 1024;
        let retained: usize = self.buffers.iter().map(Vec::capacity).sum();
        if !buffer.is_empty() && self.buffers.len() < 4
            && buffer.capacity() <= MAX_BYTES.saturating_sub(retained)
        {
            self.buffers.push(buffer);
        }
    }
}

thread_local! {
    static STORAGE_WRITEBACK_POOL: std::cell::RefCell<StorageWritebackPool> =
        std::cell::RefCell::new(StorageWritebackPool::default());
}

impl Drop for PreparedWrite {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);
        let _ = STORAGE_WRITEBACK_POOL.try_with(|pool| pool.borrow_mut().recycle(bytes));
    }
}

#[derive(Clone, Copy, Debug)]
struct TexelTarget {
    resource_index: usize,
    binding: u32,
    gpu_va: u64,
    cpu_addr: u64,
    guest_size: usize,
    raw: bool,
    raw_storage_key: Option<ComputeRawStorageKey>,
}

#[derive(Clone, Copy, Debug)]
struct MappedComputeResource {
    binding: u32,
    kind: ComputeDescriptorKind,
    writable: bool,
    tic_gpu_va: u64,
    width: u32,
    height: u32,
    depth: u32,
    mip_level: Option<u32>,
    view_base_mip: u32,
    view_mip_levels: u32,
    gpu_va: u64,
    cpu_addr: u64,
    size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageSubresourceLayout {
    mip_level: u32,
    width: u32,
    height: u32,
    depth: u32,
    storage_width: u32,
    storage_height: u32,
    block_height_log2: u32,
    stride_alignment_log2: u32,
    guest_offset: usize,
    guest_size: usize,
}

struct PreparedTexelWrite {
    target: TexelTarget,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FrontendCacheKey {
    code_sha256: [u8; 32],
    cbuf_dependencies: Option<Arc<[FrontendCbufDependency]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FrontendCbufDependency {
    binding: u8,
    byte_offset: u32,
    value: Option<u32>,
}

struct FrontendPlan {
    cfg: nexium_shader::Cfg,
    needs: Vec<ResourceNeed>,
    storage_buffers: Vec<nexium_shader::StorageBufferAddr>,
    writable_storage_buffers: Vec<bool>,
}

#[derive(Default)]
struct FrontendPlanCache {
    plans: HashMap<FrontendCacheKey, Arc<FrontendPlan>>,
    dependent_keys: HashMap<[u8; 32], Vec<FrontendCacheKey>>,
    #[cfg(test)]
    build_attempts: usize,
}

impl FrontendPlanCache {
    fn lookup(
        &self,
        code_sha256: [u8; 32],
        cbufs: &[Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS],
        indirect_hits_enabled: bool,
    ) -> Option<(FrontendCacheKey, Arc<FrontendPlan>)> {
        if let Some(keys) = self.dependent_keys.get(&code_sha256) {
            if !indirect_hits_enabled {
                return None;
            }
            return keys.iter().find_map(|key| {
                let matches = key.cbuf_dependencies.as_ref()?.iter().all(|dependency| {
                    read_frontend_cbuf(cbufs, dependency.binding, dependency.byte_offset)
                        == dependency.value
                });
                if !matches {
                    return None;
                }
                self.plans
                    .get(key)
                    .map(|plan| (key.clone(), Arc::clone(plan)))
            });
        }
        let key = frontend_cache_key(code_sha256, None);
        self.plans.get(&key).map(|plan| (key, Arc::clone(plan)))
    }

    fn insert(
        &mut self,
        code_sha256: [u8; 32],
        mut dependencies: Vec<FrontendCbufDependency>,
        plan: Arc<FrontendPlan>,
    ) -> (FrontendCacheKey, Arc<FrontendPlan>) {
        dependencies.sort_unstable_by_key(|dependency| (dependency.binding, dependency.byte_offset));
        dependencies.dedup();
        let cbuf_dependent = !dependencies.is_empty();
        let key = frontend_cache_key(
            code_sha256,
            cbuf_dependent.then(|| Arc::from(dependencies)),
        );
        if let Some(existing) = self.plans.get(&key) {
            return (key, Arc::clone(existing));
        }
        if self.plans.len() >= MAX_TRANSLATION_CACHE_ENTRIES {
            if let Some(oldest) = self.plans.keys().next().cloned() {
                let evicted_code = oldest.code_sha256;
                self.plans.remove(&oldest);
                if let Some(keys) = self.dependent_keys.get_mut(&evicted_code) {
                    keys.retain(|candidate| *candidate != oldest);
                    if keys.is_empty() {
                        self.dependent_keys.remove(&evicted_code);
                    }
                }
            }
        }
        if cbuf_dependent {
            self.dependent_keys
                .entry(code_sha256)
                .or_default()
                .push(key.clone());
        }
        let plan = Arc::clone(self.plans.entry(key.clone()).or_insert(plan));
        if cbuf_dependent {
            profile_frontend_cache_indirect_variant_insert();
        }
        (key, plan)
    }

    fn record_build_attempt(&mut self) {
        #[cfg(test)]
        {
            self.build_attempts += 1;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedRawStorageBuffer {
    base: u64,
    size: usize,
}

fn raw_storage_buffer_pointer_va(
    descriptor_index: usize,
    descriptor: nexium_shader::StorageBufferAddr,
    resolved: &[ResolvedRawStorageBuffer],
) -> Result<Option<u64>, String> {
    let Some(indirect) = descriptor.indirect else {
        return Ok(None);
    };
    let parent_index = indirect.parent_buffer_index as usize;
    if parent_index >= descriptor_index {
        return Err(format!(
            "raw compute buffer {descriptor_index} has non-topological parent {parent_index}"
        ));
    }
    let parent = resolved.get(parent_index).ok_or_else(|| {
        format!("raw compute buffer {descriptor_index} parent {parent_index} is unresolved")
    })?;
    let pointer_end = (indirect.pointer_offset as usize)
        .checked_add(8)
        .ok_or_else(|| "raw compute buffer pointer range overflow".to_string())?;
    if pointer_end > parent.size {
        return Err(format!(
            "raw compute buffer {descriptor_index} pointer exceeds parent {parent_index}"
        ));
    }
    parent
        .base
        .checked_add(u64::from(indirect.pointer_offset))
        .map(Some)
        .ok_or_else(|| "raw compute buffer pointer address overflow".to_string())
}

fn resolve_raw_storage_buffer(
    descriptor_index: usize,
    descriptor: nexium_shader::StorageBufferAddr,
    cbufs: &[Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS],
    resolved: &[ResolvedRawStorageBuffer],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<ResolvedRawStorageBuffer, String> {
    let pointer_va = raw_storage_buffer_pointer_va(descriptor_index, descriptor, resolved)?;
    let (base, size) = if let Some(pointer_va) = pointer_va {
        let bytes = read_gpu_vec(
            mappings,
            mem_read,
            pointer_va,
            8,
            "raw storage buffer pointer",
        )?;
        (
            u64::from_le_bytes(bytes.try_into().unwrap()),
            descriptor.required_size.max(4) as usize,
        )
    } else {
        let cbuf = cbufs
            .get(descriptor.cbuf_binding as usize)
            .and_then(Option::as_deref)
            .ok_or_else(|| {
                format!(
                    "raw compute buffer {descriptor_index} references unavailable cbuf {}",
                    descriptor.cbuf_binding
                )
            })?;
        let offset = descriptor.cbuf_offset as usize;
        let raw = cbuf.get(offset..offset.saturating_add(12)).ok_or_else(|| {
            format!(
                "raw compute buffer {descriptor_index} descriptor at c[{}]:{:#x} is truncated",
                descriptor.cbuf_binding, descriptor.cbuf_offset
            )
        })?;
        (
            (u64::from(u32::from_le_bytes(raw[4..8].try_into().unwrap())) << 32)
                | u64::from(u32::from_le_bytes(raw[0..4].try_into().unwrap())),
            u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize,
        )
    };
    if base == 0 || size == 0 || size > MAX_RESOURCE_BYTES {
        return Err(format!(
            "raw compute buffer {descriptor_index} has invalid base/size {base:#x}/{size:#x}"
        ));
    }
    Ok(ResolvedRawStorageBuffer { base, size })
}

fn raw_storage_snapshot_size(
    declared_size: usize,
    descriptor: nexium_shader::StorageBufferAddr,
    writable: bool,
) -> usize {
    if !writable && !descriptor.has_dynamic_offset && descriptor.required_size != 0 {
        declared_size.min(descriptor.required_size as usize)
    } else {
        declared_size
    }
}

pub(crate) fn raw_storage_cache_key(
    mappings: &GpuMappings,
    gpu_va: u64,
    size: usize,
) -> Option<ComputeRawStorageKey> {
    let size = u64::try_from(size).ok()?;
    if size == 0 {
        return None;
    }
    let (cpu_addr, available) = mappings.cpu_range_for(gpu_va)?;
    if available < size {
        return None;
    }
    Some(ComputeRawStorageKey {
        mapping_epoch: mappings.mapping_epoch_for(gpu_va)?,
        nvmap_id: mappings.nvmap_id_for(gpu_va)?,
        gpu_va,
        cpu_addr,
        size,
    })
}

fn raw_storage_mapping_is_current(key: ComputeRawStorageKey, mappings: &GpuMappings) -> bool {
    mappings.mapping_epoch_for(key.gpu_va) == Some(key.mapping_epoch)
        && mappings.nvmap_id_for(key.gpu_va) == Some(key.nvmap_id)
        && mappings
            .cpu_range_for(key.gpu_va)
            .is_some_and(|(cpu_addr, available)| cpu_addr == key.cpu_addr && available >= key.size)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ModuleCacheKey {
    frontend: FrontendCacheKey,
    local_size: [u32; 3],
    vertex_memory_store: bool,
    local_memory_low_size: u32,
    local_memory_high_size: u32,
    local_memory_crs_size: u32,
    shared_memory_size: u32,
    texture_bound_cbuf: u8,
    cbuf_sizes: [u32; nexium_spirv::COMPUTE_CBUF_SLOTS],
    resources: Vec<ComputeImageResource>,
    num_storage_buffers: u32,
    big_warp: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct QmdLocalMemory {
    low_size: u32,
    high_size: u32,
    crs_size: u32,
}

fn qmd_local_memory(qmd: &[u32; 0x40]) -> QmdLocalMemory {
    const SIZE_MASK: u32 = 0x00ff_ffff;
    QmdLocalMemory {
        low_size: qmd[0x2d] & SIZE_MASK,
        high_size: qmd[0x2e] & SIZE_MASK,
        crs_size: qmd[0x2f] & SIZE_MASK,
    }
}

fn prepare_uniform_buffers(
    module: &ComputeModule,
    cbufs: &[Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS],
) -> Result<Vec<ComputeUniformBuffer>, String> {
    let mut uniform_buffers = Vec::new();
    for descriptor in module
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.kind == ComputeDescriptorKind::UniformBuffer)
    {
        let slot = nexium_spirv::compute_cbuf_slot_for_descriptor_binding(descriptor.binding)
            .ok_or_else(|| {
                format!(
                    "SPIR-V returned unknown compute cbuf descriptor binding {}",
                    descriptor.binding
                )
            })?;
        let required = module.cbuf_required_sizes[usize::from(slot)] as usize;
        let cbuf = cbufs[usize::from(slot)]
            .as_deref()
            .ok_or_else(|| format!("Maxwell compute requires unavailable cbuf {slot}"))?;
        if required == 0 || required > cbuf.len() {
            return Err(format!(
                "Maxwell compute cbuf {slot} requires {required:#x} bytes, but QMD provides {:#x}",
                cbuf.len()
            ));
        }
        uniform_buffers.push(ComputeUniformBuffer {
            binding: descriptor.binding,
            bytes: cbuf[..required].to_vec(),
        });
    }
    Ok(uniform_buffers)
}

fn is_image_resource_descriptor(descriptor: &ComputeDescriptor) -> bool {
    matches!(
        descriptor.kind,
        ComputeDescriptorKind::CombinedSampledImage
            | ComputeDescriptorKind::UniformTexelBuffer
            | ComputeDescriptorKind::StorageTexelBuffer
            | ComputeDescriptorKind::SampledImage
            | ComputeDescriptorKind::StorageImage
    )
}

const MAX_TRANSLATION_CACHE_ENTRIES: usize = 128;

#[derive(Clone)]
enum PendingComputeId {
    Ready(u64),
    Deferred(crossbeam::channel::Receiver<Option<u64>>),
}

impl PendingComputeId {
    fn preview(&self) -> u64 {
        match self {
            PendingComputeId::Ready(id) => *id,
            PendingComputeId::Deferred(_) => 0,
        }
    }

    fn wait(self) -> Option<u64> {
        match self {
            PendingComputeId::Ready(id) => Some(id),
            PendingComputeId::Deferred(rx) => {
                match rx.recv() {
                    Ok(id) => id,
                    Err(_) => {
                        log::error!("[compute-offload] dispatch channel disconnected before completion ID");
                        None
                    }
                }
            }
        }
    }
}

fn compute_alias_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_COMPUTE_ALIAS").ok().as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        )
    })
}

fn compute_output_rt_alias(
    target: &OutputTarget,
    mappings: &GpuMappings,
) -> Option<nexium_gpu::compute::ComputeOutputRtAlias> {
    let tic = &target.tic;
    if !crate::gpu::vk_dispatch::tic_can_alias_render_target_view(tic) {
        return None;
    }
    let samples = tic.sample_grid()?;
    let nvmap_id = mappings.nvmap_id_for(tic.gpu_va)?;
    let mapping_epoch = mappings.mapping_epoch_for(tic.gpu_va)?;
    let layers = if tic.texture_type == 2 {
        tic.depth.max(1)
    } else {
        1
    };
    let key = nexium_gpu::rt_cache::RtKey::with_cpu(
        nvmap_id,
        tic.width,
        tic.height,
        tic.gpu_va,
        mappings.cpu_address_for(tic.gpu_va).unwrap_or(0),
    )
    .with_mapping_epoch(mapping_epoch)
    .with_sample_grid(samples.width, samples.height)
    .with_volume_depth(layers);
    let key = crate::gpu::vk_dispatch::with_tic_rt_alias_layout(key, tic).with_guest_size_bytes(
        nexium_gpu::texture::texture_guest_size_bytes(tic, layers)
            .and_then(|size| u64::try_from(size).ok())
            .unwrap_or(0),
    );
    Some(nexium_gpu::compute::ComputeOutputRtAlias {
        binding: target.binding,
        key,
    })
}

fn compute_offload_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("NEXIUM_COMPUTE_OFFLOAD").ok();
        match value.as_deref() {
            Some("0") | Some("false") | Some("off") | Some("no") => false,
            Some("1") | Some("true") | Some("on") | Some("yes") => true,
            _ => true,
        }
    })
}

struct PendingComputeWriteback {
    id: PendingComputeId,
    program_gpu_va: u64,
    serial: u64,
    output_targets: Vec<OutputTarget>,
    texel_targets: Vec<TexelTarget>,
    landed: Option<crossbeam::channel::Receiver<()>>,
}

struct LandingRecord {
    id: PendingComputeId,
    program_gpu_va: u64,
    output_targets: Vec<OutputTarget>,
    texel_targets: Vec<TexelTarget>,
    done: crossbeam::channel::Sender<()>,
}

static COMPUTE_WRITEBACK_FAILED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn mark_compute_writeback_failed() {
    COMPUTE_WRITEBACK_FAILED.store(true, std::sync::atomic::Ordering::Release);
}

pub(crate) fn compute_landing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_COMPUTE_LANDING").ok().as_deref(),
            Some("0") | Some("false") | Some("off") | Some("no")
        )
    })
}

pub(crate) fn schedule_pending_landings(
    renderer: std::sync::Arc<nexium_gpu::Renderer>,
    memory: crate::gpu::GuestMemoryAccess,
) -> usize {
    let landing_records: Vec<LandingRecord> = {
        let mut pending = pending_writebacks()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        pending
            .iter_mut()
            .filter(|record| record.landed.is_none())
            .map(|record| {
                let (done, landed) = crossbeam::channel::bounded(1);
                record.landed = Some(landed);
                LandingRecord {
                    id: record.id.clone(),
                    program_gpu_va: record.program_gpu_va,
                    output_targets: record.output_targets.clone(),
                    texel_targets: record.texel_targets.clone(),
                    done,
                }
            })
            .collect()
    };
    let count = landing_records.len();
    if count == 0 {
        return 0;
    }
    crate::gpu::pusher::kickprof::count(crate::gpu::pusher::kickprof::KC_LANDING, count as u64);
    let job_renderer = std::sync::Arc::clone(&renderer);
    crate::gpu::completion::submit_renderer_completion(renderer, move || {
        land_records(&job_renderer, &memory, landing_records);
    });
    count
}

fn land_records(
    renderer: &nexium_gpu::Renderer,
    memory: &crate::gpu::GuestMemoryAccess,
    records: Vec<LandingRecord>,
) {
    let writer = memory.writer();
    for record in records {
        if let Some(writer) = writer.as_ref() {
            let mem_write = |address: u64, bytes: &[u8]| writer(address, bytes);
            if let Some(id) = record.id.wait() {
                match renderer.take_pending_compute(id) {
                    Ok(result) => {
                        let mappings = memory.read_mappings();
                        if let Err(error) = write_back_outputs(
                            result,
                            record.program_gpu_va,
                            &record.output_targets,
                            &record.texel_targets,
                            renderer,
                            &mappings,
                            &mem_write,
                        ) {
                            mark_compute_writeback_failed();
                            log::error!("[compute-landing] id={} {}", id, error);
                        }
                    }
                    Err(error) => {
                        mark_compute_writeback_failed();
                        log::error!("[compute-landing] id={} {}", id, error);
                    }
                }
            } else {
                mark_compute_writeback_failed();
                log::error!("[compute-landing] dispatch did not produce a completion ID");
            }
        } else {
            mark_compute_writeback_failed();
            log::error!("[compute-landing] guest memory writer unavailable; outputs dropped");
        }
        let _ = record.done.send(());
    }
    renderer.release_unreferenced_compute_raw_storage();
}

pub(crate) fn drain_landed_writebacks() -> Vec<PendingComputeWritebackSpan> {
    let mut pending = pending_writebacks()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let mut landed = Vec::new();
    let mut index = 0;
    while index < pending.len() {
        let done = pending[index]
            .landed
            .as_ref()
            .is_some_and(|landed| landed.try_recv().is_ok());
        if done {
            landed.push(pending.remove(index));
        } else {
            index += 1;
        }
    }
    if landed.is_empty() {
        return Vec::new();
    }
    PENDING_WRITEBACK_REVISION.fetch_add(1, std::sync::atomic::Ordering::Release);
    drop(pending);
    retire_compute_serials(
        landed
            .iter()
            .map(|record| record.serial)
            .filter(|serial| *serial != 0)
            .collect(),
    );
    pending_writeback_target_spans(&landed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingComputeWritebackSpan {
    pub(crate) dispatch_id: u64,
    pub(crate) serial: u64,
    pub(crate) resource_index: u32,
    pub(crate) binding: u32,
    pub(crate) raw: bool,
    pub(crate) gpu_va: u64,
    pub(crate) cpu_addr: u64,
    pub(crate) len: usize,
}

thread_local! {
    static RESOLVED_WRITEBACK_SPANS: RefCell<Vec<PendingComputeWritebackSpan>> = RefCell::new(Vec::new());
}

fn report_resolved_writeback_spans(spans: Vec<PendingComputeWritebackSpan>) {
    if spans.is_empty() {
        return;
    }
    RESOLVED_WRITEBACK_SPANS.with(|reported| reported.borrow_mut().extend(spans));
}

pub(crate) fn take_resolved_writeback_spans() -> Vec<PendingComputeWritebackSpan> {
    RESOLVED_WRITEBACK_SPANS.with(|reported| std::mem::take(&mut *reported.borrow_mut()))
}

fn pending_writebacks() -> &'static Mutex<Vec<PendingComputeWriteback>> {
    static PENDING: OnceLock<Mutex<Vec<PendingComputeWriteback>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(Vec::new()))
}

static PENDING_WRITEBACK_REVISION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

pub(crate) fn pending_writeback_revision() -> u64 {
    PENDING_WRITEBACK_REVISION.load(std::sync::atomic::Ordering::Acquire)
}

fn push_pending_writeback(record: PendingComputeWriteback) {
    let mut pending = pending_writebacks().lock().unwrap_or_else(|error| error.into_inner());
    pending.push(record);
    PENDING_WRITEBACK_REVISION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

fn retired_compute_serials() -> &'static Mutex<Vec<u64>> {
    static RETIRED: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    RETIRED.get_or_init(|| Mutex::new(Vec::new()))
}

fn retire_compute_serials(serials: Vec<u64>) {
    if serials.is_empty() {
        return;
    }
    retired_compute_serials()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .extend(serials);
}

pub(crate) fn take_retired_compute_serials() -> Vec<u64> {
    std::mem::take(
        &mut *retired_compute_serials()
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
    )
}

static NEXT_COMPUTE_SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_compute_serial() -> u64 {
    NEXT_COMPUTE_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn lazy_compute_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_NO_LAZY_COMPUTE").is_none())
}

pub(crate) fn has_pending_writebacks() -> bool {
    !pending_writebacks()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .is_empty()
}

pub(crate) fn pending_writeback_overlaps(gpu_va: u64, cpu_addr: u64, size: usize) -> bool {
    if size == 0 {
        return false;
    }
    let pending = pending_writebacks()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    pending.iter().any(|record| {
        record.output_targets.iter().any(|target| {
            mapped_resources_overlap(
                gpu_va,
                cpu_addr,
                size,
                target.gpu_va,
                target.cpu_addr,
                target.guest_size,
            )
        }) || record.texel_targets.iter().any(|target| {
            mapped_resources_overlap(
                gpu_va,
                cpu_addr,
                size,
                target.gpu_va,
                target.cpu_addr,
                target.guest_size,
            )
        })
    })
}

fn resident_raw_overlap_is_compatible(
    current: Option<ComputeRawStorageKey>,
    pending: Option<ComputeRawStorageKey>,
    resident: bool,
) -> bool {
    resident && current.is_some() && current == pending
}

fn pending_writeback_requires_raw_storage_resolution(
    gpu_va: u64,
    cpu_addr: u64,
    size: usize,
    key: Option<ComputeRawStorageKey>,
    resident: bool,
) -> bool {
    if size == 0 {
        return false;
    }
    let pending = pending_writebacks()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    pending.iter().any(|record| {
        record.output_targets.iter().any(|target| {
            mapped_resources_overlap(
                gpu_va,
                cpu_addr,
                size,
                target.gpu_va,
                target.cpu_addr,
                target.guest_size,
            )
        }) || record.texel_targets.iter().any(|target| {
            mapped_resources_overlap(
                gpu_va,
                cpu_addr,
                size,
                target.gpu_va,
                target.cpu_addr,
                target.guest_size,
            ) && !resident_raw_overlap_is_compatible(key, target.raw_storage_key, resident)
        })
    })
}

fn pending_writeback_target_spans(
    records: &[PendingComputeWriteback],
) -> Vec<PendingComputeWritebackSpan> {
    let span_count = records
        .iter()
        .map(|record| record.output_targets.len() + record.texel_targets.len())
        .sum();
    let mut spans = Vec::with_capacity(span_count);
    for record in records {
        spans.extend(
            record
                .output_targets
                .iter()
                .map(|target| PendingComputeWritebackSpan {
                    dispatch_id: record.id.preview(),
                    serial: record.serial,
                    resource_index: u32::MAX,
                    binding: target.binding,
                    raw: false,
                    gpu_va: target.gpu_va,
                    cpu_addr: target.cpu_addr,
                    len: target.guest_size,
                }),
        );
        spans.extend(
            record
                .texel_targets
                .iter()
                .map(|target| PendingComputeWritebackSpan {
                    dispatch_id: record.id.preview(),
                    serial: record.serial,
                    resource_index: u32::try_from(target.resource_index).unwrap_or(u32::MAX),
                    binding: target.binding,
                    raw: target.raw,
                    gpu_va: target.gpu_va,
                    cpu_addr: target.cpu_addr,
                    len: target.guest_size,
                }),
        );
    }
    spans
}

pub(crate) fn pending_writeback_spans_snapshot() -> Vec<PendingComputeWritebackSpan> {
    let pending = pending_writebacks()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    pending_writeback_target_spans(&pending)
}

#[track_caller]
pub(crate) fn resolve_pending_writebacks_report(
    renderer: &nexium_gpu::Renderer,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> Vec<PendingComputeWritebackSpan> {
    let records: Vec<PendingComputeWriteback> = {
        let mut pending = pending_writebacks()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !pending.is_empty() {
            PENDING_WRITEBACK_REVISION.fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        pending.drain(..).collect()
    };
    let spans = pending_writeback_target_spans(&records);
    if records.is_empty() {
        renderer.release_unreferenced_compute_raw_storage();
        return spans;
    }
    let kp = crate::gpu::pusher::kickprof::start();
    let trace = std::env::var_os("NEXIUM_KC_RESOLVE_TRACE").is_some();
    let caller = std::panic::Location::caller();
    let record_count = records.len();
    let retired_serials: Vec<u64> = records
        .iter()
        .map(|record| record.serial)
        .filter(|serial| *serial != 0)
        .collect();
    for record in records {
        let wait_started = std::time::Instant::now();
        if let Some(landed) = &record.landed {
            if landed
                .recv_timeout(std::time::Duration::from_secs(3))
                .is_err()
            {
                mark_compute_writeback_failed();
                log::error!("[compute-landing] wait for a scheduled landing timed out");
            }
            if trace {
                log::warn!(
                    "[kc-resolve] caller={}:{} records={} landed wait_ms={:.3}",
                    caller.file(),
                    caller.line(),
                    record_count,
                    wait_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            continue;
        }
        let waited = record.id.wait();
        if trace {
            log::warn!(
                "[kc-resolve] caller={}:{} records={} id={:?} wait_ms={:.3} outputs={} texels={} bytes={}",
                caller.file(),
                caller.line(),
                record_count,
                waited,
                wait_started.elapsed().as_secs_f64() * 1000.0,
                record.output_targets.len(),
                record.texel_targets.len(),
                record
                    .output_targets
                    .iter()
                    .map(|target| target.guest_size)
                    .chain(record.texel_targets.iter().map(|target| target.guest_size))
                    .sum::<usize>()
            );
        }
        let Some(id) = waited else {
            mark_compute_writeback_failed();
            log::error!("[compute-lazy-resolve] dispatch did not produce a completion ID");
            continue;
        };
        match renderer.take_pending_compute(id) {
            Ok(result) => {
                if let Err(error) = write_back_outputs(
                    result,
                    record.program_gpu_va,
                    &record.output_targets,
                    &record.texel_targets,
                    renderer,
                    mappings,
                    mem_write,
                ) {
                    mark_compute_writeback_failed();
                    log::error!("[compute-lazy-writeback] id={} {}", id, error);
                }
            }
            Err(error) => {
                mark_compute_writeback_failed();
                log::error!("[compute-lazy-resolve] id={} {}", id, error);
            }
        }
    }
    renderer.release_unreferenced_compute_raw_storage();
    retire_compute_serials(retired_serials);
    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_RESOLVE, kp);
    spans
}

#[track_caller]
pub(crate) fn resolve_pending_writebacks(
    renderer: &nexium_gpu::Renderer,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) {
    report_resolved_writeback_spans(resolve_pending_writebacks_report(
        renderer, mappings, mem_write,
    ));
}

#[track_caller]
pub(crate) fn resolve_pending_writebacks_checked(
    renderer: &nexium_gpu::Renderer,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> bool {
    resolve_pending_writebacks(renderer, mappings, mem_write);
    !COMPUTE_WRITEBACK_FAILED.load(std::sync::atomic::Ordering::Acquire)
}

fn dispatch_requires_pending_writeback_resolution(
    raw_storage_overlap: bool,
    image_overlap: bool,
) -> bool {
    raw_storage_overlap || image_overlap
}

pub(crate) fn vertex_memory_store_code(code: &[u8]) -> Option<Vec<u8>> {
    type Entry = (u64, Vec<u8>, Option<Vec<u8>>);
    thread_local! {
        static CACHE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    }
    if code.len() > MAX_CODE_BYTES {
        return convert_vertex_memory_store_code(code);
    }
    let hash = xxhash_rust::xxh3::xxh3_64(code);
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, _, converted)) = cache
            .iter()
            .find(|(cached_hash, source, _)| *cached_hash == hash && source == code)
        {
            return converted.clone();
        }
        let converted = convert_vertex_memory_store_code(code);
        if cache.len() >= 16 {
            cache.remove(0);
        }
        cache.push((hash, code.to_vec(), converted.clone()));
        converted
    })
}

fn convert_vertex_memory_store_code(code: &[u8]) -> Option<Vec<u8>> {
    if !code.chunks_exact(8).enumerate().any(|(index, chunk)| {
        index % 4 != 0
            && nexium_shader::decode_one(u64::from_le_bytes(chunk.try_into().unwrap())).is_some_and(
                |decoded| {
                    matches!(
                        decoded.opcode,
                        nexium_shader::Opcode::SUST | nexium_shader::Opcode::STG
                    )
                },
            )
    }) {
        return None;
    }
    let cfg = nexium_shader::build_cfg(code);
    if cfg
        .blocks
        .iter()
        .any(|block| matches!(block.branch, nexium_shader::BranchKind::Indirect { .. }))
    {
        return None;
    }
    let mut stores = false;
    for inst in cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
    {
        match &inst.op {
            IrOp::Unimplemented {
                opcode: nexium_shader::Opcode::SUST,
                ..
            }
            | IrOp::StoreGlobal { .. } => stores = true,
            IrOp::LoadAttr { slot: 0x2fc } => {}
            IrOp::LoadAttr { .. }
            | IrOp::LoadAttrIndexed { .. }
            | IrOp::InterpAttr { .. }
            | IrOp::InterpAttrIndexed { .. }
            | IrOp::StoreAttr { .. }
            | IrOp::Unimplemented { .. }
            | IrOp::Shfl { .. }
            | IrOp::SubgroupVote { .. }
            | IrOp::FSwzAdd { .. }
            | IrOp::SubgroupLaneId
            | IrOp::SubgroupMask { .. }
            | IrOp::LocalInvocationId { .. }
            | IrOp::WorkgroupId { .. }
            | IrOp::LoadShared { .. }
            | IrOp::StoreShared { .. }
            | IrOp::WorkgroupBarrier
            | IrOp::SampleId => return None,
            _ => {}
        }
    }
    if !stores {
        return None;
    }
    let mut converted = code.to_vec();
    for (index, chunk) in converted.chunks_exact_mut(8).enumerate() {
        if index % 4 == 0
            || !cfg
                .blocks
                .iter()
                .any(|block| (block.start_offset..block.end_offset).contains(&(index * 8)))
        {
            continue;
        }
        let raw = u64::from_le_bytes(chunk.try_into().unwrap());
        if nexium_shader::decode_one(raw)
            .is_some_and(|decoded| decoded.opcode == nexium_shader::Opcode::S2R)
        {
            return None;
        }
        if nexium_shader::decode_one(raw)
            .is_some_and(|decoded| decoded.opcode == nexium_shader::Opcode::ALD)
            && (raw >> 20) & 0x3ff == 0x2fc
            && (raw >> 8) & 0xff == 0xff
        {
            let replacement = 0xf0c8_0000_0250_0000u64 | (raw & 0xf00ff);
            chunk.copy_from_slice(&replacement.to_le_bytes());
        }
    }
    Some(converted)
}

fn vertex_memory_store_group_size(vertex_count: u32) -> u32 {
    (1..=32)
        .rev()
        .find(|size| vertex_count % size == 0)
        .unwrap_or(1)
}

pub(crate) fn execute_vertex_memory_store(
    draw: &super::maxwell3d::DrawCall,
    sph: &[u8; 0x50],
    code: &[u8],
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> MaxwellComputeOutcome {
    let program = draw.shader_programs[1];
    let mut qmd = [0u32; 0x40];
    qmd[0x08] = program.address_lo.wrapping_add(0x50);
    qmd[0x0b] = u32::from(draw.sampler_binding == 1) << 30;
    let group_size = vertex_memory_store_group_size(draw.vertex_count);
    qmd[0x0c] = draw.vertex_count / group_size;
    qmd[0x0d] = 1 | (1 << 16);
    qmd[0x12] = group_size << 16;
    qmd[0x13] = 1 | (1 << 16);
    for slot in 0..8 {
        let (address, size) = draw.cbuf_binds[program.cbuf_group(0)][slot];
        if address != 0 && size != 0 {
            if address >> 40 != 0 || size > 0x1ffff {
                return MaxwellComputeOutcome::Unsupported("vertex cbuf exceeds QMD range".into());
            }
            qmd[0x14] |= 1 << slot;
            qmd[0x1d + slot * 2] = address as u32;
            qmd[0x1e + slot * 2] = ((address >> 32) as u32) | (size << 15);
        }
    }
    for index in 0..3 {
        qmd[0x2d + index] =
            u32::from_le_bytes(sph[4 + index * 4..8 + index * 4].try_into().unwrap()) & 0xffffff;
    }
    let texture = ComputeTextureState {
        tic_pool_gpu_va: draw.tic_pool_gpu_va,
        tic_limit: draw.tic_pool_limit,
        tsc_pool_gpu_va: draw.tsc_pool_gpu_va,
        tsc_limit: draw.tsc_pool_limit,
        tex_cb_index: draw.bindless_texture_const_buffer_slot,
    };
    match prepare_and_execute(
        &qmd,
        draw.program_region_gpu_va,
        texture,
        Some(renderer),
        mappings,
        mem_read,
        mem_write,
        &|_, _| None,
        Some(code),
    ) {
        Ok(()) => {
            resolve_pending_writebacks(renderer, mappings, mem_write);
            MaxwellComputeOutcome::Executed
        }
        Err(ExecuteError::Unsupported(reason)) => {
            if reason.contains("LoadGlobal") || reason.contains("StoreGlobal") {
                let mut cpu_qmd = qmd;
                cpu_qmd[0x0c] = draw.vertex_count;
                cpu_qmd[0x12] = 1 << 16;
                let cpu_texture = super::compute_cpu::ComputeTextureState {
                    tic_pool_gpu_va: texture.tic_pool_gpu_va,
                    tic_limit: texture.tic_limit,
                    tsc_pool_gpu_va: texture.tsc_pool_gpu_va,
                    tsc_limit: texture.tsc_limit,
                    tex_cb_index: texture.tex_cb_index,
                };
                if super::compute_cpu::try_execute_with_code(
                    &cpu_qmd,
                    draw.program_region_gpu_va,
                    cpu_texture,
                    Some(renderer.as_ref()),
                    mappings,
                    mem_read,
                    mem_write,
                    0,
                    Some(code),
                ) {
                    return MaxwellComputeOutcome::Executed;
                }
            }
            MaxwellComputeOutcome::Unsupported(reason)
        }
        Err(ExecuteError::Submitted(reason)) => MaxwellComputeOutcome::SubmittedFailure(reason),
    }
}

pub(super) fn try_execute(
    qmd: &[u32; 0x40],
    code_base: u64,
    texture: ComputeTextureState,
    renderer: Option<&std::sync::Arc<nexium_gpu::Renderer>>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    content_key: &dyn Fn(u64, usize) -> Option<u64>,
) -> MaxwellComputeOutcome {
    match prepare_and_execute(
        qmd,
        code_base,
        texture,
        renderer,
        mappings,
        mem_read,
        mem_write,
        content_key,
        None,
    ) {
        Ok(()) => MaxwellComputeOutcome::Executed,
        Err(ExecuteError::Unsupported(reason)) => MaxwellComputeOutcome::Unsupported(reason),
        Err(ExecuteError::Submitted(reason)) => MaxwellComputeOutcome::SubmittedFailure(reason),
    }
}

#[derive(Debug)]
enum ExecuteError {
    Unsupported(String),
    Submitted(String),
}

impl From<String> for ExecuteError {
    fn from(value: String) -> Self {
        Self::Unsupported(value)
    }
}

fn prepare_and_execute(
    qmd: &[u32; 0x40],
    code_base: u64,
    texture: ComputeTextureState,
    renderer: Option<&std::sync::Arc<nexium_gpu::Renderer>>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    content_key: &dyn Fn(u64, usize) -> Option<u64>,
    code_override: Option<&[u8]>,
) -> Result<(), ExecuteError> {
    let kp_frontend = crate::gpu::pusher::kickprof::start();
    let group_count = [qmd[0x0c] & 0x7fff_ffff, qmd[0x0d] & 0xffff, qmd[0x0d] >> 16];
    let local_size = [qmd[0x12] >> 16, qmd[0x13] & 0xffff, qmd[0x13] >> 16];
    let local_memory = qmd_local_memory(qmd);
    if group_count.contains(&0) || local_size.contains(&0) {
        return Err(format!(
            "zero Maxwell compute launch dimension: groups={group_count:?} local={local_size:?}"
        )
        .into());
    }

    let code_gpu = code_base
        .checked_add(qmd[0x08] as u64)
        .ok_or_else(|| "Maxwell compute code address overflow".to_string())?;
    if let Some(renderer) = renderer {
        if has_pending_writebacks() {
            let overlaps = (0..8u8).any(|slot| {
                let Some((gpu_va, size)) = qmd_cbuf_range(qmd, slot) else {
                    return false;
                };
                let Some((cpu_addr, _)) = mapped_range(mappings, gpu_va) else {
                    return false;
                };
                pending_writeback_overlaps(gpu_va, cpu_addr, size)
            });
            if overlaps {
                resolve_pending_writebacks(renderer, mappings, mem_write);
            }
        }
    }
    if let Some(renderer) = renderer {
        if has_pending_writebacks() {
            if let Some((cpu_addr, available)) = mapped_range(mappings, code_gpu) {
                let code_len = usize::try_from(available.min(MAX_CODE_BYTES as u64))
                    .unwrap_or(MAX_CODE_BYTES)
                    & !7;
                if pending_writeback_overlaps(code_gpu, cpu_addr, code_len) {
                    resolve_pending_writebacks(renderer, mappings, mem_write);
                }
            }
        }
    }
    let kp_snap = crate::gpu::pusher::kickprof::start();
    let code = match code_override {
        Some(code) => code.to_vec(),
        None => snapshot_code(mappings, mem_read, code_gpu)?,
    };
    let cbufs = snapshot_cbufs(qmd, mappings, mem_read);
    crate::gpu::pusher::kickprof::add_sized(
        crate::gpu::pusher::kickprof::KC_SNAP,
        kp_snap,
        code.len(),
    );
    if let Some(dir) = std::env::var_os("NEXIUM_DUMP_COMPUTE_SASS") {
        let path = std::path::Path::new(&dir).join(format!("cs_{:x}.bin", qmd[0x08]));
        if !path.exists() {
            let _ = std::fs::write(path, &code);
        }
    }
    let code_sha256 = if code_override.is_some() {
        Sha256::digest(&code).into()
    } else {
        memoized_code_sha256(code_gpu, &code)
    };
    let (frontend_key, frontend) = cached_frontend_plan(code_sha256, &code, &cbufs)?;
    let needs = &frontend.needs;
    let texture_bound_cbuf = u8::try_from(texture.tex_cb_index)
        .ok()
        .filter(|slot| *slot < 8)
        .ok_or_else(|| {
            format!(
                "Maxwell compute texture cbuf {} is outside the QMD cbuf table",
                texture.tex_cb_index
            )
        })?;
    let mut resolved = Vec::with_capacity(needs.len());
    for (index, need) in needs.iter().copied().enumerate() {
        let handle = resolve_handle(need.handle, texture_bound_cbuf, &cbufs)?;
        let (tic_index, tsc_index) = match need.access {
            ResourceAccess::FilteredSample => {
                let (tic_index, tsc_index) = split_sample_handle(qmd, handle);
                (tic_index, Some(tsc_index))
            }
            ResourceAccess::Sampled | ResourceAccess::Storage | ResourceAccess::Atomic => {
                (split_tic_handle(qmd, handle), None)
            }
        };
        if let Some(renderer) = renderer {
            if has_pending_writebacks() {
                let tic_gpu_va = tic_entry_gpu_va(texture, tic_index)?;
                let tic_overlap =
                    mapped_range(mappings, tic_gpu_va).is_some_and(|(cpu_addr, available)| {
                        available >= 32 && pending_writeback_overlaps(tic_gpu_va, cpu_addr, 32)
                    });
                let tsc_overlap = tsc_index
                    .map(|index| tsc_entry_gpu_va(texture, index))
                    .transpose()?
                    .is_some_and(|tsc_gpu_va| {
                        mapped_range(mappings, tsc_gpu_va).is_some_and(|(cpu_addr, available)| {
                            available >= 32 && pending_writeback_overlaps(tsc_gpu_va, cpu_addr, 32)
                        })
                    });
                if tic_overlap || tsc_overlap {
                    resolve_pending_writebacks(renderer, mappings, mem_write);
                }
            }
        }
        let (tic, null) = match read_tic(texture, tic_index, mappings, mem_read)? {
            Some(tic) => (tic, false),
            None => {
                if !matches!(
                    need.access,
                    ResourceAccess::Sampled | ResourceAccess::FilteredSample
                ) {
                    return Err(format!(
                        "null Maxwell TIC {tic_index} for {:?} access",
                        need.access
                    )
                    .into());
                }
                (null_tic(need.instruction_dimension), true)
            }
        };
        let tsc = tsc_index
            .map(|index| read_tsc(texture, index, mappings, mem_read))
            .transpose()?;
        let dimension = image_dimension(&tic)?;
        if let Some(instruction_dimension) = need.instruction_dimension {
            let compatible = instruction_dimension == dimension
                || (need.access == ResourceAccess::Sampled
                    && instruction_dimension == ImageDimension::D1
                    && dimension == ImageDimension::Buffer);
            if !compatible {
                return Err(format!(
                    "Maxwell compute resource {handle:#x} has {dimension}, but SASS expects {instruction_dimension}"
                )
                .into());
            }
        }
        let numeric_type = if need.access == ResourceAccess::Atomic {
            TextureNumericType::Uint
        } else {
            texture_numeric_type(&tic, need.referenced_components)
        };
        let kind = match (need.access, dimension) {
            (ResourceAccess::Sampled, ImageDimension::Buffer) => {
                ComputeResourceKind::UniformTexelBuffer
            }
            (ResourceAccess::Sampled, _) => ComputeResourceKind::SampledImage,
            (
                ResourceAccess::FilteredSample,
                ImageDimension::D2 | ImageDimension::D2Array | ImageDimension::D3 | ImageDimension::Cube,
            ) => ComputeResourceKind::CombinedSampledImage,
            (ResourceAccess::FilteredSample, other) => {
                return Err(
                    format!("filtered Maxwell compute sample has unsupported {other} TIC").into(),
                );
            }
            (ResourceAccess::Storage, ImageDimension::Buffer) => {
                storage_texel_format(&tic, numeric_type)?;
                ComputeResourceKind::StorageTexelBuffer
            }
            (ResourceAccess::Storage, _) => ComputeResourceKind::StorageImage,
            (ResourceAccess::Atomic, ImageDimension::Buffer) => {
                validate_storage_texel_buffer(&tic)?;
                ComputeResourceKind::StorageTexelBuffer
            }
            (ResourceAccess::Atomic, other) => {
                return Err(format!(
                    "Maxwell surface atomic has unsupported {other} TIC; only buffer views are implemented"
                )
                .into());
            }
        };
        let texel_format = match kind {
            ComputeResourceKind::UniformTexelBuffer => Some(texel_format(&tic, numeric_type)?),
            ComputeResourceKind::StorageTexelBuffer => {
                Some(storage_texel_format(&tic, numeric_type)?)
            }
            ComputeResourceKind::CombinedSampledImage
            | ComputeResourceKind::SampledImage
            | ComputeResourceKind::StorageImage => None,
        };
        resolved.push(ResolvedResource {
            metadata: ComputeImageResource {
                handle: need.handle,
                binding: index as u32 + 1 + frontend.storage_buffers.len() as u32,
                kind,
                dimension,
                numeric_type,
                texel_format: texel_format.map(ComputeTexelFormat::spirv_format),
            },
            access: need.access,
            tic,
            tsc,
            null,
        });
    }

    let cbuf_sizes = std::array::from_fn(|slot| {
        cbufs[slot]
            .as_ref()
            .map_or(0, |cbuf| u32::try_from(cbuf.len()).unwrap_or(u32::MAX))
    });
    let options = ComputeOptions {
        local_size,
        vertex_memory_store: code_override.is_some(),
        local_memory_low_size: local_memory.low_size,
        local_memory_high_size: local_memory.high_size,
        local_memory_crs_size: local_memory.crs_size,
        shared_memory_size: qmd[0x11] & 0x3ffff,
        texture_bound_cbuf,
        cbuf_sizes,
        num_storage_buffers: frontend.storage_buffers.len() as u32,
        resources: resolved.iter().map(|resource| resource.metadata).collect(),
        big_warp: nexium_common::gpu_caps::big_warp(),
    };
    let module_key = ModuleCacheKey {
        frontend: frontend_key,
        local_size,
        vertex_memory_store: options.vertex_memory_store,
        local_memory_low_size: options.local_memory_low_size,
        local_memory_high_size: options.local_memory_high_size,
        local_memory_crs_size: options.local_memory_crs_size,
        shared_memory_size: options.shared_memory_size,
        texture_bound_cbuf,
        cbuf_sizes,
        resources: options.resources.clone(),
        num_storage_buffers: options.num_storage_buffers,
        big_warp: options.big_warp,
    };
    let module = cached_compute_module(module_key, &frontend.cfg, &options)?;
    if module.texture_bound_cbuf != texture_bound_cbuf {
        return Err("SPIR-V compute manifest changed the texture-bound cbuf"
            .to_string()
            .into());
    }

    let unknown_cbufs = module.cbuf_bindings & !0xff;
    if unknown_cbufs != 0 {
        return Err(format!(
            "Maxwell compute references cbuf slots outside the QMD table ({unknown_cbufs:#x})"
        )
        .into());
    }
    for slot in 0..8u8 {
        if module.cbuf_bindings & (1 << slot) != 0 && cbufs[slot as usize].is_none() {
            return Err(format!("Maxwell compute requires unavailable cbuf {slot}").into());
        }
    }

    let uniform_buffers = prepare_uniform_buffers(&module, &cbufs)?;

    let renderer_arc = renderer.ok_or_else(|| "the Vulkan renderer is unavailable".to_string())?;
    let renderer: &nexium_gpu::Renderer = renderer_arc;
    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_FRONTEND, kp_frontend);
    let kp_sync = crate::gpu::pusher::kickprof::start();
    let offload = lazy_compute_enabled()
        && compute_offload_enabled()
        && crate::render_thread::maybe_render_thread().is_some();
    let drained = offload || vk_dispatch::sync_render_thread();
    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_SYNC, kp_sync);
    if !drained {
        return Err("could not drain the render thread for Maxwell compute"
            .to_string()
            .into());
    }
    let kp_resources = crate::gpu::pusher::kickprof::start();
    let mut resolved_storage_buffers = Vec::with_capacity(frontend.storage_buffers.len());
    let mut raw_storage_ranges = Vec::with_capacity(frontend.storage_buffers.len());
    let mut raw_storage_keys = Vec::with_capacity(frontend.storage_buffers.len());
    for (index, descriptor) in frontend.storage_buffers.iter().enumerate() {
        if has_pending_writebacks() {
            if let Some(pointer_va) =
                raw_storage_buffer_pointer_va(index, *descriptor, &resolved_storage_buffers)?
            {
                if let Some((cpu_addr, _)) = mapped_range(mappings, pointer_va) {
                    if pending_writeback_overlaps(pointer_va, cpu_addr, 8) {
                        resolve_pending_writebacks(renderer, mappings, mem_write);
                    }
                }
            }
        }
        let storage = resolve_raw_storage_buffer(
            index,
            *descriptor,
            &cbufs,
            &resolved_storage_buffers,
            mappings,
            mem_read,
        )?;
        resolved_storage_buffers.push(storage);
        let base = storage.base;
        let size = raw_storage_snapshot_size(
            storage.size, *descriptor, frontend.writable_storage_buffers[index],
        );
        let align = u64::from(descriptor.align.max(1));
        let aligned = base & !(align - 1);
        let slack = (base - aligned) as usize;
        let byte_len = size
            .checked_add(slack)
            .ok_or_else(|| "raw compute buffer size overflow".to_string())?;
        raw_storage_ranges.push((aligned, byte_len));
        raw_storage_keys.push(raw_storage_cache_key(mappings, aligned, byte_len));
    }
    let mut raw_storage_leases: Vec<_> = raw_storage_keys
        .iter()
        .map(|key| key.and_then(|key| renderer.pin_compute_raw_storage(key)))
        .collect();
    if has_pending_writebacks() {
        let image_overlap = resolved.iter().any(|resource| {
            let Ok(size) = resource_size(&resource.tic) else {
                return false;
            };
            if resource.tic.is_sparse {
                return sparse_texture_ranges(&resource.tic, mappings).is_ok_and(|ranges| {
                    ranges.into_iter().any(|(gpu, cpu, len)| {
                        cpu.is_some_and(|cpu| pending_writeback_overlaps(gpu, cpu, len))
                    })
                });
            }
            let Ok((view_tic, _)) = sampled_view_tic(&resource.tic) else {
                return false;
            };
            let Some((cpu_addr, _)) = mapped_range(mappings, view_tic.gpu_va) else {
                return false;
            };
            pending_writeback_overlaps(view_tic.gpu_va, cpu_addr, size)
        });
        let raw_storage_overlap =
            raw_storage_ranges
                .iter()
                .enumerate()
                .any(|(index, &(gpu_va, size))| {
                    let Some((cpu_addr, _)) = mapped_range(mappings, gpu_va) else {
                        return false;
                    };
                    let key = raw_storage_keys[index];
                    let resident = raw_storage_leases[index].is_some();
                    pending_writeback_requires_raw_storage_resolution(
                        gpu_va, cpu_addr, size, key, resident,
                    )
                });
        if dispatch_requires_pending_writeback_resolution(raw_storage_overlap, image_overlap) {
            resolve_pending_writebacks(renderer, mappings, mem_write);
            for (key, lease) in raw_storage_keys.iter().zip(&mut raw_storage_leases) {
                let refreshed = key.and_then(|key| renderer.pin_compute_raw_storage(key));
                *lease = refreshed;
            }
        }
    }
    let (image_aliases, overlapping_sampled) =
        validate_sampled_writable_aliases(&module, &resolved, mappings)?;

    let mut uniform_texel_buffers = Vec::new();
    let mut texel_buffers: Vec<ComputeTexelBuffer> = Vec::new();
    let mut texel_targets: Vec<TexelTarget> = Vec::new();
    let mut sampled_rts = Vec::new();
    let mut sampled_images = Vec::new();
    let mut outputs = Vec::new();
    let mut output_targets: Vec<OutputTarget> = Vec::new();

    for (index, &(aligned, byte_len)) in raw_storage_ranges.iter().enumerate() {
        let raw_storage_key = raw_storage_keys[index];
        let bytes =
            if raw_storage_leases[index].is_some() {
                Vec::new()
            } else {
                read_gpu_vec(mappings, mem_read, aligned, byte_len, "raw storage buffer")
                    .map_err(|reason| format!("{reason}; binding {index} {:?}", frontend.storage_buffers[index]))?
            };
        let resource_index = texel_buffers.len();
        let writable = frontend.writable_storage_buffers[index];
        let graphics_storage = match raw_storage_key {
            Some(key) => crate::gpu::graphics_storage::take_for_compute(key),
            None => {
                if let Some((cpu_addr, _)) = mapped_range(mappings, aligned) {
                    crate::gpu::graphics_storage::release_overlapping(
                        aligned,
                        cpu_addr,
                        byte_len as u64,
                        "compute raw storage",
                    );
                }
                None
            }
        };
        texel_buffers.push(ComputeTexelBuffer {
            bindings: vec![nexium_spirv::COMPUTE_STORAGE_BUFFER_BINDING_BASE + index as u32],
            bytes,
            byte_len,
            format: ComputeTexelFormat::R32Uint,
            raw: true,
            raw_storage_key,
            writable,
            requires_atomics: false,
            graphics_storage,
        });
        if writable {
            let (cpu_addr, available) = mapped_range(mappings, aligned)
                .ok_or_else(|| format!("raw compute output {aligned:#x} is unmapped"))?;
            if byte_len as u64 > available {
                return Err(format!(
                    "raw compute output mapping is short ({available:#x} < {byte_len:#x})"
                )
                .into());
            }
            texel_targets.push(TexelTarget {
                resource_index,
                binding: nexium_spirv::COMPUTE_STORAGE_BUFFER_BINDING_BASE + index as u32,
                gpu_va: aligned,
                cpu_addr,
                guest_size: byte_len,
                raw: true,
                raw_storage_key,
            });
        }
    }

    for descriptor in module
        .descriptors
        .iter()
        .filter(|descriptor| is_image_resource_descriptor(descriptor))
    {
        let resource = resolved
            .iter()
            .find(|resource| resource.metadata.binding == descriptor.binding)
            .ok_or_else(|| {
                format!(
                    "SPIR-V returned unknown Maxwell compute binding {}",
                    descriptor.binding
                )
            })?;
        if descriptor.handle != Some(resource.metadata.handle)
            || descriptor.dimension != Some(resource.metadata.dimension)
            || descriptor.numeric_type != Some(resource.metadata.numeric_type)
            || descriptor.texel_format != resource.metadata.texel_format
        {
            return Err(format!(
                "SPIR-V descriptor {} does not match its resolved TIC metadata",
                descriptor.binding
            )
            .into());
        }
        let expected_descriptor_kind = match resource.metadata.kind {
            ComputeResourceKind::UniformTexelBuffer => ComputeDescriptorKind::UniformTexelBuffer,
            ComputeResourceKind::StorageTexelBuffer => ComputeDescriptorKind::StorageTexelBuffer,
            ComputeResourceKind::CombinedSampledImage => {
                ComputeDescriptorKind::CombinedSampledImage
            }
            ComputeResourceKind::SampledImage => ComputeDescriptorKind::SampledImage,
            ComputeResourceKind::StorageImage => ComputeDescriptorKind::StorageImage,
        };
        if descriptor.kind != expected_descriptor_kind {
            return Err(format!(
                "SPIR-V descriptor {} has kind {:?}, expected {:?}",
                descriptor.binding, descriptor.kind, expected_descriptor_kind
            )
            .into());
        }
        match descriptor.kind {
            ComputeDescriptorKind::UniformBuffer | ComputeDescriptorKind::StorageBuffer => {
                return Err(format!(
                    "unexpected non-image descriptor {:?} at binding {}",
                    descriptor.kind, descriptor.binding
                )
                .into());
            }
            ComputeDescriptorKind::StorageTexelBuffer => {
                let requires_atomics = resource.access == ResourceAccess::Atomic;
                if requires_atomics {
                    validate_storage_texel_buffer(&resource.tic)?;
                }
                let format = storage_texel_format(&resource.tic, resource.metadata.numeric_type)?;
                let guest_size = resource_size(&resource.tic)?;
                let (cpu_addr, available) = mapped_range(mappings, resource.tic.gpu_va)
                    .ok_or_else(|| {
                        format!(
                            "storage texel buffer binding {} at {:#x} is unmapped",
                            descriptor.binding, resource.tic.gpu_va
                        )
                    })?;
                if guest_size as u64 > available {
                    return Err(format!(
                        "storage texel buffer binding {} mapping is short ({available:#x} < {guest_size:#x})",
                        descriptor.binding
                    )
                    .into());
                }

                if let Some(target) = texel_targets.iter().copied().find(|target| {
                    target.gpu_va == resource.tic.gpu_va
                        && target.cpu_addr == cpu_addr
                        && target.guest_size == guest_size
                }) {
                    let buffer = &mut texel_buffers[target.resource_index];
                    if buffer.format != format {
                        return Err(format!(
                            "storage texel buffer binding {} aliases binding {} with incompatible {:?}/{:?} views",
                            descriptor.binding, target.binding, format, buffer.format
                        )
                        .into());
                    }
                    buffer.bindings.push(descriptor.binding);
                    buffer.requires_atomics |= requires_atomics;
                } else {
                    if let Some(target) = texel_targets.iter().copied().find(|target| {
                        mapped_resources_overlap(
                            resource.tic.gpu_va,
                            cpu_addr,
                            guest_size,
                            target.gpu_va,
                            target.cpu_addr,
                            target.guest_size,
                        )
                    }) {
                        return Err(format!(
                            "storage texel buffer binding {} partially aliases binding {}",
                            descriptor.binding, target.binding
                        )
                        .into());
                    }
                    if let Some(target) = output_targets.iter().copied().find(|target| {
                        mapped_resources_overlap(
                            resource.tic.gpu_va,
                            cpu_addr,
                            guest_size,
                            target.gpu_va,
                            target.cpu_addr,
                            target.guest_size,
                        )
                    }) {
                        return Err(format!(
                            "storage texel buffer binding {} aliases storage image binding {}",
                            descriptor.binding, target.binding
                        )
                        .into());
                    }
                    let bytes = read_gpu_vec(
                        mappings,
                        mem_read,
                        resource.tic.gpu_va,
                        guest_size,
                        "storage texel buffer",
                    )?;
                    crate::gpu::graphics_storage::release_overlapping(
                        resource.tic.gpu_va,
                        cpu_addr,
                        guest_size as u64,
                        "compute texel buffer",
                    );
                    let resource_index = texel_buffers.len();
                    texel_buffers.push(ComputeTexelBuffer {
                        bindings: vec![descriptor.binding],
                        bytes,
                        byte_len: guest_size,
                        format,
                        raw: false,
                        raw_storage_key: None,
                        writable: true,
                        requires_atomics,
                        graphics_storage: None,
                    });
                    texel_targets.push(TexelTarget {
                        resource_index,
                        binding: descriptor.binding,
                        gpu_va: resource.tic.gpu_va,
                        cpu_addr,
                        guest_size,
                        raw: false,
                        raw_storage_key: None,
                    });
                }
            }
            ComputeDescriptorKind::UniformTexelBuffer => {
                let format = texel_format(&resource.tic, resource.metadata.numeric_type)?;
                let size = resource_size(&resource.tic)?;
                let bytes = read_gpu_vec(
                    mappings,
                    mem_read,
                    resource.tic.gpu_va,
                    size,
                    "uniform texel buffer",
                )?;
                uniform_texel_buffers.push(ComputeUniformTexelBuffer {
                    binding: descriptor.binding,
                    bytes,
                    format,
                });
            }
            ComputeDescriptorKind::CombinedSampledImage => {
                if resource.metadata.kind != ComputeResourceKind::CombinedSampledImage
                    || !matches!(
                        resource.metadata.dimension,
                        ImageDimension::D2 | ImageDimension::D2Array | ImageDimension::D3 | ImageDimension::Cube
                    )
                    || (resource.metadata.dimension == ImageDimension::Cube
                        && resource.tic.depth != 1)
                    || !resource.tic.normalized_coords
                {
                    return Err(format!(
                        "filtered sampled image binding {} is not a normalized 2D/3D/cube view: {:?}",
                        descriptor.binding, resource.tic
                    )
                    .into());
                }
                if resource.metadata.dimension == ImageDimension::Cube
                    && (resource.tic.width != resource.tic.height
                        || resource.tic.block_width_log2 != 0
                        || resource.tic.block_depth_log2 != 0
                        || !resource.tic.is_block_linear
                        || nexium_gpu::pitch_oracle::is_pitch_dst(resource.tic.gpu_va))
                {
                    return Err(format!(
                        "unsupported Maxwell compute cube layout at binding {}: {:?}",
                        descriptor.binding, resource.tic
                    )
                    .into());
                }
                validate_image_view(&resource.tic, false)?;
                let tsc = resource.tsc.ok_or_else(|| {
                    format!(
                        "filtered sampled image binding {} is missing its Maxwell sampler",
                        descriptor.binding
                    )
                })?;
                let size = resource_size(&resource.tic)?;
                if resource.null {
                    sampled_rts.push(ComputeSampledRt {
                        binding: descriptor.binding,
                        key: RtKey::with_cpu(0, 1, 1, 0, 0),
                        tic: resource.tic,
                        tsc,
                        sample_type: compute_sample_type(resource.metadata.numeric_type),
                        guest_bytes: Some(Arc::new(vec![0; size])),
                        guest_bytes_authoritative: true,
                        require_live: false,
                        content_key: None,
                    });
                    continue;
                }
                if resource.tic.is_sparse {
                    let (tic, bytes, key, content_key) =
                        snapshot_sparse_texture(&resource.tic, mappings, mem_read)?;
                    sampled_rts.push(ComputeSampledRt {
                        binding: descriptor.binding,
                        key,
                        tic,
                        tsc,
                        sample_type: compute_sample_type(resource.metadata.numeric_type),
                        guest_bytes: Some(bytes),
                        guest_bytes_authoritative: true,
                        require_live: false,
                        content_key: Some(content_key),
                    });
                    continue;
                }
                let (view_tic, layered) = sampled_view_tic(&resource.tic)?;
                let (guest_bytes, snapshot_identity) = read_compute_sampled_image(
                    renderer_arc, mappings, mem_read, content_key,
                    view_tic.gpu_va, size, "filtered sampled image",
                )?;
                let nvmap_id = mappings.nvmap_id_for(view_tic.gpu_va).ok_or_else(|| {
                    format!(
                        "filtered sampled image binding {} at {:#x} has no nvmap mapping",
                        descriptor.binding, view_tic.gpu_va
                    )
                })?;
                let (cpu_addr, _) = mapped_range(mappings, view_tic.gpu_va).ok_or_else(|| {
                    format!(
                        "filtered sampled image binding {} at {:#x} has no CPU mapping",
                        descriptor.binding, view_tic.gpu_va
                    )
                })?;
                let key =
                    texture_rt_key(mappings, nvmap_id, &view_tic, cpu_addr).ok_or_else(|| {
                        format!(
                            "filtered sampled image binding {} has unsupported layout identity",
                            descriptor.binding
                        )
                    })?;
                sampled_rts.push(ComputeSampledRt {
                    binding: descriptor.binding,
                    key,
                    tic: view_tic,
                    tsc,
                    sample_type: compute_sample_type(resource.metadata.numeric_type),
                    guest_bytes: Some(guest_bytes),
                    guest_bytes_authoritative: layered
                        || overlapping_sampled.contains(&descriptor.binding),
                    require_live: false,
                    content_key: snapshot_identity,
                });
            }
            ComputeDescriptorKind::SampledImage => {
                if !matches!(
                    resource.metadata.dimension,
                    ImageDimension::D2 | ImageDimension::D3
                ) {
                    return Err(format!(
                        "sampled image binding {} has unsupported dimension {}",
                        descriptor.binding, resource.metadata.dimension
                    )
                    .into());
                }
                validate_image_view(&resource.tic, false)?;
                let size = resource_size(&resource.tic)?;
                if resource.null {
                    sampled_images.push(ComputeSampledImage {
                        binding: descriptor.binding,
                        key: None,
                        tic: resource.tic,
                        sample_type: compute_sample_type(resource.metadata.numeric_type),
                        guest_bytes: Arc::new(vec![0; size]),
                        guest_bytes_authoritative: true,
                        require_live: false,
                        content_key: None,
                    });
                    continue;
                }
                if resource.tic.is_sparse {
                    let (tic, bytes, key, content_key) =
                        snapshot_sparse_texture(&resource.tic, mappings, mem_read)?;
                    sampled_images.push(ComputeSampledImage {
                        binding: descriptor.binding,
                        key: Some(key),
                        tic,
                        sample_type: compute_sample_type(resource.metadata.numeric_type),
                        guest_bytes: bytes,
                        guest_bytes_authoritative: true,
                        require_live: false,
                        content_key: Some(content_key),
                    });
                    continue;
                }
                let (view_tic, layered) = sampled_view_tic(&resource.tic)?;
                let (guest_bytes, snapshot_identity) = read_compute_sampled_image(
                    renderer_arc, mappings, mem_read, content_key,
                    view_tic.gpu_va, size, "sampled image",
                )?;
                let key = mappings.nvmap_id_for(view_tic.gpu_va).and_then(|nvmap_id| {
                    mapped_range(mappings, view_tic.gpu_va).and_then(|(cpu_addr, _)| {
                        texture_rt_key(mappings, nvmap_id, &view_tic, cpu_addr)
                    })
                });
                sampled_images.push(ComputeSampledImage {
                    binding: descriptor.binding,
                    key,
                    tic: view_tic,
                    sample_type: compute_sample_type(resource.metadata.numeric_type),
                    guest_bytes,
                    guest_bytes_authoritative: layered
                        || overlapping_sampled.contains(&descriptor.binding),
                    require_live: false,
                    content_key: snapshot_identity,
                });
            }
            ComputeDescriptorKind::StorageImage => {
                validate_image_view(&resource.tic, true)?;
                let format = storage_format(&resource.tic)?;
                if resource.tic.format.src_bpp() != format.bytes_per_pixel() {
                    return Err(format!(
                        "storage image binding {} changes guest texel size",
                        descriptor.binding
                    )
                    .into());
                }
                let subresource = storage_subresource(&resource.tic)?;
                let gpu_va = resource
                    .tic
                    .gpu_va
                    .checked_add(subresource.guest_offset as u64)
                    .ok_or_else(|| {
                        format!(
                            "storage image binding {} mip address overflow",
                            descriptor.binding
                        )
                    })?;
                let guest_size = subresource.guest_size;
                let (cpu_addr, available) = mapped_range(mappings, gpu_va).ok_or_else(|| {
                    format!(
                        "storage image binding {} mip {} at {gpu_va:#x} is unmapped",
                        descriptor.binding, subresource.mip_level
                    )
                })?;
                if guest_size as u64 > available {
                    return Err(format!(
                        "storage image binding {} mip {} mapping is short ({available:#x} < {guest_size:#x})",
                        descriptor.binding, subresource.mip_level
                    )
                    .into());
                }
                if let Some(target) = output_targets.iter().copied().find(|target| {
                    mapped_resources_overlap(
                        gpu_va,
                        cpu_addr,
                        guest_size,
                        target.gpu_va,
                        target.cpu_addr,
                        target.guest_size,
                    )
                }) {
                    return Err(format!(
                        "storage image binding {} mip {} aliases storage image binding {} mip {}",
                        descriptor.binding,
                        subresource.mip_level,
                        target.binding,
                        target.subresource.mip_level
                    )
                    .into());
                }
                if let Some(target) = texel_targets.iter().copied().find(|target| {
                    mapped_resources_overlap(
                        gpu_va,
                        cpu_addr,
                        guest_size,
                        target.gpu_va,
                        target.cpu_addr,
                        target.guest_size,
                    )
                }) {
                    return Err(format!(
                        "storage image binding {} aliases storage texel buffer binding {}",
                        descriptor.binding, target.binding
                    )
                    .into());
                }
                let initial_bytes = if subresource.mip_level == 0 {
                    if let Some(nvmap_id) = mappings.nvmap_id_for(gpu_va) {
                        let mut key_tic = resource.tic;
                        key_tic.gpu_va = gpu_va;
                        let live_seed = storage_rt_key(mappings, nvmap_id, &key_tic, cpu_addr)
                            .map(|key| renderer.readback_compute_storage_seed(key, format))
                            .transpose()?
                            .flatten();
                        match live_seed {
                            Some(live_bytes) => live_bytes,
                            None => {
                                let guest_bytes = read_gpu_vec(
                                    mappings,
                                    mem_read,
                                    gpu_va,
                                    guest_size,
                                    "storage image",
                                )?;
                                linearize_storage_image(
                                    &resource.tic,
                                    subresource,
                                    format,
                                    guest_bytes,
                                )?
                            }
                        }
                    } else {
                        let guest_bytes =
                            read_gpu_vec(mappings, mem_read, gpu_va, guest_size, "storage image")?;
                        linearize_storage_image(&resource.tic, subresource, format, guest_bytes)?
                    }
                } else {
                    let guest_bytes =
                        read_gpu_vec(mappings, mem_read, gpu_va, guest_size, "storage image mip")?;
                    linearize_storage_image(&resource.tic, subresource, format, guest_bytes)?
                };
                let resource_index = outputs.len();
                outputs.push(ComputeStorageImage {
                    binding: descriptor.binding,
                    width: subresource.width,
                    height: subresource.height,
                    depth: subresource.depth,
                    is_3d: resource.metadata.dimension == ImageDimension::D3,
                    format,
                    initial_bytes: Some(initial_bytes),
                });
                output_targets.push(OutputTarget {
                    resource_index,
                    binding: descriptor.binding,
                    tic: resource.tic,
                    subresource,
                    gpu_va,
                    cpu_addr,
                    guest_size,
                    format,
                });
            }
        }
    }

    let program_key = renderer_program_key(code_sha256, module.spirv_hash);
    let serial = if code_override.is_none()
        && lazy_compute_enabled()
        && compute_offload_enabled()
        && crate::render_thread::maybe_render_thread().is_some()
    {
        next_compute_serial()
    } else {
        0
    };
    let dispatch = ComputeDispatch {
        serial,
        program_key,
        spirv: Arc::clone(&module.spirv),
        spirv_hash: module.spirv_hash,
        group_count,
        local_size,
        shared_memory_size: qmd[0x11] & 0x3ffff,
        required_subgroup_size: if nexium_common::gpu_caps::big_warp() {
            None
        } else {
            Some(32)
        },
        requires_workgroup_explicit_layout: false,
        uniform_buffers,
        texel_buffers,
        uniform_texel_buffers,
        sampled_rts,
        sampled_images,
        outputs,
        image_aliases,
        output_rt_aliases: if compute_alias_enabled() {
            output_targets
                .iter()
                .filter_map(|target| compute_output_rt_alias(target, mappings))
                .collect()
        } else {
            Vec::new()
        },
    };

    static RESOURCE_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *RESOURCE_TRACE.get_or_init(|| {
        std::env::var_os("NEXIUM_COMPUTE_RESOURCE_TRACE").is_some_and(|value| {
            let value = value.to_string_lossy();
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
    }) {
        let sampled_rt_summary = dispatch
            .sampled_rts
            .iter()
            .map(|sampled| {
                format!(
                    "b{}:{}:{:?}",
                    sampled.binding,
                    sampled.key.label(),
                    sampled.tic.format
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let sampled_image_summary = dispatch
            .sampled_images
            .iter()
            .map(|sampled| {
                format!(
                    "b{}:{}:{:?}",
                    sampled.binding,
                    sampled
                        .key
                        .map(|key| key.label())
                        .unwrap_or_else(|| "none".to_string()),
                    sampled.tic.format
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let output_summary = output_targets
            .iter()
            .map(|target| {
                format!(
                    "b{}:{}:{:?}",
                    target.binding,
                    RtKey::with_cpu(
                        mappings.nvmap_id_for(target.gpu_va).unwrap_or_default(),
                        target.subresource.width,
                        target.subresource.height,
                        target.gpu_va,
                        target.cpu_addr,
                    )
                    .label(),
                    target.format
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        log::warn!(
            "[compute-resources] program={:#x} sampled_rts=[{}] sampled_images=[{}] outputs=[{}]",
            qmd[0x08],
            sampled_rt_summary,
            sampled_image_summary,
            output_summary,
        );
    }

    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_RESOURCES, kp_resources);
    let kp_exec = crate::gpu::pusher::kickprof::start();
    crate::gpu::pusher::kickprof::count(crate::gpu::pusher::kickprof::KC_DISPATCH, 1);
    if code_override.is_none() && lazy_compute_enabled() && compute_offload_enabled() {
        if let Some(rt) = crate::render_thread::maybe_render_thread() {
            let (id_tx, id_rx) = crossbeam::channel::bounded(1);
            let job_renderer = std::sync::Arc::clone(renderer_arc);
            rt.submit_named(
                "compute-dispatch",
                Box::new(move || {
                    let sent = match job_renderer.dispatch_compute_lazy(dispatch) {
                        ComputeDispatchOutcome::Submitted(id) => Some(id),
                        ComputeDispatchOutcome::Executed(_) => {
                            log::error!("[compute-offload] lazy dispatch executed synchronously; output dropped");
                            None
                        }
                        ComputeDispatchOutcome::Unsupported(reason)
                        | ComputeDispatchOutcome::FailedBeforeSubmit(reason)
                        | ComputeDispatchOutcome::SubmittedFailure(reason) => {
                            log::warn!("[compute-offload] dispatch failed: {}", reason);
                            None
                        }
                    };
                    drop(raw_storage_leases);
                    let _ = id_tx.send(sent);
                }),
            );
            push_pending_writeback(PendingComputeWriteback {
                    id: PendingComputeId::Deferred(id_rx),
                    program_gpu_va: code_gpu,
                    serial,
                    output_targets,
                    texel_targets,
                    landed: None,
                });
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_EXEC, kp_exec);
            return Ok(());
        }
    }
    let outcome = if code_override.is_none() && lazy_compute_enabled() {
        renderer.dispatch_compute_lazy(dispatch)
    } else {
        renderer.dispatch_compute_sync(dispatch)
    };
    drop(raw_storage_leases);
    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_EXEC, kp_exec);
    match outcome {
        ComputeDispatchOutcome::Executed(result) => {
            let kp_wb = crate::gpu::pusher::kickprof::start();
            let written = write_back_outputs(
                result,
                code_gpu,
                &output_targets,
                &texel_targets,
                renderer,
                mappings,
                mem_write,
            );
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_WB, kp_wb);
            renderer.release_unreferenced_compute_raw_storage();
            if written.is_err() {
                mark_compute_writeback_failed();
            }
            written.map_err(ExecuteError::Submitted)
        }
        ComputeDispatchOutcome::Submitted(id) => {
            static SUBMIT_TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *SUBMIT_TRACE
                .get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_SUBMIT_TRACE").is_some())
            {
                log::warn!("[compute-submit] id={} program={:#x}", id, qmd[0x08]);
            }
            push_pending_writeback(PendingComputeWriteback {
                    id: PendingComputeId::Ready(id),
                    program_gpu_va: code_gpu,
                    serial,
                    output_targets,
                    texel_targets,
                    landed: None,
                });
            Ok(())
        }
        ComputeDispatchOutcome::Unsupported(reason)
        | ComputeDispatchOutcome::FailedBeforeSubmit(reason) => {
            renderer.release_unreferenced_compute_raw_storage();
            Err(ExecuteError::Unsupported(reason))
        }
        ComputeDispatchOutcome::SubmittedFailure(reason) => {
            mark_compute_writeback_failed();
            renderer.release_unreferenced_compute_raw_storage();
            Err(ExecuteError::Submitted(reason))
        }
    }
}

fn snapshot_code(
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    gpu_va: u64,
) -> Result<Vec<u8>, String> {
    let (cpu_addr, available) = mapped_range(mappings, gpu_va)
        .ok_or_else(|| format!("Maxwell compute code at {gpu_va:#x} is unmapped"))?;
    let len = usize::try_from(available.min(MAX_CODE_BYTES as u64)).unwrap_or(MAX_CODE_BYTES) & !7;
    if len < 8 {
        return Err(format!(
            "Maxwell compute code mapping at {gpu_va:#x} is too short"
        ));
    }
    let mut code = vec![0u8; len];
    if mem_read(cpu_addr, &mut code) {
        return Ok(code);
    }
    let mut filled = 0;
    while filled < len {
        let address = gpu_va + filled as u64;
        let Some((cpu, available)) = mapped_range(mappings, address) else { break };
        let page_remaining = 0x1000 - (cpu & 0xfff) as usize;
        let count = (len - filled).min(page_remaining).min(available as usize) & !7;
        if count == 0 || !mem_read(cpu, &mut code[filled..filled + count]) {
            break;
        }
        filled += count;
    }
    if filled == 0 {
        return Err(format!("could not read Maxwell compute code at {gpu_va:#x}"));
    }
    code.truncate(filled);
    Ok(code)
}

fn snapshot_cbufs(
    qmd: &[u32; 0x40],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> [Option<Vec<u8>>; 8] {
    std::array::from_fn(|slot| snapshot_cbuf(qmd, slot as u8, mappings, mem_read).ok())
}

fn snapshot_cbuf(
    qmd: &[u32; 0x40],
    slot: u8,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<Vec<u8>, String> {
    if qmd[0x14] & (1 << slot) == 0 {
        return Err(format!("cbuf {slot} is disabled"));
    }
    let Some((gpu_va, size)) = qmd_cbuf_range(qmd, slot) else {
        return Err(format!("cbuf {slot} has an empty descriptor"));
    };
    read_gpu_vec(mappings, mem_read, gpu_va, size, "constant buffer")
}

fn qmd_cbuf_range(qmd: &[u32; 0x40], slot: u8) -> Option<(u64, usize)> {
    if qmd[0x14] & (1 << slot) == 0 {
        return None;
    }
    let base = 0x1d + slot as usize * 2;
    let lo = qmd[base] as u64;
    let hi_size = qmd[base + 1];
    let gpu_va = (((hi_size & 0xff) as u64) << 32) | lo;
    let size = ((hi_size >> 15) & 0x1ffff) as usize;
    if gpu_va == 0 || size == 0 {
        return None;
    }
    Some((gpu_va, size))
}

fn frontend_cache_key(
    code_sha256: [u8; 32],
    cbuf_dependencies: Option<Arc<[FrontendCbufDependency]>>,
) -> FrontendCacheKey {
    FrontendCacheKey {
        code_sha256,
        cbuf_dependencies,
    }
}

fn read_frontend_cbuf(
    cbufs: &[Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS],
    binding: u8,
    byte_offset: u32,
) -> Option<u32> {
    cbufs
        .get(binding as usize)
        .and_then(Option::as_deref)
        .and_then(|cbuf| cbuf_u32(cbuf, byte_offset))
}

fn indirect_frontend_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("NEXIUM_COMPUTE_INDIRECT_FRONTEND_CACHE")
                .ok()
                .as_deref(),
            Some("0")
                | Some("false")
                | Some("FALSE")
                | Some("off")
                | Some("OFF")
                | Some("no")
                | Some("NO")
        )
    })
}

fn indirect_frontend_cache_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NEXIUM_COMPUTE_INDIRECT_FRONTEND_CACHE_PROFILE")
                .ok()
                .as_deref(),
            Some("1")
                | Some("true")
                | Some("TRUE")
                | Some("on")
                | Some("ON")
                | Some("yes")
                | Some("YES")
        )
    })
}

fn profile_frontend_cache_lookup(indirect_hit: Option<bool>) {
    if !indirect_frontend_cache_profile_enabled() {
        return;
    }
    use std::sync::atomic::Ordering;

    match indirect_hit {
        Some(true) => {
            FRONTEND_CACHE_PROFILE_INDIRECT_HITS.fetch_add(1, Ordering::Relaxed);
        }
        Some(false) => {
            FRONTEND_CACHE_PROFILE_DIRECT_HITS.fetch_add(1, Ordering::Relaxed);
        }
        None => {
            FRONTEND_CACHE_PROFILE_MISSES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let lookups = FRONTEND_CACHE_PROFILE_LOOKUPS.fetch_add(1, Ordering::Relaxed) + 1;
    if lookups % 1024 == 0 {
        log::warn!(
            "[compute-frontend-cache] lookups={} direct_hits={} indirect_hits={} misses={} cfg_builds={} indirect_variant_inserts={}",
            lookups,
            FRONTEND_CACHE_PROFILE_DIRECT_HITS.load(Ordering::Relaxed),
            FRONTEND_CACHE_PROFILE_INDIRECT_HITS.load(Ordering::Relaxed),
            FRONTEND_CACHE_PROFILE_MISSES.load(Ordering::Relaxed),
            FRONTEND_CACHE_PROFILE_CFG_BUILDS.load(Ordering::Relaxed),
            FRONTEND_CACHE_PROFILE_INDIRECT_VARIANT_INSERTS.load(Ordering::Relaxed),
        );
    }
}

fn profile_frontend_cache_cfg_build() {
    if !indirect_frontend_cache_profile_enabled() {
        return;
    }
    FRONTEND_CACHE_PROFILE_CFG_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn profile_frontend_cache_indirect_variant_insert() {
    if !indirect_frontend_cache_profile_enabled() {
        return;
    }
    FRONTEND_CACHE_PROFILE_INDIRECT_VARIANT_INSERTS
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn memoized_code_sha256(code_gpu: u64, code: &[u8]) -> [u8; 32] {
    static MEMO: OnceLock<Mutex<HashMap<u64, (u64, usize, [u8; 32])>>> = OnceLock::new();
    let generation = nexium_gpu::tex_invalidate::region_gen_range(code_gpu, code.len() as u64);
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let memo = memo.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(&(cached_generation, cached_len, sha)) = memo.get(&code_gpu) {
            if cached_generation == generation && cached_len == code.len() {
                return sha;
            }
        }
    }
    let sha: [u8; 32] = Sha256::digest(code).into();
    let mut memo = memo.lock().unwrap_or_else(|error| error.into_inner());
    if memo.len() >= 512 && !memo.contains_key(&code_gpu) {
        memo.clear();
    }
    memo.insert(code_gpu, (generation, code.len(), sha));
    sha
}

fn cached_frontend_plan(
    code_sha256: [u8; 32],
    code: &[u8],
    cbufs: &[Option<Vec<u8>>; 8],
) -> Result<(FrontendCacheKey, Arc<FrontendPlan>), String> {
    static CACHE: OnceLock<Mutex<FrontendPlanCache>> = OnceLock::new();
    cached_frontend_plan_with_cache(
        CACHE.get_or_init(|| Mutex::new(FrontendPlanCache::default())),
        code_sha256,
        code,
        cbufs,
        indirect_frontend_cache_enabled(),
    )
}

fn cached_frontend_plan_with_cache(
    cache: &Mutex<FrontendPlanCache>,
    code_sha256: [u8; 32],
    code: &[u8],
    cbufs: &[Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS],
    indirect_hits_enabled: bool,
) -> Result<(FrontendCacheKey, Arc<FrontendPlan>), String> {
    {
        let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(hit) = cache.lookup(code_sha256, cbufs, indirect_hits_enabled) {
            profile_frontend_cache_lookup(Some(hit.0.cbuf_dependencies.is_some()));
            return Ok(hit);
        }
        profile_frontend_cache_lookup(None);
        cache.record_build_attempt();
    }

    profile_frontend_cache_cfg_build();
    let mut dependencies = Vec::new();
    let mut cfg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nexium_shader::build_compute_cfg_with_cbuf(code, |binding, byte_offset| {
            let value = read_frontend_cbuf(cbufs, binding, byte_offset);
            dependencies.push(FrontendCbufDependency {
                binding,
                byte_offset,
                value,
            });
            value
        })
    }))
    .map_err(|_| "Maxwell compute frontend panicked while building the CFG".to_string())?;
    if cfg.unimplemented != 0 {
        let mut details = Vec::new();
        for instruction in cfg
            .blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
        {
            if let IrOp::Unimplemented { opcode, raw } = &instruction.op {
                let detail = format!("{opcode:?}@{raw:#018x}");
                if !details.contains(&detail) {
                    details.push(detail);
                    if details.len() == 12 {
                        break;
                    }
                }
            }
        }
        return Err(format!(
            "Maxwell compute frontend left {} instruction(s) unimplemented{}",
            cfg.unimplemented,
            if details.is_empty() {
                String::new()
            } else {
                format!(": {}", details.join(", "))
            }
        ));
    }
    let storage_buffers = nexium_shader::collect_storage_buffers(&mut cfg);
    let mut writable_storage_buffers = vec![false; storage_buffers.len()];
    for instruction in cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
    {
        if let IrOp::StoreStorage { buffer_index, .. } | IrOp::StorageAtomic { buffer_index, .. } =
            instruction.op
        {
            if let Some(writable) = writable_storage_buffers.get_mut(buffer_index as usize) {
                *writable = true;
            }
        }
    }
    let needs = collect_resource_needs(&cfg)?;
    let plan = Arc::new(FrontendPlan {
        cfg,
        needs,
        storage_buffers,
        writable_storage_buffers,
    });
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    Ok(cache.insert(code_sha256, dependencies, plan))
}

struct CachedComputeModule {
    module: ComputeModule,
    spirv: Arc<[u32]>,
    spirv_hash: u64,
}

impl std::ops::Deref for CachedComputeModule {
    type Target = ComputeModule;

    fn deref(&self) -> &ComputeModule {
        &self.module
    }
}

fn cached_compute_module(
    key: ModuleCacheKey,
    cfg: &nexium_shader::Cfg,
    options: &ComputeOptions,
) -> Result<Arc<CachedComputeModule>, String> {
    static CACHE: OnceLock<Mutex<HashMap<ModuleCacheKey, Arc<CachedComputeModule>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(module) = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&key)
    {
        return Ok(Arc::clone(module));
    }

    let module = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nexium_spirv::emit_compute(cfg, options)
    }))
    .map_err(|_| "Maxwell compute SPIR-V emission panicked".to_string())?
    .map_err(|error| format!("Maxwell compute SPIR-V emission failed: {error}"))?;
    let spirv: Arc<[u32]> = Arc::from(module.words.as_slice());
    let spirv_hash = nexium_gpu::compute::compute_spirv_hash(&spirv);
    let module = Arc::new(CachedComputeModule {
        module,
        spirv,
        spirv_hash,
    });
    let mut cache = cache.lock().unwrap_or_else(|error| error.into_inner());
    if cache.len() >= MAX_TRANSLATION_CACHE_ENTRIES {
        if let Some(oldest) = cache.keys().next().cloned() {
            cache.remove(&oldest);
        }
    }
    Ok(Arc::clone(cache.entry(key).or_insert(module)))
}

fn collect_resource_needs(cfg: &nexium_shader::Cfg) -> Result<Vec<ResourceNeed>, String> {
    let mut needs = Vec::new();
    for block in &cfg.blocks {
        for instruction in &block.program.instructions {
            match &instruction.op {
                IrOp::TexelFetchHandle {
                    handle,
                    dimension,
                    component,
                    ..
                } => merge_need(
                    &mut needs,
                    *handle,
                    ResourceAccess::Sampled,
                    Some(*dimension),
                    1u8 << *component,
                )?,
                IrOp::SampleTexHandle {
                    handle,
                    dimension,
                    component,
                    ..
                } => {
                    let referenced_component =
                        1u8.checked_shl(u32::from(*component)).ok_or_else(|| {
                            format!("Maxwell compute filtered sample selects component {component}")
                        })?;
                    if referenced_component == 0 {
                        return Err(format!(
                            "Maxwell compute filtered sample selects component {component}"
                        ));
                    }
                    merge_need(
                        &mut needs,
                        *handle,
                        ResourceAccess::FilteredSample,
                        Some(*dimension),
                        referenced_component,
                    )?;
                }
                IrOp::GatherTex { tex_id, array, gather_component, .. } => merge_need(
                    &mut needs,
                    TextureHandleOrigin::Bound { cbuf_word_offset: *tex_id },
                    ResourceAccess::FilteredSample,
                    Some(if array.is_some() { ImageDimension::D2Array } else { ImageDimension::D2 }),
                    1u8 << *gather_component,
                )?,
                IrOp::TextureQueryDimension { handle, .. } => {
                    merge_need(&mut needs, *handle, ResourceAccess::Sampled, None, 0)?
                }
                IrOp::ImageWrite {
                    handle, dimension, ..
                } => merge_need(
                    &mut needs,
                    *handle,
                    ResourceAccess::Storage,
                    Some(*dimension),
                    0xf,
                )?,
                IrOp::ImageAtomic {
                    handle, dimension, ..
                } => merge_need(
                    &mut needs,
                    *handle,
                    ResourceAccess::Atomic,
                    Some(*dimension),
                    1,
                )?,
                _ => {}
            }
        }
    }
    Ok(needs)
}

fn merge_need(
    needs: &mut Vec<ResourceNeed>,
    handle: TextureHandleOrigin,
    access: ResourceAccess,
    dimension: Option<ImageDimension>,
    referenced_components: u8,
) -> Result<(), String> {
    let shares_storage_descriptor = |a: ResourceAccess, b: ResourceAccess| {
        matches!(a, ResourceAccess::Storage | ResourceAccess::Atomic)
            && matches!(b, ResourceAccess::Storage | ResourceAccess::Atomic)
    };
    if let Some(existing) = needs.iter_mut().find(|need| {
        need.handle == handle
            && (need.access == access || shares_storage_descriptor(need.access, access))
    }) {
        match (existing.instruction_dimension, dimension) {
            (Some(a), Some(b)) if a != b => {
                return Err(format!(
                    "Maxwell compute handle {handle:?} is used with both {a} and {b}"
                ));
            }
            (None, Some(dimension)) => existing.instruction_dimension = Some(dimension),
            _ => {}
        }
        if shares_storage_descriptor(existing.access, access)
            && (existing.access == ResourceAccess::Atomic || access == ResourceAccess::Atomic)
        {
            existing.access = ResourceAccess::Atomic;
        }
        existing.referenced_components |= referenced_components;
    } else {
        needs.push(ResourceNeed {
            handle,
            access,
            instruction_dimension: dimension,
            referenced_components,
        });
    }
    Ok(())
}

fn resolve_handle(
    origin: TextureHandleOrigin,
    texture_bound_cbuf: u8,
    cbufs: &[Option<Vec<u8>>; 8],
) -> Result<u32, String> {
    let read_word = |binding: u8, word_offset: u32| {
        cbufs
            .get(binding as usize)
            .and_then(Option::as_deref)
            .and_then(|cbuf| cbuf_u32(cbuf, word_offset.checked_mul(4)?))
            .ok_or_else(|| {
                format!(
                    "could not resolve Maxwell texture handle from c[{binding}][{:#x}]",
                    word_offset * 4
                )
            })
    };
    match origin {
        TextureHandleOrigin::Bound { cbuf_word_offset } => {
            read_word(texture_bound_cbuf, cbuf_word_offset)
        }
        TextureHandleOrigin::Bindless {
            cbuf_binding,
            cbuf_word_offset,
            cbuf_secondary_word_offset,
        } => {
            let primary = read_word(cbuf_binding, cbuf_word_offset)?;
            let secondary = cbuf_secondary_word_offset
                .map(|offset| read_word(cbuf_binding, offset))
                .transpose()?
                .unwrap_or(0);
            Ok(primary | secondary)
        }
    }
}

fn split_tic_handle(qmd: &[u32; 0x40], handle: u32) -> u32 {
    if qmd[0x0b] & (1 << 30) != 0 {
        handle
    } else {
        handle & 0x000f_ffff
    }
}

fn split_sample_handle(qmd: &[u32; 0x40], handle: u32) -> (u32, u32) {
    if qmd[0x0b] & (1 << 30) != 0 {
        (handle, handle)
    } else {
        (handle & 0x000f_ffff, handle >> 20)
    }
}

fn read_tic(
    texture: ComputeTextureState,
    index: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<Option<TicEntry>, String> {
    let gpu_va = tic_entry_gpu_va(texture, index)?;
    let raw = read_gpu_vec(mappings, mem_read, gpu_va, 32, "TIC")?;
    if raw.iter().all(|&byte| byte == 0) {
        return Ok(None);
    }
    TicEntry::parse(&raw)
        .map(Some)
        .ok_or_else(|| format!("could not parse Maxwell TIC {index} raw={raw:02x?}"))
}

fn null_tic(dimension: Option<ImageDimension>) -> TicEntry {
    TicEntry {
        format: TicFormat::A8B8G8R8,
        component_types: [ComponentType::Unorm; 4],
        swizzle: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
        gpu_va: 0,
        width: 1,
        height: 1,
        block_width_log2: 0,
        block_height_log2: 0,
        block_depth_log2: 0,
        tile_width_spacing: 0,
        pitch_bytes: 0,
        is_block_linear: true,
        texture_type: match dimension {
            Some(ImageDimension::Cube) => 3,
            Some(ImageDimension::D3) => 2,
            Some(ImageDimension::D1) => 0,
            _ => 1,
        },
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

fn tic_entry_gpu_va(texture: ComputeTextureState, index: u32) -> Result<u64, String> {
    if texture.tic_pool_gpu_va == 0 || index > texture.tic_limit {
        return Err(format!(
            "Maxwell TIC index {index} is outside pool limit {}",
            texture.tic_limit
        ));
    }
    texture
        .tic_pool_gpu_va
        .checked_add(index as u64 * 32)
        .ok_or_else(|| "Maxwell TIC address overflow".to_string())
}

fn read_tsc(
    texture: ComputeTextureState,
    index: u32,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<TscEntry, String> {
    let gpu_va = tsc_entry_gpu_va(texture, index)?;
    let raw = read_gpu_vec(mappings, mem_read, gpu_va, 32, "TSC")?;
    TscEntry::parse(&raw).ok_or_else(|| format!("could not parse Maxwell TSC {index}"))
}

fn tsc_entry_gpu_va(texture: ComputeTextureState, index: u32) -> Result<u64, String> {
    if texture.tsc_pool_gpu_va == 0 || index > texture.tsc_limit {
        return Err(format!(
            "Maxwell TSC index {index} is outside pool limit {}",
            texture.tsc_limit
        ));
    }
    texture
        .tsc_pool_gpu_va
        .checked_add(index as u64 * 32)
        .ok_or_else(|| "Maxwell TSC address overflow".to_string())
}

fn validate_pitch_linear_layout(tic: &TicEntry) -> Result<(), String> {
    if tic.pitch_bytes != 0
        && (!matches!(tic.texture_type, 1 | 7)
            || tic.is_block_linear
            || tic.depth != 1
            || tic.base_layer != 0)
    {
        return Err(format!(
            "unsupported Maxwell pitch-linear TIC layout: {tic:?}"
        ));
    }
    Ok(())
}

fn image_dimension(tic: &TicEntry) -> Result<ImageDimension, String> {
    validate_pitch_linear_layout(tic)?;
    Ok(if tic.is_buffer() {
        ImageDimension::Buffer
    } else {
        match tic.texture_type {
            0 => ImageDimension::D1,
            1 => ImageDimension::D2,
            5 => ImageDimension::D2Array,
            7 if tic.pitch_bytes != 0 => ImageDimension::D2,
            2 => ImageDimension::D3,
            3 => ImageDimension::Cube,
            other => {
                return Err(format!(
                    "unsupported Maxwell compute TIC texture type {other}"
                ));
            }
        }
    })
}

fn image_depth(tic: &TicEntry) -> u32 {
    if tic.texture_type == 2 {
        tic.depth.max(1)
    } else {
        1
    }
}

fn validate_image_view(tic: &TicEntry, storage: bool) -> Result<(), String> {
    validate_pitch_linear_layout(tic)?;
    let supported_type = matches!(tic.texture_type, 1 | 2)
        || (matches!(tic.texture_type, 3 | 5) && !storage)
        || (tic.texture_type == 7 && tic.pitch_bytes != 0);
    let layered_2d_view = matches!(tic.texture_type, 1 | 5)
        && tic.is_block_linear
        && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va)
        && tic.base_layer < tic.depth.max(1);
    if !supported_type || (tic.base_layer != 0 && !layered_2d_view) || (storage && tic.is_srgb) {
        return Err(format!(
            "unsupported Maxwell compute sampled/storage TIC view: {tic:?}"
        ));
    }

    if tic.res_min_mip_level > tic.res_max_mip_level || tic.res_max_mip_level > tic.max_mip_level {
        return Err(format!(
            "invalid Maxwell compute mip view min={} max={} storage_max={}: {tic:?}",
            tic.res_min_mip_level, tic.res_max_mip_level, tic.max_mip_level
        ));
    }

    let mip_levels = tic.mip_levels();
    let max_dimension = tic.width.max(tic.height).max(image_depth(tic));
    let max_physical_levels = u32::BITS - max_dimension.max(1).leading_zeros();
    if mip_levels > max_physical_levels {
        return Err(format!(
            "Maxwell compute mip count {mip_levels} exceeds {}x{}x{} image limit {max_physical_levels}",
            tic.width,
            tic.height,
            image_depth(tic)
        ));
    }

    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    if mip_levels > 1
        && (!matches!(tic.texture_type, 1 | 3 | 5)
            || !effective_block_linear
            || tic.block_width_log2 != 0
            || tic.block_depth_log2 != 0
            || block_linear_mip_layout(tic).is_none())
    {
        return Err(format!(
            "unsupported Maxwell compute multi-mip layout: {tic:?}"
        ));
    }
    Ok(())
}

fn image_view_subresources(
    tic: &TicEntry,
    storage: bool,
) -> Result<Vec<ImageSubresourceLayout>, String> {
    validate_image_view(tic, storage)?;
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    if effective_block_linear && tic.texture_type == 1 {
        let layout = block_linear_mip_layout(tic)
            .ok_or_else(|| "invalid Maxwell block-linear mip layout".to_string())?;
        let layer_offset = layout
            .layer_stride
            .checked_mul(tic.base_layer as usize)
            .ok_or_else(|| "Maxwell layered 2D view offset overflow".to_string())?;
        let base = tic.view_base_mip() as usize;
        let count = if storage {
            1
        } else {
            tic.view_mip_levels() as usize
        };
        let end = base
            .checked_add(count)
            .ok_or_else(|| "Maxwell compute mip range overflow".to_string())?;
        let levels = layout.levels.get(base..end).ok_or_else(|| {
            format!(
                "Maxwell compute mip view base={} levels={} exceeds storage {}",
                base,
                count,
                layout.levels.len()
            )
        })?;
        return Ok(levels
            .iter()
            .map(|level| ImageSubresourceLayout {
                mip_level: level.level,
                width: level.width,
                height: level.height,
                depth: 1,
                storage_width: level.storage_width,
                storage_height: level.storage_height,
                block_height_log2: level.block_height_log2,
                stride_alignment_log2: level.stride_alignment_log2,
                guest_offset: level.guest_offset.saturating_add(layer_offset),
                guest_size: level.guest_size,
            })
            .collect());
    }

    if !matches!(tic.texture_type, 3 | 5)
        && (tic.view_base_mip() != 0 || tic.view_mip_levels() != 1 || tic.mip_levels() != 1)
    {
        return Err(format!(
            "unsupported Maxwell compute non-block-linear mip view: {tic:?}"
        ));
    }
    let (storage_width, storage_height, _) = tic.format.storage_extent(tic.width, tic.height);
    Ok(vec![ImageSubresourceLayout {
        mip_level: 0,
        width: tic.width,
        height: tic.height,
        depth: image_depth(tic),
        storage_width,
        storage_height,
        block_height_log2: tic.block_height_log2,
        stride_alignment_log2: 6u32
            .saturating_sub(tic.format.src_bpp().checked_ilog2().unwrap_or(0)),
        guest_offset: 0,
        guest_size: resource_size(tic)?,
    }])
}

fn sampled_view_tic(tic: &TicEntry) -> Result<(TicEntry, bool), String> {
    if tic.texture_type != 1 || (tic.base_layer == 0 && tic.depth <= 1) {
        return Ok((*tic, false));
    }
    if tic.base_layer >= tic.depth.max(1) {
        return Err(format!(
            "Maxwell layered 2D view base_layer {} exceeds depth {}",
            tic.base_layer, tic.depth
        ));
    }
    let offset = if tic.base_layer == 0 {
        0
    } else {
        let layout = block_linear_mip_layout(tic)
            .ok_or_else(|| format!("invalid Maxwell layered 2D view layout: {tic:?}"))?;
        (layout.layer_stride as u64)
            .checked_mul(tic.base_layer as u64)
            .ok_or_else(|| "Maxwell layered 2D view offset overflow".to_string())?
    };
    let mut view = *tic;
    view.gpu_va = tic
        .gpu_va
        .checked_add(offset)
        .ok_or_else(|| "Maxwell layered 2D view address overflow".to_string())?;
    view.base_layer = 0;
    view.depth = 1;
    Ok((view, true))
}

fn storage_subresource(tic: &TicEntry) -> Result<ImageSubresourceLayout, String> {
    image_view_subresources(tic, true)?
        .into_iter()
        .next()
        .ok_or_else(|| "Maxwell storage image has no selected mip".to_string())
}

fn texture_numeric_type(tic: &TicEntry, referenced_components: u8) -> TextureNumericType {
    use nexium_gpu::texture::{SwizzleSource, TicFormat};

    let unorm = |component| {
        matches!(
            component,
            ComponentType::Unorm | ComponentType::UnormForceFp16
        )
    };
    if tic.format == TicFormat::G24R8
        && tic.component_types[0] == ComponentType::Uint
        && unorm(tic.component_types[1])
    {
        return if tic.swizzle.contains(&SwizzleSource::R) {
            TextureNumericType::Uint
        } else {
            TextureNumericType::Float
        };
    }
    if matches!(
        tic.format,
        TicFormat::Z24S8 | TicFormat::X8Z24 | TicFormat::S8Z24 | TicFormat::Z32 | TicFormat::Z16
    ) {
        return TextureNumericType::Float;
    }

    let mask = if referenced_components == 0 {
        1
    } else {
        referenced_components
    };
    let mut resolved = None;
    for output_component in 0..4usize {
        if mask & (1 << output_component) == 0 {
            continue;
        }
        let source_component = match tic.swizzle[output_component] {
            SwizzleSource::R => 0,
            SwizzleSource::G => 1,
            SwizzleSource::B => 2,
            SwizzleSource::A => 3,
            SwizzleSource::Zero | SwizzleSource::One => continue,
            SwizzleSource::Unknown(_) => return TextureNumericType::Float,
        };
        let current = match tic.component_types[source_component] {
            ComponentType::Uint => TextureNumericType::Uint,
            ComponentType::Sint => TextureNumericType::Sint,
            _ => return TextureNumericType::Float,
        };
        match resolved {
            Some(previous) if previous != current => return TextureNumericType::Float,
            None => resolved = Some(current),
            _ => {}
        }
    }
    resolved.unwrap_or(TextureNumericType::Float)
}

fn compute_sample_type(numeric_type: TextureNumericType) -> ComputeSampleType {
    match numeric_type {
        TextureNumericType::Float => ComputeSampleType::Float,
        TextureNumericType::Uint => ComputeSampleType::Uint,
        TextureNumericType::Sint => ComputeSampleType::Sint,
    }
}

fn texel_format(
    tic: &TicEntry,
    numeric_type: TextureNumericType,
) -> Result<ComputeTexelFormat, String> {
    ComputeTexelFormat::from_tic(tic, compute_sample_type(numeric_type)).ok_or_else(|| {
        format!(
            "unsupported Maxwell uniform texel buffer format {:?}/{:?} for {numeric_type:?}",
            tic.format, tic.component_types
        )
    })
}

fn storage_texel_format(
    tic: &TicEntry,
    numeric_type: TextureNumericType,
) -> Result<ComputeTexelFormat, String> {
    ComputeTexelFormat::from_tic(tic, compute_sample_type(numeric_type)).ok_or_else(|| {
        format!(
            "unsupported Maxwell storage texel buffer format {:?}/{:?} for {numeric_type:?}",
            tic.format, tic.component_types
        )
    })
}

fn texture_rt_key(
    mappings: &GpuMappings,
    nvmap_id: u32,
    tic: &TicEntry,
    cpu_addr: u64,
) -> Option<RtKey> {
    let samples = tic.sample_grid()?;
    let layers = if tic.texture_type == 2 {
        image_depth(tic)
    } else {
        1
    };
    let mut key = RtKey::with_cpu(nvmap_id, tic.width, tic.height, tic.gpu_va, cpu_addr)
        .with_mapping_epoch(mappings.mapping_epoch_for(tic.gpu_va)?)
        .with_sample_grid(samples.width, samples.height)
        .with_volume_depth(layers)
        .with_guest_size_bytes(
            texture_guest_size_bytes(tic, layers)
                .and_then(|size| u64::try_from(size).ok())
                .unwrap_or(0),
        );
    key = if tic.is_block_linear {
        key.with_block_linear_layout(
            tic.block_width_log2,
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
        )
    } else {
        key.with_pitch_linear_layout(tic.pitch_bytes)
    };
    Some(key)
}

fn storage_rt_key(
    mappings: &GpuMappings,
    nvmap_id: u32,
    tic: &TicEntry,
    cpu_addr: u64,
) -> Option<RtKey> {
    if tic.is_sparse
        || !tic.is_block_linear
        || tic.pitch_bytes != 0
        || tic.tile_width_spacing != 0
        || tic.block_depth_log2 != 0
        || tic.base_layer != 0
        || tic.max_mip_level != 0
        || tic.res_min_mip_level != 0
        || tic.res_max_mip_level != 0
        || tic.depth != 1
        || !(matches!(tic.texture_type, 1 | 7) || (tic.texture_type == 0 && tic.height == 1))
    {
        return None;
    }
    texture_rt_key(mappings, nvmap_id, tic, cpu_addr)
}

fn validate_storage_texel_buffer(tic: &TicEntry) -> Result<(), String> {
    if storage_texel_format(tic, TextureNumericType::Uint)? != ComputeTexelFormat::R32Uint {
        return Err("Maxwell surface atomics require an R32_UINT storage texel view".to_string());
    }
    Ok(())
}

fn storage_format(tic: &TicEntry) -> Result<ComputeStorageFormat, String> {
    ComputeStorageFormat::from_tic(tic).ok_or_else(|| {
        format!(
            "unsupported Maxwell storage image format {:?}/{:?}",
            tic.format, tic.component_types
        )
    })
}

fn linearize_storage_image(
    tic: &TicEntry,
    subresource: ImageSubresourceLayout,
    format: ComputeStorageFormat,
    guest_bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let tight_size = (subresource.width as usize)
        .checked_mul(subresource.height as usize)
        .and_then(|pixels| pixels.checked_mul(subresource.depth as usize))
        .and_then(|pixels| pixels.checked_mul(format.bytes_per_pixel()))
        .ok_or_else(|| "Maxwell storage image tight size overflow".to_string())?;
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let mut linear = if tic.pitch_bytes != 0 {
        nexium_gpu::texture::unpack_pitch_linear(&guest_bytes, tic, subresource.depth)
            .ok_or_else(|| "invalid Maxwell pitch-linear storage layout".to_string())?
    } else if effective_block_linear && tic.texture_type == 1 {
        nexium_gpu::texture::unswizzle_block_linear_strided(
            &guest_bytes,
            subresource.storage_width,
            subresource.storage_height,
            format.bytes_per_pixel(),
            subresource.block_height_log2,
            subresource.stride_alignment_log2,
        )
    } else if effective_block_linear {
        nexium_gpu::texture::unswizzle_block_linear_3d(
            &guest_bytes,
            subresource.width,
            subresource.height,
            subresource.depth,
            format.bytes_per_pixel(),
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
        )
    } else {
        guest_bytes
    };
    if linear.len() < tight_size {
        return Err(format!(
            "Maxwell storage image linearization is short ({:#x} < {tight_size:#x})",
            linear.len()
        ));
    }
    linear.truncate(tight_size);
    Ok(linear)
}

fn delinearize_storage_image(
    tic: &TicEntry,
    subresource: ImageSubresourceLayout,
    format: ComputeStorageFormat,
    linear_bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let tight_size = (subresource.width as usize)
        .checked_mul(subresource.height as usize)
        .and_then(|pixels| pixels.checked_mul(subresource.depth as usize))
        .and_then(|pixels| pixels.checked_mul(format.bytes_per_pixel()))
        .ok_or_else(|| "Maxwell storage image tight size overflow".to_string())?;
    if linear_bytes.len() != tight_size {
        return Err(format!(
            "Maxwell storage image readback is {:#x} bytes, expected {tight_size:#x}",
            linear_bytes.len()
        ));
    }
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let guest_bytes = if effective_block_linear && tic.texture_type == 1 {
        nexium_gpu::texture::swizzle_block_linear_strided(
            &linear_bytes,
            subresource.storage_width,
            subresource.storage_height,
            format.bytes_per_pixel(),
            subresource.block_height_log2,
            subresource.stride_alignment_log2,
        )
    } else if effective_block_linear {
        let buffer = STORAGE_WRITEBACK_POOL.with(|pool| {
            pool.borrow_mut().take(subresource.guest_size)
        });
        nexium_gpu::texture::swizzle_block_linear_3d_with_buffer(
            &linear_bytes,
            subresource.width,
            subresource.height,
            subresource.depth,
            format.bytes_per_pixel(),
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
            buffer,
        )
    } else {
        linear_bytes
    };
    if guest_bytes.len() != subresource.guest_size {
        return Err(format!(
            "Maxwell storage image guest layout is {:#x} bytes, expected {:#x}",
            guest_bytes.len(),
            subresource.guest_size
        ));
    }
    Ok(guest_bytes)
}

fn resource_size(tic: &TicEntry) -> Result<usize, String> {
    validate_pitch_linear_layout(tic)?;
    let depth = image_depth(tic) as usize;
    let (storage_width, storage_height, bpp) = tic.format.storage_extent(tic.width, tic.height);
    let tight_layer = (storage_width as usize)
        .checked_mul(storage_height as usize)
        .and_then(|texels| texels.checked_mul(bpp))
        .ok_or_else(|| "Maxwell compute resource size overflow".to_string())?;
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let size = if tic.is_buffer() {
        (tic.width as usize)
            .checked_mul(tic.format.src_bpp())
            .ok_or_else(|| "Maxwell texel-buffer size overflow".to_string())?
    } else if tic.pitch_bytes != 0 {
        tic.pitch_linear_layer_size()
            .ok_or_else(|| "invalid Maxwell pitch-linear resource size".to_string())?
    } else if effective_block_linear && tic.texture_type == 1 {
        texture_guest_size_bytes(tic, 1)
            .ok_or_else(|| "invalid Maxwell block-linear mip allocation".to_string())?
    } else if effective_block_linear && tic.texture_type == 5 {
        texture_guest_size_bytes(tic, tic.depth.max(1))
            .ok_or_else(|| "invalid Maxwell block-linear array allocation".to_string())?
    } else if effective_block_linear && tic.texture_type == 3 {
        texture_guest_size_bytes(tic, 6)
            .ok_or_else(|| "invalid Maxwell block-linear cube allocation".to_string())?
    } else if effective_block_linear {
        block_linear_byte_size_3d(
            storage_width,
            storage_height,
            image_depth(tic),
            bpp,
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
        )
    } else {
        let layers = match tic.texture_type { 3 => 6, 5 => tic.depth.max(1) as usize, _ => depth };
        tight_layer
            .checked_mul(layers)
            .ok_or_else(|| "Maxwell compute resource depth size overflow".to_string())?
    };
    if size == 0 || size > MAX_RESOURCE_BYTES {
        return Err(format!(
            "Maxwell compute resource size {size:#x} is outside the supported range"
        ));
    }
    Ok(size)
}

fn ranges_overlap(a: u64, a_size: usize, b: u64, b_size: usize) -> bool {
    if a_size == 0 || b_size == 0 {
        return false;
    }
    if a <= b {
        b - a < a_size as u64
    } else {
        a - b < b_size as u64
    }
}

fn mapped_resources_overlap(
    a_gpu: u64,
    a_cpu: u64,
    a_size: usize,
    b_gpu: u64,
    b_cpu: u64,
    b_size: usize,
) -> bool {
    ranges_overlap(a_gpu, a_size, b_gpu, b_size) || ranges_overlap(a_cpu, a_size, b_cpu, b_size)
}

fn sparse_texture_ranges(
    tic: &TicEntry,
    mappings: &GpuMappings,
) -> Result<Vec<(u64, Option<u64>, usize)>, String> {
    if !tic.is_sparse
        || !tic.is_block_linear
        || tic.texture_type != 2
        || tic.mip_levels() != 1
        || tic.base_layer != 0
        || tic.sample_count() != Some(1)
    {
        return Err(format!(
            "unsupported sparse compute texture layout: {tic:?}"
        ));
    }
    let size = resource_size(tic)?;
    let mut ranges: Vec<(u64, Option<u64>, usize)> = Vec::new();
    let mut offset = 0usize;
    while offset < size {
        let gpu = tic
            .gpu_va
            .checked_add(offset as u64)
            .ok_or_else(|| "sparse texture address overflow".to_string())?;
        let (cpu, available) = mappings
            .texture_memory_range(gpu)
            .ok_or_else(|| format!("sparse texture address {gpu:#x} is not reserved"))?;
        let len = available.min((size - offset) as u64) as usize;
        if len == 0 {
            return Err("empty sparse texture range".into());
        }
        let merge = ranges.last().is_some_and(|(previous_gpu, previous_cpu, previous_len)| {
            previous_gpu.checked_add(*previous_len as u64) == Some(gpu)
                && match (*previous_cpu, cpu) {
                    (None, None) => true,
                    (Some(previous), Some(current)) => previous.checked_add(*previous_len as u64) == Some(current),
                    _ => false,
                }
        });
        if merge {
            ranges.last_mut().unwrap().2 += len;
        } else {
            ranges.push((gpu, cpu, len));
        }
        offset += len;
    }
    Ok(ranges)
}

pub(crate) fn snapshot_sparse_texture(
    tic: &TicEntry,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Result<(TicEntry, Arc<Vec<u8>>, RtKey, u64), String> {
    snapshot_sparse_texture_observed(tic, mappings, mem_read, &|cpu, len| {
        if !nexium_memory::fastmem::write_watch_available() {
            return None;
        }
        let commit = nexium_memory::fastmem::commit_generation();
        let mut spans = Vec::new();
        let observation = nexium_memory::fastmem::take_write_watch_spans_observed(cpu, len, &mut spans);
        match observation.result {
            nexium_memory::fastmem::WriteWatchResult::Clean
            | nexium_memory::fastmem::WriteWatchResult::Dirty => Some((commit, observation.generation_after)),
            nexium_memory::fastmem::WriteWatchResult::Unavailable => None,
        }
    })
}

fn snapshot_sparse_texture_observed(
    tic: &TicEntry,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    observe: &dyn Fn(u64, usize) -> Option<(u64, u64)>,
) -> Result<(TicEntry, Arc<Vec<u8>>, RtKey, u64), String> {
    type Observation = Vec<(u64, Option<u64>, usize, u64, u64)>;
    type CacheEntry = ((u64, u64, usize), u64, Arc<Vec<u8>>, Option<Observation>);
    thread_local! {
        static CACHE: RefCell<Vec<CacheEntry>> = const { RefCell::new(Vec::new()) };
    }
    let started = compute_stage_profile_enabled().then(std::time::Instant::now);
    let ranges = sparse_texture_ranges(tic, mappings)?;
    let range_count = ranges.len();
    let size = resource_size(tic)?;
    let cache_key = (mappings.instance_id(), tic.gpu_va, size);
    let observation = ranges.iter().map(|&(gpu, cpu, len)| {
        let (commit, generation) = match cpu {
            Some(cpu) => observe(cpu, len)?,
            None => (0, 0),
        };
        Some((gpu, cpu, len, commit, generation))
    }).collect::<Option<Observation>>();
    if let Some(observation) = observation.as_ref() {
        let cached = CACHE.with(|cache| cache.borrow().iter()
            .find(|(key, _, _, previous)| *key == cache_key && previous.as_ref() == Some(observation))
            .map(|(_, hash, bytes, _)| (Arc::clone(bytes), *hash)));
        if let Some((bytes, hash)) = cached {
            profile_sparse_snapshot(started, true, range_count);
            return Ok(sparse_texture_snapshot_result(tic, bytes, hash));
        }
    }
    let mut pages = Vec::new();
    let mut hash = xxhash_rust::xxh3::Xxh3::new();
    size.hash(&mut hash);
    for (gpu, cpu, len) in ranges {
        if let Some(cpu) = cpu {
            let offset = (gpu - tic.gpu_va) as usize;
            let mut page = vec![0; len];
            if !mem_read(cpu, &mut page) {
                return Err(format!("sparse texture read failed at {gpu:#x}"));
            }
            offset.hash(&mut hash);
            page.hash(&mut hash);
            pages.push((offset, page));
        }
    }
    let content_key = hash.finish();
    let bytes = CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, _, bytes, previous)) = cache
            .iter_mut()
            .find(|(key, hash, _, _)| *key == cache_key && *hash == content_key)
        {
            *previous = observation;
            return Arc::clone(bytes);
        }
        let mut bytes = vec![0; size];
        for (offset, page) in pages {
            bytes[offset..offset + page.len()].copy_from_slice(&page);
        }
        let bytes = Arc::new(bytes);
        cache.retain(|(key, _, _, _)| *key != cache_key);
        if cache.len() >= 16
            || cache
                .iter()
                .map(|(_, _, bytes, _)| bytes.len())
                .sum::<usize>()
                .saturating_add(size)
                > MAX_RESOURCE_BYTES
        {
            cache.clear();
        }
        cache.push((cache_key, content_key, Arc::clone(&bytes), observation));
        bytes
    });
    profile_sparse_snapshot(started, false, range_count);
    Ok(sparse_texture_snapshot_result(tic, bytes, content_key))
}

fn profile_sparse_snapshot(started: Option<std::time::Instant>, reused: bool, ranges: usize) {
    let Some(started) = started else { return; };
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static HITS: AtomicU64 = AtomicU64::new(0);
    static RANGES: AtomicU64 = AtomicU64::new(0);
    static TIME_NS: AtomicU64 = AtomicU64::new(0);
    HITS.fetch_add(reused as u64, Ordering::Relaxed);
    RANGES.fetch_add(ranges as u64, Ordering::Relaxed);
    TIME_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if (CALLS.fetch_add(1, Ordering::Relaxed) + 1) % 512 == 0 {
        log::warn!("[compute-sparse-snapshot] calls=512 hits={} avg_ranges={:.1} avg_us={:.2}",
            HITS.swap(0, Ordering::Relaxed), RANGES.swap(0, Ordering::Relaxed) as f64 / 512.0,
            TIME_NS.swap(0, Ordering::Relaxed) as f64 / 512_000.0);
    }
}

fn sparse_texture_snapshot_result(
    tic: &TicEntry,
    bytes: Arc<Vec<u8>>,
    content_key: u64,
) -> (TicEntry, Arc<Vec<u8>>, RtKey, u64) {
    let mut dense = *tic;
    dense.is_sparse = false;
    let key = RtKey::with_cpu(0, tic.width, tic.height, tic.gpu_va, 0)
        .with_volume_depth(tic.depth)
        .with_guest_size_bytes(bytes.len() as u64)
        .with_block_linear_layout(
            tic.block_width_log2,
            tic.block_height_log2,
            tic.block_depth_log2,
            tic.tile_width_spacing,
        );
    (dense, bytes, key, content_key)
}

fn validate_sampled_writable_aliases(
    module: &ComputeModule,
    resolved: &[ResolvedResource],
    mappings: &GpuMappings,
) -> Result<(Vec<ComputeImageAlias>, HashSet<u32>), String> {
    let mut ranges = Vec::new();
    for descriptor in module
        .descriptors
        .iter()
        .filter(|descriptor| is_image_resource_descriptor(descriptor))
    {
        let writable = match descriptor.kind {
            ComputeDescriptorKind::StorageTexelBuffer | ComputeDescriptorKind::StorageImage => true,
            ComputeDescriptorKind::CombinedSampledImage
            | ComputeDescriptorKind::UniformTexelBuffer
            | ComputeDescriptorKind::SampledImage => false,
            ComputeDescriptorKind::UniformBuffer | ComputeDescriptorKind::StorageBuffer => continue,
        };
        let resource = resolved
            .iter()
            .find(|resource| resource.metadata.binding == descriptor.binding)
            .ok_or_else(|| {
                format!(
                    "cannot preflight unknown Maxwell compute binding {}",
                    descriptor.binding
                )
            })?;
        if resource.null {
            continue;
        }
        if resource.tic.is_sparse && !writable {
            for (gpu_va, cpu_addr, size) in sparse_texture_ranges(&resource.tic, mappings)? {
                if let Some(cpu_addr) = cpu_addr {
                    ranges.push(MappedComputeResource {
                        binding: descriptor.binding,
                        kind: descriptor.kind,
                        writable: false,
                        tic_gpu_va: resource.tic.gpu_va,
                        width: 0,
                        height: 0,
                        depth: 0,
                        mip_level: None,
                        view_base_mip: 0,
                        view_mip_levels: 1,
                        gpu_va,
                        cpu_addr,
                        size,
                    });
                }
            }
            continue;
        }
        let (cpu_addr, available) =
            mapped_range(mappings, resource.tic.gpu_va).ok_or_else(|| {
                format!(
                    "compute resource binding {} at {:#x} is unmapped: {:?}; {}",
                    descriptor.binding, resource.tic.gpu_va, resource.tic,
                    mappings.bracket(resource.tic.gpu_va)
                )
            })?;
        let required_size = resource_size(&resource.tic)?;
        if !writable && required_size as u64 > available {
            return Err(format!(
                "sampled compute resource binding {} mapping is short ({available:#x} < {required_size:#x}): {:?}; {}",
                descriptor.binding, resource.tic,
                mappings.bracket(resource.tic.gpu_va.saturating_add(available))
            ));
        }
        if resource.tic.is_buffer() {
            if required_size as u64 > available {
                return Err(format!(
                    "compute buffer binding {} mapping is short ({available:#x} < {required_size:#x})",
                    descriptor.binding
                ));
            }
            ranges.push(MappedComputeResource {
                binding: descriptor.binding,
                kind: descriptor.kind,
                writable,
                tic_gpu_va: resource.tic.gpu_va,
                width: resource.tic.width,
                height: 1,
                depth: 1,
                mip_level: None,
                view_base_mip: 0,
                view_mip_levels: 1,
                gpu_va: resource.tic.gpu_va,
                cpu_addr,
                size: required_size,
            });
            continue;
        }

        for subresource in image_view_subresources(&resource.tic, writable)? {
            let end = subresource
                .guest_offset
                .checked_add(subresource.guest_size)
                .ok_or_else(|| "Maxwell compute subresource range overflow".to_string())?;
            if end as u64 > available {
                return Err(format!(
                    "compute image binding {} mip {} mapping is short ({available:#x} < {end:#x})",
                    descriptor.binding, subresource.mip_level
                ));
            }
            ranges.push(MappedComputeResource {
                binding: descriptor.binding,
                kind: descriptor.kind,
                writable,
                tic_gpu_va: resource.tic.gpu_va,
                width: subresource.width,
                height: subresource.height,
                depth: subresource.depth,
                mip_level: Some(subresource.mip_level),
                view_base_mip: resource.tic.view_base_mip(),
                view_mip_levels: resource.tic.view_mip_levels(),
                gpu_va: resource
                    .tic
                    .gpu_va
                    .checked_add(subresource.guest_offset as u64)
                    .ok_or_else(|| {
                        "Maxwell compute subresource GPU address overflow".to_string()
                    })?,
                cpu_addr: cpu_addr
                    .checked_add(subresource.guest_offset as u64)
                    .ok_or_else(|| {
                        "Maxwell compute subresource CPU address overflow".to_string()
                    })?,
                size: subresource.guest_size,
            });
        }
    }
    let (aliases, overlapping) = collect_sampled_writable_aliases(&ranges);
    for alias in &aliases {
        let sampled = resolved
            .iter()
            .find(|resource| resource.metadata.binding == alias.sampled_binding)
            .ok_or_else(|| {
                format!(
                    "compute alias references unknown sampled binding {}",
                    alias.sampled_binding
                )
            })?;
        let storage = resolved
            .iter()
            .find(|resource| resource.metadata.binding == alias.storage_binding)
            .ok_or_else(|| {
                format!(
                    "compute alias references unknown storage binding {}",
                    alias.storage_binding
                )
            })?;
        if sampled.metadata.dimension != storage.metadata.dimension
            || sampled.metadata.numeric_type != storage.metadata.numeric_type
            || sampled.tic.format != storage.tic.format
            || sampled.tic.component_types != storage.tic.component_types
            || !compatible_cross_access_guest_layout(&sampled.tic, &storage.tic)
        {
            return Err(format!(
                "sampled binding {} and storage binding {} alias with incompatible image metadata",
                alias.sampled_binding, alias.storage_binding
            ));
        }
    }
    Ok((aliases, overlapping))
}

fn compatible_cross_access_guest_layout(sampled: &TicEntry, storage: &TicEntry) -> bool {
    let sampled_block_linear =
        sampled.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(sampled.gpu_va);
    let storage_block_linear =
        storage.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(storage.gpu_va);
    sampled_block_linear == storage_block_linear
        && sampled.texture_type == storage.texture_type
        && sampled.width == storage.width
        && sampled.height == storage.height
        && sampled.depth == storage.depth
        && sampled.view_base_mip() == storage.view_base_mip()
        && sampled.is_srgb == storage.is_srgb
        && (!sampled_block_linear
            || (sampled.block_width_log2 == storage.block_width_log2
                && sampled.block_height_log2 == storage.block_height_log2
                && sampled.block_depth_log2 == storage.block_depth_log2
                && sampled.tile_width_spacing == storage.tile_width_spacing))
}

fn collect_sampled_writable_aliases(
    ranges: &[MappedComputeResource],
) -> (Vec<ComputeImageAlias>, HashSet<u32>) {
    static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let mut aliases = Vec::new();
    let mut overlapping = HashSet::new();
    let writable_ranges = ranges.iter().filter(|resource| resource.writable).collect::<Vec<_>>();
    if writable_ranges.is_empty() {
        return (aliases, overlapping);
    }
    for sampled in ranges.iter().filter(|resource| !resource.writable) {
        for writable in &writable_ranges {
            let overlaps = mapped_resources_overlap(
                sampled.gpu_va,
                sampled.cpu_addr,
                sampled.size,
                writable.gpu_va,
                writable.cpu_addr,
                writable.size,
            );
            if !overlaps {
                continue;
            }
            if !exact_cross_access_image_alias(sampled, writable) {
                if overlapping.insert(sampled.binding)
                    && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8
                {
                    log::warn!(
                        "[compute-sampled-overlap] sampled binding {} ({:?}) TIC va={:#x} view={}/{} window {:#x}+{:#x} mip={:?} overlaps writable binding {} ({:?}) view={}/{} window {:#x}+{:#x} mip={:?}; forcing guest snapshot",
                        sampled.binding,
                        sampled.kind,
                        sampled.tic_gpu_va,
                        sampled.view_base_mip,
                        sampled.view_mip_levels,
                        sampled.gpu_va,
                        sampled.size,
                        sampled.mip_level,
                        writable.binding,
                        writable.kind,
                        writable.view_base_mip,
                        writable.view_mip_levels,
                        writable.gpu_va,
                        writable.size,
                        writable.mip_level,
                    );
                }
                continue;
            }
            let alias = ComputeImageAlias {
                sampled_binding: sampled.binding,
                storage_binding: writable.binding,
            };
            if !aliases.contains(&alias) {
                aliases.push(alias);
            }
        }
    }
    (aliases, overlapping)
}

fn exact_cross_access_image_alias(
    sampled: &MappedComputeResource,
    writable: &MappedComputeResource,
) -> bool {
    let sampled_image = matches!(
        sampled.kind,
        ComputeDescriptorKind::CombinedSampledImage | ComputeDescriptorKind::SampledImage
    );
    let storage_image = writable.kind == ComputeDescriptorKind::StorageImage;
    let exact_backing = sampled.size == writable.size && sampled.cpu_addr == writable.cpu_addr;
    sampled_image
        && storage_image
        && exact_backing
        && sampled.view_mip_levels == 1
        && sampled.width == writable.width
        && sampled.height == writable.height
        && sampled.depth == writable.depth
}

fn renderer_program_key(code_sha256: [u8; 32], spirv_hash: u64) -> u64 {
    u64::from_le_bytes(code_sha256[..8].try_into().unwrap())
        .rotate_left(17)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ spirv_hash
}

fn write_storage_image_guest(
    write: &PreparedWrite,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> Result<(), String> {
    if write.target.tic.pitch_bytes == 0 {
        if mem_write(write.target.cpu_addr, &write.bytes) {
            return Ok(());
        }
        return Err(format!(
            "Maxwell output binding {} guest writeback failed at {:#x}",
            write.target.binding, write.target.cpu_addr
        ));
    }

    let row_size = (write.target.subresource.storage_width as usize)
        .checked_mul(write.target.format.bytes_per_pixel())
        .ok_or_else(|| {
            format!(
                "Maxwell output binding {} pitch row size overflow",
                write.target.binding
            )
        })?;
    let rows = write.target.subresource.storage_height as usize;
    let layers = write.target.subresource.depth as usize;
    let pitch = write.target.tic.pitch_bytes as usize;
    if pitch < row_size {
        return Err(format!(
            "Maxwell output binding {} pitch {pitch:#x} is shorter than row {row_size:#x}",
            write.target.binding
        ));
    }
    let linear_layer_size = row_size.checked_mul(rows).ok_or_else(|| {
        format!(
            "Maxwell output binding {} linear layer size overflow",
            write.target.binding
        )
    })?;
    let expected_linear_size = linear_layer_size.checked_mul(layers).ok_or_else(|| {
        format!(
            "Maxwell output binding {} linear image size overflow",
            write.target.binding
        )
    })?;
    let guest_layer_size = pitch.checked_mul(rows).ok_or_else(|| {
        format!(
            "Maxwell output binding {} guest layer size overflow",
            write.target.binding
        )
    })?;
    let expected_guest_size = guest_layer_size.checked_mul(layers).ok_or_else(|| {
        format!(
            "Maxwell output binding {} guest image size overflow",
            write.target.binding
        )
    })?;
    if write.bytes.len() != expected_linear_size || write.target.guest_size != expected_guest_size {
        return Err(format!(
            "Maxwell output binding {} pitch layout size mismatch",
            write.target.binding
        ));
    }
    if pitch == row_size {
        if mem_write(write.target.cpu_addr, &write.bytes) {
            return Ok(());
        }
        return Err(format!(
            "Maxwell output binding {} guest writeback failed at {:#x}",
            write.target.binding, write.target.cpu_addr
        ));
    }
    for layer in 0..layers {
        let source_layer = layer.checked_mul(linear_layer_size).ok_or_else(|| {
            format!(
                "Maxwell output binding {} source layer offset overflow",
                write.target.binding
            )
        })?;
        let guest_layer = layer.checked_mul(guest_layer_size).ok_or_else(|| {
            format!(
                "Maxwell output binding {} guest layer offset overflow",
                write.target.binding
            )
        })?;
        for row in 0..rows {
            let source_offset = source_layer
                .checked_add(row.checked_mul(row_size).ok_or_else(|| {
                    format!(
                        "Maxwell output binding {} source row offset overflow",
                        write.target.binding
                    )
                })?)
                .ok_or_else(|| {
                    format!(
                        "Maxwell output binding {} source row address overflow",
                        write.target.binding
                    )
                })?;
            let guest_offset = guest_layer
                .checked_add(row.checked_mul(pitch).ok_or_else(|| {
                    format!(
                        "Maxwell output binding {} guest row offset overflow",
                        write.target.binding
                    )
                })?)
                .ok_or_else(|| {
                    format!(
                        "Maxwell output binding {} guest row address overflow",
                        write.target.binding
                    )
                })?;
            let cpu_addr = write
                .target
                .cpu_addr
                .checked_add(guest_offset as u64)
                .ok_or_else(|| {
                    format!(
                        "Maxwell output binding {} guest CPU address overflow",
                        write.target.binding
                    )
                })?;
            if !mem_write(
                cpu_addr,
                &write.bytes[source_offset..source_offset + row_size],
            ) {
                return Err(format!(
                    "Maxwell output binding {} guest row writeback failed at {cpu_addr:#x}",
                    write.target.binding
                ));
            }
        }
    }
    Ok(())
}

fn trace_compute_storage_watch(program_gpu_va: u64, write: &PreparedTexelWrite, published: bool) {
    static WATCH: OnceLock<Option<(usize, usize, u64)>> = OnceLock::new();
    let Some((offset, len, limit)) = *WATCH.get_or_init(|| {
        let spec = std::env::var("NEXIUM_GRAPHICS_SSBO_WATCH").ok()?;
        let parse = |value: &str| {
            let value = value.trim();
            if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
                u64::from_str_radix(hex, 16).ok()
            } else {
                value.parse::<u64>().ok()
            }
        };
        let parsed = (|| {
            let (offset, len) = spec.trim().split_once(':')?;
            let offset = usize::try_from(parse(offset)?).ok()?;
            let len = usize::try_from(parse(len)?).ok()?;
            offset.checked_add(len)?;
            let limit = std::env::var("NEXIUM_GRAPHICS_SSBO_WATCH_LIMIT").ok()
                .and_then(|value| parse(&value)).unwrap_or(256);
            (len != 0).then_some((offset, len, limit))
        })();
        if parsed.is_none() {
            log::warn!("[compute-ssbo-watch] invalid byte offset:length {spec:?}");
        }
        parsed
    }) else { return };
    if offset >= write.bytes.len() { return; }
    use std::sync::atomic::{AtomicU64, Ordering};
    static OUTPUTS: AtomicU64 = AtomicU64::new(0);
    let output = OUTPUTS.fetch_add(1, Ordering::Relaxed) + 1;
    if output > limit {
        if output == limit.saturating_add(1) {
            log::info!("[compute-ssbo-watch] output log limit={limit} reached");
        }
        return;
    }
    let end = offset.saturating_add(len).min(write.bytes.len());
    let mut words = Vec::new();
    for word_offset in ((offset & !3)..end).step_by(4) {
        let word_end = word_offset.saturating_add(4).min(write.bytes.len());
        let mut bytes = [0u8; 4];
        bytes[..word_end - word_offset].copy_from_slice(&write.bytes[word_offset..word_end]);
        let bits = u32::from_le_bytes(bytes);
        words.push(format!("{word_offset:#x}:{bits:#010x}({:e})", f32::from_bits(bits)));
    }
    log::info!("[compute-ssbo-watch] output={output} program_code_gpu={program_gpu_va:#x} binding={} resource={} raw={} gpu={:#x} cpu={:#x} bytes={} published={published} watch={offset:#x}..{end:#x} words=[{}]", write.target.binding, write.target.resource_index, write.target.raw, write.target.gpu_va, write.target.cpu_addr, write.bytes.len(), words.join(" "));
}

fn write_back_outputs(
    result: ComputeDispatchResult,
    program_gpu_va: u64,
    targets: &[OutputTarget],
    texel_targets: &[TexelTarget],
    renderer: &nexium_gpu::Renderer,
    mappings: &GpuMappings,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
) -> Result<(), String> {
    let stage_started = compute_stage_profile_enabled().then(std::time::Instant::now);
    let ComputeDispatchResult {
        image_readbacks,
        texel_readbacks,
    } = result;
    if image_readbacks.len() != targets.len() || texel_readbacks.len() != texel_targets.len() {
        return Err(format!(
            "Maxwell compute returned {} image and {} texel readbacks for {} image and {} texel outputs",
            image_readbacks.len(),
            texel_readbacks.len(),
            targets.len(),
            texel_targets.len()
        ));
    }
    let mut seen = HashSet::new();
    let mut prepared = Vec::with_capacity(targets.len());
    for readback in image_readbacks {
        let target = targets
            .iter()
            .copied()
            .find(|target| target.binding == readback.binding)
            .ok_or_else(|| {
                format!(
                    "Maxwell compute returned unexpected output binding {}",
                    readback.binding
                )
            })?;
        if !seen.insert(readback.binding) {
            return Err(format!(
                "Maxwell compute returned output binding {} twice",
                readback.binding
            ));
        }
        if readback.resource_index != target.resource_index
            || readback.width != target.subresource.width
            || readback.height != target.subresource.height
            || readback.depth != target.subresource.depth
            || readback.format != target.format
        {
            return Err(format!(
                "Maxwell output binding {} metadata differs from its dispatch target",
                target.binding
            ));
        }
        let tight_size = (target.subresource.width as usize)
            .checked_mul(target.subresource.height as usize)
            .and_then(|pixels| pixels.checked_mul(target.subresource.depth as usize))
            .and_then(|pixels| pixels.checked_mul(target.format.bytes_per_pixel()))
            .ok_or_else(|| format!("Maxwell output binding {} size overflow", target.binding))?;
        if readback.bytes.len() != tight_size {
            return Err(format!(
                "Maxwell output binding {} readback is {} bytes, expected {}",
                target.binding,
                readback.bytes.len(),
                tight_size
            ));
        }
        let bytes = if target.tic.pitch_bytes != 0 {
            readback.bytes
        } else {
            delinearize_storage_image(
                &target.tic,
                target.subresource,
                target.format,
                readback.bytes,
            )
            .map_err(|error| format!("Maxwell output binding {}: {error}", target.binding))?
        };
        prepared.push(PreparedWrite { target, bytes });
    }

    let mut seen_texels = HashSet::new();
    let mut prepared_texels = Vec::with_capacity(texel_targets.len());
    for readback in texel_readbacks {
        let target = texel_targets
            .iter()
            .copied()
            .find(|target| target.resource_index == readback.resource_index)
            .ok_or_else(|| {
                format!(
                    "Maxwell compute returned unexpected storage texel buffer {}",
                    readback.resource_index
                )
            })?;
        if !seen_texels.insert(readback.resource_index) {
            return Err(format!(
                "Maxwell compute returned storage texel buffer {} twice",
                readback.resource_index
            ));
        }
        if readback.bytes.len() != target.guest_size {
            return Err(format!(
                "Maxwell storage texel buffer binding {} readback is {} bytes, expected {}",
                target.binding,
                readback.bytes.len(),
                target.guest_size
            ));
        }
        prepared_texels.push(PreparedTexelWrite {
            target,
            bytes: readback.bytes,
        });
    }

    for write in &prepared {
        let (current_cpu, available) =
            mapped_range(mappings, write.target.gpu_va).ok_or_else(|| {
                format!(
                    "Maxwell output binding {} lost its guest mapping",
                    write.target.binding
                )
            })?;
        if current_cpu != write.target.cpu_addr || available < write.target.guest_size as u64 {
            return Err(format!(
                "Maxwell output binding {} guest mapping changed during dispatch",
                write.target.binding
            ));
        }
    }
    for write in &prepared_texels {
        if let Some(key) = write.target.raw_storage_key {
            if key.gpu_va != write.target.gpu_va
                || key.cpu_addr != write.target.cpu_addr
                || key.size != write.target.guest_size as u64
                || !raw_storage_mapping_is_current(key, mappings)
            {
                return Err(format!(
                    "Maxwell storage buffer binding {} guest mapping identity changed during dispatch",
                    write.target.binding
                ));
            }
        }
        let (current_cpu, available) =
            mapped_range(mappings, write.target.gpu_va).ok_or_else(|| {
                format!(
                    "Maxwell storage texel buffer binding {} lost its guest mapping",
                    write.target.binding
                )
            })?;
        if current_cpu != write.target.cpu_addr || available < write.bytes.len() as u64 {
            return Err(format!(
                "Maxwell storage texel buffer binding {} guest mapping changed during dispatch",
                write.target.binding
            ));
        }
    }
    let prep_elapsed = stage_started.map_or(std::time::Duration::ZERO, |started| started.elapsed());
    let write_started = stage_started.map(|_| std::time::Instant::now());
    let mut attempted_images = 0;
    let mut attempted_texels = 0;
    let write_result = (|| {
        for write in &prepared {
            attempted_images += 1;
            write_storage_image_guest(write, mem_write)?;
        }
        for write in &prepared_texels {
            attempted_texels += 1;
            let published = mem_write(write.target.cpu_addr, &write.bytes);
            trace_compute_storage_watch(program_gpu_va, write, published);
            if let Some(key) = write.target.raw_storage_key {
                crate::gpu::graphics_storage::note_writeback(key, &write.bytes, published);
            }
            if !published {
                return Err(format!(
                    "Maxwell storage texel buffer binding {} guest writeback failed at {:#x}",
                    write.target.binding, write.target.cpu_addr
                ));
            }
        }
        Ok(())
    })();
    let write_elapsed =
        write_started.map_or(std::time::Duration::ZERO, |started| started.elapsed());
    let invalidate_started = stage_started.map(|_| std::time::Instant::now());
    for write in prepared.into_iter().take(attempted_images) {
        if write.target.subresource.mip_level == 0 {
            if let Some(nvmap_id) = mappings.nvmap_id_for(write.target.gpu_va) {
                if let Some(key) =
                    storage_rt_key(mappings, nvmap_id, &write.target.tic, write.target.cpu_addr)
                {
                    renderer.invalidate_render_target_content(key);
                }
            }
        }
        invalidate_guest_write(
            renderer,
            mappings,
            write.target.gpu_va,
            write.target.cpu_addr,
            write.target.guest_size,
            None,
        );
        if write.target.gpu_va != write.target.tic.gpu_va {
            renderer.invalidate_texture_address(write.target.tic.gpu_va);
        }
    }
    for write in prepared_texels.into_iter().take(attempted_texels) {
        invalidate_guest_write(
            renderer,
            mappings,
            write.target.gpu_va,
            write.target.cpu_addr,
            write.bytes.len(),
            write.target.raw_storage_key,
        );
    }
    if let Some(started) = stage_started {
        profile_compute_writeback_stages(
            prep_elapsed,
            write_elapsed,
            invalidate_started.map_or(std::time::Duration::ZERO, |started| started.elapsed()),
            started.elapsed(),
        );
    }
    write_result
}

fn compute_stage_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_STAGE_PROFILE").is_some())
}

fn profile_compute_writeback_stages(
    prep: std::time::Duration,
    write: std::time::Duration,
    invalidate: std::time::Duration,
    total: std::time::Duration,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static PREP_NS: AtomicU64 = AtomicU64::new(0);
    static WRITE_NS: AtomicU64 = AtomicU64::new(0);
    static INVALIDATE_NS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);

    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    PREP_NS.fetch_add(prep.as_nanos() as u64, Ordering::Relaxed);
    WRITE_NS.fetch_add(write.as_nanos() as u64, Ordering::Relaxed);
    INVALIDATE_NS.fetch_add(invalidate.as_nanos() as u64, Ordering::Relaxed);
    TOTAL_NS.fetch_add(total.as_nanos() as u64, Ordering::Relaxed);
    if calls % 512 != 0 {
        return;
    }
    let prep_ns = PREP_NS.swap(0, Ordering::Relaxed);
    let write_ns = WRITE_NS.swap(0, Ordering::Relaxed);
    let invalidate_ns = INVALIDATE_NS.swap(0, Ordering::Relaxed);
    let total_ns = TOTAL_NS.swap(0, Ordering::Relaxed);
    let other_ns = total_ns.saturating_sub(
        prep_ns
            .saturating_add(write_ns)
            .saturating_add(invalidate_ns),
    );
    log::warn!(
        "[compute-stage-writeback] calls=512 avg_us prep={:.2} write={:.2} invalidate={:.2} other={:.2} total={:.2}",
        prep_ns as f64 / 512_000.0,
        write_ns as f64 / 512_000.0,
        invalidate_ns as f64 / 512_000.0,
        other_ns as f64 / 512_000.0,
        total_ns as f64 / 512_000.0,
    );
}

fn invalidate_guest_write(
    renderer: &nexium_gpu::Renderer,
    mappings: &GpuMappings,
    gpu_va: u64,
    cpu_addr: u64,
    size: usize,
    preserve_raw_storage: Option<ComputeRawStorageKey>,
) {
    let profile = compute_stage_profile_enabled();
    let alias_started = profile.then(std::time::Instant::now);
    let mut aliases = mappings.gpu_regions_for_cpu_range(cpu_addr, size as u64);
    if !aliases
        .iter()
        .any(|(alias, available)| *alias == gpu_va && *available >= size as u64)
    {
        aliases.push((gpu_va, size as u64));
    }
    aliases.sort_unstable();
    aliases.dedup();
    let alias_elapsed = alias_started.map_or(std::time::Duration::ZERO, |s| s.elapsed());
    renderer.invalidate_compute_raw_storage_range_except(
        cpu_addr,
        size as u64,
        &aliases,
        preserve_raw_storage,
    );
    let rt_started = profile.then(std::time::Instant::now);
    renderer.invalidate_render_target_range(cpu_addr, size as u64, &aliases);
    let rt_elapsed = rt_started.map_or(std::time::Duration::ZERO, |s| s.elapsed());
    let alias_count = aliases.len();
    let bump_started = profile.then(std::time::Instant::now);
    let mut bump_bytes = 0u64;
    for (alias, available) in &aliases {
        bump_bytes = bump_bytes.saturating_add(*available);
        nexium_gpu::tex_invalidate::bump_region(*alias, *available);
    }
    let bump_elapsed = bump_started.map_or(std::time::Duration::ZERO, |s| s.elapsed());
    let tex_started = profile.then(std::time::Instant::now);
    for (alias, _) in aliases {
        renderer.invalidate_texture_address(alias);
    }
    if profile {
        profile_invalidate_stages(
            alias_elapsed,
            rt_elapsed,
            bump_elapsed,
            tex_started.map_or(std::time::Duration::ZERO, |s| s.elapsed()),
            alias_count,
            bump_bytes,
        );
    }
}

fn profile_invalidate_stages(
    alias: std::time::Duration,
    rt: std::time::Duration,
    bump: std::time::Duration,
    tex: std::time::Duration,
    alias_count: usize,
    bump_bytes: u64,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static ALIAS_NS: AtomicU64 = AtomicU64::new(0);
    static RT_NS: AtomicU64 = AtomicU64::new(0);
    static BUMP_NS: AtomicU64 = AtomicU64::new(0);
    static TEX_NS: AtomicU64 = AtomicU64::new(0);
    static ALIASES: AtomicU64 = AtomicU64::new(0);
    static BYTES: AtomicU64 = AtomicU64::new(0);

    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    ALIAS_NS.fetch_add(alias.as_nanos() as u64, Ordering::Relaxed);
    RT_NS.fetch_add(rt.as_nanos() as u64, Ordering::Relaxed);
    BUMP_NS.fetch_add(bump.as_nanos() as u64, Ordering::Relaxed);
    TEX_NS.fetch_add(tex.as_nanos() as u64, Ordering::Relaxed);
    ALIASES.fetch_add(alias_count as u64, Ordering::Relaxed);
    BYTES.fetch_add(bump_bytes, Ordering::Relaxed);
    if calls % 512 != 0 {
        return;
    }
    log::warn!(
        "[compute-stage-invalidate] calls=512 avg_us alias_scan={:.2} rt_range={:.2} bump={:.2} tex_addr={:.2} avg_aliases={:.2} avg_bump_kib={:.1}",
        ALIAS_NS.swap(0, Ordering::Relaxed) as f64 / 512_000.0,
        RT_NS.swap(0, Ordering::Relaxed) as f64 / 512_000.0,
        BUMP_NS.swap(0, Ordering::Relaxed) as f64 / 512_000.0,
        TEX_NS.swap(0, Ordering::Relaxed) as f64 / 512_000.0,
        ALIASES.swap(0, Ordering::Relaxed) as f64 / 512.0,
        BYTES.swap(0, Ordering::Relaxed) as f64 / 512.0 / 1024.0,
    );
}

fn read_compute_sampled_image(
    renderer: &Arc<nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    content_key: &dyn Fn(u64, usize) -> Option<u64>,
    gpu_va: u64,
    len: usize,
    label: &str,
) -> Result<(Arc<Vec<u8>>, Option<u64>), String> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_COMPUTE_SNAPSHOT_CACHE")
            .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    });
    if enabled {
        let error = RefCell::new(None);
        let read = |address, size| match read_gpu_vec(mappings, mem_read, address, size, label) {
            Ok(data) => Some(data),
            Err(reason) => { *error.borrow_mut() = Some(reason); None }
        };
        return vk_dispatch::fetch_compute_texture_snapshot(mappings, renderer, &read, gpu_va, len)
            .ok_or_else(|| error.into_inner().unwrap_or_else(|| format!("could not snapshot {label}")));
    }
    let bytes = read_gpu_vec(mappings, mem_read, gpu_va, len, label)?;
    Ok((Arc::new(bytes), content_key(gpu_va, len)))
}

fn read_gpu_vec(
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    gpu_va: u64,
    len: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    if len == 0 || len > MAX_RESOURCE_BYTES {
        return Err(format!("invalid {label} snapshot size {len:#x}"));
    }
    let mut bytes = vec![0u8; len];
    let mut offset = 0usize;
    while offset < len {
        let address = gpu_va.checked_add(offset as u64)
            .ok_or_else(|| format!("{label} address overflow"))?;
        let (cpu_addr, available) = mapped_range(mappings, address)
            .filter(|(_, available)| *available != 0)
            .ok_or_else(|| format!("{label} at {address:#x} is unmapped (start={gpu_va:#x}, size={len:#x})"))?;
        let count = (len - offset).min(usize::try_from(available).unwrap_or(usize::MAX));
        if !mem_read(cpu_addr, &mut bytes[offset..offset + count]) {
            return Err(format!("could not snapshot {label} at {address:#x}"));
        }
        offset += count;
    }
    Ok(bytes)
}

fn mapped_range(mappings: &GpuMappings, gpu_va: u64) -> Option<(u64, u64)> {
    let Some((cpu_addr, mut available)) = mappings.cpu_range_for(gpu_va) else {
        return mappings.cpu_address_for_any32(gpu_va)
            .map(|(_, cpu_addr, available)| (cpu_addr, available));
    };
    while let Some(next_gpu) = gpu_va.checked_add(available) {
        let Some((next_cpu, next_size)) = mappings.cpu_range_for(next_gpu) else {
            break;
        };
        if next_size == 0 || cpu_addr.checked_add(available) != Some(next_cpu) {
            break;
        }
        let Some(combined) = available.checked_add(next_size) else {
            break;
        };
        available = combined;
    }
    Some((cpu_addr, available))
}

fn cbuf_u32(cbuf: &[u8], byte_offset: u32) -> Option<u32> {
    let offset = byte_offset as usize;
    cbuf.get(offset..offset.checked_add(4)?)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u32::from_le_bytes)
}

#[cfg(test)]
mod tests {
    #[test]
    fn storage_writeback_pool_reuses_exact_sizes_and_bounds_retention() {
        let mut pool = super::StorageWritebackPool::default();
        let bytes = vec![0x5a; 4096];
        let pointer = bytes.as_ptr();
        pool.recycle(bytes);
        assert!(pool.take(2048).is_empty());
        let reused = pool.take(4096);
        assert_eq!(reused.as_ptr(), pointer);
        assert!(reused.iter().all(|&byte| byte == 0x5a));
        assert!(pool.take(4096).is_empty());
        for _ in 0..5 {
            pool.recycle(vec![0; 4096]);
        }
        assert_eq!(pool.buffers.len(), 4);
        pool.buffers.clear();
        pool.recycle(vec![0; 12 * 1024 * 1024]);
        pool.recycle(vec![0; 8 * 1024 * 1024]);
        assert_eq!(pool.buffers.len(), 1);
        pool.recycle(Vec::new());
        assert_eq!(pool.buffers.len(), 1);
    }

    use super::*;

    fn write_shader_word(bytes: &mut [u8], offset: usize, word: u64) {
        bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
    }

    fn vertex_store_program(attribute: u64) -> Vec<u8> {
        [
            0,
            attribute,
            0x4c98078804870000,
            0xeb20000a00f70c10,
            0,
            0xe30000000007000fu64,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
    }

    #[test]
    fn vertex_memory_store_groups_preserve_exact_invocation_count() {
        for vertices in 1..=65535 {
            let size = vertex_memory_store_group_size(vertices);
            assert!((1..=32).contains(&size));
            assert_eq!(vertices % size, 0);
            assert_eq!((vertices / size - 1) * size + size - 1, vertices - 1);
        }
        assert_eq!(vertex_memory_store_group_size(5700), 30);
        assert_eq!(vertex_memory_store_group_size(11), 11);
        assert_eq!(vertex_memory_store_group_size(37), 1);
    }

    #[test]
    fn vertex_memory_store_maps_vertex_id_and_preserves_predication() {
        for pred in [0, 7, 8] {
            let code = vertex_store_program((0xefd87f802fc7ff00 & !(15 << 16)) | (pred << 16));
            let converted = vertex_memory_store_code(&code).unwrap();
            let cfg = nexium_shader::build_compute_cfg(&converted);
            assert_eq!(cfg.unimplemented, 0);
            let mut translator = nexium_shader::Translator::new_compute();
            assert!(translator.translate(u64::from_le_bytes(converted[8..16].try_into().unwrap())));
            let mut expected = nexium_shader::Translator::new();
            assert!(expected.translate(u64::from_le_bytes(code[8..16].try_into().unwrap())));
            for inst in &mut expected.program.instructions {
                if matches!(inst.op, IrOp::LoadAttr { slot: 0x2fc }) {
                    inst.op = IrOp::WorkgroupId { component: 0 };
                }
            }
            assert_eq!(
                format!("{:?}", translator.program.instructions),
                format!("{:?}", expected.program.instructions)
            );
            assert!(cfg
                .blocks
                .iter()
                .flat_map(|block| &block.program.instructions)
                .any(|inst| matches!(
                    inst.op,
                    IrOp::ImageWrite {
                        dimension: ImageDimension::D3,
                        ..
                    }
                )));
        }
    }

    #[test]
    fn vertex_memory_store_preserves_global_writes() {
        let code: Vec<u8> = [
            0u64,
            0xefd87f802fc7ff04,
            0x4c98078004470006,
            0x4c98078004570007,
            0,
            0xeedc200000070604,
            0xe30000000007000f,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
        let converted = vertex_memory_store_code(&code).unwrap();
        let mut cfg = nexium_shader::build_compute_cfg(&converted);
        let buffers = nexium_shader::collect_storage_buffers(&mut cfg);
        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].cbuf_offset, 0x110);
        assert!(
            cfg.blocks
                .iter()
                .flat_map(|block| &block.program.instructions)
                .any(|inst| matches!(inst.op, IrOp::StoreStorage { .. }))
        );
        assert!(
            cfg.blocks
                .iter()
                .flat_map(|block| &block.program.instructions)
                .any(|inst| matches!(inst.op, IrOp::WorkgroupId { component: 0 }))
        );
    }

    #[test]
    fn vertex_memory_store_cache_revalidates_modified_code() {
        let mut code = vertex_store_program(0xefd87f802fc7ff00);
        let expected = convert_vertex_memory_store_code(&code);
        assert!(expected.is_some());
        assert_eq!(vertex_memory_store_code(&code), expected);
        assert_eq!(vertex_memory_store_code(&code), expected);
        code = vertex_store_program(0xefd87f800807ff00);
        assert!(vertex_memory_store_code(&code).is_none());
        assert!(vertex_memory_store_code(&code).is_none());
        let original = vertex_store_program(0xefd87f802fc7ff00);
        assert_eq!(vertex_memory_store_code(&original), expected);
    }

    #[test]
    fn vertex_memory_store_rejects_vertex_inputs_outputs_and_special_registers() {
        for attribute in [0xefd87f800807ff00, 0xeff07f800807ff00, 0xf0c8000000070000] {
            assert!(
                vertex_memory_store_code(&vertex_store_program(attribute)).is_none(),
                "{attribute:#x}"
            );
        }
        assert!(vertex_memory_store_code(&direct_compute_program()).is_none());
    }

    #[test]
    #[ignore = "requires NEXIUM_VERTEX_STORE_TEST_SASS pointing to a local shader binary"]
    fn vertex_memory_store_captured_shader_compiles() {
        let code = std::fs::read(std::env::var_os("NEXIUM_VERTEX_STORE_TEST_SASS").unwrap()).unwrap();
        let converted = vertex_memory_store_code(&code).expect("eligible vertex memory store");
        let cbufs = std::array::from_fn(|_| Some(vec![0u8; 0x10000]));
        let (_, frontend) =
            cached_frontend_plan(Sha256::digest(&converted).into(), &converted, &cbufs).unwrap();
        assert!(
            frontend
                .writable_storage_buffers
                .iter()
                .any(|writable| *writable)
                || frontend
                    .needs
                    .iter()
                    .any(|need| need.access == ResourceAccess::Storage)
        );
        let resources = frontend
            .needs
            .iter()
            .enumerate()
            .map(|(index, need)| ComputeImageResource {
                handle: need.handle,
                binding: index as u32 + 1,
                kind: if need.access == ResourceAccess::Storage {
                    ComputeResourceKind::StorageImage
                } else if need.access == ResourceAccess::FilteredSample {
                    ComputeResourceKind::CombinedSampledImage
                } else {
                    ComputeResourceKind::SampledImage
                },
                dimension: need.instruction_dimension.unwrap_or(ImageDimension::D2),
                numeric_type: TextureNumericType::Float,
                texel_format: None,
            })
            .collect();
        let module = nexium_spirv::emit_compute(
            &frontend.cfg,
            &ComputeOptions {
                local_size: [1; 3],
                num_storage_buffers: frontend.storage_buffers.len() as u32,
                resources,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!module.words.is_empty());
        let output =
            std::path::PathBuf::from(std::env::var_os("NEXIUM_VERTEX_STORE_TEST_SASS").unwrap())
                .with_extension("spv");
        std::fs::write(
            &output,
            module
                .words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let validation = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(output)
            .output()
            .unwrap();
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
    }

    #[test]
    fn sparse_ranges_merge_only_contiguous_physical_pages() {
        let mut tic = null_tic(Some(ImageDimension::D3));
        tic.gpu_va = 0x100000;
        tic.width = 16;
        tic.height = 16;
        tic.depth = 16;
        tic.is_sparse = true;
        let mut mappings = GpuMappings::new();
        mappings.add_sparse(tic.gpu_va, 16384);
        mappings.add(tic.gpu_va, 4096, 0x20000, 7);
        mappings.add(tic.gpu_va + 4096, 4096, 0x21000, 8);
        mappings.add(tic.gpu_va + 8192, 4096, 0x30000, 9);
        assert_eq!(sparse_texture_ranges(&tic, &mappings).unwrap(), vec![
            (tic.gpu_va, Some(0x20000), 8192),
            (tic.gpu_va + 8192, Some(0x30000), 4096),
            (tic.gpu_va + 12288, None, 4096),
        ]);
    }

    #[test]
    fn sparse_snapshot_observes_writes_remaps_and_unavailable_tracking() {
        use std::cell::Cell;
        let mut tic = null_tic(Some(ImageDimension::D3));
        tic.gpu_va = 0x100000;
        tic.width = 16;
        tic.height = 16;
        tic.depth = 16;
        tic.is_sparse = true;
        let mut mappings = GpuMappings::new();
        mappings.add_sparse(tic.gpu_va, resource_size(&tic).unwrap() as u64);
        mappings.add(tic.gpu_va + 4096, 4096, 0x20000, 7);
        let reads = Cell::new(0);
        let value = Cell::new(0x5a);
        let read = |_, dst: &mut [u8]| { reads.set(reads.get() + 1); dst.fill(value.get()); true };
        let generation = Cell::new(1);
        let commit = Cell::new(1);
        let observe = |_, _| Some((commit.get(), generation.get()));
        let snapshot = |mappings: &GpuMappings| snapshot_sparse_texture_observed(&tic, mappings, &read, &observe).unwrap();
        let initial = snapshot(&mappings);
        assert_eq!(reads.get(), 1);
        assert!(Arc::ptr_eq(&initial.1, &snapshot(&mappings).1));
        assert_eq!(reads.get(), 1);
        value.set(0x6b);
        generation.set(2);
        let changed = snapshot(&mappings);
        assert_eq!(reads.get(), 2);
        assert_ne!(initial.3, changed.3);
        assert_eq!(changed.1[4096], 0x6b);
        commit.set(2);
        snapshot(&mappings);
        assert_eq!(reads.get(), 3);
        mappings.add(tic.gpu_va + 4096, 4096, 0x30000, 7);
        snapshot(&mappings);
        assert_eq!(reads.get(), 4);
        value.set(0x7c);
        let fallback = snapshot_sparse_texture_observed(&tic, &mappings, &read, &|_, _| None).unwrap();
        assert_eq!(reads.get(), 5);
        assert_eq!(fallback.1[4096], 0x7c);
        mappings.add_sparse(tic.gpu_va + 4096, 4096);
        assert!(snapshot(&mappings).1.iter().all(|byte| *byte == 0));
        assert_eq!(reads.get(), 5);
    }

    #[test]
    fn sparse_texture_snapshot_preserves_pages_and_reserved_holes() {
        let mut tic = null_tic(Some(ImageDimension::D3));
        tic.gpu_va = 0x100000;
        tic.width = 16;
        tic.height = 16;
        tic.depth = 16;
        tic.is_sparse = true;
        let size = resource_size(&tic).unwrap();
        assert_eq!(size, 16384);
        let mut mappings = GpuMappings::new();
        mappings.add_sparse(tic.gpu_va, size as u64);
        mappings.add(tic.gpu_va + 4096, 4096, 0x20000, 7);
        let read = |cpu, dst: &mut [u8]| {
            assert_eq!(cpu, 0x20000);
            dst.fill(0x5a);
            true
        };
        let (dense, bytes, _, hash) = snapshot_sparse_texture(&tic, &mappings, &read).unwrap();
        assert!(!dense.is_sparse);
        assert!(bytes[..4096].iter().all(|byte| *byte == 0));
        assert!(bytes[4096..8192].iter().all(|byte| *byte == 0x5a));
        assert!(bytes[8192..].iter().all(|byte| *byte == 0));
        let (_, repeated, _, repeated_hash) = snapshot_sparse_texture(&tic, &mappings, &read).unwrap();
        assert_eq!(hash, repeated_hash);
        assert!(Arc::ptr_eq(&bytes, &repeated));
        let changed = |_, dst: &mut [u8]| {
            dst.fill(0x6b);
            true
        };
        assert_ne!(
            hash,
            snapshot_sparse_texture(&tic, &mappings, &changed)
                .unwrap()
                .3
        );
        mappings.add_sparse(tic.gpu_va + 4096, 4096);
        let (_, cleared, _, cleared_hash) =
            snapshot_sparse_texture(&tic, &mappings, &|_, _| panic!("hole read")).unwrap();
        assert!(cleared.iter().all(|byte| *byte == 0));
        assert_ne!(hash, cleared_hash);
        let mut gap = GpuMappings::new();
        gap.add(tic.gpu_va + 4096, 4096, 0x20000, 7);
        assert!(snapshot_sparse_texture(&tic, &gap, &read).is_err());
    }

    fn direct_compute_program() -> Vec<u8> {
        let mut bytes = vec![0u8; 0x20];
        write_shader_word(&mut bytes, 0x08, 0xE300_0000_0007_000F);
        bytes
    }

    fn indirect_compute_program() -> Vec<u8> {
        let mut bytes = vec![0u8; 0x60];
        let imnmx = 0x3820_0380_0007_0000u64 | (1u64 << 20);
        let shl = 0x3848_0000_0007_0000u64 | (2u64 << 20);
        let ldc = 0xEF94_0010_0007_0000u64;
        let branch_offset = ((-0x30i32 as u32) & 0x00FF_FFFF) as u64;
        let brx = 0xE250_0000_0007_000Fu64 | (branch_offset << 20);
        write_shader_word(&mut bytes, 0x08, imnmx);
        write_shader_word(&mut bytes, 0x10, shl);
        write_shader_word(&mut bytes, 0x18, ldc);
        write_shader_word(&mut bytes, 0x28, brx);
        write_shader_word(&mut bytes, 0x30, 0xE300_0000_0007_000F);
        write_shader_word(&mut bytes, 0x38, 0xE300_0000_0007_000F);
        bytes
    }

    fn empty_compute_cbufs() -> [Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS] {
        std::array::from_fn(|_| None)
    }

    fn indirect_compute_cbufs(first: u32, second: u32) -> [Option<Vec<u8>>; 8] {
        let mut cbufs = empty_compute_cbufs();
        let mut table = vec![0u8; 8];
        table[0..4].copy_from_slice(&first.to_le_bytes());
        table[4..8].copy_from_slice(&second.to_le_bytes());
        cbufs[1] = Some(table);
        cbufs
    }

    #[test]
    fn direct_frontend_plan_cache_ignores_unrelated_cbuf_variants() {
        let cache = Mutex::new(FrontendPlanCache::default());
        let code = direct_compute_program();
        let code_sha256: [u8; 32] = Sha256::digest(&code).into();
        let first_cbufs = empty_compute_cbufs();
        let (first_key, first) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &first_cbufs, true)
                .expect("direct frontend plan");
        assert_eq!(first_key.cbuf_dependencies, None);
        assert_eq!(cache.lock().unwrap().build_attempts, 1);

        let mut changed_cbufs = empty_compute_cbufs();
        changed_cbufs[0] = Some(vec![1, 2, 3, 4]);
        let (changed_key, changed) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &changed_cbufs, true)
                .expect("cached direct frontend plan");
        assert_eq!(changed_key, first_key);
        assert!(Arc::ptr_eq(&first, &changed));
        assert_eq!(cache.lock().unwrap().build_attempts, 1);
    }

    #[test]
    fn indirect_frontend_plan_cache_hits_each_exact_cbuf_variant() {
        let cache = Mutex::new(FrontendPlanCache::default());
        let code = indirect_compute_program();
        let code_sha256: [u8; 32] = Sha256::digest(&code).into();
        let first_cbufs = indirect_compute_cbufs(0x30, 0x38);
        let (first_key, first) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &first_cbufs, true)
                .expect("first indirect frontend plan");
        assert!(first_key.cbuf_dependencies.is_some());
        assert_eq!(cache.lock().unwrap().build_attempts, 1);

        let (_, first_again) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &first_cbufs, true)
                .expect("cached first indirect frontend plan");
        assert!(Arc::ptr_eq(&first, &first_again));
        assert_eq!(cache.lock().unwrap().build_attempts, 1);

        let second_cbufs = indirect_compute_cbufs(0x38, 0x30);
        let (second_key, second) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &second_cbufs, true)
                .expect("second indirect frontend plan");
        assert_ne!(second_key, first_key);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(cache.lock().unwrap().build_attempts, 2);

        let (_, first_after_switch) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &first_cbufs, true)
                .expect("reused first indirect frontend plan");
        assert!(Arc::ptr_eq(&first, &first_after_switch));
        assert_eq!(cache.lock().unwrap().build_attempts, 2);

        let (_, disabled_hit) =
            cached_frontend_plan_with_cache(&cache, code_sha256, &code, &first_cbufs, false)
                .expect("legacy indirect frontend rebuild");
        assert!(Arc::ptr_eq(&first, &disabled_hit));
        assert_eq!(cache.lock().unwrap().build_attempts, 3);
    }

    fn reported_span(gpu_va: u64) -> PendingComputeWritebackSpan {
        PendingComputeWritebackSpan {
            dispatch_id: 1,
            serial: 1,
            resource_index: 0,
            binding: 2,
            raw: true,
            gpu_va,
            cpu_addr: gpu_va + 0x4000,
            len: 0x40,
        }
    }

    #[test]
    fn resolved_writeback_reports_drain_once() {
        let _ = take_resolved_writeback_spans();
        report_resolved_writeback_spans(vec![reported_span(0x1000), reported_span(0x2000)]);
        assert_eq!(
            take_resolved_writeback_spans(),
            vec![reported_span(0x1000), reported_span(0x2000)]
        );
        assert!(take_resolved_writeback_spans().is_empty());
    }

    #[test]
    fn resolved_writeback_reports_do_not_cross_threads() {
        let _ = take_resolved_writeback_spans();
        report_resolved_writeback_spans(vec![reported_span(0x3000)]);
        let other = std::thread::spawn(|| {
            assert!(take_resolved_writeback_spans().is_empty());
            report_resolved_writeback_spans(vec![reported_span(0x4000)]);
            let drained = take_resolved_writeback_spans();
            assert!(take_resolved_writeback_spans().is_empty());
            drained
        })
        .join()
        .expect("collector thread");
        assert_eq!(other, vec![reported_span(0x4000)]);
        assert_eq!(take_resolved_writeback_spans(), vec![reported_span(0x3000)]);
        assert!(take_resolved_writeback_spans().is_empty());
    }

    #[test]
    fn compute_snapshot_crosses_discontiguous_physical_mappings() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x10000, 1);
        mappings.add(0x2000, 0x1000, 0x30000, 2);
        let read = |address, bytes: &mut [u8]| {
            match address {
                0x10ff0 => bytes.fill(0x12),
                0x30000 => bytes.fill(0x34),
                _ => return false,
            }
            true
        };
        let bytes = read_gpu_vec(&mappings, &read, 0x1ff0, 32, "test").unwrap();
        assert_eq!(&bytes[..16], &[0x12; 16]);
        assert_eq!(&bytes[16..], &[0x34; 16]);
        let empty = GpuMappings::new();
        assert!(read_gpu_vec(&empty, &read, 0x1ff0, 32, "test").is_err());
        let mut hole = GpuMappings::new();
        hole.add(0x1000, 0x1000, 0x10000, 1);
        assert!(read_gpu_vec(&hole, &read, 0x1ff0, 32, "test").is_err());
    }

    #[test]
    fn compute_code_snapshot_preserves_readable_prefix_of_reserved_mapping() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x10000, 0x10000, 0x72000, 1);
        let read = |address: u64, bytes: &mut [u8]| {
            if address + bytes.len() as u64 > 0x78000 { return false; }
            for (offset, byte) in bytes.iter_mut().enumerate() {
                *byte = offset as u8;
            }
            true
        };
        let code = snapshot_code(&mappings, &read, 0x10100).unwrap();
        assert_eq!(code.len(), 0x5f00);
        assert!(code.iter().enumerate().all(|(offset, byte)| *byte == offset as u8));
        assert!(snapshot_code(&mappings, &read, 0x16000).is_err());
        assert!(snapshot_code(&GpuMappings::new(), &read, 0x10000).is_err());
    }

    #[test]
    fn raw_storage_buffers_resolve_static_indirection_topologically() {
        let mut cbuf = vec![0u8; 0x40];
        cbuf[0x20..0x28].copy_from_slice(&0x2003u64.to_le_bytes());
        cbuf[0x28..0x2c].copy_from_slice(&0x40u32.to_le_bytes());
        let mut cbufs: [Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS] =
            std::array::from_fn(|_| None);
        cbufs[0] = Some(cbuf);

        let mut parent_bytes = [0u8; 0x40];
        parent_bytes[8..16].copy_from_slice(&0x3005u64.to_le_bytes());
        let mut mappings = GpuMappings::new();
        mappings.add(0x2000, 0x40, 0x5000, 1);
        let read = |cpu: u64, output: &mut [u8]| {
            let Some(offset) = cpu.checked_sub(0x5000).map(|offset| offset as usize) else {
                return false;
            };
            let Some(source) = parent_bytes.get(offset..offset.saturating_add(output.len())) else {
                return false;
            };
            output.copy_from_slice(source);
            true
        };

        let mut direct = nexium_shader::StorageBufferAddr::direct(0, 0x20, 8);
        direct.required_size = 16;
        let parent = resolve_raw_storage_buffer(0, direct, &cbufs, &[], &mappings, &read)
            .expect("direct descriptor");
        assert_eq!(
            parent,
            ResolvedRawStorageBuffer {
                base: 0x2003,
                size: 0x40
            }
        );

        let child_descriptor = nexium_shader::StorageBufferAddr {
            cbuf_binding: 0,
            cbuf_offset: 0x20,
            align: 8,
            indirect: Some(nexium_shader::StorageBufferIndirection {
                parent_buffer_index: 0,
                pointer_offset: 5,
            }),
            required_size: 20,
            has_dynamic_offset: false,
        };
        let child =
            resolve_raw_storage_buffer(1, child_descriptor, &cbufs, &[parent], &mappings, &read)
                .expect("indirect descriptor");
        assert_eq!(
            child,
            ResolvedRawStorageBuffer {
                base: 0x3005,
                size: 20
            }
        );

        assert!(
            resolve_raw_storage_buffer(0, child_descriptor, &cbufs, &[], &mappings, &read).is_err()
        );

        let mut narrow_child = child_descriptor;
        narrow_child.required_size = 0;
        assert_eq!(
            resolve_raw_storage_buffer(1, narrow_child, &cbufs, &[parent], &mappings, &read)
                .expect("narrow indirect descriptor")
                .size,
            4
        );
    }

    #[test]
    fn raw_storage_cache_key_tracks_exact_mapping_identity() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x8000, 7);
        let first = raw_storage_cache_key(&mappings, 0x1100, 0x80).expect("direct mapping key");
        assert_eq!(first.nvmap_id, 7);
        assert_eq!(first.gpu_va, 0x1100);
        assert_eq!(first.cpu_addr, 0x8100);
        assert_eq!(first.size, 0x80);
        assert!(raw_storage_mapping_is_current(first, &mappings));
        assert!(raw_storage_cache_key(&mappings, 0x1f80, 0x100).is_none());

        mappings.add(0x1000, 0x1000, 0xa000, 9);
        let remapped =
            raw_storage_cache_key(&mappings, 0x1100, 0x80).expect("replacement mapping key");
        assert_ne!(remapped.mapping_epoch, first.mapping_epoch);
        assert_eq!(remapped.nvmap_id, 9);
        assert_eq!(remapped.cpu_addr, 0xa100);
        assert!(!raw_storage_mapping_is_current(first, &mappings));
        assert!(raw_storage_mapping_is_current(remapped, &mappings));
    }

    #[test]
    fn pending_writeback_report_covers_every_declared_target_span() {
        let tic = pitch_r8_tic();
        let format = storage_format(&tic).expect("R8 UNORM storage format");
        let subresource = storage_subresource(&tic).expect("base storage subresource");
        let records = vec![
            PendingComputeWriteback {
                id: PendingComputeId::Ready(1),
                program_gpu_va: 0,
                serial: 11,
                output_targets: vec![OutputTarget {
                    resource_index: 0,
                    binding: 7,
                    tic,
                    subresource,
                    gpu_va: 0x1000,
                    cpu_addr: 0x8000,
                    guest_size: 0x80,
                    format,
                }],
                texel_targets: vec![TexelTarget {
                    resource_index: 1,
                    binding: 8,
                    gpu_va: 0x2000,
                    cpu_addr: 0x9000,
                    guest_size: 0x40,
                    raw: false,
                    raw_storage_key: None,
                }],
                landed: None,
            },
            PendingComputeWriteback {
                id: PendingComputeId::Ready(2),
                program_gpu_va: 0,
                serial: 12,
                output_targets: Vec::new(),
                texel_targets: vec![TexelTarget {
                    resource_index: 0,
                    binding: 9,
                    gpu_va: 0x3000,
                    cpu_addr: 0xa000,
                    guest_size: 0x20,
                    raw: true,
                    raw_storage_key: None,
                }],
                landed: None,
            },
        ];

        assert_eq!(
            pending_writeback_target_spans(&records),
            vec![
                PendingComputeWritebackSpan {
                    dispatch_id: 1,
                    serial: 11,
                    resource_index: u32::MAX,
                    binding: 7,
                    raw: false,
                    gpu_va: 0x1000,
                    cpu_addr: 0x8000,
                    len: 0x80,
                },
                PendingComputeWritebackSpan {
                    dispatch_id: 1,
                    serial: 11,
                    resource_index: 1,
                    binding: 8,
                    raw: false,
                    gpu_va: 0x2000,
                    cpu_addr: 0x9000,
                    len: 0x40,
                },
                PendingComputeWritebackSpan {
                    dispatch_id: 2,
                    serial: 12,
                    resource_index: 0,
                    binding: 9,
                    raw: true,
                    gpu_va: 0x3000,
                    cpu_addr: 0xa000,
                    len: 0x20,
                },
            ]
        );
    }

    #[test]
    fn pending_writebacks_settle_only_for_overlapping_resources() {
        assert!(dispatch_requires_pending_writeback_resolution(true, false));
        assert!(dispatch_requires_pending_writeback_resolution(false, true));
        assert!(dispatch_requires_pending_writeback_resolution(true, true));
        assert!(!dispatch_requires_pending_writeback_resolution(
            false, false
        ));

        let key = ComputeRawStorageKey {
            mapping_epoch: 4,
            nvmap_id: 2,
            gpu_va: 0x1000,
            cpu_addr: 0x8000,
            size: 0x100,
        };
        assert!(resident_raw_overlap_is_compatible(
            Some(key),
            Some(key),
            true
        ));
        assert!(!resident_raw_overlap_is_compatible(
            Some(key),
            Some(ComputeRawStorageKey {
                gpu_va: 0x1080,
                ..key
            }),
            true,
        ));
        assert!(!resident_raw_overlap_is_compatible(
            Some(key),
            Some(key),
            false
        ));
        assert!(!resident_raw_overlap_is_compatible(Some(key), None, true));
    }

    #[test]
    fn cbuf_binding_4097_uploads_qmd_slot_one_and_skips_image_lookup() {
        let cbuf_binding = nexium_spirv::compute_cbuf_descriptor_binding(1).unwrap();
        assert_eq!(cbuf_binding, 4097);

        let mut required_sizes = [0; nexium_spirv::COMPUTE_CBUF_SLOTS];
        required_sizes[1] = 6;
        let module = ComputeModule {
            words: Vec::new(),
            cbuf_bindings: 1 << 1,
            cbuf_size: 0,
            cbuf_required_sizes: required_sizes,
            texture_bound_cbuf: 0,
            descriptors: vec![
                ComputeDescriptor {
                    binding: cbuf_binding,
                    kind: ComputeDescriptorKind::UniformBuffer,
                    handle: None,
                    dimension: None,
                    numeric_type: None,
                    texel_format: None,
                },
                ComputeDescriptor {
                    binding: 1,
                    kind: ComputeDescriptorKind::SampledImage,
                    handle: None,
                    dimension: None,
                    numeric_type: None,
                    texel_format: None,
                },
            ],
        };
        let mut cbufs: [Option<Vec<u8>>; nexium_spirv::COMPUTE_CBUF_SLOTS] =
            std::array::from_fn(|_| None);
        cbufs[1] = Some(vec![0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]);

        let uniform_buffers = prepare_uniform_buffers(&module, &cbufs).unwrap();
        assert_eq!(uniform_buffers.len(), 1);
        assert_eq!(uniform_buffers[0].binding, 4097);
        assert_eq!(
            uniform_buffers[0].bytes,
            vec![0x10, 0x11, 0x12, 0x13, 0x14, 0x15]
        );

        let image_bindings = module
            .descriptors
            .iter()
            .filter(|descriptor| is_image_resource_descriptor(descriptor))
            .map(|descriptor| descriptor.binding)
            .collect::<Vec<_>>();
        assert_eq!(image_bindings, vec![1]);
    }

    #[test]
    fn qmd_local_memory_uses_all_24_size_bits_and_keeps_crs_separate() {
        let mut qmd = [0u32; 0x40];
        qmd[0x2d] = 0xf5ab_cdef;
        qmd[0x2e] = 0xa612_3456;
        qmd[0x2f] = 0x7c65_4321;

        assert_eq!(
            qmd_local_memory(&qmd),
            QmdLocalMemory {
                low_size: 0x00ab_cdef,
                high_size: 0x0012_3456,
                crs_size: 0x0065_4321,
            }
        );
    }

    #[test]
    fn samplerless_filtered_storage_and_atomic_uses_get_distinct_bindings() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x48,
        };
        let atomic_handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x4c,
        };
        let mut needs = Vec::new();
        merge_need(
            &mut needs,
            handle,
            ResourceAccess::Sampled,
            Some(ImageDimension::D3),
            1,
        )
        .unwrap();
        merge_need(
            &mut needs,
            handle,
            ResourceAccess::FilteredSample,
            Some(ImageDimension::D2),
            4,
        )
        .unwrap();
        merge_need(
            &mut needs,
            handle,
            ResourceAccess::Storage,
            Some(ImageDimension::D3),
            0xf,
        )
        .unwrap();
        merge_need(
            &mut needs,
            atomic_handle,
            ResourceAccess::Atomic,
            Some(ImageDimension::Buffer),
            1,
        )
        .unwrap();
        assert_eq!(needs.len(), 4);
    }

    #[test]
    fn sust_and_suatom_on_the_same_buffer_share_the_atomic_descriptor() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x48,
        };
        let mut needs = Vec::new();
        merge_need(
            &mut needs,
            handle,
            ResourceAccess::Storage,
            Some(ImageDimension::Buffer),
            0xf,
        )
        .unwrap();
        merge_need(
            &mut needs,
            handle,
            ResourceAccess::Atomic,
            Some(ImageDimension::Buffer),
            1,
        )
        .unwrap();
        assert_eq!(needs.len(), 1);
        assert_eq!(needs[0].access, ResourceAccess::Atomic);
        assert_eq!(needs[0].instruction_dimension, Some(ImageDimension::Buffer));
        assert_eq!(needs[0].referenced_components, 0xf);
    }

    #[test]
    fn bounded_readonly_snapshot_avoids_unused_unmapped_tail() {
        let mut descriptor = nexium_shader::StorageBufferAddr::direct(0, 0x290, 16);
        descriptor.required_size = 284;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x2000, 1);
        let read = |_: u64, bytes: &mut [u8]| { bytes.fill(0x5a); true };
        let size = raw_storage_snapshot_size(0x750, descriptor, false);
        assert_eq!(size, 284);
        assert_eq!(read_gpu_vec(&mappings, &read, 0x1a00, size, "test").unwrap(), vec![0x5a; 284]);
        assert!(read_gpu_vec(&mappings, &read, 0x1a00, 0x750, "test").is_err());
        assert_eq!(raw_storage_snapshot_size(0x750, descriptor, true), 0x750);
        assert_eq!(raw_storage_snapshot_size(128, descriptor, false), 128);
        descriptor.has_dynamic_offset = true;
        assert_eq!(raw_storage_snapshot_size(0x750, descriptor, false), 0x750);
    }

    #[test]
    fn compute_snapshot_spans_adjacent_contiguous_mappings() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 4, 0x2000, 1);
        mappings.add(0x1004, 4, 0x2004, 2);
        assert_eq!(mapped_range(&mappings, 0x1002), Some((0x2002, 6)));
        let read = |cpu: u64, out: &mut [u8]| {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];
            let offset = (cpu - 0x2000) as usize;
            let Some(source) = data.get(offset..offset + out.len()) else { return false; };
            out.copy_from_slice(source);
            true
        };
        assert_eq!(read_gpu_vec(&mappings, &read, 0x1002, 6, "test").unwrap(), [3, 4, 5, 6, 7, 8]);
        assert!(read_gpu_vec(&mappings, &read, 0x1002, 7, "test").is_err());
        mappings.add(0x1004, 4, 0x3000, 3);
        assert_eq!(mapped_range(&mappings, 0x1002), Some((0x2002, 2)));
        assert!(read_gpu_vec(&mappings, &read, 0x1002, 6, "test").is_err());
        mappings.add_sparse(0x1004, 4);
        assert_eq!(mapped_range(&mappings, 0x1002), Some((0x2002, 2)));
    }

    #[test]
    fn writable_resource_overlap_checks_gpu_and_cpu_aliases() {
        assert!(mapped_resources_overlap(
            0x1000, 0x8000, 0x100, 0x1080, 0x9000, 0x100
        ));
        assert!(mapped_resources_overlap(
            0x1000, 0x8000, 0x100, 0x3000, 0x8080, 0x100
        ));
        assert!(!mapped_resources_overlap(
            0x1000, 0x8000, 0x100, 0x1100, 0x8100, 0x100
        ));
    }

    #[test]
    fn sampled_writable_partial_overlaps_fall_back_to_guest_snapshot() {
        let sampled = MappedComputeResource {
            binding: 1,
            kind: ComputeDescriptorKind::SampledImage,
            writable: false,
            tic_gpu_va: 0x1000,
            width: 0x40,
            height: 1,
            depth: 1,
            mip_level: Some(0),
            view_base_mip: 0,
            view_mip_levels: 1,
            gpu_va: 0x1000,
            cpu_addr: 0x8000,
            size: 0x100,
        };
        let gpu_alias = MappedComputeResource {
            binding: 2,
            kind: ComputeDescriptorKind::StorageImage,
            writable: true,
            gpu_va: 0x1080,
            cpu_addr: 0xa000,
            ..sampled
        };
        let cpu_alias = MappedComputeResource {
            binding: 3,
            kind: ComputeDescriptorKind::StorageTexelBuffer,
            writable: true,
            gpu_va: 0x3000,
            cpu_addr: 0x8080,
            ..sampled
        };
        let disjoint = MappedComputeResource {
            binding: 4,
            kind: ComputeDescriptorKind::StorageImage,
            writable: true,
            gpu_va: 0x4000,
            cpu_addr: 0xb000,
            ..sampled
        };

        let (aliases, overlapping) = collect_sampled_writable_aliases(&[sampled, gpu_alias]);
        assert!(aliases.is_empty());
        assert!(overlapping.contains(&1));
        let (aliases, overlapping) = collect_sampled_writable_aliases(&[sampled, cpu_alias]);
        assert!(aliases.is_empty());
        assert!(overlapping.contains(&1));
        let (aliases, overlapping) = collect_sampled_writable_aliases(&[sampled, disjoint]);
        assert!(aliases.is_empty());
        assert!(overlapping.is_empty());
    }

    #[test]
    fn exact_sampled_storage_image_alias_uses_one_cross_access_image() {
        let sampled = MappedComputeResource {
            binding: 1,
            kind: ComputeDescriptorKind::SampledImage,
            writable: false,
            tic_gpu_va: 0x1000,
            width: 16,
            height: 16,
            depth: 16,
            mip_level: Some(0),
            view_base_mip: 0,
            view_mip_levels: 1,
            gpu_va: 0x1000,
            cpu_addr: 0x8000,
            size: 0x4000,
        };
        let exact_storage = MappedComputeResource {
            binding: 3,
            kind: ComputeDescriptorKind::StorageImage,
            writable: true,
            ..sampled
        };
        assert_eq!(
            collect_sampled_writable_aliases(&[sampled, exact_storage]).0,
            vec![ComputeImageAlias {
                sampled_binding: 1,
                storage_binding: 3,
            }]
        );

        let cpu_alias = MappedComputeResource {
            gpu_va: 0x9000,
            tic_gpu_va: 0x9000,
            ..exact_storage
        };
        assert_eq!(
            collect_sampled_writable_aliases(&[sampled, cpu_alias]).0,
            vec![ComputeImageAlias {
                sampled_binding: 1,
                storage_binding: 3,
            }]
        );

        let partial = MappedComputeResource {
            gpu_va: 0x1080,
            cpu_addr: 0x8080,
            size: 0x3f80,
            ..exact_storage
        };
        let wrong_extent = MappedComputeResource {
            width: 8,
            ..exact_storage
        };
        let storage_buffer = MappedComputeResource {
            kind: ComputeDescriptorKind::StorageTexelBuffer,
            ..exact_storage
        };
        let multi_level = MappedComputeResource {
            view_mip_levels: 2,
            ..sampled
        };
        for ranges in [
            [sampled, partial],
            [sampled, wrong_extent],
            [multi_level, exact_storage],
            [sampled, storage_buffer],
        ] {
            let (aliases, overlapping) = collect_sampled_writable_aliases(&ranges);
            assert!(aliases.is_empty());
            assert!(overlapping.contains(&1));
        }
    }

    fn pps_r16_mip_tic(view_base: u32, view_max: u32) -> TicEntry {
        let mut raw = [
            0x9b, 0xff, 0x17, 0x70, 0x00, 0x00, 0x85, 0x26, 0x05, 0x00, 0x60, 0x00, 0x20, 0x00,
            0x07, 0x90, 0xff, 0x03, 0x80, 0xe8, 0xff, 0x01, 0x00, 0x80, 0x00, 0x00, 0x00, 0x03,
            0x33, 0x00, 0x00, 0x00,
        ];
        raw[28..32].copy_from_slice(&(view_base | (view_max << 4)).to_le_bytes());
        TicEntry::parse(&raw).unwrap()
    }

    fn mapped_image_mip(
        binding: u32,
        kind: ComputeDescriptorKind,
        tic: TicEntry,
        storage: bool,
    ) -> MappedComputeResource {
        let subresource = image_view_subresources(&tic, storage).unwrap()[0];
        MappedComputeResource {
            binding,
            kind,
            writable: storage,
            tic_gpu_va: tic.gpu_va,
            width: subresource.width,
            height: subresource.height,
            depth: subresource.depth,
            mip_level: Some(subresource.mip_level),
            view_base_mip: tic.view_base_mip(),
            view_mip_levels: tic.view_mip_levels(),
            gpu_va: tic.gpu_va + subresource.guest_offset as u64,
            cpu_addr: 0x8000 + subresource.guest_offset as u64,
            size: subresource.guest_size,
        }
    }

    #[test]
    fn adjacent_block_linear_mip_views_do_not_false_alias() {
        let sampled_tic = pps_r16_mip_tic(3, 3);
        let writable_tic = pps_r16_mip_tic(4, 4);
        assert_eq!(resource_size(&sampled_tic).unwrap(), 0x155c00);

        let sampled = mapped_image_mip(
            1,
            ComputeDescriptorKind::CombinedSampledImage,
            sampled_tic,
            false,
        );
        let writable = mapped_image_mip(2, ComputeDescriptorKind::StorageImage, writable_tic, true);
        assert_eq!(
            (sampled.mip_level, sampled.gpu_va, sampled.size),
            (Some(3), sampled_tic.gpu_va + 0x150000, 0x4000)
        );
        assert_eq!(
            (writable.mip_level, writable.gpu_va, writable.size),
            (Some(4), writable_tic.gpu_va + 0x154000, 0x1000)
        );
        let (aliases, overlapping) = collect_sampled_writable_aliases(&[sampled, writable]);
        assert!(aliases.is_empty());
        assert!(overlapping.is_empty());
    }

    #[test]
    fn bc1_cube_mip_layout_spans_six_faces() {
        let tic = TicEntry {
            format: TicFormat::BC1,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x40_0000,
            width: 64,
            height: 64,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 3,
            depth: 1,
            base_layer: 0,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 3,
            res_min_mip_level: 0,
            res_max_mip_level: 3,
        };
        assert_eq!(image_dimension(&tic), Ok(ImageDimension::Cube));
        validate_image_view(&tic, false).unwrap();
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!(
            layout
                .levels
                .iter()
                .map(|level| (level.guest_offset, level.guest_size))
                .collect::<Vec<_>>(),
            vec![(0, 2048), (2048, 512), (2560, 512), (3072, 512)]
        );
        assert_eq!(layout.layer_stride, 3584);
        assert_eq!(texture_guest_size_bytes(&tic, 6), Some(3584 * 6));
        assert_eq!(resource_size(&tic).unwrap(), 3584 * 6);
        let array = TicEntry { texture_type: 5, depth: 4, ..tic };
        assert_eq!(image_dimension(&array), Ok(ImageDimension::D2Array));
        validate_image_view(&array, false).unwrap();
        assert!(validate_image_view(&array, true).is_err());
        assert_eq!(resource_size(&array).unwrap(), 3584 * 4);
        assert_eq!(image_view_subresources(&array, false).unwrap()[0].guest_size, 3584 * 4);
        let subresources = image_view_subresources(&tic, false).unwrap();
        assert_eq!(subresources.len(), 1);
        assert_eq!(
            (subresources[0].guest_offset, subresources[0].guest_size),
            (0, 3584 * 6)
        );
    }

    #[test]
    fn layered_2d_view_slices_layer_window() {
        let mut tic = TicEntry {
            format: TicFormat::A8B8G8R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x80_0000,
            width: 64,
            height: 64,
            block_width_log2: 0,
            block_height_log2: 3,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: true,
            texture_type: 1,
            depth: 6,
            base_layer: 1,
            normalized_coords: true,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 6,
            res_min_mip_level: 0,
            res_max_mip_level: 6,
        };
        validate_image_view(&tic, false).unwrap();
        validate_image_view(&tic, true).unwrap();
        let layout = block_linear_mip_layout(&tic).unwrap();
        assert_eq!((layout.layer_size, layout.layer_stride), (23552, 24576));
        assert_eq!(resource_size(&tic).unwrap(), 23552);
        let subresources = image_view_subresources(&tic, false).unwrap();
        assert_eq!(subresources.len(), 7);
        assert_eq!(subresources[0].guest_offset, 24576);
        assert_eq!(
            (subresources[6].guest_offset, subresources[6].guest_size),
            (24576 + 23040, 512)
        );
        let (view, layered) = sampled_view_tic(&tic).unwrap();
        assert!(layered);
        assert_eq!(
            (view.gpu_va, view.base_layer, view.depth),
            (0x80_0000 + 24576, 0, 1)
        );
        let storage = storage_subresource(&tic).unwrap();
        assert_eq!(
            (
                storage.mip_level,
                storage.guest_offset,
                storage.guest_size,
                storage.width
            ),
            (0, 24576, 16384, 64)
        );
        tic.res_min_mip_level = 1;
        let storage = storage_subresource(&tic).unwrap();
        assert_eq!(
            (
                storage.mip_level,
                storage.guest_offset,
                storage.guest_size,
                storage.width
            ),
            (1, 40960, 4096, 32)
        );
        tic.res_min_mip_level = 0;
        tic.is_srgb = true;
        assert!(validate_image_view(&tic, true).is_err());
        tic.is_srgb = false;
        tic.base_layer = 6;
        assert!(sampled_view_tic(&tic).is_err());
        assert!(validate_image_view(&tic, true).is_err());
    }

    #[test]
    fn storage_image_uses_only_view_base_mip_and_exact_alias_is_shared() {
        let sampled_tic = pps_r16_mip_tic(3, 3);
        let full_storage_tic = pps_r16_mip_tic(0, 9);
        let storage_levels = image_view_subresources(&full_storage_tic, true).unwrap();
        assert_eq!(storage_levels.len(), 1);
        assert_eq!(storage_levels[0].mip_level, 0);
        assert_eq!(
            (storage_levels[0].width, storage_levels[0].height),
            (1024, 512)
        );

        let sampled = mapped_image_mip(
            1,
            ComputeDescriptorKind::CombinedSampledImage,
            sampled_tic,
            false,
        );
        let base_writable = mapped_image_mip(
            2,
            ComputeDescriptorKind::StorageImage,
            full_storage_tic,
            true,
        );
        let (aliases, overlapping) = collect_sampled_writable_aliases(&[sampled, base_writable]);
        assert!(aliases.is_empty());
        assert!(overlapping.is_empty());

        let same_writable =
            mapped_image_mip(2, ComputeDescriptorKind::StorageImage, sampled_tic, true);
        assert_eq!(
            collect_sampled_writable_aliases(&[sampled, same_writable]).0,
            vec![ComputeImageAlias {
                sampled_binding: 1,
                storage_binding: 2,
            }]
        );
    }

    #[test]
    fn linked_and_separate_tic_handles_decode_without_an_allowlist() {
        let mut qmd = [0u32; 0x40];
        assert_eq!(split_tic_handle(&qmd, 0xabc0_0456), 0x456);
        assert_eq!(split_sample_handle(&qmd, 0xabc0_0456), (0x456, 0xabc));
        qmd[0x0b] = 1 << 30;
        assert_eq!(split_tic_handle(&qmd, 0xabc0_0456), 0xabc0_0456);
        assert_eq!(
            split_sample_handle(&qmd, 0xabc0_0456),
            (0xabc0_0456, 0xabc0_0456)
        );
    }

    #[test]
    fn compute_depth_numeric_type_follows_g24r8_view_aspect() {
        use nexium_gpu::texture::{SwizzleSource, TicFormat};

        let mut tic = TicEntry {
            format: TicFormat::G24R8,
            component_types: [
                ComponentType::Uint,
                ComponentType::Unorm,
                ComponentType::Unorm,
                ComponentType::Unorm,
            ],
            swizzle: [SwizzleSource::G; 4],
            gpu_va: 1,
            width: 1,
            height: 1,
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
        };
        assert_eq!(texture_numeric_type(&tic, 0xf), TextureNumericType::Float);
        tic.swizzle = [SwizzleSource::R; 4];
        assert_eq!(texture_numeric_type(&tic, 0xf), TextureNumericType::Uint);
        tic.format = TicFormat::Z24S8;
        assert_eq!(texture_numeric_type(&tic, 0xf), TextureNumericType::Float);
    }

    #[test]
    fn buffer_tics_select_exact_texel_views_while_atomics_stay_r32_uint() {
        use nexium_gpu::texture::{SwizzleSource, TicFormat};

        let make_tic = |format, component| TicEntry {
            format,
            component_types: [component; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x1000,
            width: 64,
            height: 1,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 0,
            is_block_linear: false,
            texture_type: 6,
            depth: 1,
            base_layer: 0,
            normalized_coords: false,
            is_srgb: false,
            is_sparse: false,
            msaa_mode: 0,
            max_mip_level: 0,
            res_min_mip_level: 0,
            res_max_mip_level: 0,
        };
        for (format, component, numeric_type, expected, bytes_per_element) in [
            (
                TicFormat::R32,
                ComponentType::Float,
                TextureNumericType::Float,
                ComputeTexelFormat::R32Float,
                4,
            ),
            (
                TicFormat::R32,
                ComponentType::Uint,
                TextureNumericType::Uint,
                ComputeTexelFormat::R32Uint,
                4,
            ),
            (
                TicFormat::R32,
                ComponentType::Sint,
                TextureNumericType::Sint,
                ComputeTexelFormat::R32Sint,
                4,
            ),
            (
                TicFormat::R16,
                ComponentType::Uint,
                TextureNumericType::Uint,
                ComputeTexelFormat::R16Uint,
                2,
            ),
        ] {
            let tic = make_tic(format, component);
            assert_eq!(storage_texel_format(&tic, numeric_type), Ok(expected));
            assert_eq!(resource_size(&tic), Ok(64 * bytes_per_element));
        }

        let rgba32_float = make_tic(TicFormat::R32G32B32A32, ComponentType::Float);
        assert_eq!(
            texel_format(&rgba32_float, TextureNumericType::Float),
            Ok(ComputeTexelFormat::Rgba32Float)
        );
        assert_eq!(resource_size(&rgba32_float), Ok(64 * 16));

        assert!(
            validate_storage_texel_buffer(&make_tic(TicFormat::R32, ComponentType::Uint)).is_ok()
        );
        assert!(
            validate_storage_texel_buffer(&make_tic(TicFormat::R32, ComponentType::Float)).is_err()
        );
        assert!(
            validate_storage_texel_buffer(&make_tic(TicFormat::R32, ComponentType::Sint)).is_err()
        );
        assert!(
            validate_storage_texel_buffer(&make_tic(TicFormat::R16, ComponentType::Uint)).is_err()
        );
    }

    #[test]
    fn abgr8_unorm_storage_image_roundtrips_guest_component_order() {
        use nexium_gpu::texture::{SwizzleSource, TicFormat};

        let tic = TicEntry {
            format: TicFormat::A8B8G8R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            gpu_va: 0x1234_0000,
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
        };
        let format = storage_format(&tic).expect("A8B8G8R8 UNORM storage format");
        assert_eq!(format, ComputeStorageFormat::Abgr8Unorm);
        let subresource = storage_subresource(&tic).expect("base storage subresource");
        let mut tight = Vec::with_capacity(8 * 8 * 4);
        for pixel in 0..64u8 {
            tight.extend_from_slice(&[
                pixel,
                pixel.wrapping_add(0x40),
                pixel.wrapping_add(0x80),
                0xff,
            ]);
        }

        let guest = delinearize_storage_image(&tic, subresource, format, tight.clone())
            .expect("block-linear guest layout");
        assert_eq!(guest.len(), subresource.guest_size);
        let readback = linearize_storage_image(&tic, subresource, format, guest)
            .expect("tight storage readback");
        assert_eq!(readback, tight);
        assert_eq!(&readback[..4], &[0x00, 0x40, 0x80, 0xff]);
    }

    fn pitch_r8_tic() -> TicEntry {
        use nexium_gpu::texture::{SwizzleSource, TicFormat};

        TicEntry {
            format: TicFormat::R8,
            component_types: [ComponentType::Unorm; 4],
            swizzle: [SwizzleSource::R; 4],
            gpu_va: 0x1234_0000,
            width: 3,
            height: 2,
            block_width_log2: 0,
            block_height_log2: 0,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            pitch_bytes: 32,
            is_block_linear: false,
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

    #[test]
    fn storage_rt_key_rejects_unsupported_3d_alias_footprint() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x1234_0000, 0x20_000, 0x8000, 1);
        let mut tic = pitch_r8_tic();
        tic.is_block_linear = true;
        tic.pitch_bytes = 0;
        assert!(storage_rt_key(&mappings, 1, &tic, 0x8000).is_some());

        tic.depth = 2;
        assert!(storage_rt_key(&mappings, 1, &tic, 0x8000).is_none());
        tic.depth = 1;

        tic.texture_type = 2;
        tic.depth = 4;
        assert!(storage_rt_key(&mappings, 1, &tic, 0x8000).is_none());
    }

    #[test]
    fn pitch_linear_type_seven_is_single_layer_2d() {
        let tic = TicEntry {
            texture_type: 7,
            ..pitch_r8_tic()
        };
        assert_eq!(image_dimension(&tic), Ok(ImageDimension::D2));
        assert!(validate_image_view(&tic, false).is_ok());
        assert!(validate_image_view(&tic, true).is_ok());
        assert_eq!(resource_size(&tic), Ok(64));
        let subresource = storage_subresource(&tic).unwrap();
        assert_eq!(subresource.depth, 1);
        assert_eq!(subresource.guest_size, 64);
    }

    #[test]
    fn pitch_linear_rejects_non_2d_and_layered_layouts() {
        let base = pitch_r8_tic();
        let invalid = [
            TicEntry {
                texture_type: 2,
                depth: 2,
                ..base
            },
            TicEntry {
                texture_type: 5,
                ..base
            },
            TicEntry { depth: 2, ..base },
            TicEntry {
                base_layer: 1,
                ..base
            },
        ];
        for tic in invalid {
            assert!(image_dimension(&tic).is_err());
            assert!(validate_image_view(&tic, false).is_err());
            assert!(resource_size(&tic).is_err());
        }
    }

    #[test]
    fn pitch_linear_storage_image_roundtrips_padded_rows() {
        let tic = pitch_r8_tic();
        let format = storage_format(&tic).expect("R8 UNORM storage format");
        let subresource = storage_subresource(&tic).expect("base storage subresource");
        assert_eq!(resource_size(&tic), Ok(64));
        assert_eq!(subresource.guest_size, 64);
        let tight = vec![1, 2, 3, 4, 5, 6];
        let backing = std::cell::RefCell::new(vec![0xcc; 64]);
        let target = OutputTarget {
            resource_index: 0,
            binding: 7,
            tic,
            subresource,
            gpu_va: tic.gpu_va,
            cpu_addr: 0x8000,
            guest_size: 64,
            format,
        };
        let write = PreparedWrite {
            target,
            bytes: tight.clone(),
        };
        write_storage_image_guest(&write, &|cpu_addr, bytes| {
            let Some(offset) = cpu_addr
                .checked_sub(target.cpu_addr)
                .map(|offset| offset as usize)
            else {
                return false;
            };
            let mut backing = backing.borrow_mut();
            let Some(destination) = backing.get_mut(offset..offset.saturating_add(bytes.len()))
            else {
                return false;
            };
            destination.copy_from_slice(bytes);
            true
        })
        .expect("pitch-linear guest writeback");
        let guest = backing.into_inner();
        assert_eq!(&guest[0..3], &[1, 2, 3]);
        assert_eq!(&guest[32..35], &[4, 5, 6]);
        assert!(guest[3..32].iter().all(|byte| *byte == 0xcc));
        assert!(guest[35..].iter().all(|byte| *byte == 0xcc));
        assert_eq!(
            linearize_storage_image(&tic, subresource, format, guest)
                .expect("tight storage readback"),
            tight
        );
    }
}

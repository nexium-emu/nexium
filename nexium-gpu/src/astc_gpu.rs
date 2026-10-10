use ash::vk;
use nexium_common::astc::{AstcDecodeMode, AstcRecompression};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::bcn_encode::BcTarget;

pub(crate) const INPUT_ARENA_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const OUTPUT_ARENA_BYTES: u64 = 128 * 1024 * 1024;
const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
const ARENA_ALIGNMENT: u64 = 256;
const PUSH_CONSTANT_BYTES: u32 = 32;
const WORKGROUP_BLOCKS: u32 = 8;

static BOOT_DECODE_MODE: AtomicU8 = AtomicU8::new(0);
static BOOT_RECOMPRESSION: AtomicU8 = AtomicU8::new(0);

fn env_decode_mode() -> Option<AstcDecodeMode> {
    static VALUE: OnceLock<Option<AstcDecodeMode>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NEXIUM_ASTC_DECODE")
            .ok()
            .and_then(|value| AstcDecodeMode::parse(&value))
    })
}

fn env_recompression() -> Option<AstcRecompression> {
    static VALUE: OnceLock<Option<AstcRecompression>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("NEXIUM_ASTC_RECOMPRESSION")
            .ok()
            .and_then(|value| AstcRecompression::parse(&value))
    })
}

fn recompression_target(recompression: AstcRecompression, native_bc_formats: &[vk::Format]) -> Option<BcTarget> {
    let target = match recompression {
        AstcRecompression::Uncompressed => return None,
        AstcRecompression::Bc1 => BcTarget::Bc1,
        AstcRecompression::Bc3 => BcTarget::Bc3,
    };
    [false, true]
        .iter()
        .all(|&srgb| native_bc_formats.contains(&target.format(srgb)))
        .then_some(target)
}

pub(crate) fn configure(native_bc_formats: &[vk::Format]) {
    let mode = env_decode_mode().unwrap_or_else(nexium_common::astc::decode_mode);
    let recompression = env_recompression().unwrap_or_else(nexium_common::astc::recompression);
    let target = recompression_target(recompression, native_bc_formats);
    if recompression != AstcRecompression::Uncompressed && target.is_none() {
        log::warn!("[astc] this GPU cannot sample {recompression:?} textures; ASTC stays uncompressed");
    }
    BOOT_DECODE_MODE.store(
        match mode {
            AstcDecodeMode::Gpu => 0,
            AstcDecodeMode::Cpu => 1,
            AstcDecodeMode::CpuAsynchronous => 2,
        },
        Ordering::Relaxed,
    );
    BOOT_RECOMPRESSION.store(
        match target {
            None => 0,
            Some(BcTarget::Bc1) => 1,
            Some(BcTarget::Bc3) => 3,
        },
        Ordering::Relaxed,
    );
    log::info!(
        "[astc] decode={mode:?} recompression={}",
        target.map_or("uncompressed".to_string(), |target| format!("{target:?}"))
    );
}

pub(crate) fn decode_mode() -> AstcDecodeMode {
    match BOOT_DECODE_MODE.load(Ordering::Relaxed) {
        1 => AstcDecodeMode::Cpu,
        2 => AstcDecodeMode::CpuAsynchronous,
        _ => AstcDecodeMode::Gpu,
    }
}

pub(crate) fn bc_target() -> Option<BcTarget> {
    match BOOT_RECOMPRESSION.load(Ordering::Relaxed) {
        1 => Some(BcTarget::Bc1),
        3 => Some(BcTarget::Bc3),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AstcDecodeJob {
    pub(crate) src_offset: u64,
    pub(crate) dst_offset: u64,
    pub(crate) encoded_offset: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) block_width: u32,
    pub(crate) block_height: u32,
}

impl AstcDecodeJob {
    pub(crate) fn blocks(&self) -> (u32, u32) {
        (
            self.width.div_ceil(self.block_width.max(1)),
            self.height.div_ceil(self.block_height.max(1)),
        )
    }

    pub(crate) fn src_layer_bytes(&self) -> u64 {
        let (blocks_x, blocks_y) = self.blocks();
        u64::from(blocks_x) * u64::from(blocks_y) * 16
    }

    pub(crate) fn dst_layer_bytes(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * 4
    }

    pub(crate) fn encoded_layer_bytes(&self, target: BcTarget) -> u64 {
        target.encoded_size(self.width, self.height) as u64
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AstcUploadPlan {
    pub(crate) compressed: Vec<u8>,
    pub(crate) jobs: Vec<AstcDecodeJob>,
    pub(crate) decoded_bytes: u64,
    pub(crate) target: Option<BcTarget>,
    pub(crate) encoded_bytes: u64,
}

impl AstcUploadPlan {
    pub(crate) fn new(target: Option<BcTarget>) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    pub(crate) fn push_level(
        &mut self,
        layers: &[Vec<u8>],
        width: u32,
        height: u32,
        block_width: u32,
        block_height: u32,
    ) -> u64 {
        let job = AstcDecodeJob {
            src_offset: self.compressed.len() as u64,
            dst_offset: self.decoded_bytes,
            encoded_offset: self.encoded_bytes,
            width,
            height,
            layers: layers.len() as u32,
            block_width,
            block_height,
        };
        let layer_bytes = job.src_layer_bytes() as usize;
        for layer in layers {
            let start = self.compressed.len();
            self.compressed.extend_from_slice(&layer[..layer.len().min(layer_bytes)]);
            self.compressed.resize(start + layer_bytes, 0);
        }
        self.decoded_bytes = self
            .decoded_bytes
            .saturating_add(job.dst_layer_bytes().saturating_mul(u64::from(job.layers)));
        if let Some(target) = self.target {
            self.encoded_bytes = self
                .encoded_bytes
                .saturating_add(job.encoded_layer_bytes(target).saturating_mul(u64::from(job.layers)));
        }
        self.jobs.push(job);
        if self.target.is_some() {
            job.encoded_offset
        } else {
            job.dst_offset
        }
    }
}
pub(crate) fn allocate_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|error| format!("astc buffer: {error:?}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(memory_type) =
        crate::rt_cache::find_memory_type(mem_props, requirements.memory_type_bits, flags)
    else {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(format!("astc buffer: no memory type with {flags:?}"));
    };
    let memory = match unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    } {
        Ok(memory) => memory,
        Err(error) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(format!("astc buffer memory: {error:?}"));
        }
    };
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(format!("astc buffer bind: {error:?}"));
    }
    Ok((buffer, memory))
}

pub(crate) struct AstcDecodePipeline {
    descriptor_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    encode_pipeline: vk::Pipeline,
}

fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spirv: &[u8],
) -> Result<vk::Pipeline, String> {
    let code: Vec<_> = spirv
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
    }
    .map_err(|error| format!("shader module: {error:?}"))?;
    let info = [vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main"),
        )
        .layout(layout)];
    let pipelines =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
    unsafe { device.destroy_shader_module(module, None) };
    match pipelines {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                unsafe { device.destroy_pipeline(pipeline, None) };
            }
            Err(format!("compute pipeline: {error:?}"))
        }
    }
}

fn push_words(device: &ash::Device, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, words: [u32; 8]) {
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    unsafe {
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
    }
}

fn buffer_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size.max(4));
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

impl AstcDecodePipeline {
    pub(crate) fn new(device: &ash::Device) -> Result<Self, String> {
        let bindings = [0, 1].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| format!("astc descriptor layout: {error:?}"))?;
        let set_layouts = [descriptor_layout];
        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_CONSTANT_BYTES)];
        let layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_constants),
                None,
            )
        } {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_layout, None) };
                return Err(format!("astc pipeline layout: {error:?}"));
            }
        };
        let mut result = Self {
            descriptor_layout,
            layout,
            pipeline: vk::Pipeline::null(),
            encode_pipeline: vk::Pipeline::null(),
        };
        match create_compute_pipeline(
            device,
            layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/astc_decode_main.spv")),
        ) {
            Ok(pipeline) => result.pipeline = pipeline,
            Err(error) => {
                result.destroy(device);
                return Err(format!("astc decode {error}"));
            }
        }
        match create_compute_pipeline(
            device,
            layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bc_encode_main.spv")),
        ) {
            Ok(pipeline) => result.encode_pipeline = pipeline,
            Err(error) => {
                result.destroy(device);
                return Err(format!("bc encode {error}"));
            }
        }
        Ok(result)
    }

    pub(crate) fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        arena: &AstcDecodeArena,
        reservation: AstcReservation,
        plan: &AstcUploadPlan,
    ) {
        unsafe {
            if reservation.output_wrapped {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[reservation.descriptor_set],
                &[],
            );
        }
        for job in &plan.jobs {
            let (blocks_x, blocks_y) = job.blocks();
            push_words(
                device,
                cmd,
                self.layout,
                [
                    ((reservation.src_offset + job.src_offset) / 4) as u32,
                    ((reservation.dst_offset + job.dst_offset) / 4) as u32,
                    job.width,
                    job.height,
                    job.block_width,
                    job.block_height,
                    (job.src_layer_bytes() / 4) as u32,
                    (job.dst_layer_bytes() / 4) as u32,
                ],
            );
            unsafe {
                device.cmd_dispatch(
                    cmd,
                    blocks_x.div_ceil(WORKGROUP_BLOCKS),
                    blocks_y.div_ceil(WORKGROUP_BLOCKS),
                    job.layers.max(1),
                );
            }
        }
        let Some(target) = plan.target else {
            buffer_barrier(
                device,
                cmd,
                arena.output_buffer,
                reservation.dst_offset,
                plan.decoded_bytes,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            );
            return;
        };
        buffer_barrier(
            device,
            cmd,
            arena.output_buffer,
            reservation.dst_offset,
            plan.decoded_bytes,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.encode_pipeline);
        }
        for job in &plan.jobs {
            push_words(
                device,
                cmd,
                self.layout,
                [
                    ((reservation.dst_offset + job.dst_offset) / 4) as u32,
                    ((reservation.encoded_offset + job.encoded_offset) / 4) as u32,
                    job.width,
                    job.height,
                    (job.dst_layer_bytes() / 4) as u32,
                    (job.encoded_layer_bytes(target) / 4) as u32,
                    target.shader_mode(),
                    0,
                ],
            );
            unsafe {
                device.cmd_dispatch(
                    cmd,
                    job.width.div_ceil(4).div_ceil(WORKGROUP_BLOCKS),
                    job.height.div_ceil(4).div_ceil(WORKGROUP_BLOCKS),
                    job.layers.max(1),
                );
            }
        }
        buffer_barrier(
            device,
            cmd,
            arena.output_buffer,
            reservation.encoded_offset,
            plan.encoded_bytes,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        unsafe {
            if self.encode_pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.encode_pipeline, None);
            }
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AstcReservation {
    pub(crate) descriptor_set: vk::DescriptorSet,
    pub(crate) src_offset: u64,
    pub(crate) dst_offset: u64,
    pub(crate) encoded_offset: u64,
    pub(crate) copy_offset: u64,
    pub(crate) output_wrapped: bool,
}

struct InputChunk {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    size: u64,
    head: u64,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl InputChunk {
    fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        descriptor_layout: vk::DescriptorSetLayout,
        output_buffer: vk::Buffer,
        size: u64,
    ) -> Result<Self, String> {
        let (buffer, memory) = allocate_buffer(
            device,
            mem_props,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let mut chunk = Self {
            buffer,
            memory,
            mapped: std::ptr::null_mut(),
            size,
            head: 0,
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
        };
        let setup = (|| -> Result<(), String> {
            chunk.mapped = unsafe {
                device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            }
            .map_err(|error| format!("astc input map: {error:?}"))?
                as *mut u8;
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            chunk.descriptor_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            }
            .map_err(|error| format!("astc descriptor pool: {error:?}"))?;
            let layouts = [descriptor_layout];
            chunk.descriptor_set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(chunk.descriptor_pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(|error| format!("astc descriptor set: {error:?}"))?[0];
            let input_info = [vk::DescriptorBufferInfo::default()
                .buffer(chunk.buffer)
                .range(vk::WHOLE_SIZE)];
            let output_info = [vk::DescriptorBufferInfo::default()
                .buffer(output_buffer)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(chunk.descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&input_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(chunk.descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&output_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            Ok(())
        })();
        if let Err(error) = setup {
            chunk.destroy(device);
            return Err(error);
        }
        Ok(chunk)
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            if self.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if !self.mapped.is_null() {
                device.unmap_memory(self.memory);
            }
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub(crate) struct AstcDecodeArena {
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,
    output_size: u64,
    output_head: u64,
    descriptor_layout: vk::DescriptorSetLayout,
    chunk_size: u64,
    chunks: Vec<InputChunk>,
    current: usize,
}

unsafe impl Send for AstcDecodeArena {}
unsafe impl Sync for AstcDecodeArena {}

fn output_layout(start: u64, plan: &AstcUploadPlan) -> Option<(u64, u64, u64)> {
    let dst_offset = start.next_multiple_of(ARENA_ALIGNMENT);
    let dst_end = dst_offset.checked_add(plan.decoded_bytes)?;
    let encoded_offset = dst_end.next_multiple_of(ARENA_ALIGNMENT);
    let end = if plan.target.is_some() {
        encoded_offset.checked_add(plan.encoded_bytes)?
    } else {
        dst_end
    };
    Some((dst_offset, encoded_offset, end))
}

impl AstcDecodeArena {
    pub(crate) fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        pipeline: &AstcDecodePipeline,
        input_size: u64,
        output_size: u64,
    ) -> Result<Self, String> {
        let (output_buffer, output_memory) = allocate_buffer(
            device,
            mem_props,
            output_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let mut arena = Self {
            output_buffer,
            output_memory,
            output_size,
            output_head: 0,
            descriptor_layout: pipeline.descriptor_layout,
            chunk_size: input_size,
            chunks: Vec::new(),
            current: 0,
        };
        match InputChunk::new(
            device,
            mem_props,
            arena.descriptor_layout,
            output_buffer,
            input_size,
        ) {
            Ok(chunk) => arena.chunks.push(chunk),
            Err(error) => {
                arena.destroy(device);
                return Err(error);
            }
        }
        Ok(arena)
    }

    pub(crate) fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
    }

    pub(crate) fn reset(&mut self, device: &ash::Device) {
        let keep = (self.current + 1).min(self.chunks.len()).max(1);
        for chunk in self.chunks.drain(keep..) {
            chunk.destroy(device);
        }
        for chunk in &mut self.chunks {
            chunk.head = 0;
        }
        self.current = 0;
        self.output_head = 0;
    }

    fn input_bytes(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.size).sum()
    }

    pub(crate) fn reserve(
        &mut self,
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        plan: &AstcUploadPlan,
    ) -> Option<AstcReservation> {
        if plan.decoded_bytes == 0 {
            return None;
        }
        let (mut dst_offset, mut encoded_offset, mut output_end) =
            output_layout(self.output_head, plan)?;
        let mut output_wrapped = false;
        if output_end > self.output_size {
            (dst_offset, encoded_offset, output_end) = output_layout(0, plan)?;
            if output_end > self.output_size {
                return None;
            }
            output_wrapped = true;
        }
        let length = plan.compressed.len() as u64;
        let src_offset = loop {
            if let Some(chunk) = self.chunks.get(self.current) {
                let offset = chunk.head.next_multiple_of(ARENA_ALIGNMENT);
                if offset.checked_add(length)? <= chunk.size {
                    break offset;
                }
                self.current += 1;
                continue;
            }
            let size = self.chunk_size.max(length.next_multiple_of(ARENA_ALIGNMENT));
            if self.input_bytes().saturating_add(size) > MAX_INPUT_BYTES {
                return None;
            }
            match InputChunk::new(device, mem_props, self.descriptor_layout, self.output_buffer, size) {
                Ok(chunk) => self.chunks.push(chunk),
                Err(error) => {
                    log::warn!("[astc] decode input chunk unavailable: {error}");
                    return None;
                }
            }
        };
        let chunk = &mut self.chunks[self.current];
        unsafe {
            std::ptr::copy_nonoverlapping(
                plan.compressed.as_ptr(),
                chunk.mapped.add(src_offset as usize),
                plan.compressed.len(),
            );
        }
        chunk.head = src_offset + length;
        self.output_head = output_end;
        Some(AstcReservation {
            descriptor_set: chunk.descriptor_set,
            src_offset,
            dst_offset,
            encoded_offset,
            copy_offset: if plan.target.is_some() {
                encoded_offset
            } else {
                dst_offset
            },
            output_wrapped,
        })
    }

    pub(crate) fn destroy(self, device: &ash::Device) {
        for chunk in self.chunks {
            chunk.destroy(device);
        }
        unsafe {
            device.destroy_buffer(self.output_buffer, None);
            device.free_memory(self.output_memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{recompression_target, AstcDecodeJob, AstcUploadPlan};
    use crate::bcn_encode::BcTarget;
    use ash::vk;
    use nexium_common::astc::AstcRecompression;

    #[test]
    fn upload_plan_packs_levels_with_layer_strides() {
        let mut plan = AstcUploadPlan::default();
        assert_eq!(plan.push_level(&[vec![1; 64], vec![2; 64]], 8, 8, 4, 4), 0);
        assert_eq!(plan.push_level(&[vec![3; 16], vec![4; 10]], 4, 4, 4, 4), 512);
        assert_eq!(plan.compressed.len(), 64 + 64 + 16 + 16);
        assert_eq!(&plan.compressed[144..154], &[4; 10]);
        assert_eq!(&plan.compressed[154..160], &[0; 6]);
        assert_eq!(plan.decoded_bytes, 8 * 8 * 4 * 2 + 4 * 4 * 4 * 2);
        assert_eq!(plan.encoded_bytes, 0);
        assert_eq!(
            plan.jobs,
            vec![
                AstcDecodeJob {
                    src_offset: 0,
                    dst_offset: 0,
                    encoded_offset: 0,
                    width: 8,
                    height: 8,
                    layers: 2,
                    block_width: 4,
                    block_height: 4,
                },
                AstcDecodeJob {
                    src_offset: 128,
                    dst_offset: 512,
                    encoded_offset: 0,
                    width: 4,
                    height: 4,
                    layers: 2,
                    block_width: 4,
                    block_height: 4,
                },
            ]
        );
    }

    #[test]
    fn recompressed_plans_copy_from_the_encoded_region() {
        let mut plan = AstcUploadPlan::new(Some(BcTarget::Bc3));
        assert_eq!(plan.push_level(&[vec![0; 16 * 9], vec![0; 16 * 9]], 10, 9, 4, 4), 0);
        assert_eq!(plan.push_level(&[vec![0; 16 * 4], vec![0; 16 * 4]], 5, 5, 4, 4), 3 * 3 * 16 * 2);
        assert_eq!(plan.encoded_bytes, (3 * 3 + 2 * 2) * 16 * 2);
        assert_eq!(plan.decoded_bytes, (10 * 9 + 5 * 5) * 4 * 2);
        let mut bc1 = AstcUploadPlan::new(Some(BcTarget::Bc1));
        bc1.push_level(&[vec![0; 16]], 3, 2, 4, 4);
        assert_eq!(bc1.encoded_bytes, 8);
    }

    #[test]
    fn recompression_needs_both_bc_variants() {
        let all = [
            vk::Format::BC1_RGBA_UNORM_BLOCK,
            vk::Format::BC1_RGBA_SRGB_BLOCK,
            vk::Format::BC3_UNORM_BLOCK,
            vk::Format::BC3_SRGB_BLOCK,
        ];
        assert_eq!(recompression_target(AstcRecompression::Uncompressed, &all), None);
        assert_eq!(recompression_target(AstcRecompression::Bc1, &all), Some(BcTarget::Bc1));
        assert_eq!(recompression_target(AstcRecompression::Bc3, &all), Some(BcTarget::Bc3));
        assert_eq!(recompression_target(AstcRecompression::Bc3, &all[..3]), None);
        assert_eq!(recompression_target(AstcRecompression::Bc1, &[]), None);
    }
}

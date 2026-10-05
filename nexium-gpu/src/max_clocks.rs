use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, Once};
use std::time::{Duration, Instant};

const ACTIVE_WINDOW: Duration = Duration::from_millis(100);
const PARK_TIMEOUT: Duration = Duration::from_millis(250);
const TARGET_DISPATCH: Duration = Duration::from_micros(200);
const GROUPS_X: u32 = 64;
const GROUPS_Y: u32 = 16;
const SINK_BYTES: u64 = GROUPS_X as u64 * GROUPS_Y as u64 * 64 * 4;
const MIN_ROUNDS: u32 = 16;
const MAX_ROUNDS: u32 = 1 << 16;

struct Signal {
    parked: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

static SIGNAL: Signal = Signal {
    parked: AtomicBool::new(false),
    lock: Mutex::new(()),
    wake: Condvar::new(),
};
static LAST_SUBMISSION_NS: AtomicU64 = AtomicU64::new(0);
static WORKER: Once = Once::new();

pub(crate) fn note_submission() {
    if !nexium_common::force_max_clocks::enabled() {
        return;
    }
    LAST_SUBMISSION_NS.store(crate::renderer::monotonic_nanos().max(1), Ordering::SeqCst);
    WORKER.call_once(|| {
        if let Err(error) = std::thread::Builder::new()
            .name("nexium-max-clocks".into())
            .spawn(run)
        {
            log::warn!("force max clocks: worker spawn failed: {error}");
        }
    });
    if SIGNAL.parked.load(Ordering::SeqCst) {
        let _guard = SIGNAL.lock.lock().unwrap_or_else(|error| error.into_inner());
        SIGNAL.wake.notify_one();
    }
}

fn active() -> bool {
    nexium_common::force_max_clocks::enabled()
        && crate::renderer::monotonic_nanos()
            .saturating_sub(LAST_SUBMISSION_NS.load(Ordering::SeqCst))
            < ACTIVE_WINDOW.as_nanos() as u64
}

fn park() {
    let guard = SIGNAL.lock.lock().unwrap_or_else(|error| error.into_inner());
    SIGNAL.parked.store(true, Ordering::SeqCst);
    let guard = if active() {
        guard
    } else {
        SIGNAL
            .wake
            .wait_timeout(guard, PARK_TIMEOUT)
            .map(|(guard, _)| guard)
            .unwrap_or_else(|error| error.into_inner().0)
    };
    SIGNAL.parked.store(false, Ordering::SeqCst);
    drop(guard);
}

fn next_rounds(rounds: u32, elapsed: Duration) -> u32 {
    let next = if elapsed < TARGET_DISPATCH * 3 / 4 {
        rounds.saturating_add(rounds / 4 + 1)
    } else if elapsed > TARGET_DISPATCH * 5 / 4 {
        rounds - rounds / 5
    } else {
        rounds
    };
    next.clamp(MIN_ROUNDS, MAX_ROUNDS)
}

fn run() {
    let mut engine = match Engine::new() {
        Ok(engine) => engine,
        Err(error) => {
            log::warn!("force max clocks unavailable: {error}");
            return;
        }
    };
    let mut rounds = 256;
    loop {
        if !active() {
            park();
            continue;
        }
        let started = Instant::now();
        if let Err(error) = engine.dispatch(rounds) {
            log::warn!("force max clocks stopped: {error}");
            return;
        }
        rounds = next_rounds(rounds, started.elapsed());
    }
}

struct Engine {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: Option<ash::Device>,
    queue: vk::Queue,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    seed: u32,
}

impl Engine {
    fn new() -> Result<Self, String> {
        let entry = crate::adapter::vulkan_entry()?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"NeXium max clocks")
            .api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
        }
        .map_err(|error| format!("create instance: {error:?}"))?;
        let mut engine = Self {
            _entry: entry,
            instance,
            device: None,
            queue: vk::Queue::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            seed: 0x2545_f491,
        };
        engine.init()?;
        Ok(engine)
    }

    fn init(&mut self) -> Result<(), String> {
        let physical = crate::adapter::physical_devices(&self.instance)?;
        let infos: Vec<_> = physical.iter().map(|(_, info)| info.clone()).collect();
        let preferred = crate::adapter::preferred_device();
        let index = crate::adapter::select_index(&infos, preferred.as_deref())
            .ok_or_else(|| "no Vulkan 1.3 device".to_string())?;
        let physical_device = physical[index].0;
        let families = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(physical_device)
        };
        let has = |flags: vk::QueueFlags, without: vk::QueueFlags| {
            families.iter().position(|family| {
                family.queue_count != 0
                    && family.queue_flags.contains(flags)
                    && !family.queue_flags.intersects(without)
            })
        };
        let (family, async_compute) = match has(vk::QueueFlags::COMPUTE, vk::QueueFlags::GRAPHICS) {
            Some(family) => (family as u32, true),
            None => (
                has(vk::QueueFlags::COMPUTE, vk::QueueFlags::empty())
                    .ok_or_else(|| "no compute queue".to_string())? as u32,
                false,
            ),
        };
        let extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let priority_extension = [vk::KHR_GLOBAL_PRIORITY_NAME, vk::EXT_GLOBAL_PRIORITY_NAME]
            .into_iter()
            .find(|wanted| {
                extensions.iter().any(|extension| {
                    extension.extension_name_as_c_str().is_ok_and(|name| name == *wanted)
                })
            });
        let (device, low_priority) = match priority_extension {
            Some(extension) => match self.create_device(physical_device, family, Some(extension)) {
                Ok(device) => (device, true),
                Err(_) => (self.create_device(physical_device, family, None)?, false),
            },
            None => (self.create_device(physical_device, family, None)?, false),
        };
        self.queue = unsafe { device.get_device_queue(family, 0) };
        let device = self.device.insert(device);

        self.buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(SINK_BYTES)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|error| format!("create buffer: {error:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(self.buffer) };
        let memory_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(physical_device)
        };
        let memory_types =
            &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
        let allowed = |index: usize| requirements.memory_type_bits & (1 << index) != 0;
        let memory_type = (0..memory_types.len())
            .find(|&index| {
                allowed(index)
                    && memory_types[index]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .or_else(|| (0..memory_types.len()).find(|&index| allowed(index)))
            .ok_or_else(|| "no memory type for the sink buffer".to_string())?;
        self.memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type as u32),
                None,
            )
        }
        .map_err(|error| format!("allocate memory: {error:?}"))?;
        unsafe { device.bind_buffer_memory(self.buffer, self.memory, 0) }
            .map_err(|error| format!("bind memory: {error:?}"))?;

        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        self.descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
        }
        .map_err(|error| format!("descriptor layout: {error:?}"))?;
        let set_layouts = [self.descriptor_layout];
        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        self.pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_constants),
                None,
            )
        }
        .map_err(|error| format!("pipeline layout: {error:?}"))?;
        let code: Vec<_> = include_bytes!(concat!(env!("OUT_DIR"), "/max_clocks_main.spv"))
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
            .layout(self.pipeline_layout)];
        let pipelines =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &info, None) };
        unsafe { device.destroy_shader_module(module, None) };
        self.pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
                return Err(format!("compute pipeline: {error:?}"));
            }
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        self.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| format!("descriptor pool: {error:?}"))?;
        self.descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|error| format!("descriptor set: {error:?}"))?[0];
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(SINK_BYTES)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info)];
        unsafe { device.update_descriptor_sets(&write, &[]) };

        self.command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(family),
                None,
            )
        }
        .map_err(|error| format!("command pool: {error:?}"))?;
        self.command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| format!("command buffer: {error:?}"))?[0];
        self.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|error| format!("fence: {error:?}"))?;
        log::info!(
            "force max clocks: worker ready on {} ({} queue, {} priority)",
            infos[index].label(),
            if async_compute { "async compute" } else { "shared" },
            if low_priority { "low" } else { "default" },
        );
        Ok(())
    }

    fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        family: u32,
        priority_extension: Option<&std::ffi::CStr>,
    ) -> Result<ash::Device, String> {
        let priorities = [0.0f32];
        let mut global_priority = vk::DeviceQueueGlobalPriorityCreateInfoKHR::default()
            .global_priority(vk::QueueGlobalPriorityKHR::LOW);
        let mut queue = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities);
        if priority_extension.is_some() {
            queue = queue.push_next(&mut global_priority);
        }
        let queues = [queue];
        let extensions: Vec<_> = priority_extension.iter().map(|name| name.as_ptr()).collect();
        unsafe {
            self.instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&extensions),
                None,
            )
        }
        .map_err(|error| format!("create device: {error:?}"))
    }

    fn dispatch(&mut self, rounds: u32) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("no device")?;
        self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let constants = [rounds.to_le_bytes(), self.seed.to_le_bytes()].concat();
        let barrier = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        let command_buffers = [self.command_buffer];
        unsafe {
            device
                .begin_command_buffer(
                    self.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| format!("begin: {error:?}"))?;
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &constants,
            );
            device.cmd_dispatch(self.command_buffer, GROUPS_X, GROUPS_Y, 1);
            device
                .end_command_buffer(self.command_buffer)
                .map_err(|error| format!("end: {error:?}"))?;
            device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                    self.fence,
                )
                .map_err(|error| format!("submit: {error:?}"))?;
            loop {
                match device.wait_for_fences(&[self.fence], true, 1_000_000_000) {
                    Ok(()) => break,
                    Err(vk::Result::TIMEOUT) => continue,
                    Err(error) => return Err(format!("wait: {error:?}")),
                }
            }
            device
                .reset_fences(&[self.fence])
                .map_err(|error| format!("reset fence: {error:?}"))?;
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = self.device.take() {
                let _ = device.device_wait_idle();
                if self.fence != vk::Fence::null() {
                    device.destroy_fence(self.fence, None);
                }
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                }
                if self.descriptor_pool != vk::DescriptorPool::null() {
                    device.destroy_descriptor_pool(self.descriptor_pool, None);
                }
                if self.pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.pipeline, None);
                }
                if self.pipeline_layout != vk::PipelineLayout::null() {
                    device.destroy_pipeline_layout(self.pipeline_layout, None);
                }
                if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                    device.destroy_descriptor_set_layout(self.descriptor_layout, None);
                }
                if self.buffer != vk::Buffer::null() {
                    device.destroy_buffer(self.buffer, None);
                }
                if self.memory != vk::DeviceMemory::null() {
                    device.free_memory(self.memory, None);
                }
                device.destroy_device(None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_size_converges_on_the_target_and_stays_bounded() {
        assert!(next_rounds(256, Duration::from_micros(20)) > 256);
        assert!(next_rounds(256, Duration::from_millis(2)) < 256);
        assert_eq!(next_rounds(256, TARGET_DISPATCH), 256);
        assert_eq!(next_rounds(MAX_ROUNDS, Duration::ZERO), MAX_ROUNDS);
        assert_eq!(next_rounds(MIN_ROUNDS, Duration::from_secs(1)), MIN_ROUNDS);
    }
}

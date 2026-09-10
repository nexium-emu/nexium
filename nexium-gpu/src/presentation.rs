use ash::vk;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use crate::renderer::SubmitState;
use crate::rt_cache::find_memory_type;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct SurfaceState {
    pub width: u32,
    pub height: u32,
    pub visible: bool,
    pub vsync: bool,
    pub nearest: bool,
}

pub struct PresentationTarget {
    pub(crate) hwnd: isize,
    pub(crate) hinstance: isize,
    state: Mutex<SurfaceState>,
    stopped: AtomicBool,
    sequence: AtomicU64,
    presented: AtomicU64,
    snapshots: AtomicU64,
    dimensions: AtomicU64,
    snapshot_requested: AtomicBool,
    snapshot: Mutex<Option<Snapshot>>,
    error: Mutex<Option<String>>,
    repaint: Arc<dyn Fn() + Send + Sync>,
}

pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl PresentationTarget {
    pub unsafe fn win32(
        hwnd: isize,
        hinstance: isize,
        repaint: Arc<dyn Fn() + Send + Sync>,
    ) -> Arc<Self> {
        Arc::new(Self {
            hwnd,
            hinstance,
            state: Mutex::new(SurfaceState {
                vsync: true,
                ..Default::default()
            }),
            stopped: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
            presented: AtomicU64::new(0),
            snapshots: AtomicU64::new(0),
            dimensions: AtomicU64::new(0),
            snapshot_requested: AtomicBool::new(false),
            snapshot: Mutex::new(None),
            error: Mutex::new(None),
            repaint,
        })
    }

    pub fn configure(&self, state: SurfaceState) {
        *self.state.lock() = state;
    }

    pub fn progress(&self) -> (u64, u32, u32) {
        let sequence = self.sequence.load(Ordering::Acquire);
        let dimensions = self.dimensions.load(Ordering::Acquire);
        (sequence, (dimensions >> 32) as u32, dimensions as u32)
    }

    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
    }

    pub fn metrics(&self) -> (u64, u64, u64) {
        (
            self.sequence.load(Ordering::Acquire),
            self.presented.load(Ordering::Acquire),
            self.snapshots.load(Ordering::Acquire),
        )
    }

    pub fn request_snapshot(&self) {
        self.snapshot_requested.store(true, Ordering::Release);
    }

    pub fn take_snapshot(&self) -> Option<Snapshot> {
        self.snapshot.lock().take()
    }

    pub fn take_error(&self) -> Option<String> {
        self.error.lock().take()
    }

    fn fail(&self, error: String) {
        log::error!("[vulkan-present] {error}");
        *self.error.lock() = Some(error);
        self.stop();
        (self.repaint)();
    }
}

#[derive(Clone, Copy)]
pub struct PresentParameters {
    pub read_rect: Option<[u32; 4]>,
    pub crop: Option<[u32; 4]>,
    pub flip_y: bool,
    pub transform: u32,
    pub present_at: Instant,
}

impl PresentParameters {
    fn mapping(self, width: u32, height: u32) -> ([f32; 6], [u32; 2]) {
        let [x, y, w, h] = valid_rect(self.read_rect, width, height);
        let rotated = self.transform & 4 != 0;
        let (ow, oh) = if rotated { (h, w) } else { (w, h) };
        let [cx, cy, cw, ch] = valid_rect(self.crop, ow, oh);
        let map = |u: f32, v: f32| {
            let u = (cx as f32 + u * cw as f32) / ow as f32;
            let v = (cy as f32 + v * ch as f32) / oh as f32;
            let (mut u, mut v) = if rotated { (v, 1.0 - u) } else { (u, v) };
            if self.transform & 1 != 0 {
                u = 1.0 - u;
            }
            if (self.transform & 2 != 0) ^ self.flip_y {
                v = 1.0 - v;
            }
            [
                (x as f32 + u * w as f32) / width as f32,
                (y as f32 + v * h as f32) / height as f32,
            ]
        };
        let origin = map(0.0, 0.0);
        let right = map(1.0, 0.0);
        let bottom = map(0.0, 1.0);
        (
            [
                origin[0],
                origin[1],
                right[0] - origin[0],
                right[1] - origin[1],
                bottom[0] - origin[0],
                bottom[1] - origin[1],
            ],
            [cw, ch],
        )
    }
}

fn valid_rect(rect: Option<[u32; 4]>, width: u32, height: u32) -> [u32; 4] {
    rect.filter(|r| {
        r[2] != 0
            && r[3] != 0
            && r[0] < width
            && r[1] < height
            && r[2] <= width - r[0]
            && r[3] <= height - r[1]
    })
    .unwrap_or([0, 0, width, height])
}

pub(crate) struct FrameSlot {
    device: ash::Device,
    pub image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub generation: u64,
    parameters: PresentParameters,
}

impl FrameSlot {
    fn new(device: &ash::Device, family: u32) -> Result<Self, String> {
        let mut slot = Self {
            device: device.clone(),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            generation: 0,
            parameters: PresentParameters {
                read_rect: None,
                crop: None,
                flip_y: false,
                transform: 0,
                present_at: Instant::now(),
            },
        };
        unsafe {
            slot.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(err)?;
            slot.cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(slot.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(err)?[0];
            slot.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(err)?;
        }
        Ok(slot)
    }

    pub fn prepare(
        &mut self,
        mem: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(), String> {
        if self.extent == (vk::Extent2D { width, height }) && self.format == format {
            return Ok(());
        }
        self.destroy_image();
        self.extent = vk::Extent2D { width, height };
        self.format = format;
        unsafe {
            self.image = self
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(err)?;
            let requirements = self.device.get_image_memory_requirements(self.image);
            let memory_type_index = find_memory_type(
                mem,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or("no presentation image memory")?;
            self.memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type_index),
                    None,
                )
                .map_err(err)?;
            self.device
                .bind_image_memory(self.image, self.memory, 0)
                .map_err(err)?;
            self.view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(color_range()),
                    None,
                )
                .map_err(err)?;
        }
        Ok(())
    }

    fn destroy_image(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
        self.view = vk::ImageView::null();
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
        self.extent = vk::Extent2D::default();
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        self.destroy_image();
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

pub(crate) struct Presenter {
    target: Arc<PresentationTarget>,
    stop: Arc<AtomicBool>,
    free: Mutex<mpsc::Receiver<FrameSlot>>,
    recycle: mpsc::Sender<FrameSlot>,
    pending: mpsc::Sender<FrameSlot>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Presenter {
    pub fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        family: u32,
        queue: vk::Queue,
        timeline: vk::Semaphore,
        submit: Arc<SubmitState>,
        target: Arc<PresentationTarget>,
        present_fences: bool,
    ) -> Result<Self, String> {
        if timeline == vk::Semaphore::null() {
            return Err("native presentation requires timeline semaphores".into());
        }
        let (pending, receiver) = mpsc::channel();
        let (recycle, free) = mpsc::channel();
        for _ in 0..4 {
            recycle
                .send(FrameSlot::new(device, family)?)
                .map_err(|e| e.to_string())?;
        }
        let worker = Worker::new(
            entry,
            instance,
            device,
            physical,
            family,
            queue,
            timeline,
            submit,
            target.clone(),
            present_fences,
        )?;
        let stop = worker.stop.clone();
        let thread_target = target.clone();
        let thread_recycle = recycle.clone();
        let handle = std::thread::Builder::new()
            .name("Vulkan presentation".into())
            .spawn(move || {
                let mut worker = worker;
                if let Err(error) = worker.run(&receiver, &thread_recycle) {
                    thread_target.fail(error);
                }
                worker.drain();
                while let Ok(slot) = receiver.try_recv() {
                    unsafe {
                        let _ = worker.device.wait_for_fences(&[slot.fence], true, u64::MAX);
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        log::info!("[vulkan-present] dedicated presentation worker started; four GPU frame slots");
        Ok(Self {
            target,
            stop,
            free: Mutex::new(free),
            recycle,
            pending,
            worker: Some(handle),
        })
    }

    pub fn reserve(&self) -> Option<FrameSlot> {
        let free = self.free.lock();
        while !self.target.stopped.load(Ordering::Acquire) {
            match free.recv_timeout(Duration::from_millis(20)) {
                Ok(slot) => return Some(slot),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
        None
    }

    pub fn recycle(&self, slot: FrameSlot) {
        let _ = self.recycle.send(slot);
    }

    pub fn fail(&self, error: String) {
        self.target.fail(error);
    }

    pub fn submit(&self, mut slot: FrameSlot, parameters: PresentParameters) -> Result<(), String> {
        slot.parameters = parameters;
        self.pending.send(slot).map_err(|e| {
            unsafe {
                let _ = e.0.device.wait_for_fences(&[e.0.fence], true, u64::MAX);
            }
            "presentation worker stopped".into()
        })
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct SwapImage {
    image: vk::Image,
    view: vk::ImageView,
    complete: vk::Semaphore,
    present_fence: vk::Fence,
    pending_present: bool,
}

struct RetiredSwapchain {
    handle: vk::SwapchainKHR,
    images: Vec<SwapImage>,
}

struct Worker {
    device: ash::Device,
    surface_api: ash::khr::surface::Instance,
    swap_api: ash::khr::swapchain::Device,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<SwapImage>,
    retired: Vec<RetiredSwapchain>,
    present_fences: bool,
    extent: vk::Extent2D,
    vsync: bool,
    queue: vk::Queue,
    uses_render_queue: bool,
    timeline: vk::Semaphore,
    submit: Arc<SubmitState>,
    target: Arc<PresentationTarget>,
    stop: Arc<AtomicBool>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    fence_pending: bool,
    acquired: vk::Semaphore,
    descriptors: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    samplers: [vk::Sampler; 2],
    last: Option<FrameSlot>,
}

impl Worker {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        family: u32,
        queue: vk::Queue,
        timeline: vk::Semaphore,
        submit: Arc<SubmitState>,
        target: Arc<PresentationTarget>,
        present_fences: bool,
    ) -> Result<Self, String> {
        let mut worker = Self {
            device: device.clone(),
            surface_api: ash::khr::surface::Instance::new(entry, instance),
            swap_api: ash::khr::swapchain::Device::new(instance, device),
            instance: instance.clone(),
            physical,
            surface: vk::SurfaceKHR::null(),
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            retired: Vec::new(),
            present_fences,
            extent: vk::Extent2D::default(),
            vsync: true,
            queue,
            timeline,
            uses_render_queue: queue == unsafe { device.get_device_queue(family, 0) },
            stop: Arc::new(AtomicBool::new(false)),
            submit,
            target,
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            fence_pending: false,
            acquired: vk::Semaphore::null(),
            descriptors: vk::DescriptorPool::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            set: vk::DescriptorSet::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            samplers: [vk::Sampler::null(); 2],
            last: None,
        };
        #[cfg(windows)]
        unsafe {
            let api = ash::khr::win32_surface::Instance::new(entry, instance);
            worker.surface = api
                .create_win32_surface(
                    &vk::Win32SurfaceCreateInfoKHR::default()
                        .hwnd(worker.target.hwnd)
                        .hinstance(worker.target.hinstance),
                    None,
                )
                .map_err(err)?;
        }
        if worker.surface == vk::SurfaceKHR::null() {
            return Err("unsupported native presentation surface".into());
        }
        unsafe {
            if !worker
                .surface_api
                .get_physical_device_surface_support(physical, family, worker.surface)
                .map_err(err)?
            {
                return Err("graphics queue cannot present to the game surface".into());
            }
            worker.pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(err)?;
            worker.cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(worker.pool)
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .map_err(err)?[0];
            worker.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(err)?;
            worker.acquired = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(err)?;
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            worker.set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(err)?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                },
            ];
            worker.descriptors = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(err)?;
            worker.set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(worker.descriptors)
                        .set_layouts(&[worker.set_layout]),
                )
                .map_err(err)?[0];
            let push = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: 24,
            }];
            worker.pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[worker.set_layout])
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(err)?;
            for (i, filter) in [vk::Filter::LINEAR, vk::Filter::NEAREST]
                .into_iter()
                .enumerate()
            {
                worker.samplers[i] = device
                    .create_sampler(
                        &vk::SamplerCreateInfo::default()
                            .min_filter(filter)
                            .mag_filter(filter)
                            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                        None,
                    )
                    .map_err(err)?;
            }
        }
        Ok(worker)
    }

    fn run(
        &mut self,
        receiver: &mpsc::Receiver<FrameSlot>,
        recycle: &mpsc::Sender<FrameSlot>,
    ) -> Result<(), String> {
        while !self.target.stopped.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire) {
            let mut incoming = match receiver.recv_timeout(Duration::from_millis(8)) {
                Ok(slot) => Some(slot),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let state = *self.target.state.lock();
            if let Some(slot) = incoming.take() {
                unsafe {
                    self.device
                        .wait_for_fences(&[slot.fence], true, u64::MAX)
                        .map_err(err)?;
                }
                if let Some(previous) = self.last.replace(slot) {
                    let _ = recycle.send(previous);
                }
                let (_, size) = self.last.as_ref().unwrap().parameters.mapping(
                    self.last.as_ref().unwrap().extent.width,
                    self.last.as_ref().unwrap().extent.height,
                );
                self.target.dimensions.store(
                    (u64::from(size[0]) << 32) | u64::from(size[1]),
                    Ordering::Release,
                );
                self.target.sequence.fetch_add(1, Ordering::Release);
                (self.target.repaint)();
                if state.visible && state.width != 0 && state.height != 0 {
                    self.present(state)?;
                } else {
                    self.wait_present_time(self.last.as_ref().unwrap().parameters.present_at);
                }
            } else if self.last.is_some()
                && state.visible
                && state.width != 0
                && state.height != 0
                && (self.swapchain == vk::SwapchainKHR::null()
                    || self.extent
                        != vk::Extent2D {
                            width: state.width,
                            height: state.height,
                        }
                    || self.vsync != state.vsync)
            {
                self.present(state)?;
            }
            if self.target.snapshot_requested.swap(false, Ordering::AcqRel) && self.last.is_some() {
                let snapshot = self.snapshot()?;
                self.target.snapshots.fetch_add(1, Ordering::Relaxed);
                *self.target.snapshot.lock() = Some(snapshot);
                (self.target.repaint)();
            }
        }
        Ok(())
    }

    fn queue_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        let mutex = if self.uses_render_queue {
            &self.submit.sequence
        } else {
            &self.submit.presentation_sequence
        };
        mutex.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn wait_present_time(&self, deadline: Instant) {
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            if self.target.stopped.load(Ordering::Acquire) || self.stop.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(2)));
        }
    }

    fn wait_submission(&mut self) -> Result<(), String> {
        if self.fence_pending {
            unsafe {
                self.device
                    .wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(err)?;
            }
            self.fence_pending = false;
        }
        Ok(())
    }

    fn drain(&mut self) {
        let _ = self.wait_submission();
        if !self.present_fences {
            if self.images.iter().any(|image| image.pending_present)
                || self
                    .retired
                    .iter()
                    .any(|swapchain| swapchain.images.iter().any(|image| image.pending_present))
            {
                let _sequence = self.queue_lock();
                unsafe {
                    let _ = self.device.queue_wait_idle(self.queue);
                }
            }
            for image in self.images.iter_mut().chain(
                self.retired
                    .iter_mut()
                    .flat_map(|swapchain| &mut swapchain.images),
            ) {
                image.pending_present = false;
            }
            return;
        }
        for image in &mut self.images {
            if image.pending_present {
                unsafe {
                    let _ = self
                        .device
                        .wait_for_fences(&[image.present_fence], true, u64::MAX);
                }
                image.pending_present = false;
            }
        }
    }

    fn retire_swapchain(&mut self) {
        let _ = self.wait_submission();
        if self.present_fences {
            self.drain();
        }
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.pipeline = vk::Pipeline::null();
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            let swapchain = RetiredSwapchain {
                handle: std::mem::replace(&mut self.swapchain, vk::SwapchainKHR::null()),
                images: std::mem::take(&mut self.images),
            };
            if self.present_fences {
                self.release_swapchain(swapchain);
            } else {
                self.retired.push(swapchain);
            }
        }
    }

    fn release_swapchain(&self, swapchain: RetiredSwapchain) {
        unsafe {
            for image in swapchain.images {
                self.device.destroy_image_view(image.view, None);
                self.device.destroy_semaphore(image.complete, None);
                self.device.destroy_fence(image.present_fence, None);
            }
            self.swap_api.destroy_swapchain(swapchain.handle, None);
        }
    }

    fn release_retired(&mut self) {
        for swapchain in std::mem::take(&mut self.retired) {
            self.release_swapchain(swapchain);
        }
    }

    fn recreate(&mut self, state: SurfaceState) -> Result<(), String> {
        self.retire_swapchain();
        unsafe {
            let caps = self
                .surface_api
                .get_physical_device_surface_capabilities(self.physical, self.surface)
                .map_err(err)?;
            let formats = self
                .surface_api
                .get_physical_device_surface_formats(self.physical, self.surface)
                .map_err(err)?;
            let format = formats
                .iter()
                .copied()
                .find(|f| {
                    matches!(
                        f.format,
                        vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM
                    ) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .ok_or("surface has no RGBA/BGRA UNORM format")?;
            if !caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                return Err("surface cannot be a color attachment".into());
            }
            self.extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: state
                        .width
                        .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: state
                        .height
                        .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            if self.extent.width == 0 || self.extent.height == 0 {
                return Ok(());
            }
            self.vsync = state.vsync;
            let modes = self
                .surface_api
                .get_physical_device_surface_present_modes(self.physical, self.surface)
                .map_err(err)?;
            let mode = if !state.vsync && modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            };
            let count = caps
                .min_image_count
                .max(3)
                .min(if caps.max_image_count == 0 {
                    u32::MAX
                } else {
                    caps.max_image_count
                });
            let alpha = [
                vk::CompositeAlphaFlagsKHR::OPAQUE,
                vk::CompositeAlphaFlagsKHR::INHERIT,
                vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            ]
            .into_iter()
            .find(|f| caps.supported_composite_alpha.contains(*f))
            .ok_or("no composite alpha mode")?;
            self.swapchain = self
                .swap_api
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.surface)
                        .min_image_count(count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(self.extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(alpha)
                        .present_mode(mode)
                        .clipped(true)
                        .old_swapchain(
                            self.retired
                                .last()
                                .map_or(vk::SwapchainKHR::null(), |swapchain| swapchain.handle),
                        ),
                    None,
                )
                .map_err(err)?;
            for image in self
                .swap_api
                .get_swapchain_images(self.swapchain)
                .map_err(err)?
            {
                self.images.push(SwapImage {
                    image,
                    view: vk::ImageView::null(),
                    complete: vk::Semaphore::null(),
                    present_fence: vk::Fence::null(),
                    pending_present: false,
                });
                let out = self.images.last_mut().unwrap();
                out.view = self
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(color_range()),
                        None,
                    )
                    .map_err(err)?;
                out.complete = self
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(err)?;
                if self.present_fences {
                    out.present_fence = self
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .map_err(err)?;
                }
            }
            self.pipeline = create_pipeline(&self.device, self.pipeline_layout, format.format)?;
            log::info!(
                "[vulkan-present] swapchain {}x{} {:?}, {} images",
                self.extent.width,
                self.extent.height,
                mode,
                count
            );
        }
        Ok(())
    }

    fn present(&mut self, mut state: SurfaceState) -> Result<(), String> {
        self.wait_submission()?;
        let (index, suboptimal) = loop {
            if !state.visible
                || state.width == 0
                || state.height == 0
                || self.target.stopped.load(Ordering::Acquire)
                || self.stop.load(Ordering::Acquire)
            {
                return Ok(());
            }
            if self.swapchain == vk::SwapchainKHR::null()
                || self.extent
                    != (vk::Extent2D {
                        width: state.width,
                        height: state.height,
                    })
                || self.vsync != state.vsync
            {
                self.recreate(state)?;
            }
            if self.swapchain == vk::SwapchainKHR::null() {
                return Ok(());
            }
            match unsafe {
                self.swap_api.acquire_next_image(
                    self.swapchain,
                    20_000_000,
                    self.acquired,
                    vk::Fence::null(),
                )
            } {
                Ok(value) => break value,
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.retire_swapchain(),
                Err(error) => return Err(err(error)),
            }
            state = *self.target.state.lock();
        };
        let image = &mut self.images[index as usize];
        let release_retired = !self.present_fences && image.pending_present;
        let present_fence = image.present_fence;
        unsafe {
            if self.present_fences {
                if image.pending_present {
                    self.device
                        .wait_for_fences(&[image.present_fence], true, u64::MAX)
                        .map_err(err)?;
                }
                self.device
                    .reset_fences(&[image.present_fence])
                    .map_err(err)?;
            }
            image.pending_present = false;
            self.device.reset_fences(&[self.fence]).map_err(err)?;
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(err)?;
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(err)?;
        }
        let slot = self.last.as_ref().unwrap();
        let (mapping, _) = slot
            .parameters
            .mapping(slot.extent.width, slot.extent.height);
        let view = [vk::DescriptorImageInfo::default()
            .image_view(slot.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sampler =
            [vk::DescriptorImageInfo::default().sampler(self.samplers[state.nearest as usize])];
        unsafe {
            self.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&view),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler),
                ],
                &[],
            );
        }
        barrier(
            &self.device,
            self.cmd,
            image.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        let color = [vk::RenderingAttachmentInfo::default()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rect = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: self.extent,
        };
        unsafe {
            self.device.cmd_begin_rendering(
                self.cmd,
                &vk::RenderingInfo::default()
                    .render_area(rect)
                    .layer_count(1)
                    .color_attachments(&color),
            );
            self.device
                .cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.set],
                &[],
            );
            self.device.cmd_set_viewport(
                self.cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(self.cmd, 0, &[rect]);
            let bytes: [u8; 24] = std::array::from_fn(|i| mapping[i / 4].to_ne_bytes()[i % 4]);
            self.device.cmd_push_constants(
                self.cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &bytes,
            );
            self.device.cmd_draw(self.cmd, 3, 1, 0, 0);
            self.device.cmd_end_rendering(self.cmd);
        }
        barrier(
            &self.device,
            self.cmd,
            image.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        unsafe {
            self.device.end_command_buffer(self.cmd).map_err(err)?;
        }
        let waits = [self.acquired, self.timeline];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
        let wait_values = [0, slot.generation];
        let signal_values = [0];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let commands = [self.cmd];
        let signals = [image.complete];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signals)
            .push_next(&mut timeline);
        {
            let _sequence = self.queue_lock();
            unsafe {
                self.device
                    .queue_submit(self.queue, &[submit], self.fence)
                    .map_err(err)?;
            }
        }
        self.fence_pending = true;
        self.wait_present_time(slot.parameters.present_at);
        let swaps = [self.swapchain];
        let indices = [index];
        let fences = [present_fence];
        let mut completion = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
        let mut present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signals)
            .swapchains(&swaps)
            .image_indices(&indices);
        if self.present_fences {
            present = present.push_next(&mut completion);
        }
        let result = {
            let _sequence = self.queue_lock();
            unsafe { self.swap_api.queue_present(self.queue, &present) }
        };
        self.images[index as usize].pending_present = matches!(
            result,
            Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::ERROR_SURFACE_LOST_KHR)
        );
        if result.is_ok() {
            let count = self.target.presented.fetch_add(1, Ordering::Relaxed) + 1;
            if count % 300 == 0 {
                log::info!(
                    "[vulkan-present] presented={count} snapshots={}",
                    self.target.snapshots.load(Ordering::Relaxed)
                );
            }
        }
        self.wait_submission()?;
        if release_retired {
            self.release_retired();
        }
        match result {
            Ok(changed) if changed || suboptimal => self.retire_swapchain(),
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.retire_swapchain(),
            Err(error) => return Err(err(error)),
        }
        Ok(())
    }

    fn snapshot(&mut self) -> Result<Snapshot, String> {
        self.wait_submission()?;
        let slot = self.last.as_ref().unwrap();
        let width = slot.extent.width;
        let height = slot.extent.height;
        let len = u64::from(width) * u64::from(height) * 4;
        let memory = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical)
        };
        let mut buffer = HostBuffer {
            device: self.device.clone(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
        };
        unsafe {
            buffer.buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(len)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .map_err(err)?;
            let requirements = self.device.get_buffer_memory_requirements(buffer.buffer);
            let ty = find_memory_type(
                &memory,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or("no snapshot memory")?;
            buffer.memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(ty),
                    None,
                )
                .map_err(err)?;
            self.device
                .bind_buffer_memory(buffer.buffer, buffer.memory, 0)
                .map_err(err)?;
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(err)?;
            self.device.reset_fences(&[self.fence]).map_err(err)?;
            self.device
                .begin_command_buffer(
                    self.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(err)?;
        }
        barrier(
            &self.device,
            self.cmd,
            slot.image,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(color_layers())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.cmd,
                slot.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer,
                &[copy],
            );
        }
        barrier(
            &self.device,
            self.cmd,
            slot.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        unsafe {
            self.device.end_command_buffer(self.cmd).map_err(err)?;
        }
        let wait_values = [slot.generation];
        let wait_semaphores = [self.timeline];
        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let commands = [self.cmd];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let submission = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .push_next(&mut timeline);
        {
            let _sequence = self.queue_lock();
            unsafe {
                self.device
                    .queue_submit(self.queue, &[submission], self.fence)
                    .map_err(err)?;
            }
        }
        self.fence_pending = true;
        let parameters = slot.parameters;
        let format = slot.format;
        self.wait_submission()?;
        let bytes = unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, len, vk::MemoryMapFlags::empty())
                .map_err(err)?;
            let bytes = std::slice::from_raw_parts(ptr as *const u8, len as usize).to_vec();
            self.device.unmap_memory(buffer.memory);
            bytes
        };
        let (mapping, [out_w, out_h]) = parameters.mapping(width, height);
        let mut pixels = vec![0; out_w as usize * out_h as usize * 4];
        for y in 0..out_h {
            for x in 0..out_w {
                let u = (x as f32 + 0.5) / out_w as f32;
                let v = (y as f32 + 0.5) / out_h as f32;
                let sx = ((mapping[0] + mapping[2] * u + mapping[4] * v) * width as f32) as u32;
                let sy = ((mapping[1] + mapping[3] * u + mapping[5] * v) * height as f32) as u32;
                let src = ((sy.min(height - 1) * width + sx.min(width - 1)) * 4) as usize;
                let dst = ((y * out_w + x) * 4) as usize;
                pixels[dst..dst + 4].copy_from_slice(&bytes[src..src + 4]);
                if format == vk::Format::B8G8R8A8_UNORM {
                    pixels.swap(dst, dst + 2);
                }
                pixels[dst + 3] = 255;
            }
        }
        log::debug!("[vulkan-present] pause snapshot {}x{}", out_w, out_h);
        Ok(Snapshot {
            width: out_w,
            height: out_h,
            pixels,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.drain();
        self.retire_swapchain();
        self.release_retired();
        self.last.take();
        unsafe {
            for sampler in self.samplers {
                self.device.destroy_sampler(sampler, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_semaphore(self.acquired, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.surface_api.destroy_surface(self.surface, None);
        }
    }
}

struct HostBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn create_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    format: vk::Format,
) -> Result<vk::Pipeline, String> {
    let vertex = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/present_vertex.spv"
    ))))
    .map_err(|e| e.to_string())?;
    let fragment = ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/present_fragment.spv"
    ))))
    .map_err(|e| e.to_string())?;
    unsafe {
        let vert = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vertex), None)
            .map_err(err)?;
        let frag = match device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fragment), None)
        {
            Ok(module) => module,
            Err(error) => {
                device.destroy_shader_module(vert, None);
                return Err(err(error));
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"vertex"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"fragment"),
        ];
        let input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let formats = [format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&samples)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        let result = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
        match result {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((pipelines, error)) => {
                for pipeline in pipelines {
                    device.destroy_pipeline(pipeline, None);
                }
                Err(err(error))
            }
        }
    }
}

pub(crate) fn color_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

pub(crate) fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let src = if old == vk::ImageLayout::UNDEFINED {
        vk::AccessFlags::empty()
    } else {
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    };
    let dst = if new == vk::ImageLayout::PRESENT_SRC_KHR {
        vk::AccessFlags::empty()
    } else {
        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(color_range());
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn err(error: vk::Result) -> String {
    format!("Vulkan presentation: {error:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> PresentParameters {
        PresentParameters {
            read_rect: None,
            crop: None,
            flip_y: false,
            transform: 0,
            present_at: Instant::now(),
        }
    }

    #[test]
    fn orientation_and_crop_preserve_the_selected_pixels() {
        let mut p = parameters();
        p.flip_y = true;
        p.crop = Some([10, 20, 50, 40]);
        let (m, size) = p.mapping(100, 100);
        assert_eq!(size, [50, 40]);
        for (actual, expected) in m.into_iter().zip([0.1, 0.8, 0.5, 0.0, 0.0, -0.4]) {
            assert!((actual - expected).abs() < 0.00001);
        }
        p.transform = 2;
        assert!(p.mapping(100, 100).0[5] > 0.0);
    }

    #[test]
    fn rotation_exchanges_dimensions_and_axes() {
        let mut p = parameters();
        p.transform = 4;
        assert_eq!(
            p.mapping(120, 80),
            ([0.0, 1.0, 0.0, -1.0, 1.0, 0.0], [80, 120])
        );
    }

    #[test]
    fn invalid_crop_cannot_overflow_or_empty_a_frame() {
        assert_eq!(
            valid_rect(Some([u32::MAX, 0, 2, 2]), 100, 50),
            [0, 0, 100, 50]
        );
        assert_eq!(valid_rect(Some([0, 0, 0, 2]), 100, 50), [0, 0, 100, 50]);
    }
}

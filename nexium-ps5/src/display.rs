use std::ffi::{c_char, CStr};
use std::time::Instant;

use ash::{khr, vk};

pub fn entry() -> ash::Entry {
    nexium_gpu::adapter::vulkan_entry().expect("linked RADV entry")
}

pub fn vkerr(what: &str) -> impl Fn(vk::Result) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

pub fn name_of(raw: &[c_char]) -> String {
    unsafe { CStr::from_ptr(raw.as_ptr()) }.to_string_lossy().into_owned()
}

pub fn spv(bytes: &[u8]) -> Vec<u32> {
    ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("spv")
}

pub fn memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(flags))
}

pub struct HostBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
}

pub unsafe fn host_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<HostBuffer, String> {
    unsafe {
        let buffer = device
            .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)
            .map_err(vkerr("create_buffer"))?;
        let req = device.get_buffer_memory_requirements(buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = memory_type(memory_props, req.memory_type_bits, flags).ok_or("no host-visible memory type")?;
        let memory = device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
            .map_err(vkerr("allocate_memory"))?;
        device.bind_buffer_memory(buffer, memory, 0).map_err(vkerr("bind_buffer_memory"))?;
        let ptr = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()).map_err(vkerr("map_memory"))? as *mut u8;
        Ok(HostBuffer { buffer, memory, ptr, size })
    }
}

pub unsafe fn free_host_buffer(device: &ash::Device, b: &HostBuffer) {
    unsafe {
        device.unmap_memory(b.memory);
        device.destroy_buffer(b.buffer, None);
        device.free_memory(b.memory, None);
    }
}

pub struct Capture {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bgra: bool,
}

pub struct Display {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pd: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub family: u32,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub refresh_mhz: u32,
    pub init_ms: f64,
    pub pipeline_ms: f64,
    surface_i: khr::surface::Instance,
    swap_d: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    vs: vk::ShaderModule,
    fs: vk::ShaderModule,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    rendered: Vec<vk::Semaphore>,
    readback: HostBuffer,
    frame_image: Option<FrameImage>,
}

struct FrameImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    staging: HostBuffer,
    width: u32,
    height: u32,
}

pub const PUSH_FLOATS: usize = 8;

impl Display {
    pub fn new(log_device: bool) -> Result<Self, String> {
        let entry = entry();
        unsafe {
            let inst_exts: Vec<String> = entry
                .enumerate_instance_extension_properties(None)
                .map_err(vkerr("instance extensions"))?
                .iter()
                .map(|e| name_of(&e.extension_name))
                .collect();
            if log_device {
                let version = entry.try_enumerate_instance_version().map_err(vkerr("instance version"))?.unwrap_or(vk::API_VERSION_1_0);
                crate::klog!(
                    "vk instance {}.{}.{} extensions: {}",
                    vk::api_version_major(version),
                    vk::api_version_minor(version),
                    vk::api_version_patch(version),
                    inst_exts.join(" ")
                );
            }
            let mut enabled = vec![khr::surface::NAME.as_ptr(), khr::display::NAME.as_ptr()];
            if inst_exts.iter().any(|e| e == "VK_KHR_get_surface_capabilities2") {
                enabled.push(khr::get_surface_capabilities2::NAME.as_ptr());
            }
            let app = vk::ApplicationInfo::default().application_name(c"NeXium").engine_name(c"NeXium").api_version(vk::API_VERSION_1_3);
            let started = Instant::now();
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&enabled), None)
                .map_err(vkerr("create_instance"))?;
            let pd = *instance
                .enumerate_physical_devices()
                .map_err(vkerr("enumerate_physical_devices"))?
                .first()
                .ok_or("no physical device")?;
            if log_device {
                for line in crate::probe_vk::report_device(&instance, pd)? {
                    crate::klog!("vk {line}");
                }
            }
            let display_i = khr::display::Instance::new(&entry, &instance);
            let surface_i = khr::surface::Instance::new(&entry, &instance);
            let displays = display_i.get_physical_device_display_properties(pd).map_err(vkerr("display properties"))?;
            let display = displays.first().ok_or("no display")?;
            let modes = display_i.get_display_mode_properties(pd, display.display).map_err(vkerr("display modes"))?;
            let mode = *modes.iter().max_by_key(|m| m.parameters.refresh_rate).ok_or("no display mode")?;
            if log_device {
                let name = if display.display_name.is_null() {
                    "?".to_string()
                } else {
                    CStr::from_ptr(display.display_name).to_string_lossy().into_owned()
                };
                let list: Vec<String> = modes
                    .iter()
                    .map(|m| {
                        format!(
                            "{}x{}@{:.2}",
                            m.parameters.visible_region.width,
                            m.parameters.visible_region.height,
                            m.parameters.refresh_rate as f64 / 1000.0
                        )
                    })
                    .collect();
                crate::klog!("vk display '{name}' modes [{}]", list.join(", "));
            }
            let mut extent = mode.parameters.visible_region;
            let surface = display_i
                .create_display_plane_surface(
                    &vk::DisplaySurfaceCreateInfoKHR::default()
                        .display_mode(mode.display_mode)
                        .plane_index(0)
                        .plane_stack_index(0)
                        .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .global_alpha(1.0)
                        .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::OPAQUE)
                        .image_extent(extent),
                    None,
                )
                .map_err(vkerr("create_display_plane_surface"))?;
            let families = instance.get_physical_device_queue_family_properties(pd);
            let family = (0..families.len() as u32)
                .find(|&i| {
                    families[i as usize].queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                        && surface_i.get_physical_device_surface_support(pd, i, surface).unwrap_or(false)
                })
                .ok_or("no graphics+compute+present queue family")?;
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true);
            let features = vk::PhysicalDeviceFeatures::default().robust_buffer_access(true);
            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let device_exts = [khr::swapchain::NAME.as_ptr()];
            let device = instance
                .create_device(
                    pd,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_info)
                        .enabled_extension_names(&device_exts)
                        .enabled_features(&features)
                        .push_next(&mut f12)
                        .push_next(&mut f13),
                    None,
                )
                .map_err(vkerr("create_device"))?;
            let init_ms = started.elapsed().as_secs_f64() * 1000.0;
            let queue = device.get_device_queue(family, 0);
            let memory_props = instance.get_physical_device_memory_properties(pd);
            let swap_d = khr::swapchain::Device::new(&instance, &device);
            let caps = surface_i.get_physical_device_surface_capabilities(pd, surface).map_err(vkerr("surface caps"))?;
            let formats = surface_i.get_physical_device_surface_formats(pd, surface).map_err(vkerr("surface formats"))?;
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or_else(|| formats.first())
                .copied()
                .ok_or("no surface format")?;
            if log_device {
                let modes = surface_i.get_physical_device_surface_present_modes(pd, surface).map_err(vkerr("present modes"))?;
                crate::klog!(
                    "vk surface caps images {}..{} extent {}x{} usage {:?} formats {:?} present modes {:?}",
                    caps.min_image_count,
                    caps.max_image_count,
                    caps.current_extent.width,
                    caps.current_extent.height,
                    caps.supported_usage_flags,
                    formats.iter().map(|f| f.format).collect::<Vec<_>>(),
                    modes
                );
            }
            let image_count =
                if caps.max_image_count == 0 { caps.min_image_count.max(3) } else { caps.min_image_count.max(3).min(caps.max_image_count) };
            if caps.current_extent.width != u32::MAX {
                extent = caps.current_extent;
            }
            let swapchain = swap_d
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(image_count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(vk::PresentModeKHR::FIFO)
                        .clipped(true),
                    None,
                )
                .map_err(vkerr("create_swapchain"))?;
            let images = swap_d.get_swapchain_images(swapchain).map_err(vkerr("swapchain images"))?;
            let views = images
                .iter()
                .map(|&image| {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(color_range()),
                        None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(vkerr("image view"))?;
            let vs_code = spv(include_bytes!(concat!(env!("OUT_DIR"), "/probe_vs_main.spv")));
            let fs_code = spv(include_bytes!(concat!(env!("OUT_DIR"), "/probe_fs_main.spv")));
            let vs = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vs_code), None).map_err(vkerr("vs"))?;
            let fs = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&fs_code), None).map_err(vkerr("fs"))?;
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .offset(0)
                .size((PUSH_FLOATS * 4) as u32)];
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_range), None)
                .map_err(vkerr("pipeline layout"))?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"vs_main"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"fs_main"),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachment = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
            let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
            let color_formats = [format.format];
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
            let pipeline_started = Instant::now();
            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&vertex_input)
                        .input_assembly_state(&assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&raster)
                        .multisample_state(&multisample)
                        .color_blend_state(&blend)
                        .dynamic_state(&dynamic_state)
                        .layout(layout)
                        .push_next(&mut rendering)],
                    None,
                )
                .map_err(|(_, e)| format!("graphics pipeline: {e:?}"))?[0];
            let pipeline_ms = pipeline_started.elapsed().as_secs_f64() * 1000.0;
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(vkerr("command pool"))?;
            let cmd = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))
                .map_err(vkerr("command buffer"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("fence"))?;
            let acquired = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(vkerr("semaphore"))?;
            let rendered = images
                .iter()
                .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(vkerr("semaphore"))?;
            let readback = host_buffer(&device, &memory_props, extent.width as u64 * extent.height as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST)?;
            Ok(Self {
                entry,
                instance,
                pd,
                device,
                queue,
                family,
                memory_props,
                extent,
                format: format.format,
                refresh_mhz: mode.parameters.refresh_rate,
                init_ms,
                pipeline_ms,
                surface_i,
                swap_d,
                surface,
                swapchain,
                images,
                views,
                vs,
                fs,
                layout,
                pipeline,
                pool,
                cmd,
                fence,
                acquired,
                rendered,
                readback,
                frame_image: None,
            })
        }
    }

    pub fn frame(&mut self, push: [f32; PUSH_FLOATS], capture: bool) -> Result<Option<Capture>, String> {
        let device = &self.device;
        let extent = self.extent;
        unsafe {
            let (index, _) = self
                .swap_d
                .acquire_next_image(self.swapchain, u64::MAX, self.acquired, vk::Fence::null())
                .map_err(vkerr("acquire"))?;
            let image = self.images[index as usize];
            let cmd = self.cmd;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).map_err(vkerr("reset cmd"))?;
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(vkerr("begin cmd"))?;
            barrier(
                device,
                cmd,
                image,
                (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
                (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );
            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.views[index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })];
            device.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent })
                    .layer_count(1)
                    .color_attachments(&attachment),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 }],
            );
            device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D::default(), extent }]);
            let push_bytes: Vec<u8> = push.iter().flat_map(|v| v.to_le_bytes()).collect();
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &push_bytes);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            let mut last = (vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            if capture {
                let src = (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                barrier(device, cmd, image, last, src);
                device.cmd_copy_image_to_buffer(
                    cmd,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback.buffer,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D { width: extent.width, height: extent.height, depth: 1 },
                    }],
                );
                last = src;
            }
            barrier(device, cmd, image, last, (vk::PipelineStageFlags2::BOTTOM_OF_PIPE, vk::AccessFlags2::NONE, vk::ImageLayout::PRESENT_SRC_KHR));
            device.end_command_buffer(cmd).map_err(vkerr("end cmd"))?;
            let wait = [vk::SemaphoreSubmitInfo::default().semaphore(self.acquired).stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [vk::SemaphoreSubmitInfo::default().semaphore(self.rendered[index as usize]).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default().wait_semaphore_infos(&wait).command_buffer_infos(&cmds).signal_semaphore_infos(&signal)],
                    self.fence,
                )
                .map_err(vkerr("queue_submit2"))?;
            let swapchains = [self.swapchain];
            let indices = [index];
            let wait_present = [self.rendered[index as usize]];
            self.swap_d
                .queue_present(self.queue, &vk::PresentInfoKHR::default().wait_semaphores(&wait_present).swapchains(&swapchains).image_indices(&indices))
                .map_err(vkerr("queue_present"))?;
            device.wait_for_fences(&[self.fence], true, u64::MAX).map_err(vkerr("wait fence"))?;
            device.reset_fences(&[self.fence]).map_err(vkerr("reset fence"))?;
            if capture {
                let bytes = std::slice::from_raw_parts(self.readback.ptr, self.readback.size as usize).to_vec();
                let bgra = matches!(self.format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB);
                return Ok(Some(Capture { bytes, width: extent.width, height: extent.height, bgra }));
            }
        }
        Ok(None)
    }
}

impl Display {
    fn ensure_frame_image(&mut self, width: u32, height: u32) -> Result<(), String> {
        if let Some(f) = &self.frame_image {
            if f.width == width && f.height == height {
                return Ok(());
            }
        }
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            if let Some(f) = self.frame_image.take() {
                free_host_buffer(d, &f.staging);
                d.destroy_image(f.image, None);
                d.free_memory(f.memory, None);
            }
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .extent(vk::Extent3D { width, height, depth: 1 })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(vkerr("frame image"))?;
            let req = d.get_image_memory_requirements(image);
            let index = memory_type(&self.memory_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or("no device-local memory for frame image")?;
            let memory = d
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index), None)
                .map_err(vkerr("frame image memory"))?;
            d.bind_image_memory(image, memory, 0).map_err(vkerr("bind frame image"))?;
            let staging = host_buffer(d, &self.memory_props, width as u64 * height as u64 * 4, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.frame_image = Some(FrameImage { image, memory, staging, width, height });
        }
        Ok(())
    }

    pub fn present_rgba(&mut self, width: u32, height: u32, pixels: Option<&[u8]>) -> Result<(), String> {
        if width == 0 || height == 0 {
            return self.frame([0.0; PUSH_FLOATS], false).map(|_| ());
        }
        self.ensure_frame_image(width, height)?;
        let extent = self.extent;
        unsafe {
            let f = self.frame_image.as_ref().unwrap();
            let upload = match pixels {
                Some(p) if p.len() >= (width * height * 4) as usize => {
                    std::ptr::copy_nonoverlapping(p.as_ptr(), f.staging.ptr, (width * height * 4) as usize);
                    true
                }
                _ => false,
            };
            let device = &self.device;
            let (index, _) = self
                .swap_d
                .acquire_next_image(self.swapchain, u64::MAX, self.acquired, vk::Fence::null())
                .map_err(vkerr("acquire"))?;
            let target = self.images[index as usize];
            let cmd = self.cmd;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).map_err(vkerr("reset cmd"))?;
            device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(vkerr("begin cmd"))?;
            if upload {
                barrier(
                    device,
                    cmd,
                    f.image,
                    (vk::PipelineStageFlags2::ALL_TRANSFER, vk::AccessFlags2::TRANSFER_READ, vk::ImageLayout::UNDEFINED),
                    (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                );
                device.cmd_copy_buffer_to_image(
                    cmd,
                    f.staging.buffer,
                    f.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D { width, height, depth: 1 },
                    }],
                );
                barrier(
                    device,
                    cmd,
                    f.image,
                    (vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                    (vk::PipelineStageFlags2::BLIT, vk::AccessFlags2::TRANSFER_READ, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
                );
            }
            barrier(
                device,
                cmd,
                target,
                (vk::PipelineStageFlags2::ALL_TRANSFER, vk::AccessFlags2::NONE, vk::ImageLayout::UNDEFINED),
                (vk::PipelineStageFlags2::CLEAR, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            );
            device.cmd_clear_color_image(
                cmd,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                &[color_range()],
            );
            barrier(
                device,
                cmd,
                target,
                (vk::PipelineStageFlags2::CLEAR, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                (vk::PipelineStageFlags2::BLIT, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
            );
            let scale = (extent.width as f32 / width as f32).min(extent.height as f32 / height as f32);
            let dw = (width as f32 * scale) as i32;
            let dh = (height as f32 * scale) as i32;
            let dx = (extent.width as i32 - dw) / 2;
            let dy = (extent.height as i32 - dh) / 2;
            let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
            device.cmd_blit_image(
                cmd,
                f.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit {
                    src_subresource: layers,
                    src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: width as i32, y: height as i32, z: 1 }],
                    dst_subresource: layers,
                    dst_offsets: [vk::Offset3D { x: dx, y: dy, z: 0 }, vk::Offset3D { x: dx + dw, y: dy + dh, z: 1 }],
                }],
                vk::Filter::LINEAR,
            );
            barrier(
                device,
                cmd,
                target,
                (vk::PipelineStageFlags2::BLIT, vk::AccessFlags2::TRANSFER_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
                (vk::PipelineStageFlags2::BOTTOM_OF_PIPE, vk::AccessFlags2::NONE, vk::ImageLayout::PRESENT_SRC_KHR),
            );
            device.end_command_buffer(cmd).map_err(vkerr("end cmd"))?;
            let wait = [vk::SemaphoreSubmitInfo::default().semaphore(self.acquired).stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)];
            let signal = [vk::SemaphoreSubmitInfo::default().semaphore(self.rendered[index as usize]).stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default().wait_semaphore_infos(&wait).command_buffer_infos(&cmds).signal_semaphore_infos(&signal)],
                    self.fence,
                )
                .map_err(vkerr("queue_submit2"))?;
            let swapchains = [self.swapchain];
            let indices = [index];
            let wait_present = [self.rendered[index as usize]];
            self.swap_d
                .queue_present(self.queue, &vk::PresentInfoKHR::default().wait_semaphores(&wait_present).swapchains(&swapchains).image_indices(&indices))
                .map_err(vkerr("queue_present"))?;
            device.wait_for_fences(&[self.fence], true, u64::MAX).map_err(vkerr("wait fence"))?;
            device.reset_fences(&[self.fence]).map_err(vkerr("reset fence"))?;
        }
        Ok(())
    }

    pub fn capture_last(&mut self) -> Option<Capture> {
        let f = self.frame_image.as_ref()?;
        let bytes = unsafe { std::slice::from_raw_parts(f.staging.ptr, (f.width * f.height * 4) as usize).to_vec() };
        Some(Capture { bytes, width: f.width, height: f.height, bgra: false })
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
}

unsafe fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    from: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
    to: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(from.0)
        .src_access_mask(from.1)
        .old_layout(from.2)
        .dst_stage_mask(to.0)
        .dst_access_mask(to.1)
        .new_layout(to.2)
        .image(image)
        .subresource_range(color_range())];
    unsafe { device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}

impl Drop for Display {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;
            let _ = d.device_wait_idle();
            free_host_buffer(d, &self.readback);
            for &s in &self.rendered {
                d.destroy_semaphore(s, None);
            }
            if let Some(f) = self.frame_image.take() {
                free_host_buffer(d, &f.staging);
                d.destroy_image(f.image, None);
                d.free_memory(f.memory, None);
            }
            d.destroy_semaphore(self.acquired, None);
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_shader_module(self.vs, None);
            d.destroy_shader_module(self.fs, None);
            for &v in &self.views {
                d.destroy_image_view(v, None);
            }
            self.swap_d.destroy_swapchain(self.swapchain, None);
            d.destroy_device(None);
            self.surface_i.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn write_bmp(path: &str, cap: &Capture, step: u32) -> std::io::Result<()> {
    use std::io::Write;
    let w = cap.width / step;
    let h = cap.height / step;
    let row = (w * 3).div_ceil(4) * 4;
    let mut out = Vec::with_capacity(54 + (row * h) as usize);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54 + row * h).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(w as i32).to_le_bytes());
    out.extend_from_slice(&(h as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 24]);
    let pitch = cap.width as usize * 4;
    for y in (0..h).rev() {
        let line = &cap.bytes[(y * step) as usize * pitch..];
        let start = out.len();
        for x in 0..w {
            let p = &line[(x * step) as usize * 4..(x * step) as usize * 4 + 4];
            let (r, g, b) = if cap.bgra { (p[2], p[1], p[0]) } else { (p[0], p[1], p[2]) };
            out.extend_from_slice(&[b, g, r]);
        }
        out.resize(start + row as usize, 0);
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&out)?;
    crate::console::share(path, 0o666);
    Ok(())
}

use ash::vk;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use crate::descriptor::{DescriptorPool, DescriptorSetLayout};
use crate::pipeline::PipelineCache;
use crate::rt_cache::{find_memory_type, RtCache, RtKey};
use crate::shader::ShaderCompiler;

pub struct Renderer {
    inner: Mutex<RendererInner>,
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
    dummy_white: Option<DummyImage>,
    default_sampler: Option<vk::Sampler>,
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

impl Renderer {
    pub fn new() -> Result<Arc<Self>, String> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("Vulkan entry load failed: {:?}", e))?;

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
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let instance = unsafe {
            entry.create_instance(&inst_info, None)
                .map_err(|e| format!("create_instance: {:?}", e))?
        };

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
        let queue_family = qf
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| "no graphics queue family".to_string())? as u32;

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
        let mut features_13 = vk::PhysicalDeviceVulkan13Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            p_next: std::ptr::null_mut(),
            ..Default::default()
        };
        let dev_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            p_enabled_features: std::ptr::null(),
            p_next: &mut features_13 as *mut _ as *mut std::ffi::c_void,
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let device = unsafe {
            instance.create_device(physical_device, &dev_info, None)
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
            device.create_command_pool(&cmd_pool_info, None)
                .map_err(|e| format!("create_command_pool: {:?}", e))?
        };

        let mut rt_cache = RtCache::new();
        rt_cache.set_mem_properties(mem_props);

        let descriptor_layout = DescriptorSetLayout::new(&device)?;
        let descriptor_pool = DescriptorPool::new(&device, 256)?;
        let shader_compiler = ShaderCompiler::new();
        let pipeline_cache = PipelineCache::new(&device, descriptor_layout.layout)?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        log::info!("nexium-gpu Renderer init OK: {} (Vulkan via Ash)", name);

        Ok(Arc::new(Self {
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
                dummy_white: None,
                default_sampler: None,
            }),
        }))
    }

    pub fn clear_target(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        rgba: [f32; 4],
    ) -> Result<(), String> {
        let mut inner = self.inner.lock();
        let RendererInner { device, cmd_pool, queue, rt_cache, .. } = &mut *inner;
        let key = RtKey { nvmap_id, width, height };
        let img = rt_cache.get_or_create(key, device)?;

        let cmd = alloc_one_time_cmd(device, *cmd_pool)?;
        begin_one_time(device, cmd)?;
        transition_image(
            device, cmd, img.image, img.layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let clear = vk::ClearColorValue { float32: rgba };
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
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
        }
        img.layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        end_one_time(device, cmd)?;
        submit_and_wait(device, *queue, cmd)?;
        unsafe { device.free_command_buffers(*cmd_pool, &[cmd]) };
        Ok(())
    }

    pub fn readback_target(&self, nvmap_id: u32, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device, cmd_pool, queue, rt_cache, mem_props, staging, ..
        } = &mut *inner;
        let key = RtKey { nvmap_id, width, height };
        if !cache_contains(rt_cache, key) {
            return None;
        }
        let img = rt_cache.get_or_create(key, device).ok()?;

        let row_bytes = (width as u64) * 4;
        let total = row_bytes * (height as u64);
        let stage = ensure_staging(staging, device, mem_props, (width, height), total).ok()?;

        let cmd = alloc_one_time_cmd(device, *cmd_pool).ok()?;
        begin_one_time(device, cmd).ok()?;
        transition_image(
            device, cmd, img.image, img.layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
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
            image_extent: vk::Extent3D { width, height, depth: 1 },
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
        end_one_time(device, cmd).ok()?;
        submit_and_wait(device, *queue, cmd).ok()?;
        unsafe { device.free_command_buffers(*cmd_pool, &[cmd]) };

        let mut out = vec![0u8; total as usize];
        unsafe {
            let ptr = device
                .map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
                .ok()? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), total as usize);
            device.unmap_memory(stage.memory);
        }
        Some(out)
    }

    pub fn compile_pipeline(
        &self,
        vs_spirv: &[u32],
        fs_spirv: &[u32],
        vs_cbuf_mask: u32,
        fs_cbuf_mask: u32,
        layout: &crate::draw::VertexLayout,
        topology: vk::PrimitiveTopology,
        color_format: vk::Format,
    ) -> Result<vk::Pipeline, String> {
        let mut inner = self.inner.lock();
        let key = crate::pipeline::PipelineKey {
            vs_hash: hash_spirv(vs_spirv),
            fs_hash: hash_spirv(fs_spirv),
            topology: topology.as_raw() as u32,
            color_format: color_format.as_raw() as u32,
            vs_cbuf_mask,
            fs_cbuf_mask,
            vertex_layout_hash: layout.hash(),
        };
        if let Some(p) = inner.pipeline_cache.get(&key) {
            return Ok(p);
        }
        let RendererInner { device, shader_compiler, pipeline_cache, .. } = &mut *inner;

        let vs_mod = shader_compiler.compile_or_get(vs_spirv, device)?;
        let fs_mod = shader_compiler.compile_or_get(fs_spirv, device)?;

        let entry = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: vs_mod,
                p_name: entry.as_ptr(),
                p_specialization_info: std::ptr::null(),
                p_next: std::ptr::null(),
                flags: Default::default(),
                _marker: std::marker::PhantomData,
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fs_mod,
                p_name: entry.as_ptr(),
                p_specialization_info: std::ptr::null(),
                p_next: std::ptr::null(),
                flags: Default::default(),
                _marker: std::marker::PhantomData,
            },
        ];

        let vk_bindings: Vec<vk::VertexInputBindingDescription> = layout.bindings.iter().map(|b| {
            vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                input_rate: vk::VertexInputRate::VERTEX,
            }
        }).collect();
        let vk_attrs: Vec<vk::VertexInputAttributeDescription> = layout.attrs.iter().map(|a| {
            vk::VertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: a.format,
                offset: a.offset,
            }
        }).collect();

        let vi_state = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: vk_bindings.len() as u32,
            p_vertex_binding_descriptions: vk_bindings.as_ptr(),
            vertex_attribute_description_count: vk_attrs.len() as u32,
            p_vertex_attribute_descriptions: vk_attrs.as_ptr(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let ia_state = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology,
            primitive_restart_enable: vk::FALSE,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let vp_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            p_viewports: std::ptr::null(),
            scissor_count: 1,
            p_scissors: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let rs_state = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let ms_state = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let cb_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };
        let cb_state = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &cb_attachment,
            blend_constants: [0.0; 4],
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dyn_state = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dyn_states.len() as u32,
            p_dynamic_states: dyn_states.as_ptr(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let color_formats = [color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo {
            s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachment_formats: color_formats.as_ptr(),
            depth_attachment_format: vk::Format::UNDEFINED,
            stencil_attachment_format: vk::Format::UNDEFINED,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vi_state,
            p_input_assembly_state: &ia_state,
            p_tessellation_state: std::ptr::null(),
            p_viewport_state: &vp_state,
            p_rasterization_state: &rs_state,
            p_multisample_state: &ms_state,
            p_depth_stencil_state: std::ptr::null(),
            p_color_blend_state: &cb_state,
            p_dynamic_state: &dyn_state,
            layout: pipeline_cache.layout,
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            p_next: &mut rendering_info as *mut _ as *mut std::ffi::c_void,
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| format!("create_graphics_pipelines: {:?}", e))?
        };
        let pipeline = pipelines[0];
        pipeline_cache.insert(key, pipeline);
        Ok(pipeline)
    }

    pub fn execute_draw<F>(
        &self,
        call: &crate::draw::Maxwell3dDrawCall,
        read_guest: F,
    ) -> Result<(), String>
    where
        F: Fn(u64, usize) -> Option<Vec<u8>>,
    {
        let pipeline = self.compile_pipeline(
            &call.vs_spirv,
            &call.fs_spirv,
            call.vs_cbuf_mask,
            call.fs_cbuf_mask,
            &call.vertex_layout,
            call.state.topology,
            call.rt_format,
        )?;

        let vertex_stride = call
            .vertex_layout
            .bindings
            .first()
            .map(|b| b.stride as u64)
            .unwrap_or(0);
        let vertex_bytes = vertex_stride.saturating_mul(call.vertex_count as u64) as usize;
        let vertex_data = if vertex_bytes > 0 {
            read_guest(call.vertex_addr, vertex_bytes)
                .ok_or_else(|| format!("vertex read failed va={:#x}", call.vertex_addr))?
        } else {
            Vec::new()
        };

        let cbuf_size = call.cbuf_size as usize;
        let cbuf_data = if cbuf_size > 0 && call.cbuf_addr != 0 {
            read_guest(call.cbuf_addr, cbuf_size).unwrap_or_else(|| vec![0u8; cbuf_size])
        } else {
            vec![0u8; 256]
        };

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
            dummy_white,
            default_sampler,
            ..
        } = &mut *inner;

        if dummy_white.is_none() {
            *dummy_white = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props,
            )?);
        }
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let dummy = dummy_white.as_ref().unwrap();
        let samp = default_sampler.unwrap();

        let vertex_buf = if !vertex_data.is_empty() {
            Some(create_host_buffer(
                device,
                mem_props,
                &vertex_data,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?)
        } else {
            None
        };
        let ubo_buf = create_host_buffer(
            device,
            mem_props,
            &cbuf_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

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
            buffer: ubo_buf.buffer,
            offset: 0,
            range: cbuf_data.len() as u64,
        };
        let img_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: dummy.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let samp_info = vk::DescriptorImageInfo {
            sampler: samp,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        };
        let writes = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_buffer_info: &ubo_info,
                p_image_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: &img_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: 2,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLER,
                p_image_info: &samp_info,
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            },
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let rt = rt_cache.get_or_create(call.rt_key, device)?;
        let rt_image = rt.image;
        let rt_view = rt.view;
        let rt_extent = rt.extent;
        let rt_prev_layout = rt.layout;

        let cmd = alloc_one_time_cmd(device, *cmd_pool)?;
        begin_one_time(device, cmd)?;
        transition_image(
            device,
            cmd,
            rt_image,
            rt_prev_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: call.clear_color },
        };
        let attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: rt_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            load_op: if call.clear { vk::AttachmentLoadOp::CLEAR } else { vk::AttachmentLoadOp::LOAD },
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: rt_extent,
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
        unsafe { device.cmd_begin_rendering(cmd, &render_info) };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: rt_extent.width as f32,
            height: rt_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: rt_extent,
        };
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_cache.layout,
                0,
                &[dset],
                &[],
            );
            if let Some(vb) = &vertex_buf {
                device.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);
            }
            device.cmd_draw(cmd, call.vertex_count, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }

        transition_image(
            device,
            cmd,
            rt_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );

        end_one_time(device, cmd)?;
        submit_and_wait(device, *queue, cmd)?;

        unsafe {
            device.free_command_buffers(*cmd_pool, &[cmd]);
            let _ = device.free_descriptor_sets(descriptor_pool.pool, &[dset]);
            if let Some(vb) = vertex_buf {
                device.destroy_buffer(vb.buffer, None);
                device.free_memory(vb.memory, None);
            }
            device.destroy_buffer(ubo_buf.buffer, None);
            device.free_memory(ubo_buf.memory, None);
        }
        rt_cache.get_or_create(call.rt_key, device)?.layout =
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        Ok(())
    }
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

fn create_default_sampler(device: &ash::Device) -> Result<vk::Sampler, String> {
    let info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
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

fn create_dummy_white_image(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<DummyImage, String> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let img_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D { width: 1, height: 1, depth: 1 },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        p_next: std::ptr::null(),
        flags: Default::default(),
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

    let pixel: [u8; 4] = [255, 255, 255, 255];
    let stage = create_host_buffer(device, mem_props, &pixel, vk::BufferUsageFlags::TRANSFER_SRC)?;

    let cmd = alloc_one_time_cmd(device, cmd_pool)?;
    begin_one_time(device, cmd)?;
    transition_image(device, cmd, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
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
        image_extent: vk::Extent3D { width: 1, height: 1, depth: 1 },
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
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
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
    Ok(DummyImage { image, view, memory })
}

fn hash_spirv(spirv: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in spirv {
        h ^= *w as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn cache_contains(rt: &RtCache, key: RtKey) -> bool {
    let _ = key;
    let _ = rt;
    true
}

fn alloc_one_time_cmd(device: &ash::Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer, String> {
    let info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let v = unsafe {
        device.allocate_command_buffers(&info)
            .map_err(|e| format!("allocate_command_buffers: {:?}", e))?
    };
    Ok(v[0])
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
        device.begin_command_buffer(cmd, &begin)
            .map_err(|e| format!("begin_command_buffer: {:?}", e))
    }
}

fn end_one_time(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<(), String> {
    unsafe {
        device.end_command_buffer(cmd)
            .map_err(|e| format!("end_command_buffer: {:?}", e))
    }
}

fn submit_and_wait(device: &ash::Device, queue: vk::Queue, cmd: vk::CommandBuffer) -> Result<(), String> {
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
        device.queue_submit(queue, &[submit], vk::Fence::null())
            .map_err(|e| format!("queue_submit: {:?}", e))?;
        device.queue_wait_idle(queue)
            .map_err(|e| format!("queue_wait_idle: {:?}", e))?;
    }
    Ok(())
}

fn transition_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
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
            device.create_buffer(&buf_info, None)
                .map_err(|e| format!("create_buffer: {:?}", e))?
        };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mt = find_memory_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).ok_or_else(|| "no HOST_VISIBLE memory type".to_string())?;
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: req.size,
            memory_type_index: mt,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let memory = unsafe {
            device.allocate_memory(&alloc_info, None)
                .map_err(|e| format!("allocate_memory(staging): {:?}", e))?
        };
        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| format!("bind_buffer_memory: {:?}", e))?;
        }
        staging.insert(key, StagingBuffer { buffer, memory, size: req.size });
    }
    Ok(staging.get(&key).unwrap())
}

impl Drop for RendererInner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        if let Some(d) = self.dummy_white.take() {
            unsafe {
                self.device.destroy_image_view(d.view, None);
                self.device.destroy_image(d.image, None);
                self.device.free_memory(d.memory, None);
            }
        }
        if let Some(s) = self.default_sampler.take() {
            unsafe { self.device.destroy_sampler(s, None) };
        }
        self.pipeline_cache.clear(&self.device);
        self.shader_compiler.clear(&self.device);
        unsafe {
            self.device.destroy_descriptor_pool(self.descriptor_pool.pool, None);
            self.descriptor_pool.pool = vk::DescriptorPool::null();
            self.device.destroy_descriptor_set_layout(self.descriptor_layout.layout, None);
            self.descriptor_layout.layout = vk::DescriptorSetLayout::null();
        }
        for (_, s) in self.staging.drain() {
            unsafe {
                self.device.destroy_buffer(s.buffer, None);
                self.device.free_memory(s.memory, None);
            }
        }
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

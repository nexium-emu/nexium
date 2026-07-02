use ash::vk;
use parking_lot::Mutex;
use std::collections::{hash_map::Entry, HashMap, VecDeque};
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
    dummy_white_array: Option<DummyImage>,
    dummy_white_3d: Option<DummyImage>,
    default_sampler: Option<vk::Sampler>,
    sampler_cache: HashMap<crate::texture::TscEntry, vk::Sampler>,
    tex_cache: HashMap<TexCacheKey, CachedTexture>,
    frame_slots: [FrameSlot; 2],
    frame_index: usize,
    utility_slot: FrameSlot,
    ubo_ring: UboRing,
    min_ubo_offset_alignment: u64,
    pending_readbacks: HashMap<RtKey, VecDeque<PendingReadback>>,
    readback_slots: Vec<ReadbackSlot>,
    tele_last_emit_ns: u64,
    tele_ring_wraps: u64,
    tele_ring_waits: u64,
    tele_in_flight_mask: u32,
    depth_clamp_supported: bool,
    depth_clip_control_enabled: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TexCacheKey {
    gpu_va: u64,
    width: u32,
    height: u32,
    layers: u32,
    base_layer: u32,
    view_layers: u32,
    arrayed: bool,
    volume: bool,
    format: crate::texture::TicFormat,
    swizzle: [crate::texture::SwizzleSource; 4],
}

type PendingTexture = (TexCacheKey, crate::texture::TicEntry, usize, usize);

struct CachedTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    hash: u64,
    gen: u64,
}

#[derive(Clone, Copy)]
struct VolumeRtSlice {
    layer: u32,
    key: RtKey,
    image: vk::Image,
    layout: vk::ImageLayout,
    format: vk::Format,
    stamp: u64,
}

#[derive(Clone, Copy)]
struct RtAlias {
    key: RtKey,
    image: vk::Image,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    depth: bool,
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

struct FrameSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    in_flight: bool,
    retired_dsets: Vec<vk::DescriptorSet>,
    retired_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    retired_textures: Vec<CachedTexture>,
}

struct PendingReadback {
    slot: usize,
    width: u32,
    height: u32,
    format: vk::Format,
}

struct ReadbackSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    stage: Option<StagingBuffer>,
    in_flight: bool,
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

        let dcc_ext_supported = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map(|exts| {
                    exts.iter().any(|e| {
                        let name = std::ffi::CStr::from_ptr(e.extension_name.as_ptr());
                        name == vk::EXT_DEPTH_CLIP_CONTROL_NAME
                    })
                })
                .unwrap_or(false)
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

        let mut features_13 = vk::PhysicalDeviceVulkan13Features {
            s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            dynamic_rendering: vk::TRUE,
            synchronization2: vk::TRUE,
            p_next: std::ptr::null_mut(),
            ..Default::default()
        };
        let mut dcc_feature = vk::PhysicalDeviceDepthClipControlFeaturesEXT {
            s_type: vk::StructureType::PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
            depth_clip_control: vk::TRUE,
            p_next: std::ptr::null_mut(),
            _marker: std::marker::PhantomData,
        };
        if enable_depth_clip_control {
            dcc_feature.p_next = &mut features_13 as *mut _ as *mut std::ffi::c_void;
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
        let p_next_chain: *mut std::ffi::c_void = if enable_depth_clip_control {
            &mut dcc_feature as *mut _ as *mut std::ffi::c_void
        } else {
            &mut features_13 as *mut _ as *mut std::ffi::c_void
        };
        let core_features = unsafe { instance.get_physical_device_features(physical_device) };
        let depth_clamp_supported = core_features.depth_clamp == vk::TRUE;
        if !depth_clamp_supported {
            log::info!("Vulkan depthClamp feature unavailable; Maxwell depth clamp disabled");
        }
        let enabled_core_features = vk::PhysicalDeviceFeatures {
            robust_buffer_access: vk::TRUE,
            depth_clamp: if depth_clamp_supported {
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
        let descriptor_pool = DescriptorPool::new(&device, 1024)?;
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
        let min_ubo_offset_alignment = props.limits.min_uniform_buffer_offset_alignment.max(1);

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
                retired_buffers: Vec::new(),
                retired_textures: Vec::new(),
            },
            FrameSlot {
                fence: fence_b,
                cmd: frame_cmds[1],
                in_flight: false,
                retired_dsets: Vec::new(),
                retired_buffers: Vec::new(),
                retired_textures: Vec::new(),
            },
        ];
        let utility_slot = FrameSlot {
            fence: fence_util,
            cmd: frame_cmds[2],
            in_flight: false,
            retired_dsets: Vec::new(),
            retired_buffers: Vec::new(),
            retired_textures: Vec::new(),
        };

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

        let ubo_ring = create_ubo_ring(&device, &mem_props, 16 * 1024 * 1024)?;

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
                dummy_white: None,
                dummy_white_array: None,
                dummy_white_3d: None,
                default_sampler: None,
                sampler_cache: HashMap::new(),
                tex_cache: HashMap::new(),
                frame_slots,
                frame_index: 0,
                utility_slot,
                ubo_ring,
                min_ubo_offset_alignment,
                pending_readbacks: HashMap::new(),
                readback_slots,
                tele_last_emit_ns: 0,
                tele_ring_wraps: 0,
                tele_ring_waits: 0,
                tele_in_flight_mask: 0,
                depth_clamp_supported,
                depth_clip_control_enabled: enable_depth_clip_control,
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
            ..
        } = &mut *inner;
        let mut queued = 0usize;
        for spec in &specs {
            if pipeline_cache.get(&spec.key).is_some() {
                continue;
            }
            let vs_mod = match shader_compiler.compile_or_get(&spec.vs_spirv, device) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let fs_mod = match shader_compiler.compile_or_get(&spec.fs_spirv, device) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let req = crate::pipeline::spec_to_request(spec, vs_mod, fs_mod);
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
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create(key, device)?;

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
                .map_err(|e| format!("begin_command_buffer(utility): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(utility): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, utility_slot.fence)?;
        wait_fence(device, utility_slot.fence)?;
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
            return self.clear_target(nvmap_id, width, height, gpu_va, rgba);
        }
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create(key, device)?;

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
        submit_with_fence(device, *queue, cmd, utility_slot.fence)?;
        wait_fence(device, utility_slot.fence)?;
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

    pub fn clear_depth(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        depth: f32,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create_depth(key, device)?;

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
                .map_err(|e| format!("begin_command_buffer(depth clear): {:?}", e))?;
        }
        transition_image_aspect(
            device,
            cmd,
            img.image,
            img.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
        );
        let clear = vk::ClearDepthStencilValue { depth, stencil: 0 };
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
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
        submit_with_fence(device, *queue, cmd, utility_slot.fence)?;
        wait_fence(device, utility_slot.fence)?;
        Ok(())
    }

    pub fn rt_key_for_nvmap(&self, nvmap_id: u32, width: u32, height: u32) -> Option<(u32, u32)> {
        let inner = self.inner.lock();
        inner
            .rt_cache
            .find_color(RtKey::request(nvmap_id, width, height))
            .map(|(k, _, _, _)| (k.width, k.height))
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
        let key = rt_cache.resolve_present_key(requested_key)?;
        trace_present_key(rt_cache, requested_key, key);
        for (_, mut pending) in pending_readbacks.drain() {
            while let Some(prev) = pending.pop_front() {
                if let Some(slot) = readback_slots.get_mut(prev.slot) {
                    unsafe {
                        let _ = device.wait_for_fences(&[slot.fence], true, u64::MAX);
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
        unsafe {
            let _ = device.wait_for_fences(&[fence], true, u64::MAX);
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

    pub fn readback_target_pipelined(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        copy_rect: Option<[u32; 4]>,
    ) -> Option<(u32, u32, Vec<u8>)> {
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
        let requested_key = RtKey::request(nvmap_id, width, height);
        let key = rt_cache.resolve_present_key(requested_key)?;
        trace_present_key(rt_cache, requested_key, key);
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
            let out = readback_to_rgba8(&raw, prev.format, prev.width, prev.height);
            ready_frame = Some((prev.width, prev.height, out));
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
        _vertex_count: u32,
    ) -> Result<Option<vk::Pipeline>, String> {
        let mut inner = self.inner.lock();
        let blend_signature: u64 = (blend.enabled as u64)
            | ((blend.src_factor.as_raw() as u64 & 0xFF) << 8)
            | ((blend.dst_factor.as_raw() as u64 & 0xFF) << 16)
            | ((blend.op.as_raw() as u64 & 0xFF) << 24)
            | ((blend.src_alpha_factor.as_raw() as u64 & 0xFF) << 32)
            | ((blend.dst_alpha_factor.as_raw() as u64 & 0xFF) << 40)
            | ((blend.alpha_op.as_raw() as u64 & 0xFF) << 48)
            | ((blend.color_write_mask.as_raw() as u64 & 0xF) << 56);
        let raster_state_packed: u32 =
            (cull_test_enable as u32) | ((cull_face & 0xFF) << 8) | ((front_face & 0xFF) << 16);
        let has_depth = depth_format != vk::Format::UNDEFINED;
        let depth_state_packed: u32 = (depth.test_enabled as u32)
            | ((depth.write_enabled as u32) << 1)
            | ((has_depth as u32) << 2)
            | ((depth.compare_op.as_raw() as u32 & 0xFF) << 8);
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
            depth_clamp_enabled,
            poly_offset_packed,
            color_write_mask: blend.color_write_mask.as_raw(),
        };
        let depth_clip_control_enabled = inner.depth_clip_control_enabled;
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

        let vs_mod = shader_compiler.compile_or_get(vs_spirv, device)?;
        let fs_mod = shader_compiler.compile_or_get(fs_spirv, device)?;

        let bindings: Vec<vk::VertexInputBindingDescription> = layout
            .bindings
            .iter()
            .map(|b| vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                input_rate: vk::VertexInputRate::VERTEX,
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
            attrs,
            topology,
            color_formats: color_formats.clone(),
            depth_format,
            has_depth,
            blend,
            depth,
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
                .map(|b| (b.binding, b.stride))
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
            blend: (
                blend.enabled,
                blend.src_factor.as_raw(),
                blend.dst_factor.as_raw(),
                blend.op.as_raw(),
                blend.src_alpha_factor.as_raw(),
                blend.dst_alpha_factor.as_raw(),
                blend.alpha_op.as_raw(),
            ),
            color_write_mask: blend.color_write_mask.as_raw(),
            depth: (
                depth.test_enabled,
                depth.write_enabled,
                depth.compare_op.as_raw(),
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

        let pipeline = pipeline_cache.build(device, &req)?;
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
        let use_depth = call.depth_key.is_some();
        let depth_format = if use_depth {
            vk::Format::D32_SFLOAT
        } else {
            vk::Format::UNDEFINED
        };
        let color_formats = color_formats_for_call(call, call.color_rt_keys.len().max(1));
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
            call.vertex_count,
        )? {
            Some(p) => p,
            None => return Ok(()),
        };

        let vertex_stride = call
            .vertex_layout
            .bindings
            .iter()
            .find(|b| b.stride > 0)
            .map(|b| b.stride as u64)
            .unwrap_or(0);
        let vertex_base_addr = call
            .vertex_addr
            .wrapping_add(vertex_stride.saturating_mul(call.first_vertex as u64));
        let vertex_bytes = vertex_stride.saturating_mul(call.vertex_count as u64) as usize;
        let mut vertex_data = if vertex_bytes > 0 {
            read_guest(vertex_base_addr, vertex_bytes)
                .ok_or_else(|| format!("vertex read failed va={:#x}", vertex_base_addr))?
        } else {
            Vec::new()
        };
        let draw_vertex_count = if call.quad_expand && vertex_stride > 0 && !vertex_data.is_empty()
        {
            vertex_data = crate::draw::expand_quad_vertices(&vertex_data, vertex_stride as usize);
            (vertex_data.len() / vertex_stride as usize) as u32
        } else {
            call.vertex_count
        };

        let cbuf_size = call.cbuf_size as usize;
        let cbuf_data = if let Some(d) = &call.cbuf_data {
            d.clone()
        } else if cbuf_size > 0 && call.cbuf_addr != 0 {
            read_guest(call.cbuf_addr, cbuf_size).unwrap_or_else(|| vec![0u8; cbuf_size])
        } else {
            vec![0u8; 256]
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

        let tex_pendings = collect_tex_pendings(call, &read_guest);

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
            dummy_white_array,
            dummy_white_3d,
            default_sampler,
            sampler_cache,
            tex_cache,
            frame_slots,
            frame_index,
            ubo_ring,
            min_ubo_offset_alignment,
            tele_last_emit_ns,
            tele_ring_wraps,
            tele_ring_waits,
            tele_in_flight_mask,
            ..
        } = &mut *inner;

        let cur_idx = *frame_index;
        let other_idx = (cur_idx + 1) % 2;
        let ubo_alignment = *min_ubo_offset_alignment;
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
                for (b, m) in slot.retired_buffers.drain(..) {
                    unsafe {
                        device.destroy_buffer(b, None);
                        device.free_memory(m, None);
                    }
                }
                for t in slot.retired_textures.drain(..) {
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

        if dummy_white.is_none() {
            *dummy_white = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, false, false,
            )?);
        }
        if dummy_white_array.is_none() {
            *dummy_white_array = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, true, false,
            )?);
        }
        if dummy_white_3d.is_none() {
            *dummy_white_3d = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, false, true,
            )?);
        }
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let dummy_view = dummy_white.as_ref().unwrap().view;
        let dummy_array_view = dummy_white_array.as_ref().unwrap().view;
        let dummy_3d_view = dummy_white_3d.as_ref().unwrap().view;
        let shader_arrayed = call.fs_sampler_arrayed;
        let fallback_view = if shader_arrayed {
            dummy_array_view
        } else {
            dummy_view
        };
        let default_samp = default_sampler.unwrap();
        let color_keys = if call.color_rt_keys.is_empty() {
            vec![call.rt_key]
        } else {
            call.color_rt_keys.clone()
        };
        let tsc_entries = collect_tsc_entries(call, &read_guest);
        let rt_aliases: Vec<_> = (0..tex_pendings.len())
            .map(|slot| rt_alias_for_slot(rt_cache, call, slot, call.rt_key, false))
            .map(|alias| alias.filter(|alias| alias.depth || !color_keys.contains(&alias.key)))
            .collect();
        let mut bound_tex_views = vec![fallback_view; max_texture_descriptors()];
        let mut bound_tex_views_3d = vec![dummy_3d_view; max_texture_descriptors()];
        for (slot, pending) in tex_pendings.iter().enumerate() {
            let pending_volume = pending.map_or(false, |(k, _, _, _)| k.volume);
            if !pending_volume {
                if let Some(alias) = rt_aliases.get(slot).copied().flatten() {
                    bound_tex_views[slot] = alias.view;
                    continue;
                }
            }
            let Some((key, tic, pitch_size, read_size)) = *pending else {
                continue;
            };
            if let Some(raw) = read_guest(tic.gpu_va, read_size) {
                let tex_hash = hash_src_prefix(&raw);
                let cur_gen = crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64);
                let force_refresh = force_refresh_texture(tic.gpu_va);
                let need_upload = force_refresh
                    || match tex_cache.get(&key) {
                        Some(t) => t.gen != cur_gen || t.hash != tex_hash,
                        None => true,
                    };
                if need_upload {
                    let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH")
                        .map(|v| v == "1")
                        .unwrap_or(false);
                    let rgba8 = decode_texture_rgba8_layers(&raw, &tic, pitch_size, force_pitch);
                    log::debug!(
                        "TIC gpu_va={:#x} {}x{}x{} fmt={:?} bl={} bh={} src_bytes={} rgba8_bytes={} (cache miss -> upload)",
                        tic.gpu_va, tic.width, tic.height, key.layers, tic.format,
                        tic.is_block_linear, tic.block_height_log2, read_size, rgba8.len()
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
                        key.volume,
                        &rgba8,
                        tic.swizzle,
                        vk::Format::R8G8B8A8_UNORM,
                        tex_hash,
                        cur_gen,
                    ) {
                        Ok(tex) => {
                            if let Some(old) = tex_cache.insert(key, tex) {
                                frame_slots[cur_idx].retired_textures.push(old);
                            }
                        }
                        Err(e) => log::warn!("texture upload failed: {}", e),
                    }
                }
            }
            if key.volume {
                bound_tex_views_3d[slot] =
                    tex_cache.get(&key).map(|t| t.view).unwrap_or(dummy_3d_view);
            } else {
                bound_tex_views[slot] =
                    tex_cache.get(&key).map(|t| t.view).unwrap_or(fallback_view);
            }
        }
        let mut bound_samplers = vec![default_samp; max_texture_descriptors()];
        for (slot, tsc) in tsc_entries.iter().enumerate() {
            let Some(t) = *tsc else {
                continue;
            };
            bound_samplers[slot] = match sampler_cache.get(&t) {
                Some(s) => *s,
                None => match create_sampler_for_tsc(device, &t) {
                    Ok(s) => {
                        sampler_cache.insert(t, s);
                        s
                    }
                    Err(e) => {
                        log::warn!("tsc sampler create failed: {}", e);
                        default_samp
                    }
                },
            };
        }

        let vertex_bind: Option<(vk::Buffer, u64)> = if !vertex_data.is_empty() {
            let v_align = vertex_stride.max(16);
            let v_size = align_up(vertex_data.len() as u64, v_align);
            if ubo_ring.head + v_size > ubo_ring.size {
                *tele_ring_wraps += 1;
                let other = &mut frame_slots[other_idx];
                if other.in_flight {
                    wait_fence(device, other.fence)?;
                    *tele_ring_waits += 1;
                    if !other.retired_dsets.is_empty() {
                        unsafe {
                            let _ = device
                                .free_descriptor_sets(descriptor_pool.pool, &other.retired_dsets);
                        }
                        other.retired_dsets.clear();
                    }
                    reset_command_buffer(device, other.cmd)?;
                    other.in_flight = false;
                }
                ubo_ring.head = 0;
                ubo_ring.slot_head[other_idx] = 0;
            }
            let (vbuf, voff, vptr) = ring_alloc(ubo_ring, v_size, v_align)
                .map_err(|e| format!("ring_alloc(vertex): {}", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(vertex_data.as_ptr(), vptr, vertex_data.len());
            }
            Some((vbuf, voff))
        } else {
            None
        };

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

        let cbuf_size_aligned = align_up(cbuf_data.len() as u64, ubo_alignment);
        {
            let v_size = if !vertex_data.is_empty() {
                let v_align = vertex_stride.max(16);
                align_up(vertex_data.len() as u64, v_align)
            } else {
                0
            };
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
                reset_command_buffer(device, other.cmd)?;
                other.in_flight = false;
            }
            ubo_ring.head = 0;
            ubo_ring.slot_head[other_idx] = 0;
        }
        let (ubo_buffer, ubo_offset, ubo_ptr) =
            ring_alloc(ubo_ring, cbuf_size_aligned, ubo_alignment)
                .map_err(|e| format!("ring_alloc(ubo): {}", e))?;
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
        let image_infos: Vec<vk::DescriptorImageInfo> = bound_tex_views
            .iter()
            .map(|view| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: *view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect();
        let image_infos_3d: Vec<vk::DescriptorImageInfo> = bound_tex_views_3d
            .iter()
            .map(|view| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: *view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect();
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
                descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: image_infos.as_ptr(),
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
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: dset,
            dst_binding: crate::descriptor::IMAGE3D_BINDING,
            dst_array_element: 0,
            descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: image_infos_3d.as_ptr(),
            p_buffer_info: std::ptr::null(),
            p_texel_buffer_view: std::ptr::null(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut color_bind = Vec::with_capacity(color_keys.len());
        for (idx, key) in color_keys.iter().enumerate() {
            let format = color_formats.get(idx).copied().unwrap_or(call.rt_format);
            let rt = rt_cache.get_or_create_with_format(*key, device, format)?;
            color_bind.push((*key, rt.image, rt.view, rt.extent, rt.layout));
        }
        let (_, _, _, rt_extent, _) = color_bind[0];

        let depth_bind: Option<(vk::Image, vk::ImageView, vk::ImageLayout)> = if use_depth {
            let d = rt_cache.get_or_create_depth(call.depth_key.unwrap(), device)?;
            Some((d.image, d.view, d.layout))
        } else {
            None
        };

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
        if let Some((d_image, _, d_prev)) = depth_bind {
            transition_image_aspect(
                device,
                cmd,
                d_image,
                d_prev,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
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
                            vk::ImageAspectFlags::DEPTH,
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
        let depth_attachment = depth_bind.map(|(_, d_view, _)| vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: d_view,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            resolve_mode: vk::ResolveModeFlags::NONE,
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            load_op: vk::AttachmentLoadOp::LOAD,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
        let p_depth_attachment = match &depth_attachment {
            Some(a) => a as *const _,
            None => std::ptr::null(),
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
        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: rt_extent,
            },
            layer_count: 1,
            view_mask: 0,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment,
            p_stencil_attachment: std::ptr::null(),
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
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_cache.layout,
                0,
                &[dset],
                &[],
            );
            if let Some((vbuf, voff)) = vertex_bind {
                for b in call.vertex_layout.bindings.iter().filter(|b| b.stride > 0) {
                    device.cmd_bind_vertex_buffers(cmd, b.binding, &[vbuf], &[voff]);
                }
            }
            if let Some((wbinding, wbuf, woff)) = white_bind {
                device.cmd_bind_vertex_buffers(cmd, wbinding, &[wbuf], &[woff]);
            }
            let cmd_first_vertex = if vertex_bind.is_some() {
                0
            } else {
                call.first_vertex
            };
            if let Some((ibuf, ioff)) = index_bind {
                device.cmd_bind_index_buffer(cmd, ibuf, ioff, index_type);
                device.cmd_draw_indexed(
                    cmd,
                    index_count,
                    call.instance_count.max(1),
                    0,
                    0,
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
        if !call.blend.color_write_mask.is_empty() {
            for (key, _, _, _, _) in &color_bind {
                let stamp = rt_cache.mark_drawn(*key);
                trace_rt_stamp(stamp, *key, &[call]);
            }
        }
        if use_depth {
            if let Ok(d) = rt_cache.get_or_create_depth(call.depth_key.unwrap(), device) {
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
        if calls.is_empty() {
            return Ok(());
        }

        struct Prep {
            pipeline: vk::Pipeline,
            vertex_data: Vec<u8>,
            cbuf_data: Vec<u8>,
            vertex_stride: u64,
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
        for call in calls {
            let use_depth = call.depth_key.is_some();
            let depth_format = if use_depth {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::UNDEFINED
            };
            let color_formats = color_formats_for_call(call, call.color_rt_keys.len().max(1));
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
                call.vertex_count,
            )? {
                Some(p) => p,
                None => continue,
            };
            let vertex_stride = call
                .vertex_layout
                .bindings
                .iter()
                .find(|b| b.stride > 0)
                .map(|b| b.stride as u64)
                .unwrap_or(0);
            let vertex_base_addr = call
                .vertex_addr
                .wrapping_add(vertex_stride.saturating_mul(call.first_vertex as u64));
            let vertex_bytes = vertex_stride.saturating_mul(call.vertex_count as u64) as usize;
            let mut vertex_data = if vertex_bytes > 0 {
                read_guest(vertex_base_addr, vertex_bytes)
                    .ok_or_else(|| format!("vertex read failed va={:#x}", vertex_base_addr))?
            } else {
                Vec::new()
            };
            let draw_vertex_count =
                if call.quad_expand && vertex_stride > 0 && !vertex_data.is_empty() {
                    vertex_data =
                        crate::draw::expand_quad_vertices(&vertex_data, vertex_stride as usize);
                    (vertex_data.len() / vertex_stride as usize) as u32
                } else {
                    call.vertex_count
                };
            let cbuf_size = call.cbuf_size as usize;
            let cbuf_data = if let Some(d) = &call.cbuf_data {
                d.clone()
            } else if cbuf_size > 0 && call.cbuf_addr != 0 {
                read_guest(call.cbuf_addr, cbuf_size).unwrap_or_else(|| vec![0u8; cbuf_size])
            } else {
                vec![0u8; 256]
            };
            let tex_pendings = collect_tex_pendings(call, &read_guest);
            let tsc_entries = collect_tsc_entries(call, &read_guest);
            let (index_data, index_count, index_type) = match (&call.index_data, call.index_count) {
                (Some(d), Some(c)) if c > 0 && !d.is_empty() => (d.clone(), c, call.index_type),
                _ => (Vec::new(), 0u32, call.index_type),
            };
            if let Ok(want) = std::env::var("NEXIUM_VTX_DBG") {
                if parse_u64_value(&want) == Some(call.vs_gpu_va) {
                    let floats: Vec<f32> = vertex_data
                        .chunks_exact(4)
                        .take(24)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
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
                    log::warn!(
                        "[vtx-dbg] vs={:#x} addr={:#x} stride={} vlen={} vcount={} icount={} itype={:?} attrs=[{}] floats={:?} idx={:?}",
                        call.vs_gpu_va,
                        vertex_base_addr,
                        vertex_stride,
                        vertex_data.len(),
                        call.vertex_count,
                        index_count,
                        index_type,
                        attrs,
                        floats,
                        idx
                    );
                }
            }
            preps.push((
                call,
                Prep {
                    pipeline,
                    vertex_data,
                    cbuf_data,
                    vertex_stride,
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
            dummy_white_array,
            dummy_white_3d,
            default_sampler,
            sampler_cache,
            tex_cache,
            frame_slots,
            frame_index,
            ubo_ring,
            min_ubo_offset_alignment,
            ..
        } = &mut *inner;

        let cur_idx = *frame_index;
        let other_idx = (cur_idx + 1) % 2;
        let ubo_alignment = *min_ubo_offset_alignment;

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
                for (b, m) in slot.retired_buffers.drain(..) {
                    unsafe {
                        device.destroy_buffer(b, None);
                        device.free_memory(m, None);
                    }
                }
                for t in slot.retired_textures.drain(..) {
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
        if dummy_white.is_none() {
            *dummy_white = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, false, false,
            )?);
        }
        if dummy_white_array.is_none() {
            *dummy_white_array = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, true, false,
            )?);
        }
        if dummy_white_3d.is_none() {
            *dummy_white_3d = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props, false, true,
            )?);
        }
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let dummy_view = dummy_white.as_ref().unwrap().view;
        let dummy_array_view = dummy_white_array.as_ref().unwrap().view;
        let dummy_3d_view = dummy_white_3d.as_ref().unwrap().view;
        let default_samp = default_sampler.unwrap();

        let rt_key = calls[0].rt_key;
        let color_keys = if calls[0].color_rt_keys.is_empty() {
            vec![rt_key]
        } else {
            calls[0].color_rt_keys.clone()
        };
        let color_formats = color_formats_for_call(&calls[0], color_keys.len());
        let mut color_bind = Vec::with_capacity(color_keys.len());
        for (idx, key) in color_keys.iter().enumerate() {
            let format = color_formats
                .get(idx)
                .copied()
                .unwrap_or(calls[0].rt_format);
            let rt = rt_cache.get_or_create_with_format(*key, device, format)?;
            color_bind.push((*key, rt.image, rt.view, rt.extent, rt.layout));
        }
        let (_, _, _, rt_extent, rt_prev_layout) = color_bind[0];
        let any_depth = preps.iter().any(|p| p.1.use_depth);
        let depth_key = if any_depth { calls[0].depth_key } else { None };
        let (depth_image, depth_view, depth_prev) = if any_depth {
            let d = rt_cache.get_or_create_depth(depth_key.unwrap(), device)?;
            (Some(d.image), Some(d.view), d.layout)
        } else {
            (None, None, vk::ImageLayout::UNDEFINED)
        };

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
                vk::ImageAspectFlags::DEPTH,
            );
        }

        let clear_rt = rt_prev_layout == vk::ImageLayout::UNDEFINED;

        let mut dsets_batch: Vec<vk::DescriptorSet> = Vec::new();
        let mut alias_used: Vec<(RtKey, bool)> = Vec::new();
        let mut tex_raw_cache: HashMap<(u64, usize), Option<(u64, Vec<u8>)>> = HashMap::new();
        let mut pass_open = false;
        let mut pass_depth = false;
        let mut pass_rt_layout = vk::ImageLayout::UNDEFINED;
        let mut had_pass = false;
        for (_i, (call, prep)) in preps.iter().enumerate() {
            let call = *call;
            let feedback_loop = color_keys
                .iter()
                .copied()
                .any(|key| call_samples_rt(call, key));
            let required_rt_layout = if feedback_loop {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            };

            let rt_aliases: Vec<_> = (0..prep.tex_pendings.len())
                .map(|slot| rt_alias_for_slot(rt_cache, call, slot, rt_key, true))
                .collect();
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
                                pass_open = false;
                                for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
                                    color_layouts[idx] = pass_rt_layout;
                                    rt_cache.set_color_layout(*key, pass_rt_layout);
                                }
                            }
                            if alias.depth {
                                transition_image_aspect(
                                    device,
                                    cmd,
                                    alias.image,
                                    alias_prev,
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                    vk::ImageAspectFlags::DEPTH,
                                );
                                rt_cache.set_depth_layout(
                                    alias.key,
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

            let shader_arrayed = call.fs_sampler_arrayed;
            let fallback_view = if shader_arrayed {
                dummy_array_view
            } else {
                dummy_view
            };
            let mut bound_tex_views = vec![fallback_view; max_texture_descriptors()];
            let mut bound_tex_views_3d = vec![dummy_3d_view; max_texture_descriptors()];
            let mut bound_tex_layouts =
                vec![vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL; max_texture_descriptors()];
            for (slot, pending) in prep.tex_pendings.iter().enumerate() {
                let pending_volume = pending.map_or(false, |(k, _, _, _)| k.volume);
                if !pending_volume {
                    if let Some(alias) = rt_aliases.get(slot).copied().flatten() {
                        bound_tex_views[slot] = alias.view;
                        if !alias.depth
                            && sampled_rt_key_for_slot(call, slot)
                                .map_or(false, |key| color_keys.contains(&key))
                        {
                            bound_tex_layouts[slot] = required_rt_layout;
                        }
                        continue;
                    }
                }
                let Some((key, tic, pitch_size, read_size)) = *pending else {
                    continue;
                };
                let raw_entry = match tex_raw_cache.entry((tic.gpu_va, read_size)) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        entry.insert(read_guest(tic.gpu_va, read_size).map(|raw| {
                            let tex_hash = hash_src_prefix(&raw);
                            (tex_hash, raw)
                        }))
                    }
                };
                if let Some((tex_hash, raw)) = raw_entry.as_ref() {
                    let raw_hash = *tex_hash;
                    let mut tex_hash = raw_hash;
                    let cur_gen =
                        crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64);
                    let identity_volume =
                        key.volume && std::env::var_os("NEXIUM_VOLUME_IDENTITY").is_some();
                    let volume_slices = if key.volume && !identity_volume {
                        find_volume_rt_slices(rt_cache, &tic, pitch_size, key.layers)
                    } else {
                        None
                    };
                    if let Some(slices) = volume_slices.as_ref() {
                        tex_hash = volume_rt_slice_hash(tex_hash, slices);
                    }
                    let force_refresh = force_refresh_texture(tic.gpu_va);
                    let need_upload = force_refresh
                        || match tex_cache.get(&key) {
                            Some(t) => {
                                if key.volume && volume_slices.is_none() && t.hash != raw_hash {
                                    t.gen != cur_gen
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
                        let rgba8 = if identity_volume {
                            identity_volume_rgba8(key.width, key.height, key.layers)
                        } else {
                            decode_texture_rgba8_layers(raw, &tic, pitch_size, force_pitch)
                        };
                        if std::env::var_os("NEXIUM_TEXDUMP")
                            .map(|v| v == "1")
                            .unwrap_or(false)
                        {
                            use std::sync::{Mutex, OnceLock};
                            static SEEN: OnceLock<Mutex<std::collections::HashSet<u64>>> =
                                OnceLock::new();
                            let s =
                                SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                            if s.lock().unwrap().insert(tic.gpu_va) {
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
                                    &raw[..16.min(raw.len())],
                                );
                            }
                        }
                        if std::env::var_os("NEXIUM_TEXDUMP_IMG")
                            .map(|v| v == "1")
                            .unwrap_or(false)
                        {
                            dump_texture_bmp_once(
                                tic.gpu_va,
                                tic.width,
                                tic.height,
                                key.layers,
                                &rgba8,
                                tic.swizzle,
                            );
                        }
                        if pass_open {
                            unsafe {
                                device.cmd_end_rendering(cmd);
                            }
                            pass_open = false;
                            for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
                                color_layouts[idx] = pass_rt_layout;
                                rt_cache.set_color_layout(*key, pass_rt_layout);
                            }
                        }
                        let image_format = if identity_volume {
                            vk::Format::R8G8B8A8_UNORM
                        } else if let Some(slice) =
                            volume_slices.as_ref().and_then(|slices| slices.first())
                        {
                            slice.format
                        } else {
                            texture_image_format(tic.format, false)
                        };
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
                            key.volume,
                            &rgba8,
                            volume_slices.as_deref(),
                            tic.swizzle,
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
                            Err(e) => log::warn!("texture upload failed: {}", e),
                        }
                    }
                }
                if key.volume {
                    bound_tex_views_3d[slot] =
                        tex_cache.get(&key).map(|t| t.view).unwrap_or(dummy_3d_view);
                } else {
                    bound_tex_views[slot] =
                        tex_cache.get(&key).map(|t| t.view).unwrap_or(fallback_view);
                }
            }

            let vertex_bind: Option<(vk::Buffer, u64)> = if !prep.vertex_data.is_empty() {
                let v_align = prep.vertex_stride.max(16);
                let v_size = align_up(prep.vertex_data.len() as u64, v_align);
                if ubo_ring.head + v_size > ubo_ring.size {
                    ring_wrap_other(
                        device,
                        frame_slots,
                        other_idx,
                        descriptor_pool.pool,
                        ubo_ring,
                    )?;
                }
                let (vbuf, voff, vptr) = ring_alloc(ubo_ring, v_size, v_align)
                    .map_err(|e| format!("ring_alloc(vertex): {}", e))?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prep.vertex_data.as_ptr(),
                        vptr,
                        prep.vertex_data.len(),
                    );
                }
                Some((vbuf, voff))
            } else {
                None
            };

            let white_bind: Option<(u32, vk::Buffer, u64)> =
                if let Some(wb) = call.vertex_layout.bindings.iter().find(|b| b.stride == 0) {
                    if ubo_ring.head + 16 > ubo_ring.size {
                        ring_wrap_other(
                            device,
                            frame_slots,
                            other_idx,
                            descriptor_pool.pool,
                            ubo_ring,
                        )?;
                    }
                    let (wbuf, woff, wptr) = ring_alloc(ubo_ring, 16, 16)
                        .map_err(|e| format!("ring_alloc(white): {}", e))?;
                    unsafe {
                        let white = [1.0f32, 1.0, 1.0, 1.0];
                        std::ptr::copy_nonoverlapping(white.as_ptr() as *const u8, wptr, 16);
                    }
                    Some((wb.binding, wbuf, woff))
                } else {
                    None
                };

            let index_bind: Option<(vk::Buffer, u64)> =
                if prep.index_count > 0 && !prep.index_data.is_empty() {
                    let isz = align_up(prep.index_data.len() as u64, 4);
                    if ubo_ring.head + isz > ubo_ring.size {
                        ring_wrap_other(
                            device,
                            frame_slots,
                            other_idx,
                            descriptor_pool.pool,
                            ubo_ring,
                        )?;
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

            let cbuf_size_aligned = align_up(prep.cbuf_data.len() as u64, ubo_alignment);
            if ubo_ring.head + cbuf_size_aligned > ubo_ring.size {
                ring_wrap_other(
                    device,
                    frame_slots,
                    other_idx,
                    descriptor_pool.pool,
                    ubo_ring,
                )?;
            }
            let (ubo_buffer, ubo_offset, ubo_ptr) =
                ring_alloc(ubo_ring, cbuf_size_aligned, ubo_alignment)
                    .map_err(|e| format!("ring_alloc(ubo): {}", e))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    prep.cbuf_data.as_ptr(),
                    ubo_ptr,
                    prep.cbuf_data.len(),
                );
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
            let dset = unsafe {
                device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| format!("allocate_descriptor_sets: {:?}", e))?[0]
            };
            let ubo_info = vk::DescriptorBufferInfo {
                buffer: ubo_buffer,
                offset: ubo_offset,
                range: prep.cbuf_data.len() as u64,
            };
            let image_infos: Vec<vk::DescriptorImageInfo> = bound_tex_views
                .iter()
                .zip(bound_tex_layouts.iter())
                .map(|(view, layout)| vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: *view,
                    image_layout: *layout,
                })
                .collect();
            let image_infos_3d: Vec<vk::DescriptorImageInfo> = bound_tex_views_3d
                .iter()
                .map(|view| vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: *view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                })
                .collect();
            let mut bound_samplers = vec![default_samp; max_texture_descriptors()];
            for (slot, tsc) in prep.tsc_entries.iter().enumerate() {
                let Some(t) = *tsc else {
                    continue;
                };
                bound_samplers[slot] = match sampler_cache.get(&t) {
                    Some(s) => *s,
                    None => match create_sampler_for_tsc(device, &t) {
                        Ok(s) => {
                            sampler_cache.insert(t, s);
                            s
                        }
                        Err(e) => {
                            log::warn!("tsc sampler create failed: {}", e);
                            default_samp
                        }
                    },
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
                if ubo_ring.head + sz_al > ubo_ring.size {
                    ubo_ring.head = 0;
                    ubo_ring.slot_head[other_idx] = 0;
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
                    descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                    descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                    p_image_info: image_infos.as_ptr(),
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
            writes.push(vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                dst_set: dset,
                dst_binding: crate::descriptor::IMAGE3D_BINDING,
                dst_array_element: 0,
                descriptor_count: crate::descriptor::MAX_TEXTURE_DESCRIPTORS,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                p_image_info: image_infos_3d.as_ptr(),
                p_buffer_info: std::ptr::null(),
                p_texel_buffer_view: std::ptr::null(),
                p_next: std::ptr::null(),
                _marker: std::marker::PhantomData,
            });
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
            dsets_batch.push(dset);

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
                    for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
                        color_layouts[idx] = pass_rt_layout;
                        rt_cache.set_color_layout(*key, pass_rt_layout);
                    }
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
                                vk::ImageAspectFlags::DEPTH,
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
                    depth_view.map(|dv| vk::RenderingAttachmentInfo {
                        s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                        image_view: dv,
                        image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        resolve_mode: vk::ResolveModeFlags::NONE,
                        resolve_image_view: vk::ImageView::null(),
                        resolve_image_layout: vk::ImageLayout::UNDEFINED,
                        load_op: vk::AttachmentLoadOp::LOAD,
                        store_op: vk::AttachmentStoreOp::STORE,
                        clear_value: vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                        p_next: std::ptr::null(),
                        _marker: std::marker::PhantomData,
                    })
                } else {
                    None
                };
                let p_depth_attachment = match &depth_attachment {
                    Some(a) => a as *const _,
                    None => std::ptr::null(),
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
                let render_info = vk::RenderingInfo {
                    s_type: vk::StructureType::RENDERING_INFO,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: rt_extent,
                    },
                    layer_count: 1,
                    view_mask: 0,
                    color_attachment_count: attachments.len() as u32,
                    p_color_attachments: attachments.as_ptr(),
                    p_depth_attachment,
                    p_stencil_attachment: std::ptr::null(),
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
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_cache.layout,
                    0,
                    &[dset],
                    &[],
                );
                if let Some((vbuf, voff)) = vertex_bind {
                    for b in call.vertex_layout.bindings.iter().filter(|b| b.stride > 0) {
                        device.cmd_bind_vertex_buffers(cmd, b.binding, &[vbuf], &[voff]);
                    }
                }
                if let Some((wbinding, wbuf, woff)) = white_bind {
                    device.cmd_bind_vertex_buffers(cmd, wbinding, &[wbuf], &[woff]);
                }
                let cmd_first_vertex = if vertex_bind.is_some() {
                    0
                } else {
                    call.first_vertex
                };
                if let Some((ibuf, ioff)) = index_bind {
                    device.cmd_bind_index_buffer(cmd, ibuf, ioff, prep.index_type);
                    device.cmd_draw_indexed(
                        cmd,
                        prep.index_count,
                        call.instance_count.max(1),
                        0,
                        0,
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
        }
        if pass_open {
            unsafe {
                device.cmd_end_rendering(cmd);
            }
            for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
                color_layouts[idx] = pass_rt_layout;
                rt_cache.set_color_layout(*key, pass_rt_layout);
            }
        }

        unsafe {
            device
                .end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(batch): {:?}", e))?;
        }
        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx].retired_dsets.extend(dsets_batch);
        for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
            rt_cache.set_color_layout(*key, color_layouts[idx]);
        }
        if calls
            .iter()
            .any(|call| !call.blend.color_write_mask.is_empty())
        {
            let trace_calls: Vec<_> = calls.iter().collect();
            for (key, _, _, _, _) in &color_bind {
                let stamp = rt_cache.mark_drawn(*key);
                trace_rt_stamp(stamp, *key, &trace_calls);
            }
        }
        if any_depth {
            if let Ok(d) = rt_cache.get_or_create_depth(depth_key.unwrap(), device) {
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

fn trace_present_key(rt_cache: &RtCache, requested_key: RtKey, key: RtKey) {
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
    rgb_nonzero: u64,
    alpha_nonzero: u64,
    rgb_sum: u64,
    alpha_sum: u64,
    rgb_max: u8,
    bbox: Option<(u32, u32, u32, u32)>,
    first: Option<(u32, u32, [u8; 4])>,
    pixel_rows: Vec<String>,
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
                log::warn!(
                    "[rt-stats] seq={} key={} stamp={} rgbnz={}/{} ({:.2}%) anz={} avg_rgb={:.2} avg_a={:.2} max={} bbox={} first={}",
                    seq,
                    key.label(),
                    stamp,
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
}

fn rt_stats_seq() -> Option<u64> {
    let enabled = std::env::var_os("NEXIUM_RT_STATS")?;
    if enabled.to_string_lossy().trim() == "0" {
        return None;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
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

fn rt_pixels_enabled() -> bool {
    std::env::var_os("NEXIUM_RT_PIXELS")
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

fn read_rt_image_stats(
    device: &ash::Device,
    cmd_pool: vk::CommandPool,
    queue: vk::Queue,
    rt_cache: &mut RtCache,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    key: RtKey,
) -> Option<RtImageStats> {
    let existing = rt_cache.get_existing(key)?;
    let format = existing.format;
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
    let img = match rt_cache.get_existing(key) {
        Some(i) => i,
        None => {
            unsafe {
                let _ = device.end_command_buffer(cmd);
            }
            cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
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
    img.layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    if end_one_time(device, cmd).is_err() || submit_with_fence(device, queue, cmd, fence).is_err() {
        cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
        return None;
    }
    let mut stats = RtImageStats {
        pixels: (key.width as u64) * (key.height as u64),
        ..RtImageStats::default()
    };
    unsafe {
        let _ = device.wait_for_fences(&[fence], true, u64::MAX);
        let ptr = match device.map_memory(stage.memory, 0, stage.size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => ptr as *const u8,
            Err(_) => {
                cleanup(device, cmd_pool, Some(fence), Some(cmd), &stage);
                return None;
            }
        };
        let data = std::slice::from_raw_parts(ptr, total as usize);
        let rgba = readback_to_rgba8(data, format, key.width, key.height);
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
        if rt_pixels_enabled() {
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
        vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_SINT
        | vk::Format::R8_UINT => 1,
        _ => 4,
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
                let a = v & 0x3;
                let b = (v >> 2) & 0x3ff;
                let g = (v >> 12) & 0x3ff;
                let r = (v >> 22) & 0x3ff;
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

fn max_texture_descriptors() -> usize {
    crate::descriptor::MAX_TEXTURE_DESCRIPTORS as usize
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

fn tic_is_arrayed(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 5
}

fn tic_is_volume(tic: &crate::texture::TicEntry) -> bool {
    tic.texture_type == 2
}

fn tic_layer_count(tic: &crate::texture::TicEntry) -> u32 {
    if tic_is_arrayed(tic) {
        tic.base_layer.saturating_add(tic.depth).max(1)
    } else if tic_is_volume(tic) {
        tic.depth.max(1)
    } else {
        1
    }
}

fn tic_view_base_layer(tic: &crate::texture::TicEntry) -> u32 {
    if tic_is_arrayed(tic) {
        tic.base_layer
    } else {
        0
    }
}

fn tic_view_layer_count(tic: &crate::texture::TicEntry) -> u32 {
    if tic_is_arrayed(tic) {
        tic.depth.max(1)
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

fn decode_texture_rgba8_layers(
    raw: &[u8],
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    force_pitch: bool,
) -> Vec<u8> {
    let layers = tic_layer_count(tic) as usize;
    let layer_read_size = tic_layer_read_size(tic, pitch_size);
    let effective_block_linear =
        tic.is_block_linear && !crate::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let mut out = Vec::new();
    let layer_rgba_size = tic.width as usize * tic.height as usize * 4;
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_read_size);
        if start >= raw.len() {
            out.resize(out.len() + layer_rgba_size, 0);
            break;
        }
        let end = (start + layer_read_size).min(raw.len());
        let layer_raw = &raw[start..end];
        let linear: Vec<u8> = if effective_block_linear && !force_pitch {
            let (storage_width, storage_height, bpp) =
                tic.format.storage_extent(tic.width, tic.height);
            crate::texture::unswizzle_block_linear(
                layer_raw,
                storage_width,
                storage_height,
                bpp,
                tic.block_height_log2,
            )
        } else if layer_raw.len() >= pitch_size {
            layer_raw[..pitch_size].to_vec()
        } else {
            layer_raw.to_vec()
        };
        let mut decoded =
            crate::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        out.extend(decoded);
    }
    out.resize(layer_rgba_size.saturating_mul(layers), 0);
    out
}

fn find_volume_rt_slices(
    rt_cache: &RtCache,
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    layers: u32,
) -> Option<Vec<VolumeRtSlice>> {
    if layers == 0 {
        return None;
    }
    let nominal_slice_size = tic_layer_read_size(tic, pitch_size) as u64;
    if nominal_slice_size == 0 {
        return None;
    }
    let allow_partial = std::env::var_os("NEXIUM_VOLUME_PARTIAL").is_some();
    let Some((first_key, first_image, first_layout, first_format, first_stamp)) =
        rt_cache.find_drawn_color_at(tic.width, tic.height, tic.gpu_va)
    else {
        if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
            use std::collections::HashSet;
            use std::sync::{Mutex, OnceLock};
            static MISSING: OnceLock<Mutex<HashSet<(u64, u32)>>> = OnceLock::new();
            let missing = MISSING.get_or_init(|| Mutex::new(HashSet::new()));
            if missing.lock().unwrap().insert((tic.gpu_va, 0)) {
                log::warn!(
                    "[volume-rt-miss] va={:#x} layer=0 slice_va={:#x} {}x{}x{} slice_size={}",
                    tic.gpu_va,
                    tic.gpu_va,
                    tic.width,
                    tic.height,
                    layers,
                    nominal_slice_size
                );
            }
        }
        return None;
    };
    let slice_size = nominal_slice_size.max(
        (first_key.width as u64)
            .saturating_mul(first_key.height as u64)
            .saturating_mul(tic.format.src_bpp() as u64),
    );
    let mut out = Vec::with_capacity(layers as usize);
    out.push(VolumeRtSlice {
        layer: 0,
        key: first_key,
        image: first_image,
        layout: first_layout,
        format: first_format,
        stamp: first_stamp,
    });
    for layer in 1..layers {
        let va = tic
            .gpu_va
            .checked_add(slice_size.saturating_mul(layer as u64))?;
        let Some((key, image, layout, format, stamp)) =
            rt_cache.find_drawn_color_at(tic.width, tic.height, va)
        else {
            if std::env::var_os("NEXIUM_VOLUME_DBG").is_some() {
                use std::collections::HashSet;
                use std::sync::{Mutex, OnceLock};
                static MISSING: OnceLock<Mutex<HashSet<(u64, u32)>>> = OnceLock::new();
                let missing = MISSING.get_or_init(|| Mutex::new(HashSet::new()));
                if missing.lock().unwrap().insert((tic.gpu_va, layer)) {
                    log::warn!(
                        "[volume-rt-miss] va={:#x} layer={} slice_va={:#x} {}x{}x{} slice_size={}",
                        tic.gpu_va,
                        layer,
                        va,
                        tic.width,
                        tic.height,
                        layers,
                        slice_size
                    );
                }
            }
            if allow_partial {
                continue;
            }
            return None;
        };
        out.push(VolumeRtSlice {
            layer,
            key,
            image,
            layout,
            format,
            stamp,
        });
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
                "[volume-rt] va={:#x} {}x{}x{} found={}/{} slice_size={} first={} last={} formats={}",
                tic.gpu_va,
                tic.width,
                tic.height,
                layers,
                out.len(),
                layers,
                slice_size,
                first,
                last,
                formats.join("|")
            );
        }
    }
    Some(out)
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

fn collect_tex_pendings<F>(
    call: &crate::draw::Maxwell3dDrawCall,
    read_guest: &F,
) -> Vec<Option<PendingTexture>>
where
    F: Fn(u64, usize) -> Option<Vec<u8>>,
{
    let shader_arrayed = call.fs_sampler_arrayed;
    call.fs_tex_ids
        .iter()
        .take(max_texture_descriptors())
        .map(|tex_id| {
            if *tex_id == u32::MAX || *tex_id > call.tic_pool_limit || call.tic_pool_gpu_va == 0 {
                return None;
            }
            let tic_addr = call.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
            read_guest(tic_addr, 32).and_then(|tic_raw| {
                crate::texture::TicEntry::parse(&tic_raw).map(|tic| {
                    let pitch_size = tic.format.linear_size(tic.width, tic.height);
                    let volume = tic_is_volume(&tic);
                    let arrayed = shader_arrayed && !volume;
                    let layers = if arrayed || volume {
                        tic_layer_count(&tic)
                    } else {
                        1
                    };
                    let read_size =
                        tic_layer_read_size(&tic, pitch_size).saturating_mul(layers as usize);
                    let key = TexCacheKey {
                        gpu_va: tic.gpu_va,
                        width: tic.width,
                        height: tic.height,
                        layers,
                        base_layer: if arrayed {
                            tic_view_base_layer(&tic)
                        } else {
                            0
                        },
                        view_layers: if arrayed {
                            tic_view_layer_count(&tic)
                        } else {
                            1
                        },
                        arrayed,
                        volume,
                        format: tic.format,
                        swizzle: tic.swizzle,
                    };
                    (key, tic, pitch_size, read_size)
                })
            })
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
    call.sampled_rt_slots
        .get(slot)
        .copied()
        .flatten()
        .or_else(|| call.sampled_rt_keys.get(slot).copied())
        .or_else(|| if slot == 0 { call.sampled_rt_key } else { None })
}

fn call_samples_rt(call: &crate::draw::Maxwell3dDrawCall, rt_key: RtKey) -> bool {
    call.sampled_rt_key == Some(rt_key)
        || call.sampled_rt_keys.contains(&rt_key)
        || call
            .sampled_rt_slots
            .iter()
            .any(|slot| *slot == Some(rt_key))
}

fn rt_alias_for_slot(
    rt_cache: &RtCache,
    call: &crate::draw::Maxwell3dDrawCall,
    slot: usize,
    rt_key: RtKey,
    allow_self: bool,
) -> Option<RtAlias> {
    let sk = sampled_rt_key_for_slot(call, slot)?;
    let found = rt_cache.find_color(sk).or_else(|| {
        if call.sampled_rt_fuzzy {
            rt_cache.find_color_screen(sk)
        } else {
            None
        }
    });
    let found_key = found.as_ref().map(|(k, _, _, _)| *k);
    let filtered = found
        .filter(|(k, _, _, _)| allow_self || *k != rt_key)
        .map(|(key, image, view, layout)| RtAlias {
            key,
            image,
            view,
            layout,
            depth: false,
        })
        .or_else(|| {
            rt_cache
                .find_depth(sk)
                .and_then(|(key, image, view, layout)| {
                    if Some(key) != call.depth_key {
                        Some(RtAlias {
                            key,
                            image,
                            view,
                            layout,
                            depth: true,
                        })
                    } else {
                        None
                    }
                })
        });
    trace_rt_alias(
        slot,
        rt_key,
        sk,
        found_key,
        filtered.is_some(),
        call.sampled_rt_fuzzy,
    );
    filtered
}

fn trace_rt_alias(
    slot: usize,
    dst: RtKey,
    src: RtKey,
    found: Option<RtKey>,
    used: bool,
    fuzzy: bool,
) {
    if std::env::var_os("NEXIUM_RT_ALIAS_DBG").is_none() {
        return;
    }
    if src.width < 512 || src.height < 256 {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let limit = std::env::var("NEXIUM_RT_ALIAS_LIMIT")
        .ok()
        .and_then(|v| parse_u64_value(&v))
        .unwrap_or(300);
    if n >= limit {
        return;
    }
    let found = found.map(|k| k.label()).unwrap_or_else(|| "-".to_string());
    log::warn!(
        "[rt-alias] #{} slot={} dst={} src={} found={} used={} fuzzy={}",
        n,
        slot,
        dst.label(),
        src.label(),
        found,
        used,
        fuzzy
    );
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
        for (b, m) in other.retired_buffers.drain(..) {
            unsafe {
                device.destroy_buffer(b, None);
                device.free_memory(m, None);
            }
        }
        for t in other.retired_textures.drain(..) {
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

fn create_sampler_for_tsc(
    device: &ash::Device,
    tsc: &crate::texture::TscEntry,
) -> Result<vk::Sampler, String> {
    let mag = vk_filter(tsc.mag_filter);
    let min = vk_filter(tsc.min_filter);
    let mip = match tsc.mip_filter {
        crate::texture::TexFilter::Linear => vk::SamplerMipmapMode::LINEAR,
        _ => vk::SamplerMipmapMode::NEAREST,
    };
    let info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: mag,
        min_filter: min,
        mipmap_mode: mip,
        address_mode_u: map_wrap(tsc.wrap_u, mag),
        address_mode_v: map_wrap(tsc.wrap_v, mag),
        address_mode_w: map_wrap(tsc.wrap_p, mag),
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
            .map_err(|e| format!("create_sampler(tsc): {:?}", e))
    }
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
    volume: bool,
    rgba8: &[u8],
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
        volume,
        rgba8,
        None,
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
    volume: bool,
    rgba8: &[u8],
    volume_slices: Option<&[VolumeRtSlice]>,
    swizzle: [crate::texture::SwizzleSource; 4],
    format: vk::Format,
    hash: u64,
    gen: u64,
) -> Result<(CachedTexture, Option<(vk::Buffer, vk::DeviceMemory)>), String> {
    let layers = layers.max(1);
    let view_base_layer = if arrayed {
        base_layer.min(layers - 1)
    } else {
        0
    };
    let view_layer_count = if arrayed {
        view_layers.max(1).min(layers - view_base_layer)
    } else {
        1
    };
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
        mip_levels: 1,
        array_layers: if volume { 1 } else { layers },
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

    transition_image(
        device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
            let copy = vk::ImageCopy {
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
                dst_offset: vk::Offset3D {
                    x: 0,
                    y: 0,
                    z: slice.layer as i32,
                },
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };
            unsafe {
                device.cmd_copy_image(
                    cmd,
                    slice.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
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
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: if volume { 1 } else { layers },
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: if volume { layers } else { 1 },
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
    }
    transition_image(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: if volume {
            vk::ImageViewType::TYPE_3D
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        },
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
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
        },
        stage.map(|stage| (stage.buffer, stage.memory)),
    ))
}

fn texture_image_format(format: crate::texture::TicFormat, from_rt_slices: bool) -> vk::Format {
    if from_rt_slices && format == crate::texture::TicFormat::B10G11R11 {
        vk::Format::B10G11R11_UFLOAT_PACK32
    } else {
        vk::Format::R8G8B8A8_UNORM
    }
}

fn color_formats_for_call(
    call: &crate::draw::Maxwell3dDrawCall,
    attachment_count: usize,
) -> Vec<vk::Format> {
    let mut formats = if call.color_rt_formats.is_empty() {
        vec![call.rt_format]
    } else {
        call.color_rt_formats.clone()
    };
    let count = attachment_count.max(1).min(8);
    if formats.len() < count {
        formats.resize(count, call.rt_format);
    }
    formats.truncate(count);
    formats
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

    vk::ComponentMapping {
        r: one(swizzle[0]),
        g: one(swizzle[1]),
        b: one(swizzle[2]),
        a: one(swizzle[3]),
    }
}

fn create_dummy_white_image(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    arrayed: bool,
    volume: bool,
) -> Result<DummyImage, String> {
    let format = vk::Format::R8G8B8A8_UNORM;
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
    let stage = create_host_buffer(
        device,
        mem_props,
        &pixel,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;

    let cmd = alloc_one_time_cmd(device, cmd_pool)?;
    begin_one_time(device, cmd)?;
    transition_image(
        device,
        cmd,
        image,
        vk::ImageLayout::UNDEFINED,
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
        view_type: if volume {
            vk::ImageViewType::TYPE_3D
        } else if arrayed {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        },
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
        device
            .queue_wait_idle(queue)
            .map_err(|e| format!("queue_wait_idle: {:?}", e))?;
    }
    Ok(())
}

fn submit_with_fence(
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
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("wait_for_fences: {:?}", e))?;
        device
            .reset_fences(&[fence])
            .map_err(|e| format!("reset_fences: {:?}", e))?;
    }
    Ok(())
}

fn align_up(x: u64, align: u64) -> u64 {
    debug_assert!(align > 0);
    let mask = align - 1;
    (x + mask) & !mask
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
    let mask = align - 1;
    let aligned_head = (ring.head + mask) & !mask;
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

fn transition_image_aspect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
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
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
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
            .map_err(|e| format!("allocate_memory(readback staging): {:?}", e))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind_buffer_memory(readback): {:?}", e))?;
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
        if let Some(d) = self.dummy_white.take() {
            unsafe {
                self.device.destroy_image_view(d.view, None);
                self.device.destroy_image(d.image, None);
                self.device.free_memory(d.memory, None);
            }
        }
        if let Some(d) = self.dummy_white_array.take() {
            unsafe {
                self.device.destroy_image_view(d.view, None);
                self.device.destroy_image(d.image, None);
                self.device.free_memory(d.memory, None);
            }
        }
        if let Some(d) = self.dummy_white_3d.take() {
            unsafe {
                self.device.destroy_image_view(d.view, None);
                self.device.destroy_image(d.image, None);
                self.device.free_memory(d.memory, None);
            }
        }
        for (_, t) in self.tex_cache.drain() {
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
            for t in slot.retired_textures.drain(..) {
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
        unsafe {
            self.device.destroy_fence(self.utility_slot.fence, None);
        }
        self.utility_slot.fence = vk::Fence::null();
        for (_, mut pending) in self.pending_readbacks.drain() {
            while let Some(pr) = pending.pop_front() {
                if let Some(slot) = self.readback_slots.get_mut(pr.slot) {
                    unsafe {
                        let _ = self.device.wait_for_fences(&[slot.fence], true, u64::MAX);
                    }
                    slot.in_flight = false;
                }
            }
        }
        for slot in self.readback_slots.drain(..) {
            unsafe {
                let _ = self.device.wait_for_fences(&[slot.fence], true, u64::MAX);
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

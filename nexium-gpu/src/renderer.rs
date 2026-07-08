use ash::vk;
use parking_lot::Mutex;
use std::collections::{hash_map::DefaultHasher, hash_map::Entry, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
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
    arrayed: bool,
    volume: bool,
    format: crate::texture::TicFormat,
    swizzle: [crate::texture::SwizzleSource; 4],
}

type PendingTexture = (TexCacheKey, crate::texture::TicEntry, usize, usize);

const TEXTURE_IDENTITY_SWIZZLE: [crate::texture::SwizzleSource; 4] = [
    crate::texture::SwizzleSource::R,
    crate::texture::SwizzleSource::G,
    crate::texture::SwizzleSource::B,
    crate::texture::SwizzleSource::A,
];

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
}

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
    depth: bool,
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

struct FrameSlot {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    in_flight: bool,
    retired_dsets: Vec<vk::DescriptorSet>,
    retired_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
    retired_textures: Vec<CachedTexture>,
    retired_views: Vec<vk::ImageView>,
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
    pub fn wait_idle(&self) {
        let inner = self.inner.lock();
        unsafe {
            let _ = inner.device.device_wait_idle();
        }
    }

    pub fn clear_texture_cache(&self) {
        let mut inner = self.inner.lock();
        let drained: Vec<_> = inner.tex_cache.drain().map(|(_, t)| t).collect();
        let cleared = drained.len();
        for t in drained {
            unsafe {
                inner.device.destroy_image_view(t.view, None);
                inner.device.destroy_image(t.image, None);
                inner.device.free_memory(t.memory, None);
            }
        }
        if cleared != 0 && std::env::var_os("NEXIUM_TEX_CACHE_DBG").is_some() {
            log::warn!("[tex-cache] cleared {} cached textures", cleared);
        }
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
        if sampler_filter_minmax_supported {
            enabled_ext_names.push(vk::EXT_SAMPLER_FILTER_MINMAX_NAME.as_ptr());
            log::info!("VK_EXT_sampler_filter_minmax enabled");
        } else {
            log::info!(
                "VK_EXT_sampler_filter_minmax unavailable; min/max samplers use weighted average"
            );
        }
        let p_next_chain: *mut std::ffi::c_void = if enable_depth_clip_control {
            &mut dcc_feature as *mut _ as *mut std::ffi::c_void
        } else {
            &mut features_13 as *mut _ as *mut std::ffi::c_void
        };
        let core_features = unsafe { instance.get_physical_device_features(physical_device) };
        let depth_clamp_supported = core_features.depth_clamp == vk::TRUE;
        let independent_blend_supported = core_features.independent_blend == vk::TRUE;
        let sampler_anisotropy_supported = core_features.sampler_anisotropy == vk::TRUE;
        if !depth_clamp_supported {
            log::info!("Vulkan depthClamp feature unavailable; Maxwell depth clamp disabled");
        }
        if !independent_blend_supported {
            log::info!("Vulkan independentBlend feature unavailable; per-target masks collapsed");
        }
        if !sampler_anisotropy_supported {
            log::info!("Vulkan samplerAnisotropy feature unavailable; TSC anisotropy disabled");
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
                retired_views: Vec::new(),
            },
            FrameSlot {
                fence: fence_b,
                cmd: frame_cmds[1],
                in_flight: false,
                retired_dsets: Vec::new(),
                retired_buffers: Vec::new(),
                retired_textures: Vec::new(),
                retired_views: Vec::new(),
            },
        ];
        let utility_slot = FrameSlot {
            fence: fence_util,
            cmd: frame_cmds[2],
            in_flight: false,
            retired_dsets: Vec::new(),
            retired_buffers: Vec::new(),
            retired_textures: Vec::new(),
            retired_views: Vec::new(),
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
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create_with_format(key, device, format)?;

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
        let RendererInner {
            device,
            queue,
            rt_cache,
            utility_slot,
            ..
        } = &mut *inner;
        let key = RtKey::new(nvmap_id, width, height, gpu_va);
        let img = rt_cache.get_or_create_with_format(key, device, format)?;

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
        let mut inner = self.inner.lock();
        let RendererInner {
            device,
            cmd_pool,
            queue,
            rt_cache,
            mem_props,
            ..
        } = &mut *inner;
        let key = rt_cache.find_color_key_at_va(nvmap_id, gpu_va)?;
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

    pub fn readback_target_pipelined(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        gpu_va: u64,
        cpu_addr: u64,
        copy_rect: Option<[u32; 4]>,
    ) -> Option<(u32, u32, Vec<u8>, Option<bool>)> {
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
        let requested_key = if gpu_va != 0 {
            RtKey::with_cpu(nvmap_id, width, height, gpu_va, cpu_addr)
        } else if cpu_addr != 0 {
            RtKey::with_cpu(nvmap_id, width, height, 0, cpu_addr)
        } else {
            RtKey::request(nvmap_id, width, height)
        };
        let key = rt_cache.resolve_present_key(requested_key)?;
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
            let out = readback_to_rgba8(&raw, prev.format, prev.width, prev.height);
            ready_frame = Some((prev.width, prev.height, out, prev.flip_y));
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
        _vertex_count: u32,
    ) -> Result<Option<vk::Pipeline>, String> {
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
            call.vertex_count,
        )? {
            Some(p) => p,
            None => return Ok(()),
        };

        let (vertex_bindings, draw_vertex_count) = prepare_vertex_bindings(call, &read_guest)?;

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
            sampler_filter_minmax_supported,
            sampler_anisotropy_supported,
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
                    let swizzle = pending
                        .map(|(_, tic, _, _)| tic.swizzle)
                        .unwrap_or(TEXTURE_IDENTITY_SWIZZLE);
                    let view_format = pending
                        .map(|(_, tic, _, _)| rt_alias_view_format(alias.key, tic, alias.format))
                        .unwrap_or(alias.format);
                    bound_tex_views[slot] = rt_alias_sample_view(
                        device,
                        &mut frame_slots[cur_idx],
                        alias,
                        swizzle,
                        view_format,
                    );
                    if bind_trace_fs(call.fs_gpu_va) {
                        log::warn!(
                            "[bind-trace] EXD fs={:#x} slot={} ALIAS key={} alias_fmt={:?} view_fmt={:?} swz={:?} bound={:?}",
                            call.fs_gpu_va, slot, alias.key.label(), alias.format,
                            view_format, swizzle, bound_tex_views[slot]
                        );
                    }
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
            let Some((key, tic, pitch_size, read_size)) = *pending else {
                if bind_trace_fs(call.fs_gpu_va) {
                    log::warn!(
                        "[bind-trace] EXD fs={:#x} slot={} DUMMY no_pending",
                        call.fs_gpu_va,
                        slot
                    );
                }
                trace_vs_tex_bind_dummy(call, slot, &read_guest);
                continue;
            };
            let cur_gen = crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64);
            let identity_volume =
                key.volume && std::env::var_os("NEXIUM_VOLUME_IDENTITY").is_some();
            let volume_slices = if key.volume && !identity_volume {
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
            let raw = read_guest(tic.gpu_va, read_size);
            if raw.is_some() || volume_slices.is_some() || identity_volume {
                let raw_hash = raw.as_ref().map(|raw| hash_src_prefix(raw));
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
                        vk::Format::R8G8B8A8_UNORM
                    } else if let Some(slice) =
                        volume_slices.as_ref().and_then(|slices| slices.first())
                    {
                        slice.format
                    } else {
                        texture_image_format(tic.format, false, tic.is_srgb)
                    };
                    let texels = if volume_slices.is_some() {
                        Vec::new()
                    } else if identity_volume {
                        identity_volume_rgba8(key.width, key.height, key.layers)
                    } else if let Some(raw) = raw.as_ref() {
                        texture_upload_data(raw, &tic, pitch_size, force_pitch, image_format)
                    } else {
                        Vec::new()
                    };
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
                        key.volume,
                        &texels,
                        volume_slices.as_deref(),
                        tic.swizzle,
                        image_format,
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
            if bind_trace_fs(call.fs_gpu_va) {
                log::warn!(
                    "[bind-trace] EXD fs={:#x} slot={} TEX va={:#x} {}x{}x{} vol={} cache_hit={} bound={:?}",
                    call.fs_gpu_va, slot, tic.gpu_va, tic.width, tic.height, key.layers,
                    key.volume, tex_cache.get(&key).is_some(),
                    if key.volume { bound_tex_views_3d[slot] } else { bound_tex_views[slot] }
                );
            }
            trace_vs_tex_bind_texture(
                call,
                slot,
                key,
                tic,
                tex_cache.get(&key).is_some(),
                if key.volume {
                    bound_tex_views_3d[slot]
                } else {
                    bound_tex_views[slot]
                },
            );
        }
        let mut bound_samplers = vec![default_samp; max_texture_descriptors()];
        for (slot, tsc) in tsc_entries.iter().enumerate() {
            let Some(t) = *tsc else {
                continue;
            };
            bound_samplers[slot] = match sampler_cache.get(&t) {
                Some(s) => *s,
                None => match create_sampler_for_tsc(
                    device,
                    &t,
                    *sampler_filter_minmax_supported,
                    *sampler_anisotropy_supported,
                ) {
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

        let vertex_binds = upload_vertex_bindings(
            device,
            frame_slots,
            other_idx,
            descriptor_pool.pool,
            ubo_ring,
            &vertex_bindings,
        )?;

        let white_bind: Option<(u32, vk::Buffer, u64)> =
            if let Some(wb) = call.vertex_layout.bindings.iter().find(|b| b.stride == 0) {
                let (wbuf, woff, wptr) = ring_alloc(ubo_ring, 16, 16)
                    .map_err(|e| format!("ring_alloc(const_attr): {}", e))?;
                unsafe {
                    let const_default = [1.0f32, 1.0, 1.0, 1.0];
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
        let rt_extent = color_bind
            .first()
            .map(|(_, _, _, extent, _)| *extent)
            .unwrap_or(vk::Extent2D {
                width: call.rt_key.width,
                height: call.rt_key.height,
            });

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
                rt_cache.record_present_flip(*key, call.flip_y);
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
        for call in calls {
            let use_depth = call.depth_key.is_some();
            let depth_format = if use_depth {
                vk::Format::D32_SFLOAT
            } else {
                vk::Format::UNDEFINED
            };
            if !use_depth && !call_writes_any_color(call) {
                continue;
            }
            let pipeline = match self.compile_pipeline(
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
                call.vertex_count,
            )? {
                Some(p) => p,
                None => continue,
            };
            let (vertex_bindings, draw_vertex_count) = prepare_vertex_bindings(call, &read_guest)?;
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
                    let vertex_base_addr = call
                        .vertex_bindings
                        .first()
                        .map(|b| {
                            b.addr.wrapping_add(
                                (b.stride as u64).saturating_mul(call.first_vertex as u64),
                            )
                        })
                        .unwrap_or(call.vertex_addr);
                    let vertex_stride = first_binding.map(|b| b.stride).unwrap_or(0);
                    let vertex_len: usize = vertex_bindings.iter().map(|b| b.data.len()).sum();
                    log::warn!(
                        "[vtx-dbg] vs={:#x} addr={:#x} stride={} vlen={} vcount={} icount={} itype={:?} attrs=[{}] floats={:?} idx={:?}",
                        call.vs_gpu_va,
                        vertex_base_addr,
                        vertex_stride,
                        vertex_len,
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
            sampler_filter_minmax_supported,
            sampler_anisotropy_supported,
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
        let rt_extent = color_bind
            .first()
            .map(|(_, _, _, extent, _)| *extent)
            .unwrap_or(vk::Extent2D {
                width: rt_key.width,
                height: rt_key.height,
            });
        let rt_prev_layout = color_bind
            .first()
            .map(|(_, _, _, _, layout)| *layout)
            .unwrap_or(vk::ImageLayout::UNDEFINED);
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

        let clear_rt = !color_bind.is_empty() && rt_prev_layout == vk::ImageLayout::UNDEFINED;

        let mut dsets_batch: Vec<vk::DescriptorSet> = Vec::new();
        let mut alias_used: Vec<(RtKey, bool)> = Vec::new();
        let mut tex_raw_cache: HashMap<(u64, usize), Option<(u64, Vec<u8>)>> = HashMap::new();
        let mut pass_open = false;
        let mut pass_depth = false;
        let mut pass_rt_layout = vk::ImageLayout::UNDEFINED;
        let mut pass_dirty = vec![false; color_bind.len()];
        let mut pass_trace_calls: Vec<&crate::draw::Maxwell3dDrawCall> = Vec::new();
        let mut had_pass = false;
        for (_i, (call, prep)) in preps.iter().enumerate() {
            let call = *call;
            let mut rt_aliases: Vec<_> = (0..prep.tex_pendings.len())
                .map(|slot| rt_alias_for_slot(rt_cache, call, slot, rt_key, true))
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
                                    vk::ImageAspectFlags::DEPTH,
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
                    if let Some(sk) = sampled_rt_key_for_slot(call, slot) {
                        if pass_open && sampled_color_needs_sync(rt_cache, sk) {
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
                                *alias_slot = rt_alias_for_slot(rt_cache, call, slot, rt_key, true);
                            }
                        }
                    }
                }
                if !pending_volume {
                    if let Some(alias) = rt_aliases.get(slot).copied().flatten() {
                        let swizzle = pending
                            .map(|(_, tic, _, _)| tic.swizzle)
                            .unwrap_or(TEXTURE_IDENTITY_SWIZZLE);
                        let view_format = pending
                            .map(|(_, tic, _, _)| {
                                rt_alias_view_format(alias.key, tic, alias.format)
                            })
                            .unwrap_or(alias.format);
                        bound_tex_views[slot] = rt_alias_sample_view(
                            device,
                            &mut frame_slots[cur_idx],
                            alias,
                            swizzle,
                            view_format,
                        );
                        if bind_trace_fs(call.fs_gpu_va) {
                            log::warn!(
                                "[bind-trace] EXDS fs={:#x} slot={} ALIAS key={} alias_fmt={:?} view_fmt={:?} swz={:?} alias_view={:?} bound={:?} snap={}",
                                call.fs_gpu_va, slot, alias.key.label(), alias.format,
                                view_format, swizzle, alias.view, bound_tex_views[slot],
                                alias_snapshotted.get(slot).copied().unwrap_or(false)
                            );
                        }
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
                let Some((key, tic, pitch_size, read_size)) = *pending else {
                    if bind_trace_fs(call.fs_gpu_va) {
                        log::warn!(
                            "[bind-trace] EXDS fs={:#x} slot={} DUMMY no_pending",
                            call.fs_gpu_va,
                            slot
                        );
                    }
                    trace_vs_tex_bind_dummy(call, slot, &read_guest);
                    continue;
                };
                let cur_gen = crate::tex_invalidate::region_gen_range(tic.gpu_va, read_size as u64);
                let identity_volume =
                    key.volume && std::env::var_os("NEXIUM_VOLUME_IDENTITY").is_some();
                let volume_slices = if key.volume && !identity_volume {
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
                let raw_entry = match tex_raw_cache.entry((tic.gpu_va, read_size)) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        entry.insert(read_guest(tic.gpu_va, read_size).map(|raw| {
                            let tex_hash = hash_src_prefix(&raw);
                            (tex_hash, raw)
                        }))
                    }
                };
                let raw = raw_entry.as_ref().map(|(_, raw)| raw.as_slice());
                let raw_hash = raw_entry.as_ref().map(|(tex_hash, _)| *tex_hash);
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
                        let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                        let image_format = if identity_volume {
                            vk::Format::R8G8B8A8_UNORM
                        } else if let Some(slice) =
                            volume_slices.as_ref().and_then(|slices| slices.first())
                        {
                            slice.format
                        } else {
                            texture_image_format(tic.format, false, tic.is_srgb)
                        };
                        let texels = if volume_slices.is_some() {
                            Vec::new()
                        } else if identity_volume {
                            identity_volume_rgba8(key.width, key.height, key.layers)
                        } else if let Some(raw) = raw {
                            texture_upload_data(raw, &tic, pitch_size, force_pitch, image_format)
                        } else {
                            Vec::new()
                        };
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
                                    texels.clone()
                                } else if identity_volume {
                                    identity_volume_rgba8(key.width, key.height, key.layers)
                                } else if let Some(raw) = raw {
                                    decode_texture_rgba8_layers(raw, &tic, pitch_size, force_pitch)
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
                            key.volume,
                            &texels,
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
                if bind_trace_fs(call.fs_gpu_va) {
                    log::warn!(
                        "[bind-trace] EXDS fs={:#x} slot={} TEX va={:#x} {}x{}x{} vol={} cache_hit={} bound={:?}",
                        call.fs_gpu_va, slot, tic.gpu_va, tic.width, tic.height, key.layers,
                        key.volume, tex_cache.get(&key).is_some(),
                        if key.volume { bound_tex_views_3d[slot] } else { bound_tex_views[slot] }
                    );
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
                    if key.volume {
                        bound_tex_views_3d[slot]
                    } else {
                        bound_tex_views[slot]
                    },
                );
            }

            let vertex_binds = upload_vertex_bindings(
                device,
                frame_slots,
                other_idx,
                descriptor_pool.pool,
                ubo_ring,
                &prep.vertex_bindings,
            )?;

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
                    None => match create_sampler_for_tsc(
                        device,
                        &t,
                        *sampler_filter_minmax_supported,
                        *sampler_anisotropy_supported,
                    ) {
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
        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx].retired_dsets.extend(dsets_batch);
        for (idx, (key, _, _, _, _)) in color_bind.iter().enumerate() {
            rt_cache.set_color_layout(*key, color_layouts[idx]);
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
    let Some(base) = std::env::var_os("APPDATA") else {
        return;
    };
    if key.width == 0 || key.height == 0 {
        return;
    }
    let dir = std::path::PathBuf::from(base).join("NeXium").join("logs");
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

fn active_color_keys_for_call(call: &crate::draw::Maxwell3dDrawCall) -> Vec<RtKey> {
    if !call_writes_any_color(call) {
        Vec::new()
    } else if call.color_rt_keys.is_empty() {
        vec![call.rt_key]
    } else {
        call.color_rt_keys.clone()
    }
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
                rt_cache.record_present_flip(*key, c.flip_y);
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
    if prev_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
        transition_image(
            device,
            cmd,
            img.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            prev_layout,
        );
    }
    img.layout = prev_layout;
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

fn max_texture_descriptors() -> usize {
    crate::descriptor::MAX_TEXTURE_DESCRIPTORS as usize
}

fn bind_trace_fs(fs_gpu_va: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("NEXIUM_BIND_TRACE_FS")
            .map(|v| {
                v.split(',')
                    .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_default()
    });
    bind_trace_all() || list.contains(&fs_gpu_va)
}

fn bind_trace_all() -> bool {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALL.get_or_init(|| {
        std::env::var("NEXIUM_BIND_TRACE_FS")
            .map(|v| v.trim().eq_ignore_ascii_case("all"))
            .unwrap_or(false)
    })
}

fn tex_diag_once(fs_gpu_va: u64, tex_id: u32) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u64, u32)>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap();
    guard.get_or_insert_with(HashSet::new).insert((fs_gpu_va, tex_id))
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
    std::env::var_os("NEXIUM_VS_TEX_BIND").is_some() || bind_trace_fs(call.fs_gpu_va)
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
    let zeros = vec![0u8; size];
    let Ok(buf) = create_host_buffer(
        device,
        mem_props,
        &zeros,
        vk::BufferUsageFlags::TRANSFER_DST,
    ) else {
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

fn tic_read_size(tic: &crate::texture::TicEntry, pitch_size: usize, layers: u32) -> usize {
    if tic.is_block_linear && tic_is_volume(tic) {
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
    pitch_size: usize,
    force_pitch: bool,
) -> Vec<u8> {
    let layers = tic_layer_count(tic) as usize;
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

    let layer_read_size = tic_layer_read_size(tic, pitch_size);
    let mut out = Vec::with_capacity(layer_linear_size.saturating_mul(layers));
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_read_size);
        if start >= raw.len() {
            out.resize(out.len() + layer_linear_size, 0);
            break;
        }
        let end = (start + layer_read_size).min(raw.len());
        let layer_raw = &raw[start..end];
        let mut linear: Vec<u8> = if effective_block_linear && !force_pitch {
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
        linear.resize(layer_linear_size, 0);
        out.extend(linear);
    }
    out.resize(layer_linear_size.saturating_mul(layers), 0);
    out
}

fn decode_texture_rgba8_layers(
    raw: &[u8],
    tic: &crate::texture::TicEntry,
    pitch_size: usize,
    force_pitch: bool,
) -> Vec<u8> {
    let layers = tic_layer_count(tic) as usize;
    let linear = linear_texture_layers(raw, tic, pitch_size, force_pitch);
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
    pitch_size: usize,
    force_pitch: bool,
    format: vk::Format,
) -> Vec<u8> {
    if format == vk::Format::B10G11R11_UFLOAT_PACK32 {
        linear_texture_layers(raw, tic, pitch_size, force_pitch)
    } else {
        decode_texture_rgba8_layers(raw, tic, pitch_size, force_pitch)
    }
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
    let bpp_log2 = bytes_per_block_log2(bpp);
    let width_blocks = align_up_pow2(storage_width as u64, tic.tile_width_spacing);
    let stride_bytes = width_blocks << bpp_log2;
    let gobs_width = ceil_div_pow2(stride_bytes, 6);
    let block_height = tic.block_height_log2;
    let block_depth = tic.block_depth_log2;
    let block_size = gobs_width
        << (9u32
            .saturating_add(block_height)
            .saturating_add(block_depth));
    let slice_size = ceil_div_pow2(storage_height as u64, 3u32.saturating_add(block_height))
        .saturating_mul(block_size);
    let depth_blocks = ceil_div_pow2(layers.max(1) as u64, block_depth);
    depth_blocks.saturating_mul(slice_size) as usize
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
                if bind_trace_fs(call.fs_gpu_va) && tex_diag_once(call.fs_gpu_va, *tex_id) {
                    log::warn!(
                        "[tic] fs={:#x} tex_id={} UNBOUND -> dummy_white (pool_va={:#x} limit={})",
                        call.fs_gpu_va, tex_id, call.tic_pool_gpu_va, call.tic_pool_limit
                    );
                }
                return None;
            }
            let tic_addr = call.tic_pool_gpu_va.wrapping_add((*tex_id as u64) * 32);
            read_guest(tic_addr, 32).and_then(|tic_raw| {
                crate::texture::TicEntry::parse(&tic_raw).map(|tic| {
                    if bind_trace_fs(call.fs_gpu_va) && tex_diag_once(call.fs_gpu_va, *tex_id) {
                        let w0 = u32::from_le_bytes([tic_raw[0], tic_raw[1], tic_raw[2], tic_raw[3]]);
                        let w4 = u32::from_le_bytes([tic_raw[16], tic_raw[17], tic_raw[18], tic_raw[19]]);
                        let srgb = (w4 >> 22) & 1;
                        let (amin, amax, azero, atotal) = {
                            let sz = tic.format.linear_size(tic.width, tic.height).min(1 << 20);
                            match read_guest(tic.gpu_va, sz) {
                                Some(raw) => {
                                    let (mut mn, mut mx, mut z, mut n) = (255u8, 0u8, 0u32, 0u32);
                                    for a in raw.iter().skip(3).step_by(4) {
                                        mn = mn.min(*a);
                                        mx = mx.max(*a);
                                        if *a == 0 {
                                            z += 1;
                                        }
                                        n += 1;
                                    }
                                    (mn, mx, z, n)
                                }
                                None => (0, 0, 0, 0),
                            }
                        };
                        log::warn!(
                            "[tic] fs={:#x} tex_id={} {}x{} fmt={:?} ctypes={:?} swz={:?} bl={} bh={} gpu_va={:#x} srgb={} alpha[min={} max={} zero={}/{}] w0={:#010x}",
                            call.fs_gpu_va, tex_id, tic.width, tic.height, tic.format,
                            tic.component_types, tic.swizzle, tic.is_block_linear,
                            tic.block_height_log2, tic.gpu_va, srgb,
                            amin, amax, azero, atotal, w0
                        );
                    }
                    let pitch_size = tic.format.linear_size(tic.width, tic.height);
                    let volume = tic_is_volume(&tic);
                    let arrayed = shader_arrayed && !volume;
                    let layers = if arrayed || volume {
                        tic_layer_count(&tic)
                    } else {
                        1
                    };
                    let read_size = tic_read_size(&tic, pitch_size, layers);
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
    let found = drawn_color_alias_for_key(rt_cache, sk)
        .or_else(|| rt_cache.find_color_with_format(sk))
        .or_else(|| {
            if call.sampled_rt_fuzzy && sk.gpu_va == 0 {
                rt_cache
                    .find_color_screen(sk)
                    .map(|(key, image, view, layout)| {
                        let format = rt_cache
                            .find_color_with_format(key)
                            .map(|(_, _, _, _, format)| format)
                            .unwrap_or(vk::Format::R8G8B8A8_UNORM);
                        (key, image, view, layout, format)
                    })
            } else {
                None
            }
        });
    let found_key = found.as_ref().map(|(k, _, _, _, _)| *k);
    let filtered = found
        .filter(|(k, _, _, _, _)| allow_self || *k != rt_key)
        .map(|(key, image, view, layout, format)| RtAlias {
            key,
            image,
            view,
            layout,
            format,
            depth: false,
        })
        .or_else(|| {
            rt_cache
                .find_depth(sk)
                .or_else(|| {
                    if call.sampled_rt_fuzzy {
                        rt_cache.find_depth_fuzzy(sk)
                    } else {
                        None
                    }
                })
                .and_then(|(key, image, view, layout)| {
                    if Some(key) != call.depth_key {
                        Some(RtAlias {
                            key,
                            image,
                            view,
                            layout,
                            format: vk::Format::D32_SFLOAT,
                            depth: true,
                        })
                    } else {
                        None
                    }
                })
        });
    trace_rt_alias(
        slot,
        call,
        rt_key,
        sk,
        found_key,
        filtered.is_some(),
        call.sampled_rt_fuzzy,
    );
    filtered
}

fn drawn_color_alias_for_key(
    rt_cache: &RtCache,
    key: RtKey,
) -> Option<(RtKey, vk::Image, vk::ImageView, vk::ImageLayout, vk::Format)> {
    let found = if key.gpu_va != 0 {
        rt_cache.find_drawn_color_at(key.width, key.height, key.gpu_va)
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
        if binding.stride == 0 {
            continue;
        }
        let stride = binding.stride as u64;
        let start_vertex = if call.state.indexed {
            0
        } else {
            call.first_vertex
        };
        let vertex_span = if call.state.indexed {
            call.first_vertex.saturating_add(call.vertex_count)
        } else {
            call.vertex_count
        };
        let start_byte = stride.saturating_mul(start_vertex as u64);
        let base = binding.addr.wrapping_add(start_byte);
        let mut bytes = stride.saturating_mul(vertex_span as u64);
        if binding.size > 0 {
            if start_byte >= binding.size {
                continue;
            }
            bytes = bytes.min(binding.size - start_byte);
        }
        let bytes = bytes as usize;
        let mut data = if bytes > 0 {
            read_guest(base, bytes).ok_or_else(|| format!("vertex read failed va={:#x}", base))?
        } else {
            Vec::new()
        };
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
) -> Result<Vec<(u32, vk::Buffer, u64)>, String> {
    let mut out = Vec::with_capacity(vertex_bindings.len());
    for binding in vertex_bindings {
        if binding.data.is_empty() {
            continue;
        }
        let align = binding.stride.max(16);
        let size = align_up(binding.data.len() as u64, align);
        if ubo_ring.head + size > ubo_ring.size {
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
        volume,
        rgba8,
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

fn texture_image_format(
    format: crate::texture::TicFormat,
    from_rt_slices: bool,
    is_srgb: bool,
) -> vk::Format {
    if from_rt_slices && format == crate::texture::TicFormat::B10G11R11 {
        vk::Format::B10G11R11_UFLOAT_PACK32
    } else if is_srgb && std::env::var_os("NEXIUM_NO_TEX_SRGB").is_none() {
        vk::Format::R8G8B8A8_SRGB
    } else {
        vk::Format::R8G8B8A8_UNORM
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
        TicFormat::R16G16 => match ty {
            ComponentType::Unorm | ComponentType::UnormForceFp16 => vk::Format::R16G16_UNORM,
            ComponentType::Snorm | ComponentType::SnormForceFp16 => vk::Format::R16G16_SNORM,
            ComponentType::Uint => vk::Format::R16G16_UINT,
            ComponentType::Sint => vk::Format::R16G16_SINT,
            ComponentType::Float => vk::Format::R16G16_SFLOAT,
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
        TicFormat::B10G11R11 => vk::Format::B10G11R11_UFLOAT_PACK32,
        _ => fallback,
    }
}

fn rt_alias_sample_view(
    device: &ash::Device,
    slot: &mut FrameSlot,
    alias: RtAlias,
    swizzle: [crate::texture::SwizzleSource; 4],
    view_format: vk::Format,
) -> vk::ImageView {
    if alias.depth || (swizzle == TEXTURE_IDENTITY_SWIZZLE && view_format == alias.format) {
        return alias.view;
    }

    let view_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image: alias.image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: view_format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        components: texture_component_mapping(swizzle),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    match unsafe { device.create_image_view(&view_info, None) } {
        Ok(view) => {
            slot.retired_views.push(view);
            view
        }
        Err(e) => {
            log::warn!(
                "rt alias sample view failed key={} fmt={:?}->{:?} swz={:?}: {:?}",
                alias.key.label(),
                alias.format,
                view_format,
                swizzle,
                e
            );
            alias.view
        }
    }
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
            unsafe {
                self.device.destroy_fence(slot.fence, None);
            }
            slot.fence = vk::Fence::null();
        }
        self.utility_slot.retired_dsets.clear();
        for view in self.utility_slot.retired_views.drain(..) {
            unsafe {
                self.device.destroy_image_view(view, None);
            }
        }
        unsafe {
            self.device.destroy_fence(self.utility_slot.fence, None);
        }
        self.utility_slot.fence = vk::Fence::null();
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

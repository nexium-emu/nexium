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
    sampler_cache: HashMap<crate::texture::TscEntry, vk::Sampler>,
    tex_cache: HashMap<TexCacheKey, CachedTexture>,
    frame_slots: [FrameSlot; 2],
    frame_index: usize,
    utility_slot: FrameSlot,
    ubo_ring: UboRing,
    min_ubo_offset_alignment: u64,
    pending_readbacks: HashMap<RtKey, PendingReadback>,
    tele_last_emit_ns: u64,
    tele_ring_wraps: u64,
    tele_ring_waits: u64,
    tele_in_flight_mask: u32,
    depth_clip_control_enabled: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TexCacheKey {
    gpu_va: u64,
    width: u32,
    height: u32,
}

struct CachedTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
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
}

struct PendingReadback {
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    stage: StagingBuffer,
    width: u32,
    height: u32,
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

        let want_validation =
            std::env::var("NEXIUM_VK_VALIDATION").ok().as_deref() == Some("1");
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
        if validation_available {
            layer_ptrs.push(validation_layer.as_ptr());
            ext_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());
            log::info!("Vulkan validation layers ENABLED (guest ash instance)");
        }

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
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let instance = unsafe {
            entry.create_instance(&inst_info, None)
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
            p_enabled_features: std::ptr::null(),
            p_next: p_next_chain,
            ..Default::default()
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
        let descriptor_pool = DescriptorPool::new(&device, 1024)?;
        let shader_compiler = ShaderCompiler::new();
        let pipeline_cache = PipelineCache::new(&device, descriptor_layout.layout)?;

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
            },
            FrameSlot {
                fence: fence_b,
                cmd: frame_cmds[1],
                in_flight: false,
                retired_dsets: Vec::new(),
            },
        ];
        let utility_slot = FrameSlot {
            fence: fence_util,
            cmd: frame_cmds[2],
            in_flight: false,
            retired_dsets: Vec::new(),
        };

        let ubo_ring = create_ubo_ring(&device, &mem_props, 16 * 1024 * 1024)?;

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
                sampler_cache: HashMap::new(),
                tex_cache: HashMap::new(),
                frame_slots,
                frame_index: 0,
                utility_slot,
                ubo_ring,
                min_ubo_offset_alignment,
                pending_readbacks: HashMap::new(),
                tele_last_emit_ns: 0,
                tele_ring_wraps: 0,
                tele_ring_waits: 0,
                tele_in_flight_mask: 0,
                depth_clip_control_enabled: enable_depth_clip_control,
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
        let RendererInner { device, queue, rt_cache, utility_slot, .. } = &mut *inner;
        let key = RtKey { nvmap_id, width, height };
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
            device.begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(utility): {:?}", e))?;
        }
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
        unsafe {
            device.end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(utility): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, utility_slot.fence)?;
        wait_fence(device, utility_slot.fence)?;
        Ok(())
    }

    pub fn clear_depth(
        &self,
        nvmap_id: u32,
        width: u32,
        height: u32,
        depth: f32,
    ) -> Result<(), String> {
        let mut inner = self.inner.lock();
        let RendererInner { device, queue, rt_cache, utility_slot, .. } = &mut *inner;
        let key = RtKey { nvmap_id, width, height };
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
            device.begin_command_buffer(cmd, &begin)
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
            device.end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(depth clear): {:?}", e))?;
        }
        submit_with_fence(device, *queue, cmd, utility_slot.fence)?;
        wait_fence(device, utility_slot.fence)?;
        Ok(())
    }

    pub fn readback_target(&self, nvmap_id: u32, width: u32, height: u32) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock();
        let RendererInner {
            device, cmd_pool, queue, rt_cache, mem_props, pending_readbacks,
            frame_slots, descriptor_pool, ..
        } = &mut *inner;
        let key = RtKey { nvmap_id, width, height };
        if !cache_contains(rt_cache, key) {
            return None;
        }

        for slot in frame_slots.iter_mut() {
            if slot.in_flight {
                if wait_fence(device, slot.fence).is_err() {
                    return None;
                }
                if !slot.retired_dsets.is_empty() {
                    unsafe {
                        let _ = device.free_descriptor_sets(
                            descriptor_pool.pool, &slot.retired_dsets,
                        );
                    }
                    slot.retired_dsets.clear();
                }
                if reset_command_buffer(device, slot.cmd).is_err() {
                    return None;
                }
                slot.in_flight = false;
            }
        }

        let row_bytes = (width as u64) * 4;
        let total = row_bytes * (height as u64);

        let mut out_bytes: Option<Vec<u8>> = None;
        if let Some(prev) = pending_readbacks.remove(&key) {
            if prev.width == width && prev.height == height {
                unsafe {
                    let _ = device.wait_for_fences(&[prev.fence], true, u64::MAX);
                }
                let mut out = vec![0u8; total as usize];
                unsafe {
                    if let Ok(ptr) = device.map_memory(
                        prev.stage.memory, 0, prev.stage.size, vk::MemoryMapFlags::empty(),
                    ) {
                        std::ptr::copy_nonoverlapping(
                            ptr as *const u8, out.as_mut_ptr(), total as usize,
                        );
                        device.unmap_memory(prev.stage.memory);
                        out_bytes = Some(out);
                    }
                }
            }
            unsafe {
                device.destroy_fence(prev.fence, None);
                device.destroy_buffer(prev.stage.buffer, None);
                device.free_memory(prev.stage.memory, None);
                device.free_command_buffers(*cmd_pool, &[prev.cmd]);
            }
        }

        let stage = create_staging_owned(device, mem_props, total).ok()?;
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::empty(),
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(_) => {
                unsafe {
                    device.destroy_buffer(stage.buffer, None);
                    device.free_memory(stage.memory, None);
                }
                return out_bytes;
            }
        };
        let cmd = match alloc_one_time_cmd(device, *cmd_pool) {
            Ok(c) => c,
            Err(_) => {
                unsafe {
                    device.destroy_fence(fence, None);
                    device.destroy_buffer(stage.buffer, None);
                    device.free_memory(stage.memory, None);
                }
                return out_bytes;
            }
        };
        if begin_one_time(device, cmd).is_err() {
            unsafe {
                device.free_command_buffers(*cmd_pool, &[cmd]);
                device.destroy_fence(fence, None);
                device.destroy_buffer(stage.buffer, None);
                device.free_memory(stage.memory, None);
            }
            return out_bytes;
        }
        let img = match rt_cache.get_or_create(key, device) {
            Ok(i) => i,
            Err(_) => {
                unsafe {
                    let _ = device.end_command_buffer(cmd);
                    device.free_command_buffers(*cmd_pool, &[cmd]);
                    device.destroy_fence(fence, None);
                    device.destroy_buffer(stage.buffer, None);
                    device.free_memory(stage.memory, None);
                }
                return out_bytes;
            }
        };
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
        if end_one_time(device, cmd).is_err() {
            unsafe {
                device.free_command_buffers(*cmd_pool, &[cmd]);
                device.destroy_fence(fence, None);
                device.destroy_buffer(stage.buffer, None);
                device.free_memory(stage.memory, None);
            }
            return out_bytes;
        }
        if submit_with_fence(device, *queue, cmd, fence).is_err() {
            unsafe {
                device.free_command_buffers(*cmd_pool, &[cmd]);
                device.destroy_fence(fence, None);
                device.destroy_buffer(stage.buffer, None);
                device.free_memory(stage.memory, None);
            }
            return out_bytes;
        }
        pending_readbacks.insert(key, PendingReadback {
            fence, cmd, stage, width, height,
        });
        out_bytes
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
        blend: crate::draw::BlendState,
        cull_test_enable: bool,
        cull_face: u32,
        front_face: u32,
        poly_offset_enable: bool,
        poly_offset_units: f32,
        poly_offset_factor: f32,
        depth: crate::draw::DepthState,
        depth_format: vk::Format,
    ) -> Result<vk::Pipeline, String> {
        let mut inner = self.inner.lock();
        let blend_signature: u32 = (blend.enabled as u32)
            | ((blend.src_factor.as_raw() as u32 & 0xFF) << 8)
            | ((blend.dst_factor.as_raw() as u32 & 0xFF) << 16)
            | ((blend.op.as_raw() as u32 & 0xFF) << 24);
        let raster_state_packed: u32 = (cull_test_enable as u32)
            | ((cull_face & 0xFF) << 8)
            | ((front_face & 0xFF) << 16);
        let has_depth = depth_format != vk::Format::UNDEFINED;
        let depth_state_packed: u32 = (depth.test_enabled as u32)
            | ((depth.write_enabled as u32) << 1)
            | ((has_depth as u32) << 2)
            | ((depth.compare_op.as_raw() as u32 & 0xFF) << 8);
        let poly_offset_packed: u64 = (poly_offset_enable as u64)
            | ((poly_offset_units.to_bits() as u64) << 1)
            | ((poly_offset_factor.to_bits() as u64) << 33);
        let key = crate::pipeline::PipelineKey {
            vs_hash: hash_spirv(vs_spirv),
            fs_hash: hash_spirv(fs_spirv),
            topology: topology.as_raw() as u32,
            color_format: color_format.as_raw() as u32,
            vs_cbuf_mask,
            fs_cbuf_mask,
            vertex_layout_hash: layout.hash(),
            blend_signature,
            raster_state_packed,
            depth_state_packed,
            poly_offset_packed,
        };
        if let Some(p) = inner.pipeline_cache.get(&key) {
            return Ok(p);
        }
        let depth_clip_control_enabled = inner.depth_clip_control_enabled;
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

        let dcc_vp = vk::PipelineViewportDepthClipControlCreateInfoEXT {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
            negative_one_to_one: vk::TRUE,
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        };
        let vp_pnext: *const std::ffi::c_void = if depth_clip_control_enabled {
            &dcc_vp as *const _ as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };
        let vp_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            p_viewports: std::ptr::null(),
            scissor_count: 1,
            p_scissors: std::ptr::null(),
            p_next: vp_pnext,
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let host_front_face = match front_face {
            0x0900 => vk::FrontFace::CLOCKWISE,
            0x0901 => vk::FrontFace::COUNTER_CLOCKWISE,
            _ => vk::FrontFace::COUNTER_CLOCKWISE,
        };
        let host_cull = if !cull_test_enable {
            vk::CullModeFlags::NONE
        } else {
            match cull_face {
                0x0404 | 0x0001 => vk::CullModeFlags::FRONT,
                0x0405 | 0x0002 => vk::CullModeFlags::BACK,
                0x0408 | 0x0003 => vk::CullModeFlags::FRONT_AND_BACK,
                _ => vk::CullModeFlags::NONE,
            }
        };

        thread_local! {
            static RS_DIAG_LOGGED: std::sync::atomic::AtomicBool =
                const { std::sync::atomic::AtomicBool::new(false) };
        }
        RS_DIAG_LOGGED.with(|flag| {
            if !flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::info!(
                    "pipeline_rs_diag (first build): host_cull={:?} host_front_face={:?} \
                     guest_cull_test_enable={} guest_cull_face={:#x} guest_front_face={:#x} \
                     pipeline_key{{vs_hash=0x{:016x}, fs_hash=0x{:016x}, topology={}, \
                     raster_state_packed={:#x}}}",
                    host_cull,
                    host_front_face,
                    cull_test_enable,
                    cull_face,
                    front_face,
                    key.vs_hash,
                    key.fs_hash,
                    key.topology,
                    key.raster_state_packed,
                );
            }
        });

        let rs_state = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: host_cull,
            front_face: host_front_face,
            line_width: 1.0,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_enable: if poly_offset_enable { vk::TRUE } else { vk::FALSE },
            depth_bias_constant_factor: poly_offset_units / 2.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: poly_offset_factor,
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
            blend_enable: if blend.enabled { vk::TRUE } else { vk::FALSE },
            src_color_blend_factor: blend.src_factor,
            dst_color_blend_factor: blend.dst_factor,
            color_blend_op: blend.op,
            src_alpha_blend_factor: blend.src_factor,
            dst_alpha_blend_factor: blend.dst_factor,
            alpha_blend_op: blend.op,
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

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: if depth.test_enabled { vk::TRUE } else { vk::FALSE },
            depth_write_enable: if depth.write_enabled { vk::TRUE } else { vk::FALSE },
            depth_compare_op: depth.compare_op,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let p_depth_stencil_state: *const vk::PipelineDepthStencilStateCreateInfo =
            if has_depth { &depth_stencil_state } else { std::ptr::null() };

        let color_formats = [color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo {
            s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachment_formats: color_formats.as_ptr(),
            depth_attachment_format: depth_format,
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
            p_depth_stencil_state,
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
        let use_depth = call.depth_key.is_some();
        let depth_format = if use_depth {
            vk::Format::D32_SFLOAT
        } else {
            vk::Format::UNDEFINED
        };
        let pipeline = self.compile_pipeline(
            &call.vs_spirv,
            &call.fs_spirv,
            call.vs_cbuf_mask,
            call.fs_cbuf_mask,
            &call.vertex_layout,
            call.state.topology,
            call.rt_format,
            call.blend,
            call.cull_test_enable,
            call.cull_face,
            call.front_face,
            call.poly_offset_enable,
            call.poly_offset_units,
            call.poly_offset_factor,
            call.depth,
            depth_format,
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

        log::debug!(
            "cbuf_data: addr={:#x} size={} all_zero={}",
            call.cbuf_addr, call.cbuf_size,
            cbuf_data.iter().all(|&b| b == 0),
        );

        let tex_pending: Option<(TexCacheKey, crate::texture::TicEntry, usize, usize)> =
            if !call.fs_tex_ids.is_empty() && call.tic_pool_gpu_va != 0 {
                let tex_id = call.fs_tex_ids[0];
                if tex_id <= call.tic_pool_limit {
                    let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
                    read_guest(tic_addr, 32).and_then(|tic_raw| {
                        {
                            use std::sync::atomic::{AtomicU64, Ordering};
                            static DUMPED_MASK: AtomicU64 = AtomicU64::new(0);
                            let bit = 1u64 << ((tex_id as u64) & 63);
                            let prev = DUMPED_MASK.fetch_or(bit, Ordering::Relaxed);
                            if prev & bit == 0 && tic_raw.len() >= 32 {
                                log::debug!(
                                    "TIC_RAW id={} bytes={:02x?}",
                                    tex_id, &tic_raw[..32]
                                );
                            }
                        }
                        crate::texture::TicEntry::parse(&tic_raw).map(|tic| {
                            let bpp = tic.format.src_bpp();
                            let pitch_size = (tic.width as usize) * (tic.height as usize) * bpp;
                            let read_size = if tic.is_block_linear {
                                crate::texture::block_linear_byte_size(
                                    tic.width, tic.height, bpp, tic.block_height_log2,
                                ).max(pitch_size)
                            } else {
                                pitch_size
                            };
                            let key = TexCacheKey {
                                gpu_va: tic.gpu_va,
                                width: tic.width,
                                height: tic.height,
                            };
                            (key, tic, pitch_size, read_size)
                        })
                    })
                } else {
                    None
                }
            } else {
                None
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
                        let _ = device.free_descriptor_sets(
                            descriptor_pool.pool, &slot.retired_dsets,
                        );
                    }
                    slot.retired_dsets.clear();
                }
                reset_command_buffer(device, slot.cmd)?;
                slot.in_flight = false;
                ubo_ring.head = ubo_ring.slot_head[cur_idx];
            }
        }

        if dummy_white.is_none() {
            *dummy_white = Some(create_dummy_white_image(
                device, *queue, *cmd_pool, mem_props,
            )?);
        }
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let dummy = dummy_white.as_ref().unwrap();
        let default_samp = default_sampler.unwrap();
        let tsc: Option<crate::texture::TscEntry> =
            if !call.fs_sampler_ids.is_empty() && call.tsc_pool_gpu_va != 0 {
                let tsc_id = call.fs_sampler_ids[0];
                if tsc_id <= call.tsc_pool_limit {
                    let tsc_addr = call.tsc_pool_gpu_va.wrapping_add((tsc_id as u64) * 32);
                    read_guest(tsc_addr, 32).and_then(|r| crate::texture::TscEntry::parse(&r))
                } else { None }
            } else { None };
        let samp = match tsc {
            Some(t) => match sampler_cache.get(&t) {
                Some(s) => *s,
                None => match create_sampler_for_tsc(device, &t) {
                    Ok(s) => { sampler_cache.insert(t, s); s }
                    Err(e) => { log::warn!("tsc sampler create failed: {}", e); default_samp }
                },
            },
            None => default_samp,
        };

        let bound_tex_view: vk::ImageView = if let Some((key, tic, pitch_size, read_size)) =
            tex_pending
        {
            if !tex_cache.contains_key(&key) {
                if let Some(raw) = read_guest(tic.gpu_va, read_size) {
                    {
                        use std::collections::HashSet;
                        use std::sync::{Mutex, OnceLock};
                        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
                        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
                        let mut g = seen.lock().unwrap();
                        if g.insert(tic.gpu_va) && raw.len() >= 32 {
                            log::debug!(
                                "TIC_SRC va={:#x} {}x{} bh={} first32={:02x?}",
                                tic.gpu_va, tic.width, tic.height,
                                tic.block_height_log2, &raw[..32]
                            );
                        }
                    }
                    let bpp = tic.format.src_bpp();
                    let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH")
                        .map(|v| v == "1")
                        .unwrap_or(false);
                    let effective_block_linear =
                        tic.is_block_linear && !crate::pitch_oracle::is_pitch_dst(tic.gpu_va);
                    let linear: Vec<u8> = if effective_block_linear && !force_pitch {
                        crate::texture::unswizzle_block_linear(
                            &raw, tic.width, tic.height, bpp, tic.block_height_log2,
                        )
                    } else if raw.len() >= pitch_size {
                        raw[..pitch_size].to_vec()
                    } else {
                        raw
                    };
                    let rgba8 = crate::texture::decode_to_rgba8(
                        &linear, tic.width, tic.height, tic.format,
                    );
                    log::debug!(
                        "TIC gpu_va={:#x} {}x{} fmt={:?} bl={} bh={} src_bytes={} rgba8_bytes={} (cache miss -> upload)",
                        tic.gpu_va, tic.width, tic.height, tic.format,
                        tic.is_block_linear, tic.block_height_log2, read_size, rgba8.len()
                    );
                    match create_texture_image(
                        device, *queue, *cmd_pool, mem_props, key.width, key.height, &rgba8,
                    ) {
                        Ok(tex) => { tex_cache.insert(key, tex); }
                        Err(e) => log::warn!("texture upload failed: {}", e),
                    }
                }
            }
            tex_cache.get(&key).map(|t| t.view).unwrap_or(dummy.view)
        } else {
            dummy.view
        };

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
                            let _ = device.free_descriptor_sets(
                                descriptor_pool.pool, &other.retired_dsets,
                            );
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

        let cbuf_size_aligned = align_up(cbuf_data.len() as u64, ubo_alignment);
        {
            let v_size = if !vertex_data.is_empty() {
                let v_align = vertex_stride.max(16);
                align_up(vertex_data.len() as u64, v_align)
            } else { 0 };
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
                        let _ = device.free_descriptor_sets(
                            descriptor_pool.pool, &other.retired_dsets,
                        );
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
        let img_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: bound_tex_view,
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

        let depth_bind: Option<(vk::Image, vk::ImageView, vk::ImageLayout)> = if use_depth {
            let d = rt_cache.get_or_create_depth(call.rt_key, device)?;
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
            device.begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(slot): {:?}", e))?;
        }
        transition_image(
            device,
            cmd,
            rt_image,
            rt_prev_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
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

        let clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: call.clear_color },
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
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
            p_next: std::ptr::null(),
            _marker: std::marker::PhantomData,
        });
        let p_depth_attachment = match &depth_attachment {
            Some(a) => a as *const _,
            None => std::ptr::null(),
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
            p_depth_attachment,
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
            if let Some((vbuf, voff)) = vertex_bind {
                device.cmd_bind_vertex_buffers(cmd, 0, &[vbuf], &[voff]);
            }
            device.cmd_draw(cmd, call.vertex_count, 1, 0, 0);
            device.cmd_end_rendering(cmd);
        }

        unsafe {
            device.end_command_buffer(cmd)
                .map_err(|e| format!("end_command_buffer(slot): {:?}", e))?;
        }

        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx].retired_dsets.push(dset);

        rt_cache.get_or_create(call.rt_key, device)?.layout =
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        if use_depth {
            if let Ok(d) = rt_cache.get_or_create_depth(call.rt_key, device) {
                d.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
        }

        let next_idx = other_idx;
        ubo_ring.slot_head[next_idx] = ubo_ring.head;
        *frame_index = next_idx;

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
            tex_pending: Option<(TexCacheKey, crate::texture::TicEntry, usize, usize)>,
            use_depth: bool,
            tsc: Option<crate::texture::TscEntry>,
        }
        let mut preps: Vec<Prep> = Vec::with_capacity(calls.len());
        for call in calls {
            let use_depth = call.depth_key.is_some();
            let depth_format = if use_depth { vk::Format::D32_SFLOAT } else { vk::Format::UNDEFINED };
            let pipeline = self.compile_pipeline(
                &call.vs_spirv, &call.fs_spirv, call.vs_cbuf_mask, call.fs_cbuf_mask,
                &call.vertex_layout, call.state.topology, call.rt_format, call.blend,
                call.cull_test_enable, call.cull_face, call.front_face,
                call.poly_offset_enable, call.poly_offset_units, call.poly_offset_factor,
                call.depth, depth_format,
            )?;
            let vertex_stride = call.vertex_layout.bindings.first().map(|b| b.stride as u64).unwrap_or(0);
            let vertex_bytes = vertex_stride.saturating_mul(call.vertex_count as u64) as usize;
            let vertex_data = if vertex_bytes > 0 {
                read_guest(call.vertex_addr, vertex_bytes)
                    .ok_or_else(|| format!("vertex read failed va={:#x}", call.vertex_addr))?
            } else { Vec::new() };
            let cbuf_size = call.cbuf_size as usize;
            let cbuf_data = if cbuf_size > 0 && call.cbuf_addr != 0 {
                read_guest(call.cbuf_addr, cbuf_size).unwrap_or_else(|| vec![0u8; cbuf_size])
            } else { vec![0u8; 256] };
            let tex_pending: Option<(TexCacheKey, crate::texture::TicEntry, usize, usize)> =
                if !call.fs_tex_ids.is_empty() && call.tic_pool_gpu_va != 0 {
                    let tex_id = call.fs_tex_ids[0];
                    if tex_id <= call.tic_pool_limit {
                        let tic_addr = call.tic_pool_gpu_va.wrapping_add((tex_id as u64) * 32);
                        read_guest(tic_addr, 32).and_then(|tic_raw| {
                            crate::texture::TicEntry::parse(&tic_raw).map(|tic| {
                                let bpp = tic.format.src_bpp();
                                let pitch_size = (tic.width as usize) * (tic.height as usize) * bpp;
                                let read_size = if tic.is_block_linear {
                                    crate::texture::block_linear_byte_size(
                                        tic.width, tic.height, bpp, tic.block_height_log2,
                                    ).max(pitch_size)
                                } else { pitch_size };
                                let key = TexCacheKey { gpu_va: tic.gpu_va, width: tic.width, height: tic.height };
                                (key, tic, pitch_size, read_size)
                            })
                        })
                    } else { None }
                } else { None };
            let tsc: Option<crate::texture::TscEntry> =
                if !call.fs_sampler_ids.is_empty() && call.tsc_pool_gpu_va != 0 {
                    let tsc_id = call.fs_sampler_ids[0];
                    if tsc_id <= call.tsc_pool_limit {
                        let tsc_addr = call.tsc_pool_gpu_va.wrapping_add((tsc_id as u64) * 32);
                        read_guest(tsc_addr, 32).and_then(|r| crate::texture::TscEntry::parse(&r))
                    } else { None }
                } else { None };
            preps.push(Prep { pipeline, vertex_data, cbuf_data, vertex_stride, tex_pending, use_depth, tsc });
        }

        let mut inner = self.inner.lock();
        let RendererInner {
            device, queue, mem_props, cmd_pool, rt_cache, descriptor_layout, descriptor_pool,
            pipeline_cache, dummy_white, default_sampler, sampler_cache, tex_cache, frame_slots, frame_index,
            ubo_ring, min_ubo_offset_alignment, ..
        } = &mut *inner;

        let cur_idx = *frame_index;
        let other_idx = (cur_idx + 1) % 2;
        let ubo_alignment = *min_ubo_offset_alignment;

        {
            let slot = &mut frame_slots[cur_idx];
            if slot.in_flight {
                wait_fence(device, slot.fence)?;
                if !slot.retired_dsets.is_empty() {
                    unsafe { let _ = device.free_descriptor_sets(descriptor_pool.pool, &slot.retired_dsets); }
                    slot.retired_dsets.clear();
                }
                reset_command_buffer(device, slot.cmd)?;
                slot.in_flight = false;
                ubo_ring.head = ubo_ring.slot_head[cur_idx];
            }
        }
        if dummy_white.is_none() {
            *dummy_white = Some(create_dummy_white_image(device, *queue, *cmd_pool, mem_props)?);
        }
        if default_sampler.is_none() {
            *default_sampler = Some(create_default_sampler(device)?);
        }
        let dummy_view = dummy_white.as_ref().unwrap().view;
        let default_samp = default_sampler.unwrap();

        let rt_key = calls[0].rt_key;
        let (rt_image, rt_view, rt_extent, rt_prev_layout) = {
            let rt = rt_cache.get_or_create(rt_key, device)?;
            (rt.image, rt.view, rt.extent, rt.layout)
        };
        let any_depth = preps.iter().any(|p| p.use_depth);
        let (depth_image, depth_view, depth_prev) = if any_depth {
            let d = rt_cache.get_or_create_depth(rt_key, device)?;
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
            device.begin_command_buffer(cmd, &begin)
                .map_err(|e| format!("begin_command_buffer(batch): {:?}", e))?;
        }
        transition_image(device, cmd, rt_image, rt_prev_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        if let Some(di) = depth_image {
            transition_image_aspect(device, cmd, di, depth_prev, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageAspectFlags::DEPTH);
        }

        let mut dsets_batch: Vec<vk::DescriptorSet> = Vec::new();
        for (i, (call, prep)) in calls.iter().zip(preps.iter()).enumerate() {
            if i > 0 {
                transition_image(device, cmd, rt_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                if let Some(di) = depth_image {
                    if prep.use_depth {
                        transition_image_aspect(device, cmd, di, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageAspectFlags::DEPTH);
                    }
                }
            }

            let bound_tex_view: vk::ImageView = if let Some((key, tic, pitch_size, read_size)) = prep.tex_pending {
                if !tex_cache.contains_key(&key) {
                    if let Some(raw) = read_guest(tic.gpu_va, read_size) {
                        let bpp = tic.format.src_bpp();
                        let force_pitch = std::env::var_os("NEXIUM_FORCE_PITCH").map(|v| v == "1").unwrap_or(false);
                        let effective_block_linear =
                            tic.is_block_linear && !crate::pitch_oracle::is_pitch_dst(tic.gpu_va);
                        let linear: Vec<u8> = if effective_block_linear && !force_pitch {
                            crate::texture::unswizzle_block_linear(&raw, tic.width, tic.height, bpp, tic.block_height_log2)
                        } else if raw.len() >= pitch_size { raw[..pitch_size].to_vec() } else { raw };
                        let rgba8 = crate::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
                        if std::env::var_os("NEXIUM_TEXDUMP").map(|v| v == "1").unwrap_or(false) {
                            use std::sync::{Mutex, OnceLock};
                            static SEEN: OnceLock<Mutex<std::collections::HashSet<u64>>> = OnceLock::new();
                            let s = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                            if s.lock().unwrap().insert(tic.gpu_va) {
                                let (mut sr, mut sg, mut sb, mut sa) = (0u64, 0u64, 0u64, 0u64);
                                let (mut amin, mut amax) = (255u8, 0u8);
                                for c in rgba8.chunks_exact(4) {
                                    sr += c[0] as u64; sg += c[1] as u64; sb += c[2] as u64; sa += c[3] as u64;
                                    amin = amin.min(c[3]); amax = amax.max(c[3]);
                                }
                                let n = (rgba8.len() / 4).max(1) as u64;
                                log::warn!(
                                    "TEXDUMP va={:#x} {}x{} fmt={:?} bl={} pitchdst={} tsc={:?} avg=({},{},{},{}) a=[{}..{}]",
                                    tic.gpu_va, tic.width, tic.height, tic.format,
                                    tic.is_block_linear, crate::pitch_oracle::is_pitch_dst(tic.gpu_va),
                                    prep.tsc.map(|t| (t.wrap_u, t.mag_filter, t.min_filter)),
                                    sr / n, sg / n, sb / n, sa / n, amin, amax,
                                );
                            }
                        }
                        match create_texture_image(device, *queue, *cmd_pool, mem_props, key.width, key.height, &rgba8) {
                            Ok(tex) => { tex_cache.insert(key, tex); }
                            Err(e) => log::warn!("texture upload failed: {}", e),
                        }
                    }
                }
                tex_cache.get(&key).map(|t| t.view).unwrap_or(dummy_view)
            } else { dummy_view };

            let vertex_bind: Option<(vk::Buffer, u64)> = if !prep.vertex_data.is_empty() {
                let v_align = prep.vertex_stride.max(16);
                let v_size = align_up(prep.vertex_data.len() as u64, v_align);
                if ubo_ring.head + v_size > ubo_ring.size { ubo_ring.head = 0; }
                let (vbuf, voff, vptr) = ring_alloc(ubo_ring, v_size, v_align)
                    .map_err(|e| format!("ring_alloc(vertex): {}", e))?;
                unsafe { std::ptr::copy_nonoverlapping(prep.vertex_data.as_ptr(), vptr, prep.vertex_data.len()); }
                Some((vbuf, voff))
            } else { None };

            let cbuf_size_aligned = align_up(prep.cbuf_data.len() as u64, ubo_alignment);
            if ubo_ring.head + cbuf_size_aligned > ubo_ring.size { ubo_ring.head = 0; }
            let (ubo_buffer, ubo_offset, ubo_ptr) = ring_alloc(ubo_ring, cbuf_size_aligned, ubo_alignment)
                .map_err(|e| format!("ring_alloc(ubo): {}", e))?;
            unsafe { std::ptr::copy_nonoverlapping(prep.cbuf_data.as_ptr(), ubo_ptr, prep.cbuf_data.len()); }

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
                device.allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| format!("allocate_descriptor_sets: {:?}", e))?[0]
            };
            let ubo_info = vk::DescriptorBufferInfo { buffer: ubo_buffer, offset: ubo_offset, range: prep.cbuf_data.len() as u64 };
            let img_info = vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: bound_tex_view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL };
            let samp = match prep.tsc {
                Some(t) => match sampler_cache.get(&t) {
                    Some(s) => *s,
                    None => match create_sampler_for_tsc(device, &t) {
                        Ok(s) => { sampler_cache.insert(t, s); s }
                        Err(e) => { log::warn!("tsc sampler create failed: {}", e); default_samp }
                    },
                },
                None => default_samp,
            };
            let samp_info = vk::DescriptorImageInfo { sampler: samp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED };
            let writes = [
                vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: dset, dst_binding: 0, dst_array_element: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::UNIFORM_BUFFER, p_buffer_info: &ubo_info, p_image_info: std::ptr::null(), p_texel_buffer_view: std::ptr::null(), p_next: std::ptr::null(), _marker: std::marker::PhantomData },
                vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: dset, dst_binding: 1, dst_array_element: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::SAMPLED_IMAGE, p_image_info: &img_info, p_buffer_info: std::ptr::null(), p_texel_buffer_view: std::ptr::null(), p_next: std::ptr::null(), _marker: std::marker::PhantomData },
                vk::WriteDescriptorSet { s_type: vk::StructureType::WRITE_DESCRIPTOR_SET, dst_set: dset, dst_binding: 2, dst_array_element: 0, descriptor_count: 1, descriptor_type: vk::DescriptorType::SAMPLER, p_image_info: &samp_info, p_buffer_info: std::ptr::null(), p_texel_buffer_view: std::ptr::null(), p_next: std::ptr::null(), _marker: std::marker::PhantomData },
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]); }
            dsets_batch.push(dset);

            let clear_value = vk::ClearValue { color: vk::ClearColorValue { float32: call.clear_color } };
            let depth_attachment = if prep.use_depth {
                depth_view.map(|dv| vk::RenderingAttachmentInfo {
                    s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO, image_view: dv,
                    image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE, resolve_image_view: vk::ImageView::null(),
                    resolve_image_layout: vk::ImageLayout::UNDEFINED, load_op: vk::AttachmentLoadOp::LOAD,
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
                    p_next: std::ptr::null(), _marker: std::marker::PhantomData,
                })
            } else { None };
            let p_depth_attachment = match &depth_attachment { Some(a) => a as *const _, None => std::ptr::null() };
            let attachment = vk::RenderingAttachmentInfo {
                s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO, image_view: rt_view,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, resolve_mode: vk::ResolveModeFlags::NONE,
                resolve_image_view: vk::ImageView::null(), resolve_image_layout: vk::ImageLayout::UNDEFINED,
                load_op: vk::AttachmentLoadOp::LOAD, store_op: vk::AttachmentStoreOp::STORE, clear_value,
                p_next: std::ptr::null(), _marker: std::marker::PhantomData,
            };
            let render_info = vk::RenderingInfo {
                s_type: vk::StructureType::RENDERING_INFO,
                render_area: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: rt_extent },
                layer_count: 1, view_mask: 0, color_attachment_count: 1, p_color_attachments: &attachment,
                p_depth_attachment, p_stencil_attachment: std::ptr::null(), p_next: std::ptr::null(),
                flags: Default::default(), _marker: std::marker::PhantomData,
            };
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: rt_extent.width as f32, height: rt_extent.height as f32, min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: rt_extent };
            unsafe {
                device.cmd_begin_rendering(cmd, &render_info);
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_set_scissor(cmd, 0, &[scissor]);
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, prep.pipeline);
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_cache.layout, 0, &[dset], &[]);
                if let Some((vbuf, voff)) = vertex_bind { device.cmd_bind_vertex_buffers(cmd, 0, &[vbuf], &[voff]); }
                device.cmd_draw(cmd, call.vertex_count, 1, 0, 0);
                device.cmd_end_rendering(cmd);
            }
        }

        unsafe {
            device.end_command_buffer(cmd).map_err(|e| format!("end_command_buffer(batch): {:?}", e))?;
        }
        let slot_fence = frame_slots[cur_idx].fence;
        submit_with_fence(device, *queue, cmd, slot_fence)?;
        frame_slots[cur_idx].in_flight = true;
        frame_slots[cur_idx].retired_dsets.extend(dsets_batch);
        rt_cache.get_or_create(rt_key, device)?.layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        if any_depth {
            if let Ok(d) = rt_cache.get_or_create_depth(rt_key, device) {
                d.layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
        }
        let next_idx = other_idx;
        ubo_ring.slot_head[next_idx] = ubo_ring.head;
        *frame_index = next_idx;
        Ok(())
    }
}

fn monotonic_nanos() -> u64 {
    use std::time::Instant;
    use std::sync::OnceLock;
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
        usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
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

fn create_texture_image(
    device: &ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    rgba8: &[u8],
) -> Result<CachedTexture, String> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let img_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D { width, height, depth: 1 },
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
        device.create_image(&img_info, None)
            .map_err(|e| format!("create_image(tex {}x{}): {:?}", width, height, e))?
    };
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mt = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ).ok_or_else(|| "no DEVICE_LOCAL for texture image".to_string())?;
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mt,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let memory = unsafe {
        device.allocate_memory(&alloc, None)
            .map_err(|e| format!("allocate_memory(tex): {:?}", e))?
    };
    unsafe {
        device.bind_image_memory(image, memory, 0)
            .map_err(|e| format!("bind_image_memory(tex): {:?}", e))?;
    }

    let stage = create_host_buffer(device, mem_props, rgba8, vk::BufferUsageFlags::TRANSFER_SRC)?;

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
        image_extent: vk::Extent3D { width, height, depth: 1 },
    };
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd, stage.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[copy],
        );
    }
    transition_image(
        device, cmd, image,
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
        device.create_image_view(&view_info, None)
            .map_err(|e| format!("create_image_view(tex): {:?}", e))?
    };
    Ok(CachedTexture { image, view, memory })
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
        device.reset_fences(&[fence])
            .map_err(|e| format!("reset_fences(submit): {:?}", e))?;
        device.queue_submit(queue, &[submit], fence)
            .map_err(|e| format!("queue_submit(fence): {:?}", e))?;
    }
    Ok(())
}

fn wait_fence(device: &ash::Device, fence: vk::Fence) -> Result<(), String> {
    unsafe {
        device.wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("wait_for_fences: {:?}", e))?;
        device.reset_fences(&[fence])
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
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
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
    let end = aligned_head.checked_add(size).ok_or("ring_alloc: overflow")?;
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
        (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ) => (
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
        device.create_buffer(&buf_info, None)
            .map_err(|e| format!("create_buffer(readback): {:?}", e))?
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
            .map_err(|e| format!("allocate_memory(readback staging): {:?}", e))?
    };
    unsafe {
        device.bind_buffer_memory(buffer, memory, 0)
            .map_err(|e| format!("bind_buffer_memory(readback): {:?}", e))?;
    }
    Ok(StagingBuffer { buffer, memory, size: req.size })
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
        for slot in self.frame_slots.iter_mut() {
            slot.retired_dsets.clear();
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
        for (_, pr) in self.pending_readbacks.drain() {
            unsafe {
                let _ = self.device.wait_for_fences(&[pr.fence], true, u64::MAX);
                self.device.destroy_fence(pr.fence, None);
                self.device.destroy_buffer(pr.stage.buffer, None);
                self.device.free_memory(pr.stage.memory, None);
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

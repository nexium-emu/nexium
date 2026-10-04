use std::time::Instant;

use ash::vk;

use crate::display::{free_host_buffer, host_buffer, name_of, spv, vkerr, write_bmp, Display};

pub type Check = fn() -> Result<String, String>;

pub const CHECKS: &[(&str, Check)] = &[("vulkan-display", vulkan_display)];

const FRAMES: u32 = 360;
const CAPTURE_FRAME: u32 = 300;

pub fn report_device(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    unsafe {
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver).push_next(&mut p11);
        instance.get_physical_device_properties2(pd, &mut props2);
        let p = props2.properties;
        let api = p.api_version;
        lines.push(format!(
            "device '{}' api {}.{}.{} driver '{}' '{}' vendor {:#06x} device {:#06x} conformance {}.{}.{}.{}",
            name_of(&p.device_name),
            vk::api_version_major(api),
            vk::api_version_minor(api),
            vk::api_version_patch(api),
            name_of(&driver.driver_name),
            name_of(&driver.driver_info),
            p.vendor_id,
            p.device_id,
            driver.conformance_version.major,
            driver.conformance_version.minor,
            driver.conformance_version.subminor,
            driver.conformance_version.patch
        ));
        let l = p.limits;
        lines.push(format!(
            "limits image2D={} storageBufferRange={} pushConstants={} computeShared={} sets={} minSSBOAlign={} anisotropy={} timestampPeriod={} subgroup={}",
            l.max_image_dimension2_d,
            l.max_storage_buffer_range,
            l.max_push_constants_size,
            l.max_compute_shared_memory_size,
            l.max_bound_descriptor_sets,
            l.min_storage_buffer_offset_alignment,
            l.max_sampler_anisotropy,
            l.timestamp_period,
            p11.subgroup_size
        ));

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
        instance.get_physical_device_features2(pd, &mut f2);
        let f = f2.features;
        let flag = |b: vk::Bool32| if b == vk::TRUE { "y" } else { "N" };
        let features = [
            ("dynamicRendering", f13.dynamic_rendering),
            ("synchronization2", f13.synchronization2),
            ("subgroupSizeControl", f13.subgroup_size_control),
            ("maintenance4", f13.maintenance4),
            ("timelineSemaphore", f12.timeline_semaphore),
            ("samplerFilterMinmax", f12.sampler_filter_minmax),
            ("shaderOutputLayer", f12.shader_output_layer),
            ("shaderOutputViewportIndex", f12.shader_output_viewport_index),
            ("uniformBufferStandardLayout", f12.uniform_buffer_standard_layout),
            ("descriptorBindingPartiallyBound", f12.descriptor_binding_partially_bound),
            ("descriptorIndexing", f12.descriptor_indexing),
            ("bufferDeviceAddress", f12.buffer_device_address),
            ("scalarBlockLayout", f12.scalar_block_layout),
            ("shaderFloat16", f12.shader_float16),
            ("shaderInt8", f12.shader_int8),
            ("storageBuffer8BitAccess", f12.storage_buffer8_bit_access),
            ("vulkanMemoryModel", f12.vulkan_memory_model),
            ("shaderDrawParameters", f11.shader_draw_parameters),
            ("storageBuffer16BitAccess", f11.storage_buffer16_bit_access),
            ("robustBufferAccess", f.robust_buffer_access),
            ("depthClamp", f.depth_clamp),
            ("depthBounds", f.depth_bounds),
            ("independentBlend", f.independent_blend),
            ("dualSrcBlend", f.dual_src_blend),
            ("logicOp", f.logic_op),
            ("samplerAnisotropy", f.sampler_anisotropy),
            ("imageCubeArray", f.image_cube_array),
            ("geometryShader", f.geometry_shader),
            ("tessellationShader", f.tessellation_shader),
            ("multiViewport", f.multi_viewport),
            ("wideLines", f.wide_lines),
            ("largePoints", f.large_points),
            ("fillModeNonSolid", f.fill_mode_non_solid),
            ("shaderFloat64", f.shader_float64),
            ("shaderInt64", f.shader_int64),
            ("shaderInt16", f.shader_int16),
            ("textureCompressionBC", f.texture_compression_bc),
            ("textureCompressionASTC_LDR", f.texture_compression_astc_ldr),
            ("storageImageExtendedFormats", f.shader_storage_image_extended_formats),
            ("storageImageReadWithoutFormat", f.shader_storage_image_read_without_format),
            ("storageImageWriteWithoutFormat", f.shader_storage_image_write_without_format),
            ("fragmentStoresAndAtomics", f.fragment_stores_and_atomics),
            ("vertexPipelineStoresAndAtomics", f.vertex_pipeline_stores_and_atomics),
            ("shaderClipDistance", f.shader_clip_distance),
            ("shaderCullDistance", f.shader_cull_distance),
            ("sampleRateShading", f.sample_rate_shading),
            ("occlusionQueryPrecise", f.occlusion_query_precise),
            ("pipelineStatisticsQuery", f.pipeline_statistics_query),
        ];
        lines.push(format!(
            "features {}",
            features.iter().map(|(n, v)| format!("{n}={}", flag(*v))).collect::<Vec<_>>().join(" ")
        ));
        let required = [
            ("dynamicRendering", f13.dynamic_rendering),
            ("synchronization2", f13.synchronization2),
            ("timelineSemaphore", f12.timeline_semaphore),
            ("robustBufferAccess", f.robust_buffer_access),
        ];
        let missing: Vec<_> = required.iter().filter(|(_, v)| *v != vk::TRUE).map(|(n, _)| *n).collect();
        if !missing.is_empty() {
            return Err(format!("required features missing: {missing:?}"));
        }

        let exts: Vec<String> = instance
            .enumerate_device_extension_properties(pd)
            .map_err(vkerr("enumerate_device_extension_properties"))?
            .iter()
            .map(|e| name_of(&e.extension_name))
            .collect();
        let wanted = [
            "VK_KHR_swapchain",
            "VK_EXT_swapchain_maintenance1",
            "VK_EXT_depth_clip_control",
            "VK_KHR_workgroup_memory_explicit_layout",
            "VK_EXT_sampler_filter_minmax",
            "VK_KHR_vertex_attribute_divisor",
            "VK_EXT_vertex_attribute_divisor",
            "VK_EXT_custom_border_color",
            "VK_EXT_robustness2",
            "VK_EXT_extended_dynamic_state3",
            "VK_EXT_provoking_vertex",
            "VK_EXT_transform_feedback",
            "VK_EXT_line_rasterization",
            "VK_EXT_conditional_rendering",
            "VK_KHR_push_descriptor",
            "VK_EXT_shader_stencil_export",
            "VK_EXT_depth_range_unrestricted",
            "VK_EXT_shader_viewport_index_layer",
            "VK_KHR_shader_float_controls",
            "VK_EXT_image_view_min_lod",
        ];
        lines.push(format!(
            "device extensions ({} total): {}",
            exts.len(),
            wanted
                .iter()
                .map(|w| format!("{w}={}", if exts.iter().any(|e| e == w) { "y" } else { "N" }))
                .collect::<Vec<_>>()
                .join(" ")
        ));

        let formats = [
            ("RGBA8", vk::Format::R8G8B8A8_UNORM),
            ("RGBA8_SRGB", vk::Format::R8G8B8A8_SRGB),
            ("BGRA8", vk::Format::B8G8R8A8_UNORM),
            ("A2B10G10R10", vk::Format::A2B10G10R10_UNORM_PACK32),
            ("RGBA16F", vk::Format::R16G16B16A16_SFLOAT),
            ("RGBA32F", vk::Format::R32G32B32A32_SFLOAT),
            ("B10G11R11F", vk::Format::B10G11R11_UFLOAT_PACK32),
            ("E5B9G9R9", vk::Format::E5B9G9R9_UFLOAT_PACK32),
            ("R5G6B5", vk::Format::R5G6B5_UNORM_PACK16),
            ("A1R5G5B5", vk::Format::A1R5G5B5_UNORM_PACK16),
            ("R8", vk::Format::R8_UNORM),
            ("RG8", vk::Format::R8G8_UNORM),
            ("R16", vk::Format::R16_UNORM),
            ("R32UI", vk::Format::R32_UINT),
            ("BC1", vk::Format::BC1_RGBA_UNORM_BLOCK),
            ("BC2", vk::Format::BC2_UNORM_BLOCK),
            ("BC3", vk::Format::BC3_UNORM_BLOCK),
            ("BC4", vk::Format::BC4_UNORM_BLOCK),
            ("BC5", vk::Format::BC5_UNORM_BLOCK),
            ("BC6H", vk::Format::BC6H_UFLOAT_BLOCK),
            ("BC7", vk::Format::BC7_UNORM_BLOCK),
            ("ASTC4x4", vk::Format::ASTC_4X4_UNORM_BLOCK),
            ("D16", vk::Format::D16_UNORM),
            ("D24S8", vk::Format::D24_UNORM_S8_UINT),
            ("D32F", vk::Format::D32_SFLOAT),
            ("D32FS8", vk::Format::D32_SFLOAT_S8_UINT),
            ("S8", vk::Format::S8_UINT),
        ];
        let mut fl = Vec::new();
        for (n, fmt) in formats {
            let fp = instance.get_physical_device_format_properties(pd, fmt);
            let o = fp.optimal_tiling_features;
            let mut s = String::new();
            for (c, bit) in [
                ('s', vk::FormatFeatureFlags::SAMPLED_IMAGE),
                ('l', vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
                ('c', vk::FormatFeatureFlags::COLOR_ATTACHMENT),
                ('b', vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
                ('d', vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
                ('w', vk::FormatFeatureFlags::STORAGE_IMAGE),
                ('t', vk::FormatFeatureFlags::BLIT_DST),
            ] {
                if o.contains(bit) {
                    s.push(c);
                }
            }
            fl.push(format!("{n}={}", if s.is_empty() { "-".to_string() } else { s }));
        }
        lines.push(format!("formats (s=sampled l=linear c=color b=blend d=depth w=storage t=blit-dst) {}", fl.join(" ")));

        let mp = instance.get_physical_device_memory_properties(pd);
        let heaps: Vec<String> = (0..mp.memory_heap_count as usize)
            .map(|i| format!("heap{i}={}MiB{:?}", mp.memory_heaps[i].size >> 20, mp.memory_heaps[i].flags))
            .collect();
        let types: Vec<String> = (0..mp.memory_type_count as usize)
            .map(|i| format!("t{i}:h{}:{:?}", mp.memory_types[i].heap_index, mp.memory_types[i].property_flags))
            .collect();
        lines.push(format!("memory {} | {}", heaps.join(" "), types.join(" ")));
    }
    Ok(lines)
}

fn vulkan_display() -> Result<String, String> {
    let mut display = Display::new(true)?;
    let compute_result = unsafe { compute_test(&display.device, display.queue, display.family, &display.memory_props) };
    let loop_started = Instant::now();
    let mut last = Instant::now();
    let mut intervals = Vec::with_capacity(FRAMES as usize);
    let mut capture_ok = false;
    for frame in 0..FRAMES {
        let push = [frame as f32 / 30.0, display.extent.width as f32, display.extent.height as f32, frame as f32, 0.0, 0.0, 0.0, 0.0];
        let cap = display.frame(push, frame == CAPTURE_FRAME)?;
        let now = Instant::now();
        intervals.push(now.duration_since(last).as_secs_f64() * 1000.0);
        last = now;
        if let Some(cap) = cap {
            let centre = (cap.height as usize / 2 * cap.width as usize + cap.width as usize / 2) * 4;
            capture_ok = cap.bytes[centre..centre + 3].iter().all(|&b| b > 240);
            let path = format!("{}/vk-probe.bmp", crate::console::DATA_ROOT);
            match write_bmp(&path, &cap, 4) {
                Ok(()) => crate::klog!("vk capture frame {frame}: {path} (centre bar white={capture_ok}, corner={:?})", &cap.bytes[0..4]),
                Err(e) => crate::klog!("vk capture write failed: {e}"),
            }
        }
    }
    let total = loop_started.elapsed().as_secs_f64();
    let steady = &intervals[10..];
    let avg = steady.iter().sum::<f64>() / steady.len() as f64;
    let max = steady.iter().cloned().fold(0.0, f64::max);
    let min = steady.iter().cloned().fold(f64::MAX, f64::min);
    let detail = [
        format!("init={:.0}ms pipeline={:.1}ms", display.init_ms, display.pipeline_ms),
        format!(
            "{FRAMES} frames {}x{} {:?} in {total:.2}s: avg {avg:.2}ms ({:.2} fps) min {min:.2} max {max:.2}",
            display.extent.width,
            display.extent.height,
            display.format,
            1000.0 / avg
        ),
        compute_result.clone().unwrap_or_else(|e| format!("compute FAILED: {e}")),
    ]
    .join("; ");
    drop(display);
    if compute_result.is_err() || !capture_ok {
        return Err(detail);
    }
    Ok(detail)
}

unsafe fn compute_test(
    device: &ash::Device,
    queue: vk::Queue,
    family: u32,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
) -> Result<String, String> {
    unsafe {
        const COUNT: u64 = 4096;
        let buffer = host_buffer(device, memory_props, COUNT * 4, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        std::ptr::write_bytes(buffer.ptr, 0, (COUNT * 4) as usize);
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let set_layout = device
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding), None)
            .map_err(vkerr("set layout"))?;
        let set_layouts = [set_layout];
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
            .map_err(vkerr("compute layout"))?;
        let code = spv(include_bytes!(concat!(env!("OUT_DIR"), "/probe_cs_main.spv")));
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(vkerr("cs"))?;
        let started = Instant::now();
        let pipeline = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"cs_main"))
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| format!("compute pipeline: {e:?}"))?[0];
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let pool_sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
        let dpool = device
            .create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes), None)
            .map_err(vkerr("descriptor pool"))?;
        let set = device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dpool).set_layouts(&set_layouts))
            .map_err(vkerr("descriptor set"))?[0];
        let info = [vk::DescriptorBufferInfo { buffer: buffer.buffer, offset: 0, range: vk::WHOLE_SIZE }];
        device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info)],
            &[],
        );
        let pool = device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None)
            .map_err(vkerr("compute pool"))?;
        let cmd = device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))
            .map_err(vkerr("compute cmd"))?[0];
        device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(vkerr("begin"))?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
        device.cmd_dispatch(cmd, (COUNT / 64) as u32, 1, 1);
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
        device.end_command_buffer(cmd).map_err(vkerr("end"))?;
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("fence"))?;
        let cmds = [cmd];
        let started = Instant::now();
        device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).map_err(vkerr("submit"))?;
        device.wait_for_fences(&[fence], true, 5_000_000_000).map_err(vkerr("compute wait"))?;
        let run_us = started.elapsed().as_secs_f64() * 1e6;
        let values = std::slice::from_raw_parts(buffer.ptr as *const u32, COUNT as usize);
        let bad = values.iter().enumerate().find(|(i, &v)| v != *i as u32 * 2 + 7).map(|(i, &v)| (i, v));
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_descriptor_pool(dpool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        free_host_buffer(device, &buffer);
        match bad {
            Some((i, v)) => Err(format!("value[{i}]={v}, expected {}", i * 2 + 7)),
            None => Ok(format!("compute {COUNT} values ok (compile {compile_ms:.1}ms, submit+wait {run_us:.0}us)")),
        }
    }
}

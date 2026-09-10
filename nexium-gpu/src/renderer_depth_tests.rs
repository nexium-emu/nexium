use super::*;

struct Logger;
static ERRORS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

impl log::Log for Logger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        eprintln!("{} {}", record.level(), record.args());
        if record.level() == log::Level::Error {
            ERRORS.lock().unwrap().push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

#[test]
#[ignore = "requires a Vulkan device"]
fn depth_fallback_preserves_clears_stencil_aliases_and_quantization() {
    let _ = log::set_logger(&Logger);
    log::set_max_level(log::LevelFilter::Info);
    if let Ok(name) = std::env::var("NEXIUM_TEST_GPU") {
        let adapter = crate::adapter::available_devices()
            .iter()
            .find(|adapter| adapter.name == name)
            .expect("test GPU");
        crate::adapter::set_preferred_device(Some(adapter.id.clone()));
    }
    let renderer = Renderer::new().unwrap();
    for force_fallback in [false, true] {
        {
            let mut inner = renderer.inner.lock();
            let device = inner.device.clone();
            inner.rt_cache.clear(&device);
            if force_fallback {
                inner
                    .rt_cache
                    .set_depth_formats(crate::depth::DepthFormats::float_fallback());
            }
        }
        let key = RtKey::new(900, 65, 3, 0x12340000);
        let aspects = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
        for depth in [0.0, 0.25, 0.5, 0.75, 1.0] {
            renderer
                .clear_depth_stencil(
                    key,
                    vk::Format::D24_UNORM_S8_UINT,
                    aspects,
                    aspects,
                    depth,
                    0xab,
                )
                .unwrap();
            let (_, _, bpp, bytes) = renderer
                .readback_depth_target_raw(key.nvmap_id, key.width, key.height, key.gpu_va)
                .unwrap();
            assert_eq!(bpp, 4);
            for pixel in bytes.chunks_exact(4) {
                let value = u32::from_le_bytes(pixel.try_into().unwrap()) & 0xffffff;
                assert!(value.abs_diff(crate::depth::pack_d24(depth)) <= 1);
            }
        }
        let mut inner = renderer.inner.lock();
        let RendererInner {
            device,
            rt_cache,
            mem_props,
            cmd_pool,
            queue,
            descriptor_pool,
            frame_slots,
            ..
        } = &mut *inner;
        let (cached, fresh) = rt_cache
            .get_or_create_depth(key, device, vk::Format::D24_UNORM_S8_UINT, aspects)
            .unwrap();
        assert!(!fresh);
        assert_eq!(cached.base_format, vk::Format::D24_UNORM_S8_UINT);
        if force_fallback {
            assert_eq!(cached.format, vk::Format::D32_SFLOAT_S8_UINT);
        }
        let image = cached.image;
        assert!(!rt_cache.depth_requires_recreate(key, vk::Format::D24_UNORM_S8_UINT, aspects));
        assert!(rt_cache.depth_requires_recreate(key, vk::Format::D32_SFLOAT_S8_UINT, aspects));
        assert!(rt_cache
            .find_d24_depth_covering(RtKey { width: 64, ..key })
            .is_none());
        assert!(rt_cache.find_d24_depth_covering(key).is_some());
        let cmd = alloc_one_time_cmd(device, *cmd_pool).unwrap();
        begin_one_time(device, cmd).unwrap();
        let alias = sync_sampled_depth_as_color(
            device,
            cmd,
            rt_cache,
            mem_props,
            descriptor_pool.pool,
            &mut frame_slots[0],
            key,
        )
        .unwrap()
        .unwrap();
        let count = u64::from(key.width) * u64::from(key.height);
        let staging = create_staging_owned(device, mem_props, count * 5).unwrap();
        transition_image_aspect(
            device,
            cmd,
            image,
            rt_cache.depth_layout(key).unwrap(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            aspects,
        );
        transition_image(
            device,
            cmd,
            alias.image,
            alias.layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            });
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                alias.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.buffer,
                &[region],
            );
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.buffer,
                &[region.buffer_offset(count * 4).image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::STENCIL)
                        .layer_count(1),
                )],
            );
        }
        submit_test(device, *queue, cmd);
        let bytes = read_test_buffer(device, &staging, count as usize * 5);
        for pixel in bytes[..count as usize * 4].chunks_exact(4) {
            assert_eq!(
                u32::from_le_bytes(pixel.try_into().unwrap()) & 0xffffff,
                0xffffff
            );
        }
        assert!(bytes[count as usize * 4..]
            .iter()
            .all(|&stencil| stencil == 0xab));
        rt_cache.set_depth_layout(key, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        rt_cache.set_color_layout(alias.key, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        unsafe {
            device.destroy_buffer(staging.buffer, None);
            device.free_memory(staging.memory, None);
            device.free_command_buffers(*cmd_pool, &[cmd]);
        }
    }
    {
        let mut inner = renderer.inner.lock();
        let RendererInner {
            device,
            rt_cache,
            mem_props,
            cmd_pool,
            queue,
            descriptor_pool,
            ..
        } = &mut *inner;
        let count = 4_194_307usize;
        let mut random = 0x12345678u32;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            values.push((random >> 8) as f32 / 16_777_215.0);
        }
        values[..9].copy_from_slice(&[
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]);
        let source: Vec<_> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let upload = create_host_buffer(
            device,
            mem_props,
            &source,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .unwrap();
        let bytes = source.len() as u64;
        let transfer = create_transfer_buffer_with_usage(
            device,
            mem_props,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .unwrap();
        let readback = create_staging_owned(device, mem_props, bytes).unwrap();
        let cmd = alloc_one_time_cmd(device, *cmd_pool).unwrap();
        begin_one_time(device, cmd).unwrap();
        let copy = [vk::BufferCopy::default().size(bytes)];
        unsafe { device.cmd_copy_buffer(cmd, upload.buffer, transfer.buffer, &copy) };
        let set = rt_cache
            .pack_depth_buffer(device, cmd, descriptor_pool.pool, transfer.buffer, bytes)
            .unwrap();
        unsafe { device.cmd_copy_buffer(cmd, transfer.buffer, readback.buffer, &copy) };
        submit_test(device, *queue, cmd);
        let output = read_test_buffer(device, &readback, bytes as usize);
        for (index, (pixel, value)) in output.chunks_exact(4).zip(values).enumerate() {
            assert_eq!(
                u32::from_le_bytes(pixel.try_into().unwrap()),
                crate::depth::pack_d24(value),
                "texel {index}: {value}"
            );
        }
        unsafe {
            device
                .free_descriptor_sets(descriptor_pool.pool, &[set])
                .unwrap();
            device.free_command_buffers(*cmd_pool, &[cmd]);
            for (buffer, memory) in [
                (upload.buffer, upload.memory),
                (transfer.buffer, transfer.memory),
                (readback.buffer, readback.memory),
            ] {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
        }
    }
    drop(renderer);
    let errors = ERRORS.lock().unwrap();
    assert!(errors.is_empty(), "{errors:?}");
}

fn submit_test(device: &ash::Device, queue: vk::Queue, cmd: vk::CommandBuffer) {
    end_one_time(device, cmd).unwrap();
    let commands = [cmd];
    let info = [vk::SubmitInfo::default().command_buffers(&commands)];
    unsafe {
        device
            .queue_submit(queue, &info, vk::Fence::null())
            .unwrap();
        device.queue_wait_idle(queue).unwrap();
    }
}

fn read_test_buffer(device: &ash::Device, buffer: &StagingBuffer, bytes: usize) -> Vec<u8> {
    unsafe {
        let ptr = device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
            .unwrap();
        let result = std::slice::from_raw_parts(ptr as *const u8, bytes).to_vec();
        device.unmap_memory(buffer.memory);
        result
    }
}

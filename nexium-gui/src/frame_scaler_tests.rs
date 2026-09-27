use super::*;
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

const FILTERS: [ScalingFilter; 5] = [
    ScalingFilter::Nearest,
    ScalingFilter::Linear,
    ScalingFilter::Bicubic,
    ScalingFilter::ScaleForce,
    ScalingFilter::Fsr,
];

struct ThreadWake(std::thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn blocking_future<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(!remaining.is_zero(), "GPU setup or validation timed out");
        std::thread::park_timeout(remaining);
    }
}

fn render_state() -> RenderState {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::VULKAN;
    descriptor.flags |= wgpu::InstanceFlags::VALIDATION;
    let instance = wgpu::Instance::new(descriptor);
    let adapter =
        blocking_future(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("these ignored tests require a local Vulkan adapter");
    let (device, queue) =
        blocking_future(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("create default-feature Vulkan device");
    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = eframe::egui_wgpu::Renderer::new(&device, target_format, Default::default());
    RenderState {
        available_adapters: vec![adapter.clone()],
        adapter,
        instance,
        device,
        queue,
        target_format,
        renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
        surface_config: eframe::egui_wgpu::SurfaceConfig::LOW_LATENCY,
    }
}

fn source_texture(rs: &RenderState, size: [u32; 2]) -> wgpu::Texture {
    rs.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scaler_test_source"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn upload(rs: &RenderState, texture: &wgpu::Texture, pixels: &[[u8; 4]]) {
    assert_eq!(pixels.len(), (texture.width() * texture.height()) as usize);
    let bytes: Vec<_> = pixels.iter().flatten().copied().collect();
    rs.queue.write_texture(
        texture.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width() * 4),
            rows_per_image: Some(texture.height()),
        },
        texture.size(),
    );
}

fn scaler(rs: &RenderState) -> FrameScaler {
    let scope = rs.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let scaler = FrameScaler::new(&rs.device);
    let error = blocking_future(scope.pop());
    assert!(
        error.is_none(),
        "scaler shader or pipeline validation failed: {error:?}"
    );
    scaler
}

fn readback(rs: &RenderState, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
    let stride = (texture.width() * 4).div_ceil(256) * 256;
    let buffer = rs.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scaler_test_readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = rs.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    let submission = rs.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    rs.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .expect("wait for scaler readback");
    receiver
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let pixels = mapped
        .chunks_exact(stride as usize)
        .flat_map(|row| row[..texture.width() as usize * 4].chunks_exact(4))
        .map(|pixel| pixel.try_into().unwrap())
        .collect();
    drop(mapped);
    buffer.unmap();
    pixels
}

fn render(
    rs: &RenderState,
    scaler: &mut FrameScaler,
    source: &wgpu::Texture,
    size: [u32; 2],
    filter: ScalingFilter,
    dirty: bool,
) -> Vec<[u8; 4]> {
    render_with_sharpness(rs, scaler, source, size, filter, 87, dirty)
}

fn render_with_sharpness(
    rs: &RenderState,
    scaler: &mut FrameScaler,
    source: &wgpu::Texture,
    size: [u32; 2],
    filter: ScalingFilter,
    sharpness: u8,
    dirty: bool,
) -> Vec<[u8; 4]> {
    let scope = rs.device.push_error_scope(wgpu::ErrorFilter::Validation);
    scaler.render(rs, source, size, filter, sharpness, dirty);
    let error = blocking_future(scope.pop());
    assert!(
        error.is_none(),
        "filter {filter:?}, size {size:?}: {error:?}"
    );
    let output = &scaler.output.as_ref().unwrap().texture;
    assert_eq!([output.width(), output.height()], size);
    readback(rs, output)
}

fn assert_pixels_close(actual: &[[u8; 4]], expected: &[[u8; 4]], tolerance: u8) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.abs_diff(*b) <= tolerance),
            "pixel {index}: {actual:?}, expected {expected:?}, tolerance {tolerance}"
        );
    }
}

fn pattern(size: [u32; 2], checker: bool) -> Vec<[u8; 4]> {
    (0..size[1])
        .flat_map(|y| {
            (0..size[0]).map(move |x| {
                let bright = if checker {
                    (x + y) % 2 == 0
                } else {
                    x * 3 > y * 2 + 2
                };
                if bright {
                    [240, 191, 79, 255]
                } else {
                    [12, 37, 83, 255]
                }
            })
        })
        .collect()
}

#[test]
#[ignore = "requires a local Vulkan adapter"]
fn gpu_scaling_preserves_constant_black_white_and_color() {
    let rs = render_state();
    let source = source_texture(&rs, [7, 5]);
    let mut scaler = scaler(&rs);
    for filter in FILTERS {
        for color in [[0, 0, 0, 255], [255; 4], [31, 127, 223, 255]] {
            upload(&rs, &source, &vec![color; 35]);
            for size in [[17, 13], [3, 2], [1, 1]] {
                let pixels = render(&rs, &mut scaler, &source, size, filter, true);
                assert_pixels_close(&pixels, &vec![color; (size[0] * size[1]) as usize], 2);
            }
        }
    }
}

#[test]
#[ignore = "requires a local Vulkan adapter"]
fn gpu_scaling_identity_orientation_and_noninteger_resize() {
    let rs = render_state();
    let source = source_texture(&rs, [4, 3]);
    let input: Vec<_> = (0..3)
        .flat_map(|y| {
            (0..4).map(move |x| {
                [
                    (x * 61 + y * 3) as u8,
                    (y * 79 + x * 5) as u8,
                    (x * 17 + y * 31) as u8,
                    255,
                ]
            })
        })
        .collect();
    upload(&rs, &source, &input);
    let mut scaler = scaler(&rs);
    for filter in [ScalingFilter::Nearest, ScalingFilter::Linear] {
        assert_pixels_close(
            &render(&rs, &mut scaler, &source, [4, 3], filter, true),
            &input,
            1,
        );
        let output = render(&rs, &mut scaler, &source, [11, 7], filter, false);
        let expected: Vec<_> = (0..7)
            .flat_map(|y| {
                (0..11).map({
                    let input = &input;
                    move |x| {
                        let sx = (x as f32 + 0.5) * 4.0 / 11.0 - 0.5;
                        let sy = (y as f32 + 0.5) * 3.0 / 7.0 - 0.5;
                        let sample =
                            |x: i32, y: i32| input[(y.clamp(0, 2) * 4 + x.clamp(0, 3)) as usize];
                        if filter == ScalingFilter::Nearest {
                            return sample((sx + 0.5).floor() as i32, (sy + 0.5).floor() as i32);
                        }
                        let (x0, y0) = (sx.floor() as i32, sy.floor() as i32);
                        let (fx, fy) = (sx - sx.floor(), sy - sy.floor());
                        std::array::from_fn(|channel| {
                            let top = f32::from(sample(x0, y0)[channel]) * (1.0 - fx)
                                + f32::from(sample(x0 + 1, y0)[channel]) * fx;
                            let bottom = f32::from(sample(x0, y0 + 1)[channel]) * (1.0 - fx)
                                + f32::from(sample(x0 + 1, y0 + 1)[channel]) * fx;
                            (top * (1.0 - fy) + bottom * fy).round() as u8
                        })
                    }
                })
            })
            .collect();
        assert_pixels_close(&output, &expected, 1);
    }
}

#[test]
#[ignore = "requires a local Vulkan adapter"]
fn gpu_advanced_filters_are_not_bilinear_on_edges_and_checkerboards() {
    let rs = render_state();
    let source = source_texture(&rs, [12, 12]);
    let mut scaler = scaler(&rs);
    let step: Vec<_> = (0..144)
        .map(|index| {
            if index % 12 < 6 {
                [0, 0, 0, 255]
            } else {
                [255; 4]
            }
        })
        .collect();
    upload(&rs, &source, &step);
    for (filter, values) in [
        (ScalingFilter::Linear, [64, 191]),
        (ScalingFilter::Bicubic, [52, 203]),
        (ScalingFilter::ScaleForce, [0, 255]),
    ] {
        let output = render(&rs, &mut scaler, &source, [24, 12], filter, true);
        let actual = [output[6 * 24 + 11], output[6 * 24 + 12]];
        let expected = values.map(|value| [value, value, value, 255]);
        assert_pixels_close(&actual, &expected, 2);
    }
    for checker in [false, true] {
        upload(&rs, &source, &pattern([12, 12], checker));
        let linear = render(
            &rs,
            &mut scaler,
            &source,
            [31, 27],
            ScalingFilter::Linear,
            true,
        );
        let mut advanced = Vec::new();
        for filter in [
            ScalingFilter::Bicubic,
            ScalingFilter::ScaleForce,
            ScalingFilter::Fsr,
        ] {
            let output = render(&rs, &mut scaler, &source, [31, 27], filter, false);
            let changed = output
                .iter()
                .zip(&linear)
                .filter(|(a, b)| a[..3].iter().zip(&b[..3]).any(|(a, b)| a.abs_diff(*b) > 2))
                .count();
            assert!(
                changed >= 12,
                "{filter:?} resembles bilinear: {changed} pixels, checker={checker}"
            );
            assert!(output.iter().all(|pixel| pixel[3] == 255));
            advanced.push(output);
        }
        assert_ne!(
            advanced[0], advanced[1],
            "bicubic and ScaleForce used the same result"
        );
        assert_ne!(
            advanced[0], advanced[2],
            "bicubic and FSR used the same result"
        );
    }
}

#[test]
#[ignore = "requires a local Vulkan adapter"]
fn gpu_fsr_recovers_after_resize_filter_changes_and_source_uploads() {
    let rs = render_state();
    let source = source_texture(&rs, [12, 12]);
    upload(&rs, &source, &pattern([12, 12], false));
    let mut reused = scaler(&rs);
    let mut texture_id = None;
    for (size, filter) in [
        ([19, 17], ScalingFilter::Fsr),
        ([25, 21], ScalingFilter::Fsr),
        ([25, 21], ScalingFilter::Nearest),
        ([25, 21], ScalingFilter::ScaleForce),
        ([25, 21], ScalingFilter::Fsr),
        ([19, 17], ScalingFilter::Fsr),
    ] {
        let actual = render(&rs, &mut reused, &source, size, filter, false);
        let expected = render(&rs, &mut scaler(&rs), &source, size, filter, true);
        assert_pixels_close(&actual, &expected, 0);
        if let Some(id) = texture_id {
            assert_eq!(reused.id, Some(id));
        }
        texture_id = reused.id;
        assert_pixels_close(
            &render(&rs, &mut reused, &source, size, filter, false),
            &actual,
            0,
        );
    }
    upload(&rs, &source, &pattern([12, 12], true));
    let actual = render(
        &rs,
        &mut reused,
        &source,
        [19, 17],
        ScalingFilter::Fsr,
        true,
    );
    let expected = render(
        &rs,
        &mut scaler(&rs),
        &source,
        [19, 17],
        ScalingFilter::Fsr,
        true,
    );
    assert_pixels_close(&actual, &expected, 0);
}

fn rcas_reference(input: &[[u8; 4]], size: [u32; 2], sharpness: u8) -> Vec<[u8; 4]> {
    let sample = |x: i32, y: i32| -> [f64; 3] {
        let offset = y.clamp(0, size[1] as i32 - 1) as usize * size[0] as usize
            + x.clamp(0, size[0] as i32 - 1) as usize;
        std::array::from_fn(|channel| f64::from(input[offset][channel]) / 255.0)
    };
    let mut output = Vec::with_capacity(input.len());
    for y in 0..size[1] as i32 {
        for x in 0..size[0] as i32 {
            let center = sample(x, y);
            let ring = [sample(x, y - 1), sample(x - 1, y), sample(x + 1, y), sample(x, y + 1)];
            let mut lobe = -0.1875f64;
            for channel in 0..3 {
                let low = ring.iter().map(|pixel| pixel[channel]).fold(1.0f64, f64::min);
                let high = ring.iter().map(|pixel| pixel[channel]).fold(0.0f64, f64::max);
                let black_limit = -low.min(center[channel]) / (4.0 * high).max(0.000001);
                let white_limit = (high.max(center[channel]) - 1.0)
                    / (4.0 * (1.0 - low)).max(0.000001);
                lobe = lobe.max(black_limit).max(white_limit);
            }
            lobe = lobe.min(0.0) * f64::from(sharpness.min(100)) / 100.0;
            let mut pixel = [255; 4];
            for channel in 0..3 {
                let neighbors: f64 = ring.iter().map(|pixel| pixel[channel]).sum();
                let value = (center[channel] + lobe * neighbors) / (1.0 + 4.0 * lobe);
                pixel[channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            output.push(pixel);
        }
    }
    output
}

#[test]
#[ignore = "requires a local Vulkan adapter"]
fn gpu_fsr_sharpness_changes_paused_frame_and_matches_intermediate_reference() {
    let rs = render_state();
    let source = source_texture(&rs, [12, 12]);
    upload(&rs, &source, &pattern([12, 12], true));
    let mut scaler = scaler(&rs);
    let size = [31, 27];
    let zero = render_with_sharpness(
        &rs, &mut scaler, &source, size, ScalingFilter::Fsr, 0, true,
    );
    let intermediate = readback(&rs, &scaler.intermediate.as_ref().unwrap().texture);
    assert_pixels_close(&zero, &intermediate, 0);
    let texture_id = scaler.id;
    for sharpness in [100, 37, 0, 255] {
        let actual = render_with_sharpness(
            &rs, &mut scaler, &source, size, ScalingFilter::Fsr, sharpness, false,
        );
        assert_eq!(scaler.id, texture_id);
        assert_pixels_close(&actual, &rcas_reference(&intermediate, size, sharpness), 2);
        assert_pixels_close(
            &readback(&rs, &scaler.intermediate.as_ref().unwrap().texture), &intermediate, 0,
        );
        if sharpness == 100 {
            let changed = actual.iter().zip(&zero).filter(|(a, b)| {
                a[..3].iter().zip(&b[..3]).any(|(a, b)| a.abs_diff(*b) > 2)
            }).count();
            assert!(changed >= 12, "sharpness did not change enough pixels: {changed}");
        }
    }
    for color in [[0, 0, 0, 255], [255; 4], [31, 127, 223, 255]] {
        upload(&rs, &source, &vec![color; 144]);
        for sharpness in [0, 100] {
            let actual = render_with_sharpness(
                &rs, &mut scaler, &source, size, ScalingFilter::Fsr, sharpness, true,
            );
            assert_pixels_close(&actual, &vec![color; (size[0] * size[1]) as usize], 2);
        }
    }
}

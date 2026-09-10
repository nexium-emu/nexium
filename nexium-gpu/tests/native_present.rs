#![cfg(windows)]

use nexium_gpu::presentation::{PresentParameters, PresentationTarget, SurfaceState};
use nexium_gpu::{rt_cache::RtKey, Renderer};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

struct Logger;
static ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

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

struct Events;
impl ApplicationHandler for Events {
    fn resumed(&mut self, _: &ActiveEventLoop) {}
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

fn pump_until(event_loop: &mut EventLoop<()>, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "presentation timed out"
        );
        event_loop.pump_app_events(Some(Duration::from_millis(2)), &mut Events);
    }
}

#[test]
#[ignore = "requires a Windows Vulkan presentation device"]
fn gpu_frames_survive_resize_snapshot_backpressure_and_shutdown() {
    let _ = log::set_logger(&Logger);
    log::set_max_level(log::LevelFilter::Info);
    let mut event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_title("NeXium presentation test")
                .with_inner_size(winit::dpi::PhysicalSize::new(320, 180)),
        )
        .unwrap();
    let RawWindowHandle::Win32(handle) = window.window_handle().unwrap().as_raw() else {
        unreachable!()
    };
    let target = unsafe {
        PresentationTarget::win32(
            handle.hwnd.get(),
            handle.hinstance.unwrap().get(),
            Arc::new(|| {}),
        )
    };
    target.configure(SurfaceState {
        width: 320,
        height: 180,
        visible: true,
        vsync: true,
        nearest: true,
    });
    let renderer = Renderer::new_with_presentation(Some(target.clone())).unwrap();
    let key = RtKey::new(1, 16, 8, 0x10000);
    let mut pixels = vec![0; 16 * 8 * 4];
    for y in 0..8 {
        for x in 0..16 {
            let color = if y < 4 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            pixels[(y * 16 + x) * 4..(y * 16 + x + 1) * 4].copy_from_slice(&color);
        }
    }
    renderer
        .clear_target_with_format(
            1,
            16,
            8,
            0x10000,
            [0.0; 4],
            ash::vk::Format::A8B8G8R8_UNORM_PACK32,
        )
        .unwrap();
    renderer
        .upload_target_rgba(1, 16, 8, 0x10000, &pixels)
        .unwrap();
    renderer
        .clear_target_rect_with_format(
            1,
            16,
            8,
            0x10000,
            [1.0, 0.0, 0.0, 1.0],
            [0, 0, 8, 4],
            ash::vk::Format::A8B8G8R8_UNORM_PACK32,
        )
        .unwrap();
    let stamp = renderer.render_target_stamp(key).unwrap();
    let parameters = PresentParameters {
        read_rect: None,
        crop: None,
        flip_y: false,
        transform: 0,
        present_at: Instant::now(),
    };
    assert!(!renderer
        .present_image(key, stamp + 1, false, parameters)
        .unwrap());
    assert!(renderer
        .present_image(key, stamp, false, parameters)
        .unwrap());
    pump_until(&mut event_loop, || target.progress().0 >= 1);
    assert_eq!(target.metrics().2, 0);
    target.request_snapshot();
    let mut snapshot = None;
    pump_until(&mut event_loop, || {
        snapshot = target.take_snapshot();
        snapshot.is_some()
    });
    let snapshot = snapshot.unwrap();
    assert_eq!((snapshot.width, snapshot.height), (16, 8));
    assert_eq!(snapshot.pixels, pixels);
    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(480, 270));
    event_loop.pump_app_events(Some(Duration::from_millis(20)), &mut Events);
    target.configure(SurfaceState {
        width: 480,
        height: 270,
        visible: true,
        vsync: false,
        nearest: false,
    });
    let producer = {
        let renderer = renderer.clone();
        std::thread::spawn(move || {
            for n in 0..24 {
                assert!(renderer
                    .present_image(
                        key,
                        stamp,
                        false,
                        PresentParameters {
                            transform: if n == 23 { 2 } else { 0 },
                            ..parameters
                        }
                    )
                    .unwrap());
            }
        })
    };
    pump_until(&mut event_loop, || {
        producer.is_finished() && target.progress().0 >= 25
    });
    producer.join().unwrap();
    target.configure(SurfaceState {
        visible: false,
        ..Default::default()
    });
    target.request_snapshot();
    let mut flipped = None;
    pump_until(&mut event_loop, || {
        flipped = target.take_snapshot();
        flipped.is_some()
    });
    let flipped = flipped.unwrap();
    assert_eq!(&flipped.pixels[..4], &[0, 0, 255, 255]);
    assert!(target.metrics().1 >= 25);
    assert_eq!(target.metrics().2, 2);
    target.configure(SurfaceState {
        width: 480,
        height: 270,
        visible: true,
        vsync: true,
        nearest: true,
    });
    let initial_frames = target.progress().0;
    let started = Instant::now();
    let producer = {
        let renderer = renderer.clone();
        std::thread::spawn(move || {
            for n in 0..120 {
                assert!(renderer
                    .present_image(
                        key,
                        stamp,
                        false,
                        PresentParameters {
                            present_at: started + Duration::from_nanos(n * 16_666_667),
                            ..parameters
                        }
                    )
                    .unwrap());
            }
        })
    };
    pump_until(&mut event_loop, || {
        producer.is_finished() && target.progress().0 >= initial_frames + 120
    });
    producer.join().unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(2250),
        "60 fps producer was throttled: {:?}",
        started.elapsed()
    );
    let worker = std::thread::spawn(move || {
        target.stop();
        drop(renderer);
    });
    pump_until(&mut event_loop, || worker.is_finished());
    worker.join().unwrap();
    let errors = ERRORS.lock().unwrap();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

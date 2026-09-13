use nexium_gpu::presentation::{PresentationTarget, SurfaceState};
use std::sync::Arc;

pub struct NativeGameWindow {
    hwnd: isize,
    hinstance: isize,
    pub target: Option<Arc<PresentationTarget>>,
    pub active: bool,
    pub sequence: u64,
    pub dimensions: [u32; 2],
    visible: bool,
    placement: [i32; 4],
    holes: Vec<[i32; 4]>,
}

impl NativeGameWindow {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::UI::WindowsAndMessaging::*;
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            let RawWindowHandle::Win32(parent) = cc.window_handle().ok()?.as_raw() else {
                return None;
            };
            let hinstance = unsafe {
                windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null())
            };
            let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
            let title: Vec<u16> = "NeXium Vulkan Game\0".encode_utf16().collect();
            let hwnd = unsafe {
                CreateWindowExW(
                    0,
                    class.as_ptr(),
                    title.as_ptr(),
                    WS_CHILD | WS_DISABLED | WS_CLIPSIBLINGS,
                    0,
                    0,
                    1,
                    1,
                    parent.hwnd.get() as _,
                    std::ptr::null_mut(),
                    hinstance,
                    std::ptr::null(),
                )
            };
            if hwnd.is_null() {
                log::error!("could not create native game window");
                return None;
            }
            unsafe {
                let style = GetWindowLongPtrW(parent.hwnd.get() as _, GWL_STYLE);
                SetWindowLongPtrW(
                    parent.hwnd.get() as _,
                    GWL_STYLE,
                    style | WS_CLIPCHILDREN as isize,
                );
            }
            Some(Self {
                hwnd: hwnd as isize,
                hinstance: hinstance as isize,
                target: None,
                active: false,
                sequence: 0,
                dimensions: [0, 0],
                visible: false,
                placement: [0, 0, 1, 1],
                holes: Vec::new(),
            })
        }
        #[cfg(not(windows))]
        {
            let _ = cc;
            None
        }
    }

    pub fn begin_game(&mut self, ctx: &egui::Context) -> Arc<PresentationTarget> {
        self.hide();
        if let Some(target) = &self.target {
            target.stop();
        }
        let ctx = ctx.clone();
        let target = unsafe {
            PresentationTarget::win32(
                self.hwnd,
                self.hinstance,
                Arc::new(move || {
                    ctx.request_repaint_after(std::time::Duration::from_nanos(1));
                }),
            )
        };
        self.target = Some(target.clone());
        self.sequence = 0;
        self.dimensions = [0, 0];
        self.active = false;
        target
    }

    pub fn poll(&mut self) -> u64 {
        let Some(target) = &self.target else {
            return 0;
        };
        if let Some(error) = target.take_error() {
            log::error!("Native game presentation stopped: {error}");
            self.active = false;
        }
        let (sequence, width, height) = target.progress();
        let new_frames = sequence.saturating_sub(self.sequence);
        if new_frames == 0 {
            return 0;
        }
        self.sequence = sequence;
        self.dimensions = [width, height];
        self.active = true;
        new_frames
    }

    pub fn update(
        &mut self,
        rect: Option<egui::Rect>,
        holes: &[egui::Rect],
        scale: f32,
        vsync: bool,
        nearest: bool,
    ) {
        let rect = rect.filter(|rect| self.active && rect.is_positive());
        let Some(rect) = rect else {
            self.hide();
            return;
        };
        let x = (rect.min.x * scale).round() as i32;
        let y = (rect.min.y * scale).round() as i32;
        let w = ((rect.max.x * scale).round() as i32 - x).max(1);
        let h = ((rect.max.y * scale).round() as i32 - y).max(1);
        let placement = [x, y, w, h];
        let holes: Vec<_> = holes
            .iter()
            .map(|r| r.intersect(rect))
            .filter(|r| r.is_positive())
            .map(|r| {
                [
                    (r.min.x * scale).floor() as i32 - x,
                    (r.min.y * scale).floor() as i32 - y,
                    (r.max.x * scale).ceil() as i32 - x,
                    (r.max.y * scale).ceil() as i32 - y,
                ]
            })
            .collect();
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::Graphics::Gdi::*;
            use windows_sys::Win32::UI::WindowsAndMessaging::*;
            if self.placement != placement {
                SetWindowPos(
                    self.hwnd as _,
                    std::ptr::null_mut(),
                    x,
                    y,
                    w,
                    h,
                    SWP_NOACTIVATE | SWP_NOZORDER,
                );
            }
            if self.placement != placement || self.holes != holes {
                let region = CreateRectRgn(0, 0, w, h);
                if !region.is_null() {
                    for hole in &holes {
                        let cut = CreateRectRgn(hole[0], hole[1], hole[2], hole[3]);
                        if !cut.is_null() {
                            CombineRgn(region, region, cut, RGN_DIFF);
                            DeleteObject(cut);
                        }
                    }
                    if SetWindowRgn(self.hwnd as _, region, 1) == 0 {
                        DeleteObject(region);
                    }
                }
            }
            if !self.visible {
                ShowWindow(self.hwnd as _, SW_SHOWNOACTIVATE);
            }
        }
        self.placement = placement;
        self.holes = holes;
        self.visible = true;
        if let Some(target) = &self.target {
            target.configure(SurfaceState {
                width: w as u32,
                height: h as u32,
                visible: true,
                vsync,
                nearest,
            });
        }
    }

    pub fn hide(&mut self) {
        if self.visible {
            if let Some(target) = &self.target {
                target.request_snapshot();
            }
            #[cfg(windows)]
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(self.hwnd as _, 0);
            }
            self.visible = false;
        }
        if let Some(target) = &self.target {
            target.configure(SurfaceState::default());
        }
    }
}

impl Drop for NativeGameWindow {
    fn drop(&mut self) {
        if let Some(target) = &self.target {
            target.stop();
        }
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::DestroyWindow(self.hwnd as _);
        }
    }
}

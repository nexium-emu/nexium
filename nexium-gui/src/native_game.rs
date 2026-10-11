use nexium_gpu::presentation::{PresentationTarget, ScalingFilter, SurfaceState};
use std::sync::Arc;

/// What the window system was last told about the game window's geometry.
/// `None` means unknown: nothing applied yet, or a failed update that may
/// have taken effect only partly. Anything unknown is re-sent in full.
#[derive(Default)]
struct AppliedGeometry {
    placement: Option<[i32; 4]>,
    holes: Option<Vec<[i32; 4]>>,
}

impl AppliedGeometry {
    /// `(moved, reshaped)`: whether the placement and the region (which
    /// depends on the size as well as the holes) must be sent again.
    fn changes(&self, placement: [i32; 4], holes: &[[i32; 4]]) -> (bool, bool) {
        let moved = self.placement != Some(placement);
        (moved, moved || self.holes.as_deref() != Some(holes))
    }

    fn record(&mut self, placement: [i32; 4], holes: Vec<[i32; 4]>) {
        self.placement = Some(placement);
        self.holes = Some(holes);
    }

    fn invalidate(&mut self) {
        *self = Self::default();
    }
}

pub struct NativeGameWindow {
    #[cfg(not(target_os = "linux"))]
    hwnd: isize,
    #[cfg(not(target_os = "linux"))]
    hinstance: isize,
    #[cfg(target_os = "linux")]
    x11: x11::ChildWindow,
    pub target: Option<Arc<PresentationTarget>>,
    pub active: bool,
    pub sequence: u64,
    pub dimensions: [u32; 2],
    visible: bool,
    applied: AppliedGeometry,
}

impl NativeGameWindow {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::UI::WindowsAndMessaging::*;
            use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
            let Some(RawWindowHandle::Win32(parent)) = cc.window_handle().ok().map(|handle| handle.as_raw()) else {
                log::info!("[present] game presentation path: legacy because the GUI window has no Win32 handle");
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
                log::error!("[present] game presentation path: legacy because the native game window could not be created");
                return None;
            }
            log::info!("[present] game presentation path: native-win32 (child window of the GUI window)");
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
                applied: AppliedGeometry::default(),
            })
        }
        #[cfg(target_os = "linux")]
        {
            use winit::raw_window_handle::HasWindowHandle;
            let legacy = std::env::var_os("NEXIUM_LEGACY_PRESENT").is_some();
            let x_server = std::env::var("DISPLAY").is_ok_and(|display| !display.is_empty());
            let Ok(handle) = cc.window_handle() else {
                log::info!("[present] game presentation path: legacy because the GUI window has no native handle");
                return None;
            };
            let parent = match x11::parent_window(handle.as_raw(), legacy, x_server) {
                Ok(parent) => parent,
                Err(reason) => {
                    log::info!("[present] game presentation path: legacy because {reason}");
                    return None;
                }
            };
            match x11::ChildWindow::new(parent) {
                Ok(x11) => {
                    let wayland_session = ["WAYLAND_DISPLAY", "WAYLAND_SOCKET"]
                        .iter()
                        .any(|name| std::env::var(name).is_ok_and(|value| !value.is_empty()));
                    log::info!(
                        "[present] game presentation path: native-x11 (Vulkan on X11 child window {:#x} of {parent:#x}{})",
                        x11.window(),
                        if wayland_session {
                            "; Wayland session, GUI forced onto XWayland by NEXIUM_X11"
                        } else {
                            ""
                        }
                    );
                    Some(Self {
                        x11,
                        target: None,
                        active: false,
                        sequence: 0,
                        dimensions: [0, 0],
                        visible: false,
                        applied: AppliedGeometry::default(),
                    })
                }
                Err(error) => {
                    log::warn!(
                        "[present] game presentation path: legacy because the X11 game window could not be created ({error})"
                    );
                    None
                }
            }
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            let _ = cc;
            log::info!("[present] game presentation path: legacy because this platform has no native presenter");
            None
        }
    }

    pub fn begin_game(&mut self, ctx: &egui::Context) -> Arc<PresentationTarget> {
        self.hide();
        if let Some(target) = &self.target {
            target.stop();
        }
        let ctx = ctx.clone();
        #[cfg(not(target_os = "linux"))]
        let target = unsafe {
            PresentationTarget::win32(
                self.hwnd,
                self.hinstance,
                Arc::new(move || {
                    ctx.request_repaint_after(std::time::Duration::from_nanos(1));
                }),
            )
        };
        #[cfg(target_os = "linux")]
        let target = unsafe {
            PresentationTarget::xcb(
                self.x11.raw_connection(),
                self.x11.window(),
                Some(self.x11.keepalive()),
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

    pub fn set_title(&self, title: &str) {
        #[cfg(windows)]
        {
            let title: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowTextW(
                    self.hwnd as _,
                    title.as_ptr(),
                );
            }
        }
        #[cfg(not(windows))]
        let _ = title;
    }

    pub fn update(
        &mut self,
        rect: Option<egui::Rect>,
        holes: &[egui::Rect],
        scale: f32,
        vsync: bool,
        filter: ScalingFilter,
        sharpness: u8,
    ) {
        // Once per GUI frame, whether or not the game window is shown.
        #[cfg(target_os = "linux")]
        self.x11.drain_events();
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
        let (moved, reshaped) = self.applied.changes(placement, &holes);
        #[cfg(not(any(windows, target_os = "linux")))]
        let _ = (moved, reshaped);
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::Graphics::Gdi::*;
            use windows_sys::Win32::UI::WindowsAndMessaging::*;
            if moved {
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
            if reshaped {
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
        #[cfg(target_os = "linux")]
        {
            if let Err(error) = self.x11.apply(
                moved.then_some(placement),
                reshaped.then_some(holes.as_slice()),
                !self.visible,
                [w, h],
            ) {
                // Stopping the target makes the renderer decline native
                // presents, so frames come back through the GUI path.
                log::error!("[native-game] X11 game window update failed ({error}); using GUI frame delivery");
                // Requests before the failing one may have taken effect, so
                // the server's geometry is unknown: re-send it all next time.
                self.applied.invalidate();
                if let Some(target) = &self.target {
                    target.stop();
                }
                self.active = false;
                self.x11.unmap();
                self.hide();
                return;
            }
        }
        self.applied.record(placement, holes);
        self.visible = true;
        if let Some(target) = &self.target {
            target.configure(SurfaceState {
                width: w as u32,
                height: h as u32,
                visible: true,
                vsync,
                filter,
                sharpness,
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
            #[cfg(target_os = "linux")]
            self.x11.unmap();
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
        // The X window itself is destroyed once the last PresentationTarget
        // (and so the last Vulkan surface on it) is gone; hide it now.
        #[cfg(target_os = "linux")]
        self.x11.unmap();
    }
}

/// X11 counterpart of the win32 child window: an input-transparent child of
/// the winit window that the Vulkan presenter owns a swapchain on.
#[cfg(target_os = "linux")]
mod x11 {
    use std::sync::Arc;
    use winit::raw_window_handle::RawWindowHandle;
    use x11rb::connection::{Connection, RequestConnection};
    use x11rb::protocol::shape::{self, ConnectionExt as _};
    use x11rb::protocol::xproto::{self, ConnectionExt as _};
    use x11rb::xcb_ffi::XCBConnection;

    /// Picks the X11 parent window for the game surface, or says why the
    /// native presenter cannot be used with this window. `x_server` is
    /// whether an X server (`DISPLAY`) exists for the `NEXIUM_X11` opt-in.
    pub fn parent_window(handle: RawWindowHandle, legacy: bool, x_server: bool) -> Result<u32, &'static str> {
        if legacy {
            return Err("NEXIUM_LEGACY_PRESENT is set");
        }
        match handle {
            RawWindowHandle::Xlib(handle) => u32::try_from(handle.window)
                .ok()
                .filter(|window| *window != 0)
                .ok_or("the X11 window id is invalid"),
            RawWindowHandle::Xcb(handle) => Ok(handle.window.get()),
            RawWindowHandle::Wayland(_) if x_server => Err(
                "the GUI runs as a native Wayland client, where native game presentation is not implemented \
                 (start with NEXIUM_X11=1 to run the GUI under XWayland and use native-x11)",
            ),
            RawWindowHandle::Wayland(_) => Err(
                "the GUI runs as a native Wayland client, where native game presentation is not implemented, \
                 and there is no X server (DISPLAY) for NEXIUM_X11=1",
            ),
            _ => Err("the window system has no native game presentation path"),
        }
    }

    /// Debug-logs something the private connection received unasked, the
    /// first few times and then at powers of two.
    fn note_unsolicited(event: &x11rb::protocol::Event) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static ERRORS: AtomicU64 = AtomicU64::new(0);
        static EVENTS: AtomicU64 = AtomicU64::new(0);
        let (counter, kind) = match event {
            x11rb::protocol::Event::Error(_) => (&ERRORS, "error"),
            _ => (&EVENTS, "event"),
        };
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 8 || n.is_power_of_two() {
            log::debug!("[native-game] discarded unsolicited X11 {kind} #{n}: {event:?}");
        }
    }

    /// Bounding rectangle and hole rectangles (window-relative `[x0, y0, x1,
    /// y1]`) as X rectangles, clamped to the window and the protocol's 16-bit
    /// range. Empty holes are dropped.
    pub fn shape_rectangles(
        width: i32,
        height: i32,
        holes: &[[i32; 4]],
    ) -> (xproto::Rectangle, Vec<xproto::Rectangle>) {
        let width = width.clamp(1, i32::from(i16::MAX));
        let height = height.clamp(1, i32::from(i16::MAX));
        let bounding = xproto::Rectangle { x: 0, y: 0, width: width as u16, height: height as u16 };
        let holes = holes
            .iter()
            .filter_map(|hole| {
                let x0 = hole[0].clamp(0, width);
                let y0 = hole[1].clamp(0, height);
                let x1 = hole[2].clamp(0, width);
                let y1 = hole[3].clamp(0, height);
                (x1 > x0 && y1 > y0).then(|| xproto::Rectangle {
                    x: x0 as i16,
                    y: y0 as i16,
                    width: (x1 - x0) as u16,
                    height: (y1 - y0) as u16,
                })
            })
            .collect();
        (bounding, holes)
    }

    pub struct ChildWindow {
        shared: Arc<Shared>,
    }

    /// The connection and the window, shared with every PresentationTarget
    /// made for this window. A Vulkan surface is created on both, so neither
    /// may go away while a renderer could still hold one: the window is
    /// destroyed (and the connection closed) only when the last owner (the
    /// GUI or a presenter thread tearing down its surface) drops this.
    struct Shared {
        // A private connection: xcb is thread-safe, so the UI thread can
        // place the window here while the presenter thread's Vulkan driver
        // presents through the same connection, and nothing races winit's
        // Xlib display. Requests on one connection are ordered, and every
        // placement request is checked (a round trip) before the new size is
        // published to the presenter, so the server has applied a resize
        // before the swapchain is recreated for it.
        //
        // The connection's general event queue belongs to us and is drained
        // once per GUI frame (`drain_events`). The window selects no events,
        // but core MappingNotify goes to every client, and errors of
        // unchecked requests (the Vulkan WSI sends such requests on this
        // connection) are queued there too, so it would otherwise grow for
        // the whole session. Draining cannot steal anything from the WSI:
        // replies and checked errors go to their cookies, and the Present
        // events a WSI waits for (Mesa's wsi_common_x11; NVIDIA's driver
        // likewise works on application-owned connections, e.g. SDL's and
        // GLFW's, whose event loops drain the general queue) are routed to
        // a special event queue registered with xcb_register_for_special_xge,
        // which xcb_poll_for_event never returns.
        conn: XCBConnection,
        window: u32,
    }

    impl Drop for Shared {
        fn drop(&mut self) {
            // Fails harmlessly if the parent (and so this child) is already gone.
            if let Ok(cookie) = self.conn.destroy_window(self.window) {
                let _ = cookie.check();
            }
        }
    }

    impl ChildWindow {
        pub fn new(parent: u32) -> Result<Self, String> {
            let (conn, _) = XCBConnection::connect(None).map_err(|e| e.to_string())?;
            if conn
                .extension_information(shape::X11_EXTENSION_NAME)
                .map_err(|e| e.to_string())?
                .is_none()
            {
                return Err("the X server has no SHAPE extension".into());
            }
            let version = conn
                .shape_query_version()
                .map_err(|e| e.to_string())?
                .reply()
                .map_err(|e| e.to_string())?;
            if (version.major_version, version.minor_version) < (1, 1) {
                return Err("the X server's SHAPE extension has no input shapes".into());
            }
            let root = conn
                .get_geometry(parent)
                .map_err(|e| e.to_string())?
                .reply()
                .map_err(|e| e.to_string())?
                .root;
            let screen = conn
                .setup()
                .roots
                .iter()
                .find(|screen| screen.root == root)
                .ok_or("the application window's X screen was not found")?;
            let window = conn.generate_id().map_err(|e| e.to_string())?;
            // The screen's default visual rather than the parent's: the
            // parent may be a 32-bit ARGB window, where an alpha channel in
            // the game image would turn into desktop translucency. A depth
            // that differs from the parent needs an explicit colormap and
            // border pixel.
            conn.create_window(
                screen.root_depth,
                window,
                parent,
                0,
                0,
                1,
                1,
                0,
                xproto::WindowClass::INPUT_OUTPUT,
                screen.root_visual,
                &xproto::CreateWindowAux::new()
                    .background_pixel(screen.black_pixel)
                    .border_pixel(screen.black_pixel)
                    .colormap(screen.default_colormap)
                    .event_mask(xproto::EventMask::NO_EVENT),
            )
            .map_err(|e| e.to_string())?
            .check()
            .map_err(|e| e.to_string())?;
            let child = Self { shared: Arc::new(Shared { conn, window }) };
            // An empty input shape: the pointer is never "in" this window, so
            // every pointer event (and enter/leave state) stays with the egui
            // window, like WS_DISABLED on win32.
            child
                .shared
                .conn
                .shape_rectangles(
                    shape::SO::SET,
                    shape::SK::INPUT,
                    xproto::ClipOrdering::UNSORTED,
                    window,
                    0,
                    0,
                    &[],
                )
                .map_err(|e| e.to_string())?
                .check()
                .map_err(|e| e.to_string())?;
            Ok(child)
        }

        pub fn window(&self) -> u32 {
            self.shared.window
        }

        pub fn raw_connection(&self) -> *mut std::ffi::c_void {
            self.shared.conn.get_raw_xcb_connection()
        }

        pub fn keepalive(&self) -> Arc<dyn std::any::Any + Send + Sync> {
            self.shared.clone()
        }

        /// Moves/resizes (`placement` = `[x, y, w, h]` in parent pixels),
        /// re-cuts the holes and maps the window, each only when asked. All
        /// requests are checked so a failure is reported instead of leaving
        /// an unread error on the connection.
        pub fn apply(
            &self,
            placement: Option<[i32; 4]>,
            holes: Option<&[[i32; 4]]>,
            map: bool,
            size: [i32; 2],
        ) -> Result<(), String> {
            let mut cookies = Vec::new();
            if let Some([x, y, w, h]) = placement {
                cookies.push(
                    self.shared.conn
                        .configure_window(
                            self.shared.window,
                            &xproto::ConfigureWindowAux::new()
                                .x(x)
                                .y(y)
                                .width(w.clamp(1, i32::from(i16::MAX)) as u32)
                                .height(h.clamp(1, i32::from(i16::MAX)) as u32),
                        )
                        .map_err(|e| e.to_string())?,
                );
            }
            if let Some(holes) = holes {
                let (bounding, holes) = shape_rectangles(size[0], size[1], holes);
                cookies.push(
                    self.shared.conn
                        .shape_rectangles(
                            shape::SO::SET,
                            shape::SK::BOUNDING,
                            xproto::ClipOrdering::UNSORTED,
                            self.shared.window,
                            0,
                            0,
                            &[bounding],
                        )
                        .map_err(|e| e.to_string())?,
                );
                if !holes.is_empty() {
                    cookies.push(
                        self.shared.conn
                            .shape_rectangles(
                                shape::SO::SUBTRACT,
                                shape::SK::BOUNDING,
                                xproto::ClipOrdering::UNSORTED,
                                self.shared.window,
                                0,
                                0,
                                &holes,
                            )
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
            if map {
                cookies.push(self.shared.conn.map_window(self.shared.window).map_err(|e| e.to_string())?);
            }
            for cookie in cookies {
                cookie.check().map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        pub fn unmap(&self) {
            if let Ok(cookie) = self.shared.conn.unmap_window(self.shared.window) {
                let _ = cookie.check();
            }
        }

        /// Discards everything waiting in the connection's general event
        /// queue (see `Shared::conn`) and returns how much that was. Never
        /// blocks. A broken connection is left to the next checked request
        /// to report.
        pub fn drain_events(&self) -> usize {
            let mut drained = 0;
            while let Ok(Some(event)) = self.shared.conn.poll_for_event() {
                note_unsolicited(&event);
                drained += 1;
            }
            drained
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use winit::raw_window_handle::{WaylandWindowHandle, XcbWindowHandle, XlibWindowHandle};

        #[test]
        fn x11_windows_get_a_native_parent_and_everything_else_falls_back() {
            let xlib = RawWindowHandle::Xlib(XlibWindowHandle::new(0x42));
            let xcb = RawWindowHandle::Xcb(XcbWindowHandle::new(std::num::NonZeroU32::new(0x43).unwrap()));
            let mut surface = 0u8;
            let wayland = RawWindowHandle::Wayland(WaylandWindowHandle::new(
                std::ptr::NonNull::from(&mut surface).cast(),
            ));
            for x_server in [false, true] {
                assert_eq!(parent_window(xlib, false, x_server), Ok(0x42));
                assert_eq!(parent_window(xcb, false, x_server), Ok(0x43));
                assert!(parent_window(wayland, false, x_server).is_err());
                assert!(parent_window(RawWindowHandle::Xlib(XlibWindowHandle::new(0)), false, x_server).is_err());
                assert_eq!(parent_window(xlib, true, x_server), Err("NEXIUM_LEGACY_PRESENT is set"));
                assert_eq!(parent_window(xcb, true, x_server), Err("NEXIUM_LEGACY_PRESENT is set"));
                assert_eq!(parent_window(wayland, true, x_server), Err("NEXIUM_LEGACY_PRESENT is set"));
            }
            // The Wayland reason only suggests the XWayland opt-in when an X
            // server exists to opt into.
            assert!(parent_window(wayland, false, true).unwrap_err().contains("NEXIUM_X11=1 to run"));
            assert!(parent_window(wayland, false, false).unwrap_err().contains("no X server"));
        }

        /// Unsolicited traffic on the private connection (here an error of
        /// a request whose cookie was dropped unchecked, as the WSI's are,
        /// and a ClientMessage sent with an empty mask, which X delivers to
        /// the window's creator) is drained instead of piling up.
        #[test]
        fn drain_events_empties_the_general_event_queue() {
            let Ok((conn, screen)) = XCBConnection::connect(None) else {
                eprintln!("no X server reachable; skipping");
                return;
            };
            let screen = &conn.setup().roots[screen];
            let parent = conn.generate_id().unwrap();
            conn.create_window(
                screen.root_depth, parent, screen.root, 0, 0, 64, 64, 0,
                xproto::WindowClass::INPUT_OUTPUT, screen.root_visual,
                &xproto::CreateWindowAux::new(),
            )
            .unwrap()
            .check()
            .unwrap();
            let child = ChildWindow::new(parent).unwrap();
            assert_eq!(child.drain_events(), 0);

            // BadWindow on the child's own connection, never checked.
            drop(child.shared.conn.map_window(0x1f_ffff).unwrap());
            let message = xproto::ClientMessageEvent::new(32, child.window(), xproto::AtomEnum::STRING, [1u32, 2, 3, 4, 5]);
            conn.send_event(false, child.window(), xproto::EventMask::NO_EVENT, message)
                .unwrap()
                .check()
                .unwrap();
            // Round trips on both connections so both items have arrived.
            conn.get_input_focus().unwrap().reply().unwrap();
            child.shared.conn.get_input_focus().unwrap().reply().unwrap();

            assert_eq!(child.drain_events(), 2);
            assert_eq!(child.drain_events(), 0);
            assert!(child.shared.conn.poll_for_event().unwrap().is_none());
            // Still usable afterwards.
            child.apply(Some([0, 0, 8, 8]), Some(&[]), false, [8, 8]).unwrap();
            drop(child);
            conn.destroy_window(parent).unwrap().check().unwrap();
        }

        #[test]
        fn shape_rectangles_clamp_holes_to_the_window_and_drop_empty_ones() {
            let (bounding, holes) = shape_rectangles(
                200,
                100,
                &[[10, 20, 60, 40], [-5, -5, 20, 10], [150, 90, 400, 300], [30, 30, 30, 50], [250, 0, 300, 10]],
            );
            assert_eq!((bounding.x, bounding.y, bounding.width, bounding.height), (0, 0, 200, 100));
            let holes: Vec<_> = holes.iter().map(|r| (r.x, r.y, r.width, r.height)).collect();
            assert_eq!(holes, [(10, 20, 50, 20), (0, 0, 20, 10), (150, 90, 50, 10)]);
            let (bounding, holes) = shape_rectangles(100_000, 0, &[]);
            assert_eq!((bounding.width, bounding.height), (i16::MAX as u16, 1));
            assert!(holes.is_empty());
        }

        /// Runs against the real X server when one is reachable (including
        /// XWayland). Everything stays under a never-mapped parent, so
        /// nothing appears on screen.
        #[test]
        fn child_window_requests_are_accepted_by_the_x_server() {
            let Ok((conn, screen)) = XCBConnection::connect(None) else {
                eprintln!("no X server reachable; skipping");
                return;
            };
            let screen = &conn.setup().roots[screen];
            let parent = conn.generate_id().unwrap();
            conn.create_window(
                screen.root_depth, parent, screen.root, 0, 0, 640, 480, 0,
                xproto::WindowClass::INPUT_OUTPUT, screen.root_visual,
                &xproto::CreateWindowAux::new(),
            )
            .unwrap()
            .check()
            .unwrap();

            let child = ChildWindow::new(parent).unwrap();
            assert!(!child.raw_connection().is_null());
            child
                .apply(Some([40, 30, 320, 180]), Some(&[[0, 0, 50, 20], [300, 170, 400, 400]]), true, [320, 180])
                .unwrap();
            let tree = conn.query_tree(parent).unwrap().reply().unwrap();
            assert_eq!(tree.children, [child.window()]);
            let geometry = conn.get_geometry(child.window()).unwrap().reply().unwrap();
            assert_eq!(
                (geometry.x, geometry.y, geometry.width, geometry.height, geometry.depth),
                (40, 30, 320, 180, screen.root_depth)
            );
            let attributes = conn.get_window_attributes(child.window()).unwrap().reply().unwrap();
            assert_eq!(attributes.map_state, xproto::MapState::UNVIEWABLE);
            assert_eq!(attributes.visual, screen.root_visual);
            let input = conn.shape_get_rectangles(child.window(), shape::SK::INPUT).unwrap().reply().unwrap();
            assert!(input.rectangles.is_empty());
            let bounding = conn.shape_get_rectangles(child.window(), shape::SK::BOUNDING).unwrap().reply().unwrap();
            let area: u32 = bounding.rectangles.iter().map(|r| u32::from(r.width) * u32::from(r.height)).sum();
            assert_eq!(area, 320 * 180 - 50 * 20 - 20 * 10);

            // Holes removed again: the bounding shape is the whole window.
            child.apply(None, Some(&[]), false, [320, 180]).unwrap();
            let bounding = conn.shape_get_rectangles(child.window(), shape::SK::BOUNDING).unwrap().reply().unwrap();
            let area: u32 = bounding.rectangles.iter().map(|r| u32::from(r.width) * u32::from(r.height)).sum();
            assert_eq!(area, 320 * 180);

            child.unmap();
            let attributes = conn.get_window_attributes(child.window()).unwrap().reply().unwrap();
            assert_eq!(attributes.map_state, xproto::MapState::UNMAPPED);

            // A destroyed window is reported, not silently ignored.
            let window = child.window();
            conn.destroy_window(parent).unwrap().check().unwrap();
            assert!(child.apply(Some([0, 0, 10, 10]), None, false, [10, 10]).is_err());
            drop(child);
            assert!(conn.get_geometry(window).unwrap().reply().is_err());
        }

        /// The window a Vulkan surface was made on must outlive the GUI's
        /// handle while a presenter still holds the keepalive.
        #[test]
        fn the_x_window_lives_until_the_last_keepalive_is_dropped() {
            let Ok((conn, screen)) = XCBConnection::connect(None) else {
                eprintln!("no X server reachable; skipping");
                return;
            };
            let screen = &conn.setup().roots[screen];
            let parent = conn.generate_id().unwrap();
            conn.create_window(
                screen.root_depth, parent, screen.root, 0, 0, 64, 64, 0,
                xproto::WindowClass::INPUT_OUTPUT, screen.root_visual,
                &xproto::CreateWindowAux::new(),
            )
            .unwrap()
            .check()
            .unwrap();
            let child = ChildWindow::new(parent).unwrap();
            let window = child.window();
            let keepalive = child.keepalive();
            drop(child);
            assert!(conn.get_geometry(window).unwrap().reply().is_ok());
            drop(keepalive);
            assert!(conn.get_geometry(window).unwrap().reply().is_err());
            conn.destroy_window(parent).unwrap().check().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_geometry_is_sent_in_full_and_only_changes_after_that() {
        let mut applied = AppliedGeometry::default();
        let hole = [[1, 2, 3, 4]];
        assert_eq!(applied.changes([0, 0, 10, 10], &[]), (true, true));
        applied.record([0, 0, 10, 10], hole.to_vec());
        assert_eq!(applied.changes([0, 0, 10, 10], &hole), (false, false));
        assert_eq!(applied.changes([0, 0, 10, 10], &[]), (false, true));
        // A move or resize re-cuts the region even with the same holes.
        assert_eq!(applied.changes([5, 0, 10, 10], &hole), (true, true));
        // After a failure the cached values must not be trusted, including
        // a later request for exactly the geometry that was cached.
        applied.invalidate();
        assert_eq!(applied.changes([0, 0, 10, 10], &hole), (true, true));
    }

    /// A failed X11 update leaves the cache unknown, so a later update with
    /// the old geometry is applied again instead of being skipped.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_failed_x11_update_invalidates_the_applied_geometry() {
        use x11rb::connection::Connection as _;
        use x11rb::protocol::xproto::{self, ConnectionExt as _};
        let Ok((conn, screen)) = x11rb::xcb_ffi::XCBConnection::connect(None) else {
            eprintln!("no X server reachable; skipping");
            return;
        };
        let screen = &conn.setup().roots[screen];
        let parent = conn.generate_id().unwrap();
        conn.create_window(
            screen.root_depth, parent, screen.root, 0, 0, 640, 480, 0,
            xproto::WindowClass::INPUT_OUTPUT, screen.root_visual,
            &xproto::CreateWindowAux::new(),
        )
        .unwrap()
        .check()
        .unwrap();
        let mut window = NativeGameWindow {
            x11: x11::ChildWindow::new(parent).unwrap(),
            target: None,
            active: true,
            sequence: 0,
            dimensions: [0, 0],
            visible: false,
            applied: AppliedGeometry::default(),
        };
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(300.0, 200.0));
        let hole = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(30.0, 10.0));
        window.update(Some(rect), &[hole], 1.0, true, ScalingFilter::Linear, 0);
        assert!(window.visible);
        assert_eq!(window.applied.placement, Some([10, 20, 300, 200]));
        assert_eq!(window.applied.holes.as_deref(), Some(&[[0, 0, 30, 10]][..]));

        conn.destroy_window(parent).unwrap().check().unwrap();
        let moved = egui::Rect::from_min_size(egui::pos2(12.0, 20.0), egui::vec2(300.0, 200.0));
        window.update(Some(moved), &[hole], 1.0, true, ScalingFilter::Linear, 0);
        assert!(!window.active && !window.visible);
        assert_eq!(window.applied.placement, None);
        assert_eq!(window.applied.holes, None);
        assert_eq!(window.applied.changes([10, 20, 300, 200], &[[0, 0, 30, 10]]), (true, true));
    }
}

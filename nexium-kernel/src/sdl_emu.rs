use parking_lot::Mutex;
use std::sync::Arc;

pub struct SdlContext {
    pub initialized: bool,
    pub window: Option<SdlWindow>,
    pub surface: Option<Vec<u8>>,
}

pub struct SdlWindow {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl SdlContext {
    pub fn new() -> Self {
        Self {
            initialized: false,
            window: None,
            surface: None,
        }
    }

    pub fn init(&mut self) -> i32 {
        log::info!("SDL_Init called");
        self.initialized = true;
        0
    }

    pub fn create_window(&mut self, title: &str, width: u32, height: u32) -> bool {
        log::info!("SDL_CreateWindow '{}' {}x{}", title, width, height);
        self.window = Some(SdlWindow {
            width,
            height,
            pixels: vec![0u8; (width * height * 4) as usize],
        });
        self.surface = Some(vec![0u8; (width * height * 4) as usize]);
        true
    }

    pub fn get_window_surface(&mut self) -> Option<*mut u8> {
        if let Some(ref mut surface) = self.surface {
            Some(surface.as_mut_ptr())
        } else {
            None
        }
    }

    pub fn update_window_surface(&mut self) {
        if let (Some(ref mut surface), Some(ref mut window)) = (&mut self.surface, &mut self.window)
        {
            log::debug!(
                "SDL_UpdateWindowSurface - frame available {}x{}",
                window.width,
                window.height
            );
            window.pixels.copy_from_slice(surface);
        }
    }

    pub fn get_window_pixels(&self) -> Option<(u32, u32, Vec<u8>)> {
        self.window
            .as_ref()
            .map(|w| (w.width, w.height, w.pixels.clone()))
    }
}

pub static SDL_CONTEXT: once_cell::sync::OnceCell<Arc<Mutex<SdlContext>>> =
    once_cell::sync::OnceCell::new();

pub fn get_sdl_context() -> Arc<Mutex<SdlContext>> {
    SDL_CONTEXT
        .get_or_init(|| Arc::new(Mutex::new(SdlContext::new())))
        .clone()
}

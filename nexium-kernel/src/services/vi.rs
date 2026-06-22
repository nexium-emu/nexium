use nexium_common::result::SUCCESS;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub struct DisplayService {
    displays: HashMap<u64, Display>,
    next_display_id: u64,
    layers: HashMap<u64, Layer>,
    next_layer_id: u64,
    framebuffer_queue: Arc<Mutex<Vec<Vec<u8>>>>,
}

#[derive(Clone)]
struct Display {
    id: u64,
    name: String,
    width: u32,
    height: u32,
    open: bool,
}

#[derive(Clone)]
struct Layer {
    id: u64,
    display_id: u64,
    width: u32,
    height: u32,
}

impl DisplayService {
    pub fn new() -> Self {
        let mut displays = HashMap::new();
        let default_display = Display {
            id: 0,
            name: "Default".to_string(),
            width: 1280,
            height: 720,
            open: false,
        };
        displays.insert(0, default_display);

        Self {
            displays,
            next_display_id: 1,
            layers: HashMap::new(),
            next_layer_id: 0,
            framebuffer_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn dispatch(&mut self, cmd_id: u32) -> u32 {
        log::debug!("VI cmd: {}", cmd_id);
        match cmd_id {
            0 => self.get_display_service(),
            1010 => self.open_display(),
            1020 => self.close_display(),
            2010 => self.create_layer(),
            2020 => self.open_layer(),
            2030 => self.create_stray_layer(),
            5000 => self.create_managed_layer(),
            7000 => self.get_display_vsync_event(),
            _ => {
                log::warn!(
                    "VI.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                    cmd_id
                );
                SUCCESS
            }
        }
    }

    fn get_display_service(&mut self) -> u32 {
        log::debug!("VI::GetDisplayService");
        SUCCESS
    }

    fn open_display(&mut self) -> u32 {
        log::info!("VI::OpenDisplay");
        if let Some(display) = self.displays.get_mut(&0) {
            display.open = true;
        }
        SUCCESS
    }

    fn close_display(&mut self) -> u32 {
        log::info!("VI::CloseDisplay");
        if let Some(display) = self.displays.get_mut(&0) {
            display.open = false;
        }
        SUCCESS
    }

    fn create_layer(&mut self) -> u32 {
        log::info!("VI::CreateLayer");
        let layer = Layer {
            id: self.next_layer_id,
            display_id: 0,
            width: 1280,
            height: 720,
        };
        self.layers.insert(self.next_layer_id, layer);
        self.next_layer_id += 1;
        SUCCESS
    }

    fn open_layer(&mut self) -> u32 {
        log::info!("VI::OpenLayer");
        SUCCESS
    }

    fn create_stray_layer(&mut self) -> u32 {
        log::info!("VI::CreateStrayLayer");
        SUCCESS
    }

    fn create_managed_layer(&mut self) -> u32 {
        log::info!("VI::CreateManagedLayer");
        let layer = Layer {
            id: self.next_layer_id,
            display_id: 0,
            width: 1280,
            height: 720,
        };
        self.layers.insert(self.next_layer_id, layer);
        self.next_layer_id += 1;
        SUCCESS
    }

    fn get_display_vsync_event(&mut self) -> u32 {
        log::debug!("VI::GetDisplayVsyncEvent");
        SUCCESS
    }

    pub fn get_framebuffers(&self) -> Vec<Vec<u8>> {
        self.framebuffer_queue.lock().clone()
    }

    pub fn submit_framebuffer(&self, data: Vec<u8>) {
        let mut queue = self.framebuffer_queue.lock();
        queue.push(data);
        if queue.len() > 3 {
            queue.remove(0);
        }
    }
}

impl Default for DisplayService {
    fn default() -> Self {
        Self::new()
    }
}

use crate::common::result::SUCCESS;
use std::collections::HashMap;

pub struct DisplayService {
    displays: HashMap<u64, Display>,
    next_display_id: u64,
    layers: HashMap<u64, Layer>,
    next_layer_id: u64,
}

struct Display {
    id: u64,
    name: String,
    width: u32,
    height: u32,
}

struct Layer {
    id: u64,
    display_id: u64,
}

impl DisplayService {
    pub fn new() -> Self {
        let mut displays = HashMap::new();
        let default_display = Display {
            id: 0,
            name: "Default".to_string(),
            width: 1280,
            height: 720,
        };
        displays.insert(0, default_display);

        Self {
            displays,
            next_display_id: 1,
            layers: HashMap::new(),
            next_layer_id: 0,
        }
    }

    pub fn dispatch(&mut self, cmd_id: u32) -> u32 {
        log::info!("vi cmd: {}", cmd_id);
        match cmd_id {
            1010 => self.open_display(),
            1020 => self.close_display(),
            2020 => self.open_layer(),
            2030 => self.create_stray_layer(),
            7000 => self.get_display_vsync_event(),
            _ => {
                log::warn!("unknown vi command: {}", cmd_id);
                SUCCESS
            }
        }
    }

    fn open_display(&mut self) -> u32 {
        log::info!("VI::OpenDisplay");
        SUCCESS
    }

    fn close_display(&mut self) -> u32 {
        log::info!("VI::CloseDisplay");
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

    fn get_display_vsync_event(&mut self) -> u32 {
        log::info!("VI::GetDisplayVsyncEvent");
        SUCCESS
    }
}

impl Default for DisplayService {
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::HashMap;

pub struct ServiceManager {
    services: HashMap<String, u32>,
}

impl ServiceManager {
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
        }
    }

    pub fn register_service(&mut self, name: &str, port: u32) {
        self.services.insert(name.to_string(), port);
    }

    pub fn query_service(&self, name: &str) -> Option<u32> {
        self.services.get(name).copied()
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        match cmd_id {
            0 => 0,
            1 => 0,
            2 => 0,
            _ => {
                log::warn!("unknown sm command: {}", cmd_id);
                1
            }
        }
    }
}

impl Default for ServiceManager {
    fn default() -> Self {
        Self::new()
    }
}

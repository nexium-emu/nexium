use std::collections::HashMap;
use crate::common::result::SUCCESS;

pub struct ServiceManager {
    services: HashMap<String, u32>,
    next_port: u32,
}

impl ServiceManager {
    pub fn new() -> Self {
        let mut sm = Self {
            services: HashMap::new(),
            next_port: 0x100,
        };
        sm.register_builtin_services();
        sm
    }

    fn register_builtin_services(&mut self) {
        let services = vec![
            "hid", "time:s", "time:a", "time:r", "set", "am", "vi:m", "vi:s",
            "audio", "fsp-srv", "nifm:u", "nifm:a", "acc:u", "acc:a", "ns",
            "apm", "bsd:u", "bsd:s", "ssl", "spl:", "pl:u", "pl:s", "pctl:a",
        ];
        for name in services {
            self.register_service(name);
        }
    }

    pub fn register_service(&mut self, name: &str) -> u32 {
        let port = self.next_port;
        self.next_port = self.next_port.wrapping_add(1);
        self.services.insert(name.to_string(), port);
        log::debug!("sm: registered service '{}' on port {:#x}", name, port);
        port
    }

    pub fn query_service(&self, name: &str) -> Option<u32> {
        self.services.get(name).copied()
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        match cmd_id {
            0 => SUCCESS,
            1 => SUCCESS,
            2 => SUCCESS,
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

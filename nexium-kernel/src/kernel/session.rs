use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Session {
    pub handle: u32,
    pub port_name: String,
    pub is_domain: bool,
    pub domain_group: u32,
    pub domain_objects: HashMap<u32, String>,
    pub next_domain_object_id: u32,
}

impl Session {
    pub fn new(handle: u32, port_name: String) -> Self {
        Self {
            handle,
            port_name,
            is_domain: false,
            domain_group: handle,
            domain_objects: HashMap::new(),
            next_domain_object_id: 1,
        }
    }

    pub fn convert_to_domain(&mut self) {
        if !self.is_domain {
            self.is_domain = true;
            self.domain_group = self.handle;
            self.domain_objects.insert(1, self.port_name.clone());
            self.next_domain_object_id = 2;
        }
    }

    pub fn alloc_domain_object(&mut self, service_name: String) -> u32 {
        let id = self.next_domain_object_id;
        self.next_domain_object_id += 1;
        self.domain_objects.insert(id, service_name);
        id
    }

    pub fn service_for_object(&self, object_id: u32) -> Option<&str> {
        self.domain_objects.get(&object_id).map(|s| s.as_str())
    }

    pub fn close_object(&mut self, object_id: u32) {
        self.domain_objects.remove(&object_id);
    }
}

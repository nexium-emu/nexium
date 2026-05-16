#[derive(Clone, Debug)]
pub struct Session {
    pub handle: u32,
    pub port_name: String,
    pub is_domain: bool,
}

impl Session {
    pub fn new(handle: u32, port_name: String) -> Self {
        Self { handle, port_name, is_domain: false }
    }
}

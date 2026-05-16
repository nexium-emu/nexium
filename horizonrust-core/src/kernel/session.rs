#[derive(Clone, Debug)]
pub struct Session {
    pub handle: u32,
    pub port_name: String,
}

impl Session {
    pub fn new(handle: u32, port_name: String) -> Self {
        Self { handle, port_name }
    }
}

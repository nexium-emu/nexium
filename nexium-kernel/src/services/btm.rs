pub struct BluetoothService;

impl BluetoothService {
    pub fn new() -> Self {
        Self
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("btm cmd: {}", cmd_id);
        0
    }
}

impl Default for BluetoothService {
    fn default() -> Self {
        Self::new()
    }
}

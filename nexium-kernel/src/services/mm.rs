pub struct MmService {
    current: u32,
    id: u32,
}

impl MmService {
    pub fn new() -> Self {
        Self { current: 0, id: 1 }
    }

    pub fn dispatch(&mut self, cmd_id: u32, data: &[u8]) -> (u32, Vec<u8>) {
        match cmd_id {
            0 | 1 | 5 => (0, Vec::new()),
            2 => {
                let min = read_u32(data, 0);
                let max = read_u32(data, 4);
                self.current = min;
                log::debug!("mm:u SetAndWaitOld min={:#x} max={:#x}", min, max);
                (0, Vec::new())
            }
            3 | 7 => {
                log::debug!("mm:u Get current={:#x}", self.current);
                (0, self.current.to_le_bytes().to_vec())
            }
            4 => {
                log::debug!("mm:u Initialize id={:#x}", self.id);
                (0, self.id.to_le_bytes().to_vec())
            }
            6 => {
                let input_id = read_u32(data, 0);
                let min = read_u32(data, 4);
                let max = read_u32(data, 8);
                self.current = min;
                log::debug!(
                    "mm:u SetAndWait id={:#x} min={:#x} max={:#x}",
                    input_id,
                    min,
                    max
                );
                (0, Vec::new())
            }
            _ => {
                log::warn!("unknown mm:u command: {}", cmd_id);
                (0, Vec::new())
            }
        }
    }
}

impl Default for MmService {
    fn default() -> Self {
        Self::new()
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    if offset + 4 <= data.len() {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        0
    }
}

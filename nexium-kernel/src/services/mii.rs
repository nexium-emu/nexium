const CHAR_INFO_SIZE: usize = 0x58;

pub struct MiiService {
    next_random_id: u32,
    interface_version: u32,
}

impl MiiService {
    pub fn new() -> Self {
        Self {
            next_random_id: 0,
            interface_version: 0,
        }
    }
    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("mii cmd: {}", cmd_id);
        0
    }

    pub fn set_interface_version(&mut self, version: u32) {
        self.interface_version = version;
    }

    pub fn build_random(&mut self, requested_gender: u32) -> [u8; CHAR_INFO_SIZE] {
        let id = self.next_random_id;
        self.next_random_id = self.next_random_id.wrapping_add(1);
        build_char_info(0x100 + id, requested_gender)
    }

    pub fn build_default(&self, index: u32) -> [u8; CHAR_INFO_SIZE] {
        build_char_info(index, index & 1)
    }
}

fn build_char_info(id: u32, requested_gender: u32) -> [u8; CHAR_INFO_SIZE] {
    let mut info = [0u8; CHAR_INFO_SIZE];

    info[0..4].copy_from_slice(&id.to_le_bytes());
    info[4..16].copy_from_slice(&[
        0x4e, 0x65, 0x40, 0x69, 0x80, 0x6d, 0x69, 0x69, 0x4e, 0x58, 0x4d, 0x49,
    ]);
    let nickname = format!("NeXium{:03}", id % 1000);
    for (index, unit) in nickname.encode_utf16().take(10).enumerate() {
        let offset = 0x10 + index * 2;
        info[offset..offset + 2].copy_from_slice(&unit.to_le_bytes());
    }

    let gender = match requested_gender {
        0 | 1 => requested_gender as u8,
        _ => (id & 1) as u8,
    };
    info[0x27] = (id % 12) as u8;
    info[0x28] = gender;
    info[0x29] = 64;
    info[0x2a] = 64;
    info[0x31] = if gender == 0 { 33 } else { 12 };
    info[0x32] = 1;
    info[0x34] = if gender == 0 { 2 } else { 4 };
    info[0x35] = 8;
    info[0x36] = 4;
    info[0x37] = 3;
    info[0x38] = 4;
    info[0x39] = 2;
    info[0x3a] = 12;
    info[0x3b] = if gender == 0 { 6 } else { 0 };
    info[0x3c] = 1;
    info[0x3d] = 4;
    info[0x3e] = 3;
    info[0x3f] = 6;
    info[0x40] = 2;
    info[0x41] = 10;
    info[0x42] = 1;
    info[0x43] = 4;
    info[0x44] = 9;
    info[0x45] = 23;
    info[0x46] = 19;
    info[0x47] = 4;
    info[0x48] = 3;
    info[0x49] = 13;
    info[0x4a] = 1;
    info[0x4d] = 4;
    info[0x4e] = 10;
    info[0x50] = 8;
    info[0x51] = 4;
    info[0x52] = 10;
    info[0x54] = 4;
    info[0x55] = 2;
    info[0x56] = 20;
    info
}

impl Default for MiiService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{build_char_info, CHAR_INFO_SIZE};

    #[test]
    fn generated_char_info_has_valid_shape_and_terminated_name() {
        let info = build_char_info(7, 2);
        assert_eq!(info.len(), CHAR_INFO_SIZE);
        assert_ne!(&info[0..16], &[0; 16]);
        assert_eq!(info[6] >> 4, 4);
        assert_eq!(info[8] & 0xc0, 0x80);
        assert_eq!(&info[0x10..0x1c], b"N\0e\0X\0i\0u\0m\0");
        assert_eq!(&info[0x24..0x26], &[0, 0]);
        assert!(info[0x28] <= 1);
        assert_eq!(info[0x29], 64);
        assert_eq!(info[0x2a], 64);
    }
}

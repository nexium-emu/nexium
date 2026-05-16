pub struct Parcel {
    data: Vec<u8>,
    pos: usize,
}

impl Parcel {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    pub fn read_u32(&mut self) -> Option<u32> {
        if self.pos + 4 <= self.data.len() {
            let bytes = &self.data[self.pos..self.pos + 4];
            self.pos += 4;
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            None
        }
    }

    pub fn read_u64(&mut self) -> Option<u64> {
        if self.pos + 8 <= self.data.len() {
            let bytes = &self.data[self.pos..self.pos + 8];
            self.pos += 8;
            Some(u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            None
        }
    }

    pub fn read_bytes(&mut self, len: usize) -> Option<&[u8]> {
        if self.pos + len <= self.data.len() {
            let result = &self.data[self.pos..self.pos + len];
            self.pos += len;
            Some(result)
        } else {
            None
        }
    }

    pub fn write_u32(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_u64(&mut self, value: u64) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

pub struct ParcelReader {
    data: Vec<u8>,
    pos: usize,
}

impl ParcelReader {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data, pos: 0 }
    }

    pub fn read_u32(&mut self) -> Result<u32, &'static str> {
        if self.pos + 4 <= self.data.len() {
            let bytes = &self.data[self.pos..self.pos + 4];
            self.pos += 4;
            Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        } else {
            Err("not enough data")
        }
    }

    pub fn read_u64(&mut self) -> Result<u64, &'static str> {
        if self.pos + 8 <= self.data.len() {
            let bytes = &self.data[self.pos..self.pos + 8];
            self.pos += 8;
            Ok(u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            Err("not enough data")
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

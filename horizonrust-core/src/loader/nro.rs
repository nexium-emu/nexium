use std::io::Read;
use byteorder::{LittleEndian, ReadBytesExt};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NroHeader {
    pub magic: u32,
    pub version: u32,
    pub size: u32,
    pub flags: u32,
    pub text_offset: u32,
    pub text_size: u32,
    pub ro_offset: u32,
    pub ro_size: u32,
    pub data_offset: u32,
    pub data_size: u32,
    pub bss_size: u32,
    pub mod0_offset: u32,
    pub padding: [u8; 12],
    pub build_id: [u8; 32],
}

impl NroHeader {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 128 {
            return Err("NRO header too small".to_string());
        }

        let mut cursor = &data[..];
        let magic = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

        if magic != 0x304F524E {
            return Err(format!("Invalid NRO magic: {:#x}", magic));
        }

        cursor = &data[..];
        let magic = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let version = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let flags = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let text_offset = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let text_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let ro_offset = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let ro_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let data_offset = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let data_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let bss_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let mod0_offset = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

        let mut padding = [0u8; 12];
        cursor.read_exact(&mut padding).map_err(|e| e.to_string())?;

        let mut build_id = [0u8; 32];
        cursor.read_exact(&mut build_id).map_err(|e| e.to_string())?;

        Ok(NroHeader {
            magic,
            version,
            size,
            flags,
            text_offset,
            text_size,
            ro_offset,
            ro_size,
            data_offset,
            data_size,
            bss_size,
            mod0_offset,
            padding,
            build_id,
        })
    }
}

pub struct NroSegment {
    pub offset: u32,
    pub size: u32,
    pub data: Vec<u8>,
}

pub struct Nro {
    pub header: NroHeader,
    pub text: NroSegment,
    pub ro: NroSegment,
    pub data: NroSegment,
    pub bss_size: u32,
}

impl Nro {
    fn unwrap_homebrew(data: &[u8]) -> Result<Vec<u8>, String> {
        if data.len() < 16 {
            return Ok(data.to_vec());
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic == 0x304F524E {
            return Ok(data.to_vec());
        }

        if data.len() >= 32 {
            let magic_at_16 = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
            if magic_at_16 == 0x304F524E {
                log::debug!("Detected homebrew wrapper, skipping 16-byte header");
                return Ok(data[16..].to_vec());
            }
        }

        Ok(data.to_vec())
    }

    pub fn parse(data: &[u8]) -> Result<Self, String> {
        let nro_data = Self::unwrap_homebrew(data)?;
        let header = NroHeader::from_bytes(&nro_data)?;

        log::debug!("NRO: magic={:#x}, version={}, size={}", header.magic, header.version, header.size);

        // NRO header is 128 bytes (0x80). Segment offsets may be relative to after the header.
        // If offsets are < 0x80, assume they're relative and add 0x80.
        const HEADER_SIZE: u32 = 128;
        let text_offset = if header.text_offset < HEADER_SIZE { header.text_offset + HEADER_SIZE } else { header.text_offset };
        let ro_offset = if header.ro_offset < HEADER_SIZE { header.ro_offset + HEADER_SIZE } else { header.ro_offset };
        let data_offset = if header.data_offset < HEADER_SIZE { header.data_offset + HEADER_SIZE } else { header.data_offset };

        log::debug!("NRO: text=[{:#x}, {:#x}), ro=[{:#x}, {:#x}), data=[{:#x}, {:#x})",
                   text_offset, text_offset + header.text_size,
                   ro_offset, ro_offset + header.ro_size,
                   data_offset, data_offset + header.data_size);

        if text_offset + header.text_size > nro_data.len() as u32 {
            return Err("Text segment out of bounds".to_string());
        }
        if ro_offset + header.ro_size > nro_data.len() as u32 {
            return Err("RO segment out of bounds".to_string());
        }
        if data_offset + header.data_size > nro_data.len() as u32 {
            return Err("Data segment out of bounds".to_string());
        }

        let text = NroSegment {
            offset: text_offset,
            size: header.text_size,
            data: nro_data[text_offset as usize..(text_offset + header.text_size) as usize].to_vec(),
        };

        let ro = NroSegment {
            offset: ro_offset,
            size: header.ro_size,
            data: nro_data[ro_offset as usize..(ro_offset + header.ro_size) as usize].to_vec(),
        };

        let data_seg = NroSegment {
            offset: data_offset,
            size: header.data_size,
            data: nro_data[data_offset as usize..(data_offset + header.data_size) as usize].to_vec(),
        };

        Ok(Nro {
            header,
            text,
            ro,
            data: data_seg,
            bss_size: header.bss_size,
        })
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        Self::parse(&data)
    }

    pub fn total_memory_size(&self) -> u64 {
        (self.text.size + self.ro.size + self.data.size + self.bss_size) as u64
    }

    pub fn get_section(&self, name: &str) -> Option<&NroSegment> {
        match name {
            "text" => Some(&self.text),
            "ro" => Some(&self.ro),
            "data" => Some(&self.data),
            _ => None,
        }
    }
}

use std::io::Read;
use byteorder::{LittleEndian, ReadBytesExt};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NroSegment {
    pub file_off: u32,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NroHeader {
    pub _entry_pad: [u8; 4],
    pub module_header_offset: u32,
    pub magic_ext1: [u8; 4],
    pub magic_ext2: [u8; 4],
    pub magic: [u8; 4],
    pub _pad1: [u8; 4],
    pub file_size: u32,
    pub _pad2: [u8; 4],
    pub text: NroSegment,
    pub rodata: NroSegment,
    pub data: NroSegment,
    pub bss_size: u32,
    pub trailing_a: [u8; 32],
    pub trailing_b: [u8; 32],
    pub trailing_c: [u8; 4],
}

impl NroHeader {
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 128 {
            return Err("NRO header too small".to_string());
        }

        let mut cursor = &data[..];
        let mut entry_pad = [0u8; 4];
        cursor.read_exact(&mut entry_pad).map_err(|e| e.to_string())?;
        let module_header_offset = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let mut magic_ext1 = [0u8; 4];
        cursor.read_exact(&mut magic_ext1).map_err(|e| e.to_string())?;
        let mut magic_ext2 = [0u8; 4];
        cursor.read_exact(&mut magic_ext2).map_err(|e| e.to_string())?;
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| e.to_string())?;

        if magic != [0x4E, 0x52, 0x4F, 0x30] {  // "NRO0"
            return Err(format!("Invalid NRO magic: {:?}", magic));
        }

        let mut pad1 = [0u8; 4];
        cursor.read_exact(&mut pad1).map_err(|e| e.to_string())?;
        let file_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let mut pad2 = [0u8; 4];
        cursor.read_exact(&mut pad2).map_err(|e| e.to_string())?;

        let text_off = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let text_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let rodata_off = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let rodata_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let data_off = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let data_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
        let bss_size = cursor.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;

        let mut trailing_a = [0u8; 32];
        cursor.read_exact(&mut trailing_a).map_err(|e| e.to_string())?;
        let mut trailing_b = [0u8; 32];
        cursor.read_exact(&mut trailing_b).map_err(|e| e.to_string())?;
        let mut trailing_c = [0u8; 4];
        cursor.read_exact(&mut trailing_c).map_err(|e| e.to_string())?;

        Ok(NroHeader {
            _entry_pad: entry_pad,
            module_header_offset,
            magic_ext1,
            magic_ext2,
            magic,
            _pad1: pad1,
            file_size,
            _pad2: pad2,
            text: NroSegment { file_off: text_off, size: text_size },
            rodata: NroSegment { file_off: rodata_off, size: rodata_size },
            data: NroSegment { file_off: data_off, size: data_size },
            bss_size,
            trailing_a,
            trailing_b,
            trailing_c,
        })
    }
}

pub struct NroSegmentData {
    pub offset: u32,
    pub size: u32,
    pub data: Vec<u8>,
}

pub struct Nro {
    pub header: NroHeader,
    pub text: NroSegmentData,
    pub ro: NroSegmentData,
    pub data: NroSegmentData,
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
        log::debug!("NRO file size: {}", data.len());
        let nro_data = Self::unwrap_homebrew(data)?;
        log::debug!("After unwrap_homebrew: size={}, first 4 bytes: {:02x} {:02x} {:02x} {:02x}",
                   nro_data.len(), nro_data[0], nro_data[1], nro_data[2], nro_data[3]);
        let header = NroHeader::from_bytes(&nro_data)?;

        log::debug!("NRO: magic={:?}, file_size={}", header.magic, header.file_size);
        log::debug!("NRO: text=[{:#x}, {:#x}), rodata=[{:#x}, {:#x}), data=[{:#x}, {:#x})",
                   header.text.file_off, header.text.file_off + header.text.size,
                   header.rodata.file_off, header.rodata.file_off + header.rodata.size,
                   header.data.file_off, header.data.file_off + header.data.size);

        if header.text.file_off as usize + header.text.size as usize > nro_data.len() {
            return Err("Text segment out of bounds".to_string());
        }
        if header.rodata.file_off as usize + header.rodata.size as usize > nro_data.len() {
            return Err("RO segment out of bounds".to_string());
        }
        if header.data.file_off as usize + header.data.size as usize > nro_data.len() {
            return Err("Data segment out of bounds".to_string());
        }

        let text = NroSegmentData {
            offset: header.text.file_off,
            size: header.text.size,
            data: nro_data[header.text.file_off as usize..(header.text.file_off as usize + header.text.size as usize)].to_vec(),
        };

        let ro = NroSegmentData {
            offset: header.rodata.file_off,
            size: header.rodata.size,
            data: nro_data[header.rodata.file_off as usize..(header.rodata.file_off as usize + header.rodata.size as usize)].to_vec(),
        };

        let data_seg = NroSegmentData {
            offset: header.data.file_off,
            size: header.data.size,
            data: nro_data[header.data.file_off as usize..(header.data.file_off as usize + header.data.size as usize)].to_vec(),
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

    pub fn get_section(&self, name: &str) -> Option<&NroSegmentData> {
        match name {
            "text" => Some(&self.text),
            "ro" => Some(&self.ro),
            "data" => Some(&self.data),
            _ => None,
        }
    }
}

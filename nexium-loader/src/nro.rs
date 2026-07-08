use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

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
        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;

        if magic != 0x304F524E {
            return Err(format!("Invalid NRO magic: {:#x}", magic));
        }

        cursor = &data[..];
        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let version = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let flags = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let text_offset = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let text_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let ro_offset = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let ro_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let data_offset = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let data_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let bss_size = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;
        let mod0_offset = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| e.to_string())?;

        let mut padding = [0u8; 12];
        cursor.read_exact(&mut padding).map_err(|e| e.to_string())?;

        let mut build_id = [0u8; 32];
        cursor
            .read_exact(&mut build_id)
            .map_err(|e| e.to_string())?;

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

#[derive(Debug, Clone)]
pub struct NroSegmentView {
    pub offset: u32,
    pub size: u32,
    pub mmap_range: Range<usize>,
}

pub struct Nro {
    pub header: NroHeader,
    pub text: NroSegmentView,
    pub ro: NroSegmentView,
    pub data: NroSegmentView,
    pub bss_size: u32,
    mmap: Arc<Mmap>,
    nro_offset: usize,
    romfs_range: Option<Range<usize>>,
}

impl Nro {
    pub fn bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    pub fn nro_bytes(&self) -> &[u8] {
        &self.mmap[self.nro_offset..]
    }

    pub fn romfs(&self) -> &[u8] {
        match &self.romfs_range {
            Some(r) => &self.mmap[r.clone()],
            None => &[],
        }
    }

    pub fn mmap_arc(&self) -> Arc<Mmap> {
        self.mmap.clone()
    }

    pub fn romfs_range(&self) -> Option<Range<usize>> {
        self.romfs_range.clone()
    }

    pub fn text_data(&self) -> &[u8] {
        &self.mmap[self.text.mmap_range.clone()]
    }

    pub fn ro_data(&self) -> &[u8] {
        &self.mmap[self.ro.mmap_range.clone()]
    }

    pub fn data_data(&self) -> &[u8] {
        &self.mmap[self.data.mmap_range.clone()]
    }

    fn detect_nro_offset(mmap: &Mmap) -> Result<usize, String> {
        if mmap.len() < 16 {
            return Err("file too small to contain NRO".to_string());
        }
        let magic0 = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
        if magic0 == 0x304F524E {
            return Ok(0);
        }
        if mmap.len() >= 32 {
            let magic16 = u32::from_le_bytes([mmap[16], mmap[17], mmap[18], mmap[19]]);
            if magic16 == 0x304F524E {
                log::debug!("Detected homebrew wrapper, NRO body starts at offset 16");
                return Ok(16);
            }
        }
        Err("no NRO magic at offset 0 or 16".to_string())
    }

    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("Failed to open NRO: {}", e))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap NRO: {}", e))?;
        Self::parse_mmap(Arc::new(mmap))
    }

    pub fn parse_mmap(mmap: Arc<Mmap>) -> Result<Self, String> {
        let nro_offset = Self::detect_nro_offset(&mmap)?;
        let nro_slice = &mmap[nro_offset..];
        let header = NroHeader::from_bytes(nro_slice)?;

        log::debug!(
            "NRO: magic={:#x}, version={}, size={}, file_offset={}",
            header.magic,
            header.version,
            header.size,
            nro_offset
        );

        const HEADER_SIZE: u32 = 128;
        let text_offset = if header.text_offset < HEADER_SIZE {
            header.text_offset + HEADER_SIZE
        } else {
            header.text_offset
        };
        let ro_offset = if header.ro_offset < HEADER_SIZE {
            header.ro_offset + HEADER_SIZE
        } else {
            header.ro_offset
        };
        let data_offset = if header.data_offset < HEADER_SIZE {
            header.data_offset + HEADER_SIZE
        } else {
            header.data_offset
        };

        log::debug!(
            "NRO: text=[{:#x}, {:#x}), ro=[{:#x}, {:#x}), data=[{:#x}, {:#x})",
            text_offset,
            text_offset + header.text_size,
            ro_offset,
            ro_offset + header.ro_size,
            data_offset,
            data_offset + header.data_size
        );

        if (text_offset + header.text_size) as usize > nro_slice.len() {
            return Err("Text segment out of bounds".to_string());
        }
        if (ro_offset + header.ro_size) as usize > nro_slice.len() {
            return Err("RO segment out of bounds".to_string());
        }
        if (data_offset + header.data_size) as usize > nro_slice.len() {
            return Err("Data segment out of bounds".to_string());
        }

        let text = NroSegmentView {
            offset: text_offset,
            size: header.text_size,
            mmap_range: (nro_offset + text_offset as usize)
                ..(nro_offset + (text_offset + header.text_size) as usize),
        };
        let ro = NroSegmentView {
            offset: ro_offset,
            size: header.ro_size,
            mmap_range: (nro_offset + ro_offset as usize)
                ..(nro_offset + (ro_offset + header.ro_size) as usize),
        };
        let data = NroSegmentView {
            offset: data_offset,
            size: header.data_size,
            mmap_range: (nro_offset + data_offset as usize)
                ..(nro_offset + (data_offset + header.data_size) as usize),
        };

        let romfs_range = parse_asset_romfs(&mmap, header.size);

        Ok(Nro {
            header,
            text,
            ro,
            data,
            bss_size: header.bss_size,
            mmap,
            nro_offset,
            romfs_range,
        })
    }

    pub fn total_memory_size(&self) -> u64 {
        (self.text.size + self.ro.size + self.data.size + self.bss_size) as u64
    }
}

pub struct NroMetadata {
    pub title: String,
    pub author: String,
    pub icon_jpeg: Option<Vec<u8>>,
}

pub fn read_nro_metadata(path: &std::path::Path) -> Option<NroMetadata> {
    let file = std::fs::File::open(path).ok()?;
    let mmap = unsafe { Mmap::map(&file) }.ok()?;
    let nro_offset = Nro::detect_nro_offset(&mmap).ok()?;
    let header = NroHeader::from_bytes(&mmap[nro_offset..]).ok()?;

    let asset_start = header.size as usize;
    if mmap.len() < asset_start + 56 {
        return None;
    }
    let asset = &mmap[asset_start..];
    if u32::from_le_bytes([asset[0], asset[1], asset[2], asset[3]]) != 0x54_45_53_41 {
        return None;
    }

    let read_u64 = |o: usize| {
        u64::from_le_bytes([
            asset[o],
            asset[o + 1],
            asset[o + 2],
            asset[o + 3],
            asset[o + 4],
            asset[o + 5],
            asset[o + 6],
            asset[o + 7],
        ]) as usize
    };
    let icon_off = read_u64(8);
    let icon_size = read_u64(16);
    let nacp_off = read_u64(24);
    let nacp_size = read_u64(32);

    let icon_jpeg = (icon_size > 0)
        .then(|| mmap.get(asset_start + icon_off..asset_start + icon_off + icon_size))
        .flatten()
        .map(|b| b.to_vec());

    let (title, author) = mmap
        .get(asset_start + nacp_off..asset_start + nacp_off + nacp_size)
        .filter(|n| n.len() >= 0x300)
        .map(parse_nacp)
        .unwrap_or_default();

    Some(NroMetadata {
        title,
        author,
        icon_jpeg,
    })
}

pub(crate) fn parse_nacp(nacp: &[u8]) -> (String, String) {
    for i in 0..16 {
        let base = i * 0x300;
        if base + 0x300 > nacp.len() {
            break;
        }
        let name = read_cstr(&nacp[base..base + 0x200]);
        if !name.is_empty() {
            return (name, read_cstr(&nacp[base + 0x200..base + 0x300]));
        }
    }
    (String::new(), String::new())
}

pub(crate) fn read_cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).trim().to_string()
}

fn parse_asset_romfs(mmap: &Mmap, nro_size: u32) -> Option<Range<usize>> {
    let asset_start = nro_size as usize;
    if mmap.len() < asset_start + 56 {
        return None;
    }
    let asset = &mmap[asset_start..];
    let magic = u32::from_le_bytes([asset[0], asset[1], asset[2], asset[3]]);
    if magic != 0x54_45_53_41 {
        log::debug!("NRO: no ASET section (magic={:#x})", magic);
        return None;
    }
    let romfs_off = u64::from_le_bytes([
        asset[40], asset[41], asset[42], asset[43], asset[44], asset[45], asset[46], asset[47],
    ]) as usize;
    let romfs_size = u64::from_le_bytes([
        asset[48], asset[49], asset[50], asset[51], asset[52], asset[53], asset[54], asset[55],
    ]) as usize;
    if romfs_size == 0 {
        log::debug!("NRO: ASET section present but romfs size is 0");
        return None;
    }
    let abs_start = asset_start + romfs_off;
    let abs_end = abs_start + romfs_size;
    if abs_end > mmap.len() {
        log::warn!(
            "NRO: romfs section [{:#x}..{:#x}) exceeds mmap size {:#x}",
            abs_start,
            abs_end,
            mmap.len()
        );
        return None;
    }
    log::info!(
        "NRO: romfs ({} bytes) at file offset {:#x} (zero-copy slice)",
        romfs_size,
        abs_start
    );
    Some(abs_start..abs_end)
}

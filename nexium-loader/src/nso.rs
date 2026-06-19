use crate::bin_read::{u32at, slice};

pub const NSO0_MAGIC: u32 = 0x304F534E;
const PAGE: u32 = 0x1000;

fn page_align(size: u32) -> u32 {
    (size.wrapping_add(PAGE - 1)) & !(PAGE - 1)
}

#[derive(Clone, Copy, Debug)]
pub struct NsoSegment {
    pub file_offset: u32,
    pub mem_offset: u32,
    pub decompressed_size: u32,
    pub compressed_size: u32,
    pub compressed: bool,
}

pub struct Nso {
    pub text: NsoSegment,
    pub ro: NsoSegment,
    pub data: NsoSegment,
    pub bss_size: u32,
    pub image_size: u32,
    pub module_image: Vec<u8>,
}

impl Nso {
    pub fn parse(region: &[u8]) -> Result<Self, String> {
        let magic = u32at(region, 0)?;
        if magic != NSO0_MAGIC {
            return Err(format!("NSO magic {:#010x} is not NSO0", magic));
        }
        let flags = u32at(region, 0x0C)?;

        let read_seg = |i: usize| -> Result<NsoSegment, String> {
            let h = 0x10 + i * 0x10;
            Ok(NsoSegment {
                file_offset: u32at(region, h)?,
                mem_offset: u32at(region, h + 4)?,
                decompressed_size: u32at(region, h + 8)?,
                compressed_size: u32at(region, 0x60 + i * 4)?,
                compressed: (flags >> i) & 1 != 0,
            })
        };
        let text = read_seg(0)?;
        let ro = read_seg(1)?;
        let data = read_seg(2)?;
        let bss_size = u32at(region, 0x10 + 2 * 0x10 + 0x0C)?;

        let mut end: u32 = 0;
        for s in [&text, &ro, &data] {
            let seg_end = s.mem_offset.checked_add(s.decompressed_size).ok_or("NSO segment end overflow")?;
            end = end.max(seg_end);
        }
        let image_size = page_align(end.checked_add(bss_size).ok_or("NSO image size overflow")?);

        let mut module_image = vec![0u8; image_size as usize];
        for s in [&text, &ro, &data] {
            let src = slice(region, s.file_offset as usize..(s.file_offset as usize + s.compressed_size as usize))?;
            let bytes = if s.compressed {
                let out = lz4_flex::block::decompress(src, s.decompressed_size as usize)
                    .map_err(|e| format!("NSO LZ4 decompress failed: {}", e))?;
                if out.len() != s.decompressed_size as usize {
                    return Err(format!("NSO segment decompressed to {} bytes, expected {}", out.len(), s.decompressed_size));
                }
                out
            } else {
                src[..s.decompressed_size as usize].to_vec()
            };
            let dst = s.mem_offset as usize;
            module_image
                .get_mut(dst..dst + bytes.len())
                .ok_or("NSO segment destination out of image")?
                .copy_from_slice(&bytes);
        }

        Ok(Self { text, ro, data, bss_size, image_size, module_image })
    }

    pub fn data_region_size(&self) -> u32 {
        let raw = self.data.decompressed_size.saturating_add(self.bss_size);
        page_align(self.image_size.saturating_sub(self.data.mem_offset).max(raw))
    }
}

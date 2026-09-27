use crate::bin_read::{slice, u32at};

pub const NSO0_MAGIC: u32 = 0x304F534E;
const PAGE: u32 = 0x1000;

fn page_align(size: u32) -> Result<u32, String> {
    size.checked_add(PAGE - 1).map(|size| size & !(PAGE - 1))
        .ok_or_else(|| "NSO image alignment overflow".to_string())
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
    pub build_id: [u8; 32],
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
        let mut build_id = [0u8; 32];
        build_id.copy_from_slice(slice(region, 0x40..0x60)?);

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
            let seg_end = s
                .mem_offset
                .checked_add(s.decompressed_size)
                .ok_or("NSO segment end overflow")?;
            end = end.max(seg_end);
        }
        let image_size = page_align(end.checked_add(bss_size).ok_or("NSO image size overflow")?)?;

        for segment in [&text, &ro, &data] {
            if !segment.compressed && segment.compressed_size < segment.decompressed_size {
                return Err("NSO uncompressed segment is shorter than its declared size".to_string());
            }
        }
        let mut module_image = Vec::new();
        module_image.try_reserve_exact(image_size as usize)
            .map_err(|error| format!("NSO image allocation failed: {error}"))?;
        module_image.resize(image_size as usize, 0u8);
        for s in [&text, &ro, &data] {
            let src = slice(
                region,
                s.file_offset as usize..(s.file_offset as usize + s.compressed_size as usize),
            )?;
            let bytes = if s.compressed {
                let out = lz4_flex::block::decompress(src, s.decompressed_size as usize)
                    .map_err(|e| format!("NSO LZ4 decompress failed: {}", e))?;
                if out.len() != s.decompressed_size as usize {
                    return Err(format!(
                        "NSO segment decompressed to {} bytes, expected {}",
                        out.len(),
                        s.decompressed_size
                    ));
                }
                out
            } else {
                src.get(..s.decompressed_size as usize)
                    .ok_or("NSO uncompressed segment payload is truncated")?.to_vec()
            };
            let dst = s.mem_offset as usize;
            module_image
                .get_mut(dst..dst + bytes.len())
                .ok_or("NSO segment destination out of image")?
                .copy_from_slice(&bytes);
        }

        Ok(Self {
            build_id,
            text,
            ro,
            data,
            bss_size,
            image_size,
            module_image,
        })
    }

    pub fn data_region_size(&self) -> u32 {
        let raw = self.data.decompressed_size.saturating_add(self.bss_size);
        page_align(
            self.image_size
                .saturating_sub(self.data.mem_offset)
                .max(raw),
        ).unwrap_or(self.image_size)
    }
}

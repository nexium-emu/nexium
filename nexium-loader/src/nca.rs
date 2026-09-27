use memmap2::Mmap;
use std::ops::Range;
use std::sync::Arc;

use crate::bin_read::{u32at, u64at, u8at};
use crate::romfs;

pub const DNCA_MAGIC: u32 = 0x41434E44;
pub const NCA3_MAGIC: u32 = 0x3341434E;

const SECTOR: u64 = 0x200;
const HEADER_SIZE: usize = 0x400;
const FS_HEADER_SIZE: usize = 0x200;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NcaContentType {
    Program,
    Meta,
    Control,
    Manual,
    Data,
    PublicData,
    Unknown,
}

impl NcaContentType {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => NcaContentType::Program,
            1 => NcaContentType::Meta,
            2 => NcaContentType::Control,
            3 => NcaContentType::Manual,
            4 => NcaContentType::Data,
            5 => NcaContentType::PublicData,
            _ => NcaContentType::Unknown,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NcaFsType {
    RomFs,
    PartitionFs,
}

#[derive(Clone, Debug)]
pub struct NcaFsSection {
    pub index: usize,
    pub fs_type: NcaFsType,
    pub hash_type: u8,
    pub encryption_type: u8,
    pub section_range: Range<usize>,
    pub fs_data_range: Range<usize>,
    pub compression: Option<NcaCompressionInfo>,
    pub patch: Option<NcaPatchInfo>,
    pub sparse: bool,
}

#[derive(Clone, Debug)]
pub struct NcaPatchInfo {
    pub bucket_offset: u64,
    pub bucket_size: u64,
    pub entry_count: u32,
}

#[derive(Clone, Debug)]
pub struct NcaCompressionInfo {
    pub bucket_offset: u64,
    pub bucket_size: u64,
    pub entry_count: u32,
}

pub struct Nca {
    mmap: Arc<Mmap>,
    pub nca_base: usize,
    pub content_type: NcaContentType,
    pub program_id: u64,
    pub content_size: u64,
    pub sections: Vec<NcaFsSection>,
}

impl Nca {
    pub fn parse(mmap: Arc<Mmap>, nca_base: usize) -> Result<Self, String> {
        let buf = &mmap[..];
        let magic = u32at(buf, nca_base + 0x200)?;
        if magic != DNCA_MAGIC {
            if magic == NCA3_MAGIC {
                return Err(format!(
                    "NCA at {:#x} is encrypted (NCA3); only decrypted DNCA supported",
                    nca_base
                ));
            }
            return Err(format!(
                "NCA magic {:#010x} at {:#x} is not DNCA",
                magic,
                nca_base + 0x200
            ));
        }

        let content_type = NcaContentType::from_u8(u8at(buf, nca_base + 0x205)?);
        let content_size = u64at(buf, nca_base + 0x208)?;
        let program_id = u64at(buf, nca_base + 0x210)?;

        let mut sections = Vec::new();
        for i in 0..4 {
            let fs_info = nca_base + 0x240 + i * 0x10;
            let start_sector = u32at(buf, fs_info)? as u64;
            let end_sector = u32at(buf, fs_info + 4)? as u64;
            if end_sector <= start_sector {
                continue;
            }

            let section_start = (nca_base as u64)
                .checked_add(
                    start_sector
                        .checked_mul(SECTOR)
                        .ok_or("section start overflow")?,
                )
                .ok_or("section start overflow")?;
            let section_end = (nca_base as u64)
                .checked_add(
                    end_sector
                        .checked_mul(SECTOR)
                        .ok_or("section end overflow")?,
                )
                .ok_or("section end overflow")?;
            let section_end = if section_end > buf.len() as u64 {
                log::warn!(
                    "NCA section {} end {:#x} clamped to EOF {:#x} (over by {:#x})",
                    i,
                    section_end,
                    buf.len(),
                    section_end - buf.len() as u64
                );
                buf.len() as u64
            } else {
                section_end
            };
            if section_start > section_end {
                continue;
            }
            let section_len = section_end - section_start;

            let fh = nca_base + HEADER_SIZE + i * FS_HEADER_SIZE;
            let fs_header = crate::bin_read::slice(buf, fh..fh + FS_HEADER_SIZE)?;
            let fs_type = match u8at(fs_header, 0x02)? {
                0 => NcaFsType::RomFs,
                1 => NcaFsType::PartitionFs,
                other => return Err(format!("unknown NCA fs_type {} in section {}", other, i)),
            };
            let hash_type = u8at(fs_header, 0x03)?;
            let encryption_type = u8at(fs_header, 0x04)?;
            let indirect_offset = crate::bin_read::i64at(fs_header, 0x100)?;
            let indirect_size = crate::bin_read::i64at(fs_header, 0x108)?;
            let aes_ctr_ex_offset = crate::bin_read::i64at(fs_header, 0x120)?;
            let aes_ctr_ex_size = crate::bin_read::i64at(fs_header, 0x128)?;
            if indirect_size != 0 || aes_ctr_ex_size != 0 {
                log::info!(
                    "NCA section {} patch indirect={:#x}+{:#x} aes_ctr_ex={:#x}+{:#x}",
                    i,
                    indirect_offset,
                    indirect_size,
                    aes_ctr_ex_offset,
                    aes_ctr_ex_size,
                );
            }
            let patch = if indirect_size != 0 {
                let magic = u32at(fs_header, 0x110)?;
                let version = u32at(fs_header, 0x114)?;
                let entry_count = u32at(fs_header, 0x118)?;
                if indirect_offset < 0 || indirect_size < 0 || magic != 0x52544b42
                    || version > 1 || entry_count == 0
                {
                    return Err(format!("invalid NCA indirect table in section {i}"));
                }
                Some(NcaPatchInfo {
                    bucket_offset: indirect_offset as u64,
                    bucket_size: indirect_size as u64,
                    entry_count,
                })
            } else {
                None
            };
            let sparse_bucket_offset = crate::bin_read::i64at(fs_header, 0x148)?;
            let sparse_bucket_size = crate::bin_read::i64at(fs_header, 0x150)?;
            let sparse_physical_offset = crate::bin_read::i64at(fs_header, 0x168)?;
            let sparse_generation = u16::from_le_bytes([fs_header[0x170], fs_header[0x171]]);
            if sparse_generation != 0 {
                log::info!(
                    "NCA section {} sparse generation={} bucket={:#x}+{:#x} physical_offset={:#x}",
                    i,
                    sparse_generation,
                    sparse_bucket_offset,
                    sparse_bucket_size,
                    sparse_physical_offset,
                );
            }
            let compression_bucket_offset = crate::bin_read::i64at(fs_header, 0x178)?;
            let compression_bucket_size = crate::bin_read::i64at(fs_header, 0x180)?;
            let compression_entry_count = crate::bin_read::u32at(fs_header, 0x190)?;
            if compression_bucket_size != 0 {
                log::info!(
                    "NCA section {} compression bucket={:#x}+{:#x}",
                    i,
                    compression_bucket_offset,
                    compression_bucket_size,
                );
            }
            let compression = if compression_bucket_size > 0 {
                if compression_bucket_offset < 0 || compression_entry_count == 0 {
                    return Err(format!(
                        "invalid NCA compression table offset={} size={} entries={}",
                        compression_bucket_offset, compression_bucket_size, compression_entry_count
                    ));
                }
                Some(NcaCompressionInfo {
                    bucket_offset: compression_bucket_offset as u64,
                    bucket_size: compression_bucket_size as u64,
                    entry_count: compression_entry_count,
                })
            } else {
                None
            };
            if encryption_type != 0 && encryption_type != 1 {
                log::warn!(
                    "NCA section {} encryption_type={} (expected None on a decrypted NCA)",
                    i,
                    encryption_type
                );
            }

            let (data_off, data_size) = romfs::fs_data_extent(fs_header, hash_type, section_len)?;
            let fs_data_start = section_start
                .checked_add(data_off)
                .ok_or("fs data start overflow")?;
            let fs_data_end = fs_data_start
                .checked_add(data_size)
                .ok_or("fs data end overflow")?;
            let fs_data_end = if patch.is_some() { fs_data_end } else { fs_data_end.min(section_end) };
            if patch.is_none() && fs_data_start > section_end {
                return Err(format!(
                    "NCA section {} fs data start {:#x} beyond section end {:#x}",
                    i, fs_data_start, section_end
                ));
            }

            sections.push(NcaFsSection {
                index: i,
                fs_type,
                hash_type,
                encryption_type,
                section_range: section_start as usize..section_end as usize,
                fs_data_range: fs_data_start as usize..fs_data_end as usize,
                compression,
                patch,
                sparse: sparse_generation != 0,
            });
        }

        Ok(Self {
            mmap,
            nca_base,
            content_type,
            program_id,
            content_size,
            sections,
        })
    }

    pub fn section(&self, fs_type: NcaFsType) -> Option<&NcaFsSection> {
        self.sections.iter().find(|s| s.fs_type == fs_type)
    }

    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }
}

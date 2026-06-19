use std::sync::Arc;
use memmap2::Mmap;

use crate::bin_read::u64at;
use super::partition::PartitionFs;

pub const DXCI_MAGIC: u32 = 0x49435844;
pub const HEAD_MAGIC: u32 = 0x44414548;

pub struct Xci {
    secure: PartitionFs,
}

impl Xci {
    pub fn parse(mmap: Arc<Mmap>) -> Result<Self, String> {
        let buf = &mmap[..];
        let magic = crate::bin_read::u32at(buf, 0x100)?;
        if magic != DXCI_MAGIC {
            if magic == HEAD_MAGIC {
                return Err("XCI is encrypted (magic HEAD); only decrypted DXCI is supported".to_string());
            }
            return Err(format!("gamecard magic {:#010x} at 0x100 is not DXCI", magic));
        }

        let hfs_offset = u64at(buf, 0x130)?;
        let main = PartitionFs::parse(mmap.clone(), hfs_offset as usize)?;

        let secure_entry = main
            .find("secure")
            .ok_or_else(|| {
                let names: Vec<&str> = main.entries().iter().map(|e| e.name.as_str()).collect();
                format!("no 'secure' partition in gamecard HFS0 (found {:?})", names)
            })?
            .clone();
        let secure_range = main.entry_range(&secure_entry)?;
        let secure = PartitionFs::parse(mmap.clone(), secure_range.start)?;

        Ok(Self { secure })
    }

    pub fn ncas(&self) -> &PartitionFs {
        &self.secure
    }
}

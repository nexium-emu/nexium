use memmap2::Mmap;
use std::sync::Arc;

use super::partition::{PartitionFs, PFS0_MAGIC};

pub struct Nsp {
    pfs: PartitionFs,
}

impl Nsp {
    pub fn parse(mmap: Arc<Mmap>) -> Result<Self, String> {
        let magic = crate::bin_read::u32at(&mmap[..], 0)?;
        if magic != PFS0_MAGIC {
            return Err(format!("NSP magic {:#010x} at 0 is not PFS0", magic));
        }
        let pfs = PartitionFs::parse(mmap, 0)?;
        Ok(Self { pfs })
    }

    pub fn ncas(&self) -> &PartitionFs {
        &self.pfs
    }
}

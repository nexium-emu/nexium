use memmap2::Mmap;
use std::ops::Range;
use std::sync::Arc;

use crate::bin_read::{u32at, u64at};

pub const PFS0_MAGIC: u32 = 0x30534650;
pub const HFS0_MAGIC: u32 = 0x30534648;

#[derive(Clone, Debug)]
pub struct PartitionEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub hash_region_size: u32,
}

pub struct PartitionFs {
    mmap: Arc<Mmap>,
    base: usize,
    data_start: usize,
    is_hfs0: bool,
    entries: Vec<PartitionEntry>,
}

impl PartitionFs {
    pub fn parse(mmap: Arc<Mmap>, base: usize) -> Result<Self, String> {
        let buf = &mmap[..];
        let magic = u32at(buf, base)?;
        let is_hfs0 = match magic {
            PFS0_MAGIC => false,
            HFS0_MAGIC => true,
            _ => {
                return Err(format!(
                    "partition magic {:#010x} at {:#x} is not PFS0/HFS0",
                    magic, base
                ))
            }
        };
        let num_entries = u32at(buf, base + 4)? as usize;
        let strtab_size = u32at(buf, base + 8)? as usize;
        let entry_size = if is_hfs0 { 0x40 } else { 0x18 };

        let entry_table = base
            .checked_add(0x10)
            .ok_or("partition entry table overflow")?;
        let strtab = entry_table
            .checked_add(
                num_entries
                    .checked_mul(entry_size)
                    .ok_or("entry count overflow")?,
            )
            .ok_or("strtab offset overflow")?;
        let data_start = strtab
            .checked_add(strtab_size)
            .ok_or("data_start overflow")?;
        if data_start > buf.len() {
            return Err(format!(
                "partition data_start {:#x} exceeds file {:#x}",
                data_start,
                buf.len()
            ));
        }

        let mut entries = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let e = entry_table + i * entry_size;
            let offset = u64at(buf, e)?;
            let size = u64at(buf, e + 8)?;
            let name_off = u32at(buf, e + 0x10)? as usize;
            let hash_region_size = if is_hfs0 { u32at(buf, e + 0x14)? } else { 0 };

            if name_off >= strtab_size {
                return Err(format!("partition entry {i} name offset is outside its string table"));
            }
            let name_start = strtab.checked_add(name_off).ok_or("name offset overflow")?;
            let name = read_cstr(buf, name_start, strtab + strtab_size);
            entries.push(PartitionEntry {
                name,
                offset,
                size,
                hash_region_size,
            });
        }

        Ok(Self {
            mmap,
            base,
            data_start,
            is_hfs0,
            entries,
        })
    }

    pub fn entries(&self) -> &[PartitionEntry] {
        &self.entries
    }

    pub fn is_hfs0(&self) -> bool {
        self.is_hfs0
    }

    pub fn base(&self) -> usize {
        self.base
    }

    pub fn find(&self, name: &str) -> Option<&PartitionEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn entry_range(&self, e: &PartitionEntry) -> Result<Range<usize>, String> {
        let start = (self.data_start as u64)
            .checked_add(e.offset)
            .ok_or("entry start overflow")?;
        let limit = self.mmap.len() as u64;
        if start > limit {
            return Err(format!(
                "entry '{}' start {:#x} past EOF {:#x}",
                e.name, start, limit
            ));
        }
        let nominal_end = start.checked_add(e.size).ok_or("entry size overflow")?;
        let end = if nominal_end > limit {
            log::warn!(
                "entry '{}' end {:#x} clamped to EOF {:#x} (over by {:#x})",
                e.name,
                nominal_end,
                limit,
                nominal_end - limit
            );
            limit
        } else {
            nominal_end
        };
        Ok(start as usize..end as usize)
    }

    pub fn mmap(&self) -> &Arc<Mmap> {
        &self.mmap
    }
}

fn read_cstr(buf: &[u8], start: usize, limit: usize) -> String {
    let end = buf[start.min(buf.len())..limit.min(buf.len())]
        .iter()
        .position(|&b| b == 0)
        .map(|p| start + p)
        .unwrap_or(limit.min(buf.len()));
    String::from_utf8_lossy(&buf[start.min(buf.len())..end]).into_owned()
}

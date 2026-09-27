use memmap2::Mmap;
use std::ops::Range;
use std::sync::Arc;

use crate::bin_read::{u32at, u64at};
use crate::nca::NcaFsSection;

struct Source {
    mmap: Arc<Mmap>,
    range: Range<usize>,
}

struct Entry {
    offset: u64,
    physical: u64,
    source: usize,
}

pub(crate) struct PatchedRomfs {
    sources: [Source; 2],
    entries: Vec<Entry>,
    virtual_size: u64,
    data_offset: u64,
    data_size: u64,
}

impl PatchedRomfs {
    pub(crate) fn new(
        base_mmap: Arc<Mmap>,
        base: &NcaFsSection,
        patch_mmap: Arc<Mmap>,
        patch: &NcaFsSection,
    ) -> Result<Self, String> {
        let info = patch.patch.as_ref().ok_or("update has no indirect table")?;
        if info.entry_count == 0 || info.entry_count > i32::MAX as u32 {
            return Err("invalid update indirect entry count".into());
        }
        if base.sparse || patch.sparse || base.patch.is_some() {
            return Err("this update requires unsupported sparse or chained base storage".into());
        }
        if base.section_range.start > base.section_range.end
            || base.section_range.end > base_mmap.len()
            || patch.section_range.start > patch.section_range.end
            || patch.section_range.end > patch_mmap.len()
            || patch.fs_data_range.start > patch.fs_data_range.end
        {
            return Err("invalid update source range".into());
        }
        let start = patch
            .section_range
            .start
            .checked_add(
                usize::try_from(info.bucket_offset).map_err(|_| "update table offset overflow")?,
            )
            .ok_or("update table offset overflow")?;
        let end = start
            .checked_add(
                usize::try_from(info.bucket_size).map_err(|_| "update table size overflow")?,
            )
            .ok_or("update table size overflow")?;
        if end > patch.section_range.end {
            return Err("update indirect table extends beyond its section".into());
        }
        let table = patch_mmap
            .get(start..end)
            .ok_or("update indirect table is truncated")?;
        let entries_per_set = (0x4000usize - 0x10) / 0x14;
        let set_count = (info.entry_count as usize).div_ceil(entries_per_set);
        let offsets_per_node = (0x4000usize - 0x10) / 8;
        let extra_nodes = if set_count <= offsets_per_node {
            0
        } else {
            let groups = set_count.div_ceil(offsets_per_node);
            (set_count - (offsets_per_node - (groups - 1))).div_ceil(offsets_per_node)
        };
        let entry_start = (extra_nodes + 1)
            .checked_mul(0x4000)
            .ok_or("update node count overflow")?;
        let table_needed = set_count
            .checked_mul(0x4000)
            .and_then(|size| size.checked_add(entry_start))
            .ok_or("update table size overflow")?;
        if info.entry_count == 0 || table_needed > table.len() {
            return Err("update indirect table is incomplete".into());
        }
        let virtual_size = u64at(table, 8)?;
        let root_count = u32at(table, 4)? as usize;
        if u32at(table, 0)? != 0
            || root_count == 0
            || root_count > offsets_per_node
            || virtual_size == 0
            || virtual_size > i64::MAX as u64
        {
            return Err("invalid update indirect root node".into());
        }
        let mut previous_set_end = 0;
        let mut entries = Vec::with_capacity(info.entry_count as usize);
        for set in 0..set_count {
            let at = entry_start + set * 0x4000;
            let count = u32at(table, at + 4)? as usize;
            let remaining = info.entry_count as usize - entries.len();
            if u32at(table, at)? as usize != set || count != remaining.min(entries_per_set) {
                return Err(format!("invalid update indirect entry set {set}"));
            }
            if u64at(table, at + 0x10)? != previous_set_end {
                return Err(format!(
                    "inconsistent update indirect entry set boundary {set}"
                ));
            }
            for index in 0..count {
                let entry = at + 0x10 + index * 0x14;
                entries.push(Entry {
                    offset: u64at(table, entry)?,
                    physical: u64at(table, entry + 8)?,
                    source: u32at(table, entry + 16)? as usize,
                });
            }
            let set_end = u64at(table, at + 8)?;
            if set_end > virtual_size || entries.last().is_none_or(|entry| entry.offset >= set_end)
            {
                return Err(format!("invalid update indirect entry set boundary {set}"));
            }
            previous_set_end = set_end;
        }
        if previous_set_end != virtual_size {
            return Err("update indirect entries do not cover virtual storage".into());
        }
        let patch_data_end = patch
            .section_range
            .start
            .checked_add(
                usize::try_from(info.bucket_offset).map_err(|_| "update data size overflow")?,
            )
            .ok_or("update data size overflow")?;
        let sources = [
            Source {
                mmap: base_mmap,
                range: base.section_range.clone(),
            },
            Source {
                mmap: patch_mmap,
                range: patch.section_range.start..patch_data_end,
            },
        ];
        for (index, entry) in entries.iter().enumerate() {
            let end = entries
                .get(index + 1)
                .map_or(virtual_size, |next| next.offset);
            let source = sources
                .get(entry.source)
                .ok_or("update references an invalid source")?;
            if (index == 0 && entry.offset != 0) || end <= entry.offset || end > virtual_size {
                return Err("update indirect entries overlap or leave a gap".into());
            }
            let physical_end = entry
                .physical
                .checked_add(end - entry.offset)
                .ok_or("update source extent overflow")?;
            if source.range.end > source.mmap.len() || physical_end > source.range.len() as u64 {
                return Err(format!(
                    "update extent {index} exceeds source {}",
                    entry.source
                ));
            }
        }
        let data_offset = patch
            .fs_data_range
            .start
            .checked_sub(patch.section_range.start)
            .ok_or("invalid update data offset")? as u64;
        let data_size = patch.fs_data_range.len() as u64;
        if data_offset
            .checked_add(data_size)
            .is_none_or(|end| end > virtual_size)
        {
            return Err("update filesystem exceeds its virtual storage".into());
        }
        Ok(Self {
            sources,
            entries,
            virtual_size,
            data_offset,
            data_size,
        })
    }

    pub(crate) fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, String> {
        if offset
            .checked_add(size as u64)
            .is_none_or(|end| end > self.data_size)
        {
            return Err("update filesystem read is out of bounds".into());
        }
        let mut position = self.data_offset + offset;
        let mut output = Vec::with_capacity(size);
        while output.len() < size {
            let index = self
                .entries
                .partition_point(|entry| entry.offset <= position)
                .checked_sub(1)
                .ok_or("update read has no source extent")?;
            let entry = &self.entries[index];
            let end = self
                .entries
                .get(index + 1)
                .map_or(self.virtual_size, |next| next.offset);
            let take = (size - output.len()).min((end - position) as usize);
            let source = &self.sources[entry.source];
            let physical = source.range.start + (entry.physical + position - entry.offset) as usize;
            output.extend_from_slice(&source.mmap[physical..physical + take]);
            position += take as u64;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::LazyRomfs;
    use crate::nca::{NcaCompressionInfo, NcaFsType, NcaPatchInfo};
    use memmap2::MmapMut;

    fn mapped(bytes: &[u8]) -> Arc<Mmap> {
        let mut mmap = MmapMut::map_anon(bytes.len()).unwrap();
        mmap.copy_from_slice(bytes);
        Arc::new(mmap.make_read_only().unwrap())
    }

    fn put32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn section(range: Range<usize>, data: Range<usize>) -> NcaFsSection {
        NcaFsSection {
            index: 0,
            fs_type: NcaFsType::RomFs,
            hash_type: 3,
            encryption_type: 1,
            section_range: range,
            fs_data_range: data,
            compression: None,
            patch: None,
            sparse: false,
        }
    }

    struct Fixture {
        base: Vec<u8>,
        patch: Vec<u8>,
        base_section: NcaFsSection,
        patch_section: NcaFsSection,
        table: usize,
    }

    impl Fixture {
        fn new(
            base: &[u8],
            patch: &[u8],
            entries: &[(u64, u64, u32)],
            size: u64,
            data: Range<usize>,
        ) -> Self {
            let base_start = 0x80;
            let patch_start = 0x100;
            let per_set = (0x4000 - 0x10) / 0x14;
            let sets = entries.len().div_ceil(per_set);
            let table = patch_start + patch.len().div_ceil(0x200) * 0x200;
            let table_size = (sets + 1) * 0x4000;
            let mut base_bytes = vec![0xcc; base_start + base.len()];
            base_bytes[base_start..].copy_from_slice(base);
            let mut patch_bytes = vec![0; table + table_size];
            patch_bytes[patch_start..patch_start + patch.len()].copy_from_slice(patch);
            put32(&mut patch_bytes, table + 4, sets as u32);
            put64(&mut patch_bytes, table + 8, size);
            for (set, group) in entries.chunks(per_set).enumerate() {
                let at = table + (set + 1) * 0x4000;
                let end = entries
                    .get((set + 1) * per_set)
                    .map_or(size, |entry| entry.0);
                put64(&mut patch_bytes, table + 0x10 + set * 8, group[0].0);
                put32(&mut patch_bytes, at, set as u32);
                put32(&mut patch_bytes, at + 4, group.len() as u32);
                put64(&mut patch_bytes, at + 8, end);
                for (index, &(offset, physical, source)) in group.iter().enumerate() {
                    let at = at + 0x10 + index * 0x14;
                    put64(&mut patch_bytes, at, offset);
                    put64(&mut patch_bytes, at + 8, physical);
                    put32(&mut patch_bytes, at + 16, source);
                }
            }
            let mut patch_section = section(
                patch_start..patch_bytes.len(),
                patch_start + data.start..patch_start + data.end,
            );
            patch_section.patch = Some(NcaPatchInfo {
                bucket_offset: (table - patch_start) as u64,
                bucket_size: table_size as u64,
                entry_count: entries.len() as u32,
            });
            Self {
                base: base_bytes,
                patch: patch_bytes,
                base_section: section(
                    base_start..base_start + base.len(),
                    base_start..base_start + base.len(),
                ),
                patch_section,
                table,
            }
        }

        fn open(&self) -> Result<LazyRomfs, String> {
            LazyRomfs::from_patched_section(
                mapped(&self.base),
                &self.base_section,
                mapped(&self.patch),
                &self.patch_section,
            )
        }
    }

    #[test]
    fn bktr_cross_source_reads_apply_hash_leaf_offset_once() {
        let base: Vec<u8> = (0..64).collect();
        let patch: Vec<u8> = (128..192).collect();
        let fixture = Fixture::new(
            &base,
            &patch,
            &[(0, 0, 0), (12, 16, 1), (17, 24, 0)],
            32,
            8..28,
        );
        let storage = fixture.open().unwrap();
        let expected = [&base[8..12], &patch[16..21], &base[24..35]].concat();
        assert_eq!(storage.len(), 20);
        assert_eq!(storage.read(0, 20).unwrap(), expected);
        assert_eq!(storage.read(2, 10).unwrap(), expected[2..12]);
        assert_eq!(storage.read(18, 100).unwrap(), expected[18..]);
        assert!(storage.read(20, 1).unwrap().is_empty());
        assert!(storage.read(u64::MAX, 1).unwrap().is_empty());
        assert!(storage.as_slice().is_empty());
    }

    #[test]
    fn bktr_virtual_filesystem_can_exceed_update_file() {
        let base = vec![0x6b; 0x30000];
        let fixture = Fixture::new(
            &base,
            b"PATCH",
            &[(0, 0, 0), (0x24000, 0, 1), (0x24005, 0x24005, 0)],
            0x30000,
            0x1000..0x30000,
        );
        assert!(fixture.patch_section.fs_data_range.end > fixture.patch.len());
        let storage = fixture.open().unwrap();
        assert_eq!(storage.read(0x22ffe, 9).unwrap(), b"kkPATCHkk");
        assert_eq!(storage.read(storage.len() - 3, 8).unwrap(), b"kkk");
    }

    #[test]
    fn bktr_rejects_invalid_sources_order_extents_and_ranges() {
        for entries in [
            vec![(0, 0, 2)],
            vec![(1, 0, 0)],
            vec![(0, 0, 0), (0, 0, 1)],
            vec![(0, 0, 0), (9, 0, 1), (8, 0, 0)],
            vec![(0, u64::MAX, 0)],
            vec![(0, 60, 0)],
        ] {
            let fixture = Fixture::new(&[0; 64], &[0; 64], &entries, 16, 0..16);
            assert!(fixture.open().is_err(), "entries={entries:?}");
        }
        let mut fixture = Fixture::new(&[0; 64], &[0; 64], &[(0, 0, 0)], 16, 0..16);
        fixture.patch_section.fs_data_range = 0x110..0x100;
        assert!(fixture.open().is_err());
        fixture.patch_section.fs_data_range = 0x100..0x111;
        assert!(fixture.open().is_err());
        fixture.patch_section.fs_data_range = 0x100..0x110;
        fixture.base_section.section_range = 10..2;
        assert!(fixture.open().is_err());
        fixture.base_section.section_range = 0x80..0xc0;
        fixture.patch_section.patch.as_mut().unwrap().entry_count = u32::MAX;
        assert!(fixture.open().is_err());
    }

    #[test]
    fn bktr_rejects_truncated_tables_and_inconsistent_set_boundaries() {
        let entries: Vec<_> = (0..820).map(|index| (index, index, 0)).collect();
        let fixture = Fixture::new(&vec![0x3d; 820], &[0], &entries, 820, 0..820);
        assert_eq!(fixture.open().unwrap().read(815, 5).unwrap(), [0x3d; 5]);
        for offset in [
            fixture.table + 8,
            fixture.table + 0x4000 + 8,
            fixture.table + 0x8000 + 8,
        ] {
            let mut bytes = fixture.patch.clone();
            put64(&mut bytes, offset, 819);
            assert!(
                LazyRomfs::from_patched_section(
                    mapped(&fixture.base),
                    &fixture.base_section,
                    mapped(&bytes),
                    &fixture.patch_section
                )
                .is_err()
            );
        }
        let mut truncated = fixture.patch_section.clone();
        truncated.patch.as_mut().unwrap().bucket_size -= 1;
        assert!(
            LazyRomfs::from_patched_section(
                mapped(&fixture.base),
                &fixture.base_section,
                mapped(&fixture.patch),
                &truncated
            )
            .is_err()
        );
        let mut bytes = fixture.patch.clone();
        bytes.pop();
        assert!(
            LazyRomfs::from_patched_section(
                mapped(&fixture.base),
                &fixture.base_section,
                mapped(&bytes),
                &fixture.patch_section
            )
            .is_err()
        );
    }

    #[test]
    fn bktr_full_replacement_without_indirection_ignores_base_storage() {
        let mut base = section(0..1, 0..1);
        base.sparse = true;
        let replacement = section(2..15, 6..15);
        let storage = LazyRomfs::from_patched_section(
            mapped(b"x"),
            &base,
            mapped(b"headerREPLACED!"),
            &replacement,
        )
        .unwrap();
        assert_eq!(storage.read(0, 100).unwrap(), b"REPLACED!");
        assert_eq!(storage.as_slice(), b"REPLACED!");
    }

    fn compressed_fixture() -> (Fixture, Vec<u8>) {
        let decoded =
            b"patched compression reads both original bytes and update bytes repeatedly repeatedly"
                .repeat(3);
        let encoded = lz4_flex::block::compress(&decoded);
        let leaf = 0x80;
        let table = leaf + 0x200;
        let mut base = vec![0; table + 0x8000];
        base[leaf..leaf + encoded.len()].copy_from_slice(&encoded);
        put32(&mut base, table + 4, 1);
        put64(&mut base, table + 8, decoded.len() as u64);
        put32(&mut base, table + 0x4000 + 4, 1);
        put64(&mut base, table + 0x4000 + 8, decoded.len() as u64);
        put64(&mut base, table + 0x4000 + 0x10, 0);
        put64(&mut base, table + 0x4000 + 0x18, 0);
        base[table + 0x4000 + 0x20] = 3;
        put32(&mut base, table + 0x4000 + 0x24, encoded.len() as u32);
        let split = encoded.len() / 2;
        base[leaf..leaf + split].fill(0xa5);
        let mut fixture = Fixture::new(
            &base,
            &encoded[..split],
            &[
                (0, 0, 0),
                (leaf as u64, 0, 1),
                ((leaf + split) as u64, (leaf + split) as u64, 0),
            ],
            base.len() as u64,
            leaf..base.len(),
        );
        fixture.patch_section.compression = Some(NcaCompressionInfo {
            bucket_offset: 0x200,
            bucket_size: 0x8000,
            entry_count: 1,
        });
        (fixture, decoded)
    }

    #[test]
    fn bktr_decompresses_after_indirection_and_hash_leaf_selection() {
        let (fixture, decoded) = compressed_fixture();
        let storage = fixture.open().unwrap();
        assert_eq!(storage.len(), decoded.len() as u64);
        assert_eq!(storage.read(0, decoded.len()).unwrap(), decoded);
        assert_eq!(storage.read(13, 47).unwrap(), decoded[13..60]);
    }

    #[test]
    fn bktr_rejects_malformed_compression_metadata_before_reading() {
        for (offset, value) in [
            (4, 0),
            (0x4004, 2),
            (0x4008, 1),
            (0x4018, 0x200),
            (0x4020, 2),
            (0x4024, 0),
        ] {
            let (mut fixture, _) = compressed_fixture();
            let table = 0x80 + 0x80 + 0x200;
            put32(&mut fixture.base, table + offset, value);
            assert!(fixture.open().is_err(), "offset={offset:#x}");
        }
        let (mut fixture, _) = compressed_fixture();
        fixture
            .patch_section
            .compression
            .as_mut()
            .unwrap()
            .entry_count = 2;
        assert!(fixture.open().is_err());
    }

    #[test]
    fn bktr_rejects_short_decoded_blocks_without_panicking() {
        let (mut fixture, decoded) = compressed_fixture();
        let table = 0x80 + 0x80 + 0x200;
        put64(&mut fixture.base, table + 8, decoded.len() as u64 + 1);
        put64(&mut fixture.base, table + 0x4008, decoded.len() as u64 + 1);
        assert!(fixture.open().unwrap().read(0, decoded.len() + 1).is_err());
    }
}

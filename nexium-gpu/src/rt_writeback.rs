use crate::rt_cache::RtKey;

#[derive(Clone, Copy)]
pub(crate) struct WritebackFootprint {
    pub key: RtKey,
    pub bpp: usize,
    pub tile_mode: u32,
    pub stamp: u64,
}

impl WritebackFootprint {
    fn ranges(self) -> Option<Vec<(u64, u64)>> {
        let key = self.key;
        if key.gpu_va == 0 || key.cpu_addr == 0 || key.mapping_epoch == 0 || key.is_3d
            || key.width == 0 || key.height == 0 || self.bpp == 0 || self.stamp == 0 {
            return None;
        }
        let row = u64::from(key.width).checked_mul(self.bpp as u64)?;
        let size = if self.tile_mode & (1 << 12) != 0 {
            row.checked_mul(u64::from(key.height))?
        } else {
            let block_rows = 8u64.checked_shl((self.tile_mode >> 4) & 7)?;
            let rows = u64::from(key.height).checked_add(block_rows - 1)? / block_rows;
            (row.checked_add(63)? / 64).checked_mul(64)?
                .checked_mul(rows)?.checked_mul(block_rows)?
        };
        let layers = key.render_layer_count();
        if layers > 1 && key.array_stride_bytes < size { return None; }
        (0..layers).map(|layer| {
            let start = key.gpu_va.checked_add(key.array_stride_bytes.checked_mul(u64::from(layer))?)?;
            Some((start, start.checked_add(size)?))
        }).collect()
    }
}

pub(crate) fn superseded_writebacks(writes: &[WritebackFootprint]) -> Vec<bool> {
    let ranges: Vec<_> = writes.iter().map(|write| write.ranges()).collect();
    writes.iter().enumerate().map(|(index, write)| {
        let Some(wanted) = &ranges[index] else { return false; };
        let mut newer = Vec::new();
        for (other, candidate) in writes.iter().enumerate() {
            if candidate.stamp > write.stamp
                && candidate.key.nvmap_id == write.key.nvmap_id
                && candidate.key.mapping_epoch == write.key.mapping_epoch
                && i128::from(candidate.key.gpu_va) - i128::from(candidate.key.cpu_addr)
                    == i128::from(write.key.gpu_va) - i128::from(write.key.cpu_addr) {
                if let Some(ranges) = &ranges[other] { newer.extend_from_slice(ranges); }
            }
        }
        newer.sort_unstable();
        wanted.iter().all(|&(start, end)| {
            let mut cursor = start;
            for &(lo, hi) in &newer {
                if lo > cursor { break; }
                cursor = cursor.max(hi);
                if cursor >= end { return true; }
            }
            false
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(cpu: u64, width: u32, height: u32, bh: u32, stamp: u64) -> WritebackFootprint {
        WritebackFootprint {
            key: RtKey::with_cpu(12, width, height, cpu, cpu)
                .with_mapping_epoch(32).with_guest_size_bytes(0x6000),
            bpp: 4, tile_mode: bh << 4, stamp,
        }
    }

    #[test]
    fn scratch_views_use_written_gobs_not_allocation_size() {
        let old = write(0x10000, 128, 23, 2, 1);
        let newer = write(0x10000, 64, 64, 3, 2);
        assert_eq!(old.ranges(), Some(vec![(0x10000, 0x14000)]));
        assert_eq!(superseded_writebacks(&[old, newer]), [true, false]);
        let mip = write(0x14000, 32, 32, 2, 3);
        assert_eq!(mip.ranges(), Some(vec![(0x14000, 0x15000)]));
        assert_eq!(superseded_writebacks(&[newer, mip]), [false, false]);
    }

    #[test]
    fn coverage_can_span_faces_but_must_not_bridge_holes() {
        let old = write(0x10000, 64, 64, 3, 1);
        let faces: Vec<_> = (0..4).map(|i| write(0x10000 + i * 0x1000, 32, 32, 2, i + 2)).collect();
        let mut all = vec![old];
        all.extend_from_slice(&faces);
        assert!(superseded_writebacks(&all)[0]);
        all.remove(2);
        assert!(!superseded_writebacks(&all)[0]);
        all[1].key.guest_size_bytes = u64::MAX;
        assert!(!superseded_writebacks(&all)[0]);
    }

    #[test]
    fn aliases_require_live_identity_and_strict_write_order() {
        let old = write(0x10000, 64, 64, 3, 1);
        let mut new = old;
        new.key.gpu_va += 0x800000;
        new.stamp = 2;
        assert_eq!(superseded_writebacks(&[new, old]), [false, false]);
        new.key.gpu_va = old.key.gpu_va;
        assert_eq!(superseded_writebacks(&[new, old]), [false, true]);
        new.stamp = 1;
        assert_eq!(superseded_writebacks(&[new, old]), [false, false]);
        new.stamp = 2;
        new.key.mapping_epoch += 1;
        assert_eq!(superseded_writebacks(&[new, old]), [false, false]);
        new.key.mapping_epoch = old.key.mapping_epoch;
        new.key.nvmap_id += 1;
        assert_eq!(superseded_writebacks(&[new, old]), [false, false]);
    }

    #[test]
    fn layered_and_linear_writes_preserve_gaps_and_reject_overflow() {
        let mut layered = write(0x10000, 16, 8, 0, 2);
        layered.key.depth = 2;
        layered.key.array_stride_bytes = 0x1000;
        assert_eq!(layered.ranges(), Some(vec![(0x10000, 0x10200), (0x11000, 0x11200)]));
        let mut linear = write(0x10000, 3, 5, 0, 1);
        linear.tile_mode = 1 << 12;
        assert_eq!(linear.ranges(), Some(vec![(0x10000, 0x1003c)]));
        assert!(superseded_writebacks(&[linear, layered])[0]);
        linear.key.cpu_addr = u64::MAX - 5;
        linear.key.gpu_va = u64::MAX - 5;
        assert!(linear.ranges().is_none());
        layered.key.array_stride_bytes = 1;
        assert!(layered.ranges().is_none());
    }
}

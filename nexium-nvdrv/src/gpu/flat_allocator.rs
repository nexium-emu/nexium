pub struct FlatAllocator {
    intervals: Vec<(u64, u64, u64)>,
    virt_start: u64,
    va_limit: u64,
    linear: u64,
}

impl FlatAllocator {
    pub fn new(virt_start: u64, va_limit: u64) -> Self {
        Self {
            intervals: Vec::new(),
            virt_start,
            va_limit,
            linear: virt_start,
        }
    }

    pub fn allocate(&mut self, size: u64) -> u64 {
        self.allocate_aligned(size, 1)
    }

    pub fn allocate_aligned(&mut self, size: u64, align: u64) -> u64 {
        if size == 0 || align == 0 || !align.is_power_of_two() {
            return 0;
        }
        let mut linear_candidate = Self::align_up(self.linear, align);
        if let Some(mut start) = linear_candidate {
            for &(s, e, _) in &self.intervals {
                let Some(end) = start.checked_add(size) else {
                    linear_candidate = None;
                    break;
                };
                if s < end && e > start {
                    let Some(aligned) = Self::align_up(e, align) else {
                        linear_candidate = None;
                        break;
                    };
                    start = aligned;
                }
            }
            if linear_candidate.is_some() {
                if let Some(end) = self.checked_end(start, size) {
                    self.reserve(start, end);
                    self.linear = end;
                    return start;
                }
            }
        }
        let Some(mut cand) = Self::align_up(self.virt_start, align) else {
            return 0;
        };
        for &(s, e, _) in &self.intervals {
            if e <= cand {
                continue;
            }
            let Some(end) = cand.checked_add(size) else {
                return 0;
            };
            if end <= s {
                self.reserve(cand, end);
                return cand;
            }
            let Some(aligned) = Self::align_up(cand.max(e), align) else {
                return 0;
            };
            cand = aligned;
        }
        if let Some(end) = self.checked_end(cand, size) {
            self.reserve(cand, end);
            return cand;
        }
        0
    }

    fn align_up(value: u64, align: u64) -> Option<u64> {
        value
            .checked_add(align - 1)
            .map(|value| value & !(align - 1))
    }

    pub fn allocate_fixed(&mut self, virt: u64, size: u64) -> bool {
        let Some(end) = self.checked_end(virt, size) else {
            return false;
        };
        self.reserve(virt, end);
        true
    }

    pub fn allocate_fixed_exclusive(&mut self, virt: u64, size: u64) -> bool {
        let Some(end) = self.checked_end(virt, size) else {
            return false;
        };
        if self
            .intervals
            .iter()
            .any(|&(start, finish, _)| start < end && finish > virt)
        {
            return false;
        }
        self.reserve(virt, end);
        true
    }

    pub fn free(&mut self, virt: u64, size: u64) -> bool {
        let Some(end) = self.checked_end(virt, size) else {
            return false;
        };
        self.unreserve(virt, end);
        true
    }

    fn checked_end(&self, virt: u64, size: u64) -> Option<u64> {
        if size == 0 || virt < self.virt_start {
            return None;
        }
        let end = virt.checked_add(size)?;
        (end <= self.va_limit).then_some(end)
    }

    fn reserve(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }

        let intervals = std::mem::take(&mut self.intervals);
        let mut out = Vec::with_capacity(intervals.len() + 2);
        let mut cursor = start;
        for (s, e, owners) in intervals {
            if e <= start {
                Self::push_interval(&mut out, s, e, owners);
                continue;
            }
            if s >= end {
                if cursor < end {
                    Self::push_interval(&mut out, cursor, end, 1);
                    cursor = end;
                }
                Self::push_interval(&mut out, s, e, owners);
                continue;
            }

            if s < start {
                Self::push_interval(&mut out, s, start, owners);
            }
            let overlap_start = s.max(start);
            if cursor < overlap_start {
                Self::push_interval(&mut out, cursor, overlap_start, 1);
            }
            let overlap_end = e.min(end);
            Self::push_interval(
                &mut out,
                overlap_start,
                overlap_end,
                owners.checked_add(1).expect("reservation count overflow"),
            );
            cursor = cursor.max(overlap_end);
            if e > end {
                Self::push_interval(&mut out, end, e, owners);
            }
        }
        if cursor < end {
            Self::push_interval(&mut out, cursor, end, 1);
        }
        self.intervals = out;
    }

    fn unreserve(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }

        let intervals = std::mem::take(&mut self.intervals);
        let mut out = Vec::with_capacity(intervals.len() + 1);
        for (s, e, owners) in intervals {
            if e <= start || s >= end {
                Self::push_interval(&mut out, s, e, owners);
                continue;
            }

            if s < start {
                Self::push_interval(&mut out, s, start, owners);
            }
            if owners > 1 {
                Self::push_interval(&mut out, s.max(start), e.min(end), owners - 1);
            }
            if e > end {
                Self::push_interval(&mut out, end, e, owners);
            }
        }
        self.intervals = out;
    }

    fn push_interval(intervals: &mut Vec<(u64, u64, u64)>, start: u64, end: u64, owners: u64) {
        if start >= end || owners == 0 {
            return;
        }
        if let Some((_, previous_end, previous_owners)) = intervals.last_mut() {
            if *previous_end == start && *previous_owners == owners {
                *previous_end = end;
                return;
            }
        }
        intervals.push((start, end, owners));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_sequence() {
        let mut a = FlatAllocator::new(0x1000, 0x100000);
        assert_eq!(a.allocate(0x1000), 0x1000);
        assert_eq!(a.allocate(0x2000), 0x2000);
        assert_eq!(a.allocate(0x1000), 0x4000);
    }

    #[test]
    fn aligned_allocation_skips_and_later_reuses_linear_padding() {
        let mut a = FlatAllocator::new(0x40000, 0x80000);
        assert_eq!(a.allocate_aligned(0x10000, 0x10000), 0x40000);
        assert_eq!(a.allocate_aligned(0x20000, 0x20000), 0x60000);
        assert_eq!(a.allocate(0x10000), 0x50000);
        assert_eq!(a.allocate_aligned(0x1000, 3), 0);
    }

    #[test]
    fn fixed_reservation_is_skipped() {
        let mut a = FlatAllocator::new(0x1000, 0x100000);
        a.allocate_fixed(0x2000, 0x1000);
        assert_eq!(a.allocate(0x1000), 0x1000);
        assert_eq!(a.allocate(0x1000), 0x3000);
    }

    #[test]
    fn fixed_in_linear_path_skipped() {
        let mut a = FlatAllocator::new(0x1000, 0x100000);
        a.allocate_fixed(0x1000, 0x2000);
        assert_eq!(a.allocate(0x1000), 0x3000);
    }

    #[test]
    fn free_then_overflow_reuses() {
        let mut a = FlatAllocator::new(0x1000, 0x6000);
        let p0 = a.allocate(0x2000);
        let _p1 = a.allocate(0x2000);
        a.free(p0, 0x2000);
        assert_eq!(a.allocate(0x2000), 0x1000);
    }

    #[test]
    fn full_returns_zero() {
        let mut a = FlatAllocator::new(0x1000, 0x3000);
        assert_eq!(a.allocate(0x2000), 0x1000);
        assert_eq!(a.allocate(0x2000), 0);
    }

    #[test]
    fn fixed_reservations_require_a_valid_in_domain_range() {
        let mut a = FlatAllocator::new(0x1000, 0x5000);

        assert!(!a.allocate_fixed(0x1000, 0));
        assert!(!a.allocate_fixed(0x800, 0x800));
        assert!(!a.allocate_fixed(0x4800, 0x1000));
        assert!(!a.allocate_fixed(u64::MAX - 0x800, 0x1000));
        assert!(a.intervals.is_empty());

        assert!(a.allocate_fixed(0x1000, 0x4000));
        assert_eq!(a.intervals, vec![(0x1000, 0x5000, 1)]);
    }

    #[test]
    fn exclusive_fixed_reservations_reject_existing_owners() {
        let mut a = FlatAllocator::new(0x1000, 0x5000);

        assert!(a.allocate_fixed_exclusive(0x1000, 0x2000));
        assert!(!a.allocate_fixed_exclusive(0x1800, 0x1000));
        assert!(!a.allocate_fixed_exclusive(0x800, 0x1000));
        assert_eq!(a.intervals, vec![(0x1000, 0x3000, 1)]);

        assert!(a.free(0x1000, 0x2000));
        assert!(a.allocate_fixed_exclusive(0x1800, 0x1000));
    }

    #[test]
    fn invalid_frees_leave_reservations_unchanged() {
        let mut a = FlatAllocator::new(0x1000, 0x5000);
        assert!(a.allocate_fixed(0x1000, 0x4000));

        assert!(!a.free(0x1000, 0));
        assert!(!a.free(0x800, 0x800));
        assert!(!a.free(0x4800, 0x1000));
        assert!(!a.free(u64::MAX - 0x800, 0x1000));
        assert_eq!(a.intervals, vec![(0x1000, 0x5000, 1)]);

        assert!(a.free(0x1000, 0x4000));
        assert!(a.intervals.is_empty());
    }

    #[test]
    fn dynamic_allocation_rejects_overflowing_size() {
        let mut a = FlatAllocator::new(0x1000, u64::MAX);
        assert_eq!(a.allocate(u64::MAX), 0);
        assert!(a.intervals.is_empty());
    }

    #[test]
    fn nested_fixed_mapping_release_preserves_parent_allocation() {
        let mut a = FlatAllocator::new(0x1000, 0x5000);
        let parent = a.allocate(0x4000);
        a.allocate_fixed(parent + 0x1000, 0x1000);

        a.free(parent + 0x1000, 0x1000);

        assert_eq!(a.intervals, vec![(0x1000, 0x5000, 1)]);
        assert_eq!(a.allocate(0x1000), 0);
    }

    #[test]
    fn partial_overlapping_frees_decrement_only_their_covered_span() {
        let mut a = FlatAllocator::new(0x1000, 0x4000);
        a.allocate_fixed(0x1000, 0x2000);
        a.allocate_fixed(0x1800, 0x1000);

        a.free(0x1800, 0x800);
        assert_eq!(
            a.intervals,
            vec![
                (0x1000, 0x2000, 1),
                (0x2000, 0x2800, 2),
                (0x2800, 0x3000, 1)
            ]
        );

        a.free(0x2000, 0x800);
        assert_eq!(a.intervals, vec![(0x1000, 0x3000, 1)]);
    }

    #[test]
    fn exact_final_release_makes_overlapped_space_reusable() {
        let mut a = FlatAllocator::new(0x1000, 0x4000);
        let parent = a.allocate(0x3000);
        a.allocate_fixed(0x1800, 0x1000);

        a.free(parent, 0x3000);
        assert_eq!(a.intervals, vec![(0x1800, 0x2800, 1)]);
        assert_eq!(a.allocate(0x1801), 0);

        a.free(0x1800, 0x1000);
        assert_eq!(a.allocate(0x3000), parent);
    }
}

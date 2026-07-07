pub struct FlatAllocator {
    intervals: Vec<(u64, u64)>,
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
        if size == 0 {
            return 0;
        }
        let mut start = self.linear;
        for &(s, e) in &self.intervals {
            if s < start.saturating_add(size) && e > start {
                start = e;
            }
        }
        if start
            .checked_add(size)
            .map_or(false, |end| end <= self.va_limit)
        {
            self.reserve(start, start + size);
            self.linear = start + size;
            return start;
        }
        let mut cand = self.virt_start;
        for &(s, e) in &self.intervals {
            if e <= cand {
                continue;
            }
            if cand.saturating_add(size) <= s {
                self.reserve(cand, cand + size);
                return cand;
            }
            cand = cand.max(e);
        }
        if cand
            .checked_add(size)
            .map_or(false, |end| end <= self.va_limit)
        {
            self.reserve(cand, cand + size);
            return cand;
        }
        0
    }

    pub fn allocate_fixed(&mut self, virt: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.reserve(virt, virt.saturating_add(size));
    }

    pub fn free(&mut self, virt: u64, size: u64) {
        if size == 0 {
            return;
        }
        self.unreserve(virt, virt.saturating_add(size));
    }

    fn reserve(&mut self, start: u64, end: u64) {
        let mut lo = start;
        let mut hi = end;
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(self.intervals.len() + 1);
        let mut inserted = false;
        for &(s, e) in &self.intervals {
            if e < lo {
                out.push((s, e));
            } else if s > hi {
                if !inserted {
                    out.push((lo, hi));
                    inserted = true;
                }
                out.push((s, e));
            } else {
                lo = lo.min(s);
                hi = hi.max(e);
            }
        }
        if !inserted {
            out.push((lo, hi));
        }
        self.intervals = out;
    }

    fn unreserve(&mut self, start: u64, end: u64) {
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(self.intervals.len() + 1);
        for &(s, e) in &self.intervals {
            if e <= start || s >= end {
                out.push((s, e));
            } else {
                if s < start {
                    out.push((s, start));
                }
                if e > end {
                    out.push((end, e));
                }
            }
        }
        self.intervals = out;
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
}

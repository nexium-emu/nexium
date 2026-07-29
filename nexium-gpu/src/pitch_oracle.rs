use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

fn ranges() -> &'static Mutex<BTreeMap<u64, u64>> {
    static S: OnceLock<Mutex<BTreeMap<u64, u64>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn page_mode() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| std::env::var_os("NEXIUM_PITCH_ORACLE_PAGE").map_or(false, |v| v == "1"))
}

pub fn record_pitch_dst(gpu_va: u64, len: u64) {
    let mut start = gpu_va;
    let mut end = gpu_va.saturating_add(len.max(1));
    if page_mode() {
        start &= !0xFFFF;
        end = end.max(start + 0x10000);
    }
    let mut m = ranges().lock().unwrap();
    record_pitch_dst_in(&mut m, start, end);
}

fn record_pitch_dst_in(m: &mut BTreeMap<u64, u64>, mut start: u64, mut end: u64) {
    loop {
        let candidate = m
            .range(..=end)
            .next_back()
            .map(|(&range_start, &range_end)| (range_start, range_end));
        let Some((range_start, range_end)) = candidate else {
            break;
        };
        if range_end < start {
            break;
        }
        m.remove(&range_start);
        start = start.min(range_start);
        end = end.max(range_end);
    }
    m.insert(start, end);
}

pub fn clear_pitch_range(gpu_va: u64, len: u64) {
    if page_mode() {
        return;
    }
    let start = gpu_va;
    let end = gpu_va.saturating_add(len.max(1));
    let mut m = ranges().lock().unwrap();
    clear_pitch_range_in(&mut m, start, end);
}

fn clear_pitch_range_in(m: &mut BTreeMap<u64, u64>, start: u64, end: u64) {
    loop {
        let candidate = m
            .range(..end)
            .next_back()
            .map(|(&range_start, &range_end)| (range_start, range_end));
        let Some((range_start, range_end)) = candidate else {
            break;
        };
        if range_end <= start {
            break;
        }
        m.remove(&range_start);
        if range_start < start {
            m.insert(range_start, start);
        }
        if range_end > end {
            m.insert(end, range_end);
        }
    }
}

pub fn is_pitch_dst(gpu_va: u64) -> bool {
    let m = ranges().lock().unwrap();
    m.range(..=gpu_va)
        .next_back()
        .map_or(false, |(_, &e)| e > gpu_va)
}

#[cfg(test)]
mod tests {
    use super::{clear_pitch_range_in, record_pitch_dst_in};
    use std::collections::BTreeMap;

    #[test]
    fn record_merges_only_touching_or_overlapping_ranges() {
        let mut ranges = BTreeMap::from([(0x1000, 0x1800), (0x3000, 0x3800)]);
        record_pitch_dst_in(&mut ranges, 0x1800, 0x3200);
        assert_eq!(ranges, BTreeMap::from([(0x1000, 0x3800)]));

        record_pitch_dst_in(&mut ranges, 0x5000, 0x5800);
        assert_eq!(ranges, BTreeMap::from([(0x1000, 0x3800), (0x5000, 0x5800)]));
    }

    #[test]
    fn clear_splits_and_removes_only_overlapping_ranges() {
        let mut ranges = BTreeMap::from([(0x1000, 0x2000), (0x3000, 0x5000), (0x6000, 0x7000)]);
        clear_pitch_range_in(&mut ranges, 0x3800, 0x4800);
        assert_eq!(
            ranges,
            BTreeMap::from([
                (0x1000, 0x2000),
                (0x3000, 0x3800),
                (0x4800, 0x5000),
                (0x6000, 0x7000),
            ])
        );

        clear_pitch_range_in(&mut ranges, 0x1800, 0x6800);
        assert_eq!(ranges, BTreeMap::from([(0x1000, 0x1800), (0x6800, 0x7000)]));
    }
}

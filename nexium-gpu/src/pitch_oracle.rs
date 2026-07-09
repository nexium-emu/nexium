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
    let absorbed: Vec<u64> = m
        .range(..=end)
        .filter(|&(_, &e)| e >= start)
        .map(|(&s, _)| s)
        .collect();
    for s in absorbed {
        let e = m.remove(&s).unwrap();
        if s < start {
            start = s;
        }
        if e > end {
            end = e;
        }
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
    let overlapping: Vec<(u64, u64)> = m
        .range(..end)
        .filter(|&(_, &e)| e > start)
        .map(|(&s, &e)| (s, e))
        .collect();
    for (s, e) in overlapping {
        m.remove(&s);
        if s < start {
            m.insert(s, start);
        }
        if e > end {
            m.insert(end, e);
        }
    }
}

pub fn is_pitch_dst(gpu_va: u64) -> bool {
    let m = ranges().lock().unwrap();
    m.range(..=gpu_va)
        .next_back()
        .map_or(false, |(_, &e)| e > gpu_va)
}

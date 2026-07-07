use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

fn set() -> &'static Mutex<HashSet<u64>> {
    static S: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn record_pitch_dst(gpu_va: u64) {
    let mut s = set().lock().unwrap();
    s.insert(gpu_va);
    s.insert(gpu_va & !0xFFFF);
}

pub fn is_pitch_dst(gpu_va: u64) -> bool {
    let s = set().lock().unwrap();
    s.contains(&gpu_va) || s.contains(&(gpu_va & !0xFFFF))
}

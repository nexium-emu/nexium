use std::cell::Cell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::console::DATA_ROOT;
use crate::sys;

type Check = fn() -> Result<String, String>;

pub fn run() -> u32 {
    let checks: &[(&str, Check)] = &[
        ("memory-info", memory_info),
        ("fp-state", fp_state),
        ("alloc", alloc),
        ("threads-tls", threads_tls),
        ("mutex-condvar", mutex_condvar),
        ("hashmap-random", hashmap_random),
        ("time", time),
        ("fs", fs),
        ("parallelism", parallelism),
    ];
    let mut all: Vec<(&str, Check)> = checks.to_vec();
    #[cfg(feature = "title")]
    all.extend_from_slice(crate::probe_mem::CHECKS);
    #[cfg(feature = "title")]
    all.extend_from_slice(crate::probe_jit::CHECKS);
    #[cfg(feature = "title")]
    all.extend_from_slice(crate::probe_filemap::CHECKS);
    #[cfg(feature = "title")]
    all.extend_from_slice(crate::probe_vk::CHECKS);
    let mut failures = 0;
    for (name, check) in all {
        let started = Instant::now();
        match check() {
            Ok(detail) => crate::klog!("PASS {name} ({:.1} ms): {detail}", ms(started)),
            Err(detail) => {
                failures += 1;
                crate::klog!("FAIL {name} ({:.1} ms): {detail}", ms(started))
            }
        }
    }
    failures
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn memory_info() -> Result<String, String> {
    let page = unsafe { libc_getpagesize() };
    Ok(format!(
        "pagesize={page} direct={} MiB flexible_free={} MiB",
        sys::direct_size() >> 20,
        sys::flexible_available().map(|v| v >> 20).unwrap_or(0)
    ))
}

unsafe extern "C" {
    #[link_name = "getpagesize"]
    fn libc_getpagesize() -> i32;
}

fn fp_state() -> Result<String, String> {
    let now = sys::mxcsr();
    let tiny = sys::opaque(f64::MIN_POSITIVE);
    let denormal = sys::opaque(tiny / 4.0);
    let preserved = denormal != 0.0 && denormal * 4.0 == tiny;
    if now & 0x8040 != 0 {
        return Err(format!("mxcsr={now:#06x} has FTZ/DAZ set; denormals preserved={preserved}"));
    }
    if !preserved {
        return Err(format!("mxcsr={now:#06x} but a denormal was flushed"));
    }
    Ok(format!("mxcsr={now:#06x} denormals preserved"))
}

fn alloc() -> Result<String, String> {
    let sizes = [1usize << 10, 1 << 20, 64 << 20, 512 << 20];
    let mut detail = Vec::new();
    for size in sizes {
        let started = Instant::now();
        let mut v: Vec<u8> = Vec::new();
        v.try_reserve_exact(size).map_err(|e| format!("reserve {size}: {e}"))?;
        v.resize(size, 0);
        let mut i = 0;
        while i < size {
            v[i] = (i >> 14) as u8;
            i += 1 << 14;
        }
        let sum: u64 = v.iter().step_by(1 << 14).map(|&b| b as u64).sum();
        let expected: u64 = (0..size.div_ceil(1 << 14)).map(|p| (p as u8) as u64).sum();
        if sum != expected {
            return Err(format!("size {size}: checksum {sum} != {expected}"));
        }
        detail.push(format!("{}KiB@{:.1}ms", size >> 10, ms(started)));
    }
    let mut boxes = Vec::with_capacity(100_000);
    let started = Instant::now();
    for i in 0..100_000u64 {
        boxes.push(Box::new([i; 3]));
    }
    let small_alloc_us = started.elapsed().as_secs_f64() * 1e6 / 100_000.0;
    let started = Instant::now();
    drop(boxes);
    let small_free_us = started.elapsed().as_secs_f64() * 1e6 / 100_000.0;
    detail.push(format!("small malloc {small_alloc_us:.2}us free {small_free_us:.2}us"));
    let started = Instant::now();
    for _ in 0..10_000 {
        unsafe { libc::getpid() };
    }
    detail.push(format!("getpid syscall {:.2}us", started.elapsed().as_secs_f64() * 1e6 / 10_000.0));
    let aligned = std::alloc::Layout::from_size_align(1 << 20, 1 << 16).unwrap();
    let p = unsafe { std::alloc::alloc(aligned) };
    if p.is_null() || (p as usize) & 0xffff != 0 {
        return Err(format!("64 KiB aligned alloc gave {p:p}"));
    }
    unsafe { std::alloc::dealloc(p, aligned) };
    Ok(detail.join(" "))
}

thread_local! {
    static COUNTER: Cell<u64> = const { Cell::new(0) };
}

fn threads_tls() -> Result<String, String> {
    let shared = Arc::new(AtomicU64::new(0));
    let threads = 8;
    let iterations = 1_000_000u64;
    let handles: Vec<_> = (0..threads)
        .map(|i| {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("probe-{i}"))
                .stack_size(4 << 20)
                .spawn(move || {
                    for _ in 0..iterations {
                        COUNTER.with(|c| c.set(c.get() + 1));
                        shared.fetch_add(1, Ordering::Relaxed);
                    }
                    let name = std::thread::current().name().map(str::to_owned);
                    (COUNTER.with(|c| c.get()), name, sys::mxcsr())
                })
                .map_err(|e| format!("spawn {i}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    let mut thread_mxcsr = Vec::new();
    for (i, h) in handles.into_iter().enumerate() {
        let (local, name, mxcsr) = h.join().map_err(|_| format!("join {i} panicked"))?;
        thread_mxcsr.push(mxcsr);
        if local != iterations {
            return Err(format!("thread {i} TLS counter {local} != {iterations}"));
        }
        if name.as_deref() != Some(format!("probe-{i}").as_str()) {
            return Err(format!("thread {i} name {name:?}"));
        }
    }
    let main_local = COUNTER.with(|c| c.get());
    let total = shared.load(Ordering::Relaxed);
    if main_local != 0 || total != threads * iterations {
        return Err(format!("main TLS {main_local}, total {total}"));
    }
    thread_mxcsr.dedup();
    if thread_mxcsr.iter().any(|m| m & 0x8040 != 0) && cfg!(feature = "title") {
        return Err(format!("spawned threads run with FTZ/DAZ: mxcsr {thread_mxcsr:x?}"));
    }
    Ok(format!(
        "{threads} threads x {iterations}, TLS isolated, names kept, thread mxcsr {thread_mxcsr:x?}"
    ))
}

fn mutex_condvar() -> Result<String, String> {
    let pair = Arc::new((Mutex::new(0u32), Condvar::new()));
    let rounds = 20_000u32;
    let other = pair.clone();
    let started = Instant::now();
    let h = std::thread::spawn(move || {
        let (lock, cv) = &*other;
        for i in 0..rounds {
            let mut g = lock.lock().unwrap();
            while *g != i * 2 + 1 {
                g = cv.wait(g).unwrap();
            }
            *g += 1;
            cv.notify_one();
        }
    });
    let (lock, cv) = &*pair;
    for i in 0..rounds {
        let mut g = lock.lock().unwrap();
        while *g != i * 2 {
            g = cv.wait(g).unwrap();
        }
        *g += 1;
        cv.notify_one();
    }
    h.join().map_err(|_| "ping-pong thread panicked".to_string())?;
    let elapsed = started.elapsed();
    let (g, timeout) = cv
        .wait_timeout(lock.lock().unwrap(), Duration::from_millis(20))
        .map_err(|_| "poisoned".to_string())?;
    if *g != rounds * 2 || !timeout.timed_out() {
        return Err(format!("final {} timed_out={}", *g, timeout.timed_out()));
    }
    Ok(format!(
        "{rounds} round trips in {:.1} ms ({:.2} us each), wait_timeout ok",
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1e6 / rounds as f64
    ))
}

fn hashmap_random() -> Result<String, String> {
    let mut m = HashMap::new();
    for i in 0..10_000u32 {
        m.insert(i, i * 3);
    }
    if (0..10_000u32).any(|i| m.get(&i) != Some(&(i * 3))) {
        return Err("lookup mismatch".into());
    }
    let a = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 42u32);
    let b = std::hash::BuildHasher::hash_one(&std::collections::hash_map::RandomState::new(), 42u32);
    Ok(format!("10000 entries; two RandomState seeds differ={}", a != b))
}

fn time() -> Result<String, String> {
    let started = Instant::now();
    std::thread::sleep(Duration::from_millis(50));
    let slept = started.elapsed();
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("SystemTime before epoch: {e}"))?;
    let process = unsafe { sys::sceKernelGetProcessTime() };
    if slept < Duration::from_millis(50) || slept > Duration::from_millis(500) {
        return Err(format!("sleep(50ms) took {slept:?}"));
    }
    if wall.as_secs() < 1_700_000_000 {
        return Err(format!("wall clock {wall:?}"));
    }
    let mut min_step = u128::MAX;
    let mut last = Instant::now();
    for _ in 0..1000 {
        let now = Instant::now();
        let step = now.duration_since(last).as_nanos();
        if step > 0 {
            min_step = min_step.min(step);
        }
        last = now;
    }
    Ok(format!(
        "sleep(50ms)={slept:?} unix={} process_us={process} instant_min_step={min_step}ns",
        wall.as_secs()
    ))
}

fn fs() -> Result<String, String> {
    let dir = format!("{DATA_ROOT}/probe");
    crate::console::ensure_shared_dir(&dir).map_err(|e| format!("create_dir_all {dir}: {e}"))?;
    let path = format!("{dir}/probe.bin");
    let payload: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
    {
        let mut f = std::fs::File::create(&path).map_err(|e| format!("create {path}: {e}"))?;
        f.write_all(&payload).map_err(|e| format!("write: {e}"))?;
        f.sync_all().map_err(|e| format!("sync: {e}"))?;
    }
    let meta = std::fs::metadata(&path).map_err(|e| format!("metadata: {e}"))?;
    if meta.len() != payload.len() as u64 || !meta.is_file() {
        return Err(format!("metadata len {} is_file {}", meta.len(), meta.is_file()));
    }
    let modified = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok());
    let mut back = Vec::new();
    std::fs::File::open(&path)
        .and_then(|mut f| f.read_to_end(&mut back))
        .map_err(|e| format!("read: {e}"))?;
    if back != payload {
        return Err("read-back differs".into());
    }
    let renamed = format!("{dir}/probe-renamed.bin");
    std::fs::rename(&path, &renamed).map_err(|e| format!("rename: {e}"))?;
    let listed: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| format!("read_dir: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    if !listed.iter().any(|n| n == "probe-renamed.bin") {
        return Err(format!("read_dir missing renamed file: {listed:?}"));
    }
    std::fs::remove_file(&renamed).map_err(|e| format!("remove: {e}"))?;
    let dir_meta = std::fs::metadata(&dir).map_err(|e| format!("dir metadata: {e}"))?;
    Ok(format!(
        "write/stat/read/rename/readdir/remove ok; mtime={:?} dir={} entries={}",
        modified.map(|d| d.as_secs()),
        dir_meta.is_dir(),
        listed.len()
    ))
}

fn parallelism() -> Result<String, String> {
    let n = std::thread::available_parallelism().map_err(|e| format!("{e}"))?;
    Ok(format!("available_parallelism={n}"))
}

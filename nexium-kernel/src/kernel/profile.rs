use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::Instant;

pub struct ProfileBucket {
    pub count: u64,
    pub total_ns: u64,
}

pub struct Profiler {
    pub svcs: HashMap<u16, ProfileBucket>,
    pub ipcs: HashMap<String, ProfileBucket>,
    pub wait_handles: HashMap<u32, u64>,
}

impl Profiler {
    fn new() -> Self {
        Self { svcs: HashMap::new(), ipcs: HashMap::new(), wait_handles: HashMap::new() }
    }

    pub fn record_svc(&mut self, imm: u16, ns: u64) {
        let b = self.svcs.entry(imm).or_insert(ProfileBucket { count: 0, total_ns: 0 });
        b.count += 1;
        b.total_ns += ns;
    }

    pub fn record_ipc(&mut self, port: &str, ns: u64) {
        let b = self.ipcs.entry(port.to_string()).or_insert(ProfileBucket { count: 0, total_ns: 0 });
        b.count += 1;
        b.total_ns += ns;
    }

    pub fn snapshot_and_reset(&mut self) -> (Vec<(u16, u64, u64)>, Vec<(String, u64, u64)>, Vec<(u32, u64)>) {
        let mut svcs: Vec<(u16, u64, u64)> = self.svcs.drain().map(|(k, v)| (k, v.count, v.total_ns)).collect();
        svcs.sort_by(|a, b| b.2.cmp(&a.2));
        let mut ipcs: Vec<(String, u64, u64)> = self.ipcs.drain().map(|(k, v)| (k, v.count, v.total_ns)).collect();
        ipcs.sort_by(|a, b| b.2.cmp(&a.2));
        let mut waits: Vec<(u32, u64)> = self.wait_handles.drain().collect();
        waits.sort_by(|a, b| b.1.cmp(&a.1));
        (svcs, ipcs, waits)
    }
}

static PROFILER: Mutex<Option<Profiler>> = Mutex::new(None);

pub fn record_svc(imm: u16, start: Instant) {
    let ns = start.elapsed().as_nanos() as u64;
    let mut g = PROFILER.lock();
    if g.is_none() { *g = Some(Profiler::new()); }
    g.as_mut().unwrap().record_svc(imm, ns);
}

pub fn record_ipc(port: &str, start: Instant) {
    let ns = start.elapsed().as_nanos() as u64;
    let mut g = PROFILER.lock();
    if g.is_none() { *g = Some(Profiler::new()); }
    g.as_mut().unwrap().record_ipc(port, ns);
}

pub fn record_wait_handle(handle: u32) {
    let mut g = PROFILER.lock();
    if g.is_none() { *g = Some(Profiler::new()); }
    *g.as_mut().unwrap().wait_handles.entry(handle).or_insert(0) += 1;
}

pub fn dump_heartbeat_with_kernel(kernel: &crate::kernel::Kernel) {
    let (svcs, ipcs, waits) = {
        let mut g = PROFILER.lock();
        if g.is_none() { return; }
        g.as_mut().unwrap().snapshot_and_reset()
    };
    if svcs.is_empty() && ipcs.is_empty() && waits.is_empty() { return; }
    log::warn!("[profile] top SVCs by total_ns (last interval):");
    for (imm, count, total_ns) in svcs.iter().take(8) {
        log::warn!("  svc {:#04x} count={} total_ms={:.2} avg_us={:.2}",
            imm, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
    log::warn!("[profile] top IPC ports by total_ns (last interval):");
    for (port, count, total_ns) in ipcs.iter().take(8) {
        log::warn!("  ipc {} count={} total_ms={:.2} avg_us={:.2}",
            port, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
    log::warn!("[profile] top wait handles (last interval):");
    for (h, count) in waits.iter().take(8) {
        let mut tags: Vec<&'static str> = Vec::new();
        if kernel.vsync_handles.contains(h) { tags.push("vsync"); }
        if Some(*h) == kernel.applet_message_event { tags.push("applet_msg"); }
        if kernel.audio_buffer_events.values().any(|v| *v == *h) { tags.push("audio_buf"); }
        if kernel.event_signals.contains_key(h) { tags.push("event"); }
        let ty = kernel.handles.get_handle(*h).map(|hh| format!("{:?}", hh.handle_type)).unwrap_or_else(|| "Unknown".into());
        log::warn!("  wait handle {:#x} count={} type={} tags={:?}", h, count, ty, tags);
    }
}

pub fn dump_heartbeat() {
    let (svcs, ipcs, waits) = {
        let mut g = PROFILER.lock();
        if g.is_none() { return; }
        g.as_mut().unwrap().snapshot_and_reset()
    };
    if svcs.is_empty() && ipcs.is_empty() && waits.is_empty() { return; }
    log::warn!("[profile] top SVCs by total_ns (last interval):");
    for (imm, count, total_ns) in svcs.iter().take(8) {
        log::warn!("  svc {:#04x} count={} total_ms={:.2} avg_us={:.2}",
            imm, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
    log::warn!("[profile] top IPC ports by total_ns (last interval):");
    for (port, count, total_ns) in ipcs.iter().take(8) {
        log::warn!("  ipc {} count={} total_ms={:.2} avg_us={:.2}",
            port, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
    log::warn!("[profile] top wait handles (last interval):");
    for (h, count) in waits.iter().take(8) {
        log::warn!("  wait handle {:#x} count={}", h, count);
    }
}

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
}

impl Profiler {
    fn new() -> Self {
        Self { svcs: HashMap::new(), ipcs: HashMap::new() }
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

    pub fn snapshot_and_reset(&mut self) -> (Vec<(u16, u64, u64)>, Vec<(String, u64, u64)>) {
        let mut svcs: Vec<(u16, u64, u64)> = self.svcs.drain().map(|(k, v)| (k, v.count, v.total_ns)).collect();
        svcs.sort_by(|a, b| b.2.cmp(&a.2));
        let mut ipcs: Vec<(String, u64, u64)> = self.ipcs.drain().map(|(k, v)| (k, v.count, v.total_ns)).collect();
        ipcs.sort_by(|a, b| b.2.cmp(&a.2));
        (svcs, ipcs)
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

pub fn dump_heartbeat() {
    let (svcs, ipcs) = {
        let mut g = PROFILER.lock();
        if g.is_none() { return; }
        g.as_mut().unwrap().snapshot_and_reset()
    };
    if svcs.is_empty() && ipcs.is_empty() { return; }
    log::info!("[profile] top SVCs by total_ns (last interval):");
    for (imm, count, total_ns) in svcs.iter().take(8) {
        log::info!("  svc {:#04x} count={} total_ms={:.2} avg_us={:.2}",
            imm, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
    log::info!("[profile] top IPC ports by total_ns (last interval):");
    for (port, count, total_ns) in ipcs.iter().take(8) {
        log::info!("  ipc {} count={} total_ms={:.2} avg_us={:.2}",
            port, count, *total_ns as f64 / 1e6, (*total_ns as f64) / (*count as f64) / 1e3);
    }
}

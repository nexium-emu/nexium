use ash::vk;

const PAIRS: u32 = 4096;
const KINDS: usize = 3;
const KIND_NAMES: [&str; KINDS] = ["graphics", "compute", "other"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StampKind {
    Graphics = 0,
    Compute = 1,
    Other = 2,
}

#[derive(Clone, Copy)]
struct PendingStamp {
    pair: u32,
    kind: StampKind,
    generation: u64,
}

pub(crate) struct GpuTimestamps {
    pool: vk::QueryPool,
    period_ns: f64,
    valid_mask: u64,
    next: u32,
    in_use: Vec<bool>,
    pending: std::collections::VecDeque<PendingStamp>,
    window: GpuTimeWindow,
}

#[derive(Default)]
struct GpuTimeWindow {
    busy_ns: [f64; KINDS],
    count: [u64; KINDS],
    union_ns: f64,
    first_start: Option<u64>,
    last_end: u64,
    started: Option<std::time::Instant>,
    skipped: u64,
}

pub(crate) fn gpu_timestamps_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_GPU_TIMESTAMPS").is_ok_and(|value| value != "0")
    })
}

impl GpuTimestamps {
    pub(crate) fn new(device: &ash::Device, valid_bits: u32, period: f32) -> Option<Self> {
        if !gpu_timestamps_enabled() || valid_bits == 0 || period <= 0.0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(PAIRS * 2);
        let pool = match unsafe { device.create_query_pool(&info, None) } {
            Ok(pool) => pool,
            Err(error) => {
                log::warn!("[gpu-ts] query pool unavailable: {error:?}");
                return None;
            }
        };
        log::warn!("[gpu-ts] enabled: {} pairs, period {:.3} ns, {} valid bits", PAIRS, period, valid_bits);
        Some(Self {
            pool,
            period_ns: f64::from(period),
            valid_mask: if valid_bits >= 64 { u64::MAX } else { (1u64 << valid_bits) - 1 },
            next: 0,
            in_use: vec![false; PAIRS as usize],
            pending: std::collections::VecDeque::new(),
            window: GpuTimeWindow::default(),
        })
    }

    pub(crate) fn begin(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) -> Option<u32> {
        let pair = self.next;
        if self.in_use[pair as usize] {
            self.window.skipped += 1;
            return None;
        }
        self.next = (self.next + 1) % PAIRS;
        self.in_use[pair as usize] = true;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, pair * 2, 2);
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, pair * 2);
        }
        Some(pair)
    }

    pub(crate) fn end(&self, device: &ash::Device, cmd: vk::CommandBuffer, pair: u32) {
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                pair * 2 + 1,
            );
        }
    }

    pub(crate) fn submitted(&mut self, pair: u32, kind: StampKind, generation: u64) {
        self.pending.push_back(PendingStamp { pair, kind, generation });
    }

    pub(crate) fn abandon(&mut self, pair: u32) {
        if let Some(slot) = self.in_use.get_mut(pair as usize) {
            *slot = false;
        }
    }

    pub(crate) fn poll(&mut self, device: &ash::Device, completed: u64) {
        while let Some(stamp) = self.pending.front().copied() {
            if stamp.generation > completed {
                break;
            }
            let mut ticks = [0u64; 2];
            let read = unsafe {
                device.get_query_pool_results(
                    self.pool,
                    stamp.pair * 2,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if read.is_err() {
                break;
            }
            self.pending.pop_front();
            self.in_use[stamp.pair as usize] = false;
            let start = ticks[0] & self.valid_mask;
            let end = ticks[1] & self.valid_mask;
            if end >= start {
                self.record(stamp.kind, start, end);
            }
        }
        self.maybe_report();
    }

    fn record(&mut self, kind: StampKind, start: u64, end: u64) {
        let window = &mut self.window;
        window.busy_ns[kind as usize] += (end - start) as f64 * self.period_ns;
        window.count[kind as usize] += 1;
        if window.first_start.is_none_or(|first| start < first) {
            window.first_start = Some(start);
        }
        if start >= window.last_end {
            window.union_ns += (end - start) as f64 * self.period_ns;
        } else if end > window.last_end {
            window.union_ns += (end - window.last_end) as f64 * self.period_ns;
        }
        window.last_end = window.last_end.max(end);
    }

    fn maybe_report(&mut self) {
        let now = std::time::Instant::now();
        let started = *self.window.started.get_or_insert(now);
        let elapsed = now.duration_since(started);
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let window = std::mem::take(&mut self.window);
        let span_ns = window
            .first_start
            .map_or(0.0, |first| window.last_end.saturating_sub(first) as f64 * self.period_ns);
        let mut parts = String::new();
        for kind in 0..KINDS {
            parts.push_str(&format!(
                " {}={}/{:.1}ms",
                KIND_NAMES[kind],
                window.count[kind],
                window.busy_ns[kind] / 1_000_000.0
            ));
        }
        log::warn!(
            "[gpu-ts] wall_ms={:.0} busy_ms={:.1} span_ms={:.1} busy_pct={:.1}{} skipped={}",
            elapsed.as_secs_f64() * 1000.0,
            window.union_ns / 1_000_000.0,
            span_ns / 1_000_000.0,
            window.union_ns * 100.0 / (elapsed.as_secs_f64() * 1e9),
            parts,
            window.skipped
        );
        self.window.started = Some(now);
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_query_pool(self.pool, None);
        }
        self.pool = vk::QueryPool::null();
    }
}

use super::super::PipelineStats;
use super::engines::{
    sw_renderer, Fermi2D, KeplerCompute, KeplerMemory, Maxwell3D, MaxwellDma, FERMI_2D_CLASS,
    KEPLER_COMPUTE_CLASS, KEPLER_MEMORY_CLASS, MACRO_REGISTERS_START, MAXWELL_DMA_CLASS,
};
use super::GpuMappings;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::Duration;

pub(crate) mod kickprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    pub const LOCKS: usize = 0;
    pub const ELIST: usize = 1;
    pub const PBREAD: usize = 2;
    pub const M3D: usize = 3;
    pub const MACRO: usize = 4;
    pub const CBUFWB: usize = 5;
    pub const SEMACQ: usize = 6;
    pub const SEMREL: usize = 7;
    pub const BARRIER: usize = 8;
    pub const ENQ: usize = 9;
    pub const FLUSHP: usize = 10;
    pub const DMA: usize = 11;
    pub const FERMI: usize = 12;
    pub const KEPLER: usize = 13;
    pub const PULLER: usize = 14;
    pub const SMALLRT: usize = 15;
    pub const DISJOINT: usize = 16;
    pub const KM_FLUSH: usize = 16;
    pub const KCU_FLUSH: usize = 17;
    pub const KC_LAUNCH: usize = 18;
    pub const KC_SYNC: usize = 19;
    pub const KC_EXEC: usize = 20;
    pub const KC_WB: usize = 21;
    pub const KC_RESOLVE: usize = 22;
    pub const COUNT: usize = 23;

    const NAMES: [&str; COUNT] = [
        "locks", "elist", "pbread", "m3d", "macro", "cbufwb", "semacq", "semrel", "barrier",
        "enq", "flushp", "dma", "fermi", "kepler", "puller", "smallrt", "kmflush", "kcuflush",
        "kclaunch", "kcsync", "kcexec", "kcwb", "kcresolve",
    ];

    static NS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
    static CALLS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];
    static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    static KICKS: AtomicU64 = AtomicU64::new(0);
    static WINDOW_KICKS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            let on = std::env::var_os("NEXIUM_KICKOFF_PROFILE").is_some();
            if on {
                log::warn!("[kickprof] armed");
            }
            on
        })
    }

    #[inline]
    pub fn start() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    #[inline]
    pub fn add(phase: usize, started: Option<Instant>) {
        let Some(started) = started else {
            return;
        };
        NS[phase].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CALLS[phase].fetch_add(1, Ordering::Relaxed);
    }

    pub fn kick_done(started: Option<Instant>) {
        let Some(started) = started else {
            return;
        };
        TOTAL_NS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let total_kicks = KICKS.fetch_add(1, Ordering::Relaxed) + 1;
        let window = WINDOW_KICKS.fetch_add(1, Ordering::Relaxed) + 1;
        if window < 64 {
            return;
        }
        WINDOW_KICKS.store(0, Ordering::Relaxed);
        let total = TOTAL_NS.swap(0, Ordering::Relaxed).max(1);
        let kicks = window as f64;
        let mut accounted = 0u64;
        let mut parts = String::new();
        for i in 0..COUNT {
            let ns = NS[i].swap(0, Ordering::Relaxed);
            let n = CALLS[i].swap(0, Ordering::Relaxed);
            if i < DISJOINT {
                accounted += ns;
            }
            if ns == 0 && n == 0 {
                continue;
            }
            parts.push_str(&format!(
                " {}={:.2}ms/{:.0}%/n{}",
                NAMES[i],
                ns as f64 / kicks / 1_000_000.0,
                ns as f64 * 100.0 / total as f64,
                n
            ));
        }
        let other = total.saturating_sub(accounted);
        log::warn!(
            "[kickprof] kicks={} (window {}) avg_ms total={:.2} other={:.2}/{:.0}% |{}",
            total_kicks,
            window,
            total as f64 / kicks / 1_000_000.0,
            other as f64 / kicks / 1_000_000.0,
            other as f64 * 100.0 / total as f64,
            parts
        );
    }
}

static UNMAPPED_PB_WARNS: AtomicU64 = AtomicU64::new(0);

fn warn_unmapped_pushbuffer(kind: &str, gpu_va: u64, mappings: &GpuMappings) {
    let n = UNMAPPED_PB_WARNS.fetch_add(1, Ordering::Relaxed);
    if n < 64 || n % 4096 == 0 {
        log::warn!(
            "pusher: {} gpu_va={:#x} not in GMMU — skipping (#{}) {}",
            kind,
            gpu_va,
            n,
            mappings.bracket(gpu_va)
        );
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CommandListHeader {
    pub address_lo: u32,
    pub address_hi_and_count: u32,
}

impl CommandListHeader {
    pub fn address(&self) -> u64 {
        ((self.address_hi_and_count as u64 & 0xFF) << 32) | (self.address_lo as u64 & 0xFFFF_FFFC)
    }

    pub fn entry_count(&self) -> u32 {
        (self.address_hi_and_count >> 10) & 0xFFFFF
    }

    pub fn no_prefetch(&self) -> bool {
        (self.address_hi_and_count & 0x8000_0000) != 0
    }

    pub fn not_main(&self) -> bool {
        (self.address_hi_and_count & 0x4000_0000) != 0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    IncreasingOld,
    Increasing,
    NonIncreasingOld,
    NonIncreasing,
    Inline,
    IncreaseOnce,
}

impl Mode {
    fn from_bits(v: u32) -> Option<Mode> {
        Some(match v {
            0 => Mode::IncreasingOld,
            1 => Mode::Increasing,
            2 => Mode::NonIncreasingOld,
            3 => Mode::NonIncreasing,
            4 => Mode::Inline,
            5 => Mode::IncreaseOnce,
            _ => return None,
        })
    }
}

const METHOD_BIND_OBJECT: u32 = 0x00;
const METHOD_SEMAPHORE_ADDR_HIGH: u32 = 0x04;
const METHOD_SEMAPHORE_ADDR_LOW: u32 = 0x05;
const METHOD_SEMAPHORE_PAYLOAD: u32 = 0x06;
const METHOD_SEMAPHORE_OPERATION: u32 = 0x07;
const METHOD_SEMAPHORE_ACQUIRE: u32 = 0x1A;
const METHOD_SEMAPHORE_RELEASE: u32 = 0x1B;
const METHOD_SYNCPOINT_PAYLOAD: u32 = 0x1C;
const METHOD_SYNCPOINT_OPERATION: u32 = 0x1D;
const NON_PULLER_METHODS: u32 = 0x40;

const POISON_SENTINEL: u32 = 0xBEEF_2929;

static GPU_SEM_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn ordered_gpu_sync_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXIUM_RELAXED_GPU_SYNC")
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

fn gpu_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_NVDRV_PROFILE").is_some())
}

fn profile_method_if_slow(
    started: Option<std::time::Instant>,
    entry_gpu_va: u64,
    word_index: usize,
    bound_class: u32,
    method: u32,
) {
    let Some(started) = started else {
        return;
    };
    let elapsed = started.elapsed();
    if elapsed >= Duration::from_millis(1) {
        log::warn!(
            "[nvprof] method entry={:#x} word={} class={:#x} method={:#x} elapsed_ms={:.3}",
            entry_gpu_va,
            word_index,
            bound_class,
            method,
            elapsed.as_secs_f64() * 1000.0,
        );
    }
}

#[derive(Default)]
struct DmaState {
    method: u32,
    subchannel: u32,
    method_count: u32,
    non_incrementing: bool,
    increment_once: bool,
}

#[derive(Default)]
struct PullerState {
    semaphore_addr_high: u32,
    semaphore_addr_low: u32,
    semaphore_payload: u32,
    syncpoint_payload: u32,
}

pub struct Pusher {
    pub syncpt_value: u32,
    bound_classes: [u32; 8],
    state: DmaState,
    puller: PullerState,
    entries_logged: u32,
    active_entry_gpu_va: u64,
    active_entry_cpu_va: u64,
    active_word_index: usize,
    active_header: u32,
    pub entry_word_limit: u32,
    pub renderer: Option<Arc<nexium_gpu::Renderer>>,
    vk_batch: Vec<nexium_gpu::draw::Maxwell3dDrawCall>,
}

impl Pusher {
    pub fn new() -> Self {
        Self {
            syncpt_value: 0,
            bound_classes: [0; 8],
            state: DmaState::default(),
            puller: PullerState::default(),
            entries_logged: 0,
            active_entry_gpu_va: 0,
            active_entry_cpu_va: 0,
            active_word_index: 0,
            active_header: 0,
            entry_word_limit: 0,
            renderer: None,
            vk_batch: Vec::new(),
        }
    }

    pub fn set_renderer(&mut self, r: Option<Arc<nexium_gpu::Renderer>>) {
        self.renderer = r;
    }

    pub(crate) fn flush_vk(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if self.vk_batch.is_empty() {
            return;
        }
        let kp = kickprof::start();
        if let Some(r) = self.renderer.clone() {
            super::vk_dispatch::flush_accum(
                &mut self.vk_batch,
                &r,
                mappings,
                mem_read,
                mem_write,
            );
        } else {
            self.vk_batch.clear();
        }
        kickprof::add(kickprof::FLUSHP, kp);
    }

    pub(crate) fn resolve_pending_compute(
        &self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        if !super::engines::maxwell_compute::has_pending_writebacks() {
            return;
        }
        let Some(renderer) = self.renderer.as_deref() else {
            return;
        };
        super::engines::maxwell_compute::resolve_pending_writebacks(renderer, mappings, mem_write);
    }

    fn sync_renderer_idle(&self, reason: &str) {
        let profile = gpu_profile_enabled();
        if !ordered_gpu_sync_enabled() {
            if profile {
                static SKIPS: AtomicU64 = AtomicU64::new(0);
                if SKIPS.fetch_add(1, Ordering::Relaxed) < 8 {
                    log::warn!("[gpu-sync] {} skipped (relaxed)", reason);
                }
            }
            return;
        }
        let started = profile.then(std::time::Instant::now);
        let Some(renderer) = self.renderer.clone() else {
            return;
        };
        if let Some(rt) = crate::render_thread::maybe_render_thread() {
            let (tx, rx) = mpsc::sync_channel(1);
            let r = renderer.clone();
            let job = Box::new(move || {
                r.wait_idle_if_dirty();
                let _ = tx.send(());
            }) as crate::render_thread::RenderJob;
            if !rt.submit_timeout(job, Duration::from_secs(3)) {
                log::warn!("[gpu-sync] {} render thread submit timeout", reason);
                return;
            }
            if rx.recv_timeout(Duration::from_secs(10)).is_err() {
                log::warn!("[gpu-sync] {} render thread idle timeout", reason);
            }
        } else if !renderer.wait_idle_if_dirty() {
            return;
        }
        if let Some(started) = started {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(1) {
                log::warn!(
                    "[gpu-sync] {} elapsed_ms={:.3}",
                    reason,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
    }

    pub fn process_gpfifo(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let cpu_addr = match mappings.cpu_address_for(address) {
            Some(c) => c,
            None => {
                warn_unmapped_pushbuffer("GPFIFO entry list", address, mappings);
                return;
            }
        };

        let bytes_needed = (num_entries as usize) * 8;
        if direct_forensics() {
            let remaining = mappings
                .cpu_range_for(address)
                .map(|(_, sz)| sz)
                .unwrap_or(0);
            if (bytes_needed as u64) > remaining {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 32 {
                    log::warn!(
                        "[el-overrun] entry_list gpu_va={:#x} num_entries={} need={:#x} remaining={:#x}",
                        address,
                        num_entries,
                        bytes_needed,
                        remaining
                    );
                }
            }
        }
        let _ = cpu_addr;
        let kp_elist = kickprof::start();
        let mut buf = vec![0u8; bytes_needed];
        read_gpu_scattered(mappings, address, &mut buf, mem_read);

        let decoded: Vec<CommandListHeader> = (0..num_entries as usize)
            .map(|i| {
                let off = i * 8;
                CommandListHeader {
                    address_lo: u32::from_le_bytes([
                        buf[off],
                        buf[off + 1],
                        buf[off + 2],
                        buf[off + 3],
                    ]),
                    address_hi_and_count: u32::from_le_bytes([
                        buf[off + 4],
                        buf[off + 5],
                        buf[off + 6],
                        buf[off + 7],
                    ]),
                }
            })
            .collect();
        let addrs: Vec<u64> = decoded.iter().map(|e| e.address()).collect();
        kickprof::add(kickprof::ELIST, kp_elist);

        for i in 0..num_entries as usize {
            let entry = decoded[i];
            self.entry_word_limit = if entry.entry_count() > 4096 {
                nearest_forward_gap(&addrs, i)
            } else {
                0
            };
            self.process_entry(
                &entry,
                mappings,
                maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_compute,
                kepler_memory,
                stats,
                mem_read,
                mem_write,
            );
        }
        self.entry_word_limit = 0;
        self.resolve_pending_compute(mappings, mem_write);
        self.flush_vk(mappings, mem_read, mem_write);
        if let Some(r) = self.renderer.clone() {
            let kp = kickprof::start();
            super::vk_dispatch::writeback_small_rts(&r, mappings, mem_write);
            kickprof::add(kickprof::SMALLRT, kp);
        }
    }

    pub fn process_entry(
        &mut self,
        entry: &CommandListHeader,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let address = entry.address();
        let mut word_count = entry.entry_count();
        let profile = gpu_profile_enabled();
        let started = profile.then(std::time::Instant::now);
        let methods_before = profile.then(|| stats.methods_dispatched.load(Ordering::Relaxed));

        if self.entry_word_limit != 0 && word_count > self.entry_word_limit {
            if direct_forensics() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 48 {
                    log::warn!(
                        "[pb-clamp] gpu_va={:#x} declared={} clamped_to={} (next entry overlaps)",
                        address,
                        word_count,
                        self.entry_word_limit
                    );
                }
            }
            word_count = self.entry_word_limit;
        }

        if self.entries_logged < 16 {
            log::info!(
                "gpfifo[{}]: gpu_va={:#x} word_count={} no_prefetch={} not_main={} raw_lo={:#010x} raw_hi={:#010x}",
                self.entries_logged,
                address,
                word_count,
                entry.no_prefetch(),
                entry.not_main(),
                entry.address_lo,
                entry.address_hi_and_count
            );
            self.entries_logged += 1;
        }

        if word_count == 0 || word_count > 0x100000 {
            return;
        }

        let cpu_addr = match mappings.cpu_address_for(address) {
            Some(c) => c,
            None => {
                warn_unmapped_pushbuffer("pushbuffer", address, mappings);
                return;
            }
        };

        let bytes_needed = (word_count as usize) * 4;
        if direct_forensics() {
            let remaining = mappings
                .cpu_range_for(address)
                .map(|(_, sz)| sz)
                .unwrap_or(0);
            if (bytes_needed as u64) > remaining {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 64 {
                    log::warn!(
                        "[pb-overrun] gpu_va={:#x} cpu={:#x} need={:#x} remaining_in_mapping={:#x} {}",
                        address,
                        cpu_addr,
                        bytes_needed,
                        remaining,
                        mappings.bracket(address)
                    );
                }
            }
            if word_count > 16384 {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 48 {
                    log::warn!(
                        "[pb-bogus-entry] gpu_va={:#x} word_count={} raw_lo={:#010x} raw_hi={:#010x} no_prefetch={} not_main={}",
                        address,
                        word_count,
                        entry.address_lo,
                        entry.address_hi_and_count,
                        entry.no_prefetch(),
                        entry.not_main()
                    );
                }
            }
        }
        let _ = cpu_addr;
        let kp_pb = kickprof::start();
        let mut buf = vec![0u8; bytes_needed];
        read_gpu_scattered(mappings, address, &mut buf, mem_read);

        let mut words: Vec<u32> = Vec::with_capacity(word_count as usize);
        for i in 0..word_count as usize {
            let off = i * 4;
            let w = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
            words.push(if w == POISON_SENTINEL { 0 } else { w });
        }
        kickprof::add(kickprof::PBREAD, kp_pb);

        self.active_entry_gpu_va = address;
        self.active_entry_cpu_va = cpu_addr;
        self.process_commands(
            &words,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_compute,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
        );
        if direct_forensics() && self.state.method_count > 0 {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 64 {
                log::warn!(
                    "[pb-leak] entry_gpu={:#x} words={} ended with method={:#x} count_left={} noninc={} subch={}",
                    address,
                    word_count,
                    self.state.method,
                    self.state.method_count,
                    self.state.non_incrementing,
                    self.state.subchannel
                );
            }
        }
        self.active_entry_gpu_va = 0;
        self.active_entry_cpu_va = 0;
        self.active_word_index = 0;
        self.active_header = 0;
        self.entry_word_limit = 0;
        if let (Some(started), Some(methods_before)) = (started, methods_before) {
            let elapsed = started.elapsed();
            if elapsed >= Duration::from_millis(5) {
                let methods = stats
                    .methods_dispatched
                    .load(Ordering::Relaxed)
                    .saturating_sub(methods_before);
                log::warn!(
                    "[nvprof] entry gpu_va={:#x} words={} methods={} elapsed_ms={:.3}",
                    address,
                    word_count,
                    methods,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }
    }

    fn process_commands(
        &mut self,
        commands: &[u32],
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let mut i = 0;
        let mut methods_dispatched = 0u64;
        while i < commands.len() {
            let header = commands[i];

            if self.state.method_count > 0 {
                self.active_word_index = i;
                let cls = self.bound_classes[self.state.subchannel as usize & 7];
                if suspicious_direct(cls, self.state.method, header) {
                    dump_direct_ctx(
                        self.active_entry_gpu_va,
                        self.active_entry_cpu_va,
                        self.state.method,
                        header,
                        cls,
                        self.state.method_count,
                        self.state.non_incrementing,
                        commands,
                        i,
                    );
                }
                self.dispatch_method(
                    header,
                    self.active_entry_gpu_va
                        .checked_add((self.active_word_index as u64) * 4),
                    mappings,
                    maxwell,
                    maxwell_dma,
                    fermi_2d,
                    kepler_compute,
                    kepler_memory,
                    stats,
                    mem_read,
                    mem_write,
                );
                methods_dispatched += 1;
                if !self.state.non_incrementing {
                    self.state.method = self.state.method.wrapping_add(1);
                }
                if self.state.increment_once {
                    self.state.non_incrementing = true;
                }
                self.state.method_count -= 1;
                i += 1;
                continue;
            }

            let method = header & 0x1FFF;
            let subchannel = (header >> 13) & 0x7;
            let arg_count = (header >> 16) & 0x1FFF;
            let mode_bits = (header >> 29) & 0x7;
            self.active_word_index = i;
            self.active_header = header;
            let Some(mode) = Mode::from_bits(mode_bits) else {
                log::trace!(
                    "pusher: unknown mode {} in header {:#010x}",
                    mode_bits,
                    header
                );
                i += 1;
                continue;
            };

            self.state.method = method;
            self.state.subchannel = subchannel;
            self.state.method_count = arg_count;

            if direct_forensics() && arg_count > 512 && mode != Mode::Inline {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 64 {
                    let cls = self.bound_classes[subchannel as usize & 7];
                    log::warn!(
                        "[pb-bighdr] entry_gpu={:#x} word={} of {} header={:#010x} method={:#x} count={} mode={:?} subch={} class={:#x} {}",
                        self.active_entry_gpu_va,
                        i,
                        commands.len(),
                        header,
                        method,
                        arg_count,
                        mode,
                        subchannel,
                        cls,
                        mappings.bracket(self.active_entry_gpu_va)
                    );
                }
            }

            match mode {
                Mode::Increasing | Mode::IncreasingOld => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = false;
                }
                Mode::NonIncreasing | Mode::NonIncreasingOld => {
                    self.state.non_incrementing = true;
                    self.state.increment_once = false;
                }
                Mode::IncreaseOnce => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = true;
                }
                Mode::Inline => {
                    self.state.method_count = 0;
                    self.active_word_index = i;
                    self.dispatch_method(
                        arg_count,
                        None,
                        mappings,
                        maxwell,
                        maxwell_dma,
                        fermi_2d,
                        kepler_compute,
                        kepler_memory,
                        stats,
                        mem_read,
                        mem_write,
                    );
                    methods_dispatched += 1;
                }
            }
            i += 1;
        }
        if methods_dispatched != 0 {
            stats
                .methods_dispatched
                .fetch_add(methods_dispatched, Ordering::Relaxed);
        }
    }

    fn dispatch_method(
        &mut self,
        arg: u32,
        arg_gpu_va: Option<u64>,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_compute: &mut KeplerCompute,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let method = self.state.method;
        let subchannel = self.state.subchannel as usize;
        let bound_class = self.bound_classes[subchannel & 7];
        let profile_started = gpu_profile_enabled().then(std::time::Instant::now);

        if method < NON_PULLER_METHODS {
            self.flush_vk(mappings, mem_read, mem_write);
            let kp = kickprof::start();
            self.handle_puller_method(method, arg, subchannel, mappings, mem_write);
            kickprof::add(kickprof::PULLER, kp);
            profile_method_if_slow(
                profile_started,
                self.active_entry_gpu_va,
                self.active_word_index,
                bound_class,
                method,
            );
            return;
        }

        if bound_class == 0xB197 {
            let arg = if method >= MACRO_REGISTERS_START {
                let kp = kickprof::start();
                let live = arg_gpu_va.and_then(|gpu_va| {
                    read_live_word(mappings, gpu_va, mem_read).map(|value| (gpu_va, value))
                });
                kickprof::add(kickprof::MACRO, kp);
                if let Some((gpu_va, value)) = live {
                    if value != arg && mme_param_trace() {
                        use std::sync::atomic::{AtomicU32, Ordering};
                        static N: AtomicU32 = AtomicU32::new(0);
                        if N.fetch_add(1, Ordering::Relaxed) < 256 {
                            log::warn!(
                                "[mme-param-refresh] method={:#x} gpu_va={:#x} snapshot={:#010x} live={:#010x}",
                                method,
                                gpu_va,
                                arg,
                                value
                            );
                        }
                    }
                    value
                } else {
                    arg
                }
            } else {
                arg
            };
            let is_last = self.state.method_count <= 1;
            let pre_draws = maxwell.regs.draw_count;
            let pre_clears = maxwell.regs.clear_count;
            let kp_m3d = kickprof::start();
            maxwell.dispatch_method(method, arg, is_last);
            kickprof::add(kickprof::M3D, kp_m3d);
            let upload_launch = maxwell.inline_upload_launch_pending();
            if upload_launch {
                self.flush_vk(mappings, mem_read, mem_write);
            }
            maxwell.process_inline_uploads(mappings, mem_read, mem_write);
            let d = maxwell.regs.draw_count - pre_draws;
            let c = maxwell.regs.clear_count - pre_clears;
            if d > 0 {
                stats.maxwell3d_draws.fetch_add(d, Ordering::Relaxed);
            }
            if c > 0 {
                stats.maxwell3d_clears.fetch_add(c, Ordering::Relaxed);
            }
            maxwell.record_method(method);

            if !maxwell.regs.pending_constbuf_writes.is_empty() {
                let kp = kickprof::start();
                let writes = std::mem::take(&mut maxwell.regs.pending_constbuf_writes);
                for (gpu_va, dword) in writes {
                    if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                        self.trace_constbuf_upload(maxwell, gpu_va, cpu, dword);
                        mem_write(cpu, &dword.to_le_bytes());
                    } else {
                        static DROPPED: AtomicU64 = AtomicU64::new(0);
                        let n = DROPPED.fetch_add(1, Ordering::Relaxed);
                        if n < 32 || n % 4096 == 0 {
                            log::warn!(
                                "[cbuf-upload-drop] #{} gpu_va={:#x} dword={:#010x}",
                                n,
                                gpu_va,
                                dword
                            );
                        }
                    }
                }
                kickprof::add(kickprof::CBUFWB, kp);
            }
            if !maxwell.regs.pending_semaphore_acquires.is_empty() {
                let kp = kickprof::start();
                let acquires = std::mem::take(&mut maxwell.regs.pending_semaphore_acquires);
                static NO_ACQUIRE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let skip =
                    *NO_ACQUIRE.get_or_init(|| std::env::var_os("NEXIUM_NO_SEM_ACQUIRE").is_some());
                if !skip {
                    static ACQUIRES: AtomicU64 = AtomicU64::new(0);
                    for (gpu_va, payload, _mode) in acquires {
                        let Some(cpu) = mappings.cpu_address_for(gpu_va) else {
                            static UNMAPPED: AtomicU64 = AtomicU64::new(0);
                            let n = UNMAPPED.fetch_add(1, Ordering::Relaxed);
                            if n < 32 {
                                log::warn!(
                                    "[sem-acquire] unmapped gpu_va={:#x} payload={:#x}",
                                    gpu_va,
                                    payload
                                );
                            }
                            continue;
                        };
                        let n = ACQUIRES.fetch_add(1, Ordering::Relaxed);
                        let start = std::time::Instant::now();
                        let mut last = 0u32;
                        loop {
                            let mut buf = [0u8; 4];
                            if mem_read(cpu, &mut buf) {
                                last = u32::from_le_bytes(buf);
                                if last == payload || (last.wrapping_sub(payload) as i32) >= 0 {
                                    if n < 8 {
                                        log::warn!(
                                            "[sem-acquire] #{} gpu_va={:#x} payload={:#x} value={:#x}",
                                            n,
                                            gpu_va,
                                            payload,
                                            last
                                        );
                                    }
                                    break;
                                }
                            }
                            if start.elapsed() >= std::time::Duration::from_millis(100) {
                                static TIMEOUTS: AtomicU64 = AtomicU64::new(0);
                                let t = TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                                if t < 32 || t % 1024 == 0 {
                                    log::warn!(
                                        "[sem-acquire] timeout gpu_va={:#x} payload={:#x} last={:#x}",
                                        gpu_va,
                                        payload,
                                        last
                                    );
                                }
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_micros(100));
                        }
                    }
                }
                kickprof::add(kickprof::SEMACQ, kp);
            }
            if !maxwell.regs.pending_semaphore_writes.is_empty() {
                self.resolve_pending_compute(mappings, mem_write);
                let writes = std::mem::take(&mut maxwell.regs.pending_semaphore_writes);
                let mut renderer_ordered = false;
                for write in writes {
                    if write.requires_renderer_completion() && !renderer_ordered {
                        self.flush_vk(mappings, mem_read, mem_write);
                        let kp = kickprof::start();
                        self.sync_renderer_idle("report-semaphore");
                        kickprof::add(kickprof::SEMREL, kp);
                        renderer_ordered = true;
                    }
                    if let Some(cpu) = mappings.cpu_address_for(write.gpu_va) {
                        let ok = if write.long {
                            let ts = GPU_SEM_TICK.fetch_add(1, Ordering::Relaxed);
                            let mut buf = [0u8; 16];
                            buf[0..8].copy_from_slice(&(write.payload as u64).to_le_bytes());
                            buf[8..16].copy_from_slice(&ts.to_le_bytes());
                            mem_write(cpu, &buf)
                        } else {
                            mem_write(cpu, &write.payload.to_le_bytes())
                        };
                        log::trace!(
                            "pusher: fence release gpu_va={:#x} cpu={:#x} payload={:#x} long={} write_ok={}",
                            write.gpu_va,
                            cpu,
                            write.payload,
                            write.long,
                            ok
                        );
                        stats.fence_releases.fetch_add(1, Ordering::Relaxed);
                    } else {
                        log::warn!(
                            "pusher: fence release gpu_va={:#x} not mapped — payload={:#x} dropped",
                            write.gpu_va,
                            write.payload
                        );
                    }
                }
            }
            if !maxwell.pending_draws.is_empty() {
                let draws = std::mem::take(&mut maxwell.pending_draws);
                if let Some(r) = self.renderer.clone() {
                    let kp = kickprof::start();
                    super::vk_dispatch::enqueue_draws(
                        &draws,
                        &mut self.vk_batch,
                        mappings,
                        maxwell,
                        &r,
                        maxwell_dma,
                        mem_read,
                        mem_write,
                    );
                    kickprof::add(kickprof::ENQ, kp);
                } else {
                    sw_renderer::execute_draws(&draws, mappings, maxwell_dma, mem_read, mem_write);
                }
            }
            let barrier_flushes = std::mem::take(&mut maxwell.regs.pending_barrier_flushes);
            let texture_invalidates =
                std::mem::take(&mut maxwell.regs.pending_texture_cache_invalidates);
            if barrier_flushes != 0 || texture_invalidates != 0 {
                self.flush_vk(mappings, mem_read, mem_write);
                let kp = kickprof::start();
                self.sync_renderer_idle("maxwell-barrier");
                if texture_invalidates != 0 {
                    static CLEAR_ON_TIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    let clear_on_tic = *CLEAR_ON_TIC
                        .get_or_init(|| std::env::var_os("NEXIUM_TIC_INVALIDATE_CLEAR").is_some());
                    if clear_on_tic {
                        if let Some(r) = self.renderer.clone() {
                            r.clear_texture_cache();
                        }
                    }
                }
                kickprof::add(kickprof::BARRIER, kp);
                if std::env::var_os("NEXIUM_MW3D_SYNC_DBG").is_some() {
                    log::warn!(
                        "[gpu-sync] maxwell barriers={} texture_invalidates={}",
                        barrier_flushes,
                        texture_invalidates
                    );
                }
            }
        } else if bound_class == MAXWELL_DMA_CLASS {
            self.flush_vk(mappings, mem_read, mem_write);
            let kp = kickprof::start();
            if method == super::engines::maxwell_dma::M_LAUNCH_DMA {
                if super::engines::maxwell_compute::has_pending_writebacks() {
                    let spans = maxwell_dma.transfer_spans();
                    let overlaps = spans.iter().any(|&(gpu_va, size)| {
                        let cpu_addr = mappings
                            .cpu_range_for(gpu_va)
                            .map(|(cpu, _)| cpu)
                            .unwrap_or(0);
                        super::engines::maxwell_compute::pending_writeback_overlaps(
                            gpu_va, cpu_addr, size,
                        )
                    });
                    if overlaps {
                        self.resolve_pending_compute(mappings, mem_write);
                    }
                }
                if let Some(r) = self.renderer.clone() {
                    maxwell_dma.stage_rt_source(arg, mappings, &r, mem_write);
                }
            }
            let pre = maxwell_dma.blit_count;
            maxwell_dma.dispatch_method(method, arg, mappings, mem_read, mem_write);
            let n = maxwell_dma.blit_count - pre;
            if n > 0 {
                stats.maxwell_dma_blits.fetch_add(n, Ordering::Relaxed);
            }
            kickprof::add(kickprof::DMA, kp);
        } else if bound_class == FERMI_2D_CLASS {
            self.flush_vk(mappings, mem_read, mem_write);
            let kp = kickprof::start();
            let r = self.renderer.clone();
            let pre = fermi_2d.blit_count;
            fermi_2d.dispatch_method(method, arg, mappings, r.as_deref(), mem_read, mem_write);
            let n = fermi_2d.blit_count - pre;
            if n > 0 {
                stats.fermi_2d_blits.fetch_add(n, Ordering::Relaxed);
            }
            kickprof::add(kickprof::FERMI, kp);
        } else if bound_class == KEPLER_MEMORY_CLASS {
            self.flush_vk(mappings, mem_read, mem_write);
            let kp = kickprof::start();
            kepler_memory.dispatch_method(method, arg, mappings, mem_read, mem_write);
            kickprof::add(kickprof::KEPLER, kp);
        } else if bound_class == KEPLER_COMPUTE_CLASS {
            self.flush_vk(mappings, mem_read, mem_write);
            let kp = kickprof::start();
            let is_last = self.state.method_count <= 1;
            kepler_compute.dispatch_method(
                method,
                arg,
                is_last,
                self.renderer.as_ref(),
                mappings,
                mem_read,
                mem_write,
            );
            kickprof::add(kickprof::KEPLER, kp);
        } else {
            if bound_class == 0xB1C0 {
                use std::sync::atomic::{AtomicU64, Ordering as O2};
                use std::sync::{Mutex, OnceLock};
                static COMPUTE_METHODS: AtomicU64 = AtomicU64::new(0);
                let n = COMPUTE_METHODS.fetch_add(1, O2::Relaxed) + 1;
                if method == 0xAF {
                    log::warn!("[compute] LAUNCH (0xAF) arg={:#x} count={}", arg, n);
                }
                if std::env::var_os("NEXIUM_COMPUTE_DBG").is_some() {
                    static METHS: OnceLock<Mutex<std::collections::BTreeMap<u32, u64>>> =
                        OnceLock::new();
                    let m = METHS.get_or_init(|| Mutex::new(std::collections::BTreeMap::new()));
                    if let Ok(mut map) = m.lock() {
                        *map.entry(method).or_insert(0) += 1;
                        if n % 10000 == 0 {
                            let s: Vec<String> =
                                map.iter().map(|(k, v)| format!("{:#x}:{}", k, v)).collect();
                            log::warn!(
                                "[compute-methods] n={} distinct={} [{}]",
                                n,
                                map.len(),
                                s.join(" ")
                            );
                        }
                    }
                    if n <= 90 {
                        log::warn!("[compute-seq] #{} method={:#x} arg={:#x}", n, method, arg);
                    }
                }
            }
            if bound_class != 0 {
                use std::sync::{Mutex, OnceLock};
                static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
                let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
                if let Ok(mut s) = seen.lock() {
                    if s.insert(bound_class) {
                        log::warn!(
                            "[gpu-unhandled-class] subch={} class={:#x} method={:#x} arg={:#x} NOT dispatched (0xb1c0=KeplerCompute) — fence may never release",
                            subchannel,
                            bound_class,
                            method,
                            arg
                        );
                    }
                }
            }
            log::trace!(
                "pusher: subch={} class={:#x} method={:#x} arg={:#x} (unsupported class)",
                subchannel,
                bound_class,
                method,
                arg
            );
        }
        profile_method_if_slow(
            profile_started,
            self.active_entry_gpu_va,
            self.active_word_index,
            bound_class,
            method,
        );
    }

    fn handle_puller_method(
        &mut self,
        method: u32,
        arg: u32,
        subchannel: usize,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        match method {
            METHOD_BIND_OBJECT => {
                let class = arg & 0xFFFF;
                if is_known_gpu_class(class) {
                    self.bound_classes[subchannel & 7] = class;
                    log::debug!("puller: BindObject subch={} class={:#x}", subchannel, class);
                } else {
                    log::trace!(
                        "puller: BindObject subch={} class={:#x} rejected (unknown) keeping {:#x}",
                        subchannel,
                        class,
                        self.bound_classes[subchannel & 7]
                    );
                }
            }
            METHOD_SEMAPHORE_ADDR_HIGH => {
                if arg <= 0xFF {
                    self.puller.semaphore_addr_high = arg;
                }
            }
            METHOD_SEMAPHORE_ADDR_LOW => self.puller.semaphore_addr_low = arg,
            METHOD_SEMAPHORE_PAYLOAD => self.puller.semaphore_payload = arg,
            METHOD_SEMAPHORE_OPERATION => {
                if arg & 0xF == 0x2 {
                    self.resolve_pending_compute(mappings, mem_write);
                    let payload = self.puller.semaphore_payload;
                    self.write_semaphore(mappings, mem_write, payload, true);
                }
            }
            METHOD_SEMAPHORE_RELEASE => {
                self.resolve_pending_compute(mappings, mem_write);
                self.write_semaphore(mappings, mem_write, arg, false);
            }
            METHOD_SEMAPHORE_ACQUIRE => {}
            METHOD_SYNCPOINT_PAYLOAD => self.puller.syncpoint_payload = arg,
            METHOD_SYNCPOINT_OPERATION => {
                let op = arg & 0xFF;
                if op == 1 {
                    self.syncpt_value = self.syncpt_value.wrapping_add(1);
                    log::trace!("puller: SyncpointIncrement → {}", self.syncpt_value);
                }
            }
            _ => {
                log::trace!("puller: method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn write_semaphore(
        &self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        payload: u32,
        long: bool,
    ) {
        let gpu_va = ((self.puller.semaphore_addr_high as u64) << 32)
            | (self.puller.semaphore_addr_low as u64);
        if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
            if long {
                let ts = GPU_SEM_TICK.fetch_add(1, Ordering::Relaxed);
                let mut buf = [0u8; 16];
                buf[0..8].copy_from_slice(&(payload as u64).to_le_bytes());
                buf[8..16].copy_from_slice(&ts.to_le_bytes());
                mem_write(cpu, &buf);
            } else {
                mem_write(cpu, &payload.to_le_bytes());
            }
            log::trace!(
                "puller: semaphore write gpu_va={:#x} cpu={:#x} payload={:#x} long={}",
                gpu_va,
                cpu,
                payload,
                long
            );
        } else {
            log::warn!(
                "puller: semaphore write gpu_va={:#x} not mapped — payload={:#x} dropped",
                gpu_va,
                payload
            );
        }
    }

    fn trace_constbuf_upload(&self, maxwell: &Maxwell3D, gpu_va: u64, cpu: u64, dword: u32) {
        let Some((watch_va, watch_len)) = constbuf_upload_watch() else {
            return;
        };
        if gpu_va >= watch_va.saturating_add(watch_len) || gpu_va.saturating_add(4) <= watch_va {
            return;
        }
        let cb_addr = ((maxwell.regs.constbuf_selector_addr_hi as u64) << 32)
            | maxwell.regs.constbuf_selector_addr_lo as u64;
        log::warn!(
            "[cbuf-upload] entry_gpu={:#x} entry_cpu={:#x} word={} header={:#010x} subch={} method={:#x} target={:#x} cpu={:#x} cb={:#x} size={:#x} off={:#x} dword={:#010x} float={:.6}",
            self.active_entry_gpu_va,
            self.active_entry_cpu_va,
            self.active_word_index,
            self.active_header,
            self.state.subchannel,
            self.state.method,
            gpu_va,
            cpu,
            cb_addr,
            maxwell.regs.constbuf_selector_size,
            gpu_va.saturating_sub(cb_addr),
            dword,
            f32::from_bits(dword)
        );
    }
}

fn direct_forensics() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_FORENSICS").is_some())
}

pub(crate) fn nearest_forward_gap(addrs: &[u64], i: usize) -> u32 {
    let cur = addrs[i];
    let mut best: Option<u64> = None;
    for (j, &a) in addrs.iter().enumerate() {
        if j == i || a <= cur {
            continue;
        }
        let d = a - cur;
        if best.map_or(true, |b| d < b) {
            best = Some(d);
        }
    }
    match best {
        Some(d) if d <= 0x100000 * 4 => (d / 4) as u32,
        _ => 0,
    }
}

fn read_gpu_scattered(
    mappings: &GpuMappings,
    gpu_va: u64,
    out: &mut [u8],
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) {
    let mut off = 0usize;
    let mut va = gpu_va;
    while off < out.len() {
        match mappings.cpu_range_for(va) {
            Some((cpu, remain)) if remain > 0 => {
                let take = (remain as usize).min(out.len() - off);
                let _ = mem_read(cpu, &mut out[off..off + take]);
                off += take;
                va = va.wrapping_add(take as u64);
            }
            _ => {
                let page_left = (0x1000 - (va & 0xFFF)) as usize;
                let step = page_left.min(out.len() - off).max(1);
                off += step;
                va = va.wrapping_add(step as u64);
            }
        }
    }
}

fn read_live_word(
    mappings: &GpuMappings,
    gpu_va: u64,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> Option<u32> {
    let cpu_addr = mappings.cpu_address_for(gpu_va)?;
    let mut bytes = [0u8; 4];
    if !mem_read(cpu_addr, &mut bytes) {
        return None;
    }
    Some(u32::from_le_bytes(bytes))
}

fn mme_param_trace() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MME_PARAM_TRACE").is_some())
}

fn suspicious_direct(class: u32, method: u32, arg: u32) -> bool {
    if !direct_forensics() {
        return false;
    }
    if class == 0xB197 {
        let hi_reg = method == 0x582
            || method == 0x6c0
            || method == 0x8e1
            || method == 0x554
            || (method >= 0x200 && method < 0x280 && (method & 0xF) == 0);
        if hi_reg && arg > 0xFF {
            return true;
        }
        if (method == 0x47 || method == 0x48) && arg > 0x1000 {
            return true;
        }
    }
    if method == METHOD_SEMAPHORE_ADDR_HIGH && arg > 0xFF {
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn dump_direct_ctx(
    entry_gpu_va: u64,
    entry_cpu_va: u64,
    method: u32,
    arg: u32,
    class: u32,
    method_count: u32,
    non_incrementing: bool,
    commands: &[u32],
    i: usize,
) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 64 {
        return;
    }
    let lo = i.saturating_sub(3);
    let hi = (i + 6).min(commands.len());
    log::warn!(
        "[pb-garbage] entry_gpu={:#x} cpu={:#x} word={} class={:#x} method={:#x} arg={:#010x} mcount={} noninc={} words[{}..{}]={:08x?}",
        entry_gpu_va,
        entry_cpu_va,
        i,
        class,
        method,
        arg,
        method_count,
        non_incrementing,
        lo,
        hi,
        &commands[lo..hi]
    );
}

fn is_known_gpu_class(class: u32) -> bool {
    matches!(
        class,
        0xB197 | MAXWELL_DMA_CLASS | FERMI_2D_CLASS | KEPLER_MEMORY_CLASS | KEPLER_COMPUTE_CLASS
    )
}

fn constbuf_upload_watch() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static WATCH: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *WATCH.get_or_init(|| {
        let spec = std::env::var("NEXIUM_CBUF_UPLOAD_WATCH").ok()?;
        let (va, len) = spec.trim().split_once(':')?;
        let va = parse_u64ish(va.trim())?;
        let len = parse_u64ish(len.trim()).unwrap_or(4);
        (va != 0 && len != 0).then_some((va, len))
    })
}

fn parse_u64ish(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

impl Default for Pusher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn macro_argument_refresh_reads_the_live_pushbuffer_word() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x5000, 0x1000, 0x9000, 1);
        let expected = 0x1234_5678u32;
        let read = |cpu_addr: u64, out: &mut [u8]| {
            if cpu_addr != 0x9020 || out.len() != 4 {
                return false;
            }
            out.copy_from_slice(&expected.to_le_bytes());
            true
        };

        assert_eq!(read_live_word(&mappings, 0x5020, &read), Some(expected));
    }

    #[test]
    fn macro_argument_refresh_preserves_snapshot_when_live_read_fails() {
        let mappings = GpuMappings::new();
        let read = |_: u64, _: &mut [u8]| false;
        assert_eq!(read_live_word(&mappings, 0x5020, &read), None);
    }

    #[test]
    fn split_non_incrementing_upload_preserves_payload_state() {
        let mut pusher = Pusher::new();
        let mappings = GpuMappings::new();
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let mem_read = |_: u64, _: &mut [u8]| true;
        let mem_write = |_: u64, _: &[u8]| true;

        let mut first = vec![0x8100_0000; 9];
        first.push(0x6300_006D);
        pusher.process_commands(
            &first,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );
        assert_eq!(pusher.state.method, 0x6D);
        assert_eq!(pusher.state.method_count, 768);
        assert!(pusher.state.non_incrementing);

        let payload: Vec<u32> = (0..768).map(|i| 0x1000_0000 | i).collect();
        pusher.process_commands(
            &payload,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );
        assert_eq!(pusher.state.method_count, 0);

        pusher.process_commands(
            &[0x2001_0100, 0x1234_5678],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );
        assert_eq!(pusher.state.method, 0x101);
        assert_eq!(pusher.state.method_count, 0);
    }

    #[test]
    fn method_count_above_entry_limit_survives_boundary() {
        let mut pusher = Pusher::new();
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let mem_read = |_: u64, _: &mut [u8]| true;
        let mem_write = |_: u64, _: &[u8]| true;
        let entry = CommandListHeader {
            address_lo: 0x4000,
            address_hi_and_count: 1 << 10,
        };
        pusher.state.method = 0x101;
        pusher.state.method_count = 4097;

        pusher.process_entry(
            &entry,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );
        assert_eq!(pusher.state.method_count, 4096);
    }

    #[test]
    fn maxwell3d_inline_upload_writes_mapped_bytes() {
        let mut pusher = Pusher::new();
        pusher.bound_classes[0] = 0xB197;
        let mut mappings = GpuMappings::new();
        mappings.add(0x6000, 0x100, 0x1000, 1);
        let mut maxwell = Maxwell3D::new();
        let mut maxwell_dma = MaxwellDma::new();
        let mut fermi_2d = Fermi2D::new();
        let mut kepler_compute = KeplerCompute::new();
        let mut kepler_memory = KeplerMemory::new();
        let stats = PipelineStats::default();
        let memory = Arc::new(Mutex::new(vec![0u8; 0x100]));
        let read_memory = memory.clone();
        let mem_read = move |cpu: u64, out: &mut [u8]| {
            let memory = read_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(out.len()) > memory.len() {
                return false;
            }
            out.copy_from_slice(&memory[start..start + out.len()]);
            true
        };
        let write_memory = memory.clone();
        let mem_write = move |cpu: u64, data: &[u8]| {
            let mut memory = write_memory.lock().unwrap();
            let start = cpu.saturating_sub(0x1000) as usize;
            if start.saturating_add(data.len()) > memory.len() {
                return false;
            }
            memory[start..start + data.len()].copy_from_slice(data);
            true
        };

        let setup = [
            0x200D_0060,
            6,
            1,
            0,
            0x6000,
            6,
            0,
            6,
            1,
            1,
            0,
            0,
            0,
            1,
        ];
        pusher.process_commands(
            &setup,
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );
        pusher.process_commands(
            &[0x6002_006D, 0x1122_3344, 0x5566],
            &mappings,
            &mut maxwell,
            &mut maxwell_dma,
            &mut fermi_2d,
            &mut kepler_compute,
            &mut kepler_memory,
            &stats,
            &mem_read,
            &mem_write,
        );

        let memory = memory.lock().unwrap();
        assert_eq!(&memory[..6], &[0x44, 0x33, 0x22, 0x11, 0x66, 0x55]);
        assert_eq!(maxwell.reg_file[0x47], 0);
        assert_eq!(maxwell.reg_file[0x48], 0);
    }
}

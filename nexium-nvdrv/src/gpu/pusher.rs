use super::super::PipelineStats;
use super::engines::{
    sw_renderer, Fermi2D, KeplerMemory, Maxwell3D, MaxwellDma, FERMI_2D_CLASS, KEPLER_MEMORY_CLASS,
    MAXWELL_DMA_CLASS,
};
use super::GpuMappings;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CommandListHeader {
    pub address_lo: u32,
    pub address_hi_and_count: u32,
}

impl CommandListHeader {
    pub fn address(&self) -> u64 {
        ((self.address_hi_and_count as u64 & 0xFF) << 32) | self.address_lo as u64
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

static GPU_SEM_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

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
    ) {
        if self.vk_batch.is_empty() {
            return;
        }
        if let Some(r) = self.renderer.clone() {
            super::vk_dispatch::flush_accum(&mut self.vk_batch, &r, mappings, mem_read);
        } else {
            self.vk_batch.clear();
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
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let cpu_addr = mappings.cpu_address_for(address).unwrap_or(address);

        let bytes_needed = (num_entries as usize) * 8;
        let mut buf = vec![0u8; bytes_needed];
        if !mem_read(cpu_addr, &mut buf) {
            log::debug!(
                "pusher: failed to read GPFIFO entries at cpu {:#x} (input addr {:#x})",
                cpu_addr,
                address
            );
            return;
        }

        for i in 0..num_entries as usize {
            let off = i * 8;
            let entry = CommandListHeader {
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
            };
            self.process_entry(
                &entry,
                mappings,
                maxwell,
                maxwell_dma,
                fermi_2d,
                kepler_memory,
                stats,
                mem_read,
                mem_write,
            );
        }
        self.flush_vk(mappings, mem_read);
    }

    pub fn process_entry(
        &mut self,
        entry: &CommandListHeader,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let address = entry.address();
        let word_count = entry.entry_count();

        if self.entries_logged < 16 {
            log::info!(
                "gpfifo[{}]: gpu_va={:#x} word_count={} no_prefetch={} not_main={} raw_lo={:#010x} raw_hi={:#010x}",
                self.entries_logged, address, word_count, entry.no_prefetch(), entry.not_main(),
                entry.address_lo, entry.address_hi_and_count
            );
            self.entries_logged += 1;
        }

        if word_count == 0 || word_count > 0x100000 {
            return;
        }

        let cpu_addr = mappings.cpu_address_for(address).unwrap_or(address);

        let bytes_needed = (word_count as usize) * 4;
        let mut buf = vec![0u8; bytes_needed];
        if !mem_read(cpu_addr, &mut buf) {
            log::debug!("pusher: failed to read cmd buffer at cpu {:#x}", cpu_addr);
            return;
        }

        let mut words: Vec<u32> = Vec::with_capacity(word_count as usize);
        for i in 0..word_count as usize {
            let off = i * 4;
            words.push(u32::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
            ]));
        }

        self.process_commands(
            &words,
            mappings,
            maxwell,
            maxwell_dma,
            fermi_2d,
            kepler_memory,
            stats,
            mem_read,
            mem_write,
        );
    }

    fn process_commands(
        &mut self,
        commands: &[u32],
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
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
                self.dispatch_method(
                    header,
                    mappings,
                    maxwell,
                    maxwell_dma,
                    fermi_2d,
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
                    self.dispatch_method(
                        arg_count,
                        mappings,
                        maxwell,
                        maxwell_dma,
                        fermi_2d,
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
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        maxwell_dma: &mut MaxwellDma,
        fermi_2d: &mut Fermi2D,
        kepler_memory: &mut KeplerMemory,
        stats: &PipelineStats,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let method = self.state.method;
        let subchannel = self.state.subchannel as usize;

        if method < NON_PULLER_METHODS {
            self.flush_vk(mappings, mem_read);
            self.handle_puller_method(method, arg, subchannel, mappings, mem_write);
            return;
        }

        let bound_class = self.bound_classes[subchannel & 7];
        if bound_class == 0xB197 {
            let is_last = self.state.method_count <= 1;
            let pre_draws = maxwell.regs.draw_count;
            let pre_clears = maxwell.regs.clear_count;
            maxwell.dispatch_method(method, arg, is_last);
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
                let writes = std::mem::take(&mut maxwell.regs.pending_constbuf_writes);
                for (gpu_va, dword) in writes {
                    if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                        mem_write(cpu, &dword.to_le_bytes());
                    }
                }
            }
            if !maxwell.regs.pending_semaphore_writes.is_empty() {
                self.flush_vk(mappings, mem_read);
                let writes = std::mem::take(&mut maxwell.regs.pending_semaphore_writes);
                for (gpu_va, payload) in writes {
                    if let Some(cpu) = mappings.cpu_address_for(gpu_va) {
                        let ok = mem_write(cpu, &payload.to_le_bytes());
                        log::trace!(
                            "pusher: fence release gpu_va={:#x} cpu={:#x} payload={:#x} write_ok={}",
                            gpu_va, cpu, payload, ok
                        );
                        stats.fence_releases.fetch_add(1, Ordering::Relaxed);
                    } else {
                        log::warn!(
                            "pusher: fence release gpu_va={:#x} not mapped — payload={:#x} dropped",
                            gpu_va,
                            payload
                        );
                    }
                }
            }
            if !maxwell.pending_draws.is_empty() {
                let draws = std::mem::take(&mut maxwell.pending_draws);
                if let Some(r) = self.renderer.clone() {
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
                } else {
                    sw_renderer::execute_draws(&draws, mappings, maxwell_dma, mem_read, mem_write);
                }
            }
        } else if bound_class == MAXWELL_DMA_CLASS {
            self.flush_vk(mappings, mem_read);
            if method == super::engines::maxwell_dma::M_LAUNCH_DMA {
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
        } else if bound_class == FERMI_2D_CLASS {
            self.flush_vk(mappings, mem_read);
            let pre = fermi_2d.blit_count;
            fermi_2d.dispatch_method(method, arg, mappings, mem_read, mem_write);
            let n = fermi_2d.blit_count - pre;
            if n > 0 {
                stats.fermi_2d_blits.fetch_add(n, Ordering::Relaxed);
            }
        } else if bound_class == KEPLER_MEMORY_CLASS {
            self.flush_vk(mappings, mem_read);
            kepler_memory.dispatch_method(method, arg, mappings, mem_write);
        } else {
            if bound_class == 0xB1C0 {
                use std::sync::atomic::{AtomicU64, Ordering as O2};
                static COMPUTE_METHODS: AtomicU64 = AtomicU64::new(0);
                let n = COMPUTE_METHODS.fetch_add(1, O2::Relaxed) + 1;
                if n <= 3 || n % 2000 == 0 {
                    log::warn!("[compute] 0xb1c0 method={:#x} count={} (dropped)", method, n);
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
                            subchannel, bound_class, method, arg
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
                self.bound_classes[subchannel & 7] = arg & 0xFFFF;
                log::debug!(
                    "puller: BindObject subch={} class={:#x}",
                    subchannel,
                    arg & 0xFFFF
                );
            }
            METHOD_SEMAPHORE_ADDR_HIGH => self.puller.semaphore_addr_high = arg,
            METHOD_SEMAPHORE_ADDR_LOW => self.puller.semaphore_addr_low = arg,
            METHOD_SEMAPHORE_PAYLOAD => self.puller.semaphore_payload = arg,
            METHOD_SEMAPHORE_OPERATION => {
                if arg & 0xF == 0x2 {
                    let payload = self.puller.semaphore_payload;
                    self.write_semaphore(mappings, mem_write, payload, true);
                }
            }
            METHOD_SEMAPHORE_RELEASE => {
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
}

impl Default for Pusher {
    fn default() -> Self {
        Self::new()
    }
}

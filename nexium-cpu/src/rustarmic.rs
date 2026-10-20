use nexium_memory::Perm;
use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::jit::page_table::PageTableMem;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::{CpuEvent, FaultSnapshot, HaltHandle, MappingView};

const NULL_SKIP_MAX: u32 = 64;

/// Upper bound on the guest address width covered by the fastmem page table.
/// The Switch address space is 39 bits; a larger arena only means addresses
/// above this take the callbacks.
const FASTMEM_TABLE_MAX_BITS: u32 = 39;

const CODE_CACHE_DEFAULT_MB: usize = 1024;
const CODE_CACHE_MAX_MB: usize = 2047;

#[derive(Clone, Copy, Debug)]
struct Region {
    va: u64,
    end: u64,
    host_ptr: *mut u8,
    perm: Perm,
}
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

/// Mapped regions.  `list` keeps insertion order because lookups use
/// first-match semantics when regions overlap.  `sorted` holds the non-empty
/// regions ordered by address and is used for binary search whenever no two
/// of them overlap (the normal case), where "first match" and "the only
/// match" coincide.
struct Regions {
    list: Vec<Region>,
    sorted: Vec<Region>,
    disjoint: bool,
}

impl Default for Regions {
    fn default() -> Self {
        Self {
            list: Vec::new(),
            sorted: Vec::new(),
            disjoint: true,
        }
    }
}

impl Regions {
    fn rebuild_index(&mut self) {
        self.sorted.clear();
        self.sorted
            .extend(self.list.iter().copied().filter(|r| r.end > r.va));
        self.sorted.sort_by_key(|r| r.va);
        self.disjoint = self.sorted.windows(2).all(|w| w[0].end <= w[1].va);
    }

    fn push(&mut self, region: Region) {
        self.list.push(region);
        if self.disjoint {
            if region.end <= region.va {
                return;
            }
            let index = self.sorted.partition_point(|r| r.va < region.va);
            let after_previous = index == 0 || self.sorted[index - 1].end <= region.va;
            let before_next = self.sorted.get(index).is_none_or(|next| region.end <= next.va);
            if after_previous && before_next {
                self.sorted.insert(index, region);
                return;
            }
        }
        self.rebuild_index();
    }

    fn remove_sorted_range(&mut self, va: u64, end: u64) {
        let first = self.sorted.partition_point(|r| r.end <= va);
        let last = first
            + self.sorted[first..]
                .iter()
                .take_while(|r| r.va < end)
                .count();
        if first == last {
            return;
        }
        let mut pieces = Vec::with_capacity(2);
        let head = self.sorted[first];
        if head.va < va {
            pieces.push(Region { end: va, ..head });
        }
        let tail = self.sorted[last - 1];
        if tail.end > end {
            let host_ptr = tail.host_ptr.wrapping_add((end - tail.va) as usize);
            pieces.push(Region { va: end, host_ptr, ..tail });
        }
        self.sorted.splice(first..last, pieces);
    }

    /// Unmaps exactly `[va, end)`: a region partly inside keeps its pieces
    /// outside the range (with the host pointer advanced for the upper
    /// piece), as the Dynarmic backend's page-granular unmap does.  Returns
    /// the unmapped part of the removed extent, the only pages whose
    /// translation can change.
    fn remove_range(&mut self, va: u64, end: u64) -> Option<(u64, u64)> {
        if end <= va {
            return None;
        }
        let mut extent: Option<(u64, u64)> = None;
        let mut kept = Vec::with_capacity(self.list.len() + 1);
        for r in self.list.drain(..) {
            if r.end <= va || r.va >= end || r.end <= r.va {
                kept.push(r);
                continue;
            }
            let (lo, hi) = (r.va.max(va), r.end.min(end));
            extent = Some(match extent {
                Some((a, b)) => (a.min(lo), b.max(hi)),
                None => (lo, hi),
            });
            if r.va < va {
                kept.push(Region { end: va, ..r });
            }
            if r.end > end {
                let host_ptr = r.host_ptr.wrapping_add((end - r.va) as usize);
                kept.push(Region { va: end, host_ptr, ..r });
            }
        }
        self.list = kept;
        if extent.is_some() {
            if self.disjoint {
                self.remove_sorted_range(va, end);
            } else {
                self.rebuild_index();
            }
        }
        extent
    }

    /// Removes every region overlapping `[va, end)` and returns the removed
    /// regions' combined extent.  This was `unmap_host` before it became
    /// exact (`remove_range`); kept for the tests that pin the page-table
    /// refresh over a whole removed extent.
    #[cfg(test)]
    fn remove_overlapping(&mut self, va: u64, end: u64) -> Option<(u64, u64)> {
        let mut extent: Option<(u64, u64)> = None;
        self.list.retain(|r| {
            let keep = r.end <= va || r.va >= end;
            if !keep {
                extent = Some(match extent {
                    Some((lo, hi)) => (lo.min(r.va), hi.max(r.end)),
                    None => (r.va, r.end),
                });
            }
            keep
        });
        if extent.is_some() {
            self.rebuild_index();
        }
        extent
    }

    fn find(&self, va: u64) -> Option<&Region> {
        if self.disjoint {
            let idx = self.sorted.partition_point(|r| r.va <= va);
            let r = self.sorted.get(idx.checked_sub(1)?)?;
            (va < r.end).then_some(r)
        } else {
            self.list.iter().find(|r| va >= r.va && va < r.end)
        }
    }

    /// True when any executable region overlaps `[va, va + len)`.
    fn exec_overlaps(&self, va: u64, len: u64) -> bool {
        let Some(end) = va.checked_add(len) else {
            return false;
        };
        if len == 0 {
            return false;
        }
        if self.disjoint {
            // Disjoint and sorted by start, so ends are sorted too.
            let first = self.sorted.partition_point(|r| r.end <= va);
            self.sorted[first..]
                .iter()
                .take_while(|r| r.va < end)
                .any(|r| r.perm.contains(Perm::X))
        } else {
            self.list
                .iter()
                .any(|r| r.perm.contains(Perm::X) && va < r.end && end > r.va)
        }
    }

    fn len(&self) -> usize {
        self.list.len()
    }

    fn iter(&self) -> std::slice::Iter<'_, Region> {
        self.list.iter()
    }
}

/// Per-page fastmem permissions for the generated code (see
/// `rustarmic::JitConfig::fastmem_page_bits`).  A page gets a direct bit
/// only when it lies entirely inside one region, that region is the
/// first-match region for the whole page, and its host memory is the
/// fastmem arena at `mem_base + va`.  Writes additionally require that no
/// executable region and no watched range touches the page, so
/// self-modifying code and write watches keep going through
/// `mem_write_hook`.  Every other access takes the hooks.
///
/// The table (128 MiB for a 39-bit arena) is a reservation that only costs
/// the pages actually written or probed (`PageTableMem`), on Windows too, so
/// one per core is cheap.  It is per core rather than shared because each
/// core's region list is updated independently (`map_host` / `unmap_host`
/// reach cores at different times), and a page may only be direct for a
/// core whose own hooks would have allowed the access.
struct FastmemTable {
    bits: u32,
    table: PageTableMem,
    mem_base: *mut u8,
}

impl FastmemTable {
    fn new() -> Option<Self> {
        if !rustarmic_fastmem_enabled() {
            log::info!("rustarmic: fastmem disabled by NEXIUM_RUSTARMIC_FASTMEM=0");
            return None;
        }
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if crate::lockstep::enabled() && !crate::lockstep::config().fastmem_reads {
            log::info!("rustarmic: fastmem disabled for lockstep (NEXIUM_LOCKSTEP_FASTMEM=0)");
            return None;
        }
        // Direct (NCE) mode places guest VAs at host addresses; the
        // `mem_base + va` arithmetic below assumes arena offset == va.
        if nexium_memory::fastmem::direct_va_base().is_some() {
            return None;
        }
        let base = nexium_memory::fastmem::base()?;
        let bits = nexium_memory::fastmem::arena_bits().min(FASTMEM_TABLE_MAX_BITS);
        if bits < 13 {
            return None;
        }
        // Reserved, zero-filled, committed only where touched.
        let Some(table) = PageTableMem::new(1usize << (bits - 12)) else {
            log::warn!("rustarmic: could not reserve the fastmem page table; fastmem off");
            return None;
        };
        log::info!(
            "rustarmic: page-table fastmem enabled, arena base={:p} bits={} (NEXIUM_RUSTARMIC_FASTMEM=0 disables)",
            base,
            bits
        );
        Some(Self {
            bits,
            table,
            mem_base: base,
        })
    }

    fn limit(&self) -> u64 {
        1u64 << self.bits
    }

    fn arena_backed(&self, r: &Region) -> bool {
        r.end <= self.limit()
            && (self.mem_base as u64).checked_add(r.va) == Some(r.host_ptr as u64)
    }

    fn direct_bits(&self, r: &Region) -> u8 {
        if !self.arena_backed(r) {
            return 0;
        }
        let mut bits = 0;
        if r.perm.contains(Perm::R) {
            bits |= PAGE_DIRECT_READ;
        }
        if r.perm.contains(Perm::W) && !r.perm.contains(Perm::X) {
            bits |= PAGE_DIRECT_WRITE;
        }
        bits
    }

    /// Recomputes the entries of every page overlapping `[lo, hi)` from the
    /// full region list.
    fn refresh(&mut self, regions: &Regions, lo: u64, hi: u64) {
        let limit = self.limit();
        let lo_page = (lo.min(limit) >> 12) as usize;
        let hi_page = (hi.min(limit).saturating_add(0xFFF) >> 12) as usize;
        if lo_page >= hi_page {
            return;
        }
        self.table.prepare_write(lo_page..hi_page);
        self.table[lo_page..hi_page].fill(0);
        // Pages a region only partly covers, or covers but `[a, b)` clips.
        let pages_touching = |va: u64, end: u64| -> (usize, usize) {
            let a = ((va.min(limit)) >> 12) as usize;
            let b = ((end.min(limit)).saturating_add(0xFFF) >> 12) as usize;
            (a.max(lo_page), b.min(hi_page))
        };
        // Lowest priority first: the earliest region touching a page (the
        // one the hooks would find first) is applied last and wins.
        for r in regions.iter().rev() {
            if r.end <= r.va {
                continue;
            }
            let (a, b) = pages_touching(r.va, r.end);
            if a >= b {
                continue;
            }
            self.table[a..b].fill(0);
            let full_a = ((r.va.saturating_add(0xFFF).min(limit)) >> 12) as usize;
            let full_b = ((r.end.min(limit)) >> 12) as usize;
            let (fa, fb) = (full_a.max(a), full_b.min(b));
            if fa < fb {
                let bits = self.direct_bits(r);
                self.table[fa..fb].fill(bits);
            }
        }
        let mut clear_write = |va: u64, end: u64| {
            let (a, b) = pages_touching(va, end);
            if a < b {
                for e in &mut self.table[a..b] {
                    *e &= !PAGE_DIRECT_WRITE;
                }
            }
        };
        for r in regions.iter().filter(|r| r.perm.contains(Perm::X)) {
            clear_write(r.va, r.end);
        }
        if let Some((wlo, whi)) = rustarmic_watch_range() {
            clear_write(wlo, whi);
        }
    }
}

fn env_disabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

fn rustarmic_fastmem_enabled() -> bool {
    !env_disabled("NEXIUM_RUSTARMIC_FASTMEM")
}

fn code_cache_bytes() -> usize {
    let mb = std::env::var("NEXIUM_JIT_CACHE_MB")
        .ok()
        .and_then(|mb| mb.trim().parse::<usize>().ok())
        .unwrap_or(CODE_CACHE_DEFAULT_MB);
    mb.clamp(16, CODE_CACHE_MAX_MB) << 20
}

#[derive(Clone, Debug)]
pub struct RustarmicThreadContext {
    v: [[u64; 2]; 32],
    tpidr_el0: u64,
    fpcr: u32,
    fpsr: u32,
    nzcv: u8,
}

#[repr(C)]
struct State {
    ctx: CpuContext,
    regions: parking_lot::RwLock<Regions>,
    control: Arc<Control>,
    last_event: Mutex<Option<CpuEvent>>,
    /// Set (under the `last_event` lock) whenever an event is stored, so the
    /// common entry path does not take the lock.
    has_event: AtomicBool,
    last_fault: Mutex<Option<FaultSnapshot>>,
    continue_on_null: AtomicBool,
    null_skip_count: AtomicU32,
    pending_invalidations: Mutex<Vec<(u64, u64)>>,
    /// Set (under the `pending_invalidations` lock) whenever an entry is
    /// queued.
    has_pending_invalidations: AtomicBool,
    /// After a data abort `ctx.pc` reports the faulting instruction; the
    /// next run resumes past the access (`JitConfig::resume_after_fault`)
    /// unless the PC was changed in between: (fault pc, resume pc).
    fault_resume: Option<(u64, u64)>,
    fastmem: Option<FastmemTable>,
    /// Lockstep write/timer observation (`crate::lockstep`); `None` unless
    /// `NEXIUM_LOCKSTEP` is on.
    #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
    lockstep_hooks: Option<Box<LockstepHooks>>,
}

impl State {
    fn set_event(&self, event: CpuEvent) {
        let mut slot = self.last_event.lock().unwrap();
        *slot = Some(event);
        self.has_event.store(true, Ordering::Release);
    }

    fn take_event(&self) -> Option<CpuEvent> {
        if !self.has_event.load(Ordering::Acquire) {
            return None;
        }
        let mut slot = self.last_event.lock().unwrap();
        self.has_event.store(false, Ordering::Release);
        slot.take()
    }

    fn queue_invalidation(&self, va: u64, len: u64) {
        let mut queue = self.pending_invalidations.lock().unwrap();
        queue.push((va, len));
        self.has_pending_invalidations
            .store(true, Ordering::Release);
    }

    fn take_invalidations(&self) -> Vec<(u64, u64)> {
        if !self.has_pending_invalidations.load(Ordering::Acquire) {
            return Vec::new();
        }
        let mut queue = self.pending_invalidations.lock().unwrap();
        self.has_pending_invalidations
            .store(false, Ordering::Release);
        std::mem::take(&mut *queue)
    }

    fn refresh_fastmem(&mut self, lo: u64, hi: u64) {
        let Self {
            regions, fastmem, ..
        } = self;
        if let Some(table) = fastmem.as_mut() {
            table.refresh(regions.get_mut(), lo, hi);
        }
        // Lockstep must see every store, so none may bypass the hooks.
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if let (Some(table), Some(_)) = (self.fastmem.as_mut(), self.lockstep_hooks.as_ref()) {
            let limit = table.limit();
            let lo_page = (lo.min(limit) >> 12) as usize;
            let hi_page = (hi.min(limit).saturating_add(0xFFF) >> 12) as usize;
            if lo_page < hi_page {
                table.table.prepare_write(lo_page..hi_page);
                for e in &mut table.table[lo_page..hi_page] {
                    *e &= !PAGE_DIRECT_WRITE;
                }
            }
        }
    }
}

#[repr(C)]
struct Control {
    halt: AtomicBool,
    snapshot_seq: AtomicU64,
    peek_pc: AtomicU64,
    peek_lr: AtomicU64,
    peek_sp: AtomicU64,
    /// Address of the owning core's `ctx.should_halt` (0 once the core is
    /// dropped).  Generated code checks that byte at every block prologue,
    /// so a halt requested from another thread lands mid-slice instead of
    /// at the next `run`.  The lock keeps a halt from writing into a
    /// dropped core.
    live_should_halt: Mutex<usize>,
}

impl Control {
    fn request_halt(&self) {
        self.halt.store(true, Ordering::Release);
        let live = self.live_should_halt.lock().unwrap_or_else(|e| e.into_inner());
        if *live != 0 {
            // SAFETY: non-zero only while the owning `State` is alive (the
            // core clears it under this lock before dropping it), and the
            // byte is only ever read and written as a plain byte.
            unsafe { (*(*live as *const std::sync::atomic::AtomicU8)).store(1, Ordering::Release) };
        }
    }
}

pub struct RustarmicCpu {
    state: Box<State>,
    jit: Jit,
    /// Shadow Dynarmic JIT checked against every slice (`NEXIUM_LOCKSTEP`).
    #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
    lockstep: Option<Box<crate::lockstep::Lockstep>>,
}

impl Drop for RustarmicCpu {
    fn drop(&mut self) {
        *self
            .state
            .control
            .live_should_halt
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = 0;
    }
}

unsafe impl Send for RustarmicCpu {}
unsafe impl Sync for RustarmicCpu {}

impl RustarmicCpu {
    pub fn new() -> Result<Self, String> {
        Self::new_with_jit_config(JitConfig {
            code_cache_bytes: code_cache_bytes(),
            ..JitConfig::default()
        })
    }

    pub fn new_with_engine_config(config: &rustarmic::EngineConfig) -> Result<Self, String> {
        let mut jit = JitConfig::default();
        jit.code_cache_bytes = config.code_cache_bytes;
        jit.use_fastmem = matches!(config.memory_mode, rustarmic::MemoryMode::Fastmem);
        jit.translate.max_insts = config.max_block_insts;
        jit.host_features = Some(config.host_features);
        Self::new_with_jit_config(jit)
    }

    fn new_with_jit_config(mut jit_config: JitConfig) -> Result<Self, String> {
        // Escape hatches for the dispatcher optimisations (default on).
        if env_disabled("NEXIUM_RUSTARMIC_MULTIBLOCK") {
            jit_config.translate.multiblock = false;
        }
        if env_disabled("NEXIUM_RUSTARMIC_LINK") {
            jit_config.link_blocks = false;
        }
        if !env_disabled("NEXIUM_RUSTARMIC_RWX") {
            jit_config.rwx_code_cache = true;
        }
        // A data abort resumes past the faulting access, as on Dynarmic
        // (whose unmapped callback completes the access and stops after
        // it).  Retrying it would fault forever: the aux-core loop re-enters
        // the faulting thread after logging the exception.
        jit_config.resume_after_fault = true;
        let fastmem = FastmemTable::new();
        if let Some(table) = &fastmem {
            jit_config.fastmem_page_bits = Some(table.bits as u8);
        }
        let mut state = Box::new(State {
            ctx: CpuContext::default(),
            regions: parking_lot::RwLock::new(Regions::default()),
            control: Arc::new(Control {
                halt: AtomicBool::new(false),
                snapshot_seq: AtomicU64::new(0),
                peek_pc: AtomicU64::new(0),
                peek_lr: AtomicU64::new(0),
                peek_sp: AtomicU64::new(0),
                live_should_halt: Mutex::new(0),
            }),
            last_event: Mutex::new(None),
            has_event: AtomicBool::new(false),
            last_fault: Mutex::new(None),
            continue_on_null: AtomicBool::new(false),
            null_skip_count: AtomicU32::new(0),
            pending_invalidations: Mutex::new(Vec::new()),
            has_pending_invalidations: AtomicBool::new(false),
            fault_resume: None,
            fastmem,
            #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
            lockstep_hooks: None,
        });
        state.ctx.stop_token = Arc::as_ptr(&state.control).cast();
        // `state` is boxed and never moved out, so the byte stays put.
        *state.control.live_should_halt.lock().unwrap() =
            std::ptr::addr_of_mut!(state.ctx.should_halt) as usize;
        if let Some(table) = &state.fastmem {
            // The table is never reallocated, so this pointer stays valid
            // for the lifetime of `state`.
            state.ctx.page_table = table.table.as_ptr();
            state.ctx.mem_base = table.mem_base;
        }
        state.ctx.mem_read = mem_read_hook;
        state.ctx.mem_write = mem_write_hook;
        state.ctx.mem_atomic = mem_atomic_hook;
        let jit = Jit::new(jit_config).map_err(|e| format!("rustarmic Jit init: {:?}", e))?;
        #[allow(unused_mut)]
        let mut cpu = Self {
            state,
            jit,
            #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
            lockstep: None,
        };
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if crate::lockstep::enabled() {
            cpu.attach_lockstep(crate::lockstep::config().clone());
        }
        Ok(cpu)
    }

    /// This core's view of `va`: the region its callbacks use, the nearest
    /// neighbours and the page's fastmem bits.
    pub fn mapping_view(&self, va: u64) -> MappingView {
        let regions = self.state.regions.read();
        let mut below: Option<(u64, u64)> = None;
        let mut above: Option<(u64, u64)> = None;
        for r in regions.iter().filter(|r| r.end > r.va) {
            if r.end <= va && below.is_none_or(|(_, end)| r.end > end) {
                below = Some((r.va, r.end));
            }
            if r.va > va && above.is_none_or(|(base, _)| r.va < base) {
                above = Some((r.va, r.end));
            }
        }
        let page_direct = self.state.fastmem.as_ref().and_then(|t| {
            (va < t.limit()).then(|| {
                let bits = t.table[(va >> 12) as usize];
                (bits & PAGE_DIRECT_READ != 0, bits & PAGE_DIRECT_WRITE != 0)
            })
        });
        MappingView {
            region: region_at(&regions, va).map(|r| (r.va, r.end, r.perm)),
            below,
            above,
            page_direct,
            region_count: regions.len(),
        }
    }

    /// Whether this core has the fastmem page table (inline accesses).
    pub fn has_fastmem_table(&self) -> bool {
        self.state.fastmem.is_some()
    }

    pub fn take_fault(&self) -> Option<FaultSnapshot> {
        self.state.last_fault.lock().unwrap().take()
    }

    pub fn guest_cntpct(&self) -> u64 {
        CpuContext::default_cntpct()
    }

    pub fn set_continue_on_null(&self, enable: bool) {
        self.state.continue_on_null.store(enable, Ordering::Relaxed);
    }

    pub fn null_skip_count(&self) -> u32 {
        self.state.null_skip_count.load(Ordering::Relaxed)
    }

    pub fn jit_stats(&self) -> String {
        format!(
            "jit_blocks={} jit_bytes={}",
            self.jit.cache.len(),
            self.jit.cache.used_bytes()
        )
    }

    pub fn halt_handle(&self) -> HaltHandle {
        let control = Arc::clone(&self.state.control);
        let peek_control = Arc::clone(&self.state.control);
        HaltHandle {
            inner: Arc::new(move || control.request_halt()),
            peek: Arc::new(move || loop {
                let before = peek_control.snapshot_seq.load(Ordering::Acquire);
                if before & 1 != 0 {
                    std::hint::spin_loop();
                    continue;
                }
                let p = peek_control.peek_pc.load(Ordering::Relaxed);
                let l = peek_control.peek_lr.load(Ordering::Relaxed);
                let s = peek_control.peek_sp.load(Ordering::Relaxed);
                let after = peek_control.snapshot_seq.load(Ordering::Acquire);
                if before == after {
                    break (p, l, s);
                }
            }),
            peek_dump: Arc::new(|| String::from("(peek_dump unsupported on rustarmic)")),
        }
    }

    pub unsafe fn map_host(
        &mut self,
        va: u64,
        len: u64,
        perm: Perm,
        ptr: *mut u8,
    ) -> Result<(), String> {
        let end = va
            .checked_add(len)
            .ok_or_else(|| format!("map_host overflow va={:#x} len={:#x}", va, len))?;
        log::debug!(
            "rustarmic map_host va={:#x}..{:#x} perm={} ptr={:p}",
            va,
            end,
            perm,
            ptr
        );
        if perm.contains(Perm::X) {
            self.jit.invalidate_range(va, len);
        }
        self.state.regions.get_mut().push(Region {
            va,
            end,
            host_ptr: ptr,
            perm,
        });
        self.state.refresh_fastmem(va, end);
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if let Some(lockstep) = self.lockstep.as_mut() {
            unsafe { lockstep.map_host(va, len, perm, ptr) };
        }
        Ok(())
    }

    pub unsafe fn unmap_host(&mut self, va: u64, len: u64) -> Result<(), String> {
        let end = va
            .checked_add(len)
            .ok_or_else(|| format!("unmap_host overflow va={:#x} len={:#x}", va, len))?;
        if let Some((lo, hi)) = self.state.regions.get_mut().remove_range(va, end) {
            self.state.refresh_fastmem(lo, hi);
        }
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if let Some(lockstep) = self.lockstep.as_mut() {
            unsafe { lockstep.unmap_host(va, len) };
        }
        Ok(())
    }

    pub fn write_bytes(&self, va: u64, bytes: &[u8]) -> Result<(), String> {
        let regions = self.state.regions.read();
        copy_to_regions(&regions, va, bytes, true)?;
        if regions.exec_overlaps(va, bytes.len() as u64) {
            self.state.queue_invalidation(va, bytes.len() as u64);
        }
        Ok(())
    }

    pub fn invalidate_range(&mut self, va: u64, len: u64) {
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if let Some(lockstep) = self.lockstep.as_mut() {
            for (qva, qlen) in self.state.take_invalidations() {
                self.jit.invalidate_range(qva, qlen);
                lockstep.invalidate_range(qva, qlen);
            }
            lockstep.invalidate_range(va, len);
        }
        for (qva, qlen) in self.state.take_invalidations() {
            self.jit.invalidate_range(qva, qlen);
        }
        self.jit.invalidate_range(va, len);
    }

    pub fn read_bytes(&self, va: u64, buf: &mut [u8]) -> Result<(), String> {
        let regions = self.state.regions.read();
        copy_from_regions(&regions, va, buf, false)
    }

    pub fn set_register(&mut self, reg: u32, val: u64) {
        if reg < 31 {
            self.state.ctx.x[reg as usize] = val;
        } else if reg == 31 {
            self.state.ctx.sp = val;
        }
    }

    pub fn get_register(&self, reg: u32) -> u64 {
        if reg < 31 {
            self.state.ctx.x[reg as usize]
        } else if reg == 31 {
            self.state.ctx.sp
        } else {
            0
        }
    }

    pub fn set_pc(&mut self, pc: u64) {
        self.state.ctx.pc = pc;
        self.state.fault_resume = None;
    }

    pub fn set_core_id(&mut self, core_id: u64) {
        self.state.ctx.core_id = core_id;
    }
    pub fn get_pc(&self) -> u64 {
        self.state.ctx.pc
    }
    pub fn set_sp(&mut self, sp: u64) {
        self.state.ctx.sp = sp;
    }
    pub fn get_sp(&self) -> u64 {
        self.state.ctx.sp
    }

    pub fn set_tpidrro_el0(&mut self, val: u64) {
        self.state.ctx.tpidrro_el0 = val;
    }
    pub fn get_tpidrro_el0(&self) -> u64 {
        self.state.ctx.tpidrro_el0
    }

    pub fn save_thread_context(&self) -> RustarmicThreadContext {
        RustarmicThreadContext {
            v: self.state.ctx.v,
            tpidr_el0: self.state.ctx.tpidr_el0,
            fpcr: self.state.ctx.fpcr,
            fpsr: self.state.ctx.fpsr,
            nzcv: self.state.ctx.nzcv,
        }
    }

    pub fn restore_thread_context(&mut self, context: &RustarmicThreadContext) {
        self.state.ctx.v = context.v;
        self.state.ctx.tpidr_el0 = context.tpidr_el0;
        self.state.ctx.fpcr = context.fpcr;
        self.state.ctx.fpsr = context.fpsr;
        self.state.ctx.nzcv = context.nzcv;
        self.state.ctx.exclusive_addr = 0;
        self.state.ctx.exclusive_size = 0;
    }

    pub fn reset_thread_context(&mut self) {
        self.state.ctx.v = [[0; 2]; 32];
        self.state.ctx.tpidr_el0 = 0;
        self.state.ctx.fpcr = 0;
        self.state.ctx.fpsr = 0;
        self.state.ctx.nzcv = 0;
        self.state.ctx.exclusive_addr = 0;
        self.state.ctx.exclusive_size = 0;
    }

    pub fn run_with_count(&mut self, max_insn: u64) -> (CpuEvent, u64) {
        #[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
        if let Some(mut lockstep) = self.lockstep.take() {
            let result = lockstep.run(self, max_insn);
            self.lockstep = Some(lockstep);
            return result;
        }
        let queued = self.state.take_invalidations();
        if !queued.is_empty() {
            log::debug!(
                "rustarmic: applying {} queued SMC invalidations",
                queued.len()
            );
            for (va, len) in queued {
                self.jit.invalidate_range(va, len);
            }
        }

        if let Some(ev) = self.state.take_event() {
            return (ev, 0);
        }
        if let Some((fault_pc, resume_pc)) = self.state.fault_resume.take() {
            if self.state.ctx.pc == fault_pc {
                self.state.ctx.pc = resume_pc;
            }
        }
        if self.state.control.halt.swap(false, Ordering::AcqRel) {
            self.state.ctx.should_halt = 1;
        }

        self.state
            .control
            .snapshot_seq
            .fetch_add(1, Ordering::Release);
        let exit = {
            let state = &mut *self.state;
            // Regions only change through `&mut self` (`map_host` /
            // `unmap_host`), so borrowing them for the slice cannot miss an
            // update.  The hooks take further recursive read locks.
            let mut mem = RegionMemory {
                regions: state.regions.read_recursive(),
            };
            self.jit.run_bounded(&mut state.ctx, &mut mem, max_insn)
        };

        self.state
            .control
            .peek_pc
            .store(self.state.ctx.pc, Ordering::Relaxed);
        self.state
            .control
            .peek_lr
            .store(self.state.ctx.x[30], Ordering::Relaxed);
        self.state
            .control
            .peek_sp
            .store(self.state.ctx.sp, Ordering::Relaxed);
        self.state
            .control
            .snapshot_seq
            .fetch_add(1, Ordering::Release);

        match exit {
            Ok(outcome) => {
                let event = match outcome.stop {
                    StopReason::Unsupported(info) => {
                        log::warn!(
                            "rustarmic unsupported stop: pc={:#x} opcode={:#010x} class={} retired={}",
                            info.pc,
                            info.opcode,
                            info.decoded_class,
                            outcome.retired,
                        );
                        CpuEvent::Stalled
                    }
                    StopReason::MemoryFault(fault) => {
                        let mut pending = self.state.last_event.lock().unwrap();
                        if matches!(*pending, Some(CpuEvent::Exception(0x0E))) {
                            *pending = None;
                        }
                        drop(pending);
                        self.state.ctx.should_halt = 0;
                        self.state.fault_resume = Some((fault.pc, self.state.ctx.pc));
                        self.state.ctx.pc = fault.pc;
                        static MEMORY_FAULTS: AtomicU64 = AtomicU64::new(0);
                        if let Some(n) = log_occurrence(&MEMORY_FAULTS) {
                            log::warn!(
                                "rustarmic memory fault #{}: pc={:#x} address={:#x} size={} access={:?} cause={:?} retired={}",
                                n,
                                fault.pc,
                                fault.address,
                                fault.size,
                                fault.access,
                                fault.cause,
                                outcome.retired,
                            );
                        }
                        CpuEvent::Exception(0x0E)
                    }
                    StopReason::BudgetExhausted => CpuEvent::Running,
                    // A halt is a cooperative exit (self-modifying-code invalidation,
                    // the spin watchdog, a queued fault), not a stop request: report
                    // the queued event if there is one, otherwise keep running, as
                    // the Dynarmic backend does. Core 0 ends emulation on Interrupted.
                    StopReason::Halted => {
                        self.state.control.halt.store(false, Ordering::Release);
                        self.state.take_event().unwrap_or(CpuEvent::Running)
                    }
                    StopReason::Svc(imm) => CpuEvent::Svc(imm as u16),
                    StopReason::Brk(imm) => CpuEvent::Exception(0x100 | imm),
                    StopReason::Hvc(imm) => CpuEvent::Exception(0x200 | imm),
                    StopReason::Yield | StopReason::Wait => CpuEvent::Interrupted,
                };
                return (event, outcome.retired);
            }
            Err(e) => {
                match &e {
                    rustarmic::Error::Unsupported { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() {
                            u32::from_le_bytes(b)
                        } else {
                            *opcode
                        };
                        log::warn!(
                            "rustarmic Jit::run unsupported: pc={:#x} opcode={:#010x}",
                            pc,
                            word
                        );
                    }
                    rustarmic::Error::Decode { pc, opcode } => {
                        let mut b = [0u8; 4];
                        let word = if self.read_bytes(*pc, &mut b).is_ok() {
                            u32::from_le_bytes(b)
                        } else {
                            *opcode
                        };
                        log::warn!(
                            "rustarmic Jit::run decode-fail: pc={:#x} opcode={:#010x}",
                            pc,
                            word
                        );
                    }
                    other => log::warn!("rustarmic Jit::run error: {:?}", other),
                }
                (CpuEvent::Stalled, 0)
            }
        }
    }

    pub fn run(&mut self, max_insn: u64) -> CpuEvent {
        self.run_with_count(max_insn).0
    }

    pub fn step(&mut self) -> CpuEvent {
        self.run(1)
    }

    pub fn inject_svc(&mut self, imm: u16) {
        self.state.set_event(CpuEvent::Svc(imm));
    }
}

struct RegionMemory<'a> {
    regions: parking_lot::RwLockReadGuard<'a, Regions>,
}

impl Memory for RegionMemory<'_> {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let mut buf = [0u8; 4];
        if copy_from_regions(&self.regions, addr, &mut buf, true).is_ok() {
            return Some(u32::from_le_bytes(buf));
        }
        static FETCH_MISSES: AtomicU64 = AtomicU64::new(0);
        let n = log_occurrence(&FETCH_MISSES)?;
        log::warn!(
            "rustarmic fetch_inst miss #{} at {:#x} ({} regions)",
            n,
            addr,
            self.regions.len()
        );
        if n == 1 {
            for r in self.regions.iter() {
                log::warn!(
                    "  region {:#x}..{:#x} perm={} x={}",
                    r.va,
                    r.end,
                    r.perm,
                    r.perm.contains(Perm::X)
                );
            }
        }
        None
    }
}

fn log_occurrence(counter: &AtomicU64) -> Option<u64> {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    (n <= 16 || n.is_power_of_two()).then_some(n)
}

fn region_at(regions: &Regions, va: u64) -> Option<&Region> {
    regions.find(va)
}

fn copy_from_regions(
    regions: &Regions,
    va: u64,
    out: &mut [u8],
    execute: bool,
) -> Result<(), String> {
    va.checked_add(out.len() as u64)
        .ok_or_else(|| format!("read: address overflow va={va:#x} len={}", out.len()))?;
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < out.len() {
        let region =
            region_at(regions, cursor).ok_or_else(|| format!("read: unmapped va={cursor:#x}"))?;
        let perm = if execute { Perm::X } else { Perm::R };
        if !region.perm.contains(perm) {
            return Err(format!("read: permission fault va={cursor:#x}"));
        }
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(out.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                region.host_ptr.add((cursor - region.va) as usize),
                out.as_mut_ptr().add(copied),
                count,
            );
        }
        copied += count;
        cursor = cursor.saturating_add(count as u64);
    }
    Ok(())
}

fn copy_to_regions(regions: &Regions, va: u64, input: &[u8], write: bool) -> Result<(), String> {
    va.checked_add(input.len() as u64)
        .ok_or_else(|| format!("write: address overflow va={va:#x} len={}", input.len()))?;
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < input.len() {
        let region =
            region_at(regions, cursor).ok_or_else(|| format!("write: unmapped va={cursor:#x}"))?;
        if write && !region.perm.contains(Perm::W) {
            return Err(format!("write: permission fault va={cursor:#x}"));
        }
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(input.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr().add(copied),
                region.host_ptr.add((cursor - region.va) as usize),
                count,
            );
        }
        copied += count;
        cursor = cursor.saturating_add(count as u64);
    }
    Ok(())
}

/// First byte of `[va, va + len)` an access with `perm` cannot touch, and
/// the fault cause (0 = unmapped, 1 = permission).  Walks the regions the
/// same way as `copy_from_regions` / `copy_to_regions`, so a multi-byte
/// access that runs from a mapped region into an unmapped page reports the
/// first unmapped byte, not the access base (an ARM data abort reports the
/// address that faulted; the bytes before it may already have been
/// written).
fn first_fault(regions: &Regions, va: u64, len: u64, perm: Perm) -> Option<(u64, u8)> {
    let end = va.checked_add(len)?;
    let mut cursor = va;
    while cursor < end {
        let Some(region) = region_at(regions, cursor) else {
            return Some((cursor, 0));
        };
        if !region.perm.contains(perm) {
            return Some((cursor, 1));
        }
        cursor = region.end.min(end);
    }
    None
}

/// Fault address and cause for a failed access at `addr`.
fn fault_site(regions: &Regions, addr: u64, size: u8, perm: Perm) -> (u64, u8) {
    first_fault(regions, addr, u64::from(size), perm).unwrap_or_else(|| {
        // Overflowing or oversized access: report the base.
        let cause = u8::from(region_at(regions, addr).is_some());
        (addr, cause)
    })
}

/// Host pointer for an aligned 1/2/4/8-byte access inside one region with
/// `perm`, so the hooks can use one host access of that width (single-copy
/// atomic, as on hardware) instead of a byte copy.
#[inline(always)]
fn scalar_host_ptr(regions: &Regions, addr: u64, size: u8, perm: Perm) -> Option<*mut u8> {
    if !matches!(size, 1 | 2 | 4 | 8) || addr & (u64::from(size) - 1) != 0 {
        return None;
    }
    let region = region_at(regions, addr)?;
    if !region.perm.contains(perm) || addr.checked_add(u64::from(size))? > region.end {
        return None;
    }
    let ptr = unsafe { region.host_ptr.add((addr - region.va) as usize) };
    ((ptr as usize) & (usize::from(size) - 1) == 0).then_some(ptr)
}

/// The parts a load/store callback is performed as: a 16-byte access as
/// its two 8-byte halves (each single-copy atomic when aligned, and the
/// unit a fault is reported for, as Unicorn does), anything else whole.
fn access_parts(addr: u64, size: u8) -> ([(u64, u8); 2], usize) {
    if size == 16 {
        ([(addr, 8), (addr.wrapping_add(8), 8)], 2)
    } else {
        ([(addr, size), (0, 0)], 1)
    }
}

/// One load of at most 8 bytes; `None` once the fault is reported.
fn load_part(state: &State, regions: &Regions, ctx_ptr: *mut CpuContext, addr: u64, size: u8) -> Option<u64> {
    if let Some(ptr) = scalar_host_ptr(regions, addr, size, Perm::R) {
        return Some(unsafe {
            match size {
                1 => (&*ptr.cast::<AtomicU8>()).load(Ordering::Relaxed) as u64,
                2 => (&*ptr.cast::<AtomicU16>()).load(Ordering::Relaxed) as u64,
                4 => (&*ptr.cast::<AtomicU32>()).load(Ordering::Relaxed) as u64,
                8 => (&*ptr.cast::<AtomicU64>()).load(Ordering::Relaxed),
                _ => unreachable!(),
            }
        });
    }
    let mut buf = [0u8; 8];
    if size <= 8 && copy_from_regions(regions, addr, &mut buf[..size as usize], false).is_ok() {
        return Some(u64::from_le_bytes(buf));
    }
    let (fault_addr, cause) = fault_site(regions, addr, size, Perm::R);
    handle_unmapped(state, ctx_ptr, fault_addr, size, false, 0, cause);
    None
}

/// One store of at most 8 bytes; `false` once the fault is reported.
fn store_part(
    state: &State,
    regions: &Regions,
    ctx_ptr: *mut CpuContext,
    addr: u64,
    size: u8,
    value: u64,
) -> bool {
    let written = if let Some(ptr) = scalar_host_ptr(regions, addr, size, Perm::W) {
        unsafe {
            match size {
                1 => (&*ptr.cast::<AtomicU8>()).store(value as u8, Ordering::Relaxed),
                2 => (&*ptr.cast::<AtomicU16>()).store(value as u16, Ordering::Relaxed),
                4 => (&*ptr.cast::<AtomicU32>()).store(value as u32, Ordering::Relaxed),
                8 => (&*ptr.cast::<AtomicU64>()).store(value, Ordering::Relaxed),
                _ => unreachable!(),
            }
        }
        true
    } else {
        size <= 8 && copy_to_regions(regions, addr, &value.to_le_bytes()[..size as usize], true).is_ok()
    };
    if written {
        if regions.exec_overlaps(addr, u64::from(size)) {
            state.queue_invalidation(addr, u64::from(size));
            unsafe {
                (*ctx_ptr).should_halt = 1;
            }
        }
        return true;
    }
    let (fault_addr, cause) = fault_site(regions, addr, size, Perm::W);
    handle_unmapped(state, ctx_ptr, fault_addr, size, true, value, cause);
    false
}

unsafe extern "C" fn mem_read_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let regions = state.regions.read_recursive();
    let (parts, count) = access_parts(addr, size);
    let mut value = [0u64; 2];
    for (slot, &(part_addr, part_size)) in value.iter_mut().zip(&parts[..count]) {
        match load_part(state, &regions, ctx_ptr, part_addr, part_size) {
            Some(loaded) => *slot = loaded,
            None => break,
        }
    }
    unsafe {
        (*ctx_ptr).io_value = value;
    }
}

unsafe extern "C" fn mem_write_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let io = unsafe { (*ctx_ptr).io_value };
    if rustarmic_watch_overlaps(addr, size as u64) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HITS: AtomicU64 = AtomicU64::new(0);
        let n = HITS.fetch_add(1, Ordering::SeqCst);
        if n < 128 {
            let ctx = unsafe { &*ctx_ptr };
            log::warn!(
                "[watch-write] #{} addr={:#x} size={} val={:#x}/{:#x} pc={:#x} lr={:#x}",
                n,
                addr,
                size,
                io[0],
                io[1],
                ctx.pc,
                ctx.x[30]
            );
        }
    }
    let regions = state.regions.read_recursive();
    let (parts, count) = access_parts(addr, size);
    for (&value, &(part_addr, part_size)) in io.iter().zip(&parts[..count]) {
        if !store_part(state, &regions, ctx_ptr, part_addr, part_size, value) {
            break;
        }
    }
}

/// Guest atomic read-modify-write (LSE atomics, CAS, store-exclusive) for
/// accesses the generated code did not perform inline.  Must be atomic with
/// respect to the other cores' host threads: an access inside one readable
/// and writable region is done with host atomics on the backing memory
/// (`rustarmic::jit::atomic::context_host_atomic`; misaligned or 128-bit
/// without `cmpxchg16b` falls back to a process-wide lock).  Accesses that
/// span regions are serialised under that lock and use the copy helpers.
/// Stores into executable regions still invalidate translated code.
unsafe extern "C" fn mem_atomic_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8, op: u8) {
    use rustarmic::jit::atomic::{AtomicOp, FALLBACK_LOCK, apply, context_host_atomic};
    let state = unsafe { &*(ctx_ptr as *mut State) };
    // Like the load/store callbacks, skip once an earlier access of this
    // run faulted, so the first fault is the one reported and this access
    // runs when the guest resumes (generated code sets `ctx.pc` to it).
    unsafe {
        let ctx = &mut *ctx_ptr;
        if ctx.mem_fault != 0 {
            if ctx.mem_fault_skip_pc == 0 {
                ctx.mem_fault_skip_pc = ctx.pc;
            }
            ctx.io_value = [0, 0];
            return;
        }
    }
    let len = u64::from(size);
    if rustarmic_watch_overlaps(addr, len) {
        let ctx = unsafe { &*ctx_ptr };
        log::warn!(
            "[watch-atomic] addr={:#x} size={} op={} operand={:#x}/{:#x} pc={:#x} lr={:#x}",
            addr,
            size,
            op,
            ctx.io_value[0],
            ctx.io_value[1],
            ctx.pc,
            ctx.x[30]
        );
    }
    let regions = state.regions.read_recursive();
    let rw = Perm::R | Perm::W;
    let single = region_at(&regions, addr).filter(|r| {
        size <= 16 && addr.checked_add(len).is_some_and(|end| end <= r.end) && r.perm.contains(rw)
    });
    if let Some(r) = single {
        unsafe {
            let ptr = r.host_ptr.add((addr - r.va) as usize);
            context_host_atomic(ctx_ptr, ptr, size, op);
        }
    } else {
        if let Some((fault_addr, cause)) = first_fault(&regions, addr, len, Perm::R)
            .or_else(|| first_fault(&regions, addr, len, Perm::W))
            .or_else(|| (size > 16).then_some((addr, 0)))
        {
            let value = unsafe { (*ctx_ptr).io_value[0] };
            handle_unmapped(state, ctx_ptr, fault_addr, size, true, value, cause);
            unsafe { (*ctx_ptr).io_value = [0, 0] };
            return;
        }
        let _guard = FALLBACK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut buf = [0u8; 16];
        let n = size as usize;
        if copy_from_regions(&regions, addr, &mut buf[..n], false).is_err() {
            return;
        }
        let to128 = |v: [u64; 2]| u128::from(v[0]) | (u128::from(v[1]) << 64);
        let old = u128::from_le_bytes(buf);
        let (operand, expected) = unsafe { (to128((*ctx_ptr).io_value), to128((*ctx_ptr).atomic_expected)) };
        let op = AtomicOp::from_u8(op).unwrap_or(AtomicOp::Cas);
        if let Some(new) = apply(op, size, old, operand, expected) {
            let _ = copy_to_regions(&regions, addr, &new.to_le_bytes()[..n], true);
        }
        unsafe { (*ctx_ptr).io_value = [old as u64, (old >> 64) as u64] };
    }
    if regions.exec_overlaps(addr, len) {
        state.queue_invalidation(addr, len);
        unsafe {
            (*ctx_ptr).should_halt = 1;
        }
    }
}

#[cfg(test)]
mod halt_request_tests {
    use super::*;

    const CODE: u64 = 0x10_0000;
    const PAGE: usize = 0x1000;
    const COUNTER: u64 = 0x1234_5678_90ab_cdef;
    const GUARD: u64 = 0xfacd_8537_1942_6000;

    unsafe extern "C" fn post_halt_in_flight(context: *mut CpuContext) -> u64 {
        let state = unsafe { &*context.cast::<State>() };
        state.control.halt.store(true, Ordering::Release);
        state.control.halt.store(true, Ordering::Release);
        COUNTER
    }

    struct Fixture {
        cpu: RustarmicCpu,
        code: Box<[u8; PAGE]>,
        initial_code: Box<[u8; PAGE]>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut code = Box::new([0x55; PAGE]);
            for (index, word) in [
                0xd520_0000u32 | (u32::from(rustarmic::arch::sysreg::CNTPCT_EL0) << 5) | 9,
                0x9100_0400,
                0xd400_0fc1,
                0xd420_0020,
            ]
            .iter()
            .enumerate()
            {
                code[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            let mut cpu = RustarmicCpu::new_with_jit_config(JitConfig {
                code_cache_bytes: 64 * 1024,
                ..JitConfig::default()
            })
            .expect("CPU initialization");
            unsafe {
                cpu.map_host(CODE, PAGE as u64, Perm::RX, code.as_mut_ptr())
                    .unwrap();
            }
            cpu.state.ctx.read_cntpct = post_halt_in_flight;
            Self {
                initial_code: code.clone(),
                cpu,
                code,
            }
        }

        fn seed(&mut self, iteration: u64) {
            assert!(!self.cpu.state.control.halt.load(Ordering::Acquire));
            let context = &mut self.cpu.state.ctx;
            for (index, register) in context.x.iter_mut().enumerate() {
                *register = GUARD ^ (index as u64).wrapping_mul(0x1020_3040_5060_7081);
            }
            context.x[0] = iteration;
            for (index, vector) in context.v.iter_mut().enumerate() {
                *vector = [GUARD ^ index as u64, !GUARD ^ ((index as u64) << 32)];
            }
            context.pc = CODE;
            context.sp = GUARD;
            context.tpidr_el0 = GUARD ^ 1;
            context.tpidrro_el0 = GUARD ^ 2;
            context.fpcr = 0x0340_0000;
            context.fpsr = 0x0800_009f;
            context.nzcv = 0xb;
            context.exclusive_addr = GUARD ^ 3;
            context.exclusive_value = GUARD ^ 4;
            context.exclusive_size = 8;
        }

        fn guards(&self, iteration: u64, increment: u64, pc: u64) {
            let context = &self.cpu.state.ctx;
            for (index, &register) in context.x.iter().enumerate() {
                let expected = match index {
                    0 => iteration + increment,
                    9 => COUNTER,
                    _ => GUARD ^ (index as u64).wrapping_mul(0x1020_3040_5060_7081),
                };
                assert_eq!(register, expected, "X{index}");
            }
            for (index, &vector) in context.v.iter().enumerate() {
                assert_eq!(
                    vector,
                    [GUARD ^ index as u64, !GUARD ^ ((index as u64) << 32)],
                    "V{index}"
                );
            }
            assert_eq!(context.pc, pc, "PC");
            assert_eq!(context.sp, GUARD, "SP");
            assert_eq!(context.tpidr_el0, GUARD ^ 1, "TPIDR_EL0");
            assert_eq!(context.tpidrro_el0, GUARD ^ 2, "TPIDRRO_EL0");
            assert_eq!(context.fpcr, 0x0340_0000, "FPCR");
            assert_eq!(context.fpsr, 0x0800_009f, "FPSR");
            assert_eq!(context.nzcv, 0xb, "NZCV");
            assert_eq!(context.exclusive_addr, GUARD ^ 3, "exclusive address");
            assert_eq!(context.exclusive_value, GUARD ^ 4, "exclusive value");
            assert_eq!(context.exclusive_size, 8, "exclusive size");
            assert_eq!(context.should_halt, 0, "stale context halt");
            assert_eq!(context.mem_fault, 0, "memory fault");
            assert!(self.cpu.take_fault().is_none());
            assert_eq!(&self.code[..], &self.initial_code[..], "code canaries");
        }
    }

    #[test]
    fn in_flight_host_halt_survives_budget_exit_and_coalesces_for_one_slice() {
        let mut fixture = Fixture::new();
        for iteration in 0..32 {
            fixture.seed(iteration);
            let (event, retired) = fixture.cpu.run_with_count(1);
            assert!(matches!(event, CpuEvent::Running), "{event:?}");
            assert_eq!(retired, 1);
            fixture.guards(iteration, 0, CODE + 4);
            assert!(fixture.cpu.state.control.halt.load(Ordering::Acquire));

            let (event, retired) = fixture.cpu.run_with_count(64);
            assert!(matches!(event, CpuEvent::Running), "{event:?}");
            assert_eq!(retired, 0, "in-flight halt request was lost");
            fixture.guards(iteration, 0, CODE + 4);
            assert!(!fixture.cpu.state.control.halt.load(Ordering::Acquire));

            let (event, retired) = fixture.cpu.run_with_count(64);
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
            assert_eq!(retired, 2, "coalesced halt must not stop a second slice");
            fixture.guards(iteration, 1, CODE + 12);
        }
    }

    #[test]
    fn in_flight_host_halt_survives_svc_exit_without_replaying_the_svc() {
        let mut fixture = Fixture::new();
        for iteration in 0..32 {
            fixture.seed(iteration);
            let (event, retired) = fixture.cpu.run_with_count(64);
            assert!(matches!(event, CpuEvent::Svc(0x7e)), "{event:?}");
            assert_eq!(retired, 3);
            fixture.guards(iteration, 1, CODE + 12);
            assert!(fixture.cpu.state.control.halt.load(Ordering::Acquire));

            let (event, retired) = fixture.cpu.run_with_count(64);
            assert!(matches!(event, CpuEvent::Running), "{event:?}");
            assert_eq!(retired, 0, "SVC exit erased an in-flight halt request");
            fixture.guards(iteration, 1, CODE + 12);
            assert!(!fixture.cpu.state.control.halt.load(Ordering::Acquire));

            let (event, retired) = fixture.cpu.run_with_count(64);
            assert!(matches!(event, CpuEvent::Exception(0x101)), "{event:?}");
            assert_eq!(retired, 1, "resume must continue after, not replay, the SVC");
            fixture.guards(iteration, 1, CODE + 12);
        }
    }
}

#[cfg(test)]
mod executable_overlap_tests {
    use super::*;

    fn overlaps_executable(ranges: &[(u64, u64)], addr: u64, size: u64) -> bool {
        let mut regions = Regions::default();
        for &(va, end) in ranges {
            regions.push(Region {
                va,
                end,
                host_ptr: std::ptr::null_mut(),
                perm: Perm::RX,
            });
        }
        regions.exec_overlaps(addr, size)
    }

    #[test]
    fn executable_overlap_uses_half_open_ranges_and_exact_widths() {
        let ranges = [(0x2000, 0x3000), (0x5000, 0x6000)];
        assert!(!overlaps_executable(&ranges, 0x1000, 0x1000));
        assert!(overlaps_executable(&ranges, 0x1fff, 2));
        assert!(overlaps_executable(&ranges, 0x2fff, 1));
        assert!(!overlaps_executable(&ranges, 0x3000, 0x2000));
        assert!(overlaps_executable(&ranges, 0x3000, 0x2001));
        assert!(!overlaps_executable(&ranges, 0x6000, 1));
        assert!(!overlaps_executable(&ranges, 0x2000, 0));
        assert!(!overlaps_executable(&ranges, u64::MAX - 1, 4));
    }

    #[test]
    fn executable_overlap_preserves_unsorted_and_overlapping_range_semantics() {
        let ranges = [(0x5000, 0x6000), (0x2000, 0x3000), (0x2800, 0x4000)];
        assert!(overlaps_executable(&ranges, 0x3fff, 1));
        assert!(overlaps_executable(&ranges, 0x5fff, 1));
        assert!(!overlaps_executable(&ranges, 0x4000, 0x1000));
        assert!(!overlaps_executable(&[], 0x2000, 8));
    }
}

fn rustarmic_watch_overlaps(addr: u64, size: u64) -> bool {
    let Some((lo, hi)) = rustarmic_watch_range() else {
        return false;
    };
    addr < hi && addr.saturating_add(size) > lo
}

fn rustarmic_watch_range() -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static RANGE: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *RANGE.get_or_init(|| {
        let spec = std::env::var("NEXIUM_WATCH_WRITE_CPU").ok()?;
        let (va, len) = spec.trim().split_once(':')?;
        let va = parse_watch_u64(va.trim())?;
        let len = parse_watch_u64(len.trim()).unwrap_or(0x80);
        (va != 0 && len != 0).then_some((va, va.saturating_add(len)))
    })
}

fn parse_watch_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u64>()
            .ok()
            .or_else(|| u64::from_str_radix(s, 16).ok())
    }
}

fn handle_unmapped(
    state: &State,
    ctx_ptr: *mut CpuContext,
    addr: u64,
    size: u8,
    is_write: bool,
    value: u64,
    cause: u8,
) {
    if unsafe { (*ctx_ptr).mem_fault != 0 } {
        return;
    }
    let is_null_zone = addr < 0x1000;
    let mut regs = [0u64; 31];
    let (live_pc, live_lr, live_sp);
    unsafe {
        let ctx = &*ctx_ptr;
        for i in 0..31 {
            regs[i] = ctx.x[i];
        }
        live_pc = ctx.pc;
        live_lr = ctx.x[30];
        live_sp = ctx.sp;
    }
    let snap = FaultSnapshot {
        pc: live_pc,
        lr: live_lr,
        sp: live_sp,
        addr,
        size: size as u32,
        is_write,
        value,
        regs,
    };
    *state.last_fault.lock().unwrap() = Some(snap);
    // Legacy null policy: a null-page access is dropped (a load reads 0)
    // and the guest keeps running, as the Dynarmic backend's unmapped
    // callback does, until the skip cap.  Everything else is a data abort.
    let continue_null = state.continue_on_null.load(Ordering::Relaxed);
    let skip_null = is_null_zone
        && continue_null
        && state.null_skip_count.fetch_add(1, Ordering::Relaxed) + 1 <= NULL_SKIP_MAX;
    if !skip_null {
        // The JIT reports this as `StopReason::MemoryFault`, which
        // `run_with_count` turns into `CpuEvent::Exception(0x0E)`; queueing
        // the event as well would deliver the same fault twice.
        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.mem_fault = 1;
            ctx.mem_fault_access = u8::from(is_write);
            ctx.mem_fault_size = size;
            ctx.mem_fault_cause = cause;
            ctx.mem_fault_addr = addr;
            ctx.mem_fault_pc = ctx.pc;
            ctx.should_halt = 1;
        }
    }
    if is_null_zone {
        log::error!(
            "[null-deref] addr={:#x} size={} write={} val={:#x}",
            addr,
            size,
            is_write,
            value
        );
        if continue_null && !skip_null {
            log::error!("[null-deref] skip cap ({}) exceeded, reporting the fault", NULL_SKIP_MAX);
        }
    } else {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNT: AtomicU64 = AtomicU64::new(0);
        const RUNAWAY_THRESHOLD: u64 = 1_000_000;

        let n = COUNT.fetch_add(1, Ordering::Relaxed);
        if n < 16 {
            log::warn!(
                "rustarmic: unmapped {:#x} size={} write={}",
                addr,
                size,
                is_write
            );
        } else if n & 0xFFFF == 0 {
            log::warn!("rustarmic: unmapped {:#x} size={} write={} (total {} so far — likely runaway loop)",
                addr, size, is_write, n + 1);
        }
        if n == RUNAWAY_THRESHOLD {
            let (pc, sp, regs) = unsafe {
                let ctx = &*ctx_ptr;
                let mut r = [0u64; 31];
                for i in 0..31 {
                    r[i] = ctx.x[i];
                }
                (ctx.pc, ctx.sp, r)
            };
            log::error!(
                "rustarmic: runaway-loop threshold ({} unmapped accesses) reached at addr={:#x}",
                RUNAWAY_THRESHOLD,
                addr
            );
            log::error!(
                "  PC = {:#018x}  SP = {:#018x}  LR(x30) = {:#018x}",
                pc,
                sp,
                regs[30]
            );
            log::error!("  x0..x7   {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[0], regs[1], regs[2], regs[3], regs[4], regs[5], regs[6], regs[7]);
            log::error!("  x8..x15  {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[8], regs[9], regs[10], regs[11], regs[12], regs[13], regs[14], regs[15]);
            log::error!("  x16..x23 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[16], regs[17], regs[18], regs[19], regs[20], regs[21], regs[22], regs[23]);
            log::error!(
                "  x24..x30 {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x} {:#018x}",
                regs[24],
                regs[25],
                regs[26],
                regs[27],
                regs[28],
                regs[29],
                regs[30]
            );
            log::error!("  (each access is reported as a data abort)");
        }
    }
}

// ---------------------------------------------------------------------------
// Lockstep differential execution (`crate::lockstep`).  Everything below is
// only reached when `NEXIUM_LOCKSTEP` attached a shadow to this core; the
// normal path keeps its own hooks and never consults any of it.

/// Write and timer observation for one lockstepped core.  The hooks below
/// wrap the normal ones and record into `log` while it is enabled.
#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
pub(crate) struct LockstepHooks {
    log: parking_lot::Mutex<crate::lockstep::HookLog>,
    orig_cntpct: unsafe extern "C" fn(*mut CpuContext) -> u64,
    /// All-zero page table swapped in for capture runs, so that every load
    /// takes `mem_read` (and is logged) as well.
    no_direct: Option<PageTableMem>,
}

/// Copies guest memory out without permission checks.
#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
fn raw_copy_from(regions: &Regions, va: u64, out: &mut [u8]) -> bool {
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < out.len() {
        let Some(region) = region_at(regions, cursor) else {
            return false;
        };
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(out.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                region.host_ptr.add((cursor - region.va) as usize),
                out.as_mut_ptr().add(copied),
                count,
            );
        }
        copied += count;
        let Some(next) = cursor.checked_add(count as u64) else {
            return copied == out.len();
        };
        cursor = next;
    }
    true
}

/// Copies into guest memory without permission checks.
#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
fn raw_copy_to(regions: &Regions, va: u64, input: &[u8]) -> bool {
    let mut cursor = va;
    let mut copied = 0usize;
    while copied < input.len() {
        let Some(region) = region_at(regions, cursor) else {
            return false;
        };
        let available = usize::try_from(region.end - cursor).unwrap_or(usize::MAX);
        let count = available.min(input.len() - copied);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr().add(copied),
                region.host_ptr.add((cursor - region.va) as usize),
                count,
            );
        }
        copied += count;
        let Some(next) = cursor.checked_add(count as u64) else {
            return copied == input.len();
        };
        cursor = next;
    }
    true
}

#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
unsafe extern "C" fn ls_mem_read_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    if let Some(hooks) = state.lockstep_hooks.as_deref() {
        let mut log = hooks.log.lock();
        if log.enabled && log.log_reads {
            log.reads.push((addr, size));
        }
    }
    unsafe { mem_read_hook(ctx_ptr, addr, size) }
}

#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
unsafe extern "C" fn ls_mem_write_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    if let Some(hooks) = state.lockstep_hooks.as_deref() {
        let mut log = hooks.log.lock();
        if log.enabled && size <= 16 {
            let regions = state.regions.read_recursive();
            let n = size as usize;
            let mut old = [0u8; 16];
            // Only stores the normal hook will perform are recorded.
            if first_fault(&regions, addr, u64::from(size), Perm::W).is_none()
                && raw_copy_from(&regions, addr, &mut old[..n])
            {
                let io = unsafe { (*ctx_ptr).io_value };
                let mut new = [0u8; 16];
                new[..8].copy_from_slice(&io[0].to_le_bytes());
                new[8..].copy_from_slice(&io[1].to_le_bytes());
                log.writes.push(crate::lockstep::MemWrite {
                    addr,
                    size,
                    old,
                    new,
                    step: u32::MAX,
                });
            }
        }
    }
    unsafe { mem_write_hook(ctx_ptr, addr, size) }
}

#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
unsafe extern "C" fn ls_mem_atomic_hook(ctx_ptr: *mut CpuContext, addr: u64, size: u8, op: u8) {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let hooks = state.lockstep_hooks.as_deref();
    let logging = hooks.is_some_and(|h| {
        let mut log = h.log.lock();
        if log.enabled {
            log.atomics += 1;
        }
        log.enabled
    });
    let skipped = unsafe { (*ctx_ptr).mem_fault != 0 };
    let n = (size as usize).min(16);
    let mut old = [0u8; 16];
    let before = logging
        && !skipped
        && raw_copy_from(&state.regions.read_recursive(), addr, &mut old[..n]);
    unsafe { mem_atomic_hook(ctx_ptr, addr, size, op) };
    if let (true, Some(hooks)) = (before, hooks) {
        let mut new = [0u8; 16];
        if raw_copy_from(&state.regions.read_recursive(), addr, &mut new[..n]) {
            let mut log = hooks.log.lock();
            if log.log_reads {
                log.reads.push((addr, size));
            }
            if new[..n] != old[..n] {
                log.writes.push(crate::lockstep::MemWrite {
                    addr,
                    size: n as u8,
                    old,
                    new,
                    step: u32::MAX,
                });
            }
        }
    }
}

#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
unsafe extern "C" fn ls_read_cntpct(ctx_ptr: *mut CpuContext) -> u64 {
    let state = unsafe { &*(ctx_ptr as *mut State) };
    let Some(hooks) = state.lockstep_hooks.as_deref() else {
        return 0;
    };
    let value = unsafe { (hooks.orig_cntpct)(ctx_ptr) };
    let mut log = hooks.log.lock();
    if log.enabled {
        log.timer.push(value);
    }
    value
}

/// Instruction fetch that records every address the translator reads.
#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
struct LoggingFetch<'a, 'b> {
    inner: RegionMemory<'a>,
    log: &'b mut Vec<u64>,
}

#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
impl Memory for LoggingFetch<'_, '_> {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.log.push(addr);
        self.inner.fetch_inst(addr)
    }
}

/// The primitives `crate::lockstep` drives this core with.
#[cfg(all(feature = "backend-dynarmic", target_os = "linux", target_arch = "x86_64"))]
impl RustarmicCpu {
    /// Attaches a shadow Dynarmic and routes stores and atomics through the
    /// recording hooks, timer reads too.  Must run before any `map_host`.
    pub(crate) fn attach_lockstep(&mut self, config: crate::lockstep::Config) {
        match crate::lockstep::Lockstep::new(config) {
            Ok(lockstep) => {
                let no_direct = self
                    .state
                    .fastmem
                    .as_ref()
                    .and_then(|t| PageTableMem::new(1usize << (t.bits - 12)));
                self.state.lockstep_hooks = Some(Box::new(LockstepHooks {
                    log: parking_lot::Mutex::new(crate::lockstep::HookLog::default()),
                    orig_cntpct: self.state.ctx.read_cntpct,
                    no_direct,
                }));
                self.state.ctx.mem_read = ls_mem_read_hook;
                self.state.ctx.mem_write = ls_mem_write_hook;
                self.state.ctx.mem_atomic = ls_mem_atomic_hook;
                self.state.ctx.read_cntpct = ls_read_cntpct;
                self.lockstep = Some(Box::new(lockstep));
            }
            Err(error) => {
                log::error!("lockstep: no shadow Dynarmic for this core, lockstep off: {error}");
            }
        }
    }

    pub(crate) fn ls_ctx(&self) -> &CpuContext {
        &self.state.ctx
    }

    pub(crate) fn ls_ctx_mut(&mut self) -> &mut CpuContext {
        &mut self.state.ctx
    }

    /// Applies queued self-modifying-code invalidations to this JIT and
    /// returns them so the caller can apply them to the shadow too.
    pub(crate) fn ls_take_invalidations(&mut self) -> Vec<(u64, u64)> {
        let queued = self.state.take_invalidations();
        for &(va, len) in &queued {
            self.jit.invalidate_range(va, len);
        }
        queued
    }

    pub(crate) fn ls_take_event(&self) -> Option<CpuEvent> {
        self.state.take_event()
    }

    pub(crate) fn ls_halt_requested(&self) -> bool {
        self.state.control.halt.load(Ordering::Acquire)
    }

    pub(crate) fn ls_clear_halt(&self) {
        self.state.control.halt.store(false, Ordering::Release);
    }

    pub(crate) fn ls_peek_begin(&self) {
        self.state.control.snapshot_seq.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn ls_peek_end(&self) {
        let control = &self.state.control;
        control.peek_pc.store(self.state.ctx.pc, Ordering::Relaxed);
        control.peek_lr.store(self.state.ctx.x[30], Ordering::Relaxed);
        control.peek_sp.store(self.state.ctx.sp, Ordering::Relaxed);
        control.snapshot_seq.fetch_add(1, Ordering::Release);
    }

    /// Starts recording what the guest stores plus the timer values it reads (loads too when
    /// `log_reads`).
    pub(crate) fn ls_log_begin(&self, log_reads: bool) {
        if let Some(hooks) = self.state.lockstep_hooks.as_deref() {
            let mut log = hooks.log.lock();
            log.writes.clear();
            log.reads.clear();
            log.timer.clear();
            log.atomics = 0;
            log.enabled = true;
            log.log_reads = log_reads;
        }
    }

    /// Stops recording and returns what was recorded.
    pub(crate) fn ls_log_end(&self) -> crate::lockstep::HookLog {
        match self.state.lockstep_hooks.as_deref() {
            Some(hooks) => {
                let mut log = hooks.log.lock();
                log.enabled = false;
                log.log_reads = false;
                std::mem::take(&mut *log)
            }
            None => crate::lockstep::HookLog::default(),
        }
    }

    /// One `run_bounded` call.  `fetches` (capture runs) also forces every
    /// load through the hooks and records every instruction fetch.
    pub(crate) fn ls_run(
        &mut self,
        budget: u64,
        fetches: Option<&mut Vec<u64>>,
    ) -> Result<rustarmic::RunOutcome, rustarmic::Error> {
        let state = &mut *self.state;
        let normal_table = state.ctx.page_table;
        let capture = fetches.is_some();
        if capture {
            if let Some(table) = state.lockstep_hooks.as_ref().and_then(|h| h.no_direct.as_ref()) {
                state.ctx.page_table = table.as_ptr();
            }
        }
        let result = {
            let inner = RegionMemory {
                regions: state.regions.read_recursive(),
            };
            match fetches {
                Some(log) => {
                    let mut mem = LoggingFetch { inner, log };
                    self.jit.run_bounded(&mut state.ctx, &mut mem, budget)
                }
                None => {
                    let mut mem = inner;
                    self.jit.run_bounded(&mut state.ctx, &mut mem, budget)
                }
            }
        };
        state.ctx.page_table = normal_table;
        result
    }

    /// Reads guest memory ignoring permissions.
    pub(crate) fn ls_read(&self, va: u64, buf: &mut [u8]) -> bool {
        raw_copy_from(&self.state.regions.read(), va, buf)
    }

    /// Writes guest memory ignoring permissions; translated code covering
    /// it is invalidated.  `None` when unmapped, else whether it touched an
    /// executable region.
    pub(crate) fn ls_write(&mut self, va: u64, bytes: &[u8]) -> Option<bool> {
        let (ok, exec) = {
            let regions = self.state.regions.read();
            (
                raw_copy_to(&regions, va, bytes),
                regions.exec_overlaps(va, bytes.len() as u64),
            )
        };
        if exec {
            self.jit.invalidate_range(va, bytes.len() as u64);
        }
        ok.then_some(exec)
    }

    pub(crate) fn ls_exec_overlaps(&self, va: u64, len: u64) -> bool {
        self.state.regions.read().exec_overlaps(va, len)
    }

    pub(crate) fn ls_invalidate_jit(&mut self, va: u64, len: u64) {
        self.jit.invalidate_range(va, len);
    }

    #[cfg(test)]
    pub(crate) fn ls_jit_mut(&mut self) -> &mut Jit {
        &mut self.jit
    }

    #[cfg(test)]
    pub(crate) fn ls_lockstep(&self) -> Option<&crate::lockstep::Lockstep> {
        self.lockstep.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn ls_lockstep_mut(&mut self) -> Option<&mut crate::lockstep::Lockstep> {
        self.lockstep.as_deref_mut()
    }

    #[cfg(test)]
    pub(crate) fn ls_new_with_lockstep(config: crate::lockstep::Config) -> Result<Self, String> {
        let mut cpu = Self::new_with_jit_config(JitConfig::default())?;
        if cpu.lockstep.is_none() {
            cpu.attach_lockstep(config);
        }
        Ok(cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(va: u64, end: u64, perm: Perm, host: u64) -> Region {
        Region {
            va,
            end,
            host_ptr: host as *mut u8,
            perm,
        }
    }

    fn regions(list: &[Region]) -> Regions {
        let mut r = Regions::default();
        for &x in list {
            r.push(x);
        }
        r
    }

    #[test]
    fn region_lookup_matches_first_match_semantics() {
        let disjoint = regions(&[
            region(0x3000, 0x4000, Perm::RW, 0),
            region(0x1000, 0x2000, Perm::RX, 0),
            region(0x2000, 0x2800, Perm::RO, 0),
        ]);
        assert!(disjoint.disjoint);
        assert_eq!(disjoint.find(0x0fff).map(|r| r.va), None);
        assert_eq!(disjoint.find(0x1000).map(|r| r.va), Some(0x1000));
        assert_eq!(disjoint.find(0x1fff).map(|r| r.va), Some(0x1000));
        assert_eq!(disjoint.find(0x2000).map(|r| r.va), Some(0x2000));
        assert_eq!(disjoint.find(0x2800).map(|r| r.va), None);
        assert_eq!(disjoint.find(0x3fff).map(|r| r.va), Some(0x3000));
        assert_eq!(disjoint.find(0x4000).map(|r| r.va), None);
        assert!(disjoint.exec_overlaps(0x0ff8, 16));
        assert!(disjoint.exec_overlaps(0x1ff8, 8));
        assert!(!disjoint.exec_overlaps(0x2000, 16));
        assert!(!disjoint.exec_overlaps(0x0ff0, 16));
        assert!(!disjoint.exec_overlaps(0x1000, 0));

        // Overlap: the earlier-inserted region wins, as before.
        let overlapping = regions(&[
            region(0x1800, 0x2000, Perm::RO, 0),
            region(0x1000, 0x3000, Perm::RWX, 0),
        ]);
        assert!(!overlapping.disjoint);
        assert_eq!(overlapping.find(0x1900).unwrap().perm, Perm::RO);
        assert_eq!(overlapping.find(0x1000).unwrap().perm, Perm::RWX);
        assert!(overlapping.exec_overlaps(0x1900, 4));
    }

    #[test]
    fn incremental_index_matches_a_full_rebuild() {
        let mut r = Regions::default();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let key = |x: &Region| (x.va, x.end, x.host_ptr as u64);
        for _ in 0..4000 {
            let va = next(64) * 0x1000;
            let end = va + next(8) * 0x1000;
            match next(6) {
                0 | 1 => {
                    r.remove_range(va, end);
                }
                2 => r.push(region(va, end, Perm::RW, 0x10_0000 + va)),
                _ => {
                    r.remove_range(va, end);
                    r.push(region(va, end, Perm::RW, 0x10_0000 + va));
                }
            }
            let mut rebuilt = Regions {
                list: r.list.clone(),
                ..Regions::default()
            };
            rebuilt.rebuild_index();
            assert_eq!(r.disjoint, rebuilt.disjoint);
            if r.disjoint {
                assert_eq!(
                    r.sorted.iter().map(key).collect::<Vec<_>>(),
                    rebuilt.sorted.iter().map(key).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn remove_overlapping_reports_whole_removed_extent() {
        let mut r = regions(&[
            region(0x1000, 0x5000, Perm::RW, 0),
            region(0x8000, 0x9000, Perm::RW, 0),
        ]);
        assert_eq!(r.remove_overlapping(0x4000, 0x4001), Some((0x1000, 0x5000)));
        assert_eq!(r.remove_overlapping(0x4000, 0x4001), None);
        assert_eq!(r.len(), 1);
        assert_eq!(r.find(0x8000).map(|r| r.va), Some(0x8000));
    }

    const BASE: u64 = 0x7000_0000_0000;

    fn table() -> FastmemTable {
        FastmemTable {
            bits: 20,
            table: PageTableMem::new(1 << 8).unwrap(),
            mem_base: BASE as *mut u8,
        }
    }

    const RW: u8 = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;

    #[test]
    fn page_table_only_grants_whole_arena_backed_pages() {
        let mut t = table();
        let r = regions(&[
            // Arena-backed, unaligned on both ends.
            region(0x1800, 0x4800, Perm::RW, BASE + 0x1800),
            // Heap-backed: never direct.
            region(0x10000, 0x12000, Perm::RW, 0x5555_0000),
            // Read-only, arena-backed.
            region(0x20000, 0x21000, Perm::RO, BASE + 0x20000),
            // Past the table limit.
            region(0xFF000, 0x101000, Perm::RW, BASE + 0xFF000),
        ]);
        t.refresh(&r, 0, u64::MAX);
        assert_eq!(&t.table[0..6], &[0, 0, RW, RW, 0, 0]);
        assert_eq!(&t.table[0x10..0x12], &[0, 0]);
        assert_eq!(t.table[0x20], PAGE_DIRECT_READ);
        assert_eq!(t.table[0xFF], 0, "region crossing the limit is not direct");
    }

    #[test]
    fn page_table_keeps_code_and_watch_writes_on_the_hook() {
        let mut t = table();
        let r = regions(&[
            region(0x1000, 0x3000, Perm::RW, BASE + 0x1000),
            region(0x3000, 0x4000, Perm::RWX, BASE + 0x3000),
            region(0x4000, 0x5000, Perm::RX, BASE + 0x4000),
            // An executable region shadowed by an earlier RW one still
            // makes stores take the hook (the hook's SMC check is "any").
            region(0x1800, 0x1900, Perm::RX, BASE + 0x1800),
        ]);
        t.refresh(&r, 0, u64::MAX);
        assert_eq!(t.table[2], RW);
        assert_eq!(t.table[1], PAGE_DIRECT_READ);
        assert_eq!(t.table[3], PAGE_DIRECT_READ);
        assert_eq!(t.table[4], PAGE_DIRECT_READ);
    }

    #[test]
    fn page_table_follows_first_match_priority_and_unmap() {
        let mut t = table();
        let mut r = regions(&[
            region(0x1000, 0x3000, Perm::RO, BASE + 0x1000),
            region(0x0000, 0x8000, Perm::RW, BASE),
        ]);
        t.refresh(&r, 0, u64::MAX);
        assert_eq!(
            &t.table[0..8],
            &[RW, PAGE_DIRECT_READ, PAGE_DIRECT_READ, RW, RW, RW, RW, RW]
        );
        // Partial overlap by the higher-priority region: that page is slow.
        let mut t2 = table();
        let r2 = regions(&[
            region(0x1800, 0x2000, Perm::RO, BASE + 0x1800),
            region(0x0000, 0x8000, Perm::RW, BASE),
        ]);
        t2.refresh(&r2, 0, u64::MAX);
        assert_eq!(&t2.table[0..3], &[RW, 0, RW]);

        // Unmap removes every overlapping region (both here) and the
        // removed extent is recomputed from what is left.
        let (lo, hi) = r.remove_overlapping(0x1000, 0x1001).unwrap();
        assert_eq!((lo, hi), (0, 0x8000));
        t.refresh(&r, lo, hi);
        assert!(t.table.iter().all(|&e| e == 0));
        // Re-mapping the low-priority region alone restores it.
        r.push(region(0x0000, 0x8000, Perm::RW, BASE));
        t.refresh(&r, 0, 0x8000);
        assert_eq!(&t.table[0..8], &[RW; 8]);
    }

    fn put_code(ptr: *mut u8, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), ptr.add(i * 4), 4)
            };
        }
    }

    /// End to end through the real arena: direct loads/stores, a store into
    /// an executable page still invalidates translated code, and an access
    /// to an unmapped page inside the arena is a guest fault.
    #[test]
    fn arena_fastmem_end_to_end() {
        const CODE: u64 = 0x7f_0000_0000;
        const DATA: u64 = 0x7f_0010_0000;
        if nexium_memory::fastmem::base().is_none()
            || nexium_memory::fastmem::direct_va_base().is_some()
            || !rustarmic_fastmem_enabled()
        {
            eprintln!("SKIPPED: fastmem arena unavailable");
            return;
        }
        let code = nexium_memory::fastmem::commit(CODE, 0x2000).expect("commit code");
        let data = nexium_memory::fastmem::commit(DATA, 0x2000).expect("commit data");
        let mut cpu = RustarmicCpu::new().expect("init");
        assert!(cpu.state.fastmem.is_some());
        unsafe {
            cpu.map_host(CODE, 0x2000, Perm::RWX, code).unwrap();
            cpu.map_host(DATA, 0x1000, Perm::RW, data).unwrap();
        }
        let page = |va: u64| cpu.state.fastmem.as_ref().unwrap().table[(va >> 12) as usize];
        assert_eq!(page(DATA), RW);
        assert_eq!(page(CODE), PAGE_DIRECT_READ);
        assert_eq!(page(DATA + 0x1000), 0);

        // str x1, [x0]; ldr x2, [x0, #8]; svc 0
        put_code(code, &[0xF900_0001, 0xF940_0402, 0xD400_0001]);
        unsafe { (data.add(8) as *mut u64).write_unaligned(0x1234_5678) };
        cpu.set_register(0, DATA);
        cpu.set_register(1, 0xAABB_CCDD_EEFF_0011);
        cpu.set_pc(CODE);
        let ev = cpu.run(1000);
        assert!(matches!(ev, CpuEvent::Svc(0)), "{ev:?} fault={:?}", cpu.take_fault());
        assert_eq!(unsafe { (data as *const u64).read_unaligned() }, 0xAABB_CCDD_EEFF_0011);
        assert_eq!(cpu.get_register(2), 0x1234_5678);

        // Code at CODE+0x100: movz x3, #1; svc 0.  Translate it once.
        put_code(unsafe { code.add(0x100) }, &[0xD280_0023, 0xD400_0001]);
        cpu.set_pc(CODE + 0x100);
        let ev = cpu.run(1000);
        assert!(matches!(ev, CpuEvent::Svc(0)), "{ev:?} fault={:?}", cpu.take_fault());
        assert_eq!(cpu.get_register(3), 1);
        // Guest overwrites it with movz x3, #2 (str w1, [x0]; svc 0).
        cpu.set_register(0, CODE + 0x100);
        cpu.set_register(1, 0xD280_0043);
        cpu.set_pc(CODE + 0x200);
        put_code(unsafe { code.add(0x200) }, &[0xB900_0001, 0xD400_0001]);
        let ev = cpu.run(1000);
        assert!(
            matches!(ev, CpuEvent::Svc(0) | CpuEvent::Interrupted | CpuEvent::Running),
            "{ev:?}"
        );
        // The SMC store requests a halt; the scheduler simply re-enters.
        cpu.set_pc(CODE + 0x100);
        let mut ev = cpu.run(1000);
        if matches!(ev, CpuEvent::Interrupted | CpuEvent::Running) {
            ev = cpu.run(1000);
        }
        assert!(matches!(ev, CpuEvent::Svc(0)), "{ev:?}");
        assert_eq!(cpu.get_register(3), 2, "stale translation after SMC");

        // Unmapped page inside the arena window: guest fault, not a crash.
        put_code(code, &[0xF940_0002, 0xD400_0001]); // ldr x2, [x0]
        cpu.invalidate_range(CODE, 8);
        cpu.set_register(0, DATA + 0x1000);
        cpu.set_pc(CODE);
        let first = cpu.run(1000);
        let ev = if matches!(first, CpuEvent::Exception(0x0E)) {
            first
        } else {
            cpu.run(1000)
        };
        assert!(matches!(ev, CpuEvent::Exception(0x0E)), "{first:?} {ev:?}");
        let fault = cpu.take_fault().expect("fault snapshot");
        assert_eq!(fault.addr, DATA + 0x1000);
        assert!(!fault.is_write);
    }

    /// Page-aligned heap memory standing in for guest RAM (not the fastmem
    /// arena, so every access takes the hooks).
    struct HostPages {
        ptr: *mut u8,
        layout: std::alloc::Layout,
    }
    impl HostPages {
        fn new(len: usize) -> Self {
            let layout = std::alloc::Layout::from_size_align(len, 0x1000).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }
    }
    impl Drop for HostPages {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    fn run_to_event(cpu: &mut RustarmicCpu) -> CpuEvent {
        for _ in 0..8 {
            let ev = cpu.run(10_000);
            if !matches!(ev, CpuEvent::Interrupted | CpuEvent::Running) {
                return ev;
            }
        }
        panic!("no event");
    }

    /// A 16-byte access running from a mapped page into an unmapped one
    /// faults at the first unmapped byte (not the access base); the store's
    /// mapped part may be performed.
    #[test]
    fn straddling_q_access_reports_first_unmapped_byte() {
        const CODE: u64 = 0x10_0000;
        const DATA: u64 = 0x20_0000;
        let code = HostPages::new(0x1000);
        let data = HostPages::new(0x1000);
        for (insn, is_write) in [(0x3D80_0000u32, true), (0x3DC0_0001, false)] {
            for off in [0xFF8u64, 0xFFC, 0xFF1] {
                let mut cpu = RustarmicCpu::new().expect("init");
                unsafe {
                    cpu.map_host(CODE, 0x1000, Perm::RX, code.ptr).unwrap();
                    cpu.map_host(DATA, 0x1000, Perm::RW, data.ptr).unwrap();
                }
                // str q0, [x0] / ldr q1, [x0]; svc 0
                put_code(code.ptr, &[insn, 0xD400_0001]);
                cpu.invalidate_range(CODE, 8);
                unsafe { std::ptr::write_bytes(data.ptr, 0, 0x1000) };
                cpu.state.ctx.v[0] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
                cpu.set_register(0, DATA + off);
                cpu.set_pc(CODE);
                let ev = run_to_event(&mut cpu);
                assert!(matches!(ev, CpuEvent::Exception(0x0E)), "{ev:?}");
                let fault = cpu.take_fault().expect("fault snapshot");
                assert_eq!(fault.addr, DATA + 0x1000, "insn {insn:#x} off {off:#x}");
                assert_eq!(fault.is_write, is_write);
                if is_write {
                    // The mapped low part was stored.
                    let n = (0x1000 - off) as usize;
                    let mut got = vec![0u8; n];
                    unsafe { std::ptr::copy_nonoverlapping(data.ptr.add(off as usize), got.as_mut_ptr(), n) };
                    let mut want = [0u8; 16];
                    want[..8].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
                    want[8..].copy_from_slice(&0x5555_6666_7777_8888u64.to_le_bytes());
                    assert_eq!(&got[..], &want[..n]);
                }
            }
        }
    }

    /// A data abort is reported once and the guest then continues past the
    /// faulting access (dropped, a load reads 0), as on Dynarmic, instead of
    /// re-entering the same access forever (the aux-core loop re-runs the
    /// faulting thread after an exception).  Exactly one exception per
    /// fault, not a queued duplicate.
    #[test]
    fn data_abort_resumes_past_the_faulting_access() {
        const CODE: u64 = 0x10_0000;
        const DATA: u64 = 0x20_0000;
        // ldr x1, [x0] / str x1, [x0]; add x2, x2, #1; svc 0
        for (insn, is_write) in [(0xF940_0001u32, false), (0xF900_0001, true)] {
            let code = HostPages::new(0x1000);
            let data = HostPages::new(0x1000);
            let mut cpu = RustarmicCpu::new().expect("init");
            unsafe {
                cpu.map_host(CODE, 0x1000, Perm::RX, code.ptr).unwrap();
                cpu.map_host(DATA, 0x1000, Perm::RW, data.ptr).unwrap();
            }
            put_code(code.ptr, &[insn, 0x9100_0442, 0xD400_0001]);
            cpu.set_register(0, DATA + 0x1000);
            cpu.set_register(1, 0x55);
            cpu.set_pc(CODE);
            let ev = run_to_event(&mut cpu);
            assert!(matches!(ev, CpuEvent::Exception(0x0E)), "{ev:?}");
            let fault = cpu.take_fault().expect("fault snapshot");
            assert_eq!((fault.addr, fault.is_write), (DATA + 0x1000, is_write));
            let ev = run_to_event(&mut cpu);
            assert!(matches!(ev, CpuEvent::Svc(0)), "retried the faulting access: {ev:?}");
            assert_eq!(cpu.get_register(2), 1, "instructions after the fault ran once");
            if !is_write {
                assert_eq!(cpu.get_register(1), 0, "a faulting load reads 0");
            }
        }
    }

    /// Faults keep making progress when the faulting access is followed by
    /// another callback access in the same block (skipped after the first
    /// fault): each re-entry moves past one faulting access.
    #[test]
    fn consecutive_faulting_accesses_each_report_once() {
        const CODE: u64 = 0x10_0000;
        let code = HostPages::new(0x1000);
        let mut cpu = RustarmicCpu::new().expect("init");
        unsafe { cpu.map_host(CODE, 0x1000, Perm::RX, code.ptr).unwrap() };
        // ldr x1, [x0]; str x1, [x0, #8]; add x2, x2, #1; svc 0
        put_code(code.ptr, &[0xF940_0001, 0xF900_0401, 0x9100_0442, 0xD400_0001]);
        cpu.set_register(0, 0x30_0000);
        cpu.set_pc(CODE);
        let mut faults = Vec::new();
        let ev = loop {
            let ev = run_to_event(&mut cpu);
            if !matches!(ev, CpuEvent::Exception(0x0E)) {
                break ev;
            }
            faults.push(cpu.take_fault().unwrap().addr);
            assert!(faults.len() <= 2, "fault storm: {faults:x?}");
        };
        assert!(matches!(ev, CpuEvent::Svc(0)), "{ev:?}");
        assert_eq!(faults, [0x30_0000, 0x30_0008]);
    }

    /// The legacy null policy drops null-page accesses and keeps running.
    #[test]
    fn continue_on_null_skips_without_an_exception() {
        const CODE: u64 = 0x10_0000;
        let code = HostPages::new(0x1000);
        let mut cpu = RustarmicCpu::new().expect("init");
        unsafe { cpu.map_host(CODE, 0x1000, Perm::RX, code.ptr).unwrap() };
        cpu.set_continue_on_null(true);
        // ldr x1, [x0]; add x2, x2, #1; svc 0
        put_code(code.ptr, &[0xF940_0001, 0x9100_0442, 0xD400_0001]);
        cpu.set_register(0, 0x20);
        cpu.set_register(1, 0x55);
        cpu.set_pc(CODE);
        let ev = run_to_event(&mut cpu);
        assert!(matches!(ev, CpuEvent::Svc(0)), "{ev:?}");
        assert_eq!((cpu.get_register(1), cpu.get_register(2)), (0, 1));
        assert_eq!(cpu.null_skip_count(), 1);
    }

    #[test]
    fn unmap_is_exact_and_refreshes_the_page_table() {
        let mut t = table();
        let mut r = regions(&[region(0x1000, 0x9000, Perm::RW, BASE + 0x1000)]);
        t.refresh(&r, 0, u64::MAX);
        assert_eq!(r.remove_range(0x3000, 0x5000), Some((0x3000, 0x5000)));
        assert_eq!(r.remove_range(0x3000, 0x5000), None);
        t.refresh(&r, 0x3000, 0x5000);
        assert_eq!(&t.table[0..10], &[0, RW, RW, 0, 0, RW, RW, RW, RW, 0]);
        let upper = r.find(0x5000).unwrap();
        assert_eq!((upper.va, upper.end, upper.host_ptr as u64), (0x5000, 0x9000, BASE + 0x5000));
        assert_eq!(r.find(0x2fff).map(|x| x.end), Some(0x3000));
        // Unaligned, inside one page: that page loses its direct bits only.
        assert_eq!(r.remove_range(0x6800, 0x6900), Some((0x6800, 0x6900)));
        t.refresh(&r, 0x6800, 0x6900);
        assert_eq!(&t.table[5..9], &[RW, 0, RW, RW]);
        assert!(r.find(0x6800).is_none() && r.find(0x6900).is_some() && r.find(0x67ff).is_some());
    }

    /// Several cores (one host thread and one `RustarmicCpu` each) running
    /// LDAXR/STLXR and LDADDAL increment loops on shared guest memory end
    /// with exact counts, through the hook path and, when the fastmem arena
    /// is available, through the inline path.
    #[test]
    fn multicore_exclusives_and_lse_are_atomic() {
        const CODE: u64 = 0x7f_0020_0000;
        const DATA: u64 = 0x7f_0030_0000;
        const THREADS: u64 = 4;
        const ITERS: u64 = 20_000;
        let loops: [(&str, Vec<u32>); 2] = [
            (
                "ldaxr/stlxr",
                vec![
                    0x885F_FC02, // ldaxr w2, [x0]
                    0x1100_0442, // add w2, w2, #1
                    0x8803_FC02, // stlxr w3, w2, [x0]
                    0x35FF_FFA3, // cbnz w3, -12
                    0xF100_0421, // subs x1, x1, #1
                    0x54FF_FF61, // b.ne -20
                    0xD400_0001, // svc 0
                ],
            ),
            (
                "ldaddal",
                vec![
                    0xB8E5_0006, // ldaddal w5, w6, [x0]
                    0xF100_0421, // subs x1, x1, #1
                    0x54FF_FFC1, // b.ne -8
                    0xD400_0001, // svc 0
                ],
            ),
        ];
        let arena = nexium_memory::fastmem::base().is_some()
            && nexium_memory::fastmem::direct_va_base().is_none()
            && rustarmic_fastmem_enabled();
        let mut backings: Vec<(&str, *mut u8, *mut u8, Option<HostPages>)> = Vec::new();
        let heap_code = HostPages::new(0x1000);
        let heap_data = HostPages::new(0x1000);
        backings.push(("heap", heap_code.ptr, heap_data.ptr, None));
        if arena {
            let c = nexium_memory::fastmem::commit(CODE, 0x1000).expect("commit code");
            let d = nexium_memory::fastmem::commit(DATA, 0x1000).expect("commit data");
            backings.push(("arena", c, d, None));
        } else {
            eprintln!("fastmem arena unavailable: hook path only");
        }
        for (name, code_ptr, data_ptr, _) in &backings {
            for (loop_name, words) in &loops {
                put_code(*code_ptr, words);
                unsafe { std::ptr::write_bytes(*data_ptr, 0, 0x1000) };
                let (code_addr, data_addr) = (*code_ptr as usize, *data_ptr as usize);
                std::thread::scope(|s| {
                    for _ in 0..THREADS {
                        s.spawn(move || {
                            let mut cpu = RustarmicCpu::new().expect("init");
                            unsafe {
                                cpu.map_host(CODE, 0x1000, Perm::RX, code_addr as *mut u8).unwrap();
                                cpu.map_host(DATA, 0x1000, Perm::RW, data_addr as *mut u8).unwrap();
                            }
                            cpu.set_register(0, DATA);
                            cpu.set_register(1, ITERS);
                            cpu.set_register(5, 1);
                            cpu.set_pc(CODE);
                            loop {
                                match cpu.run(5_000) {
                                    CpuEvent::Svc(0) => break,
                                    CpuEvent::Running | CpuEvent::Interrupted => {}
                                    other => panic!("{other:?} {:?}", cpu.take_fault()),
                                }
                            }
                        });
                    }
                });
                let got = unsafe { (*data_ptr as *const u32).read() };
                assert_eq!(u64::from(got), THREADS * ITERS, "{name} {loop_name}");
            }
        }
    }

    /// An atomic store into an executable region invalidates translated
    /// code, like a plain store.
    #[test]
    fn atomic_store_into_code_invalidates() {
        const CODE: u64 = 0x10_0000;
        let code = HostPages::new(0x1000);
        let mut cpu = RustarmicCpu::new().expect("init");
        unsafe { cpu.map_host(CODE, 0x1000, Perm::RWX, code.ptr).unwrap() };
        // +0x100: movz x3, #1; svc 0     +0x200: swpal w1, w2, [x0]; svc 0
        put_code(unsafe { code.ptr.add(0x100) }, &[0xD280_0023, 0xD400_0001]);
        put_code(unsafe { code.ptr.add(0x200) }, &[0xB8E1_8002, 0xD400_0001]);
        cpu.set_pc(CODE + 0x100);
        assert!(matches!(run_to_event(&mut cpu), CpuEvent::Svc(0)));
        assert_eq!(cpu.get_register(3), 1);
        cpu.set_register(0, CODE + 0x100);
        cpu.set_register(1, 0xD280_0043); // movz x3, #2
        cpu.set_pc(CODE + 0x200);
        assert!(matches!(run_to_event(&mut cpu), CpuEvent::Svc(0)));
        assert_eq!(cpu.get_register(2), 0xD280_0023, "swp old value");
        cpu.set_pc(CODE + 0x100);
        assert!(matches!(run_to_event(&mut cpu), CpuEvent::Svc(0)));
        assert_eq!(cpu.get_register(3), 2, "stale translation after atomic SMC");
    }
}

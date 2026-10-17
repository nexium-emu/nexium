//! Lockstep differential execution: Rustarmic checked against Dynarmic.
//!
//! With `NEXIUM_LOCKSTEP=1` (Rustarmic backend, `NEXIUM_SINGLECORE=1`) every
//! Rustarmic core carries a shadow Dynarmic JIT that mirrors its memory map.
//! Each slice is cut into chunks; for each chunk:
//!
//! 1. the guest state is snapshotted and the exclusive monitor cleared;
//! 2. Rustarmic runs up to `NEXIUM_LOCKSTEP_SLICE` instructions with every
//!    store and atomic recorded, along with each timer value it reads (inline fastmem stores are
//!    disabled so no store bypasses the hooks; inline loads stay on unless
//!    `NEXIUM_LOCKSTEP_FASTMEM=0`);
//! 3. its stores are rolled back and Dynarmic runs the same number of
//!    instructions from the same state over the same memory, its stores
//!    reported by dynarmic-sys's write callback.  Dynarmic first runs the
//!    whole count in one go (exact when its block boundary matches); if that
//!    disagrees it is single-stepped (`Jit::Step`) for the exact count, and
//!    every `MRS CNTPCT/CNTVCT` it executes is given the value Rustarmic read;
//! 4. GPRs, SP, PC, NZCV, FPCR, FPSR, V0-V31, TPIDR(RO), the stop reason and
//!    SVC number, the retired count and the net memory effect of both runs
//!    are compared.  Dynarmic's stores stay in memory: they are the commit.
//!
//! A mismatch is re-run from the snapshot to rule out races with host
//! threads, then bisected down to the single instruction whose execution
//! makes the states differ.  That instruction is classified (real JIT
//! divergence, FP NaN payload, FPCR mode, FPSR flags only, system register
//! value, Dynarmic's no-op `DC ZVA`, timer), reported once with full context
//! to the log and `~/.config/NeXium/lockstep/`, together with a minimized
//! repro that `replay` (and rustarmic's `lockstep_replay` Unicorn test) can
//! re-execute.  Real divergences stall the core (`NEXIUM_LOCKSTEP_ON_DIVERGE=
//! continue` adopts Dynarmic's state instead); benign classes adopt the
//! reference state and continue (`NEXIUM_LOCKSTEP_BENIGN=halt` stops).
//!
//! Memory faults are not compared blindly: the prefix before the faulting
//! instruction is checked, and the fault is classified as a guest fault or,
//! when the kernel has the address mapped, a core mapping-sync issue.
//!
//! Environment (read once at startup):
//!
//! | variable | default | meaning |
//! |---|---|---|
//! | `NEXIUM_LOCKSTEP=1` | off | enable (Rustarmic backend only) |
//! | `NEXIUM_SINGLECORE=1` | | required (or `NEXIUM_LOCKSTEP_MULTICORE=1`) |
//! | `NEXIUM_LOCKSTEP_SLICE` | 4096 | instructions per checked chunk |
//! | `NEXIUM_LOCKSTEP_ON_DIVERGE` | `halt` | `continue` adopts Dynarmic's state |
//! | `NEXIUM_LOCKSTEP_BENIGN` | `continue` | `halt` also stops on benign classes |
//! | `NEXIUM_LOCKSTEP_START` | 0 | instructions per core run unchecked first |
//! | `NEXIUM_LOCKSTEP_SAMPLE` | 1 | check one chunk in N |
//! | `NEXIUM_LOCKSTEP_DYN_RUN=0` | on | always single-step Dynarmic |
//! | `NEXIUM_LOCKSTEP_FASTMEM=0` | on | no inline Rustarmic loads either |
//! | `NEXIUM_LOCKSTEP_DIR` | `~/.config/NeXium/lockstep` | report/repro directory |
//! | `NEXIUM_LOCKSTEP_MAX_REPORTS` | 64 | report files per process |
//!
//! Every chunk starts with the exclusive monitor cleared on both JITs
//! (Dynarmic clears its own at every run).  That is architecturally allowed
//! and real code retries, but an LDXR/STXR pair split by a chunk boundary
//! fails where it would not without lockstep.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::c_void;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock, RwLock, Weak};
use std::time::Instant;

use nexium_memory::{AddressSpace, Perm};
use rustarmic::{CpuContext, MemoryFaultCause, StopReason};

use crate::dynarmic::DynarmicCpu;
use crate::rustarmic::RustarmicCpu;
use crate::CpuEvent;

/// The dynarmic-sys build whose internal layouts (`struct dynarmic`,
/// `struct context64`) the shadow relies on.
const SUPPORTED_DYNARMIC_VERSION: u32 = 20260209;
/// FPCR mode bits Rustarmic's inline FP ops ignore: AHP, DN, FZ, RMode, FZ16.
const FPCR_MODES: u32 = (1 << 26) | (1 << 25) | (1 << 24) | (3 << 22) | (1 << 19);
const ARM_DEFAULT_NAN32: u32 = 0x7FC0_0000;
const ARM_DEFAULT_NAN64: u64 = 0x7FF8_0000_0000_0000;

// ---------------------------------------------------------------------------
// Configuration.

#[derive(Clone, Debug)]
pub(crate) struct Config {
    /// Instructions per lockstep chunk.
    pub(crate) slice: u64,
    /// Stall the core on a real divergence (else adopt Dynarmic's state).
    pub(crate) halt_on_divergence: bool,
    /// Stall on benign classes too (else adopt the reference state).
    pub(crate) halt_on_benign: bool,
    /// Try one bounded Dynarmic run per chunk before single-stepping.
    pub(crate) dyn_fast_run: bool,
    /// Keep Rustarmic's inline (fastmem) loads; stores always use the hooks.
    pub(crate) fastmem_reads: bool,
    /// Instructions per core to run unchecked before checking starts.
    pub(crate) start_after: u64,
    /// Check one chunk in `sample`.
    pub(crate) sample: u64,
    /// Where reports and repros go; `None` disables files.
    pub(crate) out_dir: Option<PathBuf>,
    /// Report files written per process before only logging.
    pub(crate) max_reports: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            slice: 4096,
            halt_on_divergence: true,
            halt_on_benign: false,
            dyn_fast_run: true,
            fastmem_reads: true,
            start_after: 0,
            sample: 1,
            out_dir: default_out_dir(),
            max_reports: 64,
        }
    }
}

impl Config {
    fn from_env() -> Self {
        let mut c = Self::default();
        c.slice = env_u64("NEXIUM_LOCKSTEP_SLICE").unwrap_or(c.slice).clamp(1, 1 << 24);
        if let Ok(v) = std::env::var("NEXIUM_LOCKSTEP_ON_DIVERGE") {
            c.halt_on_divergence = !v.trim().eq_ignore_ascii_case("continue");
        }
        if let Ok(v) = std::env::var("NEXIUM_LOCKSTEP_BENIGN") {
            c.halt_on_benign = v.trim().eq_ignore_ascii_case("halt");
        }
        c.dyn_fast_run = !env_off("NEXIUM_LOCKSTEP_DYN_RUN");
        c.fastmem_reads = !env_off("NEXIUM_LOCKSTEP_FASTMEM");
        c.start_after = env_u64("NEXIUM_LOCKSTEP_START").unwrap_or(0);
        c.sample = env_u64("NEXIUM_LOCKSTEP_SAMPLE").unwrap_or(1).max(1);
        if let Some(dir) = std::env::var_os("NEXIUM_LOCKSTEP_DIR") {
            c.out_dir = Some(PathBuf::from(dir));
        }
        if let Some(n) = env_u64("NEXIUM_LOCKSTEP_MAX_REPORTS") {
            c.max_reports = n.min(u64::from(u32::MAX)) as u32;
        }
        c
    }
}

fn default_out_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("NeXium").join("lockstep"))
}

fn env_u64(name: &str) -> Option<u64> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => v.parse().ok(),
    }
}

fn env_off(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("0") | Ok("false") | Ok("off"))
}

fn env_on(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true") | Ok("on"))
}

struct Global {
    enabled: bool,
    config: Config,
}

fn global() -> &'static Global {
    static GLOBAL: OnceLock<Global> = OnceLock::new();
    GLOBAL.get_or_init(|| {
        let config = Config::from_env();
        let enabled = env_on("NEXIUM_LOCKSTEP") && lockstep_allowed(&config);
        Global { enabled, config }
    })
}

fn lockstep_allowed(config: &Config) -> bool {
    if std::env::var_os("NEXIUM_SINGLECORE").is_none() && !env_on("NEXIUM_LOCKSTEP_MULTICORE") {
        log::error!(
            "lockstep: NEXIUM_LOCKSTEP needs NEXIUM_SINGLECORE=1 (cores racing on shared memory make both JITs disagree); set NEXIUM_LOCKSTEP_MULTICORE=1 to run it anyway. Lockstep is OFF."
        );
        return false;
    }
    let version = dynarmic_sys::dynarmic_version();
    if version != SUPPORTED_DYNARMIC_VERSION {
        log::error!(
            "lockstep: dynarmic-sys version {version} is not the supported {SUPPORTED_DYNARMIC_VERSION}; lockstep is OFF"
        );
        return false;
    }
    log::warn!(
        "lockstep: ON (Rustarmic vs Dynarmic) slice={} on_diverge={} benign={} dyn_run={} fastmem_loads={} start_after={} sample=1/{} reports={}",
        config.slice,
        if config.halt_on_divergence { "halt" } else { "continue" },
        if config.halt_on_benign { "halt" } else { "continue" },
        config.dyn_fast_run,
        config.fastmem_reads,
        config.start_after,
        config.sample,
        config
            .out_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".into()),
    );
    true
}

thread_local! {
    /// Set while building replay CPUs, which must not get a shadow of their own.
    static SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}

struct Suppress(bool);

impl Drop for Suppress {
    fn drop(&mut self) {
        SUPPRESSED.with(|s| s.set(self.0));
    }
}

fn suppress() -> Suppress {
    Suppress(SUPPRESSED.with(|s| s.replace(true)))
}

/// Whether new Rustarmic cores get a shadow (`NEXIUM_LOCKSTEP=1` and its
/// requirements met).
pub(crate) fn enabled() -> bool {
    global().enabled && !SUPPRESSED.with(Cell::get)
}

pub(crate) fn config() -> &'static Config {
    &global().config
}

static KERNEL_SPACE: RwLock<Option<Weak<AddressSpace>>> = RwLock::new(None);

pub(crate) fn register_address_space(space: &Arc<AddressSpace>) {
    if let Ok(mut slot) = KERNEL_SPACE.write() {
        *slot = Some(Arc::downgrade(space));
    }
}

/// Whether the kernel's address space has `addr` readable (`None` when no
/// address space was registered).
fn kernel_mapped(addr: u64) -> Option<bool> {
    let space = KERNEL_SPACE.read().ok()?.as_ref()?.upgrade()?;
    let mut probe = [0u8; 1];
    Some(space.read(addr, &mut probe).is_ok())
}

// ---------------------------------------------------------------------------
// Guest state and memory effects.

/// Everything both JITs expose that an instruction can change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestState {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub v: [[u64; 2]; 32],
    /// N, Z, C, V in bits 3..0.
    pub nzcv: u8,
    pub fpcr: u32,
    pub fpsr: u32,
    pub tpidr: u64,
    pub tpidrro: u64,
}

impl GuestState {
    fn from_rust(ctx: &CpuContext) -> Self {
        Self {
            x: ctx.x,
            sp: ctx.sp,
            pc: ctx.pc,
            v: ctx.v,
            nzcv: ctx.nzcv & 0xF,
            fpcr: ctx.fpcr,
            fpsr: ctx.fpsr,
            tpidr: ctx.tpidr_el0,
            tpidrro: ctx.tpidrro_el0,
        }
    }

    /// Loads the state and clears the exclusive monitor (Dynarmic clears
    /// its own at every run, so an exclusive pair may not span a chunk).
    fn apply_rust(&self, ctx: &mut CpuContext) {
        ctx.x = self.x;
        ctx.sp = self.sp;
        ctx.pc = self.pc;
        ctx.v = self.v;
        ctx.nzcv = self.nzcv & 0xF;
        ctx.fpcr = self.fpcr;
        ctx.fpsr = self.fpsr;
        ctx.tpidr_el0 = self.tpidr;
        ctx.tpidrro_el0 = self.tpidrro;
        ctx.exclusive_addr = 0;
        ctx.exclusive_size = 0;
    }

    fn from_dyn(c: &Context64) -> Self {
        Self {
            x: c.x,
            sp: c.sp,
            pc: c.pc,
            v: c.v,
            nzcv: ((c.pstate >> 28) & 0xF) as u8,
            fpcr: c.fpcr,
            fpsr: c.fpsr,
            tpidr: c.tpidr,
            tpidrro: c.tpidrro,
        }
    }

    fn apply_dyn(&self, c: &mut Context64) {
        c.x = self.x;
        c.sp = self.sp;
        c.pc = self.pc;
        c.v = self.v;
        c.pstate = u32::from(self.nzcv & 0xF) << 28;
        c.fpcr = self.fpcr;
        c.fpsr = self.fpsr;
        c.tpidr = self.tpidr;
        c.tpidrro = self.tpidrro;
    }

    fn dump(&self, out: &mut String, indent: &str) {
        for row in 0..8 {
            let _ = write!(out, "{indent}");
            for i in row * 4..(row * 4 + 4).min(31) {
                let _ = write!(out, "x{:<2}={:#018x}  ", i, self.x[i]);
            }
            if row == 7 {
                let _ = write!(out, "sp ={:#018x}", self.sp);
            }
            let _ = writeln!(out);
        }
        let _ = writeln!(
            out,
            "{indent}pc={:#x} nzcv={} fpcr={:#010x} fpsr={:#010x} tpidr={:#x} tpidrro={:#x}",
            self.pc,
            nzcv_str(self.nzcv),
            self.fpcr,
            self.fpsr,
            self.tpidr,
            self.tpidrro
        );
        for (i, v) in self.v.iter().enumerate() {
            if v[0] != 0 || v[1] != 0 {
                let _ = writeln!(out, "{indent}v{:<2}={:#018x}_{:016x}", i, v[1], v[0]);
            }
        }
    }

    fn write_text(&self, prefix: &str, out: &mut String) {
        let _ = writeln!(out, "{prefix}.pc {:#x}", self.pc);
        let _ = writeln!(out, "{prefix}.sp {:#x}", self.sp);
        for (i, x) in self.x.iter().enumerate() {
            let _ = writeln!(out, "{prefix}.x{i} {x:#x}");
        }
        let _ = writeln!(out, "{prefix}.nzcv {:#x}", self.nzcv);
        let _ = writeln!(out, "{prefix}.fpcr {:#x}", self.fpcr);
        let _ = writeln!(out, "{prefix}.fpsr {:#x}", self.fpsr);
        let _ = writeln!(out, "{prefix}.tpidr {:#x}", self.tpidr);
        let _ = writeln!(out, "{prefix}.tpidrro {:#x}", self.tpidrro);
        for (i, v) in self.v.iter().enumerate() {
            let _ = writeln!(out, "{prefix}.v{i} {:#x} {:#x}", v[0], v[1]);
        }
    }

    /// Applies one `key value...` line of `write_text` output.
    fn read_field(&mut self, field: &str, values: &[u64]) -> Result<(), String> {
        let one = || values.first().copied().ok_or_else(|| format!("{field}: missing value"));
        match field {
            "pc" => self.pc = one()?,
            "sp" => self.sp = one()?,
            "nzcv" => self.nzcv = one()? as u8 & 0xF,
            "fpcr" => self.fpcr = one()? as u32,
            "fpsr" => self.fpsr = one()? as u32,
            "tpidr" => self.tpidr = one()?,
            "tpidrro" => self.tpidrro = one()?,
            _ => {
                if let Some(i) = field.strip_prefix('x').and_then(|i| i.parse::<usize>().ok()) {
                    *self.x.get_mut(i).ok_or_else(|| format!("bad register {field}"))? = one()?;
                } else if let Some(i) = field.strip_prefix('v').and_then(|i| i.parse::<usize>().ok()) {
                    if values.len() != 2 {
                        return Err(format!("{field}: expected lo hi"));
                    }
                    *self.v.get_mut(i).ok_or_else(|| format!("bad register {field}"))? =
                        [values[0], values[1]];
                } else {
                    return Err(format!("unknown field {field}"));
                }
            }
        }
        Ok(())
    }
}

fn nzcv_str(nzcv: u8) -> String {
    let flag = |bit: u8, c: char| if nzcv & bit != 0 { c } else { '-' };
    [flag(8, 'N'), flag(4, 'Z'), flag(2, 'C'), flag(1, 'V')].iter().collect()
}

/// One guest store as either JIT performed it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemWrite {
    pub(crate) addr: u64,
    pub(crate) size: u8,
    pub(crate) old: [u8; 16],
    pub(crate) new: [u8; 16],
    /// Dynarmic single-step index that performed it (`u32::MAX` otherwise).
    pub(crate) step: u32,
}

/// What Rustarmic's lockstep hooks recorded during a run.
#[derive(Debug, Default)]
pub(crate) struct HookLog {
    pub(crate) enabled: bool,
    pub(crate) log_reads: bool,
    pub(crate) writes: Vec<MemWrite>,
    pub(crate) reads: Vec<(u64, u8)>,
    pub(crate) timer: Vec<u64>,
    /// Atomic / exclusive-store callbacks taken (whether or not they wrote).
    pub(crate) atomics: u32,
}

/// What happened to one aligned 8-byte word: the pre-run value of every
/// byte either side stored to (the first `old` seen) and the last value
/// Rustarmic stored.
#[derive(Clone, Copy, Default)]
struct WordEffect {
    old: [u8; 8],
    old_mask: u8,
    new: [u8; 8],
    new_mask: u8,
}

/// Multiplicative hash for word addresses (the default SipHash dominated
/// the per-chunk cost).
#[derive(Default)]
struct WordHasher(u64);

impl std::hash::Hasher for WordHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (v ^ (v >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type Effects = HashMap<u64, WordEffect, std::hash::BuildHasherDefault<WordHasher>>;

/// Net memory effect of Rustarmic's stores `rust` and Dynarmic's `dynarmic`
/// (only the pre-run values of the latter are used).
fn effects(rust: &[MemWrite], dynarmic: &[MemWrite]) -> Effects {
    let mut map = Effects::default();
    for (writes, is_rust) in [(rust, true), (dynarmic, false)] {
        for w in writes {
            let n = w.size as usize;
            let mut i = 0;
            while i < n {
                let word = w.addr.wrapping_add(i as u64) & !7;
                let e = map.entry(word).or_default();
                while i < n && w.addr.wrapping_add(i as u64) & !7 == word {
                    let lane = (w.addr.wrapping_add(i as u64) & 7) as usize;
                    let bit = 1u8 << lane;
                    if e.old_mask & bit == 0 {
                        e.old[lane] = w.old[i];
                        e.old_mask |= bit;
                    }
                    if is_rust {
                        e.new[lane] = w.new[i];
                        e.new_mask |= bit;
                    }
                    i += 1;
                }
            }
        }
    }
    map
}

/// Contiguous runs of the masked lanes of one word.
fn word_runs(word: u64, mask: u8, bytes: &[u8; 8]) -> impl Iterator<Item = (u64, Vec<u8>)> + '_ {
    let mut lane = 0usize;
    std::iter::from_fn(move || {
        while lane < 8 && mask & (1 << lane) == 0 {
            lane += 1;
        }
        if lane >= 8 {
            return None;
        }
        let start = lane;
        while lane < 8 && mask & (1 << lane) != 0 {
            lane += 1;
        }
        Some((word + start as u64, bytes[start..lane].to_vec()))
    })
}

/// Splits sorted byte addresses into contiguous runs.
fn runs<I: IntoIterator<Item = (u64, u8)>>(bytes: I) -> Vec<(u64, Vec<u8>)> {
    let mut out: Vec<(u64, Vec<u8>)> = Vec::new();
    for (addr, byte) in bytes {
        match out.last_mut() {
            Some((start, data)) if start.wrapping_add(data.len() as u64) == addr => data.push(byte),
            _ => out.push((addr, vec![byte])),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The shadow Dynarmic.

/// `struct context64` of dynarmic-sys (`dynarmic.h`).
#[repr(C)]
struct Context64 {
    sp: u64,
    pc: u64,
    x: [u64; 31],
    v: [[u64; 2]; 32],
    fpcr: u32,
    fpsr: u32,
    pstate: u32,
    tpidr: u64,
    tpidrro: u64,
}

const _: () = assert!(std::mem::size_of::<Context64>() == 808);
const _: () = assert!(
    std::mem::size_of::<dynarmic_sys::DynarmicContextInner>() == std::mem::size_of::<*mut c_void>()
);

extern "C" {
    /// `Dynarmic::A64::Jit::Step()`: executes exactly one instruction.
    #[link_name = "_ZN8Dynarmic3A643Jit4StepEv"]
    fn dynarmic_a64_jit_step(jit: *mut c_void) -> u32;
    /// `Dynarmic::A64::Jit::ClearExclusiveState()`.
    #[link_name = "_ZN8Dynarmic3A643Jit19ClearExclusiveStateEv"]
    fn dynarmic_a64_jit_clear_exclusive_state(jit: *mut c_void);
}

/// The `Dynarmic::A64::Jit*` inside dynarmic-sys's `struct dynarmic`
/// (`khash *memory; size_t entries; void **page_table; cb64; jit64`), found
/// the same way `DirectPageTable::discover` finds the page table.
fn discover_jit64(emu: &dynarmic_sys::Dynarmic<'static, ()>) -> Option<*mut c_void> {
    if std::mem::size_of::<dynarmic_sys::Dynarmic<'static, ()>>() != 2 * std::mem::size_of::<usize>() {
        return None;
    }
    let words = unsafe {
        std::ptr::read((emu as *const dynarmic_sys::Dynarmic<'static, ()>).cast::<[usize; 2]>())
    };
    words.into_iter().find_map(|word| {
        if word == 0 || word % std::mem::align_of::<usize>() != 0 {
            return None;
        }
        let fields = word as *const usize;
        let (memory, len, table, cb64, jit64) = unsafe {
            (*fields, *fields.add(1), *fields.add(2), *fields.add(3), *fields.add(4))
        };
        (memory != 0 && len == 1 << (40 - 12) && table != 0 && cb64 != 0 && jit64 != 0 && jit64 % 8 == 0)
            .then_some(jit64 as *mut c_void)
    })
}

#[derive(Default)]
struct ShadowLog {
    writes: Vec<MemWrite>,
    unmapped: Vec<(u64, usize, u64)>,
    svc: Option<u32>,
    step: u32,
}

/// What a Dynarmic run did.
struct DynRun {
    state: GuestState,
    /// Exact instruction count when known (single-stepping, or a bounded
    /// run that stopped early); `None` when a bounded run used its whole
    /// budget (it may have overshot to the end of its block).
    executed: Option<u64>,
    svc: Option<u32>,
    /// Stopped on an instruction Dynarmic raised an exception for (BRK,
    /// UDF, ...).
    exception: bool,
    writes: Vec<MemWrite>,
    unmapped: Vec<(u64, usize, u64)>,
    trace: Vec<u64>,
}

struct Shadow {
    cpu: DynarmicCpu,
    jit64: *mut c_void,
    context: dynarmic_sys::DynarmicContext,
    log: Rc<RefCell<ShadowLog>>,
    map_errors: u32,
}

impl Shadow {
    fn new() -> Result<Self, String> {
        let version = dynarmic_sys::dynarmic_version();
        if version != SUPPORTED_DYNARMIC_VERSION {
            return Err(format!("unsupported dynarmic-sys version {version}"));
        }
        let cpu = DynarmicCpu::new_lockstep_shadow()?;
        let jit64 = discover_jit64(cpu.raw()).ok_or("could not locate the Dynarmic A64 JIT")?;
        let context = cpu.alloc_thread_context();
        let log = Rc::new(RefCell::new(ShadowLog::default()));
        let svc_log = Rc::clone(&log);
        cpu.raw().set_svc_callback(move |dy, swi, _until, _pc| {
            svc_log.borrow_mut().svc = Some(swi);
            let _ = dy.emu_stop();
        });
        let mem_log = Rc::clone(&log);
        cpu.raw().set_unmapped_mem_callback(move |dy, addr, size, value| {
            // In the shadow's watch mode every store arrives here first; a
            // load only does when it misses the mapping.
            let n = size.min(8);
            let mut old = [0u8; 16];
            if n > 0 && dy.mem_read(addr, &mut old[..n]).is_ok() {
                let mut new = [0u8; 16];
                new[..8].copy_from_slice(&value.to_le_bytes());
                let mut log = mem_log.borrow_mut();
                let step = log.step;
                log.writes.push(MemWrite {
                    addr,
                    size: n as u8,
                    old,
                    new,
                    step,
                });
                return false;
            }
            mem_log.borrow_mut().unmapped.push((addr, size, value));
            true
        });
        let shadow = Self {
            cpu,
            jit64,
            context,
            log,
            map_errors: 0,
        };
        if shadow.raw_context().is_null() {
            return Err("dynarmic context allocation failed".into());
        }
        Ok(shadow)
    }

    fn raw_context(&self) -> *mut Context64 {
        unsafe { *(Arc::as_ptr(&self.context) as *const *mut Context64) }
    }

    unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) {
        if let Err(error) = unsafe { self.cpu.map_host(va, len, perm, ptr) } {
            self.map_errors += 1;
            if self.map_errors <= 8 {
                log::warn!("lockstep: shadow map {va:#x}+{len:#x} failed: {error}");
            }
        }
    }

    unsafe fn unmap_host(&mut self, va: u64, len: u64) {
        let _ = unsafe { self.cpu.unmap_host(va, len) };
    }

    fn invalidate(&mut self, va: u64, len: u64) {
        self.cpu.invalidate_range(va, len);
    }

    fn set_state(&mut self, s: &GuestState) {
        s.apply_dyn(unsafe { &mut *self.raw_context() });
        let _ = self.cpu.restore_thread_context(&self.context);
    }

    fn state(&mut self) -> GuestState {
        let _ = self.cpu.save_thread_context(&mut self.context);
        GuestState::from_dyn(unsafe { &*self.raw_context() })
    }

    fn fetch(&self, pc: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.cpu.read_bytes(pc, &mut b).ok()?;
        Some(u32::from_le_bytes(b))
    }

    fn reset_log(&self) {
        let mut log = self.log.borrow_mut();
        log.writes.clear();
        log.unmapped.clear();
        log.svc = None;
        log.step = 0;
    }

    fn take_log(&self) -> ShadowLog {
        std::mem::take(&mut *self.log.borrow_mut())
    }

    /// One bounded `Run` of `count` instructions.
    fn run(&mut self, s: &GuestState, count: u64) -> DynRun {
        self.set_state(s);
        self.reset_log();
        let _ = self.cpu.raw().emu_start_bounded(s.pc, u64::MAX - 16, count);
        let remaining = self.cpu.raw().emu_ticks_remaining();
        let state = self.state();
        let log = self.take_log();
        DynRun {
            state,
            executed: (remaining > 0).then(|| count.saturating_sub(remaining)),
            svc: log.svc,
            exception: false,
            writes: log.writes,
            unmapped: log.unmapped,
            trace: Vec::new(),
        }
    }

    /// Exactly `count` single steps (fewer if an SVC or exception stops
    /// it).  Timer reads get `timer`'s values in order.
    fn step(&mut self, s: &GuestState, count: u64, timer: &[u64], trace: bool) -> DynRun {
        self.set_state(s);
        self.reset_log();
        unsafe { dynarmic_a64_jit_clear_exclusive_state(self.jit64) };
        let mut timer = timer.iter();
        let mut pcs = Vec::new();
        let mut executed = 0u64;
        let mut exception = false;
        let raw = self.cpu.raw();
        for i in 0..count {
            self.log.borrow_mut().step = i.min(u64::from(u32::MAX - 1)) as u32;
            let pc = raw.reg_read_pc().unwrap_or(0);
            let word = self.fetch(pc);
            if trace {
                pcs.push(pc);
            }
            let exclusive = word.and_then(|w| store_exclusive_target(w)).map(|(rn, len)| {
                let base = if rn == 31 {
                    raw.reg_read_sp().unwrap_or(0)
                } else {
                    raw.reg_read(rn as usize).unwrap_or(0)
                };
                let mut before = [0u8; 16];
                let ok = raw.mem_read(base, &mut before[..len]).is_ok();
                (base, len, before, ok)
            });
            unsafe { dynarmic_a64_jit_step(self.jit64) };
            executed = i + 1;
            if let Some((base, len, before, true)) = exclusive {
                let mut after = [0u8; 16];
                if raw.mem_read(base, &mut after[..len]).is_ok() && after[..len] != before[..len] {
                    let mut log = self.log.borrow_mut();
                    let step = log.step;
                    log.writes.push(MemWrite {
                        addr: base,
                        size: len as u8,
                        old: before,
                        new: after,
                        step,
                    });
                }
            }
            if let Some(rt) = word.and_then(timer_read_rt) {
                if let Some(&value) = timer.next() {
                    let _ = raw.reg_write_raw(rt as usize, value);
                }
            }
            if self.log.borrow().svc.is_some() {
                break;
            }
            let after = raw.reg_read_pc().unwrap_or(0);
            if after == pc && !word.is_some_and(is_branch) {
                exception = true;
                break;
            }
        }
        let state = self.state();
        let log = self.take_log();
        DynRun {
            state,
            executed: Some(executed),
            svc: log.svc,
            exception,
            writes: log.writes,
            unmapped: log.unmapped,
            trace: pcs,
        }
    }

    /// Steps from `s` until the instruction at `target` has executed.
    fn step_past(&mut self, s: &GuestState, target: u64, max: u64) -> DynRun {
        self.set_state(s);
        self.reset_log();
        let raw = self.cpu.raw();
        let mut executed = 0;
        for i in 0..max {
            self.log.borrow_mut().step = i as u32;
            let pc = raw.reg_read_pc().unwrap_or(0);
            unsafe { dynarmic_a64_jit_step(self.jit64) };
            executed = i + 1;
            if pc == target || self.log.borrow().svc.is_some() {
                break;
            }
        }
        let state = self.state();
        let log = self.take_log();
        DynRun {
            state,
            executed: Some(executed),
            svc: log.svc,
            exception: false,
            writes: log.writes,
            unmapped: log.unmapped,
            trace: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction helpers.

/// `(Rn, bytes)` of a store-exclusive (STXR/STLXR/STXP/STLXP), whose store
/// dynarmic-sys performs without the write callback.
fn store_exclusive_target(w: u32) -> Option<(u32, usize)> {
    if w & 0x3F00_0000 != 0x0800_0000 || (w >> 22) & 1 != 0 || (w >> 23) & 1 != 0 {
        return None;
    }
    let size = 1usize << (w >> 30);
    let pair = (w >> 21) & 1 != 0;
    Some(((w >> 5) & 31, if pair { size * 2 } else { size }))
}

/// `(op0, op1, CRn, CRm, op2, Rt)` of an MRS.
fn mrs_fields(w: u32) -> Option<(u32, u32, u32, u32, u32, u32)> {
    (w & 0xFFF0_0000 == 0xD530_0000).then(|| {
        (
            (w >> 19) & 3,
            (w >> 16) & 7,
            (w >> 12) & 15,
            (w >> 8) & 15,
            (w >> 5) & 7,
            w & 31,
        )
    })
}

/// `Rt` of an `MRS Xt, CNTPCT_EL0 / CNTVCT_EL0`.
fn timer_read_rt(w: u32) -> Option<u32> {
    match mrs_fields(w)? {
        (3, 3, 14, 0, 1 | 2, rt) if rt != 31 => Some(rt),
        _ => None,
    }
}

fn is_branch(w: u32) -> bool {
    w & 0x7C00_0000 == 0x1400_0000
        || w & 0xFF00_0010 == 0x5400_0000
        || w & 0x7E00_0000 == 0x3400_0000
        || w & 0x7E00_0000 == 0x3600_0000
        || w & 0xFE00_0000 == 0xD600_0000
}

/// Scalar FP / Advanced SIMD data processing.
fn is_fp_simd_dp(w: u32) -> bool {
    (w >> 25) & 7 == 7
}

pub(crate) fn disasm(pc: u64, word: u32) -> String {
    match disarm64::decoder::decode(word) {
        Some(op) => {
            let mut s = String::new();
            match disarm64::format_insn::format_insn_pc(pc, &mut s, &op) {
                Ok(()) => s,
                Err(_) => format!("{op}"),
            }
        }
        None => "<undecodable>".to_string(),
    }
}

fn is_nan32(v: u32) -> bool {
    v & 0x7F80_0000 == 0x7F80_0000 && v & 0x007F_FFFF != 0
}

fn is_nan64(v: u64) -> bool {
    v & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000 && v & 0x000F_FFFF_FFFF_FFFF != 0
}

/// Whether two differing 128-bit values only differ in NaN encodings,
/// read either as four f32 lanes or as two f64 lanes.
fn nan_only_difference(a: [u64; 2], b: [u64; 2]) -> bool {
    let lanes64 = (0..2).all(|i| a[i] == b[i] || (is_nan64(a[i]) && is_nan64(b[i])));
    let lanes32 = (0..4).all(|i| {
        let (x, y) = ((a[i / 2] >> (32 * (i % 2))) as u32, (b[i / 2] >> (32 * (i % 2))) as u32);
        x == y || (is_nan32(x) && is_nan32(y))
    });
    lanes64 || lanes32
}

// ---------------------------------------------------------------------------
// Comparison.

#[derive(Clone, Debug, PartialEq, Eq)]
enum Field {
    X(usize, u64, u64),
    Sp(u64, u64),
    Pc(u64, u64),
    Nzcv(u8, u8),
    Fpcr(u32, u32),
    Tpidr(u64, u64),
    Tpidrro(u64, u64),
    V(usize, [u64; 2], [u64; 2]),
}

impl Field {
    fn describe(&self) -> String {
        match self {
            Field::X(i, r, d) => format!("x{i}: rustarmic={r:#018x} dynarmic={d:#018x} (xor {:#x})", r ^ d),
            Field::Sp(r, d) => format!("sp: rustarmic={r:#x} dynarmic={d:#x}"),
            Field::Pc(r, d) => format!("pc: rustarmic={r:#x} dynarmic={d:#x}"),
            Field::Nzcv(r, d) => format!("nzcv: rustarmic={} dynarmic={}", nzcv_str(*r), nzcv_str(*d)),
            Field::Fpcr(r, d) => format!("fpcr: rustarmic={r:#010x} dynarmic={d:#010x}"),
            Field::Tpidr(r, d) => format!("tpidr_el0: rustarmic={r:#x} dynarmic={d:#x}"),
            Field::Tpidrro(r, d) => format!("tpidrro_el0: rustarmic={r:#x} dynarmic={d:#x}"),
            Field::V(i, r, d) => format!(
                "v{i}: rustarmic={:#018x}_{:016x} dynarmic={:#018x}_{:016x}",
                r[1], r[0], d[1], d[0]
            ),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Diff {
    fields: Vec<Field>,
    /// Rustarmic, Dynarmic FPSR when they differ.
    fpsr: Option<(u32, u32)>,
    stop: Option<(String, String)>,
    retired: Option<(u64, u64)>,
    /// Bytes whose value after the run differs: (address, rustarmic, dynarmic).
    mem: Vec<(u64, u8, u8)>,
}

impl Diff {
    fn is_empty(&self) -> bool {
        self.ignoring_fpsr_empty() && self.fpsr.is_none()
    }

    fn ignoring_fpsr_empty(&self) -> bool {
        self.fields.is_empty() && self.stop.is_none() && self.retired.is_none() && self.mem.is_empty()
    }

    fn only_fpsr(&self) -> bool {
        self.ignoring_fpsr_empty() && self.fpsr.is_some()
    }

    fn describe(&self, out: &mut String, indent: &str) {
        for f in &self.fields {
            let _ = writeln!(out, "{indent}{}", f.describe());
        }
        if let Some((r, d)) = self.fpsr {
            let _ = writeln!(out, "{indent}fpsr: rustarmic={r:#010x} dynarmic={d:#010x}");
        }
        if let Some((r, d)) = &self.stop {
            let _ = writeln!(out, "{indent}stop: rustarmic={r} dynarmic={d}");
        }
        if let Some((r, d)) = self.retired {
            let _ = writeln!(out, "{indent}retired: rustarmic={r} dynarmic={d}");
        }
        for (addr, rbytes) in runs(self.mem.iter().map(|&(a, r, _)| (a, r))) {
            let dbytes: Vec<u8> = (0..rbytes.len())
                .map(|i| {
                    let a = addr + i as u64;
                    self.mem.iter().find(|m| m.0 == a).map(|m| m.2).unwrap_or(0)
                })
                .collect();
            let _ = writeln!(
                out,
                "{indent}memory {addr:#x}+{}: rustarmic={} dynarmic={}",
                rbytes.len(),
                hex(&rbytes),
                hex(&dbytes)
            );
        }
    }

    fn summary(&self) -> String {
        let mut parts: Vec<String> = self.fields.iter().take(3).map(Field::describe).collect();
        if let Some((r, d)) = self.fpsr {
            parts.push(format!("fpsr {r:#x}/{d:#x}"));
        }
        if let Some((r, d)) = &self.stop {
            parts.push(format!("stop {r}/{d}"));
        }
        if let Some((r, d)) = self.retired {
            parts.push(format!("retired {r}/{d}"));
        }
        if !self.mem.is_empty() {
            parts.push(format!("{} memory bytes (first at {:#x})", self.mem.len(), self.mem[0].0));
        }
        parts.join("; ")
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn compare_states(r: &GuestState, d: &GuestState, diff: &mut Diff) {
    for i in 0..31 {
        if r.x[i] != d.x[i] {
            diff.fields.push(Field::X(i, r.x[i], d.x[i]));
        }
    }
    if r.sp != d.sp {
        diff.fields.push(Field::Sp(r.sp, d.sp));
    }
    if r.pc != d.pc {
        diff.fields.push(Field::Pc(r.pc, d.pc));
    }
    if r.nzcv != d.nzcv {
        diff.fields.push(Field::Nzcv(r.nzcv, d.nzcv));
    }
    if r.fpcr != d.fpcr {
        diff.fields.push(Field::Fpcr(r.fpcr, d.fpcr));
    }
    if r.tpidr != d.tpidr {
        diff.fields.push(Field::Tpidr(r.tpidr, d.tpidr));
    }
    if r.tpidrro != d.tpidrro {
        diff.fields.push(Field::Tpidrro(r.tpidrro, d.tpidrro));
    }
    for i in 0..32 {
        if r.v[i] != d.v[i] {
            diff.fields.push(Field::V(i, r.v[i], d.v[i]));
        }
    }
    if r.fpsr != d.fpsr {
        diff.fpsr = Some((r.fpsr, d.fpsr));
    }
}

fn rust_stop_label(stop: &StopReason) -> String {
    match stop {
        StopReason::BudgetExhausted | StopReason::Halted => "none".into(),
        StopReason::Svc(imm) => format!("svc #{imm:#x}"),
        StopReason::Brk(imm) => format!("brk #{imm:#x}"),
        StopReason::Hvc(imm) => format!("hvc #{imm:#x}"),
        StopReason::Yield => "yield".into(),
        StopReason::Wait => "wait".into(),
        StopReason::MemoryFault(f) => format!("fault {:#x}", f.address),
        StopReason::Unsupported(info) => format!("unsupported {:#010x}", info.opcode),
    }
}

fn dyn_stop_label(d: &DynRun) -> String {
    match (d.svc, d.exception) {
        (Some(imm), _) => format!("svc #{imm:#x}"),
        (None, true) => "exception".into(),
        (None, false) => "none".into(),
    }
}

fn stops_agree(r: &StopReason, d: &DynRun) -> bool {
    match r {
        StopReason::Svc(imm) => d.svc == Some(*imm),
        StopReason::Brk(_) | StopReason::Hvc(_) => d.svc.is_none() && d.exception,
        _ => d.svc.is_none() && !d.exception,
    }
}

// ---------------------------------------------------------------------------
// Classification.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Class {
    /// Rustarmic computed something Dynarmic did not.
    Divergence,
    /// A NaN result whose encoding (payload / sign / default NaN) differs.
    FpNan,
    /// An FP result that differs while FPCR has FZ/DN/RMode/AHP/FZ16 set,
    /// which Rustarmic's inline FP ops ignore.
    FpcrMode,
    /// Only the FPSR cumulative flags differ.
    FpStatus,
    /// An `MRS` of an ID / configuration register returned another value.
    SystemRegister,
    /// A counter read (`MRS CNTPCT/CNTVCT`) differs (normally masked).
    Timer,
    /// Dynarmic raised an exception for an instruction Rustarmic executes
    /// (not implemented in this dynarmic build, e.g. SQDMLSL).
    DynarmicUnsupported,
    /// Rustarmic cannot translate an instruction Dynarmic runs.
    Unsupported,
    /// Rustarmic faulted on an address the kernel has mapped.
    MappingSync,
    /// A guest memory fault both JITs agree on up to the faulting access.
    GuestFault,
    /// A mismatch that did not reproduce from the saved state.
    NonReproducible,
}

impl Class {
    fn label(self) -> &'static str {
        match self {
            Class::Divergence => "divergence",
            Class::FpNan => "fp-nan-encoding",
            Class::FpcrMode => "fpcr-mode",
            Class::FpStatus => "fpsr-flags",
            Class::SystemRegister => "system-register",
            Class::Timer => "timer",
            Class::DynarmicUnsupported => "dynarmic-unsupported",
            Class::Unsupported => "unsupported",
            Class::MappingSync => "mapping-sync",
            Class::GuestFault => "guest-fault",
            Class::NonReproducible => "non-reproducible",
        }
    }

    fn explain(self) -> &'static str {
        match self {
            Class::Divergence => "Rustarmic's result differs from Dynarmic's for this instruction: a Rustarmic miscalculation.",
            Class::FpNan => "Both results are NaN but encoded differently (Rustarmic inline FP ops produce the x86 default NaN; ARM's is 0x7FC00000 / 0x7FF8000000000000). Benign unless a later comparison or bit-cast of the NaN changes control flow.",
            Class::FpcrMode => "FPCR has FZ/DN/RMode/AHP/FZ16 set and Rustarmic's inline FP ops ignore FPCR modes.",
            Class::FpStatus => "Only FPSR cumulative exception flags differ (Rustarmic's inline FP ops do not accumulate them).",
            Class::SystemRegister => "An ID/configuration system register reads differently on the two JITs (a configuration constant, not arithmetic).",
            Class::Timer => "A counter read differs; the timer is host time on both JITs.",
            Class::DynarmicUnsupported => "Dynarmic raised an exception for this instruction while Rustarmic executed it. Usually this dynarmic build does not implement it (its decoder lacks e.g. SQDMLAL/SQDMLSL/SQRSHL/UQRSHRN); if the encoding is really unallocated, Rustarmic should fault instead. Rustarmic's result is kept but cannot be verified here: replay the repro against Unicorn (rustarmic's lockstep_replay test).",
            Class::Unsupported => "Rustarmic cannot translate this instruction; Dynarmic can.",
            Class::MappingSync => "Rustarmic faulted on an address the kernel has mapped: this core's map lagged the kernel (a mapping-sync issue, not a JIT divergence).",
            Class::GuestFault => "A guest memory fault; both JITs agree on the state up to the faulting access.",
            Class::NonReproducible => "The mismatch did not reproduce from the saved state (likely another host thread wrote guest memory during the chunk).",
        }
    }

    fn benign(self) -> bool {
        !matches!(self, Class::Divergence | Class::Unsupported)
    }
}

/// Classifies the instruction `word` whose execution produced `diff` from
/// a state with FPCR `fpcr`.
fn classify(word: u32, fpcr: u32, diff: &Diff) -> Class {
    if let Some((op0, op1, crn, crm, op2, _)) = mrs_fields(word) {
        if (op0, op1, crn, crm) == (3, 3, 14, 0) && matches!(op2, 1 | 2) {
            return Class::Timer;
        }
        // Only the destination register differs: a configuration value.
        let rt = (word & 31) as usize;
        if diff.fields.iter().all(|f| matches!(f, Field::X(i, ..) if *i == rt))
            && diff.mem.is_empty()
            && diff.stop.is_none()
            && !(op0 == 3 && op1 == 3 && crn == 4)
        {
            return Class::SystemRegister;
        }
    }
    if diff.stop.as_ref().is_some_and(|(r, d)| r == "none" && d == "exception") {
        return Class::DynarmicUnsupported;
    }
    if is_fp_simd_dp(word) && diff.stop.is_none() && diff.retired.is_none() && diff.mem.is_empty() {
        if diff.fields.is_empty() {
            if diff.fpsr.is_some() {
                return Class::FpStatus;
            }
        } else {
            let nan_only = diff.fields.iter().all(|f| match f {
                Field::V(_, r, d) => nan_only_difference(*r, *d),
                Field::X(_, r, d) => {
                    (is_nan64(*r) && is_nan64(*d)) || (is_nan32(*r as u32) && is_nan32(*d as u32) && r >> 32 == d >> 32)
                }
                _ => false,
            });
            if nan_only {
                return Class::FpNan;
            }
            if fpcr & FPCR_MODES != 0 {
                return Class::FpcrMode;
            }
        }
    }
    Class::Divergence
}

/// Whether a NaN involved in `diff` is ARM's default NaN on Dynarmic's side
/// and something else on Rustarmic's (the documented x86-default-NaN case).
fn default_nan_note(diff: &Diff) -> Option<String> {
    for f in &diff.fields {
        if let Field::V(i, r, d) = f {
            for lane in 0..4 {
                let (rv, dv) = ((r[lane / 2] >> (32 * (lane % 2))) as u32, (d[lane / 2] >> (32 * (lane % 2))) as u32);
                if dv == ARM_DEFAULT_NAN32 && rv != dv && is_nan32(rv) {
                    return Some(format!("v{i} lane {lane}: Rustarmic {rv:#010x} vs ARM default NaN {dv:#010x}"));
                }
            }
            for lane in 0..2 {
                if d[lane] == ARM_DEFAULT_NAN64 && r[lane] != d[lane] && is_nan64(r[lane]) {
                    return Some(format!(
                        "v{i} lane {lane}: Rustarmic {:#018x} vs ARM default NaN {:#018x}",
                        r[lane], d[lane]
                    ));
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The driver.

/// What a Rustarmic run did.
struct RustRun {
    state: GuestState,
    stop: StopReason,
    retired: u64,
    writes: Vec<MemWrite>,
    reads: Vec<(u64, u8)>,
    timer: Vec<u64>,
    fetches: Vec<u64>,
    atomics: u32,
}

/// The located first diverging instruction.
struct Located {
    /// 1-based index of the instruction within the chunk.
    k: u64,
    pc: u64,
    word: u32,
    before: GuestState,
    rust_after: GuestState,
    dyn_after: DynRun,
    diff: Diff,
    /// The instruction alone (in a one-instruction block) matches Dynarmic:
    /// only the block it was compiled in diverges.
    block_context: bool,
    trace: Vec<u64>,
}

#[derive(Default, Debug, Clone)]
pub(crate) struct Stats {
    pub(crate) chunks: u64,
    pub(crate) checked: u64,
    pub(crate) retired_checked: u64,
    pub(crate) fast_hits: u64,
    pub(crate) stepped: u64,
    pub(crate) nonrepro: u64,
    pub(crate) by_class: HashMap<&'static str, u64>,
}

/// One reported event, kept for tests and the stats line.
#[derive(Clone, Debug)]
pub(crate) struct Record {
    pub(crate) class: Class,
    pub(crate) pc: u64,
    pub(crate) word: u32,
    pub(crate) k: u64,
    pub(crate) chunk_pc: u64,
    pub(crate) block_context: bool,
    pub(crate) report: Option<PathBuf>,
    pub(crate) repro: Option<PathBuf>,
    pub(crate) repro_verified: Option<bool>,
    pub(crate) text: String,
}

static REPORTS_WRITTEN: AtomicU32 = AtomicU32::new(0);
static EVENT_SEQ: AtomicU32 = AtomicU32::new(0);

pub(crate) struct Lockstep {
    cfg: Config,
    shadow: Shadow,
    stalled: bool,
    retired_seen: u64,
    pub(crate) stats: Stats,
    pub(crate) records: Vec<Record>,
    reported_pcs: HashMap<(Class, u64), u32>,
    last_stats: Instant,
    replay_rig: Option<Box<ReplayRig>>,
    #[cfg(test)]
    pub(crate) replay_injection: Option<rustarmic::frontend::fault_injection::FaultInjection>,
}

impl Drop for Lockstep {
    /// One summary of everything this core reported, at the end of emulation.
    fn drop(&mut self) {
        if self.stats.checked == 0 {
            return;
        }
        let mut classes: Vec<_> = self.stats.by_class.iter().collect();
        classes.sort();
        log::warn!(
            "lockstep summary: {} chunks checked ({} instructions, {} by one Dynarmic run), events {:?}",
            self.stats.checked,
            self.stats.retired_checked,
            self.stats.fast_hits,
            classes
        );
        for r in &self.records {
            let where_ = match (&r.report, r.text.lines().next()) {
                (Some(path), _) => path.display().to_string(),
                (None, Some(first)) => first.to_string(),
                (None, None) => String::new(),
            };
            log::warn!(
                "  [{}] pc={:#x} {:08x} {} (#{} of chunk at {:#x}{}) repro={} verified={:?} {}",
                r.class.label(),
                r.pc,
                r.word,
                disasm(r.pc, r.word),
                r.k,
                r.chunk_pc,
                if r.block_context { ", block context" } else { "" },
                r.repro.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "-".into()),
                r.repro_verified,
                where_
            );
        }
    }
}

/// How a chunk ended.
enum Chunk {
    /// Keep going with the next chunk.
    Continue(u64),
    /// Return this event to the scheduler.
    Event(CpuEvent, u64),
}

impl Lockstep {
    pub(crate) fn new(cfg: Config) -> Result<Self, String> {
        let shadow = Shadow::new()?;
        Ok(Self {
            cfg,
            shadow,
            stalled: false,
            retired_seen: 0,
            stats: Stats::default(),
            records: Vec::new(),
            reported_pcs: HashMap::new(),
            last_stats: Instant::now(),
            replay_rig: None,
            #[cfg(test)]
            replay_injection: None,
        })
    }

    pub(crate) unsafe fn map_host(&mut self, va: u64, len: u64, perm: Perm, ptr: *mut u8) {
        unsafe { self.shadow.map_host(va, len, perm, ptr) };
    }

    pub(crate) unsafe fn unmap_host(&mut self, va: u64, len: u64) {
        unsafe { self.shadow.unmap_host(va, len) };
    }

    pub(crate) fn invalidate_range(&mut self, va: u64, len: u64) {
        self.shadow.invalidate(va, len);
    }

    /// `RustarmicCpu::run_with_count` with every chunk checked.
    pub(crate) fn run(&mut self, cpu: &mut RustarmicCpu, max: u64) -> (CpuEvent, u64) {
        if self.stalled {
            return (CpuEvent::Stalled, 0);
        }
        for (va, len) in cpu.ls_take_invalidations() {
            self.shadow.invalidate(va, len);
        }
        if let Some(event) = cpu.ls_take_event() {
            return (event, 0);
        }
        let mut total = 0u64;
        loop {
            if cpu.ls_halt_requested() {
                // As `run_bounded` would: the halt is consumed.
                cpu.ls_clear_halt();
                cpu.ls_ctx_mut().should_halt = 0;
                return (cpu.ls_take_event().unwrap_or(CpuEvent::Running), total);
            }
            let budget = max.saturating_sub(total).min(self.cfg.slice);
            if budget == 0 {
                return (CpuEvent::Running, total);
            }
            cpu.ls_peek_begin();
            let chunk = self.chunk(cpu, budget);
            cpu.ls_peek_end();
            self.maybe_log_stats();
            match chunk {
                Chunk::Continue(n) => total += n,
                Chunk::Event(event, n) => return (event, total + n),
            }
        }
    }

    fn maybe_log_stats(&mut self) {
        if self.last_stats.elapsed().as_secs() < 10 {
            return;
        }
        self.last_stats = Instant::now();
        let mut classes: Vec<_> = self.stats.by_class.iter().collect();
        classes.sort();
        log::info!(
            "lockstep: chunks={} checked={} retired_checked={} dyn_run_hits={} stepped={} non_repro={} events={:?}",
            self.stats.chunks,
            self.stats.checked,
            self.stats.retired_checked,
            self.stats.fast_hits,
            self.stats.stepped,
            self.stats.nonrepro,
            classes
        );
    }

    /// Maps a Rustarmic stop to the event `run_with_count` reports, with the
    /// same logging; `None` means the chunk ran to its budget.
    fn event_for(&self, cpu: &RustarmicCpu, stop: &StopReason, retired: u64) -> Option<CpuEvent> {
        Some(match stop {
            StopReason::BudgetExhausted => return None,
            StopReason::Unsupported(info) => {
                log::warn!(
                    "rustarmic unsupported stop: pc={:#x} opcode={:#010x} class={} retired={}",
                    info.pc,
                    info.opcode,
                    info.decoded_class,
                    retired,
                );
                CpuEvent::Stalled
            }
            StopReason::MemoryFault(fault) => {
                log::warn!(
                    "rustarmic memory fault: pc={:#x} address={:#x} size={} access={:?} cause={:?} retired={}",
                    fault.pc,
                    fault.address,
                    fault.size,
                    fault.access,
                    fault.cause,
                    retired,
                );
                CpuEvent::Exception(0x0E)
            }
            StopReason::Halted => cpu.ls_take_event().unwrap_or(CpuEvent::Running),
            StopReason::Svc(imm) => CpuEvent::Svc(*imm as u16),
            StopReason::Brk(imm) => CpuEvent::Exception(0x100 | imm),
            StopReason::Hvc(imm) => CpuEvent::Exception(0x200 | imm),
            StopReason::Yield | StopReason::Wait => CpuEvent::Interrupted,
        })
    }

    fn finish(&self, cpu: &RustarmicCpu, stop: &StopReason, retired: u64) -> Chunk {
        match self.event_for(cpu, stop, retired) {
            Some(event) => Chunk::Event(event, retired),
            None => Chunk::Continue(retired),
        }
    }

    fn should_check(&self) -> bool {
        self.retired_seen >= self.cfg.start_after && self.stats.chunks % self.cfg.sample == 0
    }

    fn chunk(&mut self, cpu: &mut RustarmicCpu, budget: u64) -> Chunk {
        let s = GuestState::from_rust(cpu.ls_ctx());
        s.apply_rust(cpu.ls_ctx_mut());
        let check = self.should_check();
        self.stats.chunks += 1;
        let r = match self.rust_primary(cpu, budget, check) {
            Ok(r) => r,
            Err(error) => {
                log::warn!("rustarmic Jit::run error: {error}");
                return Chunk::Event(CpuEvent::Stalled, 0);
            }
        };
        cpu.ls_clear_halt();
        self.retired_seen = self.retired_seen.saturating_add(r.retired);
        if !check {
            return self.finish(cpu, &r.stop, r.retired);
        }
        self.stats.checked += 1;
        self.stats.retired_checked += r.retired;
        if let StopReason::MemoryFault(fault) = &r.stop {
            let fault = *fault;
            return self.on_fault(cpu, &s, budget, r, fault);
        }
        if r.retired == 0 {
            if let StopReason::Unsupported(_) = &r.stop {
                return self.on_unsupported(cpu, &s, r);
            }
            return self.finish(cpu, &r.stop, 0);
        }
        self.rollback(cpu, &r.writes);
        let (d, diff) = self.dyn_exact(cpu, &s, &r);
        if diff.is_empty() || diff.only_fpsr() {
            if let Some((_, dyn_fpsr)) = diff.fpsr {
                self.count(Class::FpStatus);
                cpu.ls_ctx_mut().fpsr = dyn_fpsr;
            }
            if let StopReason::Unsupported(_) = &r.stop {
                return self.on_unsupported(cpu, &s, r);
            }
            return self.finish(cpu, &r.stop, r.retired);
        }
        self.on_mismatch(cpu, &s, r, d)
    }

    fn count(&mut self, class: Class) {
        *self.stats.by_class.entry(class.label()).or_default() += 1;
    }

    // -- Rustarmic runs ----------------------------------------------------

    /// The chunk's real run: one `run_bounded` like `run_with_count`.
    fn rust_primary(&mut self, cpu: &mut RustarmicCpu, budget: u64, log: bool) -> Result<RustRun, String> {
        for (va, len) in cpu.ls_take_invalidations() {
            self.shadow.invalidate(va, len);
        }
        if log {
            cpu.ls_log_begin(false);
        }
        let result = cpu.ls_run(budget, None);
        let hook_log = cpu.ls_log_end();
        let out = result.map_err(|e| format!("{e:?}"))?;
        Ok(RustRun {
            state: GuestState::from_rust(cpu.ls_ctx()),
            stop: out.stop,
            retired: out.retired,
            writes: hook_log.writes,
            reads: hook_log.reads,
            timer: hook_log.timer,
            atomics: hook_log.atomics,
            fetches: Vec::new(),
        })
    }

    /// Exactly `count` instructions from `start` (or from the current state
    /// when `None`), continuing through self-modifying-code and external
    /// halts.  `capture` also records loads and instruction fetches.
    fn rust_exact(
        &mut self,
        cpu: &mut RustarmicCpu,
        start: Option<&GuestState>,
        count: u64,
        capture: bool,
    ) -> Result<RustRun, String> {
        if let Some(s) = start {
            s.apply_rust(cpu.ls_ctx_mut());
        }
        for (va, len) in cpu.ls_take_invalidations() {
            self.shadow.invalidate(va, len);
        }
        cpu.ls_log_begin(capture);
        let mut fetches = Vec::new();
        let mut retired = 0u64;
        let mut stop = StopReason::BudgetExhausted;
        let mut idle = 0;
        while retired < count {
            let out = match cpu.ls_run(count - retired, capture.then_some(&mut fetches)) {
                Ok(out) => out,
                Err(e) => {
                    cpu.ls_log_end();
                    return Err(format!("{e:?}"));
                }
            };
            retired += out.retired;
            stop = out.stop;
            match stop {
                StopReason::BudgetExhausted => {}
                StopReason::Halted => {
                    for (va, len) in cpu.ls_take_invalidations() {
                        self.shadow.invalidate(va, len);
                    }
                    if out.retired == 0 {
                        idle += 1;
                        if idle > 4 {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
        if retired >= count && matches!(stop, StopReason::Halted) {
            stop = StopReason::BudgetExhausted;
        }
        let log = cpu.ls_log_end();
        Ok(RustRun {
            state: GuestState::from_rust(cpu.ls_ctx()),
            stop,
            retired,
            writes: log.writes,
            reads: log.reads,
            timer: log.timer,
            atomics: log.atomics,
            fetches,
        })
    }

    // -- memory --------------------------------------------------------------

    /// Undoes `writes` (newest first).
    fn rollback(&mut self, cpu: &mut RustarmicCpu, writes: &[MemWrite]) {
        for w in writes.iter().rev() {
            let n = w.size as usize;
            if cpu.ls_write(w.addr, &w.old[..n]) == Some(true) {
                self.shadow.invalidate(w.addr, n as u64);
            }
        }
    }

    /// Restores every byte either run wrote to its pre-chunk value.
    fn restore(&mut self, cpu: &mut RustarmicCpu, a: &[MemWrite], b: &[MemWrite]) {
        for (word, e) in effects(a, b) {
            for (addr, bytes) in word_runs(word, e.old_mask, &e.old) {
                if cpu.ls_write(addr, &bytes) == Some(true) {
                    self.shadow.invalidate(addr, bytes.len() as u64);
                }
            }
        }
    }

    /// After a Dynarmic run: code it stored to must be retranslated by both.
    fn after_dyn(&mut self, cpu: &mut RustarmicCpu, d: &DynRun) {
        for w in &d.writes {
            if cpu.ls_exec_overlaps(w.addr, u64::from(w.size)) {
                self.shadow.invalidate(w.addr, u64::from(w.size));
                cpu.ls_invalidate_jit(w.addr, u64::from(w.size));
            }
        }
    }

    /// Compares a Rustarmic run with the Dynarmic run whose stores are in
    /// memory now.
    fn compare(&self, cpu: &RustarmicCpu, r: &RustRun, d: &DynRun, mask_fpsr: bool) -> Diff {
        let mut diff = Diff::default();
        compare_states(&r.state, &d.state, &mut diff);
        if mask_fpsr {
            diff.fpsr = None;
        }
        if !stops_agree(&r.stop, d) {
            diff.stop = Some((rust_stop_label(&r.stop), dyn_stop_label(d)));
        }
        if let Some(executed) = d.executed {
            if executed != r.retired {
                diff.retired = Some((r.retired, executed));
            }
        }
        // Net memory effect over every byte either side stored to: what
        // Rustarmic left there (or the pre-run value) against what is in
        // memory now.
        for (word, e) in effects(&r.writes, &d.writes) {
            let mut actual = [0u8; 8];
            if !cpu.ls_read(word, &mut actual) {
                for lane in 0..8 {
                    if e.old_mask & (1 << lane) != 0 {
                        cpu.ls_read(word + lane as u64, &mut actual[lane..lane + 1]);
                    }
                }
            }
            for lane in 0..8 {
                let bit = 1u8 << lane;
                if e.old_mask & bit == 0 {
                    continue;
                }
                let expected = if e.new_mask & bit != 0 { e.new[lane] } else { e.old[lane] };
                if expected != actual[lane] {
                    diff.mem.push((word + lane as u64, expected, actual[lane]));
                }
            }
        }
        diff.mem.sort_unstable_by_key(|m| m.0);
        diff
    }

    /// Runs Dynarmic for exactly Rustarmic's instruction count from `s`
    /// (memory must be at the chunk's start).  Leaves Dynarmic's stores in
    /// memory.
    fn dyn_exact(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, r: &RustRun) -> (DynRun, Diff) {
        // One bounded run can overshoot Rustarmic's stop point to the end of
        // Dynarmic's block.  That is harmless (it is detected and the
        // chunk is single-stepped) except for store-exclusives, which
        // dynarmic-sys performs without the write callback and so could not
        // be rolled back; timer reads need the per-step fixup.
        if self.cfg.dyn_fast_run
            && r.timer.is_empty()
            && r.atomics == 0
            && !self.exclusive_store_ahead(cpu, r.state.pc)
        {
            let d = self.shadow.run(s, r.retired);
            self.after_dyn(cpu, &d);
            let diff = self.compare(cpu, r, &d, false);
            if diff.is_empty() || diff.only_fpsr() {
                self.stats.fast_hits += 1;
                return (d, diff);
            }
            self.restore(cpu, &r.writes, &d.writes);
        }
        self.stats.stepped += 1;
        let d = self.shadow.step(s, r.retired, &r.timer, false);
        self.after_dyn(cpu, &d);
        let diff = self.compare(cpu, r, &d, false);
        (d, diff)
    }

    /// Whether the straight-line code from `pc` up to the next branch (where
    /// a Dynarmic block that overshoots `pc` must end) has a store-exclusive.
    fn exclusive_store_ahead(&self, cpu: &RustarmicCpu, pc: u64) -> bool {
        const SCAN: usize = 128;
        let mut bytes = [0u8; SCAN * 4];
        let readable = cpu.ls_read(pc, &mut bytes);
        for i in 0..SCAN {
            let word = if readable {
                u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            } else {
                let mut b = [0u8; 4];
                if !cpu.ls_read(pc.wrapping_add(4 * i as u64), &mut b) {
                    return false;
                }
                u32::from_le_bytes(b)
            };
            if store_exclusive_target(word).is_some() {
                return true;
            }
            // Branches and exception-generating instructions end the block.
            if is_branch(word) || word & 0xFF00_0000 == 0xD400_0000 {
                return false;
            }
        }
        true
    }

    // -- mismatch handling ---------------------------------------------------

    fn on_mismatch(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, r: RustRun, d: DynRun) -> Chunk {
        let first_summary = self.compare(cpu, &r, &d, false).summary();
        self.restore(cpu, &r.writes, &d.writes);
        let count = r.retired;
        // Reproduce from the saved state.
        let r2 = match self.rust_exact(cpu, Some(s), count, false) {
            Ok(r2) => r2,
            Err(error) => {
                log::warn!("lockstep: re-run failed: {error}");
                return Chunk::Event(CpuEvent::Stalled, 0);
            }
        };
        self.rollback(cpu, &r2.writes);
        let d2 = self.shadow.step(s, count, &r2.timer, false);
        self.after_dyn(cpu, &d2);
        let diff2 = self.compare(cpu, &r2, &d2, true);
        if diff2.ignoring_fpsr_empty() {
            // Memory holds Dynarmic's stores, equal to Rustarmic's.
            self.stats.nonrepro += 1;
            self.count(Class::NonReproducible);
            if self.stats.nonrepro <= 16 || self.stats.nonrepro % 1000 == 0 {
                let text = format!(
                    "lockstep: [{}] chunk at pc={:#x} ({} instructions) disagreed ({}) but agrees when re-run from the saved state (count {})",
                    Class::NonReproducible.label(),
                    s.pc,
                    count,
                    first_summary,
                    self.stats.nonrepro
                );
                log::warn!("{text}");
                if self.stats.nonrepro <= 16 {
                    self.records.push(Record {
                        class: Class::NonReproducible,
                        pc: s.pc,
                        word: 0,
                        k: 0,
                        chunk_pc: s.pc,
                        block_context: false,
                        report: None,
                        repro: None,
                        repro_verified: None,
                        text,
                    });
                }
            }
            let mut state = r2.state.clone();
            state.fpsr = d2.state.fpsr;
            state.apply_rust(cpu.ls_ctx_mut());
            return self.finish(cpu, &r2.stop, r2.retired);
        }
        self.restore(cpu, &r2.writes, &d2.writes);
        let timer = r2.timer;
        // Dynarmic stopped on an instruction it raised an exception for (and
        // left its state as it was before it): no bisection needed once the
        // prefix is confirmed, and a hot unimplemented instruction does not
        // cost a full search every chunk.
        if d2.exception && !matches!(r2.stop, StopReason::Brk(_) | StopReason::Hvc(_)) {
            if let Some(chunk) = self.dynarmic_unsupported_shortcut(cpu, s, &d2, &timer) {
                return chunk;
            }
        }
        let k = self.bisect(cpu, s, count, &timer);
        let loc = match self.locate(cpu, s, k, &timer) {
            Ok(loc) => loc,
            Err(error) => {
                log::error!("lockstep: locating the divergence failed: {error}");
                self.stalled = true;
                s.apply_rust(cpu.ls_ctx_mut());
                return Chunk::Event(CpuEvent::Stalled, 0);
            }
        };
        let class = classify(loc.word, loc.before.fpcr, &loc.diff);
        self.resolve(cpu, s, loc, class, &timer)
    }

    /// `d` stopped on an exception at instruction `k = d.executed`: when
    /// Rustarmic agrees up to it, keep Rustarmic's result for it (Dynarmic
    /// cannot run it).  The first sighting at a pc gets the full report.
    fn dynarmic_unsupported_shortcut(
        &mut self,
        cpu: &mut RustarmicCpu,
        s: &GuestState,
        d: &DynRun,
        timer: &[u64],
    ) -> Option<Chunk> {
        let k = d.executed.filter(|&k| k >= 1)?;
        let pc = d.state.pc;
        let first = !self.reported_pcs.contains_key(&(Class::DynarmicUnsupported, pc));
        if first {
            // The general path locates it and writes the full report.
            return None;
        }
        let prefix = self.rust_exact(cpu, Some(s), k - 1, false).ok()?;
        self.rollback(cpu, &prefix.writes);
        let d_prefix = self.shadow.step(s, k - 1, timer, false);
        self.after_dyn(cpu, &d_prefix);
        let agrees = self.compare(cpu, &prefix, &d_prefix, true).ignoring_fpsr_empty();
        self.restore(cpu, &prefix.writes, &d_prefix.writes);
        if !agrees || prefix.state.pc != pc {
            return None;
        }
        self.count(Class::DynarmicUnsupported);
        *self.reported_pcs.entry((Class::DynarmicUnsupported, pc)).or_default() += 1;
        if self.cfg.halt_on_benign {
            let _ = self.rust_exact(cpu, Some(s), k - 1, false);
            self.stalled = true;
            return Some(Chunk::Event(CpuEvent::Stalled, k - 1));
        }
        let r = self.rust_exact(cpu, Some(s), k, false).ok()?;
        Some(self.finish(cpu, &r.stop, r.retired))
    }

    /// Smallest `k` in `1..=count` whose state after `k` instructions
    /// differs, given that `count` differs (and 0 trivially matches).
    fn bisect(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, count: u64, timer: &[u64]) -> u64 {
        let (mut lo, mut hi) = (0u64, count);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.differs_after(cpu, s, mid, timer) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    fn differs_after(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, k: u64, timer: &[u64]) -> bool {
        let r = match self.rust_exact(cpu, Some(s), k, false) {
            Ok(r) => r,
            Err(_) => return true,
        };
        self.rollback(cpu, &r.writes);
        let d = self.shadow.step(s, k, timer, false);
        self.after_dyn(cpu, &d);
        let diff = self.compare(cpu, &r, &d, true);
        self.restore(cpu, &r.writes, &d.writes);
        !diff.ignoring_fpsr_empty()
    }

    fn locate(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, k: u64, timer: &[u64]) -> Result<Located, String> {
        let before = if k > 1 {
            let d = self.shadow.step(s, k - 1, timer, false);
            self.after_dyn(cpu, &d);
            self.restore(cpu, &[], &d.writes);
            d.state
        } else {
            s.clone()
        };
        let mut word_bytes = [0u8; 4];
        cpu.ls_read(before.pc, &mut word_bytes);
        let word = u32::from_le_bytes(word_bytes);
        let r = self.rust_exact(cpu, Some(s), k, false)?;
        self.rollback(cpu, &r.writes);
        let d = self.shadow.step(s, k, timer, true);
        self.after_dyn(cpu, &d);
        let diff = self.compare(cpu, &r, &d, false);
        self.restore(cpu, &r.writes, &d.writes);
        // The same instruction in a block of its own.
        let prefix = self.rust_exact(cpu, Some(s), k - 1, false)?;
        let single = self.rust_exact(cpu, None, 1, false)?;
        let combined = RustRun {
            state: single.state,
            stop: single.stop,
            retired: prefix.retired + single.retired,
            writes: prefix.writes.iter().chain(&single.writes).copied().collect(),
            reads: Vec::new(),
            timer: Vec::new(),
            fetches: Vec::new(),
            atomics: prefix.atomics + single.atomics,
        };
        self.rollback(cpu, &combined.writes);
        let d_again = self.shadow.step(s, k, timer, false);
        self.after_dyn(cpu, &d_again);
        let block_context = self.compare(cpu, &combined, &d_again, true).ignoring_fpsr_empty();
        self.restore(cpu, &combined.writes, &d_again.writes);
        let trace = d.trace.clone();
        Ok(Located {
            k,
            pc: before.pc,
            word,
            before,
            rust_after: r.state,
            dyn_after: d,
            diff,
            block_context,
            trace,
        })
    }

    /// Reports `loc` and applies the policy for its class.
    fn resolve(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, loc: Located, class: Class, timer: &[u64]) -> Chunk {
        self.count(class);
        let seen = {
            let n = self.reported_pcs.entry((class, loc.pc)).or_default();
            *n += 1;
            *n
        };
        let class_total = self.stats.by_class.get(class.label()).copied().unwrap_or(0);
        let full = !class.benign() || (seen == 1 && class_total <= 8);
        if full {
            let repro = self.capture_repro(cpu, s, &loc, class, timer);
            self.report(cpu, s, &loc, class, repro);
        } else if seen <= 2 && class_total <= 256 {
            log::warn!(
                "lockstep: [{}] pc={:#x} {:08x} {} : {}",
                class.label(),
                loc.pc,
                loc.word,
                disasm(loc.pc, loc.word),
                loc.diff.summary()
            );
        }
        let halt = if class.benign() {
            self.cfg.halt_on_benign
        } else {
            self.cfg.halt_on_divergence
        };
        if halt {
            // Leave the core right before the instruction.
            match self.rust_exact(cpu, Some(s), loc.k - 1, false) {
                Ok(_) => {}
                Err(_) => s.apply_rust(cpu.ls_ctx_mut()),
            }
            self.stalled = true;
            log::error!(
                "lockstep: core stalled at pc={:#x} before the diverging instruction (NEXIUM_LOCKSTEP_ON_DIVERGE=continue / NEXIUM_LOCKSTEP_BENIGN=continue keep going)",
                cpu.ls_ctx().pc
            );
            return Chunk::Event(CpuEvent::Stalled, loc.k - 1);
        }
        if class == Class::DynarmicUnsupported {
            // Dynarmic cannot run it: keep Rustarmic's result.
            return match self.rust_exact(cpu, Some(s), loc.k, false) {
                Ok(r) => Chunk::Continue(r.retired),
                Err(_) => Chunk::Event(CpuEvent::Stalled, 0),
            };
        }
        let d = self.shadow.step(s, loc.k, timer, false);
        self.after_dyn(cpu, &d);
        d.state.apply_rust(cpu.ls_ctx_mut());
        let executed = d.executed.unwrap_or(loc.k);
        match d.svc {
            Some(imm) => Chunk::Event(CpuEvent::Svc(imm as u16), executed),
            None => Chunk::Continue(executed),
        }
    }

    fn on_unsupported(&mut self, cpu: &mut RustarmicCpu, s: &GuestState, r: RustRun) -> Chunk {
        let StopReason::Unsupported(info) = &r.stop else {
            return self.finish(cpu, &r.stop, r.retired);
        };
        let info = info.clone();
        self.count(Class::Unsupported);
        let here = GuestState::from_rust(cpu.ls_ctx());
        let mut text = String::new();
        let _ = writeln!(
            text,
            "=== NeXium lockstep: [{}] guest pc={:#x} ===",
            Class::Unsupported.label(),
            info.pc
        );
        let _ = writeln!(text, "{}", Class::Unsupported.explain());
        let _ = writeln!(
            text,
            "instruction: {:08x}  {}   (decoder class {})",
            info.opcode,
            disasm(info.pc, info.opcode),
            info.decoded_class
        );
        let _ = writeln!(
            text,
            "block starts at {:#x}; {} instructions of this chunk (from {:#x}) matched Dynarmic",
            here.pc, r.retired, s.pc
        );
        let _ = writeln!(text, "state at the block start:");
        here.dump(&mut text, "  ");
        self.code_listing(cpu, info.pc, &[], &mut text);
        let first = self.reported_pcs.entry((Class::Unsupported, info.pc)).or_default();
        *first += 1;
        if *first == 1 {
            let (report, _) = self.write_files(Class::Unsupported, &text, None);
            log::error!("{text}");
            self.records.push(Record {
                class: Class::Unsupported,
                pc: info.pc,
                word: info.opcode,
                k: r.retired + 1,
                chunk_pc: s.pc,
                block_context: false,
                report,
                repro: None,
                repro_verified: None,
                text,
            });
        }
        if self.cfg.halt_on_divergence {
            self.stalled = true;
            return Chunk::Event(CpuEvent::Stalled, r.retired);
        }
        // Let Dynarmic execute up to and including the instruction.
        let d = self.shadow.step_past(&here, info.pc, 1024);
        self.after_dyn(cpu, &d);
        d.state.apply_rust(cpu.ls_ctx_mut());
        let executed = r.retired + d.executed.unwrap_or(0);
        match d.svc {
            Some(imm) => Chunk::Event(CpuEvent::Svc(imm as u16), executed),
            None => Chunk::Continue(executed),
        }
    }

    fn on_fault(
        &mut self,
        cpu: &mut RustarmicCpu,
        s: &GuestState,
        budget: u64,
        r: RustRun,
        fault: rustarmic::MemoryFault,
    ) -> Chunk {
        self.rollback(cpu, &r.writes);
        // The longest prefix that does not fault.
        let (mut lo, mut hi) = (0u64, r.retired.max(1));
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let faults = match self.rust_exact(cpu, Some(s), mid, false) {
                Ok(run) => {
                    self.rollback(cpu, &run.writes);
                    matches!(run.stop, StopReason::MemoryFault(_))
                }
                Err(_) => true,
            };
            if faults {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        if lo > 0 {
            if let Ok(prefix) = self.rust_exact(cpu, Some(s), lo, false) {
                self.rollback(cpu, &prefix.writes);
                let d = self.shadow.step(s, lo, &prefix.timer, false);
                self.after_dyn(cpu, &d);
                let diff = self.compare(cpu, &prefix, &d, true);
                if !diff.ignoring_fpsr_empty() {
                    // The fault follows a divergence: report that instead.
                    return self.on_mismatch(cpu, s, prefix, d);
                }
                self.restore(cpu, &prefix.writes, &d.writes);
            }
        }
        let kernel = kernel_mapped(fault.address);
        let class = if fault.cause == MemoryFaultCause::Unmapped && kernel == Some(true) {
            Class::MappingSync
        } else {
            Class::GuestFault
        };
        self.count(class);
        let n = self.stats.by_class.get(class.label()).copied().unwrap_or(0);
        if n <= 16 || n % 1000 == 0 {
            log::warn!(
                "lockstep: [{}] pc={:#x} addr={:#x} size={} access={:?} cause={:?} kernel={} core={} | prefix of {} instructions matched Dynarmic (count {})",
                class.label(),
                fault.pc,
                fault.address,
                fault.size,
                fault.access,
                fault.cause,
                match kernel {
                    Some(true) => "mapped",
                    Some(false) => "unmapped",
                    None => "n/a",
                },
                cpu.mapping_view(fault.address),
                lo,
                n
            );
        }
        // Re-run the chunk for real so the fault is delivered exactly as
        // without lockstep.
        s.apply_rust(cpu.ls_ctx_mut());
        match self.rust_primary(cpu, budget, false) {
            Ok(run) => self.finish(cpu, &run.stop, run.retired),
            Err(error) => {
                log::warn!("rustarmic Jit::run error: {error}");
                Chunk::Event(CpuEvent::Stalled, 0)
            }
        }
    }

    // -- reporting -----------------------------------------------------------

    fn code_listing(&self, cpu: &RustarmicCpu, pc: u64, trace: &[u64], out: &mut String) {
        let _ = writeln!(out, "code around pc:");
        for i in -12i64..=12 {
            let addr = pc.wrapping_add((i * 4) as u64);
            let mut b = [0u8; 4];
            if !cpu.ls_read(addr, &mut b) {
                continue;
            }
            let w = u32::from_le_bytes(b);
            let _ = writeln!(
                out,
                "  {} {addr:#012x}: {w:08x}  {}",
                if addr == pc { "=>" } else { "  " },
                disasm(addr, w)
            );
        }
        if !trace.is_empty() {
            let from = trace.len().saturating_sub(48);
            let _ = writeln!(out, "last {} executed instructions (oldest first):", trace.len() - from);
            for (i, &addr) in trace.iter().enumerate().skip(from) {
                let mut b = [0u8; 4];
                cpu.ls_read(addr, &mut b);
                let w = u32::from_le_bytes(b);
                let _ = writeln!(
                    out,
                    "  #{:<5} {addr:#012x}: {w:08x}  {}",
                    i + 1,
                    disasm(addr, w)
                );
            }
        }
    }

    fn report(&mut self, cpu: &RustarmicCpu, s: &GuestState, loc: &Located, class: Class, repro: Option<(Repro, Option<bool>)>) {
        let mut text = String::new();
        let _ = writeln!(
            text,
            "=== NeXium lockstep: [{}] guest pc={:#x} core={} ===",
            class.label(),
            loc.pc,
            cpu.ls_ctx().core_id
        );
        let _ = writeln!(text, "{}", class.explain());
        let _ = writeln!(text, "instruction: {:08x}  {}", loc.word, disasm(loc.pc, loc.word));
        let _ = writeln!(
            text,
            "it is instruction #{} of a lockstep chunk that started at pc={:#x} (core has retired ~{} checked-or-not instructions including this chunk)",
            loc.k,
            s.pc,
            self.retired_seen
        );
        if loc.block_context {
            let _ = writeln!(
                text,
                "BLOCK CONTEXT: this instruction translated on its own matches Dynarmic; only the block it was compiled in (with its neighbours) diverges: suspect an optimisation that spans instructions, such as flag fusion or register allocation. See the executed instructions below."
            );
        } else {
            let _ = writeln!(text, "the instruction alone (one-instruction block) reproduces the difference");
        }
        if class == Class::FpNan {
            if let Some(note) = default_nan_note(&loc.diff) {
                let _ = writeln!(text, "default NaN: {note}");
            }
        }
        if class == Class::FpcrMode {
            let _ = writeln!(text, "fpcr before = {:#010x}", loc.before.fpcr);
        }
        let _ = writeln!(text, "differences after the instruction:");
        loc.diff.describe(&mut text, "  ");
        let own: Vec<&MemWrite> = loc.dyn_after.writes.iter().filter(|w| u64::from(w.step) + 1 == loc.k).collect();
        if !own.is_empty() {
            let _ = writeln!(text, "dynarmic stores by this instruction:");
            for w in own {
                let n = w.size as usize;
                let _ = writeln!(
                    text,
                    "  [{:#x}] size={} old={} new={}",
                    w.addr,
                    n,
                    hex(&w.old[..n]),
                    hex(&w.new[..n])
                );
            }
        }
        if !loc.dyn_after.unmapped.is_empty() {
            let _ = writeln!(text, "dynarmic accesses to memory this core has unmapped (address, size, value):");
            for (addr, size, value) in loc.dyn_after.unmapped.iter().take(8) {
                let _ = writeln!(text, "  [{addr:#x}] size={size} value={value:#x}");
            }
        }
        let _ = writeln!(text, "state before the instruction:");
        loc.before.dump(&mut text, "  ");
        let _ = writeln!(text, "rustarmic after:");
        loc.rust_after.dump(&mut text, "  ");
        let _ = writeln!(text, "dynarmic after:");
        loc.dyn_after.state.dump(&mut text, "  ");
        let view = cpu.mapping_view(loc.pc);
        let _ = writeln!(text, "core map at pc: {view}");
        self.code_listing(cpu, loc.pc, &loc.trace, &mut text);
        let verified = repro.as_ref().and_then(|(_, v)| *v);
        if let Some((_, v)) = &repro {
            let _ = writeln!(
                text,
                "repro verified by an in-process replay (fresh JITs, plain memory, no fastmem): {}",
                match v {
                    Some(true) => "yes, it diverges the same way",
                    Some(false) => "NO: it does not diverge outside this process state (suspect the inline fastmem load path, or state this repro does not capture)",
                    None => "replay failed",
                }
            );
        }
        let (report_path, repro_path) = self.write_files(class, &text, repro.as_ref().map(|(r, _)| r));
        if let Some(p) = &report_path {
            let _ = writeln!(text, "report: {}", p.display());
        }
        if let Some(p) = &repro_path {
            let _ = writeln!(text, "repro: {}", p.display());
        }
        if class.benign() {
            log::warn!("{text}");
        } else {
            log::error!("{text}");
        }
        self.records.push(Record {
            class,
            pc: loc.pc,
            word: loc.word,
            k: loc.k,
            chunk_pc: s.pc,
            block_context: loc.block_context,
            report: report_path,
            repro: repro_path,
            repro_verified: verified,
            text,
        });
    }

    fn write_files(&self, class: Class, text: &str, repro: Option<&Repro>) -> (Option<PathBuf>, Option<PathBuf>) {
        let Some(dir) = &self.cfg.out_dir else {
            return (None, None);
        };
        if REPORTS_WRITTEN.fetch_add(1, Ordering::Relaxed) >= self.cfg.max_reports {
            return (None, None);
        }
        if let Err(error) = std::fs::create_dir_all(dir) {
            log::warn!("lockstep: cannot create {}: {error}", dir.display());
            return (None, None);
        }
        let stem = format!(
            "{}-{}-{:03}-{}",
            timestamp(),
            std::process::id(),
            EVENT_SEQ.fetch_add(1, Ordering::Relaxed),
            class.label()
        );
        let report = dir.join(format!("{stem}.txt"));
        let report = std::fs::write(&report, text).ok().map(|_| report);
        let repro = repro.and_then(|repro| {
            let path = dir.join(format!("{stem}.repro"));
            std::fs::write(&path, repro.to_text()).ok().map(|_| path)
        });
        (report, repro)
    }

    // -- repro capture -------------------------------------------------------

    fn capture_repro(
        &mut self,
        cpu: &mut RustarmicCpu,
        s: &GuestState,
        loc: &Located,
        class: Class,
        timer: &[u64],
    ) -> Option<(Repro, Option<bool>)> {
        // Start at the last branch target before the instruction (where a
        // block is likely to start); fall back to the chunk start, which is
        // exactly where Rustarmic's run began.
        let mut starts = Vec::new();
        let k = loc.k as usize;
        if loc.trace.len() >= k && k >= 2 {
            let j = (1..k)
                .rev()
                .find(|&j| loc.trace[j] != loc.trace[j - 1].wrapping_add(4))
                .unwrap_or(0);
            if j > 0 {
                starts.push(j as u64);
            }
        }
        starts.push(0);
        let mut last = None;
        for j in starts {
            let repro = match self.build_repro(cpu, s, j, loc, class, timer) {
                Ok(repro) => repro,
                Err(error) => {
                    log::warn!("lockstep: repro capture failed: {error}");
                    continue;
                }
            };
            #[cfg(test)]
            let injection = self.replay_injection;
            let verified = {
                let prepare = |_cpu: &mut RustarmicCpu| {
                    #[cfg(test)]
                    {
                        _cpu.ls_jit_mut().fault_injection = injection;
                    }
                };
                if self.replay_rig.is_none() {
                    match ReplayRig::new() {
                        Ok(rig) => self.replay_rig = Some(Box::new(rig)),
                        Err(error) => log::warn!("lockstep: no replay JITs: {error}"),
                    }
                }
                let result = match self.replay_rig.as_deref_mut() {
                    Some(rig) => replay_on(rig, &repro, &prepare),
                    None => Err("no replay JITs".to_string()),
                };
                match result {
                    Ok(outcome) => Some(outcome.diverged),
                    Err(error) => {
                        log::warn!("lockstep: repro replay failed: {error}");
                        None
                    }
                }
            };
            if verified == Some(true) {
                return Some((repro, verified));
            }
            last = Some((repro, verified));
        }
        last
    }

    fn build_repro(
        &mut self,
        cpu: &mut RustarmicCpu,
        s: &GuestState,
        j: u64,
        loc: &Located,
        class: Class,
        timer: &[u64],
    ) -> Result<Repro, String> {
        let _ = timer;
        let prefix = self.rust_exact(cpu, Some(s), j, false)?;
        let start = prefix.state.clone();
        let window = loc.k - j;
        // Retranslate so every instruction fetch is seen.
        let pages: BTreeSet<u64> = loc.trace.iter().map(|pc| pc & !0xFFF).collect();
        for &page in &pages {
            cpu.ls_invalidate_jit(page, 0x1000);
        }
        let capture = self.rust_exact(cpu, None, window, true);
        let capture = match capture {
            Ok(c) => c,
            Err(e) => {
                self.rollback(cpu, &prefix.writes);
                return Err(e);
            }
        };
        self.rollback(cpu, &capture.writes);
        let mut code_bytes: BTreeSet<u64> = BTreeSet::new();
        for &a in capture.fetches.iter().chain(loc.trace.iter().skip(j as usize)) {
            for i in 0..4 {
                code_bytes.insert(a.wrapping_add(i));
            }
        }
        let mut data_bytes: BTreeSet<u64> = BTreeSet::new();
        let mut add_data = |addr: u64, size: u64| {
            // Whole 16-byte lines: cheap and robust to small address skews.
            let lo = addr & !15;
            let hi = (addr.wrapping_add(size).wrapping_add(15)) & !15;
            let mut a = lo;
            while a < hi {
                data_bytes.insert(a);
                a += 1;
            }
        };
        for &(addr, size) in &capture.reads {
            add_data(addr, u64::from(size));
        }
        for w in &capture.writes {
            add_data(w.addr, u64::from(w.size));
        }
        for w in &loc.dyn_after.writes {
            add_data(w.addr, u64::from(w.size));
        }
        let read_runs = |cpu: &RustarmicCpu, set: &BTreeSet<u64>| -> Vec<(u64, Vec<u8>)> {
            runs(set.iter().filter_map(|&a| {
                let mut b = [0u8; 1];
                cpu.ls_read(a, &mut b).then_some((a, b[0]))
            }))
        };
        let data_bytes: BTreeSet<u64> = data_bytes.difference(&code_bytes).copied().collect();
        let code = read_runs(cpu, &code_bytes);
        let data = read_runs(cpu, &data_bytes);
        self.rollback(cpu, &prefix.writes);
        let rust_mem = runs(
            loc.diff
                .mem
                .iter()
                .map(|&(a, r, _)| (a, r))
                .collect::<BTreeMap<_, _>>(),
        );
        let dyn_mem = runs(
            loc.diff
                .mem
                .iter()
                .map(|&(a, _, d)| (a, d))
                .collect::<BTreeMap<_, _>>(),
        );
        Ok(Repro {
            class: class.label().to_string(),
            pc: loc.pc,
            word: loc.word,
            start,
            insns: window,
            code,
            data,
            expect_dyn: loc.dyn_after.state.clone(),
            expect_rust: loc.rust_after.clone(),
            dyn_mem,
            rust_mem,
            note: format!(
                "instruction #{} of a chunk starting at {:#x}; window starts {} instructions in",
                loc.k, s.pc, j
            ),
        })
    }
}

fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

// ---------------------------------------------------------------------------
// Repro files.

/// A minimized, self-contained reproduction: a start state, the memory the
/// window touches, the instruction count, and both JITs' end states.
#[derive(Clone, Debug, Default)]
pub struct Repro {
    pub class: String,
    pub pc: u64,
    pub word: u32,
    pub start: GuestState,
    pub insns: u64,
    /// Instruction bytes (mapped RWX).
    pub code: Vec<(u64, Vec<u8>)>,
    /// Data bytes (mapped RW).
    pub data: Vec<(u64, Vec<u8>)>,
    pub expect_dyn: GuestState,
    pub expect_rust: GuestState,
    /// Final values of the bytes whose final values differ.
    pub dyn_mem: Vec<(u64, Vec<u8>)>,
    pub rust_mem: Vec<(u64, Vec<u8>)>,
    pub note: String,
}

impl Repro {
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# NeXium lockstep repro v1");
        let _ = writeln!(out, "# class: {}", self.class);
        let _ = writeln!(
            out,
            "# diverging instruction: {:#x}: {:08x}  {}",
            self.pc,
            self.word,
            disasm(self.pc, self.word)
        );
        let _ = writeln!(out, "# {}", self.note);
        let _ = writeln!(
            out,
            "# Replay: NEXIUM_LOCKSTEP_REPLAY=<this file> cargo test -p nexium-cpu --lib lockstep::tests::replay_file -- --nocapture"
        );
        let _ = writeln!(
            out,
            "#         RUSTARMIC_LOCKSTEP_REPRO=<this file> cargo test --features unicorn --test lockstep_replay -- --nocapture   (in rustarmic)"
        );
        let _ = writeln!(out, "class {}", self.class);
        let _ = writeln!(out, "pc {:#x}", self.pc);
        let _ = writeln!(out, "word {:#x}", self.word);
        let _ = writeln!(out, "insns {}", self.insns);
        self.start.write_text("start", &mut out);
        for (addr, bytes) in &self.code {
            let _ = writeln!(out, "code {addr:#x} {}", hex(bytes));
        }
        for (addr, bytes) in &self.data {
            let _ = writeln!(out, "data {addr:#x} {}", hex(bytes));
        }
        self.expect_dyn.write_text("dyn", &mut out);
        self.expect_rust.write_text("rust", &mut out);
        for (addr, bytes) in &self.dyn_mem {
            let _ = writeln!(out, "dyn.mem {addr:#x} {}", hex(bytes));
        }
        for (addr, bytes) in &self.rust_mem {
            let _ = writeln!(out, "rust.mem {addr:#x} {}", hex(bytes));
        }
        out
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        fn num(s: &str) -> Result<u64, String> {
            match s.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16),
                None => s.parse(),
            }
            .map_err(|e| format!("bad number {s:?}: {e}"))
        }
        fn bytes(s: &str) -> Result<Vec<u8>, String> {
            if s.len() % 2 != 0 {
                return Err("odd hex length".into());
            }
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
                .collect()
        }
        let mut r = Repro::default();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let key = parts.next().unwrap_or_default();
            let rest: Vec<&str> = parts.collect();
            let err = |e: String| format!("line {}: {e}", lineno + 1);
            match key {
                "class" => r.class = rest.join(" "),
                "pc" => r.pc = num(rest.first().ok_or("pc")?).map_err(err)?,
                "word" => r.word = num(rest.first().ok_or("word")?).map_err(err)? as u32,
                "insns" => r.insns = num(rest.first().ok_or("insns")?).map_err(err)?,
                "code" | "data" | "dyn.mem" | "rust.mem" => {
                    if rest.len() != 2 {
                        return Err(err(format!("{key}: expected address and bytes")));
                    }
                    let entry = (num(rest[0]).map_err(err)?, bytes(rest[1]).map_err(err)?);
                    match key {
                        "code" => r.code.push(entry),
                        "data" => r.data.push(entry),
                        "dyn.mem" => r.dyn_mem.push(entry),
                        _ => r.rust_mem.push(entry),
                    }
                }
                _ => {
                    let (prefix, field) = key.split_once('.').ok_or_else(|| err(format!("unknown key {key}")))?;
                    let values = rest.iter().map(|v| num(v)).collect::<Result<Vec<_>, _>>().map_err(err)?;
                    let target = match prefix {
                        "start" => &mut r.start,
                        "dyn" => &mut r.expect_dyn,
                        "rust" => &mut r.expect_rust,
                        _ => return Err(err(format!("unknown key {key}"))),
                    };
                    target.read_field(field, &values).map_err(err)?;
                }
            }
        }
        if r.insns == 0 {
            return Err("repro has no instructions".into());
        }
        Ok(r)
    }
}

/// Page-aligned host memory for a replay.
struct ReplayPages {
    runs: Vec<(u64, usize, *mut u8, std::alloc::Layout, Perm)>,
}

impl ReplayPages {
    fn new(repro: &Repro) -> Result<Self, String> {
        let mut pages: BTreeMap<u64, Perm> = BTreeMap::new();
        let mut add = |chunks: &[(u64, Vec<u8>)], perm: Perm| {
            for (addr, bytes) in chunks {
                let lo = addr & !0xFFF;
                let hi = (addr + bytes.len() as u64 + 0xFFF) & !0xFFF;
                let mut p = lo;
                while p < hi {
                    let slot = pages.entry(p).or_insert(perm);
                    if perm.contains(Perm::X) {
                        *slot = Perm::RWX;
                    }
                    p += 0x1000;
                }
            }
        };
        add(&repro.data, Perm::RW);
        add(&repro.code, Perm::RWX);
        let mut runs: Vec<(u64, usize, *mut u8, std::alloc::Layout, Perm)> = Vec::new();
        let mut spans: Vec<(u64, u64, Perm)> = Vec::new();
        for (&page, &perm) in &pages {
            match spans.last_mut() {
                Some((_, end, p)) if *end == page && *p == perm => *end += 0x1000,
                _ => spans.push((page, page + 0x1000, perm)),
            }
        }
        for (start, end, perm) in spans {
            let len = (end - start) as usize;
            let layout = std::alloc::Layout::from_size_align(len, 0x1000).map_err(|e| e.to_string())?;
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                return Err("replay allocation failed".into());
            }
            runs.push((start, len, ptr, layout, perm));
        }
        let pages = Self { runs };
        for (addr, bytes) in repro.code.iter().chain(&repro.data) {
            pages.write(*addr, bytes)?;
        }
        Ok(pages)
    }

    fn locate(&self, addr: u64) -> Option<(*mut u8, usize)> {
        self.runs.iter().find_map(|&(start, len, ptr, _, _)| {
            (addr >= start && addr < start + len as u64)
                .then(|| unsafe { (ptr.add((addr - start) as usize), len - (addr - start) as usize) })
        })
    }

    fn write(&self, addr: u64, bytes: &[u8]) -> Result<(), String> {
        let (ptr, room) = self.locate(addr).ok_or_else(|| format!("{addr:#x} not in replay memory"))?;
        if room < bytes.len() {
            for (i, &b) in bytes.iter().enumerate() {
                self.write(addr + i as u64, &[b])?;
            }
            return Ok(());
        }
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        Ok(())
    }

    fn snapshot(&self) -> Vec<Vec<u8>> {
        self.runs
            .iter()
            .map(|&(_, len, ptr, _, _)| unsafe { std::slice::from_raw_parts(ptr, len).to_vec() })
            .collect()
    }

    fn restore(&self, snap: &[Vec<u8>]) {
        for (&(_, len, ptr, _, _), data) in self.runs.iter().zip(snap) {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len) };
        }
    }
}

impl Drop for ReplayPages {
    fn drop(&mut self) {
        for &(_, _, ptr, layout, _) in &self.runs {
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }
}

/// Result of re-executing a repro on fresh Rustarmic and Dynarmic JITs.
#[derive(Debug)]
pub struct ReplayOutcome {
    pub rust: GuestState,
    pub dynarmic: GuestState,
    pub rust_stop: String,
    pub dyn_stop: String,
    /// The two JITs disagree.
    pub diverged: bool,
    /// Dynarmic matches the end state recorded in the repro.
    pub dyn_matches_recorded: bool,
    /// Rustarmic matches the (diverging) end state recorded in the repro.
    pub rust_matches_recorded: bool,
    pub summary: String,
}

/// Re-executes `repro` on a fresh Rustarmic core and a fresh shadow
/// Dynarmic, over private memory.
pub fn replay(repro: &Repro) -> Result<ReplayOutcome, String> {
    replay_with(repro, &|_| {})
}

/// A Rustarmic core and a Dynarmic JIT for replays.  dynarmic-sys never
/// frees a JIT (its callbacks keep the handle alive), so a lockstep core
/// keeps one rig and remaps it for every replay instead of leaking a JIT
/// per report.
pub(crate) struct ReplayRig {
    cpu: RustarmicCpu,
    shadow: Shadow,
}

impl ReplayRig {
    fn new() -> Result<Self, String> {
        let cpu = {
            let _quiet = suppress();
            RustarmicCpu::new()?
        };
        Ok(Self {
            cpu,
            shadow: Shadow::new()?,
        })
    }
}

fn replay_with(repro: &Repro, prepare: &dyn Fn(&mut RustarmicCpu)) -> Result<ReplayOutcome, String> {
    replay_on(&mut ReplayRig::new()?, repro, prepare)
}

fn replay_on(rig: &mut ReplayRig, repro: &Repro, prepare: &dyn Fn(&mut RustarmicCpu)) -> Result<ReplayOutcome, String> {
    let pages = ReplayPages::new(repro)?;
    prepare(&mut rig.cpu);
    for &(start, len, ptr, _, perm) in &pages.runs {
        unsafe {
            rig.cpu.map_host(start, len as u64, perm, ptr)?;
            rig.shadow.map_host(start, len as u64, perm, ptr);
        }
    }
    let result = replay_mapped(rig, repro, &pages);
    for &(start, len, _, _, _) in &pages.runs {
        unsafe {
            let _ = rig.cpu.unmap_host(start, len as u64);
            rig.shadow.unmap_host(start, len as u64);
        }
        rig.cpu.ls_invalidate_jit(start, len as u64);
        rig.shadow.invalidate(start, len as u64);
    }
    let _ = rig.cpu.ls_take_invalidations();
    result
}

fn replay_mapped(rig: &mut ReplayRig, repro: &Repro, pages: &ReplayPages) -> Result<ReplayOutcome, String> {
    let cpu = &mut rig.cpu;
    let initial = pages.snapshot();
    repro.start.apply_rust(cpu.ls_ctx_mut());
    let mut retired = 0u64;
    let mut rust_stop = StopReason::BudgetExhausted;
    while retired < repro.insns {
        let out = cpu.ls_run(repro.insns - retired, None).map_err(|e| format!("{e:?}"))?;
        retired += out.retired;
        rust_stop = out.stop;
        cpu.ls_take_invalidations();
        match rust_stop {
            StopReason::BudgetExhausted => {}
            StopReason::Halted if out.retired > 0 => {}
            _ => break,
        }
    }
    let rust = GuestState::from_rust(cpu.ls_ctx());
    let rust_mem = pages.snapshot();
    pages.restore(&initial);
    let d = rig.shadow.step(&repro.start, repro.insns, &[], false);
    let dyn_mem = pages.snapshot();
    pages.restore(&initial);
    let mut diff = Diff::default();
    compare_states(&rust, &d.state, &mut diff);
    if !stops_agree(&rust_stop, &d) {
        diff.stop = Some((rust_stop_label(&rust_stop), dyn_stop_label(&d)));
    }
    if d.executed != Some(retired) {
        diff.retired = Some((retired, d.executed.unwrap_or(0)));
    }
    for (run, (r, dm)) in pages.runs.iter().zip(rust_mem.iter().zip(&dyn_mem)) {
        for i in 0..r.len() {
            if r[i] != dm[i] {
                diff.mem.push((run.0 + i as u64, r[i], dm[i]));
            }
        }
    }
    let mut summary = String::new();
    diff.describe(&mut summary, "  ");
    let mut recorded = Diff::default();
    compare_states(&d.state, &repro.expect_dyn, &mut recorded);
    let dyn_matches_recorded = recorded.ignoring_fpsr_empty();
    let mut recorded_rust = Diff::default();
    compare_states(&rust, &repro.expect_rust, &mut recorded_rust);
    Ok(ReplayOutcome {
        rust,
        dynarmic: d.state.clone(),
        rust_stop: rust_stop_label(&rust_stop),
        dyn_stop: dyn_stop_label(&d),
        diverged: !diff.ignoring_fpsr_empty(),
        dyn_matches_recorded,
        rust_matches_recorded: recorded_rust.ignoring_fpsr_empty(),
        summary,
    })
}

#[cfg(test)]
#[path = "../../rustarmic/tests/control_flow_fuzz.rs"]
#[allow(dead_code, unexpected_cfgs, clippy::all)]
mod control_flow_fuzz;

#[cfg(test)]
#[path = "lockstep_tests.rs"]
mod tests;

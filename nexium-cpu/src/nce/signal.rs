use super::context::{CoreState, GuestContext, NativeExecutionParameters};
use crate::nce_layout::*;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

#[repr(C, align(16))]
pub struct ReservedArea(pub [u8; 4096]);

#[repr(C, align(16))]
pub struct McontextAarch64 {
    pub fault_address: u64,
    pub regs: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub pstate: u64,
    pub reserved: ReservedArea,
}

#[repr(C)]
pub struct UcontextAarch64 {
    pub uc_flags: u64,
    pub uc_link: usize,
    pub uc_stack: [u64; 3],
    pub uc_sigmask: u64,
    pub unused: [u8; 120],
    pub uc_mcontext: McontextAarch64,
}

#[repr(C)]
pub struct AarchCtxHeader {
    pub magic: u32,
    pub size: u32,
}

#[repr(C, align(16))]
pub struct FpsimdContext {
    pub head: AarchCtxHeader,
    pub fpsr: u32,
    pub fpcr: u32,
    pub vregs: [u128; 32],
}

const FPSIMD_MAGIC: u32 = 0x4650_8001;
const SVE_MAGIC: u32 = 0x5356_4501;

const _: () = {
    assert!(std::mem::offset_of!(UcontextAarch64, uc_mcontext) == 176);
    assert!(std::mem::offset_of!(McontextAarch64, reserved) == 288);
    assert!(std::mem::size_of::<FpsimdContext>() == 16 + 32 * 16);
};

unsafe fn find_fpsimd(mc: *mut McontextAarch64) -> Option<*mut FpsimdContext> {
    let base = std::ptr::addr_of_mut!((*mc).reserved) as *mut u8;
    let mut offset = 0usize;
    while offset + 8 <= 4096 {
        let header = base.add(offset) as *mut AarchCtxHeader;
        let magic = (*header).magic;
        let size = (*header).size as usize;
        if magic == 0 || size == 0 {
            return None;
        }
        if magic == FPSIMD_MAGIC {
            return Some(header as *mut FpsimdContext);
        }
        offset += size;
    }
    None
}

unsafe fn find_sve_payload(mc: *mut McontextAarch64) -> Option<(*mut u8, usize)> {
    let base = std::ptr::addr_of_mut!((*mc).reserved) as *mut u8;
    let mut offset = 0usize;
    while offset + 8 <= 4096 {
        let header = base.add(offset) as *mut AarchCtxHeader;
        let magic = (*header).magic;
        let size = (*header).size as usize;
        if magic == 0 || size == 0 {
            return None;
        }
        if magic == SVE_MAGIC {
            let vl = *(base.add(offset + 8) as *const u16) as usize;
            let flags = *(base.add(offset + 10) as *const u16);
            let has_data = size > 16 && vl != 0 && flags & 1 == 0;
            return has_data.then_some((base.add(offset), vl));
        }
        offset += size;
    }
    None
}

unsafe fn write_sve_low_bits(mc: *mut McontextAarch64, vregs: &[u128; 32]) {
    let Some((sve, vl)) = find_sve_payload(mc) else {
        return;
    };
    let z_offset = (16 + 15) & !15;
    for (i, value) in vregs.iter().enumerate() {
        let dst = sve.add(z_offset + i * vl) as *mut u128;
        std::ptr::write_unaligned(dst, *value);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_restore_guest_context(raw: *mut libc::c_void) -> *mut u8 {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let nep = (*mc).regs[9] as *mut NativeExecutionParameters;
    let ctx = (*nep).native_context;
    let fp = find_fpsimd(mc);
    let host = &mut (*ctx).host;
    let frame_regs: &mut [u64; 31] = &mut (*mc).regs;
    host.regs.copy_from_slice(&frame_regs[19..31]);
    if let Some(fp) = fp {
        let frame_vregs: &mut [u128; 32] = &mut (*fp).vregs;
        host.vregs.copy_from_slice(&frame_vregs[8..16]);
    }
    host.sp = (*mc).sp;
    (*mc).sp = (*ctx).sp;
    (*mc).pc = (*ctx).pc;
    (*mc).pstate = (*ctx).pstate as u64;
    frame_regs.copy_from_slice(&(*ctx).x);
    if let Some(fp) = fp {
        (*fp).fpcr = (*ctx).fpcr;
        (*fp).fpsr = (*ctx).fpsr;
        let frame_vregs: &mut [u128; 32] = &mut (*fp).vregs;
        frame_vregs.copy_from_slice(&(*ctx).v);
    }
    write_sve_low_bits(mc, &(*ctx).v);
    nep as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_save_guest_context(ctx: *mut GuestContext, raw: *mut libc::c_void) {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let fp = find_fpsimd(mc);
    let frame_regs: &mut [u64; 31] = &mut (*mc).regs;
    (*ctx).x.copy_from_slice(frame_regs);
    if let Some(fp) = fp {
        let frame_vregs: &mut [u128; 32] = &mut (*fp).vregs;
        (*ctx).v.copy_from_slice(frame_vregs);
        (*ctx).fpsr = (*fp).fpsr;
        (*ctx).fpcr = (*fp).fpcr;
    }
    (*ctx).pstate = (*mc).pstate as u32;
    (*ctx).pc = (*mc).pc;
    (*ctx).sp = (*mc).sp;
    let host = &(*ctx).host;
    (*mc).sp = host.sp;
    frame_regs[19..31].copy_from_slice(&host.regs);
    if let Some(fp) = fp {
        let frame_vregs: &mut [u128; 32] = &mut (*fp).vregs;
        frame_vregs[8..16].copy_from_slice(&host.vregs);
    }
    (*mc).pc = host.regs[11];
    (*mc).regs[0] = (*ctx).esr.swap(0, Ordering::AcqRel);
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_unlock_nep(nep: *mut NativeExecutionParameters) {
    (*nep).lock.store(LOCK_UNLOCKED, Ordering::Release);
}

unsafe fn record_fault(ctx: *mut GuestContext, mc: *mut McontextAarch64, addr: u64, is_write: bool) {
    let core = (*ctx).core;
    if core.is_null() {
        return;
    }
    let fault = &(*core).fault;
    fault.pc.store((*mc).pc, Ordering::Relaxed);
    fault.addr.store(addr, Ordering::Relaxed);
    fault.lr.store((*mc).regs[30], Ordering::Relaxed);
    fault.sp.store((*mc).sp, Ordering::Relaxed);
    fault.is_write.store(is_write as u32, Ordering::Relaxed);
    for (slot, value) in fault.regs.iter().zip((*mc).regs.iter()) {
        slot.store(*value, Ordering::Relaxed);
    }
    fault.valid.store(1, Ordering::Release);
}

unsafe fn leave_guest(ctx: *mut GuestContext, raw: *mut libc::c_void, reason: u64) -> u64 {
    (*ctx).esr.fetch_or(reason, Ordering::AcqRel);
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let nep = current_nep();
    if !nep.is_null() {
        (*nep).lock.store(LOCK_LOCKED, Ordering::Release);
    }
    let _ = mc;
    nexium_nce_save_guest_context(ctx, raw);
    0
}

unsafe fn current_nep() -> *mut NativeExecutionParameters {
    CURRENT_NEP.with(|slot| slot.get())
}

thread_local! {
    pub static CURRENT_NEP: std::cell::Cell<*mut NativeExecutionParameters> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

fn is_store_instruction(insn: u32) -> bool {
    if insn & 0x0A00_0000 == 0x0800_0000 {
        let op = (insn >> 22) & 0x3;
        return op == 0 || (insn >> 23) & 0x7F == 0x10 && (insn >> 22) & 1 == 0;
    }
    false
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_guest_segv(
    ctx: *mut GuestContext,
    info: *mut libc::siginfo_t,
    raw: *mut libc::c_void,
) -> u64 {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let addr = (*info).si_addr() as u64;
    let pc = (*mc).pc;
    if pc == addr {
        return leave_guest(ctx, raw, HALT_PREFETCH_ABORT);
    }
    let insn = if pc % 4 == 0 { std::ptr::read_volatile(pc as *const u32) } else { 0 };
    record_fault(ctx, mc, addr, is_store_instruction(insn));
    let core = (*ctx).core;
    if !core.is_null() && (*core).continue_on_null.load(Ordering::Relaxed) != 0 {
        let skips = (*core).null_skips.fetch_add(1, Ordering::Relaxed) + 1;
        if skips <= (*core).null_skip_limit {
            (*mc).pc = pc.wrapping_add(4);
            return 1;
        }
    }
    leave_guest(ctx, raw, HALT_DATA_ABORT)
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_guest_bus(
    ctx: *mut GuestContext,
    info: *mut libc::siginfo_t,
    raw: *mut libc::c_void,
) -> u64 {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let addr = (*info).si_addr() as u64;
    record_fault(ctx, mc, addr, false);
    leave_guest(ctx, raw, HALT_ALIGNMENT)
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_guest_trap(
    ctx: *mut GuestContext,
    _info: *mut libc::siginfo_t,
    raw: *mut libc::c_void,
) -> u64 {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    let pc = (*mc).pc;
    let insn = if pc % 4 == 0 { std::ptr::read_volatile(pc as *const u32) } else { 0 };
    if let Some(imm) = crate::nce_patch::is_brk(insn) {
        if imm == EXIT_STUB_BRK_IMM as u32 {
            (*mc).pc = pc.wrapping_add(4);
            (*ctx).svc = 7;
            return leave_guest(ctx, raw, HALT_SUPERVISOR_CALL);
        }
    }
    record_fault(ctx, mc, pc, false);
    leave_guest(ctx, raw, HALT_BREAKPOINT)
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_guest_ill(
    ctx: *mut GuestContext,
    _info: *mut libc::siginfo_t,
    raw: *mut libc::c_void,
) -> u64 {
    let uc = raw as *mut UcontextAarch64;
    let mc = std::ptr::addr_of_mut!((*uc).uc_mcontext);
    record_fault(ctx, mc, (*mc).pc, false);
    leave_guest(ctx, raw, HALT_BREAKPOINT)
}

struct SavedActions {
    segv: libc::sigaction,
    bus: libc::sigaction,
    trap: libc::sigaction,
    ill: libc::sigaction,
}

unsafe impl Send for SavedActions {}
unsafe impl Sync for SavedActions {}

static SAVED: OnceLock<SavedActions> = OnceLock::new();

unsafe fn chain(sig: i32, saved: &libc::sigaction, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    let handler = saved.sa_sigaction;
    if handler == libc::SIG_IGN {
        return;
    }
    if handler == libc::SIG_DFL {
        let mut default: libc::sigaction = std::mem::zeroed();
        default.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut default.sa_mask);
        libc_sigaction()(sig, &default, std::ptr::null_mut());
        return;
    }
    if saved.sa_flags & libc::SA_SIGINFO != 0 {
        let f: extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) = std::mem::transmute(handler);
        f(sig, info, raw);
    } else {
        let f: extern "C" fn(i32) = std::mem::transmute(handler);
        f(sig);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_host_segv(sig: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    if let Some(saved) = SAVED.get() {
        chain(sig, &saved.segv, info, raw);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_host_bus(sig: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    if let Some(saved) = SAVED.get() {
        chain(sig, &saved.bus, info, raw);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_host_trap(sig: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    if let Some(saved) = SAVED.get() {
        chain(sig, &saved.trap, info, raw);
    }
}

#[no_mangle]
pub unsafe extern "C" fn nexium_nce_host_ill(sig: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    if let Some(saved) = SAVED.get() {
        chain(sig, &saved.ill, info, raw);
    }
}

pub const SIGNAL_ENTER: i32 = libc::SIGUSR2;
pub const SIGNAL_BREAK: i32 = libc::SIGURG;

type SigactionFn =
    unsafe extern "C" fn(i32, *const libc::sigaction, *mut libc::sigaction) -> libc::c_int;

fn libc_sigaction() -> SigactionFn {
    static RESOLVED: OnceLock<usize> = OnceLock::new();
    let address = *RESOLVED.get_or_init(|| unsafe {
        let name = b"sigaction\0".as_ptr() as *const libc::c_char;
        let mut symbol = std::ptr::null_mut();
        for library in [b"libc.so\0".as_ptr(), b"libc.so.6\0".as_ptr()] {
            let handle = libc::dlopen(library as *const libc::c_char, libc::RTLD_NOW);
            if !handle.is_null() {
                symbol = libc::dlsym(handle, name);
                if !symbol.is_null() {
                    break;
                }
            }
        }
        if symbol.is_null() {
            log::warn!("nce: could not resolve libc sigaction directly; using the interposed one");
            libc::sigaction as usize
        } else {
            symbol as usize
        }
    });
    unsafe { std::mem::transmute::<usize, SigactionFn>(address) }
}

pub fn install_handlers() -> Result<(), String> {
    if SAVED.get().is_some() {
        return Ok(());
    }
    let sigaction = libc_sigaction();
    log::info!(
        "nce: installing signal handlers through {}",
        if sigaction as usize == libc::sigaction as usize {
            "the process sigaction"
        } else {
            "libc.so sigaction (sigchain bypassed)"
        }
    );
    unsafe {
        let mut mask: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut mask);
        for sig in [SIGNAL_ENTER, SIGNAL_BREAK, libc::SIGSEGV, libc::SIGBUS, libc::SIGTRAP, libc::SIGILL] {
            libc::sigaddset(&mut mask, sig);
        }
        let install = |sig: i32,
                       handler: unsafe extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void),
                       restart: bool|
         -> Result<libc::sigaction, String> {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | if restart { libc::SA_RESTART } else { 0 };
            action.sa_sigaction = handler as usize;
            action.sa_mask = mask;
            let mut old: libc::sigaction = std::mem::zeroed();
            if sigaction(sig, &action, &mut old) != 0 {
                return Err(format!("sigaction({sig}) failed: {}", std::io::Error::last_os_error()));
            }
            Ok(old)
        };
        install(SIGNAL_ENTER, super::asm::nexium_nce_sigusr2_handler, false)?;
        install(SIGNAL_BREAK, super::asm::nexium_nce_sigurg_handler, false)?;
        let segv = install(libc::SIGSEGV, super::asm::nexium_nce_sigsegv_handler, true)?;
        let bus = install(libc::SIGBUS, super::asm::nexium_nce_sigbus_handler, false)?;
        let trap = install(libc::SIGTRAP, super::asm::nexium_nce_sigtrap_handler, false)?;
        let ill = install(libc::SIGILL, super::asm::nexium_nce_sigill_handler, false)?;
        let _ = SAVED.set(SavedActions { segv, bus, trap, ill });
    }
    Ok(())
}

pub fn ensure_alt_stack() {
    thread_local! {
        static INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    INSTALLED.with(|installed| {
        if installed.get() {
            return;
        }
        const STACK_SIZE: usize = 256 * 1024;
        let stack: Box<[u8]> = vec![0u8; STACK_SIZE].into_boxed_slice();
        let ptr = Box::into_raw(stack) as *mut u8;
        unsafe {
            let ss = libc::stack_t {
                ss_sp: ptr as *mut libc::c_void,
                ss_flags: 0,
                ss_size: STACK_SIZE,
            };
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }
        installed.set(true);
    });
}

pub fn core_state_of(ctx: &GuestContext) -> Option<&CoreState> {
    unsafe { ctx.core.as_ref() }
}

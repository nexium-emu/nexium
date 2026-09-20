use crate::nce_layout::*;
use std::sync::atomic::{AtomicU32, AtomicU64};

#[repr(C, align(16))]
pub struct HostContext {
    pub regs: [u64; 12],
    pub vregs: [u128; 8],
    pub sp: u64,
    pub tpidr_el0: u64,
}

#[repr(C, align(16))]
pub struct GuestContext {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub fpcr: u32,
    pub fpsr: u32,
    pub v: [u128; 32],
    pub pstate: u32,
    pub pad0: [u32; 3],
    pub host: HostContext,
    pub tpidrro_el0: u64,
    pub tpidr_el0: u64,
    pub esr: AtomicU64,
    pub nzcv: u32,
    pub svc: u32,
    pub core: *mut CoreState,
    pub pad1: u64,
}

#[repr(C)]
pub struct NativeExecutionParameters {
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub native_context: *mut GuestContext,
    pub lock: AtomicU32,
    pub is_running: AtomicU32,
    pub magic: u32,
    pub pad: [u32; 3],
}

#[repr(C)]
pub struct FaultRecord {
    pub valid: AtomicU32,
    pub is_write: AtomicU32,
    pub pc: AtomicU64,
    pub addr: AtomicU64,
    pub lr: AtomicU64,
    pub sp: AtomicU64,
    pub regs: [AtomicU64; 31],
}

#[repr(C)]
pub struct CoreState {
    pub continue_on_null: AtomicU32,
    pub null_skips: AtomicU32,
    pub null_skip_limit: u32,
    pub tid: std::sync::atomic::AtomicI32,
    pub fault: FaultRecord,
    pub entries_trampoline: AtomicU64,
    pub entries_signal: AtomicU64,
    pub exits_svc: AtomicU64,
    pub exits_break: AtomicU64,
    pub exits_fault: AtomicU64,
    pub exits_idle: AtomicU64,
}

const _: () = {
    assert!(std::mem::offset_of!(GuestContext, sp) == GUEST_SP);
    assert!(std::mem::offset_of!(GuestContext, pc) == GUEST_PC);
    assert!(std::mem::offset_of!(GuestContext, fpcr) == GUEST_FPCR);
    assert!(std::mem::offset_of!(GuestContext, fpsr) == GUEST_FPSR);
    assert!(std::mem::offset_of!(GuestContext, v) == GUEST_V);
    assert!(std::mem::offset_of!(GuestContext, pstate) == GUEST_PSTATE);
    assert!(std::mem::offset_of!(GuestContext, host) == GUEST_HOST_CTX);
    assert!(std::mem::offset_of!(GuestContext, tpidrro_el0) == GUEST_TPIDRRO);
    assert!(std::mem::offset_of!(GuestContext, tpidr_el0) == GUEST_TPIDR);
    assert!(std::mem::offset_of!(GuestContext, esr) == GUEST_ESR);
    assert!(std::mem::offset_of!(GuestContext, nzcv) == GUEST_NZCV);
    assert!(std::mem::offset_of!(GuestContext, svc) == GUEST_SVC);
    assert!(std::mem::offset_of!(GuestContext, core) == GUEST_CTX_SIZE);
    assert!(std::mem::offset_of!(HostContext, regs) == HOST_REGS);
    assert!(std::mem::offset_of!(HostContext, vregs) == HOST_VREGS);
    assert!(std::mem::offset_of!(HostContext, sp) == HOST_SP);
    assert!(std::mem::offset_of!(HostContext, tpidr_el0) == HOST_TPIDR);
    assert!(std::mem::size_of::<HostContext>() == HOST_CTX_SIZE);
    assert!(std::mem::offset_of!(NativeExecutionParameters, tpidr_el0) == NEP_TPIDR);
    assert!(std::mem::offset_of!(NativeExecutionParameters, tpidrro_el0) == NEP_TPIDRRO);
    assert!(std::mem::offset_of!(NativeExecutionParameters, native_context) == NEP_NATIVE_CONTEXT);
    assert!(std::mem::offset_of!(NativeExecutionParameters, lock) == NEP_LOCK);
    assert!(std::mem::offset_of!(NativeExecutionParameters, is_running) == NEP_IS_RUNNING);
    assert!(std::mem::offset_of!(NativeExecutionParameters, magic) == NEP_MAGIC);
    assert!(std::mem::size_of::<NativeExecutionParameters>() == NEP_SIZE);
};

impl GuestContext {
    pub fn new(core: *mut CoreState) -> Self {
        Self {
            x: [0; 31],
            sp: 0,
            pc: 0,
            fpcr: 0,
            fpsr: 0,
            v: [0; 32],
            pstate: 0,
            pad0: [0; 3],
            host: HostContext {
                regs: [0; 12],
                vregs: [0; 8],
                sp: 0,
                tpidr_el0: 0,
            },
            tpidrro_el0: 0,
            tpidr_el0: 0,
            esr: AtomicU64::new(0),
            nzcv: 0,
            svc: 0,
            core,
            pad1: 0,
        }
    }

    pub fn reset_thread_state(&mut self) {
        self.x = [0; 31];
        self.sp = 0;
        self.pc = 0;
        self.fpcr = 0;
        self.fpsr = 0;
        self.v = [0; 32];
        self.pstate = 0;
        self.nzcv = 0;
        self.tpidr_el0 = 0;
    }
}

impl NativeExecutionParameters {
    pub fn new() -> Self {
        Self {
            tpidr_el0: 0,
            tpidrro_el0: 0,
            native_context: std::ptr::null_mut(),
            lock: AtomicU32::new(LOCK_UNLOCKED),
            is_running: AtomicU32::new(0),
            magic: TLS_MAGIC,
            pad: [0; 3],
        }
    }
}

impl FaultRecord {
    pub const fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            valid: AtomicU32::new(0),
            is_write: AtomicU32::new(0),
            pc: AtomicU64::new(0),
            addr: AtomicU64::new(0),
            lr: AtomicU64::new(0),
            sp: AtomicU64::new(0),
            regs: [ZERO; 31],
        }
    }
}

impl CoreState {
    pub fn new(null_skip_limit: u32) -> Self {
        Self {
            continue_on_null: AtomicU32::new(0),
            null_skips: AtomicU32::new(0),
            null_skip_limit,
            tid: std::sync::atomic::AtomicI32::new(-1),
            fault: FaultRecord::new(),
            entries_trampoline: AtomicU64::new(0),
            entries_signal: AtomicU64::new(0),
            exits_svc: AtomicU64::new(0),
            exits_break: AtomicU64::new(0),
            exits_fault: AtomicU64::new(0),
            exits_idle: AtomicU64::new(0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NceThreadContext {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub fpcr: u32,
    pub fpsr: u32,
    pub v: [u128; 32],
    pub pstate: u32,
    pub nzcv: u32,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
}

impl NceThreadContext {
    pub fn capture(ctx: &GuestContext) -> Self {
        Self {
            x: ctx.x,
            sp: ctx.sp,
            pc: ctx.pc,
            fpcr: ctx.fpcr,
            fpsr: ctx.fpsr,
            v: ctx.v,
            pstate: ctx.pstate,
            nzcv: ctx.nzcv,
            tpidr_el0: ctx.tpidr_el0,
            tpidrro_el0: ctx.tpidrro_el0,
        }
    }

    pub fn apply(&self, ctx: &mut GuestContext) {
        ctx.x = self.x;
        ctx.sp = self.sp;
        ctx.pc = self.pc;
        ctx.fpcr = self.fpcr;
        ctx.fpsr = self.fpsr;
        ctx.v = self.v;
        ctx.pstate = self.pstate;
        ctx.nzcv = self.nzcv;
        ctx.tpidr_el0 = self.tpidr_el0;
        ctx.tpidrro_el0 = self.tpidrro_el0;
    }
}

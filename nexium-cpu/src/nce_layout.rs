pub const GUEST_X: usize = 0x0;
pub const GUEST_SP: usize = 0xF8;
pub const GUEST_PC: usize = 0x100;
pub const GUEST_FPCR: usize = 0x108;
pub const GUEST_FPSR: usize = 0x10C;
pub const GUEST_V: usize = 0x110;
pub const GUEST_PSTATE: usize = 0x310;
pub const GUEST_HOST_CTX: usize = 0x320;
pub const HOST_REGS: usize = 0x0;
pub const HOST_VREGS: usize = 0x60;
pub const HOST_SP: usize = 0xE0;
pub const HOST_TPIDR: usize = 0xE8;
pub const HOST_CTX_SIZE: usize = 0xF0;
pub const GUEST_TPIDRRO: usize = 0x410;
pub const GUEST_TPIDR: usize = 0x418;
pub const GUEST_ESR: usize = 0x420;
pub const GUEST_NZCV: usize = 0x428;
pub const GUEST_SVC: usize = 0x42C;
pub const GUEST_CTX_SIZE: usize = 0x430;

pub const NEP_TPIDR: usize = 0x0;
pub const NEP_TPIDRRO: usize = 0x8;
pub const NEP_NATIVE_CONTEXT: usize = 0x10;
pub const NEP_LOCK: usize = 0x18;
pub const NEP_IS_RUNNING: usize = 0x1C;
pub const NEP_MAGIC: usize = 0x20;
pub const NEP_SIZE: usize = 0x30;

pub const TLS_MAGIC: u32 = 0x4E45_5855;
pub const LOCK_LOCKED: u32 = 0;
pub const LOCK_UNLOCKED: u32 = 1;

pub const HALT_SUPERVISOR_CALL: u64 = 1 << 0;
pub const HALT_BREAK_LOOP: u64 = 1 << 1;
pub const HALT_PREFETCH_ABORT: u64 = 1 << 2;
pub const HALT_DATA_ABORT: u64 = 1 << 3;
pub const HALT_BREAKPOINT: u64 = 1 << 4;
pub const HALT_ALIGNMENT: u64 = 1 << 5;

pub const EXIT_STUB_BRK_IMM: u16 = 0xF07;
pub const GUEST_CNTFRQ_HZ: u64 = 19_200_000;

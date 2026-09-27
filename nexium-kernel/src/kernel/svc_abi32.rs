use std::sync::OnceLock;

pub const REG_COUNT: usize = 8;

const PLAN_TABLE_LEN: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Word,
    Dword,
    Pointer,
    ThreadContext,
    PhysicalMemoryInfo,
    SecureMonitorArguments,
}

use Kind::{Dword, PhysicalMemoryInfo, Pointer, SecureMonitorArguments, ThreadContext, Word};

impl Kind {
    const fn size(self, wide: bool) -> usize {
        match self {
            Word => 4,
            Dword => 8,
            Pointer => {
                if wide {
                    8
                } else {
                    4
                }
            }
            ThreadContext => {
                if wide {
                    32
                } else {
                    16
                }
            }
            PhysicalMemoryInfo => {
                if wide {
                    24
                } else {
                    16
                }
            }
            SecureMonitorArguments => {
                if wide {
                    64
                } else {
                    32
                }
            }
        }
    }

    fn layout(self) -> &'static [(u8, bool)] {
        match self {
            Word | Pointer => &[(0, false)],
            Dword => &[(0, false), (0, true)],
            ThreadContext => &[(0, false), (1, false), (2, false), (3, false)],
            PhysicalMemoryInfo => &[(0, false), (0, true), (1, false), (2, false)],
            SecureMonitorArguments => &[
                (0, false),
                (1, false),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, false),
                (7, false),
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct Param {
    kind: Kind,
    out: bool,
}

const fn arg(kind: Kind) -> Param {
    Param { kind, out: false }
}

const fn out(kind: Kind) -> Param {
    Param { kind, out: true }
}

struct Decl {
    id: u16,
    ret: Option<Kind>,
    params: &'static [Param],
}

const VOID: Option<Kind> = None;
const RESULT: Option<Kind> = Some(Word);
const INT32: Option<Kind> = Some(Word);
const INT64: Option<Kind> = Some(Dword);

static TABLE: &[Decl] = &[
    Decl {
        id: 0x01,
        ret: RESULT,
        params: &[out(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x02,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x03,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x04,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x05,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x06,
        ret: RESULT,
        params: &[arg(Pointer), out(Word), arg(Pointer)],
    },
    Decl {
        id: 0x07,
        ret: VOID,
        params: &[],
    },
    Decl {
        id: 0x08,
        ret: RESULT,
        params: &[
            out(Word),
            arg(Pointer),
            arg(Pointer),
            arg(Pointer),
            arg(Word),
            arg(Word),
        ],
    },
    Decl {
        id: 0x09,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x0A,
        ret: VOID,
        params: &[],
    },
    Decl {
        id: 0x0B,
        ret: VOID,
        params: &[arg(Dword)],
    },
    Decl {
        id: 0x0C,
        ret: RESULT,
        params: &[out(Word), arg(Word)],
    },
    Decl {
        id: 0x0D,
        ret: RESULT,
        params: &[arg(Word), arg(Word)],
    },
    Decl {
        id: 0x0E,
        ret: RESULT,
        params: &[out(Word), out(Dword), arg(Word)],
    },
    Decl {
        id: 0x0F,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x10,
        ret: INT32,
        params: &[],
    },
    Decl {
        id: 0x11,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x12,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x13,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x14,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x15,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x16,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x17,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x18,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x19,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x1A,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x1B,
        ret: RESULT,
        params: &[arg(Pointer)],
    },
    Decl {
        id: 0x1C,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x1D,
        ret: VOID,
        params: &[arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x1E,
        ret: INT64,
        params: &[],
    },
    Decl {
        id: 0x1F,
        ret: RESULT,
        params: &[out(Word), arg(Pointer)],
    },
    Decl {
        id: 0x20,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x21,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x22,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x23,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x24,
        ret: RESULT,
        params: &[out(Dword), arg(Word)],
    },
    Decl {
        id: 0x25,
        ret: RESULT,
        params: &[out(Dword), arg(Word)],
    },
    Decl {
        id: 0x26,
        ret: VOID,
        params: &[arg(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x27,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x28,
        ret: VOID,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x29,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x2A,
        ret: VOID,
        params: &[],
    },
    Decl {
        id: 0x2B,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x2C,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x2D,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x2E,
        ret: RESULT,
        params: &[out(ThreadContext), out(Dword), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x2F,
        ret: RESULT,
        params: &[out(ThreadContext), out(Pointer), out(Word)],
    },
    Decl {
        id: 0x30,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x31,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x32,
        ret: RESULT,
        params: &[arg(Word), arg(Word)],
    },
    Decl {
        id: 0x33,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x34,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x35,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x36,
        ret: VOID,
        params: &[],
    },
    Decl {
        id: 0x37,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x39,
        ret: RESULT,
        params: &[out(Word), arg(Word)],
    },
    Decl {
        id: 0x3A,
        ret: RESULT,
        params: &[
            out(Word),
            arg(Word),
            arg(Dword),
            arg(Pointer),
            arg(Word),
            arg(Word),
        ],
    },
    Decl {
        id: 0x3C,
        ret: VOID,
        params: &[arg(Word), arg(Dword), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x3D,
        ret: VOID,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x40,
        ret: RESULT,
        params: &[out(Word), out(Word), arg(Word), arg(Pointer)],
    },
    Decl {
        id: 0x41,
        ret: RESULT,
        params: &[out(Word), arg(Word)],
    },
    Decl {
        id: 0x42,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x43,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x44,
        ret: RESULT,
        params: &[
            out(Word),
            arg(Pointer),
            arg(Pointer),
            arg(Pointer),
            arg(Word),
            arg(Word),
            arg(Dword),
        ],
    },
    Decl {
        id: 0x45,
        ret: RESULT,
        params: &[out(Word), out(Word)],
    },
    Decl {
        id: 0x46,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x47,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x48,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x49,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x4A,
        ret: RESULT,
        params: &[arg(Pointer)],
    },
    Decl {
        id: 0x4B,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x4C,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Dword), arg(Dword), arg(Word)],
    },
    Decl {
        id: 0x4D,
        ret: VOID,
        params: &[],
    },
    Decl {
        id: 0x4E,
        ret: RESULT,
        params: &[out(Word), arg(Dword), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x4F,
        ret: RESULT,
        params: &[arg(Word), arg(Word)],
    },
    Decl {
        id: 0x50,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x51,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x52,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x53,
        ret: RESULT,
        params: &[out(Word), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x54,
        ret: RESULT,
        params: &[out(PhysicalMemoryInfo), arg(Pointer)],
    },
    Decl {
        id: 0x55,
        ret: RESULT,
        params: &[out(Pointer), out(Pointer), arg(Dword), arg(Pointer)],
    },
    Decl {
        id: 0x56,
        ret: RESULT,
        params: &[out(Word), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x57,
        ret: RESULT,
        params: &[arg(Word), arg(Word)],
    },
    Decl {
        id: 0x58,
        ret: RESULT,
        params: &[arg(Word), arg(Word)],
    },
    Decl {
        id: 0x59,
        ret: RESULT,
        params: &[
            arg(Word),
            arg(Word),
            arg(Dword),
            arg(Pointer),
            arg(Dword),
            arg(Word),
        ],
    },
    Decl {
        id: 0x5A,
        ret: RESULT,
        params: &[
            arg(Word),
            arg(Word),
            arg(Dword),
            arg(Pointer),
            arg(Dword),
            arg(Word),
        ],
    },
    Decl {
        id: 0x5C,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Dword), arg(Pointer), arg(Dword)],
    },
    Decl {
        id: 0x5D,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x5E,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x5F,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x60,
        ret: RESULT,
        params: &[out(Word), arg(Dword)],
    },
    Decl {
        id: 0x61,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x62,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x63,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x64,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x65,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x66,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x67,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Dword), arg(Word)],
    },
    Decl {
        id: 0x68,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x69,
        ret: RESULT,
        params: &[arg(Pointer), out(Word), arg(Word), arg(Pointer)],
    },
    Decl {
        id: 0x6A,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x6B,
        ret: RESULT,
        params: &[arg(Word), arg(Pointer), arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x6C,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x6D,
        ret: RESULT,
        params: &[out(Dword), out(Word), arg(Word), arg(Dword), arg(Word)],
    },
    Decl {
        id: 0x6F,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x70,
        ret: RESULT,
        params: &[out(Word), out(Word), arg(Word), arg(Word), arg(Pointer)],
    },
    Decl {
        id: 0x71,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x72,
        ret: RESULT,
        params: &[out(Word), arg(Word)],
    },
    Decl {
        id: 0x73,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword), arg(Word)],
    },
    Decl {
        id: 0x74,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Dword), arg(Pointer)],
    },
    Decl {
        id: 0x75,
        ret: RESULT,
        params: &[arg(Pointer), arg(Word), arg(Dword), arg(Pointer)],
    },
    Decl {
        id: 0x76,
        ret: RESULT,
        params: &[arg(Pointer), out(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x77,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x78,
        ret: RESULT,
        params: &[arg(Word), arg(Dword), arg(Dword), arg(Dword)],
    },
    Decl {
        id: 0x79,
        ret: RESULT,
        params: &[out(Word), arg(Pointer), arg(Pointer), arg(Word)],
    },
    Decl {
        id: 0x7A,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x7B,
        ret: RESULT,
        params: &[arg(Word)],
    },
    Decl {
        id: 0x7C,
        ret: RESULT,
        params: &[out(Dword), arg(Word), arg(Word)],
    },
    Decl {
        id: 0x7D,
        ret: RESULT,
        params: &[out(Word)],
    },
    Decl {
        id: 0x7E,
        ret: RESULT,
        params: &[arg(Word), arg(Word), arg(Dword)],
    },
    Decl {
        id: 0x7F,
        ret: VOID,
        params: &[arg(SecureMonitorArguments)],
    },
    Decl {
        id: 0x90,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
    Decl {
        id: 0x91,
        ret: RESULT,
        params: &[arg(Pointer), arg(Pointer)],
    },
];

#[derive(Clone, Copy, Default)]
struct RegList {
    regs: [u8; REG_COUNT],
    len: usize,
}

impl RegList {
    fn push(&mut self, reg: u8) -> Option<()> {
        let slot = self.regs.get_mut(self.len)?;
        *slot = reg;
        self.len += 1;
        Some(())
    }

    fn get(&self, index: usize) -> Option<u8> {
        self.regs[..self.len].get(index).copied()
    }
}

struct RegisterFile {
    byte_size: usize,
    parameter_count: usize,
    used: [bool; REG_COUNT],
}

impl RegisterFile {
    fn narrow() -> Self {
        Self {
            byte_size: 4,
            parameter_count: 4,
            used: [false; REG_COUNT],
        }
    }

    fn wide() -> Self {
        Self {
            byte_size: 8,
            parameter_count: 8,
            used: [false; REG_COUNT],
        }
    }

    fn add_single(&mut self, n: usize) -> Option<(u8, usize)> {
        if n >= self.parameter_count {
            let reg = self.used.iter().position(|used| !*used)?;
            self.used[reg] = true;
            Some((reg as u8, 0))
        } else {
            let slot = self.used.get_mut(n)?;
            if *slot {
                return None;
            }
            *slot = true;
            Some((n as u8, 1))
        }
    }

    fn add(&mut self, ngrn: &mut usize, size: usize, align: bool) -> Option<RegList> {
        let mut regs = RegList::default();
        if size <= self.byte_size {
            let (reg, increment) = self.add_single(*ngrn)?;
            regs.push(reg)?;
            *ngrn += increment;
        } else {
            let mut increment = if align { *ngrn % 2 } else { 0 };
            for _ in 0..size / self.byte_size {
                let (reg, single) = self.add_single(*ngrn + increment)?;
                increment += single;
                regs.push(reg)?;
            }
            *ngrn += increment;
        }
        Some(regs)
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Move {
    r: u8,
    x: u8,
    high: bool,
}

impl Move {
    fn shift(self) -> u32 {
        if self.high {
            32
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct MoveList {
    moves: [Move; REG_COUNT],
    len: usize,
}

impl MoveList {
    fn push(&mut self, m: Move) -> Option<()> {
        let slot = self.moves.get_mut(self.len)?;
        *slot = m;
        self.len += 1;
        Some(())
    }

    fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Plan {
    gather: MoveList,
    scatter: MoveList,
}

fn push_moves(kind: Kind, narrow: &RegList, wide: &RegList, into: &mut MoveList) -> Option<()> {
    let layout = kind.layout();
    if layout.len() != narrow.len {
        return None;
    }
    for (word, &(slot, high)) in layout.iter().enumerate() {
        let r = narrow.get(word)?;
        let x = wide.get(usize::from(slot))?;
        into.push(Move { r, x, high })?;
    }
    Some(())
}

fn build_plan(decl: &Decl) -> Option<Plan> {
    let mut plan = Plan::default();
    let mut narrow = RegisterFile::narrow();
    let mut wide = RegisterFile::wide();
    let mut ngrn_narrow = 0;
    let mut ngrn_wide = 0;
    for param in decl.params {
        if param.out {
            ngrn_narrow += 1;
            ngrn_wide += 1;
            continue;
        }
        let r = narrow.add(&mut ngrn_narrow, param.kind.size(false), true)?;
        let x = wide.add(&mut ngrn_wide, param.kind.size(true), true)?;
        push_moves(param.kind, &r, &x, &mut plan.gather)?;
    }
    let mut narrow = RegisterFile::narrow();
    let mut wide = RegisterFile::wide();
    let mut ngrn_narrow = 0;
    let mut ngrn_wide = 0;
    let outputs = decl
        .ret
        .into_iter()
        .chain(decl.params.iter().filter(|p| p.out).map(|p| p.kind));
    for kind in outputs {
        let r = narrow.add(&mut ngrn_narrow, kind.size(false), false)?;
        let x = wide.add(&mut ngrn_wide, kind.size(true), false)?;
        push_moves(kind, &r, &x, &mut plan.scatter)?;
    }
    Some(plan)
}

fn plans() -> &'static [Option<Plan>; PLAN_TABLE_LEN] {
    static PLANS: OnceLock<[Option<Plan>; PLAN_TABLE_LEN]> = OnceLock::new();
    PLANS.get_or_init(|| {
        let mut plans = [None; PLAN_TABLE_LEN];
        for decl in TABLE {
            if let Some(slot) = plans.get_mut(usize::from(decl.id)) {
                *slot = build_plan(decl);
            }
        }
        plans
    })
}

fn plan(imm: u16) -> Option<&'static Plan> {
    plans().get(usize::from(imm))?.as_ref()
}

pub fn gather(imm: u16, r: &[u32; REG_COUNT]) -> [u64; REG_COUNT] {
    let mut x = [0u64; REG_COUNT];
    match plan(imm) {
        Some(plan) => {
            for m in plan.gather.as_slice() {
                x[usize::from(m.x)] |= u64::from(r[usize::from(m.r)]) << m.shift();
            }
        }
        None => {
            for (x, r) in x.iter_mut().zip(r) {
                *x = u64::from(*r);
            }
        }
    }
    x
}

pub fn scatter(imm: u16, x: &[u64; REG_COUNT], r: &mut [u32; REG_COUNT]) {
    match plan(imm) {
        Some(plan) => {
            for m in plan.scatter.as_slice() {
                r[usize::from(m.r)] = (x[usize::from(m.x)] >> m.shift()) as u32;
            }
        }
        None => r[0] = x[0] as u32,
    }
}

pub fn is_known(imm: u16) -> bool {
    plan(imm).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> [u32; REG_COUNT] {
        let mut r = [0u32; REG_COUNT];
        for (i, v) in r.iter_mut().enumerate() {
            *v = 0x1111_1111 * (i as u32 + 1);
        }
        r
    }

    fn expect_gather(imm: u16, spec: &[(usize, usize, Option<usize>)]) {
        let r = inputs();
        let mut expected = [0u64; REG_COUNT];
        for &(x, low, high) in spec {
            let high = high.map_or(0, |high| u64::from(r[high]) << 32);
            expected[x] = u64::from(r[low]) | high;
        }
        assert_eq!(gather(imm, &r), expected, "svc {imm:#04x}");
    }

    fn expect_scatter(imm: u16, spec: &[(usize, usize, bool)]) {
        let mut x = [0u64; REG_COUNT];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((0xA0 + i as u64) << 32) | (0xB0 + i as u64);
        }
        let mut r = [0u32; REG_COUNT];
        for (i, v) in r.iter_mut().enumerate() {
            *v = 0xDEAD_0000 | i as u32;
        }
        let mut expected = r;
        for &(reg, slot, high) in spec {
            let shift = if high { 32 } else { 0 };
            expected[reg] = (x[slot] >> shift) as u32;
        }
        scatter(imm, &x, &mut r);
        assert_eq!(r, expected, "svc {imm:#04x}");
    }

    #[test]
    fn set_heap_size() {
        expect_gather(0x01, &[(1, 1, None)]);
        expect_scatter(0x01, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn create_thread() {
        expect_gather(
            0x08,
            &[
                (1, 1, None),
                (2, 2, None),
                (3, 3, None),
                (4, 0, None),
                (5, 4, None),
            ],
        );
        expect_scatter(0x08, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn sleep_thread() {
        expect_gather(0x0B, &[(0, 0, Some(1))]);
        expect_scatter(0x0B, &[]);
    }

    #[test]
    fn get_thread_core_mask() {
        expect_gather(0x0E, &[(2, 2, None)]);
        expect_scatter(
            0x0E,
            &[(0, 0, false), (1, 1, false), (2, 2, false), (3, 2, true)],
        );
    }

    #[test]
    fn set_thread_core_mask() {
        expect_gather(0x0F, &[(0, 0, None), (1, 1, None), (2, 2, Some(3))]);
        expect_scatter(0x0F, &[(0, 0, false)]);
    }

    #[test]
    fn wait_synchronization() {
        expect_gather(0x18, &[(1, 1, None), (2, 2, None), (3, 0, Some(3))]);
        expect_scatter(0x18, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn wait_process_wide_key_atomic() {
        expect_gather(
            0x1C,
            &[(0, 0, None), (1, 1, None), (2, 2, None), (3, 3, Some(4))],
        );
        expect_scatter(0x1C, &[(0, 0, false)]);
    }

    #[test]
    fn get_system_tick() {
        expect_gather(0x1E, &[]);
        expect_scatter(0x1E, &[(0, 0, false), (1, 0, true)]);
    }

    #[test]
    fn get_thread_id() {
        expect_gather(0x25, &[(1, 1, None)]);
        expect_scatter(0x25, &[(0, 0, false), (1, 1, false), (2, 1, true)]);
    }

    #[test]
    fn get_process_id() {
        expect_gather(0x24, &[(1, 1, None)]);
        expect_scatter(0x24, &[(0, 0, false), (1, 1, false), (2, 1, true)]);
    }

    #[test]
    fn break_svc() {
        expect_gather(0x26, &[(0, 0, None), (1, 1, None), (2, 2, None)]);
        expect_scatter(0x26, &[]);
    }

    #[test]
    fn output_debug_string() {
        expect_gather(0x27, &[(0, 0, None), (1, 1, None)]);
        expect_scatter(0x27, &[(0, 0, false)]);
    }

    #[test]
    fn get_info() {
        expect_gather(0x29, &[(1, 1, None), (2, 2, None), (3, 0, Some(3))]);
        expect_scatter(0x29, &[(0, 0, false), (1, 1, false), (2, 1, true)]);
    }

    #[test]
    fn query_memory() {
        expect_gather(0x06, &[(0, 0, None), (2, 2, None)]);
        expect_scatter(0x06, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn get_last_thread_info() {
        expect_gather(0x2F, &[]);
        expect_scatter(
            0x2F,
            &[
                (0, 0, false),
                (1, 1, false),
                (2, 2, false),
                (3, 3, false),
                (4, 4, false),
                (5, 5, false),
                (6, 6, false),
            ],
        );
    }

    #[test]
    fn get_thread_context3() {
        expect_gather(0x33, &[(0, 0, None), (1, 1, None)]);
        expect_scatter(0x33, &[(0, 0, false)]);
    }

    #[test]
    fn wait_for_address() {
        expect_gather(
            0x34,
            &[(0, 0, None), (1, 1, None), (2, 2, None), (3, 3, Some(4))],
        );
        expect_scatter(0x34, &[(0, 0, false)]);
    }

    #[test]
    fn signal_to_address() {
        expect_gather(
            0x35,
            &[(0, 0, None), (1, 1, None), (2, 2, None), (3, 3, None)],
        );
        expect_scatter(0x35, &[(0, 0, false)]);
    }

    #[test]
    fn create_transfer_memory() {
        expect_gather(0x15, &[(1, 1, None), (2, 2, None), (3, 3, None)]);
        expect_scatter(0x15, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn reply_and_receive() {
        expect_gather(
            0x43,
            &[(1, 1, None), (2, 2, None), (3, 3, None), (4, 0, Some(4))],
        );
        expect_scatter(0x43, &[(0, 0, false), (1, 1, false)]);
    }

    #[test]
    fn get_current_processor_number() {
        expect_gather(0x10, &[]);
        expect_scatter(0x10, &[(0, 0, false)]);
    }

    #[test]
    fn query_physical_address() {
        expect_gather(0x54, &[(1, 1, None)]);
        expect_scatter(
            0x54,
            &[
                (0, 0, false),
                (1, 1, false),
                (2, 1, true),
                (3, 2, false),
                (4, 3, false),
            ],
        );
    }

    #[test]
    fn get_debug_future_thread_info() {
        expect_gather(0x2E, &[(2, 2, None), (3, 0, Some(1))]);
        expect_scatter(
            0x2E,
            &[
                (0, 0, false),
                (1, 1, false),
                (2, 2, false),
                (3, 3, false),
                (4, 4, false),
                (5, 5, false),
                (6, 5, true),
            ],
        );
    }

    #[test]
    fn call_secure_monitor() {
        expect_gather(
            0x7F,
            &[
                (0, 0, None),
                (1, 1, None),
                (2, 2, None),
                (3, 3, None),
                (4, 4, None),
                (5, 5, None),
                (6, 6, None),
                (7, 7, None),
            ],
        );
        expect_scatter(0x7F, &[]);
    }

    #[test]
    fn kernel_debug() {
        expect_gather(
            0x3C,
            &[
                (0, 0, None),
                (1, 2, Some(3)),
                (2, 1, Some(4)),
                (3, 5, Some(6)),
            ],
        );
        expect_scatter(0x3C, &[]);
    }

    #[test]
    fn unknown_is_identity() {
        for imm in [
            0x00u16, 0x38, 0x3B, 0x3E, 0x3F, 0x5B, 0x6E, 0x80, 0x92, 0xFF, 0x100, 0xFFFF,
        ] {
            assert!(!is_known(imm), "svc {imm:#04x}");
            let r = inputs();
            let x = gather(imm, &r);
            for i in 0..REG_COUNT {
                assert_eq!(x[i], u64::from(r[i]));
            }
            let x = [0xAAAA_BBBB_CCCC_DDDDu64; REG_COUNT];
            let mut r = inputs();
            let mut expected = r;
            expected[0] = 0xCCCC_DDDD;
            scatter(imm, &x, &mut r);
            assert_eq!(r, expected);
        }
    }

    #[test]
    fn every_table_entry_has_a_plan() {
        for decl in TABLE {
            assert!(is_known(decl.id), "svc {:#04x}", decl.id);
            let plan = plan(decl.id).expect("plan");
            for list in [&plan.gather, &plan.scatter] {
                let moves = list.as_slice();
                for (i, a) in moves.iter().enumerate() {
                    assert!(usize::from(a.r) < REG_COUNT);
                    assert!(usize::from(a.x) < REG_COUNT);
                    for b in &moves[i + 1..] {
                        assert_ne!(a.r, b.r, "svc {:#04x} reuses r{}", decl.id, a.r);
                        assert!(
                            (a.x, a.high) != (b.x, b.high),
                            "svc {:#04x} reuses x{} half",
                            decl.id,
                            a.x
                        );
                    }
                }
            }
        }
        let ids: Vec<u16> = TABLE.iter().map(|decl| decl.id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids, sorted);
    }
}

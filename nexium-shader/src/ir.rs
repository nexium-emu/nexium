

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {

    Inst(ValueId),

    GprIn(u8),

    ImmU32(u32),

    ImmF32(f32),

    Zero,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Inst(v) => write!(f, "v{}", v.0),
            Value::GprIn(r) => write!(f, "R{r}_in"),
            Value::ImmU32(v) => write!(f, "{v:#x}"),
            Value::ImmF32(v) => write!(f, "{v}f"),
            Value::Zero => write!(f, "0"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FComp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    T,
}

impl FComp {
    pub fn from_bits(v: u64) -> Self {
        match v & 0xF {
            0  => Self::F,
            1  => Self::Lt,
            2  => Self::Eq,
            3  => Self::Le,
            4  => Self::Gt,
            5  => Self::Ne,
            6  => Self::Ge,
            7  => Self::Num,
            8  => Self::Nan,
            9  => Self::Ltu,
            10 => Self::Equ,
            11 => Self::Leu,
            12 => Self::Gtu,
            13 => Self::Neu,
            14 => Self::Geu,
            _  => Self::T,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp { And, Or, Xor }

impl BoolOp {
    pub fn from_bits(v: u64) -> Self {
        match v & 3 {
            0 => Self::And,
            1 => Self::Or,
            _ => Self::Xor,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MufuFunc {
    Cos,
    Sin,
    Ex2,
    Lg2,
    Rcp,
    Rsq,
    Rcp64h,
    Rsq64h,
    Sqrt,
    Unknown(u8),
}

impl MufuFunc {
    pub fn from_bits(v: u32) -> Self {
        match v & 0xF {
            0 => Self::Cos,
            1 => Self::Sin,
            2 => Self::Ex2,
            3 => Self::Lg2,
            4 => Self::Rcp,
            5 => Self::Rsq,
            6 => Self::Rcp64h,
            7 => Self::Rsq64h,
            8 => Self::Sqrt,
            n => Self::Unknown(n as u8),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cos => "cos",
            Self::Sin => "sin",
            Self::Ex2 => "ex2",
            Self::Lg2 => "lg2",
            Self::Rcp => "rcp",
            Self::Rsq => "rsq",
            Self::Rcp64h => "rcp_64h",
            Self::Rsq64h => "rsq_64h",
            Self::Sqrt => "sqrt",
            Self::Unknown(_) => "?",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Predicate {
    pub idx: u8,
    pub negate: bool,
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negate {
            write!(f, "@!P{}", self.idx)
        } else {
            write!(f, "@P{}", self.idx)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Op {

    Mov(Value),

    FMul { a: Value, b: Value },

    FAdd { a: Value, b: Value },

    FFma { a: Value, b: Value, c: Value },

    MultiFunc { src: Value, func: MufuFunc },

    LoadCbuf { binding: u8, byte_offset: u32 },

    LoadAttr { slot: u32 },

    StoreAttr { slot: u32, src: Value },

    InterpAttr { slot: u32, perspective: Value },

    SampleTex { tex_id: u32, u: Value, v: Value, component: u8 },

    FSetPred {
        cmp: FComp,
        bop: BoolOp,
        src_a: Value,
        src_b: Value,
        neg_a: bool,
        abs_a: bool,
        neg_b: bool,
        abs_b: bool,
        src_pred: u8,
        src_pred_inv: bool,
        dest_p: u8,
        dest_np: u8,
    },

    Kill,

    Phi { sources: Vec<(super::cfg::BlockId, Value)> },

    Exit,

    Unimplemented {
        opcode: super::opcodes::Opcode,
        raw: u64,
    },
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Op,

    pub result: Option<ValueId>,

    pub dest_reg: Option<u8>,

    pub pred: Option<Predicate>,
}

#[derive(Default, Debug)]
pub struct Program {
    pub instructions: Vec<Inst>,
    next_value: u32,

    pub exit_reg_state: Option<HashMap<u8, Value>>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_offset(start: u32) -> Self {
        Self {
            instructions: Vec::new(),
            next_value: start,
            exit_reg_state: None,
        }
    }

    pub fn next_value_id(&self) -> u32 {
        self.next_value
    }

    pub fn alloc_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value = self.next_value.wrapping_add(1);
        id
    }

    pub fn emit(&mut self, op: Op, dest_reg: Option<u8>) -> ValueId {
        self.emit_pred(op, dest_reg, None)
    }

    pub fn emit_pred(
        &mut self,
        op: Op,
        dest_reg: Option<u8>,
        pred: Option<Predicate>,
    ) -> ValueId {
        let id = self.alloc_value();
        self.instructions.push(Inst {
            op,
            result: Some(id),
            dest_reg,
            pred,
        });
        id
    }

    pub fn emit_void(&mut self, op: Op) {
        self.emit_void_pred(op, None);
    }

    pub fn emit_void_pred(&mut self, op: Op, pred: Option<Predicate>) {
        self.instructions.push(Inst {
            op,
            result: None,
            dest_reg: None,
            pred,
        });
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for inst in &self.instructions {
            inst.fmt_oneline(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Inst {

    pub fn fmt_oneline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pred_tag = match self.pred {
            Some(p) => format!("{p:<5} "),
            None => "      ".to_string(),
        };
        let dest_tag = match (self.result, self.dest_reg) {
            (Some(v), Some(r)) => format!("v{:<3} (R{}) = ", v.0, r),
            (Some(v), None) => format!("v{:<3}      = ", v.0),
            (None, _) => "             ".to_string(),
        };
        write!(f, "{pred_tag}{dest_tag}")?;
        match &self.op {
            Op::Mov(s) => write!(f, "Mov   {s}"),
            Op::FMul { a, b } => write!(f, "FMul  {a}, {b}"),
            Op::FAdd { a, b } => write!(f, "FAdd  {a}, {b}"),
            Op::FFma { a, b, c } => write!(f, "FFma  {a}, {b}, {c}"),
            Op::MultiFunc { src, func } => write!(f, "MFn.{} {src}", func.name()),
            Op::LoadCbuf { binding, byte_offset } => {
                write!(f, "LdCbuf c[{binding:#x}]:{byte_offset:#x}")
            }
            Op::LoadAttr { slot } => write!(f, "LdAttr a[{slot:#x}]"),
            Op::StoreAttr { slot, src } => write!(f, "StAttr a[{slot:#x}], {src}"),
            Op::InterpAttr { slot, perspective } => {
                write!(f, "Interp a[{slot:#x}], persp={perspective}")
            }
            Op::SampleTex { tex_id, u, v, component } => {
                write!(f, "TexSamp t[{tex_id:#x}], ({u}, {v}).{}",
                    ["r","g","b","a"].get(*component as usize).copied().unwrap_or("?"))
            }
            Op::Phi { sources } => {
                write!(f, "Phi   ")?;
                for (i, (bid, v)) in sources.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[{bid}:{v}]")?;
                }
                Ok(())
            }
            Op::FSetPred { cmp, bop, src_a, src_b, dest_p, src_pred, .. } => {
                write!(f, "FSetP.{cmp:?}.{bop:?} P{dest_p}, {src_a}, {src_b}, P{src_pred}")
            }
            Op::Kill => write!(f, "Kill"),
            Op::Exit => write!(f, "Exit"),
            Op::Unimplemented { opcode, raw } => {
                write!(f, "<unimpl {opcode:?} raw={raw:016x}>")
            }
        }
    }
}

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
            0 => Self::F,
            1 => Self::Lt,
            2 => Self::Eq,
            3 => Self::Le,
            4 => Self::Gt,
            5 => Self::Ne,
            6 => Self::Ge,
            7 => Self::Num,
            8 => Self::Nan,
            9 => Self::Ltu,
            10 => Self::Equ,
            11 => Self::Leu,
            12 => Self::Gtu,
            13 => Self::Neu,
            14 => Self::Geu,
            _ => Self::T,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfSwizzle {
    H1H0,
    F32,
    H0H0,
    H1H1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfMerge {
    H1H0,
    F32,
    MrgH0,
    MrgH1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfPrecision {
    None,
    FTZ,
    FMZ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBarrierScope {
    Workgroup,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageDimension {
    D1,
    Buffer,
    D2,
    D3,
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageAtomicOp {
    Add,
    Min,
    Max,
    Increment,
    Decrement,
    And,
    Or,
    Xor,
    Exchange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageAtomicType {
    U32,
    S32,
    Sd32,
}

impl ImageAtomicType {
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::S32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureHandleOrigin {
    Bound {
        cbuf_word_offset: u32,
    },
    Bindless {
        cbuf_binding: u8,
        cbuf_word_offset: u32,
        cbuf_secondary_word_offset: Option<u32>,
    },
}

impl fmt::Display for ImageDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::D1 => "1d",
            Self::Buffer => "buffer",
            Self::D2 => "2d",
            Self::D3 => "3d",
            Self::Cube => "cube",
        })
    }
}

impl fmt::Display for TextureHandleOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bound { cbuf_word_offset } => {
                write!(f, "bound[{:#x}]", cbuf_word_offset * 4)
            }
            Self::Bindless {
                cbuf_binding,
                cbuf_word_offset,
                cbuf_secondary_word_offset,
            } => {
                write!(f, "c[{cbuf_binding}][{:#x}]", cbuf_word_offset * 4)?;
                if let Some(secondary) = cbuf_secondary_word_offset {
                    write!(f, "|c[{cbuf_binding}][{:#x}]", secondary * 4)?;
                }
                Ok(())
            }
        }
    }
}

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
pub enum LogicOp {
    And,
    Or,
    Xor,
    PassB,
}

impl LogicOp {
    pub fn from_bits(v: u64) -> Self {
        match v & 3 {
            0 => Self::And,
            1 => Self::Or,
            2 => Self::Xor,
            _ => Self::PassB,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ICmp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    T,
}

impl ICmp {
    pub fn from_bits(v: u64) -> Self {
        match v & 0x7 {
            0 => Self::F,
            1 => Self::Lt,
            2 => Self::Eq,
            3 => Self::Le,
            4 => Self::Gt,
            5 => Self::Ne,
            6 => Self::Ge,
            _ => Self::T,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubgroupMask {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteMode {
    All,
    Any,
    Equal,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FMods {
    pub neg_a: bool,
    pub abs_a: bool,
    pub neg_b: bool,
    pub abs_b: bool,
    pub neg_c: bool,
    pub sat: bool,
    pub scale: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CbufAddressMode {
    #[default]
    Default,
    Segmented,
}

#[derive(Clone, Debug)]
pub enum Op {
    Mov(Value),

    FMul {
        a: Value,
        b: Value,
        mods: FMods,
    },

    FAdd {
        a: Value,
        b: Value,
        mods: FMods,
    },

    FFma {
        a: Value,
        b: Value,
        c: Value,
        mods: FMods,
    },

    DpdxFine {
        src: Value,
    },

    DpdyFine {
        src: Value,
    },

    YDirection,

    FMin {
        a: Value,
        b: Value,
        mods: FMods,
    },

    FMax {
        a: Value,
        b: Value,
        mods: FMods,
    },

    FMinMaxPred {
        a: Value,
        b: Value,
        mods: FMods,
        pred: u8,
        neg_pred: bool,
    },

    MultiFunc {
        src: Value,
        func: MufuFunc,
    },

    LoadCbuf {
        binding: u8,
        byte_offset: u32,
    },

    LoadCbufIndexed {
        binding: u8,
        byte_offset: u32,
        index: Value,
        address_mode: CbufAddressMode,
    },

    LoadGlobal {
        addr_lo: Value,
        offset: i32,
    },

    StoreGlobal {
        addr_lo: Value,
        offset: i32,
        value: Value,
    },

    GlobalAtomic {
        addr_lo: Value,
        offset: i32,
        value: Value,
        op: ImageAtomicOp,
        is_signed: bool,
    },

    StorageAtomic {
        buffer_index: u32,
        addr_lo: Value,
        base_addr_lo: Value,
        imm: i32,
        value: Value,
        op: ImageAtomicOp,
        is_signed: bool,
        cbuf_binding: u8,
        cbuf_offset: u32,
        align: u32,
    },

    LoadLocal {
        addr: Value,
    },

    StoreLocal {
        addr: Value,
        value: Value,
    },

    LoadShared {
        addr: Value,
    },

    StoreShared {
        addr: Value,
        value: Value,
    },

    SharedAtomic {
        addr: Value,
        value: Value,
        op: ImageAtomicOp,
    },

    WorkgroupBarrier,

    MemoryBarrier {
        scope: MemoryBarrierScope,
    },

    LoadStorage {
        buffer_index: u32,
        addr_lo: Value,
        base_addr_lo: Value,
        imm: i32,
        cbuf_binding: u8,
        cbuf_offset: u32,
        align: u32,
    },

    StoreStorage {
        buffer_index: u32,
        addr_lo: Value,
        base_addr_lo: Value,
        imm: i32,
        value: Value,
        cbuf_binding: u8,
        cbuf_offset: u32,
        align: u32,
    },

    LoadAttr {
        slot: u32,
    },

    StoreAttr {
        slot: u32,
        src: Value,
    },

    InterpAttr {
        slot: u32,
        perspective: Value,
        mode: u8,
        sat: bool,
    },

    SampleTex {
        tex_id: u32,
        u: Value,
        v: Value,
        array: Option<Value>,
        volume: Option<Value>,
        cube: Option<Value>,
        implicit_lod: bool,
        lod_bias: Option<Value>,
        explicit_lod: Option<Value>,
        texel_offset: Option<(Value, Value)>,
        dref: Option<Value>,
        component: u8,
    },

    SampleTexHandle {
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        u: Value,
        v: Option<Value>,
        w: Option<Value>,
        implicit_lod: bool,
        lod_bias: Option<Value>,
        explicit_lod: Option<Value>,
        texel_offset: Option<(Value, Value)>,
        dref: Option<Value>,
        component: u8,
    },

    TexelFetch {
        cbuf_binding: u8,
        cbuf_word_offset: u32,
        cbuf_secondary_word_offset: Option<u32>,
        x: Value,
        y: Option<Value>,
        z: Option<Value>,
        component: u8,
    },

    TexelFetchHandle {
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        x: Value,
        y: Option<Value>,
        z: Option<Value>,
        component: u8,
    },

    TextureQueryDimension {
        handle: TextureHandleOrigin,
        lod: Value,
        component: u8,
    },

    LocalInvocationId {
        component: u8,
    },

    WorkgroupId {
        component: u8,
    },

    SubgroupLaneId,

    SubgroupMask {
        kind: SubgroupMask,
    },

    ImageWrite {
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        x: Value,
        y: Option<Value>,
        z: Option<Value>,
        values: [Value; 4],
    },

    ImageAtomic {
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        x: Value,
        y: Option<Value>,
        z: Option<Value>,
        value: Value,
        op: ImageAtomicOp,
        data_type: ImageAtomicType,
    },

    HAdd {
        a: Value,
        b: Value,
        old: Value,
        merge: HalfMerge,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        abs_a: bool,
        neg_a: bool,
        abs_b: bool,
        neg_b: bool,
        sat: bool,
        ftz: bool,
    },

    HMul {
        a: Value,
        b: Value,
        old: Value,
        merge: HalfMerge,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        abs_a: bool,
        neg_a: bool,
        abs_b: bool,
        neg_b: bool,
        sat: bool,
        precision: HalfPrecision,
    },

    HFma {
        a: Value,
        b: Value,
        c: Value,
        old: Value,
        merge: HalfMerge,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        swizzle_c: HalfSwizzle,
        neg_b: bool,
        neg_c: bool,
        sat: bool,
        precision: HalfPrecision,
    },

    PackHalf2 {
        lo: Value,
        hi: Value,
    },

    GatherTex {
        tex_id: u32,
        u: Value,
        v: Value,
        gather_component: u8,
        lane: u8,
    },

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

    F2F {
        src: Value,
        neg: bool,
        abs: bool,
        sat: bool,
        round: u8,
    },

    I2F {
        src: Value,
        signed: bool,
        neg: bool,
        abs: bool,
        int_format: u8,
        selector: u8,
    },

    FSet {
        cmp: FComp,
        bop: BoolOp,
        src_a: Value,
        src_b: Value,
        neg_a: bool,
        abs_a: bool,
        neg_b: bool,
        abs_b: bool,
        bf: bool,
        src_pred: u8,
        src_pred_inv: bool,
    },

    ISetPred {
        cmp: ICmp,
        signed: bool,
        bop: BoolOp,
        src_a: Value,
        src_b: Value,
        src_pred: u8,
        src_pred_inv: bool,
        dest_p: u8,
        dest_np: u8,
    },

    HSetPred {
        cmp: FComp,
        bop: BoolOp,
        src_a: Value,
        src_b: Value,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        neg_a: bool,
        abs_a: bool,
        neg_b: bool,
        abs_b: bool,
        src_pred: u8,
        src_pred_inv: bool,
        dest_p: u8,
        dest_np: u8,
        h_and: bool,
        ftz: bool,
    },

    PSetPred {
        dest_p: u8,
        dest_np: u8,
        pred_a: u8,
        neg_pred_a: bool,
        pred_b: u8,
        neg_pred_b: bool,
        pred_c: u8,
        neg_pred_c: bool,
        bop_1: BoolOp,
        bop_2: BoolOp,
    },

    CSetPred {
        dest_p: u8,
        dest_np: u8,
        flow_test: u8,
        bop_pred: u8,
        neg_bop_pred: bool,
        bop: BoolOp,
    },

    PSet {
        pred_a: u8,
        neg_pred_a: bool,
        pred_b: u8,
        neg_pred_b: bool,
        pred_c: u8,
        neg_pred_c: bool,
        bop_1: BoolOp,
        bop_2: BoolOp,
        bool_float: bool,
    },

    IAdd {
        a: Value,
        b: Value,
        neg_a: bool,
        neg_b: bool,
    },

    IMul {
        a: Value,
        b: Value,
    },

    IMulHigh {
        a: Value,
        b: Value,
        signed_a: bool,
        signed_b: bool,
    },

    IMinMaxPred {
        a: Value,
        b: Value,
        signed: bool,
        pred: u8,
        neg_pred: bool,
    },

    IScAdd {
        a: Value,
        b: Value,
        shift: u8,
        neg_a: bool,
        neg_b: bool,
    },

    ILop {
        a: Value,
        b: Value,
        op: LogicOp,
        not_a: bool,
        not_b: bool,
    },

    ILop3 {
        a: Value,
        b: Value,
        c: Value,
        lut: u8,
    },

    FindUMsb {
        value: Value,
    },

    BitCount {
        value: Value,
    },

    IShl {
        a: Value,
        b: Value,
    },

    IShr {
        a: Value,
        b: Value,
        signed: bool,
    },

    F2I {
        src: Value,
        signed: bool,
        round: u8,
    },

    Bfe {
        a: Value,
        b: Value,
        signed: bool,
    },

    Bfi {
        base: Value,
        insert: Value,
        control: Value,
    },

    Shfl {
        value: Value,
        index: Value,
        mask: Value,
        mode: u8,
        pred_dest: u8,
    },

    SubgroupVote {
        source_pred: Predicate,
        mode: VoteMode,
        pred_dest: u8,
        old: Value,
    },

    FSwzAdd {
        a: Value,
        b: Value,
        swizzle: u32,
    },

    ISet {
        cmp: ICmp,
        signed: bool,
        a: Value,
        b: Value,
        bool_float: bool,
    },

    Kill,

    Phi {
        sources: Vec<(super::cfg::BlockId, Value)>,
    },

    SelectPred {
        pred: Predicate,
        if_true: Value,
        if_false: Value,
    },

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

    pub fn emit_pred(&mut self, op: Op, dest_reg: Option<u8>, pred: Option<Predicate>) -> ValueId {
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
            Op::FMul { a, b, .. } => write!(f, "FMul  {a}, {b}"),
            Op::FAdd { a, b, .. } => write!(f, "FAdd  {a}, {b}"),
            Op::FFma { a, b, c, .. } => write!(f, "FFma  {a}, {b}, {c}"),
            Op::DpdxFine { src } => write!(f, "DPdxFine {src}"),
            Op::DpdyFine { src } => write!(f, "DPdyFine {src}"),
            Op::YDirection => write!(f, "YDirection"),
            Op::HAdd { a, b, .. } => write!(f, "HAdd  {a}, {b}"),
            Op::HMul { a, b, .. } => write!(f, "HMul  {a}, {b}"),
            Op::HFma { a, b, c, .. } => write!(f, "HFma  {a}, {b}, {c}"),
            Op::PackHalf2 { lo, hi } => write!(f, "PackH {lo}, {hi}"),
            Op::FMin { a, b, .. } => write!(f, "FMin  {a}, {b}"),
            Op::FMax { a, b, .. } => write!(f, "FMax  {a}, {b}"),
            Op::FMinMaxPred {
                a,
                b,
                pred,
                neg_pred,
                ..
            } => write!(f, "FMnMx {a}, {b}, P{pred}, neg={neg_pred}"),
            Op::MultiFunc { src, func } => write!(f, "MFn.{} {src}", func.name()),
            Op::LoadCbuf {
                binding,
                byte_offset,
            } => {
                write!(f, "LdCbuf c[{binding:#x}]:{byte_offset:#x}")
            }
            Op::LoadCbufIndexed {
                binding,
                byte_offset,
                index,
                address_mode,
            } => {
                match address_mode {
                    CbufAddressMode::Default => {
                        write!(f, "LdCbufIdx c[{binding:#x}]:{byte_offset:#x}+{index}")
                    }
                    CbufAddressMode::Segmented => write!(
                        f,
                        "LdCbufIS c[{binding:#x}+(({index}+{byte_offset:#x})>>16)]:(({index}+{byte_offset:#x})&0xffff)"
                    ),
                }
            }
            Op::LoadGlobal { addr_lo, offset } => {
                write!(f, "LdGbl [{addr_lo}+{offset:#x}]")
            }
            Op::StoreGlobal {
                addr_lo,
                offset,
                value,
            } => write!(f, "StGbl [{addr_lo}+{offset:#x}], {value}"),
            Op::GlobalAtomic {
                addr_lo,
                offset,
                value,
                op,
                is_signed,
            } => write!(
                f,
                "GblAtomic.{op:?}{} [{addr_lo}+{offset:#x}], {value}",
                if *is_signed { ".s32" } else { ".u32" }
            ),
            Op::StorageAtomic {
                buffer_index,
                addr_lo,
                imm,
                value,
                op,
                is_signed,
                ..
            } => write!(
                f,
                "StorageAtomic.{op:?}{} ssbo{buffer_index}[{addr_lo}+{imm:#x}], {value}",
                if *is_signed { ".s32" } else { ".u32" }
            ),
            Op::LoadLocal { addr } => write!(f, "LdLcl [{addr}]"),
            Op::StoreLocal { addr, value } => write!(f, "StLcl [{addr}], {value}"),
            Op::LoadShared { addr } => write!(f, "LdShared [{addr}]"),
            Op::StoreShared { addr, value } => write!(f, "StShared [{addr}], {value}"),
            Op::SharedAtomic { addr, value, op } => {
                write!(f, "SharedAtomic.{op:?} [{addr}], {value}")
            }
            Op::WorkgroupBarrier => write!(f, "Barrier.Workgroup"),
            Op::MemoryBarrier { scope } => write!(f, "MemoryBarrier.{scope:?}"),
            Op::LoadStorage {
                buffer_index,
                addr_lo,
                base_addr_lo,
                imm,
                cbuf_binding,
                cbuf_offset,
                align,
            } => {
                write!(
                    f,
                    "LdStor ssbo{buffer_index}[{addr_lo}+{imm:#x} - ({base_addr_lo}&~{align:#x}); c[{cbuf_binding:#x}]:{cbuf_offset:#x}]"
                )
            }
            Op::StoreStorage {
                buffer_index,
                addr_lo,
                base_addr_lo,
                imm,
                value,
                cbuf_binding,
                cbuf_offset,
                align,
            } => write!(
                f,
                "StStor ssbo{buffer_index}[{addr_lo}+{imm:#x} - ({base_addr_lo}&~{align:#x}); c[{cbuf_binding:#x}]:{cbuf_offset:#x}], {value}"
            ),
            Op::LoadAttr { slot } => write!(f, "LdAttr a[{slot:#x}]"),
            Op::StoreAttr { slot, src } => write!(f, "StAttr a[{slot:#x}], {src}"),
            Op::InterpAttr {
                slot,
                perspective,
                mode,
                sat,
            } => {
                write!(
                    f,
                    "Interp a[{slot:#x}], persp={perspective}, mode={mode}, sat={sat}"
                )
            }
            Op::SampleTex {
                tex_id,
                u,
                v,
                array,
                volume,
                cube,
                implicit_lod,
                lod_bias,
                explicit_lod,
                texel_offset,
                dref,
                component,
            } => {
                let coords = if let Some(w) = cube {
                    if let Some(array) = array {
                        format!("({u}, {v}, {w}, {array})cube[]")
                    } else {
                        format!("({u}, {v}, {w})cube")
                    }
                } else if let Some(w) = volume {
                    format!("({u}, {v}, {w})3d")
                } else if let Some(array) = array {
                    format!("({u}, {v}, {array})")
                } else {
                    format!("({u}, {v})")
                };
                let offset = texel_offset
                    .map(|(x, y)| format!(".offset({x}, {y})"))
                    .unwrap_or_default();
                write!(f, "TexSamp")?;
                if *implicit_lod {
                    write!(f, ".implicit")?;
                    if let Some(bias) = lod_bias {
                        write!(f, ".bias({bias})")?;
                    }
                } else if let Some(lod) = explicit_lod {
                    write!(f, ".lod({lod})")?;
                } else {
                    write!(f, ".lod0")?;
                }
                if let Some(dref) = dref {
                    write!(f, ".dref({dref})")?;
                }
                write!(
                    f,
                    "{offset} t[{tex_id:#x}], {coords}.{}",
                    ["r", "g", "b", "a"]
                        .get(*component as usize)
                        .copied()
                        .unwrap_or("?")
                )
            }
            Op::SampleTexHandle {
                handle,
                dimension,
                u,
                v,
                w,
                implicit_lod,
                lod_bias,
                explicit_lod,
                texel_offset,
                dref,
                component,
            } => {
                let coords = match (v, w) {
                    (None, None) => format!("({u})"),
                    (Some(v), None) => format!("({u}, {v})"),
                    (Some(v), Some(w)) => format!("({u}, {v}, {w})"),
                    (None, Some(w)) => format!("({u}, ?, {w})"),
                };
                write!(f, "TexSampHandle")?;
                if *implicit_lod {
                    write!(f, ".implicit")?;
                    if let Some(bias) = lod_bias {
                        write!(f, ".bias({bias})")?;
                    }
                } else if let Some(lod) = explicit_lod {
                    write!(f, ".lod({lod})")?;
                } else {
                    write!(f, ".lod0")?;
                }
                if let Some((x, y)) = texel_offset {
                    write!(f, ".offset({x}, {y})")?;
                }
                if let Some(reference) = dref {
                    write!(f, ".dref({reference})")?;
                }
                write!(
                    f,
                    " {handle}, {coords}{dimension}.{}",
                    ["r", "g", "b", "a"]
                        .get(*component as usize)
                        .copied()
                        .unwrap_or("?")
                )
            }
            Op::TexelFetch {
                cbuf_binding,
                cbuf_word_offset,
                cbuf_secondary_word_offset,
                x,
                y,
                z,
                component,
            } => {
                let coords = match (y, z) {
                    (None, None) => format!("({x})1d"),
                    (Some(y), None) => format!("({x}, {y})2d"),
                    (Some(y), Some(z)) => format!("({x}, {y}, {z})3d"),
                    (None, Some(z)) => format!("({x}, ?, {z})invalid"),
                };
                let handle = if let Some(secondary) = cbuf_secondary_word_offset {
                    format!(
                        "c[{cbuf_binding}][{:#x}]|c[{cbuf_binding}][{:#x}]",
                        cbuf_word_offset * 4,
                        secondary * 4,
                    )
                } else {
                    format!("c[{cbuf_binding}][{:#x}]", cbuf_word_offset * 4)
                };
                write!(
                    f,
                    "TexelFetch {handle}, {coords}.{}",
                    ["r", "g", "b", "a"]
                        .get(*component as usize)
                        .copied()
                        .unwrap_or("?")
                )
            }
            Op::TexelFetchHandle {
                handle,
                dimension,
                x,
                y,
                z,
                component,
            } => {
                let coords = match (y, z) {
                    (None, None) => format!("({x})"),
                    (Some(y), None) => format!("({x}, {y})"),
                    (Some(y), Some(z)) => format!("({x}, {y}, {z})"),
                    (None, Some(z)) => format!("({x}, ?, {z})"),
                };
                write!(
                    f,
                    "TexelFetch {handle}, {coords}{dimension}.{}",
                    ["r", "g", "b", "a"]
                        .get(*component as usize)
                        .copied()
                        .unwrap_or("?")
                )
            }
            Op::TextureQueryDimension {
                handle,
                lod,
                component,
            } => write!(f, "TexQueryDim {handle}, lod={lod}[{component}]"),
            Op::LocalInvocationId { component } => {
                write!(f, "LocalInvocationId[{component}]")
            }
            Op::WorkgroupId { component } => write!(f, "WorkgroupId[{component}]"),
            Op::SubgroupLaneId => write!(f, "SubgroupLaneId"),
            Op::SubgroupMask { kind } => write!(f, "SubgroupMask.{kind:?}"),
            Op::ImageWrite {
                handle,
                dimension,
                x,
                y,
                z,
                values,
            } => {
                let coords = match (y, z) {
                    (None, None) => format!("({x})"),
                    (Some(y), None) => format!("({x}, {y})"),
                    (Some(y), Some(z)) => format!("({x}, {y}, {z})"),
                    (None, Some(z)) => format!("({x}, ?, {z})"),
                };
                write!(
                    f,
                    "ImageWrite {handle}, {coords}{dimension}, ({}, {}, {}, {})",
                    values[0], values[1], values[2], values[3]
                )
            }
            Op::ImageAtomic {
                handle,
                dimension,
                x,
                y,
                z,
                value,
                op,
                data_type,
            } => {
                let coords = match (y, z) {
                    (None, None) => format!("({x})"),
                    (Some(y), None) => format!("({x}, {y})"),
                    (Some(y), Some(z)) => format!("({x}, {y}, {z})"),
                    (None, Some(z)) => format!("({x}, ?, {z})"),
                };
                write!(
                    f,
                    "ImageAtomic.{op:?}.{data_type:?} {handle}, {coords}{dimension}, {value}"
                )
            }
            Op::GatherTex {
                tex_id,
                u,
                v,
                gather_component,
                lane,
            } => write!(
                f,
                "TexGather t[{tex_id:#x}], ({u}, {v}).{}[{lane}]",
                ["r", "g", "b", "a"]
                    .get(*gather_component as usize)
                    .copied()
                    .unwrap_or("?")
            ),
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
            Op::SelectPred {
                pred,
                if_true,
                if_false,
            } => write!(f, "SelPred {pred}, {if_true}, {if_false}"),
            Op::FSetPred {
                cmp,
                bop,
                src_a,
                src_b,
                dest_p,
                src_pred,
                ..
            } => {
                write!(
                    f,
                    "FSetP.{cmp:?}.{bop:?} P{dest_p}, {src_a}, {src_b}, P{src_pred}"
                )
            }
            Op::F2F {
                src, sat, round, ..
            } => write!(f, "F2F   {src} sat={sat} round={round}"),
            Op::I2F { src, signed, .. } => write!(f, "I2F   {src} signed={signed}"),
            Op::FSet {
                cmp,
                bop,
                src_a,
                src_b,
                ..
            } => {
                write!(f, "FSet.{cmp:?}.{bop:?} {src_a}, {src_b}")
            }
            Op::ISetPred {
                cmp,
                bop,
                src_a,
                src_b,
                dest_p,
                ..
            } => {
                write!(f, "ISetP.{cmp:?}.{bop:?} P{dest_p}, {src_a}, {src_b}")
            }
            Op::HSetPred {
                cmp,
                bop,
                src_a,
                src_b,
                dest_p,
                src_pred,
                ..
            } => write!(
                f,
                "HSetP.{cmp:?}.{bop:?} P{dest_p}, {src_a}, {src_b}, P{src_pred}"
            ),
            Op::PSetPred {
                dest_p,
                pred_a,
                pred_b,
                pred_c,
                ..
            } => write!(f, "PSetP P{dest_p}, P{pred_a}, P{pred_b}, P{pred_c}"),
            Op::CSetPred {
                dest_p, flow_test, ..
            } => write!(f, "CSetP P{dest_p}, flow={flow_test}"),
            Op::PSet {
                pred_a,
                pred_b,
                pred_c,
                ..
            } => write!(f, "PSet P{pred_a}, P{pred_b}, P{pred_c}"),
            Op::IAdd { a, b, .. } => write!(f, "IAdd  {a}, {b}"),
            Op::IMul { a, b } => write!(f, "IMul  {a}, {b}"),
            Op::IMulHigh {
                a,
                b,
                signed_a,
                signed_b,
            } => write!(f, "IMulHi {a}, {b} signed={signed_a}/{signed_b}"),
            Op::IMinMaxPred {
                a, b, signed, pred, ..
            } => write!(f, "IMnMx {a}, {b} signed={signed} P{pred}"),
            Op::IScAdd { a, b, shift, .. } => write!(f, "IScAdd {a}, {b} << {shift}"),
            Op::ILop { a, b, op, .. } => write!(f, "ILop.{op:?} {a}, {b}"),
            Op::ILop3 { a, b, c, lut } => write!(f, "ILop3.{lut:#04x} {a}, {b}, {c}"),
            Op::FindUMsb { value } => write!(f, "FindUMsb {value}"),
            Op::BitCount { value } => write!(f, "BitCount {value}"),
            Op::IShl { a, b } => write!(f, "IShl  {a}, {b}"),
            Op::IShr { a, b, signed } => write!(f, "IShr  {a}, {b} signed={signed}"),
            Op::F2I { src, signed, round } => {
                write!(f, "F2I   {src} signed={signed} round={round}")
            }
            Op::Bfe { a, b, signed } => write!(f, "Bfe   {a}, {b} signed={signed}"),
            Op::Bfi {
                base,
                insert,
                control,
            } => write!(f, "Bfi   {base}, {insert}, {control}"),
            Op::Shfl {
                value,
                index,
                mask,
                mode,
                pred_dest,
            } => write!(f, "Shfl.{mode} {value}, {index}, {mask} -> p{pred_dest}"),
            Op::SubgroupVote {
                source_pred,
                mode,
                pred_dest,
                ..
            } => write!(
                f,
                "Vote.{mode:?} P{}{} -> p{pred_dest}",
                if source_pred.negate { "!" } else { "" },
                source_pred.idx
            ),
            Op::FSwzAdd { a, b, swizzle } => write!(f, "FSwzAdd {a}, {b} sw={swizzle:#x}"),
            Op::ISet { cmp, a, b, .. } => write!(f, "ISet.{cmp:?} {a}, {b}"),
            Op::Kill => write!(f, "Kill"),
            Op::Exit => write!(f, "Exit"),
            Op::Unimplemented { opcode, raw } => {
                write!(f, "<unimpl {opcode:?} raw={raw:016x}>")
            }
        }
    }
}

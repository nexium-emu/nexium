use std::collections::{HashMap, HashSet};

use super::bindless_texture_id_pair;
use super::decode::decode_one;
use super::ir::{
    BoolOp, CbufAddressMode, DoubleOp, FComp, FMods, HalfMerge, HalfPrecision, HalfSwizzle, ICmp,
    ImageAtomicOp, ImageAtomicType, ImageDimension, Inst, LogicOp, MemoryBarrierScope, MufuFunc,
    Op, Predicate, Program, ShaderStage, SubgroupMask, TextureHandleOrigin, Value, ValueId,
    VoteMode,
};
use super::opcodes::Opcode;
use super::operand::{
    ald_num_elements, attr_slot_ald, attr_slot_ipa, bfe_signed, cbuf, csetp_bop, csetp_bop_pred,
    csetp_flow_test, csetp_neg_bop_pred, decoded_pred, f2f_mods, f2i_rounding, f2i_signed,
    fadd32i_mods, fadd_mods, ffma32i_mods, ffma_mods, float_imm20, fmnmx_mods, fmnmx_neg_pred,
    fmnmx_pred, fmul32i_mods, fmul_mods, fset_abs_a, fset_abs_b, fset_bf, fset_bop, fset_cmp,
    fset_neg_a, fset_neg_b, fset_src_pred, fset_src_pred_inv, fsetp_abs_a, fsetp_abs_b, fsetp_bop,
    fsetp_cmp, fsetp_dest_np, fsetp_dest_p, fsetp_neg_a, fsetp_neg_b, fsetp_src_pred,
    fsetp_src_pred_inv, half_bop, half_compare, half_dest_np, half_dest_p, half_h_and, half_merge,
    half_precision, half_src_pred, half_src_pred_inv, half_swizzle_a, half_swizzle_b,
    half_swizzle_c, i2f_abs, i2f_int_format, i2f_neg, i2f_selector, i2f_signed, i2i_abs, i2i_cc,
    i2i_dst_format, i2i_dst_signed, i2i_neg, i2i_sat, i2i_selector, i2i_src_format, i2i_src_signed,
    iadd32i_neg_a, iadd32i_po, iadd3_half_a, iadd3_half_b, iadd3_half_c, iadd3_neg_a, iadd3_neg_b,
    iadd3_neg_c, iadd3_shift, iadd_neg_a, iadd_neg_b, imm20, imm32, ipa_interpolation_mode,
    ipa_saturate, iscadd_shift, iset_bf, iset_cmp, iset_signed, isetp_bop, isetp_cmp,
    isetp_dest_np, isetp_dest_p, isetp_signed, isetp_src_pred, isetp_src_pred_inv, ldc_mode,
    ldc_ref, ldc_size, ldc_src_reg, ldg_addr_reg, ldg_offset, ldg_size, ldls_word_count,
    lop32i_not_a, lop32i_not_b, lop32i_op, lop_not_a, lop_not_b, lop_op, mufu_func_bits,
    pset_bool_float, psetp_bop_1, psetp_bop_2, psetp_dest_np, psetp_dest_p, psetp_neg_pred_a,
    psetp_neg_pred_b, psetp_neg_pred_c, psetp_pred_a, psetp_pred_b, psetp_pred_c, reg_a, reg_b,
    reg_c, reg_dest, sel_neg_pred, sel_pred, shr_signed, texs_tex_id, xmad_cr_mrg, xmad_cr_psl,
    xmad_half_a, xmad_imm_src_b, xmad_rc_half_b, xmad_rc_select, xmad_reg_half_b, xmad_reg_mrg,
    xmad_reg_psl, xmad_reg_select, xmad_signed_a, xmad_signed_b, LdcMode, RZ,
};

const PT: u8 = 7;

#[derive(Clone)]
struct ValueDef {
    op: Op,
    dest_reg: Option<u8>,
}

#[derive(Default)]
pub(crate) struct ValueDefs(HashMap<ValueId, ValueDef>);

impl ValueDefs {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn insert(&mut self, id: ValueId, op: Op) {
        self.0.insert(id, ValueDef { op, dest_reg: None });
    }

    pub(crate) fn insert_inst(&mut self, inst: &Inst) {
        if let Some(id) = inst.result {
            self.0.insert(
                id,
                ValueDef {
                    op: inst.op.clone(),
                    dest_reg: inst.dest_reg,
                },
            );
        }
    }

    fn get(&self, id: &ValueId) -> Option<&ValueDef> {
        self.0.get(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CbufHandleOrigin {
    binding: u8,
    word_offset: u32,
    secondary_word_offset: Option<u32>,
    secondary_binding: Option<u8>,
}

impl CbufHandleOrigin {
    pub(crate) fn texture_id(self) -> u32 {
        let same_binding_secondary = self
            .secondary_binding
            .is_none_or(|binding| binding == self.binding)
            .then_some(self.secondary_word_offset)
            .flatten();
        bindless_texture_id_pair(self.binding, self.word_offset, same_binding_secondary)
    }

    pub(crate) fn cross_binding_partner_id(self) -> Option<u32> {
        let binding = self.secondary_binding?;
        (binding != self.binding).then(|| {
            bindless_texture_id_pair(
                binding,
                self.secondary_word_offset
                    .expect("cross-buffer texture handle requires a partner word"),
                None,
            )
        })
    }

    pub(crate) fn as_texture_handle(self) -> TextureHandleOrigin {
        debug_assert!(
            self.secondary_binding
                .is_none_or(|binding| binding == self.binding),
            "cross-buffer handles are lowered to graphics descriptor IDs before SPIR-V"
        );
        TextureHandleOrigin::Bindless {
            cbuf_binding: self.binding,
            cbuf_word_offset: self.word_offset,
            cbuf_secondary_word_offset: self.secondary_word_offset,
        }
    }
}

#[derive(Clone, Copy)]
enum CbufHandleTrace {
    Origin {
        origin: CbufHandleOrigin,
        deferred: bool,
    },
    Cycle,
}

pub(crate) struct PendingBindlessOriginCheck {
    pub(crate) opcode: Opcode,
    pub(crate) raw: u64,
    pub(crate) handle: Value,
    pub(crate) consumer_pred: Option<Predicate>,
    pub(crate) samples: Vec<(usize, Value)>,
}

struct CbufOriginTracer<'a> {
    local: Option<&'a Program>,
    defs: Option<&'a ValueDefs>,
    consumer_pred: Option<Predicate>,
    allow_back_edge_placeholder: bool,
    visiting: HashSet<ValueId>,
}

impl CbufOriginTracer<'_> {
    fn trace_root(&mut self, value: &Value) -> Option<(CbufHandleOrigin, bool)> {
        match self.trace_inner(value)? {
            CbufHandleTrace::Origin { origin, deferred } => Some((origin, deferred)),
            CbufHandleTrace::Cycle => None,
        }
    }

    fn find_def(&self, id: ValueId) -> Option<ValueDef> {
        self.local
            .and_then(|program| {
                program
                    .instructions
                    .iter()
                    .find(|inst| inst.result == Some(id))
            })
            .map(|inst| ValueDef {
                op: inst.op.clone(),
                dest_reg: inst.dest_reg,
            })
            .or_else(|| self.defs.and_then(|defs| defs.get(&id).cloned()))
    }

    fn trace_inner(&mut self, value: &Value) -> Option<CbufHandleTrace> {
        let Value::Inst(id) = *value else {
            return None;
        };
        if !self.visiting.insert(id) {
            return Some(CbufHandleTrace::Cycle);
        }

        let result = self.find_def(id).and_then(|def| self.trace_def(id, def));
        self.visiting.remove(&id);
        result
    }

    fn trace_def(&mut self, id: ValueId, def: ValueDef) -> Option<CbufHandleTrace> {
        match def.op {
            Op::LoadCbuf {
                binding,
                byte_offset,
            } => Some(CbufHandleTrace::Origin {
                origin: CbufHandleOrigin {
                    binding,
                    word_offset: byte_offset / 4,
                    secondary_word_offset: None,
                    secondary_binding: None,
                },
                deferred: false,
            }),
            Op::Mov(inner) => self.trace_inner(&inner),
            Op::SelectPred {
                pred,
                if_true,
                if_false,
            } => {
                if self.consumer_pred == Some(pred) {
                    self.trace_inner(&if_true)
                } else if self.consumer_pred.is_some_and(|consumer| {
                    consumer.idx == pred.idx && consumer.negate != pred.negate
                }) {
                    self.trace_inner(&if_false)
                } else {
                    let if_true = self.trace_inner(&if_true)?;
                    let if_false = self.trace_inner(&if_false)?;
                    match (if_true, if_false) {
                        (
                            CbufHandleTrace::Origin {
                                origin: true_origin,
                                deferred: true_deferred,
                            },
                            CbufHandleTrace::Origin {
                                origin: false_origin,
                                deferred: false_deferred,
                            },
                        ) if true_origin == false_origin => Some(CbufHandleTrace::Origin {
                            origin: true_origin,
                            deferred: true_deferred || false_deferred,
                        }),
                        (CbufHandleTrace::Cycle, CbufHandleTrace::Cycle) => {
                            Some(CbufHandleTrace::Cycle)
                        }
                        _ => None,
                    }
                }
            }
            Op::Phi { sources } => {
                let mut origin = None;
                let mut deferred = false;
                let mut saw_cycle = false;
                for (_, source) in sources {
                    if source == Value::Inst(id) {
                        deferred = true;
                        saw_cycle = true;
                        continue;
                    }
                    if self.allow_back_edge_placeholder
                        && def.dest_reg.is_some_and(|reg| source == Value::GprIn(reg))
                    {
                        deferred = true;
                        saw_cycle = true;
                        continue;
                    }
                    match self.trace_inner(&source)? {
                        CbufHandleTrace::Origin {
                            origin: source_origin,
                            deferred: source_deferred,
                        } => {
                            if origin.is_some_and(|known| known != source_origin) {
                                return None;
                            }
                            origin = Some(source_origin);
                            deferred |= source_deferred;
                        }
                        CbufHandleTrace::Cycle => {
                            deferred = true;
                            saw_cycle = true;
                        }
                    }
                }
                origin.map_or_else(
                    || saw_cycle.then_some(CbufHandleTrace::Cycle),
                    |origin| Some(CbufHandleTrace::Origin { origin, deferred }),
                )
            }
            Op::ILop {
                a,
                b,
                op: LogicOp::And,
                not_a: false,
                not_b: false,
            } => match (a, b) {
                (Value::ImmU32(mask), value) | (value, Value::ImmU32(mask)) if mask != 0 => {
                    self.trace_inner(&value)
                }
                _ => None,
            },
            Op::ILop {
                a,
                b,
                op: LogicOp::Or,
                not_a: false,
                not_b: false,
            } => {
                let CbufHandleTrace::Origin {
                    origin: a,
                    deferred: a_deferred,
                } = self.trace_inner(&a)?
                else {
                    return None;
                };
                let CbufHandleTrace::Origin {
                    origin: b,
                    deferred: b_deferred,
                } = self.trace_inner(&b)?
                else {
                    return None;
                };
                let origin = if a.secondary_word_offset.is_some()
                    || b.secondary_word_offset.is_some()
                {
                    return None;
                } else if a.binding != b.binding {
                    CbufHandleOrigin {
                        binding: a.binding,
                        word_offset: a.word_offset,
                        secondary_word_offset: Some(b.word_offset),
                        secondary_binding: Some(b.binding),
                    }
                } else if a.word_offset == b.word_offset {
                    a
                } else {
                    let (word_offset, secondary_word_offset) = if a.word_offset < b.word_offset {
                        (a.word_offset, b.word_offset)
                    } else {
                        (b.word_offset, a.word_offset)
                    };
                    CbufHandleOrigin {
                        binding: a.binding,
                        word_offset,
                        secondary_word_offset: Some(secondary_word_offset),
                        secondary_binding: None,
                    }
                };
                Some(CbufHandleTrace::Origin {
                    origin,
                    deferred: a_deferred || b_deferred,
                })
            }
            _ => None,
        }
    }
}

pub(crate) fn resolve_cbuf_handle_origin(
    value: &Value,
    consumer_pred: Option<Predicate>,
    defs: &ValueDefs,
) -> Option<CbufHandleOrigin> {
    CbufOriginTracer {
        local: None,
        defs: Some(defs),
        consumer_pred,
        allow_back_edge_placeholder: false,
        visiting: HashSet::new(),
    }
    .trace_root(value)
    .map(|(origin, _)| origin)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TextureSampleForm {
    tex_type: u8,
    mask: u8,
    implicit_lod: bool,
    bias_reg: Option<u8>,
    lod_reg: Option<u8>,
    offset_reg: Option<u8>,
    dref_reg: Option<u8>,
}

fn decode_texture_sample_form(raw: u64, bindless: bool) -> Option<TextureSampleForm> {
    let (aoffi_bit, blod_shift, lc_bit) = if bindless { (36, 37, 40) } else { (54, 55, 58) };
    let aoffi = ((raw >> aoffi_bit) & 1) != 0;
    let blod = ((raw >> blod_shift) & 0x7) as u8;
    let lc = ((raw >> lc_bit) & 1) != 0;
    let ndv = ((raw >> 35) & 1) != 0;
    let dc = ((raw >> 50) & 1) != 0;
    let sparse_pred = ((raw >> 51) & 0x7) as u8;
    let tex_type = ((raw >> 28) & 0x7) as u8;
    let mask = ((raw >> 31) & 0xF) as u8;
    if lc
        || ndv
        || sparse_pred != PT
        || !matches!(tex_type, 0 | 2 | 3 | 4 | 6 | 7)
        || (aoffi && matches!(tex_type, 6 | 7))
        || mask == 0
    {
        return None;
    }
    let mut meta_reg = reg_b(raw).wrapping_add(if bindless { 1 } else { 0 });
    let (implicit_lod, bias_reg, lod_reg) = match blod {
        0 => (true, None, None),
        1 => (false, None, None),
        2 => {
            let reg = meta_reg;
            meta_reg = meta_reg.wrapping_add(1);
            (true, Some(reg), None)
        }
        3 => {
            let reg = meta_reg;
            meta_reg = meta_reg.wrapping_add(1);
            (false, None, Some(reg))
        }
        _ => return None,
    };
    let offset_reg = if aoffi {
        let reg = meta_reg;
        meta_reg = meta_reg.wrapping_add(1);
        Some(reg)
    } else {
        None
    };
    let dref_reg = dc.then_some(meta_reg);
    Some(TextureSampleForm {
        tex_type,
        mask,
        implicit_lod,
        bias_reg,
        lod_reg,
        offset_reg,
        dref_reg,
    })
}

#[derive(Clone, Copy)]
struct FineDerivativeShuffle {
    src: Value,
    index: u8,
}

pub struct Translator {
    pub program: Program,
    reg_state: HashMap<u8, Value>,
    fine_derivative_shuffles: HashMap<u8, FineDerivativeShuffle>,
    stage: ShaderStage,

    pred_state: HashMap<u8, ValueId>,
    cc_source: Option<Value>,
    carry_source: Option<Value>,
    carry_guard: Option<(Predicate, Option<ValueId>)>,
    pending_bindless_origin_checks: Vec<PendingBindlessOriginCheck>,
    pub finished: bool,
    pub unimplemented_count: u32,
    pub bindless_or_partners: HashMap<u32, u32>,
}

fn flow_test_cmp(test: u64) -> Option<ICmp> {
    match test {
        1 | 9 => Some(ICmp::Lt),
        2 | 10 => Some(ICmp::Eq),
        3 | 11 => Some(ICmp::Le),
        4 | 12 => Some(ICmp::Gt),
        5 | 13 => Some(ICmp::Ne),
        6 | 14 => Some(ICmp::Ge),
        _ => None,
    }
}

impl Translator {
    pub fn new() -> Self {
        Self::with_offset(0)
    }

    pub fn new_fragment() -> Self {
        Self::with_initial_stage(HashMap::new(), HashMap::new(), 0, ShaderStage::Fragment)
    }

    pub fn new_compute() -> Self {
        Self::with_initial_stage(HashMap::new(), HashMap::new(), 0, ShaderStage::Compute)
    }

    pub fn with_offset(start: u32) -> Self {
        Self::with_initial(HashMap::new(), HashMap::new(), start)
    }

    pub fn with_initial(
        initial: HashMap<u8, Value>,
        initial_pred: HashMap<u8, ValueId>,
        start: u32,
    ) -> Self {
        Self::with_initial_stage(initial, initial_pred, start, ShaderStage::Vertex)
    }

    pub(crate) fn with_initial_stage(
        initial: HashMap<u8, Value>,
        initial_pred: HashMap<u8, ValueId>,
        start: u32,
        stage: ShaderStage,
    ) -> Self {
        Self {
            program: Program::with_offset(start),
            reg_state: initial,
            fine_derivative_shuffles: HashMap::new(),
            stage,
            pred_state: initial_pred,
            cc_source: None,
            carry_source: None,
            carry_guard: None,
            pending_bindless_origin_checks: Vec::new(),
            finished: false,
            unimplemented_count: 0,
            bindless_or_partners: HashMap::new(),
        }
    }

    fn read_reg(&self, r: u8) -> Value {
        if r == RZ {
            return Value::Zero;
        }
        self.reg_state.get(&r).copied().unwrap_or(Value::GprIn(r))
    }

    fn invalidates_cc_source(opcode: Opcode, raw: u64) -> bool {
        let regular_cc = ((raw >> 47) & 1) != 0
            && matches!(
                opcode,
                Opcode::BFE_reg
                    | Opcode::BFE_cbuf
                    | Opcode::BFE_imm
                    | Opcode::BFI_reg
                    | Opcode::BFI_rc
                    | Opcode::BFI_cr
                    | Opcode::BFI_imm
                    | Opcode::CSET
                    | Opcode::DADD_reg
                    | Opcode::DADD_cbuf
                    | Opcode::DADD_imm
                    | Opcode::DFMA_reg
                    | Opcode::DFMA_rc
                    | Opcode::DFMA_cr
                    | Opcode::DFMA_imm
                    | Opcode::DMNMX_reg
                    | Opcode::DMNMX_cbuf
                    | Opcode::DMNMX_imm
                    | Opcode::DMUL_reg
                    | Opcode::DMUL_cbuf
                    | Opcode::DMUL_imm
                    | Opcode::DSET_reg
                    | Opcode::DSET_cbuf
                    | Opcode::DSET_imm
                    | Opcode::F2F_reg
                    | Opcode::F2F_cbuf
                    | Opcode::F2F_imm
                    | Opcode::F2I_reg
                    | Opcode::F2I_cbuf
                    | Opcode::F2I_imm
                    | Opcode::FADD_reg
                    | Opcode::FADD_cbuf
                    | Opcode::FADD_imm
                    | Opcode::FFMA_reg
                    | Opcode::FFMA_rc
                    | Opcode::FFMA_cr
                    | Opcode::FFMA_imm
                    | Opcode::FLO_reg
                    | Opcode::FLO_cbuf
                    | Opcode::FLO_imm
                    | Opcode::FMNMX_reg
                    | Opcode::FMNMX_cbuf
                    | Opcode::FMNMX_imm
                    | Opcode::FMUL_reg
                    | Opcode::FMUL_cbuf
                    | Opcode::FMUL_imm
                    | Opcode::FSET_reg
                    | Opcode::FSET_cbuf
                    | Opcode::FSET_imm
                    | Opcode::FSWZADD
                    | Opcode::I2F_reg
                    | Opcode::I2F_cbuf
                    | Opcode::I2F_imm
                    | Opcode::IADD_reg
                    | Opcode::IADD_cbuf
                    | Opcode::IADD_imm
                    | Opcode::IADD3_reg
                    | Opcode::IADD3_cbuf
                    | Opcode::IADD3_imm
                    | Opcode::IMNMX_reg
                    | Opcode::IMNMX_cbuf
                    | Opcode::IMNMX_imm
                    | Opcode::ISCADD_reg
                    | Opcode::ISCADD_cbuf
                    | Opcode::ISCADD_imm
                    | Opcode::ISET_reg
                    | Opcode::ISET_cbuf
                    | Opcode::ISET_imm
                    | Opcode::LEA_hi_reg
                    | Opcode::LEA_hi_cbuf
                    | Opcode::LEA_lo_reg
                    | Opcode::LEA_lo_cbuf
                    | Opcode::LEA_lo_imm
                    | Opcode::LOP_reg
                    | Opcode::LOP_cbuf
                    | Opcode::LOP_imm
                    | Opcode::LOP3_reg
                    | Opcode::LOP3_cbuf
                    | Opcode::LOP3_imm
                    | Opcode::PSET
                    | Opcode::SHF_l_reg
                    | Opcode::SHF_l_imm
                    | Opcode::SHF_r_reg
                    | Opcode::SHF_r_imm
                    | Opcode::SHL_reg
                    | Opcode::SHL_cbuf
                    | Opcode::SHL_imm
                    | Opcode::SHR_reg
                    | Opcode::SHR_cbuf
                    | Opcode::SHR_imm
                    | Opcode::VMAD
                    | Opcode::VMNMX
                    | Opcode::XMAD_reg
                    | Opcode::XMAD_rc
                    | Opcode::XMAD_cr
                    | Opcode::XMAD_imm
            );
        let immediate_cc = ((raw >> 52) & 1) != 0
            && matches!(
                opcode,
                Opcode::FADD32I
                    | Opcode::FFMA32I
                    | Opcode::FMUL32I
                    | Opcode::IADD32I
                    | Opcode::ISCADD32I
                    | Opcode::LOP32I
            );
        let r2p_cc = ((raw >> 40) & 1) != 0
            && matches!(opcode, Opcode::R2P_reg | Opcode::R2P_cbuf | Opcode::R2P_imm);
        let control_ambiguity = matches!(
            opcode,
            Opcode::BAR
                | Opcode::BRA
                | Opcode::BRK
                | Opcode::BRX
                | Opcode::CAL
                | Opcode::CONT
                | Opcode::EXIT
                | Opcode::JCAL
                | Opcode::JMP
                | Opcode::JMX
                | Opcode::KIL
                | Opcode::LONGJMP
                | Opcode::PCNT
                | Opcode::PEXIT
                | Opcode::PLONGJMP
                | Opcode::PRET
                | Opcode::RET
                | Opcode::RTT
                | Opcode::SYNC
        );
        regular_cc || immediate_cc || r2p_cc || control_ambiguity
    }

    fn trace_cbuf_handle_origin(
        &self,
        value: &Value,
        consumer_pred: Option<Predicate>,
        defs: Option<&ValueDefs>,
    ) -> Option<(CbufHandleOrigin, bool)> {
        CbufOriginTracer {
            local: Some(&self.program),
            defs,
            consumer_pred,
            allow_back_edge_placeholder: true,
            visiting: HashSet::new(),
        }
        .trace_root(value)
    }

    fn direct_value_op(&self, value: Value, defs: Option<&ValueDefs>) -> Option<Op> {
        let Value::Inst(id) = value else {
            return None;
        };
        self.program
            .instructions
            .iter()
            .find(|inst| inst.result == Some(id))
            .map(|inst| inst.op.clone())
            .or_else(|| defs.and_then(|defs| defs.get(&id).map(|def| def.op.clone())))
    }

    fn fold_y_direction_dpdy(
        &self,
        a: Value,
        b: Value,
        mods: FMods,
        defs: Option<&ValueDefs>,
    ) -> Option<Value> {
        if mods != FMods::default() {
            return None;
        }
        let a_op = self.direct_value_op(a, defs)?;
        let b_op = self.direct_value_op(b, defs)?;
        match (a_op, b_op) {
            (Op::YDirection, Op::DpdyFine { .. }) => Some(b),
            (Op::DpdyFine { .. }, Op::YDirection) => Some(a),
            _ => None,
        }
    }

    pub(crate) fn take_pending_bindless_origin_checks(
        &mut self,
    ) -> Vec<PendingBindlessOriginCheck> {
        std::mem::take(&mut self.pending_bindless_origin_checks)
    }

    pub fn snapshot_reg_state(&self) -> HashMap<u8, Value> {
        self.reg_state.clone()
    }

    pub fn snapshot_pred_state(&self) -> HashMap<u8, ValueId> {
        self.pred_state.clone()
    }

    fn write_reg(&mut self, r: u8, op: Op, pred: Option<Predicate>) -> ValueId {
        self.fine_derivative_shuffles.remove(&r);
        if let Some(pred) = pred.filter(|_| r != RZ) {
            let old = self.read_reg(r);
            let new_id = self.program.emit(op, None);
            let id = self.program.emit(
                Op::SelectPred {
                    pred,
                    if_true: Value::Inst(new_id),
                    if_false: old,
                },
                Some(r),
            );
            self.reg_state.insert(r, Value::Inst(id));
            id
        } else {
            let id = self.program.emit_pred(op, Some(r), pred);
            if r != RZ {
                self.reg_state.insert(r, Value::Inst(id));
            }
            id
        }
    }

    fn write_side_effecting_reg(
        &mut self,
        r: u8,
        op: Op,
        pred: Option<Predicate>,
    ) -> (ValueId, usize) {
        self.fine_derivative_shuffles.remove(&r);
        let old = pred.filter(|_| r != RZ).map(|_| self.read_reg(r));
        let instruction_index = self.program.instructions.len();
        let result = self
            .program
            .emit_pred(op, (old.is_none()).then_some(r), pred);
        if let (Some(pred), Some(old)) = (pred, old) {
            let selected = self.program.emit(
                Op::SelectPred {
                    pred,
                    if_true: Value::Inst(result),
                    if_false: old,
                },
                Some(r),
            );
            self.reg_state.insert(r, Value::Inst(selected));
            (selected, instruction_index)
        } else {
            if r != RZ {
                self.reg_state.insert(r, Value::Inst(result));
            }
            (result, instruction_index)
        }
    }

    fn emit_texture_sample(
        &mut self,
        raw: u64,
        tex_id: u32,
        compute_handle: Option<(TextureHandleOrigin, ImageDimension)>,
        form: TextureSampleForm,
        pred: Option<Predicate>,
    ) -> Option<Vec<(usize, Value)>> {
        let coord = reg_a(raw);
        let arrayed = matches!(form.tex_type, 3 | 7);
        let u = self.read_reg(coord.wrapping_add(u8::from(arrayed)));
        let v = match form.tex_type {
            0 => Value::Zero,
            3 | 7 => self.read_reg(coord.wrapping_add(2)),
            _ => self.read_reg(coord.wrapping_add(1)),
        };
        let array = match form.tex_type {
            3 | 7 => Some(self.read_reg(coord)),
            _ => None,
        };
        let volume = (form.tex_type == 4).then(|| self.read_reg(coord.wrapping_add(2)));
        let cube = match form.tex_type {
            6 => Some(self.read_reg(coord.wrapping_add(2))),
            7 => Some(self.read_reg(coord.wrapping_add(3))),
            _ => None,
        };
        let explicit_lod = if form.implicit_lod {
            None
        } else {
            Some(
                form.lod_reg
                    .map_or(Value::Zero, |lod_reg| self.read_reg(lod_reg)),
            )
        };
        let lod_bias = form.bias_reg.map(|bias_reg| self.read_reg(bias_reg));
        let texel_offset = if let Some(offset_reg) = form.offset_reg {
            let packed = self.immediate_u32(self.read_reg(offset_reg))?;
            let signed_nibble = |shift: u32| {
                let nibble = ((packed >> shift) & 0xf) as i32;
                (if nibble & 0x8 != 0 {
                    nibble - 0x10
                } else {
                    nibble
                }) as u32
            };
            Some((
                Value::ImmU32(signed_nibble(0)),
                Value::ImmU32(if form.tex_type == 0 {
                    0
                } else {
                    signed_nibble(4)
                }),
                Value::ImmU32(if form.tex_type == 4 {
                    signed_nibble(8)
                } else {
                    0
                }),
            ))
        } else {
            None
        };
        let dref = form.dref_reg.map(|reg| self.read_reg(reg));
        let sample_site = self.program.next_value_id();
        let mut dst = reg_dest(raw);
        let mut samples = Vec::new();
        for component in 0..4u8 {
            if (form.mask >> component) & 1 == 0 {
                continue;
            }
            let old_value = self.read_reg(dst);
            let instruction_index = self.program.instructions.len();
            if dref.is_some() && component == 3 {
                self.write_reg(dst, Op::Mov(Value::ImmF32(1.0)), pred);
                dst = dst.wrapping_add(1);
                continue;
            }
            let op = if let Some((handle, dimension)) = compute_handle {
                Op::SampleTexHandle {
                    sample_site: Some(sample_site),
                    handle,
                    dimension,
                    u,
                    v: (dimension != ImageDimension::D1).then_some(v),
                    w: match dimension {
                        ImageDimension::D2Array => array,
                        ImageDimension::D3 => Some(
                            volume.expect("3D compute texture sample must have a W coordinate"),
                        ),
                        _ => None,
                    },
                    implicit_lod: form.implicit_lod,
                    lod_bias,
                    explicit_lod,
                    texel_offset,
                    dref,
                    component,
                }
            } else {
                Op::SampleTex {
                    sample_site: Some(sample_site),
                    tex_id,
                    u,
                    v,
                    array,
                    volume,
                    cube,
                    implicit_lod: form.implicit_lod,
                    lod_bias,
                    explicit_lod,
                    texel_offset,
                    dref,
                    component,
                }
            };
            self.write_reg(dst, op, pred);
            samples.push((instruction_index, old_value));
            dst = dst.wrapping_add(1);
        }
        Some(samples)
    }

    fn immediate_u32(&self, mut value: Value) -> Option<u32> {
        for _ in 0..16 {
            match value {
                Value::Zero => return Some(0),
                Value::ImmU32(bits) => return Some(bits),
                Value::ImmF32(value) => return Some(value.to_bits()),
                Value::Inst(id) => {
                    let op = self
                        .program
                        .instructions
                        .iter()
                        .find(|inst| inst.result == Some(id))
                        .map(|inst| &inst.op)?;
                    let Op::Mov(next) = op else {
                        return None;
                    };
                    value = *next;
                }
                Value::GprIn(_) => return None,
            }
        }
        None
    }

    fn load_cbuf(&mut self, raw: u64) -> ValueId {
        let cb = cbuf(raw);
        self.program.emit(
            Op::LoadCbuf {
                binding: cb.binding,
                byte_offset: cb.byte_offset,
            },
            None,
        )
    }

    fn double_pair(&self, reg: u8) -> [Value; 2] {
        if reg == RZ { [Value::Zero; 2] } else { [self.read_reg(reg), self.read_reg(reg + 1)] }
    }

    fn write_double(
        &mut self, dest: u8, op: DoubleOp, a: [Value; 2], b: [Value; 2], c: [Value; 2],
        mods: FMods, pred: Option<Predicate>,
    ) {
        for component in 0..if op == DoubleOp::ToFloat32 { 1 } else { 2 } {
            let reg = if dest == RZ { RZ } else { dest + component };
            self.write_reg(reg, Op::Double { op, a, b, c, mods, component }, pred);
        }
    }

    fn emit_f2f(&mut self, raw: u64, src: Value, pred: Option<Predicate>) -> bool {
        let dest = reg_dest(raw);
        let m = f2f_mods(raw);
        let src_size = (raw >> 10) & 3;
        let dst_size = (raw >> 8) & 3;
        if src_size == 3 || dst_size == 3 {
            let opcode = decode_one(raw).unwrap().opcode;
            if !matches!(src_size, 2 | 3) || !matches!(dst_size, 2 | 3)
                || ((raw >> 47) & 1) != 0 || ((raw >> 39) & 15) != 0
                || (dst_size == 3 && dest != RZ && dest % 2 != 0)
                || (src_size == 3 && opcode == Opcode::F2F_imm)
                || (src_size == 3 && opcode == Opcode::F2F_reg && reg_b(raw) != RZ && reg_b(raw) % 2 != 0)
            {
                self.program.emit_void(Op::Unimplemented { opcode, raw });
                self.unimplemented_count += 1;
                return false;
            }
            let a = if src_size == 2 { [src, Value::Zero] } else if opcode == Opcode::F2F_reg {
                self.double_pair(reg_b(raw))
            } else {
                let cb = cbuf(raw);
                let high = self.program.emit(Op::LoadCbuf { binding: cb.binding, byte_offset: cb.byte_offset + 4 }, None);
                [src, Value::Inst(high)]
            };
            let op = if src_size == 2 { DoubleOp::FromFloat32 } else if dst_size == 2 { DoubleOp::ToFloat32 } else { DoubleOp::Move };
            self.write_double(dest, op, a, [Value::Zero; 2], [Value::Zero; 2],
                FMods { neg_a: m.neg, abs_a: m.abs, ..FMods::default() }, pred);
            return true;
        }
        self.write_reg(
            dest,
            Op::F2F {
                src,
                neg: m.neg,
                abs: m.abs,
                sat: m.sat,
                round: m.round,
            },
            pred,
        );
        true
    }

    fn emit_i2f(&mut self, raw: u64, src: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        self.write_reg(
            dest,
            Op::I2F {
                src,
                signed: i2f_signed(raw),
                neg: i2f_neg(raw),
                abs: i2f_abs(raw),
                int_format: i2f_int_format(raw),
                selector: i2f_selector(raw),
            },
            pred,
        );
    }

    fn emit_fset(&mut self, raw: u64, src_b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let src_a = self.read_reg(reg_a(raw));
        let id = self.write_reg(
            dest,
            Op::FSet {
                cmp: FComp::from_bits(fset_cmp(raw)),
                bop: BoolOp::from_bits(fset_bop(raw)),
                src_a,
                src_b,
                neg_a: fset_neg_a(raw),
                abs_a: fset_abs_a(raw),
                neg_b: fset_neg_b(raw),
                abs_b: fset_abs_b(raw),
                bf: fset_bf(raw),
                src_pred: fset_src_pred(raw),
                src_pred_inv: fset_src_pred_inv(raw),
            },
            pred,
        );
        if raw & (1 << 47) != 0 && pred.is_none() {
            self.cc_source = Some(Value::Inst(id));
        }
    }

    fn half_swizzle(bits: u8) -> HalfSwizzle {
        match bits & 3 {
            0 => HalfSwizzle::H1H0,
            1 => HalfSwizzle::F32,
            2 => HalfSwizzle::H0H0,
            _ => HalfSwizzle::H1H1,
        }
    }

    fn half_merge(bits: u8) -> HalfMerge {
        match bits & 3 {
            0 => HalfMerge::H1H0,
            1 => HalfMerge::F32,
            2 => HalfMerge::MrgH0,
            _ => HalfMerge::MrgH1,
        }
    }

    fn half_precision(bits: u8) -> HalfPrecision {
        match bits & 3 {
            1 => HalfPrecision::FTZ,
            2 => HalfPrecision::FMZ,
            _ => HalfPrecision::None,
        }
    }

    fn half_imm(raw: u64) -> Value {
        let low = ((raw >> 20) & 0x1ff) as u32;
        let high = ((raw >> 30) & 0x1ff) as u32;
        let value = (low << 6)
            | ((((raw >> 29) & 1) as u32) << 15)
            | (high << 22)
            | ((((raw >> 56) & 1) as u32) << 31);
        Value::ImmU32(value)
    }

    fn emit_hadd2(
        &mut self,
        raw: u64,
        src_b: Value,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        abs_a: bool,
        neg_a: bool,
        abs_b: bool,
        neg_b: bool,
        merge: HalfMerge,
        sat: bool,
        ftz: bool,
        pred: Option<Predicate>,
    ) {
        let old = self.read_reg(reg_dest(raw));
        self.write_reg(
            reg_dest(raw),
            Op::HAdd {
                a: self.read_reg(reg_a(raw)),
                b: src_b,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                abs_a,
                neg_a,
                abs_b,
                neg_b,
                sat,
                ftz,
            },
            pred,
        );
    }

    fn emit_hmul2(
        &mut self,
        raw: u64,
        src_b: Value,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        abs_a: bool,
        neg_a: bool,
        abs_b: bool,
        neg_b: bool,
        merge: HalfMerge,
        sat: bool,
        precision: HalfPrecision,
        pred: Option<Predicate>,
    ) {
        let old = self.read_reg(reg_dest(raw));
        self.write_reg(
            reg_dest(raw),
            Op::HMul {
                a: self.read_reg(reg_a(raw)),
                b: src_b,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                abs_a,
                neg_a,
                abs_b,
                neg_b,
                sat,
                precision,
            },
            pred,
        );
    }

    fn emit_hfma2(
        &mut self,
        raw: u64,
        src_b: Value,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        src_c: Value,
        swizzle_c: HalfSwizzle,
        neg_b: bool,
        neg_c: bool,
        merge: HalfMerge,
        sat: bool,
        precision: HalfPrecision,
        pred: Option<Predicate>,
    ) {
        let old = self.read_reg(reg_dest(raw));
        self.write_reg(
            reg_dest(raw),
            Op::HFma {
                a: self.read_reg(reg_a(raw)),
                b: src_b,
                c: src_c,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                swizzle_c,
                neg_b,
                neg_c,
                sat,
                precision,
            },
            pred,
        );
    }

    fn emit_hsetp2(
        &mut self,
        raw: u64,
        src_b: Value,
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        neg_a: bool,
        abs_a: bool,
        neg_b: bool,
        abs_b: bool,
        cmp: FComp,
        bop: BoolOp,
        src_pred: u8,
        src_pred_inv: bool,
        h_and: bool,
        ftz: bool,
        pred: Option<Predicate>,
    ) {
        let dest_p = half_dest_p(raw);
        let dest_np = half_dest_np(raw);
        let id = self.program.emit_pred(
            Op::HSetPred {
                cmp,
                bop,
                src_a: self.read_reg(reg_a(raw)),
                src_b,
                swizzle_a,
                swizzle_b,
                neg_a,
                abs_a,
                neg_b,
                abs_b,
                src_pred,
                src_pred_inv,
                dest_p,
                dest_np,
                h_and,
                ftz,
            },
            None,
            pred,
        );
        if dest_p != PT {
            self.pred_state.insert(dest_p, id);
        }
        if dest_np != PT {
            self.pred_state.insert(dest_np, id);
        }
    }

    fn video_operand(&mut self, value: Value, width: u32, selector: u32, signed: bool) -> Value {
        let (bits, offset) = match width {
            0 | 1 => (8, selector * 8),
            2 => (16, (selector & 1) * 16),
            3 => return value,
            _ => panic!("invalid video operand width"),
        };
        self.emit_value(Op::Bfe { a: value, b: Value::ImmU32((bits << 8) | offset), signed })
    }

    fn emit_isetp(&mut self, raw: u64, src_b: Value, pred: Option<Predicate>) {
        let src_a = self.read_reg(reg_a(raw));
        let dest_p = isetp_dest_p(raw);
        let dest_np = isetp_dest_np(raw);
        let op = Op::ISetPred {
            cmp: ICmp::from_bits(isetp_cmp(raw)),
            signed: isetp_signed(raw),
            bop: BoolOp::from_bits(isetp_bop(raw)),
            src_a,
            src_b,
            src_pred: isetp_src_pred(raw),
            src_pred_inv: isetp_src_pred_inv(raw),
            dest_p,
            dest_np,
        };
        let id = self.program.emit_pred(op, None, pred);
        if dest_p != PT {
            self.pred_state.insert(dest_p, id);
        }
        if dest_np != PT {
            self.pred_state.insert(dest_np, id);
        }
    }

    fn emit_iadd(&mut self, raw: u64, b: Value, pred: Option<Predicate>) -> bool {
        let opcode = decode_one(raw).unwrap().opcode;
        let immediate32 = opcode == Opcode::IADD32I;
        let cc = ((raw >> if immediate32 { 52 } else { 47 }) & 1) != 0;
        let x = ((raw >> if immediate32 { 53 } else { 43 }) & 1) != 0;
        let sat = ((raw >> if immediate32 { 54 } else { 50 }) & 1) != 0;
        let po = if immediate32 {
            iadd32i_po(raw)
        } else {
            ((raw >> 48) & 3) == 3
        };
        let carry_available = self.carry_source.is_some()
            && self.carry_guard.is_none_or(|(guard, version)| {
                Some(guard) == pred && self.pred_state.get(&guard.idx).copied() == version
            });
        if sat || (cc && (x || po)) || (x && (po || !carry_available)) {
            self.program.emit_void(Op::Unimplemented { opcode, raw });
            self.unimplemented_count += 1;
            return false;
        }
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        let neg_a = if immediate32 {
            iadd32i_neg_a(raw)
        } else {
            iadd_neg_a(raw)
        };
        let neg_b = !immediate32 && !po && iadd_neg_b(raw);
        let add = Op::IAdd { a, b, neg_a, neg_b };
        let result = if po || x {
            let sum = self.emit_value(add);
            self.write_reg(
                dest,
                Op::IAdd {
                    a: sum,
                    b: if po {
                        Value::ImmU32(1)
                    } else {
                        self.carry_source.unwrap()
                    },
                    neg_a: false,
                    neg_b: false,
                },
                pred,
            )
        } else {
            self.write_reg(dest, add, pred)
        };
        if cc {
            let unsigned_a = if neg_a { self.emit_ineg_value(a) } else { a };
            self.record_add_flags(result, unsigned_a, pred);
        }
        true
    }

    fn record_add_flags(&mut self, result: ValueId, unsigned_a: Value, pred: Option<Predicate>) {
        self.cc_source = pred.is_none().then_some(Value::Inst(result));
        let carry_mask = self.emit_value(Op::ISet {
            cmp: ICmp::Lt,
            signed: false,
            a: Value::Inst(result),
            b: unsigned_a,
            bool_float: false,
        });
        self.carry_source = Some(self.emit_bfe_value(carry_mask, 0, 1, false));
        self.carry_guard = pred.map(|guard| (guard, self.pred_state.get(&guard.idx).copied()));
    }

    fn emit_imad(&mut self, raw: u64, b: Value, c: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        let prod = self.emit_imul_value(a, b);
        self.write_reg(
            dest,
            Op::IAdd {
                a: prod,
                b: c,
                neg_a: false,
                neg_b: false,
            },
            pred,
        );
    }

    fn emit_imul(
        &mut self,
        raw: u64,
        b: Value,
        pred: Option<Predicate>,
        high: bool,
        signed_a: bool,
        signed_b: bool,
    ) {
        let a = self.read_reg(reg_a(raw));
        let op = if high {
            Op::IMulHigh {
                a,
                b,
                signed_a,
                signed_b,
            }
        } else {
            Op::IMul { a, b }
        };
        self.write_reg(reg_dest(raw), op, pred);
    }

    fn emit_prmt_imm(&mut self, raw: u64, pred: Option<Predicate>) -> bool {
        if ((raw >> 47) & 1) != 0 || ((raw >> 48) & 0x7) != 0 {
            return false;
        }

        let a = self.read_reg(reg_a(raw));
        let b = self.read_reg(reg_c(raw));
        let selector = ((raw >> 20) & 0xffff) as u16;
        let mut result = Value::Zero;

        for output_byte in 0..4u32 {
            let select = u32::from((selector >> (output_byte * 4)) & 0xf);
            let source = if select & 0x7 < 4 { a } else { b };
            let source_byte = select & 0x3;
            let mut byte = if select & 0x8 != 0 {
                self.emit_bfe_value(source, source_byte * 8 + 7, 1, true)
            } else {
                self.emit_bfe_value(source, source_byte * 8, 8, false)
            };
            byte = self.emit_ilop_imm_value(byte, 0xff, LogicOp::And, false, false);
            if output_byte != 0 {
                byte = self.emit_ishl_imm_value(byte, output_byte * 8);
            }
            result = self.emit_ilop_value(result, byte, LogicOp::Or, false, false);
        }

        self.write_reg(reg_dest(raw), Op::Mov(result), pred);
        true
    }

    fn emit_imnmx(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
            dest,
            Op::IMinMaxPred {
                a,
                b,
                signed: (raw >> 48) & 1 == 1,
                pred: fmnmx_pred(raw),
                neg_pred: fmnmx_neg_pred(raw),
            },
            pred,
        );
    }

    fn emit_sel(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
            dest,
            Op::SelectPred {
                pred: Predicate {
                    idx: sel_pred(raw),
                    negate: sel_neg_pred(raw),
                },
                if_true: a,
                if_false: b,
            },
            pred,
        );
    }

    fn emit_psetp(&mut self, raw: u64, pred: Option<Predicate>) {
        let dest_p = psetp_dest_p(raw);
        let dest_np = psetp_dest_np(raw);
        let op = Op::PSetPred {
            dest_p,
            dest_np,
            pred_a: psetp_pred_a(raw),
            neg_pred_a: psetp_neg_pred_a(raw),
            pred_b: psetp_pred_b(raw),
            neg_pred_b: psetp_neg_pred_b(raw),
            pred_c: psetp_pred_c(raw),
            neg_pred_c: psetp_neg_pred_c(raw),
            bop_1: BoolOp::from_bits(psetp_bop_1(raw)),
            bop_2: BoolOp::from_bits(psetp_bop_2(raw)),
        };
        let id = self.program.emit_pred(op, None, pred);
        if dest_p != PT {
            self.pred_state.insert(dest_p, id);
        }
        if dest_np != PT {
            self.pred_state.insert(dest_np, id);
        }
    }

    fn emit_csetp(&mut self, raw: u64, pred: Option<Predicate>) -> bool {
        let dest_p = psetp_dest_p(raw);
        let dest_np = psetp_dest_np(raw);
        if pred.is_some()
            || csetp_flow_test(raw) != 13
            || csetp_bop_pred(raw) != PT
            || csetp_neg_bop_pred(raw)
            || BoolOp::from_bits(csetp_bop(raw)) != BoolOp::And
        {
            return false;
        }
        let Some(cc_source) = self.cc_source else {
            return false;
        };
        let op = Op::ISetPred {
            cmp: ICmp::Ne,
            signed: false,
            bop: BoolOp::And,
            src_a: cc_source,
            src_b: Value::Zero,
            src_pred: PT,
            src_pred_inv: false,
            dest_p,
            dest_np,
        };
        let id = self.program.emit_pred(op, None, None);
        if dest_p != PT {
            self.pred_state.insert(dest_p, id);
        }
        if dest_np != PT {
            self.pred_state.insert(dest_np, id);
        }
        true
    }

    fn emit_pset(&mut self, raw: u64, pred: Option<Predicate>) {
        self.write_reg(
            reg_dest(raw),
            Op::PSet {
                pred_a: psetp_pred_a(raw),
                neg_pred_a: psetp_neg_pred_a(raw),
                pred_b: psetp_pred_b(raw),
                neg_pred_b: psetp_neg_pred_b(raw),
                pred_c: psetp_pred_c(raw),
                neg_pred_c: psetp_neg_pred_c(raw),
                bop_1: BoolOp::from_bits(psetp_bop_1(raw)),
                bop_2: BoolOp::from_bits(psetp_bop_2(raw)),
                bool_float: pset_bool_float(raw),
            },
            pred,
        );
    }

    fn emit_value(&mut self, op: Op) -> Value {
        Value::Inst(self.program.emit(op, None))
    }

    fn emit_bfe_value(&mut self, a: Value, pos: u32, count: u32, signed: bool) -> Value {
        self.emit_value(Op::Bfe {
            a,
            b: Value::ImmU32(pos | (count << 8)),
            signed,
        })
    }

    fn emit_iadd_value(&mut self, a: Value, b: Value, neg_a: bool, neg_b: bool) -> Value {
        self.emit_value(Op::IAdd { a, b, neg_a, neg_b })
    }

    fn emit_ineg_value(&mut self, value: Value) -> Value {
        self.emit_iadd_value(Value::Zero, value, false, true)
    }

    fn emit_imul_value(&mut self, a: Value, b: Value) -> Value {
        self.emit_value(Op::IMul { a, b })
    }

    fn emit_ishl_imm_value(&mut self, a: Value, shift: u32) -> Value {
        self.emit_value(Op::IShl {
            a,
            b: Value::ImmU32(shift),
        })
    }

    fn emit_ishr_imm_value(&mut self, a: Value, shift: u32) -> Value {
        self.emit_value(Op::IShr {
            a,
            b: Value::ImmU32(shift),
            signed: false,
        })
    }

    fn shf_common_supported(raw: u64) -> bool {
        ((raw >> 37) & 3) == 0 && ((raw >> 47) & 1) == 0 && ((raw >> 48) & 3) == 0
    }

    fn emit_shf_l_imm(&mut self, raw: u64, pred: Option<Predicate>) -> bool {
        if !Self::shf_common_supported(raw) || ((raw >> 50) & 1) != 0 {
            return false;
        }

        let dest = reg_dest(raw);
        let low_reg = reg_a(raw);
        let high_reg = reg_c(raw);
        let low = self.read_reg(low_reg);
        let high = self.read_reg(high_reg);
        let shift = (imm20(raw) as u32).min(32);

        let op = match shift {
            0 => Op::Mov(high),
            32 => Op::Mov(low),
            shift if low_reg == RZ => Op::IShl {
                a: high,
                b: Value::ImmU32(shift),
            },
            shift if high_reg == RZ => Op::IShr {
                a: low,
                b: Value::ImmU32(32 - shift),
                signed: false,
            },
            shift => {
                let left = self.emit_ishl_imm_value(high, shift);
                let right = self.emit_ishr_imm_value(low, 32 - shift);
                Op::ILop {
                    a: left,
                    b: right,
                    op: LogicOp::Or,
                    not_a: false,
                    not_b: false,
                }
            }
        };
        self.write_reg(dest, op, pred);
        true
    }

    fn emit_shf_l_reg(&mut self, raw: u64, pred: Option<Predicate>) -> bool {
        if !Self::shf_common_supported(raw) || ((raw >> 50) & 1) == 0 || reg_a(raw) != RZ {
            return false;
        }

        let shift = self.read_reg(reg_b(raw));
        let masked_shift = self.emit_ilop_imm_value(shift, 31, LogicOp::And, false, false);
        let high = self.read_reg(reg_c(raw));
        self.write_reg(
            reg_dest(raw),
            Op::IShl {
                a: high,
                b: masked_shift,
            },
            pred,
        );
        true
    }

    fn emit_shf_r_imm(&mut self, raw: u64, pred: Option<Predicate>) -> bool {
        if !Self::shf_common_supported(raw) || ((raw >> 50) & 1) != 0 || reg_c(raw) != RZ {
            return false;
        }

        let low = self.read_reg(reg_a(raw));
        let shift = (imm20(raw) as u32).min(32);
        let op = match shift {
            0 => Op::Mov(low),
            32 => Op::Mov(Value::Zero),
            shift => Op::IShr {
                a: low,
                b: Value::ImmU32(shift),
                signed: false,
            },
        };
        self.write_reg(reg_dest(raw), op, pred);
        true
    }

    fn emit_ilop_imm_value(
        &mut self,
        a: Value,
        b: u32,
        op: LogicOp,
        not_a: bool,
        not_b: bool,
    ) -> Value {
        self.emit_value(Op::ILop {
            a,
            b: Value::ImmU32(b),
            op,
            not_a,
            not_b,
        })
    }

    fn emit_ilop_value(
        &mut self,
        a: Value,
        b: Value,
        op: LogicOp,
        not_a: bool,
        not_b: bool,
    ) -> Value {
        self.emit_value(Op::ILop {
            a,
            b,
            op,
            not_a,
            not_b,
        })
    }

    fn emit_mask_select(&mut self, mask: Value, if_true: Value, if_false: Value) -> Value {
        let selected_true = self.emit_ilop_value(mask, if_true, LogicOp::And, false, false);
        let selected_false = self.emit_ilop_value(mask, if_false, LogicOp::And, true, false);
        self.emit_ilop_value(selected_true, selected_false, LogicOp::Or, false, false)
    }

    fn emit_ftz_value(&mut self, value: Value) -> Value {
        let exponent = self.emit_ilop_imm_value(value, 0x7f80_0000, LogicOp::And, false, false);
        let flush_mask = self.emit_value(Op::ISet {
            cmp: ICmp::Eq,
            signed: false,
            a: exponent,
            b: Value::Zero,
            bool_float: false,
        });
        let sign = self.emit_ilop_imm_value(value, 0x8000_0000, LogicOp::And, false, false);
        self.emit_mask_select(flush_mask, sign, value)
    }

    fn emit_fcmp(&mut self, raw: u64, src_a: Value, operand: Value, pred: Option<Predicate>) {
        let operand = if ((raw >> 47) & 1) != 0 {
            self.emit_ftz_value(operand)
        } else {
            operand
        };
        let compare_mask = self.emit_value(Op::FSet {
            cmp: FComp::from_bits((raw >> 48) & 0xf),
            bop: BoolOp::And,
            src_a: operand,
            src_b: Value::Zero,
            neg_a: false,
            abs_a: false,
            neg_b: false,
            abs_b: false,
            bf: false,
            src_pred: PT,
            src_pred_inv: false,
        });
        let selected = self.emit_mask_select(compare_mask, self.read_reg(reg_a(raw)), src_a);
        self.write_reg(reg_dest(raw), Op::Mov(selected), pred);
    }

    fn emit_icmp(&mut self, raw: u64, src_a: Value, operand: Value, pred: Option<Predicate>) {
        let compare_mask = self.emit_value(Op::ISet {
            cmp: ICmp::from_bits((raw >> 49) & 0x7),
            signed: ((raw >> 48) & 1) != 0,
            a: operand,
            b: Value::Zero,
            bool_float: false,
        });
        let selected = self.emit_mask_select(compare_mask, self.read_reg(reg_a(raw)), src_a);
        self.write_reg(reg_dest(raw), Op::Mov(selected), pred);
    }

    fn emit_predicate_mask(&mut self, pred: Predicate) -> Value {
        self.emit_value(Op::PSet {
            pred_a: pred.idx,
            neg_pred_a: pred.negate,
            pred_b: PT,
            neg_pred_b: false,
            pred_c: PT,
            neg_pred_c: false,
            bop_1: BoolOp::And,
            bop_2: BoolOp::And,
            bool_float: false,
        })
    }

    fn emit_r2p(
        &mut self,
        raw: u64,
        mask: Value,
        known_mask: Option<u32>,
        pred: Option<Predicate>,
    ) {
        let src = self.read_reg(reg_a(raw));
        let offset_base = (((raw >> 41) & 3) as u32) * 8;
        let guard_bit = pred.map(|guard| {
            let guard_mask = self.emit_predicate_mask(guard);
            self.emit_bfe_value(guard_mask, 0, 1, false)
        });

        for index in 0..7u8 {
            if known_mask.is_some_and(|known| ((known >> index) & 1) == 0) {
                continue;
            }

            let src_bit = self.emit_bfe_value(src, offset_base + u32::from(index), 1, false);
            let update_bit = match known_mask {
                Some(_) => guard_bit,
                None => {
                    let mask_bit = self.emit_bfe_value(mask, u32::from(index), 1, false);
                    Some(match guard_bit {
                        Some(guard) => {
                            self.emit_ilop_value(mask_bit, guard, LogicOp::And, false, false)
                        }
                        None => mask_bit,
                    })
                }
            };
            let result = if let Some(update_bit) = update_bit {
                let old_mask = self.emit_predicate_mask(Predicate {
                    idx: index,
                    negate: false,
                });
                let old_bit = self.emit_bfe_value(old_mask, 0, 1, false);
                let changed = self.emit_ilop_value(old_bit, src_bit, LogicOp::Xor, false, false);
                let delta = self.emit_ilop_value(update_bit, changed, LogicOp::And, false, false);
                self.emit_ilop_value(old_bit, delta, LogicOp::Xor, false, false)
            } else {
                src_bit
            };

            let id = self.program.emit_pred(
                Op::ISetPred {
                    cmp: ICmp::Ne,
                    signed: false,
                    bop: BoolOp::And,
                    src_a: result,
                    src_b: Value::Zero,
                    src_pred: PT,
                    src_pred_inv: false,
                    dest_p: index,
                    dest_np: PT,
                },
                None,
                None,
            );
            self.pred_state.insert(index, id);
        }
    }

    fn iadd3_half(&mut self, value: Value, half: u8) -> Value {
        match half {
            1 => self.emit_bfe_value(value, 0, 16, false),
            2 => self.emit_bfe_value(value, 16, 16, false),
            _ => value,
        }
    }

    fn xmad_half(&mut self, value: Value, half: u8, signed: bool) -> Value {
        let pos = if half != 0 { 16 } else { 0 };
        self.emit_bfe_value(value, pos, 16, signed)
    }

    fn emit_iadd3_common(
        &mut self,
        raw: u64,
        mut op_a: Value,
        mut op_b: Value,
        mut op_c: Value,
        shift: u8,
        pred: Option<Predicate>,
    ) {
        if iadd3_neg_a(raw) {
            op_a = self.emit_ineg_value(op_a);
        }
        if iadd3_neg_b(raw) {
            op_b = self.emit_ineg_value(op_b);
        }
        if iadd3_neg_c(raw) {
            op_c = self.emit_ineg_value(op_c);
        }
        let lhs = self.emit_iadd_value(op_a, op_b, false, false);
        let lhs = match shift {
            1 => self.emit_ishr_imm_value(lhs, 16),
            2 => self.emit_ishl_imm_value(lhs, 16),
            _ => lhs,
        };
        let result = self.emit_iadd_value(lhs, op_c, false, false);
        self.write_reg(reg_dest(raw), Op::Mov(result), pred);
    }

    fn emit_iadd3_reg(&mut self, raw: u64, pred: Option<Predicate>) {
        let a = self.iadd3_half(self.read_reg(reg_a(raw)), iadd3_half_a(raw));
        let b = self.iadd3_half(self.read_reg(reg_b(raw)), iadd3_half_b(raw));
        let c = self.iadd3_half(self.read_reg(reg_c(raw)), iadd3_half_c(raw));
        self.emit_iadd3_common(raw, a, b, c, iadd3_shift(raw), pred);
    }

    fn emit_iadd3(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        self.emit_iadd3_common(
            raw,
            self.read_reg(reg_a(raw)),
            b,
            self.read_reg(reg_c(raw)),
            0,
            pred,
        );
    }

    fn emit_xmad_common(
        &mut self,
        raw: u64,
        src_b: Value,
        src_c: Value,
        select: u8,
        half_b: u8,
        psl: bool,
        mrg: bool,
        pred: Option<Predicate>,
    ) {
        let a = self.xmad_half(
            self.read_reg(reg_a(raw)),
            xmad_half_a(raw),
            xmad_signed_a(raw),
        );
        let b = self.xmad_half(src_b, half_b, xmad_signed_b(raw));
        let mut product = self.emit_imul_value(a, b);
        if psl {
            product = self.emit_ishl_imm_value(product, 16);
        }
        let c = match select {
            1 => self.emit_bfe_value(src_c, 0, 16, false),
            2 => self.emit_bfe_value(src_c, 16, 16, false),
            4 => {
                let shifted_b = self.emit_ishl_imm_value(src_b, 16);
                self.emit_iadd_value(shifted_b, src_c, false, false)
            }
            _ => src_c,
        };
        let mut result = self.emit_iadd_value(product, c, false, false);
        if mrg {
            let low_result =
                self.emit_ilop_imm_value(result, 0x0000_ffff, LogicOp::And, false, false);
            let low_b = self.emit_bfe_value(src_b, 0, 16, false);
            let high_b = self.emit_ishl_imm_value(low_b, 16);
            result = self.emit_value(Op::ILop {
                a: low_result,
                b: high_b,
                op: LogicOp::Or,
                not_a: false,
                not_b: false,
            });
        }
        self.write_reg(reg_dest(raw), Op::Mov(result), pred);
    }

    fn emit_xmad_reg(&mut self, raw: u64, pred: Option<Predicate>) {
        self.emit_xmad_common(
            raw,
            self.read_reg(reg_b(raw)),
            self.read_reg(reg_c(raw)),
            xmad_reg_select(raw),
            xmad_reg_half_b(raw),
            xmad_reg_psl(raw),
            xmad_reg_mrg(raw),
            pred,
        );
    }

    fn emit_xmad_rc(&mut self, raw: u64, pred: Option<Predicate>) {
        let cbuf = Value::Inst(self.load_cbuf(raw));
        self.emit_xmad_common(
            raw,
            self.read_reg(reg_c(raw)),
            cbuf,
            xmad_rc_select(raw),
            xmad_rc_half_b(raw),
            false,
            false,
            pred,
        );
    }

    fn emit_xmad_cr(&mut self, raw: u64, pred: Option<Predicate>) {
        let cbuf = Value::Inst(self.load_cbuf(raw));
        self.emit_xmad_common(
            raw,
            cbuf,
            self.read_reg(reg_c(raw)),
            xmad_rc_select(raw),
            xmad_rc_half_b(raw),
            xmad_cr_psl(raw),
            xmad_cr_mrg(raw),
            pred,
        );
    }

    fn emit_xmad_imm(&mut self, raw: u64, pred: Option<Predicate>) {
        self.emit_xmad_common(
            raw,
            Value::ImmU32(xmad_imm_src_b(raw)),
            self.read_reg(reg_c(raw)),
            xmad_reg_select(raw),
            0,
            xmad_reg_psl(raw),
            xmad_reg_mrg(raw),
            pred,
        );
    }

    fn emit_iscadd(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        let result = self.write_reg(
            dest,
            Op::IScAdd {
                a,
                b,
                shift: iscadd_shift(raw),
                neg_a: iadd_neg_a(raw),
                neg_b: iadd_neg_b(raw),
            },
            pred,
        );
        if ((raw >> 47) & 1) != 0 {
            let a = if iadd_neg_a(raw) {
                self.emit_ineg_value(a)
            } else {
                a
            };
            let scaled = self.emit_value(Op::IShl {
                a,
                b: Value::ImmU32(u32::from(iscadd_shift(raw))),
            });
            self.record_add_flags(result, scaled, pred);
        }
    }

    fn emit_ilop(
        &mut self,
        raw: u64,
        b: Value,
        op: LogicOp,
        not_a: bool,
        not_b: bool,
        pred: Option<Predicate>,
    ) -> ValueId {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
            dest,
            Op::ILop {
                a,
                b,
                op,
                not_a,
                not_b,
            },
            pred,
        )
    }

    fn emit_lop_predicate(&mut self, raw: u64, result: ValueId, pred: Option<Predicate>) {
        let dest_p = ((raw >> 48) & 7) as u8;
        self.emit_logic_predicate(dest_p, ((raw >> 44) & 3) as u8, result, pred);
    }

    fn emit_logic_predicate(
        &mut self,
        dest_p: u8,
        mode: u8,
        result: ValueId,
        pred: Option<Predicate>,
    ) {
        if dest_p == PT {
            return;
        }
        let cmp = match mode {
            0 => ICmp::F,
            1 => ICmp::T,
            2 => ICmp::Eq,
            _ => ICmp::Ne,
        };
        let id = self.program.emit_pred(
            Op::ISetPred {
                cmp,
                signed: false,
                bop: BoolOp::And,
                src_a: Value::Inst(result),
                src_b: Value::Zero,
                src_pred: PT,
                src_pred_inv: false,
                dest_p,
                dest_np: PT,
            },
            None,
            pred,
        );
        self.pred_state.insert(dest_p, id);
    }

    fn emit_ishl(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(dest, Op::IShl { a, b }, pred);
    }

    fn emit_ishr(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
            dest,
            Op::IShr {
                a,
                b,
                signed: shr_signed(raw),
            },
            pred,
        );
    }

    fn emit_bfe(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let mut a = self.read_reg(reg_a(raw));
        if raw & (1 << 40) != 0 {
            a = self.emit_value(Op::BitReverse { value: a });
        }
        self.write_reg(
            dest,
            Op::Bfe {
                a,
                b,
                signed: bfe_signed(raw),
            },
            pred,
        );
    }

    fn ldl_addr(&mut self, raw: u64) -> Value {
        let off_reg = reg_a(raw);
        let raw_off = ((raw >> 20) & 0x00FF_FFFF) as u32;
        if off_reg == RZ {
            Value::ImmU32(raw_off)
        } else {
            let rel = ((raw_off << 8) as i32) >> 8;
            let base = self.read_reg(off_reg);
            self.emit_value(Op::IAdd {
                a: base,
                b: Value::ImmU32(rel as u32),
                neg_a: false,
                neg_b: false,
            })
        }
    }

    fn shared_addr(&mut self, raw: u64) -> Value {
        let base_reg = reg_a(raw);
        let immediate = ((raw >> 20) & 0x00ff_ffff) as u32;
        if base_reg == RZ {
            Value::ImmU32(immediate)
        } else {
            let displacement = ((immediate << 8) as i32) >> 8;
            let base = self.read_reg(base_reg);
            self.emit_value(Op::IAdd {
                a: base,
                b: Value::ImmU32(displacement as u32),
                neg_a: false,
                neg_b: false,
            })
        }
    }

    fn shared_word_count(raw: u64) -> Option<u32> {
        match (raw >> 48) & 0x7 {
            4 => Some(1),
            5 => Some(2),
            6 => Some(4),
            _ => None,
        }
    }

    fn shared_atomic_addr(&mut self, raw: u64) -> Value {
        let base_reg = reg_a(raw);
        let word_offset = ((raw >> 30) & 0x003f_ffff) as u32;
        if base_reg == RZ {
            Value::ImmU32(word_offset << 2)
        } else {
            let relative_words = ((word_offset << 10) as i32) >> 10;
            let base = self.read_reg(base_reg);
            self.emit_value(Op::IAdd {
                a: base,
                b: Value::ImmU32(relative_words.wrapping_mul(4) as u32),
                neg_a: false,
                neg_b: false,
            })
        }
    }

    fn emit_f2i(&mut self, raw: u64, mut src: Value, pred: Option<Predicate>) {
        let captured_u16 = ((raw >> 8) & 3) == 1
            && ((raw >> 10) & 3) == 2
            && !f2i_signed(raw)
            && f2i_rounding(raw) == 0
            && ((raw >> 41) & 1) == 0
            && ((raw >> 45) & 1) == 0
            && ((raw >> 47) & 1) == 0
            && ((raw >> 49) & 1) == 0;
        if captured_u16 {
            src = self.emit_value(Op::FMax {
                a: src,
                b: Value::ImmF32(0.0),
                mods: FMods::default(),
            });
            src = self.emit_value(Op::FMin {
                a: src,
                b: Value::ImmF32(65535.0),
                mods: FMods::default(),
            });
        }
        let dest = reg_dest(raw);
        self.write_reg(
            dest,
            Op::F2I {
                src,
                signed: f2i_signed(raw),
                round: f2i_rounding(raw),
            },
            pred,
        );
    }

    fn emit_i2i(&mut self, raw: u64, src: Value, pred: Option<Predicate>) -> bool {
        const WORD: u8 = 2;

        let writes_cc = i2i_cc(raw);
        if writes_cc {
            self.cc_source = None;
        }
        let src_signed = i2i_src_signed(raw);
        let dst_signed = i2i_dst_signed(raw);
        let abs = i2i_abs(raw);
        let neg = i2i_neg(raw);
        if i2i_src_format(raw) != WORD
            || i2i_dst_format(raw) != WORD
            || i2i_selector(raw) != 0
            || src_signed != dst_signed
            || i2i_sat(raw)
            || ((abs || neg) && !src_signed)
        {
            return false;
        }
        let mut value = src;
        if abs {
            let sign = self.emit_value(Op::IShr {
                a: value.clone(),
                b: Value::ImmU32(31),
                signed: true,
            });
            let flipped = self.emit_value(Op::ILop {
                a: value,
                b: sign.clone(),
                op: LogicOp::Xor,
                not_a: false,
                not_b: false,
            });
            value = self.emit_iadd_value(flipped, sign, false, true);
        }
        if neg {
            value = self.emit_ineg_value(value);
        }

        if writes_cc && pred.is_none() && !abs && !neg {
            self.cc_source = Some(value);
        }

        self.write_reg(reg_dest(raw), Op::Mov(value), pred);
        true
    }

    fn emit_iset(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        let id = self.write_reg(
            dest,
            Op::ISet {
                cmp: ICmp::from_bits(iset_cmp(raw)),
                signed: iset_signed(raw),
                a,
                b,
                bool_float: iset_bf(raw),
            },
            pred,
        );
        if raw & (1 << 47) != 0 && pred.is_none() {
            self.cc_source = Some(Value::Inst(id));
        }
    }

    pub(crate) fn emit_exit_flow_predicate(&mut self, raw: u64) -> bool {
        let Some(source) = self.cc_source else { return false; };
        if raw & 0x1f != 13 { return false; }
        let predicate = decoded_pred(raw);
        self.program.emit_pred(Op::ISetPred {
            cmp: ICmp::Ne,
            signed: false,
            bop: BoolOp::And,
            src_a: source,
            src_b: Value::Zero,
            src_pred: predicate.map_or(PT, |pred| pred.idx),
            src_pred_inv: predicate.is_some_and(|pred| pred.negate),
            dest_p: 8,
            dest_np: PT,
        }, None, None);
        true
    }

    pub(crate) fn emit_bra_flow_predicate(&mut self, raw: u64) {
        let predicate = decoded_pred(raw);
        let (cmp, src_a) = match (flow_test_cmp(raw & 0x1f), self.cc_source) {
            (Some(cmp), Some(source)) => (cmp, source),
            _ => (ICmp::T, Value::Zero),
        };
        self.program.emit_pred(Op::ISetPred {
            cmp,
            signed: true,
            bop: BoolOp::And,
            src_a,
            src_b: Value::Zero,
            src_pred: predicate.map_or(PT, |pred| pred.idx),
            src_pred_inv: predicate.is_some_and(|pred| pred.negate),
            dest_p: 8,
            dest_np: PT,
        }, None, None);
    }

    pub fn translate(&mut self, raw: u64) -> bool {
        self.translate_impl(raw, None)
    }

    pub(crate) fn translate_with_defs(&mut self, raw: u64, defs: &ValueDefs) -> bool {
        self.translate_impl(raw, Some(defs))
    }

    fn translate_impl(&mut self, raw: u64, defs: Option<&ValueDefs>) -> bool {
        if self.finished {
            return true;
        }
        let Some(decoded) = decode_one(raw) else {
            self.cc_source = None;
            log::debug!(
                "SASS decode failure raw={:#018x} top16={:#x} top13={:#x}",
                raw,
                (raw >> 48) as u16,
                (raw >> 51) as u16,
            );
            self.program.emit_void(Op::Unimplemented {
                opcode: Opcode::NOP,
                raw,
            });
            self.unimplemented_count += 1;
            return false;
        };
        let pred = decoded_pred(raw);
        if Self::invalidates_cc_source(decoded.opcode, raw) {
            self.cc_source = None;
            self.carry_source = None;
        }
        match decoded.opcode {
            Opcode::NOP => {}

            Opcode::MOV_reg => {
                let dest = reg_dest(raw);
                let src = self.read_reg(reg_b(raw));
                self.write_reg(dest, Op::Mov(src), pred);
            }
            Opcode::MOV_cbuf => {
                let dest = reg_dest(raw);
                let cb_id = self.load_cbuf(raw);
                self.write_reg(dest, Op::Mov(Value::Inst(cb_id)), pred);
            }
            Opcode::MOV_imm => {
                let dest = reg_dest(raw);
                self.write_reg(dest, Op::Mov(Value::ImmU32(imm20(raw) as u32)), pred);
            }
            Opcode::MOV32I => {
                let dest = reg_dest(raw);
                self.write_reg(dest, Op::Mov(Value::ImmU32(imm32(raw))), pred);
            }

            Opcode::SHFL => {
                let dest = reg_dest(raw);
                let src = self.read_reg(reg_a(raw));
                let index = ((raw >> 20) & 0x1F) as u8;
                let index_imm = ((raw >> 28) & 1) != 0;
                let mask_imm = ((raw >> 29) & 1) != 0;
                let mode = (raw >> 30) & 3;
                let mask = (raw >> 34) & 0x1FFF;
                let dest_pred = (raw >> 48) & 7;
                if self.stage == ShaderStage::Fragment
                    && pred.is_none()
                    && index_imm
                    && mask_imm
                    && mode == 3
                    && mask == 0x1C03
                    && dest_pred == PT as u64
                    && matches!(index, 1 | 2)
                {
                    self.write_reg(dest, Op::Mov(src), pred);
                    self.fine_derivative_shuffles
                        .insert(dest, FineDerivativeShuffle { src, index });
                } else {
                    let index_value = if index_imm {
                        Value::ImmU32(u32::from(index))
                    } else {
                        self.read_reg(reg_b(raw))
                    };
                    let mask_value = if mask_imm {
                        Value::ImmU32(mask as u32)
                    } else {
                        self.read_reg(reg_c(raw))
                    };
                    let id = self.program.emit(
                        Op::Shfl {
                            value: src,
                            index: index_value,
                            mask: mask_value,
                            mode: mode as u8,
                            pred_dest: dest_pred as u8,
                        },
                        None,
                    );
                    self.write_reg(dest, Op::Mov(Value::Inst(id)), pred);
                }
            }

            Opcode::FSWZADD => {
                let dest = reg_dest(raw);
                let src_a = reg_a(raw);
                let src_b = self.read_reg(reg_b(raw));
                let swizzle = ((raw >> 28) & 0xFF) as u8;
                let ndv = ((raw >> 38) & 1) != 0;
                let round = (raw >> 39) & 3;
                let ftz = ((raw >> 44) & 1) != 0;
                let cc = ((raw >> 47) & 1) != 0;
                let derivative = self
                    .fine_derivative_shuffles
                    .get(&src_a)
                    .copied()
                    .filter(|shuffle| shuffle.src == src_b)
                    .and_then(|shuffle| match (shuffle.index, swizzle) {
                        (1, 0x99) => Some(Op::DpdxFine { src: src_b }),
                        (2, 0xa5) => Some(Op::DpdyFine { src: src_b }),
                        _ => None,
                    });
                if !ndv && round == 0 && !ftz && !cc {
                    if let Some(op) = derivative {
                        self.write_reg(dest, op, pred);
                        return true;
                    }
                }
                let a = self.read_reg(src_a);
                self.write_reg(
                    dest,
                    Op::FSwzAdd {
                        a,
                        b: src_b,
                        swizzle: u32::from(swizzle),
                    },
                    pred,
                );
            }

            Opcode::FMUL_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                let mods = fmul_mods(raw);
                if let Some(derivative) = self.fold_y_direction_dpdy(a, b, mods, defs) {
                    self.write_reg(dest, Op::Mov(derivative), pred);
                    return true;
                }
                self.write_reg(dest, Op::FMul { a, b, mods }, pred);
            }
            Opcode::FMUL_cbuf => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                self.write_reg(
                    dest,
                    Op::FMul {
                        a,
                        b: Value::Inst(cb_id),
                        mods: fmul_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FMUL_imm => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                self.write_reg(
                    dest,
                    Op::FMul {
                        a,
                        b: Value::ImmF32(float_imm20(raw)),
                        mods: fmul_mods(raw),
                    },
                    pred,
                );
            }

            Opcode::FADD_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                self.write_reg(
                    dest,
                    Op::FAdd {
                        a,
                        b,
                        mods: fadd_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FADD_cbuf => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                self.write_reg(
                    dest,
                    Op::FAdd {
                        a,
                        b: Value::Inst(cb_id),
                        mods: fadd_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FADD_imm => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                self.write_reg(
                    dest,
                    Op::FAdd {
                        a,
                        b: Value::ImmF32(float_imm20(raw)),
                        mods: fadd_mods(raw),
                    },
                    pred,
                );
            }

            Opcode::FCMP_reg => {
                let src_a = self.read_reg(reg_b(raw));
                let operand = self.read_reg(reg_c(raw));
                self.emit_fcmp(raw, src_a, operand, pred);
            }
            Opcode::FCMP_rc => {
                let src_a = self.read_reg(reg_c(raw));
                let operand = Value::Inst(self.load_cbuf(raw));
                self.emit_fcmp(raw, src_a, operand, pred);
            }
            Opcode::FCMP_cr => {
                let src_a = Value::Inst(self.load_cbuf(raw));
                let operand = self.read_reg(reg_c(raw));
                self.emit_fcmp(raw, src_a, operand, pred);
            }
            Opcode::FCMP_imm => {
                let src_a = Value::ImmU32(float_imm20(raw).to_bits());
                let operand = self.read_reg(reg_c(raw));
                self.emit_fcmp(raw, src_a, operand, pred);
            }

            Opcode::FFMA_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                let c = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::FFma {
                        a,
                        b,
                        c,
                        mods: ffma_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FFMA_cr => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                let c = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::FFma {
                        a,
                        b: Value::Inst(cb_id),
                        c,
                        mods: ffma_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FFMA_rc => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_c(raw));
                let cb_id = self.load_cbuf(raw);
                self.write_reg(
                    dest,
                    Op::FFma {
                        a,
                        b,
                        c: Value::Inst(cb_id),
                        mods: ffma_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FFMA_imm => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let c = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::FFma {
                        a,
                        b: Value::ImmF32(float_imm20(raw)),
                        c,
                        mods: ffma_mods(raw),
                    },
                    pred,
                );
            }

            Opcode::FADD32I => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                self.write_reg(
                    dest,
                    Op::FAdd {
                        a,
                        b: Value::ImmF32(f32::from_bits(imm32(raw))),
                        mods: fadd32i_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FMUL32I => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                self.write_reg(
                    dest,
                    Op::FMul {
                        a,
                        b: Value::ImmF32(f32::from_bits(imm32(raw))),
                        mods: fmul32i_mods(raw),
                    },
                    pred,
                );
            }
            Opcode::FFMA32I => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let c = self.read_reg(dest);
                self.write_reg(
                    dest,
                    Op::FFma {
                        a,
                        b: Value::ImmF32(f32::from_bits(imm32(raw))),
                        c,
                        mods: ffma32i_mods(raw),
                    },
                    pred,
                );
            }

            Opcode::HADD2_reg => self.emit_hadd2(
                raw,
                self.read_reg(reg_b(raw)),
                Self::half_swizzle(half_swizzle_a(raw)),
                Self::half_swizzle(half_swizzle_b(raw)),
                ((raw >> 44) & 1) != 0,
                ((raw >> 43) & 1) != 0,
                ((raw >> 30) & 1) != 0,
                ((raw >> 31) & 1) != 0,
                Self::half_merge(half_merge(raw)),
                ((raw >> 32) & 1) != 0,
                ((raw >> 39) & 1) != 0,
                pred,
            ),
            Opcode::HADD2_cbuf => {
                let b = Value::Inst(self.load_cbuf(raw));
                self.emit_hadd2(
                    raw,
                    b,
                    Self::half_swizzle(half_swizzle_a(raw)),
                    HalfSwizzle::F32,
                    ((raw >> 44) & 1) != 0,
                    ((raw >> 43) & 1) != 0,
                    ((raw >> 54) & 1) != 0,
                    ((raw >> 56) & 1) != 0,
                    Self::half_merge(half_merge(raw)),
                    ((raw >> 52) & 1) != 0,
                    ((raw >> 39) & 1) != 0,
                    pred,
                );
            }
            Opcode::HADD2_imm => self.emit_hadd2(
                raw,
                Self::half_imm(raw),
                Self::half_swizzle(half_swizzle_a(raw)),
                HalfSwizzle::H1H0,
                ((raw >> 44) & 1) != 0,
                ((raw >> 43) & 1) != 0,
                false,
                false,
                Self::half_merge(half_merge(raw)),
                ((raw >> 52) & 1) != 0,
                ((raw >> 39) & 1) != 0,
                pred,
            ),
            Opcode::HADD2_32I => self.emit_hadd2(
                raw,
                Value::ImmU32(imm32(raw)),
                Self::half_swizzle(((raw >> 53) & 3) as u8),
                HalfSwizzle::H1H0,
                false,
                ((raw >> 56) & 1) != 0,
                false,
                false,
                HalfMerge::H1H0,
                ((raw >> 52) & 1) != 0,
                ((raw >> 55) & 1) != 0,
                pred,
            ),

            Opcode::HMUL2_reg => self.emit_hmul2(
                raw,
                self.read_reg(reg_b(raw)),
                Self::half_swizzle(half_swizzle_a(raw)),
                Self::half_swizzle(half_swizzle_b(raw)),
                ((raw >> 44) & 1) != 0,
                false,
                ((raw >> 30) & 1) != 0,
                ((raw >> 31) & 1) != 0,
                Self::half_merge(half_merge(raw)),
                ((raw >> 32) & 1) != 0,
                Self::half_precision(half_precision(raw, 39)),
                pred,
            ),
            Opcode::HMUL2_cbuf => {
                let b = Value::Inst(self.load_cbuf(raw));
                self.emit_hmul2(
                    raw,
                    b,
                    Self::half_swizzle(half_swizzle_a(raw)),
                    HalfSwizzle::F32,
                    ((raw >> 44) & 1) != 0,
                    ((raw >> 43) & 1) != 0,
                    ((raw >> 54) & 1) != 0,
                    false,
                    Self::half_merge(half_merge(raw)),
                    ((raw >> 52) & 1) != 0,
                    Self::half_precision(half_precision(raw, 39)),
                    pred,
                );
            }
            Opcode::HMUL2_imm => self.emit_hmul2(
                raw,
                Self::half_imm(raw),
                Self::half_swizzle(half_swizzle_a(raw)),
                HalfSwizzle::H1H0,
                ((raw >> 44) & 1) != 0,
                ((raw >> 43) & 1) != 0,
                false,
                false,
                Self::half_merge(half_merge(raw)),
                ((raw >> 52) & 1) != 0,
                Self::half_precision(half_precision(raw, 39)),
                pred,
            ),
            Opcode::HMUL2_32I => self.emit_hmul2(
                raw,
                Value::ImmU32(imm32(raw)),
                Self::half_swizzle(((raw >> 53) & 3) as u8),
                HalfSwizzle::H1H0,
                false,
                false,
                false,
                false,
                HalfMerge::H1H0,
                ((raw >> 52) & 1) != 0,
                Self::half_precision(half_precision(raw, 55)),
                pred,
            ),

            Opcode::HFMA2_reg => self.emit_hfma2(
                raw,
                self.read_reg(reg_b(raw)),
                Self::half_swizzle(half_swizzle_a(raw)),
                Self::half_swizzle(half_swizzle_b(raw)),
                self.read_reg(reg_c(raw)),
                Self::half_swizzle(half_swizzle_c(raw)),
                ((raw >> 31) & 1) != 0,
                ((raw >> 30) & 1) != 0,
                Self::half_merge(half_merge(raw)),
                ((raw >> 32) & 1) != 0,
                Self::half_precision(half_precision(raw, 37)),
                pred,
            ),
            Opcode::HFMA2_rc => {
                let c = Value::Inst(self.load_cbuf(raw));
                self.emit_hfma2(
                    raw,
                    self.read_reg(reg_c(raw)),
                    Self::half_swizzle(half_swizzle_a(raw)),
                    Self::half_swizzle(((raw >> 53) & 3) as u8),
                    c,
                    HalfSwizzle::F32,
                    ((raw >> 56) & 1) != 0,
                    ((raw >> 51) & 1) != 0,
                    Self::half_merge(half_merge(raw)),
                    ((raw >> 52) & 1) != 0,
                    Self::half_precision(half_precision(raw, 57)),
                    pred,
                );
            }
            Opcode::HFMA2_cr => {
                let b = Value::Inst(self.load_cbuf(raw));
                self.emit_hfma2(
                    raw,
                    b,
                    Self::half_swizzle(half_swizzle_a(raw)),
                    HalfSwizzle::F32,
                    self.read_reg(reg_c(raw)),
                    Self::half_swizzle(((raw >> 53) & 3) as u8),
                    ((raw >> 56) & 1) != 0,
                    ((raw >> 51) & 1) != 0,
                    Self::half_merge(half_merge(raw)),
                    ((raw >> 52) & 1) != 0,
                    Self::half_precision(half_precision(raw, 57)),
                    pred,
                );
            }
            Opcode::HFMA2_imm => self.emit_hfma2(
                raw,
                Self::half_imm(raw),
                Self::half_swizzle(half_swizzle_a(raw)),
                HalfSwizzle::H1H0,
                self.read_reg(reg_c(raw)),
                Self::half_swizzle(((raw >> 53) & 3) as u8),
                false,
                ((raw >> 51) & 1) != 0,
                Self::half_merge(half_merge(raw)),
                ((raw >> 52) & 1) != 0,
                Self::half_precision(half_precision(raw, 57)),
                pred,
            ),
            Opcode::HFMA2_32I => self.emit_hfma2(
                raw,
                Value::ImmU32(imm32(raw)),
                Self::half_swizzle(((raw >> 53) & 3) as u8),
                HalfSwizzle::H1H0,
                self.read_reg(reg_dest(raw)),
                HalfSwizzle::H1H0,
                false,
                ((raw >> 52) & 1) != 0,
                HalfMerge::H1H0,
                false,
                Self::half_precision(half_precision(raw, 55)),
                pred,
            ),

            Opcode::HSETP2_reg => self.emit_hsetp2(
                raw,
                self.read_reg(reg_b(raw)),
                Self::half_swizzle(half_swizzle_a(raw)),
                Self::half_swizzle(half_swizzle_b(raw)),
                ((raw >> 43) & 1) != 0,
                ((raw >> 44) & 1) != 0,
                ((raw >> 31) & 1) != 0,
                ((raw >> 30) & 1) != 0,
                FComp::from_bits(half_compare(raw, 35) as u64),
                BoolOp::from_bits(half_bop(raw) as u64),
                half_src_pred(raw),
                half_src_pred_inv(raw),
                half_h_and(raw, 49),
                half_h_and(raw, 6),
                pred,
            ),
            Opcode::HSETP2_cbuf => {
                let b = Value::Inst(self.load_cbuf(raw));
                self.emit_hsetp2(
                    raw,
                    b,
                    Self::half_swizzle(half_swizzle_a(raw)),
                    HalfSwizzle::F32,
                    ((raw >> 43) & 1) != 0,
                    ((raw >> 44) & 1) != 0,
                    ((raw >> 56) & 1) != 0,
                    ((raw >> 54) & 1) != 0,
                    FComp::from_bits(half_compare(raw, 49) as u64),
                    BoolOp::from_bits(half_bop(raw) as u64),
                    half_src_pred(raw),
                    half_src_pred_inv(raw),
                    half_h_and(raw, 53),
                    half_h_and(raw, 6),
                    pred,
                );
            }
            Opcode::HSETP2_imm => self.emit_hsetp2(
                raw,
                Self::half_imm(raw),
                Self::half_swizzle(half_swizzle_a(raw)),
                HalfSwizzle::H1H0,
                ((raw >> 43) & 1) != 0,
                ((raw >> 44) & 1) != 0,
                false,
                false,
                FComp::from_bits(half_compare(raw, 49) as u64),
                BoolOp::from_bits(half_bop(raw) as u64),
                half_src_pred(raw),
                half_src_pred_inv(raw),
                half_h_and(raw, 53),
                half_h_and(raw, 6),
                pred,
            ),

            opcode @ (Opcode::DADD_reg | Opcode::DMUL_reg | Opcode::DFMA_reg) => {
                let fma = opcode == Opcode::DFMA_reg;
                let round = (raw >> if fma { 50 } else { 39 }) & 3;
                let registers = [reg_dest(raw), reg_a(raw), reg_b(raw), if fma { reg_c(raw) } else { RZ }];
                if ((raw >> 47) & 1) != 0 || round != 0 || registers.iter().any(|reg| *reg != RZ && reg % 2 != 0) {
                    self.program.emit_void(Op::Unimplemented { opcode, raw });
                    self.unimplemented_count += 1;
                    return false;
                }
                let op = match opcode { Opcode::DADD_reg => DoubleOp::Add, Opcode::DMUL_reg => DoubleOp::Multiply, _ => DoubleOp::Fma };
                let mods = if fma {
                    FMods { neg_b: ((raw >> 48) & 1) != 0, neg_c: ((raw >> 49) & 1) != 0, ..FMods::default() }
                } else {
                    FMods {
                        neg_a: ((raw >> 48) & 1) != 0,
                        abs_a: opcode == Opcode::DADD_reg && ((raw >> 46) & 1) != 0,
                        neg_b: opcode == Opcode::DADD_reg && ((raw >> 45) & 1) != 0,
                        abs_b: opcode == Opcode::DADD_reg && ((raw >> 49) & 1) != 0,
                        ..FMods::default()
                    }
                };
                self.write_double(reg_dest(raw), op, self.double_pair(reg_a(raw)), self.double_pair(reg_b(raw)),
                    if fma { self.double_pair(reg_c(raw)) } else { [Value::Zero; 2] }, mods, pred);
            }

            Opcode::F2F_reg => {
                let s = self.read_reg(reg_b(raw));
                if !self.emit_f2f(raw, s, pred) { return false; }
            }
            Opcode::F2F_cbuf => {
                let cb = self.load_cbuf(raw);
                if !self.emit_f2f(raw, Value::Inst(cb), pred) { return false; }
            }
            Opcode::F2F_imm => {
                if !self.emit_f2f(raw, Value::ImmF32(float_imm20(raw)), pred) { return false; }
            }

            Opcode::RRO_reg => {
                let s = self.read_reg(reg_b(raw));
                self.write_reg(
                    reg_dest(raw),
                    Op::F2F {
                        src: s,
                        neg: i2f_neg(raw),
                        abs: i2f_abs(raw),
                        sat: false,
                        round: 0,
                    },
                    pred,
                );
            }
            Opcode::RRO_cbuf => {
                let cb = self.load_cbuf(raw);
                self.write_reg(
                    reg_dest(raw),
                    Op::F2F {
                        src: Value::Inst(cb),
                        neg: i2f_neg(raw),
                        abs: i2f_abs(raw),
                        sat: false,
                        round: 0,
                    },
                    pred,
                );
            }

            Opcode::I2F_reg => {
                let s = self.read_reg(reg_b(raw));
                self.emit_i2f(raw, s, pred);
            }
            Opcode::I2F_cbuf => {
                let cb = self.load_cbuf(raw);
                self.emit_i2f(raw, Value::Inst(cb), pred);
            }
            Opcode::I2F_imm => {
                self.emit_i2f(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::FSET_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_fset(raw, b, pred);
            }
            Opcode::FSET_cbuf => {
                let cb = self.load_cbuf(raw);
                self.emit_fset(raw, Value::Inst(cb), pred);
            }
            Opcode::FSET_imm => {
                self.emit_fset(raw, Value::ImmF32(float_imm20(raw)), pred);
            }

            Opcode::ISETP_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_isetp(raw, b, pred);
            }
            Opcode::ISETP_cbuf => {
                let cb = self.load_cbuf(raw);
                self.emit_isetp(raw, Value::Inst(cb), pred);
            }
            Opcode::ISETP_imm => {
                self.emit_isetp(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::SSY
            | Opcode::SYNC
            | Opcode::PBK
            | Opcode::BRK
            | Opcode::PCNT
            | Opcode::CONT => {}

            Opcode::IMAD_reg => {
                let b = self.read_reg(reg_b(raw));
                let c = self.read_reg(reg_c(raw));
                self.emit_imad(raw, b, c, pred);
            }
            Opcode::IMAD_cr => {
                let cb = self.load_cbuf(raw);
                let c = self.read_reg(reg_c(raw));
                self.emit_imad(raw, Value::Inst(cb), c, pred);
            }
            Opcode::IMAD_rc => {
                let b = self.read_reg(reg_c(raw));
                let cb = self.load_cbuf(raw);
                self.emit_imad(raw, b, Value::Inst(cb), pred);
            }
            Opcode::IMAD_imm => {
                let c = self.read_reg(reg_c(raw));
                self.emit_imad(raw, Value::ImmU32(imm20(raw) as u32), c, pred);
            }

            Opcode::IMNMX_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_imnmx(raw, b, pred);
            }
            Opcode::IMNMX_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_imnmx(raw, Value::Inst(id), pred);
            }
            Opcode::IMNMX_imm => {
                self.emit_imnmx(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::FMNMX_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                let mods = fmnmx_mods(raw);
                let op = Op::FMinMaxPred {
                    a,
                    b,
                    mods,
                    pred: fmnmx_pred(raw),
                    neg_pred: fmnmx_neg_pred(raw),
                };
                self.write_reg(dest, op, pred);
            }
            Opcode::FMNMX_cbuf => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                let b = Value::Inst(cb_id);
                let mods = fmnmx_mods(raw);
                let op = Op::FMinMaxPred {
                    a,
                    b,
                    mods,
                    pred: fmnmx_pred(raw),
                    neg_pred: fmnmx_neg_pred(raw),
                };
                self.write_reg(dest, op, pred);
            }
            Opcode::FMNMX_imm => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = Value::ImmF32(float_imm20(raw));
                let mods = fmnmx_mods(raw);
                let op = Op::FMinMaxPred {
                    a,
                    b,
                    mods,
                    pred: fmnmx_pred(raw),
                    neg_pred: fmnmx_neg_pred(raw),
                };
                self.write_reg(dest, op, pred);
            }

            Opcode::AL2P => {
                if ((raw >> 47) & 3) != 0 || self.stage == ShaderStage::Compute {
                    self.program.emit_void(Op::Unimplemented { opcode: Opcode::AL2P, raw });
                    self.unimplemented_count += 1;
                    return false;
                }
                let source = self.read_reg(reg_a(raw));
                let offset = (((raw >> 20) & 0x7ff) as i32) << 21 >> 21;
                self.write_reg(reg_dest(raw), Op::IAdd {
                    a: source, b: Value::ImmU32(offset as u32), neg_a: false, neg_b: false,
                }, pred);
            }

            Opcode::ISBERD if self.stage == ShaderStage::Geometry && raw & 0x0001_ffff_ffff_0000 == 0x0000_0000_0007_0000 => {
                self.write_reg(reg_dest(raw), Op::Mov(self.read_reg(reg_a(raw))), pred);
            }
            Opcode::OUT_reg if self.stage == ShaderStage::Geometry && reg_b(raw) == RZ => {
                if raw & (1 << 39) != 0 { self.program.emit_void_pred(Op::EmitVertex, pred); }
                if raw & (1 << 40) != 0 { self.program.emit_void_pred(Op::EndPrimitive, pred); }
                self.write_reg(reg_dest(raw), Op::Mov(Value::Zero), pred);
            }
            Opcode::VMAD if raw & ((1 << 47) | (0x1f << 51)) == 0 => {
                let a = self.video_operand(self.read_reg(reg_a(raw)), ((raw >> 37) & 3) as u32,
                    ((raw >> 36) & 3) as u32, raw & (1 << 48) != 0);
                let immediate = raw & (1 << 50) == 0;
                let b = if immediate { Value::ImmU32(((raw >> 20) & 0xffff) as u32) } else { self.read_reg(reg_b(raw)) };
                let b = self.video_operand(b, if immediate { 2 } else { ((raw >> 29) & 3) as u32 },
                    if immediate { 0 } else { ((raw >> 28) & 3) as u32 }, raw & (1 << 49) != 0);
                let product = self.emit_value(Op::IMul { a, b });
                self.write_reg(reg_dest(raw), Op::IAdd { a: product, b: self.read_reg(reg_c(raw)), neg_a: false, neg_b: false }, pred);
            }
            Opcode::VSETP => {
                let a = self.video_operand(self.read_reg(reg_a(raw)), ((raw >> 37) & 3) as u32,
                    ((raw >> 36) & 3) as u32, raw & (1 << 48) != 0);
                let immediate = raw & (1 << 50) == 0;
                let b = if immediate { Value::ImmU32(((raw >> 20) & 0xffff) as u32) } else { self.read_reg(reg_b(raw)) };
                let b = self.video_operand(b, if immediate { 2 } else { ((raw >> 29) & 3) as u32 },
                    if immediate { 0 } else { ((raw >> 28) & 3) as u32 }, raw & (1 << 49) != 0);
                let compare = ((raw >> 43) & 0x1f) as u64;
                let compare = (compare & 3) | ((compare >> 2) & 4);
                let dest_p = ((raw >> 3) & 7) as u8;
                let dest_np = (raw & 7) as u8;
                let id = self.program.emit_pred(Op::ISetPred {
                    cmp: ICmp::from_bits(compare), signed: raw & (1 << 49) != 0,
                    bop: BoolOp::from_bits((raw >> 45) & 3), src_a: a, src_b: b,
                    src_pred: ((raw >> 39) & 7) as u8, src_pred_inv: raw & (1 << 42) != 0,
                    dest_p, dest_np,
                }, None, pred);
                if dest_p != PT { self.pred_state.insert(dest_p, id); }
                if dest_np != PT { self.pred_state.insert(dest_np, id); }
            }
            Opcode::ALD => {
                let base_dest = reg_dest(raw);
                let base_slot = attr_slot_ald(raw);
                let n = ald_num_elements(raw);
                for elem in 0..n {
                    let dest = base_dest.wrapping_add(elem as u8);
                    let slot = base_slot + elem * 4;
                    let op = if self.stage == ShaderStage::Geometry {
                        Op::LoadGeometryAttr { slot, vertex: self.read_reg(reg_c(raw)) }
                    } else if reg_a(raw) != RZ {
                        let address = self.read_reg(reg_a(raw));
                        let address = self.emit_iadd_value(address, Value::ImmU32(elem * 4), false, false);
                        Op::LoadAttrIndexed { address }
                    } else {
                        Op::LoadAttr { slot }
                    };
                    self.write_reg(dest, op, pred);
                }
            }
            Opcode::AST => {
                let base_src = reg_dest(raw);
                let base_slot = attr_slot_ald(raw);
                let n = ald_num_elements(raw);
                for elem in 0..n {
                    let src_reg = base_src.wrapping_add(elem as u8);
                    let src = self.read_reg(src_reg);
                    let slot = base_slot + elem * 4;
                    self.program
                        .emit_void_pred(Op::StoreAttr { slot, src }, pred);
                }
            }
            Opcode::IPA => {
                let dest = reg_dest(raw);
                let perspective = self.read_reg(reg_b(raw));
                self.write_reg(
                    dest,
                    if ((raw >> 38) & 1) != 0 && reg_a(raw) != RZ {
                        Op::InterpAttrIndexed {
                            address: self.read_reg(reg_a(raw)),
                            perspective,
                            mode: ipa_interpolation_mode(raw),
                            sat: ipa_saturate(raw),
                        }
                    } else {
                        Op::InterpAttr {
                            slot: attr_slot_ipa(raw),
                            perspective,
                            mode: ipa_interpolation_mode(raw),
                            sat: ipa_saturate(raw),
                        }
                    },
                    pred,
                );
            }

            Opcode::TEX => {
                let Some(form) = decode_texture_sample_form(raw, false) else {
                    log::debug!("TEX unsupported form raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                let tex_id = texs_tex_id(raw);
                let compute_handle = if self.stage == ShaderStage::Compute {
                    let dimension = match form.tex_type {
                        0 => ImageDimension::D1,
                        2 => ImageDimension::D2,
                        3 if form.bias_reg.is_none() => ImageDimension::D2Array,
                        4 => ImageDimension::D3,
                        _ => {
                            self.program.emit_void(Op::Unimplemented {
                                opcode: Opcode::TEX,
                                raw,
                            });
                            self.unimplemented_count += 1;
                            return false;
                        }
                    };
                    Some((
                        TextureHandleOrigin::Bound {
                            cbuf_word_offset: tex_id,
                        },
                        dimension,
                    ))
                } else {
                    None
                };
                if self
                    .emit_texture_sample(raw, tex_id, compute_handle, form, pred)
                    .is_none()
                {
                    log::debug!("TEX runtime AOFFI unsupported raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }

            Opcode::TEX_b => {
                let Some(form) = decode_texture_sample_form(raw, true) else {
                    log::debug!("TEX_b unsupported form raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX_b,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                let handle_reg = reg_b(raw);
                let handle = self.read_reg(handle_reg);
                let Some((origin, deferred)) = self.trace_cbuf_handle_origin(&handle, pred, defs)
                else {
                    log::debug!("TEX_b handle not traceable to LDC raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX_b,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                if let Some(partner) = origin.cross_binding_partner_id() {
                    self.bindless_or_partners
                        .insert(origin.texture_id(), partner);
                }
                let compute_handle = if self.stage == ShaderStage::Compute {
                    let dimension = match form.tex_type {
                        0 => ImageDimension::D1,
                        2 => ImageDimension::D2,
                        3 if form.bias_reg.is_none() => ImageDimension::D2Array,
                        4 => ImageDimension::D3,
                        _ => {
                            self.program.emit_void(Op::Unimplemented {
                                opcode: Opcode::TEX_b,
                                raw,
                            });
                            self.unimplemented_count += 1;
                            return false;
                        }
                    };
                    Some((origin.as_texture_handle(), dimension))
                } else {
                    None
                };
                let Some(samples) =
                    self.emit_texture_sample(raw, origin.texture_id(), compute_handle, form, pred)
                else {
                    log::debug!("TEX_b runtime AOFFI unsupported raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX_b,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                if deferred {
                    self.pending_bindless_origin_checks
                        .push(PendingBindlessOriginCheck {
                            opcode: Opcode::TEX_b,
                            raw,
                            handle,
                            consumer_pred: pred,
                            samples,
                        });
                }
            }

            Opcode::TXD => {
                let aoffi = ((raw >> 35) & 1) != 0;
                let lc = ((raw >> 50) & 1) != 0;
                let sparse_pred = ((raw >> 51) & 0x7) as u8;
                let tex_type = ((raw >> 28) & 0x7) as u8;
                let mask = ((raw >> 31) & 0xf) as u8;
                if self.stage != ShaderStage::Fragment
                    || aoffi
                    || lc
                    || sparse_pred != PT
                    || tex_type != 2
                    || mask == 0
                {
                    log::debug!(
                        "TXD unsupported form raw={:#018x} stage={:?} type={} mask={:#x} aoffi={} lc={} sparse_pred={}",
                        raw,
                        self.stage,
                        tex_type,
                        mask,
                        aoffi,
                        lc,
                        sparse_pred,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TXD,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let coord = reg_a(raw);
                let derivative = reg_b(raw);
                let u = self.read_reg(coord);
                let v = self.read_reg(coord.wrapping_add(1));
                let dpdx = (
                    self.read_reg(derivative),
                    self.read_reg(derivative.wrapping_add(2)),
                );
                let dpdy = (
                    self.read_reg(derivative.wrapping_add(1)),
                    self.read_reg(derivative.wrapping_add(3)),
                );
                let sample_site = self.program.next_value_id();
                self.program.emit_void(Op::TextureGradients {
                    sample_site,
                    dpdx,
                    dpdy,
                });

                let tex_id = texs_tex_id(raw);
                let mut dst = reg_dest(raw);
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(
                        dst,
                        Op::SampleTex {
                            sample_site: Some(sample_site),
                            tex_id,
                            u,
                            v,
                            array: None,
                            volume: None,
                            cube: None,
                            implicit_lod: false,
                            lod_bias: None,
                            explicit_lod: None,
                            texel_offset: None,
                            dref: None,
                            component,
                        },
                        pred,
                    );
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::TLD | Opcode::TLD_b => {
                let bindless = decoded.opcode == Opcode::TLD_b;
                let lod = ((raw >> 55) & 1) != 0;
                let multisample = ((raw >> 50) & 1) != 0;
                let aoffi = ((raw >> 35) & 1) != 0;
                let clamp = ((raw >> 54) & 1) != 0;
                let sparse_pred = ((raw >> 51) & 0x7) as u8;
                let tex_type = ((raw >> 28) & 0x7) as u8;
                let mask = ((raw >> 31) & 0xF) as u8;
                if lod
                    || multisample
                    || aoffi
                    || clamp
                    || sparse_pred != PT
                    || !matches!(tex_type, 0 | 2 | 4)
                    || mask == 0
                {
                    log::debug!(
                        "TLD unsupported form raw={:#018x} bindless={} type={} mask={:#x} lod={} ms={} aoffi={} clamp={} sparse_pred={}",
                        raw,
                        bindless,
                        tex_type,
                        mask,
                        lod,
                        multisample,
                        aoffi,
                        clamp,
                        sparse_pred,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                if !bindless && self.stage != ShaderStage::Compute {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TLD,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let (handle_origin, pending_handle) = if bindless {
                    let handle = self.read_reg(reg_b(raw));
                    let Some((origin, deferred)) =
                        self.trace_cbuf_handle_origin(&handle, pred, defs)
                    else {
                        log::debug!("TLD_b handle not traceable to LDC raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TLD_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    if origin.cross_binding_partner_id().is_some() {
                        log::debug!("TLD_b cross-buffer handle is unsupported raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TLD_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                    (origin.as_texture_handle(), deferred.then_some(handle))
                } else {
                    (
                        TextureHandleOrigin::Bound {
                            cbuf_word_offset: ((raw >> 36) & 0x1fff) as u32,
                        },
                        None,
                    )
                };

                let coord = reg_a(raw);
                let x = self.read_reg(coord);
                let (dimension, y, z) = match tex_type {
                    0 => (ImageDimension::D1, None, None),
                    2 => (
                        ImageDimension::D2,
                        Some(self.read_reg(coord.wrapping_add(1))),
                        None,
                    ),
                    4 => (
                        ImageDimension::D3,
                        Some(self.read_reg(coord.wrapping_add(1))),
                        Some(self.read_reg(coord.wrapping_add(2))),
                    ),
                    _ => unreachable!(),
                };
                let mut dst = reg_dest(raw);
                let mut fetches = Vec::new();
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    let old_value = self.read_reg(dst);
                    let instruction_index = self.program.instructions.len();
                    self.write_reg(
                        dst,
                        if self.stage == ShaderStage::Compute || !bindless {
                            Op::TexelFetchHandle {
                                handle: handle_origin,
                                dimension,
                                x,
                                y,
                                z,
                                component,
                            }
                        } else {
                            let TextureHandleOrigin::Bindless {
                                cbuf_binding,
                                cbuf_word_offset,
                                cbuf_secondary_word_offset,
                            } = handle_origin
                            else {
                                unreachable!()
                            };
                            Op::TexelFetch {
                                cbuf_binding,
                                cbuf_word_offset,
                                cbuf_secondary_word_offset,
                                x,
                                y,
                                z,
                                component,
                            }
                        },
                        pred,
                    );
                    fetches.push((instruction_index, old_value));
                    dst = dst.wrapping_add(1);
                }
                if let Some(handle) = pending_handle {
                    self.pending_bindless_origin_checks
                        .push(PendingBindlessOriginCheck {
                            opcode: Opcode::TLD_b,
                            raw,
                            handle,
                            consumer_pred: pred,
                            samples: fetches,
                        });
                }
            }

            Opcode::TLD4S => {
                let dest_a = reg_dest(raw);
                let dest_b = ((raw >> 28) & 0xFF) as u8;
                let ra = reg_a(raw);
                let rb = reg_b(raw);
                let tex_id = texs_tex_id(raw);
                let gather_component = ((raw >> 52) & 0x3) as u8;
                let aoffi = ((raw >> 51) & 0x1) != 0;
                let dc = ((raw >> 50) & 0x1) != 0;
                let fp16 = ((raw >> 55) & 0x1) != 0;
                if aoffi
                    || dc
                    || (!fp16
                        && [dest_a, dest_b]
                            .into_iter()
                            .any(|dest| dest != RZ && dest & 1 != 0))
                {
                    log::debug!(
                        "TLD4S unsupported form raw={:#018x} aoffi={} dc={} fp16={} dest_a={} dest_b={}",
                        raw,
                        aoffi,
                        dc,
                        fp16,
                        dest_a,
                        dest_b,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TLD4S,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let u = self.read_reg(ra);
                let v = self.read_reg(rb);
                if fp16 {
                    let gathered: [Value; 4] = std::array::from_fn(|lane| {
                        Value::Inst(self.program.emit(
                            Op::GatherTex {
                                tex_id,
                                u,
                                v,
                                array: None,
                                gather_component,
                                lane: lane as u8,
                            },
                            None,
                        ))
                    });
                    self.write_reg(
                        dest_a,
                        Op::PackHalf2 {
                            lo: gathered[0],
                            hi: gathered[1],
                        },
                        pred,
                    );
                    self.write_reg(
                        dest_b,
                        Op::PackHalf2 {
                            lo: gathered[2],
                            hi: gathered[3],
                        },
                        pred,
                    );
                } else {
                    for lane in 0..4u8 {
                        let pair_dest = if lane < 2 { dest_a } else { dest_b };
                        let dst_reg = if pair_dest == RZ {
                            RZ
                        } else {
                            pair_dest + (lane & 1)
                        };
                        self.write_reg(
                            dst_reg,
                            Op::GatherTex {
                                tex_id,
                                u,
                                v,
                                array: None,
                                gather_component,
                                lane,
                            },
                            pred,
                        );
                    }
                }
            }

            Opcode::TEXS | Opcode::TLDS => {
                let dest_a = reg_dest(raw);
                let dest_b = ((raw >> 28) & 0xFF) as u8;
                let ra = reg_a(raw);
                let rb = reg_b(raw);
                let enc = (raw >> 53) & 0xF;
                let is_texs = matches!(decoded.opcode, Opcode::TEXS);
                let compute_sample = if is_texs && self.stage == ShaderStage::Compute {
                    let sample = match enc {
                        0 => (
                            ImageDimension::D1,
                            self.read_reg(ra),
                            None,
                            None,
                            false,
                            Some(Value::Zero),
                        ),
                        1 => (
                            ImageDimension::D2,
                            self.read_reg(ra),
                            Some(self.read_reg(rb)),
                            None,
                            true,
                            None,
                        ),
                        2 => (
                            ImageDimension::D2,
                            self.read_reg(ra),
                            Some(self.read_reg(rb)),
                            None,
                            false,
                            Some(Value::Zero),
                        ),
                        3 => (
                            ImageDimension::D2,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            None,
                            false,
                            Some(self.read_reg(rb)),
                        ),
                        7 | 8 | 9 => (
                            ImageDimension::D2Array,
                            self.read_reg(ra.wrapping_add(1)),
                            Some(self.read_reg(rb)),
                            Some(self.read_reg(ra)),
                            enc == 7,
                            (enc != 7).then_some(Value::Zero),
                        ),
                        10 => (
                            ImageDimension::D3,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            Some(self.read_reg(rb)),
                            true,
                            None,
                        ),
                        11 => (
                            ImageDimension::D3,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            Some(self.read_reg(rb)),
                            false,
                            Some(Value::Zero),
                        ),
                        12 => (
                            ImageDimension::Cube,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            Some(self.read_reg(rb)),
                            true,
                            None,
                        ),
                        13 => (
                            ImageDimension::Cube,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            Some(self.read_reg(rb)),
                            false,
                            Some(self.read_reg(rb.wrapping_add(1))),
                        ),
                        _ => {
                            self.program.emit_void(Op::Unimplemented {
                                opcode: decoded.opcode,
                                raw,
                            });
                            self.unimplemented_count += 1;
                            return false;
                        }
                    };
                    Some(sample)
                } else {
                    None
                };
                let compute_fetch = if !is_texs {
                    let fetch = match enc {
                        0 => (ImageDimension::D1, self.read_reg(ra), None, None),
                        2 => (
                            ImageDimension::D2,
                            self.read_reg(ra),
                            Some(self.read_reg(rb)),
                            None,
                        ),
                        4 if ra == RZ || ra & 1 == 0 => {
                            let offsets = self.read_reg(rb);
                            let dx = self.emit_bfe_value(offsets, 0, 4, true);
                            let dy = self.emit_bfe_value(offsets, 4, 4, true);
                            let x = self.program.emit(Op::IAdd {
                                a: self.read_reg(ra), b: dx, neg_a: false, neg_b: false,
                            }, None);
                            let y = self.program.emit(Op::IAdd {
                                a: if ra == RZ { Value::Zero } else { self.read_reg(ra + 1) },
                                b: dy, neg_a: false, neg_b: false,
                            }, None);
                            (ImageDimension::D2, Value::Inst(x), Some(Value::Inst(y)), None)
                        }
                        7 => (
                            ImageDimension::D3,
                            self.read_reg(ra),
                            Some(self.read_reg(ra.wrapping_add(1))),
                            Some(self.read_reg(rb)),
                        ),
                        _ => {
                            self.program.emit_void(Op::Unimplemented {
                                opcode: decoded.opcode,
                                raw,
                            });
                            self.unimplemented_count += 1;
                            return false;
                        }
                    };
                    Some(fetch)
                } else {
                    None
                };
                let array_2d = is_texs && matches!(enc, 7 | 8 | 9);
                let tex_3d = is_texs && matches!(enc, 10 | 11);
                let tex_cube = is_texs && matches!(enc, 12 | 13);
                let paired_uv = is_texs && matches!(enc, 3 | 4 | 5 | 6);
                let (u, v) = if array_2d {
                    (self.read_reg(ra.wrapping_add(1)), self.read_reg(rb))
                } else if tex_3d || tex_cube || paired_uv {
                    (self.read_reg(ra), self.read_reg(ra.wrapping_add(1)))
                } else {
                    (self.read_reg(ra), self.read_reg(rb))
                };
                let array = if array_2d {
                    Some(self.read_reg(ra))
                } else {
                    None
                };
                let volume = if tex_3d {
                    Some(self.read_reg(rb))
                } else {
                    None
                };
                let cube = if tex_cube {
                    Some(self.read_reg(rb))
                } else {
                    None
                };
                let tex_id = texs_tex_id(raw);
                let swizzle = ((raw >> 50) & 0x7) as usize;
                let implicit_lod = is_texs && matches!(enc, 1 | 4 | 7 | 10 | 12);
                let explicit_lod = if is_texs {
                    match enc {
                        3 | 5 => Some(self.read_reg(rb)),
                        13 => Some(self.read_reg(rb.wrapping_add(1))),
                        _ => None,
                    }
                } else {
                    None
                };
                let dref = if is_texs {
                    match enc {
                        4 | 6 => Some(self.read_reg(rb)),
                        5 | 9 => Some(self.read_reg(rb.wrapping_add(1))),
                        _ => None,
                    }
                } else {
                    None
                };
                let handle = TextureHandleOrigin::Bound {
                    cbuf_word_offset: tex_id,
                };
                let sample_site = self.program.next_value_id();
                let compact_texture_op = |component| {
                    if dref.is_some() && component == 3 {
                        Op::Mov(Value::ImmF32(1.0))
                    } else if let Some((dimension, u, v, w, implicit_lod, explicit_lod)) =
                        compute_sample
                    {
                        Op::SampleTexHandle {
                            sample_site: Some(sample_site),
                            handle,
                            dimension,
                            u,
                            v,
                            w,
                            implicit_lod,
                            lod_bias: None,
                            explicit_lod,
                            texel_offset: None,
                            dref,
                            component,
                        }
                    } else if let Some((dimension, x, y, z)) = compute_fetch {
                        Op::TexelFetchHandle {
                            handle,
                            dimension,
                            x,
                            y,
                            z,
                            component,
                        }
                    } else {
                        Op::SampleTex {
                            sample_site: Some(sample_site),
                            tex_id,
                            u,
                            v,
                            array,
                            volume,
                            cube,
                            implicit_lod,
                            lod_bias: None,
                            explicit_lod,
                            texel_offset: None,
                            dref,
                            component,
                        }
                    }
                };

                const RG_LUT: [u8; 8] = [1, 2, 4, 8, 1 | 2, 1 | 8, 2 | 8, 4 | 8];
                const RGBA_LUT: [u8; 5] =
                    [1 | 2 | 4, 1 | 2 | 8, 1 | 4 | 8, 2 | 4 | 8, 1 | 2 | 4 | 8];
                let mask = if dest_b == super::operand::RZ {
                    RG_LUT[swizzle.min(7)]
                } else if swizzle < RGBA_LUT.len() {
                    RGBA_LUT[swizzle]
                } else {
                    1
                };

                let fp16 = ((raw >> 59) & 1) == 0;
                if fp16 {
                    let mut sampled: Vec<Value> = Vec::new();
                    for component in 0..4u8 {
                        if (mask >> component) & 1 == 0 {
                            continue;
                        }
                        let id = self.program.emit(compact_texture_op(component), None);
                        sampled.push(Value::Inst(id));
                    }
                    for (i, pair) in sampled.chunks(2).enumerate() {
                        let dst_reg = if i == 0 { dest_a } else { dest_b };
                        let lo = pair[0];
                        let hi = pair.get(1).copied().unwrap_or(Value::Zero);
                        self.write_reg(dst_reg, Op::PackHalf2 { lo, hi }, pred);
                    }
                } else {
                    let mut store_index: u8 = 0;
                    for component in 0..4u8 {
                        if (mask >> component) & 1 == 0 {
                            continue;
                        }
                        let dst_reg = match store_index {
                            0 => dest_a,
                            1 => dest_a.wrapping_add(1),
                            2 => dest_b,
                            _ => dest_b.wrapping_add(1),
                        };
                        self.write_reg(dst_reg, compact_texture_op(component), pred);
                        store_index += 1;
                    }
                }
            }

            Opcode::TMML | Opcode::TMML_b => {
                let bindless = decoded.opcode == Opcode::TMML_b;
                let tex_type = ((raw >> 28) & 0x7) as u8;
                let mask = ((raw >> 31) & 0xf) as u8;
                if self.stage != ShaderStage::Fragment
                    || !matches!(tex_type, 2 | 3)
                    || mask & 0b1100 != 0
                {
                    log::debug!(
                        "TMML unsupported form opcode={:?} raw={:#018x} stage={:?} type={} mask={:#x}",
                        decoded.opcode,
                        raw,
                        self.stage,
                        tex_type,
                        mask,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let handle = if bindless {
                    let meta_reg = ((raw >> 20) & 0xff) as u8;
                    let handle_value = self.read_reg(meta_reg);
                    let Some((origin, false)) =
                        self.trace_cbuf_handle_origin(&handle_value, pred, defs)
                    else {
                        log::debug!(
                            "TMML_b handle is not an exact static cbuf origin raw={:#018x}",
                            raw,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TMML_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    if origin.cross_binding_partner_id().is_some() {
                        log::debug!(
                            "TMML_b cross-buffer handle is unsupported raw={:#018x}",
                            raw,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TMML_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                    origin.as_texture_handle()
                } else {
                    TextureHandleOrigin::Bound {
                        cbuf_word_offset: ((raw >> 36) & 0x1fff) as u32,
                    }
                };
                let arrayed = tex_type == 3;
                let coord_reg = reg_a(raw).wrapping_add(u8::from(arrayed));
                let u = self.read_reg(coord_reg);
                let v = self.read_reg(coord_reg.wrapping_add(1));
                let mut dst = reg_dest(raw);
                for component in 0..2u8 {
                    if mask >> component & 1 == 0 {
                        continue;
                    }
                    let queried = self.emit_value(Op::TextureQueryLod {
                        handle,
                        u,
                        v,
                        arrayed,
                        component,
                    });
                    let integer = self.emit_value(Op::F2I {
                        src: queried,
                        signed: false,
                        round: 3,
                    });
                    self.write_reg(
                        dst,
                        Op::IShl {
                            a: integer,
                            b: Value::ImmU32(8),
                        },
                        pred,
                    );
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::TXQ | Opcode::TXQ_b => {
                let mask = ((raw >> 31) & 0xF) as u8;
                let mode = ((raw >> 22) & 0x7) as u8;
                if mode != 1 || mask == 0 {
                    log::debug!(
                        "TXQ unsupported form raw={:#018x} mode={} mask={:#x}",
                        raw,
                        mode,
                        mask,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let bindless = decoded.opcode == Opcode::TXQ_b;
                let source = reg_a(raw);
                let handle = if bindless {
                    let handle_value = self.read_reg(source);
                    let Some((origin, false)) =
                        self.trace_cbuf_handle_origin(&handle_value, pred, defs)
                    else {
                        log::debug!("TXQ_b handle not traceable to LDC raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TXQ_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    if origin.cross_binding_partner_id().is_some() {
                        log::debug!("TXQ_b cross-buffer handle is unsupported raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TXQ_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                    origin.as_texture_handle()
                } else {
                    TextureHandleOrigin::Bound {
                        cbuf_word_offset: ((raw >> 36) & 0x1fff) as u32,
                    }
                };
                let lod = self.read_reg(source.wrapping_add(u8::from(bindless)));
                let mut dst = reg_dest(raw);
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(
                        dst,
                        Op::TextureQueryDimension {
                            handle,
                            lod,
                            component,
                        },
                        pred,
                    );
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::TLD4 | Opcode::TLD4_b => {
                let bindless = decoded.opcode == Opcode::TLD4_b;
                let mask = ((raw >> 31) & 0xF) as u8;
                let tex_type = ((raw >> 28) & 0x7) as u8;
                let dc = ((raw >> 50) & 0x1) != 0;
                let sparse_pred = ((raw >> 51) & 0x7) as u8;
                let (offset_type, gather_component) = if bindless {
                    (((raw >> 36) & 0x3) as u8, ((raw >> 38) & 0x3) as u8)
                } else {
                    (((raw >> 54) & 0x3) as u8, ((raw >> 56) & 0x3) as u8)
                };
                if (self.stage == ShaderStage::Compute && bindless)
                    || mask == 0
                    || !matches!(tex_type, 2 | 3)
                    || dc
                    || sparse_pred != PT
                    || offset_type != 0
                {
                    log::debug!(
                        "TLD4 unsupported form raw={:#018x} bindless={} stage={:?} type={} mask={:#x} dc={} sparse_pred={} offset={}",
                        raw,
                        bindless,
                        self.stage,
                        tex_type,
                        mask,
                        dc,
                        sparse_pred,
                        offset_type,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let tex_id = if bindless {
                    let handle = self.read_reg(reg_b(raw));
                    let Some((origin, deferred)) =
                        self.trace_cbuf_handle_origin(&handle, pred, defs)
                    else {
                        log::debug!("TLD4_b handle not traceable to LDC raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TLD4_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    if deferred {
                        log::debug!(
                            "TLD4_b deferred handle origin is unsupported raw={:#018x}",
                            raw,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::TLD4_b,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                    if let Some(partner) = origin.cross_binding_partner_id() {
                        self.bindless_or_partners
                            .insert(origin.texture_id(), partner);
                    }
                    origin.texture_id()
                } else {
                    texs_tex_id(raw)
                };

                let coord = reg_a(raw);
                let arrayed = tex_type == 3;
                let array = arrayed.then(|| self.read_reg(coord));
                let u = self.read_reg(coord.wrapping_add(u8::from(arrayed)));
                let v = self.read_reg(coord.wrapping_add(1 + u8::from(arrayed)));
                let mut dst = reg_dest(raw);
                for lane in 0..4u8 {
                    if (mask >> lane) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(
                        dst,
                        Op::GatherTex {
                            tex_id,
                            u,
                            v,
                            array,
                            gather_component,
                            lane,
                        },
                        pred,
                    );
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::MUFU => {
                let dest = reg_dest(raw);
                let src = self.read_reg(reg_a(raw));
                let func = MufuFunc::from_bits(mufu_func_bits(raw));
                let mods = FMods {
                    abs_a: ((raw >> 46) & 1) != 0,
                    neg_a: ((raw >> 48) & 1) != 0,
                    sat: ((raw >> 50) & 1) != 0,
                    ..FMods::default()
                };
                self.write_reg(dest, Op::MultiFunc { src, func, mods }, pred);
            }

            Opcode::EXIT => {
                let flow_test = (raw & 0x1F) as u32;
                if flow_test == 0x1C {
                    log::debug!("EXIT FCSM_TR treated as fall-through raw={:#018x}", raw);
                } else if pred.is_none() {
                    self.program.emit_void(Op::Exit);
                    self.program.exit_reg_state = Some(self.reg_state.clone());
                    self.finished = true;
                }
            }

            Opcode::LDC => {
                let mode = ldc_mode(raw);
                if matches!(mode, LdcMode::Il | LdcMode::Isl) {
                    log::warn!("LDC addressing mode {mode:?} is unsupported raw={raw:#018x}",);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LDC,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let dest = reg_dest(raw);
                let src_reg = ldc_src_reg(raw);
                let size = ldc_size(raw);
                if size == 5 && dest != RZ && dest & 1 != 0 {
                    log::warn!(
                        "LDC B64 requires an even destination register raw={raw:#018x} dest=R{dest}",
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LDC,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let count = match size {
                    0..=4 => 1u32,
                    5 => 2,
                    6 => 4,
                    _ => {
                        log::warn!(
                            "LDC with unsupported size raw={:#018x} size={}",
                            raw,
                            size,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::LDC,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                };
                let r = ldc_ref(raw);
                let index = if src_reg != RZ {
                    self.read_reg(src_reg)
                } else {
                    Value::Zero
                };
                for w in 0..count {
                    let bo = (r.byte_offset as u32).wrapping_add(w * 4);
                    let cb_id = match mode {
                        LdcMode::Default if src_reg == RZ => self.program.emit(
                            Op::LoadCbuf {
                                binding: r.binding,
                                byte_offset: bo,
                            },
                            None,
                        ),
                        LdcMode::Default => self.program.emit(
                            Op::LoadCbufIndexed {
                                binding: r.binding,
                                byte_offset: bo,
                                index,
                                address_mode: CbufAddressMode::Default,
                            },
                            None,
                        ),
                        LdcMode::Is => self.program.emit(
                            Op::LoadCbufIndexed {
                                binding: r.binding,
                                byte_offset: bo,
                                index,
                                address_mode: CbufAddressMode::Segmented,
                            },
                            None,
                        ),
                        LdcMode::Il | LdcMode::Isl => unreachable!(),
                    };
                    let dst = if dest == RZ {
                        RZ
                    } else {
                        dest.wrapping_add(w as u8)
                    };
                    let value = if size < 4 {
                        let address = self.emit_iadd_value(index, Value::ImmU32(bo), false, false);
                        let shifted = self.emit_value(Op::IShl {
                            a: address,
                            b: Value::ImmU32(3),
                        });
                        let position = self.emit_value(Op::ILop {
                            a: shifted,
                            b: Value::ImmU32(if size < 2 { 24 } else { 16 }),
                            op: LogicOp::And,
                            not_a: false,
                            not_b: false,
                        });
                        let control = self.emit_iadd_value(
                            position,
                            Value::ImmU32(if size < 2 { 8 << 8 } else { 16 << 8 }),
                            false,
                            false,
                        );
                        self.emit_value(Op::Bfe {
                            a: Value::Inst(cb_id),
                            b: control,
                            signed: size & 1 != 0,
                        })
                    } else {
                        Value::Inst(cb_id)
                    };
                    self.write_reg(dst, Op::Mov(value), pred);
                }
            }

            Opcode::LDG => {
                let dest = reg_dest(raw);
                let addr_reg = ldg_addr_reg(raw);
                let offset = ldg_offset(raw);
                let size = ldg_size(raw);
                let count = match size {
                    4 => 1u32,
                    5 => 2,
                    6 | 7 => 4,
                    _ => {
                        log::warn!(
                            "LDG sub-word size not yet lifted raw={:#018x} size={}",
                            raw,
                            size,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::LDG,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                };
                let addr_lo = self.read_reg(addr_reg);
                for w in 0..count {
                    let off = offset.wrapping_add((w * 4) as i32);
                    let id = self.program.emit_pred(
                        Op::LoadGlobal {
                            addr_lo,
                            offset: off,
                        },
                        None,
                        pred,
                    );
                    let dst = if dest == RZ {
                        RZ
                    } else {
                        dest.wrapping_add(w as u8)
                    };
                    self.write_reg(dst, Op::Mov(Value::Inst(id)), pred);
                }
            }

            Opcode::STG => {
                let src = reg_dest(raw);
                let addr_reg = ldg_addr_reg(raw);
                let offset = ldg_offset(raw);
                let size = ldg_size(raw);
                let count = match size {
                    4 => 1u32,
                    5 => 2,
                    6 | 7 => 4,
                    _ => {
                        log::warn!(
                            "STG sub-word size not yet lifted raw={:#018x} size={}",
                            raw,
                            size,
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::STG,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                };
                let addr_lo = self.read_reg(addr_reg);
                for w in 0..count {
                    let off = offset.wrapping_add((w * 4) as i32);
                    let value = self.read_reg(if src == RZ {
                        RZ
                    } else {
                        src.wrapping_add(w as u8)
                    });
                    self.program.emit_void_pred(
                        Op::StoreGlobal {
                            addr_lo,
                            offset: off,
                            value,
                        },
                        pred,
                    );
                }
            }

            Opcode::ATOM => {
                let size = ((raw >> 49) & 7) as u8;
                let operation = ((raw >> 52) & 15) as u8;
                if size > 1 || operation > 8 {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::ATOM,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let address = reg_a(raw);
                let immediate = ((raw >> 28) & 0xfffff) as u32;
                let offset = if address == RZ {
                    immediate as i32
                } else {
                    ((immediate << 12) as i32) >> 12
                };
                let addr_lo = self.read_reg(address);
                let value = self.read_reg(reg_b(raw));
                let op = match operation {
                    0 => ImageAtomicOp::Add,
                    1 => ImageAtomicOp::Min,
                    2 => ImageAtomicOp::Max,
                    3 => ImageAtomicOp::Increment,
                    4 => ImageAtomicOp::Decrement,
                    5 => ImageAtomicOp::And,
                    6 => ImageAtomicOp::Or,
                    7 => ImageAtomicOp::Xor,
                    _ => ImageAtomicOp::Exchange,
                };
                let operation =
                    if size == 1 && matches!(op, ImageAtomicOp::Increment | ImageAtomicOp::Decrement) {
                        Op::LoadGlobal { addr_lo, offset }
                    } else {
                        Op::GlobalAtomic {
                            addr_lo,
                            offset,
                            value,
                            op,
                            is_signed: size == 1,
                        }
                    };
                self.write_side_effecting_reg(reg_dest(raw), operation, pred);
            }
            Opcode::RED => {
                let operand = reg_dest(raw);
                let addr_reg = reg_a(raw);
                let size = ((raw >> 20) & 7) as u8;
                let atomic_op = ((raw >> 23) & 7) as u8;
                if size > 1 || (self.stage != ShaderStage::Compute && new_fs_ops_disabled()) {
                    log::warn!(
                        "RED non-32-bit size not yet lifted raw={:#018x} size={}",
                        raw,
                        size,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::RED,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let is_signed = size == 1;
                let op = match atomic_op {
                    0 => ImageAtomicOp::Add,
                    1 => ImageAtomicOp::Min,
                    2 => ImageAtomicOp::Max,
                    3 => ImageAtomicOp::Increment,
                    4 => ImageAtomicOp::Decrement,
                    5 => ImageAtomicOp::And,
                    6 => ImageAtomicOp::Or,
                    _ => ImageAtomicOp::Xor,
                };
                if is_signed && matches!(op, ImageAtomicOp::Increment | ImageAtomicOp::Decrement) {
                    return true;
                }
                let offset = if addr_reg == RZ {
                    ((raw >> 28) & 0xf_ffff) as i32
                } else {
                    (((((raw >> 28) & 0xf_ffff) as i64) << 44) >> 44) as i32
                };
                let addr_lo = self.read_reg(addr_reg);
                let value = self.read_reg(operand);
                self.program.emit_void_pred(
                    Op::GlobalAtomic {
                        addr_lo,
                        offset,
                        value,
                        op,
                        is_signed,
                    },
                    pred,
                );
            }

            Opcode::LDL => {
                let dest = reg_dest(raw);
                let addr = self.ldl_addr(raw);
                let count = ldls_word_count(raw);
                for w in 0..count {
                    let a = if w == 0 {
                        addr
                    } else {
                        self.emit_value(Op::IAdd {
                            a: addr,
                            b: Value::ImmU32(4 * w),
                            neg_a: false,
                            neg_b: false,
                        })
                    };
                    let id = self.program.emit(Op::LoadLocal { addr: a }, None);
                    let dst = if dest == RZ {
                        RZ
                    } else {
                        dest.wrapping_add(w as u8)
                    };
                    self.write_reg(dst, Op::Mov(Value::Inst(id)), pred);
                }
            }
            Opcode::STL => {
                let src = reg_dest(raw);
                let addr = self.ldl_addr(raw);
                let count = ldls_word_count(raw);
                for w in 0..count {
                    let a = if w == 0 {
                        addr
                    } else {
                        self.emit_value(Op::IAdd {
                            a: addr,
                            b: Value::ImmU32(4 * w),
                            neg_a: false,
                            neg_b: false,
                        })
                    };
                    let value = self.read_reg(src.wrapping_add(w as u8));
                    self.program
                        .emit_void_pred(Op::StoreLocal { addr: a, value }, pred);
                }
            }

            Opcode::LDS if self.stage == ShaderStage::Compute => {
                let Some(count) = Self::shared_word_count(raw) else {
                    log::debug!(
                        "LDS unsupported shared-memory width raw={:#018x} size={}",
                        raw,
                        (raw >> 48) & 0x7,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LDS,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                let dest = reg_dest(raw);
                if dest.checked_add((count - 1) as u8).is_none() {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LDS,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let base = self.shared_addr(raw);
                for word in 0..count {
                    let addr = if word == 0 {
                        base
                    } else {
                        self.emit_value(Op::IAdd {
                            a: base,
                            b: Value::ImmU32(word * 4),
                            neg_a: false,
                            neg_b: false,
                        })
                    };
                    let value = self.program.emit(Op::LoadShared { addr }, None);
                    self.write_reg(
                        dest.wrapping_add(word as u8),
                        Op::Mov(Value::Inst(value)),
                        pred,
                    );
                }
            }
            Opcode::STS if self.stage == ShaderStage::Compute => {
                let Some(count) = Self::shared_word_count(raw) else {
                    log::debug!(
                        "STS unsupported shared-memory width raw={:#018x} size={}",
                        raw,
                        (raw >> 48) & 0x7,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::STS,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                let source = reg_dest(raw);
                if source.checked_add((count - 1) as u8).is_none() {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::STS,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let base = self.shared_addr(raw);
                for word in 0..count {
                    let addr = if word == 0 {
                        base
                    } else {
                        self.emit_value(Op::IAdd {
                            a: base,
                            b: Value::ImmU32(word * 4),
                            neg_a: false,
                            neg_b: false,
                        })
                    };
                    let value = self.read_reg(source.wrapping_add(word as u8));
                    self.program
                        .emit_void_pred(Op::StoreShared { addr, value }, pred);
                }
            }

            Opcode::ATOMS if self.stage == ShaderStage::Compute => {
                let size = ((raw >> 28) & 0x3) as u8;
                let atomic_op = ((raw >> 52) & 0xf) as u8;
                if size != 0 || atomic_op != 6 {
                    log::debug!(
                        "ATOMS unsupported form raw={:#018x} size={} op={}",
                        raw,
                        size,
                        atomic_op,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::ATOMS,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let dest = reg_dest(raw);
                let addr = self.shared_atomic_addr(raw);
                let value = self.read_reg(reg_b(raw));
                self.write_side_effecting_reg(
                    dest,
                    Op::SharedAtomic {
                        addr,
                        value,
                        op: ImageAtomicOp::Or,
                    },
                    pred,
                );
            }

            Opcode::BAR if self.stage == ShaderStage::Compute => {
                const WORKGROUP_SYNC: u64 = 0xf0a8_1b80_0007_0000;
                if raw != WORKGROUP_SYNC || pred.is_some() {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::BAR,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                self.program.emit_void(Op::WorkgroupBarrier);
            }

            Opcode::MEMBAR if self.stage == ShaderStage::Compute => {
                if pred.is_some() {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::MEMBAR,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let scope = if ((raw >> 8) & 0x3) == 0 {
                    MemoryBarrierScope::Workgroup
                } else {
                    MemoryBarrierScope::Device
                };
                self.program.emit_void(Op::MemoryBarrier { scope });
            }

            Opcode::SUATOM if self.stage == ShaderStage::Compute => {
                let is_bindless = ((raw >> 54) & 1) != 0;
                let atomic_op = ((raw >> 29) & 0xf) as u8;
                let surface_type = ((raw >> 33) & 0x7) as u8;
                let size = ((raw >> 51) & 0x7) as u8;
                let clamp = ((raw >> 49) & 0x3) as u8;
                let op = match atomic_op {
                    0 => Some(ImageAtomicOp::Add),
                    1 => Some(ImageAtomicOp::Min),
                    2 => Some(ImageAtomicOp::Max),
                    3 => Some(ImageAtomicOp::Increment),
                    4 => Some(ImageAtomicOp::Decrement),
                    5 => Some(ImageAtomicOp::And),
                    6 => Some(ImageAtomicOp::Or),
                    7 => Some(ImageAtomicOp::Xor),
                    8 => Some(ImageAtomicOp::Exchange),
                    _ => None,
                };
                let data_type = match size {
                    0 => Some(ImageAtomicType::U32),
                    1 => Some(ImageAtomicType::S32),
                    6 => Some(ImageAtomicType::Sd32),
                    _ => None,
                };
                if surface_type != 1 || data_type.is_none() || clamp != 0 || op.is_none() {
                    log::debug!(
                        "SUATOM unsupported form raw={:#018x} type={} size={} clamp={} op={}",
                        raw,
                        surface_type,
                        size,
                        clamp,
                        atomic_op,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SUATOM,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let (handle, pending_handle) = if is_bindless {
                    let handle_reg = ((raw >> 39) & 0xff) as u8;
                    let handle_value = self.read_reg(handle_reg);
                    let Some((origin, deferred)) =
                        self.trace_cbuf_handle_origin(&handle_value, pred, defs)
                    else {
                        log::debug!("SUATOM handle not traceable to LDC raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::SUATOM,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    (origin.as_texture_handle(), deferred.then_some(handle_value))
                } else {
                    (
                        TextureHandleOrigin::Bound {
                            cbuf_word_offset: ((raw >> 36) & 0x1fff) as u32,
                        },
                        None,
                    )
                };

                let dest = (raw & 0xff) as u8;
                let coord = ((raw >> 8) & 0xff) as u8;
                let operand = ((raw >> 20) & 0xff) as u8;
                let (_, instruction_index) = self.write_side_effecting_reg(
                    dest,
                    Op::ImageAtomic {
                        handle,
                        dimension: ImageDimension::Buffer,
                        x: self.read_reg(coord),
                        y: None,
                        z: None,
                        value: self.read_reg(operand),
                        op: op.unwrap(),
                        data_type: data_type.unwrap(),
                    },
                    pred,
                );
                if let Some(handle) = pending_handle {
                    self.pending_bindless_origin_checks
                        .push(PendingBindlessOriginCheck {
                            opcode: Opcode::SUATOM,
                            raw,
                            handle,
                            consumer_pred: pred,
                            samples: vec![(instruction_index, Value::Zero)],
                        });
                }
            }

            Opcode::SUST if self.stage == ShaderStage::Compute => {
                let is_bound = ((raw >> 51) & 1) != 0;
                let typed = ((raw >> 52) & 1) != 0;
                let surface_type = ((raw >> 33) & 0x7) as u8;
                let cache = ((raw >> 24) & 0x3) as u8;
                let swizzle = ((raw >> 20) & 0xf) as u8;
                let clamp = ((raw >> 49) & 0x3) as u8;
                let Some(dimension) = (match surface_type {
                    0 => Some(ImageDimension::D1),
                    1 => Some(ImageDimension::Buffer),
                    3 => Some(ImageDimension::D2),
                    5 => Some(ImageDimension::D3),
                    _ => None,
                }) else {
                    log::debug!(
                        "SUST unsupported surface type raw={:#018x} type={}",
                        raw,
                        surface_type,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SUST,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                };
                if typed || swizzle != 0xf || clamp != 0 || !matches!(cache, 0 | 1) {
                    log::debug!(
                        "SUST unsupported form raw={:#018x} typed={} swizzle={:#x} clamp={} cache={}",
                        raw,
                        typed,
                        swizzle,
                        clamp,
                        cache,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SUST,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let (handle, pending_handle) = if is_bound {
                    (TextureHandleOrigin::Bound {
                        cbuf_word_offset: ((raw >> 36) & 0x1fff) as u32,
                    }, None)
                } else {
                    let handle_reg = ((raw >> 39) & 0xff) as u8;
                    let handle_value = self.read_reg(handle_reg);
                    let Some((origin, deferred)) =
                        self.trace_cbuf_handle_origin(&handle_value, pred, defs)
                    else {
                        log::debug!("SUST handle not traceable to LDC raw={:#018x}", raw);
                        self.program.emit_void(Op::Unimplemented {
                            opcode: Opcode::SUST,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    };
                    (origin.as_texture_handle(), deferred.then_some(handle_value))
                };

                let coord = reg_a(raw);
                let x = self.read_reg(coord);
                let (y, z) = match dimension {
                    ImageDimension::D1 | ImageDimension::Buffer => (None, None),
                    ImageDimension::D2 => (Some(self.read_reg(coord.wrapping_add(1))), None),
                    ImageDimension::D2Array | ImageDimension::D3 | ImageDimension::Cube => (
                        Some(self.read_reg(coord.wrapping_add(1))),
                        Some(self.read_reg(coord.wrapping_add(2))),
                    ),
                };
                let data = reg_dest(raw);
                let values = [
                    self.read_reg(data),
                    self.read_reg(data.wrapping_add(1)),
                    self.read_reg(data.wrapping_add(2)),
                    self.read_reg(data.wrapping_add(3)),
                ];
                let instruction_index = self.program.instructions.len();
                self.program.emit_void_pred(
                    Op::ImageWrite {
                        handle,
                        dimension,
                        x,
                        y,
                        z,
                        values,
                    },
                    pred,
                );
                if let Some(handle) = pending_handle {
                    self.pending_bindless_origin_checks.push(PendingBindlessOriginCheck {
                        opcode: Opcode::SUST, raw, handle, consumer_pred: pred,
                        samples: vec![(instruction_index, Value::Zero)],
                    });
                }
            }

            Opcode::FSETP_reg | Opcode::FSETP_cbuf | Opcode::FSETP_imm => {
                let dest_p = fsetp_dest_p(raw);
                let dest_np = fsetp_dest_np(raw);
                let src_a = self.read_reg(reg_a(raw));
                let src_b = match decoded.opcode {
                    Opcode::FSETP_cbuf => {
                        let cb = self.load_cbuf(raw);
                        Value::Inst(cb)
                    }
                    Opcode::FSETP_imm => Value::ImmF32(float_imm20(raw)),
                    _ => self.read_reg(reg_b(raw)),
                };
                let cmp = FComp::from_bits(fsetp_cmp(raw));
                let bop = BoolOp::from_bits(fsetp_bop(raw));
                let src_pred = fsetp_src_pred(raw);
                let src_pred_inv = fsetp_src_pred_inv(raw);
                let fsp = Op::FSetPred {
                    cmp,
                    bop,
                    src_a,
                    src_b,
                    neg_a: fsetp_neg_a(raw),
                    abs_a: fsetp_abs_a(raw),
                    neg_b: fsetp_neg_b(raw),
                    abs_b: fsetp_abs_b(raw),
                    src_pred,
                    src_pred_inv,
                    dest_p,
                    dest_np,
                };
                let id = self.program.emit_pred(fsp, None, pred);
                if dest_p != PT {
                    self.pred_state.insert(dest_p, id);
                }

                if dest_np != PT {
                    self.pred_state.insert(dest_np, id);
                }
            }

            Opcode::PSETP => {
                self.emit_psetp(raw, pred);
            }
            Opcode::CSETP => {
                if !self.emit_csetp(raw, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::CSETP,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }
            Opcode::PSET => {
                self.emit_pset(raw, pred);
            }

            Opcode::R2P_reg | Opcode::R2P_cbuf | Opcode::R2P_imm => {
                if ((raw >> 40) & 1) != 0 {
                    log::debug!(
                        "R2P condition-code mode is not yet modeled raw={:#018x}",
                        raw,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                match decoded.opcode {
                    Opcode::R2P_reg => {
                        let mask = self.read_reg(reg_b(raw));
                        self.emit_r2p(raw, mask, None, pred);
                    }
                    Opcode::R2P_cbuf => {
                        let mask = Value::Inst(self.load_cbuf(raw));
                        self.emit_r2p(raw, mask, None, pred);
                    }
                    Opcode::R2P_imm => {
                        let mask = imm20(raw) as u32;
                        self.emit_r2p(raw, Value::ImmU32(mask), Some(mask), pred);
                    }
                    _ => unreachable!(),
                }
            }

            Opcode::KIL => {
                self.program.emit_void_pred(Op::Kill, pred);
            }

            Opcode::IADD_reg => {
                let b = self.read_reg(reg_b(raw));
                if !self.emit_iadd(raw, b, pred) { return false; }
            }
            Opcode::IADD_cbuf => {
                let id = self.load_cbuf(raw);
                if !self.emit_iadd(raw, Value::Inst(id), pred) { return false; }
            }
            Opcode::IADD_imm => {
                if !self.emit_iadd(raw, Value::ImmU32(imm20(raw) as u32), pred) { return false; }
            }
            Opcode::IADD32I => {
                if !self.emit_iadd(raw, Value::ImmU32(imm32(raw)), pred) { return false; }
            }

            Opcode::IMUL_imm => {
                if ((raw >> 39) & 0x1ff) != 0 {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::IMUL_imm,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                self.emit_imul(
                    raw,
                    Value::ImmU32(imm20(raw) as u32),
                    pred,
                    false,
                    false,
                    false,
                );
            }
            Opcode::IMUL_reg => {
                self.emit_imul(
                    raw,
                    self.read_reg(reg_b(raw)),
                    pred,
                    ((raw >> 53) & 1) != 0,
                    ((raw >> 54) & 1) != 0,
                    ((raw >> 55) & 1) != 0,
                );
            }
            Opcode::IMUL32I => {
                self.emit_imul(
                    raw,
                    Value::ImmU32(imm32(raw)),
                    pred,
                    ((raw >> 53) & 1) != 0,
                    ((raw >> 54) & 1) != 0,
                    ((raw >> 55) & 1) != 0,
                );
            }

            Opcode::PRMT_imm => {
                if !self.emit_prmt_imm(raw, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::PRMT_imm,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }

            Opcode::IADD3_reg => {
                self.emit_iadd3_reg(raw, pred);
            }
            Opcode::IADD3_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_iadd3(raw, Value::Inst(id), pred);
            }
            Opcode::IADD3_imm => {
                self.emit_iadd3(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::SEL_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_sel(raw, b, pred);
            }
            Opcode::SEL_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_sel(raw, Value::Inst(id), pred);
            }
            Opcode::SEL_imm => {
                self.emit_sel(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::ISCADD_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_iscadd(raw, b, pred);
            }
            Opcode::ISCADD_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_iscadd(raw, Value::Inst(id), pred);
            }
            Opcode::ISCADD_imm => {
                self.emit_iscadd(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::SHF_l_imm => {
                if !self.emit_shf_l_imm(raw, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SHF_l_imm,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }
            Opcode::SHF_l_reg => {
                if !self.emit_shf_l_reg(raw, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SHF_l_reg,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }
            Opcode::SHF_r_imm => {
                if !self.emit_shf_r_imm(raw, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::SHF_r_imm,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }

            Opcode::SHL_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_ishl(raw, b, pred);
            }
            Opcode::SHL_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_ishl(raw, Value::Inst(id), pred);
            }
            Opcode::SHL_imm => {
                self.emit_ishl(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::SHR_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_ishr(raw, b, pred);
            }
            Opcode::SHR_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_ishr(raw, Value::Inst(id), pred);
            }
            Opcode::SHR_imm => {
                self.emit_ishr(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::LOP_reg => {
                let b = self.read_reg(reg_b(raw));
                let result = self.emit_ilop(
                    raw,
                    b,
                    LogicOp::from_bits(lop_op(raw)),
                    lop_not_a(raw),
                    lop_not_b(raw),
                    pred,
                );
                self.emit_lop_predicate(raw, result, pred);
            }
            Opcode::LOP_cbuf => {
                let id = self.load_cbuf(raw);
                let result = self.emit_ilop(
                    raw,
                    Value::Inst(id),
                    LogicOp::from_bits(lop_op(raw)),
                    lop_not_a(raw),
                    lop_not_b(raw),
                    pred,
                );
                self.emit_lop_predicate(raw, result, pred);
            }
            Opcode::LOP_imm => {
                let result = self.emit_ilop(
                    raw,
                    Value::ImmU32(imm20(raw) as u32),
                    LogicOp::from_bits(lop_op(raw)),
                    lop_not_a(raw),
                    lop_not_b(raw),
                    pred,
                );
                self.emit_lop_predicate(raw, result, pred);
            }
            Opcode::LOP32I => {
                self.emit_ilop(
                    raw,
                    Value::ImmU32(imm32(raw)),
                    LogicOp::from_bits(lop32i_op(raw)),
                    lop32i_not_a(raw),
                    lop32i_not_b(raw),
                    pred,
                );
            }
            opcode @ (Opcode::LOP3_reg | Opcode::LOP3_cbuf | Opcode::LOP3_imm) => {
                let cc = ((raw >> 47) & 1) != 0;
                let register_form = opcode == Opcode::LOP3_reg;
                let extended = register_form && ((raw >> 38) & 1) != 0;
                if cc || extended {
                    self.program.emit_void(Op::Unimplemented {
                        opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }

                let a = self.read_reg(reg_a(raw));
                let b = match opcode {
                    Opcode::LOP3_reg => self.read_reg(reg_b(raw)),
                    Opcode::LOP3_cbuf => Value::Inst(self.load_cbuf(raw)),
                    _ => Value::ImmU32(imm20(raw) as u32),
                };
                let c = self.read_reg(reg_c(raw));
                let lut = ((raw >> if register_form { 28 } else { 48 }) & 0xff) as u8;
                let result = self.write_reg(
                    reg_dest(raw),
                    Op::ILop3 {
                        a,
                        b,
                        c,
                        lut,
                    },
                    pred,
                );
                if register_form {
                    self.emit_logic_predicate(
                        ((raw >> 48) & 7) as u8,
                        ((raw >> 36) & 3) as u8,
                        result,
                        pred,
                    );
                }
            }

            Opcode::VMNMX => {
                let operation = ((raw >> 51) & 7) as u8;
                if ((raw >> 47) & 1) != 0 || ((raw >> 55) & 1) != 0 || !matches!(operation, 5 | 6) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::VMNMX,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let immediate = ((raw >> 50) & 1) == 0;
                let mut a = self.read_reg(reg_a(raw));
                let mut b = if immediate {
                    Value::ImmU32(((raw >> 20) & 0xffff) as u32)
                } else {
                    self.read_reg(reg_b(raw))
                };
                let c = self.read_reg(reg_c(raw));
                let a_width = (raw >> 37) & 3;
                let b_width = if immediate { 2 } else { (raw >> 29) & 3 };
                let a_signed = ((raw >> 48) & 1) != 0;
                let b_signed = ((raw >> 49) & 1) != 0;
                if a_width != 3 {
                    let bits = if a_width == 2 { 16 } else { 8 };
                    a = self.emit_bfe_value(a, ((raw >> 36) & 3) as u32 * bits, bits, a_signed);
                }
                if b_width != 3 {
                    let bits = if b_width == 2 { 16 } else { 8 };
                    let selector = if immediate {
                        0
                    } else {
                        ((raw >> 28) & 3) as u32
                    };
                    b = self.emit_bfe_value(b, selector * bits, bits, b_signed);
                }
                let first = self.emit_value(Op::IMinMaxPred {
                    a,
                    b,
                    signed: b_signed,
                    pred: PT,
                    neg_pred: ((raw >> 56) & 1) != 0,
                });
                self.write_reg(
                    reg_dest(raw),
                    Op::IMinMaxPred {
                        a: first,
                        b: c,
                        signed: ((raw >> 54) & 1) != 0,
                        pred: PT,
                        neg_pred: operation == 6,
                    },
                    pred,
                );
            }
            Opcode::FLO_reg => {
                let tilde = ((raw >> 40) & 0x1) != 0;
                let shift = ((raw >> 41) & 0x1) != 0;
                let unsupported_modifiers = ((raw >> 47) & 0x1) != 0
                    || ((raw >> 48) & 0x1) != 0
                    || ((tilde || shift) && new_fs_ops_disabled());
                if unsupported_modifiers {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::FLO_reg,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let mut source = self.read_reg(reg_b(raw));
                if tilde {
                    source = self.emit_value(Op::ILop {
                        a: Value::Zero,
                        b: source,
                        op: LogicOp::PassB,
                        not_a: false,
                        not_b: true,
                    });
                }
                if shift {
                    let msb = self.emit_value(Op::FindUMsb { value: source });
                    let found_mask = self.emit_value(Op::ISet {
                        cmp: ICmp::Ne,
                        signed: false,
                        a: msb,
                        b: Value::ImmU32(0xFFFF_FFFF),
                        bool_float: false,
                    });
                    let xor_mask = self.emit_value(Op::ILop {
                        a: found_mask,
                        b: Value::ImmU32(31),
                        op: LogicOp::And,
                        not_a: false,
                        not_b: false,
                    });
                    self.write_reg(
                        reg_dest(raw),
                        Op::ILop {
                            a: msb,
                            b: xor_mask,
                            op: LogicOp::Xor,
                            not_a: false,
                            not_b: false,
                        },
                        pred,
                    );
                } else {
                    self.write_reg(reg_dest(raw), Op::FindUMsb { value: source }, pred);
                }
            }

            opcode @ (Opcode::LEA_lo_reg | Opcode::LEA_lo_cbuf | Opcode::LEA_lo_imm) => {
                let scale = ((raw >> 39) & 0x1F) as u8;
                let neg = ((raw >> 45) & 1) != 0;
                let x = ((raw >> 46) & 1) != 0;
                let cc = ((raw >> 47) & 1) != 0;
                let src_pred = ((raw >> 48) & 0x7) as u8;
                if x || cc || src_pred != PT || new_fs_ops_disabled() {
                    log::debug!(
                        "LEA_lo unsupported form opcode={:?} raw={:#018x} x={} cc={} src_pred={}",
                        opcode,
                        raw,
                        x,
                        cc,
                        src_pred,
                    );
                    self.program.emit_void(Op::Unimplemented { opcode, raw });
                    self.unimplemented_count += 1;
                    return false;
                }
                let base = match opcode {
                    Opcode::LEA_lo_reg => self.read_reg(reg_b(raw)),
                    Opcode::LEA_lo_cbuf => Value::Inst(self.load_cbuf(raw)),
                    Opcode::LEA_lo_imm => Value::ImmU32(imm20(raw) as u32),
                    _ => unreachable!(),
                };
                let offset = self.read_reg(reg_a(raw));
                self.write_reg(
                    reg_dest(raw),
                    Op::IScAdd {
                        a: offset,
                        b: base,
                        shift: scale,
                        neg_a: neg,
                        neg_b: false,
                    },
                    pred,
                );
            }

            Opcode::LEA_hi_reg => {
                let neg = ((raw >> 37) & 1) != 0;
                let x = ((raw >> 38) & 1) != 0;
                let cc = ((raw >> 47) & 1) != 0;
                let src_pred = ((raw >> 48) & 0x7) as u8;
                if neg || x || cc || src_pred != PT || new_fs_ops_disabled() {
                    log::debug!("LEA_hi_reg unsupported form raw={:#018x}", raw);
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LEA_hi_reg,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                let scale = ((raw >> 28) & 0x1F) as u32;
                let rc = ((raw >> 39) & 0xFF) as u8;
                let base = self.read_reg(reg_b(raw));
                let shifted = if scale == 0 {
                    self.read_reg(rc)
                } else {
                    let lo = self.read_reg(reg_a(raw));
                    let lo_part = self.emit_value(Op::IShr {
                        a: lo,
                        b: Value::ImmU32(32 - scale),
                        signed: false,
                    });
                    if rc == RZ {
                        lo_part
                    } else {
                        let hi = self.read_reg(rc);
                        let hi_part = self.emit_value(Op::IShl {
                            a: hi,
                            b: Value::ImmU32(scale),
                        });
                        self.emit_value(Op::ILop {
                            a: lo_part,
                            b: hi_part,
                            op: LogicOp::Or,
                            not_a: false,
                            not_b: false,
                        })
                    }
                };
                self.write_reg(
                    reg_dest(raw),
                    Op::IAdd {
                        a: base,
                        b: shifted,
                        neg_a: false,
                        neg_b: false,
                    },
                    pred,
                );
            }

            Opcode::POPC_reg => {
                let mut source = self.read_reg(reg_b(raw));
                if ((raw >> 40) & 1) != 0 {
                    source = self.emit_value(Op::ILop {
                        a: Value::Zero,
                        b: source,
                        op: LogicOp::PassB,
                        not_a: false,
                        not_b: true,
                    });
                }
                self.write_reg(reg_dest(raw), Op::BitCount { value: source }, pred);
            }

            Opcode::XMAD_reg => {
                self.emit_xmad_reg(raw, pred);
            }
            Opcode::XMAD_rc => {
                self.emit_xmad_rc(raw, pred);
            }
            Opcode::XMAD_cr => {
                self.emit_xmad_cr(raw, pred);
            }
            Opcode::XMAD_imm => {
                self.emit_xmad_imm(raw, pred);
            }

            Opcode::BFE_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_bfe(raw, b, pred);
            }
            Opcode::BFE_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_bfe(raw, Value::Inst(id), pred);
            }
            Opcode::BFE_imm => {
                self.emit_bfe(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }

            Opcode::BFI_reg => {
                let dest = reg_dest(raw);
                let insert = self.read_reg(reg_a(raw));
                let control = self.read_reg(reg_b(raw));
                let base = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::Bfi {
                        base,
                        insert,
                        control,
                    },
                    pred,
                );
            }
            Opcode::BFI_cr => {
                let dest = reg_dest(raw);
                let insert = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                let base = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::Bfi {
                        base,
                        insert,
                        control: Value::Inst(cb_id),
                    },
                    pred,
                );
            }
            Opcode::BFI_rc => {
                let dest = reg_dest(raw);
                let insert = self.read_reg(reg_a(raw));
                let control = self.read_reg(reg_c(raw));
                let cb_id = self.load_cbuf(raw);
                self.write_reg(
                    dest,
                    Op::Bfi {
                        base: Value::Inst(cb_id),
                        insert,
                        control,
                    },
                    pred,
                );
            }
            Opcode::BFI_imm => {
                let dest = reg_dest(raw);
                let insert = self.read_reg(reg_a(raw));
                let base = self.read_reg(reg_c(raw));
                self.write_reg(
                    dest,
                    Op::Bfi {
                        base,
                        insert,
                        control: Value::ImmU32(imm20(raw) as u32),
                    },
                    pred,
                );
            }

            Opcode::F2I_reg => {
                let src = self.read_reg(reg_b(raw));
                self.emit_f2i(raw, src, pred);
            }
            Opcode::F2I_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_f2i(raw, Value::Inst(id), pred);
            }
            Opcode::F2I_imm => {
                self.emit_f2i(raw, Value::ImmF32(float_imm20(raw)), pred);
            }

            Opcode::I2I_reg => {
                let src = self.read_reg(reg_b(raw));
                if !self.emit_i2i(raw, src, pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }
            Opcode::I2I_cbuf => {
                let id = self.load_cbuf(raw);
                if !self.emit_i2i(raw, Value::Inst(id), pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }
            Opcode::I2I_imm => {
                if !self.emit_i2i(raw, Value::ImmU32(imm20(raw) as u32), pred) {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: decoded.opcode,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
            }

            Opcode::ISET_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_iset(raw, b, pred);
            }
            Opcode::ISET_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_iset(raw, Value::Inst(id), pred);
            }
            Opcode::ISET_imm => {
                self.emit_iset(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }
            Opcode::ICMP_reg => {
                let select_src = self.read_reg(reg_b(raw));
                let compare_src = self.read_reg(reg_c(raw));
                self.emit_icmp(raw, select_src, compare_src, pred);
            }
            Opcode::ICMP_rc => {
                let select_src = self.read_reg(reg_c(raw));
                let compare_src = Value::Inst(self.load_cbuf(raw));
                self.emit_icmp(raw, select_src, compare_src, pred);
            }
            Opcode::ICMP_cr => {
                let select_src = Value::Inst(self.load_cbuf(raw));
                let compare_src = self.read_reg(reg_c(raw));
                self.emit_icmp(raw, select_src, compare_src, pred);
            }
            Opcode::ICMP_imm => {
                let select_src = Value::ImmU32(imm20(raw) as u32);
                let compare_src = self.read_reg(reg_c(raw));
                self.emit_icmp(raw, select_src, compare_src, pred);
            }

            Opcode::DEPBAR => {}

            Opcode::PIXLD => {
                let mode = (raw >> 31) & 7;
                let addr_reg = ((raw >> 8) & 0xff) as u8;
                let offset = (raw >> 20) & 0xff;
                let dest_pred = ((raw >> 45) & 7) as u8;
                if self.stage != ShaderStage::Fragment
                    || mode != 5
                    || addr_reg != RZ
                    || offset != 0
                    || dest_pred != PT
                {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::PIXLD,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                self.write_reg(reg_dest(raw), Op::SampleId, pred);
            }

            Opcode::S2R => {
                let dest = reg_dest(raw);
                let sr = ((raw >> 20) & 0xFF) as u32;
                if self.stage == ShaderStage::Compute {
                    match sr {
                        0 => {
                            self.write_reg(dest, Op::SubgroupLaneId, pred);
                        }
                        0x20 => {
                            let x = self.emit_value(Op::LocalInvocationId { component: 0 });
                            let y = self.emit_value(Op::LocalInvocationId { component: 1 });
                            let z = self.emit_value(Op::LocalInvocationId { component: 2 });
                            let xy = self.emit_value(Op::Bfi {
                                base: x,
                                insert: y,
                                control: Value::ImmU32((8 << 8) | 16),
                            });
                            let packed = self.emit_value(Op::Bfi {
                                base: xy,
                                insert: z,
                                control: Value::ImmU32((6 << 8) | 26),
                            });
                            self.write_reg(dest, Op::Mov(packed), pred);
                        }
                        0x21..=0x23 => {
                            self.write_reg(
                                dest,
                                Op::LocalInvocationId {
                                    component: (sr - 0x21) as u8,
                                },
                                pred,
                            );
                        }
                        0x25..=0x27 => {
                            self.write_reg(
                                dest,
                                Op::WorkgroupId {
                                    component: (sr - 0x25) as u8,
                                },
                                pred,
                            );
                        }
                        0x38..=0x3c => {
                            let kind = match sr {
                                0x38 => SubgroupMask::Eq,
                                0x39 => SubgroupMask::Lt,
                                0x3a => SubgroupMask::Le,
                                0x3b => SubgroupMask::Gt,
                                0x3c => SubgroupMask::Ge,
                                _ => unreachable!(),
                            };
                            self.write_reg(dest, Op::SubgroupMask { kind }, pred);
                        }
                        _ => {
                            log::debug!("unsupported compute S2R sr={} raw={:#018x}", sr, raw);
                            self.program.emit_void(Op::Unimplemented {
                                opcode: Opcode::S2R,
                                raw,
                            });
                            self.unimplemented_count += 1;
                            return false;
                        }
                    }
                } else {
                    match sr {
                        0 => {
                            self.write_reg(dest, Op::SubgroupLaneId, pred);
                        }
                        0x38..=0x3c => {
                            let kind = match sr {
                                0x38 => SubgroupMask::Eq,
                                0x39 => SubgroupMask::Lt,
                                0x3a => SubgroupMask::Le,
                                0x3b => SubgroupMask::Gt,
                                0x3c => SubgroupMask::Ge,
                                _ => unreachable!(),
                            };
                            self.write_reg(dest, Op::SubgroupMask { kind }, pred);
                        }
                        0x1d if self.stage == ShaderStage::Geometry => {
                            self.write_reg(dest, Op::GeometryInvocationInfo, pred);
                        }
                        18 => {
                            self.write_reg(dest, Op::YDirection, pred);
                        }
                        19 if self.stage == ShaderStage::Fragment => {
                            self.write_reg(dest, Op::HelperInvocation, pred);
                        }
                        30 | 31 => {
                            self.write_reg(dest, Op::Mov(Value::ImmU32(0x3F80_0000)), pred);
                        }
                        _ => {
                            log::debug!("S2R sr={} â†’ 0 (approx) raw={:#018x}", sr, raw);
                            self.write_reg(dest, Op::Mov(Value::ImmU32(0)), pred);
                        }
                    }
                }
            }

            Opcode::VOTE_vtg if self.stage != ShaderStage::Compute => {}

            Opcode::VOTE => {
                let vote_mode = ((raw >> 48) & 0x3) as u8;
                let vote_gated = self.stage != ShaderStage::Compute && new_fs_ops_disabled();
                let mode = match (vote_mode, vote_gated) {
                    (0, false) => VoteMode::All,
                    (1, false) => VoteMode::Any,
                    (2, false) => VoteMode::Equal,
                    _ => {
                        log::debug!(
                            "unsupported {:?} mode={} stage={:?} raw={:#018x}",
                            decoded.opcode,
                            vote_mode,
                            self.stage,
                            raw
                        );
                        self.program.emit_void(Op::Unimplemented {
                            opcode: decoded.opcode,
                            raw,
                        });
                        self.unimplemented_count += 1;
                        return false;
                    }
                };
                let dest = reg_dest(raw);
                let pred_dest = ((raw >> 45) & 0x7) as u8;
                let old = self.read_reg(dest);
                let id = self.program.emit_pred(
                    Op::SubgroupVote {
                        source_pred: Predicate {
                            idx: ((raw >> 39) & 0x7) as u8,
                            negate: ((raw >> 42) & 1) != 0,
                        },
                        mode,
                        pred_dest,
                        old,
                    },
                    Some(dest),
                    pred,
                );
                if dest != RZ {
                    self.reg_state.insert(dest, Value::Inst(id));
                }
                if pred_dest != PT {
                    self.pred_state.insert(pred_dest, id);
                }
            }

            other => {
                log::debug!(
                    "SASS opcode decoded but not lowered opcode={:?} raw={:#018x}",
                    other,
                    raw,
                );
                self.program
                    .emit_void(Op::Unimplemented { opcode: other, raw });
                self.unimplemented_count += 1;
                return false;
            }
        }
        true
    }
}

fn new_fs_ops_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_DISABLE_NEW_FS_OPS").is_some())
}

impl Default for Translator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn translate_shader(bytes: &[u8]) -> Translator {
    let mut t = Translator::new();
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        if t.finished {
            break;
        }
        let offset = i * 8;
        if offset % 0x20 == 0 {
            continue;
        }
        let raw = u64::from_le_bytes(chunk.try_into().unwrap());
        t.translate(raw);
    }
    t
}

pub fn translate_compute_shader(bytes: &[u8]) -> Translator {
    let mut t = Translator::new_compute();
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        if t.finished {
            break;
        }
        let offset = i * 8;
        if offset % 0x20 == 0 {
            continue;
        }
        let raw = u64::from_le_bytes(chunk.try_into().unwrap());
        t.translate(raw);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::super::ir::Op;
    use super::*;

    fn static_ldc(dest: u8, binding: u8, byte_offset: u32) -> u64 {
        (0xEF94u64 << 48)
            | (u64::from(binding) << 36)
            | (u64::from(byte_offset & 0xFFFF) << 20)
            | (u64::from(PT) << 16)
            | (u64::from(RZ) << 8)
            | u64::from(dest)
    }

    fn ldc_with_mode(dest: u8, src: u8, binding: u8, byte_offset: i16, mode: u8) -> u64 {
        (0xEF94u64 << 48)
            | (u64::from(mode & 0x3) << 44)
            | (u64::from(binding & 0x1f) << 36)
            | (u64::from(byte_offset as u16) << 20)
            | (u64::from(PT) << 16)
            | (u64::from(src) << 8)
            | u64::from(dest)
    }

    #[allow(clippy::too_many_arguments)]
    fn tld4_raw(
        bindless: bool,
        dest: u8,
        coord: u8,
        meta_or_tex_id: u32,
        tex_type: u8,
        mask: u8,
        gather_component: u8,
        offset_type: u8,
        dc: bool,
        sparse_pred: u8,
    ) -> u64 {
        let common = (u64::from(dc) << 50)
            | (u64::from(sparse_pred & 0x7) << 51)
            | (u64::from(mask & 0xf) << 31)
            | (u64::from(tex_type & 0x7) << 28)
            | (u64::from(PT) << 16)
            | (u64::from(coord) << 8)
            | u64::from(dest);
        if bindless {
            (0b1101_1110_11u64 << 54)
                | (u64::from(gather_component & 0x3) << 38)
                | (u64::from(offset_type & 0x3) << 36)
                | (u64::from(meta_or_tex_id & 0xff) << 20)
                | common
        } else {
            (0b1100_10u64 << 58)
                | (u64::from(gather_component & 0x3) << 56)
                | (u64::from(offset_type & 0x3) << 54)
                | (u64::from(meta_or_tex_id & 0x1fff) << 36)
                | common
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tld4s_raw(
        dest_a: u8,
        dest_b: u8,
        coord_a: u8,
        coord_b: u8,
        tex_id: u32,
        gather_component: u8,
        fp16: bool,
        aoffi: bool,
        dc: bool,
    ) -> u64 {
        (0xdfu64 << 56)
            | (u64::from(fp16) << 55)
            | (u64::from(gather_component & 0x3) << 52)
            | (u64::from(aoffi) << 51)
            | (u64::from(dc) << 50)
            | (u64::from(tex_id & 0x1fff) << 36)
            | (u64::from(dest_b) << 28)
            | (u64::from(coord_b) << 20)
            | (u64::from(PT) << 16)
            | (u64::from(coord_a) << 8)
            | u64::from(dest_a)
    }

    fn tmml_raw(
        bindless: bool,
        dest: u8,
        coord: u8,
        meta_or_tex_id: u32,
        tex_type: u8,
        mask: u8,
    ) -> u64 {
        let common = (u64::from(mask & 0xf) << 31)
            | (u64::from(tex_type & 0x7) << 28)
            | (u64::from(PT) << 16)
            | (u64::from(coord) << 8)
            | u64::from(dest);
        if bindless {
            (0b1101_1111_0110_0u64 << 51) | (u64::from(meta_or_tex_id & 0xff) << 20) | common
        } else {
            (0b1101_1111_0101_1u64 << 51) | (u64::from(meta_or_tex_id & 0x1fff) << 36) | common
        }
    }

    fn translate_tld(raw: u64, binding: u8, byte_offset: u32) -> Translator {
        let mut t = Translator::new();
        assert!(t.translate(static_ldc(reg_b(raw), binding, byte_offset)));
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        t
    }

    fn direct_tex(blod: u8, tex_type: u8, mask: u8) -> u64 {
        0xC000_0000_0000_0000
            | (u64::from(blod) << 55)
            | (u64::from(PT) << 51)
            | (1 << 49)
            | (0x24 << 36)
            | (u64::from(mask) << 31)
            | (u64::from(tex_type) << 28)
            | (6 << 20)
            | (u64::from(PT) << 16)
            | (4 << 8)
            | 8
    }

    #[test]
    fn fmul_reg_emits_fmul() {
        let mut t = Translator::new();

        let raw = 0x5C68_1000_0007_0203u64;
        assert!(t.translate(raw));
        assert_eq!(t.program.instructions.len(), 1);
        match &t.program.instructions[0].op {
            Op::FMul { a, b, .. } => {
                assert!(matches!(a, Value::GprIn(2)));
                assert!(matches!(b, Value::GprIn(0)));
            }
            other => panic!("expected FMul, got {other:?}"),
        }
        assert_eq!(t.program.instructions[0].dest_reg, Some(3));
    }

    #[test]
    fn dusk_nvk_prmt_immediates_translate_generic_byte_permutations() {
        for (raw, dest, source) in [
            (0x36c0_7f84_4437_0403, 3, 4),
            (0x36c0_7f84_4437_0504, 4, 5),
            (0x36c0_7f84_4437_0605, 5, 6),
        ] {
            assert_eq!(decode_one(raw).unwrap().opcode, Opcode::PRMT_imm);
            let mut t = Translator::new();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::Bfe {
                    a: Value::GprIn(reg),
                    b: Value::ImmU32(0x818),
                    signed: false,
                } if reg == source
            )));
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::Mov(Value::Inst(_)),
                    dest_reg: Some(reg),
                    ..
                }) if *reg == dest
            ));
        }
    }

    #[test]
    fn prmt_sign_selectors_replicate_only_the_selected_sign_bit() {
        const BASE_RAW: u64 = 0x36c0_7f84_4437_0403;
        const SELECTOR_MASK: u64 = 0xffff << 20;
        let raw = (BASE_RAW & !SELECTOR_MASK) | (0x8888 << 20);
        assert_eq!(decode_one(raw).unwrap().opcode, Opcode::PRMT_imm);

        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(
            t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::Bfe {
                    a: Value::GprIn(4),
                    b: Value::ImmU32(0x107),
                    signed: true,
                }
            )),
            "{:#?}",
            t.program.instructions
        );
        assert!(!t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::IShr { signed: true, .. })));
    }

    #[test]
    fn dusk_nvk_imul_immediate_multiplies_vertex_index_by_stride() {
        const RAW: u64 = 0x3838_0000_0307_0000;
        assert_eq!(decode_one(RAW).unwrap().opcode, Opcode::IMUL_imm);

        let mut t = Translator::new();
        assert!(t.translate(RAW));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.as_slice(),
            [Inst {
                op: Op::IMul {
                    a: Value::GprIn(0),
                    b: Value::ImmU32(48),
                },
                dest_reg: Some(0),
                ..
            }]
        ));
    }

    #[test]
    fn dusk_nvk_imul32i_high_extracts_unsigned_product_high_word() {
        const RAW: u64 = 0x1f2a_aaaa_aab7_0101;
        assert_eq!(decode_one(RAW).unwrap().opcode, Opcode::IMUL32I);

        let mut t = Translator::new();
        assert!(t.translate(RAW));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.as_slice(),
            [Inst {
                op: Op::IMulHigh {
                    a: Value::GprIn(1),
                    b: Value::ImmU32(0xaaaa_aaab),
                    signed_a: false,
                    signed_b: false,
                },
                dest_reg: Some(1),
                ..
            }]
        ));
    }

    #[test]
    fn ffma_cr_emits_load_then_ffma() {
        let raw = 0x49A0_0100_0047_0102u64;
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.program.instructions.len(), 2);
        assert!(matches!(t.program.instructions[0].op, Op::LoadCbuf { .. }));
        match &t.program.instructions[1].op {
            Op::FFma { a, b, c, .. } => {
                assert!(matches!(a, Value::GprIn(1)));
                assert!(matches!(b, Value::Inst(_)));
                assert!(matches!(c, Value::GprIn(2)));
            }
            other => panic!("expected FFma, got {other:?}"),
        }
    }

    #[test]
    fn exit_marks_finished() {
        let mut t = Translator::new();
        let exit = 0xE300_0000_0007_000Fu64;
        assert!(t.translate(exit));
        assert!(t.finished);
        assert!(t.program.exit_reg_state.is_some());
        assert!(matches!(
            t.program.instructions.as_slice(),
            [Inst {
                op: Op::Exit,
                pred: None,
                ..
            }]
        ));
    }

    #[test]
    fn predicated_exit_is_cfg_only_and_falls_through() {
        for exit in [0xE300_0000_0000_000Fu64, 0xE300_0000_0008_000Fu64] {
            let mut t = Translator::new();
            assert!(t.translate(exit));
            assert!(!t.finished);
            assert!(t.program.exit_reg_state.is_none());
            assert!(t.program.instructions.is_empty());

            assert!(t.translate(0x5C68_1000_0007_0203));
            assert!(matches!(
                t.program.instructions.as_slice(),
                [Inst {
                    op: Op::FMul { .. },
                    ..
                }]
            ));
        }
    }

    #[test]
    fn fcsm_tr_exit_remains_fallthrough() {
        let mut t = Translator::new();
        assert!(t.translate(0xE300_0000_0007_001Cu64));
        assert!(!t.finished);
        assert!(t.program.exit_reg_state.is_none());
        assert!(t.program.instructions.is_empty());

        assert!(t.translate(0x5C68_1000_0007_0203));
        assert!(matches!(
            t.program.instructions.as_slice(),
            [Inst {
                op: Op::FMul { .. },
                ..
            }]
        ));
    }

    #[test]
    fn pps_predicated_exit_preserves_bindless_fallthrough_translation() {
        let mut t = Translator::new();
        for raw in [
            0xe300_0000_0000_000f,
            0x4c98_0788_05a7_0004,
            0x4c47_0208_15c7_0404,
            0x5c98_0780_0047_000c,
            0x5c98_0780_0047_0012,
            0x5c98_0780_0047_0010,
            0xdeb8_0060_a047_0a04,
            0xdeb8_0060_a107_0e0e,
            0xdeb8_0060_a127_0808,
            0xdeb8_0060_a0c7_0607,
        ] {
            assert!(t.translate(raw), "raw={raw:#018x}");
        }

        assert!(!t.finished);
        assert!(t.program.exit_reg_state.is_none());
        assert!(t
            .program
            .instructions
            .iter()
            .all(|inst| !matches!(inst.op, Op::Exit)));
        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::SampleTex { tex_id, .. } => Some(tex_id),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![bindless_texture_id_pair(2, 0x5a, Some(0x15c)); 4]
        );
        assert_eq!(t.unimplemented_count, 0);
    }

    #[test]
    fn ald_emits_loadattr() {
        let raw = 0xEFD8_FF80_0807_FF00u64;
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(
            t.program.instructions.len(),
            2,
            "vec2 ALD lifts into 2 LoadAttr ops"
        );
        match &t.program.instructions[0].op {
            Op::LoadAttr { slot } => assert_eq!(*slot, 0x80),
            other => panic!("expected LoadAttr at 0x80, got {other:?}"),
        }
        match &t.program.instructions[1].op {
            Op::LoadAttr { slot } => assert_eq!(*slot, 0x84),
            other => panic!("expected LoadAttr at 0x84, got {other:?}"),
        }
    }

    #[test]
    fn mufu_emits_multifunc_with_func() {
        let raw = 0x5080_0000_0047_0004u64;
        let mut t = Translator::new();
        assert!(t.translate(raw));
        match &t.program.instructions[0].op {
            Op::MultiFunc {
                func: MufuFunc::Rcp,
                mods,
                ..
            } => assert_eq!(*mods, FMods::default()),
            other => panic!("expected MultiFunc Rcp, got {other:?}"),
        }
    }

    #[test]
    fn mufu_carries_abs_neg_and_sat_modifiers_into_ir() {
        let raw = 0x5084_0000_0027_230bu64 | (1u64 << 46) | (1u64 << 48);
        let mut t = Translator::new();
        assert!(t.translate(raw));
        match &t.program.instructions[0].op {
            Op::MultiFunc {
                src: Value::GprIn(35),
                func: MufuFunc::Ex2,
                mods,
            } => assert_eq!(
                *mods,
                FMods {
                    abs_a: true,
                    neg_a: true,
                    sat: true,
                    ..FMods::default()
                }
            ),
            other => panic!("expected modified MultiFunc Ex2, got {other:?}"),
        }
    }

    #[test]
    fn ldc_static_emits_load_cbuf_then_mov() {
        let raw: u64 = (0xEF94u64 << 48)
            | (0x0000u64 << 44)
            | (0x0000u64 << 36)
            | (0x0010u64 << 20)
            | (PT as u64) << 16
            | (0x00FFu64 << 8)
            | 0x0003u64;
        let mut t = Translator::new();
        assert!(
            t.translate(raw),
            "LDC (static B32) should translate successfully"
        );

        assert_eq!(
            t.program.instructions.len(),
            2,
            "static LDC lifts to LoadCbuf + Mov"
        );
        match &t.program.instructions[0].op {
            Op::LoadCbuf {
                binding,
                byte_offset,
            } => {
                assert_eq!(*binding, 0, "cbuf binding should be 0");
                assert_eq!(*byte_offset, 0x10, "byte_offset should be 0x10 (16)");
            }
            other => panic!("expected LoadCbuf, got {other:?}"),
        }
        match &t.program.instructions[1].op {
            Op::Mov(Value::Inst(_)) => {}
            other => panic!("expected Mov(Inst), got {other:?}"),
        }
        assert_eq!(
            t.program.instructions[1].dest_reg,
            Some(3),
            "dest reg should be R3"
        );
    }

    #[test]
    fn ldc_indexed_emits_load_cbuf_indexed_then_mov() {
        let raw: u64 = (0xEF94u64 << 48)
            | (0x0000u64 << 44)
            | (0x0000u64 << 36)
            | (0x0010u64 << 20)
            | (PT as u64) << 16
            | (0x0001u64 << 8)
            | 0x0003u64;
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.program.instructions.len(), 2);
        match &t.program.instructions[0].op {
            Op::LoadCbufIndexed {
                binding,
                byte_offset,
                index,
                address_mode,
            } => {
                assert_eq!(*binding, 0);
                assert_eq!(*byte_offset, 0x10);
                assert!(matches!(index, Value::GprIn(1)));
                assert_eq!(*address_mode, CbufAddressMode::Default);
            }
            other => panic!("expected LoadCbufIndexed, got {other:?}"),
        }
        assert!(matches!(
            t.program.instructions[1].op,
            Op::Mov(Value::Inst(_))
        ));
        assert_eq!(t.program.instructions[1].dest_reg, Some(3));
    }

    #[test]
    fn ldc_b64_odd_destination_fails_closed() {
        let raw = static_ldc(3, 6, 0x2aa0) | (1u64 << 48);
        assert_eq!(ldc_size(raw), 5);
        let mut translator = Translator::new_fragment();
        assert!(!translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 1);
        assert!(matches!(
            translator.program.instructions.as_slice(),
            [Inst {
                op: Op::Unimplemented {
                    opcode: Opcode::LDC,
                    raw: emitted_raw,
                },
                ..
            }] if *emitted_raw == raw
        ));
    }

    #[test]
    fn ldc_b64_even_destination_emits_two_independent_word_reads() {
        let raw = static_ldc(2, 6, 0x2aa0) | (1u64 << 48);
        let mut translator = Translator::new_fragment();
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);
        let reads = translator
            .program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction.op {
                Op::LoadCbuf {
                    binding,
                    byte_offset,
                } => Some((binding, byte_offset)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(reads, vec![(6, 0x2aa0), (6, 0x2aa4)]);
    }

    #[test]
    fn pps_captured_b64_ldc_tables_preserve_full_and_signed_offsets() {
        for (raw, expected_src, expected_offsets) in [
            (0xef95_0060_0007_2500u64, 37, [0, 4]),
            (0xef95_0061_5507_2500, 37, [0x1550, 0x1554]),
            (0xef95_0062_aa07_2500, 37, [0x2aa0, 0x2aa4]),
            (0xef95_0069_5307_2724, 39, [0xffff_9530, 0xffff_9534]),
        ] {
            let mut translator = Translator::new_fragment();
            assert!(translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 0);
            let reads = translator
                .program
                .instructions
                .iter()
                .filter_map(|instruction| match &instruction.op {
                    Op::LoadCbufIndexed {
                        binding,
                        byte_offset,
                        index,
                        address_mode,
                    } => Some((*binding, *byte_offset, *index, *address_mode)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(reads.len(), 2, "raw={raw:#018x}");
            for (read, expected_offset) in reads.iter().zip(expected_offsets) {
                assert_eq!(read.0, 6);
                assert_eq!(read.1, expected_offset);
                assert_eq!(read.2, Value::GprIn(expected_src));
                assert_eq!(read.3, CbufAddressMode::Default);
            }
        }
    }

    #[test]
    fn ldc_is_preserves_segmented_addressing_in_every_stage() {
        let raw = ldc_with_mode(3, 1, 5, -16, 2);
        for (stage, mut translator) in [
            ("vertex", Translator::new()),
            ("fragment", Translator::new_fragment()),
            ("compute", Translator::new_compute()),
        ] {
            assert!(translator.translate(raw), "{stage} LDC.IS translation");
            assert_eq!(translator.unimplemented_count, 0, "{stage}");
            assert_eq!(translator.program.instructions.len(), 2, "{stage}");
            match &translator.program.instructions[0].op {
                Op::LoadCbufIndexed {
                    binding,
                    byte_offset,
                    index,
                    address_mode,
                } => {
                    assert_eq!(*binding, 5, "{stage}");
                    assert_eq!(*byte_offset, 0xffff_fff0, "{stage}");
                    assert!(matches!(index, Value::GprIn(1)), "{stage}: {index:?}");
                    assert_eq!(*address_mode, CbufAddressMode::Segmented, "{stage}");
                }
                other => panic!("{stage}: expected segmented LoadCbufIndexed, got {other:?}"),
            }
        }
    }

    #[test]
    fn ldc_is_with_rz_keeps_segmented_binding_semantics() {
        let raw = ldc_with_mode(3, RZ, 5, -16, 2);
        let mut translator = Translator::new_compute();
        assert!(translator.translate(raw));
        match &translator.program.instructions[0].op {
            Op::LoadCbufIndexed {
                binding,
                byte_offset,
                index,
                address_mode,
            } => {
                assert_eq!(*binding, 5);
                assert_eq!(*byte_offset, 0xffff_fff0);
                assert!(matches!(index, Value::Zero));
                assert_eq!(*address_mode, CbufAddressMode::Segmented);
            }
            other => panic!("expected segmented LoadCbufIndexed, got {other:?}"),
        }
    }

    #[test]
    fn ldc_il_and_isl_fail_closed_in_every_stage() {
        for mode in [1, 3] {
            let raw = ldc_with_mode(3, 1, 5, 0x10, mode);
            for (stage, mut translator) in [
                ("vertex", Translator::new()),
                ("fragment", Translator::new_fragment()),
                ("compute", Translator::new_compute()),
            ] {
                assert!(
                    !translator.translate(raw),
                    "{stage} LDC mode {mode} must fail"
                );
                assert_eq!(translator.unimplemented_count, 1, "{stage} mode {mode}");
                assert_eq!(
                    translator.program.instructions.len(),
                    1,
                    "{stage} mode {mode}"
                );
                assert!(matches!(
                    &translator.program.instructions[0].op,
                    Op::Unimplemented {
                        opcode: Opcode::LDC,
                        raw: rejected,
                    } if *rejected == raw
                ));
            }
        }
    }

    #[test]
    fn fine_x_derivative_idiom_emits_dpdx() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0xEF17_700C_F017_0103));
        assert!(t.translate(0x50F8_0009_9017_0303));
        assert_eq!(t.unimplemented_count, 0);
        match &t.program.instructions[1].op {
            Op::DpdxFine { src } => assert!(matches!(src, Value::GprIn(1))),
            other => panic!("expected DPdxFine, got {other:?}"),
        }
    }

    #[test]
    fn pps_fine_y_derivative_idioms_emit_dpdy() {
        for (shfl, fswzadd, src, dest) in [
            (0xef17_700c_f027_1106, 0x50f8_000a_5117_0606, 17, 6),
            (0xef17_700c_f027_1301, 0x50f8_000a_5137_0101, 19, 1),
            (0xef17_700c_f027_0508, 0x50f8_000a_5057_0808, 5, 8),
        ] {
            let mut t = Translator::new_fragment();
            assert!(t.translate(shfl), "SHFL raw={shfl:#018x}");
            assert!(t.translate(fswzadd), "FSWZADD raw={fswzadd:#018x}");
            assert_eq!(t.unimplemented_count, 0);
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::DpdyFine {
                        src: Value::GprIn(actual_src)
                    },
                    dest_reg: Some(actual_dest),
                    ..
                }) if *actual_src == src && *actual_dest == dest
            ));
        }
    }

    #[test]
    fn derivative_idiom_lowers_generically_outside_fragment_stage() {
        let mut t = Translator::new();

        assert!(t.translate(0xef17_700c_f027_1106));
        assert!(t.translate(0x50f8_000a_5117_0606));

        assert_eq!(t.unimplemented_count, 0);
        assert!(!t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::DpdxFine { .. } | Op::DpdyFine { .. })));
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::Shfl { .. })));
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::FSwzAdd { .. })));
    }

    #[test]
    fn y_direction_times_dpdy_folds_in_either_operand_order() {
        let original = 0x5c68_1000_0067_0306u64;
        let reversed =
            (original & !((0xffu64 << 8) | (0xffu64 << 20))) | (6u64 << 8) | (3u64 << 20);

        for fmul in [original, reversed] {
            let mut t = Translator::new_fragment();
            assert!(t.translate(0xf0c8_0000_0127_0003));
            assert!(t.translate(0xef17_700c_f027_1106));
            assert!(t.translate(0x50f8_000a_5117_0606));
            let derivative = t.program.instructions.last().unwrap().result.unwrap();

            assert!(t.translate(fmul), "FMUL raw={fmul:#018x}");

            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::Mov(Value::Inst(actual)),
                    dest_reg: Some(6),
                    ..
                }) if *actual == derivative
            ));
        }
    }

    #[test]
    fn y_direction_derivative_fold_rejects_dpdx_and_modified_fmul() {
        let mut dpdx = Translator::new_fragment();
        assert!(dpdx.translate(0xf0c8_0000_0127_0003));
        assert!(dpdx.translate(0xef17_700c_f017_1106));
        assert!(dpdx.translate(0x50f8_0009_9117_0606));
        assert!(dpdx.translate(0x5c68_1000_0067_0306));
        assert!(matches!(
            dpdx.program.instructions.last(),
            Some(Inst {
                op: Op::FMul { .. },
                ..
            })
        ));

        let mut modified = Translator::new_fragment();
        assert!(modified.translate(0xf0c8_0000_0127_0003));
        assert!(modified.translate(0xef17_700c_f027_1106));
        assert!(modified.translate(0x50f8_000a_5117_0606));
        assert!(modified.translate(0x5c69_1000_0067_0306));
        assert!(matches!(
            modified.program.instructions.last(),
            Some(Inst {
                op: Op::FMul {
                    mods: FMods { neg_b: true, .. },
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn pps_s2r_y_direction_emits_dynamic_float_sign() {
        let mut t = Translator::new();

        assert!(t.translate(0xf0c8_0000_0127_0003));

        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::YDirection,
                dest_reg: Some(3),
                ..
            })
        ));
    }

    #[test]
    fn pps_compute_s2r_uses_local_and_workgroup_xyz_builtins() {
        for (raw, dest, local, component) in [
            (0xf0c8_0000_0217_0001, 1, true, 0),
            (0xf0c8_0000_0227_0003, 3, true, 1),
            (0xf0c8_0000_0237_0005, 5, true, 2),
            (0xf0c8_0000_0257_0000, 0, false, 0),
            (0xf0c8_0000_0267_0001, 1, false, 1),
            (0xf0c8_0000_0277_0004, 4, false, 2),
        ] {
            let mut t = Translator::new_compute();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0);
            assert!(
                matches!(
                    t.program.instructions.last(),
                    Some(Inst {
                        op: Op::LocalInvocationId {
                            component: actual
                        },
                        dest_reg: Some(actual_dest),
                        ..
                    }) if local && *actual == component && *actual_dest == dest
                ) || matches!(
                    t.program.instructions.last(),
                    Some(Inst {
                        op: Op::WorkgroupId {
                            component: actual
                        },
                        dest_reg: Some(actual_dest),
                        ..
                    }) if !local && *actual == component && *actual_dest == dest
                )
            );
        }

        let mut graphics = Translator::new();
        assert!(graphics.translate(0xf0c8_0000_0217_0001));
        assert!(matches!(
            graphics.program.instructions.last(),
            Some(Inst {
                op: Op::Mov(Value::ImmU32(0)),
                dest_reg: Some(1),
                ..
            })
        ));
    }

    #[test]
    fn compute_s2r_packed_tid_preserves_maxwell_bit_layout() {
        let mut t = Translator::new_compute();
        assert!(t.translate(0xf0c8_0000_0207_0006));
        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter(|inst| matches!(inst.op, Op::LocalInvocationId { .. }))
                .count(),
            3
        );
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::Bfi {
                control: Value::ImmU32(0x810),
                ..
            }
        )));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::Bfi {
                control: Value::ImmU32(0x61a),
                ..
            }
        )));
        assert_eq!(t.program.instructions.last().unwrap().dest_reg, Some(6));
    }

    #[test]
    fn compute_s2r_lane_id_and_maxwell_lane_masks_are_explicit_ir() {
        let mut lane = Translator::new_compute();
        assert!(lane.translate(0xf0c8_0000_0007_0004));
        assert_eq!(lane.unimplemented_count, 0);
        assert!(matches!(
            lane.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupLaneId,
                dest_reg: Some(4),
                ..
            })
        ));

        for (sr, expected) in [
            (0x38u64, SubgroupMask::Eq),
            (0x39, SubgroupMask::Lt),
            (0x3a, SubgroupMask::Le),
            (0x3b, SubgroupMask::Gt),
            (0x3c, SubgroupMask::Ge),
        ] {
            let raw = 0xf0c8_0000_0007_0004 | (sr << 20);
            let mut t = Translator::new_compute();
            assert!(t.translate(raw), "S2R sr={sr:#x} raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0);
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::SubgroupMask { kind },
                    dest_reg: Some(4),
                    ..
                }) if *kind == expected
            ));
        }

        let mut captured = Translator::new_compute();
        assert!(captured.translate(0xf0c8_0000_0397_0004));
        assert!(matches!(
            captured.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupMask {
                    kind: SubgroupMask::Lt
                },
                dest_reg: Some(4),
                ..
            })
        ));
    }

    #[test]
    fn unsupported_compute_s2r_fails_closed_instead_of_returning_zero() {
        let raw = 0xf0c8_0000_07f7_0001;
        let mut t = Translator::new_compute();
        assert!(!t.translate(raw));
        assert_eq!(t.unimplemented_count, 1);
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::Unimplemented {
                    opcode: Opcode::S2R,
                    raw: actual
                },
                ..
            }) if *actual == raw
        ));
    }

    #[test]
    fn pps_vote_lowers_ballot_with_all_any_and_equal() {
        let captured = 0x50d8_e380_0007_0001;
        assert_eq!(decode_one(captured).unwrap().opcode, Opcode::VOTE);
        let mut all = Translator::new_compute();
        assert!(all.translate(captured));
        assert_eq!(all.unimplemented_count, 0);
        assert!(matches!(
            all.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupVote {
                    source_pred: Predicate {
                        idx: PT,
                        negate: false
                    },
                    mode: VoteMode::All,
                    pred_dest: PT,
                    old: Value::GprIn(1),
                },
                dest_reg: Some(1),
                pred: None,
                ..
            })
        ));

        let any_raw = 0x50d9_e380_0007_0002;
        assert_eq!(decode_one(any_raw).unwrap().opcode, Opcode::VOTE);
        let mut any = Translator::new_compute();
        assert!(any.translate(any_raw));
        assert_eq!(any.unimplemented_count, 0);
        assert!(matches!(
            any.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupVote {
                    mode: VoteMode::Any,
                    ..
                },
                dest_reg: Some(2),
                ..
            })
        ));

        let equal_raw = 0x50da_e380_0007_0003;
        assert_eq!(decode_one(equal_raw).unwrap().opcode, Opcode::VOTE);
        let mut equal = Translator::new_compute();
        assert!(equal.translate(equal_raw));
        assert_eq!(equal.unimplemented_count, 0);
        assert!(matches!(
            equal.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupVote {
                    mode: VoteMode::Equal,
                    ..
                },
                dest_reg: Some(3),
                ..
            })
        ));
    }

    #[test]
    fn unsupported_vote_mode_fails_closed() {
        let unsupported_mode = 0x50db_e380_0007_0001;
        assert_eq!(decode_one(unsupported_mode).unwrap().opcode, Opcode::VOTE);
        let mut compute = Translator::new_compute();
        assert!(!compute.translate(unsupported_mode));
        assert_eq!(compute.unimplemented_count, 1);
        assert!(matches!(
            compute.program.instructions.last(),
            Some(Inst {
                op: Op::Unimplemented {
                    opcode: Opcode::VOTE,
                    raw
                },
                ..
            }) if *raw == unsupported_mode
        ));
    }

    #[test]
    fn dredge_vote_vtg_is_a_graphics_noop_but_compute_fails_closed() {
        let dredge_vtg = 0x50e2_4321_1117_0000;
        assert_eq!(decode_one(dredge_vtg).unwrap().opcode, Opcode::VOTE_vtg);

        for mut graphics in [Translator::new(), Translator::new_fragment()] {
            let registers_before = graphics.reg_state.clone();
            let predicates_before = graphics.pred_state.clone();
            assert!(graphics.translate(dredge_vtg));
            assert_eq!(graphics.unimplemented_count, 0);
            assert!(graphics.program.instructions.is_empty());
            assert_eq!(graphics.reg_state, registers_before);
            assert_eq!(graphics.pred_state, predicates_before);
        }

        let mut compute = Translator::new_compute();
        assert!(!compute.translate(dredge_vtg));
        assert_eq!(compute.unimplemented_count, 1);
        assert!(matches!(
            compute.program.instructions.last(),
            Some(Inst {
                op: Op::Unimplemented {
                    opcode: Opcode::VOTE_vtg,
                    raw
                },
                ..
            }) if *raw == dredge_vtg
        ));
    }

    #[test]
    fn graphics_vote_lowers() {
        let mut graphics = Translator::new();
        assert!(graphics.translate(0x50d8_e380_0007_0001));
        assert_eq!(graphics.unimplemented_count, 0);
        assert!(matches!(
            graphics.program.instructions.last(),
            Some(Inst {
                op: Op::SubgroupVote {
                    mode: VoteMode::All,
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn texs_2d_encoding_uses_implicit_lod() {
        let mut t = Translator::new();
        assert!(t.translate(0xD822_00A0_5087_0500));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().all(|inst| matches!(
            inst.op,
            Op::SampleTex {
                implicit_lod: true,
                ..
            }
        )));
    }

    #[test]
    fn captured_direct_2d_txd_preserves_interleaved_gradients() {
        let raw = 0xde38_0081_a047_0e0cu64;
        assert_eq!(decode_one(raw).unwrap().opcode, Opcode::TXD);
        let mut translator = Translator::new_fragment();
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);
        assert!(matches!(
            translator.program.instructions.as_slice(),
            [
                Inst {
                    op: Op::TextureGradients {
                        sample_site: 0,
                        dpdx: (Value::GprIn(4), Value::GprIn(6)),
                        dpdy: (Value::GprIn(5), Value::GprIn(7)),
                    },
                    result: None,
                    dest_reg: None,
                    pred: None,
                },
                Inst {
                    op: Op::SampleTex {
                        sample_site: Some(0),
                        tex_id: 8,
                        u: Value::GprIn(14),
                        v: Value::GprIn(15),
                        array: None,
                        volume: None,
                        cube: None,
                        implicit_lod: false,
                        lod_bias: None,
                        explicit_lod: None,
                        texel_offset: None,
                        dref: None,
                        component: 0,
                    },
                    dest_reg: Some(12),
                    ..
                },
                Inst {
                    op: Op::SampleTex {
                        sample_site: Some(0),
                        tex_id: 8,
                        component: 1,
                        ..
                    },
                    dest_reg: Some(13),
                    ..
                },
            ]
        ));
    }

    #[test]
    fn all_captured_direct_2d_txd_instructions_translate() {
        for raw in [
            0xde38_0081_a047_0e0c,
            0xde38_0081_a047_0a08,
            0xde38_0081_a040_0e08,
            0xde38_0081_a041_0a00,
            0xde38_0081_a041_0e08,
            0xde38_0081_a040_0a00,
            0xde38_0081_a041_0e00,
            0xde38_0081_a040_0a08,
        ] {
            assert_eq!(decode_one(raw).unwrap().opcode, Opcode::TXD);
            let mut translator = Translator::new_fragment();
            assert!(translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 0, "raw={raw:#018x}");
        }
    }

    #[test]
    fn unhandled_txd_modes_fail_closed() {
        let captured = 0xde38_0081_a047_0e0cu64;
        let unsupported = [
            captured | (1u64 << 35),
            captured | (1u64 << 50),
            captured & !(7u64 << 51),
            (captured & !(7u64 << 28)) | (3u64 << 28),
            captured & !(0xfu64 << 31),
        ];
        for raw in unsupported {
            assert_eq!(decode_one(raw).unwrap().opcode, Opcode::TXD);
            let mut translator = Translator::new_fragment();
            assert!(!translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                translator.program.instructions.last(),
                Some(Inst {
                    op: Op::Unimplemented {
                        opcode: Opcode::TXD,
                        raw: rejected,
                    },
                    ..
                }) if *rejected == raw
            ));
        }

        let mut compute = Translator::new_compute();
        assert!(!compute.translate(captured));
        assert_eq!(compute.unimplemented_count, 1);

        let bindless = captured | (1u64 << 54);
        assert_eq!(decode_one(bindless).unwrap().opcode, Opcode::TXD_b);
        let mut translator = Translator::new_fragment();
        assert!(!translator.translate(bindless));
        assert_eq!(translator.unimplemented_count, 1);
    }

    #[test]
    fn captured_texs_sparse_and_rgba_lanes_share_one_unique_sample_site() {
        let title_sparse = 0xd836_022f_f087_0908u64;
        let mut sparse = Translator::new_fragment();
        assert!(sparse.translate(title_sparse));
        let sparse_lanes = sparse
            .program
            .instructions
            .iter()
            .filter_map(|inst| match inst.op {
                Op::SampleTex {
                    sample_site: Some(site),
                    component,
                    ..
                } => Some((site, component, inst.dest_reg)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sparse_lanes, vec![(0, 0, Some(8)), (0, 3, Some(9))]);

        let material_rgba = 0xd9b2_01a0_8087_0a0au64;
        let mut rgba = Translator::new_fragment();
        assert!(rgba.translate(material_rgba));
        let rgba_lanes = rgba
            .program
            .instructions
            .iter()
            .filter_map(|inst| match inst.op {
                Op::SampleTex {
                    sample_site: Some(site),
                    component,
                    ..
                } => Some((site, component, inst.dest_reg)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            rgba_lanes,
            vec![
                (0, 0, Some(10)),
                (0, 1, Some(11)),
                (0, 2, Some(8)),
                (0, 3, Some(9)),
            ]
        );
    }

    #[test]
    fn texture_sample_sites_are_unique_within_a_merged_cfg_block() {
        let first = 0xd822_00a0_5087_0500u64;
        let second = (first & !0xff) | 8;
        let exit = 0xe300_0000_0007_000fu64;
        let nop = 0x50b0_0000_0007_0f00u64;
        let bundle = |sample| {
            [0, sample, exit, nop]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let merged = crate::merge_dual_vertex_sass(&bundle(first), &bundle(second))
            .expect("dual-vertex SASS merge");
        let cfg = crate::build_cfg(&merged);
        assert_eq!(cfg.blocks.len(), 1);
        let sites = cfg.blocks[0]
            .program
            .instructions
            .iter()
            .filter_map(|inst| match inst.op {
                Op::SampleTex {
                    sample_site: Some(site),
                    ..
                } => Some(site),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sites.len(), 6);
        assert!(sites[..3].iter().all(|site| *site == sites[0]));
        assert!(sites[3..].iter().all(|site| *site == sites[3]));
        assert_ne!(sites[0], sites[3]);
    }

    #[test]
    fn depth_compare_alpha_is_one_instead_of_repeating_comparison() {
        let raw = (0xc03e_00c0_b077_0406u64 & !(0xf << 31)) | (0b1001 << 31);
        let mut translator = Translator::new_fragment();
        assert!(translator.translate(raw));
        let results = translator
            .program
            .instructions
            .iter()
            .map(|inst| (&inst.op, inst.dest_reg))
            .collect::<Vec<_>>();
        assert!(matches!(
            results.as_slice(),
            [
                (
                    Op::SampleTex {
                        sample_site: Some(_),
                        dref: Some(Value::GprIn(7)),
                        component: 0,
                        ..
                    },
                    Some(6)
                ),
                (Op::Mov(Value::ImmF32(alpha)), Some(7)),
            ] if *alpha == 1.0
        ));
    }

    #[test]
    fn compute_texs_2d_uses_bound_handle_and_metadata() {
        let raw = 0xD822_00A0_5087_0500;
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().all(|inst| matches!(
            inst.op,
            Op::SampleTexHandle {
                handle: TextureHandleOrigin::Bound { cbuf_word_offset },
                dimension: ImageDimension::D2,
                u: Value::GprIn(u),
                v: Some(Value::GprIn(v)),
                w: None,
                implicit_lod: true,
                explicit_lod: None,
                ..
            } if cbuf_word_offset == texs_tex_id(raw) && u == reg_a(raw) && v == reg_b(raw)
        )));
    }

    #[test]
    fn compute_texs_2d_lod_level_uses_separate_v_and_lod() {
        let raw = 0xD800_0000_0000_0000
            | (3 << 53)
            | (0x24 << 36)
            | (u64::from(RZ) << 28)
            | (6 << 20)
            | (u64::from(PT) << 16)
            | (4 << 8)
            | 8;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEXS)
        );
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTexHandle {
                handle: TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x24
                },
                dimension: ImageDimension::D2,
                u: Value::GprIn(4),
                v: Some(Value::GprIn(5)),
                w: None,
                implicit_lod: false,
                explicit_lod: Some(Value::GprIn(6)),
                ..
            }
        )));
    }

    #[test]
    fn compute_tlds_lod_zero_uses_texel_fetch_handle() {
        let raw = 0xDA00_0000_0000_0000
            | (2 << 53)
            | (0x24 << 36)
            | (u64::from(RZ) << 28)
            | (6 << 20)
            | (u64::from(PT) << 16)
            | (4 << 8)
            | 8;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TLDS)
        );
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::TexelFetchHandle {
                handle: TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x24
                },
                dimension: ImageDimension::D2,
                x: Value::GprIn(4),
                y: Some(Value::GprIn(6)),
                z: None,
                ..
            }
        )));
    }

    #[test]
    fn texs_cube_encoding_uses_xyz_and_implicit_lod() {
        let raw = 0xD800_0000_0000_0000
            | (12 << 53)
            | (0x24 << 36)
            | (u64::from(RZ) << 28)
            | (6 << 20)
            | (u64::from(PT) << 16)
            | (4 << 8)
            | 8;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEXS)
        );
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                tex_id: 0x24,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                volume: None,
                cube: Some(Value::GprIn(6)),
                implicit_lod: true,
                explicit_lod: None,
                ..
            }
        )));
    }

    #[test]
    fn texs_cube_ll_encoding_uses_register_after_z() {
        let raw = 0xD800_0000_0000_0000
            | (13 << 53)
            | (0x24 << 36)
            | (u64::from(RZ) << 28)
            | (6 << 20)
            | (u64::from(PT) << 16)
            | (4 << 8)
            | 8;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEXS)
        );
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                tex_id: 0x24,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                volume: None,
                cube: Some(Value::GprIn(6)),
                implicit_lod: false,
                explicit_lod: Some(Value::GprIn(7)),
                ..
            }
        )));
    }

    #[test]
    fn graphics_texs_encodings_match_hardware_operand_table() {
        struct Expect {
            enc: u64,
            u: Value,
            v: Value,
            array: Option<Value>,
            implicit_lod: bool,
            explicit_lod: Option<Value>,
            dref: Option<Value>,
        }
        let cases = [
            Expect {
                enc: 3,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                implicit_lod: false,
                explicit_lod: Some(Value::GprIn(6)),
                dref: None,
            },
            Expect {
                enc: 4,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                implicit_lod: true,
                explicit_lod: None,
                dref: Some(Value::GprIn(6)),
            },
            Expect {
                enc: 5,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                implicit_lod: false,
                explicit_lod: Some(Value::GprIn(6)),
                dref: Some(Value::GprIn(7)),
            },
            Expect {
                enc: 6,
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                array: None,
                implicit_lod: false,
                explicit_lod: None,
                dref: Some(Value::GprIn(6)),
            },
            Expect {
                enc: 9,
                u: Value::GprIn(5),
                v: Value::GprIn(6),
                array: Some(Value::GprIn(4)),
                implicit_lod: false,
                explicit_lod: None,
                dref: Some(Value::GprIn(7)),
            },
        ];
        for case in cases {
            let raw = 0xD800_0000_0000_0000
                | (case.enc << 53)
                | (0x24 << 36)
                | (u64::from(RZ) << 28)
                | (6 << 20)
                | (u64::from(PT) << 16)
                | (4 << 8)
                | 8;
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::TEXS)
            );
            let mut t = Translator::new();
            assert!(t.translate(raw), "enc {}", case.enc);
            assert_eq!(t.unimplemented_count, 0, "enc {}", case.enc);
            assert!(
                t.program.instructions.iter().any(|inst| matches!(
                    inst.op,
                    Op::SampleTex {
                        tex_id: 0x24,
                        u,
                        v,
                        array,
                        volume: None,
                        cube: None,
                        implicit_lod,
                        explicit_lod,
                        dref,
                        ..
                    } if u == case.u
                        && v == case.v
                        && array == case.array
                        && implicit_lod == case.implicit_lod
                        && explicit_lod == case.explicit_lod
                        && dref == case.dref
                )),
                "enc {} lowered as {:?}",
                case.enc,
                t.program.instructions
            );
        }
    }

    #[test]
    fn pps_shf_subset_lowers_all_observed_encodings() {
        for raw in [
            0x36f8_0300_0107_0607,
            0x36f8_0300_0107_0609,
            0x36f8_0400_0107_0810,
            0x36f8_0680_0047_ff0d,
            0x36f8_0680_0067_ff0c,
            0x36f8_0700_0047_ff0e,
            0x36f8_0780_0047_ff0f,
            0x36f8_0780_0107_0f06,
            0x36f8_0880_0067_ff10,
            0x36f8_0900_0047_ff12,
            0x36f8_0980_0107_1319,
            0x36f8_0e00_0107_1c06,
            0x36f8_0e00_0107_1c07,
            0x36f8_0e00_0107_1c08,
            0x5bfc_0380_0027_ff07,
            0x38f8_7f80_0017_0909,
        ] {
            let mut t = Translator::new();
            assert!(t.translate(raw), "SHF raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0, "SHF raw={raw:#018x}");
            assert!(!t
                .program
                .instructions
                .iter()
                .any(|inst| matches!(inst.op, Op::Unimplemented { .. })));
        }
    }

    #[test]
    fn pps_shf_subset_emits_safe_existing_integer_ops() {
        let mut shift = Translator::new();
        assert!(shift.translate(0x36f8_0900_0047_ff12));
        assert_eq!(shift.program.instructions.len(), 1);
        assert!(matches!(
            shift.program.instructions[0].op,
            Op::IShl {
                a: Value::GprIn(18),
                b: Value::ImmU32(4),
            }
        ));
        assert_eq!(shift.program.instructions[0].dest_reg, Some(18));

        let mut rotate = Translator::new();
        assert!(rotate.translate(0x36f8_0300_0107_0607));
        assert_eq!(rotate.program.instructions.len(), 3);
        assert!(matches!(
            rotate.program.instructions[0].op,
            Op::IShl {
                a: Value::GprIn(6),
                b: Value::ImmU32(16),
            }
        ));
        assert!(matches!(
            rotate.program.instructions[1].op,
            Op::IShr {
                a: Value::GprIn(6),
                b: Value::ImmU32(16),
                signed: false,
            }
        ));
        assert!(matches!(
            rotate.program.instructions[2].op,
            Op::ILop {
                a: Value::Inst(_),
                b: Value::Inst(_),
                op: LogicOp::Or,
                not_a: false,
                not_b: false,
            }
        ));
        assert_eq!(rotate.program.instructions[2].dest_reg, Some(7));

        let mut register = Translator::new();
        assert!(register.translate(0x5bfc_0380_0027_ff07));
        assert_eq!(register.program.instructions.len(), 2);
        let mask_id = register.program.instructions[0].result.unwrap();
        assert!(matches!(
            register.program.instructions[0].op,
            Op::ILop {
                a: Value::GprIn(2),
                b: Value::ImmU32(31),
                op: LogicOp::And,
                not_a: false,
                not_b: false,
            }
        ));
        assert!(matches!(
            register.program.instructions[1].op,
            Op::IShl {
                a: Value::GprIn(7),
                b: Value::Inst(id),
            } if id == mask_id
        ));
        assert_eq!(register.program.instructions[1].dest_reg, Some(7));

        let mut right = Translator::new();
        assert!(right.translate(0x38f8_7f80_0017_0909));
        assert_eq!(right.program.instructions.len(), 1);
        assert!(matches!(
            right.program.instructions[0].op,
            Op::IShr {
                a: Value::GprIn(9),
                b: Value::ImmU32(1),
                signed: false,
            }
        ));
        assert_eq!(right.program.instructions[0].dest_reg, Some(9));
    }

    #[test]
    fn unsupported_shf_modes_fail_closed() {
        let left_imm = 0x36f8_0300_0107_0607u64;
        let left_reg = 0x5bfc_0380_0027_ff07u64;
        let right_imm = 0x38f8_7f80_0017_0909u64;
        for raw in [
            left_imm | (1 << 47),
            left_imm | (1 << 48),
            left_imm | (2 << 37),
            left_imm | (1 << 50),
            left_reg & !(1 << 50),
            (left_reg & !(0xff << 8)) | (1 << 8),
            right_imm | (1 << 50),
            (right_imm & !(0xff << 39)) | (1 << 39),
        ] {
            let mut t = Translator::new();
            assert!(!t.translate(raw), "SHF raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "SHF raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented { raw: failed, .. }) if *failed == raw
            ));
        }
    }

    #[test]
    fn pps_tld_b_first_fragment_fetch_preserves_handle_origin() {
        let raw = 0xdd38_0000_8047_1515;
        let t = translate_tld(raw, 2, 0x1a0);
        let fetches = t
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(&inst.op, Op::TexelFetch { .. }))
            .collect::<Vec<_>>();
        assert_eq!(fetches.len(), 1);
        assert_eq!(fetches[0].dest_reg, Some(21));
        match &fetches[0].op {
            Op::TexelFetch {
                cbuf_binding,
                cbuf_word_offset,
                cbuf_secondary_word_offset,
                x,
                y,
                z,
                component,
            } => {
                assert_eq!(*cbuf_binding, 2);
                assert_eq!(*cbuf_word_offset, 0x68);
                assert_eq!(*cbuf_secondary_word_offset, None);
                assert_eq!(*x, Value::GprIn(21));
                assert_eq!(*y, None);
                assert_eq!(*z, None);
                assert_eq!(*component, 0);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn pps_tld_b_preserves_exact_same_binding_or_origins() {
        for (setup, raw, primary, secondary) in [
            (
                [0x4c98_0788_06c7_0004, 0x4c47_0208_1627_0404],
                0xdd38_0000_8047_1515,
                0x6c,
                0x162,
            ),
            (
                [0x4c98_0788_05a7_0001, 0x4c47_0208_15a7_0101],
                0xdd3a_0000_a017_1208,
                0x5a,
                0x15a,
            ),
        ] {
            let mut t = Translator::new();
            assert!(t.translate(setup[0]));
            assert!(t.translate(setup[1]));
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            let fetch = t
                .program
                .instructions
                .iter()
                .find(|inst| matches!(&inst.op, Op::TexelFetch { .. }))
                .expect("texel fetch");
            match &fetch.op {
                Op::TexelFetch {
                    cbuf_binding,
                    cbuf_word_offset,
                    cbuf_secondary_word_offset,
                    ..
                } => {
                    assert_eq!(*cbuf_binding, 2);
                    assert_eq!(*cbuf_word_offset, primary);
                    assert_eq!(*cbuf_secondary_word_offset, Some(secondary));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn dusk_tld_b_preserves_masked_handle_origin() {
        let mut t = Translator::new();
        for raw in [
            0x4c98_0784_0047_0003,
            0x0400_00ff_fff7_0303,
            0x4c98_0784_0057_0006,
            0x040f_ff00_0007_0606,
            0x5c47_0200_0067_0306,
        ] {
            assert!(t.translate(raw), "raw={raw:#018x}");
        }
        assert_eq!(
            t.trace_cbuf_handle_origin(&t.read_reg(6), None, None),
            Some((
                CbufHandleOrigin {
                    binding: 1,
                    word_offset: 4,
                    secondary_word_offset: Some(5),
                    secondary_binding: None,
                },
                false,
            ))
        );

        assert!(t.translate(0x5c98_0780_0037_0006));
        assert!(t.translate(0xdd38_0003_a067_1400));
        assert_eq!(t.unimplemented_count, 0);

        let fetches: Vec<_> = t
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TexelFetch { .. }))
            .collect();
        assert_eq!(fetches.len(), 3);
        for (component, fetch) in fetches.into_iter().enumerate() {
            assert_eq!(fetch.dest_reg, Some(component as u8));
            assert!(matches!(
                fetch.op,
                Op::TexelFetch {
                    cbuf_binding: 1,
                    cbuf_word_offset: 4,
                    cbuf_secondary_word_offset: None,
                    x: Value::GprIn(20),
                    y: Some(Value::GprIn(21)),
                    z: None,
                    component: actual,
                } if actual == component as u8
            ));
        }
    }

    #[test]
    fn pps_tld_b_pair_survives_mov_select_and_unanimous_phi() {
        let mut defs = ValueDefs::new();
        defs.insert(
            ValueId(1),
            Op::LoadCbuf {
                binding: 2,
                byte_offset: 0x1b0,
            },
        );
        defs.insert(
            ValueId(2),
            Op::LoadCbuf {
                binding: 2,
                byte_offset: 0x588,
            },
        );
        defs.insert(
            ValueId(3),
            Op::ILop {
                a: Value::Inst(ValueId(1)),
                b: Value::Inst(ValueId(2)),
                op: LogicOp::Or,
                not_a: false,
                not_b: false,
            },
        );
        defs.insert(ValueId(4), Op::Mov(Value::Inst(ValueId(3))));
        defs.insert(
            ValueId(5),
            Op::SelectPred {
                pred: Predicate {
                    idx: 0,
                    negate: false,
                },
                if_true: Value::Inst(ValueId(4)),
                if_false: Value::Inst(ValueId(3)),
            },
        );
        defs.insert(
            ValueId(6),
            Op::Phi {
                sources: vec![(1, Value::Inst(ValueId(5))), (2, Value::Inst(ValueId(4)))],
            },
        );
        let mut initial = HashMap::new();
        initial.insert(4, Value::Inst(ValueId(6)));
        let mut t = Translator::with_initial(initial, HashMap::new(), 100);
        assert!(t.translate_with_defs(0xdd38_0000_8047_1515, &defs));
        let fetch = t
            .program
            .instructions
            .iter()
            .find(|inst| matches!(&inst.op, Op::TexelFetch { .. }))
            .expect("texel fetch");
        assert!(matches!(
            &fetch.op,
            Op::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x6c,
                cbuf_secondary_word_offset: Some(0x162),
                ..
            }
        ));
    }

    #[test]
    fn pps_tld_b_rejects_different_binding_and_nested_pairs() {
        let mov = 0x4c98_0788_06c7_0004u64;
        let or = 0x4c47_0208_1627_0404u64;
        let tld = 0xdd38_0000_8047_1515u64;

        let mut different_binding = Translator::new();
        assert!(different_binding.translate(mov));
        let other_binding = (or & !(0x1f << 34)) | (3 << 34);
        assert!(different_binding.translate(other_binding));
        assert!(!different_binding.translate(tld));
        assert_eq!(different_binding.unimplemented_count, 1);

        let mut nested = Translator::new();
        assert!(nested.translate(mov));
        assert!(nested.translate(or));
        let third_origin = (or & !(0x3fff << 20)) | (0x164 << 20);
        assert!(nested.translate(third_origin));
        assert!(!nested.translate(tld));
        assert_eq!(nested.unimplemented_count, 1);
    }

    #[test]
    fn pps_tld_b_2d_masks_use_signed_xy_and_decimated_destinations() {
        for (raw, expected_components, expected_dest) in [
            (0xdd3a_0000_a017_1208, vec![0], 8u8),
            (0xdd3a_0001_2017_1e09, vec![1], 9),
            (0xdd3a_0002_2017_220a, vec![2], 10),
            (0xdd3a_0007_a017_0604, vec![0, 1, 2, 3], 4),
        ] {
            let t = translate_tld(raw, 1, 0x44);
            let fetches = t
                .program
                .instructions
                .iter()
                .filter(|inst| matches!(&inst.op, Op::TexelFetch { .. }))
                .collect::<Vec<_>>();
            assert_eq!(fetches.len(), expected_components.len());
            for (index, (inst, expected_component)) in fetches
                .iter()
                .zip(expected_components.into_iter())
                .enumerate()
            {
                assert_eq!(inst.dest_reg, Some(expected_dest + index as u8));
                match &inst.op {
                    Op::TexelFetch {
                        cbuf_binding,
                        cbuf_word_offset,
                        cbuf_secondary_word_offset,
                        x,
                        y,
                        z,
                        component,
                    } => {
                        assert_eq!(*cbuf_binding, 1);
                        assert_eq!(*cbuf_word_offset, 0x11);
                        assert_eq!(*cbuf_secondary_word_offset, None);
                        assert_eq!(*x, Value::GprIn(reg_a(raw)));
                        assert_eq!(*y, Some(Value::GprIn(reg_a(raw).wrapping_add(1))));
                        assert_eq!(*z, None);
                        assert_eq!(*component, expected_component);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn pps_tld_b_scene_1d_and_3d_forms_lower_exactly() {
        for (raw, expected_y, expected_z) in [
            (0xdd38_0007_82b7_0404, None, None),
            (
                0xdd38_0007_c277_0000,
                Some(Value::GprIn(1)),
                Some(Value::GprIn(2)),
            ),
        ] {
            let t = translate_tld(raw, 3, 0x90);
            let fetches = t
                .program
                .instructions
                .iter()
                .filter(|inst| matches!(&inst.op, Op::TexelFetch { .. }))
                .collect::<Vec<_>>();
            assert_eq!(fetches.len(), 4);
            for (component, inst) in fetches.iter().enumerate() {
                match &inst.op {
                    Op::TexelFetch {
                        cbuf_binding,
                        cbuf_word_offset,
                        cbuf_secondary_word_offset,
                        x,
                        y,
                        z,
                        component: actual_component,
                    } => {
                        assert_eq!(*cbuf_binding, 3);
                        assert_eq!(*cbuf_word_offset, 0x24);
                        assert_eq!(*cbuf_secondary_word_offset, None);
                        assert_eq!(*x, Value::GprIn(reg_a(raw)));
                        assert_eq!(*y, expected_y);
                        assert_eq!(*z, expected_z);
                        assert_eq!(*actual_component, component as u8);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn pps_tld_b_standard_instruction_predicate_guards_writes() {
        let raw = 0xdd38_0000_a000_1200;
        let t = translate_tld(raw, 0, 0x30);
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(&inst.op, Op::TexelFetch { .. })));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            &inst.op,
            Op::SelectPred {
                pred: Predicate {
                    idx: 0,
                    negate: false
                },
                ..
            }
        )));
    }

    #[test]
    fn pps_tld_b_unobserved_modes_and_dynamic_handles_fail_closed() {
        let base = 0xdd38_0000_8047_1515u64;
        let mut unsupported = vec![
            base | (1 << 55),
            base | (1 << 50),
            base | (1 << 35),
            base | (1 << 54),
            base & !(0x7 << 51),
            base & !(0xF << 31),
        ];
        for tex_type in [1u64, 3, 5, 6, 7] {
            unsupported.push((base & !(0x7 << 28)) | (tex_type << 28));
        }
        for raw in unsupported {
            let mut t = Translator::new();
            assert!(t.translate(static_ldc(reg_b(raw), 2, 0x1a0)));
            assert!(!t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented {
                    opcode: Opcode::TLD_b,
                    raw: failed,
                }) if *failed == raw
            ));
        }

        let mut t = Translator::new();
        assert!(!t.translate(base));
        assert_eq!(t.unimplemented_count, 1);
        assert!(matches!(
            t.program.instructions.last().map(|inst| &inst.op),
            Some(Op::Unimplemented {
                opcode: Opcode::TLD_b,
                raw: failed,
            }) if *failed == base
        ));
    }

    #[test]
    fn pps_compute_bindless_tld_keeps_exact_handle_origin_and_dimensions() {
        let mut one_d = Translator::new_compute();
        for raw in [
            0x4c98_0788_05a7_0007,
            0x4c47_0208_15a7_0707,
            0xdd3a_0000_8077_0404,
        ] {
            assert!(one_d.translate(raw), "raw={raw:#018x}");
        }
        assert!(matches!(
            one_d.program.instructions.last(),
            Some(Inst {
                op: Op::TexelFetchHandle {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x5a,
                        cbuf_secondary_word_offset: Some(0x15a),
                    },
                    dimension: ImageDimension::D1,
                    x: Value::GprIn(4),
                    y: None,
                    z: None,
                    component: 0,
                },
                dest_reg: Some(4),
                ..
            })
        ));

        let mut three_d = Translator::new_compute();
        for raw in [
            0x4c98_0788_05e7_0017,
            0x4c47_0208_15a7_1717,
            0xdd3a_0003_c177_0c04,
        ] {
            assert!(three_d.translate(raw), "raw={raw:#018x}");
        }
        let fetches = three_d
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TexelFetchHandle { .. }))
            .collect::<Vec<_>>();
        assert_eq!(fetches.len(), 3);
        for (component, fetch) in fetches.into_iter().enumerate() {
            assert_eq!(fetch.dest_reg, Some(4 + component as u8));
            assert!(matches!(
                fetch.op,
                Op::TexelFetchHandle {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x5e,
                        cbuf_secondary_word_offset: Some(0x15a),
                    },
                    dimension: ImageDimension::D3,
                    x: Value::GprIn(12),
                    y: Some(Value::GprIn(13)),
                    z: Some(Value::GprIn(14)),
                    component: actual,
                } if actual == component as u8
            ));
        }
    }

    #[test]
    fn compute_direct_tld_retains_bound_handle_word() {
        let bindless_template = 0xdd3a_0000_8077_0404u64;
        let direct = (bindless_template & !(0xffu64 << 56) & !(0x1fffu64 << 36))
            | (0xdcu64 << 56)
            | (0x48u64 << 36);
        assert_eq!(
            decode_one(direct).map(|decoded| decoded.opcode),
            Some(Opcode::TLD)
        );

        let mut t = Translator::new_compute();
        assert!(t.translate(direct));
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::TexelFetchHandle {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x48,
                    },
                    dimension: ImageDimension::D1,
                    x: Value::GprIn(4),
                    y: None,
                    z: None,
                    component: 0,
                },
                dest_reg: Some(4),
                ..
            })
        ));
    }

    #[test]
    fn captured_compute_filtered_sample_retains_bindless_handle_and_lod_zero() {
        let mut translator = Translator::new_compute();
        for raw in [
            0x4c98_0788_05e7_000a,
            0x5c98_0780_0ff7_000b,
            0x4c47_0208_15a7_0a0a,
            0xdeb8_0030_a0a7_0009,
        ] {
            assert!(translator.translate(raw), "raw={raw:#018x}");
        }
        assert_eq!(translator.unimplemented_count, 0);
        assert!(translator.program.instructions.iter().any(|instruction| {
            matches!(
                instruction,
                Inst {
                    op: Op::SampleTexHandle {
                        sample_site: Some(_),
                        handle: TextureHandleOrigin::Bindless {
                            cbuf_binding: 2,
                            cbuf_word_offset: 0x5e,
                            cbuf_secondary_word_offset: Some(0x15a),
                        },
                        dimension: ImageDimension::D2,
                        u: Value::GprIn(0),
                        v: Some(Value::GprIn(1)),
                        w: None,
                        implicit_lod: false,
                        lod_bias: None,
                        explicit_lod: Some(Value::Zero),
                        texel_offset: Some((Value::ImmU32(0), Value::ImmU32(0), Value::ImmU32(0))),
                        dref: None,
                        component: 0,
                    },
                    dest_reg: Some(9),
                    ..
                }
            )
        }));
    }

    #[test]
    fn pps_compute_txq_dimension_uses_exact_bound_handle_and_lod() {
        let mut t = Translator::new_compute();
        assert!(t.translate(0xdf48_0483_8047_0808));
        assert_eq!(t.unimplemented_count, 0);
        let queries = t
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TextureQueryDimension { .. }))
            .collect::<Vec<_>>();
        assert_eq!(queries.len(), 3);
        for (component, query) in queries.into_iter().enumerate() {
            assert_eq!(query.dest_reg, Some(8 + component as u8));
            assert!(matches!(
                query.op,
                Op::TextureQueryDimension {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x48,
                    },
                    lod: Value::GprIn(8),
                    component: actual,
                } if actual == component as u8
            ));
        }
    }

    #[test]
    fn fragment_tmml_2d_queries_rg_and_converts_to_maxwell_fixed_point() {
        let raw = tmml_raw(false, 8, 4, 0x24, 2, 0b11);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TMML)
        );
        let mut translator = Translator::new_fragment();
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);

        let queries = translator
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TextureQueryLod { .. }))
            .collect::<Vec<_>>();
        assert_eq!(queries.len(), 2);
        for (component, query) in queries.into_iter().enumerate() {
            assert_eq!(query.dest_reg, None);
            assert!(matches!(
                query.op,
                Op::TextureQueryLod {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x24,
                    },
                    u: Value::GprIn(4),
                    v: Value::GprIn(5),
                    arrayed: false,
                    component: actual,
                } if actual == component as u8
            ));
            let query_id = query.result.expect("query result");
            let conversion = translator
                .program
                .instructions
                .iter()
                .find(|inst| {
                    matches!(
                        inst.op,
                        Op::F2I {
                            src: Value::Inst(source),
                            signed: false,
                            round: 3,
                        } if source == query_id
                    )
                })
                .expect("TMML float-to-unsigned conversion");
            let conversion_id = conversion.result.expect("conversion result");
            assert!(translator.program.instructions.iter().any(|inst| {
                inst.dest_reg == Some(8 + component as u8)
                    && matches!(
                        inst.op,
                        Op::IShl {
                            a: Value::Inst(source),
                            b: Value::ImmU32(8),
                        } if source == conversion_id
                    )
            }));
        }
    }

    #[test]
    fn fragment_tmml_bindless_2d_array_uses_static_handle_and_drops_layer_coordinate() {
        let raw = tmml_raw(true, 20, 4, 12, 3, 0b10);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TMML_b)
        );
        let mut translator = Translator::new_fragment();
        assert!(translator.translate(static_ldc(12, 2, 0x1a0)));
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);
        assert!(translator.program.instructions.iter().any(|inst| {
            matches!(
                inst.op,
                Op::TextureQueryLod {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x68,
                        cbuf_secondary_word_offset: None,
                    },
                    u: Value::GprIn(5),
                    v: Value::GprIn(6),
                    arrayed: true,
                    component: 1,
                }
            )
        }));
        assert!(translator
            .program
            .instructions
            .iter()
            .any(|inst| { inst.dest_reg == Some(20) && matches!(inst.op, Op::IShl { .. }) }));
    }

    #[test]
    fn tmml_unsupported_stage_type_lanes_and_handles_fail_closed() {
        let direct_2d = tmml_raw(false, 8, 4, 0x24, 2, 0b11);
        let cases = [
            tmml_raw(false, 8, 4, 0x24, 6, 0b11),
            tmml_raw(false, 8, 4, 0x24, 2, 0b100),
        ];
        for raw in cases {
            let mut translator = Translator::new_fragment();
            assert!(!translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                translator.program.instructions.last(),
                Some(Inst {
                    op: Op::Unimplemented { raw: failed, .. },
                    ..
                }) if *failed == raw
            ));
        }

        let mut vertex = Translator::new();
        assert!(!vertex.translate(direct_2d));
        assert_eq!(vertex.unimplemented_count, 1);

        let dynamic = tmml_raw(true, 8, 4, 12, 2, 0b11);
        let mut bindless = Translator::new_fragment();
        assert!(!bindless.translate(dynamic));
        assert_eq!(bindless.unimplemented_count, 1);

        let mov = 0x4c98_0788_06c7_0004u64;
        let or = 0x4c47_0208_1627_0404u64;
        let other_binding = (or & !(0x1f << 34)) | (3 << 34);
        let cross_cbuf = tmml_raw(true, 8, 4, 4, 2, 0b11);
        let mut cross = Translator::new_fragment();
        assert!(cross.translate(mov));
        assert!(cross.translate(other_binding));
        assert!(!cross.translate(cross_cbuf));
        assert_eq!(cross.unimplemented_count, 1);
    }

    #[test]
    fn graphics_txq_dimension_preserves_bound_and_bindless_handles() {
        let direct_raw = (0b1101_1111_0100_1u64 << 51)
            | (1u64 << 22)
            | (0xdu64 << 31)
            | (0x48u64 << 36)
            | (u64::from(PT) << 16)
            | (8u64 << 8)
            | 4;
        assert_eq!(
            decode_one(direct_raw).map(|decoded| decoded.opcode),
            Some(Opcode::TXQ)
        );
        let mut direct = Translator::new_fragment();
        assert!(direct.translate(direct_raw));
        assert_eq!(direct.unimplemented_count, 0);
        let direct_queries = direct
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TextureQueryDimension { .. }))
            .collect::<Vec<_>>();
        assert_eq!(direct_queries.len(), 3);
        for ((dest, component), query) in [(4, 0), (5, 2), (6, 3)].into_iter().zip(direct_queries) {
            assert_eq!(query.dest_reg, Some(dest));
            assert!(matches!(
                query.op,
                Op::TextureQueryDimension {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x48,
                    },
                    lod: Value::GprIn(8),
                    component: actual,
                } if actual == component
            ));
        }

        let bindless_raw = (0b1101_1111_0101_0u64 << 51)
            | (1u64 << 22)
            | (0xbu64 << 31)
            | (u64::from(PT) << 16)
            | (12u64 << 8)
            | 20;
        assert_eq!(
            decode_one(bindless_raw).map(|decoded| decoded.opcode),
            Some(Opcode::TXQ_b)
        );
        let mut bindless = Translator::new_fragment();
        assert!(bindless.translate(static_ldc(12, 2, 0x1a0)));
        assert!(bindless.translate(bindless_raw));
        assert_eq!(bindless.unimplemented_count, 0);
        let bindless_queries = bindless
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::TextureQueryDimension { .. }))
            .collect::<Vec<_>>();
        assert_eq!(bindless_queries.len(), 3);
        for ((dest, component), query) in [(20, 0), (21, 1), (22, 3)]
            .into_iter()
            .zip(bindless_queries)
        {
            assert_eq!(query.dest_reg, Some(dest));
            assert!(matches!(
                query.op,
                Op::TextureQueryDimension {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x68,
                        cbuf_secondary_word_offset: None,
                    },
                    lod: Value::GprIn(13),
                    component: actual,
                } if actual == component
            ));
        }
    }

    #[test]
    fn graphics_tld4_2d_gather_preserves_mask_component_and_bound_handle() {
        let raw = tld4_raw(false, 8, 4, 0x24, 2, 0b1011, 2, 0, false, PT);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TLD4)
        );

        let mut translator = Translator::new_fragment();
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);
        let gathers = translator
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::GatherTex { .. }))
            .collect::<Vec<_>>();
        assert_eq!(gathers.len(), 3);
        for ((expected_dest, expected_lane), gather) in
            [(8, 0), (9, 1), (10, 3)].into_iter().zip(gathers)
        {
            assert_eq!(gather.dest_reg, Some(expected_dest));
            assert!(matches!(
                gather.op,
                Op::GatherTex {
                    tex_id: 0x24,
                    u: Value::GprIn(4),
                    v: Value::GprIn(5),
                    array: None,
                    gather_component: 2,
                    lane,
                } if lane == expected_lane
            ));
        }
    }

    #[test]
    fn graphics_tld4_bindless_gather_uses_exact_static_cbuf_origin() {
        let raw = tld4_raw(true, 8, 4, 12, 2, 0xf, 3, 0, false, PT);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TLD4_b)
        );

        let mut translator = Translator::new_fragment();
        assert!(translator.translate(static_ldc(12, 2, 0x1a0)));
        assert!(translator.translate(raw));
        assert_eq!(translator.unimplemented_count, 0);
        let expected_tex_id = bindless_texture_id_pair(2, 0x1a0 / 4, None);
        let gathers = translator
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::GatherTex { .. }))
            .collect::<Vec<_>>();
        assert_eq!(gathers.len(), 4);
        for (lane, gather) in gathers.into_iter().enumerate() {
            assert_eq!(gather.dest_reg, Some(8 + lane as u8));
            assert!(matches!(
                gather.op,
                Op::GatherTex {
                    tex_id,
                    u: Value::GprIn(4),
                    v: Value::GprIn(5),
                    array: None,
                    gather_component: 3,
                    lane: actual_lane,
                } if tex_id == expected_tex_id && actual_lane == lane as u8
            ));
        }
    }

    #[test]
    fn tld4_unsupported_forms_fail_closed_without_zero_results() {
        let direct_base = tld4_raw(false, 8, 4, 0x24, 2, 0xf, 0, 0, false, PT);
        let unsupported = [
            tld4_raw(false, 8, 4, 0x24, 2, 0xf, 0, 1, false, PT),
            tld4_raw(false, 8, 4, 0x24, 2, 0xf, 0, 0, true, PT),
            tld4_raw(false, 8, 4, 0x24, 4, 0xf, 0, 0, false, PT),
            tld4_raw(false, 8, 4, 0x24, 2, 0xf, 0, 0, false, 6),
            tld4_raw(false, 8, 4, 0x24, 2, 0, 0, 0, false, PT),
        ];
        for raw in unsupported {
            let opcode = decode_one(raw).expect("TLD4 must decode").opcode;
            let mut translator = Translator::new_fragment();
            assert!(!translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                translator.program.instructions.last(),
                Some(Inst {
                    op: Op::Unimplemented {
                        opcode: failed_opcode,
                        raw: failed_raw,
                    },
                    ..
                }) if *failed_opcode == opcode && *failed_raw == raw
            ));
            assert!(!translator.program.instructions.iter().any(|inst| {
                matches!(inst.op, Op::Mov(Value::Zero)) && inst.dest_reg.is_some()
            }));
        }

        let mut compute = Translator::new_compute();
        assert!(compute.translate(direct_base));
        assert_eq!(compute.unimplemented_count, 0);

        let dynamic_bindless = tld4_raw(true, 8, 4, 12, 2, 0xf, 0, 0, false, PT);
        let mut bindless = Translator::new_fragment();
        assert!(!bindless.translate(dynamic_bindless));
        assert_eq!(bindless.unimplemented_count, 1);
    }

    #[test]
    fn tld4s_preserves_f32_destinations_and_packs_fp16_results() {
        let f32_raw = tld4s_raw(8, 12, 4, 6, 0x24, 1, false, false, false);
        assert_eq!(
            decode_one(f32_raw).map(|decoded| decoded.opcode),
            Some(Opcode::TLD4S)
        );
        let mut f32_translator = Translator::new_fragment();
        assert!(f32_translator.translate(f32_raw));
        let f32_gathers = f32_translator
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::GatherTex { .. }))
            .collect::<Vec<_>>();
        assert_eq!(f32_gathers.len(), 4);
        for ((expected_dest, expected_lane), gather) in [(8, 0), (9, 1), (12, 2), (13, 3)]
            .into_iter()
            .zip(f32_gathers)
        {
            assert_eq!(gather.dest_reg, Some(expected_dest));
            assert!(matches!(
                gather.op,
                Op::GatherTex {
                    tex_id: 0x24,
                    u: Value::GprIn(4),
                    v: Value::GprIn(6),
                    array: None,
                    gather_component: 1,
                    lane,
                } if lane == expected_lane
            ));
        }

        let fp16_raw = tld4s_raw(8, 12, 4, 6, 0x24, 1, true, false, false);
        assert_eq!(
            decode_one(fp16_raw).map(|decoded| decoded.opcode),
            Some(Opcode::TLD4S)
        );
        let mut fp16_translator = Translator::new_fragment();
        assert!(fp16_translator.translate(fp16_raw));
        assert_eq!(
            fp16_translator
                .program
                .instructions
                .iter()
                .filter(|inst| matches!(inst.op, Op::GatherTex { .. }))
                .count(),
            4
        );
        let packs = fp16_translator
            .program
            .instructions
            .iter()
            .filter(|inst| matches!(inst.op, Op::PackHalf2 { .. }))
            .collect::<Vec<_>>();
        assert_eq!(packs.len(), 2);
        assert_eq!(packs[0].dest_reg, Some(8));
        assert_eq!(packs[1].dest_reg, Some(12));
        assert_eq!(fp16_translator.unimplemented_count, 0);
    }

    #[test]
    fn tld4s_offset_depth_and_unaligned_f32_forms_fail_closed() {
        let cases = [
            tld4s_raw(8, 12, 4, 6, 0x24, 1, false, true, false),
            tld4s_raw(8, 12, 4, 6, 0x24, 1, false, false, true),
            tld4s_raw(9, 12, 4, 6, 0x24, 1, false, false, false),
        ];
        for raw in cases {
            let mut translator = Translator::new_fragment();
            assert!(!translator.translate(raw), "raw={raw:#018x}");
            assert_eq!(translator.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                translator.program.instructions.last(),
                Some(Inst {
                    op: Op::Unimplemented {
                        opcode: Opcode::TLD4S,
                        raw: failed_raw,
                    },
                    ..
                }) if *failed_raw == raw
            ));
        }
    }

    #[test]
    fn pps_compute_typeless_bindless_sust_emits_3d_image_write() {
        let mut t = Translator::new_compute();
        assert!(t.translate(0x4c98_0788_0487_000b));
        assert!(t.translate(0xeb20_058a_00f7_0400));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::ImageWrite {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x48,
                        cbuf_secondary_word_offset: None,
                    },
                    dimension: ImageDimension::D3,
                    x: Value::GprIn(4),
                    y: Some(Value::GprIn(5)),
                    z: Some(Value::GprIn(6)),
                    values: [
                        Value::GprIn(0),
                        Value::GprIn(1),
                        Value::GprIn(2),
                        Value::GprIn(3),
                    ],
                },
                result: None,
                ..
            })
        ));
    }

    #[test]
    fn compute_sust_models_1d_buffer_2d_and_3d_coordinates() {
        let base = 0xeb20_058a_00f7_0400u64;
        for (surface_type, dimension, y, z) in [
            (0u64, ImageDimension::D1, None, None),
            (1, ImageDimension::Buffer, None, None),
            (3, ImageDimension::D2, Some(Value::GprIn(5)), None),
            (
                5,
                ImageDimension::D3,
                Some(Value::GprIn(5)),
                Some(Value::GprIn(6)),
            ),
        ] {
            let raw = (base & !(0x7u64 << 33)) | (surface_type << 33);
            let mut t = Translator::new_compute();
            assert!(t.translate(0x4c98_0788_0487_000b));
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::ImageWrite {
                        dimension: actual_dimension,
                        x: Value::GprIn(4),
                        y: actual_y,
                        z: actual_z,
                        ..
                    },
                    ..
                }) if *actual_dimension == dimension && *actual_y == y && *actual_z == z
            ));
        }
    }

    #[test]
    fn compute_sust_buffer_preserves_bound_handle_data_and_predicate() {
        let base = 0xeb20_058a_00f7_0400u64;
        let raw = (base & !(0x7u64 << 33) & !(0x1fffu64 << 36) & !(0xfu64 << 16))
            | (1u64 << 33)
            | (0x123u64 << 36)
            | (1u64 << 51)
            | (2u64 << 16)
            | (1u64 << 19);
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::ImageWrite {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x123,
                    },
                    dimension: ImageDimension::Buffer,
                    x: Value::GprIn(4),
                    y: None,
                    z: None,
                    values: [
                        Value::GprIn(0),
                        Value::GprIn(1),
                        Value::GprIn(2),
                        Value::GprIn(3),
                    ],
                },
                pred: Some(Predicate {
                    idx: 2,
                    negate: true,
                }),
                result: None,
                ..
            })
        ));
    }

    #[test]
    fn compute_suatom_captures_lower_buffer_add_and_exchange() {
        for (raw, expected_op, dest, coord, operand, handle_reg) in [
            (0xea70_0382_0020_0502, ImageAtomicOp::Add, 2, 5, 2, 7),
            (0xea70_0203_00a7_0d04, ImageAtomicOp::Exchange, 4, 13, 10, 4),
            (0xea70_0583_0097_0404, ImageAtomicOp::Exchange, 4, 4, 9, 11),
            (0xea70_0282_0022_0701, ImageAtomicOp::Add, 1, 7, 2, 5),
            (0xea70_0282_0022_0705, ImageAtomicOp::Add, 5, 7, 2, 5),
        ] {
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::SUATOM)
            );
            let mut t = Translator::new_compute();
            assert!(t.translate(static_ldc(handle_reg, 2, 0x120)));
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0);

            let atomic = t
                .program
                .instructions
                .iter()
                .find(|inst| matches!(inst.op, Op::ImageAtomic { .. }))
                .expect("SUATOM must produce an image atomic");
            assert_eq!(atomic.pred, decoded_pred(raw));
            assert!(matches!(
                atomic.op,
                Op::ImageAtomic {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x48,
                        cbuf_secondary_word_offset: None,
                    },
                    dimension: ImageDimension::Buffer,
                    x: Value::GprIn(actual_coord),
                    y: None,
                    z: None,
                    value: Value::GprIn(actual_operand),
                    op: actual_op,
                    data_type: ImageAtomicType::Sd32,
                } if actual_coord == coord && actual_operand == operand && actual_op == expected_op
            ));
            if let Some(predicate) = decoded_pred(raw) {
                assert_eq!(atomic.dest_reg, None);
                assert!(t.program.instructions.iter().any(|inst| matches!(
                    inst,
                    Inst {
                        op: Op::SelectPred { pred, .. },
                        dest_reg: Some(actual_dest),
                        ..
                    } if *pred == predicate && *actual_dest == dest
                )));
            } else {
                assert_eq!(atomic.dest_reg, Some(dest));
            }
        }
    }

    #[test]
    fn compute_suatom_supports_u32_bound_buffers() {
        let captured = 0xea70_0382_0020_0502u64;
        let raw =
            (captured & !(1u64 << 54) & !(0x1fffu64 << 36) & !(0x7u64 << 51)) | (0x123u64 << 36);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::SUATOM)
        );
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert!(matches!(
            t.program
                .instructions
                .iter()
                .find(|inst| matches!(inst.op, Op::ImageAtomic { .. })),
            Some(Inst {
                op: Op::ImageAtomic {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x123,
                    },
                    op: ImageAtomicOp::Add,
                    data_type: ImageAtomicType::U32,
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn compute_suatom_lowers_all_int32_types_and_operations() {
        let base = 0xea70_0382_0020_0502u64;
        let base = (base & !(1u64 << 54) & !(0x1fffu64 << 36)) | (0x123u64 << 36);
        for (size, expected_type) in [
            (0, ImageAtomicType::U32),
            (1, ImageAtomicType::S32),
            (6, ImageAtomicType::Sd32),
        ] {
            for (atomic_op, expected_op) in [
                (0, ImageAtomicOp::Add),
                (1, ImageAtomicOp::Min),
                (2, ImageAtomicOp::Max),
                (3, ImageAtomicOp::Increment),
                (4, ImageAtomicOp::Decrement),
                (5, ImageAtomicOp::And),
                (6, ImageAtomicOp::Or),
                (7, ImageAtomicOp::Xor),
                (8, ImageAtomicOp::Exchange),
            ] {
                let raw =
                    (base & !(0x7u64 << 51) & !(0xfu64 << 29)) | (size << 51) | (atomic_op << 29);
                let mut t = Translator::new_compute();
                assert!(t.translate(raw), "raw={raw:#018x}");
                assert_eq!(t.unimplemented_count, 0);
                assert!(matches!(
                    t.program.instructions.first(),
                    Some(Inst {
                        op: Op::ImageAtomic {
                            op,
                            data_type,
                            ..
                        },
                        ..
                    }) if *op == expected_op && *data_type == expected_type
                ));
            }
        }
        assert!(!ImageAtomicType::U32.is_signed());
        assert!(ImageAtomicType::S32.is_signed());
        assert!(!ImageAtomicType::Sd32.is_signed());
    }

    #[test]
    fn compute_suatom_unsupported_forms_fail_closed() {
        let base = 0xea70_0382_0020_0502u64;
        for raw in [
            (base & !(0x7u64 << 33)) | (3u64 << 33),
            (base & !(0x7u64 << 51)) | (2u64 << 51),
            base | (1u64 << 49),
            (base & !(0xfu64 << 29)) | (9u64 << 29),
        ] {
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::SUATOM)
            );
            let mut t = Translator::new_compute();
            assert!(!t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1);
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::Unimplemented {
                        opcode: Opcode::SUATOM,
                        raw: rejected,
                    },
                    ..
                }) if *rejected == raw
            ));
        }

        let mut untraceable = Translator::new_compute();
        assert!(!untraceable.translate(base));
        assert_eq!(untraceable.unimplemented_count, 1);
    }

    #[test]
    fn pps_icmp_imm_lowers_all_observed_encodings() {
        for raw in [
            0x3646_0000_0077_ff04,
            0x3646_0000_0077_ff03,
            0x3646_0800_0077_ff14,
            0x3646_0000_0077_ff10,
            0x3646_0200_0077_ff07,
            0x3646_0580_0077_ff0b,
            0x3646_0a00_0077_ff22,
            0x3646_0000_0077_ff16,
            0x3646_0180_0077_ff03,
            0x3647_0a80_0017_1717,
            0x3646_0180_0077_ff04,
            0x3646_0600_0077_ff0b,
            0x3646_0500_0077_ff0a,
            0x3647_0500_0017_1919,
            0x3646_0300_0077_ff06,
            0x3646_0800_0077_ff13,
            0x3646_0c00_0077_ff1c,
            0x3647_0600_0017_1313,
            0x3646_0c00_0077_ff23,
            0x3647_0400_0017_1509,
            0x3646_0100_0077_ff05,
            0x3646_0200_0077_ff08,
            0x3646_0200_0077_ff10,
            0x3646_0a00_0077_ff16,
            0x3646_0200_0077_ff14,
        ] {
            let mut t = Translator::new();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0, "raw={raw:#018x}");
            assert!(!t
                .program
                .instructions
                .iter()
                .any(|inst| matches!(inst.op, Op::Unimplemented { .. })));
        }
    }

    #[test]
    fn pps_icmp_imm_preserves_unsigned_and_signed_selects() {
        for (raw, signed, compare_reg, select_src, immediate, dest) in [
            (0x3646_0000_0077_ff04, false, 0, Value::Zero, 7, 4),
            (0x3647_0a80_0017_1717, true, 21, Value::GprIn(23), 1, 23),
        ] {
            let mut t = Translator::new();
            assert!(t.translate(raw));
            assert_eq!(t.program.instructions.len(), 5);
            assert!(matches!(
                t.program.instructions[0].op,
                Op::ISet {
                    cmp: ICmp::Le,
                    signed: actual_signed,
                    a: Value::GprIn(actual_compare_reg),
                    b: Value::Zero,
                    bool_float: false,
                } if actual_signed == signed && actual_compare_reg == compare_reg
            ));
            assert!(matches!(
                t.program.instructions[1].op,
                Op::ILop {
                    a: Value::Inst(_),
                    b,
                    op: LogicOp::And,
                    not_a: false,
                    not_b: false,
                } if b == select_src
            ));
            assert!(matches!(
                t.program.instructions[2].op,
                Op::ILop {
                    a: Value::Inst(_),
                    b: Value::ImmU32(actual_immediate),
                    op: LogicOp::And,
                    not_a: true,
                    not_b: false,
                } if actual_immediate == immediate
            ));
            assert!(matches!(
                t.program.instructions[4].op,
                Op::Mov(Value::Inst(_))
            ));
            assert_eq!(t.program.instructions[4].dest_reg, Some(dest));
        }
    }

    #[test]
    fn pps_icmp_forms_lower_with_yuzu_operand_order() {
        let cr = 0x4b4a_1680_0767_2828u64;
        let forms = [
            (cr & 0x0000_ffff_ffff_ffff) | (0x5b4au64 << 48),
            (cr & 0x0000_ffff_ffff_ffff) | (0x534au64 << 48),
            cr,
            (cr & 0x0000_ffff_ffff_ffff) | (0x364au64 << 48),
        ];
        for raw in forms {
            let mut t = Translator::new();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0, "raw={raw:#018x}");
            assert!(!t
                .program
                .instructions
                .iter()
                .any(|inst| matches!(inst.op, Op::Unimplemented { .. })));
        }

        let mut t = Translator::new();
        assert!(t.translate(0x4b4a_0880_0a17_0a0du64));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| {
            matches!(
                inst.op,
                Op::LoadCbuf {
                    binding: 0,
                    byte_offset: 644,
                }
            )
        }));
        assert!(t.program.instructions.iter().any(|inst| {
            matches!(
                inst.op,
                Op::ISet {
                    cmp: ICmp::Ne,
                    signed: false,
                    a: Value::GprIn(17),
                    b: Value::Zero,
                    bool_float: false,
                }
            )
        }));
        assert!(t.program.instructions.iter().any(|inst| {
            matches!(
                inst.op,
                Op::ILop {
                    b: Value::GprIn(10),
                    op: LogicOp::And,
                    not_a: false,
                    not_b: false,
                    ..
                }
            )
        }));
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| inst.dest_reg == Some(13)));
    }

    #[test]
    fn captured_nop_flo_and_popc_lower_without_fallback() {
        let mut nop = Translator::new_compute();
        assert!(nop.translate(0x50b0_0000_0007_0f00));
        assert!(nop.program.instructions.is_empty());
        assert_eq!(nop.unimplemented_count, 0);

        let mut flo = Translator::new_compute();
        assert!(flo.translate(0x5c30_0000_0017_0003));
        assert_eq!(flo.unimplemented_count, 0);
        assert_eq!(flo.program.instructions.len(), 1);
        assert!(matches!(
            flo.program.instructions[0].op,
            Op::FindUMsb {
                value: Value::GprIn(1)
            }
        ));
        assert_eq!(flo.program.instructions[0].dest_reg, Some(3));

        for (raw, dest, source) in [(0x5c08_0000_0017_0002, 2, 1), (0x5c08_0000_0047_0004, 4, 4)] {
            let mut popc = Translator::new_compute();
            assert!(popc.translate(raw), "raw={raw:#018x}");
            assert_eq!(popc.unimplemented_count, 0, "raw={raw:#018x}");
            assert_eq!(popc.program.instructions.len(), 1, "raw={raw:#018x}");
            assert!(matches!(
                popc.program.instructions[0].op,
                Op::BitCount {
                    value: Value::GprIn(actual_source)
                } if actual_source == source
            ));
            assert_eq!(popc.program.instructions[0].dest_reg, Some(dest));
        }
    }

    #[test]
    fn flo_reg_unsupported_modifiers_fail_closed() {
        let base = 0x5c30_0000_0017_0003u64;
        for bit in [47, 48] {
            let raw = base | (1u64 << bit);
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::FLO_reg)
            );
            let mut t = Translator::new_compute();
            assert!(!t.translate(raw), "bit={bit} raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "bit={bit} raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented {
                    opcode: Opcode::FLO_reg,
                    raw: failed,
                }) if *failed == raw
            ));
        }
    }

    #[test]
    fn flo_reg_tilde_and_shift_forms_lower() {
        let tilde = 0x5c30_0000_0017_0003u64 | (1u64 << 40);
        let mut t = Translator::new_compute();
        assert!(t.translate(tilde));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::ILop {
                op: LogicOp::PassB,
                not_b: true,
                ..
            }
        ));
        assert!(matches!(t.program.instructions[1].op, Op::FindUMsb { .. }));

        let shift = 0x5c30_0000_0017_0003u64 | (1u64 << 41);
        let mut t = Translator::new_compute();
        assert!(t.translate(shift));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(t.program.instructions[0].op, Op::FindUMsb { .. }));
        assert!(matches!(
            t.program.instructions[1].op,
            Op::ISet {
                cmp: ICmp::Ne,
                b: Value::ImmU32(0xFFFF_FFFF),
                ..
            }
        ));
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::ILop {
                op: LogicOp::Xor,
                ..
            }
        ));
    }

    #[test]
    fn lea_hi_reg_scaled_pair_add_lowers() {
        let raw = 0x5bdf_7f81_f037_0404u64;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::LEA_hi_reg)
        );
        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::IShr {
                a: Value::GprIn(4),
                b: Value::ImmU32(1),
                signed: false,
            }
        ));
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::IAdd {
                a: Value::GprIn(3),
                neg_a: false,
                neg_b: false,
                ..
            }
        ));
        assert_eq!(t.program.instructions.last().unwrap().dest_reg, Some(4));
    }

    #[test]
    fn lea_lo_imm_cappy_tower_capture_lowers() {
        let raw = 0x36d7_0200_00f7_1112u64;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::LEA_lo_imm)
        );

        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.program.instructions.len(), 1);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::IScAdd {
                a: Value::GprIn(17),
                b: Value::ImmU32(15),
                shift: 4,
                neg_a: false,
                neg_b: false,
            }
        ));
        assert_eq!(t.program.instructions[0].dest_reg, Some(18));
    }

    #[test]
    fn lea_lo_reg_and_cbuf_base_forms_lower() {
        let reg_raw = (0x5bd7u64 << 48)
            | (4u64 << 39)
            | (3u64 << 20)
            | (u64::from(PT) << 16)
            | (17u64 << 8)
            | 18;
        assert_eq!(
            decode_one(reg_raw).map(|decoded| decoded.opcode),
            Some(Opcode::LEA_lo_reg)
        );
        let mut reg_t = Translator::new_compute();
        assert!(reg_t.translate(reg_raw));
        assert_eq!(reg_t.unimplemented_count, 0);
        assert!(matches!(
            reg_t.program.instructions[0].op,
            Op::IScAdd {
                a: Value::GprIn(17),
                b: Value::GprIn(3),
                shift: 4,
                neg_a: false,
                neg_b: false,
            }
        ));

        let cbuf_raw = (0x4bd7u64 << 48)
            | (2u64 << 39)
            | (3u64 << 34)
            | (5u64 << 20)
            | (u64::from(PT) << 16)
            | (17u64 << 8)
            | 18;
        assert_eq!(
            decode_one(cbuf_raw).map(|decoded| decoded.opcode),
            Some(Opcode::LEA_lo_cbuf)
        );
        let mut cbuf_t = Translator::new_compute();
        assert!(cbuf_t.translate(cbuf_raw));
        assert_eq!(cbuf_t.unimplemented_count, 0);
        assert!(matches!(
            cbuf_t.program.instructions[0].op,
            Op::LoadCbuf {
                binding: 3,
                byte_offset: 20,
            }
        ));
        assert!(matches!(
            cbuf_t.program.instructions[1].op,
            Op::IScAdd {
                a: Value::GprIn(17),
                b: Value::Inst(_),
                shift: 2,
                neg_a: false,
                neg_b: false,
            }
        ));
        assert_eq!(cbuf_t.program.instructions[1].dest_reg, Some(18));
    }

    #[test]
    fn lea_lo_imm_supports_negation_and_outer_predication() {
        const CAPTURE: u64 = 0x36d7_0200_00f7_1112;
        let raw = (CAPTURE & !(0x7u64 << 16)) | (2u64 << 16) | (1u64 << 19) | (1u64 << 45);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::LEA_lo_imm)
        );

        let mut t = Translator::new_compute();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.program.instructions.len(), 2);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::IScAdd {
                a: Value::GprIn(17),
                b: Value::ImmU32(15),
                shift: 4,
                neg_a: true,
                neg_b: false,
            }
        ));
        assert!(matches!(
            t.program.instructions[1].op,
            Op::SelectPred {
                pred: Predicate {
                    idx: 2,
                    negate: true,
                },
                if_true: Value::Inst(_),
                if_false: Value::GprIn(18),
            }
        ));
        assert_eq!(t.program.instructions[1].dest_reg, Some(18));
    }

    #[test]
    fn lea_lo_imm_rejects_x_cc_and_auxiliary_predicate_forms() {
        const CAPTURE: u64 = 0x36d7_0200_00f7_1112;
        for raw in [
            CAPTURE | (1u64 << 46),
            CAPTURE | (1u64 << 47),
            (CAPTURE & !(0x7u64 << 48)) | (1u64 << 48),
        ] {
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::LEA_lo_imm),
                "raw={raw:#018x}"
            );

            let mut t = Translator::new_compute();
            assert!(!t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().unwrap().op,
                Op::Unimplemented {
                    opcode: Opcode::LEA_lo_imm,
                    raw: actual,
                } if actual == raw
            ));
        }
    }

    #[test]
    fn red_global_atomic_lowers_and_rejects_wide_sizes() {
        for (raw, op, is_signed, offset) in [
            (0xebf9_0000_0098_0002u64, ImageAtomicOp::Min, true, 0i32),
            (0xebf9_0000_4118_0002u64, ImageAtomicOp::Max, true, 4),
            (0xebf9_0000_0307_0002u64, ImageAtomicOp::Or, false, 0),
        ] {
            assert_eq!(
                decode_one(raw).map(|decoded| decoded.opcode),
                Some(Opcode::RED),
                "raw={raw:#018x}"
            );
            let mut t = Translator::new_compute();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0, "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().unwrap().op,
                Op::GlobalAtomic {
                    addr_lo: Value::GprIn(0),
                    value: Value::GprIn(2),
                    offset: actual_offset,
                    op: actual_op,
                    is_signed: actual_signed,
                } if actual_op == op && actual_signed == is_signed && actual_offset == offset
            ));
        }

        let wide = 0xebf9_0000_0098_0002u64 | (2u64 << 20);
        let mut t = Translator::new_compute();
        assert!(!t.translate(wide));
        assert_eq!(t.unimplemented_count, 1);
    }

    #[test]
    fn lop_forms_write_all_predicate_result_modes() {
        for opcode in [0x5c40u64, 0x4c40, 0x3840] {
            for (mode, expected) in [ICmp::F, ICmp::T, ICmp::Eq, ICmp::Ne]
                .into_iter()
                .enumerate()
            {
                for destination in [0, RZ] {
                    let raw = (opcode << 48)
                        | ((mode as u64) << 44)
                        | (1 << 41)
                        | (2 << 20)
                        | (7 << 16)
                        | (1 << 8)
                        | u64::from(destination);
                    let mut t = Translator::new_fragment();
                    assert!(t.translate(raw), "raw={raw:#018x}");
                    let logical = t
                        .program
                        .instructions
                        .iter()
                        .find(|inst| matches!(inst.op, Op::ILop { .. }))
                        .unwrap();
                    let predicate = t.program.instructions.last().unwrap();
                    assert!(matches!(predicate.op, Op::ISetPred {
                        cmp, src_a: Value::Inst(result), src_b: Value::Zero,
                        dest_p: 0, dest_np: PT, ..
                    } if cmp == expected && Some(result) == logical.result));
                    assert_eq!(t.snapshot_pred_state().get(&0), predicate.result.as_ref());
                }
            }
        }
    }

    #[test]
    fn lop_predicate_write_keeps_its_instruction_guard() {
        let raw = 0x5c40_3380_0028_01ff;
        let mut t = Translator::new_fragment();
        assert!(t.translate(raw));
        let predicate = t.program.instructions.last().unwrap();
        assert_eq!(
            predicate.pred,
            Some(Predicate {
                idx: 0,
                negate: true
            })
        );
        assert!(matches!(
            predicate.op,
            Op::ISetPred {
                cmp: ICmp::Ne,
                dest_p: 0,
                ..
            }
        ));
    }

    #[test]
    fn lop_immediate_honors_complement_flags() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0x3847_0180_0017_0100));
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::ILop {
                a: Value::GprIn(1),
                b: Value::ImmU32(1),
                not_a: true,
                not_b: true,
                ..
            }
        ));
        assert!(t.snapshot_pred_state().is_empty());
    }

    #[test]
    fn lop32i_does_not_treat_immediate_bits_as_predicates() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0x0400_3000_0017_0100));
        assert!(t.snapshot_pred_state().is_empty());
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::ILop { .. }
        ));
    }

    #[test]
    fn lop3_register_decodes_captured_operands_and_all_truth_tables() {
        const CAPTURED: u64 = 0x5be7_060f_e0b7_0d08;
        for lut in 0..=255u64 {
            let raw = (CAPTURED & !(0xff << 28)) | (lut << 28);
            let mut t = Translator::new_fragment();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0);
            assert_eq!(t.program.instructions.len(), 1);
            assert!(matches!(t.program.instructions[0].op,
                    Op::ILop3 {
                        a: Value::GprIn(13), b: Value::GprIn(11),
                        c: Value::GprIn(12), lut: actual,
                    } if u64::from(actual) == lut));
            assert_eq!(t.program.instructions[0].dest_reg, Some(8));
            assert!(t.snapshot_pred_state().is_empty());
        }
    }

    #[test]
    fn tlds_offsets_are_signed_nibbles_and_snapshot_aliased_coordinates() {
        let raw = 0xda8c_008f_f077_0000;
        for mut t in [Translator::new_compute(), Translator::new()] {
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            let mut offsets = Vec::new();
            for inst in &t.program.instructions {
                if let Op::Bfe { a, b: Value::ImmU32(control), signed } = inst.op {
                    if a == Value::GprIn(7) {
                        offsets.push((control & 255, (control >> 8) & 255, signed));
                    }
                }
            }
            assert_eq!(offsets, vec![(0, 4, true), (4, 4, true)]);
            let fetch = t.program.instructions.iter().find(|inst| matches!(inst.op, Op::TexelFetchHandle { .. })).unwrap();
            let Op::TexelFetchHandle { dimension, x: Value::Inst(x), y: Some(Value::Inst(y)), component, .. } = fetch.op else { panic!("expected offset fetch") };
            assert_eq!(dimension, ImageDimension::D2);
            assert_eq!(component, 3);
            for (id, reg) in [(x, 0), (y, 1)] {
                assert!(t.program.instructions.iter().any(|inst| inst.result == Some(id)
                    && matches!(inst.op, Op::IAdd { a: Value::GprIn(actual), neg_b: false, .. } if actual == reg)));
            }
        }
    }

    #[test]
    fn lop3_register_predicate_modes_and_guard_survive_rz_destination() {
        for (mode, expected) in [(0, ICmp::F), (1, ICmp::T), (2, ICmp::Eq), (3, ICmp::Ne)] {
            let raw = (0x5be7_060f_e0b7_0d08u64 & !((7 << 48) | (3 << 36) | (15 << 16) | 0xff))
                | ((mode as u64) << 36)
                | (8 << 16)
                | 0xff;
            let mut t = Translator::new_fragment();
            assert!(t.translate(raw));
            let logical = t
                .program
                .instructions
                .iter()
                .find(|inst| matches!(inst.op, Op::ILop3 { .. }))
                .unwrap();
            let predicate = t.program.instructions.last().unwrap();
            assert_eq!(
                predicate.pred,
                Some(Predicate {
                    idx: 0,
                    negate: true
                })
            );
            assert!(matches!(predicate.op, Op::ISetPred {
                    cmp, src_a: Value::Inst(result), dest_p: 0, dest_np: PT, ..
                } if cmp == expected && Some(result) == logical.result));
        }
        for flag in [38, 47] {
            let mut t = Translator::new_fragment();
            assert!(!t.translate(0x5be7_060f_e0b7_0d08 | (1 << flag)));
            assert_eq!(t.unimplemented_count, 1);
        }
    }

    #[test]
    fn lop3_constant_buffer_uses_lut48_without_predicate_output() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0x02f8_0100_0027_0302));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::LoadCbuf { .. })));
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::ILop3 {
                a: Value::GprIn(3),
                b: Value::Inst(_),
                c: Value::GprIn(2),
                lut: 0xf8
            }
        ));
        assert!(t.snapshot_pred_state().is_empty());
    }

    #[test]
    fn lop3_imm_preserves_captured_truth_table_and_aliasing() {
        for (raw, dest, a, c) in [
            (0x3cf8_0100_0027_0302, 2, 3, 2),
            (0x3cf8_0200_0027_0505, 5, 5, 4),
            (0x3cf8_0000_0027_0100, 0, 1, 0),
        ] {
            let mut t = Translator::new_compute();
            assert!(t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 0, "raw={raw:#018x}");
            assert_eq!(t.program.instructions.len(), 1, "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions[0].op,
                Op::ILop3 {
                    a: Value::GprIn(actual_a),
                    b: Value::ImmU32(2),
                    c: Value::GprIn(actual_c),
                    lut: 0xf8,
                } if actual_a == a && actual_c == c
            ));
            assert_eq!(t.program.instructions[0].dest_reg, Some(dest));
        }
    }

    #[test]
    fn lop3_imm_accepts_all_luts_but_cc_fails_closed() {
        let base = 0x3cf4_0000_0017_0100u64;
        let mut generic_lut = Translator::new_compute();
        assert!(generic_lut.translate(base ^ (1 << 48)));
        assert!(matches!(
            generic_lut.program.instructions[0].op,
            Op::ILop3 { lut: 0xf5, .. }
        ));

        let raw = base | (1 << 47);
        let mut cc = Translator::new_compute();
        assert!(!cc.translate(raw));
        assert_eq!(cc.unimplemented_count, 1);
        assert!(matches!(
            cc.program.instructions.last().map(|inst| &inst.op),
            Some(Op::Unimplemented {
                opcode: Opcode::LOP3_imm,
                raw: failed,
            }) if *failed == raw
        ));
    }

    #[test]
    fn pps_tex_b_ll_uses_register_after_bindless_handle() {
        for raw in [
            0xdeb8_0060_a047_0a04,
            0xdeb8_0060_a107_0e0e,
            0xdeb8_0060_a127_0808,
            0xdeb8_0060_a0c7_0607,
        ] {
            let t = translate_tld(raw, 2, 0x1a0);
            let samples = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match &inst.op {
                    Op::SampleTex {
                        implicit_lod,
                        explicit_lod,
                        ..
                    } => Some((*implicit_lod, *explicit_lod)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!samples.is_empty(), "raw={raw:#018x}");
            assert!(samples.iter().all(|(implicit_lod, explicit_lod)| {
                !implicit_lod && *explicit_lod == Some(Value::GprIn(reg_b(raw).wrapping_add(1)))
            }));
        }
    }

    #[test]
    fn pps_tex_b_lb_uses_register_after_bindless_handle_as_bias() {
        for raw in [
            0xdeba_0044_2087_0600,
            0xdeba_0040_a087_0a08,
            0xdeb8_0041_a0c7_0606,
        ] {
            let t = translate_tld(raw, 2, 0x1a0);
            let samples = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match &inst.op {
                    Op::SampleTex {
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        ..
                    } => Some((*implicit_lod, *lod_bias, *explicit_lod)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!samples.is_empty(), "raw={raw:#018x}");
            assert!(samples
                .iter()
                .all(|(implicit_lod, lod_bias, explicit_lod)| {
                    *implicit_lod
                        && *lod_bias == Some(Value::GprIn(reg_b(raw).wrapping_add(1)))
                        && explicit_lod.is_none()
                }));
        }
    }

    #[test]
    fn pps_tex_b_none_is_implicit_and_lz_is_explicit_zero() {
        let none = 0xdeba_0007_a0e7_0400u64;
        let implicit = translate_tld(none, 2, 0x1a0);
        assert!(implicit.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                implicit_lod: true,
                explicit_lod: None,
                ..
            }
        )));

        let lz = none | (1 << 37);
        let explicit = translate_tld(lz, 2, 0x1a0);
        assert!(explicit.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                implicit_lod: false,
                explicit_lod: Some(Value::Zero),
                ..
            }
        )));
    }

    #[test]
    fn direct_tex_ll_uses_meta_register_without_bindless_increment() {
        let raw = direct_tex(3, 2, 1);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEX)
        );
        let mut t = Translator::new();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                implicit_lod: false,
                explicit_lod: Some(Value::GprIn(6)),
                ..
            }
        )));
    }

    #[test]
    fn direct_tex_2d_array_color_consumes_layer_before_xy() {
        let raw = direct_tex(0, 3, 1);
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEX)
        );

        let mut t = Translator::new_fragment();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                sample_site: Some(_),
                tex_id: 0x24,
                u: Value::GprIn(5),
                v: Value::GprIn(6),
                array: Some(Value::GprIn(4)),
                volume: None,
                cube: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                dref: None,
                component: 0,
            }
        )));
    }

    #[test]
    fn direct_tex_2d_array_shadow_consumes_layer_before_xy_and_dref() {
        let raw = 0xc03e_00c0_b077_0406u64;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::TEX)
        );

        let mut t = Translator::new_fragment();
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.program.instructions.len(), 1);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst,
            Inst {
                op: Op::SampleTex {
                    sample_site: Some(_),
                    tex_id: 12,
                    u: Value::GprIn(5),
                    v: Value::GprIn(6),
                    array: Some(Value::GprIn(4)),
                    volume: None,
                    cube: None,
                    implicit_lod: true,
                    lod_bias: None,
                    explicit_lod: None,
                    texel_offset: None,
                    dref: Some(Value::GprIn(7)),
                    component: 0,
                },
                dest_reg: Some(6),
                ..
            }
        )));
    }

    #[test]
    fn tex_1d_and_3d_safe_forms_use_exact_coordinates() {
        let one_d = direct_tex(0, 0, 1);
        let mut t = Translator::new();
        assert!(t.translate(one_d));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                u: Value::GprIn(4),
                v: Value::Zero,
                volume: None,
                ..
            }
        )));

        let three_d = direct_tex(1, 4, 1);
        let mut t = Translator::new();
        assert!(t.translate(three_d));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                u: Value::GprIn(4),
                v: Value::GprIn(5),
                volume: Some(Value::GprIn(6)),
                explicit_lod: Some(Value::Zero),
                ..
            }
        )));
    }

    #[test]
    fn pps_tex_b_cube_uses_xyz_and_all_observed_lod_modes() {
        for (raw, implicit_lod, lod_bias, explicit_lod) in [
            (0xdeba_0003_e037_0404, true, None, None),
            (0xdeba_0027_e030_1000, false, None, Some(Value::Zero)),
            (
                0xdeba_0040_e167_082e,
                true,
                Some(Value::GprIn(reg_b(0xdeba_0040_e167_082e).wrapping_add(1))),
                None,
            ),
            (
                0xdeba_0067_e067_0808,
                false,
                None,
                Some(Value::GprIn(reg_b(0xdeba_0067_e067_0808).wrapping_add(1))),
            ),
        ] {
            let coord = reg_a(raw);
            let t = translate_tld(raw, 2, 0x1a0);
            let samples = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::SampleTex {
                        u,
                        v,
                        array,
                        volume,
                        cube,
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                        ..
                    } => Some((
                        u,
                        v,
                        array,
                        volume,
                        cube,
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!samples.is_empty(), "raw={raw:#018x}");
            assert!(samples.iter().all(|sample| {
                *sample
                    == (
                        Value::GprIn(coord),
                        Value::GprIn(coord.wrapping_add(1)),
                        None,
                        None,
                        Some(Value::GprIn(coord.wrapping_add(2))),
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        None,
                    )
            }));
        }
    }

    #[test]
    fn pps_tex_b_cube_array_ll_uses_layer_xyz_and_meta_plus_one() {
        for raw in [
            0xdeb8_0067_f107_1414,
            0xdeb8_0067_f147_1010,
            0xdeb8_0067_f047_0000,
            0xdeb8_0067_f207_1010,
        ] {
            let coord = reg_a(raw);
            let meta = reg_b(raw);
            let t = translate_tld(raw, 2, 0x1a0);
            let samples = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::SampleTex {
                        u,
                        v,
                        array,
                        volume,
                        cube,
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                        ..
                    } => Some((
                        u,
                        v,
                        array,
                        volume,
                        cube,
                        implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                    )),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!samples.is_empty(), "raw={raw:#018x}");
            assert!(samples.iter().all(|sample| {
                *sample
                    == (
                        Value::GprIn(coord.wrapping_add(1)),
                        Value::GprIn(coord.wrapping_add(2)),
                        Some(Value::GprIn(coord)),
                        None,
                        Some(Value::GprIn(coord.wrapping_add(3))),
                        false,
                        None,
                        Some(Value::GprIn(meta.wrapping_add(1))),
                        None,
                    )
            }));
        }
    }

    #[test]
    fn cube_and_cube_array_offsets_fail_closed() {
        for raw in [
            0xdeba_0003_e037_0404 | (1 << 36),
            direct_tex(0, 6, 1) | (1 << 54),
            0xdeb8_0067_f147_1010 | (1 << 36),
        ] {
            let mut t = Translator::new();
            if decode_one(raw).is_some_and(|decoded| decoded.opcode == Opcode::TEX_b) {
                assert!(t.translate(static_ldc(reg_b(raw), 2, 0x1a0)));
            }
            assert!(!t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "raw={raw:#018x}");
        }
    }

    #[test]
    fn pps_tex_b_cube_depth_compare_uses_dref_after_handle() {
        let raw = 0xdebe_0000_e0e7_080a;
        assert_eq!(
            decode_texture_sample_form(raw, true).map(|form| form.dref_reg),
            Some(Some(15))
        );
        let mut t = Translator::new();
        assert!(t.translate(static_ldc(reg_b(raw), 2, 0x570)));
        assert_eq!(
            t.trace_cbuf_handle_origin(&t.read_reg(14), None, None),
            Some((
                CbufHandleOrigin {
                    binding: 2,
                    word_offset: 0x15c,
                    secondary_word_offset: None,
                    secondary_binding: None,
                },
                false,
            ))
        );
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst,
            Inst {
                op: Op::SampleTex {
                    sample_site: Some(_),
                    tex_id,
                    u: Value::GprIn(8),
                    v: Value::GprIn(9),
                    array: None,
                    volume: None,
                    cube: Some(Value::GprIn(10)),
                    implicit_lod: true,
                    lod_bias: None,
                    explicit_lod: None,
                    texel_offset: None,
                    dref: Some(Value::GprIn(15)),
                    component: 0,
                },
                dest_reg: Some(10),
                ..
            } if *tex_id == crate::bindless_texture_id(2, 0x15c)
        )));
    }

    #[test]
    fn tex_b_handle_from_high_cbuf_bank_is_tagged() {
        let raw = 0xdeba_0003_e037_0404;
        let t = translate_tld(raw, 17, 0x1a0);
        let texture_ids = t
            .program
            .instructions
            .iter()
            .filter_map(|inst| match inst.op {
                Op::SampleTex { tex_id, .. } => Some(tex_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert!(!texture_ids.is_empty());
        assert!(texture_ids.iter().all(|texture_id| {
            crate::decode_bindless_texture_id(*texture_id) == Some((17, 0x68, None))
        }));
    }

    #[test]
    fn pps_tex_b_aoffi_extracts_signed_xy_after_bindless_handle() {
        for (raw, offset_mov, expected_x) in [
            (0xdeba_0033_a1e7_180c, 0x0100_0000_0017_f01f, 1),
            (0xdeba_0033_a047_1804, 0x010f_ffff_f0f7_f005, u32::MAX),
        ] {
            let mut t = Translator::new();
            assert!(t.translate(static_ldc(reg_b(raw), 2, 0x1a0)));
            assert!(t.translate(offset_mov));
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::SampleTex {
                    implicit_lod: false,
                    explicit_lod: Some(Value::Zero),
                    texel_offset: Some((Value::ImmU32(x), Value::ImmU32(0), Value::ImmU32(0))),
                    ..
                } if x == expected_x
            )));
        }
    }

    #[test]
    fn tex_b_dynamic_aoffi_fails_closed() {
        let raw = 0xdeba_0033_a1e7_180c;
        let mut t = Translator::new();
        assert!(t.translate(static_ldc(reg_b(raw), 2, 0x1a0)));
        assert!(!t.translate(raw));
        assert_eq!(t.unimplemented_count, 1);
    }

    #[test]
    fn direct_tex_1d_aoffi_ignores_packed_y_offset() {
        let raw = direct_tex(1, 0, 1) | (1 << 54);
        let mut t = Translator::new();
        t.write_reg(6, Op::Mov(Value::ImmU32(0x71)), None);
        assert!(t.translate(raw));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                texel_offset: Some((Value::ImmU32(1), Value::ImmU32(0), Value::ImmU32(0))),
                ..
            }
        )));
    }

    #[test]
    fn botw_volume_sample_preserves_signed_xyz_offsets() {
        let mut t = Translator::new_fragment();
        t.write_reg(13, Op::Mov(Value::ImmU32(0xf21)), None);
        assert!(t.translate(0xc1f8_0080_c0c7_0003));
        assert_eq!(t.unimplemented_count, 0);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SampleTex {
                u: Value::GprIn(0),
                v: Value::GprIn(1),
                volume: Some(Value::GprIn(2)),
                explicit_lod: Some(Value::GprIn(12)),
                texel_offset: Some((Value::ImmU32(1), Value::ImmU32(2), Value::ImmU32(u32::MAX))),
                ..
            }
        )));
    }

    #[test]
    fn botw_compute_shadow_and_gather_preserve_operands() {
        let mut shadow = Translator::new_compute();
        assert!(shadow.translate(0xd922_020f_f047_0202));
        assert_eq!(shadow.unimplemented_count, 0);
        assert!(shadow.program.instructions.iter().any(|inst| matches!(inst.op,
            Op::SampleTexHandle {
                dimension: ImageDimension::D2Array,
                u: Value::GprIn(3), v: Some(Value::GprIn(4)),
                w: Some(Value::GprIn(2)), dref: Some(Value::GprIn(5)),
                implicit_lod: false, explicit_lod: Some(Value::Zero), ..
            })));
        let mut gather = Translator::new_compute();
        assert!(gather.translate(0xc838_0146_aff7_0220));
        assert_eq!(gather.unimplemented_count, 0);
        assert_eq!(gather.program.instructions.iter().filter(|inst| matches!(inst.op,
            Op::GatherTex { tex_id: 0x14, u: Value::GprIn(2), v: Value::GprIn(3),
                array: None, gather_component: 0, .. })).count(), 3);
    }

    #[test]
    fn botw_array_gather_preserves_layer_and_coordinates() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0xc83a_0087_bff7_0400));
        assert_eq!(t.unimplemented_count, 0);
        let gathers: Vec<_> = t.program.instructions.iter().filter(|inst| {
            matches!(inst.op, Op::GatherTex { .. })
        }).collect();
        assert_eq!(gathers.len(), 4);
        for (lane, inst) in gathers.into_iter().enumerate() {
            assert!(matches!(inst.op, Op::GatherTex {
                u: Value::GprIn(5),
                v: Value::GprIn(6),
                array: Some(Value::GprIn(4)),
                lane: actual,
                ..
            } if actual == lane as u8));
        }
    }

    #[test]
    fn direct_tex_3d_dynamic_aoffi_fails_closed() {
        let raw = direct_tex(1, 4, 1) | (1 << 54);
        let mut t = Translator::new();
        assert!(!t.translate(raw));
        assert_eq!(t.unimplemented_count, 1);
    }

    #[test]
    fn tex_and_tex_b_unobserved_modes_fail_closed() {
        let bindless = 0xdeba_0007_a0e7_0400u64;
        let direct = direct_tex(0, 2, 1);
        let mut unsupported_bindless = vec![
            bindless | (1 << 40),
            bindless | (1 << 35),
            bindless & !(0x7 << 51),
            bindless & !(0xF << 31),
        ];
        let mut unsupported_direct = vec![
            direct | (1 << 58),
            direct | (1 << 35),
            direct & !(0x7 << 51),
            direct & !(0xF << 31),
        ];
        for tex_type in [1u64, 5] {
            unsupported_bindless.push((bindless & !(0x7 << 28)) | (tex_type << 28));
            unsupported_direct.push((direct & !(0x7 << 28)) | (tex_type << 28));
        }
        for blod in [4u64, 5, 6, 7] {
            unsupported_bindless.push((bindless & !(0x7 << 37)) | (blod << 37));
            unsupported_direct.push((direct & !(0x7 << 55)) | (blod << 55));
        }
        for (opcode, raws) in [
            (Opcode::TEX_b, unsupported_bindless),
            (Opcode::TEX, unsupported_direct),
        ] {
            for raw in raws {
                assert_eq!(
                    decode_one(raw).map(|decoded| decoded.opcode),
                    Some(opcode),
                    "raw={raw:#018x}"
                );
                let mut t = Translator::new();
                assert!(!t.translate(raw), "raw={raw:#018x}");
                assert_eq!(t.unimplemented_count, 1, "raw={raw:#018x}");
                assert!(matches!(
                    t.program.instructions.last().map(|inst| &inst.op),
                    Some(Op::Unimplemented {
                        opcode: failed_opcode,
                        raw: failed,
                    }) if *failed_opcode == opcode && *failed == raw
                ));
            }
        }
    }

    #[test]
    fn pps_tex_b_traces_matching_predicated_handle() {
        let mut t = Translator::new();
        for raw in [
            0x4c98_0788_0683_0012,
            0x4c47_0208_1683_1212,
            0xdeba_0007_a123_1010,
        ] {
            assert!(t.translate(raw));
        }

        let tex_ids = t
            .program
            .instructions
            .iter()
            .filter_map(|inst| match inst.op {
                Op::SampleTex { tex_id, .. } => Some(tex_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let tex_id = bindless_texture_id_pair(2, 0x68, Some(0x168));
        assert_eq!(tex_ids, vec![tex_id; 4]);
        assert_eq!(
            crate::decode_bindless_texture_id(tex_id),
            Some((2, 0x68, Some(0x168)))
        );
        assert_eq!(t.unimplemented_count, 0);
        assert!(!t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::Unimplemented { .. })));
    }

    #[test]
    fn dusk_tex_b_traces_masked_cross_cbuf_texture_sampler_handle() {
        let mut t = Translator::new_fragment();
        for raw in [
            0x4c98_0788_0007_0000,
            0x4c98_0784_0007_0001,
            0x0400_00ff_fff7_0000,
            0x040f_ff00_0007_0101,
            0x5c47_0200_0017_0000,
            0xdeba_0003_a007_0204,
        ] {
            assert!(t.translate(raw), "raw={raw:#018x}");
        }

        let texture_id = crate::bindless_texture_id(2, 0);
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(
            t.bindless_or_partners.get(&texture_id),
            Some(&crate::bindless_texture_id(1, 0))
        );
        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::SampleTex { tex_id, .. } => Some(tex_id),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![texture_id; 3]
        );
    }

    #[test]
    fn pps_tex_b_predicate_mismatch_fails_closed() {
        let mut t = Translator::new();
        assert!(t.translate(0x4c98_0788_0683_0012));
        assert!(t.translate(0x4c47_0208_1683_1212));
        assert!(!t.translate(0xdeba_0007_a122_1010));

        assert_eq!(t.unimplemented_count, 1);
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::Unimplemented {
                opcode: Opcode::TEX_b,
                ..
            }
        )));
    }

    #[test]
    fn pps_r2p_immediates_update_masked_predicates() {
        for (raw, src_reg) in [(0x38f0_0000_0607_1400, 20), (0x38f0_0000_0607_0f00, 15)] {
            let mut t = Translator::new();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);

            let extracts: Vec<(u8, u32)> = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::Bfe {
                        a: Value::GprIn(reg),
                        b: Value::ImmU32(selector),
                        signed: false,
                    } => Some((reg, selector)),
                    _ => None,
                })
                .collect();
            assert_eq!(extracts, vec![(src_reg, 0x105), (src_reg, 0x106)]);

            let destinations: Vec<u8> = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::ISetPred {
                        cmp: ICmp::Ne,
                        dest_p,
                        dest_np: PT,
                        ..
                    } => Some(dest_p),
                    _ => None,
                })
                .collect();
            assert_eq!(destinations, vec![5, 6]);
        }
    }

    #[test]
    fn pps_fcmp_immediates_flush_and_select() {
        for (raw, dest, operand_reg) in [
            (0x36a9_86bf_8007_2222, 34, 13),
            (0x36a9_863f_8007_2121, 33, 12),
            (0x36a9_90bf_8007_2222, 34, 33),
            (0x36a9_86bf_8007_2323, 35, 13),
        ] {
            let mut t = Translator::new();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            assert_eq!(t.program.instructions.last().unwrap().dest_reg, Some(dest));
            assert!(matches!(
                t.program.instructions.last().unwrap().op,
                Op::Mov(Value::Inst(_))
            ));

            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::ILop {
                    a: Value::GprIn(reg),
                    b: Value::ImmU32(0x7f80_0000),
                    op: LogicOp::And,
                    not_a: false,
                    not_b: false,
                } if reg == operand_reg
            )));
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::FSet {
                    cmp: FComp::Ltu,
                    bop: BoolOp::And,
                    src_a: Value::Inst(_),
                    src_b: Value::Zero,
                    bf: false,
                    src_pred: PT,
                    src_pred_inv: false,
                    ..
                }
            )));
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::ILop {
                    b: Value::ImmU32(0x3f80_0000),
                    op: LogicOp::And,
                    not_a: true,
                    not_b: false,
                    ..
                }
            )));
        }
    }

    #[test]
    fn pps_b64_fixed_local_store_and_load_expand_to_two_words() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0xef55_0000_0007_ff24));
        assert!(t.translate(0xef45_1000_0007_ff24));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::StoreLocal {
                addr: Value::ImmU32(0),
                value: Value::GprIn(36),
            }
        ));
        assert!(t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::StoreLocal {
                addr: Value::Inst(_),
                value: Value::GprIn(37),
            }
        )));
        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter(|inst| matches!(inst.op, Op::LoadLocal { .. }))
                .count(),
            2
        );
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::Mov(Value::Inst(_))) && inst.dest_reg == Some(36)));
        assert!(t
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::Mov(Value::Inst(_))) && inst.dest_reg == Some(37)));
    }

    #[test]
    fn local_memory_dynamic_address_and_unaligned_b64_lower_generically() {
        for raw in [0xef55_0000_0007_0124, 0xef45_1000_0007_ff25] {
            let mut t = Translator::new_fragment();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            assert!(t
                .program
                .instructions
                .iter()
                .any(|inst| matches!(inst.op, Op::LoadLocal { .. } | Op::StoreLocal { .. })));
        }
    }

    #[test]
    fn compute_shared_b32_and_b128_expand_to_word_ir() {
        let mut t = Translator::new_compute();
        assert!(t.translate(0xef5c_0000_0007_010c));
        assert!(t.translate(0xef4c_1000_1407_0105));
        assert!(t.translate(0xef5e_0000_0007_0d08));
        assert!(t.translate(0xef4e_1000_8007_0d00));
        assert_eq!(t.unimplemented_count, 0);

        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter(|inst| matches!(inst.op, Op::StoreShared { .. }))
                .count(),
            5
        );
        assert_eq!(
            t.program
                .instructions
                .iter()
                .filter(|inst| matches!(inst.op, Op::LoadShared { .. }))
                .count(),
            5
        );
        for register in 0..4 {
            assert!(t.program.instructions.iter().any(|inst| {
                inst.dest_reg == Some(register) && matches!(inst.op, Op::Mov(Value::Inst(_)))
            }));
        }
        for register in 8..12 {
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::StoreShared {
                    value: Value::GprIn(source),
                    ..
                } if source == register
            )));
        }
    }

    #[test]
    fn compute_shared_b64_preserves_both_words_and_offsets() {
        for (raw, load) in [
            (0xef4d_1000_4287_ff0c, true),
            (0xef5d_1000_4287_ff0c, false),
        ] {
            let mut t = Translator::new_compute();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            let memory_ops: Vec<_> = t
                .program
                .instructions
                .iter()
                .filter_map(|inst| match inst.op {
                    Op::LoadShared { addr } if load => Some(addr),
                    Op::StoreShared { addr, .. } if !load => Some(addr),
                    _ => None,
                })
                .collect();
            assert_eq!(memory_ops.len(), 2);
            assert_eq!(memory_ops[0], Value::ImmU32(0x428));
            assert!(t.program.instructions.iter().any(|inst| matches!(
                inst.op,
                Op::IAdd {
                    a: Value::ImmU32(0x428),
                    b: Value::ImmU32(4),
                    ..
                }
            )));
            if load {
                for reg in [12, 13] {
                    assert!(t
                        .program
                        .instructions
                        .iter()
                        .any(|inst| inst.dest_reg == Some(reg)));
                }
            } else {
                for reg in [12, 13] {
                    assert!(t.program.instructions.iter().any(|inst| matches!(inst.op,
                            Op::StoreShared { value: Value::GprIn(source), .. } if source == reg)));
                }
            }
        }
    }

    #[test]
    fn compute_shared_rejects_unknown_widths() {
        let unsupported = 0xef5e_0000_0007_0d08 | (1u64 << 48);
        let mut t = Translator::new_compute();
        assert!(!t.translate(unsupported));
        assert_eq!(t.unimplemented_count, 1);
        assert!(matches!(
            t.program.instructions.last().map(|inst| &inst.op),
            Some(Op::Unimplemented {
                opcode: Opcode::STS,
                ..
            })
        ));
    }

    #[test]
    fn captured_atoms_or_u32_preserves_predication_and_rz_result() {
        const CAPTURED: u64 = 0xec60_0000_0081_ffff;
        assert_eq!(decode_one(CAPTURED).unwrap().opcode, Opcode::ATOMS);

        let mut t = Translator::new_compute();
        assert!(t.translate(CAPTURED));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::SharedAtomic {
                    addr: Value::ImmU32(0),
                    value: Value::GprIn(8),
                    op: ImageAtomicOp::Or,
                },
                dest_reg: Some(RZ),
                pred: Some(Predicate {
                    idx: 1,
                    negate: false,
                }),
                ..
            })
        ));
    }

    #[test]
    fn predicated_atoms_non_rz_selects_the_old_register_value() {
        const CAPTURED: u64 = 0xec60_0000_0081_ffff;
        const DEST: u8 = 3;
        let raw = (CAPTURED & !0xff) | u64::from(DEST);

        let mut t = Translator::new_compute();
        let old = t.write_reg(DEST, Op::Mov(Value::ImmU32(0x1234_5678)), None);
        assert!(t.translate(raw));
        assert_eq!(t.unimplemented_count, 0);

        let atomic_index = t
            .program
            .instructions
            .iter()
            .position(|inst| matches!(inst.op, Op::SharedAtomic { .. }))
            .expect("shared atomic");
        let atomic = &t.program.instructions[atomic_index];
        let atomic_result = atomic.result.expect("atomic old value");
        assert_eq!(atomic.dest_reg, None);
        assert_eq!(
            atomic.pred,
            Some(Predicate {
                idx: 1,
                negate: false,
            })
        );
        assert!(matches!(
            t.program.instructions.get(atomic_index + 1),
            Some(Inst {
                op: Op::SelectPred {
                    pred: Predicate {
                        idx: 1,
                        negate: false,
                    },
                    if_true: Value::Inst(result),
                    if_false: Value::Inst(previous),
                },
                dest_reg: Some(DEST),
                pred: None,
                ..
            }) if *result == atomic_result && *previous == old
        ));
    }

    #[test]
    fn atoms_or_u32_decodes_absolute_and_signed_relative_offsets() {
        const TEMPLATE: u64 = 0xec60_0000_0081_ffff;

        let absolute = TEMPLATE | (0x123u64 << 30);
        let mut absolute_t = Translator::new_compute();
        assert!(absolute_t.translate(absolute));
        assert!(absolute_t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::SharedAtomic {
                addr: Value::ImmU32(0x48c),
                ..
            }
        )));

        let negative_words = (1u64 << 22) - 2;
        let relative = (TEMPLATE & !((0x003f_ffffu64 << 30) | (0xffu64 << 8)))
            | (negative_words << 30)
            | (4u64 << 8);
        let mut relative_t = Translator::new_compute();
        assert!(relative_t.translate(relative));
        assert!(relative_t.program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::IAdd {
                a: Value::GprIn(4),
                b: Value::ImmU32(0xffff_fff8),
                ..
            }
        )));
    }

    #[test]
    fn atoms_non_or_or_non_u32_forms_fail_closed() {
        const CAPTURED: u64 = 0xec60_0000_0081_ffff;
        for raw in [
            (CAPTURED & !(0xfu64 << 52)) | (5u64 << 52),
            CAPTURED | (1u64 << 28),
        ] {
            let mut t = Translator::new_compute();
            assert!(!t.translate(raw));
            assert_eq!(t.unimplemented_count, 1);
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented {
                    opcode: Opcode::ATOMS,
                    ..
                })
            ));
        }
    }

    #[test]
    fn captured_smo_iadd32i_minus_one_uses_immediate_bits_only_as_data() {
        const CAPTURED: u64 = 0x1c0f_ffff_fff7_2222;

        let mut t = Translator::new();
        assert!(t.translate(CAPTURED));
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::IAdd {
                    a: Value::GprIn(0x22),
                    b: Value::ImmU32(u32::MAX),
                    neg_a: false,
                    neg_b: false,
                },
                dest_reg: Some(0x22),
                ..
            })
        ));
    }

    #[test]
    fn iadd32i_po_adds_one_without_treating_po_as_negate_a() {
        const CAPTURED: u64 = 0x1c0f_ffff_fff7_2222;
        let po = (CAPTURED & !(3u64 << 55)) | (3u64 << 55);

        let mut t = Translator::new();
        assert!(t.translate(po));
        let first = &t.program.instructions[t.program.instructions.len() - 2];
        let first_result = first.result.expect("IADD32I base sum");
        assert!(matches!(
            first.op,
            Op::IAdd {
                a: Value::GprIn(0x22),
                b: Value::ImmU32(u32::MAX),
                neg_a: false,
                neg_b: false,
            }
        ));
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::IAdd {
                    a: Value::Inst(id),
                    b: Value::ImmU32(1),
                    neg_a: false,
                    neg_b: false,
                },
                dest_reg: Some(0x22),
                ..
            }) if *id == first_result
        ));
    }

    #[test]
    fn iadd32i_unknown_carry_and_saturation_fail_closed() {
        const CAPTURED: u64 = 0x1c0f_ffff_fff7_2222;
        for flag in [53, 54] {
            let mut t = Translator::new();
            let raw = CAPTURED | (1u64 << flag);
            assert!(!t.translate(raw));
            assert_eq!(t.unimplemented_count, 1);
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented {
                    opcode: Opcode::IADD32I,
                    raw: actual,
                }) if *actual == raw
            ));
        }
    }

    #[test]
    fn captured_integer_address_add_propagates_carry_across_u32_boundary() {
        fn value(value: Value, results: &HashMap<ValueId, u32>) -> u32 {
            match value {
                Value::Zero => 0,
                Value::ImmU32(word) => word,
                Value::Inst(id) => results[&id],
                _ => panic!("unexpected value {value:?}"),
            }
        }
        for low in [0, 0xffff_ffe3, 0xffff_ffe4, u32::MAX] {
            let mut t = Translator::new_fragment();
            t.reg_state.insert(4, Value::ImmU32(low));
            t.reg_state.insert(3, Value::ImmU32(9));
            assert!(t.translate(0x1c10_0000_01c7_0404));
            assert!(t.translate(0x5c10_0800_0037_ff05));
            let mut results = HashMap::new();
            for inst in &t.program.instructions {
                let result = match inst.op {
                    Op::IAdd { a, b, neg_a, neg_b } => {
                        let a = value(a, &results);
                        let b = value(b, &results);
                        (if neg_a { a.wrapping_neg() } else { a }).wrapping_add(if neg_b {
                            b.wrapping_neg()
                        } else {
                            b
                        })
                    }
                    Op::ISet {
                        a,
                        b,
                        cmp: ICmp::Lt,
                        signed: false,
                        ..
                    } => {
                        if value(a, &results) < value(b, &results) {
                            u32::MAX
                        } else {
                            0
                        }
                    }
                    Op::Bfe {
                        a,
                        b,
                        signed: false,
                    } => {
                        let control = value(b, &results);
                        (value(a, &results) >> (control & 0xff)) & ((1 << ((control >> 8) & 0xff)) - 1)
                    }
                    _ => panic!("unexpected operation {:?}", inst.op),
                };
                results.insert(inst.result.unwrap(), result);
            }
            let actual = (u64::from(value(t.reg_state[&5], &results)) << 32)
                | u64::from(value(t.reg_state[&4], &results));
            assert_eq!(actual, ((9u64 << 32) | u64::from(low)) + 0x1c);
        }
    }

    #[test]
    fn captured_global_atomic_keeps_guard_result_and_operand_aliasing() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0xed01_0000_0058_0205));
        assert_eq!(t.unimplemented_count, 0);
        let atomic = &t.program.instructions[0];
        assert_eq!(
            atomic.pred,
            Some(Predicate {
                idx: 0,
                negate: true
            })
        );
        assert!(matches!(
            atomic.op,
            Op::GlobalAtomic {
                addr_lo: Value::GprIn(2),
                offset: 0,
                value: Value::GprIn(5),
                op: ImageAtomicOp::Add,
                is_signed: false,
            }
        ));
        assert!(matches!(t.program.instructions[1].op, Op::SelectPred {
                if_true: Value::Inst(id), if_false: Value::GprIn(5), ..
            } if Some(id) == atomic.result));
    }

    #[test]
    fn scaled_and_predicated_pointer_adds_keep_compatible_carry() {
        for pair in [
            [0x4c18_8200_0500_0404, 0x4c10_0800_0510_ff05],
            [0x4c18_8200_0447_0600, 0x4c10_0800_0457_ff01],
            [0x4c10_8000_1440_0606, 0x4c10_0800_1450_ff07],
        ] {
            let mut t = Translator::new();
            assert!(t.translate(pair[0]));
            assert!(t.carry_source.is_some());
            assert!(t.translate(pair[1]));
            assert_eq!(t.unimplemented_count, 0);
        }
        let mut t = Translator::new();
        assert!(t.translate(0x4c10_8000_1440_0606));
        assert!(!t.translate(0x4c10_0800_1457_ff07));
        let mut t = Translator::new();
        assert!(t.translate(0x4c10_8000_1440_0606));
        t.pred_state.insert(0, ValueId(999));
        assert!(!t.translate(0x4c10_0800_1450_ff07));
    }

    #[test]
    fn captured_vmnmx_lowers_unsigned_word_max_chain() {
        let mut t = Translator::new_fragment();
        assert!(t.translate(0x3b34_0060_6037_0102));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.program.instructions.len(), 2);
        assert!(matches!(
            t.program.instructions[0].op,
            Op::IMinMaxPred {
                a: Value::GprIn(1),
                b: Value::GprIn(3),
                signed: false,
                pred: PT,
                neg_pred: true,
            }
        ));
        assert!(matches!(
            t.program.instructions[1].op,
            Op::IMinMaxPred {
                a: Value::Inst(_),
                b: Value::GprIn(0),
                signed: false,
                pred: PT,
                neg_pred: true,
            }
        ));
        assert_eq!(t.program.instructions[1].dest_reg, Some(2));
    }

    #[test]
    fn captured_smo_i2i_s32_abs_is_not_lowered_as_a_move() {
        const CAPTURED: u64 = 0x5ce2_0000_0057_3a04;

        let mut t = Translator::new();
        assert!(t.translate(CAPTURED));
        assert_eq!(t.program.instructions.len(), 4);

        let sign = t.program.instructions[0].result.expect("I2I sign mask");
        assert!(matches!(
            t.program.instructions[0].op,
            Op::IShr {
                a: Value::GprIn(5),
                b: Value::ImmU32(31),
                signed: true,
            }
        ));
        let flipped = t.program.instructions[1]
            .result
            .expect("I2I sign-flipped value");
        assert!(matches!(
            t.program.instructions[1].op,
            Op::ILop {
                a: Value::GprIn(5),
                b: Value::Inst(id),
                op: LogicOp::Xor,
                not_a: false,
                not_b: false,
            } if id == sign
        ));
        let absolute = t.program.instructions[2]
            .result
            .expect("I2I absolute value");
        assert!(matches!(
            t.program.instructions[2].op,
            Op::IAdd {
                a: Value::Inst(lhs),
                b: Value::Inst(rhs),
                neg_a: false,
                neg_b: true,
            } if lhs == flipped && rhs == sign
        ));
        assert!(matches!(
            t.program.instructions[3],
            Inst {
                op: Op::Mov(Value::Inst(id)),
                dest_reg: Some(4),
                ..
            } if id == absolute
        ));
    }

    #[test]
    fn captured_smo_i2i_applies_neg_after_abs() {
        const CAPTURED: u64 = 0x5ce2_2000_0097_3a09;

        let mut t = Translator::new();
        assert!(t.translate(CAPTURED));
        assert_eq!(t.program.instructions.len(), 5);
        let absolute = t.program.instructions[2]
            .result
            .expect("I2I absolute value");
        let negated = t.program.instructions[3]
            .result
            .expect("I2I negated absolute value");
        assert!(matches!(
            t.program.instructions[3].op,
            Op::IAdd {
                a: Value::Zero,
                b: Value::Inst(id),
                neg_a: false,
                neg_b: true,
            } if id == absolute
        ));
        assert!(matches!(
            t.program.instructions[4],
            Inst {
                op: Op::Mov(Value::Inst(id)),
                dest_reg: Some(9),
                ..
            } if id == negated
        ));
    }

    #[test]
    fn i2i_unproven_formats_selector_sat_and_unsigned_modifiers_fail_closed() {
        const CAPTURED: u64 = 0x5ce2_0000_0057_3a04;
        for raw in [
            CAPTURED | (1u64 << 41),
            (CAPTURED & !(3u64 << 10)) | (1u64 << 10),
            CAPTURED & !(3u64 << 8),
            CAPTURED | (1u64 << 50),
            CAPTURED & !((1u64 << 12) | (1u64 << 13)),
        ] {
            let mut t = Translator::new();
            assert!(!t.translate(raw), "raw={raw:#018x}");
            assert_eq!(t.unimplemented_count, 1, "raw={raw:#018x}");
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::Unimplemented {
                    opcode: Opcode::I2I_reg,
                    raw: actual,
                }) if *actual == raw
            ));
        }
    }

    #[test]
    fn captured_smo_i2i_cc_identity_keeps_translating() {
        const CAPTURED: u64 = 0x5ce0_8000_0017_0aff;

        let mut t = Translator::new();
        assert!(t.translate(CAPTURED));
        assert_eq!(t.unimplemented_count, 0);
        assert_eq!(t.cc_source, Some(Value::GprIn(1)));
        assert!(matches!(
            t.program.instructions.last(),
            Some(Inst {
                op: Op::Mov(Value::GprIn(1)),
                dest_reg: Some(RZ),
                ..
            })
        ));
    }

    #[test]
    fn captured_smo_fs_and_vs_i2i_cc_feed_csetp_neu() {
        const CSETP_NEU_P0: u64 = 0x50a0_0380_0007_0d07;
        for (i2i, source) in [(0x5ce0_8000_0017_0aff, 1), (0x5ce0_8000_0127_3aff, 0x12)] {
            let mut t = Translator::new();
            assert!(t.translate(i2i), "I2I raw={i2i:#018x}");
            assert!(t.translate(CSETP_NEU_P0));
            assert_eq!(t.unimplemented_count, 0);
            assert!(matches!(
                t.program.instructions.last(),
                Some(Inst {
                    op: Op::ISetPred {
                        cmp: ICmp::Ne,
                        signed: false,
                        bop: BoolOp::And,
                        src_a: Value::GprIn(actual),
                        src_b: Value::Zero,
                        src_pred: PT,
                        src_pred_inv: false,
                        dest_p: 0,
                        dest_np: PT,
                    },
                    pred: None,
                    ..
                }) if *actual == source
            ));
        }
    }

    #[test]
    fn botw_reconvergence_setup_preserves_condition_code() {
        let mut t = Translator::new();
        for raw in [0x5ce0800000c73aff, 0xe290000064000000,
                    0x5b5c038000d7130d, 0x5c1200000ff71815,
                    0x50a0038000070d07] {
            assert!(t.translate(raw), "{raw:#x}");
        }
        assert_eq!(t.unimplemented_count, 0);
    }

    #[test]
    fn break_setup_preserves_condition_flags_for_later_csetp() {
        let mut t = Translator::new();
        for raw in [
            0x5ce0_8000_00f7_0aff,
            0xe2a0_0000_b600_0000,
            0x50a0_0380_0007_0d07,
        ] {
            assert!(t.translate(raw), "raw={raw:#018x}");
        }
        assert!(matches!(
            t.program.instructions.last().unwrap().op,
            Op::ISetPred {
                src_a: Value::GprIn(15),
                cmp: ICmp::Ne,
                ..
            }
        ));
    }

    #[test]
    fn double_arithmetic_snapshots_pairs_before_overwriting_aliases() {
        let mut t = Translator::new();
        assert!(t.translate(0x5b70_0200_0027_1402));
        assert_eq!(t.program.instructions.len(), 2);
        for (component, inst) in t.program.instructions.iter().enumerate() {
            assert!(matches!(inst.op, Op::Double {
                op: DoubleOp::Fma,
                a: [Value::GprIn(20), Value::GprIn(21)],
                b: [Value::GprIn(2), Value::GprIn(3)],
                c: [Value::GprIn(4), Value::GprIn(5)],
                component: actual, ..
            } if usize::from(actual) == component));
            assert_eq!(inst.dest_reg, Some(2 + component as u8));
        }
        for raw in [0x5c80_0000_0007_1600u64, 0x5c70_0000_0027_0c12] {
            let mut t = Translator::new();
            assert!(t.translate(raw));
            assert_eq!(t.unimplemented_count, 0);
            assert_eq!(t.program.instructions.len(), 2);
            let mut unsupported = Translator::new();
            assert!(!unsupported.translate(raw | (1 << 47)));
        }
    }

    #[test]
    fn double_conversion_preserves_register_pair_width() {
        let mut widen = Translator::new();
        assert!(widen.translate(0x4ca8_0010_0017_0b00));
        assert_eq!(widen.program.instructions.len(), 3);
        for component in [0, 1] {
            assert!(matches!(widen.program.instructions[component + 1].op,
                Op::Double { op: DoubleOp::FromFloat32, component: actual, .. }
                if usize::from(actual) == component));
        }
        let mut narrow = Translator::new();
        assert!(narrow.translate(0x5ca8_0000_0067_0e01));
        assert_eq!(narrow.program.instructions.len(), 1);
        assert!(matches!(narrow.program.instructions[0].op, Op::Double {
            op: DoubleOp::ToFloat32, a: [Value::GprIn(6), Value::GprIn(7)], component: 0, ..
        }));
        assert_eq!(narrow.program.instructions[0].dest_reg, Some(1));
    }

    #[test]
    fn i2i_cc_source_is_replaced_by_later_integer_add() {
        const I2I_CC: u64 = 0x5ce0_8000_0017_0aff;
        const IADD_CC: u64 = 0x5c10_8000_0037_0404;
        const CSETP_NEU_P0: u64 = 0x50a0_0380_0007_0d07;

        assert!(matches!(
            decode_one(IADD_CC).map(|decoded| decoded.opcode),
            Some(Opcode::IADD_reg)
        ));
        let mut t = Translator::new();
        assert!(t.translate(I2I_CC));
        assert_eq!(t.cc_source, Some(Value::GprIn(1)));
        assert!(t.translate(IADD_CC));
        let result = t.cc_source.unwrap();
        assert_ne!(result, Value::GprIn(1));
        assert!(t.translate(CSETP_NEU_P0));
        assert_eq!(t.unimplemented_count, 0);
        assert!(matches!(
            t.program.instructions.last().map(|inst| &inst.op),
            Some(Op::ISetPred { src_a, cmp: ICmp::Ne, .. }) if *src_a == result
        ));
    }

    #[test]
    fn i2i_cc_source_is_invalidated_by_r2p_cc() {
        const I2I_CC: u64 = 0x5ce0_8000_0017_0aff;
        const R2P_CC: u64 = 0x38f0_0100_0607_1400;
        const CSETP_NEU_P0: u64 = 0x50a0_0380_0007_0d07;

        assert!(matches!(
            decode_one(R2P_CC).map(|decoded| decoded.opcode),
            Some(Opcode::R2P_imm)
        ));
        let mut t = Translator::new();
        assert!(t.translate(I2I_CC));
        assert_eq!(t.cc_source, Some(Value::GprIn(1)));
        assert!(!t.translate(R2P_CC));
        assert_eq!(t.cc_source, None);
        assert!(!t.translate(CSETP_NEU_P0));
        assert_eq!(t.unimplemented_count, 2);
        assert!(matches!(
            t.program.instructions.last().map(|inst| &inst.op),
            Some(Op::Unimplemented {
                opcode: Opcode::CSETP,
                raw: CSETP_NEU_P0,
            })
        ));
    }

    #[test]
    fn captured_smo_f2i_u16_clamps_before_conversion() {
        const CAPTURED: u64 = 0x5cb0_1000_0047_0904;

        let mut t = Translator::new_fragment();
        assert!(t.translate(CAPTURED));
        assert_eq!(t.program.instructions.len(), 3);
        let lower = t.program.instructions[0]
            .result
            .expect("F2I U16 lower clamp");
        assert!(matches!(
            t.program.instructions[0].op,
            Op::FMax {
                a: Value::GprIn(4),
                b: Value::ImmF32(0.0),
                mods: FMods {
                    neg_a: false,
                    abs_a: false,
                    neg_b: false,
                    abs_b: false,
                    neg_c: false,
                    sat: false,
                    scale: 0,
                },
            }
        ));
        let upper = t.program.instructions[1]
            .result
            .expect("F2I U16 upper clamp");
        assert!(matches!(
            t.program.instructions[1].op,
            Op::FMin {
                a: Value::Inst(id),
                b: Value::ImmF32(65535.0),
                mods: FMods {
                    neg_a: false,
                    abs_a: false,
                    neg_b: false,
                    abs_b: false,
                    neg_c: false,
                    sat: false,
                    scale: 0,
                },
            } if id == lower
        ));
        assert!(matches!(
            t.program.instructions[2],
            Inst {
                op: Op::F2I {
                    src: Value::Inst(id),
                    signed: false,
                    round: 0,
                },
                dest_reg: Some(4),
                ..
            } if id == upper
        ));
    }

    #[test]
    fn compute_barrier_accepts_only_the_observed_full_workgroup_sync() {
        const BARRIER: u64 = 0xf0a8_1b80_0007_0000;
        let mut t = Translator::new_compute();
        assert!(t.translate(BARRIER));
        assert!(matches!(
            t.program.instructions.last().map(|inst| &inst.op),
            Some(Op::WorkgroupBarrier)
        ));

        let mut invalid = Translator::new_compute();
        assert!(!invalid.translate(BARRIER ^ (1 << 8)));
        assert_eq!(invalid.unimplemented_count, 1);
    }

    #[test]
    fn compute_membar_preserves_cta_and_device_scopes() {
        const MEMBAR: u64 = 0xef98_0000_0007_0000;
        assert_eq!(decode_one(MEMBAR).unwrap().opcode, Opcode::MEMBAR);
        for (scope_bits, expected) in [
            (0, MemoryBarrierScope::Workgroup),
            (1, MemoryBarrierScope::Device),
            (2, MemoryBarrierScope::Device),
            (3, MemoryBarrierScope::Device),
        ] {
            let mut t = Translator::new_compute();
            assert!(t.translate(MEMBAR | (scope_bits << 8)));
            assert!(matches!(
                t.program.instructions.last().map(|inst| &inst.op),
                Some(Op::MemoryBarrier { scope }) if *scope == expected
            ));
        }

        let mut predicated = Translator::new_compute();
        assert!(!predicated.translate(MEMBAR & !(7 << 16)));
        assert_eq!(predicated.unimplemented_count, 1);
    }
}

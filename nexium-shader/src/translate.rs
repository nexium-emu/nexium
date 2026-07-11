use std::collections::HashMap;

use super::decode::decode_one;
use super::ir::{
    BoolOp, FComp, HalfMerge, HalfPrecision, HalfSwizzle, ICmp, LogicOp, MufuFunc, Op,
    Predicate, Program, Value, ValueId,
};
use super::opcodes::Opcode;
use super::operand::{
    ald_num_elements, attr_slot_ald, attr_slot_ipa, bfe_signed, cbuf, csetp_bop, csetp_bop_pred,
    csetp_flow_test, csetp_neg_bop_pred, decoded_pred, f2f_mods, f2i_rounding, f2i_signed,
    fadd32i_mods, fadd_mods, ffma32i_mods, ffma_mods, float_imm20, fmnmx_mods, fmnmx_neg_pred,
    fmnmx_pred, fmul32i_mods, fmul_mods, fset_abs_a, fset_abs_b, fset_bf, fset_bop, fset_cmp,
    fset_neg_a, fset_neg_b, fset_src_pred, fset_src_pred_inv, fsetp_abs_a, fsetp_abs_b, fsetp_bop,
    fsetp_cmp, fsetp_dest_np, fsetp_dest_p, fsetp_neg_a, fsetp_neg_b, fsetp_src_pred,
    fsetp_src_pred_inv, i2f_abs, i2f_int_format, i2f_neg, i2f_selector, i2f_signed, iadd3_half_a,
    iadd3_half_b, iadd3_half_c, iadd3_neg_a, iadd3_neg_b, iadd3_neg_c, iadd3_shift, iadd_neg_a,
    iadd_neg_b, imm20, imm32, ipa_interpolation_mode, ipa_saturate, iscadd_shift, iset_bf,
    iset_cmp, iset_signed, isetp_bop, isetp_cmp, isetp_dest_np, isetp_dest_p, isetp_signed,
    isetp_src_pred, isetp_src_pred_inv, ldc_ref, ldc_size, ldc_src_reg, ldg_addr_reg, ldg_offset,
    ldg_size, lop32i_not_a, lop32i_not_b, lop32i_op, lop_not_a, lop_not_b, lop_op, mufu_func_bits,
    pset_bool_float, psetp_bop_1, psetp_bop_2, psetp_dest_np, psetp_dest_p, psetp_neg_pred_a,
    psetp_neg_pred_b, psetp_neg_pred_c, psetp_pred_a, psetp_pred_b, psetp_pred_c, reg_a, reg_b,
    reg_c, reg_dest, sel_neg_pred, sel_pred, shr_signed, texs_tex_id, xmad_cr_mrg, xmad_cr_psl,
    xmad_half_a, xmad_imm_src_b, xmad_rc_half_b, xmad_rc_select, xmad_reg_half_b, xmad_reg_mrg,
    xmad_reg_psl, xmad_reg_select, xmad_signed_a, xmad_signed_b, half_bop, half_compare,
    half_dest_np, half_dest_p, half_h_and, half_merge, half_precision, half_src_pred,
    half_src_pred_inv, half_swizzle_a, half_swizzle_b, half_swizzle_c, RZ,
};

const PT: u8 = 7;

pub struct Translator {
    pub program: Program,
    reg_state: HashMap<u8, Value>,

    pred_state: HashMap<u8, ValueId>,
    pub finished: bool,
    pub unimplemented_count: u32,
}

impl Translator {
    pub fn new() -> Self {
        Self::with_offset(0)
    }

    pub fn with_offset(start: u32) -> Self {
        Self::with_initial(HashMap::new(), HashMap::new(), start)
    }

    pub fn with_initial(
        initial: HashMap<u8, Value>,
        initial_pred: HashMap<u8, ValueId>,
        start: u32,
    ) -> Self {
        Self {
            program: Program::with_offset(start),
            reg_state: initial,
            pred_state: initial_pred,
            finished: false,
            unimplemented_count: 0,
        }
    }

    fn read_reg(&self, r: u8) -> Value {
        if r == RZ {
            return Value::Zero;
        }
        self.reg_state.get(&r).copied().unwrap_or(Value::GprIn(r))
    }

    pub fn snapshot_reg_state(&self) -> HashMap<u8, Value> {
        self.reg_state.clone()
    }

    pub fn snapshot_pred_state(&self) -> HashMap<u8, ValueId> {
        self.pred_state.clone()
    }

    fn write_reg(&mut self, r: u8, op: Op, pred: Option<Predicate>) -> ValueId {
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

    fn emit_f2f(&mut self, raw: u64, src: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let m = f2f_mods(raw);
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
        self.write_reg(
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
    }

    fn half_swizzle(bits: u8) -> HalfSwizzle {
        match bits & 3 {
            0 => HalfSwizzle::H1_H0,
            1 => HalfSwizzle::F32,
            2 => HalfSwizzle::H0_H0,
            _ => HalfSwizzle::H1_H1,
        }
    }

    fn half_merge(bits: u8) -> HalfMerge {
        match bits & 3 {
            0 => HalfMerge::H1_H0,
            1 => HalfMerge::F32,
            2 => HalfMerge::MRG_H0,
            _ => HalfMerge::MRG_H1,
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

    fn emit_iadd(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
            dest,
            Op::IAdd {
                a,
                b,
                neg_a: iadd_neg_a(raw),
                neg_b: iadd_neg_b(raw),
            },
            pred,
        );
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

    fn emit_csetp(&mut self, raw: u64, pred: Option<Predicate>) {
        let dest_p = psetp_dest_p(raw);
        let dest_np = psetp_dest_np(raw);
        let op = Op::CSetPred {
            dest_p,
            dest_np,
            flow_test: csetp_flow_test(raw),
            bop_pred: csetp_bop_pred(raw),
            neg_bop_pred: csetp_neg_bop_pred(raw),
            bop: BoolOp::from_bits(csetp_bop(raw)),
        };
        let id = self.program.emit_pred(op, None, pred);
        if dest_p != PT {
            self.pred_state.insert(dest_p, id);
        }
        if dest_np != PT {
            self.pred_state.insert(dest_np, id);
        }
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
        self.write_reg(
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
    }

    fn emit_ilop(
        &mut self,
        raw: u64,
        b: Value,
        op: LogicOp,
        not_a: bool,
        not_b: bool,
        pred: Option<Predicate>,
    ) {
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
        );
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
        let a = self.read_reg(reg_a(raw));
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

    fn emit_f2i(&mut self, raw: u64, src: Value, pred: Option<Predicate>) {
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

    fn emit_iset(&mut self, raw: u64, b: Value, pred: Option<Predicate>) {
        let dest = reg_dest(raw);
        let a = self.read_reg(reg_a(raw));
        self.write_reg(
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
    }

    pub fn translate(&mut self, raw: u64) -> bool {
        if self.finished {
            return true;
        }
        let Some(decoded) = decode_one(raw) else {
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
        match decoded.opcode {
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

            Opcode::FMUL_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                self.write_reg(
                    dest,
                    Op::FMul {
                        a,
                        b,
                        mods: fmul_mods(raw),
                    },
                    pred,
                );
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
                HalfSwizzle::H1_H0,
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
                HalfSwizzle::H1_H0,
                false,
                ((raw >> 56) & 1) != 0,
                false,
                false,
                HalfMerge::H1_H0,
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
                HalfSwizzle::H1_H0,
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
                HalfSwizzle::H1_H0,
                false,
                false,
                false,
                false,
                HalfMerge::H1_H0,
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
                HalfSwizzle::H1_H0,
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
                HalfSwizzle::H1_H0,
                self.read_reg(reg_dest(raw)),
                HalfSwizzle::H1_H0,
                false,
                ((raw >> 52) & 1) != 0,
                HalfMerge::H1_H0,
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
                HalfSwizzle::H1_H0,
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

            Opcode::F2F_reg => {
                let s = self.read_reg(reg_b(raw));
                self.emit_f2f(raw, s, pred);
            }
            Opcode::F2F_cbuf => {
                let cb = self.load_cbuf(raw);
                self.emit_f2f(raw, Value::Inst(cb), pred);
            }
            Opcode::F2F_imm => {
                self.emit_f2f(raw, Value::ImmF32(float_imm20(raw)), pred);
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

            Opcode::SSY | Opcode::SYNC | Opcode::PBK | Opcode::BRK => {}

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

            Opcode::ALD => {
                let base_dest = reg_dest(raw);
                let base_slot = attr_slot_ald(raw);
                let n = ald_num_elements(raw);
                for elem in 0..n {
                    let dest = base_dest.wrapping_add(elem as u8);
                    let slot = base_slot + elem * 4;
                    self.write_reg(dest, Op::LoadAttr { slot }, pred);
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
                    Op::InterpAttr {
                        slot: attr_slot_ipa(raw),
                        perspective,
                        mode: ipa_interpolation_mode(raw),
                        sat: ipa_saturate(raw),
                    },
                    pred,
                );
            }

            Opcode::TEX => {
                let tex_id = texs_tex_id(raw);
                let coord = reg_a(raw);
                let tex_type = ((raw >> 28) & 0x7) as u32;
                let u = self.read_reg(coord);
                let v = self.read_reg(coord.wrapping_add(1));
                let volume = if tex_type == 4 {
                    Some(self.read_reg(coord.wrapping_add(2)))
                } else {
                    None
                };
                let mask = ((raw >> 31) & 0xF) as u8;
                let mut dst = reg_dest(raw);
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(
                        dst,
                        Op::SampleTex {
                            tex_id,
                            u,
                            v,
                            array: None,
                            volume,
                            component,
                        },
                        pred,
                    );
                    dst = dst.wrapping_add(1);
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
                let (u, v) = if aoffi || dc {
                    (self.read_reg(ra), self.read_reg(ra.wrapping_add(1)))
                } else {
                    (self.read_reg(ra), self.read_reg(rb))
                };
                for lane in 0..4u8 {
                    let dst_reg = match lane {
                        0 => dest_a,
                        1 => dest_a.wrapping_add(1),
                        2 => dest_b,
                        _ => dest_b.wrapping_add(1),
                    };
                    self.write_reg(
                        dst_reg,
                        Op::GatherTex {
                            tex_id,
                            u,
                            v,
                            gather_component,
                            lane,
                        },
                        pred,
                    );
                }
            }

            Opcode::TEXS | Opcode::TLDS => {
                let dest_a = reg_dest(raw);
                let dest_b = ((raw >> 28) & 0xFF) as u8;
                let ra = reg_a(raw);
                let rb = reg_b(raw);
                let enc = (raw >> 53) & 0xF;
                let is_texs = matches!(decoded.opcode, Opcode::TEXS);
                let array_2d = is_texs && matches!(enc, 7 | 8);
                let tex_3d = is_texs && matches!(enc, 10 | 11);
                let (u, v) = if array_2d {
                    (self.read_reg(ra.wrapping_add(1)), self.read_reg(rb))
                } else if tex_3d {
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
                let tex_id = texs_tex_id(raw);
                let swizzle = ((raw >> 50) & 0x7) as usize;

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
                        let id = self.program.emit(
                            Op::SampleTex {
                                tex_id,
                                u,
                                v,
                                array,
                                volume,
                                component,
                            },
                            None,
                        );
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
                        self.write_reg(
                            dst_reg,
                            Op::SampleTex {
                                tex_id,
                                u,
                                v,
                                array,
                                volume,
                                component,
                            },
                            pred,
                        );
                        store_index += 1;
                    }
                }
            }

            Opcode::TMML | Opcode::TMML_b => {
                let mask = ((raw >> 31) & 0xF) as u8;
                let mut dst = reg_dest(raw);
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(dst, Op::Mov(Value::Zero), pred);
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::TXQ | Opcode::TXQ_b | Opcode::TLD4 | Opcode::TLD4_b => {
                let mask = ((raw >> 31) & 0xF) as u8;
                let mut dst = reg_dest(raw);
                for component in 0..4u8 {
                    if (mask >> component) & 1 == 0 {
                        continue;
                    }
                    self.write_reg(dst, Op::Mov(Value::Zero), pred);
                    dst = dst.wrapping_add(1);
                }
            }

            Opcode::MUFU => {
                let dest = reg_dest(raw);
                let src = self.read_reg(reg_a(raw));
                let func = MufuFunc::from_bits(mufu_func_bits(raw));
                self.write_reg(dest, Op::MultiFunc { src, func }, pred);
            }

            Opcode::EXIT => {
                self.program.exit_reg_state = Some(self.reg_state.clone());
                self.program.emit_void_pred(Op::Exit, pred);
                self.finished = true;
            }

            Opcode::LDC => {
                let dest = reg_dest(raw);
                let src_reg = ldc_src_reg(raw);
                let size = ldc_size(raw);
                let count = match size {
                    4 => 1u32,
                    5 => 2,
                    6 => 4,
                    _ => {
                        log::warn!(
                            "LDC with sub-word size not yet lifted raw={:#018x} size={}",
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
                    Some(self.read_reg(src_reg))
                } else {
                    None
                };
                for w in 0..count {
                    let bo = (r.byte_offset as u32).wrapping_add(w * 4);
                    let cb_id = match index {
                        Some(idx) => self.program.emit(
                            Op::LoadCbufIndexed {
                                binding: r.binding,
                                byte_offset: bo,
                                index: idx,
                            },
                            None,
                        ),
                        None => self.program.emit(
                            Op::LoadCbuf {
                                binding: r.binding,
                                byte_offset: bo,
                            },
                            None,
                        ),
                    };
                    let dst = if dest == RZ {
                        RZ
                    } else {
                        dest.wrapping_add(w as u8)
                    };
                    self.write_reg(dst, Op::Mov(Value::Inst(cb_id)), pred);
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
                    let id = self.program.emit(
                        Op::LoadGlobal {
                            addr_lo,
                            offset: off,
                        },
                        None,
                    );
                    let dst = if dest == RZ {
                        RZ
                    } else {
                        dest.wrapping_add(w as u8)
                    };
                    self.write_reg(dst, Op::Mov(Value::Inst(id)), pred);
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
                self.emit_csetp(raw, pred);
            }
            Opcode::PSET => {
                self.emit_pset(raw, pred);
            }

            Opcode::KIL => {
                self.program.emit_void_pred(Op::Kill, pred);
            }

            Opcode::IADD_reg => {
                let b = self.read_reg(reg_b(raw));
                self.emit_iadd(raw, b, pred);
            }
            Opcode::IADD_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_iadd(raw, Value::Inst(id), pred);
            }
            Opcode::IADD_imm => {
                self.emit_iadd(raw, Value::ImmU32(imm20(raw) as u32), pred);
            }
            Opcode::IADD32I => {
                self.emit_iadd(raw, Value::ImmU32(imm32(raw)), pred);
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
                self.emit_ilop(
                    raw,
                    b,
                    LogicOp::from_bits(lop_op(raw)),
                    lop_not_a(raw),
                    lop_not_b(raw),
                    pred,
                );
            }
            Opcode::LOP_cbuf => {
                let id = self.load_cbuf(raw);
                self.emit_ilop(
                    raw,
                    Value::Inst(id),
                    LogicOp::from_bits(lop_op(raw)),
                    lop_not_a(raw),
                    lop_not_b(raw),
                    pred,
                );
            }
            Opcode::LOP_imm => {
                self.emit_ilop(
                    raw,
                    Value::ImmU32(imm20(raw) as u32),
                    LogicOp::from_bits(lop_op(raw)),
                    false,
                    false,
                    pred,
                );
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
                let dest = reg_dest(raw);
                let src = self.read_reg(reg_b(raw));
                self.write_reg(dest, Op::Mov(src), pred);
            }
            Opcode::I2I_cbuf => {
                let dest = reg_dest(raw);
                let id = self.load_cbuf(raw);
                self.write_reg(dest, Op::Mov(Value::Inst(id)), pred);
            }
            Opcode::I2I_imm => {
                let dest = reg_dest(raw);
                self.write_reg(dest, Op::Mov(Value::ImmU32(imm20(raw) as u32)), pred);
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

            Opcode::DEPBAR => {}

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

#[cfg(test)]
mod tests {
    use super::super::ir::Op;
    use super::*;

    #[test]
    fn fmul_reg_emits_fmul() {
        let mut t = Translator::new();

        let raw = 0x5C68_1000_0000_0203u64;
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
        assert_eq!(t.program.instructions.len(), 1);
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
                ..
            } => {}
            other => panic!("expected MultiFunc Rcp, got {other:?}"),
        }
    }

    #[test]
    fn ldc_static_emits_load_cbuf_then_mov() {
        let raw: u64 = (0xEF94u64 << 48)
            | (0x0000u64 << 44)
            | (0x0000u64 << 36)
            | (0x0010u64 << 20)
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
    fn ldc_indexed_emits_unimplemented() {
        let raw: u64 = (0xEF94u64 << 48)
            | (0x0000u64 << 44)
            | (0x0000u64 << 36)
            | (0x0010u64 << 20)
            | (0x0001u64 << 8)
            | 0x0003u64;
        let mut t = Translator::new();
        assert!(!t.translate(raw), "indexed LDC should return false");
        assert_eq!(t.unimplemented_count, 1);
    }
}

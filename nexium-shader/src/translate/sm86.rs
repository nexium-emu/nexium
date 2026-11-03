use super::*;
use crate::ampere::sm70::{Inst, Operand, Source};

mod alu2;
mod mem;
mod tex;

const URZ: u8 = 63;
const UP_BASE: u8 = 16;
pub(crate) const BRANCH_PRED: u8 = 15;
const SCRATCH_PRED: u8 = 14;
const GUARD_PRED: u8 = 13;
const TWO_PI: f32 = std::f32::consts::TAU;
const UNBOUND_BANK: u8 = 24;
const SPARE_BANK: u8 = 17;

pub(crate) struct Sm86Context {
    ur_base: u8,
}

impl Sm86Context {
    pub(crate) fn new(instructions: &[Inst]) -> Option<Self> {
        let highest = instructions
            .iter()
            .filter(|inst| crate::ampere::lower::touches_uniforms(inst))
            .filter_map(crate::ampere::lower::highest_uniform)
            .max();
        let ur_base = match highest {
            Some(highest) => (254 - (highest + 3).min(URZ - 1)) & !3,
            None => RZ,
        };
        Some(Self { ur_base })
    }
}

#[derive(Clone, Copy)]
struct Ctx {
    inst: Inst,
    pred: Option<Predicate>,
    uniform: bool,
    ur_base: u8,
}

impl Ctx {
    fn ureg(&self, index: u8) -> u8 {
        if index >= URZ || self.ur_base == RZ {
            RZ
        } else {
            self.ur_base + index
        }
    }
}

fn predicate_outputs(inst: &Inst) -> &'static [u32] {
    match inst.alu_opcode() {
        0x00b | 0x00c | 0x010 | 0x01c | 0x02a | 0x034 | 0x08c | 0x090 | 0x09c => &[81, 84],
        0x011 | 0x012 | 0x024 | 0x025 | 0x006 | 0x086 | 0x091 | 0x092 | 0x0a4 | 0x0a5 | 0x189 | 0x1a1 | 0x102 => &[81],
        _ => &[],
    }
}

fn uniform_datapath(inst: &Inst) -> bool {
    (0x080..0x0c0).contains(&inst.alu_opcode()) && inst.alu_opcode() != 0x086
}

fn bank(bank: u8) -> u8 {
    if bank == UNBOUND_BANK {
        SPARE_BANK
    } else {
        bank
    }
}

impl Translator {
    pub(crate) fn translate_sm86(
        &mut self,
        context: &Sm86Context,
        inst: Inst,
        _offset: usize,
        defs: &ValueDefs,
    ) -> bool {
        let uniform = uniform_datapath(&inst);
        let (guard, negate) = inst.guard();
        if guard == PT && negate {
            return true;
        }
        let mut pred = (guard != PT).then(|| Predicate { idx: Self::sm86_pred_key(guard, uniform), negate });
        if let Some(Predicate { idx, negate }) = pred {
            if predicate_outputs(&inst).iter().any(|&start| Self::sm86_pred_key(inst.field(start, 3) as u8, uniform) == idx) {
                let id = self.program.emit(
                    Op::PSetPred {
                        dest_p: GUARD_PRED,
                        dest_np: PT,
                        pred_a: idx,
                        neg_pred_a: false,
                        pred_b: PT,
                        neg_pred_b: false,
                        pred_c: PT,
                        neg_pred_c: false,
                        bop_1: BoolOp::And,
                        bop_2: BoolOp::And,
                    },
                    None,
                );
                self.sm86_set_pred(GUARD_PRED, id);
                pred = Some(Predicate { idx: GUARD_PRED, negate });
            }
        }
        let ctx = Ctx { inst, pred, uniform, ur_base: context.ur_base };
        let translated = if uniform { self.sm86_uniform(ctx, defs) } else { self.sm86_dispatch(ctx, defs) };
        if translated.is_none() {
            log::debug!(
                "[sm86] native translation stops at {:#05x} ({:016x}_{:016x})",
                inst.opcode(),
                inst.hi,
                inst.lo
            );
            self.unimplemented_count += 1;
        }
        translated.is_some()
    }

    pub(crate) fn sm86_emit_exit(&mut self) {
        self.program.emit_void(Op::Exit);
        self.program.exit_reg_state = Some(self.reg_state.clone());
        self.finished = true;
    }
    pub(crate) fn sm86_branch_condition(&mut self, inst: Inst) {
        let (guard, guard_negate) = inst.guard();
        let (condition, condition_negate) = inst.pred(87, 90);
        let id = self.program.emit(
            Op::PSetPred {
                dest_p: BRANCH_PRED,
                dest_np: PT,
                pred_a: guard,
                neg_pred_a: guard_negate,
                pred_b: condition,
                neg_pred_b: condition_negate,
                pred_c: PT,
                neg_pred_c: false,
                bop_1: BoolOp::And,
                bop_2: BoolOp::And,
            },
            None,
        );
        self.pred_state.insert(BRANCH_PRED, id);
    }

    fn sm86_pred_key(index: u8, uniform: bool) -> u8 {
        if index == PT || !uniform {
            index
        } else {
            UP_BASE + index
        }
    }

    fn sm86_gpr(&self, ctx: Ctx, register: u8) -> Option<u8> {
        (register == RZ || register < ctx.ur_base || ctx.uniform).then_some(register)
    }

    fn sm86_read(&mut self, ctx: Ctx, register: u8) -> Option<Value> {
        let register = self.sm86_gpr(ctx, register)?;
        Some(self.read_reg(register))
    }

    fn sm86_write(&mut self, ctx: Ctx, register: u8, op: Op) -> Option<ValueId> {
        let register = self.sm86_gpr(ctx, register)?;
        Some(self.write_reg(register, op, ctx.pred))
    }

    fn sm86_operand(&mut self, ctx: Ctx, operand: Operand) -> Option<Value> {
        Some(match operand {
            Operand::Reg(register) => self.sm86_read(ctx, register)?,
            Operand::UReg(index) => self.read_reg(ctx.ureg(index)),
            Operand::Imm(value) => Value::ImmU32(value),
            Operand::Cbuf { bank: index, offset } => {
                if offset % 4 != 0 {
                    return None;
                }
                Value::Inst(self.program.emit(
                    Op::LoadCbuf { binding: bank(index), byte_offset: u32::from(offset) },
                    None,
                ))
            }
        })
    }

    fn sm86_float(&mut self, value: Value, neg: bool, abs: bool) -> Value {
        if !neg && !abs {
            return value;
        }
        self.emit_value(Op::FAdd {
            a: value,
            b: Value::Zero,
            mods: FMods { neg_a: neg, abs_a: abs, ..FMods::default() },
        })
    }

    fn sm86_pred_value(&mut self, ctx: Ctx, (index, negate): (u8, bool)) -> Value {
        self.emit_predicate_mask(Predicate { idx: Self::sm86_pred_key(index, ctx.uniform), negate })
    }

    fn sm86_set_pred(&mut self, key: u8, id: ValueId) {
        if key != PT {
            self.pred_state.insert(key, id);
        }
    }

    fn sm86_uniform(&mut self, ctx: Ctx, defs: &ValueDefs) -> Option<()> {
        let inst = ctx.inst;
        let opcode = inst.alu_opcode();
        let fields: &[u32] = match opcode {
            0x0b9 | 0x0bb | 0x083 | 0x09c => &[],
            _ if inst.form() == 1 => &[16, 24, 32, 64],
            _ => &[16, 24, 64],
        };
        let mut vector = inst;
        for &start in fields {
            vector = vector.with_field(start, 8, u64::from(ctx.ureg(inst.field(start, 8) as u8)));
        }
        let ctx = Ctx { inst: vector, ..ctx };
        match opcode {
            0x0b9 | 0x0bb => self.sm86_uldc(Ctx { inst, ..ctx }),
            0x083 => self.sm86_up2ur(Ctx { inst, ..ctx }),
            0x09c => self.sm86_plop3(ctx),
            0x0a5 => self.sm86_uimad_wide(Ctx { inst, ..ctx }),
            0x0bd | 0x0be => {
                let lowered = vector.with_field(0, 9, u64::from(opcode + 0x043));
                self.sm86_alu(Ctx { inst: lowered, ..ctx }, defs)
            }
            vector_opcode => {
                let lowered = vector.with_field(0, 9, u64::from(vector_opcode - 0x080));
                self.sm86_alu(Ctx { inst: lowered, ..ctx }, defs)
            }
        }
    }

    fn sm86_dispatch(&mut self, ctx: Ctx, defs: &ValueDefs) -> Option<()> {
        match ctx.inst.opcode() {
            0x918 | 0x91a | 0x941 | 0x945 => Some(()),
            0x94d => self.sm86_exit(ctx),
            0x95b => self.sm86_kill(ctx),
            0x919 => self.sm86_s2r(ctx, ctx.inst.dst(), false),
            0x805 => self.sm86_cs2r(ctx),
            0x822 => self.sm86_fswzadd(ctx),
            0x3a1 => self.sm86_match(ctx),
            0xb82 => self.sm86_ldc(ctx),
            _ => match self.sm86_memory(ctx) {
                Some(result) => result,
                None => match self.sm86_texture(ctx, defs) {
                    Some(result) => result,
                    None => self.sm86_alu(ctx, defs),
                },
            },
        }
    }

    fn sm86_alu(&mut self, ctx: Ctx, _defs: &ValueDefs) -> Option<()> {
        match ctx.inst.alu_opcode() {
            0x002 => self.sm86_mov(ctx),
            0x003 => self.sm86_p2r(ctx),
            0x004 => self.sm86_r2p(ctx),
            0x007 => self.sm86_sel(ctx, false),
            0x008 => self.sm86_sel(ctx, true),
            0x009 => self.sm86_fmnmx(ctx),
            0x00a => self.sm86_fset(ctx),
            0x00b => self.sm86_fsetp(ctx),
            0x00c => self.sm86_isetp(ctx),
            0x010 => self.sm86_iadd3(ctx),
            0x011 => self.sm86_lea(ctx),
            0x012 => self.sm86_lop3(ctx),
            0x013 => self.sm86_iabs(ctx),
            0x016 => self.sm86_prmt(ctx),
            0x017 => self.sm86_imnmx(ctx),
            0x019 => self.sm86_shf(ctx),
            0x01a => self.sm86_sgxt(ctx),
            0x01c => self.sm86_plop3(ctx),
            0x020 => self.sm86_fmul(ctx),
            0x021 => self.sm86_fadd(ctx),
            0x023 => self.sm86_ffma(ctx),
            0x024 => self.sm86_imad(ctx),
            0x045 | 0x106 => self.sm86_i2f(ctx),
            0x105 => self.sm86_f2i(ctx),
            0x107 => self.sm86_frnd(ctx),
            0x108 => self.sm86_mufu(ctx),
            0x100 => self.sm86_flo(ctx),
            0x101 => self.sm86_brev(ctx),
            0x109 => self.sm86_popc(ctx),
            0x1c2 => self.sm86_r2ur(ctx),
            0x1c3 => self.sm86_s2r(ctx, ctx.inst.dst(), true),
            0x1c4 => self.sm86_redux(ctx),
            0x006 => self.sm86_vote(ctx, false),
            0x086 => self.sm86_vote(ctx, true),
            0x189 => self.sm86_shfl(ctx),
            0x030 => self.sm86_hadd2(ctx),
            0x034 => self.sm86_hsetp2(ctx),
            0x03e => self.sm86_f2fp(ctx),
            0x148 => Some(()),
            _ => None,
        }
    }

    fn sm86_exit(&mut self, ctx: Ctx) -> Option<()> {
        if ctx.inst.pred(87, 90) != (PT, false) || ctx.pred.is_some() {
            return None;
        }
        self.program.emit_void(Op::Exit);
        self.program.exit_reg_state = Some(self.reg_state.clone());
        self.finished = true;
        Some(())
    }

    fn sm86_kill(&mut self, ctx: Ctx) -> Option<()> {
        if ctx.inst.pred(87, 90) != (PT, false) {
            return None;
        }
        self.program.emit_void_pred(Op::Kill, ctx.pred);
        Some(())
    }

    fn sm86_s2r(&mut self, ctx: Ctx, dst: u8, uniform_dst: bool) -> Option<()> {
        let dst = if uniform_dst { ctx.ureg(dst) } else { self.sm86_gpr(ctx, dst)? };
        if dst == RZ {
            return Some(());
        }
        self.emit_s2r(ctx.inst.lo, dst, ctx.inst.field(72, 8) as u32, ctx.pred).then_some(())
    }

    fn sm86_cs2r(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let count = 1 + u8::from(inst.bit(80));
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        if dst == RZ {
            return Some(());
        }
        let register = inst.field(72, 8) as u32;
        for index in 0..count {
            if register == 0xFF {
                self.write_reg(dst + index, Op::Mov(Value::Zero), ctx.pred);
            } else if !self.emit_s2r(inst.lo, dst + index, register + u32::from(index), ctx.pred) {
                return None;
            }
        }
        Some(())
    }

    fn sm86_ldc_mode(mode: u64) -> LdcMode {
        match mode {
            0 => LdcMode::Default,
            1 => LdcMode::Il,
            2 => LdcMode::Is,
            _ => LdcMode::Isl,
        }
    }

    fn sm86_ldc(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.bit(91) || inst.field(73, 3) > 5 {
            return None;
        }
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        let index = self.sm86_gpr(ctx, inst.reg(24))?;
        self.emit_ldc(
            inst.lo,
            Self::sm86_ldc_mode(inst.field(78, 2)),
            dst,
            index,
            inst.field(73, 3) as u32,
            bank(inst.field(54, 5) as u8),
            inst.field(38, 16) as i32,
            ctx.pred,
        )
        .then_some(())
    }

    fn sm86_uldc(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let indexed = inst.alu_opcode() == 0x0bb;
        if inst.field(73, 3) > 5 || inst.bit(91) != indexed {
            return None;
        }
        let index = if indexed { ctx.ureg(inst.reg(24)) } else { RZ };
        self.emit_ldc(
            inst.lo,
            LdcMode::Default,
            ctx.ureg(inst.dst()),
            index,
            inst.field(73, 3) as u32,
            bank(inst.field(54, 5) as u8),
            inst.field(38, 16) as i32,
            ctx.pred,
        )
        .then_some(())
    }

    fn sm86_r2ur(&mut self, ctx: Ctx) -> Option<()> {
        let source = self.sm86_read(ctx, ctx.inst.reg(24))?;
        let dst = ctx.ureg(ctx.inst.dst());
        if dst != RZ {
            self.write_reg(dst, Op::Mov(source), ctx.pred);
        }
        Some(())
    }

    fn sm86_up2ur(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(32, 32) as u32;
        if inst.form() != 4 || inst.field(64, 32) != 0x0800_0000 || mask > 0x7F || ctx.pred.is_some() {
            return None;
        }
        let merged = self.read_reg(ctx.ureg(inst.reg(24)));
        let mut bits = Value::Zero;
        for index in 0..7u8 {
            if mask >> index & 1 == 0 {
                continue;
            }
            let flag = self.emit_predicate_mask(Predicate { idx: UP_BASE + index, negate: false });
            let bit = self.emit_ilop_imm_value(flag, 1 << index, LogicOp::And, false, false);
            bits = self.emit_ilop_value(bits, bit, LogicOp::Or, false, false);
        }
        let kept = self.emit_ilop_imm_value(merged, !mask, LogicOp::And, false, false);
        let result = self.emit_ilop_value(kept, bits, LogicOp::Or, false, false);
        let dst = ctx.ureg(inst.dst());
        if dst != RZ {
            self.write_reg(dst, Op::Mov(result), None);
        }
        Some(())
    }

    fn sm86_slot32(&self, ctx: Ctx) -> Option<Source> {
        ctx.inst.slot32()
    }

    fn sm86_mov(&mut self, ctx: Ctx) -> Option<()> {
        let source = self.sm86_slot32(ctx)?;
        let value = self.sm86_operand(ctx, source.operand)?;
        self.sm86_write(ctx, ctx.inst.dst(), Op::Mov(value))?;
        Some(())
    }

    fn sm86_sel(&mut self, ctx: Ctx, float: bool) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let mut a_value = self.sm86_operand(ctx, a.operand)?;
        let mut b_value = self.sm86_operand(ctx, b.operand)?;
        if float {
            a_value = self.sm86_float(a_value, a.neg, a.abs);
            b_value = self.sm86_float(b_value, b.neg, b.abs);
        }
        let (index, negate) = inst.pred(87, 90);
        let pred = Predicate { idx: Self::sm86_pred_key(index, ctx.uniform), negate };
        self.sm86_write(ctx, inst.dst(), Op::SelectPred { pred, if_true: a_value, if_false: b_value })?;
        Some(())
    }

    fn sm86_fmnmx(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let a_value = self.sm86_operand(ctx, a.operand)?;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let (index, negate) = inst.pred(87, 90);
        let op = Op::FMinMaxPred {
            a: a_value,
            b: b_value,
            mods: FMods { neg_a: a.neg, abs_a: a.abs, neg_b: b.neg, abs_b: b.abs, ..FMods::default() },
            pred: Self::sm86_pred_key(index, ctx.uniform),
            neg_pred: negate,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_fsetp(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let src_a = self.sm86_operand(ctx, a.operand)?;
        let src_b = self.sm86_operand(ctx, b.operand)?;
        let (src_pred, src_pred_inv) = inst.pred(87, 90);
        let dest_p = Self::sm86_pred_key(inst.field(81, 3) as u8, ctx.uniform);
        let dest_np = Self::sm86_pred_key(inst.field(84, 3) as u8, ctx.uniform);
        let id = self.program.emit_pred(
            Op::FSetPred {
                cmp: FComp::from_bits(inst.field(76, 4)),
                bop: BoolOp::from_bits(inst.field(74, 2)),
                src_a,
                src_b,
                neg_a: a.neg,
                abs_a: a.abs,
                neg_b: b.neg,
                abs_b: b.abs,
                src_pred: Self::sm86_pred_key(src_pred, ctx.uniform),
                src_pred_inv,
                dest_p,
                dest_np,
            },
            None,
            ctx.pred,
        );
        self.sm86_set_pred(dest_p, id);
        self.sm86_set_pred(dest_np, id);
        Some(())
    }

    fn sm86_fset(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(74, 2) == 3 {
            return None;
        }
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let src_a = self.sm86_operand(ctx, a.operand)?;
        let src_b = self.sm86_operand(ctx, b.operand)?;
        let (src_pred, src_pred_inv) = inst.pred(87, 90);
        let op = Op::FSet {
            cmp: FComp::from_bits(inst.field(76, 4)),
            bop: BoolOp::from_bits(inst.field(74, 2)),
            src_a,
            src_b,
            neg_a: a.neg,
            abs_a: a.abs,
            neg_b: b.neg,
            abs_b: b.abs,
            bf: true,
            src_pred: Self::sm86_pred_key(src_pred, ctx.uniform),
            src_pred_inv,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_isetp(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let b = self.sm86_slot32(ctx)?;
        if inst.bit(72) || b.modified() {
            return None;
        }
        let src_a = self.sm86_read(ctx, inst.reg(24))?;
        let src_b = self.sm86_operand(ctx, b.operand)?;
        let (src_pred, src_pred_inv) = inst.pred(87, 90);
        let dest_p = Self::sm86_pred_key(inst.field(81, 3) as u8, ctx.uniform);
        let dest_np = Self::sm86_pred_key(inst.field(84, 3) as u8, ctx.uniform);
        let id = self.program.emit_pred(
            Op::ISetPred {
                cmp: ICmp::from_bits(inst.field(76, 3)),
                signed: inst.bit(73),
                bop: BoolOp::from_bits(inst.field(74, 2)),
                src_a,
                src_b,
                src_pred: Self::sm86_pred_key(src_pred, ctx.uniform),
                src_pred_inv,
                dest_p,
                dest_np,
            },
            None,
            ctx.pred,
        );
        self.sm86_set_pred(dest_p, id);
        self.sm86_set_pred(dest_np, id);
        Some(())
    }

    fn sm86_compare_pred(&mut self, ctx: Ctx, target: u8, a: Value, b: Value, cmp: ICmp) {
        let key = Self::sm86_pred_key(target, ctx.uniform);
        let id = self.program.emit_pred(
            Op::ISetPred {
                cmp,
                signed: false,
                bop: BoolOp::And,
                src_a: a,
                src_b: b,
                src_pred: PT,
                src_pred_inv: false,
                dest_p: key,
                dest_np: PT,
            },
            None,
            ctx.pred,
        );
        self.sm86_set_pred(key, id);
    }

    fn sm86_signed_term(&mut self, ctx: Ctx, source: Source) -> Option<(Value, bool)> {
        Some((self.sm86_operand(ctx, source.operand)?, source.neg))
    }

    fn sm86_iadd3(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let (b, c) = inst.alu_sources()?;
        if a.abs || b.abs {
            return None;
        }
        let carry_in = [inst.pred(87, 90), inst.pred(77, 80)];
        let carry_out = [inst.field(81, 3) as u8, inst.field(84, 3) as u8];
        let (a_value, a_neg) = self.sm86_signed_term(ctx, a)?;
        let (b_value, b_neg) = self.sm86_signed_term(ctx, b)?;
        let (c_value, c_neg) = self.sm86_signed_term(ctx, c)?;
        if inst.bit(74) {
            if carry_out != [PT, PT] {
                return None;
            }
            let mut sum = Value::Zero;
            for (value, negate) in [(a_value, a_neg), (b_value, b_neg), (c_value, c_neg)] {
                let term = if negate {
                    self.emit_ilop_value(Value::Zero, value, LogicOp::PassB, false, true)
                } else {
                    value
                };
                sum = self.emit_iadd_value(sum, term, false, false);
            }
            for carry in carry_in {
                if carry != (PT, true) {
                    let mask = self.sm86_pred_value(ctx, carry);
                    sum = self.emit_iadd_value(sum, mask, false, true);
                }
            }
            self.sm86_write(ctx, inst.dst(), Op::Mov(sum))?;
            return Some(());
        }
        if carry_in != [(PT, true); 2] {
            return None;
        }
        let partial = self.emit_iadd_value(a_value, b_value, a_neg, b_neg);
        if carry_out == [PT, PT] {
            self.sm86_write(ctx, inst.dst(), Op::IAdd { a: partial, b: c_value, neg_a: false, neg_b: c_neg })?;
            return Some(());
        }
        let third_zero = matches!(c.operand, Operand::Reg(RZ) | Operand::Imm(0));
        if carry_out[1] != PT || !third_zero || (a_neg && b_neg) {
            return None;
        }
        match (a_neg, b_neg) {
            (false, false) => self.sm86_compare_pred(ctx, carry_out[0], partial, a_value, ICmp::Lt),
            (false, true) => self.sm86_compare_pred(ctx, carry_out[0], a_value, b_value, ICmp::Ge),
            (true, _) => self.sm86_compare_pred(ctx, carry_out[0], a_value, b_value, ICmp::Le),
        }
        self.sm86_write(ctx, inst.dst(), Op::Mov(partial))?;
        Some(())
    }

    fn sm86_lea(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let (b, _) = inst.alu_sources()?;
        let negated = matches!(inst.form(), 1 | 5 | 6) && b.neg;
        let carry_out = inst.field(81, 3) as u8;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let shift = inst.field(75, 5) as u32;
        if inst.bit(80) {
            if carry_out != PT || inst.bit(72) || b.modified() {
                return None;
            }
            let low = self.sm86_read(ctx, inst.reg(24))?;
            let high = if inst.bit(73) {
                self.emit_value(Op::IShr { a: low, b: Value::ImmU32(31), signed: true })
            } else {
                self.sm86_read(ctx, inst.reg(64))?
            };
            let shf = Shf { bits: 64, wrap: false, right: false, high_half: false, signed: false };
            let shifted_op = self.shf_const_op(&shf, low, high, shift);
            let shifted = self.emit_value(shifted_op);
            let mut sum = self.emit_iadd_value(shifted, b_value, false, false);
            let carry = inst.pred(87, 90);
            if inst.bit(74) && carry != (PT, true) {
                let mask = self.sm86_pred_value(ctx, carry);
                sum = self.emit_iadd_value(sum, mask, false, true);
            }
            self.sm86_write(ctx, inst.dst(), Op::Mov(sum))?;
            return Some(());
        }
        if inst.bit(74) || (negated && carry_out != PT) || (carry_out != PT && inst.bit(72)) {
            return None;
        }
        let a_value = self.sm86_read(ctx, inst.reg(24))?;
        let op = Op::IScAdd { a: a_value, b: b_value, shift: shift as u8, neg_a: inst.bit(72), neg_b: negated };
        if carry_out == PT {
            self.sm86_write(ctx, inst.dst(), op)?;
            return Some(());
        }
        let sum = self.emit_value(op);
        self.sm86_compare_pred(ctx, carry_out, sum, b_value, ICmp::Lt);
        self.sm86_write(ctx, inst.dst(), Op::Mov(sum))?;
        Some(())
    }

    fn sm86_lop3(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.bit(80) || inst.pred(87, 90) != (PT, true) {
            return None;
        }
        let (b, c) = inst.alu_sources()?;
        let a = self.sm86_read(ctx, inst.reg(24))?;
        let b = self.sm86_operand(ctx, b.operand)?;
        let c = self.sm86_operand(ctx, c.operand)?;
        let lut = inst.field(72, 8) as u8;
        let dest_p = Self::sm86_pred_key(inst.field(81, 3) as u8, ctx.uniform);
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        let id = self.write_reg(dst, Op::ILop3 { a, b, c, lut }, ctx.pred);
        self.emit_logic_predicate(dest_p, 3, id, ctx.pred);
        Some(())
    }

    fn sm86_iabs(&mut self, ctx: Ctx) -> Option<()> {
        let source = self.sm86_slot32(ctx)?;
        if source.modified() {
            return None;
        }
        let value = self.sm86_operand(ctx, source.operand)?;
        let negated = self.emit_ineg_value(value);
        let op = Op::IMinMaxPred { a: value, b: negated, signed: true, pred: PT, neg_pred: true };
        self.sm86_write(ctx, ctx.inst.dst(), op)?;
        Some(())
    }

    fn sm86_prmt(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let selector = inst.field(32, 32);
        if inst.form() != 4 || inst.field(72, 24) != 0 || selector > 0xFFFF {
            return None;
        }
        let a = self.sm86_read(ctx, inst.reg(24))?;
        let b = self.sm86_read(ctx, inst.reg(64))?;
        let mut result = Value::Zero;
        for output_byte in 0..4u32 {
            let select = ((selector >> (output_byte * 4)) & 0xf) as u32;
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
        self.sm86_write(ctx, inst.dst(), Op::Mov(result))?;
        Some(())
    }

    fn sm86_imnmx(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let b = self.sm86_slot32(ctx)?;
        if b.modified() {
            return None;
        }
        let a = self.sm86_read(ctx, inst.reg(24))?;
        let b = self.sm86_operand(ctx, b.operand)?;
        let (index, negate) = inst.pred(87, 90);
        let op = Op::IMinMaxPred {
            a,
            b,
            signed: inst.bit(73),
            pred: Self::sm86_pred_key(index, ctx.uniform),
            neg_pred: negate,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_shf(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let (shift, high) = inst.alu_sources()?;
        let low = self.sm86_read(ctx, inst.reg(24))?;
        let shift = self.sm86_operand(ctx, shift.operand)?;
        let kind = inst.field(73, 2);
        let signed = kind == 0 || kind == 2;
        let max_shift = match kind {
            0 => 3,
            1 => 2,
            _ => 0,
        };
        let op = match (inst.bit(76), inst.bit(80)) {
            (false, false) => Op::IShl { a: low, b: shift },
            (true, true) => {
                let high = self.sm86_operand(ctx, high.operand)?;
                Op::IShr { a: high, b: shift, signed }
            }
            (right, _) => {
                let high = self.sm86_operand(ctx, high.operand)?;
                let shf = Shf {
                    bits: if max_shift == 0 { 32 } else { 64 },
                    wrap: inst.bit(75),
                    right,
                    high_half: false,
                    signed: right && max_shift == 3,
                };
                match shift {
                    Value::Zero => self.shf_const_op(&shf, low, high, 0),
                    Value::ImmU32(amount) => {
                        let amount = if shf.wrap { amount & (shf.bits - 1) } else { amount.min(shf.bits) };
                        self.shf_const_op(&shf, low, high, amount)
                    }
                    shift => self.shf_dynamic_op(&shf, low, high, shift),
                }
            }
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_sgxt(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let bits = inst.field(32, 32) as u32;
        if inst.form() != 4 || inst.field(64, 32) != 0 || bits > 32 {
            return None;
        }
        let value = self.sm86_read(ctx, inst.reg(24))?;
        self.sm86_write(ctx, inst.dst(), Op::Bfe { a: value, b: Value::ImmU32(bits << 8), signed: false })?;
        Some(())
    }

    fn sm86_plop3(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if ctx.uniform && inst.bit(67) {
            return None;
        }
        let sources = [inst.pred(87, 90), inst.pred(77, 80), inst.pred(68, 71)];
        let mut masks = [Value::Zero; 3];
        for (index, (predicate, negate)) in sources.into_iter().enumerate() {
            let uniform = ctx.uniform || (index == 2 && inst.bit(67));
            masks[index] =
                self.emit_predicate_mask(Predicate { idx: Self::sm86_pred_key(predicate, uniform), negate });
        }
        let luts = [(inst.field(64, 3) | (inst.field(72, 5) << 3)) as u8, inst.field(16, 8) as u8];
        for (target, lut) in [inst.field(81, 3) as u8, inst.field(84, 3) as u8].into_iter().zip(luts) {
            if target == PT {
                continue;
            }
            let value = self.program.emit(Op::ILop3 { a: masks[0], b: masks[1], c: masks[2], lut }, None);
            self.emit_logic_predicate(Self::sm86_pred_key(target, ctx.uniform), 3, value, ctx.pred);
        }
        Some(())
    }

    fn sm86_fmul(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let scale = match inst.field(84, 3) {
            0 => return None,
            scale => [0, 3, 2, 1, 0, 6, 5, 4][scale as usize],
        };
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let a_value = self.sm86_operand(ctx, a.operand)?;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let mods = FMods {
            neg_a: a.neg,
            abs_a: a.abs,
            neg_b: b.neg,
            abs_b: b.abs,
            sat: inst.bit(77),
            scale,
            ..FMods::default()
        };
        self.sm86_write(ctx, inst.dst(), Op::FMul { a: a_value, b: b_value, mods })?;
        Some(())
    }

    fn sm86_fadd(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let b = self.sm86_slot32(ctx)?;
        let a_value = self.sm86_operand(ctx, a.operand)?;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let mods = FMods {
            neg_a: a.neg,
            abs_a: a.abs,
            neg_b: b.neg,
            abs_b: b.abs,
            sat: inst.bit(77),
            ..FMods::default()
        };
        self.sm86_write(ctx, inst.dst(), Op::FAdd { a: a_value, b: b_value, mods })?;
        Some(())
    }

    fn sm86_ffma(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let a = inst.src0();
        let (b, c) = inst.alu_sources()?;
        let a_value = self.sm86_operand(ctx, a.operand)?;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let mut c_value = self.sm86_operand(ctx, c.operand)?;
        if c.abs {
            c_value = self.sm86_float(c_value, false, true);
        }
        let mods = FMods {
            neg_a: a.neg,
            abs_a: a.abs,
            neg_b: b.neg,
            abs_b: b.abs,
            neg_c: c.neg,
            sat: inst.bit(77),
            ..FMods::default()
        };
        self.sm86_write(ctx, inst.dst(), Op::FFma { a: a_value, b: b_value, c: c_value, mods })?;
        Some(())
    }

    fn sm86_imad(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(81, 3) != u64::from(PT) {
            return None;
        }
        let a = inst.src0();
        let (b, c) = inst.alu_sources()?;
        let a_value = self.sm86_operand(ctx, a.operand)?;
        let b_value = self.sm86_operand(ctx, b.operand)?;
        let c_value = self.sm86_operand(ctx, c.operand)?;
        let product = self.emit_imul_value(a_value, b_value);
        let op = Op::IAdd { a: product, b: c_value, neg_a: a.neg ^ b.neg, neg_b: c.neg };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_i2f(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let source = self.sm86_slot32(ctx)?;
        let size = inst.field(84, 2);
        let selector = if matches!(inst.form(), 2 | 4) { 0 } else { inst.field(60, 2) };
        let byte = match size {
            0 => selector,
            1 if selector < 2 => selector * 2,
            2 if selector == 0 => 0,
            _ => return None,
        };
        if inst.field(75, 2) != 2 || source.modified() {
            return None;
        }
        let src = self.sm86_operand(ctx, source.operand)?;
        let op = Op::I2F {
            src,
            signed: inst.bit(74),
            neg: false,
            abs: false,
            int_format: size as u8,
            selector: byte as u8,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_f2i(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let size = inst.field(75, 2);
        let narrow = size == 1 && !inst.bit(72) && inst.field(78, 2) == 0;
        if !(size == 2 || narrow) || inst.field(84, 2) != 2 {
            return None;
        }
        let source = self.sm86_slot32(ctx)?;
        let value = self.sm86_operand(ctx, source.operand)?;
        let mut src = self.sm86_float(value, source.neg, source.abs);
        if narrow {
            src = self.emit_value(Op::FMax { a: src, b: Value::ImmF32(0.0), mods: FMods::default() });
            src = self.emit_value(Op::FMin { a: src, b: Value::ImmF32(65535.0), mods: FMods::default() });
        }
        let op = Op::F2I { src, signed: inst.bit(72), round: inst.field(78, 2) as u8 };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_frnd(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(75, 2) != 2 || inst.field(84, 2) != 2 {
            return None;
        }
        let source = self.sm86_slot32(ctx)?;
        let src = self.sm86_operand(ctx, source.operand)?;
        let op = Op::F2F {
            src,
            neg: source.neg,
            abs: source.abs,
            sat: false,
            round: inst.field(78, 2) as u8 + 1,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_mufu(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let function = inst.field(74, 6) as u32;
        if inst.bit(73) || function > 8 {
            return None;
        }
        let source = self.sm86_slot32(ctx)?;
        let mut src = self.sm86_operand(ctx, source.operand)?;
        let (mut neg, mut abs) = (source.neg, source.abs);
        if function <= 1 {
            src = self.emit_value(Op::FMul {
                a: src,
                b: Value::ImmF32(TWO_PI),
                mods: FMods { neg_a: neg, abs_a: abs, ..FMods::default() },
            });
            neg = false;
            abs = false;
        }
        let op = Op::MultiFunc {
            src,
            func: MufuFunc::from_bits(function),
            mods: FMods { neg_a: neg, abs_a: abs, ..FMods::default() },
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_unary(&mut self, ctx: Ctx) -> Option<(Value, Source)> {
        let source = self.sm86_slot32(ctx)?;
        if source.abs {
            return None;
        }
        Some((self.sm86_operand(ctx, source.operand)?, source))
    }

    fn sm86_invert(&mut self, value: Value) -> Value {
        self.emit_ilop_value(Value::Zero, value, LogicOp::PassB, false, true)
    }

    fn sm86_flo(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(81, 3) != u64::from(PT) {
            return None;
        }
        let (mut source, operand) = self.sm86_unary(ctx)?;
        if operand.neg {
            source = self.sm86_invert(source);
        }
        if inst.bit(73) {
            let sign = self.emit_value(Op::IShr { a: source, b: Value::ImmU32(31), signed: true });
            source = self.emit_ilop_value(source, sign, LogicOp::Xor, false, false);
        }
        let msb = self.emit_value(Op::FindUMsb { value: source });
        if !inst.bit(74) {
            self.sm86_write(ctx, inst.dst(), Op::Mov(msb))?;
            return Some(());
        }
        let found = self.emit_value(Op::ISet {
            cmp: ICmp::Ne,
            signed: false,
            a: msb,
            b: Value::ImmU32(0xFFFF_FFFF),
            bool_float: false,
        });
        let xor_mask = self.emit_ilop_imm_value(found, 31, LogicOp::And, false, false);
        self.sm86_write(
            ctx,
            inst.dst(),
            Op::ILop { a: msb, b: xor_mask, op: LogicOp::Xor, not_a: false, not_b: false },
        )?;
        Some(())
    }

    fn sm86_brev(&mut self, ctx: Ctx) -> Option<()> {
        let (value, source) = self.sm86_unary(ctx)?;
        if source.neg {
            return None;
        }
        self.sm86_write(ctx, ctx.inst.dst(), Op::BitReverse { value })?;
        Some(())
    }

    fn sm86_popc(&mut self, ctx: Ctx) -> Option<()> {
        let (mut value, source) = self.sm86_unary(ctx)?;
        if source.neg {
            value = self.sm86_invert(value);
        }
        self.sm86_write(ctx, ctx.inst.dst(), Op::BitCount { value })?;
        Some(())
    }

    fn sm86_p2r(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(32, 32) as u32;
        if inst.form() != 4 || mask > 0x7F || inst.field(64, 32) & !0x3000 != 0 {
            return None;
        }
        let offset = inst.field(76, 2) as u32 * 8;
        let mut insert = Value::Zero;
        for index in 0..7u8 {
            if mask >> index & 1 == 0 {
                continue;
            }
            let flag = self.emit_predicate_mask(Predicate {
                idx: Self::sm86_pred_key(index, ctx.uniform),
                negate: false,
            });
            let bit = self.emit_ilop_imm_value(flag, 1 << (u32::from(index) + offset), LogicOp::And, false, false);
            insert = self.emit_ilop_value(insert, bit, LogicOp::Or, false, false);
        }
        let merge = self.sm86_read(ctx, inst.reg(24))?;
        let kept = self.emit_ilop_imm_value(merge, !(mask << offset), LogicOp::And, false, false);
        let result = self.emit_ilop_value(kept, insert, LogicOp::Or, false, false);
        self.sm86_write(ctx, inst.dst(), Op::Mov(result))?;
        Some(())
    }

    fn sm86_r2p(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.form() != 4 || inst.field(72, 4) != 0 || inst.field(78, 2) != 0 {
            return None;
        }
        let source = self.sm86_read(ctx, inst.reg(24))?;
        let mask = inst.field(32, 32) as u32 & 0x7F;
        let offset = inst.field(76, 2) as u32 * 8;
        let guard_bit = ctx.pred.map(|guard| {
            let guard_mask = self.emit_predicate_mask(guard);
            self.emit_bfe_value(guard_mask, 0, 1, false)
        });
        for index in 0..7u8 {
            if mask >> index & 1 == 0 {
                continue;
            }
            let key = Self::sm86_pred_key(index, ctx.uniform);
            let bit = self.emit_bfe_value(source, offset + u32::from(index), 1, false);
            let result = match guard_bit {
                Some(guard) => {
                    let old_mask = self.emit_predicate_mask(Predicate { idx: key, negate: false });
                    let old = self.emit_bfe_value(old_mask, 0, 1, false);
                    let changed = self.emit_ilop_value(old, bit, LogicOp::Xor, false, false);
                    let delta = self.emit_ilop_value(guard, changed, LogicOp::And, false, false);
                    self.emit_ilop_value(old, delta, LogicOp::Xor, false, false)
                }
                None => bit,
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
                    dest_p: key,
                    dest_np: PT,
                },
                None,
                None,
            );
            self.pred_state.insert(key, id);
        }
        Some(())
    }
}

#[cfg(test)]
mod tests;

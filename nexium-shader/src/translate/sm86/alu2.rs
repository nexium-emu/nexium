use super::*;

impl Translator {
    pub(super) fn sm86_vote(&mut self, ctx: Ctx, uniform: bool) -> Option<()> {
        let inst = ctx.inst;
        let mode = match inst.field(72, 2) {
            0 => VoteMode::All,
            1 => VoteMode::Any,
            2 => VoteMode::Equal,
            _ => return None,
        };
        if inst.field(32, 32) != 0 || inst.field(64, 32) & !0x078e_0300 != 0 {
            return None;
        }
        let dest = if uniform { ctx.ureg(inst.dst()) } else { self.sm86_gpr(ctx, inst.dst())? };
        let (source, negate) = inst.pred(87, 90);
        let pred_dest = Self::sm86_pred_key(inst.field(81, 3) as u8, uniform);
        let old = self.read_reg(dest);
        let id = self.program.emit_pred(
            Op::SubgroupVote {
                source_pred: Predicate { idx: Self::sm86_pred_key(source, false), negate },
                mode,
                pred_dest,
                old,
            },
            Some(dest),
            ctx.pred,
        );
        if dest != RZ {
            self.reg_state.insert(dest, Value::Inst(id));
        }
        self.sm86_set_pred(pred_dest, id);
        Some(())
    }

    pub(super) fn sm86_shfl(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let form = inst.form();
        let lane_immediate = matches!(form, 4 | 7);
        let mask_immediate = matches!(form, 2 | 7);
        if !matches!(form, 1 | 2 | 4 | 7)
            || inst.field(60, 4) != 0
            || inst.field(72, 9) != 0
            || inst.field(84, 12) != 0
            || (lane_immediate && inst.reg(32) != 0)
        {
            return None;
        }
        let dest = self.sm86_gpr(ctx, inst.dst())?;
        let src = self.sm86_read(ctx, inst.reg(24))?;
        let mode = inst.field(58, 2) as u8;
        let pred_dest = inst.field(81, 3) as u8;
        let lane = inst.field(53, 5) as u8;
        let mask = inst.field(40, 13) as u32;
        if self.stage == ShaderStage::Fragment
            && ctx.pred.is_none()
            && lane_immediate
            && mask_immediate
            && mode == 3
            && mask == 0x1C03
            && pred_dest == PT
            && matches!(lane, 1 | 2)
        {
            self.write_reg(dest, Op::Mov(src), None);
            self.fine_derivative_shuffles.insert(dest, FineDerivativeShuffle { src, index: lane });
            return Some(());
        }
        let index = if lane_immediate { Value::ImmU32(u32::from(lane)) } else { self.sm86_read(ctx, inst.reg(32))? };
        let mask = if mask_immediate { Value::ImmU32(mask) } else { self.sm86_read(ctx, inst.reg(64))? };
        let id = self.program.emit(Op::Shfl { value: src, index, mask, mode, pred_dest }, None);
        self.write_reg(dest, Op::Mov(Value::Inst(id)), ctx.pred);
        Some(())
    }

    pub(super) fn sm86_fswzadd(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let dest = self.sm86_gpr(ctx, inst.dst())?;
        let src_a = self.sm86_gpr(ctx, inst.reg(24))?;
        let src_b = self.sm86_read(ctx, inst.reg(64))?;
        let table = inst.field(32, 8);
        let swizzle = (0..4).fold(0, |lanes, lane| lanes | ((table >> (6 - 2 * lane)) & 3) << (2 * lane)) as u8;
        let plain = !inst.bit(77) && inst.field(78, 2) == 0 && !inst.bit(80);
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
        if plain {
            if let Some(op) = derivative {
                self.write_reg(dest, op, ctx.pred);
                return Some(());
            }
        }
        let a = self.read_reg(src_a);
        self.write_reg(dest, Op::FSwzAdd { a, b: src_b, swizzle: u32::from(swizzle) }, ctx.pred);
        Some(())
    }

    fn sm86_half_b(&mut self, ctx: Ctx) -> Option<(Value, HalfSwizzle, bool, bool)> {
        let source = self.sm86_slot32(ctx)?;
        let value = self.sm86_operand(ctx, source.operand)?;
        let (swizzle, abs, neg) = if ctx.inst.form() == 2 {
            (0, false, false)
        } else {
            (ctx.inst.field(60, 2), ctx.inst.bit(62), ctx.inst.bit(63))
        };
        Some((value, Self::half_swizzle(swizzle as u8), abs, neg))
    }

    pub(super) fn sm86_hadd2(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if !matches!(inst.form(), 1..=3) || inst.bit(76) || inst.bit(79) || inst.field(81, 10) != 0 || inst.bit(91) {
            return None;
        }
        let dest = self.sm86_gpr(ctx, inst.dst())?;
        let a_register = self.sm86_gpr(ctx, inst.reg(24))?;
        let (b, swizzle_b, abs_b, neg_b) = self.sm86_half_b(ctx)?;
        let (swizzle_a, abs_a, neg_a) = (Self::half_swizzle(inst.field(74, 2) as u8), inst.bit(73), inst.bit(72));
        if inst.bit(78) {
            let b_bits = inst.field(60, 2);
            if a_register != RZ || !neg_a || abs_a || inst.field(74, 2) != 0 || b_bits < 2 || inst.bit(77) || inst.bit(80) {
                return None;
            }
            let half = self.emit_value(Op::HAdd {
                a: b,
                b: Value::Zero,
                old: Value::Zero,
                merge: HalfMerge::F32,
                swizzle_a: if b_bits == 3 { HalfSwizzle::H1H1 } else { HalfSwizzle::H0H0 },
                swizzle_b: HalfSwizzle::F32,
                abs_a: false,
                neg_a: false,
                abs_b: false,
                neg_b: true,
                sat: false,
                ftz: false,
            });
            self.write_reg(dest, Op::F2F { src: half, neg: neg_b, abs: abs_b, sat: false, round: 0 }, ctx.pred);
            return Some(());
        }
        let old = self.read_reg(dest);
        let a = self.read_reg(a_register);
        let op = Op::HAdd {
            a,
            b,
            old,
            merge: HalfMerge::H1H0,
            swizzle_a,
            swizzle_b,
            abs_a,
            neg_a,
            abs_b,
            neg_b,
            sat: inst.bit(77),
            ftz: inst.bit(80),
        };
        self.write_reg(dest, op, ctx.pred);
        Some(())
    }

    pub(super) fn sm86_hsetp2(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if !matches!(inst.form(), 1..=3) || inst.bit(65) || inst.bit(91) {
            return None;
        }
        let src_a = self.sm86_read(ctx, inst.reg(24))?;
        let (src_b, swizzle_b, abs_b, neg_b) = self.sm86_half_b(ctx)?;
        let dest_p = inst.field(81, 3) as u8;
        let dest_np = inst.field(84, 3) as u8;
        let id = self.program.emit_pred(
            Op::HSetPred {
                cmp: FComp::from_bits(inst.field(76, 4)),
                bop: BoolOp::from_bits(inst.field(69, 2)),
                src_a,
                src_b,
                swizzle_a: Self::half_swizzle(inst.field(74, 2) as u8),
                swizzle_b,
                neg_a: inst.bit(72),
                abs_a: inst.bit(73),
                neg_b,
                abs_b,
                src_pred: inst.field(87, 3) as u8,
                src_pred_inv: inst.bit(90),
                dest_p,
                dest_np,
                h_and: inst.bit(80),
                ftz: inst.bit(71),
            },
            None,
            ctx.pred,
        );
        self.sm86_set_pred(dest_p, id);
        self.sm86_set_pred(dest_np, id);
        Some(())
    }

    pub(super) fn sm86_f2fp(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(62, 2) != 0 || inst.field(72, 6) != 0 || inst.field(79, 13) != 0 {
            return None;
        }
        let low = self.sm86_slot32(ctx)?;
        let low = self.sm86_operand(ctx, low.operand)?;
        let op = if inst.bit(78) {
            let high = self.sm86_read(ctx, inst.reg(64))?;
            let packed = self.emit_value(Op::PackHalf2 { lo: low, hi: Value::Zero });
            let shifted = self.emit_ishl_imm_value(high, 16);
            Op::ILop { a: shifted, b: packed, op: LogicOp::Or, not_a: false, not_b: false }
        } else {
            let high = self.sm86_read(ctx, inst.reg(24))?;
            Op::PackHalf2 { lo: low, hi: high }
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    pub(super) fn sm86_redux(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let maximum = match inst.field(64, 32) {
            0xC000 => false,
            0x1_4000 => true,
            _ => return None,
        };
        let dest = ctx.ureg(inst.dst());
        if ctx.pred.is_some() || inst.field(32, 32) != 0 || dest == RZ {
            return None;
        }
        let source = self.sm86_read(ctx, inst.reg(24))?;
        let always = Predicate { idx: PT, negate: false };
        let active = self.emit_value(Op::SubgroupVote { source_pred: always, mode: VoteMode::Any, pred_dest: PT, old: Value::Zero });
        let mut total = Value::Zero;
        for lane in 0..32 {
            let value = self.emit_value(Op::Shfl { value: source, index: Value::ImmU32(lane), mask: Value::ImmU32(0x1F), mode: 0, pred_dest: PT });
            let bit = self.emit_bfe_value(active, lane, 1, true);
            let value = self.emit_ilop_value(value, bit, LogicOp::And, false, false);
            total = if maximum {
                self.emit_value(Op::IMinMaxPred { a: total, b: value, signed: false, pred: PT, neg_pred: true })
            } else {
                self.emit_iadd_value(total, value, false, false)
            };
        }
        self.write_reg(dest, Op::Mov(total), None);
        Some(())
    }

    pub(super) fn sm86_match(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let target = inst.field(81, 3) as u8;
        if inst.dst() != RZ
            || inst.reg(24) & 1 != 0
            || ctx.pred.is_some()
            || inst.field(32, 32) != 0
            || inst.field(64, 32) & !0xE_0000 != 0x200
        {
            return None;
        }
        let source = self.sm86_gpr(ctx, inst.reg(24))?;
        let always = Predicate { idx: PT, negate: false };
        let active = self.emit_value(Op::SubgroupVote { source_pred: always, mode: VoteMode::Any, pred_dest: PT, old: Value::Zero });
        let negated = self.emit_ineg_value(active);
        let lowest = self.emit_ilop_value(active, negated, LogicOp::And, false, false);
        let leader = self.emit_value(Op::FindUMsb { value: lowest });
        for word in 0..2u8 {
            let value = self.read_reg(if source == RZ { RZ } else { source + word });
            let shuffled = self.program.emit(
                Op::Shfl { value, index: leader, mask: Value::ImmU32(0x1F), mode: 0, pred_dest: PT },
                None,
            );
            let id = self.program.emit(
                Op::ISetPred {
                    cmp: ICmp::Eq,
                    signed: false,
                    bop: BoolOp::And,
                    src_a: value,
                    src_b: Value::Inst(shuffled),
                    src_pred: if word == 0 { PT } else { SCRATCH_PRED },
                    src_pred_inv: false,
                    dest_p: SCRATCH_PRED,
                    dest_np: PT,
                },
                None,
            );
            self.sm86_set_pred(SCRATCH_PRED, id);
        }
        let id = self.program.emit(
            Op::SubgroupVote {
                source_pred: Predicate { idx: SCRATCH_PRED, negate: false },
                mode: VoteMode::All,
                pred_dest: target,
                old: Value::Zero,
            },
            None,
        );
        self.sm86_set_pred(target, id);
        Some(())
    }

    pub(super) fn sm86_uimad_wide(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let (d, c) = (inst.dst(), inst.field(64, 8) as u8);
        if ctx.pred.is_some()
            || inst.field(40, 24) != 0
            || inst.field(72, 24) != 0x0F_8E00
            || d & 1 != 0
            || d >= URZ
            || (c < URZ && c & 1 != 0)
        {
            return None;
        }
        let a = self.read_reg(ctx.ureg(inst.reg(24)));
        let b = self.read_reg(ctx.ureg(inst.reg(32)));
        let (c_low, c_high) = if c >= URZ {
            (Value::Zero, Value::Zero)
        } else {
            (self.read_reg(ctx.ureg(c)), self.read_reg(ctx.ureg(c + 1)))
        };
        let product_low = self.emit_imul_value(a, b);
        let product_high = self.emit_value(Op::IMulHigh { a, b, signed_a: false, signed_b: false });
        let low = self.emit_iadd_value(product_low, c_low, false, false);
        let carry = self.emit_value(Op::ISet { cmp: ICmp::Lt, signed: false, a: low, b: c_low, bool_float: false });
        let high = self.emit_iadd_value(product_high, c_high, false, false);
        let high = self.emit_iadd_value(high, carry, false, true);
        self.write_reg(ctx.ureg(d), Op::Mov(low), None);
        self.write_reg(ctx.ureg(d + 1), Op::Mov(high), None);
        Some(())
    }
}

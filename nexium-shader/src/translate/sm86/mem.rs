use super::*;

impl Translator {
    pub(super) fn sm86_memory(&mut self, ctx: Ctx) -> Option<Option<()>> {
        let inst = ctx.inst;
        Some(match inst.opcode() {
            0x321 => self.sm86_ald(ctx),
            0x322 => self.sm86_ast(ctx),
            0x326 => self.sm86_ipa(ctx),
            0x381 => self.sm86_ldg(ctx, None),
            0x981 => self.sm86_ldg(ctx, Some(inst.field(32, 6) as u8)),
            0x386 => self.sm86_stg(ctx, None),
            0x986 => self.sm86_stg(ctx, Some(inst.field(64, 6) as u8)),
            0x3a8 => self.sm86_atomg(ctx, None),
            0x9a8 => self.sm86_atomg(ctx, Some(inst.field(64, 6) as u8)),
            0x98e => self.sm86_red(ctx),
            0x984 => self.sm86_shared(ctx, true),
            0x388 => self.sm86_shared(ctx, false),
            0xb1d => self.sm86_barrier(ctx),
            0x992 => self.sm86_membar(ctx, false),
            0x9ab => self.sm86_membar(ctx, true),
            _ => return None,
        })
    }

    fn sm86_ald(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.bit(76) || inst.bit(77) || inst.bit(79) {
            return None;
        }
        let count = inst.field(74, 2) as u32 + 1;
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        let index = self.sm86_gpr(ctx, inst.reg(24))?;
        let vertex = self.sm86_read(ctx, inst.reg(32))?;
        let base_slot = inst.field(40, 10) as u32;
        for element in 0..count {
            let slot = base_slot + element * 4;
            let op = if self.stage == ShaderStage::Geometry {
                Op::LoadGeometryAttr { slot, vertex }
            } else if index != RZ {
                let address = self.read_reg(index);
                let address = self.emit_iadd_value(address, Value::ImmU32(element * 4), false, false);
                Op::LoadAttrIndexed { address }
            } else {
                Op::LoadAttr { slot }
            };
            self.write_reg(dst.wrapping_add(element as u8), op, ctx.pred);
        }
        Some(())
    }

    fn sm86_ast(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.bit(76) || inst.bit(77) || inst.reg(24) != RZ {
            return None;
        }
        let count = inst.field(74, 2) as u32 + 1;
        let data = self.sm86_gpr(ctx, inst.reg(32))?;
        let base_slot = inst.field(40, 10) as u32;
        for element in 0..count {
            let src = self.read_reg(data.wrapping_add(element as u8));
            self.program.emit_void_pred(Op::StoreAttr { slot: base_slot + element * 4, src }, ctx.pred);
        }
        Some(())
    }

    fn sm86_ipa(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mode = match inst.field(78, 2) {
            0 => 0,
            1 => 2,
            _ => return None,
        };
        if inst.field(76, 2) == 3 || inst.field(81, 3) != u64::from(PT) || inst.reg(24) != 0 {
            return None;
        }
        let op = Op::InterpAttr {
            slot: inst.field(64, 8) as u32 * 4,
            perspective: Value::Zero,
            mode,
            sat: false,
        };
        self.sm86_write(ctx, inst.dst(), op)?;
        Some(())
    }

    fn sm86_global_address(&mut self, ctx: Ctx, uniform: Option<u8>) -> Option<Value> {
        let inst = ctx.inst;
        if !inst.bit(72) {
            return None;
        }
        let register = self.sm86_gpr(ctx, inst.reg(24))?;
        let base = uniform.map_or(RZ, |index| ctx.ureg(index));
        if base == RZ {
            return Some(self.read_reg(register));
        }
        let base = self.read_reg(base);
        if register == RZ {
            return Some(base);
        }
        let low = self.read_reg(register);
        Some(self.emit_iadd_value(base, low, false, false))
    }

    fn sm86_global_count(size: u64) -> Option<u32> {
        match size {
            0..=4 => Some(1),
            5 => Some(2),
            6 => Some(4),
            _ => None,
        }
    }

    fn sm86_ldg(&mut self, ctx: Ctx, uniform: Option<u8>) -> Option<()> {
        let inst = ctx.inst;
        if (uniform.is_some() && inst.field(38, 2) != 0)
            || (uniform.is_none() && inst.reg(32) != 0)
            || inst.field(81, 3) != u64::from(PT)
        {
            return None;
        }
        let size = inst.field(73, 3) as u32;
        let count = Self::sm86_global_count(u64::from(size))?;
        let dest = self.sm86_gpr(ctx, inst.dst())?;
        let addr_lo = self.sm86_global_address(ctx, uniform)?;
        let offset = inst.signed_offset(40, 24) as i32;
        for word in 0..count {
            let off = offset.wrapping_add((word * 4) as i32);
            let id = self.program.emit_pred(Op::LoadGlobal { addr_lo, offset: off }, None, ctx.pred);
            let value = if size < 4 {
                let address = self.emit_iadd_value(addr_lo, Value::ImmU32(off as u32), false, false);
                let lane_bits = self.emit_ishl_imm_value(address, 3);
                let position = self.emit_ilop_imm_value(
                    lane_bits,
                    if size < 2 { 24 } else { 16 },
                    LogicOp::And,
                    false,
                    false,
                );
                let control = self.emit_iadd_value(
                    position,
                    Value::ImmU32(if size < 2 { 8 << 8 } else { 16 << 8 }),
                    false,
                    false,
                );
                self.emit_value(Op::Bfe { a: Value::Inst(id), b: control, signed: size & 1 != 0 })
            } else {
                Value::Inst(id)
            };
            let dst = if dest == RZ { RZ } else { dest.wrapping_add(word as u8) };
            self.write_reg(dst, Op::Mov(value), ctx.pred);
        }
        Some(())
    }

    fn sm86_stg(&mut self, ctx: Ctx, uniform: Option<u8>) -> Option<()> {
        let inst = ctx.inst;
        let size = inst.field(73, 3) as u32;
        let count = Self::sm86_global_count(u64::from(size))?;
        let source = self.sm86_gpr(ctx, inst.reg(32))?;
        let addr_lo = self.sm86_global_address(ctx, uniform)?;
        let offset = inst.signed_offset(40, 24) as i32;
        if size < 4 {
            let value = self.read_reg(source);
            self.emit_subword_global_store(addr_lo, offset, value, size, ctx.pred);
            return Some(());
        }
        for word in 0..count {
            let value = self.read_reg(if source == RZ { RZ } else { source.wrapping_add(word as u8) });
            let off = offset.wrapping_add((word * 4) as i32);
            self.program.emit_void_pred(Op::StoreGlobal { addr_lo, offset: off, value }, ctx.pred);
        }
        Some(())
    }

    fn sm86_atomg(&mut self, ctx: Ctx, uniform: Option<u8>) -> Option<()> {
        let inst = ctx.inst;
        let mode = match uniform {
            Some(_) => inst.field(64, 32) & !0x3F == 0x081E_E140 && inst.reg(24) == RZ,
            None => inst.field(64, 32) == 0x001E_E100,
        };
        let offset = inst.signed_offset(40, 24);
        if !mode || !(-0x8_0000..0x8_0000).contains(&offset) {
            return None;
        }
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        let value = self.sm86_read(ctx, inst.reg(32))?;
        let addr_lo = self.sm86_global_address(ctx, uniform)?;
        let op = Op::GlobalAtomic {
            addr_lo,
            offset: offset as i32,
            value,
            op: ImageAtomicOp::Add,
            is_signed: false,
        };
        self.write_side_effecting_reg(dst, op, ctx.pred);
        Some(())
    }

    fn sm86_red(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mode = inst.field(64, 32) & !0x0380_0000;
        let uniform = (inst.reg(24) == RZ && mode & !0x3F == 0x0C10_E100).then(|| inst.field(64, 6) as u8);
        let offset = inst.signed_offset(40, 24);
        if (uniform.is_none() && mode != 0x0010_E100) || !(-0x8_0000..0x8_0000).contains(&offset) {
            return None;
        }
        let op = match inst.field(87, 3) {
            0 => ImageAtomicOp::Add,
            1 => ImageAtomicOp::Min,
            2 => ImageAtomicOp::Max,
            3 => ImageAtomicOp::Increment,
            4 => ImageAtomicOp::Decrement,
            5 => ImageAtomicOp::And,
            6 => ImageAtomicOp::Or,
            _ => ImageAtomicOp::Xor,
        };
        let value = self.sm86_read(ctx, inst.reg(32))?;
        let addr_lo = self.sm86_global_address(ctx, uniform)?;
        self.program.emit_void_pred(
            Op::GlobalAtomic { addr_lo, offset: offset as i32, value, op, is_signed: false },
            ctx.pred,
        );
        Some(())
    }

    fn sm86_shared(&mut self, ctx: Ctx, load: bool) -> Option<()> {
        let inst = ctx.inst;
        if self.stage != ShaderStage::Compute {
            return None;
        }
        let width = inst.field(73, 3);
        let stride = inst.field(78, 2);
        let uniform_base = load && inst.bit(91);
        let expected = (width << 9) | (stride << 14) | if uniform_base { 0x0800_0000 } else { 0 };
        if !(4..=6).contains(&width)
            || inst.field(64, 32) != expected
            || (load && !uniform_base && inst.reg(32) != 0)
        {
            return None;
        }
        let count = match width {
            4 => 1u32,
            5 => 2,
            _ => 4,
        };
        let data = self.sm86_gpr(ctx, if load { inst.dst() } else { inst.reg(32) })?;
        let register = self.sm86_gpr(ctx, inst.reg(24))?;
        let immediate = inst.field(40, 24) as u32;
        let mut address = if register == RZ { Value::Zero } else { self.read_reg(register) };
        if stride != 0 && register != RZ {
            address = self.emit_ishl_imm_value(address, stride as u32 + 1);
        }
        if uniform_base {
            let base = self.read_reg(ctx.ureg(inst.reg(32)));
            address = self.emit_iadd_value(address, base, false, false);
        }
        let displacement = if register == RZ && !uniform_base { immediate } else { (((immediate << 8) as i32) >> 8) as u32 };
        let base = self.emit_iadd_value(address, Value::ImmU32(displacement), false, false);
        for word in 0..count {
            let addr = if word == 0 { base } else { self.emit_iadd_value(base, Value::ImmU32(word * 4), false, false) };
            let register = data.wrapping_add(word as u8);
            if load {
                let value = self.program.emit(Op::LoadShared { addr }, None);
                self.write_reg(register, Op::Mov(Value::Inst(value)), ctx.pred);
            } else {
                let value = self.read_reg(register);
                self.program.emit_void_pred(Op::StoreShared { addr, value }, ctx.pred);
            }
        }
        Some(())
    }

    fn sm86_barrier(&mut self, ctx: Ctx) -> Option<()> {
        if ctx.inst.field(32, 64) != 0 || ctx.pred.is_some() || self.stage != ShaderStage::Compute {
            return None;
        }
        self.program.emit_void(Op::WorkgroupBarrier);
        Some(())
    }

    fn sm86_membar(&mut self, ctx: Ctx, error: bool) -> Option<()> {
        let inst = ctx.inst;
        let scope = if error { 1 } else { inst.field(76, 3) };
        let expected = if error { 0 } else { scope << 12 };
        if inst.field(32, 32) != 0 || inst.field(64, 32) != expected || ctx.pred.is_some() {
            return None;
        }
        if self.stage != ShaderStage::Compute {
            return Some(());
        }
        let scope = if scope == 0 { MemoryBarrierScope::Workgroup } else { MemoryBarrierScope::Device };
        self.program.emit_void(Op::MemoryBarrier { scope });
        Some(())
    }
}

use super::*;

const STAGING: u8 = 248;

impl Translator {
    pub(super) fn sm86_texture(&mut self, ctx: Ctx, defs: &ValueDefs) -> Option<Option<()>> {
        Some(match ctx.inst.opcode() {
            0xb60 => self.sm86_tex(ctx, defs),
            0xb69 => self.sm86_tmml(ctx),
            0xb6f => self.sm86_txq(ctx),
            0xb66 => self.sm86_tld(ctx),
            0xb63 => self.sm86_tld4(ctx),
            0xb97 => self.sm86_suld(ctx),
            0xb9b => self.sm86_sust(ctx),
            0x59f => self.sm86_sured(ctx),
            _ => return None,
        })
    }

    fn sm86_bound_handle(&self, ctx: Ctx) -> CbufHandleOrigin {
        CbufHandleOrigin {
            binding: bank(ctx.inst.field(54, 5) as u8),
            word_offset: ctx.inst.field(40, 14) as u32,
            secondary_word_offset: None,
            secondary_binding: None,
        }
    }

    fn sm86_texture_components(&self, ctx: Ctx, coordinates: u8, extras: u8) -> Option<Vec<u8>> {
        let inst = ctx.inst;
        let total = coordinates + extras;
        let (first, second) = (inst.reg(24), inst.reg(32));
        let offset = |base: u8, index: u8| if base == RZ { RZ } else { base.saturating_add(index) };
        let components: Vec<u8> = if inst.bit(60) {
            if total > 4 {
                return None;
            }
            let split = total.div_ceil(2);
            (0..split)
                .map(|index| offset(first, index))
                .chain((0..total - split).map(|index| offset(second, index)))
                .collect()
        } else {
            (0..coordinates)
                .map(|index| offset(first, index))
                .chain((0..extras).map(|index| offset(second, index)))
                .collect()
        };
        components.iter().all(|&register| self.sm86_gpr(ctx, register).is_some()).then_some(components)
    }

    fn sm86_texture_targets(&self, ctx: Ctx, mask: u8) -> Option<Vec<u8>> {
        let (dst0, dst1) = (ctx.inst.dst(), ctx.inst.reg(64));
        let offset = |base: u8, index: u8| if base == RZ { RZ } else { base.saturating_add(index) };
        let targets: Vec<u8> = (0..mask.count_ones() as u8)
            .map(|index| if index < 2 || dst1 == RZ { offset(dst0, index) } else { offset(dst1, index - 2) })
            .collect();
        targets.iter().all(|&register| self.sm86_gpr(ctx, register).is_some()).then_some(targets)
    }

    fn sm86_half_targets(&self, ctx: Ctx, count: u8) -> Option<Vec<u8>> {
        let (dst0, dst1) = (ctx.inst.dst(), ctx.inst.reg(64));
        let targets: Vec<u8> = (0..count.div_ceil(2))
            .map(|index| match index {
                0 => dst0,
                _ if dst1 != RZ => dst1,
                _ if dst0 == RZ => RZ,
                _ => dst0.saturating_add(1),
            })
            .collect();
        targets.iter().all(|&register| self.sm86_gpr(ctx, register).is_some()).then_some(targets)
    }

    fn sm86_sample_into(
        &mut self,
        ctx: Ctx,
        mask: u8,
        half: bool,
        sample: impl FnOnce(&mut Self, &[u8]) -> Option<Vec<(usize, Value)>>,
    ) -> Option<()> {
        let count = mask.count_ones() as u8;
        if !half {
            let targets = self.sm86_texture_targets(ctx, mask)?;
            sample(self, &targets)?;
            return Some(());
        }
        let targets = self.sm86_half_targets(ctx, count)?;
        let staging: Vec<u8> = (0..count).map(|index| STAGING + index).collect();
        let saved: Vec<Option<Value>> = staging.iter().map(|register| self.reg_state.get(register).copied()).collect();
        sample(self, &staging)?;
        let values: Vec<Value> = staging.iter().map(|&register| self.read_reg(register)).collect();
        for (register, old) in staging.iter().zip(saved) {
            match old {
                Some(value) => self.reg_state.insert(*register, value),
                None => self.reg_state.remove(register),
            };
        }
        for (index, target) in targets.into_iter().enumerate() {
            if target == RZ {
                continue;
            }
            let lo = values[2 * index];
            let hi = values.get(2 * index + 1).copied().unwrap_or(Value::Zero);
            self.write_reg(target, Op::PackHalf2 { lo, hi }, ctx.pred);
        }
        Some(())
    }

    fn sm86_compute_dimension(&self, form: &TextureSampleForm) -> Option<Option<ImageDimension>> {
        if self.stage != ShaderStage::Compute {
            return Some(None);
        }
        Some(Some(match form.tex_type {
            0 => ImageDimension::D1,
            2 => ImageDimension::D2,
            3 if form.bias_reg.is_none() => ImageDimension::D2Array,
            4 => ImageDimension::D3,
            6 if form.dref_reg.is_none() => ImageDimension::Cube,
            7 if form.dref_reg.is_none() && form.bias_reg.is_none() => ImageDimension::CubeArray,
            _ => return None,
        }))
    }

    fn sm86_tex(&mut self, ctx: Ctx, defs: &ValueDefs) -> Option<()> {
        let inst = ctx.inst;
        let (tex_type, coordinates) = match inst.field(61, 3) {
            0 => (0, 1u8),
            1 => (2, 2),
            2 => (4, 3),
            3 => (6, 3),
            5 => (3, 3),
            7 => (7, 4),
            _ => return None,
        };
        let lod = inst.field(87, 3);
        let mask = inst.field(72, 4) as u8;
        if lod > 3 || inst.bit(59) || inst.bit(77) || inst.field(81, 3) != u64::from(PT) || mask == 0 {
            return None;
        }
        let offsets = inst.bit(76);
        let compare = inst.bit(78);
        if offsets && matches!(tex_type, 6 | 7) {
            return None;
        }
        let extras = u8::from(lod >= 2) + u8::from(offsets) + u8::from(compare);
        let components = self.sm86_texture_components(ctx, coordinates, extras)?;
        let mut extra = components[usize::from(coordinates)..].iter().copied();
        let bias_reg = (lod == 2).then(|| extra.next()).flatten();
        let lod_reg = (lod == 3).then(|| extra.next()).flatten();
        let offset_reg = offsets.then(|| extra.next()).flatten();
        let dref_reg = compare.then(|| extra.next()).flatten();
        let form = TextureSampleForm {
            tex_type,
            mask,
            implicit_lod: lod == 0 || lod == 2,
            bias_reg,
            lod_reg,
            offset_reg,
            dref_reg,
        };
        let mut coords = [RZ; 4];
        coords[..usize::from(coordinates)].copy_from_slice(&components[..usize::from(coordinates)]);
        let origin = self.sm86_bound_handle(ctx);
        let compute_handle = self.sm86_compute_dimension(&form)?.map(|dimension| (origin.as_texture_handle(), dimension));
        let tex_id = origin.texture_id();
        let half = inst.field(79, 2) != 0;
        self.sm86_sample_into(ctx, mask, half, |this, targets| {
            this.emit_texture_sample_regs(tex_id, compute_handle, form, coords, targets, Some(defs), ctx.pred)
        })
    }

    fn sm86_tmml(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(72, 4) as u8;
        if inst.field(61, 3) != 1
            || inst.field(76, 4) != 0
            || inst.field(80, 10) != 0
            || inst.bit(59)
            || mask == 0
            || mask & 0b1100 != 0
            || self.stage != ShaderStage::Fragment
        {
            return None;
        }
        let (first, second) = (inst.reg(24), inst.reg(32));
        let second = if second == RZ { first.saturating_add(1) } else { second };
        let u = self.sm86_read(ctx, first)?;
        let v = self.sm86_read(ctx, second)?;
        let handle = self.sm86_bound_handle(ctx).as_texture_handle();
        let targets = self.sm86_texture_targets(ctx, mask)?;
        let mut target = targets.into_iter();
        for component in 0..2u8 {
            if mask >> component & 1 == 0 {
                continue;
            }
            let queried = self.emit_value(Op::TextureQueryLod { handle, u, v, arrayed: false, component });
            let integer = self.emit_value(Op::F2I { src: queried, signed: false, round: 3 });
            self.write_reg(target.next()?, Op::IShl { a: integer, b: Value::ImmU32(8) }, ctx.pred);
        }
        Some(())
    }

    fn sm86_txq(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(72, 4) as u8;
        if mask == 0 || inst.field(59, 5) != 0 || inst.field(76, 14) != 0 || inst.bit(91) || inst.reg(64) != RZ {
            return None;
        }
        let lod = self.sm86_read(ctx, inst.reg(24))?;
        let handle = self.sm86_bound_handle(ctx).as_texture_handle();
        let targets = self.sm86_texture_targets(ctx, mask)?;
        let mut target = targets.into_iter();
        for component in 0..4u8 {
            if mask >> component & 1 == 0 {
                continue;
            }
            self.write_reg(target.next()?, Op::TextureQueryDimension { handle, lod, component }, ctx.pred);
        }
        Some(())
    }

    fn sm86_tld(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let (dimension, coordinates) = match inst.field(61, 3) {
            0 => (ImageDimension::D1, 1u8),
            1 => (ImageDimension::D2, 2),
            2 => (ImageDimension::D3, 3),
            5 => (ImageDimension::D2Array, 3),
            _ => return None,
        };
        let mask = inst.field(72, 4) as u8;
        if mask == 0 || inst.field(87, 3) != 1 || inst.field(76, 3) != 0 || inst.field(81, 3) != u64::from(PT) || inst.bit(59) {
            return None;
        }
        let components = self.sm86_texture_components(ctx, coordinates, 0)?;
        let read = |this: &mut Self, index: usize| this.read_reg(components[index]);
        let (x, y, z) = match dimension {
            ImageDimension::D1 => (read(self, 0), None, None),
            ImageDimension::D2 => (read(self, 0), Some(read(self, 1)), None),
            ImageDimension::D2Array => (read(self, 1), Some(read(self, 2)), Some(read(self, 0))),
            _ => (read(self, 0), Some(read(self, 1)), Some(read(self, 2))),
        };
        let origin = self.sm86_bound_handle(ctx);
        let handle = origin.as_texture_handle();
        let half = inst.field(79, 2) != 0;
        self.sm86_sample_into(ctx, mask, half, |this, targets| {
            let mut target = targets.iter().copied();
            let mut fetches = Vec::new();
            for component in 0..4u8 {
                if mask >> component & 1 == 0 {
                    continue;
                }
                let dst = target.next()?;
                let old = this.read_reg(dst);
                let index = this.program.instructions.len();
                let op = if this.stage == ShaderStage::Compute || dimension == ImageDimension::D2Array {
                    Op::TexelFetchHandle { handle, dimension, x, y, z, component }
                } else {
                    Op::TexelFetch {
                        cbuf_binding: origin.binding,
                        cbuf_word_offset: origin.word_offset,
                        cbuf_secondary_word_offset: None,
                        x,
                        y,
                        z,
                        component,
                    }
                };
                this.write_reg(dst, op, ctx.pred);
                fetches.push((index, old));
            }
            Some(fetches)
        })
    }

    fn sm86_surface(&mut self, ctx: Ctx) -> Option<(ImageDimension, Value, Option<Value>, Option<Value>)> {
        if self.stage != ShaderStage::Compute {
            return None;
        }
        let dimension = match ctx.inst.field(61, 3) {
            0 => ImageDimension::D1,
            1 => ImageDimension::Buffer,
            3 => ImageDimension::D2,
            4 => ImageDimension::D2Array,
            5 => ImageDimension::D3,
            _ => return None,
        };
        let coord = self.sm86_gpr(ctx, ctx.inst.reg(24))?;
        let read = |this: &mut Self, index: u8| this.read_reg(if coord == RZ { RZ } else { coord.saturating_add(index) });
        let x = read(self, 0);
        let (y, z) = match dimension {
            ImageDimension::D1 | ImageDimension::Buffer => (None, None),
            ImageDimension::D2 => (Some(read(self, 1)), None),
            _ => (Some(read(self, 1)), Some(read(self, 2))),
        };
        Some((dimension, x, y, z))
    }

    fn sm86_suld(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(72, 4) as u8;
        if mask == 0 || inst.field(59, 2) != 0 || inst.field(64, 32) & !0xF00 != 0x001e_0000 {
            return None;
        }
        let (dimension, x, y, z) = self.sm86_surface(ctx)?;
        let handle = self.sm86_bound_handle(ctx).as_texture_handle();
        let dst = self.sm86_gpr(ctx, inst.dst())?;
        let mut target = dst;
        for component in 0..4u8 {
            if mask >> component & 1 == 0 {
                continue;
            }
            self.write_reg(target, Op::ImageRead { handle, dimension, x, y, z, component }, ctx.pred);
            if target != RZ {
                target = target.wrapping_add(1);
            }
        }
        Some(())
    }

    fn sm86_sust(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(59, 2) != 0 || inst.field(64, 32) != 0x0010_0f00 {
            return None;
        }
        let (dimension, x, y, z) = self.sm86_surface(ctx)?;
        let handle = self.sm86_bound_handle(ctx).as_texture_handle();
        let data = self.sm86_gpr(ctx, inst.reg(32))?;
        let values = [0u8, 1, 2, 3].map(|index| self.read_reg(if data == RZ { RZ } else { data.wrapping_add(index) }));
        self.program.emit_void_pred(Op::ImageWrite { handle, dimension, x, y, z, values }, ctx.pred);
        Some(())
    }

    fn sm86_sured(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        if inst.field(73, 3) != 0 || inst.field(59, 2) != 0 {
            return None;
        }
        let (dimension, x, y, z) = self.sm86_surface(ctx)?;
        if !matches!(dimension, ImageDimension::Buffer | ImageDimension::D2) {
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
        let handle = self.sm86_bound_handle(ctx).as_texture_handle();
        self.program.emit_void_pred(
            Op::ImageAtomic { handle, dimension, x, y, z, value, op, data_type: ImageAtomicType::U32 },
            ctx.pred,
        );
        Some(())
    }

    fn sm86_tld4(&mut self, ctx: Ctx) -> Option<()> {
        let inst = ctx.inst;
        let mask = inst.field(72, 4) as u8;
        let offsets = inst.field(76, 2) as u8;
        if inst.field(61, 3) != 1
            || offsets == 3
            || inst.bit(78)
            || inst.field(81, 3) != u64::from(PT)
            || inst.bit(59)
            || inst.bit(89)
            || mask == 0
        {
            return None;
        }
        let components = self.sm86_texture_components(ctx, 2, offsets)?;
        let u = self.read_reg(components[0]);
        let v = self.read_reg(components[1]);
        let tex_id = self.sm86_bound_handle(ctx).texture_id();
        let gather_component = inst.field(87, 2) as u8;
        let lanes: [(Value, Value, u8); 4] = match offsets {
            1 => {
                let offset = self.read_reg(components[2]);
                let u = self.shift_gather_coord(tex_id, u, offset, 0, 0);
                let v = self.shift_gather_coord(tex_id, v, offset, 8, 1);
                [(u, v, 0), (u, v, 1), (u, v, 2), (u, v, 3)]
            }
            2 => {
                let words = [self.read_reg(components[2]), self.read_reg(components[3])];
                let mut lanes = [(u, v, 3); 4];
                for (index, lane) in lanes.iter_mut().enumerate() {
                    let offset = words[index / 2];
                    let position = (index as u32 % 2) * 16;
                    lane.0 = self.shift_gather_coord(tex_id, u, offset, position, 0);
                    lane.1 = self.shift_gather_coord(tex_id, v, offset, position + 8, 1);
                }
                lanes
            }
            _ => [(u, v, 0), (u, v, 1), (u, v, 2), (u, v, 3)],
        };
        let half = inst.field(79, 2) != 0;
        self.sm86_sample_into(ctx, mask, half, |this, targets| {
            let mut target = targets.iter().copied();
            let mut samples = Vec::new();
            for (index, (u, v, lane)) in lanes.into_iter().enumerate() {
                if mask >> index & 1 == 0 {
                    continue;
                }
                let dst = target.next()?;
                let old = this.read_reg(dst);
                let position = this.program.instructions.len();
                this.write_reg(
                    dst,
                    Op::GatherTex { tex_id, u, v, array: None, gather_component, lane, dref: None },
                    ctx.pred,
                );
                samples.push((position, old));
            }
            Some(samples)
        })
    }
}
use std::collections::HashMap;

use super::decode::decode_one;
use super::ir::{BoolOp, FComp, ICmp, MufuFunc, Op, Predicate, Program, Value, ValueId};
use super::opcodes::Opcode;
use super::operand::{
    ald_num_elements, attr_slot_ald, attr_slot_ipa, cbuf, decoded_pred, f2f_mods, fadd32i_mods,
    fadd_mods, ffma32i_mods, ffma_mods, float_imm20, fmnmx_is_min, fmnmx_mods, fmul32i_mods,
    fmul_mods, fset_abs_a, fset_abs_b, fset_bop, fset_cmp, fset_neg_a, fset_neg_b, fset_src_pred,
    fset_src_pred_inv, fsetp_abs_a, fsetp_abs_b, fsetp_bop, fsetp_cmp, fsetp_dest_np, fsetp_dest_p,
    fsetp_neg_a, fsetp_neg_b, fsetp_src_pred, fsetp_src_pred_inv, i2f_abs, i2f_int_format, i2f_neg,
    i2f_selector, i2f_signed, imm20, imm32, isetp_bop, isetp_cmp, isetp_dest_np, isetp_dest_p,
    isetp_signed, isetp_src_pred, isetp_src_pred_inv, ldc_ref, ldc_size, ldc_src_reg,
    mufu_func_bits, reg_a, reg_b, reg_c, reg_dest, texs_tex_id, RZ,
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
        Self::with_initial(HashMap::new(), start)
    }

    pub fn with_initial(initial: HashMap<u8, Value>, start: u32) -> Self {
        Self {
            program: Program::with_offset(start),
            reg_state: initial,
            pred_state: HashMap::new(),
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
        let id = self.program.emit_pred(op, Some(r), pred);
        if r != RZ {
            self.reg_state.insert(r, Value::Inst(id));
        }
        id
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

    fn load_cbuf_ldc(&mut self, raw: u64) -> ValueId {
        let r = ldc_ref(raw);
        self.program.emit(
            Op::LoadCbuf {
                binding: r.binding,
                byte_offset: r.byte_offset as u32,
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
                src_pred: fset_src_pred(raw),
                src_pred_inv: fset_src_pred_inv(raw),
            },
            pred,
        );
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

            Opcode::SSY | Opcode::SYNC => {}

            Opcode::FMNMX_reg => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = self.read_reg(reg_b(raw));
                let mods = fmnmx_mods(raw);
                let op = if fmnmx_is_min(raw) {
                    Op::FMin { a, b, mods }
                } else {
                    Op::FMax { a, b, mods }
                };
                self.write_reg(dest, op, pred);
            }
            Opcode::FMNMX_cbuf => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let cb_id = self.load_cbuf(raw);
                let b = Value::Inst(cb_id);
                let mods = fmnmx_mods(raw);
                let op = if fmnmx_is_min(raw) {
                    Op::FMin { a, b, mods }
                } else {
                    Op::FMax { a, b, mods }
                };
                self.write_reg(dest, op, pred);
            }
            Opcode::FMNMX_imm => {
                let dest = reg_dest(raw);
                let a = self.read_reg(reg_a(raw));
                let b = Value::ImmF32(float_imm20(raw));
                let mods = fmnmx_mods(raw);
                let op = if fmnmx_is_min(raw) {
                    Op::FMin { a, b, mods }
                } else {
                    Op::FMax { a, b, mods }
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
                    },
                    pred,
                );
            }

            Opcode::TEX => {
                let tex_type = ((raw >> 28) & 0x7) as u32;
                if tex_type == 2 {
                    let tex_id = texs_tex_id(raw);
                    let coord = reg_a(raw);
                    let u = self.read_reg(coord);
                    let v = self.read_reg(coord.wrapping_add(1));
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
                                component,
                            },
                            pred,
                        );
                        dst = dst.wrapping_add(1);
                    }
                } else {
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::TEX,
                        raw,
                    });
                    self.unimplemented_count += 1;
                }
            }

            Opcode::TEXS | Opcode::TLDS | Opcode::TLD4S => {
                let dest_a = reg_dest(raw);
                let dest_b = ((raw >> 28) & 0xFF) as u8;
                let u = self.read_reg(reg_a(raw));
                let v = self.read_reg(reg_b(raw));
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
                            component,
                        },
                        pred,
                    );
                    store_index += 1;
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
                if src_reg != RZ {
                    log::warn!(
                        "LDC with dynamic index register (Ra != RZ) not yet lifted raw={:#018x} src_reg={}",
                        raw, src_reg,
                    );
                    self.program.emit_void(Op::Unimplemented {
                        opcode: Opcode::LDC,
                        raw,
                    });
                    self.unimplemented_count += 1;
                    return false;
                }
                if size != 4 {
                    log::warn!(
                        "LDC with non-B32 size not yet lifted raw={:#018x} size={}",
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

                let cb_id = self.load_cbuf_ldc(raw);
                self.write_reg(dest, Op::Mov(Value::Inst(cb_id)), pred);
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

            Opcode::KIL => {
                self.program.emit_void_pred(Op::Kill, pred);
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

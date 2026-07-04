use super::super::GpuMappings;
use std::collections::HashMap;

const RZ: u8 = 0xff;
const PT: u8 = 7;

#[derive(Clone, Copy, Default)]
pub struct ComputeTextureState {
    pub tic_pool_gpu_va: u64,
    pub tic_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_limit: u32,
    pub tex_cb_index: u32,
}

pub fn try_execute(
    qmd: &[u32; 0x40],
    code_base: u64,
    texture: ComputeTextureState,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    launch_count: u64,
) -> bool {
    if std::env::var_os("NEXIUM_NO_COMPUTE_CPU").is_some() {
        return false;
    }

    let grid_x = qmd[0x0c] & 0x7fff_ffff;
    let grid_y = qmd[0x0d] & 0xffff;
    let grid_z = qmd[0x0d] >> 16;
    let block_x = qmd[0x12] >> 16;
    let block_y = qmd[0x13] & 0xffff;
    let block_z = qmd[0x13] >> 16;
    let invocations = grid_x as u64
        * grid_y.max(1) as u64
        * grid_z.max(1) as u64
        * block_x.max(1) as u64
        * block_y.max(1) as u64
        * block_z.max(1) as u64;
    if invocations == 0 || invocations > 4096 {
        return false;
    }

    let code_gpu = code_base.wrapping_add(qmd[0x08] as u64);
    let Some(code) = read_gpu_vec(mappings, mem_read, code_gpu, 0x3000) else {
        return false;
    };

    let mut exec = ComputeExec {
        qmd,
        code: &code,
        mappings,
        mem_read,
        mem_write,
        texture,
        tex_cache: HashMap::new(),
        tex_trace: std::env::var_os("NEXIUM_COMPUTE_TEX_TRACE").is_some(),
        tex_logs: 0,
        writes: 0,
        unsupported: None,
    };

    for gz in 0..grid_z.max(1) {
        for gy in 0..grid_y.max(1) {
            for gx in 0..grid_x {
                for lz in 0..block_z.max(1) {
                    for ly in 0..block_y.max(1) {
                        for lx in 0..block_x.max(1) {
                            exec.run_lane([gx, gy, gz], [lx, ly, lz]);
                            if exec.unsupported.is_some() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((pc, raw, opcode)) = exec.unsupported {
        if launch_count < 8 {
            log::warn!(
                "KeplerCompute::cpu unsupported pc={:#x} raw={:#018x} opcode={:?}",
                pc,
                raw,
                opcode
            );
        }
        return false;
    }

    if exec.writes != 0 {
        if launch_count < 8 || std::env::var_os("NEXIUM_COMPUTE_CPU_TRACE").is_some() {
            log::warn!(
                "KeplerCompute::cpu executed program={:#x} invocations={} writes={}",
                code_gpu,
                invocations,
                exec.writes
            );
        }
        true
    } else {
        false
    }
}

struct ComputeExec<'a> {
    qmd: &'a [u32; 0x40],
    code: &'a [u8],
    mappings: &'a GpuMappings,
    mem_read: &'a dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &'a dyn Fn(u64, &[u8]) -> bool,
    texture: ComputeTextureState,
    tex_cache: HashMap<u32, TextureData>,
    tex_trace: bool,
    tex_logs: u32,
    writes: u64,
    unsupported: Option<(usize, u64, nexium_shader::Opcode)>,
}

struct TextureData {
    tic: nexium_gpu::texture::TicEntry,
    rgba: Vec<u8>,
    layers: usize,
    layer_size: usize,
}

impl ComputeExec<'_> {
    fn run_lane(&mut self, group: [u32; 3], local: [u32; 3]) {
        let mut regs = [0u32; 256];
        let mut preds = [false; 8];
        preds[PT as usize] = true;
        let mut pc = 0usize;
        let mut steps = 0u32;

        while pc + 8 <= self.code.len() && steps < 200_000 {
            steps += 1;
            if pc % 0x20 == 0 {
                pc += 8;
                continue;
            }
            let raw = u64::from_le_bytes(self.code[pc..pc + 8].try_into().unwrap());
            if raw == 0 {
                pc += 8;
                continue;
            }
            let Some(decoded) = nexium_shader::decode_one(raw) else {
                self.unsupported = Some((pc, raw, nexium_shader::Opcode::NOP));
                return;
            };
            let active = pred_active(raw, &preds);
            if !active {
                pc += 8;
                continue;
            }

            use nexium_shader::Opcode::*;
            match decoded.opcode {
                NOP | DEPBAR | SSY | SYNC => {}
                EXIT => return,
                BRA | JMP => {
                    pc = bra_target(pc, raw);
                    continue;
                }
                S2R => {
                    let val = match bits(raw, 20, 27) as u32 {
                        0 => local[0] & 31,
                        0x20 => local[0] | (local[1] << 16) | (local[2] << 26),
                        0x21 => local[0],
                        0x22 => local[1],
                        0x23 => local[2],
                        0x25 => group[0],
                        0x26 => group[1],
                        0x27 => group[2],
                        0x28 => self.block_dim_word(),
                        _ => 0,
                    };
                    set_reg(&mut regs, reg_dest(raw), val);
                }
                MOV_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                MOV_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                MOV_imm => set_reg(&mut regs, reg_dest(raw), imm20(raw) as u32),
                MOV32I => set_reg(&mut regs, reg_dest(raw), imm32(raw)),
                FADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fadd(&mut regs, raw, b);
                }
                FADD_cbuf => self.fadd(&mut regs, raw, self.cbuf_u32(raw)),
                FADD_imm => self.fadd(&mut regs, raw, float_imm20(raw).to_bits()),
                FMUL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fmul(&mut regs, raw, b);
                }
                FMUL_cbuf => self.fmul(&mut regs, raw, self.cbuf_u32(raw)),
                FMUL_imm => self.fmul(&mut regs, raw, float_imm20(raw).to_bits()),
                FMUL32I => self.fmul(&mut regs, raw, imm32(raw)),
                FADD32I => self.fadd(&mut regs, raw, imm32(raw)),
                FFMA_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_cr => {
                    let b = self.cbuf_u32(raw);
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_rc => {
                    let b = get_reg(&regs, reg_c(raw));
                    let c = self.cbuf_u32(raw);
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_imm => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, float_imm20(raw).to_bits(), c);
                }
                IADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iadd(&mut regs, raw, b);
                }
                IADD_cbuf => self.iadd(&mut regs, raw, self.cbuf_u32(raw)),
                IADD_imm => self.iadd(&mut regs, raw, imm20(raw) as u32),
                IADD32I => self.iadd(&mut regs, raw, imm32(raw)),
                IADD3_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, b, c, true);
                }
                IADD3_cbuf => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, self.cbuf_u32(raw), c, false);
                }
                IADD3_imm => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, imm20(raw) as u32, c, false);
                }
                ISCADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iscadd(&mut regs, raw, b);
                }
                ISCADD_cbuf => self.iscadd(&mut regs, raw, self.cbuf_u32(raw)),
                ISCADD_imm => self.iscadd(&mut regs, raw, imm20(raw) as u32),
                XMAD_reg => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 52) as u8,
                        bit(raw, 35) as u8,
                        bit(raw, 36),
                        bit(raw, 37),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_rc => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = get_reg(&regs, reg_c(raw));
                    let c = self.cbuf_u32(raw);
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 51) as u8,
                        bit(raw, 52) as u8,
                        false,
                        false,
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_cr => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = self.cbuf_u32(raw);
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 51) as u8,
                        bit(raw, 52) as u8,
                        bit(raw, 55),
                        bit(raw, 56),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_imm => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = bits(raw, 20, 35) as u32;
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 52) as u8,
                        0,
                        bit(raw, 36),
                        bit(raw, 37),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                SHL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.ishl(&mut regs, raw, b);
                }
                SHL_cbuf => self.ishl(&mut regs, raw, self.cbuf_u32(raw)),
                SHL_imm => self.ishl(&mut regs, raw, imm20(raw) as u32),
                SHR_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.ishr(&mut regs, raw, b);
                }
                SHR_cbuf => self.ishr(&mut regs, raw, self.cbuf_u32(raw)),
                SHR_imm => self.ishr(&mut regs, raw, imm20(raw) as u32),
                LOP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.lop(
                        &mut regs,
                        raw,
                        b,
                        lop_op(raw),
                        lop_not_a(raw),
                        lop_not_b(raw),
                    );
                }
                LOP_cbuf => self.lop(
                    &mut regs,
                    raw,
                    self.cbuf_u32(raw),
                    lop_op(raw),
                    lop_not_a(raw),
                    lop_not_b(raw),
                ),
                LOP_imm => self.lop(&mut regs, raw, imm20(raw) as u32, lop_op(raw), false, false),
                LOP32I => self.lop(
                    &mut regs,
                    raw,
                    imm32(raw),
                    lop32i_op(raw),
                    lop32i_not_a(raw),
                    lop32i_not_b(raw),
                ),
                F2I_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    self.f2i(&mut regs, raw, src);
                }
                F2I_cbuf => self.f2i(&mut regs, raw, self.cbuf_u32(raw)),
                F2I_imm => self.f2i(&mut regs, raw, float_imm20(raw).to_bits()),
                I2F_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    self.i2f(&mut regs, raw, src);
                }
                I2F_cbuf => self.i2f(&mut regs, raw, self.cbuf_u32(raw)),
                I2F_imm => self.i2f(&mut regs, raw, imm20(raw) as u32),
                ISET_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iset(&mut regs, raw, b);
                }
                ISET_cbuf => self.iset(&mut regs, raw, self.cbuf_u32(raw)),
                ISET_imm => self.iset(&mut regs, raw, imm20(raw) as u32),
                ISETP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.isetp(&mut regs, &mut preds, raw, b);
                }
                ISETP_cbuf => self.isetp(&mut regs, &mut preds, raw, self.cbuf_u32(raw)),
                ISETP_imm => self.isetp(&mut regs, &mut preds, raw, imm20(raw) as u32),
                FSETP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fsetp(&mut regs, &mut preds, raw, b);
                }
                FSETP_cbuf => self.fsetp(&mut regs, &mut preds, raw, self.cbuf_u32(raw)),
                FSETP_imm => self.fsetp(&mut regs, &mut preds, raw, float_imm20(raw).to_bits()),
                PSETP => self.psetp(&mut preds, raw),
                SEL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.sel(&mut regs, &preds, raw, b);
                }
                SEL_cbuf => self.sel(&mut regs, &preds, raw, self.cbuf_u32(raw)),
                SEL_imm => self.sel(&mut regs, &preds, raw, imm20(raw) as u32),
                MUFU => self.mufu(&mut regs, raw),
                RRO_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                RRO_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_imm => set_reg(&mut regs, reg_dest(raw), imm20(raw) as u32),
                LDG => self.ldg(&mut regs, raw),
                STG => self.stg(&regs, raw),
                TEXS | TLDS | TLD4S => self.texs(&mut regs, raw),
                other => {
                    self.unsupported = Some((pc, raw, other));
                    return;
                }
            }
            pc += 8;
        }
    }

    fn block_dim_word(&self) -> u32 {
        (self.qmd[0x12] >> 16) | ((self.qmd[0x13] & 0xffff) << 16)
    }

    fn cbuf_u32(&self, raw: u64) -> u32 {
        let c = nexium_shader::cbuf(raw);
        self.cbuf_slot_u32(c.binding, c.byte_offset)
    }

    fn cbuf_slot_u32(&self, slot: u8, offset: u32) -> u32 {
        let base = 0x1d + slot as usize * 2;
        if base + 1 >= self.qmd.len() {
            return 0;
        }
        let lo = self.qmd[base] as u64;
        let hi_size = self.qmd[base + 1];
        let gpu = (((hi_size & 0xff) as u64) << 32) | lo;
        self.read_gpu_u32(gpu.wrapping_add(offset as u64))
    }

    fn read_gpu_u32(&self, gpu: u64) -> u32 {
        let Some(cpu) = map_gpu(self.mappings, gpu) else {
            return 0;
        };
        let mut b = [0u8; 4];
        if (self.mem_read)(cpu, &mut b) {
            u32::from_le_bytes(b)
        } else {
            0
        }
    }

    fn write_gpu_u32(&mut self, gpu: u64, value: u32) {
        let Some(cpu) = map_gpu(self.mappings, gpu) else {
            return;
        };
        if (self.mem_write)(cpu, &value.to_le_bytes()) {
            nexium_gpu::tex_invalidate::bump_region(gpu, 4);
            self.writes += 1;
        }
    }

    fn ldg(&self, regs: &mut [u32; 256], raw: u64) {
        let addr_reg = reg_a(raw);
        let base = addr_pair(regs, addr_reg);
        let offset = ldg_offset(raw);
        let count = match ldg_size(raw) {
            5 => 2,
            6 | 7 => 4,
            _ => 1,
        };
        let dest = reg_dest(raw);
        for i in 0..count {
            let gpu = base
                .wrapping_add(offset as u64)
                .wrapping_add((i * 4) as u64);
            set_reg(regs, dest.wrapping_add(i as u8), self.read_gpu_u32(gpu));
        }
    }

    fn stg(&mut self, regs: &[u32; 256], raw: u64) {
        let base = addr_pair(regs, reg_a(raw));
        let gpu = base.wrapping_add(ldg_offset(raw) as u64);
        self.write_gpu_u32(gpu, get_reg(regs, reg_dest(raw)));
    }

    fn fadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 46) {
            a = a.abs();
        }
        if bit(raw, 48) {
            a = -a;
        }
        if bit(raw, 49) {
            b = b.abs();
        }
        if bit(raw, 45) {
            b = -b;
        }
        set_reg(regs, reg_dest(raw), fsat(a + b, bit(raw, 50)).to_bits());
    }

    fn fmul(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 48) {
            b = -b;
        }
        set_reg(regs, reg_dest(raw), fsat(a * b, bit(raw, 50)).to_bits());
    }

    fn ffma(&self, regs: &mut [u32; 256], raw: u64, b: u32, c: u32) {
        let a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        let mut c = f32::from_bits(c);
        if bit(raw, 48) {
            b = -b;
        }
        if bit(raw, 49) {
            c = -c;
        }
        set_reg(
            regs,
            reg_dest(raw),
            fsat(a.mul_add(b, c), bit(raw, 50)).to_bits(),
        );
    }

    fn iadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if bit(raw, 49) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 48) {
            b = (!b).wrapping_add(1);
        }
        set_reg(regs, reg_dest(raw), a.wrapping_add(b));
    }

    fn iadd3(&self, regs: &mut [u32; 256], raw: u64, b: u32, c: u32, use_halves: bool) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        let mut c = c;
        if use_halves {
            a = half(a, bits(raw, 35, 36) as u8);
            b = half(b, bits(raw, 33, 34) as u8);
            c = half(c, bits(raw, 31, 32) as u8);
        }
        if bit(raw, 51) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 50) {
            b = (!b).wrapping_add(1);
        }
        if bit(raw, 49) {
            c = (!c).wrapping_add(1);
        }
        let mut v = a.wrapping_add(b);
        match bits(raw, 37, 38) {
            1 => v >>= 16,
            2 => v <<= 16,
            _ => {}
        }
        set_reg(regs, reg_dest(raw), v.wrapping_add(c));
    }

    fn iscadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if bit(raw, 49) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 48) {
            b = (!b).wrapping_add(1);
        }
        set_reg(
            regs,
            reg_dest(raw),
            (a << bits(raw, 39, 43)).wrapping_add(b),
        );
    }

    fn ishl(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        set_reg(regs, reg_dest(raw), get_reg(regs, reg_a(raw)) << (b & 31));
    }

    fn ishr(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let v = if bit(raw, 48) {
            ((a as i32) >> (b & 31)) as u32
        } else {
            a >> (b & 31)
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn lop(&self, regs: &mut [u32; 256], raw: u64, b: u32, op: u64, not_a: bool, not_b: bool) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if not_a {
            a = !a;
        }
        if not_b {
            b = !b;
        }
        let v = match op & 3 {
            0 => a & b,
            1 => a | b,
            2 => a ^ b,
            _ => b,
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn f2i(&self, regs: &mut [u32; 256], raw: u64, src: u32) {
        let f = f32::from_bits(src);
        let v = if bit(raw, 12) {
            f as i32 as u32
        } else {
            f as u32
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn i2f(&self, regs: &mut [u32; 256], raw: u64, src: u32) {
        let int_format = bits(raw, 10, 11) as u8;
        let selector = bits(raw, 41, 42) as u32;
        let extracted = match int_format {
            0 => (src >> (selector * 8)) & 0xff,
            1 => (src >> (selector * 8)) & 0xffff,
            _ => src,
        };
        let mut f = if bit(raw, 13) {
            sign_extend(
                extracted,
                match int_format {
                    0 => 8,
                    1 => 16,
                    _ => 32,
                },
            ) as f32
        } else {
            extracted as f32
        };
        if bit(raw, 49) {
            f = f.abs();
        }
        if bit(raw, 45) {
            f = -f;
        }
        set_reg(regs, reg_dest(raw), f.to_bits());
    }

    fn iset(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let pass = icmp(bits(raw, 49, 51), bit(raw, 48), a, b);
        let v = if bit(raw, 44) {
            if pass {
                1.0f32.to_bits()
            } else {
                0
            }
        } else if pass {
            u32::MAX
        } else {
            0
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn isetp(&self, regs: &mut [u32; 256], preds: &mut [bool; 8], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let cmp = icmp(bits(raw, 49, 51), bit(raw, 48), a, b);
        let src = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let yes = boolop(bits(raw, 45, 46), cmp, src);
        let no = boolop(bits(raw, 45, 46), !cmp, src);
        set_pred(preds, bits(raw, 3, 5) as u8, yes);
        set_pred(preds, bits(raw, 0, 2) as u8, no);
    }

    fn fsetp(&self, regs: &mut [u32; 256], preds: &mut [bool; 8], raw: u64, b: u32) {
        let mut a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 7) {
            a = a.abs();
        }
        if bit(raw, 43) {
            a = -a;
        }
        if bit(raw, 44) {
            b = b.abs();
        }
        if bit(raw, 6) {
            b = -b;
        }
        let cmp = fcmp(bits(raw, 48, 51), a, b);
        let src = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let yes = boolop(bits(raw, 45, 46), cmp, src);
        let no = boolop(bits(raw, 45, 46), !cmp, src);
        set_pred(preds, bits(raw, 3, 5) as u8, yes);
        set_pred(preds, bits(raw, 0, 2) as u8, no);
    }

    fn psetp(&self, preds: &mut [bool; 8], raw: u64) {
        let pa = pred_value(preds, bits(raw, 12, 14) as u8, bit(raw, 15));
        let pb = pred_value(preds, bits(raw, 29, 31) as u8, bit(raw, 32));
        let pc = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let lhs_a = boolop(bits(raw, 24, 25), pa, pb);
        let lhs_b = boolop(bits(raw, 24, 25), !pa, pb);
        let result_a = boolop(bits(raw, 45, 46), lhs_a, pc);
        let result_b = boolop(bits(raw, 45, 46), lhs_b, pc);
        set_pred(preds, bits(raw, 3, 5) as u8, result_a);
        set_pred(preds, bits(raw, 0, 2) as u8, result_b);
    }

    fn sel(&self, regs: &mut [u32; 256], preds: &[bool; 8], raw: u64, b: u32) {
        let p = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let v = if p { get_reg(regs, reg_a(raw)) } else { b };
        set_reg(regs, reg_dest(raw), v);
    }

    fn mufu(&self, regs: &mut [u32; 256], raw: u64) {
        let x = f32::from_bits(get_reg(regs, reg_a(raw)));
        let y = match bits(raw, 20, 23) {
            0 => x.cos(),
            1 => x.sin(),
            2 => x.exp2(),
            3 => x.log2(),
            4 => 1.0 / x,
            5 => 1.0 / x.sqrt(),
            8 => x.sqrt(),
            _ => 0.0,
        };
        set_reg(regs, reg_dest(raw), y.to_bits());
    }

    fn texs(&mut self, regs: &mut [u32; 256], raw: u64) {
        let dest_a = reg_dest(raw);
        let dest_b = bits(raw, 28, 35) as u8;
        let swizzle = bits(raw, 50, 52) as usize;
        let mask = if dest_b == RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][swizzle.min(7)]
        } else {
            [7, 11, 13, 14, 15].get(swizzle).copied().unwrap_or(1)
        };
        let sample = self.sample_tex(regs, raw).unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let mut store = 0;
        for component in 0..4 {
            if (mask >> component) & 1 == 0 {
                continue;
            }
            let dst = match store {
                0 => dest_a,
                1 => dest_a.wrapping_add(1),
                2 => dest_b,
                _ => dest_b.wrapping_add(1),
            };
            set_reg(regs, dst, sample[component].to_bits());
            store += 1;
        }
    }

    fn sample_tex(&mut self, regs: &[u32; 256], raw: u64) -> Option<[f32; 4]> {
        let handle_offset = (bits(raw, 36, 48) as u32).wrapping_mul(4);
        let handle = self.cbuf_slot_u32(self.texture.tex_cb_index as u8, handle_offset);
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let tic_index = if linked_tsc {
            handle
        } else {
            handle & 0x000f_ffff
        };
        if tic_index > self.texture.tic_limit || self.texture.tic_pool_gpu_va == 0 {
            return None;
        }
        if !self.tex_cache.contains_key(&tic_index) {
            let data = self.load_texture(tic_index)?;
            self.tex_cache.insert(tic_index, data);
        }
        let data = self.tex_cache.get(&tic_index)?;
        let encoding = bits(raw, 53, 56);
        let reg_a_id = reg_a(raw);
        let reg_b_id = reg_b(raw);
        let fx = |r| f32::from_bits(get_reg(regs, r));
        let (u, v, layer_f) = match encoding {
            7 | 8 | 9 => {
                let layer = (get_reg(regs, reg_a_id) & 0xffff) as f32;
                (fx(reg_a_id.wrapping_add(1)), fx(reg_b_id), layer)
            }
            10 | 11 | 12 | 13 => (fx(reg_a_id), fx(reg_a_id.wrapping_add(1)), fx(reg_b_id)),
            _ => (fx(reg_a_id), fx(reg_b_id), 0.0),
        };
        let x = coord_to_index(u, data.tic.width, data.tic.normalized_coords);
        let y = coord_to_index(v, data.tic.height, data.tic.normalized_coords);
        let layer = if data.layers <= 1 {
            0
        } else if data.tic.texture_type == 2 && data.tic.normalized_coords {
            coord_to_index(layer_f, data.layers as u32, true)
        } else {
            layer_f.floor().clamp(0.0, (data.layers - 1) as f32) as usize
        };
        let off = layer
            .saturating_mul(data.layer_size)
            .saturating_add((y * data.tic.width as usize + x) * 4);
        if off + 4 > data.rgba.len() {
            return None;
        }
        let px = [
            data.rgba[off],
            data.rgba[off + 1],
            data.rgba[off + 2],
            data.rgba[off + 3],
        ];
        if self.tex_trace && self.tex_logs < 16 {
            log::warn!(
                "KeplerCompute::texs handle={:#x} tic={} fmt={:?} {}x{}x{} type={} coord=({:.4},{:.4},{:.4}) px={:?}",
                handle,
                tic_index,
                data.tic.format,
                data.tic.width,
                data.tic.height,
                data.layers,
                data.tic.texture_type,
                u,
                v,
                layer_f,
                px
            );
            self.tex_logs += 1;
        }
        let px = apply_swizzle(px, data.tic.swizzle);
        Some([
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
            px[3] as f32 / 255.0,
        ])
    }

    fn load_texture(&self, tic_index: u32) -> Option<TextureData> {
        let tic_gpu = self
            .texture
            .tic_pool_gpu_va
            .wrapping_add((tic_index as u64).saturating_mul(32));
        let tic_cpu = map_gpu(self.mappings, tic_gpu)?;
        let mut tic_raw = [0u8; 32];
        if !(self.mem_read)(tic_cpu, &mut tic_raw) {
            return None;
        }
        let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw)?;
        let pitch_size = tic.format.linear_size(tic.width, tic.height);
        let layers = texture_layer_count(&tic);
        let layer_read_size = texture_layer_read_size(&tic, pitch_size);
        let read_size = layer_read_size.saturating_mul(layers);
        let tex_cpu = map_gpu(self.mappings, tic.gpu_va)?;
        let mut raw = vec![0u8; read_size];
        if !(self.mem_read)(tex_cpu, &mut raw) {
            return None;
        }
        let rgba = decode_texture_layers(&raw, &tic, pitch_size, layer_read_size, layers);
        let layer_size = tic.width as usize * tic.height as usize * 4;
        Some(TextureData {
            tic,
            rgba,
            layers,
            layer_size,
        })
    }
}

fn texture_layer_count(tic: &nexium_gpu::texture::TicEntry) -> usize {
    match tic.texture_type {
        2 => tic.depth.max(1) as usize,
        5 => tic.base_layer.saturating_add(tic.depth).max(1) as usize,
        _ => 1,
    }
}

fn texture_layer_read_size(tic: &nexium_gpu::texture::TicEntry, pitch_size: usize) -> usize {
    if tic.is_block_linear {
        tic.format
            .block_linear_size(tic.width, tic.height, tic.block_height_log2)
            .max(pitch_size)
    } else {
        pitch_size
    }
}

fn decode_texture_layers(
    raw: &[u8],
    tic: &nexium_gpu::texture::TicEntry,
    pitch_size: usize,
    layer_read_size: usize,
    layers: usize,
) -> Vec<u8> {
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let layer_rgba_size = tic.width as usize * tic.height as usize * 4;
    let mut out = Vec::with_capacity(layer_rgba_size.saturating_mul(layers));
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_read_size);
        if start >= raw.len() {
            out.resize(out.len() + layer_rgba_size, 0);
            continue;
        }
        let end = (start + layer_read_size).min(raw.len());
        let layer_raw = &raw[start..end];
        let linear = if effective_block_linear {
            let (storage_width, storage_height, bpp) =
                tic.format.storage_extent(tic.width, tic.height);
            nexium_gpu::texture::unswizzle_block_linear(
                layer_raw,
                storage_width,
                storage_height,
                bpp,
                tic.block_height_log2,
            )
        } else if layer_raw.len() >= pitch_size {
            layer_raw[..pitch_size].to_vec()
        } else {
            layer_raw.to_vec()
        };
        let mut decoded =
            nexium_gpu::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        out.extend(decoded);
    }
    out.resize(layer_rgba_size.saturating_mul(layers), 0);
    out
}

fn coord_to_index(v: f32, size: u32, normalized: bool) -> usize {
    if size == 0 {
        return 0;
    }
    let max = (size - 1) as f32;
    let f = if normalized { v * size as f32 } else { v };
    f.floor().clamp(0.0, max) as usize
}

fn apply_swizzle(src: [u8; 4], swizzle: [nexium_gpu::texture::SwizzleSource; 4]) -> [u8; 4] {
    fn one(src: [u8; 4], s: nexium_gpu::texture::SwizzleSource) -> u8 {
        match s {
            nexium_gpu::texture::SwizzleSource::Zero => 0,
            nexium_gpu::texture::SwizzleSource::R => src[0],
            nexium_gpu::texture::SwizzleSource::G => src[1],
            nexium_gpu::texture::SwizzleSource::B => src[2],
            nexium_gpu::texture::SwizzleSource::A => src[3],
            nexium_gpu::texture::SwizzleSource::One => 255,
            nexium_gpu::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    [
        one(src, swizzle[0]),
        one(src, swizzle[1]),
        one(src, swizzle[2]),
        one(src, swizzle[3]),
    ]
}

fn read_gpu_vec(
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    gpu: u64,
    len: usize,
) -> Option<Vec<u8>> {
    let cpu = map_gpu(mappings, gpu)?;
    let mut bytes = vec![0u8; len];
    mem_read(cpu, &mut bytes).then_some(bytes)
}

fn map_gpu(mappings: &GpuMappings, gpu: u64) -> Option<u64> {
    mappings
        .cpu_address_for(gpu)
        .or_else(|| mappings.cpu_address_for_any32(gpu).map(|(_, cpu, _)| cpu))
}

fn get_reg(regs: &[u32; 256], r: u8) -> u32 {
    if r == RZ {
        0
    } else {
        regs[r as usize]
    }
}

fn set_reg(regs: &mut [u32; 256], r: u8, v: u32) {
    if r != RZ {
        regs[r as usize] = v;
    }
}

fn addr_pair(regs: &[u32; 256], r: u8) -> u64 {
    if r == RZ {
        0
    } else {
        get_reg(regs, r) as u64 | ((get_reg(regs, r.wrapping_add(1)) as u64) << 32)
    }
}

fn bits(insn: u64, lo: u32, hi: u32) -> u64 {
    (insn >> lo) & ((1u64 << (hi - lo + 1)) - 1)
}

fn bit(insn: u64, n: u32) -> bool {
    ((insn >> n) & 1) != 0
}

fn xmad_half(v: u32, half: u8, signed: bool) -> u32 {
    let x = (v >> (16 * half as u32)) & 0xffff;
    if signed {
        (((x as i32) << 16) >> 16) as u32
    } else {
        x
    }
}

fn xmad_eval(
    raw: u64,
    a_full: u32,
    src_b: u32,
    src_c: u32,
    select: u8,
    half_b: u8,
    psl: bool,
    mrg: bool,
) -> u32 {
    let a = xmad_half(a_full, bit(raw, 53) as u8, bit(raw, 48));
    let b = xmad_half(src_b, half_b, bit(raw, 49));
    let mut product = a.wrapping_mul(b);
    if psl {
        product <<= 16;
    }
    let c = match select {
        1 => src_c & 0xffff,
        2 => src_c >> 16,
        4 => (src_b << 16).wrapping_add(src_c),
        _ => src_c,
    };
    let mut result = product.wrapping_add(c);
    if mrg {
        result = (result & 0xffff) | (src_b << 16);
    }
    result
}

fn reg_dest(insn: u64) -> u8 {
    bits(insn, 0, 7) as u8
}

fn reg_a(insn: u64) -> u8 {
    bits(insn, 8, 15) as u8
}

fn reg_b(insn: u64) -> u8 {
    bits(insn, 20, 27) as u8
}

fn reg_c(insn: u64) -> u8 {
    bits(insn, 39, 46) as u8
}

fn imm20(insn: u64) -> i32 {
    let raw = bits(insn, 20, 38) as i32;
    if bit(insn, 56) {
        raw - (1 << 19)
    } else {
        raw
    }
}

fn imm32(insn: u64) -> u32 {
    bits(insn, 20, 51) as u32
}

fn float_imm20(insn: u64) -> f32 {
    let value = (bits(insn, 20, 38) as u32) << 12;
    let sign = if bit(insn, 56) { 1u32 << 31 } else { 0 };
    f32::from_bits(value | sign)
}

fn ldg_offset(insn: u64) -> i32 {
    let raw = bits(insn, 20, 43) as u32;
    if raw & (1 << 23) != 0 {
        (raw | 0xff00_0000) as i32
    } else {
        raw as i32
    }
}

fn ldg_size(insn: u64) -> u32 {
    bits(insn, 48, 50) as u32
}

fn lop_op(insn: u64) -> u64 {
    bits(insn, 41, 42)
}

fn lop_not_a(insn: u64) -> bool {
    bit(insn, 39)
}

fn lop_not_b(insn: u64) -> bool {
    bit(insn, 40)
}

fn lop32i_op(insn: u64) -> u64 {
    bits(insn, 53, 54)
}

fn lop32i_not_a(insn: u64) -> bool {
    bit(insn, 55)
}

fn lop32i_not_b(insn: u64) -> bool {
    bit(insn, 56)
}

fn pred_active(raw: u64, preds: &[bool; 8]) -> bool {
    pred_value(preds, bits(raw, 16, 18) as u8, bit(raw, 19))
}

fn pred_value(preds: &[bool; 8], pred: u8, neg: bool) -> bool {
    let v = if pred == PT {
        true
    } else {
        preds[pred as usize]
    };
    if neg {
        !v
    } else {
        v
    }
}

fn set_pred(preds: &mut [bool; 8], pred: u8, v: bool) {
    if pred != PT {
        preds[pred as usize] = v;
    }
}

fn boolop(op: u64, a: bool, b: bool) -> bool {
    match op & 3 {
        0 => a & b,
        1 => a | b,
        _ => a ^ b,
    }
}

fn icmp(cmp: u64, signed: bool, a: u32, b: u32) -> bool {
    match cmp & 7 {
        0 => false,
        1 => {
            if signed {
                (a as i32) < (b as i32)
            } else {
                a < b
            }
        }
        2 => a == b,
        3 => {
            if signed {
                (a as i32) <= (b as i32)
            } else {
                a <= b
            }
        }
        4 => {
            if signed {
                (a as i32) > (b as i32)
            } else {
                a > b
            }
        }
        5 => a != b,
        6 => {
            if signed {
                (a as i32) >= (b as i32)
            } else {
                a >= b
            }
        }
        _ => true,
    }
}

fn fcmp(cmp: u64, a: f32, b: f32) -> bool {
    let ord = !a.is_nan() && !b.is_nan();
    match cmp & 0xf {
        0 => false,
        1 => ord && a < b,
        2 => ord && a == b,
        3 => ord && a <= b,
        4 => ord && a > b,
        5 => ord && a != b,
        6 => ord && a >= b,
        7 => ord,
        8 => !ord,
        9 => !ord || a < b,
        10 => !ord || a == b,
        11 => !ord || a <= b,
        12 => !ord || a > b,
        13 => !ord || a != b,
        14 => !ord || a >= b,
        _ => true,
    }
}

fn fsat(v: f32, sat: bool) -> f32 {
    if sat {
        v.clamp(0.0, 1.0)
    } else {
        v
    }
}

fn half(v: u32, half: u8) -> u32 {
    match half {
        1 => v & 0xffff,
        2 => (v >> 16) & 0xffff,
        _ => v,
    }
}

fn sign_extend(v: u32, bits: u32) -> i32 {
    if bits >= 32 {
        v as i32
    } else {
        let shift = 32 - bits;
        ((v << shift) as i32) >> shift
    }
}

fn bra_target(pc: usize, raw: u64) -> usize {
    let raw_24 = bits(raw, 20, 43) as u32;
    let signed = if raw_24 & 0x0080_0000 != 0 {
        (raw_24 | 0xff00_0000) as i32
    } else {
        raw_24 as i32
    };
    (pc as i64 + signed as i64 + 8) as usize
}

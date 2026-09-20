use crate::nce_layout::*;

pub const SYSREG_TPIDR_EL0: u32 = 0x5E82;
pub const SYSREG_TPIDRRO_EL0: u32 = 0x5E83;
pub const SYSREG_CNTFRQ_EL0: u32 = 0x5F00;
pub const SYSREG_CNTPCT_EL0: u32 = 0x5F01;
pub const SYSREG_CNTVCT_EL0: u32 = 0x5F02;
pub const SYSREG_FPCR: u32 = 0x5A20;
pub const SYSREG_FPSR: u32 = 0x5A21;
pub const SYSREG_NZCV: u32 = 0x5A10;

pub const MAX_BRANCH_REACH: u64 = 128 << 20;

pub mod a64 {
    pub const SP: u32 = 31;
    pub const XZR: u32 = 31;

    fn imm26(offset: i64) -> u32 {
        debug_assert!(offset % 4 == 0 && (-(1 << 27)..(1 << 27)).contains(&offset));
        ((offset >> 2) as u32) & 0x03FF_FFFF
    }

    fn imm19(offset: i64) -> u32 {
        debug_assert!(offset % 4 == 0 && (-(1 << 20)..(1 << 20)).contains(&offset));
        ((offset >> 2) as u32) & 0x7FFFF
    }

    pub fn b(offset: i64) -> u32 {
        0x1400_0000 | imm26(offset)
    }

    pub fn bl(offset: i64) -> u32 {
        0x9400_0000 | imm26(offset)
    }

    pub fn ret() -> u32 {
        0xD65F_03C0
    }

    pub fn br(rn: u32) -> u32 {
        0xD61F_0000 | (rn << 5)
    }

    pub fn nop() -> u32 {
        0xD503_201F
    }

    pub fn ldr_x(rt: u32, rn: u32, offset: usize) -> u32 {
        debug_assert!(offset % 8 == 0 && offset / 8 < 4096);
        0xF940_0000 | (((offset / 8) as u32) << 10) | (rn << 5) | rt
    }

    pub fn str_x(rt: u32, rn: u32, offset: usize) -> u32 {
        debug_assert!(offset % 8 == 0 && offset / 8 < 4096);
        0xF900_0000 | (((offset / 8) as u32) << 10) | (rn << 5) | rt
    }

    pub fn ldr_w(rt: u32, rn: u32, offset: usize) -> u32 {
        debug_assert!(offset % 4 == 0 && offset / 4 < 4096);
        0xB940_0000 | (((offset / 4) as u32) << 10) | (rn << 5) | rt
    }

    pub fn str_w(rt: u32, rn: u32, offset: usize) -> u32 {
        debug_assert!(offset % 4 == 0 && offset / 4 < 4096);
        0xB900_0000 | (((offset / 4) as u32) << 10) | (rn << 5) | rt
    }

    fn imm7(offset: i64, scale: i64) -> u32 {
        debug_assert!(offset % scale == 0 && (-64..64).contains(&(offset / scale)));
        (((offset / scale) as u32) & 0x7F) << 15
    }

    pub fn stp_x(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xA900_0000 | imm7(offset, 8) | (rt2 << 10) | (rn << 5) | rt
    }

    pub fn ldp_x(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xA940_0000 | imm7(offset, 8) | (rt2 << 10) | (rn << 5) | rt
    }

    pub fn stp_q(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xAD00_0000 | imm7(offset, 16) | (rt2 << 10) | (rn << 5) | rt
    }

    pub fn ldp_q(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xAD40_0000 | imm7(offset, 16) | (rt2 << 10) | (rn << 5) | rt
    }

    pub fn stp_x_pre(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xA980_0000 | imm7(offset, 8) | (rt2 << 10) | (rn << 5) | rt
    }

    pub fn ldp_x_post(rt: u32, rt2: u32, rn: u32, offset: i64) -> u32 {
        0xA8C0_0000 | imm7(offset, 8) | (rt2 << 10) | (rn << 5) | rt
    }

    fn imm9(offset: i64) -> u32 {
        debug_assert!((-256..256).contains(&offset));
        ((offset as u32) & 0x1FF) << 12
    }

    pub fn str_x_pre(rt: u32, rn: u32, offset: i64) -> u32 {
        0xF800_0C00 | imm9(offset) | (rn << 5) | rt
    }

    pub fn ldr_x_post(rt: u32, rn: u32, offset: i64) -> u32 {
        0xF840_0400 | imm9(offset) | (rn << 5) | rt
    }

    pub fn mrs(rt: u32, sysreg: u32) -> u32 {
        0xD530_0000 | (sysreg << 5) | rt
    }

    pub fn msr(sysreg: u32, rt: u32) -> u32 {
        0xD510_0000 | (sysreg << 5) | rt
    }

    pub fn movz_w(rd: u32, imm16: u32) -> u32 {
        0x5280_0000 | (imm16 << 5) | rd
    }

    pub fn movz_x(rd: u32, imm16: u32) -> u32 {
        0xD280_0000 | (imm16 << 5) | rd
    }

    pub fn movk_x(rd: u32, imm16: u32, shift: u32) -> u32 {
        debug_assert!(shift % 16 == 0 && shift < 64);
        0xF280_0000 | ((shift / 16) << 21) | (imm16 << 5) | rd
    }

    pub fn add_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
        debug_assert!(imm12 < 4096);
        0x9100_0000 | (imm12 << 10) | (rn << 5) | rd
    }

    pub fn mov_sp_from(rn: u32) -> u32 {
        add_imm(SP, rn, 0)
    }

    pub fn mov_from_sp(rd: u32) -> u32 {
        add_imm(rd, SP, 0)
    }

    pub fn orr_x(rd: u32, rn: u32, rm: u32) -> u32 {
        0xAA00_0000 | (rm << 16) | (rn << 5) | rd
    }

    pub fn mov_x(rd: u32, rm: u32) -> u32 {
        orr_x(rd, XZR, rm)
    }

    pub fn ldaxr_x(rt: u32, rn: u32) -> u32 {
        0xC85F_FC00 | (rn << 5) | rt
    }

    pub fn stlxr_x(rs: u32, rt: u32, rn: u32) -> u32 {
        0xC800_FC00 | (rs << 16) | (rn << 5) | rt
    }

    pub fn ldaxr_w(rt: u32, rn: u32) -> u32 {
        0x885F_FC00 | (rn << 5) | rt
    }

    pub fn stxr_w(rs: u32, rt: u32, rn: u32) -> u32 {
        0x8800_7C00 | (rs << 16) | (rn << 5) | rt
    }

    pub fn stlr_w(rt: u32, rn: u32) -> u32 {
        0x889F_FC00 | (rn << 5) | rt
    }

    pub fn clrex() -> u32 {
        0xD503_3F5F
    }

    pub fn cbz_w(rt: u32, offset: i64) -> u32 {
        0x3400_0000 | (imm19(offset) << 5) | rt
    }

    pub fn cbnz_w(rt: u32, offset: i64) -> u32 {
        0x3500_0000 | (imm19(offset) << 5) | rt
    }

    pub fn ldr_lit_x(rt: u32, offset: i64) -> u32 {
        0x5800_0000 | (imm19(offset) << 5) | rt
    }

    pub fn umulh(rd: u32, rn: u32, rm: u32) -> u32 {
        0x9BC0_7C00 | (rm << 16) | (rn << 5) | rd
    }

    pub fn madd(rd: u32, rn: u32, rm: u32, ra: u32) -> u32 {
        0x9B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd
    }

    pub fn brk(imm16: u32) -> u32 {
        0xD420_0000 | (imm16 << 5)
    }

    pub fn svc(imm16: u32) -> u32 {
        0xD400_0001 | (imm16 << 5)
    }
}

pub fn decode_svc(insn: u32) -> Option<u32> {
    ((insn & 0xFFE0_001F) == 0xD400_0001).then_some((insn >> 5) & 0xFFFF)
}

pub fn decode_mrs(insn: u32) -> Option<(u32, u32)> {
    ((insn >> 20) == 0xD53).then_some((insn & 0x1F, (insn >> 5) & 0x7FFF))
}

pub fn decode_msr(insn: u32) -> Option<(u32, u32)> {
    ((insn >> 20) == 0xD51).then_some((insn & 0x1F, (insn >> 5) & 0x7FFF))
}

pub fn is_exclusive(insn: u32) -> bool {
    (insn >> 23) & 0x7F == 0x10
}

pub fn exclusive_as_ordered(insn: u32) -> u32 {
    insn | (1 << 15)
}

pub fn is_brk(insn: u32) -> Option<u32> {
    ((insn & 0xFFE0_001F) == 0xD420_0000).then_some((insn >> 5) & 0xFFFF)
}

pub fn cntpct_factor(host_hz: u64) -> (u64, u64) {
    if host_hz == 0 {
        return (0, 1);
    }
    let numerator = (GUEST_CNTFRQ_HZ as u128) << 64;
    let factor = numerator / host_hz as u128;
    (factor as u64, (factor >> 64) as u64)
}

pub fn scale_counter(counter: u64, factor: (u64, u64)) -> u64 {
    let (lo, hi) = factor;
    let low_bits = ((counter as u128 * lo as u128) >> 64) as u64;
    counter.wrapping_mul(hi).wrapping_add(low_bits)
}

#[derive(Clone, Debug, Default)]
pub struct PatchOutput {
    pub text: Vec<u32>,
    pub section: Vec<u8>,
    pub post_handlers: Vec<(u64, u64)>,
    pub svc_count: usize,
    pub mrs_count: usize,
    pub msr_count: usize,
    pub counter_count: usize,
    pub exclusive_count: usize,
}

struct Section {
    words: Vec<u32>,
    base: u64,
}

impl Section {
    fn offset(&self) -> usize {
        self.words.len() * 4
    }

    fn va(&self) -> u64 {
        self.base + self.offset() as u64
    }

    fn emit(&mut self, word: u32) {
        self.words.push(word);
    }

    fn emit_literal_u64(&mut self, value: u64) {
        if self.offset() % 8 != 0 {
            self.emit(a64::nop());
        }
        self.emit(value as u32);
        self.emit((value >> 32) as u32);
    }

    fn branch_to(&mut self, target_va: u64) {
        let offset = target_va as i64 - self.va() as i64;
        self.emit(a64::b(offset));
    }

    fn call_to(&mut self, target_va: u64) {
        let offset = target_va as i64 - self.va() as i64;
        self.emit(a64::bl(offset));
    }

    fn patch_word(&mut self, offset: usize, word: u32) {
        self.words[offset / 4] = word;
    }
}

fn write_save_context(section: &mut Section) {
    section.emit(a64::str_x(30, a64::SP, 8));
    section.emit(a64::mrs(30, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(30, 30, NEP_NATIVE_CONTEXT));
    for i in (0..=28).step_by(2) {
        section.emit(a64::stp_x(i, i + 1, 30, (GUEST_X + 8 * i as usize) as i64));
    }
    for i in (0..=30).step_by(2) {
        section.emit(a64::stp_q(i, i + 1, 30, (GUEST_V + 16 * i as usize) as i64));
    }
    section.emit(a64::str_x_pre(0, a64::SP, -16));
    section.emit(a64::ldr_x(0, a64::SP, 16));
    section.emit(a64::str_x(0, 30, GUEST_X + 8 * 30));
    section.emit(a64::add_imm(0, a64::SP, 32));
    section.emit(a64::str_x(0, 30, GUEST_SP));
    section.emit(a64::mrs(0, SYSREG_FPSR));
    section.emit(a64::str_w(0, 30, GUEST_FPSR));
    section.emit(a64::mrs(0, SYSREG_FPCR));
    section.emit(a64::str_w(0, 30, GUEST_FPCR));
    section.emit(a64::mrs(0, SYSREG_NZCV));
    section.emit(a64::str_w(0, 30, GUEST_NZCV));
    section.emit(a64::ldr_x_post(0, a64::SP, 16));
    section.emit(a64::ldr_x(30, a64::SP, 8));
    section.emit(a64::ret());
}

fn write_load_context(section: &mut Section) {
    section.emit(a64::str_x(30, a64::SP, 8));
    section.emit(a64::mrs(30, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(30, 30, NEP_NATIVE_CONTEXT));
    section.emit(a64::ldr_w(0, 30, GUEST_FPSR));
    section.emit(a64::msr(SYSREG_FPSR, 0));
    section.emit(a64::ldr_w(0, 30, GUEST_FPCR));
    section.emit(a64::msr(SYSREG_FPCR, 0));
    section.emit(a64::ldr_w(0, 30, GUEST_NZCV));
    section.emit(a64::msr(SYSREG_NZCV, 0));
    for i in (0..=30).step_by(2) {
        section.emit(a64::ldp_q(i, i + 1, 30, (GUEST_V + 16 * i as usize) as i64));
    }
    for i in (0..=28).step_by(2) {
        section.emit(a64::ldp_x(i, i + 1, 30, (GUEST_X + 8 * i as usize) as i64));
    }
    section.emit(a64::ldr_x(30, a64::SP, 8));
    section.emit(a64::ret());
}

fn write_lock_context(section: &mut Section) {
    section.emit(a64::stp_x_pre(0, 1, a64::SP, -16));
    let retry = section.offset();
    section.emit(a64::clrex());
    section.emit(a64::mrs(0, SYSREG_TPIDR_EL0));
    section.emit(a64::add_imm(0, 0, NEP_LOCK as u32));
    section.emit(a64::ldaxr_w(1, 0));
    let back = retry as i64 - section.offset() as i64;
    section.emit(a64::cbz_w(1, back));
    section.emit(a64::stxr_w(1, a64::XZR, 0));
    let back = retry as i64 - section.offset() as i64;
    section.emit(a64::cbnz_w(1, back));
    section.emit(a64::ldp_x_post(0, 1, a64::SP, 16));
}

fn write_unlock_context(section: &mut Section) {
    section.emit(a64::stp_x_pre(0, 1, a64::SP, -16));
    section.emit(a64::mrs(0, SYSREG_TPIDR_EL0));
    section.emit(a64::add_imm(0, 0, NEP_LOCK as u32));
    section.emit(a64::movz_w(1, LOCK_UNLOCKED));
    section.emit(a64::stlr_w(1, 0));
    section.emit(a64::ldp_x_post(0, 1, a64::SP, 16));
}

fn write_svc_trampoline(
    section: &mut Section,
    save_ctx: u64,
    load_ctx: u64,
    svc_id: u32,
    resume_va: u64,
) -> u64 {
    write_lock_context(section);
    section.emit(a64::str_x_pre(30, a64::SP, -16));
    section.call_to(save_ctx);
    section.emit(a64::ldr_x_post(30, a64::SP, 16));
    section.emit(a64::mrs(1, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(1, 1, NEP_NATIVE_CONTEXT));
    let literal_fixup = section.offset();
    section.emit(0);
    section.emit(a64::str_x(2, 1, GUEST_PC));
    section.emit(a64::movz_w(2, svc_id));
    section.emit(a64::str_w(2, 1, GUEST_SVC));
    section.emit(a64::add_imm(2, 1, GUEST_ESR as u32));
    let retry = section.offset();
    section.emit(a64::ldaxr_x(0, 2));
    section.emit(a64::stlxr_x(3, a64::XZR, 2));
    let back = retry as i64 - section.offset() as i64;
    section.emit(a64::cbnz_w(3, back));
    section.emit(a64::movz_x(3, HALT_SUPERVISOR_CALL as u32));
    section.emit(a64::orr_x(0, 0, 3));
    section.emit(a64::add_imm(1, 1, GUEST_HOST_CTX as u32));
    section.emit(a64::ldp_x(2, 3, 1, HOST_SP as i64));
    section.emit(a64::mov_sp_from(2));
    section.emit(a64::msr(SYSREG_TPIDR_EL0, 3));
    for i in (0..12).step_by(2) {
        section.emit(a64::ldp_x(19 + i, 20 + i, 1, (HOST_REGS + 8 * i as usize) as i64));
    }
    for i in (0..8).step_by(2) {
        section.emit(a64::ldp_q(8 + i, 9 + i, 1, (HOST_VREGS + 16 * i as usize) as i64));
    }
    section.emit(a64::ret());

    let post_handler = section.va();
    section.emit(a64::mrs(2, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(2, 2, NEP_NATIVE_CONTEXT));
    section.emit(a64::add_imm(0, 2, GUEST_HOST_CTX as u32));
    section.emit(a64::str_x(30, 0, HOST_REGS + 8 * 11));
    section.emit(a64::str_x_pre(30, a64::SP, -16));
    section.call_to(load_ctx);
    section.emit(a64::ldr_x_post(30, a64::SP, 16));
    section.emit(a64::str_x_pre(1, a64::SP, -16));
    section.emit(a64::mrs(1, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(1, 1, NEP_NATIVE_CONTEXT));
    section.emit(a64::ldr_x(30, 1, GUEST_X + 8 * 30));
    section.emit(a64::ldr_x_post(1, a64::SP, 16));
    write_unlock_context(section);
    section.branch_to(resume_va);
    if section.offset() % 8 != 0 {
        section.emit(a64::nop());
    }
    let literal_offset = section.offset();
    section.emit_literal_u64(resume_va);
    let delta = literal_offset as i64 - literal_fixup as i64;
    section.patch_word(literal_fixup, a64::ldr_lit_x(2, delta));
    post_handler
}

fn write_mrs_handler(section: &mut Section, rt: u32, sysreg: u32, resume_va: u64) {
    let field = if sysreg == SYSREG_TPIDRRO_EL0 {
        NEP_TPIDRRO
    } else {
        NEP_TPIDR
    };
    section.emit(a64::mrs(rt, SYSREG_TPIDR_EL0));
    section.emit(a64::ldr_x(rt, rt, field));
    section.branch_to(resume_va);
}

fn write_msr_handler(section: &mut Section, rt: u32, resume_va: u64) {
    let scratch = if rt == 0 { 1 } else { 0 };
    section.emit(a64::str_x_pre(scratch, a64::SP, -16));
    section.emit(a64::mrs(scratch, SYSREG_TPIDR_EL0));
    section.emit(a64::str_x(rt, scratch, NEP_TPIDR));
    section.emit(a64::ldr_x_post(scratch, a64::SP, 16));
    section.branch_to(resume_va);
}

fn write_cntpct_handler(section: &mut Section, rt: u32, factor: (u64, u64), resume_va: u64) {
    let (scratch0, scratch1) = if rt == 0 || rt == 1 { (2, 3) } else { (0, 1) };
    section.emit(a64::stp_x_pre(scratch0, scratch1, a64::SP, -16));
    section.emit(a64::mrs(rt, SYSREG_CNTVCT_EL0));
    let lo_fixup = section.offset();
    section.emit(0);
    let hi_fixup = section.offset();
    section.emit(0);
    section.emit(a64::umulh(scratch0, rt, scratch0));
    section.emit(a64::madd(rt, rt, scratch1, scratch0));
    section.emit(a64::ldp_x_post(scratch0, scratch1, a64::SP, 16));
    section.branch_to(resume_va);
    if section.offset() % 8 != 0 {
        section.emit(a64::nop());
    }
    let lo_offset = section.offset();
    section.emit_literal_u64(factor.0);
    let hi_offset = section.offset();
    section.emit_literal_u64(factor.1);
    section.patch_word(lo_fixup, a64::ldr_lit_x(scratch0, lo_offset as i64 - lo_fixup as i64));
    section.patch_word(hi_fixup, a64::ldr_lit_x(scratch1, hi_offset as i64 - hi_fixup as i64));
}

fn write_cntfrq_handler(section: &mut Section, rt: u32, resume_va: u64) {
    let hz = GUEST_CNTFRQ_HZ as u32;
    section.emit(a64::movz_x(rt, hz & 0xFFFF));
    section.emit(a64::movk_x(rt, hz >> 16, 16));
    section.branch_to(resume_va);
}

pub fn patch_module(
    text: &[u32],
    module_base: u64,
    patch_base: u64,
    host_counter_hz: u64,
) -> Result<PatchOutput, String> {
    if module_base % 4 != 0 || patch_base % 4 != 0 {
        return Err("module and patch bases must be word aligned".to_string());
    }
    let mut out = PatchOutput {
        text: text.to_vec(),
        ..PatchOutput::default()
    };
    let mut section = Section {
        words: Vec::new(),
        base: patch_base,
    };
    section.emit(a64::nop());
    let save_ctx = section.va();
    write_save_context(&mut section);
    let load_ctx = section.va();
    write_load_context(&mut section);
    let factor = cntpct_factor(host_counter_hz);

    for (index, &insn) in text.iter().enumerate() {
        let insn_va = module_base + (index as u64) * 4;
        let resume_va = insn_va + 4;
        let trampoline = section.va();
        let reach = trampoline.abs_diff(insn_va);
        if reach >= MAX_BRANCH_REACH {
            return Err(format!(
                "instruction at {:#x} cannot reach patch section at {:#x}",
                insn_va, trampoline
            ));
        }
        if let Some(svc_id) = decode_svc(insn) {
            let post = write_svc_trampoline(&mut section, save_ctx, load_ctx, svc_id, resume_va);
            out.post_handlers.push((resume_va, post));
            out.text[index] = a64::b(trampoline as i64 - insn_va as i64);
            out.svc_count += 1;
            continue;
        }
        if let Some((rt, sysreg)) = decode_mrs(insn) {
            match sysreg {
                SYSREG_TPIDR_EL0 | SYSREG_TPIDRRO_EL0 => {
                    write_mrs_handler(&mut section, rt, sysreg, resume_va);
                    out.mrs_count += 1;
                }
                SYSREG_CNTPCT_EL0 => {
                    write_cntpct_handler(&mut section, rt, factor, resume_va);
                    out.counter_count += 1;
                }
                SYSREG_CNTFRQ_EL0 => {
                    write_cntfrq_handler(&mut section, rt, resume_va);
                    out.counter_count += 1;
                }
                _ => continue,
            }
            out.text[index] = a64::b(trampoline as i64 - insn_va as i64);
            continue;
        }
        if let Some((rt, sysreg)) = decode_msr(insn) {
            if sysreg == SYSREG_TPIDR_EL0 {
                write_msr_handler(&mut section, rt, resume_va);
                out.text[index] = a64::b(trampoline as i64 - insn_va as i64);
                out.msr_count += 1;
            }
            continue;
        }
        if is_exclusive(insn) {
            out.text[index] = exclusive_as_ordered(insn);
            out.exclusive_count += 1;
        }
    }

    let mut bytes = Vec::with_capacity(section.words.len() * 4);
    for word in &section.words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    out.section = bytes;
    Ok(out)
}

pub fn patch_section_size(text: &[u32]) -> usize {
    match patch_module(text, 0x1000_0000, 0x1000_0000 + text.len() as u64 * 4, 19_200_000) {
        Ok(out) => page_align(out.section.len()),
        Err(_) => 0,
    }
}

pub fn page_align(len: usize) -> usize {
    (len + 0xFFF) & !0xFFF
}

pub fn words_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub fn bytes_from_words(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_reference_encodings() {
        assert_eq!(decode_svc(0xD40000C1), Some(0x6));
        assert_eq!(decode_svc(0xD40000E1), Some(0x7));
        assert_eq!(decode_mrs(0xD53BE020), Some((0, SYSREG_CNTPCT_EL0)));
        assert_eq!(decode_mrs(0xD53BD040), Some((0, SYSREG_TPIDR_EL0)));
        assert_eq!(decode_mrs(0xD53BD060), Some((0, SYSREG_TPIDRRO_EL0)));
        assert_eq!(decode_msr(0xD51BD040), Some((0, SYSREG_TPIDR_EL0)));
        assert!(is_exclusive(0xC85FFC00));
        assert_eq!(exclusive_as_ordered(0xC85F7C00), 0xC85FFC00);
        assert_eq!(exclusive_as_ordered(0xC8200440), 0xC8208440);
        assert!(!is_exclusive(0xD503201F));
        assert_eq!(is_brk(a64::brk(0xF07)), Some(0xF07));
    }

    #[test]
    fn encoder_matches_known_words() {
        assert_eq!(a64::svc(6), 0xD40000C1);
        assert_eq!(a64::mrs(0, SYSREG_CNTPCT_EL0), 0xD53BE020);
        assert_eq!(a64::msr(SYSREG_TPIDR_EL0, 0), 0xD51BD040);
        assert_eq!(a64::ldaxr_x(0, 0), 0xC85FFC00);
        assert_eq!(a64::ret(), 0xD65F03C0);
        assert_eq!(a64::nop(), 0xD503201F);
        assert_eq!(a64::b(0), 0x14000000);
        assert_eq!(a64::b(-4), 0x17FFFFFF);
        assert_eq!(a64::bl(8), 0x94000002);
        assert_eq!(a64::str_x(30, a64::SP, 8), 0xF90007FE);
        assert_eq!(a64::ldr_x(30, a64::SP, 8), 0xF94007FE);
        assert_eq!(a64::stp_x(19, 20, 5, 0), 0xA90050B3);
        assert_eq!(a64::ldp_q(8, 9, 1, 0x60), 0xAD432428);
        assert_eq!(a64::str_x_pre(30, a64::SP, -16), 0xF81F0FFE);
        assert_eq!(a64::ldr_x_post(30, a64::SP, 16), 0xF84107FE);
        assert_eq!(a64::stp_x_pre(0, 1, a64::SP, -16), 0xA9BF07E0);
        assert_eq!(a64::ldp_x_post(0, 1, a64::SP, 16), 0xA8C107E0);
        assert_eq!(a64::movz_w(2, 0x1234), 0x52824682);
        assert_eq!(a64::movz_x(3, 1), 0xD2800023);
        assert_eq!(a64::movk_x(4, 0x124, 16), 0xF2A02484);
        assert_eq!(a64::add_imm(2, 1, 0x420), 0x91108022);
        assert_eq!(a64::mov_sp_from(2), 0x9100005F);
        assert_eq!(a64::orr_x(0, 0, 3), 0xAA030000);
        assert_eq!(a64::stlxr_x(3, a64::XZR, 2), 0xC803FC5F);
        assert_eq!(a64::ldaxr_w(1, 0), 0x885FFC01);
        assert_eq!(a64::stxr_w(1, a64::XZR, 0), 0x88017C1F);
        assert_eq!(a64::stlr_w(1, 0), 0x889FFC01);
        assert_eq!(a64::clrex(), 0xD5033F5F);
        assert_eq!(a64::cbz_w(1, -16), 0x34FFFF81);
        assert_eq!(a64::cbnz_w(3, -8), 0x35FFFFC3);
        assert_eq!(a64::ldr_lit_x(2, 0x100), 0x58000802);
        assert_eq!(a64::umulh(2, 0, 2), 0x9BC27C02);
        assert_eq!(a64::madd(0, 0, 3, 2), 0x9B030800);
        assert_eq!(a64::brk(0xF07), 0xD421E0E0);
        assert_eq!(a64::str_w(0, 30, 0x10C), 0xB9010FC0);
        assert_eq!(a64::ldr_w(0, 30, 0x108), 0xB9410BC0);
        assert_eq!(a64::mrs(0, SYSREG_NZCV), 0xD53B4200);
        assert_eq!(a64::msr(SYSREG_FPCR, 0), 0xD51B4400);
        assert_eq!(a64::mrs(0, SYSREG_FPSR), 0xD53B4420);
        assert_eq!(a64::msr(SYSREG_FPSR, 0), 0xD51B4420);
        assert_eq!(a64::mrs(0, SYSREG_TPIDR_EL0), 0xD53BD040);
        assert_eq!(a64::mrs(0, SYSREG_TPIDRRO_EL0), 0xD53BD060);
        assert_eq!(a64::mrs(0, SYSREG_CNTFRQ_EL0), 0xD53BE000);
        assert_eq!(a64::mrs(0, SYSREG_CNTVCT_EL0), 0xD53BE040);
        assert_eq!(a64::stp_q(30, 31, 30, 0x300), 0xAD187FDE);
        assert_eq!(a64::ldp_x(28, 29, 30, 0xE0), 0xA94E77DC);
        assert_eq!(a64::str_x(0, 30, 0xF8), 0xF9007FC0);
        assert_eq!(a64::ldr_x(30, 1, 0xF0), 0xF940783E);
        assert_eq!(a64::add_imm(1, 1, 0x320), 0x910C8021);
        assert_eq!(a64::ldp_x(2, 3, 1, 0xE0), 0xA94E0C22);
        assert_eq!(a64::msr(SYSREG_TPIDR_EL0, 3), 0xD51BD043);
    }

    #[test]
    #[ignore]
    fn dump_sample_patch_section() {
        let text = vec![
            a64::nop(),
            a64::svc(0x6),
            a64::mrs(5, SYSREG_TPIDRRO_EL0),
            a64::msr(SYSREG_TPIDR_EL0, 0),
            a64::mrs(1, SYSREG_CNTPCT_EL0),
            a64::mrs(7, SYSREG_CNTFRQ_EL0),
            0xC85F7C00,
            a64::ret(),
        ];
        let out = patch_module(&text, 0x8000_0000, 0x8000_1000, 24_000_000).unwrap();
        let dir = std::env::var("NEXIUM_PATCH_DUMP_DIR").unwrap_or_else(|_| ".".to_string());
        std::fs::write(format!("{dir}/nce-sample-text.bin"), bytes_from_words(&out.text)).unwrap();
        std::fs::write(format!("{dir}/nce-sample-patch.bin"), &out.section).unwrap();
        std::fs::write(
            format!("{dir}/nce-sample-post.txt"),
            out.post_handlers
                .iter()
                .map(|(resume, post)| format!("{resume:#x} -> {post:#x}\n"))
                .collect::<String>(),
        )
        .unwrap();
    }

    #[test]
    fn counter_factor_is_identity_for_matching_frequency() {
        let factor = cntpct_factor(19_200_000);
        assert_eq!(factor, (0, 1));
        assert_eq!(scale_counter(123_456_789, factor), 123_456_789);
        let half = cntpct_factor(38_400_000);
        assert_eq!(half, (1 << 63, 0));
        assert_eq!(scale_counter(1000, half), 500);
        let odd = cntpct_factor(24_000_000);
        assert_eq!(scale_counter(24_000_000, odd), 19_200_000 - 1);
    }

    #[test]
    fn patches_svc_mrs_msr_and_exclusives() {
        let text = vec![
            a64::nop(),
            a64::svc(0x6),
            a64::mrs(5, SYSREG_TPIDRRO_EL0),
            a64::msr(SYSREG_TPIDR_EL0, 0),
            a64::mrs(1, SYSREG_CNTPCT_EL0),
            a64::mrs(7, SYSREG_CNTFRQ_EL0),
            0xC85F7C00,
            a64::ret(),
        ];
        let module_base = 0x8000_0000;
        let patch_base = module_base + 0x1000;
        let out = patch_module(&text, module_base, patch_base, 19_200_000).unwrap();
        assert_eq!(out.svc_count, 1);
        assert_eq!(out.mrs_count, 1);
        assert_eq!(out.msr_count, 1);
        assert_eq!(out.counter_count, 2);
        assert_eq!(out.exclusive_count, 1);
        assert_eq!(out.text[0], a64::nop());
        assert_eq!(out.text[7], a64::ret());
        assert_eq!(out.text[6], 0xC85FFC00);
        for index in 1..=5 {
            let word = out.text[index];
            assert_eq!(word >> 26, 0x5, "instruction {index} must become a branch");
            let target = module_base + index as u64 * 4 + branch_offset(word);
            assert!(target >= patch_base && target < patch_base + out.section.len() as u64);
        }
        assert_eq!(out.post_handlers.len(), 1);
        let (resume, post) = out.post_handlers[0];
        assert_eq!(resume, module_base + 8);
        assert!(post > patch_base && post < patch_base + out.section.len() as u64);
        let words = words_from_bytes(&out.section);
        let branches_back = words
            .iter()
            .enumerate()
            .filter(|(i, w)| {
                *w >> 26 == 0x5 && {
                    let va = patch_base + *i as u64 * 4;
                    let target = va.wrapping_add(branch_offset(**w));
                    target >= module_base && target < module_base + text.len() as u64 * 4
                }
            })
            .count();
        assert_eq!(branches_back, 5);
        assert_eq!(out.section.len() % 4, 0);
    }

    #[test]
    fn section_size_is_independent_of_bases() {
        let text = vec![a64::svc(1), a64::svc(2), a64::mrs(0, SYSREG_TPIDR_EL0)];
        let a = patch_module(&text, 0x8000_0000, 0x8010_0000, 19_200_000).unwrap();
        let b = patch_module(&text, 0x1_0000_0000, 0x1_0000_0000 + 0x2000, 24_000_000).unwrap();
        assert_eq!(a.section.len(), b.section.len());
        assert_eq!(patch_section_size(&text), page_align(a.section.len()));
    }

    #[test]
    fn rejects_unreachable_patch_section() {
        let text = vec![a64::svc(1)];
        assert!(patch_module(&text, 0, MAX_BRANCH_REACH + 0x1000, 19_200_000).is_err());
    }

    fn branch_offset(word: u32) -> u64 {
        let imm = word & 0x03FF_FFFF;
        let signed = if imm & 0x0200_0000 != 0 {
            (imm as i64) - (1 << 26)
        } else {
            imm as i64
        };
        (signed * 4) as u64
    }
}

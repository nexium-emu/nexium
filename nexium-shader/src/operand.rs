use crate::ir::FMods;

#[inline]
fn bits(insn: u64, lo: u32, hi: u32) -> u64 {
    let len = hi - lo + 1;
    let mask = if len == 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    };
    (insn >> lo) & mask
}

#[inline]
pub fn reg_dest(insn: u64) -> u8 {
    bits(insn, 0, 7) as u8
}

#[inline]
pub fn reg_a(insn: u64) -> u8 {
    bits(insn, 8, 15) as u8
}

#[inline]
pub fn reg_b(insn: u64) -> u8 {
    bits(insn, 20, 27) as u8
}

#[inline]
pub fn reg_c(insn: u64) -> u8 {
    bits(insn, 39, 46) as u8
}

#[inline]
pub fn pred_idx(insn: u64) -> u8 {
    bits(insn, 16, 18) as u8
}

#[inline]
pub fn pred_negate(insn: u64) -> bool {
    bits(insn, 19, 19) != 0
}

#[derive(Clone, Copy, Debug)]
pub struct CbufRef {
    pub binding: u8,
    pub byte_offset: u32,
}

#[inline]
pub fn cbuf(insn: u64) -> CbufRef {
    let raw_off = bits(insn, 20, 33) as u32;
    let binding = bits(insn, 34, 38) as u8;
    CbufRef {
        binding,
        byte_offset: raw_off * 4,
    }
}

#[inline]
pub fn imm20(insn: u64) -> i32 {
    let raw = bits(insn, 20, 38) as i32;
    if bits(insn, 56, 56) != 0 {
        raw - (1 << 19)
    } else {
        raw
    }
}

#[inline]
pub fn imm32(insn: u64) -> u32 {
    bits(insn, 20, 51) as u32
}

pub const RZ: u8 = 0xFF;

pub const PT: u8 = 7;

pub fn decoded_pred(insn: u64) -> Option<super::ir::Predicate> {
    let idx = pred_idx(insn);
    if idx == PT {
        return None;
    }
    Some(super::ir::Predicate {
        idx,
        negate: pred_negate(insn),
    })
}

#[inline]
pub fn attr_slot_ald(insn: u64) -> u32 {
    bits(insn, 20, 29) as u32
}

#[inline]
pub fn attr_slot_ipa(insn: u64) -> u32 {
    (bits(insn, 30, 37) as u32) * 4
}

#[inline]
pub fn ipa_saturate(insn: u64) -> bool {
    bits(insn, 51, 51) != 0
}

#[inline]
pub fn ipa_interpolation_mode(insn: u64) -> u8 {
    bits(insn, 54, 55) as u8
}

#[inline]
pub fn ald_num_elements(insn: u64) -> u32 {
    match bits(insn, 47, 48) {
        0 => 1,
        1 => 2,
        2 => 3,
        _ => 4,
    }
}

#[inline]
pub fn texs_tex_id(insn: u64) -> u32 {
    bits(insn, 36, 48) as u32
}

#[inline]
pub fn mufu_func_bits(insn: u64) -> u32 {
    bits(insn, 20, 23) as u32
}

#[derive(Clone, Copy, Debug)]
pub struct LdcRef {
    pub binding: u8,
    pub byte_offset: i32,
}

#[inline]
pub fn ldc_ref(insn: u64) -> LdcRef {
    let raw_off = bits(insn, 20, 35) as u32;

    let byte_offset = if raw_off & (1 << 15) != 0 {
        (raw_off | 0xFFFF_0000) as i32
    } else {
        raw_off as i32
    };
    let binding = bits(insn, 36, 40) as u8;
    LdcRef {
        binding,
        byte_offset,
    }
}

#[inline]
pub fn ldc_src_reg(insn: u64) -> u8 {
    bits(insn, 8, 15) as u8
}

#[inline]
pub fn ldc_size(insn: u64) -> u32 {
    bits(insn, 48, 50) as u32
}

#[inline]
pub fn ldg_addr_reg(insn: u64) -> u8 {
    bits(insn, 8, 15) as u8
}

#[inline]
pub fn ldg_offset(insn: u64) -> i32 {
    let raw = bits(insn, 20, 43) as u32;
    if raw & (1 << 23) != 0 {
        (raw | 0xFF00_0000) as i32
    } else {
        raw as i32
    }
}

#[inline]
pub fn ldg_e(insn: u64) -> bool {
    bits(insn, 45, 45) != 0
}

#[inline]
pub fn ldg_size(insn: u64) -> u32 {
    bits(insn, 48, 50) as u32
}

#[inline]
pub fn stg_data_reg(insn: u64) -> u8 {
    bits(insn, 0, 7) as u8
}

#[inline]
pub fn fsetp_dest_p(insn: u64) -> u8 {
    bits(insn, 3, 5) as u8
}
#[inline]
pub fn fsetp_dest_np(insn: u64) -> u8 {
    bits(insn, 0, 2) as u8
}

#[inline]
pub fn fsetp_src_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}
#[inline]
pub fn fsetp_src_pred_inv(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn fsetp_bop(insn: u64) -> u64 {
    bits(insn, 45, 46)
}

#[inline]
pub fn fsetp_cmp(insn: u64) -> u64 {
    bits(insn, 48, 51)
}

#[inline]
pub fn fsetp_neg_a(insn: u64) -> bool {
    bits(insn, 43, 43) != 0
}
#[inline]
pub fn fsetp_abs_a(insn: u64) -> bool {
    bits(insn, 7, 7) != 0
}
#[inline]
pub fn fsetp_neg_b(insn: u64) -> bool {
    bits(insn, 6, 6) != 0
}
#[inline]
pub fn fsetp_abs_b(insn: u64) -> bool {
    bits(insn, 44, 44) != 0
}

#[inline]
pub fn float_imm20(insn: u64) -> f32 {
    let value = (bits(insn, 20, 38) as u32) << 12;
    let sign = if bits(insn, 56, 56) != 0 {
        1u32 << 31
    } else {
        0
    };
    f32::from_bits(value | sign)
}

#[inline]
pub fn fadd_mods(insn: u64) -> FMods {
    FMods {
        neg_a: bits(insn, 48, 48) != 0,
        abs_a: bits(insn, 46, 46) != 0,
        neg_b: bits(insn, 45, 45) != 0,
        abs_b: bits(insn, 49, 49) != 0,
        neg_c: false,
        sat: bits(insn, 50, 50) != 0,
        scale: 0,
    }
}

#[inline]
pub fn fmul_mods(insn: u64) -> FMods {
    FMods {
        neg_b: bits(insn, 48, 48) != 0,
        sat: bits(insn, 50, 50) != 0,
        scale: bits(insn, 41, 43) as u8,
        ..FMods::default()
    }
}

#[inline]
pub fn ffma_mods(insn: u64) -> FMods {
    FMods {
        neg_b: bits(insn, 48, 48) != 0,
        neg_c: bits(insn, 49, 49) != 0,
        sat: bits(insn, 50, 50) != 0,
        ..FMods::default()
    }
}

#[inline]
pub fn fadd32i_mods(insn: u64) -> FMods {
    FMods {
        neg_a: bits(insn, 56, 56) != 0,
        abs_a: bits(insn, 54, 54) != 0,
        neg_b: bits(insn, 53, 53) != 0,
        abs_b: bits(insn, 57, 57) != 0,
        neg_c: false,
        sat: false,
        scale: 0,
    }
}

#[inline]
pub fn fmul32i_mods(insn: u64) -> FMods {
    FMods {
        sat: bits(insn, 55, 55) != 0,
        ..FMods::default()
    }
}

#[inline]
pub fn ffma32i_mods(insn: u64) -> FMods {
    FMods {
        neg_a: bits(insn, 56, 56) != 0,
        neg_c: bits(insn, 57, 57) != 0,
        sat: bits(insn, 55, 55) != 0,
        ..FMods::default()
    }
}

pub struct F2fMods {
    pub neg: bool,
    pub abs: bool,
    pub sat: bool,
    pub round: u8,
}

#[inline]
pub fn f2f_mods(insn: u64) -> F2fMods {
    let src_size = bits(insn, 10, 11);
    let dst_size = bits(insn, 8, 9);
    let round = if src_size == dst_size {
        match bits(insn, 39, 42) & 0x0B {
            8 => 1,
            9 => 2,
            10 => 3,
            11 => 4,
            _ => 0,
        }
    } else {
        0
    };
    F2fMods {
        neg: bits(insn, 45, 45) != 0,
        abs: bits(insn, 49, 49) != 0,
        sat: bits(insn, 50, 50) != 0,
        round,
    }
}

#[inline]
pub fn i2f_signed(insn: u64) -> bool {
    bits(insn, 13, 13) != 0
}
#[inline]
pub fn i2f_neg(insn: u64) -> bool {
    bits(insn, 45, 45) != 0
}
#[inline]
pub fn i2f_abs(insn: u64) -> bool {
    bits(insn, 49, 49) != 0
}
#[inline]
pub fn i2f_int_format(insn: u64) -> u8 {
    bits(insn, 10, 11) as u8
}
#[inline]
pub fn i2f_selector(insn: u64) -> u8 {
    bits(insn, 41, 42) as u8
}

#[inline]
pub fn fset_neg_a(insn: u64) -> bool {
    bits(insn, 43, 43) != 0
}
#[inline]
pub fn fset_abs_a(insn: u64) -> bool {
    bits(insn, 54, 54) != 0
}
#[inline]
pub fn fset_neg_b(insn: u64) -> bool {
    bits(insn, 53, 53) != 0
}
#[inline]
pub fn fset_abs_b(insn: u64) -> bool {
    bits(insn, 44, 44) != 0
}
#[inline]
pub fn fset_cmp(insn: u64) -> u64 {
    bits(insn, 48, 51)
}
#[inline]
pub fn fset_bop(insn: u64) -> u64 {
    bits(insn, 45, 46)
}
#[inline]
pub fn fset_src_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}
#[inline]
pub fn fset_src_pred_inv(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn isetp_signed(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}
#[inline]
pub fn isetp_cmp(insn: u64) -> u64 {
    bits(insn, 49, 51)
}
#[inline]
pub fn isetp_bop(insn: u64) -> u64 {
    bits(insn, 45, 46)
}
#[inline]
pub fn isetp_src_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}
#[inline]
pub fn isetp_src_pred_inv(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}
#[inline]
pub fn isetp_dest_p(insn: u64) -> u8 {
    bits(insn, 3, 5) as u8
}
#[inline]
pub fn isetp_dest_np(insn: u64) -> u8 {
    bits(insn, 0, 2) as u8
}

#[inline]
pub fn fmnmx_mods(insn: u64) -> FMods {
    FMods {
        neg_a: bits(insn, 48, 48) != 0,
        abs_a: bits(insn, 46, 46) != 0,
        neg_b: bits(insn, 45, 45) != 0,
        abs_b: bits(insn, 49, 49) != 0,
        neg_c: false,
        sat: false,
        scale: 0,
    }
}

#[inline]
pub fn fmnmx_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}

#[inline]
pub fn fmnmx_neg_pred(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn sel_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}

#[inline]
pub fn sel_neg_pred(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn psetp_dest_np(insn: u64) -> u8 {
    bits(insn, 0, 2) as u8
}

#[inline]
pub fn psetp_dest_p(insn: u64) -> u8 {
    bits(insn, 3, 5) as u8
}

#[inline]
pub fn psetp_pred_a(insn: u64) -> u8 {
    bits(insn, 12, 14) as u8
}

#[inline]
pub fn psetp_neg_pred_a(insn: u64) -> bool {
    bits(insn, 15, 15) != 0
}

#[inline]
pub fn psetp_bop_1(insn: u64) -> u64 {
    bits(insn, 24, 25)
}

#[inline]
pub fn psetp_pred_b(insn: u64) -> u8 {
    bits(insn, 29, 31) as u8
}

#[inline]
pub fn psetp_neg_pred_b(insn: u64) -> bool {
    bits(insn, 32, 32) != 0
}

#[inline]
pub fn psetp_pred_c(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}

#[inline]
pub fn psetp_neg_pred_c(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn psetp_bop_2(insn: u64) -> u64 {
    bits(insn, 45, 46)
}

#[inline]
pub fn pset_bool_float(insn: u64) -> bool {
    bits(insn, 44, 44) != 0
}

#[inline]
pub fn csetp_flow_test(insn: u64) -> u8 {
    bits(insn, 8, 12) as u8
}

#[inline]
pub fn csetp_bop_pred(insn: u64) -> u8 {
    bits(insn, 39, 41) as u8
}

#[inline]
pub fn csetp_neg_bop_pred(insn: u64) -> bool {
    bits(insn, 42, 42) != 0
}

#[inline]
pub fn csetp_bop(insn: u64) -> u64 {
    bits(insn, 45, 46)
}

#[inline]
pub fn iadd3_shift(insn: u64) -> u8 {
    bits(insn, 37, 38) as u8
}

#[inline]
pub fn iadd3_half_a(insn: u64) -> u8 {
    bits(insn, 35, 36) as u8
}

#[inline]
pub fn iadd3_half_b(insn: u64) -> u8 {
    bits(insn, 33, 34) as u8
}

#[inline]
pub fn iadd3_half_c(insn: u64) -> u8 {
    bits(insn, 31, 32) as u8
}

#[inline]
pub fn iadd3_neg_a(insn: u64) -> bool {
    bits(insn, 51, 51) != 0
}

#[inline]
pub fn iadd3_neg_b(insn: u64) -> bool {
    bits(insn, 50, 50) != 0
}

#[inline]
pub fn iadd3_neg_c(insn: u64) -> bool {
    bits(insn, 49, 49) != 0
}

#[inline]
pub fn xmad_half_a(insn: u64) -> u8 {
    bits(insn, 53, 53) as u8
}

#[inline]
pub fn xmad_signed_a(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}

#[inline]
pub fn xmad_signed_b(insn: u64) -> bool {
    bits(insn, 49, 49) != 0
}

#[inline]
pub fn xmad_reg_half_b(insn: u64) -> u8 {
    bits(insn, 35, 35) as u8
}

#[inline]
pub fn xmad_reg_psl(insn: u64) -> bool {
    bits(insn, 36, 36) != 0
}

#[inline]
pub fn xmad_reg_mrg(insn: u64) -> bool {
    bits(insn, 37, 37) != 0
}

#[inline]
pub fn xmad_reg_select(insn: u64) -> u8 {
    bits(insn, 50, 52) as u8
}

#[inline]
pub fn xmad_rc_half_b(insn: u64) -> u8 {
    bits(insn, 52, 52) as u8
}

#[inline]
pub fn xmad_rc_select(insn: u64) -> u8 {
    bits(insn, 50, 51) as u8
}

#[inline]
pub fn xmad_cr_psl(insn: u64) -> bool {
    bits(insn, 55, 55) != 0
}

#[inline]
pub fn xmad_cr_mrg(insn: u64) -> bool {
    bits(insn, 56, 56) != 0
}

#[inline]
pub fn xmad_imm_src_b(insn: u64) -> u32 {
    bits(insn, 20, 35) as u32
}

#[inline]
pub fn iscadd_shift(insn: u64) -> u8 {
    bits(insn, 39, 43) as u8
}
#[inline]
pub fn iadd_neg_a(insn: u64) -> bool {
    bits(insn, 49, 49) != 0
}
#[inline]
pub fn iadd_neg_b(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}
#[inline]
pub fn lop_op(insn: u64) -> u64 {
    bits(insn, 41, 42)
}
#[inline]
pub fn lop_not_a(insn: u64) -> bool {
    bits(insn, 39, 39) != 0
}
#[inline]
pub fn lop_not_b(insn: u64) -> bool {
    bits(insn, 40, 40) != 0
}
#[inline]
pub fn lop32i_op(insn: u64) -> u64 {
    bits(insn, 53, 54)
}
#[inline]
pub fn lop32i_not_a(insn: u64) -> bool {
    bits(insn, 55, 55) != 0
}
#[inline]
pub fn lop32i_not_b(insn: u64) -> bool {
    bits(insn, 56, 56) != 0
}
#[inline]
pub fn shr_signed(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}
#[inline]
pub fn f2i_signed(insn: u64) -> bool {
    bits(insn, 12, 12) != 0
}
#[inline]
pub fn bfe_signed(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}
#[inline]
pub fn iset_cmp(insn: u64) -> u64 {
    bits(insn, 49, 51)
}
#[inline]
pub fn iset_signed(insn: u64) -> bool {
    bits(insn, 48, 48) != 0
}
#[inline]
pub fn iset_bf(insn: u64) -> bool {
    bits(insn, 44, 44) != 0
}

pub fn fmt_reg(r: u8) -> String {
    if r == RZ {
        "RZ".into()
    } else {
        format!("R{r}")
    }
}

pub fn fmt_cbuf(c: CbufRef) -> String {
    format!("c[{:#x}]:{:#x}", c.binding, c.byte_offset)
}

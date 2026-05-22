

#[inline]
fn bits(insn: u64, lo: u32, hi: u32) -> u64 {
    let len = hi - lo + 1;
    let mask = if len == 64 { u64::MAX } else { (1u64 << len) - 1 };
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
    let raw = bits(insn, 20, 39) as u32;

    if raw & (1 << 19) != 0 {
        (raw | 0xFFF0_0000) as i32
    } else {
        raw as i32
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
    LdcRef { binding, byte_offset }
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
pub fn fsetp_neg_a(insn: u64) -> bool { bits(insn, 43, 43) != 0 }
#[inline]
pub fn fsetp_abs_a(insn: u64) -> bool { bits(insn, 7, 7) != 0 }
#[inline]
pub fn fsetp_neg_b(insn: u64) -> bool { bits(insn, 6, 6) != 0 }
#[inline]
pub fn fsetp_abs_b(insn: u64) -> bool { bits(insn, 44, 44) != 0 }

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

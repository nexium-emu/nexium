use crate::decode::opcode_bits;
use crate::opcodes::Opcode;

pub(super) const RZ: u8 = 0xFF;
pub(super) const PT: u8 = 7;
pub(super) const FLOW_ALWAYS: u64 = 0xF;
const UNBOUND_BANK: u8 = 24;
const SPARE_BANK: u8 = 17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Word(pub u64);

impl Word {
    pub fn new(opcode: Opcode) -> Self {
        Self(opcode_bits(opcode).unwrap_or_else(|| panic!("{opcode:?} missing from the opcode table")))
    }

    pub fn set(self, start: u32, len: u32, value: u64) -> Self {
        let mask = if len == 64 { u64::MAX } else { ((1u64 << len) - 1) << start };
        Self((self.0 & !mask) | ((value << start) & mask))
    }

    pub fn flag(self, at: u32, on: bool) -> Self {
        self.set(at, 1, u64::from(on))
    }

    pub fn dst(self, reg: u8) -> Self {
        self.set(0, 8, u64::from(reg))
    }

    pub fn a(self, reg: u8) -> Self {
        self.set(8, 8, u64::from(reg))
    }

    pub fn b(self, reg: u8) -> Self {
        self.set(20, 8, u64::from(reg))
    }

    pub fn c(self, reg: u8) -> Self {
        self.set(39, 8, u64::from(reg))
    }

    pub fn guard(self, pred: u8, neg: bool) -> Self {
        self.set(16, 3, u64::from(pred)).flag(19, neg)
    }

    pub fn cbuf(self, bank: u8, offset: u16) -> Self {
        self.set(20, 14, u64::from(offset) / 4).set(34, 5, u64::from(cbuf_bank(bank)))
    }

    pub fn float_imm20(self, bits: u32) -> Self {
        self.set(20, 19, u64::from(bits >> 12) & 0x7FFFF).flag(56, bits >> 31 != 0)
    }

    pub fn int_imm20(self, value: i32) -> Self {
        self.set(20, 19, (value as u32 & 0x7FFFF) as u64).flag(56, value < 0)
    }

    pub fn imm32(self, value: u32) -> Self {
        self.set(20, 32, u64::from(value))
    }

    pub fn offset24(self, value: i64) -> Self {
        self.set(20, 24, (value as u64) & 0xFF_FFFF)
    }
}

pub(super) fn cbuf_bank(bank: u8) -> u8 {
    if bank == UNBOUND_BANK {
        SPARE_BANK
    } else {
        bank
    }
}

pub(super) fn float_fits_imm20(bits: u32) -> bool {
    bits & 0xFFF == 0
}

pub(super) fn int_fits_imm20(value: u32) -> bool {
    let signed = value as i32;
    (-(1 << 19)..(1 << 19)).contains(&signed)
}

pub(super) fn nop() -> Word {
    Word::new(Opcode::NOP).set(8, 4, FLOW_ALWAYS).guard(PT, false)
}

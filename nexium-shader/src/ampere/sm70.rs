pub(super) const PT: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Inst {
    pub lo: u64,
    pub hi: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Operand {
    Reg(u8),
    UReg(u8),
    Imm(u32),
    Cbuf { bank: u8, offset: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub operand: Operand,
    pub neg: bool,
    pub abs: bool,
}

impl Source {
    pub fn plain(operand: Operand) -> Self {
        Self {
            operand,
            neg: false,
            abs: false,
        }
    }

    pub fn modified(&self) -> bool {
        self.neg || self.abs
    }
}

impl Inst {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            hi: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        }
    }

    pub fn field(&self, start: u32, len: u32) -> u64 {
        let wide = u128::from(self.lo) | (u128::from(self.hi) << 64);
        ((wide >> start) & ((1u128 << len) - 1)) as u64
    }

    pub fn with_field(self, start: u32, len: u32, value: u64) -> Self {
        let wide = u128::from(self.lo) | (u128::from(self.hi) << 64);
        let mask = ((1u128 << len) - 1) << start;
        let wide = (wide & !mask) | ((u128::from(value) << start) & mask);
        Self {
            lo: wide as u64,
            hi: (wide >> 64) as u64,
        }
    }

    pub fn bit(&self, at: u32) -> bool {
        self.field(at, 1) != 0
    }

    pub fn opcode(&self) -> u16 {
        self.field(0, 12) as u16
    }

    pub fn alu_opcode(&self) -> u16 {
        self.field(0, 9) as u16
    }

    pub fn form(&self) -> u8 {
        self.field(9, 3) as u8
    }

    pub fn guard(&self) -> (u8, bool) {
        (self.field(12, 3) as u8, self.bit(15))
    }

    pub fn dst(&self) -> u8 {
        self.field(16, 8) as u8
    }

    pub fn reg(&self, start: u32) -> u8 {
        self.field(start, 8) as u8
    }

    pub fn pred(&self, start: u32, not_bit: u32) -> (u8, bool) {
        (self.field(start, 3) as u8, self.bit(not_bit))
    }

    pub fn signed_offset(&self, start: u32, len: u32) -> i64 {
        let raw = self.field(start, len);
        let shift = 64 - len;
        ((raw << shift) as i64) >> shift
    }

    pub fn src0(&self) -> Source {
        Source {
            operand: Operand::Reg(self.reg(24)),
            neg: self.bit(72),
            abs: self.bit(73),
        }
    }

    pub fn slot32(&self) -> Option<Source> {
        let neg = self.bit(63);
        let abs = self.bit(62);
        let operand = match self.form() {
            1 => Operand::Reg(self.reg(32)),
            2 | 4 => return Some(Source::plain(Operand::Imm(self.field(32, 32) as u32))),
            3 | 5 if !self.bit(91) => Operand::Cbuf {
                bank: self.field(54, 5) as u8,
                offset: self.field(38, 16) as u16,
            },
            6 | 7 => Operand::UReg(self.field(32, 6) as u8),
            _ => return None,
        };
        Some(Source { operand, neg, abs })
    }

    pub fn slot64(&self) -> Source {
        Source {
            operand: Operand::Reg(self.reg(64)),
            neg: self.bit(75),
            abs: self.bit(74),
        }
    }

    pub fn alu_sources(&self) -> Option<(Source, Source)> {
        let wide = self.slot32()?;
        Some(match self.form() {
            1 | 4 | 5 | 6 => (wide, self.slot64()),
            _ => (self.slot64(), wide),
        })
    }

    pub fn branch_target(&self, offset: usize, start: u32, len: u32) -> i64 {
        offset as i64 + 16 + self.signed_offset(start, len) * 4
    }

    pub fn is_end(&self) -> bool {
        self.opcode() == 0x947
            && self.guard() == (PT, false)
            && self.pred(87, 90) == (PT, false)
            && self.signed_offset(34, 48) == -4
    }
}

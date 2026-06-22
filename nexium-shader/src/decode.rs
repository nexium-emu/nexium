use std::sync::OnceLock;

use super::opcodes::{Opcode, OpcodeEntry, OPCODE_TABLE};

#[derive(Clone, Copy, Debug)]
pub struct Decoded {
    pub opcode: Opcode,
    pub display: &'static str,
    pub raw: u64,
}

fn mask_value_from_encoding(encoding: &str) -> (u64, u64) {
    let mut mask: u64 = 0;
    let mut value: u64 = 0;
    let mut bit_pos: i32 = 63;
    for c in encoding.chars() {
        match c {
            '0' => {
                mask |= 1u64 << bit_pos;
                bit_pos -= 1;
            }
            '1' => {
                mask |= 1u64 << bit_pos;
                value |= 1u64 << bit_pos;
                bit_pos -= 1;
            }
            '-' => {
                bit_pos -= 1;
            }
            _ => {}
        }
        if bit_pos < 0 {
            break;
        }
    }
    (mask, value)
}

#[derive(Clone, Copy)]
struct CompiledEntry {
    mask: u64,
    value: u64,
    opcode: Opcode,
    display: &'static str,
}

fn compiled_table() -> &'static [CompiledEntry] {
    static TABLE: OnceLock<Vec<CompiledEntry>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut entries: Vec<CompiledEntry> = OPCODE_TABLE
            .iter()
            .map(|e: &OpcodeEntry| {
                let (mask, value) = mask_value_from_encoding(e.encoding);
                CompiledEntry {
                    mask,
                    value,
                    opcode: e.opcode,
                    display: e.display,
                }
            })
            .collect();

        entries.sort_by_key(|e| std::cmp::Reverse(e.mask.count_ones()));
        entries
    })
}

pub fn decode_one(insn: u64) -> Option<Decoded> {
    let table = compiled_table();
    for e in table {
        if (insn & e.mask) == e.value {
            return Some(Decoded {
                opcode: e.opcode,
                display: e.display,
                raw: insn,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_parser_basic() {
        let (mask, value) = mask_value_from_encoding("0101 1100 1001 1---");

        assert_eq!(mask >> 51 & 0x1FFF, 0x1FFF);
        assert_eq!((value >> 51) & 0x1FFF, 0b0101_1100_1001_1);
    }

    #[test]
    fn decode_known_opcodes() {
        let exit_insn = 0xE300_0000_0000_0000u64;
        let d = decode_one(exit_insn).expect("decoded");
        assert_eq!(d.opcode, Opcode::EXIT);

        let mov_cbuf = 0x4C98_0000_0000_0000u64;
        let d = decode_one(mov_cbuf).expect("decoded");
        assert_eq!(d.opcode, Opcode::MOV_cbuf);
    }
}

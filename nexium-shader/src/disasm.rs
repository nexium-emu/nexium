use super::decode::{decode_one, Decoded};

#[derive(Clone, Debug)]
pub struct DisasmLine {
    pub offset: usize,

    pub raw: u64,
    pub kind: DisasmKind,
}

#[derive(Clone, Debug)]
pub enum DisasmKind {
    Insn(Decoded),

    Schedule,

    Unknown,
}

pub fn disassemble(bytes: &[u8]) -> Vec<DisasmLine> {
    let mut out = Vec::new();
    let mut hit_exit = false;
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        if hit_exit {
            break;
        }
        let raw = u64::from_le_bytes(chunk.try_into().unwrap());
        let offset = i * 8;

        let is_schedule = offset % 0x20 == 0;
        if is_schedule {
            out.push(DisasmLine {
                offset,
                raw,
                kind: DisasmKind::Schedule,
            });
            continue;
        }
        let kind = match decode_one(raw) {
            Some(d) => {
                if matches!(d.opcode, super::opcodes::Opcode::EXIT) {
                    hit_exit = true;
                }
                DisasmKind::Insn(d)
            }
            None => DisasmKind::Unknown,
        };
        out.push(DisasmLine { offset, raw, kind });
    }
    out
}

impl DisasmLine {
    pub fn to_string_compact(&self) -> String {
        match &self.kind {
            DisasmKind::Insn(d) => {
                let mnemonic = mnemonic_only(d.display);
                let operands = super::pretty::pretty_operands(d.opcode, d.raw);
                match operands {
                    Some(s) if !s.is_empty() => format!(
                        "  +{:04x}  {:016x}  {:<8} {}",
                        self.offset, self.raw, mnemonic, s
                    ),
                    Some(_) => format!("  +{:04x}  {:016x}  {}", self.offset, self.raw, mnemonic),
                    None => format!("  +{:04x}  {:016x}  {}", self.offset, self.raw, d.display),
                }
            }
            DisasmKind::Schedule => {
                format!("  +{:04x}  {:016x}  ; sched", self.offset, self.raw)
            }
            DisasmKind::Unknown => {
                format!("  +{:04x}  {:016x}  ?", self.offset, self.raw)
            }
        }
    }
}

fn mnemonic_only(display: &str) -> &str {
    display.split_once(' ').map(|(m, _)| m).unwrap_or(display)
}

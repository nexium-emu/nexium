use super::decode::{decode_one, Decoded};
use super::opcodes::Opcode;
use super::operand::{decoded_pred, exit_never_taken, ldc_mode, ldc_ref, texs_tex_id, LdcMode};

#[derive(Clone, Copy, Debug)]
pub struct Instruction {
    pub byte_offset: usize,
    pub decoded: Decoded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsTexId {
    ImmediateTic(u32),

    BindlessCbufOffset(u32),
}

pub fn walk_instructions(code: &[u8]) -> Vec<Instruction> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 32 <= code.len() {
        for slot in 0..3 {
            let inst_off = off + 8 + slot * 8;
            if inst_off + 8 > code.len() {
                break;
            }
            let bytes: [u8; 8] = code[inst_off..inst_off + 8].try_into().unwrap();
            let insn = u64::from_le_bytes(bytes);
            if let Some(decoded) = decode_one(insn) {
                let exit = matches!(decoded.opcode, Opcode::EXIT)
                    && !exit_never_taken(insn)
                    && decoded_pred(insn).is_none();
                out.push(Instruction {
                    byte_offset: inst_off,
                    decoded,
                });
                if exit {
                    return out;
                }
            }
        }
        off += 32;
    }
    out
}

pub fn shader_uses_ldg(code: &[u8]) -> bool {
    walk_instructions(code)
        .iter()
        .any(|i| matches!(i.decoded.opcode, Opcode::LDG))
}

pub fn shader_uses_stg(code: &[u8]) -> bool {
    walk_instructions(code)
        .iter()
        .any(|i| matches!(i.decoded.opcode, Opcode::STG))
}

pub fn extract_fs_tex_ids(code: &[u8], bindless_slot: u8) -> Vec<FsTexId> {
    let mut out: Vec<FsTexId> = Vec::new();
    let push = |id: FsTexId, out: &mut Vec<FsTexId>| {
        if !out.contains(&id) {
            out.push(id);
        }
    };

    for ins in walk_instructions(code) {
        let raw = ins.decoded.raw;
        match ins.decoded.opcode {
            Opcode::TEXS | Opcode::TLDS | Opcode::TLD4S => {
                let idx = texs_tex_id(raw);
                push(FsTexId::ImmediateTic(idx), &mut out);
            }

            Opcode::TEX | Opcode::TLD | Opcode::TLD4 | Opcode::TXQ | Opcode::TXD => {
                let idx = texs_tex_id(raw);
                push(FsTexId::ImmediateTic(idx), &mut out);
            }

            Opcode::TEX_b | Opcode::TLD_b | Opcode::TLD4_b | Opcode::TXQ_b | Opcode::TXD_b => {}

            Opcode::LDC => {
                let r = ldc_ref(raw);
                if ldc_mode(raw) == LdcMode::Default && r.binding == bindless_slot {
                    let off = r.byte_offset.max(0) as u32;
                    push(FsTexId::BindlessCbufOffset(off), &mut out);
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ldc(byte_off: u32, binding: u8) -> u64 {
        let top = 0xEF90u64 << 48;
        let off_field = (byte_off as u64) & 0xFFFF;
        let cbuf_part = (off_field << 20) | ((binding as u64 & 0x1F) << 36);
        top | cbuf_part
    }

    fn make_group(i0: u64, i1: u64, i2: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(32);
        v.extend_from_slice(&[0u8; 8]);
        v.extend_from_slice(&i0.to_le_bytes());
        v.extend_from_slice(&i1.to_le_bytes());
        v.extend_from_slice(&i2.to_le_bytes());
        v
    }

    #[test]
    fn extract_skips_sched_words() {
        let code = make_group(make_ldc(0x14, 15), make_ldc(0x18, 15), make_ldc(0x14, 15));
        let ids = extract_fs_tex_ids(&code, 15);
        assert_eq!(ids.len(), 2, "dedup expected: got {:?}", ids);
        assert_eq!(ids[0], FsTexId::BindlessCbufOffset(0x14));
        assert_eq!(ids[1], FsTexId::BindlessCbufOffset(0x18));
    }

    #[test]
    fn ldc_from_wrong_binding_ignored() {
        let code = make_group(make_ldc(0x14, 0), make_ldc(0x18, 1), make_ldc(0x1C, 15));
        let ids = extract_fs_tex_ids(&code, 15);
        assert_eq!(ids, vec![FsTexId::BindlessCbufOffset(0x1C)]);
    }

    #[test]
    fn segmented_ldc_is_not_misidentified_as_a_static_bindless_handle() {
        let segmented = make_ldc(0x14, 15) | (2 << 44);
        let code = make_group(segmented, 0, 0);
        assert!(extract_fs_tex_ids(&code, 15).is_empty());
    }

    #[test]
    fn direct_txd_contributes_its_immediate_texture_id() {
        let direct = 0xde38_0081_a047_0e0cu64;
        let code = make_group(direct, direct | (1u64 << 54), 0);
        assert_eq!(
            extract_fs_tex_ids(&code, 15),
            vec![FsTexId::ImmediateTic(8)]
        );
    }

    #[test]
    fn empty_code_returns_empty() {
        let ids = extract_fs_tex_ids(&[], 15);
        assert!(ids.is_empty());
    }

    #[test]
    fn extract_stops_at_exit() {
        let code = make_group(0xe30000000007000f, make_ldc(0x14, 15), make_ldc(0x18, 15));
        let ids = extract_fs_tex_ids(&code, 15);
        assert!(
            ids.is_empty(),
            "post-exit bytes must not be walked: {ids:?}"
        );
    }

    #[test]
    fn extract_continues_after_predicated_exit_until_unpredicated_exit() {
        let mut code = make_group(0xe30000000000000f, make_ldc(0x14, 15), make_ldc(0x18, 15));
        code.extend(make_group(
            0xe30000000007000f,
            make_ldc(0x1c, 15),
            make_ldc(0x20, 15),
        ));

        let ids = extract_fs_tex_ids(&code, 15);

        assert_eq!(
            ids,
            vec![
                FsTexId::BindlessCbufOffset(0x14),
                FsTexId::BindlessCbufOffset(0x18),
            ]
        );
    }
}

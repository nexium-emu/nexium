pub mod cfg;
mod decode;
mod disasm;
pub mod ir;
mod opcodes;
mod operand;
mod pretty;
mod translate;
mod walk;

pub use cfg::{
    build_cfg, build_cfg_with_cbuf, build_compute_cfg, build_compute_cfg_with_cbuf,
    build_fragment_cfg, build_fragment_cfg_with_cbuf, collect_storage_buffers,
    merge_dual_vertex_sass, BasicBlock, BlockId, BranchKind, Cfg, IndirectBranchTarget,
    StorageBufferAddr, StorageBufferIndirection, MAX_INDIRECT_BRANCH_TARGETS,
};
pub use decode::{decode_one, Decoded};
pub use disasm::{disassemble, DisasmKind, DisasmLine};
pub use ir::{
    BoolOp, CbufAddressMode, FComp, HalfMerge, HalfPrecision, HalfSwizzle, ICmp, ImageAtomicOp,
    ImageAtomicType, ImageDimension, Inst as IrInst, LogicOp, MemoryBarrierScope, MufuFunc,
    Op as IrOp, Predicate, Program as IrProgram, ShaderStage, SubgroupMask, TextureHandleOrigin,
    Value as IrValue, ValueId, VoteMode,
};
pub use opcodes::{Opcode, OPCODE_TABLE};
pub use operand::{cbuf, fmt_cbuf, fmt_reg, imm20, imm32, reg_a, reg_b, reg_c, reg_dest, CbufRef};
pub use pretty::pretty_operands;
pub use translate::{translate_compute_shader, translate_shader, Translator};
pub use walk::{
    extract_fs_tex_ids, shader_uses_ldg, shader_uses_stg, walk_instructions, FsTexId, Instruction,
};

const BINDLESS_TEXTURE_ID_TAG: u32 = 1 << 31;
const BINDLESS_TEXTURE_BINDING_SHIFT: u32 = 26;
const BINDLESS_TEXTURE_PRIMARY_SHIFT: u32 = 13;
const BINDLESS_TEXTURE_BINDING_MASK: u32 = (1 << 5) - 1;
const BINDLESS_TEXTURE_WORD_MASK: u32 = (1 << 13) - 1;

pub fn bindless_texture_id(cbuf_binding: u8, cbuf_word_offset: u32) -> u32 {
    bindless_texture_id_pair(cbuf_binding, cbuf_word_offset, None)
}

pub fn bindless_texture_id_pair(
    cbuf_binding: u8,
    cbuf_word_offset: u32,
    cbuf_secondary_word_offset: Option<u32>,
) -> u32 {
    assert!(u32::from(cbuf_binding) <= BINDLESS_TEXTURE_BINDING_MASK);
    let (primary, secondary) = match cbuf_secondary_word_offset {
        Some(secondary) if secondary < cbuf_word_offset => (secondary, Some(cbuf_word_offset)),
        Some(secondary) if secondary == cbuf_word_offset => (cbuf_word_offset, None),
        secondary => (cbuf_word_offset, secondary),
    };
    assert!(primary <= BINDLESS_TEXTURE_WORD_MASK);
    assert!(secondary.is_none_or(|word_offset| word_offset <= BINDLESS_TEXTURE_WORD_MASK));
    let secondary = secondary.unwrap_or(primary);
    BINDLESS_TEXTURE_ID_TAG
        | (u32::from(cbuf_binding) << BINDLESS_TEXTURE_BINDING_SHIFT)
        | (primary << BINDLESS_TEXTURE_PRIMARY_SHIFT)
        | secondary
}

pub fn decode_bindless_texture_id(texture_id: u32) -> Option<(u8, u32, Option<u32>)> {
    if texture_id & BINDLESS_TEXTURE_ID_TAG == 0 {
        return None;
    }
    let primary = (texture_id >> BINDLESS_TEXTURE_PRIMARY_SHIFT) & BINDLESS_TEXTURE_WORD_MASK;
    let secondary = texture_id & BINDLESS_TEXTURE_WORD_MASK;
    if secondary < primary {
        return None;
    }
    Some((
        ((texture_id >> BINDLESS_TEXTURE_BINDING_SHIFT) & BINDLESS_TEXTURE_BINDING_MASK) as u8,
        primary,
        (secondary != primary).then_some(secondary),
    ))
}

pub fn texture_handle_for_id(texture_id: u32) -> TextureHandleOrigin {
    match decode_bindless_texture_id(texture_id) {
        Some((cbuf_binding, cbuf_word_offset, cbuf_secondary_word_offset)) => {
            TextureHandleOrigin::Bindless {
                cbuf_binding,
                cbuf_word_offset,
                cbuf_secondary_word_offset,
            }
        }
        None => TextureHandleOrigin::Bound {
            cbuf_word_offset: texture_id,
        },
    }
}

pub fn texel_fetch_buffer_coordinates_compatible(y: Option<&IrValue>, z: Option<&IrValue>) -> bool {
    if z.is_some() {
        return false;
    }
    match y {
        None | Some(IrValue::Zero) => true,
        Some(IrValue::ImmU32(value)) => *value == 0,
        Some(IrValue::ImmF32(value)) => value.to_bits() == 0,
        Some(IrValue::Inst(_) | IrValue::GprIn(_)) => false,
    }
}

#[derive(Clone, Debug, Default)]
pub struct IrConstantFacts {
    values: std::collections::HashMap<ValueId, u32>,
}

impl IrConstantFacts {
    pub fn analyze(cfg: &Cfg) -> Self {
        let mut facts = Self::default();
        let instruction_count = cfg
            .blocks
            .iter()
            .map(|block| block.program.instructions.len())
            .sum::<usize>();

        for _ in 0..=instruction_count {
            let mut changed = false;
            for instruction in cfg
                .blocks
                .iter()
                .flat_map(|block| &block.program.instructions)
            {
                let Some(result) = instruction.result else {
                    continue;
                };
                if facts.values.contains_key(&result)
                    || instruction
                        .pred
                        .is_some_and(|pred| pred.idx != 7 || pred.negate)
                {
                    continue;
                }
                if let Some(value) = facts.evaluate_op(&instruction.op) {
                    facts.values.insert(result, value);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        facts
    }

    pub fn value_u32(&self, value: &IrValue) -> Option<u32> {
        match value {
            IrValue::Zero => Some(0),
            IrValue::ImmU32(value) => Some(*value),
            IrValue::ImmF32(value) => Some(value.to_bits()),
            IrValue::Inst(value) => self.values.get(value).copied(),
            IrValue::GprIn(_) => None,
        }
    }

    pub fn value_is_zero(&self, value: &IrValue) -> bool {
        self.value_u32(value) == Some(0)
    }

    pub fn texel_fetch_buffer_coordinates_compatible(
        &self,
        y: Option<&IrValue>,
        z: Option<&IrValue>,
    ) -> bool {
        z.is_none() && y.is_none_or(|value| self.value_is_zero(value))
    }

    fn evaluate_op(&self, op: &IrOp) -> Option<u32> {
        let value = |value: &IrValue| self.value_u32(value);
        match op {
            IrOp::Mov(source) => value(source),
            IrOp::SelectPred {
                if_true, if_false, ..
            } if value(if_true) == value(if_false) => value(if_true),
            IrOp::Phi { sources } if !sources.is_empty() => {
                let first = value(&sources[0].1)?;
                sources
                    .iter()
                    .all(|(_, source)| value(source) == Some(first))
                    .then_some(first)
            }
            _ => None,
        }
    }
}

pub fn texture_ids(cfg: &Cfg) -> Vec<u32> {
    let mut ids = std::collections::BTreeSet::new();
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            match &inst.op {
                IrOp::SampleTex { tex_id, .. } | IrOp::GatherTex { tex_id, .. } => {
                    ids.insert(*tex_id);
                }
                IrOp::TexelFetch {
                    cbuf_binding,
                    cbuf_word_offset,
                    cbuf_secondary_word_offset,
                    ..
                } => {
                    ids.insert(bindless_texture_id_pair(
                        *cbuf_binding,
                        *cbuf_word_offset,
                        *cbuf_secondary_word_offset,
                    ));
                }
                IrOp::SampleTexHandle { handle, .. }
                | IrOp::TexelFetchHandle { handle, .. }
                | IrOp::TextureQueryDimension { handle, .. }
                | IrOp::TextureQueryLod { handle, .. }
                | IrOp::ImageWrite { handle, .. }
                | IrOp::ImageRead { handle, .. }
                | IrOp::ImageAtomic { handle, .. } => match handle {
                    TextureHandleOrigin::Bound { cbuf_word_offset } => {
                        ids.insert(*cbuf_word_offset);
                    }
                    TextureHandleOrigin::Bindless {
                        cbuf_binding,
                        cbuf_word_offset,
                        cbuf_secondary_word_offset,
                    } => {
                        ids.insert(bindless_texture_id_pair(
                            *cbuf_binding,
                            *cbuf_word_offset,
                            *cbuf_secondary_word_offset,
                        ));
                    }
                },
                _ => {}
            }
        }
    }
    ids.into_iter().collect()
}

pub fn shader_uses_texel_fetch(cfg: &Cfg) -> bool {
    cfg.blocks.iter().any(|block| {
        block.program.instructions.iter().any(|inst| {
            matches!(
                inst.op,
                IrOp::TexelFetch { .. } | IrOp::TexelFetchHandle { .. }
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_block_cfg(program: IrProgram) -> Cfg {
        Cfg {
            blocks: vec![BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 0,
                branch: BranchKind::Exit,
                program,
                reg_exit: Default::default(),
                pred_phis: Vec::new(),
                pred_exit: Default::default(),
            }],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    #[test]
    fn bindless_texture_ids_round_trip_five_bit_cbuf_banks() {
        for binding in [0, 15, 16, 17, 31] {
            for word_offset in [0, 1, 0x1ffe, 0x1fff] {
                let texture_id = bindless_texture_id(binding, word_offset);
                assert_eq!(
                    decode_bindless_texture_id(texture_id),
                    Some((binding, word_offset, None))
                );
            }

            for (first, second) in [(0, 1), (0, 0x1fff), (0x1ffe, 0x1fff)] {
                let texture_id = bindless_texture_id_pair(binding, second, Some(first));
                assert_eq!(
                    decode_bindless_texture_id(texture_id),
                    Some((binding, first, Some(second)))
                );
                assert_eq!(
                    texture_id,
                    bindless_texture_id_pair(binding, first, Some(second))
                );
            }
        }

        assert_ne!(bindless_texture_id(0, 0), bindless_texture_id(16, 0));
        assert_ne!(bindless_texture_id(1, 0), bindless_texture_id(17, 0));
    }

    #[test]
    fn malformed_bindless_texture_id_fails_closed() {
        let texture_id = BINDLESS_TEXTURE_ID_TAG
            | (17 << BINDLESS_TEXTURE_BINDING_SHIFT)
            | (5 << BINDLESS_TEXTURE_PRIMARY_SHIFT)
            | 4;
        assert_eq!(decode_bindless_texture_id(texture_id), None);
    }

    #[test]
    fn buffer_fetch_coordinates_accept_only_absent_or_literal_zero_y() {
        assert!(texel_fetch_buffer_coordinates_compatible(None, None));
        assert!(texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::Zero),
            None
        ));
        assert!(texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::ImmU32(0)),
            None
        ));
        assert!(texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::ImmF32(0.0)),
            None
        ));

        assert!(!texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::ImmU32(1)),
            None
        ));
        assert!(!texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::ImmF32(-0.0)),
            None
        ));
        assert!(!texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::GprIn(4)),
            None
        ));
        assert!(!texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::Zero),
            Some(&IrValue::Zero)
        ));
    }

    #[test]
    fn buffer_fetch_coordinates_follow_only_proven_zero_ssa_chains() {
        let mut program = IrProgram::new();
        let moved_zero = program.emit(IrOp::Mov(IrValue::Zero), Some(4));
        let copied_zero = program.emit(IrOp::Mov(IrValue::Inst(moved_zero)), Some(5));
        let selected_zero = program.emit(
            IrOp::SelectPred {
                pred: Predicate {
                    idx: 0,
                    negate: false,
                },
                if_true: IrValue::Inst(copied_zero),
                if_false: IrValue::Zero,
            },
            Some(6),
        );
        let dynamic = program.emit(IrOp::Mov(IrValue::GprIn(7)), Some(7));
        let arithmetic_zero = program.emit(
            IrOp::IAdd {
                a: IrValue::Zero,
                b: IrValue::Zero,
                neg_a: false,
                neg_b: false,
            },
            Some(9),
        );
        let predicated_zero = program.emit_pred(
            IrOp::Mov(IrValue::Zero),
            Some(8),
            Some(Predicate {
                idx: 0,
                negate: false,
            }),
        );
        let cfg = single_block_cfg(program);
        let facts = IrConstantFacts::analyze(&cfg);

        assert!(facts
            .texel_fetch_buffer_coordinates_compatible(Some(&IrValue::Inst(selected_zero)), None,));
        assert!(
            !facts.texel_fetch_buffer_coordinates_compatible(Some(&IrValue::Inst(dynamic)), None,)
        );
        assert!(!facts.texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::Inst(arithmetic_zero)),
            None,
        ));
        assert!(!facts.texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::Inst(predicated_zero)),
            None,
        ));
        assert!(!facts.texel_fetch_buffer_coordinates_compatible(
            Some(&IrValue::Inst(selected_zero)),
            Some(&IrValue::Zero),
        ));
    }
}

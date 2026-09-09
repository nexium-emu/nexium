use std::collections::HashMap;

use nexium_shader::cfg::{BasicBlock, BranchKind, Cfg};
use nexium_shader::{IrProgram, IrValue, Predicate};
use nexium_spirv::{emit_fragment_full_with_options, FragmentOptions};
use rspirv::dr::{Module, Operand};
use rspirv::spirv::{BuiltIn, Decoration, ExecutionMode, Op};

fn block(id: u32, branch: BranchKind, registers: &[(u8, f32)]) -> BasicBlock {
    let mut program = IrProgram::new();
    program.exit_reg_state = Some(
        registers
            .iter()
            .map(|&(register, value)| (register, IrValue::ImmF32(value)))
            .collect(),
    );
    BasicBlock {
        id,
        start_offset: id as usize * 8,
        end_offset: id as usize * 8 + 8,
        branch,
        program,
        reg_exit: HashMap::new(),
        pred_phis: Vec::new(),
        pred_exit: HashMap::new(),
    }
}

fn emit(blocks: Vec<BasicBlock>, output_map: u32, writes_depth: bool) -> Module {
    let cfg = Cfg {
        blocks,
        unimplemented: 0,
        bindless_or_partners: HashMap::new(),
    };
    emit_cfg(&cfg, output_map, writes_depth)
}

fn emit_cfg(cfg: &Cfg, output_map: u32, writes_depth: bool) -> Module {
    let (words, _, _, _, _) = emit_fragment_full_with_options(
        cfg,
        [0; 32],
        8,
        output_map,
        false,
        0,
        0,
        FragmentOptions {
            writes_depth,
            ..FragmentOptions::default()
        },
    );
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let naga = naga::front::spv::parse_u8_slice(&bytes, &Default::default()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&naga)
    .unwrap();
    rspirv::dr::load_words(words).unwrap()
}

fn depth_variable(module: &Module) -> Option<u32> {
    module.annotations.iter().find_map(|instruction| {
        match instruction.operands.as_slice() {
            [Operand::IdRef(variable), Operand::Decoration(Decoration::BuiltIn), Operand::BuiltIn(BuiltIn::FragDepth)] => {
                Some(*variable)
            }
            _ => None,
        }
    })
}

fn stored_depth_values(module: &Module) -> Vec<u32> {
    let depth = depth_variable(module).expect("FragDepth output");
    module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| match instruction.operands.as_slice() {
            [Operand::IdRef(variable), Operand::IdRef(value)]
                if instruction.class.opcode == Op::Store && *variable == depth =>
            {
                Some(*value)
            }
            _ => None,
        })
        .collect()
}

fn constant_bits(module: &Module, id: u32) -> u32 {
    let constant = module
        .types_global_values
        .iter()
        .find(|instruction| instruction.result_id == Some(id))
        .expect("constant depth");
    assert_eq!(constant.class.opcode, Op::Constant);
    match constant.operands.as_slice() {
        [Operand::LiteralBit32(bits)] => *bits,
        other => panic!("unexpected depth constant: {other:?}"),
    }
}

#[test]
fn depth_only_shader_exports_r1_without_color_fallback() {
    let module = emit(
        vec![block(0, BranchKind::Exit, &[(0, 0.9), (1, 0.25)])],
        0,
        true,
    );
    let values = stored_depth_values(&module);
    assert_eq!(values.len(), 1);
    assert_eq!(constant_bits(&module, values[0]), 0.25f32.to_bits());
    assert!(module.execution_modes.iter().any(|instruction| {
        instruction.operands.get(1) == Some(&Operand::ExecutionMode(ExecutionMode::DepthReplacing))
    }));
    assert!(!module.annotations.iter().any(|instruction| {
        instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Location))
    }));
}

#[test]
fn sparse_color_outputs_reserve_four_registers_per_active_target_before_depth() {
    let registers = (0..34)
        .map(|register| (register, f32::from(register) / 64.0))
        .collect::<Vec<_>>();
    let module = emit(vec![block(0, BranchKind::Exit, &registers)], 0x801, true);
    let values = stored_depth_values(&module);
    assert_eq!(values.len(), 1);
    assert_eq!(constant_bits(&module, values[0]), (9.0f32 / 64.0).to_bits());
}

#[test]
fn conditional_exits_keep_their_own_depth_register_values() {
    let module = emit(
        vec![
            block(
                0,
                BranchKind::Conditional {
                    target: 2,
                    pred: Predicate {
                        idx: 7,
                        negate: false,
                    },
                },
                &[],
            ),
            block(1, BranchKind::Exit, &[(5, 0.25)]),
            block(2, BranchKind::Exit, &[(5, 0.75)]),
        ],
        0xf,
        true,
    );
    let mut depths = stored_depth_values(&module)
        .into_iter()
        .map(|id| constant_bits(&module, id))
        .collect::<Vec<_>>();
    depths.sort_unstable();
    assert_eq!(depths, [0.25f32.to_bits(), 0.75f32.to_bits()]);
}

#[test]
fn ordinary_color_shader_keeps_rasterized_depth() {
    let module = emit(
        vec![block(0, BranchKind::Exit, &[(0, 0.8), (5, 0.25)])],
        0xf,
        false,
    );
    assert!(depth_variable(&module).is_none());
    assert!(!module.execution_modes.iter().any(|instruction| {
        instruction.operands.get(1) == Some(&Operand::ExecutionMode(ExecutionMode::DepthReplacing))
    }));
}

#[test]
fn gathered_scene_depth_reaches_export_after_color_registers_are_replaced() {
    let instructions = [
        0x0100_0000_0007_f002u64,
        0x0103_f000_0007_f001,
        0xdf02_0080_2017_0200,
        0x5c60_1380_0037_0203,
        0x5c60_1380_0037_0103,
        0x5c60_1380_0037_0005,
        0x0103_f600_0007_f000,
        0x0103_f600_0007_f001,
        0x0103_f600_0007_f002,
        0x0103_f600_0007_f003,
        0xe300_0000_0007_000f,
    ];
    let mut bytes = Vec::new();
    for chunk in instructions.chunks(3) {
        bytes.extend_from_slice(&[0; 8]);
        for word in chunk {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    let cfg = nexium_shader::build_fragment_cfg(&bytes);
    assert_eq!(cfg.unimplemented, 0);
    let module = emit_cfg(&cfg, 0xf, true);
    let values = stored_depth_values(&module);
    assert_eq!(values.len(), 1);
    let definitions = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect::<HashMap<_, _>>();
    let mut pending = values;
    let mut visited = std::collections::HashSet::new();
    let mut depends_on_gather = false;
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let Some(instruction) = definitions.get(&id) else {
            continue;
        };
        depends_on_gather |= instruction.class.opcode == Op::ImageGather;
        pending.extend(
            instruction
                .operands
                .iter()
                .filter_map(|operand| match operand {
                    Operand::IdRef(id) => Some(*id),
                    _ => None,
                }),
        );
    }
    assert!(
        depends_on_gather,
        "depth export lost the gathered scene depth"
    );
}

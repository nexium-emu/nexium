use nexium_shader::{BasicBlock, BranchKind, Cfg, IrOp, IrValue, LogicOp, Translator, ValueId};
use nexium_spirv::{emit_fragment_full_with_options, FragmentOptions};
use rspirv::dr::Operand;
use rspirv::spirv::{BuiltIn, Capability};
use std::collections::HashMap;

fn value(v: IrValue, values: &HashMap<ValueId, u32>, index: u32) -> u32 {
    match v {
        IrValue::Inst(id) => values[&id],
        IrValue::GprIn(4) => index,
        IrValue::ImmU32(v) => v,
        IrValue::Zero => 0,
        other => panic!("unexpected input {other:?}"),
    }
}

#[test]
fn ldc_subwords_preserve_address_lanes_and_sign_extension() {
    let bytes = [0x01u8, 0x82, 0x7f, 0xfe, 0x55, 0xa6, 0x80, 0xff];
    for size in 0..4u64 {
        for offset in [-4i16, 0, 4] {
            for lane in 0..8u32 {
                let index = lane.wrapping_sub(offset as i32 as u32);
                let raw = (0xef90u64 << 48)
                    | (size << 48)
                    | ((offset as u16 as u64) << 20)
                    | (7 << 16)
                    | (4 << 8)
                    | 4;
                let mut translator = Translator::new_fragment();
                assert!(translator.translate(raw));
                let mut values = HashMap::new();
                let mut result = None;
                for inst in &translator.program.instructions {
                    let v = |x| value(x, &values, index);
                    let output = match inst.op {
                        IrOp::LoadCbufIndexed {
                            byte_offset,
                            index: source,
                            ..
                        } => {
                            let address = v(source).wrapping_add(byte_offset) as usize & !3;
                            u32::from_le_bytes(bytes[address..address + 4].try_into().unwrap())
                        }
                        IrOp::IAdd {
                            a,
                            b,
                            neg_a: false,
                            neg_b: false,
                        } => v(a).wrapping_add(v(b)),
                        IrOp::IShl { a, b } => v(a) << v(b),
                        IrOp::ILop {
                            a,
                            b,
                            op: LogicOp::And,
                            not_a: false,
                            not_b: false,
                        } => v(a) & v(b),
                        IrOp::Bfe { a, b, signed } => {
                            let control = v(b);
                            let width = (control >> 8) & 0xff;
                            let shifted = v(a) >> (control & 0xff);
                            if signed {
                                ((shifted << (32 - width)) as i32 >> (32 - width)) as u32
                            } else {
                                shifted & ((1 << width) - 1)
                            }
                        }
                        IrOp::Mov(source) => v(source),
                        ref other => panic!("unexpected operation {other:?}"),
                    };
                    values.insert(inst.result.unwrap(), output);
                    if inst.dest_reg == Some(4) {
                        result = Some(output);
                    }
                }
                let half = u16::from_le_bytes([
                    bytes[lane as usize & !1],
                    bytes[(lane as usize & !1) + 1],
                ]);
                let expected = match size {
                    0 => bytes[lane as usize] as u32,
                    1 => bytes[lane as usize] as i8 as i32 as u32,
                    2 => half as u32,
                    3 => half as i16 as i32 as u32,
                    _ => unreachable!(),
                };
                assert_eq!(
                    result,
                    Some(expected),
                    "size={size} offset={offset} lane={lane}"
                );
            }
        }
    }
}

#[test]
fn pixld_rejects_unimplemented_modes_and_nonfragment_stages() {
    const RAW: u64 = 0xefe8e0028007ff00;
    for raw in [
        RAW & !(7 << 31),
        RAW ^ (1 << 8),
        RAW | (1 << 20),
        RAW & !(7 << 45),
    ] {
        assert!(!Translator::new_fragment().translate(raw));
    }
    assert!(!Translator::new().translate(RAW));
    assert!(!Translator::new_compute().translate(RAW));
}

#[test]
fn al2p_preserves_index_and_signed_attribute_offset() {
    for (raw, offset) in [(0xefa0700009c70916, 0x9c), (0xefa070007ff70916, u32::MAX)] {
        let mut translator = Translator::new();
        assert!(translator.translate(raw));
        assert!(matches!(translator.program.instructions[0].op,
            IrOp::IAdd { a: IrValue::GprIn(9), b: IrValue::ImmU32(value), neg_a: false, neg_b: false }
                if value == offset));
        assert_eq!(translator.program.instructions[0].dest_reg, Some(22));
    }
    assert!(!Translator::new().translate(0xefa0f00009c70916));
    assert!(!Translator::new_compute().translate(0xefa0700009c70916));
}

#[test]
fn captured_mpr_operations_emit_valid_fragment_spirv() {
    for single_sample in [false, true] {
        let mut translator = Translator::new_fragment();
        for raw in [0xefe8e0028007ff00, 0xef90000062070404, 0xefa0000008070209, 0xe06001c800470900, 0xda4000aff2c72717, 0xe30000000007000f] {
            assert!(translator.translate(raw), "{raw:#018x}");
        }
        let cfg = Cfg {
            blocks: vec![BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 24,
                branch: BranchKind::Exit,
                program: translator.program,
                reg_exit: HashMap::new(),
                pred_phis: Vec::new(),
                pred_exit: HashMap::new(),
            }],
            unimplemented: 0,
            bindless_or_partners: HashMap::new(),
        };
        let (words, ..) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                single_sample,
                texture_numeric_manifest: vec![nexium_spirv::GraphicsTextureResource::new(10, 0, nexium_spirv::TextureNumericType::Uint)],
                ..Default::default()
            },
        );
        let module = rspirv::dr::load_words(&words).unwrap();
        let sample_builtin = module
            .annotations
            .iter()
            .any(|inst| inst.operands.contains(&Operand::BuiltIn(BuiltIn::SampleId)));
        let sample_capability = module.capabilities.iter().any(|inst| {
            inst.operands
                .contains(&Operand::Capability(Capability::SampleRateShading))
        });
        assert_eq!(sample_builtin, !single_sample);
        assert_eq!(sample_capability, !single_sample);
        let path = std::env::temp_dir().join(format!(
            "nexium-mpr-{}-{single_sample}.spv",
            std::process::id()
        ));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        let result = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output();
        std::fs::remove_file(path).unwrap();
        match result {
            Ok(output) => assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("spirv-val: {error}"),
        }
    }
}

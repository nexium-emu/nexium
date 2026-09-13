use nexium_shader::{ICmp, IrOp, IrValue, LogicOp, Translator};
use std::collections::HashMap;

fn light_index(mask: u32, reverse: bool) -> u32 {
    let mut translator = Translator::new_fragment();
    let bfe =
        (0x3800u64 << 48) | (u64::from(reverse) << 40) | (0x2000 << 20) | (7 << 16) | (4 << 8);
    assert!(translator.translate(bfe));
    assert!(translator.translate(0x5c30_0000_0007_0000 | (1 << 41)));
    let mut values = HashMap::new();
    let mut result = None;
    for inst in &translator.program.instructions {
        let v = |source| match source {
            IrValue::GprIn(4) => mask,
            IrValue::Inst(id) => values[&id],
            IrValue::ImmU32(value) => value,
            IrValue::Zero => 0,
            other => panic!("unexpected source {other:?}"),
        };
        let output = match inst.op {
            IrOp::BitReverse { value } => v(value).reverse_bits(),
            IrOp::Bfe {
                a,
                b,
                signed: false,
            } => {
                assert_eq!(v(b), 0x2000);
                v(a)
            }
            IrOp::FindUMsb { value } => 31u32.wrapping_sub(v(value).leading_zeros()),
            IrOp::ISet {
                cmp: ICmp::Ne,
                a,
                b,
                bool_float: false,
                ..
            } => {
                if v(a) != v(b) {
                    u32::MAX
                } else {
                    0
                }
            }
            IrOp::ILop {
                a,
                b,
                op,
                not_a: false,
                not_b: false,
            } => match op {
                LogicOp::And => v(a) & v(b),
                LogicOp::Xor => v(a) ^ v(b),
                other => panic!("unexpected logic {other:?}"),
            },
            ref other => panic!("unexpected operation {other:?}"),
        };
        values.insert(inst.result.unwrap(), output);
        if inst.dest_reg == Some(0) {
            result = Some(output);
        }
    }
    result.unwrap()
}

#[test]
fn bfe_brev_flo_consumes_each_light_once() {
    for original in [9u32, 0x8000_0001, 0xa5f0_9018, u32::MAX, 0] {
        let mut mask = original;
        let mut iterations = 0;
        while mask != 0 {
            let index = light_index(mask, true);
            assert_eq!(index, mask.trailing_zeros(), "mask={mask:#010x}");
            mask &= !(1 << index);
            iterations += 1;
            assert!(iterations <= 32);
        }
        assert_eq!(iterations, original.count_ones());
    }
    assert_eq!(light_index(0, true), u32::MAX);
    assert_eq!(light_index(9, false), 28);
}

use rspirv::binary::Assemble;
use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

#[derive(PartialEq, Eq, Hash)]
struct CanonKey {
    opcode: Op,
    result_type: Option<Word>,
    operands: Vec<Operand>,
}

fn resolve_id(mut id: Word, remap: &HashMap<Word, Word>) -> Word {
    while let Some(&next) = remap.get(&id) {
        if next == id {
            break;
        }
        id = next;
    }
    id
}

fn operand_id(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) => Some(*id),
        _ => None,
    }
}

fn remap_operand(operand: &mut Operand, remap: &HashMap<Word, Word>) {
    match operand {
        Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) => {
            *id = resolve_id(*id, remap);
        }
        _ => {}
    }
}

fn is_deduplicable_constant(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::ConstantTrue
            | Op::ConstantFalse
            | Op::Constant
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::ConstantNull
    )
}

fn make_canon_key(inst: &Instruction, remap: &HashMap<Word, Word>) -> CanonKey {
    let mut operands = inst.operands.clone();
    for operand in &mut operands {
        remap_operand(operand, remap);
    }
    CanonKey {
        opcode: inst.class.opcode,
        result_type: inst.result_type.map(|id| resolve_id(id, remap)),
        operands,
    }
}

pub fn dedup_constants(words: Vec<u32>) -> Vec<u32> {
    let Ok(mut module) = rspirv::dr::load_words(&words) else {
        return words;
    };

    let mut protected = HashSet::new();
    for inst in module.debug_names.iter().chain(&module.annotations) {
        protected.extend(inst.operands.iter().filter_map(operand_id));
    }
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::TypeForwardPointer {
            protected.extend(inst.operands.iter().filter_map(operand_id));
        }
    }

    let has_continued_constants = module.types_global_values.iter().any(|inst| {
        matches!(
            inst.class.opcode,
            Op::ConstantCompositeContinuedINTEL | Op::SpecConstantCompositeContinuedINTEL
        )
    });
    let mut remap = HashMap::new();
    let mut canon = HashMap::new();

    for inst in &module.types_global_values {
        let Some(result_id) = inst.result_id else {
            continue;
        };
        if protected.contains(&result_id) {
            continue;
        }
        let opcode = inst.class.opcode;
        let deduplicable = rspirv::grammar::reflect::is_type(opcode)
            || is_deduplicable_constant(opcode)
                && !(has_continued_constants && opcode == Op::ConstantComposite);
        if !deduplicable {
            continue;
        }

        let key = make_canon_key(inst, &remap);
        if let Some(&survivor) = canon.get(&key) {
            remap.insert(result_id, survivor);
        } else {
            canon.insert(key, result_id);
        }
    }

    if remap.is_empty() {
        return words;
    }

    module
        .types_global_values
        .retain(|inst| inst.result_id.is_none_or(|id| !remap.contains_key(&id)));
    for inst in module.all_inst_iter_mut() {
        if let Some(result_type) = &mut inst.result_type {
            *result_type = resolve_id(*result_type, &remap);
        }
        for operand in &mut inst.operands {
            remap_operand(operand, &remap);
        }
    }

    module.assemble()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::{ExecutionMode, StorageClass};

    fn spv_header(bound: u32) -> Vec<u32> {
        vec![0x07230203, 0x00010000, 0, bound, 0]
    }

    fn op_type_float(result_id: u32) -> Vec<u32> {
        vec![(3 << 16) | Op::TypeFloat as u32, result_id, 32]
    }

    fn op_type_int(result_id: u32, width: u32, signedness: u32) -> Vec<u32> {
        vec![(4 << 16) | Op::TypeInt as u32, result_id, width, signedness]
    }

    fn op_type_pointer(result_id: u32, storage_class: StorageClass, pointee: u32) -> Vec<u32> {
        vec![
            (4 << 16) | Op::TypePointer as u32,
            result_id,
            storage_class as u32,
            pointee,
        ]
    }

    fn op_constant_f32(type_id: u32, result_id: u32, bits: u32) -> Vec<u32> {
        vec![(4 << 16) | Op::Constant as u32, type_id, result_id, bits]
    }

    fn op_execution_mode(entry_point: u32, x: u32, y: u32, z: u32) -> Vec<u32> {
        vec![
            (6 << 16) | Op::ExecutionMode as u32,
            entry_point,
            ExecutionMode::LocalSize as u32,
            x,
            y,
            z,
        ]
    }

    #[test]
    fn dedup_removes_identical_type() {
        let mut words = spv_header(4);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_type_float(2));

        let out = dedup_constants(words);
        assert_eq!(count_opcode(&out, Op::TypeFloat), 1);
    }

    #[test]
    fn dedup_removes_identical_constant() {
        let mut words = spv_header(5);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_constant_f32(1, 2, 0));
        words.extend_from_slice(&op_constant_f32(1, 3, 0));

        let out = dedup_constants(words);
        assert_eq!(count_opcode(&out, Op::Constant), 1);
    }

    #[test]
    fn distinct_constants_are_kept() {
        let mut words = spv_header(5);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_constant_f32(1, 2, 0));
        words.extend_from_slice(&op_constant_f32(1, 3, 1065353216));

        let out = dedup_constants(words);
        assert_eq!(count_opcode(&out, Op::Constant), 2);
    }

    #[test]
    fn literal_equal_to_remapped_id_is_unchanged() {
        let mut words = spv_header(34);
        words.extend_from_slice(&op_type_float(31));
        words.extend_from_slice(&op_type_float(32));
        words.extend_from_slice(&op_type_int(33, 32, 0));

        let out = dedup_constants(words);
        let module = rspirv::dr::load_words(out).unwrap();
        let int_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::TypeInt)
            .unwrap();
        assert_eq!(int_type.operands[0], Operand::LiteralBit32(32));
    }

    #[test]
    fn enum_equal_to_remapped_id_is_unchanged() {
        let mut words = spv_header(4);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_type_float(2));
        words.extend_from_slice(&op_type_pointer(3, StorageClass::Uniform, 2));

        let out = dedup_constants(words);
        let module = rspirv::dr::load_words(out).unwrap();
        let pointer = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::TypePointer)
            .unwrap();
        assert_eq!(
            pointer.operands[0],
            Operand::StorageClass(StorageClass::Uniform)
        );
        assert_eq!(pointer.operands[1], Operand::IdRef(1));
    }

    #[test]
    fn instruction_literals_equal_to_remapped_id_are_unchanged() {
        let mut words = spv_header(5);
        words.extend_from_slice(&op_execution_mode(4, 2, 3, 4));
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_type_float(2));

        let out = dedup_constants(words);
        let module = rspirv::dr::load_words(out).unwrap();
        assert_eq!(
            module.execution_modes[0].operands,
            [
                Operand::IdRef(4),
                Operand::ExecutionMode(ExecutionMode::LocalSize),
                Operand::LiteralBit32(2),
                Operand::LiteralBit32(3),
                Operand::LiteralBit32(4),
            ]
        );
    }

    #[test]
    fn constant_literal_equal_to_remapped_id_is_unchanged() {
        let mut words = spv_header(4);
        words.extend_from_slice(&op_type_float(1));
        words.extend_from_slice(&op_type_float(2));
        words.extend_from_slice(&op_constant_f32(1, 3, 2));

        let out = dedup_constants(words);
        let module = rspirv::dr::load_words(out).unwrap();
        let constant = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::Constant)
            .unwrap();
        assert_eq!(constant.operands[0], Operand::LiteralBit32(2));
    }

    #[test]
    fn passthrough_on_non_spirv() {
        let garbage = vec![0xDEAD_BEEFu32; 4];
        let out = dedup_constants(garbage.clone());
        assert_eq!(out, garbage);
    }

    fn count_opcode(words: &[u32], opcode: Op) -> usize {
        let mut i = 5;
        let mut count = 0;
        while i < words.len() {
            let w0 = words[i];
            let wc = (w0 >> 16) as usize;
            if wc == 0 {
                break;
            }
            if (w0 & 0xFFFF) == opcode as u32 {
                count += 1;
            }
            i += wc;
        }
        count
    }
}

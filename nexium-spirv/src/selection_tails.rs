use rspirv::dr::{Function, Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

fn escaping_edge(function: &Function) -> Option<(usize, usize)> {
    let labels: HashMap<Word, usize> = function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.label.as_ref()?.result_id.map(|id| (id, i)))
        .collect();
    let successors: Vec<Vec<usize>> = function
        .blocks
        .iter()
        .map(|block| {
            block
                .instructions
                .last()
                .map(|term| {
                    super::structured_branch_targets(term)
                        .iter()
                        .filter_map(|id| labels.get(id).copied())
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    let mut reachable = vec![false; function.blocks.len()];
    let mut pending = vec![0];
    while let Some(i) = pending.pop() {
        if reachable[i] {
            continue;
        }
        reachable[i] = true;
        pending.extend(successors[i].iter().copied());
    }
    let mut predecessors = vec![Vec::new(); function.blocks.len()];
    for (i, targets) in successors.iter().enumerate() {
        if reachable[i] {
            for &target in targets {
                predecessors[target].push(i);
            }
        }
    }
    let dominators = super::structured_dominators(&reachable, &predecessors);
    for (header, block) in function.blocks.iter().enumerate() {
        if !reachable[header] {
            continue;
        }
        let merge = block.instructions.iter().find_map(|inst| {
            if inst.class.opcode != Op::SelectionMerge {
                return None;
            }
            match inst.operands.first()? {
                Operand::IdRef(id) => labels.get(id).copied(),
                _ => None,
            }
        });
        let Some(merge) = merge else {
            continue;
        };
        for (source, targets) in successors.iter().enumerate() {
            if !reachable[source] || !dominators[source][header] || dominators[source][merge] {
                continue;
            }
            for &target in targets {
                if target != merge && !dominators[target][header] {
                    return Some((source, target));
                }
            }
        }
    }
    None
}

fn trim_phi(inst: &mut Instruction, keep: impl Fn(Word) -> bool) -> Result<(), String> {
    if inst.class.opcode != Op::Phi {
        return Ok(());
    }
    let incoming: Vec<_> = inst
        .operands
        .chunks_exact(2)
        .filter(|pair| matches!(pair[1], Operand::IdRef(parent) if keep(parent)))
        .flat_map(|pair| pair.iter().cloned())
        .collect();
    if incoming.is_empty() {
        return Err("tail split removed every phi input".into());
    }
    if incoming.len() == 2 {
        *inst = Instruction::new(
            Op::CopyObject,
            inst.result_type,
            inst.result_id,
            vec![incoming[0].clone()],
        );
    } else {
        inst.operands = incoming;
    }
    Ok(())
}

fn split_tail(
    function: &mut Function,
    source: usize,
    target: usize,
    bound: &mut Word,
) -> Result<HashMap<Word, Word>, String> {
    let label = |index: usize| {
        function.blocks[index]
            .label
            .as_ref()
            .unwrap()
            .result_id
            .unwrap()
    };
    let source_label = label(source);
    let target_label = label(target);
    let index: HashMap<_, _> = (0..function.blocks.len()).map(|i| (label(i), i)).collect();
    let mut region = HashSet::new();
    let mut pending = vec![target_label];
    while let Some(id) = pending.pop() {
        if !region.insert(id) {
            continue;
        }
        let block = &function.blocks[index[&id]];
        if let Some(term) = block.instructions.last() {
            pending.extend(super::structured_branch_targets(term));
        }
    }
    if region.contains(&source_label) {
        return Err("selection tail contains a cycle".into());
    }
    let mut copies: Vec<_> = function
        .blocks
        .iter()
        .filter(|block| region.contains(&block.label.as_ref().unwrap().result_id.unwrap()))
        .cloned()
        .collect();
    let mut ids = HashMap::new();
    for block in &copies {
        for inst in block.label.iter().chain(&block.instructions) {
            if let Some(id) = inst.result_id {
                ids.insert(id, *bound);
                *bound = bound.checked_add(1).ok_or("SPIR-V id overflow")?;
            }
        }
    }
    let dead_merges: Vec<_> = copies
        .iter()
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::SelectionMerge)
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::IdRef(id)) if !region.contains(id) => Some(*id),
            _ => None,
        })
        .collect();
    for id in dead_merges {
        if ids.contains_key(&id) {
            continue;
        }
        ids.insert(id, *bound);
        *bound = bound.checked_add(1).ok_or("SPIR-V id overflow")?;
        copies.push(rspirv::dr::Block {
            label: Some(Instruction::new(Op::Label, None, Some(id), vec![])),
            instructions: vec![Instruction::new(Op::Unreachable, None, None, vec![])],
        });
    }
    for block in &mut copies {
        let entry = block.label.as_ref().unwrap().result_id == Some(target_label);
        for inst in &mut block.instructions {
            trim_phi(inst, |parent| {
                region.contains(&parent) || (entry && parent == source_label)
            })?;
        }
        for inst in block.label.iter_mut().chain(&mut block.instructions) {
            if let Some(id) = &mut inst.result_id {
                *id = ids[id];
            }
            for operand in &mut inst.operands {
                if let Operand::IdRef(id) = operand {
                    if let Some(replacement) = ids.get(id) {
                        *id = *replacement;
                    }
                }
            }
        }
    }
    let term = function.blocks[source].instructions.last_mut().unwrap();
    for operand in &mut term.operands {
        if *operand == Operand::IdRef(target_label) {
            *operand = Operand::IdRef(ids[&target_label]);
        }
    }
    for inst in &mut function.blocks[target].instructions {
        trim_phi(inst, |parent| parent != source_label)?;
    }
    function.blocks.extend(copies);
    Ok(ids)
}

pub(super) fn repair(module: &mut Module) -> Result<(), String> {
    let mut bound = module.header.as_ref().ok_or("missing SPIR-V header")?.bound;
    let mut mappings = Vec::new();
    for function in &mut module.functions {
        if function.blocks.is_empty()
            || function
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|inst| inst.class.opcode == Op::LoopMerge)
        {
            continue;
        }
        let limit = function.blocks.len().saturating_mul(16).max(128);
        while let Some((source, target)) = escaping_edge(function) {
            if function.blocks.len() >= limit {
                return Err("selection tail split exceeded block budget".into());
            }
            mappings.push(split_tail(function, source, target, &mut bound)?);
        }
    }
    for ids in mappings {
        let additions: Vec<_> = module
            .annotations
            .iter()
            .filter_map(|inst| {
                let Some(Operand::IdRef(target)) = inst.operands.first() else {
                    return None;
                };
                let replacement = *ids.get(target)?;
                let mut copy = inst.clone();
                copy.operands[0] = Operand::IdRef(replacement);
                Some(copy)
            })
            .collect();
        module.annotations.extend(additions);
    }
    module.header.as_mut().unwrap().bound = bound;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::binary::Assemble;
    use rspirv::dr::Builder;
    use rspirv::spirv::*;

    fn shared_tail_shader() -> (Module, [Word; 4]) {
        let mut b = Builder::new();
        b.set_version(1, 0);
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = b.type_void();
        let float = b.type_float(32, None);
        let boolean = b.type_bool();
        let ptr = b.type_pointer(None, StorageClass::Output, float);
        let output = b.variable(ptr, None, StorageClass::Output, None);
        b.decorate(output, Decoration::Location, [Operand::LiteralBit32(0)]);
        let conditions = std::array::from_fn(|_| b.constant_true(boolean));
        let values: [Word; 5] =
            std::array::from_fn(|i| b.constant_bit32(float, ((i + 1) as f32).to_bits()));
        let ty = b.type_function(void, vec![]);
        let main = b
            .begin_function(void, None, FunctionControl::NONE, ty)
            .unwrap();
        let [h0, h1, h2, h3, a, tail_b, m3, m2, m1, join] = std::array::from_fn(|_| b.id());
        for (header, merge, yes, no, condition) in [
            (h0, join, tail_b, h1, conditions[0]),
            (h1, m1, a, h2, conditions[1]),
            (h2, m2, tail_b, h3, conditions[2]),
            (h3, m3, m3, a, conditions[3]),
        ] {
            b.begin_block(Some(header)).unwrap();
            b.selection_merge(merge, SelectionControl::NONE).unwrap();
            b.branch_conditional(condition, yes, no, []).unwrap();
        }
        b.begin_block(Some(a)).unwrap();
        let av = b
            .phi(float, None, [(values[0], h1), (values[1], h3)])
            .unwrap();
        b.branch(m1).unwrap();
        b.begin_block(Some(tail_b)).unwrap();
        let bv = b
            .phi(float, None, [(values[2], h0), (values[3], h2)])
            .unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(m3)).unwrap();
        b.branch(m2).unwrap();
        b.begin_block(Some(m2)).unwrap();
        b.branch(m1).unwrap();
        b.begin_block(Some(m1)).unwrap();
        let mv = b.phi(float, None, [(av, a), (values[4], m2)]).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(join)).unwrap();
        let result = b.phi(float, None, [(bv, tail_b), (mv, m1)]).unwrap();
        b.store(output, result, None, []).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::Fragment, main, "main", [output]);
        b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
        (b.module(), conditions)
    }

    fn evaluate(module: &Module, conditions: [Word; 4], mask: u32) -> u32 {
        let mut values = HashMap::new();
        for inst in &module.types_global_values {
            if let (Some(id), Some(Operand::LiteralBit32(value))) =
                (inst.result_id, inst.operands.first())
            {
                if inst.class.opcode == Op::Constant {
                    values.insert(id, *value);
                }
            }
        }
        for (i, id) in conditions.into_iter().enumerate() {
            values.insert(id, (mask >> i) & 1);
        }
        let f = &module.functions[0];
        let labels: HashMap<_, _> = f
            .blocks
            .iter()
            .map(|b| (b.label.as_ref().unwrap().result_id.unwrap(), b))
            .collect();
        let mut current = f.blocks[0].label.as_ref().unwrap().result_id.unwrap();
        let mut parent = 0;
        let mut result = 0;
        for _ in 0..128 {
            let block = labels[&current];
            let mut next = None;
            for inst in &block.instructions {
                let id = |i| match inst.operands[i] {
                    Operand::IdRef(id) => id,
                    _ => panic!("id"),
                };
                match inst.class.opcode {
                    Op::Phi => {
                        let pair = inst
                            .operands
                            .chunks_exact(2)
                            .find(|p| p[1] == Operand::IdRef(parent))
                            .unwrap();
                        let Operand::IdRef(value) = pair[0] else {
                            panic!("phi")
                        };
                        values.insert(inst.result_id.unwrap(), values[&value]);
                    }
                    Op::CopyObject => {
                        values.insert(inst.result_id.unwrap(), values[&id(0)]);
                    }
                    Op::Store => result = values[&id(1)],
                    Op::Branch => next = Some(id(0)),
                    Op::BranchConditional => {
                        next = Some(id(if values[&id(0)] != 0 { 1 } else { 2 }))
                    }
                    Op::Return => return result,
                    Op::SelectionMerge => (),
                    op => panic!("unexpected {op:?}"),
                }
            }
            parent = current;
            current = next.unwrap();
        }
        panic!("shader did not terminate")
    }

    #[test]
    fn shared_tails_preserve_every_branch_and_phi_value() {
        let (original, conditions) = shared_tail_shader();
        assert!(escaping_edge(&original.functions[0]).is_some());
        let mut repaired = original.clone();
        repair(&mut repaired).unwrap();
        assert!(escaping_edge(&repaired.functions[0]).is_none());
        for mask in 0..16 {
            assert_eq!(
                evaluate(&original, conditions, mask),
                evaluate(&repaired, conditions, mask),
                "mask={mask}"
            );
        }
        let words = repaired.assemble();
        assert!(crate::phi_preds_consistent(&words));
        assert!(crate::validate_structured_cfg(&words).is_ok());
        crate::tests::validates_with_spirv_val_if_available(&words);
    }
}

use rspirv::dr::{Function, Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

type Dominance = (HashMap<Word, usize>, Vec<Vec<usize>>, Vec<bool>, Vec<Vec<bool>>);

fn escaping_edge(function: &Function) -> Option<(usize, usize)> {
    let (labels, successors, reachable, dominators) = dominance(function);
    escaping_edge_in(function, &labels, &successors, &reachable, &dominators)
}

fn dominance(function: &Function) -> Dominance {
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
    (labels, successors, reachable, dominators)
}

fn escaping_edge_in(
    function: &Function,
    labels: &HashMap<Word, usize>,
    successors: &[Vec<usize>],
    reachable: &[bool],
    dominators: &[Vec<bool>],
) -> Option<(usize, usize)> {
    let loops: Vec<_> = function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(header, block)| {
            let inst = block.instructions.iter().find(|inst| inst.class.opcode == Op::LoopMerge)?;
            let (Operand::IdRef(merge), Operand::IdRef(cont)) = (&inst.operands[0], &inst.operands[1]) else {
                return None;
            };
            Some((header, labels[merge], labels[cont]))
        })
        .collect();
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
        let cases = if block.instructions.last().is_some_and(|term| term.class.opcode == Op::Switch) {
            successors[header].clone()
        } else {
            Vec::new()
        };
        for owner in std::iter::once(header).chain(cases.iter().copied()) {
            for (source, targets) in successors.iter().enumerate() {
                if !reachable[source] || !dominators[source][owner] || dominators[source][merge] {
                    continue;
                }
                for &target in targets {
                    let loop_exit = loops.iter().any(|&(loop_header, loop_merge, cont)| {
                        dominators[source][loop_header]
                            && !dominators[source][loop_merge]
                            && (target == loop_merge || target == cont
                                || (target == loop_header && dominators[source][cont]))
                    });
                    let case_entry = owner != header && cases.contains(&target);
                    if target != merge && !dominators[target][owner] && !loop_exit && !case_entry {
                        return Some((source, target));
                    }
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

fn phi_value(inst: &Instruction, parent: Word) -> Option<Word> {
    inst.operands.chunks_exact(2).find_map(|pair| match (&pair[0], &pair[1]) {
        (Operand::IdRef(value), Operand::IdRef(from)) if *from == parent => Some(*value),
        _ => None,
    })
}

fn sink_tail(
    function: &mut Function,
    source: usize,
    target: usize,
    bound: &mut Word,
    in_place: bool,
) -> Option<HashMap<Word, Word>> {
    let label = |function: &Function, index: usize| {
        function.blocks[index].label.as_ref()?.result_id
    };
    let (labels, _, _, dominators) = dominance(function);
    let plain = |block: &rspirv::dr::Block| {
        block.instructions.last().is_some_and(|term| term.class.opcode == Op::Branch)
            && !block.instructions.iter().any(|inst| {
                matches!(inst.class.opcode, Op::SelectionMerge | Op::LoopMerge)
            })
    };
    if in_place && !plain(&function.blocks[source]) {
        return None;
    }
    let tail = &function.blocks[if in_place { source } else { target }];
    let term = tail.instructions.last()?;
    if term.class.opcode != Op::Branch
        || tail.instructions.iter().any(|inst| {
            matches!(inst.class.opcode, Op::SelectionMerge | Op::LoopMerge)
        })
    {
        return None;
    }
    let Operand::IdRef(join) = term.operands[0] else {
        return None;
    };
    let merge = function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(header, block)| {
            let inst = block.instructions.iter().find(|inst| inst.class.opcode == Op::SelectionMerge)?;
            let Operand::IdRef(merge) = inst.operands[0] else {
                return None;
            };
            let merge = *labels.get(&merge)?;
            (dominators[source][header] && !dominators[source][merge]).then_some((header, merge))
        })
        .max_by_key(|&(header, _)| dominators[header].iter().filter(|&&d| d).count())?
        .1;
    let mut chain = vec![merge];
    loop {
        let block = &function.blocks[*chain.last()?];
        let (term, body) = block.instructions.split_last()?;
        if term.class.opcode != Op::Branch || body.iter().any(|inst| inst.class.opcode != Op::Phi) {
            return None;
        }
        let Operand::IdRef(next) = term.operands[0] else {
            return None;
        };
        if next == join {
            break;
        }
        if chain.len() > 64 {
            return None;
        }
        chain.push(*labels.get(&next)?);
    }
    let target_label = label(function, target)?;
    let source_label = label(function, source)?;
    let tail_label = if in_place { source_label } else { target_label };
    let join_index = labels[&join];
    let last_label = label(function, *chain.last()?)?;
    let chain_labels: Vec<Word> =
        chain.iter().map(|&index| label(function, index)).collect::<Option<_>>()?;
    let predecessors = |function: &Function, block: Word| -> Vec<Word> {
        function
            .blocks
            .iter()
            .filter(|candidate| {
                candidate.instructions.last().is_some_and(|term| {
                    super::structured_branch_targets(term).contains(&block)
                })
            })
            .filter_map(|candidate| candidate.label.as_ref()?.result_id)
            .collect()
    };
    let chain_preds: Vec<Vec<Word>> =
        chain_labels.iter().map(|&block| predecessors(function, block)).collect();
    struct Feed {
        join_phi: usize,
        result_type: Option<Word>,
        from_tail: Word,
        values: Vec<Word>,
        phis: Vec<Option<usize>>,
    }
    let mut feeds = Vec::new();
    for (join_phi, inst) in function.blocks[join_index].instructions.iter().enumerate() {
        if inst.class.opcode != Op::Phi {
            continue;
        }
        let from_tail = phi_value(inst, tail_label)?;
        let mut values = vec![0; chain.len()];
        let mut phis = vec![None; chain.len()];
        values[chain.len() - 1] = phi_value(inst, last_label)?;
        for position in (0..chain.len()).rev() {
            let value = values[position];
            phis[position] = function.blocks[chain[position]]
                .instructions
                .iter()
                .position(|inst| inst.class.opcode == Op::Phi && inst.result_id == Some(value));
            if position > 0 {
                values[position - 1] = match phis[position] {
                    Some(phi) => phi_value(
                        &function.blocks[chain[position]].instructions[phi],
                        chain_labels[position - 1],
                    )?,
                    None => value,
                };
            }
        }
        feeds.push(Feed { join_phi, result_type: inst.result_type, from_tail, values, phis });
    }
    let mut copy = function.blocks[if in_place { source } else { target }].clone();
    let mut ids = HashMap::new();
    if !in_place {
        for inst in copy.label.iter().chain(&copy.instructions) {
            if let Some(id) = inst.result_id {
                ids.insert(id, *bound);
                *bound = bound.checked_add(1)?;
            }
        }
        for inst in &mut copy.instructions {
            trim_phi(inst, |parent| parent == source_label).ok()?;
        }
    }
    let copy_label = if in_place { source_label } else { ids[&target_label] };
    for inst in copy.label.iter_mut().chain(&mut copy.instructions) {
        if let Some(id) = &mut inst.result_id {
            *id = *ids.get(id).unwrap_or(id);
        }
        for operand in &mut inst.operands {
            if let Operand::IdRef(id) = operand {
                if let Some(replacement) = ids.get(id) {
                    *id = *replacement;
                }
            }
        }
    }
    let mut fed = HashMap::new();
    for feed in &feeds {
        if let Some(phi) = feed.phis[0] {
            if fed.insert(phi, feed.from_tail).is_some_and(|other| other != feed.from_tail) {
                return None;
            }
        }
    }
    let mut undefs = Vec::new();
    let mut merge_entries = Vec::new();
    for (position, inst) in function.blocks[merge].instructions.iter().enumerate() {
        if inst.class.opcode != Op::Phi {
            continue;
        }
        let value = match fed.get(&position) {
            Some(from_tail) => *ids.get(from_tail).unwrap_or(from_tail),
            None => {
                let undef = *bound;
                *bound = bound.checked_add(1)?;
                undefs.push(Instruction::new(Op::Undef, inst.result_type, Some(undef), vec![]));
                undef
            }
        };
        merge_entries.push((position, value));
    }
    let mut new_phis: Vec<(usize, Instruction)> = Vec::new();
    let mut join_rewrites = Vec::new();
    let mut chain_rewrites = Vec::new();
    for feed in &feeds {
        if feed.phis[0].is_some() && feed.phis[1..].iter().all(Option::is_some) {
            continue;
        }
        let tail_value = *ids.get(&feed.from_tail).unwrap_or(&feed.from_tail);
        let (mut current, mut changed) = match feed.phis[0] {
            Some(_) => (feed.values[0], false),
            None => {
                let id = *bound;
                *bound = bound.checked_add(1)?;
                let mut operands: Vec<Operand> = chain_preds[0]
                    .iter()
                    .flat_map(|&parent| [Operand::IdRef(feed.values[0]), Operand::IdRef(parent)])
                    .collect();
                operands.extend([Operand::IdRef(tail_value), Operand::IdRef(copy_label)]);
                new_phis.push((chain[0], Instruction::new(Op::Phi, feed.result_type, Some(id), operands)));
                (id, true)
            }
        };
        for position in 1..chain.len() {
            if !changed {
                current = feed.values[position];
                continue;
            }
            match feed.phis[position] {
                Some(phi) => {
                    chain_rewrites.push((chain[position], phi, chain_labels[position - 1], current));
                    current = feed.values[position];
                    changed = false;
                }
                None => {
                    let id = *bound;
                    *bound = bound.checked_add(1)?;
                    let operands = chain_preds[position]
                        .iter()
                        .flat_map(|&parent| {
                            let value = if parent == chain_labels[position - 1] {
                                current
                            } else {
                                feed.values[position]
                            };
                            [Operand::IdRef(value), Operand::IdRef(parent)]
                        })
                        .collect();
                    new_phis.push((
                        chain[position],
                        Instruction::new(Op::Phi, feed.result_type, Some(id), operands),
                    ));
                    current = id;
                }
            }
        }
        if changed {
            join_rewrites.push((feed.join_phi, current));
        }
    }
    let last = copy.instructions.len() - 1;
    copy.instructions[last].operands[0] = Operand::IdRef(label(function, merge)?);
    copy.instructions.splice(last..last, undefs);
    for (position, value) in merge_entries {
        function.blocks[merge].instructions[position]
            .operands
            .extend([Operand::IdRef(value), Operand::IdRef(copy_label)]);
    }
    for (block, phi, parent, value) in chain_rewrites {
        for pair in function.blocks[block].instructions[phi].operands.chunks_exact_mut(2) {
            if pair[1] == Operand::IdRef(parent) {
                pair[0] = Operand::IdRef(value);
            }
        }
    }
    for (join_phi, value) in join_rewrites {
        for pair in function.blocks[join_index].instructions[join_phi].operands.chunks_exact_mut(2) {
            if pair[1] == Operand::IdRef(last_label) {
                pair[0] = Operand::IdRef(value);
            }
        }
    }
    for (block, phi) in new_phis.into_iter().rev() {
        function.blocks[block].instructions.insert(0, phi);
    }
    if !in_place {
        for operand in &mut function.blocks[source].instructions.last_mut()?.operands {
            if *operand == Operand::IdRef(target_label) {
                *operand = Operand::IdRef(copy_label);
            }
        }
    }
    for inst in &mut function.blocks[target].instructions {
        trim_phi(inst, |parent| parent != source_label).ok()?;
    }
    if in_place {
        function.blocks[source] = copy;
    } else {
        function.blocks.push(copy);
    }
    Some(ids)
}

pub(super) fn repair(module: &mut Module) -> Result<(), String> {
    let original = std::env::var_os("NEXIUM_DUMP_SPIRV").map(|dir| (dir, module.clone()));
    let result = repair_functions(module);
    if let (Err(_), Some((dir, original))) = (&result, original) {
        use rspirv::binary::Assemble;
        let words = original.assemble();
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let path = std::path::Path::new(&dir).join(format!("tail_failed_{:08x}.spv", words.len()));
        let _ = std::fs::write(path, bytes);
    }
    result
}

fn repair_functions(module: &mut Module) -> Result<(), String> {
    let mut bound = module.header.as_ref().ok_or("missing SPIR-V header")?.bound;
    let mut mappings = Vec::new();
    for function in &mut module.functions {
        if function.blocks.is_empty() {
            continue;
        }
        let limit = function.blocks.len().saturating_mul(16).max(128);
        while let Some((source, target)) = escaping_edge(function) {
            if function.blocks.len() >= limit {
                return Err("selection tail split exceeded block budget".into());
            }
            if let Some(ids) = sink_tail(function, source, target, &mut bound, true)
                .or_else(|| sink_tail(function, source, target, &mut bound, false))
            {
                mappings.push(ids);
                continue;
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
        for inst in &module.types_global_values {
            if let Some(id) = inst.result_id {
                match inst.class.opcode {
                    Op::ConstantFalse => { values.insert(id, 0); }
                    Op::ConstantTrue => { values.insert(id, 1); }
                    _ => {}
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
                    Op::Switch => {
                        let selector = values[&id(0)];
                        next = Some(inst.operands[2..].chunks_exact(2)
                            .find_map(|case| match (&case[0], &case[1]) {
                                (Operand::LiteralBit32(value), Operand::IdRef(target))
                                    if *value == selector => Some(*target),
                                _ => None,
                            })
                            .unwrap_or_else(|| id(1)));
                    }
                    Op::Return => return result,
                    Op::SelectionMerge | Op::LoopMerge => (),
                    op => panic!("unexpected {op:?}"),
                }
            }
            parent = current;
            current = next.unwrap();
        }
        panic!("shader did not terminate")
    }

    fn shared_switch_tail_shader(selector_value: u32) -> Module {
        let mut b = Builder::new();
        b.set_version(1, 0);
        b.capability(Capability::Shader);
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void = b.type_void();
        let float = b.type_float(32, None);
        let uint = b.type_int(32, 0);
        let selector = b.constant_bit32(uint, selector_value);
        let values: [Word; 3] =
            std::array::from_fn(|i| b.constant_bit32(float, ((i + 1) as f32).to_bits()));
        let ptr = b.type_pointer(None, StorageClass::Output, float);
        let output = b.variable(ptr, None, StorageClass::Output, None);
        b.decorate(output, Decoration::Location, [Operand::LiteralBit32(0)]);
        let ty = b.type_function(void, vec![]);
        let main = b.begin_function(void, None, FunctionControl::NONE, ty).unwrap();
        let [header, a, c, default, tail, merge] = std::array::from_fn(|_| b.id());
        b.begin_block(Some(header)).unwrap();
        b.selection_merge(merge, SelectionControl::NONE).unwrap();
        b.switch(selector, default, [(Operand::LiteralBit32(0), a), (Operand::LiteralBit32(1), c)]).unwrap();
        for label in [a, c] {
            b.begin_block(Some(label)).unwrap();
            b.branch(tail).unwrap();
        }
        b.begin_block(Some(default)).unwrap();
        b.branch(merge).unwrap();
        b.begin_block(Some(tail)).unwrap();
        let value = b.phi(float, None, [(values[0], a), (values[1], c)]).unwrap();
        b.branch(merge).unwrap();
        b.begin_block(Some(merge)).unwrap();
        let result = b.phi(float, None, [(value, tail), (values[2], default)]).unwrap();
        b.store(output, result, None, []).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::Fragment, main, "main", [output]);
        b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
        b.module()
    }

    #[test]
    fn switch_case_tails_preserve_all_case_and_default_values() {
        for selector in [0, 1, 2, u32::MAX] {
            let original = shared_switch_tail_shader(selector);
            let mut repaired = original.clone();
            assert!(escaping_edge(&original.functions[0]).is_some());
            repair(&mut repaired).unwrap();
            assert!(escaping_edge(&repaired.functions[0]).is_none());
            let expected = ((selector.min(2) + 1) as f32).to_bits();
            assert_eq!(evaluate(&original, [0; 4], 0), expected);
            assert_eq!(evaluate(&repaired, [0; 4], 0), expected);
            let words = repaired.assemble();
            assert!(crate::phi_preds_consistent(&words));
            assert!(crate::validate_structured_cfg(&words).is_ok());
            crate::tests::validates_with_spirv_val_if_available(&words);
        }
    }

    #[test]
    #[ignore = "set NEXIUM_TAIL_REPRO to a tail_failed_*.spv dump"]
    fn replay_failed_tail_dump() {
        let path = std::env::var("NEXIUM_TAIL_REPRO").unwrap();
        let bytes = std::fs::read(path).unwrap();
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_words(&words, &mut loader).unwrap();
        let mut module = loader.module();
        repair_functions(&mut module).unwrap();
        let words = module.assemble();
        assert!(crate::phi_preds_consistent(&words));
        crate::validate_structured_cfg(&words).unwrap();
        crate::tests::validates_with_spirv_val_if_available(&words);
    }

    #[test]
    fn escaping_tail_sinks_into_the_inner_merge_chain() {
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
        let conditions: [Word; 4] = std::array::from_fn(|_| b.constant_true(boolean));
        let values: [Word; 3] =
            std::array::from_fn(|i| b.constant_bit32(float, ((i + 1) as f32).to_bits()));
        let ty = b.type_function(void, vec![]);
        let main = b.begin_function(void, None, FunctionControl::NONE, ty).unwrap();
        let [outer, inner, other, merge, tail, join] = std::array::from_fn(|_| b.id());
        b.begin_block(Some(outer)).unwrap();
        b.selection_merge(join, SelectionControl::NONE).unwrap();
        b.branch_conditional(conditions[0], tail, inner, []).unwrap();
        b.begin_block(Some(inner)).unwrap();
        b.selection_merge(merge, SelectionControl::NONE).unwrap();
        b.branch_conditional(conditions[1], tail, other, []).unwrap();
        b.begin_block(Some(other)).unwrap();
        b.branch(merge).unwrap();
        b.begin_block(Some(merge)).unwrap();
        let forwarded = b.phi(float, None, [(values[2], other)]).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(tail)).unwrap();
        let shared = b.phi(float, None, [(values[0], outer), (values[1], inner)]).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(join)).unwrap();
        let result = b.phi(float, None, [(shared, tail), (forwarded, merge)]).unwrap();
        b.store(output, result, None, []).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::Fragment, main, "main", [output]);
        b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
        let original = b.module();
        let mut repaired = original.clone();
        assert!(escaping_edge(&original.functions[0]).is_some());
        repair(&mut repaired).unwrap();
        assert!(escaping_edge(&repaired.functions[0]).is_none());
        assert_eq!(repaired.functions[0].blocks.len(), original.functions[0].blocks.len() + 1);
        for mask in 0..4 {
            assert_eq!(evaluate(&repaired, conditions, mask), evaluate(&original, conditions, mask));
        }
        let words = repaired.assemble();
        assert!(crate::phi_preds_consistent(&words));
        assert!(crate::validate_structured_cfg(&words).is_ok());
        crate::tests::validates_with_spirv_val_if_available(&words);
    }

    #[test]
    fn plain_escaping_block_retargets_to_its_merge_with_a_new_phi() {
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
        let conditions: [Word; 4] = std::array::from_fn(|_| b.constant_true(boolean));
        let values: [Word; 3] =
            std::array::from_fn(|i| b.constant_bit32(float, ((i + 1) as f32).to_bits()));
        let ty = b.type_function(void, vec![]);
        let main = b.begin_function(void, None, FunctionControl::NONE, ty).unwrap();
        let [outer, early, inner, escape, other, merge, join] = std::array::from_fn(|_| b.id());
        b.begin_block(Some(outer)).unwrap();
        b.selection_merge(join, SelectionControl::NONE).unwrap();
        b.branch_conditional(conditions[0], early, inner, []).unwrap();
        b.begin_block(Some(early)).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(inner)).unwrap();
        b.selection_merge(merge, SelectionControl::NONE).unwrap();
        b.branch_conditional(conditions[1], escape, other, []).unwrap();
        b.begin_block(Some(escape)).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(other)).unwrap();
        b.branch(merge).unwrap();
        b.begin_block(Some(merge)).unwrap();
        b.branch(join).unwrap();
        b.begin_block(Some(join)).unwrap();
        let result = b
            .phi(float, None, [(values[0], early), (values[1], escape), (values[2], merge)])
            .unwrap();
        b.store(output, result, None, []).unwrap();
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(ExecutionModel::Fragment, main, "main", [output]);
        b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
        let original = b.module();
        let mut repaired = original.clone();
        assert!(escaping_edge(&original.functions[0]).is_some());
        repair(&mut repaired).unwrap();
        assert!(escaping_edge(&repaired.functions[0]).is_none());
        assert_eq!(repaired.functions[0].blocks.len(), original.functions[0].blocks.len());
        for mask in 0..4 {
            assert_eq!(evaluate(&repaired, conditions, mask), evaluate(&original, conditions, mask));
        }
        let words = repaired.assemble();
        assert!(crate::phi_preds_consistent(&words));
        assert!(crate::validate_structured_cfg(&words).is_ok());
        crate::tests::validates_with_spirv_val_if_available(&words);
    }

    fn shared_tail_shader_with_loop() -> (Module, [Word; 4]) {
        let (mut module, conditions) = shared_tail_shader();
        let start = module.header.as_ref().unwrap().bound;
        let [never, header, body, cont, merge, body_merge, always, first_pass] =
            std::array::from_fn(|i| start + i as u32);
        let boolean = module.types_global_values.iter()
            .find(|inst| inst.result_id == Some(conditions[0]))
            .unwrap().result_type;
        module.types_global_values.push(Instruction::new(
            Op::ConstantFalse, boolean, Some(never), Vec::new(),
        ));
        module.types_global_values.push(Instruction::new(
            Op::ConstantTrue, boolean, Some(always), Vec::new(),
        ));
        let function = &mut module.functions[0];
        let join = function.blocks.last_mut().unwrap();
        let join_label = join.label.as_ref().unwrap().result_id.unwrap();
        *join.instructions.last_mut().unwrap() = Instruction::new(
            Op::Branch, None, None, vec![Operand::IdRef(header)],
        );
        for (label, instructions) in [
            (header, vec![
                Instruction::new(Op::Phi, boolean, Some(first_pass), vec![
                    Operand::IdRef(always), Operand::IdRef(join_label),
                    Operand::IdRef(never), Operand::IdRef(cont),
                ]),
                Instruction::new(Op::LoopMerge, None, None, vec![
                    Operand::IdRef(merge), Operand::IdRef(cont), Operand::LoopControl(LoopControl::NONE),
                ]),
                Instruction::new(Op::BranchConditional, None, None, vec![
                    Operand::IdRef(first_pass), Operand::IdRef(body), Operand::IdRef(merge),
                ]),
            ]),
            (body, vec![
                Instruction::new(Op::SelectionMerge, None, None, vec![
                    Operand::IdRef(body_merge), Operand::SelectionControl(SelectionControl::NONE),
                ]),
                Instruction::new(Op::BranchConditional, None, None, vec![
                    Operand::IdRef(conditions[0]), Operand::IdRef(merge), Operand::IdRef(body_merge),
                ]),
            ]),
            (body_merge, vec![Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(cont)])]),
            (cont, vec![Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(header)])]),
            (merge, vec![Instruction::new(Op::Return, None, None, Vec::new())]),
        ] {
            function.blocks.push(rspirv::dr::Block {
                label: Some(Instruction::new(Op::Label, None, Some(label), Vec::new())),
                instructions,
            });
        }
        module.header.as_mut().unwrap().bound = start + 8;
        (module, conditions)
    }

    #[test]
    fn loop_tails_preserve_phi_values_and_allow_structured_loop_exits() {
        let (original, conditions) = shared_tail_shader_with_loop();
        let mut repaired = original.clone();
        assert!(escaping_edge(&original.functions[0]).is_some());
        repair(&mut repaired).unwrap();
        assert!(escaping_edge(&repaired.functions[0]).is_none());
        for mask in 0..16 {
            assert_eq!(evaluate(&original, conditions, mask), evaluate(&repaired, conditions, mask));
        }
        let words = repaired.assemble();
        assert!(crate::phi_preds_consistent(&words));
        assert!(crate::validate_structured_cfg(&words).is_ok());
        crate::tests::validates_with_spirv_val_if_available(&words);
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

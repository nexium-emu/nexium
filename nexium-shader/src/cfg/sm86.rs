use std::cell::Cell;
use std::collections::{BTreeSet, HashMap, HashSet};

use super::{
    build_cfg_blocks_with, compute_predecessors, loop_carried_phi_gaps, BlockId, BlockInfo,
    BranchKind, Cfg, IndirectBranchTarget, LoopCarriedPhis, MAX_INDIRECT_BRANCH_TARGETS, RZ,
};
use crate::ampere::sm70::{Inst, PT};
use crate::ir::{Predicate, ShaderStage};
use crate::translate::sm86::{Sm86Context, BRANCH_PRED};

const SIZE: usize = 16;
const BRA: u16 = 0x947;
const EXIT: u16 = 0x94d;
const BRX: u16 = 0x949;
const LDC: u16 = 0xb82;

fn decode(code: &[u8]) -> Option<Vec<Inst>> {
    let mut instructions = Vec::new();
    for chunk in code.chunks_exact(SIZE) {
        let inst = Inst::from_bytes(chunk);
        instructions.push(inst);
        if inst.is_end() {
            return Some(instructions);
        }
    }
    None
}

fn guard(inst: &Inst) -> Option<Predicate> {
    let (idx, negate) = inst.guard();
    (idx != PT || negate).then_some(Predicate { idx, negate })
}

fn branch_target(index: usize, inst: &Inst, count: usize) -> Option<usize> {
    let target = inst.branch_target(index * SIZE, 34, 48);
    (target >= 0 && (target as usize) < count * SIZE && target as usize % SIZE == 0)
        .then_some(target as usize / SIZE)
}

fn branch_predicate(inst: &Inst) -> (Option<Predicate>, bool) {
    let (condition, negate) = inst.pred(87, 90);
    if (condition, negate) == (PT, false) {
        return (guard(inst), false);
    }
    match guard(inst) {
        None => (Some(Predicate { idx: condition, negate }), false),
        Some(_) => (Some(Predicate { idx: BRANCH_PRED, negate: false }), true),
    }
}

pub(crate) fn build(code: &[u8], stage: ShaderStage, read_cbuf: &mut dyn FnMut(u8, u32) -> Option<u32>) -> Option<Cfg> {
    probe_with_cbuf(code, stage, read_cbuf).ok()
}

const CALL: u16 = 0x944;
const RET: u16 = 0x950;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    None,
    Branch { target: usize, unconditional: bool, invert: bool },
    Exit,
    Indirect(usize),
}

struct Table {
    register: u8,
    base: u32,
    binding: u8,
    offset: u32,
    entries: u8,
    cases: Vec<(u32, usize)>,
}

#[derive(Clone, Copy)]
struct Item {
    inst: Inst,
    flow: Flow,
}

fn plain(inst: &Inst) -> bool {
    inst.guard() == (PT, false) && inst.pred(87, 90) == (PT, false)
}

fn writer(instructions: &[Inst], before: usize, register: u8) -> Option<(usize, Inst)> {
    (0..before)
        .rev()
        .map(|at| (at, instructions[at]))
        .take_while(|(_, inst)| !matches!(inst.opcode(), BRA | CALL | RET | EXIT | BRX | 0x942))
        .find(|(_, inst)| inst.dst() == register)
        .filter(|(_, inst)| inst.guard() == (PT, false))
}

fn jump_table(instructions: &[Inst], index: usize, read_cbuf: &mut dyn FnMut(u8, u32) -> Option<u32>) -> Option<Table> {
    let inst = instructions[index];
    if !plain(&inst) || inst.field(32, 2) != 0 || inst.field(82, 5) != 0 {
        return None;
    }
    let register = inst.reg(24);
    let (at, ldc) = writer(instructions, index, register)?;
    if ldc.opcode() != LDC || ldc.field(73, 3) != 4 || ldc.field(78, 2) != 0 || ldc.bit(91) {
        return None;
    }
    let (binding, offset) = (ldc.field(54, 5) as u8, ldc.field(38, 16) as u32);
    let (at, scale) = writer(instructions, at, ldc.reg(24))?;
    let scaled = match scale.opcode() {
        0x824 => scale.field(32, 32) == 4 && scale.reg(64) == RZ && scale.field(72, 8) == 0,
        0x819 => scale.field(32, 32) == 2 && scale.reg(64) == RZ && !scale.bit(76) && !scale.bit(80),
        _ => false,
    };
    let (_, clamp) = writer(instructions, at, scale.reg(24)).filter(|_| scaled)?;
    let limit = clamp.field(32, 32) as usize;
    if clamp.opcode() != 0x817 || clamp.pred(87, 90) != (PT, false) || clamp.bit(73) || limit >= MAX_INDIRECT_BRANCH_TARGETS {
        return None;
    }
    let base = inst.branch_target(index * SIZE, 34, 48) as u32;
    let mut cases: Vec<(u32, usize)> = Vec::new();
    for entry in 0..=limit as u32 {
        let selector = base.wrapping_add(read_cbuf(binding, offset.checked_add(entry * 4)?)?);
        let target = selector as usize;
        if target % SIZE != 0 || target / SIZE >= instructions.len() {
            return None;
        }
        if !cases.iter().any(|&(known, _)| known == selector) {
            cases.push((selector, target / SIZE));
        }
    }
    Some(Table { register, base, binding, offset, entries: limit as u8 + 1, cases })
}

fn expand(instructions: &[Inst], read_cbuf: &mut dyn FnMut(u8, u32) -> Option<u32>) -> Result<(Vec<Item>, Vec<Table>), String> {
    let count = instructions.len();
    let target_of = |index: usize| branch_target(index, &instructions[index], count).ok_or("branch target");
    let mut items: Vec<Item> = Vec::with_capacity(count);
    let mut main = vec![0usize; count];
    let mut pending = Vec::new();
    let mut tables = Vec::new();
    let targeted: HashSet<usize> = (0..count)
        .filter(|&index| matches!(instructions[index].opcode(), BRA | CALL))
        .filter_map(|index| branch_target(index, &instructions[index], count))
        .collect();
    let mut skip = false;
    for (index, inst) in instructions.iter().enumerate() {
        main[index] = items.len();
        if std::mem::take(&mut skip) {
            continue;
        }
        let next = instructions.get(index + 1);
        let fused = inst.opcode() == CALL
            && inst.field(82, 5) == 0x10
            && branch_target(index, inst, count) == Some(index + 2)
            && !targeted.contains(&(index + 1))
            && next.is_some_and(|next| next.opcode() == BRA && !next.is_end() && plain(next) && next.field(82, 5) == 0 && next.field(32, 2) == 0);
        if fused {
            skip = true;
            if branch_predicate(inst).0.is_some() {
                pending.push((items.len(), target_of(index + 1)?));
                items.push(Item { inst: *inst, flow: Flow::Branch { target: 0, unconditional: false, invert: true } });
            }
            continue;
        }
        let flow = match inst.opcode() {
            BRA if inst.is_end() => Flow::Exit,
            BRA => {
                if inst.field(82, 5) != 0 || inst.field(32, 2) != 0 {
                    return Err(format!("branch modifiers at {index}"));
                }
                pending.push((items.len(), target_of(index)?));
                Flow::Branch { target: 0, unconditional: false, invert: false }
            }
            EXIT => Flow::Exit,
            RET => {
                if !plain(inst) || inst.field(82, 5) != 0x10 {
                    return Err(format!("return form at {index}"));
                }
                Flow::Exit
            }
            CALL if inst.field(82, 5) == 0x10 && branch_target(index, inst, count) == Some(index + 2) => {
                pending.push((items.len(), target_of(index)?));
                Flow::Branch { target: 0, unconditional: false, invert: false }
            }
            CALL => {
                if !plain(inst) || !matches!(inst.field(82, 5), 0 | 0x10) {
                    return Err(format!("call form at {index}"));
                }
                let start = target_of(index)?;
                let end = (start..count).find(|&at| instructions[at].opcode() == RET).ok_or("call without return")?;
                let base = items.len();
                for at in start..=end {
                    let body = instructions[at];
                    let flow = match body.opcode() {
                        RET if at == end => Flow::Branch { target: base + end - start + 1, unconditional: true, invert: false },
                        BRA if !body.is_end() => {
                            if body.field(82, 5) != 0 || body.field(32, 2) != 0 {
                                return Err(format!("branch modifiers at {at}"));
                            }
                            let target = target_of(at)?;
                            if !(start..=end).contains(&target) {
                                return Err(format!("callee branch at {at} leaves its body"));
                            }
                            Flow::Branch { target: base + target - start, unconditional: false, invert: false }
                        }
                        EXIT => Flow::Exit,
                        opcode @ (CALL | RET | 0x949 | 0x942) => {
                            return Err(format!("control flow {opcode:#05x} inside a callee"))
                        }
                        _ => Flow::None,
                    };
                    items.push(Item { inst: body, flow });
                }
                continue;
            }
            BRX => {
                tables.push(jump_table(instructions, index, read_cbuf).ok_or_else(|| format!("jump table at {index}"))?);
                Flow::Indirect(tables.len() - 1)
            }
            opcode @ 0x942 => return Err(format!("control flow {opcode:#05x}")),
            _ => Flow::None,
        };
        items.push(Item { inst: *inst, flow });
    }
    for (item, target) in pending {
        if let Flow::Branch { unconditional, invert, .. } = items[item].flow {
            items[item].flow = Flow::Branch { target: main[target], unconditional, invert };
        }
    }
    for table in &mut tables {
        for case in &mut table.cases {
            case.1 = main[case.1];
        }
    }
    Ok(prune(items, tables))
}

fn prune(items: Vec<Item>, mut tables: Vec<Table>) -> (Vec<Item>, Vec<Table>) {
    let mut reachable = vec![false; items.len()];
    let mut stack = vec![0];
    while let Some(index) = stack.pop() {
        if index >= items.len() || std::mem::replace(&mut reachable[index], true) {
            continue;
        }
        let item = items[index];
        let falls = match item.flow {
            Flow::None => true,
            Flow::Branch { target, unconditional, .. } => {
                stack.push(target);
                !unconditional && branch_predicate(&item.inst).0.is_some()
            }
            Flow::Exit => item.inst.opcode() == EXIT && branch_predicate(&item.inst).0.is_some(),
            Flow::Indirect(table) => {
                stack.extend(tables[table].cases.iter().map(|&(_, target)| target));
                false
            }
        };
        if falls {
            stack.push(index + 1);
        }
    }
    let mut remap = vec![0; items.len()];
    let mut next = 0;
    for (index, &live) in reachable.iter().enumerate() {
        remap[index] = next;
        next += usize::from(live);
    }
    for table in &mut tables {
        for case in &mut table.cases {
            case.1 = remap[case.1];
        }
    }
    let items = items
        .into_iter()
        .zip(reachable)
        .filter(|(_, live)| *live)
        .map(|(mut item, _)| {
            if let Flow::Branch { target, unconditional, invert } = item.flow {
                item.flow = Flow::Branch { target: remap[target], unconditional, invert };
            }
            item
        })
        .collect();
    (items, tables)
}
pub fn probe(code: &[u8], stage: ShaderStage) -> Result<Cfg, String> {
    probe_with_cbuf(code, stage, &mut |_, _| None)
}

pub fn probe_with_cbuf(code: &[u8], stage: ShaderStage, read_cbuf: &mut dyn FnMut(u8, u32) -> Option<u32>) -> Result<Cfg, String> {
    let instructions = decode(code).ok_or("no end marker")?;
    let context = Sm86Context::new(&instructions).ok_or("context")?;
    let (items, tables) = expand(&instructions, read_cbuf)?;
    let count = items.len();
    let mut leaders = BTreeSet::from([0]);
    for (index, item) in items.iter().enumerate() {
        if let Flow::Branch { target, .. } = item.flow {
            leaders.insert(target);
        }
        if let Flow::Indirect(table) = item.flow {
            leaders.extend(tables[table].cases.iter().map(|&(_, target)| target));
        }
        if item.flow != Flow::None {
            leaders.insert(index + 1);
        }
    }
    leaders.retain(|&leader| leader < count);
    let leader_list: Vec<usize> = leaders.iter().copied().collect();
    let block_of: HashMap<usize, BlockId> = leader_list
        .iter()
        .enumerate()
        .map(|(block, &leader)| (leader, block as BlockId))
        .collect();
    let mut topology = Vec::with_capacity(leader_list.len());
    let mut predicated_exits = Vec::new();
    let mut combined = vec![false; leader_list.len()];
    for (block, &start) in leader_list.iter().enumerate() {
        let end = leader_list.get(block + 1).copied().unwrap_or(count);
        let last = end - 1;
        let item = items[last];
        let (branch, terminator) = match item.flow {
            Flow::Branch { target, unconditional, invert } => {
                let target = block_of[&target];
                let (predicate, combine) = if unconditional { (None, false) } else { branch_predicate(&item.inst) };
                let predicate = predicate.map(|pred| Predicate { negate: pred.negate != invert, ..pred });
                combined[block] = combine;
                match predicate {
                    None => (BranchKind::Unconditional { target }, Some(last)),
                    Some(pred) => (BranchKind::Conditional { target, pred }, Some(last)),
                }
            }
            Flow::Exit if item.inst.opcode() == EXIT => match branch_predicate(&item.inst) {
                (None, _) => (BranchKind::Exit, Some(last)),
                (Some(pred), combine) => {
                    combined[block] = combine;
                    predicated_exits.push(block);
                    (BranchKind::Conditional { target: 0, pred }, Some(last))
                }
            },
            Flow::Exit => (BranchKind::Exit, Some(last)),
            Flow::Indirect(table) => {
                let table = &tables[table];
                let mut targets = [IndirectBranchTarget::default(); MAX_INDIRECT_BRANCH_TARGETS];
                for (slot, &(selector, target)) in targets.iter_mut().zip(&table.cases) {
                    *slot = IndirectBranchTarget { selector, target: block_of[&target] };
                }
                let branch = BranchKind::Indirect {
                    register: table.register,
                    base: table.base,
                    cbuf_binding: table.binding,
                    cbuf_offset: table.offset,
                    table_entries: table.entries,
                    count: table.cases.len() as u8,
                    targets,
                };
                (branch, Some(last))
            }
            Flow::None => (BranchKind::FallThrough, None),
        };
        topology.push(BlockInfo {
            start: start * SIZE,
            end: end * SIZE,
            branch,
            terminator_offset: terminator.map(|index| index * SIZE),
            synthetic_exit: false,
        });
    }
    let synthetic_base = topology.len() as BlockId;
    for (index, source) in predicated_exits.into_iter().enumerate() {
        let BranchKind::Conditional { pred, .. } = topology[source].branch else {
            unreachable!()
        };
        topology[source].branch = BranchKind::Conditional { target: synthetic_base + index as BlockId, pred };
        let offset = topology[source].end;
        topology.push(BlockInfo {
            start: offset,
            end: offset,
            branch: BranchKind::Exit,
            terminator_offset: None,
            synthetic_exit: true,
        });
    }
    let preds = compute_predecessors(&topology);
    let failed = Cell::new(None);
    let translate = |forced: &LoopCarriedPhis| {
        build_cfg_blocks_with(stage, &topology, &preds, forced, |t, info, defs| {
            if info.synthetic_exit {
                return;
            }
            let block = leader_list.binary_search(&(info.start / SIZE)).ok();
            for index in info.start / SIZE..info.end / SIZE {
                let terminator = Some(index * SIZE) == info.terminator_offset;
                let item = items[index];
                if terminator && block.is_some_and(|block| combined[block]) {
                    t.sm86_branch_condition(item.inst);
                    continue;
                }
                if terminator {
                    match item.flow {
                        Flow::Branch { .. } | Flow::Indirect(_) => continue,
                        Flow::Exit if item.inst.opcode() != EXIT => {
                            t.sm86_emit_exit();
                            continue;
                        }
                        Flow::Exit if !matches!(info.branch, BranchKind::Exit) => continue,
                        _ => {}
                    }
                }
                if !t.translate_sm86(&context, item.inst, index * SIZE, defs) && failed.get().is_none() {
                    failed.set(Some(item.inst));
                }
            }
        })
    };
    let mut forced = LoopCarriedPhis::empty(topology.len());
    let mut cfg = translate(&forced);
    let gaps = loop_carried_phi_gaps(&cfg.blocks, &preds);
    if !gaps.is_empty() {
        forced = gaps;
        cfg = translate(&forced);
    }
    if let Some(inst) = failed.get() {
        return Err(format!("opcode {:#05x} ({:016x}_{:016x})", inst.opcode(), inst.hi, inst.lo));
    }
    if cfg.unimplemented != 0 {
        return Err(format!("{} unimplemented IR ops", cfg.unimplemented));
    }
    Ok(cfg)
}
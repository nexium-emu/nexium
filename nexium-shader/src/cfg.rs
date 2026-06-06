

use std::collections::{BTreeSet, HashMap};

use super::ir::{Inst, Op, Predicate, Program, Value, ValueId};
use super::operand::{decoded_pred, RZ};
use super::translate::Translator;
use super::decode::decode_one; use super::opcodes::Opcode;

pub type BlockId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchKind {

    FallThrough,

    Unconditional { target: BlockId },

    Conditional { target: BlockId, pred: Predicate },

    Exit,
}

#[derive(Debug)]
pub struct BasicBlock {
    pub id: BlockId,

    pub start_offset: usize,

    pub end_offset: usize,
    pub branch: BranchKind,
    pub program: Program,

    pub reg_exit: HashMap<u8, Value>,
}

pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub unimplemented: u32,
}

impl Cfg {
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id as usize]
    }

    pub fn successors(&self, id: BlockId) -> Vec<BlockId> {
        match self.blocks[id as usize].branch {
            BranchKind::Exit => vec![],
            BranchKind::Unconditional { target } => vec![target],
            BranchKind::Conditional { target, .. } => {
                let fall = id + 1;
                if (fall as usize) < self.blocks.len() {
                    vec![target, fall]
                } else {
                    vec![target]
                }
            }
            BranchKind::FallThrough => {
                let fall = id + 1;
                if (fall as usize) < self.blocks.len() {
                    vec![fall]
                } else {
                    vec![]
                }
            }
        }
    }

    pub fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for src in 0..self.blocks.len() as u32 {
            for dst in self.successors(src) {
                preds[dst as usize].push(src);
            }
        }
        preds
    }
}

fn is_schedule(offset: usize) -> bool {
    offset % 0x20 == 0
}

fn bra_target(pc: usize, raw: u64) -> usize {
    let raw_24 = ((raw >> 20) & 0x00FF_FFFF) as u32;
    let signed = if raw_24 & 0x0080_0000 != 0 {
        (raw_24 | 0xFF00_0000) as i32
    } else {
        raw_24 as i32
    };
    (pc as i64 + signed as i64 + 8) as usize
}

fn discover_leaders(bytes: &[u8]) -> BTreeSet<usize> {
    let mut leaders: BTreeSet<usize> = BTreeSet::new();
    let mut worklist: Vec<usize> = vec![0];
    leaders.insert(0);

    while let Some(start) = worklist.pop() {
        let mut offset = start;
        while offset + 8 <= bytes.len() {
            if is_schedule(offset) {
                offset += 8;
                continue;
            }
            let raw = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let next = offset + 8;
            let Some(d) = decode_one(raw) else {
                offset = next;
                continue;
            };
            let pred = decoded_pred(raw);
            match d.opcode {
                Opcode::EXIT if pred.is_none() => break,
                Opcode::BRA | Opcode::JMP => {
                    let target = bra_target(offset, raw);
                    if target < bytes.len() && leaders.insert(target) {
                        worklist.push(target);
                    }
                    if pred.is_some() {
                        if next < bytes.len() && leaders.insert(next) {
                            worklist.push(next);
                        }
                        offset = next;
                        continue;
                    }
                    break;
                }
                _ => {}
            }
            offset = next;
        }
    }
    leaders
}

fn make_offset_to_block(leaders: &BTreeSet<usize>) -> HashMap<usize, BlockId> {
    leaders
        .iter()
        .enumerate()
        .map(|(i, &off)| (off, i as BlockId))
        .collect()
}

pub fn build_cfg(bytes: &[u8]) -> Cfg {
    let leaders = discover_leaders(bytes);
    let offset_to_block = make_offset_to_block(&leaders);
    let leader_vec: Vec<usize> = leaders.iter().copied().collect();

    let topology = discover_topology(bytes, &leader_vec, &offset_to_block);
    let preds = compute_predecessors(&topology);

    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(topology.len());
    let mut total_unimpl: u32 = 0;
    let mut next_value: u32 = 0;

    for (bid, info) in topology.iter().enumerate() {
        let (initial_state, phis, after_phis) =
            compute_initial_reg_state(&blocks, &preds, bid as BlockId, next_value);
        next_value = after_phis;

        let mut t = Translator::with_initial(initial_state, next_value);
        for phi in phis {
            t.program.instructions.push(phi);
        }

        let mut offset = info.start;
        while offset + 8 <= info.end {
            if is_schedule(offset) {
                offset += 8;
                continue;
            }
            if Some(offset) == info.terminator_offset {
                break;
            }
            let raw = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            t.translate(raw);
            offset += 8;
        }

        if let Some(term_off) = info.terminator_offset {
            if term_off + 8 <= bytes.len() && matches!(info.branch, BranchKind::Exit) {
                let raw = u64::from_le_bytes(bytes[term_off..term_off + 8].try_into().unwrap());
                t.translate(raw);
            }
        }

        total_unimpl += t.unimplemented_count;
        next_value = t.program.next_value_id();
        let reg_exit = t.snapshot_reg_state();
        blocks.push(BasicBlock {
            id: bid as BlockId,
            start_offset: info.start,
            end_offset: info.terminator_offset.map(|o| o + 8).unwrap_or(info.end),
            branch: info.branch,
            program: std::mem::take(&mut t.program),
            reg_exit,
        });
    }

    patch_back_edge_phi_sources(&mut blocks);

    Cfg { blocks, unimplemented: total_unimpl }
}

struct BlockInfo {
    start: usize,
    end: usize,
    branch: BranchKind,

    terminator_offset: Option<usize>,
}

fn discover_topology(
    bytes: &[u8],
    leader_vec: &[usize],
    offset_to_block: &HashMap<usize, BlockId>,
) -> Vec<BlockInfo> {
    let mut out = Vec::with_capacity(leader_vec.len());
    for (i, &start) in leader_vec.iter().enumerate() {
        let end = leader_vec.get(i + 1).copied().unwrap_or(bytes.len());
        let mut branch = BranchKind::FallThrough;
        let mut terminator_offset: Option<usize> = None;
        let mut offset = start;
        while offset + 8 <= end {
            if is_schedule(offset) {
                offset += 8;
                continue;
            }
            let raw = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            if let Some(d) = decode_one(raw) {
                match d.opcode {
                    Opcode::EXIT if decoded_pred(raw).is_none() => {
                        branch = BranchKind::Exit;
                        terminator_offset = Some(offset);
                        break;
                    }
                    Opcode::BRA | Opcode::JMP => {
                        let target_off = bra_target(offset, raw);
                        let target = *offset_to_block.get(&target_off).unwrap_or(&(i as u32));
                        match decoded_pred(raw) {
                            None => branch = BranchKind::Unconditional { target },
                            Some(pred) => branch = BranchKind::Conditional { target, pred },
                        }
                        terminator_offset = Some(offset);
                        break;
                    }
                    _ => {}
                }
            }
            offset += 8;
        }
        out.push(BlockInfo { start, end, branch, terminator_offset });
    }
    out
}

fn compute_predecessors(topology: &[BlockInfo]) -> Vec<Vec<BlockId>> {
    let mut preds = vec![Vec::new(); topology.len()];
    for (src, info) in topology.iter().enumerate() {
        let src = src as BlockId;
        match info.branch {
            BranchKind::Exit => {}
            BranchKind::Unconditional { target } => {
                preds[target as usize].push(src);
            }
            BranchKind::Conditional { target, .. } => {
                preds[target as usize].push(src);
                let fall = src + 1;
                if (fall as usize) < topology.len() {
                    preds[fall as usize].push(src);
                }
            }
            BranchKind::FallThrough => {
                let fall = src + 1;
                if (fall as usize) < topology.len() {
                    preds[fall as usize].push(src);
                }
            }
        }
    }
    preds
}

fn compute_initial_reg_state(
    built_blocks: &[BasicBlock],
    preds: &[Vec<BlockId>],
    bid: BlockId,
    mut next_value: u32,
) -> (HashMap<u8, Value>, Vec<Inst>, u32) {
    let pred_ids = &preds[bid as usize];
    if pred_ids.is_empty() {
        return (HashMap::new(), Vec::new(), next_value);
    }

    let has_back_edge = pred_ids.iter().any(|&p| p >= bid);

    if pred_ids.len() == 1 && !has_back_edge {
        let p = pred_ids[0] as usize;
        if p < built_blocks.len() {
            return (built_blocks[p].reg_exit.clone(), Vec::new(), next_value);
        }
        return (HashMap::new(), Vec::new(), next_value);
    }

    let mut all_regs: BTreeSet<u8> = BTreeSet::new();
    for &p in pred_ids {
        if let Some(b) = built_blocks.get(p as usize) {
            for &r in b.reg_exit.keys() {
                all_regs.insert(r);
            }
        }
    }

    let mut initial = HashMap::new();
    let mut phis = Vec::new();
    for r in all_regs {
        if r == RZ {
            continue;
        }
        let sources: Vec<(BlockId, Value)> = pred_ids
            .iter()
            .map(|&p| {
                let v = built_blocks
                    .get(p as usize)
                    .and_then(|b| b.reg_exit.get(&r).copied())
                    .unwrap_or(Value::GprIn(r));
                (p, v)
            })
            .collect();

        if !has_back_edge {
            let all_same = sources.windows(2).all(|w| values_equal(&w[0].1, &w[1].1));
            if all_same {
                initial.insert(r, sources[0].1);
                continue;
            }
        }
        let id = ValueId(next_value);
        next_value = next_value.wrapping_add(1);
        phis.push(Inst {
            op: Op::Phi { sources },
            result: Some(id),
            dest_reg: Some(r),
            pred: None,
        });
        initial.insert(r, Value::Inst(id));
    }
    (initial, phis, next_value)
}

fn patch_back_edge_phi_sources(blocks: &mut [BasicBlock]) {
    let reg_exits: Vec<HashMap<u8, Value>> =
        blocks.iter().map(|b| b.reg_exit.clone()).collect();
    for block in blocks.iter_mut() {
        let bid = block.id;
        for inst in block.program.instructions.iter_mut() {
            let Op::Phi { sources } = &mut inst.op else { continue };
            let Some(reg) = inst.dest_reg else { continue };
            if reg == RZ {
                continue;
            }
            for (pred_id, value) in sources.iter_mut() {
                if *pred_id >= bid {
                    if let Some(exit) = reg_exits.get(*pred_id as usize) {
                        *value = exit.get(&reg).copied().unwrap_or(Value::GprIn(reg));
                    }
                }
            }
        }
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    matches!(
        (a, b),
        (Value::Zero, Value::Zero)
    ) || match (a, b) {
        (Value::Inst(a), Value::Inst(b)) => a.0 == b.0,
        (Value::GprIn(a), Value::GprIn(b)) => a == b,
        (Value::ImmU32(a), Value::ImmU32(b)) => a == b,
        (Value::ImmF32(a), Value::ImmF32(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_exit() -> u64 {

        0xE300_0000_0007_000Fu64
    }

    fn enc_fmul_reg(rd: u8, ra: u8, rb: u8) -> u64 {

        0x5C68_1000_0000_0000u64
            | ((rb as u64) << 20)
            | ((ra as u64) << 8)
            | (rd as u64)
            | 0x0007_0000
    }

    fn build_program(words: &[u64]) -> Vec<u8> {

        let mut bytes = Vec::new();
        let mut idx = 0;
        for chunk in words.chunks(3) {
            bytes.extend_from_slice(&[0u8; 8]);
            for w in chunk {
                bytes.extend_from_slice(&w.to_le_bytes());
                idx += 1;
            }

            let _ = idx;
        }
        bytes
    }

    fn enc_fadd_reg(rd: u8, ra: u8, rb: u8) -> u64 {

        0x5C58_0000_0000_0000u64
            | ((rb as u64) << 20)
            | ((ra as u64) << 8)
            | (rd as u64)
            | 0x0007_0000
    }

    fn enc_bra_p0(ofs: i32) -> u64 {
        let raw_24 = (ofs as u32) & 0x00FF_FFFF;

        0xE240_0000_0000_0000u64 | ((raw_24 as u64) << 20)
    }

    #[test]
    fn conditional_branch_creates_three_blocks_with_phi() {

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_fmul_reg(2, 0, 1).to_le_bytes());
        bytes.extend_from_slice(&enc_bra_p0(0x10).to_le_bytes());
        bytes.extend_from_slice(&enc_fadd_reg(2, 0, 1).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_fmul_reg(3, 2, 2).to_le_bytes());
        bytes.extend_from_slice(&enc_exit().to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);

        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 3, "expected 3 blocks");
        assert!(matches!(cfg.blocks[0].branch, BranchKind::Conditional { .. }));
        assert!(matches!(cfg.blocks[1].branch, BranchKind::FallThrough));
        assert!(matches!(cfg.blocks[2].branch, BranchKind::Exit));

        let preds = cfg.predecessors();
        assert_eq!(preds[2].len(), 2, "block 2 should be a join");

        let phi_inst = cfg.blocks[2]
            .program
            .instructions
            .iter()
            .find(|i| matches!(i.op, Op::Phi { .. }))
            .expect("expected a phi at the join");
        let phi_id = phi_inst.result.expect("phi has a result");
        if let Op::Phi { sources } = &phi_inst.op {
            assert_eq!(sources.len(), 2);
        }

        let fmul = cfg.blocks[2]
            .program
            .instructions
            .iter()
            .find(|i| matches!(i.op, Op::FMul { .. }))
            .expect("expected the FMul in the merge block");
        if let Op::FMul { a, b, .. } = &fmul.op {
            assert!(matches!(a, Value::Inst(id) if *id == phi_id),
                "FMul.a should reference the phi result, got {a:?}");
            assert!(matches!(b, Value::Inst(id) if *id == phi_id),
                "FMul.b should reference the phi result, got {b:?}");
        }
    }

    fn enc_bra_pt(ofs: i32) -> u64 {
        let raw_24 = (ofs as u32) & 0x00FF_FFFF;
        0xE240_0000_0000_0000u64 | ((raw_24 as u64) << 20) | 0x0007_0000
    }

    #[test]
    fn back_edge_creates_phi_with_patched_source() {

        let mut bytes = Vec::new();

        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_fmul_reg(2, 0, 1).to_le_bytes());
        bytes.extend_from_slice(&enc_bra_pt(0x8).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);

        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_fadd_reg(2, 2, 0).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_bra_p0(-32).to_le_bytes());

        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_fmul_reg(3, 2, 2).to_le_bytes());
        bytes.extend_from_slice(&enc_exit().to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);

        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 3);
        assert!(matches!(cfg.blocks[0].branch, BranchKind::Unconditional { target: 1 }));
        assert!(matches!(cfg.blocks[1].branch, BranchKind::Conditional { target: 1, .. }));
        assert!(matches!(cfg.blocks[2].branch, BranchKind::Exit));

        let preds = cfg.predecessors();
        assert_eq!(preds[1], vec![0, 1], "B1 should have B0 + back-edge B1");
        assert_eq!(preds[2], vec![1]);

        let phi = cfg.blocks[1]
            .program
            .instructions
            .iter()
            .find(|i| matches!(i.op, Op::Phi { .. }) && i.dest_reg == Some(2))
            .expect("expected phi for R2 in B1");
        let phi_id = phi.result.unwrap();
        let Op::Phi { sources } = &phi.op else { unreachable!() };
        assert_eq!(sources.len(), 2);

        for (pred_id, val) in sources {
            assert!(
                matches!(val, Value::Inst(_)),
                "phi source from B{pred_id} should be Inst, got {val:?}"
            );
        }

        let fadd = cfg.blocks[1]
            .program
            .instructions
            .iter()
            .find(|i| matches!(i.op, Op::FAdd { .. }))
            .expect("expected FAdd in B1");
        if let Op::FAdd { a, b: _, .. } = &fadd.op {
            assert!(
                matches!(a, Value::Inst(id) if *id == phi_id),
                "FAdd.a should be phi result, got {a:?}"
            );
        }

        let fadd_id = fadd.result.unwrap();
        let back_source = sources.iter().find(|(p, _)| *p == 1).unwrap();
        assert!(
            matches!(back_source.1, Value::Inst(id) if id == fadd_id),
            "back-edge phi source should be FAdd result, got {:?}",
            back_source.1
        );
    }

    #[test]
    fn straight_line_one_block() {
        let bytes = build_program(&[
            enc_fmul_reg(2, 0, 1),
            enc_fmul_reg(3, 0, 1),
            enc_exit(),
        ]);
        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 1);
        assert!(matches!(cfg.blocks[0].branch, BranchKind::Exit));
    }
}

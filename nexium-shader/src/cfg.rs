use std::collections::{BTreeSet, HashMap, HashSet};

use super::decode::decode_one;
use super::ir::{Inst, Op, Predicate, Program, ShaderStage, Value, ValueId};
use super::opcodes::Opcode;
use super::operand::{
    decoded_pred, exit_never_taken, imm20, ldc_mode, ldc_ref, ldc_size, ldc_src_reg, pred_negate,
    reg_a, reg_dest, LdcMode, RZ,
};
use super::translate::{
    resolve_cbuf_handle_origin, PendingBindlessOriginCheck, Translator, ValueDefs,
};

pub type BlockId = u32;
pub const MAX_INDIRECT_BRANCH_TARGETS: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndirectBranchTarget {
    pub selector: u32,
    pub target: BlockId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchKind {
    FallThrough,

    Unconditional {
        target: BlockId,
    },

    Conditional {
        target: BlockId,
        pred: Predicate,
    },

    Indirect {
        register: u8,
        base: u32,
        cbuf_binding: u8,
        cbuf_offset: u32,
        table_entries: u8,
        count: u8,
        targets: [IndirectBranchTarget; MAX_INDIRECT_BRANCH_TARGETS],
    },

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
    pub pred_phis: Vec<PredPhi>,
    pub pred_exit: HashMap<u8, ValueId>,
}

#[derive(Debug, Clone)]
pub struct PredPhi {
    pub pred: u8,
    pub result: ValueId,
    pub sources: Vec<(BlockId, Option<ValueId>)>,
}

pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub unimplemented: u32,
    pub bindless_or_partners: std::collections::HashMap<u32, u32>,
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
            BranchKind::Indirect { count, targets, .. } => {
                let mut successors = Vec::with_capacity(count as usize);
                for entry in targets.iter().take(count as usize) {
                    if !successors.contains(&entry.target) {
                        successors.push(entry.target);
                    }
                }
                successors
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageBufferAddr {
    pub cbuf_binding: u8,
    pub cbuf_offset: u32,
    pub align: u32,
}

fn track_cbuf_base(start: Value, defs: &HashMap<u32, Op>) -> Option<(u8, u32, u32)> {
    if let Some((b, o)) = track_dfs(start, defs, true, 0) {
        return Some((b, o, 16));
    }
    track_dfs(start, defs, false, 0).map(|(b, o)| (b, o, 8))
}

fn track_dfs(v: Value, defs: &HashMap<u32, Op>, biased: bool, depth: u32) -> Option<(u8, u32)> {
    if depth > 24 {
        return None;
    }
    let Value::Inst(id) = v else {
        return None;
    };
    match defs.get(&id.0)? {
        Op::LoadCbuf {
            binding,
            byte_offset,
        } => {
            let align = if biased { 16 } else { 8 };
            if *byte_offset % align != 0 {
                return None;
            }
            if biased && !(*binding == 0 && *byte_offset >= 0x110 && *byte_offset < 0x610) {
                return None;
            }
            Some((*binding, *byte_offset))
        }
        Op::Mov(s) => track_dfs(*s, defs, biased, depth + 1),
        Op::IAdd { a, b, .. } => track_dfs(*a, defs, biased, depth + 1)
            .or_else(|| track_dfs(*b, defs, biased, depth + 1)),
        Op::IScAdd { a, b, .. } => track_dfs(*a, defs, biased, depth + 1)
            .or_else(|| track_dfs(*b, defs, biased, depth + 1)),
        _ => None,
    }
}

pub fn collect_storage_buffers(cfg: &mut Cfg) -> Vec<StorageBufferAddr> {
    let mut defs: HashMap<u32, Op> = HashMap::new();
    for b in &cfg.blocks {
        for inst in &b.program.instructions {
            if let Some(r) = inst.result {
                defs.insert(r.0, inst.op.clone());
            }
        }
    }

    let mut buffers: Vec<StorageBufferAddr> = Vec::new();
    let mut rewrites: Vec<(usize, usize, Op)> = Vec::new();
    for (bi, b) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in b.program.instructions.iter().enumerate() {
            if let Op::LoadGlobal { addr_lo, offset } = inst.op {
                if let Some((binding, coff, align)) = track_cbuf_base(addr_lo, &defs) {
                    let sba = StorageBufferAddr {
                        cbuf_binding: binding,
                        cbuf_offset: coff,
                        align,
                    };
                    let buffer_index = match buffers.iter().position(|x| *x == sba) {
                        Some(p) => p as u32,
                        None => {
                            buffers.push(sba);
                            (buffers.len() - 1) as u32
                        }
                    };
                    rewrites.push((
                        bi,
                        ii,
                        Op::LoadStorage {
                            buffer_index,
                            addr_lo,
                            imm: offset,
                            cbuf_binding: binding,
                            cbuf_offset: coff,
                            align,
                        },
                    ));
                }
            }
        }
    }
    for (bi, ii, op) in rewrites {
        cfg.blocks[bi].program.instructions[ii].op = op;
    }
    buffers
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

fn signed_24(raw: u64) -> i32 {
    let value = ((raw >> 20) & 0x00FF_FFFF) as u32;
    if value & 0x0080_0000 != 0 {
        (value | 0xFF00_0000) as i32
    } else {
        value as i32
    }
}

fn previous_instruction(mut offset: usize) -> Option<usize> {
    loop {
        offset = offset.checked_sub(8)?;
        if !is_schedule(offset) {
            return Some(offset);
        }
    }
}

fn find_previous<F>(bytes: &[u8], offset: &mut usize, mut matches: F) -> Option<u64>
where
    F: FnMut(u64, Opcode) -> bool,
{
    while let Some(candidate) = previous_instruction(*offset) {
        *offset = candidate;
        let raw = u64::from_le_bytes(bytes[candidate..candidate + 8].try_into().ok()?);
        if let Some(decoded) = decode_one(raw) {
            if matches(raw, decoded.opcode) {
                return Some(raw);
            }
        }
    }
    None
}

fn track_indirect_branch<F>(
    bytes: &[u8],
    pc: usize,
    raw: u64,
    read_cbuf: &mut F,
) -> Option<BranchKind>
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    if raw & 0x1F != 0x0F || decoded_pred(raw).is_some() || pred_negate(raw) || raw & 0x60 != 0 {
        return None;
    }

    let branch_register = reg_a(raw);
    let mut scan = pc;
    let ldc = find_previous(bytes, &mut scan, |candidate, opcode| {
        opcode == Opcode::LDC
            && reg_dest(candidate) == branch_register
            && ldc_size(candidate) == 4
            && ldc_mode(candidate) == LdcMode::Default
            && decoded_pred(candidate).is_none()
            && !pred_negate(candidate)
    })?;
    let reference = ldc_ref(ldc);
    let table_offset = u32::try_from(reference.byte_offset).ok()?;
    let ldc_register = ldc_src_reg(ldc);

    let shl = find_previous(bytes, &mut scan, |candidate, opcode| {
        opcode == Opcode::SHL_imm
            && reg_dest(candidate) == ldc_register
            && imm20(candidate) == 2
            && decoded_pred(candidate).is_none()
            && !pred_negate(candidate)
    })?;
    let index_register = reg_a(shl);

    let imnmx = find_previous(bytes, &mut scan, |candidate, opcode| {
        opcode == Opcode::IMNMX_imm
            && reg_dest(candidate) == index_register
            && decoded_pred(candidate).is_none()
            && !pred_negate(candidate)
    })?;
    if (imnmx >> 56) & 1 != 0 {
        return None;
    }
    let entry_count = ((imnmx >> 20) & 0x7_FFFF) as usize + 1;
    if entry_count == 0 || entry_count > MAX_INDIRECT_BRANCH_TARGETS {
        return None;
    }

    let base = (pc as u32)
        .wrapping_add(8)
        .wrapping_add(signed_24(raw) as u32);
    let mut targets = [IndirectBranchTarget::default(); MAX_INDIRECT_BRANCH_TARGETS];
    let mut count = 0usize;
    for index in 0..entry_count {
        let byte_offset = table_offset.checked_add((index as u32).checked_mul(4)?)?;
        let Some(table_value) = read_cbuf(reference.binding, byte_offset) else {
            if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
                log::warn!(
                    "[brx-track] pc={:#x} c[{}]+{:#x} unavailable",
                    pc,
                    reference.binding,
                    byte_offset
                );
            }
            return None;
        };
        let selector = base.wrapping_add(table_value);
        let raw_target = selector as usize;
        let target = if is_schedule(raw_target) {
            raw_target.saturating_add(8)
        } else {
            raw_target
        };
        if target <= pc || target + 8 > bytes.len() || target % 8 != 0 {
            if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
                log::warn!(
                    "[brx-track] pc={:#x} index={} table={:#010x} base={:#010x} target={:#x}->{:#x} len={:#x} rejected",
                    pc,
                    index,
                    table_value,
                    base,
                    raw_target,
                    target,
                    bytes.len()
                );
            }
            return None;
        }
        if targets[..count]
            .iter()
            .any(|entry| entry.selector == selector)
        {
            continue;
        }
        targets[count] = IndirectBranchTarget {
            selector,
            target: target as BlockId,
        };
        count += 1;
    }
    if count == 0 {
        return None;
    }

    if std::env::var_os("NEXIUM_BRX_TRACE").is_some() {
        log::warn!(
            "[brx-track] pc={:#x} register=R{} base={:#010x} c[{}]+{:#x} entries={} targets={}",
            pc,
            branch_register,
            base,
            reference.binding,
            table_offset,
            entry_count,
            count
        );
    }

    Some(BranchKind::Indirect {
        register: branch_register,
        base,
        cbuf_binding: reference.binding,
        cbuf_offset: table_offset,
        table_entries: entry_count as u8,
        count: count as u8,
        targets,
    })
}

fn discover_indirect_branches<F>(bytes: &[u8], read_cbuf: &mut F) -> HashMap<usize, BranchKind>
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    let mut branches = HashMap::new();
    for offset in (0..bytes.len().saturating_sub(7)).step_by(8) {
        if is_schedule(offset) {
            continue;
        }
        let raw = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        if decode_one(raw).is_some_and(|decoded| decoded.opcode == Opcode::BRX) {
            if let Some(branch) = track_indirect_branch(bytes, offset, raw, read_cbuf) {
                branches.insert(offset, branch);
            }
        }
    }
    branches
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum FlowToken {
    Ssy,
    Pbk,
}

type FlowStack = Vec<(FlowToken, usize)>;

fn pop_flow_token(stack: &FlowStack, token: FlowToken) -> Option<(usize, FlowStack)> {
    let idx = stack.iter().rposition(|(t, _)| *t == token)?;
    let target = stack[idx].1;
    let mut next = stack.clone();
    next.truncate(idx);
    Some((target, next))
}

fn push_flow_state(
    worklist: &mut Vec<(usize, FlowStack)>,
    offset: usize,
    stack: &FlowStack,
    len: usize,
) {
    if offset < len {
        worklist.push((offset, stack.clone()));
    }
}

fn discover_sync_targets(
    bytes: &[u8],
    indirect_branches: &HashMap<usize, BranchKind>,
) -> HashMap<usize, usize> {
    let mut targets = HashMap::new();
    let mut worklist = vec![(0usize, FlowStack::new())];
    let mut visited: HashSet<(usize, FlowStack)> = HashSet::new();
    while let Some((start, mut stack)) = worklist.pop() {
        if start >= bytes.len() || !visited.insert((start, stack.clone())) {
            continue;
        }
        let mut offset = start;
        while offset + 8 <= bytes.len() {
            if is_schedule(offset) {
                offset += 8;
                continue;
            }
            let raw = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let next = offset + 8;
            if let Some(d) = decode_one(raw) {
                match d.opcode {
                    Opcode::SSY => stack.push((FlowToken::Ssy, bra_target(offset, raw))),
                    Opcode::PBK => stack.push((FlowToken::Pbk, bra_target(offset, raw))),
                    Opcode::SYNC | Opcode::BRK => {
                        let token = if matches!(d.opcode, Opcode::SYNC) {
                            FlowToken::Ssy
                        } else {
                            FlowToken::Pbk
                        };
                        if let Some((target, popped)) = pop_flow_token(&stack, token) {
                            targets.entry(offset).or_insert(target);
                            push_flow_state(&mut worklist, target, &popped, bytes.len());
                            if decoded_pred(raw).is_some() {
                                push_flow_state(&mut worklist, next, &stack, bytes.len());
                            }
                            break;
                        }
                    }
                    Opcode::BRA | Opcode::JMP => {
                        let target = bra_target(offset, raw);
                        push_flow_state(&mut worklist, target, &stack, bytes.len());
                        if decoded_pred(raw).is_some() {
                            push_flow_state(&mut worklist, next, &stack, bytes.len());
                        }
                        break;
                    }
                    Opcode::BRX => {
                        if let Some(BranchKind::Indirect { count, targets, .. }) =
                            indirect_branches.get(&offset)
                        {
                            for entry in targets.iter().take(*count as usize) {
                                push_flow_state(
                                    &mut worklist,
                                    entry.target as usize,
                                    &stack,
                                    bytes.len(),
                                );
                            }
                            break;
                        }
                    }
                    Opcode::EXIT if decoded_pred(raw).is_none() && !exit_never_taken(raw) => break,
                    _ => {}
                }
            }
            offset = next;
        }
    }
    targets
}

fn discover_leaders(
    bytes: &[u8],
    sync_targets: &HashMap<usize, usize>,
    indirect_branches: &HashMap<usize, BranchKind>,
) -> BTreeSet<usize> {
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
                Opcode::EXIT if pred.is_none() && !exit_never_taken(raw) => break,
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
                Opcode::SYNC | Opcode::BRK => {
                    if let Some(&target) = sync_targets.get(&offset) {
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
                }
                Opcode::BRX => {
                    if let Some(BranchKind::Indirect { count, targets, .. }) =
                        indirect_branches.get(&offset)
                    {
                        for entry in targets.iter().take(*count as usize) {
                            let target = entry.target as usize;
                            if leaders.insert(target) {
                                worklist.push(target);
                            }
                        }
                        break;
                    }
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
    build_cfg_with_cbuf_stage(bytes, |_, _| None, ShaderStage::Vertex)
}

pub fn build_fragment_cfg(bytes: &[u8]) -> Cfg {
    build_cfg_with_cbuf_stage(bytes, |_, _| None, ShaderStage::Fragment)
}

pub fn build_compute_cfg(bytes: &[u8]) -> Cfg {
    build_cfg_with_cbuf_stage(bytes, |_, _| None, ShaderStage::Compute)
}

pub fn build_cfg_with_cbuf<F>(bytes: &[u8], mut read_cbuf: F) -> Cfg
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    build_cfg_with_cbuf_stage(bytes, &mut read_cbuf, ShaderStage::Vertex)
}

pub fn build_fragment_cfg_with_cbuf<F>(bytes: &[u8], mut read_cbuf: F) -> Cfg
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    build_cfg_with_cbuf_stage(bytes, &mut read_cbuf, ShaderStage::Fragment)
}

pub fn build_compute_cfg_with_cbuf<F>(bytes: &[u8], mut read_cbuf: F) -> Cfg
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    build_cfg_with_cbuf_stage(bytes, &mut read_cbuf, ShaderStage::Compute)
}

fn build_cfg_with_cbuf_stage<F>(bytes: &[u8], mut read_cbuf: F, stage: ShaderStage) -> Cfg
where
    F: FnMut(u8, u32) -> Option<u32>,
{
    let indirect_branches = discover_indirect_branches(bytes, &mut read_cbuf);
    let sync_targets = discover_sync_targets(bytes, &indirect_branches);
    let leaders = discover_leaders(bytes, &sync_targets, &indirect_branches);
    let offset_to_block = make_offset_to_block(&leaders);
    let leader_vec: Vec<usize> = leaders.iter().copied().collect();

    let topology = discover_topology(
        bytes,
        &leader_vec,
        &offset_to_block,
        &sync_targets,
        &indirect_branches,
    );
    let preds = compute_predecessors(&topology);

    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(topology.len());
    let mut total_unimpl: u32 = 0;
    let mut next_value: u32 = 0;
    let mut bindless_or_partners: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut value_defs = ValueDefs::new();
    let mut pending_bindless_checks: Vec<(BlockId, PendingBindlessOriginCheck)> = Vec::new();

    for (bid, info) in topology.iter().enumerate() {
        let (initial_state, phis, after_phis) =
            compute_initial_reg_state(&blocks, &preds, bid as BlockId, next_value);
        next_value = after_phis;

        let (initial_pred_state, pred_phis, after_pred_phis) =
            compute_initial_pred_state(&blocks, &preds, bid as BlockId, next_value);
        next_value = after_pred_phis;

        let mut t =
            Translator::with_initial_stage(initial_state, initial_pred_state, next_value, stage);
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
            t.translate_with_defs(raw, &value_defs);
            offset += 8;
        }

        if let Some(term_off) = info.terminator_offset {
            if term_off + 8 <= bytes.len() && matches!(info.branch, BranchKind::Exit) {
                let raw = u64::from_le_bytes(bytes[term_off..term_off + 8].try_into().unwrap());
                t.translate_with_defs(raw, &value_defs);
            }
        }

        for inst in &t.program.instructions {
            value_defs.insert_inst(inst);
        }
        pending_bindless_checks.extend(
            t.take_pending_bindless_origin_checks()
                .into_iter()
                .map(|check| (bid as BlockId, check)),
        );

        total_unimpl += t.unimplemented_count;
        for (k, v) in t.bindless_or_partners.drain() {
            bindless_or_partners.insert(k, v);
        }
        next_value = t.program.next_value_id();
        let reg_exit = t.snapshot_reg_state();
        let pred_exit = t.snapshot_pred_state();
        blocks.push(BasicBlock {
            id: bid as BlockId,
            start_offset: info.start,
            end_offset: info.terminator_offset.map(|o| o + 8).unwrap_or(info.end),
            branch: info.branch,
            program: std::mem::take(&mut t.program),
            reg_exit,
            pred_phis,
            pred_exit,
        });
    }

    patch_back_edge_phi_sources(&mut blocks);
    patch_back_edge_pred_phi_sources(&mut blocks);
    total_unimpl += finalize_bindless_origin_checks(&mut blocks, pending_bindless_checks);

    Cfg {
        blocks,
        unimplemented: total_unimpl,
        bindless_or_partners,
    }
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
    sync_targets: &HashMap<usize, usize>,
    indirect_branches: &HashMap<usize, BranchKind>,
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
                    Opcode::EXIT if decoded_pred(raw).is_none() && !exit_never_taken(raw) => {
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
                    Opcode::SYNC | Opcode::BRK => {
                        if let Some(&target_off) = sync_targets.get(&offset) {
                            let target = *offset_to_block.get(&target_off).unwrap_or(&(i as u32));
                            match decoded_pred(raw) {
                                None => branch = BranchKind::Unconditional { target },
                                Some(pred) => branch = BranchKind::Conditional { target, pred },
                            }
                            terminator_offset = Some(offset);
                            break;
                        }
                    }
                    Opcode::BRX => {
                        if let Some(BranchKind::Indirect {
                            register,
                            base,
                            cbuf_binding,
                            cbuf_offset,
                            table_entries,
                            count,
                            targets,
                        }) = indirect_branches.get(&offset)
                        {
                            let mut resolved = *targets;
                            let mut all_resolved = true;
                            for entry in resolved.iter_mut().take(*count as usize) {
                                let target_offset = entry.target as usize;
                                match offset_to_block.get(&target_offset).copied() {
                                    Some(target) => entry.target = target,
                                    None => all_resolved = false,
                                }
                            }
                            if !all_resolved {
                                offset += 8;
                                continue;
                            }
                            branch = BranchKind::Indirect {
                                register: *register,
                                base: *base,
                                cbuf_binding: *cbuf_binding,
                                cbuf_offset: *cbuf_offset,
                                table_entries: *table_entries,
                                count: *count,
                                targets: resolved,
                            };
                            terminator_offset = Some(offset);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            offset += 8;
        }
        out.push(BlockInfo {
            start,
            end,
            branch,
            terminator_offset,
        });
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
            BranchKind::Indirect { count, targets, .. } => {
                for entry in targets.iter().take(count as usize) {
                    let incoming = &mut preds[entry.target as usize];
                    if !incoming.contains(&src) {
                        incoming.push(src);
                    }
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

fn compute_initial_pred_state(
    built_blocks: &[BasicBlock],
    preds: &[Vec<BlockId>],
    bid: BlockId,
    mut next_value: u32,
) -> (HashMap<u8, ValueId>, Vec<PredPhi>, u32) {
    let pred_ids = &preds[bid as usize];
    if pred_ids.is_empty() {
        return (HashMap::new(), Vec::new(), next_value);
    }

    let has_back_edge = pred_ids.iter().any(|&p| p >= bid);

    if pred_ids.len() == 1 && !has_back_edge {
        let p = pred_ids[0] as usize;
        if p < built_blocks.len() {
            return (built_blocks[p].pred_exit.clone(), Vec::new(), next_value);
        }
        return (HashMap::new(), Vec::new(), next_value);
    }

    let mut all_preds: BTreeSet<u8> = BTreeSet::new();
    for &p in pred_ids {
        if let Some(b) = built_blocks.get(p as usize) {
            for &r in b.pred_exit.keys() {
                all_preds.insert(r);
            }
        }
    }

    let mut initial = HashMap::new();
    let mut phis = Vec::new();
    for pidx in all_preds {
        if pidx >= 7 {
            continue;
        }
        let sources: Vec<(BlockId, Option<ValueId>)> = pred_ids
            .iter()
            .map(|&p| {
                let v = built_blocks
                    .get(p as usize)
                    .and_then(|b| b.pred_exit.get(&pidx).copied());
                (p, v)
            })
            .collect();

        if !has_back_edge {
            let all_same = sources.windows(2).all(|w| w[0].1 == w[1].1);
            if all_same {
                if let Some(id) = sources[0].1 {
                    initial.insert(pidx, id);
                }
                continue;
            }
        }

        let id = ValueId(next_value);
        next_value = next_value.wrapping_add(1);
        phis.push(PredPhi {
            pred: pidx,
            result: id,
            sources,
        });
        initial.insert(pidx, id);
    }

    (initial, phis, next_value)
}

fn patch_back_edge_phi_sources(blocks: &mut [BasicBlock]) {
    let reg_exits: Vec<HashMap<u8, Value>> = blocks.iter().map(|b| b.reg_exit.clone()).collect();
    for block in blocks.iter_mut() {
        let bid = block.id;
        for inst in block.program.instructions.iter_mut() {
            let Op::Phi { sources } = &mut inst.op else {
                continue;
            };
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

fn patch_back_edge_pred_phi_sources(blocks: &mut [BasicBlock]) {
    let pred_exits: Vec<HashMap<u8, ValueId>> =
        blocks.iter().map(|b| b.pred_exit.clone()).collect();
    for block in blocks.iter_mut() {
        let bid = block.id;
        for phi in block.pred_phis.iter_mut() {
            for (pred_id, value) in phi.sources.iter_mut() {
                if *pred_id >= bid {
                    if let Some(exit) = pred_exits.get(*pred_id as usize) {
                        *value = exit.get(&phi.pred).copied();
                    }
                }
            }
        }
    }
}

fn finalize_bindless_origin_checks(
    blocks: &mut [BasicBlock],
    checks: Vec<(BlockId, PendingBindlessOriginCheck)>,
) -> u32 {
    if checks.is_empty() {
        return 0;
    }

    let mut defs = ValueDefs::new();
    for block in blocks.iter() {
        for inst in &block.program.instructions {
            defs.insert_inst(inst);
        }
    }

    let mut rejected = 0;
    for (block_id, check) in checks {
        let resolved = resolve_cbuf_handle_origin(&check.handle, check.consumer_pred, &defs);
        let Some(block) = blocks.get_mut(block_id as usize) else {
            rejected += 1;
            continue;
        };
        let sample_indices_valid = check.samples.iter().all(|(index, _)| {
            block
                .program
                .instructions
                .get(*index)
                .is_some_and(|inst| match check.opcode {
                    Opcode::TEX_b => {
                        matches!(inst.op, Op::SampleTex { .. } | Op::SampleTexHandle { .. })
                    }
                    Opcode::TLD_b => matches!(inst.op, Op::TexelFetchHandle { .. }),
                    Opcode::SUATOM => matches!(inst.op, Op::ImageAtomic { .. }),
                    _ => false,
                })
        });

        if let Some(origin) = resolved.filter(|_| sample_indices_valid) {
            for (index, _) in &check.samples {
                match &mut block.program.instructions[*index].op {
                    Op::SampleTex { tex_id, .. } if check.opcode == Opcode::TEX_b => {
                        *tex_id = origin.texture_id();
                    }
                    Op::SampleTexHandle { handle, .. } if check.opcode == Opcode::TEX_b => {
                        *handle = origin.as_texture_handle();
                    }
                    Op::TexelFetchHandle { handle, .. } if check.opcode == Opcode::TLD_b => {
                        *handle = origin.as_texture_handle();
                    }
                    Op::ImageAtomic { handle, .. } if check.opcode == Opcode::SUATOM => {
                        *handle = origin.as_texture_handle();
                    }
                    _ => unreachable!(),
                }
            }
            continue;
        }

        for (index, old_value) in check.samples {
            if let Some(inst) = block.program.instructions.get_mut(index) {
                inst.op = Op::Mov(old_value);
            }
        }
        block.program.emit_void(Op::Unimplemented {
            opcode: check.opcode,
            raw: check.raw,
        });
        log::debug!(
            "{:?} finalized handle origin rejected raw={:#018x}",
            check.opcode,
            check.raw,
        );
        rejected += 1;
    }
    rejected
}

fn values_equal(a: &Value, b: &Value) -> bool {
    matches!((a, b), (Value::Zero, Value::Zero))
        || match (a, b) {
            (Value::Inst(a), Value::Inst(b)) => a.0 == b.0,
            (Value::GprIn(a), Value::GprIn(b)) => a == b,
            (Value::ImmU32(a), Value::ImmU32(b)) => a == b,
            (Value::ImmF32(a), Value::ImmF32(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
}

pub fn merge_dual_vertex_sass(vertex_a: &[u8], vertex_b: &[u8]) -> Option<Vec<u8>> {
    const NOP: u64 = 0x50B0_0000_0007_0F00;
    let mut exit_offset = None;
    let mut offset = 0usize;
    while offset + 8 <= vertex_a.len() {
        if !is_schedule(offset) {
            let raw = u64::from_le_bytes(vertex_a[offset..offset + 8].try_into().ok()?);
            if let Some(decoded) = decode_one(raw) {
                match decoded.opcode {
                    Opcode::EXIT if decoded_pred(raw).is_none() && !exit_never_taken(raw) => {
                        exit_offset = Some(offset);
                        break;
                    }
                    Opcode::BRA
                    | Opcode::JMP
                    | Opcode::BRX
                    | Opcode::SSY
                    | Opcode::PBK
                    | Opcode::SYNC
                    | Opcode::BRK => {
                        return None;
                    }
                    _ => {}
                }
            }
        }
        offset += 8;
    }
    let exit_offset = exit_offset?;
    let bundle_end = (exit_offset & !0x1F) + 0x20;
    if bundle_end > vertex_a.len() {
        return None;
    }
    let mut merged = vertex_a[..bundle_end].to_vec();
    let mut slot = exit_offset;
    while slot < bundle_end {
        if !is_schedule(slot) {
            merged[slot..slot + 8].copy_from_slice(&NOP.to_le_bytes());
        }
        slot += 8;
    }
    merged.extend_from_slice(vertex_b);
    Some(merged)
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
        assert!(matches!(
            cfg.blocks[0].branch,
            BranchKind::Conditional { .. }
        ));
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
            assert!(
                matches!(a, Value::Inst(id) if *id == phi_id),
                "FMul.a should reference the phi result, got {a:?}"
            );
            assert!(
                matches!(b, Value::Inst(id) if *id == phi_id),
                "FMul.b should reference the phi result, got {b:?}"
            );
        }
    }

    fn enc_bra_pt(ofs: i32) -> u64 {
        let raw_24 = (ofs as u32) & 0x00FF_FFFF;
        0xE240_0000_0000_0000u64 | ((raw_24 as u64) << 20) | 0x0007_0000
    }

    fn write_word(bytes: &mut [u8], offset: usize, word: u64) {
        bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
    }

    fn sample_tex_ids(cfg: &Cfg) -> Vec<u32> {
        cfg.blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
            .filter_map(|inst| match inst.op {
                Op::SampleTex { tex_id, .. } => Some(tex_id),
                _ => None,
            })
            .collect()
    }

    fn tex_b_phi_program(second_mov: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; 0x78];
        write_word(&mut bytes, 0x08, 0xe240_0000_0380_0000);
        write_word(&mut bytes, 0x10, 0x4c98_0788_0687_0012);
        write_word(&mut bytes, 0x18, 0xe240_0000_0487_0000);
        write_word(&mut bytes, 0x48, second_mov);
        write_word(&mut bytes, 0x50, 0xe240_0000_0107_0000);
        write_word(&mut bytes, 0x68, 0xdeba_0007_a127_1010);
        write_word(&mut bytes, 0x70, enc_exit());
        bytes
    }

    fn pps_loop_tex_b_program() -> Vec<u8> {
        [
            0x001f_c400_e220_07f0,
            0x4c98_078c_0007_0009,
            0xe003_ff87_4ff7_ff00,
            0x5c98_0780_0ff7_000b,
            0x001f_c800_fe20_07f1,
            0x5c98_0780_0ff7_0008,
            0x5c98_0780_0ff7_0007,
            0x5c98_0780_0ff7_0006,
            0x003f_c000_fda0_07e6,
            0x38f8_7f80_0017_0909,
            0x5b66_0380_0ff7_090f,
            0x36b1_83bf_8007_0007,
            0x001c_4400_fe00_07ed,
            0xe240_0000_0a01_000f,
            0x4c98_0788_05a7_000e,
            0xe003_ff88_0ff7_ff04,
            0x001f_c800_fec0_07f5,
            0x5c98_0780_0ff7_000a,
            0x4c47_0208_15a7_0e0e,
            0x3818_0080_0017_0a0c,
            0x103e_fc02_fe40_073d,
            0x5cb8_0000_00c7_0a05,
            0x4c68_1010_0017_0505,
            0xdeba_0007_a0e7_0400,
            0x041f_c400_fda0_07f6,
            0x1c00_0000_0017_0a0a,
            0x5b6c_0380_0097_0a0f,
            0x5c58_1000_00b7_000b,
            0x001f_c000_fe20_07e1,
            0x5c58_1000_0087_0108,
            0x5c58_1000_0077_0207,
            0x5c58_1000_0067_0306,
            0x001f_c400_fe20_07fd,
            0xe240_0fff_f889_000f,
            0x5c98_0780_00b0_0000,
            0x5c98_0780_0080_0001,
            0x001f_f400_fe00_07f1,
            0x5c98_0780_0070_0002,
            0x5c98_0780_0060_0003,
            0xe300_0000_0000_000f,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
    }

    fn indirect_program(entry_count: usize, valid_shift: bool) -> (Vec<u8>, Vec<u32>) {
        let all_targets = [0x30u32, 0x38, 0x48, 0x50, 0x58];
        let targets = all_targets[..entry_count].to_vec();
        let mut bytes = vec![0u8; 0x60];
        let imnmx = 0x3820_0380_0007_0000u64 | (((entry_count - 1) as u64) << 20);
        let shift = if valid_shift { 2u64 } else { 1u64 };
        let shl = 0x3848_0000_0007_0000u64 | (shift << 20);
        let ldc = 0xEF94_0010_0007_0000u64;
        let branch_offset = ((-0x30i32 as u32) & 0x00FF_FFFF) as u64;
        let brx = 0xE250_0000_0007_000Fu64 | (branch_offset << 20);
        write_word(&mut bytes, 0x08, imnmx);
        write_word(&mut bytes, 0x10, shl);
        write_word(&mut bytes, 0x18, ldc);
        write_word(&mut bytes, 0x28, brx);
        for &target in &targets {
            write_word(&mut bytes, target as usize, enc_exit());
        }
        (bytes, targets)
    }

    fn tracked_indirect_cfg(entry_count: usize) -> (Cfg, Vec<u32>) {
        let (bytes, targets) = indirect_program(entry_count, true);
        let cfg = build_cfg_with_cbuf(&bytes, |binding, offset| {
            (binding == 1)
                .then(|| targets.get((offset / 4) as usize).copied())
                .flatten()
        });
        (cfg, targets)
    }

    #[test]
    fn decodes_canonical_brx_raw() {
        let raw = 0xE250_0FFF_FC07_000Fu64;
        assert_eq!(
            decode_one(raw).map(|decoded| decoded.opcode),
            Some(Opcode::BRX)
        );
    }

    #[test]
    fn tracks_five_entry_indirect_branch() {
        let (cfg, selectors) = tracked_indirect_cfg(5);
        let BranchKind::Indirect {
            register,
            base,
            cbuf_binding,
            cbuf_offset,
            table_entries,
            count,
            targets,
        } = cfg.blocks[0].branch
        else {
            panic!("expected tracked BRX")
        };
        assert_eq!(register, 0);
        assert_eq!(base, 0);
        assert_eq!((cbuf_binding, cbuf_offset, table_entries), (1, 0, 5));
        assert_eq!(count, 5);
        assert_eq!(
            targets[..count as usize]
                .iter()
                .map(|entry| entry.selector)
                .collect::<Vec<_>>(),
            selectors
        );
        assert_eq!(cfg.successors(0), vec![1, 2, 3, 4, 5]);
        assert_eq!(cfg.unimplemented, 0);
    }

    #[test]
    fn tracks_four_entry_indirect_branch() {
        let (cfg, _) = tracked_indirect_cfg(4);
        assert!(matches!(
            cfg.blocks[0].branch,
            BranchKind::Indirect {
                table_entries: 4,
                count: 4,
                ..
            }
        ));
        assert_eq!(cfg.successors(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn indirect_scheduler_target_advances_to_first_instruction() {
        let (mut bytes, _) = indirect_program(1, true);
        write_word(&mut bytes, 0x48, enc_exit());
        let cfg = build_cfg_with_cbuf(&bytes, |binding, offset| {
            (binding == 1 && offset == 0).then_some(0x40)
        });
        let BranchKind::Indirect { count, targets, .. } = cfg.blocks[0].branch else {
            panic!("expected tracked BRX")
        };
        assert_eq!(count, 1);
        assert_eq!(targets[0].selector, 0x40);
        assert_eq!(cfg.block(targets[0].target).start_offset, 0x48);
    }

    #[test]
    fn invalid_indirect_pattern_stays_unimplemented() {
        let (bytes, targets) = indirect_program(5, false);
        let cfg = build_cfg_with_cbuf(&bytes, |binding, offset| {
            (binding == 1)
                .then(|| targets.get((offset / 4) as usize).copied())
                .flatten()
        });
        assert!(cfg.blocks.iter().any(|block| {
            block.program.instructions.iter().any(|instruction| {
                matches!(
                    instruction.op,
                    Op::Unimplemented {
                        opcode: Opcode::BRX,
                        ..
                    }
                )
            })
        }));
    }

    #[test]
    fn pps_tex_b_traces_predicated_handle_across_blocks() {
        let mut bytes = vec![0u8; 0x40];
        write_word(&mut bytes, 0x08, 0x4c98_0788_0683_0012);
        write_word(&mut bytes, 0x10, 0xe240_0000_0107_0000);
        write_word(&mut bytes, 0x28, 0x4c47_0208_1683_1212);
        write_word(&mut bytes, 0x30, 0xdeba_0007_a123_1010);
        write_word(&mut bytes, 0x38, enc_exit());

        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 2);
        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(
            sample_tex_ids(&cfg),
            vec![crate::bindless_texture_id_pair(2, 0x68, Some(0x168)); 4]
        );
    }

    #[test]
    fn tex_b_unanimous_phi_preserves_cbuf_origin() {
        let bytes = tex_b_phi_program(0x4c98_0788_0687_0012);
        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.unimplemented, 0);
        assert!(cfg.blocks[3]
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::Phi { .. }) && inst.dest_reg == Some(18)));
        assert_eq!(
            sample_tex_ids(&cfg),
            vec![crate::bindless_texture_id_pair(2, 0x68, None); 4]
        );
    }

    #[test]
    fn tex_b_differing_phi_origins_fail_closed() {
        let bytes = tex_b_phi_program(0x4c98_0788_0697_0012);
        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.unimplemented, 1);
        assert!(cfg
            .blocks
            .iter()
            .any(
                |block| block.program.instructions.iter().any(|inst| matches!(
                    inst.op,
                    Op::Unimplemented {
                        opcode: Opcode::TEX_b,
                        ..
                    }
                ))
            ));
        assert!(sample_tex_ids(&cfg).is_empty());
    }

    #[test]
    fn pps_tex_b_resolves_loop_invariant_self_phi() {
        let cfg = build_cfg(&pps_loop_tex_b_program());

        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(
            sample_tex_ids(&cfg),
            vec![crate::bindless_texture_id_pair(2, 0x5a, Some(0x15a)); 4]
        );
        let loop_block = cfg
            .blocks
            .iter()
            .find(|block| block.start_offset == 0x98)
            .expect("PPS loop block");
        let phi = loop_block
            .program
            .instructions
            .iter()
            .find(|inst| matches!(inst.op, Op::Phi { .. }) && inst.dest_reg == Some(14))
            .expect("R14 loop phi");
        let phi_id = phi.result.expect("R14 phi result");
        let Op::Phi { sources } = &phi.op else {
            unreachable!()
        };
        assert!(sources
            .iter()
            .any(|(_, source)| *source == Value::Inst(phi_id)));
    }

    #[test]
    fn bindless_tld_resolves_loop_invariant_self_phi() {
        let mut bytes = pps_loop_tex_b_program();
        write_word(&mut bytes, 0xb8, 0xdd3a_0007_a0e7_0400);

        let cfg = build_compute_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let expected = crate::ir::TextureHandleOrigin::Bindless {
            cbuf_binding: 2,
            cbuf_word_offset: 0x5a,
            cbuf_secondary_word_offset: Some(0x15a),
        };
        let handles = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
            .filter_map(|inst| match inst.op {
                Op::TexelFetchHandle { handle, .. } => Some(handle),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(handles, vec![expected; 4]);
    }

    #[test]
    fn pps_tex_b_loop_phi_revalidation_fails_closed_on_changed_handle() {
        let mut bytes = pps_loop_tex_b_program();
        write_word(&mut bytes, 0xf8, 0x4c98_0788_05b7_000e);

        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.unimplemented, 1);
        assert!(sample_tex_ids(&cfg).is_empty());
        assert!(cfg
            .blocks
            .iter()
            .any(
                |block| block.program.instructions.iter().any(|inst| matches!(
                    inst.op,
                    Op::Unimplemented {
                        opcode: Opcode::TEX_b,
                        raw: 0xdeba_0007_a0e7_0400,
                    }
                ))
            ));
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
        assert!(matches!(
            cfg.blocks[0].branch,
            BranchKind::Unconditional { target: 1 }
        ));
        assert!(matches!(
            cfg.blocks[1].branch,
            BranchKind::Conditional { target: 1, .. }
        ));
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
        let Op::Phi { sources } = &phi.op else {
            unreachable!()
        };
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
        let bytes = build_program(&[enc_fmul_reg(2, 0, 1), enc_fmul_reg(3, 0, 1), enc_exit()]);
        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 1);
        assert!(matches!(cfg.blocks[0].branch, BranchKind::Exit));
    }

    #[test]
    fn merge_dual_vertex_replaces_exit_and_appends_b() {
        let nop = 0x50B0_0000_0007_0F00u64;
        let exit = 0xE300_0000_0007_000Fu64;
        let mut a = Vec::new();
        a.extend_from_slice(&0u64.to_le_bytes());
        a.extend_from_slice(&nop.to_le_bytes());
        a.extend_from_slice(&exit.to_le_bytes());
        a.extend_from_slice(&exit.to_le_bytes());
        let mut b = Vec::new();
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&nop.to_le_bytes());
        b.extend_from_slice(&exit.to_le_bytes());
        b.extend_from_slice(&nop.to_le_bytes());
        let merged = merge_dual_vertex_sass(&a, &b).unwrap();
        assert_eq!(merged.len(), a.len() + b.len());
        for slot in [0x10usize, 0x18] {
            let raw = u64::from_le_bytes(merged[slot..slot + 8].try_into().unwrap());
            assert_eq!(raw, nop);
        }
        let b_exit = u64::from_le_bytes(merged[0x20 + 0x10..0x20 + 0x18].try_into().unwrap());
        assert_eq!(b_exit, exit);
        let cfg = build_cfg(&merged);
        assert!(matches!(
            cfg.blocks.last().unwrap().branch,
            BranchKind::Exit
        ));
    }

    #[test]
    fn merge_dual_vertex_bails_without_terminal_exit() {
        let nop = 0x50B0_0000_0007_0F00u64;
        let predicated_exit = 0xE300_0000_0000_000Fu64;
        let mut a = Vec::new();
        a.extend_from_slice(&0u64.to_le_bytes());
        a.extend_from_slice(&nop.to_le_bytes());
        a.extend_from_slice(&predicated_exit.to_le_bytes());
        a.extend_from_slice(&nop.to_le_bytes());
        let b = a.clone();
        assert!(merge_dual_vertex_sass(&a, &b).is_none());
    }
}

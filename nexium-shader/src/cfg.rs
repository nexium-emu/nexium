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
pub struct StorageBufferIndirection {
    pub parent_buffer_index: u32,
    pub pointer_offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageBufferAddr {
    pub cbuf_binding: u8,
    pub cbuf_offset: u32,
    pub align: u32,
    pub indirect: Option<StorageBufferIndirection>,
    pub required_size: u32,
    pub has_dynamic_offset: bool,
}

impl StorageBufferAddr {
    pub const fn direct(cbuf_binding: u8, cbuf_offset: u32, align: u32) -> Self {
        Self {
            cbuf_binding,
            cbuf_offset,
            align,
            indirect: None,
            required_size: 0,
            has_dynamic_offset: false,
        }
    }

    pub const fn is_direct(self) -> bool {
        self.indirect.is_none()
    }
}

#[derive(Clone, Copy, Debug)]
struct TrackedStorageAddr {
    descriptor: StorageBufferAddr,
    base_addr_lo: Value,
    relative_offset: Option<i64>,
}

fn constant_offset(v: Value, defs: &HashMap<u32, Op>, depth: u32) -> Option<i64> {
    if depth > 24 {
        return None;
    }
    match v {
        Value::Zero => Some(0),
        Value::ImmU32(value) => Some(value as i32 as i64),
        Value::ImmF32(value) => Some(value.to_bits() as i32 as i64),
        Value::GprIn(_) => None,
        Value::Inst(id) => match defs.get(&id.0)? {
            Op::Mov(source) => constant_offset(*source, defs, depth + 1),
            Op::Bfe { a, b, signed: false } => {
                let value = constant_offset(*a, defs, depth + 1)? as u32;
                let control = constant_offset(*b, defs, depth + 1)? as u32;
                let shift = control & 0xff;
                let width = (control >> 8) & 0xff;
                if shift >= 32 || width == 0 { return Some(0); }
                let mask = u32::MAX.checked_shr(32u32.saturating_sub(width)).unwrap_or(0);
                Some(((value >> shift) & mask) as i64)
            },
            Op::IAdd { a, b, neg_a, neg_b } => {
                let a = constant_offset(*a, defs, depth + 1)?;
                let b = constant_offset(*b, defs, depth + 1)?;
                let a = if *neg_a { a.checked_neg()? } else { a };
                let b = if *neg_b { b.checked_neg()? } else { b };
                a.checked_add(b)
            }
            Op::IScAdd {
                a,
                b,
                shift,
                neg_a,
                neg_b,
            } => {
                let a = constant_offset(*a, defs, depth + 1)?;
                let b = constant_offset(*b, defs, depth + 1)?;
                let a = if *neg_a { a.checked_neg()? } else { a };
                let b = if *neg_b { b.checked_neg()? } else { b };
                a.checked_shl(u32::from(*shift))?.checked_add(b)
            }
            _ => None,
        },
    }
}

fn add_relative_offset(base: Option<i64>, delta: Option<i64>) -> Option<i64> {
    base.and_then(|base| delta.and_then(|delta| base.checked_add(delta)))
}

fn static_offset_from_base(
    v: Value,
    base: Value,
    defs: &HashMap<u32, Op>,
    depth: u32,
) -> Option<i64> {
    if depth > 24 {
        return None;
    }
    if values_equal(&v, &base) {
        return Some(0);
    }
    let Value::Inst(id) = v else {
        return None;
    };
    match defs.get(&id.0)? {
        Op::Mov(source) => static_offset_from_base(*source, base, defs, depth + 1),
        Op::IAdd { a, b, neg_a, neg_b } => {
            if !*neg_a {
                if let Some(offset) = static_offset_from_base(*a, base, defs, depth + 1) {
                    let delta = constant_offset(*b, defs, depth + 1)?;
                    let delta = if *neg_b { delta.checked_neg()? } else { delta };
                    return offset.checked_add(delta);
                }
            }
            if !*neg_b {
                let offset = static_offset_from_base(*b, base, defs, depth + 1)?;
                let delta = constant_offset(*a, defs, depth + 1)?;
                let delta = if *neg_a { delta.checked_neg()? } else { delta };
                return offset.checked_add(delta);
            }
            None
        }
        Op::IScAdd {
            a,
            b,
            shift,
            neg_a,
            neg_b,
        } => {
            if *shift == 0 && !*neg_a {
                if let Some(offset) = static_offset_from_base(*a, base, defs, depth + 1) {
                    let delta = constant_offset(*b, defs, depth + 1)?;
                    let delta = if *neg_b { delta.checked_neg()? } else { delta };
                    return offset.checked_add(delta);
                }
            }
            if !*neg_b {
                let offset = static_offset_from_base(*b, base, defs, depth + 1)?;
                let delta = constant_offset(*a, defs, depth + 1)?;
                let delta = if *neg_a { delta.checked_neg()? } else { delta };
                return delta.checked_shl(u32::from(*shift))?.checked_add(offset);
            }
            None
        }
        _ => None,
    }
}

fn track_storage_base(
    start: Value,
    defs: &HashMap<u32, Op>,
    buffers: &[StorageBufferAddr],
) -> Option<TrackedStorageAddr> {
    track_dfs(start, defs, buffers, true, 0).or_else(|| track_dfs(start, defs, buffers, false, 0))
}

fn writes_predicate(op: &Op, predicate: u8) -> bool {
    match op {
        Op::FSetPred {
            dest_p, dest_np, ..
        }
        | Op::ISetPred {
            dest_p, dest_np, ..
        }
        | Op::HSetPred {
            dest_p, dest_np, ..
        }
        | Op::PSetPred {
            dest_p, dest_np, ..
        }
        | Op::CSetPred {
            dest_p, dest_np, ..
        } => *dest_p == predicate || *dest_np == predicate,
        Op::Shfl { pred_dest, .. } | Op::SubgroupVote { pred_dest, .. } => *pred_dest == predicate,
        _ => false,
    }
}

fn track_storage_access(
    start: Value,
    consumer_pred: Option<Predicate>,
    block_index: usize,
    instruction_index: usize,
    defs: &HashMap<u32, Op>,
    def_locations: &HashMap<u32, (usize, usize)>,
    blocks: &[BasicBlock],
    buffers: &[StorageBufferAddr],
) -> Option<(TrackedStorageAddr, Value)> {
    if let Some(tracked) = track_storage_base(start, defs, buffers) {
        return Some((tracked, start));
    }

    let consumer_pred = consumer_pred?;
    let Value::Inst(address_id) = start else {
        return None;
    };
    let &(definition_block, definition_index) = def_locations.get(&address_id.0)?;
    if definition_block != block_index || definition_index >= instruction_index {
        return None;
    }
    let Op::SelectPred {
        pred,
        if_true,
        if_false,
    } = defs.get(&address_id.0)?
    else {
        return None;
    };
    if pred.idx != consumer_pred.idx
        || blocks[block_index].program.instructions[definition_index + 1..instruction_index]
            .iter()
            .any(|instruction| writes_predicate(&instruction.op, pred.idx))
    {
        return None;
    }
    let active_address = if pred.negate == consumer_pred.negate {
        *if_true
    } else {
        *if_false
    };
    let tracked = track_storage_base(active_address, defs, buffers)?;
    Some((tracked, active_address))
}

fn track_dfs(
    v: Value,
    defs: &HashMap<u32, Op>,
    buffers: &[StorageBufferAddr],
    biased: bool,
    depth: u32,
) -> Option<TrackedStorageAddr> {
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
            Some(TrackedStorageAddr {
                descriptor: StorageBufferAddr::direct(*binding, *byte_offset, align),
                base_addr_lo: v,
                relative_offset: Some(0),
            })
        }
        Op::LoadStorage {
            buffer_index,
            addr_lo,
            base_addr_lo,
            imm,
            ..
        } => {
            let parent = *buffers.get(*buffer_index as usize)?;
            let pointer_offset = static_offset_from_base(*addr_lo, *base_addr_lo, defs, depth + 1)?;
            let pointer_offset = pointer_offset.checked_add(i64::from(*imm))?;
            let pointer_offset = u32::try_from(pointer_offset).ok()?;
            Some(TrackedStorageAddr {
                descriptor: StorageBufferAddr {
                    cbuf_binding: parent.cbuf_binding,
                    cbuf_offset: parent.cbuf_offset,
                    align: parent.align,
                    indirect: Some(StorageBufferIndirection {
                        parent_buffer_index: *buffer_index,
                        pointer_offset,
                    }),
                    required_size: 0,
                    has_dynamic_offset: false,
                },
                base_addr_lo: v,
                relative_offset: Some(0),
            })
        }
        Op::Mov(source) => track_dfs(*source, defs, buffers, biased, depth + 1),
        Op::Phi { sources } => {
            let mut tracked = None;
            for (_, source) in sources {
                if *source == v {
                    continue;
                }
                if let Some(t) = track_dfs(*source, defs, buffers, biased, depth + 1) {
                    tracked = Some(t);
                    break;
                }
            }
            let mut tracked: TrackedStorageAddr = tracked?;
            tracked.relative_offset = None;
            Some(tracked)
        }
        Op::SelectPred {
            if_true, if_false, ..
        } => {
            let if_true = track_dfs(*if_true, defs, buffers, biased, depth + 1);
            let if_false = track_dfs(*if_false, defs, buffers, biased, depth + 1);
            match (if_true, if_false) {
                (Some(mut left), Some(right))
                    if same_storage_origin(left.descriptor, right.descriptor) =>
                {
                    if left.relative_offset != right.relative_offset {
                        left.relative_offset = None;
                    }
                    Some(left)
                }
                _ => None,
            }
        }
        Op::IAdd { a, b, neg_a, neg_b } => {
            if !*neg_a {
                if let Some(mut tracked) = track_dfs(*a, defs, buffers, biased, depth + 1) {
                    let delta = constant_offset(*b, defs, depth + 1).and_then(|delta| {
                        if *neg_b {
                            delta.checked_neg()
                        } else {
                            Some(delta)
                        }
                    });
                    tracked.relative_offset = add_relative_offset(tracked.relative_offset, delta);
                    return Some(tracked);
                }
            }
            if !*neg_b {
                let mut tracked = track_dfs(*b, defs, buffers, biased, depth + 1)?;
                let delta = constant_offset(*a, defs, depth + 1).and_then(|delta| {
                    if *neg_a {
                        delta.checked_neg()
                    } else {
                        Some(delta)
                    }
                });
                tracked.relative_offset = add_relative_offset(tracked.relative_offset, delta);
                Some(tracked)
            } else {
                None
            }
        }
        Op::IScAdd {
            a,
            b,
            shift,
            neg_a,
            neg_b,
        } => {
            if *shift == 0 && !*neg_a {
                if let Some(mut tracked) = track_dfs(*a, defs, buffers, biased, depth + 1) {
                    let delta = constant_offset(*b, defs, depth + 1).and_then(|delta| {
                        if *neg_b {
                            delta.checked_neg()
                        } else {
                            Some(delta)
                        }
                    });
                    tracked.relative_offset = add_relative_offset(tracked.relative_offset, delta);
                    return Some(tracked);
                }
            }
            if !*neg_b {
                let mut tracked = track_dfs(*b, defs, buffers, biased, depth + 1)?;
                let delta = constant_offset(*a, defs, depth + 1)
                    .and_then(|delta| {
                        if *neg_a {
                            delta.checked_neg()
                        } else {
                            Some(delta)
                        }
                    })
                    .and_then(|delta| delta.checked_shl(u32::from(*shift)));
                tracked.relative_offset = add_relative_offset(tracked.relative_offset, delta);
                Some(tracked)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn required_access_end(relative_offset: Option<i64>, immediate: i32) -> Option<u32> {
    let start = relative_offset?.checked_add(i64::from(immediate))?;
    u32::try_from(start).ok()?.checked_add(4)
}

fn same_storage_origin(a: StorageBufferAddr, b: StorageBufferAddr) -> bool {
    a.cbuf_binding == b.cbuf_binding
        && a.cbuf_offset == b.cbuf_offset
        && a.align == b.align
        && a.indirect == b.indirect
}

fn intern_storage_buffer(
    buffers: &mut Vec<StorageBufferAddr>,
    mut descriptor: StorageBufferAddr,
) -> u32 {
    if let Some(indirect) = descriptor.indirect {
        if let Some(parent) = buffers.get_mut(indirect.parent_buffer_index as usize) {
            parent.required_size = parent
                .required_size
                .max(indirect.pointer_offset.saturating_add(8));
        }
    }

    if let Some(index) = buffers
        .iter()
        .position(|existing| same_storage_origin(*existing, descriptor))
    {
        buffers[index].required_size = buffers[index].required_size.max(descriptor.required_size);
        buffers[index].has_dynamic_offset |= descriptor.has_dynamic_offset;
        return index as u32;
    }

    let index = buffers.len() as u32;
    descriptor.required_size = descriptor.required_size.max(4);
    buffers.push(descriptor);
    index
}

fn fold_stable_predicate_choices(cfg: &mut Cfg) {
    for block in &mut cfg.blocks {
        let mut versions = [0u64; 8];
        let mut choices: HashMap<u32, (Predicate, u64, Value, Value)> = HashMap::new();
        for inst in &mut block.program.instructions {
            if let Op::SelectPred { pred, if_true, if_false } = &mut inst.op {
                let version = versions[pred.idx as usize];
                for (value, active) in [(if_true, true), (if_false, false)] {
                    for _ in 0..24 {
                        let Value::Inst(id) = *value else { break };
                        let Some(&(inner, inner_version, yes, no)) = choices.get(&id.0) else { break };
                        if inner.idx != pred.idx || inner_version != version { break; }
                        *value = if active == (inner.negate == pred.negate) { yes } else { no };
                    }
                }
            }
            match (inst.result, &inst.op) {
                (Some(id), Op::SelectPred { pred, if_true, if_false }) => {
                    choices.insert(id.0, (*pred, versions[pred.idx as usize], *if_true, *if_false));
                }
                (Some(id), Op::Mov(Value::Inst(source))) if inst.pred.is_none() => {
                    if let Some(&choice) = choices.get(&source.0) {
                        choices.insert(id.0, choice);
                    }
                }
                _ => {}
            }
            for pred in 0..8 {
                if writes_predicate(&inst.op, pred) { versions[pred as usize] += 1; }
            }
        }
    }
}

fn shared_slot(v: Value, defs: &HashMap<u32, Op>, depth: u32) -> Option<(Value, u64, u64, u64)> {
    if depth > 24 { return None; }
    let Value::Inst(id) = v else { return None };
    let result = match defs.get(&id.0)? {
        Op::Mov(source) => return shared_slot(*source, defs, depth + 1),
        Op::Bfe { b, signed: false, .. } => {
            let control = constant_offset(*b, defs, 0)? as u32;
            let width = (control >> 8) & 0xff;
            if width == 0 || width > 16 || control & 0xff >= 32 { return None; }
            (v, 1, 0, (1u64 << width) - 1)
        }
        Op::LocalInvocationId { .. } => (v, 1, 0, 1023),
        Op::IAdd { a, b, neg_a: false, neg_b: false } => {
            let (source, offset) = if let Some(offset) = constant_offset(*b, defs, 0) {
                (*a, u64::try_from(offset).ok()?)
            } else { (*b, u64::try_from(constant_offset(*a, defs, 0)?).ok()?) };
            let (root, stride, base, bound) = shared_slot(source, defs, depth + 1)?;
            (root, stride, base.checked_add(offset)?, bound)
        }
        Op::IMul { a, b } => {
            let (source, scale) = if let Some(scale) = constant_offset(*b, defs, 0) {
                (*a, u64::try_from(scale).ok()?)
            } else { (*b, u64::try_from(constant_offset(*a, defs, 0)?).ok()?) };
            let (root, stride, base, bound) = shared_slot(source, defs, depth + 1)?;
            (root, stride.checked_mul(scale)?, base.checked_mul(scale)?, bound)
        }
        _ => return None,
    };
    (result.1.checked_mul(result.3)?.checked_add(result.2)? <= u32::MAX as u64).then_some(result)
}

fn track_shared_pointer_spills(cfg: &Cfg, defs: &mut HashMap<u32, Op>) {
    let mut stores = Vec::new();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.program.instructions.iter().enumerate() {
            match inst.op {
                Op::SharedAtomic { .. } => return,
                Op::StoreShared { addr, value } => {
                    let Some(slot) = shared_slot(addr, defs, 0) else { return };
                    stores.push((bi, ii, inst.pred, slot, value));
                }
                _ => {}
            }
        }
    }
    if stores.is_empty() { return; }
    let predecessors = cfg.predecessors();
    let all: HashSet<usize> = (0..cfg.blocks.len()).collect();
    let mut dominators = vec![all; cfg.blocks.len()];
    dominators[0] = HashSet::from([0]);
    loop {
        let mut changed = false;
        for bi in 1..cfg.blocks.len() {
            let mut next = if let Some(&first) = predecessors[bi].first() {
                dominators[first as usize].clone()
            } else { HashSet::new() };
            for &pred in predecessors[bi].iter().skip(1) {
                next.retain(|entry| dominators[pred as usize].contains(entry));
            }
            next.insert(bi);
            if next != dominators[bi] { dominators[bi] = next; changed = true; }
        }
        if !changed { break; }
    }
    let mut origins = Vec::new();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for (ii, inst) in block.program.instructions.iter().enumerate() {
            let (Some(id), Op::LoadShared { addr }) = (inst.result, &inst.op) else { continue };
            let Some(slot) = shared_slot(*addr, defs, 0) else { continue };
            if slot.1 < 4 { continue; }
            let mut source = None;
            let mut safe = true;
            for &(sbi, sii, pred, other, value) in &stores {
                if slot.0 != other.0 || slot.1 != other.1 { safe = false; break; }
                let residue = slot.2.abs_diff(other.2) % slot.1;
                if slot.2 == other.2 {
                    if source.is_some() || pred.is_some() || !dominators[bi].contains(&sbi)
                        || (bi == sbi && sii >= ii) { safe = false; break; }
                    source = Some(value);
                } else if residue < 4 || slot.1 - residue < 4 { safe = false; break; }
            }
            if safe {
                if let Some(value) = source { origins.push((id.0, Op::Mov(value))); }
            }
        }
    }
    defs.extend(origins);
}

pub fn collect_storage_buffers(cfg: &mut Cfg) -> Vec<StorageBufferAddr> {
    fold_stable_predicate_choices(cfg);
    let mut defs: HashMap<u32, Op> = HashMap::new();
    let mut def_locations: HashMap<u32, (usize, usize)> = HashMap::new();
    for (block_index, block) in cfg.blocks.iter().enumerate() {
        for (instruction_index, inst) in block.program.instructions.iter().enumerate() {
            if let Some(r) = inst.result {
                defs.insert(r.0, inst.op.clone());
                def_locations.insert(r.0, (block_index, instruction_index));
            }
        }
    }

    track_shared_pointer_spills(cfg, &mut defs);
    let mut buffers: Vec<StorageBufferAddr> = Vec::new();
    loop {
        let mut rewrites: Vec<(usize, usize, Op, Option<ValueId>)> = Vec::new();
        for (bi, block) in cfg.blocks.iter().enumerate() {
            for (ii, inst) in block.program.instructions.iter().enumerate() {
                let global = match inst.op {
                    Op::LoadGlobal { addr_lo, offset } => {
                        Some((addr_lo, offset, None, inst.pred, None))
                    }
                    Op::StoreGlobal {
                        addr_lo,
                        offset,
                        value,
                    } => Some((addr_lo, offset, Some(value), inst.pred, None)),
                    Op::GlobalAtomic {
                        addr_lo,
                        offset,
                        value,
                        op,
                        is_signed,
                    } => Some((
                        addr_lo,
                        offset,
                        Some(value),
                        inst.pred,
                        Some((op, is_signed)),
                    )),
                    _ => None,
                };
                let Some((addr_lo, offset, value, predicate, atomic)) = global else {
                    continue;
                };
                let Some((mut tracked, access_addr_lo)) = track_storage_access(
                    addr_lo,
                    predicate,
                    bi,
                    ii,
                    &defs,
                    &def_locations,
                    &cfg.blocks,
                    &buffers,
                ) else {
                    continue;
                };

                let required_size = required_access_end(tracked.relative_offset, offset);
                if tracked.descriptor.indirect.is_some() && required_size.is_none() {
                    continue;
                }
                tracked.descriptor.required_size = required_size.unwrap_or(0);
                tracked.descriptor.has_dynamic_offset = required_size.is_none();
                let buffer_index = intern_storage_buffer(&mut buffers, tracked.descriptor);
                let descriptor = buffers[buffer_index as usize];
                let op = if let (Some(value), Some((atomic_op, is_signed))) = (value, atomic) {
                    Op::StorageAtomic {
                        buffer_index,
                        addr_lo: access_addr_lo,
                        base_addr_lo: tracked.base_addr_lo,
                        imm: offset,
                        value,
                        op: atomic_op,
                        is_signed,
                        cbuf_binding: descriptor.cbuf_binding,
                        cbuf_offset: descriptor.cbuf_offset,
                        align: descriptor.align,
                    }
                } else if let Some(value) = value {
                    Op::StoreStorage {
                        buffer_index,
                        addr_lo: access_addr_lo,
                        base_addr_lo: tracked.base_addr_lo,
                        imm: offset,
                        value,
                        cbuf_binding: descriptor.cbuf_binding,
                        cbuf_offset: descriptor.cbuf_offset,
                        align: descriptor.align,
                    }
                } else {
                    Op::LoadStorage {
                        buffer_index,
                        addr_lo: access_addr_lo,
                        base_addr_lo: tracked.base_addr_lo,
                        imm: offset,
                        cbuf_binding: descriptor.cbuf_binding,
                        cbuf_offset: descriptor.cbuf_offset,
                        align: descriptor.align,
                    }
                };
                rewrites.push((bi, ii, op, inst.result));
            }
        }

        if rewrites.is_empty() {
            break;
        }
        for (bi, ii, op, result) in rewrites {
            cfg.blocks[bi].program.instructions[ii].op = op.clone();
            if let Some(result) = result {
                defs.insert(result.0, op);
            }
        }
    }
    if std::env::var_os("NEXIUM_TRACK_GLOBAL_DBG").is_some() {
        for block in cfg.blocks.iter() {
            for inst in block.program.instructions.iter() {
                let addr = match inst.op {
                    Op::LoadGlobal { addr_lo, .. }
                    | Op::StoreGlobal { addr_lo, .. }
                    | Op::GlobalAtomic { addr_lo, .. } => addr_lo,
                    _ => continue,
                };
                let mut chain = String::new();
                let mut v = addr;
                for _ in 0..8 {
                    let Value::Inst(id) = v else {
                        chain.push_str(&format!(" <- {v:?}"));
                        break;
                    };
                    let Some(op) = defs.get(&id.0) else {
                        chain.push_str(" <- <no def>");
                        break;
                    };
                    chain.push_str(&format!(" <- {op:?}"));
                    v = match op {
                        Op::Mov(source) => *source,
                        Op::IAdd { a, .. } | Op::IScAdd { a, .. } => *a,
                        _ => break,
                    };
                }
                log::warn!("[track-global-miss] {:?} addr chain:{}", inst.op, chain);
            }
        }
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

fn bra_flow_conditional(raw: u64) -> bool {
    !matches!(raw & 0x1f, 0 | 15)
}

fn flow_branch_predicate(raw: u64) -> Option<Predicate> {
    if bra_flow_conditional(raw) {
        Some(Predicate { idx: 8, negate: false })
    } else {
        decoded_pred(raw)
    }
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
    Pcnt,
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
                    Opcode::PCNT => stack.push((FlowToken::Pcnt, bra_target(offset, raw))),
                    Opcode::SYNC | Opcode::BRK | Opcode::CONT => {
                        let token = match d.opcode {
                            Opcode::SYNC => FlowToken::Ssy,
                            Opcode::BRK => FlowToken::Pbk,
                            Opcode::CONT => FlowToken::Pcnt,
                            _ => unreachable!(),
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
                        if decoded_pred(raw).is_some() || bra_flow_conditional(raw) {
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
                    Opcode::EXIT if decoded_pred(raw).is_none() && raw & 0x1f == 15 && !exit_never_taken(raw) => break,
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
                Opcode::EXIT if !exit_never_taken(raw) => {
                    if pred.is_some() || raw & 0x1f == 13 {
                        if next < bytes.len() && leaders.insert(next) {
                            worklist.push(next);
                        }
                    }
                    break;
                }
                Opcode::BRA | Opcode::JMP => {
                    let target = bra_target(offset, raw);
                    if target < bytes.len() && leaders.insert(target) {
                        worklist.push(target);
                    }
                    if pred.is_some() || bra_flow_conditional(raw) {
                        if next < bytes.len() && leaders.insert(next) {
                            worklist.push(next);
                        }
                        offset = next;
                        continue;
                    }
                    break;
                }
                Opcode::SYNC | Opcode::BRK | Opcode::CONT => {
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

pub fn build_geometry_cfg(bytes: &[u8]) -> Cfg {
    build_cfg_with_cbuf_stage(bytes, |_, _| None, ShaderStage::Geometry)
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

    let mut forced = LoopCarriedPhis::empty(topology.len());
    let cfg = build_cfg_blocks(bytes, stage, &topology, &preds, &forced);
    let gaps = loop_carried_phi_gaps(&cfg.blocks, &preds);
    if gaps.is_empty() {
        return cfg;
    }
    forced = gaps;
    build_cfg_blocks(bytes, stage, &topology, &preds, &forced)
}

struct LoopCarriedPhis {
    regs: Vec<BTreeSet<u8>>,
    preds: Vec<BTreeSet<u8>>,
}

impl LoopCarriedPhis {
    fn empty(blocks: usize) -> Self {
        Self {
            regs: vec![BTreeSet::new(); blocks],
            preds: vec![BTreeSet::new(); blocks],
        }
    }

    fn is_empty(&self) -> bool {
        self.regs.iter().all(BTreeSet::is_empty) && self.preds.iter().all(BTreeSet::is_empty)
    }
}

fn natural_loop_body(header: usize, preds: &[Vec<BlockId>]) -> Option<HashSet<BlockId>> {
    let mut body: HashSet<BlockId> = HashSet::new();
    let mut stack: Vec<BlockId> = preds[header]
        .iter()
        .copied()
        .filter(|&p| p as usize >= header)
        .collect();
    if stack.is_empty() {
        return None;
    }
    body.insert(header as BlockId);
    while let Some(block) = stack.pop() {
        if !body.insert(block) {
            continue;
        }
        stack.extend(preds[block as usize].iter().copied());
    }
    Some(body)
}

fn loop_carried_phi_gaps(blocks: &[BasicBlock], preds: &[Vec<BlockId>]) -> LoopCarriedPhis {
    let mut gaps = LoopCarriedPhis::empty(blocks.len());
    for header in 0..blocks.len() {
        let Some(body) = natural_loop_body(header, preds) else {
            continue;
        };
        let mut reg_defs: BTreeSet<u8> = BTreeSet::new();
        let mut pred_defs: BTreeSet<u8> = BTreeSet::new();
        for &block in &body {
            let block = &blocks[block as usize];
            let mut results: HashSet<u32> = HashSet::new();
            for inst in &block.program.instructions {
                if let Some(r) = inst.dest_reg.filter(|&r| r != RZ) {
                    reg_defs.insert(r);
                }
                if let Some(id) = inst.result {
                    results.insert(id.0);
                }
            }
            for phi in &block.pred_phis {
                results.insert(phi.result.0);
            }
            for (&pred, id) in &block.pred_exit {
                if pred < 7 && results.contains(&id.0) {
                    pred_defs.insert(pred);
                }
            }
        }
        let forward: Vec<&BasicBlock> = preds[header]
            .iter()
            .filter(|&&p| (p as usize) < header)
            .map(|&p| &blocks[p as usize])
            .collect();
        for r in reg_defs {
            if !forward.iter().any(|b| b.reg_exit.contains_key(&r)) {
                gaps.regs[header].insert(r);
            }
        }
        for p in pred_defs {
            if !forward.iter().any(|b| b.pred_exit.contains_key(&p)) {
                gaps.preds[header].insert(p);
            }
        }
    }
    gaps
}

fn build_cfg_blocks(
    bytes: &[u8],
    stage: ShaderStage,
    topology: &[BlockInfo],
    preds: &[Vec<BlockId>],
    forced: &LoopCarriedPhis,
) -> Cfg {
    let mut blocks: Vec<BasicBlock> = Vec::with_capacity(topology.len());
    let mut total_unimpl: u32 = 0;
    let mut next_value: u32 = 0;
    let mut bindless_or_partners: std::collections::HashMap<u32, u32> =
        std::collections::HashMap::new();
    let mut value_defs = ValueDefs::new();
    let mut pending_bindless_checks: Vec<(BlockId, PendingBindlessOriginCheck)> = Vec::new();

    for (bid, info) in topology.iter().enumerate() {
        let (initial_state, phis, after_phis) = compute_initial_reg_state(
            &blocks,
            preds,
            bid as BlockId,
            next_value,
            &forced.regs[bid],
        );
        next_value = after_phis;

        let (initial_pred_state, pred_phis, after_pred_phis) = compute_initial_pred_state(
            &blocks,
            preds,
            bid as BlockId,
            next_value,
            &forced.preds[bid],
        );
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
            if term_off + 8 <= bytes.len() {
                let raw = u64::from_le_bytes(bytes[term_off..term_off + 8].try_into().unwrap());
                if decode_one(raw).is_some_and(|decoded| decoded.opcode == Opcode::EXIT)
                    && raw & 0x1f == 13
                    && !t.emit_exit_flow_predicate(raw)
                {
                    t.unimplemented_count += 1;
                    t.program.emit_void(Op::Unimplemented { opcode: Opcode::EXIT, raw });
                }
                if decode_one(raw).is_some_and(|decoded| matches!(decoded.opcode, Opcode::BRA | Opcode::JMP))
                    && bra_flow_conditional(raw)
                {
                    t.emit_bra_flow_predicate(raw);
                }
            }
            if term_off + 8 <= bytes.len() && matches!(info.branch, BranchKind::Exit) {
                let raw = u64::from_le_bytes(bytes[term_off..term_off + 8].try_into().unwrap());
                t.translate_with_defs(raw, &value_defs);
            }
        }

        for inst in &t.program.instructions {
            value_defs.insert_inst(inst);
        }
        value_defs.insert_select_pred_defs(t.select_predicate_defs());
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
        if info.synthetic_exit {
            t.program.exit_reg_state = Some(reg_exit.clone());
        }
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
    total_unimpl += finalize_bindless_origin_checks(&mut blocks, pending_bindless_checks, value_defs);

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
    synthetic_exit: bool,
}

fn discover_topology(
    bytes: &[u8],
    leader_vec: &[usize],
    offset_to_block: &HashMap<usize, BlockId>,
    sync_targets: &HashMap<usize, usize>,
    indirect_branches: &HashMap<usize, BranchKind>,
) -> Vec<BlockInfo> {
    let mut out = Vec::with_capacity(leader_vec.len());
    let mut predicated_exit_sources = Vec::new();
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
                    Opcode::EXIT if !exit_never_taken(raw) => {
                        let predicate = if raw & 0x1f == 13 {
                            Some(Predicate { idx: 8, negate: false })
                        } else {
                            decoded_pred(raw)
                        };
                        match predicate {
                            None => branch = BranchKind::Exit,
                            Some(pred) => {
                                branch = BranchKind::Conditional { target: 0, pred };
                                predicated_exit_sources.push(i);
                            }
                        }
                        terminator_offset = Some(offset);
                        break;
                    }
                    Opcode::BRA | Opcode::JMP => {
                        let target_off = bra_target(offset, raw);
                        let target = *offset_to_block.get(&target_off).unwrap_or(&(i as u32));
                        match flow_branch_predicate(raw) {
                            None => branch = BranchKind::Unconditional { target },
                            Some(pred) => branch = BranchKind::Conditional { target, pred },
                        }
                        terminator_offset = Some(offset);
                        break;
                    }
                    Opcode::SYNC | Opcode::BRK | Opcode::CONT => {
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
            synthetic_exit: false,
        });
    }

    let synthetic_base = out.len() as BlockId;
    for (index, source) in predicated_exit_sources.into_iter().enumerate() {
        let target = synthetic_base + index as BlockId;
        let pred = match out[source].branch {
            BranchKind::Conditional { pred, .. } => pred,
            _ => unreachable!(),
        };
        out[source].branch = BranchKind::Conditional { target, pred };
        let offset = out[source]
            .terminator_offset
            .map(|offset| offset.saturating_add(8).min(bytes.len()))
            .unwrap_or(bytes.len());
        out.push(BlockInfo {
            start: offset,
            end: offset,
            branch: BranchKind::Exit,
            terminator_offset: None,
            synthetic_exit: true,
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
    forced: &BTreeSet<u8>,
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
    all_regs.extend(forced.iter().copied());

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
    forced: &BTreeSet<u8>,
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
    all_preds.extend(forced.iter().copied());

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
    mut defs: ValueDefs,
) -> u32 {
    if checks.is_empty() {
        return 0;
    }

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
                    Opcode::TLD_b => {
                        matches!(inst.op, Op::TexelFetch { .. } | Op::TexelFetchHandle { .. })
                    }
                    Opcode::SUATOM => matches!(inst.op, Op::ImageAtomic { .. }),
                    Opcode::SUST => matches!(inst.op, Op::ImageWrite { .. }),
                    Opcode::SULD => matches!(inst.op, Op::ImageRead { .. }),
                    _ => false,
                })
        });

        if let Some(origin) = resolved.filter(|origin| {
            sample_indices_valid
                && (check.opcode != Opcode::TLD_b || origin.cross_binding_partner_id().is_none())
        }) {
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
                    Op::TexelFetch {
                        cbuf_binding,
                        cbuf_word_offset,
                        cbuf_secondary_word_offset,
                        ..
                    } if check.opcode == Opcode::TLD_b => {
                        let super::ir::TextureHandleOrigin::Bindless {
                            cbuf_binding: binding,
                            cbuf_word_offset: word_offset,
                            cbuf_secondary_word_offset: secondary_word_offset,
                        } = origin.as_texture_handle()
                        else {
                            unreachable!()
                        };
                        *cbuf_binding = binding;
                        *cbuf_word_offset = word_offset;
                        *cbuf_secondary_word_offset = secondary_word_offset;
                    }
                    Op::ImageAtomic { handle, .. } if check.opcode == Opcode::SUATOM => {
                        *handle = origin.as_texture_handle();
                    }
                    Op::ImageWrite { handle, .. } if check.opcode == Opcode::SUST => {
                        *handle = origin.as_texture_handle();
                    }
                    Op::ImageRead { handle, .. } if check.opcode == Opcode::SULD => {
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
                    Opcode::EXIT if decoded_pred(raw).is_none() && raw & 0x1f == 15 && !exit_never_taken(raw) => {
                        exit_offset = Some(offset);
                        break;
                    }
                    Opcode::BRA
                    | Opcode::JMP
                    | Opcode::BRX
                    | Opcode::SSY
                    | Opcode::PBK
                    | Opcode::PCNT
                    | Opcode::SYNC
                    | Opcode::BRK
                    | Opcode::CONT => {
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

    fn enc_psetp_p2() -> u64 {
        0x5090_0000_0007_0017u64
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

    #[test]
    fn complementary_address_writes_respect_predicate_versions() {
        for changed in [false, true] {
            let mut p = Program::new();
            let base = p.emit(Op::LoadCbuf { binding: 0, byte_offset: 0x2d0 }, None);
            let first = p.emit(Op::SelectPred { pred: Predicate { idx: 0, negate: false },
                if_true: Value::Inst(base), if_false: Value::GprIn(8) }, None);
            if changed {
                p.emit_void(Op::PSetPred { dest_p: 0, dest_np: 7,
                    pred_a: 7, neg_pred_a: false, pred_b: 7, neg_pred_b: false,
                    pred_c: 7, neg_pred_c: false, bop_1: super::super::ir::BoolOp::And,
                    bop_2: super::super::ir::BoolOp::And });
            }
            let second = p.emit(Op::SelectPred { pred: Predicate { idx: 0, negate: true },
                if_true: Value::Inst(base), if_false: Value::Inst(first) }, None);
            p.emit(Op::LoadGlobal { addr_lo: Value::Inst(second), offset: 32 }, Some(0));
            let mut cfg = cfg_with_program(p);
            let buffers = collect_storage_buffers(&mut cfg);
            assert_eq!(buffers.len(), usize::from(!changed));
            assert_eq!(cfg.blocks[0].program.instructions.iter().any(|i|
                matches!(i.op, Op::LoadGlobal { .. })), changed);
        }
    }

    #[test]
    fn shared_pointer_origin_requires_dominating_disjoint_writes() {
        for case in 0..5 {
            let mut p = Program::new();
            let lane = p.emit(Op::LocalInvocationId { component: 0 }, None);
            let slot = p.emit(Op::IMul { a: Value::Inst(lane), b: Value::ImmU32(12) }, None);
            let pointer = p.emit(Op::LoadCbuf { binding: 0, byte_offset: 0x310 }, None);
            let store = Op::StoreShared { addr: Value::Inst(slot), value: Value::Inst(pointer) };
            if case != 4 {
                p.emit_pred(store.clone(), None,
                    (case == 3).then_some(Predicate { idx: 0, negate: false }));
            }
            let other = p.emit(Op::IAdd { a: Value::Inst(slot),
                b: Value::ImmU32(if case == 1 { 1 } else { 8 }), neg_a: false, neg_b: false }, None);
            p.emit_void(Op::StoreShared { addr: Value::Inst(other), value: Value::Zero });
            if case == 2 {
                p.emit_void(Op::StoreShared { addr: Value::GprIn(4), value: Value::Zero });
            }
            let read = p.emit(Op::LoadShared { addr: Value::Inst(slot) }, None);
            if case == 4 { p.emit_void(store); }
            p.emit_void(Op::StoreGlobal { addr_lo: Value::Inst(read), offset: 0, value: Value::Zero });
            let mut cfg = cfg_with_program(p);
            let buffers = collect_storage_buffers(&mut cfg);
            assert_eq!(buffers.len(), usize::from(case == 0), "case {case}");
            assert_eq!(cfg.blocks[0].program.instructions.iter().any(|i|
                matches!(i.op, Op::StoreGlobal { .. })), case != 0, "case {case}");
            assert!(cfg.blocks[0].program.instructions.iter().any(|i|
                matches!(i.op, Op::LoadShared { .. })));
        }
    }

    fn cfg_with_program(program: Program) -> Cfg {
        Cfg {
            blocks: vec![BasicBlock {
                id: 0,
                start_offset: 0,
                end_offset: 0,
                branch: BranchKind::Exit,
                program,
                reg_exit: HashMap::new(),
                pred_phis: Vec::new(),
                pred_exit: HashMap::new(),
            }],
            unimplemented: 0,
            bindless_or_partners: HashMap::new(),
        }
    }

    #[test]
    fn storage_buffer_collection_rewrites_nested_pointer_loads_iteratively() {
        let mut program = Program::new();
        let root = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x128,
            },
            Some(8),
        );
        let root_addr = program.emit(Op::Mov(Value::Inst(root)), Some(8));
        let child_pointer = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root_addr),
                offset: 0,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root_addr),
                offset: 4,
            },
            Some(9),
        );
        let child_addr = program.emit(Op::Mov(Value::Inst(child_pointer)), Some(8));
        let child_value = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(child_addr),
                offset: 12,
            },
            Some(12),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(
            buffers,
            vec![
                StorageBufferAddr {
                    cbuf_binding: 0,
                    cbuf_offset: 0x128,
                    align: 8,
                    indirect: None,
                    required_size: 8,
                    has_dynamic_offset: false,
                },
                StorageBufferAddr {
                    cbuf_binding: 0,
                    cbuf_offset: 0x128,
                    align: 8,
                    indirect: Some(StorageBufferIndirection {
                        parent_buffer_index: 0,
                        pointer_offset: 0,
                    }),
                    required_size: 16,
                    has_dynamic_offset: false,
                },
            ]
        );
        let instructions = &cfg.blocks[0].program.instructions;
        assert!(matches!(
            instructions[2].op,
            Op::LoadStorage {
                buffer_index: 0,
                base_addr_lo: Value::Inst(id),
                ..
            } if id == root
        ));
        assert!(matches!(
            instructions[5].op,
            Op::LoadStorage {
                buffer_index: 1,
                base_addr_lo: Value::Inst(id),
                ..
            } if id == child_pointer
        ));
        assert_eq!(instructions[5].result, Some(child_value));
    }

    #[test]
    fn storage_buffer_collection_records_recursive_parent_chain_and_store_span() {
        let mut program = Program::new();
        let root = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x128,
            },
            Some(8),
        );
        let first_pointer = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 0,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 4,
            },
            Some(9),
        );
        let first_addr = program.emit(Op::Mov(Value::Inst(first_pointer)), Some(8));
        let second_pointer = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(first_addr),
                offset: 8,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(first_addr),
                offset: 12,
            },
            Some(9),
        );
        let second_addr = program.emit(Op::Mov(Value::Inst(second_pointer)), Some(8));
        program.emit_void(Op::StoreGlobal {
            addr_lo: Value::Inst(second_addr),
            offset: 12,
            value: Value::ImmU32(0x1234_5678),
        });
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 3);
        assert_eq!(buffers[0].required_size, 8);
        assert_eq!(
            buffers[1].indirect,
            Some(StorageBufferIndirection {
                parent_buffer_index: 0,
                pointer_offset: 0,
            })
        );
        assert_eq!(buffers[1].required_size, 16);
        assert_eq!(
            buffers[2].indirect,
            Some(StorageBufferIndirection {
                parent_buffer_index: 1,
                pointer_offset: 8,
            })
        );
        assert_eq!(buffers[2].required_size, 16);
        assert!(cfg.blocks[0]
            .program
            .instructions
            .iter()
            .all(|inst| !matches!(inst.op, Op::LoadGlobal { .. } | Op::StoreGlobal { .. })));
        assert!(matches!(
            cfg.blocks[0].program.instructions[7].op,
            Op::StoreStorage {
                buffer_index: 2,
                base_addr_lo: Value::Inst(id),
                ..
            } if id == second_pointer
        ));
    }

    #[test]
    fn storage_buffer_collection_leaves_dynamic_indirect_span_unresolved() {
        let mut program = Program::new();
        let root = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x128,
            },
            Some(8),
        );
        let child_pointer = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 0,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 4,
            },
            Some(9),
        );
        let dynamic_addr = program.emit(
            Op::IAdd {
                a: Value::Inst(child_pointer),
                b: Value::GprIn(4),
                neg_a: false,
                neg_b: false,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(dynamic_addr),
                offset: 0,
            },
            Some(12),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 1);
        assert!(matches!(
            cfg.blocks[0].program.instructions[4].op,
            Op::LoadGlobal { .. }
        ));
    }

    #[test]
    fn storage_buffer_collection_rejects_conditional_unknown_pointer() {
        let mut program = Program::new();
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: Predicate {
                    idx: 2,
                    negate: false,
                },
                if_true: Value::Inst(pointer),
                if_false: Value::GprIn(4),
            },
            Some(4),
        );
        for offset in [0x60, 0x64, 0x68, 0x6c] {
            program.emit(
                Op::LoadGlobal {
                    addr_lo: Value::Inst(selected),
                    offset,
                },
                Some(4),
            );
        }
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert!(buffers.is_empty());
        for instruction in &cfg.blocks[0].program.instructions[2..] {
            assert!(matches!(instruction.op, Op::LoadGlobal { .. }));
        }
    }

    #[test]
    fn storage_buffer_bounds_remember_dynamic_accesses() {
        for dynamic in [false, true] {
            let mut program = Program::new();
            let pointer = program.emit(Op::LoadCbuf { binding: 0, byte_offset: 0x290 }, Some(4));
            program.emit(Op::LoadGlobal { addr_lo: Value::Inst(pointer), offset: 128 }, Some(5));
            if dynamic {
                let index = program.emit(Op::LocalInvocationId { component: 0 }, Some(6));
                let address = program.emit(Op::IAdd {
                    a: Value::Inst(pointer), b: Value::Inst(index), neg_a: false, neg_b: false,
                }, Some(7));
                program.emit(Op::LoadGlobal { addr_lo: Value::Inst(address), offset: 0 }, Some(8));
            }
            program.emit(Op::LoadGlobal { addr_lo: Value::Inst(pointer), offset: 280 }, Some(9));
            let mut cfg = cfg_with_program(program);
            let buffers = collect_storage_buffers(&mut cfg);
            assert_eq!(buffers.len(), 1);
            assert_eq!(buffers[0].required_size, 284);
            assert_eq!(buffers[0].has_dynamic_offset, dynamic);
        }
    }

    #[test]
    fn storage_buffer_collection_accepts_matching_predicated_pointer() {
        let mut program = Program::new();
        let predicate = Predicate {
            idx: 2,
            negate: false,
        };
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: predicate,
                if_true: Value::Inst(pointer),
                if_false: Value::GprIn(4),
            },
            Some(4),
        );
        for offset in [0x60, 0x64, 0x68, 0x6c] {
            program.emit_pred(
                Op::LoadGlobal {
                    addr_lo: Value::Inst(selected),
                    offset,
                },
                Some(4),
                Some(predicate),
            );
        }
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(
            buffers,
            vec![StorageBufferAddr {
                cbuf_binding: 0,
                cbuf_offset: 0x2d0,
                align: 16,
                indirect: None,
                required_size: 0x70,
                has_dynamic_offset: false,
            }]
        );
        for instruction in &cfg.blocks[0].program.instructions[2..] {
            assert_eq!(instruction.pred, Some(predicate));
            assert!(matches!(
                instruction.op,
                Op::LoadStorage {
                    addr_lo: Value::Inst(id),
                    base_addr_lo: Value::Inst(base),
                    ..
                } if id == pointer && base == pointer
            ));
        }
    }

    #[test]
    fn storage_buffer_collection_accepts_complementary_predicated_pointer() {
        let mut program = Program::new();
        let selector = Predicate {
            idx: 2,
            negate: false,
        };
        let consumer = Predicate {
            idx: 2,
            negate: true,
        };
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: selector,
                if_true: Value::GprIn(4),
                if_false: Value::Inst(pointer),
            },
            Some(4),
        );
        program.emit_pred(
            Op::LoadGlobal {
                addr_lo: Value::Inst(selected),
                offset: 0x60,
            },
            Some(4),
            Some(consumer),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].cbuf_offset, 0x2d0);
        assert_eq!(buffers[0].required_size, 0x64);
        assert!(matches!(
            cfg.blocks[0].program.instructions[2].op,
            Op::LoadStorage {
                addr_lo: Value::Inst(id),
                ..
            } if id == pointer
        ));
    }

    #[test]
    fn storage_buffer_collection_rejects_redefined_pointer_predicate() {
        let mut program = Program::new();
        let predicate = Predicate {
            idx: 2,
            negate: false,
        };
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: predicate,
                if_true: Value::Inst(pointer),
                if_false: Value::GprIn(4),
            },
            Some(4),
        );
        program.emit(
            Op::Shfl {
                value: Value::Zero,
                index: Value::Zero,
                mask: Value::Zero,
                mode: 0,
                pred_dest: 2,
            },
            None,
        );
        program.emit_pred(
            Op::LoadGlobal {
                addr_lo: Value::Inst(selected),
                offset: 0x60,
            },
            Some(4),
            Some(predicate),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert!(buffers.is_empty());
        assert!(matches!(
            cfg.blocks[0].program.instructions[3].op,
            Op::LoadGlobal { .. }
        ));
    }

    #[test]
    fn storage_buffer_collection_resolves_predicated_conditional_store() {
        let mut program = Program::new();
        let predicate = Predicate {
            idx: 2,
            negate: false,
        };
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: predicate,
                if_true: Value::Inst(pointer),
                if_false: Value::GprIn(4),
            },
            Some(4),
        );
        program.emit_void_pred(
            Op::StoreGlobal {
                addr_lo: Value::Inst(selected),
                offset: 0x60,
                value: Value::ImmU32(7),
            },
            Some(predicate),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].cbuf_binding, 0);
        assert_eq!(buffers[0].cbuf_offset, 0x2d0);
        assert!(matches!(
            cfg.blocks[0].program.instructions[2].op,
            Op::StoreStorage { .. }
        ));

        let mismatched = Predicate {
            idx: 3,
            negate: false,
        };
        let mut program = Program::new();
        let pointer = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x2d0,
            },
            Some(4),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: predicate,
                if_true: Value::Inst(pointer),
                if_false: Value::GprIn(4),
            },
            Some(4),
        );
        program.emit_void_pred(
            Op::StoreGlobal {
                addr_lo: Value::Inst(selected),
                offset: 0x60,
                value: Value::ImmU32(7),
            },
            Some(mismatched),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert!(buffers.is_empty());
        assert!(matches!(
            cfg.blocks[0].program.instructions[2].op,
            Op::StoreGlobal { .. }
        ));
    }

    #[test]
    fn odyssey_predicated_vector_global_load_rewrites_to_storage() {
        let mut bytes = vec![0u8; 0x20];
        write_word(&mut bytes, 0x08, 0x4c98_0780_0b42_0004);
        write_word(&mut bytes, 0x10, 0xeed6_a000_0602_0404);
        write_word(&mut bytes, 0x18, enc_exit());
        let mut cfg = build_compute_cfg(&bytes);
        let predicate = Predicate {
            idx: 2,
            negate: false,
        };
        let globals_before = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
            .filter(|instruction| matches!(instruction.op, Op::LoadGlobal { .. }))
            .collect::<Vec<_>>();
        assert_eq!(globals_before.len(), 4);
        assert!(globals_before
            .iter()
            .all(|instruction| instruction.pred == Some(predicate)));

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].cbuf_binding, 0);
        assert_eq!(buffers[0].cbuf_offset, 0x2d0);
        assert_eq!(buffers[0].required_size, 0x70);
        assert!(cfg
            .blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
            .all(|instruction| !matches!(instruction.op, Op::LoadGlobal { .. })));
    }

    #[test]
    fn storage_buffer_collection_keeps_conditional_indirect_span_unresolved() {
        let mut program = Program::new();
        let root = program.emit(
            Op::LoadCbuf {
                binding: 0,
                byte_offset: 0x128,
            },
            Some(8),
        );
        let child_pointer = program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 0,
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(root),
                offset: 4,
            },
            Some(9),
        );
        let selected = program.emit(
            Op::SelectPred {
                pred: Predicate {
                    idx: 0,
                    negate: false,
                },
                if_true: Value::Inst(child_pointer),
                if_false: Value::GprIn(8),
            },
            Some(8),
        );
        program.emit(
            Op::LoadGlobal {
                addr_lo: Value::Inst(selected),
                offset: 0,
            },
            Some(12),
        );
        let mut cfg = cfg_with_program(program);

        let buffers = collect_storage_buffers(&mut cfg);

        assert_eq!(buffers.len(), 1);
        assert!(matches!(
            cfg.blocks[0].program.instructions[4].op,
            Op::LoadGlobal { .. }
        ));
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
    fn nested_predicated_handle_is_resolved_across_predicate_redefinition_and_blocks() {
        let mut bytes = vec![0u8; 0x60];
        write_word(&mut bytes, 0x08, 0x5090_0380_2007_0017);
        write_word(&mut bytes, 0x10, 0x4c98_0788_05a2_000c);
        write_word(&mut bytes, 0x18, 0x4c47_0208_15a2_0c0c);
        write_word(&mut bytes, 0x28, 0x5c10_0000_00c2_ff0c);
        write_word(&mut bytes, 0x30, 0x4c42_3004_0007_15ff);
        write_word(&mut bytes, 0x38, 0xe240_0000_0087_0000);
        write_word(&mut bytes, 0x48, 0xdd38_0000_80c7_0101);
        write_word(&mut bytes, 0x50, enc_exit());
        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 2);
        assert_eq!(cfg.unimplemented, 0);
        assert!(cfg.blocks[1].program.instructions.iter().any(|inst| matches!(
            inst.op,
            Op::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x5a,
                cbuf_secondary_word_offset: Some(0x15a),
                ..
            }
        )));

        write_word(&mut bytes, 0x18, 0x4c42_3004_0007_15ff);
        write_word(&mut bytes, 0x28, 0x4c47_0208_15a2_0c0c);
        write_word(&mut bytes, 0x30, 0x5c10_0000_00c2_ff0c);
        let changed_predicate = build_cfg(&bytes);
        assert_eq!(changed_predicate.unimplemented, 1);
    }

    #[test]
    fn initialized_different_handle_paths_remain_unresolved() {
        let mut bytes = vec![0u8; 0x60];
        write_word(&mut bytes, 0x08, 0x4c98_0788_0587_000c);
        write_word(&mut bytes, 0x10, 0x5090_0380_2007_0017);
        write_word(&mut bytes, 0x18, 0x4c98_0788_05a2_000c);
        write_word(&mut bytes, 0x28, 0x4c47_0208_15a2_0c0c);
        write_word(&mut bytes, 0x30, 0x5c10_0000_00c2_ff0c);
        write_word(&mut bytes, 0x38, 0xe240_0000_0087_0000);
        write_word(&mut bytes, 0x48, 0xdd38_0000_80c7_0101);
        write_word(&mut bytes, 0x50, enc_exit());
        assert_eq!(build_cfg(&bytes).unimplemented, 1);
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
    fn register_first_written_inside_loop_gets_loop_carried_phi() {
        let bytes = build_program(&[
            0x5C98_0780_0FF7_0001,
            0x366B_0380_0017_0107,
            0x5C98_0780_0038_0002,
            0x1C00_0000_0017_0101,
            0x3663_0380_0037_0107,
            enc_bra_p0(-0x30),
            enc_exit(),
        ]);
        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let header = cfg
            .blocks
            .iter()
            .find(|block| block.start_offset == 0x10)
            .expect("loop header block");
        let phi = header
            .program
            .instructions
            .iter()
            .find(|inst| matches!(inst.op, Op::Phi { .. }) && inst.dest_reg == Some(2))
            .expect("R2 loop-carried phi");
        let Op::Phi { sources } = &phi.op else {
            unreachable!()
        };
        assert!(sources
            .iter()
            .any(|(pred, value)| *pred >= header.id && matches!(value, Value::Inst(_))));
        assert!(sources
            .iter()
            .any(|(pred, value)| *pred < header.id && matches!(value, Value::GprIn(2))));
        let exit = cfg.blocks.last().expect("exit block");
        assert!(matches!(exit.reg_exit.get(&2), Some(Value::Inst(_))));
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
    fn bindless_sust_validates_loop_carried_image_handle() {
        for changed_handle in [false, true] {
            let mut bytes = pps_loop_tex_b_program();
            write_word(&mut bytes, 0xb8, 0xeb20000a00f70c10 | (14 << 39));
            if changed_handle {
                write_word(&mut bytes, 0xf8, 0x4c98_0788_05b7_000e);
            }
            let cfg = build_compute_cfg(&bytes);
            let stores: Vec<_> = cfg.blocks.iter().flat_map(|block| &block.program.instructions)
                .filter_map(|inst| match inst.op { Op::ImageWrite { handle, .. } => Some(handle), _ => None }).collect();
            if changed_handle {
                assert_eq!(cfg.unimplemented, 1);
                assert!(stores.is_empty());
            } else {
                assert_eq!(cfg.unimplemented, 0);
                assert_eq!(stores, vec![crate::ir::TextureHandleOrigin::Bindless {
                    cbuf_binding: 2, cbuf_word_offset: 0x5a, cbuf_secondary_word_offset: Some(0x15a),
                }]);
            }
        }
    }

    #[test]
    fn fragment_tld_b_resolves_dusk_loop_invariant_self_phi() {
        let bytes = build_program(&[
            0x4c98_0784_0047_0003,
            0x0400_00ff_fff7_0303,
            0x4c98_0784_0057_0006,
            0x040f_ff00_0007_0606,
            0x5c47_0200_0067_0306,
            0x5c98_0780_0037_0006,
            enc_bra_pt(24),
            0x50b0_0000_0007_0f00,
            0x50b0_0000_0007_0f00,
            0xdd38_0003_a067_1400,
            enc_bra_p0(16),
            enc_bra_pt(-24),
            enc_exit(),
        ]);
        let cfg = build_fragment_cfg(&bytes);

        assert_eq!(cfg.unimplemented, 0);
        let fetches = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.program.instructions)
            .filter_map(|inst| match inst.op {
                Op::TexelFetch {
                    cbuf_binding,
                    cbuf_word_offset,
                    cbuf_secondary_word_offset,
                    component,
                    ..
                } => Some((
                    cbuf_binding,
                    cbuf_word_offset,
                    cbuf_secondary_word_offset,
                    component,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            fetches,
            vec![(1, 4, None, 0), (1, 4, None, 1), (1, 4, None, 2)]
        );
        assert!(cfg.blocks.iter().any(|block| {
            block.program.instructions.iter().any(|inst| {
                let Some(result) = inst.result else {
                    return false;
                };
                matches!(&inst.op, Op::Phi { sources } if inst.dest_reg == Some(6)
                    && sources.iter().any(|(_, value)| *value == Value::Inst(result)))
            })
        }));
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
    fn pcnt_cont_forms_loop_with_pbk_break_exit() {
        const PBK_EXIT: u64 = 0xE2A0_0000_0207_000F;
        const PCNT_SELF: u64 = 0xE2B0_0FFF_FF87_000F;
        const BRK_P0: u64 = 0xE340_0000_0000_000F;
        const CONT_PT: u64 = 0xE350_0000_0007_000F;

        assert_eq!(decode_one(PCNT_SELF).unwrap().opcode, Opcode::PCNT);
        assert_eq!(decode_one(CONT_PT).unwrap().opcode, Opcode::CONT);

        let bytes = build_program(&[PBK_EXIT, PCNT_SELF, BRK_P0, CONT_PT, enc_exit()]);
        let cfg = build_fragment_cfg(&bytes);

        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(cfg.blocks.len(), 4);
        assert_eq!(cfg.blocks[0].branch, BranchKind::FallThrough);
        assert_eq!(
            cfg.blocks[1].branch,
            BranchKind::Conditional {
                target: 3,
                pred: Predicate {
                    idx: 0,
                    negate: false,
                },
            }
        );
        assert_eq!(
            cfg.blocks[2].branch,
            BranchKind::Unconditional { target: 1 }
        );
        assert_eq!(cfg.blocks[3].branch, BranchKind::Exit);
        assert_eq!(
            cfg.predecessors(),
            vec![vec![], vec![0, 2], vec![1], vec![1]]
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
    fn exit_neu_combines_condition_flags_and_instruction_predicate() {
        for exit in [0xe300_0000_0008_000d, 0xe300_0000_0007_000d] {
            let cfg = build_cfg(&build_program(&[
                0x4b5c_838c_0127_06ff, exit, enc_fadd_reg(3, 2, 1), enc_exit(),
            ]));
            assert_eq!(cfg.unimplemented, 0);
            assert_eq!(cfg.blocks.len(), 3);
            assert!(matches!(cfg.blocks[0].branch, BranchKind::Conditional {
                pred: Predicate { idx: 8, negate: false }, ..
            }));
            let compare = cfg.blocks[0].program.instructions.iter()
                .find(|inst| matches!(inst.op, Op::ISet { .. })).unwrap();
            let condition = cfg.blocks[0].program.instructions.last().unwrap();
            assert!(matches!(condition.op, Op::ISetPred {
                cmp: super::super::ir::ICmp::Ne,
                src_a: Value::Inst(id), src_b: Value::Zero,
                src_pred, src_pred_inv, dest_p: 8, ..
            } if Some(id) == compare.result
                && src_pred == if exit == 0xe300_0000_0008_000d { 0 } else { 7 }
                && src_pred_inv == (exit == 0xe300_0000_0008_000d)));
            assert!(!cfg.blocks[0].pred_exit.contains_key(&8));
        }
        let unsupported = build_cfg(&build_program(&[
            0xe300_0000_0007_000d, enc_exit(),
        ]));
        assert_eq!(unsupported.unimplemented, 1);
    }

    #[test]
    fn negated_predicated_exit_splits_to_stateful_synthetic_exit() {
        let predicated_exit = 0xe300_0000_0008_000f;
        let bytes = build_program(&[
            enc_psetp_p2(),
            enc_fmul_reg(2, 0, 1),
            predicated_exit,
            enc_fadd_reg(3, 2, 1),
            enc_exit(),
        ]);
        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.blocks.len(), 3);
        assert_eq!(
            cfg.blocks[0].branch,
            BranchKind::Conditional {
                target: 2,
                pred: Predicate {
                    idx: 0,
                    negate: true,
                },
            }
        );
        assert_eq!(cfg.successors(0), vec![2, 1]);
        assert!(matches!(cfg.blocks[1].branch, BranchKind::Exit));
        assert!(matches!(cfg.blocks[2].branch, BranchKind::Exit));
        assert!(cfg.blocks[2].program.instructions.is_empty());
        assert_eq!(cfg.blocks[2].reg_exit, cfg.blocks[0].reg_exit);
        assert_eq!(cfg.blocks[2].pred_exit, cfg.blocks[0].pred_exit);
        assert!(cfg.blocks[0].pred_exit.contains_key(&2));
        assert_eq!(
            cfg.blocks[2].program.exit_reg_state.as_ref(),
            Some(&cfg.blocks[0].reg_exit)
        );
        assert!(cfg.blocks[1]
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::FAdd { .. })));
    }

    #[test]
    fn exit_fcsm_tr_remains_fallthrough() {
        let fcsm_tr = 0xe300_0000_0008_001c;
        let bytes = build_program(&[
            enc_fmul_reg(2, 0, 1),
            fcsm_tr,
            enc_fadd_reg(3, 2, 1),
            enc_exit(),
        ]);
        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.blocks.len(), 1);
        assert!(matches!(cfg.blocks[0].branch, BranchKind::Exit));
        assert!(cfg.blocks[0]
            .program
            .instructions
            .iter()
            .any(|inst| matches!(inst.op, Op::FAdd { .. })));
    }

    #[test]
    fn multiple_predicated_exits_get_unique_synthetic_targets() {
        let bytes = build_program(&[
            enc_fmul_reg(2, 0, 1),
            0xe300_0000_0008_000f,
            enc_fadd_reg(3, 2, 1),
            0xe300_0000_0001_000f,
            enc_fmul_reg(4, 3, 1),
            enc_exit(),
        ]);
        let cfg = build_cfg(&bytes);

        assert_eq!(cfg.blocks.len(), 5);
        assert_eq!(
            cfg.blocks[0].branch,
            BranchKind::Conditional {
                target: 3,
                pred: Predicate {
                    idx: 0,
                    negate: true,
                },
            }
        );
        assert_eq!(
            cfg.blocks[1].branch,
            BranchKind::Conditional {
                target: 4,
                pred: Predicate {
                    idx: 1,
                    negate: false,
                },
            }
        );
        assert_eq!(cfg.successors(0), vec![3, 1]);
        assert_eq!(cfg.successors(1), vec![4, 2]);
        assert_eq!(cfg.predecessors()[3], vec![0]);
        assert_eq!(cfg.predecessors()[4], vec![1]);
        assert_eq!(cfg.blocks[3].reg_exit, cfg.blocks[0].reg_exit);
        assert_eq!(cfg.blocks[4].reg_exit, cfg.blocks[1].reg_exit);
        assert_eq!(
            cfg.blocks[3].program.exit_reg_state.as_ref(),
            Some(&cfg.blocks[0].reg_exit)
        );
        assert_eq!(
            cfg.blocks[4].program.exit_reg_state.as_ref(),
            Some(&cfg.blocks[1].reg_exit)
        );
        assert!(!cfg.blocks[3].reg_exit.contains_key(&3));
        assert!(cfg.blocks[4].reg_exit.contains_key(&3));
        assert!(!cfg.blocks[4].reg_exit.contains_key(&4));
        assert!(cfg.blocks[2].reg_exit.contains_key(&4));
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
    #[test]
    fn bra_with_condition_code_test_branches_on_flow_predicate() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&0x5881_8380_0FF7_02FFu64.to_le_bytes());
        bytes.extend_from_slice(&0xE240_0000_0100_000Du64.to_le_bytes());
        bytes.extend_from_slice(&enc_fmul_reg(3, 2, 2).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        bytes.extend_from_slice(&enc_exit().to_le_bytes());

        let cfg = build_cfg(&bytes);
        assert_eq!(cfg.blocks.len(), 3);
        assert!(matches!(
            cfg.blocks[0].branch,
            BranchKind::Conditional {
                target: 2,
                pred: Predicate { idx: 8, negate: false }
            }
        ));
        assert!(matches!(cfg.blocks[1].branch, BranchKind::FallThrough));
        let flow = cfg.blocks[0]
            .program
            .instructions
            .iter()
            .find(|i| matches!(i.op, Op::ISetPred { dest_p: 8, .. }))
            .expect("expected flow predicate for BRA CC.NEU");
        assert!(matches!(
            flow.op,
            Op::ISetPred { cmp: crate::ir::ICmp::Ne, src_pred: 0, src_pred_inv: false, src_a: Value::Inst(_), .. }
        ));
    }
}

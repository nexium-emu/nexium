use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

use super::maxwell::{cbuf_bank, float_fits_imm20, int_fits_imm20, nop, Word, FLOW_ALWAYS, PT, RZ};
use super::sm70::{Inst, Operand, Source};
use super::HEADER_SIZE;
use crate::opcodes::Opcode;

const INSTRUCTION_SIZE: usize = 16;
const SCRATCH_BASE: u8 = 240;
const SCRATCH_END: u8 = 255;
const TWO_PI: u32 = 0x40C9_0FDB;
const COPY_LIMIT: usize = 32;
const UNIFORM_BASE: u8 = 176;
const URZ: u8 = 63;
const UNIFORM_PREDICATES: u8 = 239;
const SPARE_PREDICATE: u8 = 6;
const LUT_A: u8 = 0xF0;
const LUT_B: u8 = 0xCC;
const LUT_C: u8 = 0xAA;
const MERGE_A_BY_B: u8 = (LUT_A & LUT_B) | (LUT_C & !LUT_B);
const SELECT_A_BY_C: u8 = (LUT_A & LUT_C) | (LUT_B & !LUT_C);
const WORKGROUP_BARRIER: u64 = 0xf0a8_1b80_0007_0000;
const COMPARE_LT: u64 = 1;
const COMPARE_EQ: u64 = 2;
const COMPARE_LE: u64 = 3;
const COMPARE_NE: u64 = 5;
const COMPARE_GE: u64 = 6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Truncated,
    MissingEnd,
    Unsupported { offset: usize, lo: u64, hi: u64 },
    Register { offset: usize, register: u8 },
    BranchTarget { offset: usize, target: i64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "program header truncated"),
            Self::MissingEnd => write!(f, "no end of program marker"),
            Self::Unsupported { offset, lo, hi } => write!(
                f,
                "unsupported instruction {:#05x} at {offset:#x} ({hi:016x}_{lo:016x})",
                lo & 0xFFF
            ),
            Self::Register { offset, register } => {
                write!(f, "register R{register} at {offset:#x} overlaps the scratch registers")
            }
            Self::BranchTarget { offset, target } => {
                write!(f, "branch at {offset:#x} targets {target:#x}")
            }
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Copy)]
enum Target {
    Instruction(usize),
    Landing(usize),
    Copy(usize),
}

enum Item {
    Word(Word),
    Branch { word: Word, target: Target },
    Indirect { word: Word, base: i64 },
}

pub(super) fn end_of_program(binary: &[u8]) -> usize {
    let mut offset = HEADER_SIZE;
    while offset + INSTRUCTION_SIZE <= binary.len() {
        let end = Inst::from_bytes(&binary[offset..]).is_end();
        offset += INSTRUCTION_SIZE;
        if end {
            return offset;
        }
    }
    binary.len()
}

pub(super) fn lower(code: &[u8]) -> Result<Vec<u8>, Error> {
    let mut instructions = Vec::new();
    for chunk in code.chunks_exact(INSTRUCTION_SIZE) {
        let inst = Inst::from_bytes(chunk);
        instructions.push(inst);
        if inst.is_end() {
            break;
        }
    }
    if !instructions.last().is_some_and(Inst::is_end) {
        return Err(Error::MissingEnd);
    }
    let targets: Vec<Option<usize>> = instructions
        .iter()
        .enumerate()
        .map(|(index, inst)| branch_index(index, inst, instructions.len()))
        .collect();
    let fused: HashMap<usize, usize> = (0..instructions.len().saturating_sub(1))
        .filter(|index| {
            let (exit, back) = (&instructions[*index], &instructions[index + 1]);
            exit.opcode() == 0x944
                && exit.field(82, 5) == 0x10
                && targets[*index] == Some(index + 2)
                && back.opcode() == 0x947
                && back.guard() == (PT, false)
                && back.pred(87, 90) == (PT, false)
                && back.field(82, 5) == 0
        })
        .filter_map(|index| Some((index, targets[index + 1]?)))
        .collect();
    let skipped: HashSet<usize> = fused.keys().map(|index| index + 1).collect();
    let calls: Vec<usize> = (0..instructions.len())
        .filter(|index| instructions[*index].opcode() == 0x944 && !fused.contains_key(index))
        .collect();
    let landings: HashSet<usize> = instructions
        .iter()
        .enumerate()
        .filter(|(index, inst)| inst.opcode() == 0x944 && !fused.contains_key(index))
        .filter_map(|(index, _)| targets[index])
        .collect();
    let (redirects, copies) = plan_copies(&instructions, &targets, &landings, &skipped);
    let mut lowering = Lowering {
        items: Vec::new(),
        starts: Vec::with_capacity(instructions.len()),
        landings: HashMap::new(),
        redirects,
        copies: vec![0; copies.len()],
        fused,
        skipped,
        count: instructions.len(),
        return_to: (calls.len() == 1).then(|| calls[0] + 1),
        offset: 0,
        inst: instructions[0],
        scratch: SCRATCH_BASE,
        native_limit: if instructions.iter().any(touches_uniforms) {
            UNIFORM_BASE
        } else {
            SCRATCH_BASE
        },
        uniform_fields: false,
        guard: None,
    };
    for (index, inst) in instructions.iter().enumerate() {
        if landings.contains(&index) {
            lowering.landings.insert(index, lowering.items.len());
            lowering.items.push(Item::Word(nop()));
        }
        lowering.starts.push(lowering.items.len());
        lowering.lower_at(index, *inst)?;
        for (copy, (_, range)) in copies.iter().enumerate().filter(|(_, (after, _))| *after == index) {
            lowering.copies[copy] = lowering.items.len();
            for source in range.clone() {
                lowering.lower_at(source, instructions[source])?;
            }
        }
    }
    Ok(lowering.assemble())
}

fn branch_index(index: usize, inst: &Inst, count: usize) -> Option<usize> {
    if !matches!(inst.opcode(), 0x944 | 0x947) {
        return None;
    }
    let target = inst.branch_target(index * INSTRUCTION_SIZE, 34, 48);
    let valid = target >= 0
        && (target as usize) < count * INSTRUCTION_SIZE
        && target as usize % INSTRUCTION_SIZE == 0;
    valid.then_some(target as usize / INSTRUCTION_SIZE)
}

fn touches_uniforms(inst: &Inst) -> bool {
    matches!(inst.form(), 6 | 7)
        || (0x080..0x0c0).contains(&inst.alu_opcode())
        || matches!(inst.alu_opcode(), 0x1c2 | 0x1c3)
        || matches!(inst.opcode(), 0x981 | 0x986)
}

fn uniform_datapath(inst: &Inst) -> bool {
    (0x080..0x0c0).contains(&inst.alu_opcode()) && inst.alu_opcode() != 0x086
}

fn terminates(inst: &Inst) -> bool {
    match inst.opcode() {
        0x947 | 0x94d | 0x950 => inst.guard() == (PT, false) && inst.pred(87, 90) == (PT, false),
        0x949 => true,
        _ => false,
    }
}

fn ends_block(inst: &Inst) -> bool {
    matches!(inst.opcode(), 0x944 | 0x947 | 0x949 | 0x94d | 0x950)
}

fn plan_copies(
    instructions: &[Inst],
    targets: &[Option<usize>],
    landings: &HashSet<usize>,
    skipped: &HashSet<usize>,
) -> (HashMap<usize, usize>, Vec<(usize, Range<usize>)>) {
    let count = instructions.len();
    let terminates = |index: usize| !skipped.contains(&index) && terminates(&instructions[index]);
    let mut leaders = BTreeSet::from([0]);
    for (index, inst) in instructions.iter().enumerate() {
        leaders.extend(targets[index]);
        if ends_block(inst) {
            leaders.insert(index + 1);
        }
    }
    let mut redirects = HashMap::new();
    let mut copies = Vec::new();
    for &start in &leaders {
        if start >= count || landings.contains(&start) {
            continue;
        }
        let mut end = start;
        let leaf = loop {
            let inst = &instructions[end];
            end += 1;
            if terminates(end - 1) {
                break matches!(inst.opcode(), 0x947 | 0x94d);
            }
            if ends_block(inst) || end >= count || end - start >= COPY_LIMIT {
                break false;
            }
        };
        if !leaf {
            continue;
        }
        let mut branches: Vec<usize> = (0..start)
            .filter(|index| targets[*index] == Some(start) && instructions[*index].opcode() == 0x947)
            .collect();
        let falls = start > 0 && !terminates(start - 1);
        if !falls && !branches.is_empty() {
            branches.remove(0);
        }
        for branch in branches {
            let Some(after) = (branch..start).find(|index| terminates(*index)) else {
                continue;
            };
            redirects.insert(branch, copies.len());
            copies.push((after, start..end));
        }
    }
    (redirects, copies)
}

fn swap_lut_bc(lut: u8) -> u8 {
    (0..8).fold(0, |out, index| {
        let source = (index & 4) | ((index & 2) >> 1) | ((index & 1) << 1);
        out | (((lut >> source) & 1) << index)
    })
}

fn offset_register(base: u8, index: u8) -> u8 {
    if base == RZ {
        RZ
    } else {
        base.saturating_add(index)
    }
}

fn contiguous(registers: &[u8]) -> bool {
    registers
        .iter()
        .enumerate()
        .all(|(index, register)| *register != RZ && u16::from(*register) == u16::from(registers[0]) + index as u16)
}

struct Lowering {
    items: Vec<Item>,
    starts: Vec<usize>,
    landings: HashMap<usize, usize>,
    redirects: HashMap<usize, usize>,
    copies: Vec<usize>,
    fused: HashMap<usize, usize>,
    skipped: HashSet<usize>,
    count: usize,
    return_to: Option<usize>,
    offset: usize,
    inst: Inst,
    scratch: u8,
    native_limit: u8,
    uniform_fields: bool,
    guard: Option<(u8, bool)>,
}

impl Lowering {
    fn lower_at(&mut self, index: usize, inst: Inst) -> Result<(), Error> {
        self.offset = index * INSTRUCTION_SIZE;
        self.inst = inst;
        self.scratch = SCRATCH_BASE;
        if self.skipped.contains(&index) {
            return Ok(());
        }
        if let Some(&head) = self.fused.get(&index) {
            let (pred, negate) = inst.guard();
            if pred != PT {
                self.items.push(Item::Branch {
                    word: Word::new(Opcode::BRA).set(0, 5, FLOW_ALWAYS).guard(pred, !negate),
                    target: Target::Instruction(head),
                });
            }
            return Ok(());
        }
        self.instruction()
    }

    fn unsupported<T>(&self) -> Result<T, Error> {
        Err(Error::Unsupported {
            offset: self.offset,
            lo: self.inst.lo,
            hi: self.inst.hi,
        })
    }

    fn emit(&mut self, word: Word) {
        let (pred, negate) = self.guard.unwrap_or(self.inst.guard());
        self.items.push(Item::Word(word.guard(pred, negate)));
    }

    fn always(&mut self, word: Word) {
        self.items.push(Item::Word(word.guard(PT, false)));
    }

    fn regs(&self, base: u8, count: u8) -> Result<u8, Error> {
        let end = u16::from(base) + u16::from(count);
        let uniform = self.uniform_fields && base >= UNIFORM_BASE && end <= u16::from(UNIFORM_BASE + URZ);
        if base == RZ || end <= u16::from(self.native_limit) || uniform {
            Ok(base)
        } else {
            Err(Error::Register {
                offset: self.offset,
                register: base,
            })
        }
    }

    fn reg(&self, register: u8) -> Result<u8, Error> {
        self.regs(register, 1)
    }

    fn ureg(&self, index: u8) -> Result<u8, Error> {
        match index {
            URZ => Ok(RZ),
            index if index < URZ && self.native_limit == UNIFORM_BASE => Ok(UNIFORM_BASE + index),
            _ => self.unsupported(),
        }
    }

    fn scratch(&mut self, count: u8) -> Result<u8, Error> {
        let base = self.scratch;
        if u16::from(base) + u16::from(count) > u16::from(SCRATCH_END) {
            return self.unsupported();
        }
        self.scratch += count;
        Ok(base)
    }

    fn cbuf(&self, offset: u16) -> Result<u16, Error> {
        if offset % 4 == 0 {
            Ok(offset)
        } else {
            self.unsupported()
        }
    }

    fn move_into(&mut self, dst: u8, operand: Operand) -> Result<(), Error> {
        let word = match operand {
            Operand::Reg(register) => Word::new(Opcode::MOV_reg).b(self.reg(register)?).set(39, 4, 0xF),
            Operand::Imm(value) => Word::new(Opcode::MOV32I).imm32(value).set(12, 4, 0xF),
            Operand::Cbuf { bank, offset } => Word::new(Opcode::MOV_cbuf)
                .cbuf(bank, self.cbuf(offset)?)
                .set(39, 4, 0xF),
            Operand::UReg(index) => Word::new(Opcode::MOV_reg).b(self.ureg(index)?).set(39, 4, 0xF),
        };
        self.emit(word.dst(dst));
        Ok(())
    }

    fn copy(&mut self, dst: u8, source: u8) {
        if dst != source && dst != RZ {
            self.emit(Word::new(Opcode::MOV_reg).dst(dst).b(source).set(39, 4, 0xF));
        }
    }

    fn register(&mut self, operand: Operand) -> Result<u8, Error> {
        match operand {
            Operand::Reg(register) => self.reg(register),
            Operand::UReg(index) => self.ureg(index),
            other => {
                let scratch = self.scratch(1)?;
                self.move_into(scratch, other)?;
                Ok(scratch)
            }
        }
    }

    fn float_mods(&mut self, register: u8, negate: bool, abs: bool) -> Result<u8, Error> {
        if !negate && !abs {
            return Ok(register);
        }
        let scratch = self.scratch(1)?;
        self.emit(
            Word::new(Opcode::FADD_reg)
                .dst(scratch)
                .a(register)
                .b(RZ)
                .flag(48, negate)
                .flag(46, abs),
        );
        Ok(scratch)
    }

    fn float_register(&mut self, source: Source) -> Result<u8, Error> {
        let register = self.register(source.operand)?;
        self.float_mods(register, source.neg, source.abs)
    }

    fn with_b(&mut self, forms: [Opcode; 3], operand: Operand, float: bool) -> Result<Word, Error> {
        let [reg, cbuf, imm] = forms;
        Ok(match operand {
            Operand::Reg(register) => Word::new(reg).b(self.reg(register)?),
            Operand::UReg(index) => Word::new(reg).b(self.ureg(index)?),
            Operand::Cbuf { bank, offset } => Word::new(cbuf).cbuf(bank, self.cbuf(offset)?),
            Operand::Imm(value) if float && float_fits_imm20(value) => Word::new(imm).float_imm20(value),
            Operand::Imm(value) if !float && int_fits_imm20(value) => Word::new(imm).int_imm20(value as i32),
            other => Word::new(reg).b(self.register(other)?),
        })
    }

    fn slot32(&self) -> Result<Source, Error> {
        match self.inst.slot32() {
            Some(source) => Ok(source),
            None => self.unsupported(),
        }
    }

    fn sources(&self) -> Result<(Source, Source), Error> {
        match self.inst.alu_sources() {
            Some(sources) => Ok(sources),
            None => self.unsupported(),
        }
    }

    fn target_index(&self, target: i64) -> Result<usize, Error> {
        let limit = (self.count * INSTRUCTION_SIZE) as i64;
        if target < 0 || target >= limit || target % INSTRUCTION_SIZE as i64 != 0 {
            return Err(Error::BranchTarget {
                offset: self.offset,
                target,
            });
        }
        Ok(target as usize / INSTRUCTION_SIZE)
    }

    fn assemble(&self) -> Vec<u8> {
        let byte_of = |index: usize| ((index + index / 3 + 1) * 8) as i64;
        let groups = self.items.len().div_ceil(3);
        let table = (groups * 32 + 8) as i64;
        let indirect = self.items.iter().any(|item| matches!(item, Item::Indirect { .. }));
        let mut out = Vec::with_capacity(groups * 32);
        for group in 0..groups {
            out.extend_from_slice(&0u64.to_le_bytes());
            for slot in 0..3 {
                let index = group * 3 + slot;
                let word = match self.items.get(index) {
                    None => nop(),
                    Some(Item::Word(word)) => *word,
                    Some(Item::Branch { word, target }) => {
                        let target = match *target {
                            Target::Instruction(target) => self.starts[target],
                            Target::Landing(target) => self.landings[&target],
                            Target::Copy(copy) => self.copies[copy],
                        };
                        word.offset24(byte_of(target) - (byte_of(index) + 8))
                    }
                    Some(Item::Indirect { word, base }) => {
                        word.offset24(table + base - (byte_of(index) + 8))
                    }
                };
                out.extend_from_slice(&word.0.to_le_bytes());
            }
        }
        if indirect {
            for pair in 0..self.count.div_ceil(2) {
                out.extend_from_slice(&0u64.to_le_bytes());
                for slot in 0..3 {
                    let index = pair * 2 + slot / 2;
                    let word = if slot == 1 || index >= self.count {
                        nop()
                    } else {
                        let position = table + (index * INSTRUCTION_SIZE) as i64;
                        Word::new(Opcode::BRA)
                            .set(0, 5, FLOW_ALWAYS)
                            .guard(PT, false)
                            .offset24(byte_of(self.starts[index]) - (position + 8))
                    };
                    out.extend_from_slice(&word.0.to_le_bytes());
                }
            }
        }
        out
    }

    fn instruction(&mut self) -> Result<(), Error> {
        if self.inst.guard() != (PT, false) && uniform_datapath(&self.inst) {
            return self.uniform_guarded();
        }
        match self.inst.opcode() {
            0x918 | 0x91a | 0x941 | 0x945 => Ok(()),
            0x944 => self.branch(0x10),
            0x947 => self.branch(0),
            0x949 => self.indirect(),
            0x950 => self.ret(),
            0x94d => self.exit(),
            0x95b => self.kill(),
            0x919 => self.s2r(),
            0x805 => self.cs2r(),
            0x321 => self.ald(),
            0x322 => self.ast(),
            0x326 => self.ipa(),
            0xb82 => self.ldc(),
            0x381 => self.ldg(None),
            0x981 => self.ldg(Some(self.inst.field(32, 6) as u8)),
            0x386 => self.stg(None),
            0x986 => self.stg(Some(self.inst.field(64, 6) as u8)),
            0xb60 => self.tex(),
            0xb69 => self.tmml(),
            0xb63 => self.tld4(),
            0xb66 => self.tld(),
            0xb6f => self.txq(),
            0xb97 => self.suld(),
            0xb9b => self.sust(),
            0x984 => self.shared(true),
            0x388 => self.shared(false),
            0xb1d => self.barrier(),
            0x992 => self.memory_barrier(false),
            0x9ab => self.memory_barrier(true),
            _ => self.alu(),
        }
    }

    fn alu(&mut self) -> Result<(), Error> {
        match self.inst.alu_opcode() {
            opcode @ (0x082 | 0x087 | 0x08c | 0x090 | 0x091 | 0x092 | 0x099 | 0x0a4) => {
                self.uniform_alu(opcode - 0x080)
            }
            0x0bd => self.uniform_alu(0x100),
            0x0be => self.uniform_alu(0x101),
            0x083 => self.up2ur(),
            0x006 => self.vote(false),
            0x086 => self.vote(true),
            0x09c => self.plop3(true),
            0x01c => self.plop3(false),
            0x013 => self.iabs(),
            0x016 => self.prmt(),
            0x01a => self.sgxt(),
            0x03e => self.f2fp(),
            0x100 => self.flo(),
            0x101 => self.brev(),
            0x109 => self.popc(),
            0x189 => self.shfl(),
            0x0b9 | 0x0bb => self.uldc(),
            0x1c2 => self.r2ur(),
            0x1c3 => self.s2ur(),
            0x002 => self.mov(),
            0x004 => self.r2p(),
            0x007 => self.sel(false),
            0x008 => self.sel(true),
            0x009 => self.fmnmx(),
            0x00a => self.fset(),
            0x00b => self.fsetp(),
            0x00c => self.isetp(),
            0x010 => self.iadd3(),
            0x011 => self.lea(),
            0x012 => self.lop3(),
            0x017 => self.imnmx(),
            0x019 => self.shf(),
            0x020 => self.fmul(),
            0x021 => self.fadd(),
            0x023 => self.ffma(),
            0x024 => self.imad(),
            0x045 | 0x106 => self.i2f(),
            0x105 => self.f2i(),
            0x107 => self.frnd(),
            0x108 => self.mufu(),
            0x148 => Ok(()),
            _ => self.unsupported(),
        }
    }

    fn uniform_alu(&mut self, vector_opcode: u16) -> Result<(), Error> {
        let inst = self.inst;
        if inst.guard() != (PT, false) {
            return self.unsupported();
        }
        let map = |index: u64| if index as u8 >= URZ { u64::from(RZ) } else { u64::from(UNIFORM_BASE) + index };
        let mut vector = inst.with_field(0, 9, u64::from(vector_opcode));
        let fields: &[u32] = if inst.form() == 1 { &[16, 24, 32, 64] } else { &[16, 24, 64] };
        for &start in fields {
            vector = vector.with_field(start, 8, map(inst.field(start, 8)));
        }
        self.inst = vector;
        self.uniform_fields = true;
        let lowered = match inst.alu_opcode() {
            0x08c => self.uisetp(),
            0x087 if inst.pred(87, 90).0 != PT => self.usel(),
            0x092 if inst.field(81, 3) != u64::from(PT) => self.ulop3_predicate(),
            _ => self.alu(),
        };
        self.uniform_fields = false;
        self.inst = inst;
        lowered
    }

    fn uniform_guarded(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let (predicate, negate) = inst.guard();
        if predicate == PT {
            return Ok(());
        }
        if self.guard.is_some() {
            return self.unsupported();
        }
        let saved = self.scratch(1)?;
        self.always(Word::new(Opcode::P2R_imm).dst(saved).a(RZ).int_imm20(1 << SPARE_PREDICATE));
        let shift = SPARE_PREDICATE - predicate;
        let source = if shift == 0 {
            UNIFORM_PREDICATES
        } else {
            let shifted = self.scratch(1)?;
            self.always(Word::new(Opcode::SHL_imm).dst(shifted).a(UNIFORM_PREDICATES).int_imm20(i32::from(shift)));
            shifted
        };
        self.always(Word::new(Opcode::R2P_imm).a(source).int_imm20(1 << SPARE_PREDICATE));
        self.inst = inst.with_field(12, 4, u64::from(PT));
        self.guard = Some((SPARE_PREDICATE, negate));
        let lowered = self.instruction();
        self.guard = None;
        self.inst = inst;
        self.always(Word::new(Opcode::R2P_imm).a(saved).int_imm20(1 << SPARE_PREDICATE));
        lowered
    }

    fn uniform_predicate_mask(&mut self, predicate: u8, negate: bool) -> Result<u8, Error> {
        let mask = self.scratch(1)?;
        if predicate == PT {
            self.emit(Word::new(Opcode::MOV32I).dst(mask).imm32(if negate { 0 } else { u32::MAX }).set(12, 4, 0xF));
            return Ok(mask);
        }
        self.emit(
            Word::new(Opcode::LOP3_imm)
                .dst(mask)
                .a(UNIFORM_PREDICATES)
                .int_imm20(1 << predicate)
                .c(RZ)
                .set(48, 8, u64::from(LUT_A & LUT_B)),
        );
        self.emit(
            Word::new(Opcode::ISET_reg)
                .dst(mask)
                .a(mask)
                .b(RZ)
                .set(39, 3, u64::from(PT))
                .set(49, 3, if negate { COMPARE_EQ } else { COMPARE_NE }),
        );
        Ok(mask)
    }

    fn predicate_mask(&mut self, predicate: u8, negate: bool) -> Result<u8, Error> {
        let mask = self.scratch(1)?;
        self.emit(
            Word::new(Opcode::SEL_imm)
                .dst(mask)
                .a(RZ)
                .int_imm20(-1)
                .set(39, 3, u64::from(predicate))
                .flag(42, !negate),
        );
        Ok(mask)
    }

    fn set_uniform_predicate(&mut self, predicate: u8, value: u8) {
        self.emit(
            Word::new(Opcode::LOP3_imm)
                .dst(UNIFORM_PREDICATES)
                .a(value)
                .int_imm20(1 << predicate)
                .c(UNIFORM_PREDICATES)
                .set(48, 8, u64::from(MERGE_A_BY_B)),
        );
    }

    fn logic(&mut self, dst: u8, sources: [u8; 3], lut: u8, predicate: u8) {
        self.emit(
            Word::new(Opcode::LOP3_reg)
                .dst(dst)
                .a(sources[0])
                .b(sources[1])
                .c(sources[2])
                .set(28, 8, u64::from(lut))
                .set(36, 2, 3)
                .set(48, 3, u64::from(predicate)),
        );
    }

    fn uisetp(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let b = self.slot32()?;
        let operation = inst.field(74, 2);
        if inst.bit(72) || b.modified() || operation == 3 {
            return self.unsupported();
        }
        let a = self.reg(inst.reg(24))?;
        let compare = self.scratch(1)?;
        let word = self.with_b([Opcode::ISET_reg, Opcode::ISET_cbuf, Opcode::ISET_imm], b.operand, false)?;
        self.emit(
            word.dst(compare)
                .a(a)
                .set(39, 3, u64::from(PT))
                .flag(48, inst.bit(73))
                .set(49, 3, inst.field(76, 3)),
        );
        let (combine, invert) = inst.pred(87, 90);
        let mask = self.uniform_predicate_mask(combine, invert)?;
        for (target, compare_lut) in [(inst.field(81, 3) as u8, LUT_A), (inst.field(84, 3) as u8, !LUT_A)] {
            if target == PT {
                continue;
            }
            let lut = match operation {
                0 => compare_lut & LUT_B,
                1 => compare_lut | LUT_B,
                _ => compare_lut ^ LUT_B,
            };
            let value = self.scratch(1)?;
            self.logic(value, [compare, mask, RZ], lut, PT);
            self.set_uniform_predicate(target, value);
        }
        Ok(())
    }

    fn usel(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let b = self.slot32()?;
        if b.modified() {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = self.register(inst.src0().operand)?;
        let b = self.register(b.operand)?;
        let (predicate, negate) = inst.pred(87, 90);
        let mask = self.uniform_predicate_mask(predicate, negate)?;
        self.logic(dst, [a, b, mask], SELECT_A_BY_C, PT);
        Ok(())
    }

    fn ulop3_predicate(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let spare = if self.guard.is_some() { SPARE_PREDICATE - 1 } else { SPARE_PREDICATE };
        let saved = self.scratch(1)?;
        self.always(Word::new(Opcode::P2R_imm).dst(saved).a(RZ).int_imm20(1 << spare));
        self.inst = inst.with_field(81, 3, u64::from(spare));
        let lowered = self.lop3();
        self.inst = inst;
        lowered?;
        let mask = self.predicate_mask(spare, false)?;
        self.set_uniform_predicate(inst.field(81, 3) as u8, mask);
        self.always(Word::new(Opcode::R2P_imm).a(saved).int_imm20(1 << spare));
        Ok(())
    }

    fn up2ur(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mask = inst.field(32, 32);
        if inst.form() != 4 || inst.field(64, 32) != 0x0800_0000 || mask > 0x7F {
            return self.unsupported();
        }
        let dst = self.ureg(inst.dst())?;
        let merged = self.ureg(inst.reg(24))?;
        self.emit(
            Word::new(Opcode::LOP3_imm)
                .dst(dst)
                .a(UNIFORM_PREDICATES)
                .int_imm20(mask as i32)
                .c(merged)
                .set(48, 8, u64::from(MERGE_A_BY_B)),
        );
        Ok(())
    }

    fn plop3(&mut self, uniform: bool) -> Result<(), Error> {
        let inst = self.inst;
        if uniform && inst.bit(67) {
            return self.unsupported();
        }
        let sources = [inst.pred(87, 90), inst.pred(77, 80), inst.pred(68, 71)];
        let mut masks = [RZ; 3];
        for (index, (predicate, negate)) in sources.into_iter().enumerate() {
            masks[index] = if uniform || (index == 2 && inst.bit(67)) {
                self.uniform_predicate_mask(predicate, negate)?
            } else {
                self.predicate_mask(predicate, negate)?
            };
        }
        let luts = [(inst.field(64, 3) | (inst.field(72, 5) << 3)) as u8, inst.field(16, 8) as u8];
        for (target, lut) in [inst.field(81, 3) as u8, inst.field(84, 3) as u8].into_iter().zip(luts) {
            if target == PT {
                continue;
            }
            if uniform {
                let value = self.scratch(1)?;
                self.logic(value, masks, lut, PT);
                self.set_uniform_predicate(target, value);
            } else {
                self.logic(RZ, masks, lut, target);
            }
        }
        Ok(())
    }

    fn unary_source(&mut self) -> Result<(u8, Source), Error> {
        let source = self.slot32()?;
        if source.abs {
            return self.unsupported();
        }
        Ok((self.register(source.operand)?, source))
    }

    fn iabs(&mut self) -> Result<(), Error> {
        let dst = self.reg(self.inst.dst())?;
        let (value, source) = self.unary_source()?;
        if source.neg {
            return self.unsupported();
        }
        let negated = self.scratch(1)?;
        self.emit(Word::new(Opcode::IADD3_reg).dst(negated).a(RZ).b(value).c(RZ).flag(50, true));
        self.emit(
            Word::new(Opcode::IMNMX_reg)
                .dst(dst)
                .a(value)
                .b(negated)
                .flag(48, true)
                .set(39, 3, u64::from(PT))
                .flag(42, true),
        );
        Ok(())
    }

    fn popc(&mut self) -> Result<(), Error> {
        let dst = self.reg(self.inst.dst())?;
        let (value, source) = self.unary_source()?;
        self.emit(Word::new(Opcode::POPC_reg).dst(dst).b(value).flag(40, source.neg));
        Ok(())
    }

    fn flo(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.field(81, 3) != u64::from(PT) {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let (value, source) = self.unary_source()?;
        self.emit(
            Word::new(Opcode::FLO_reg)
                .dst(dst)
                .b(value)
                .flag(40, source.neg)
                .flag(41, inst.bit(74))
                .flag(48, inst.bit(73)),
        );
        Ok(())
    }

    fn brev(&mut self) -> Result<(), Error> {
        let dst = self.reg(self.inst.dst())?;
        let (value, source) = self.unary_source()?;
        if source.neg {
            return self.unsupported();
        }
        self.emit(Word::new(Opcode::BFE_imm).dst(dst).a(value).int_imm20(32 << 8).flag(40, true));
        Ok(())
    }

    fn sgxt(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let bits = inst.field(32, 32);
        if inst.form() != 4 || inst.field(64, 32) != 0 || bits > 32 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let value = self.reg(inst.reg(24))?;
        self.emit(Word::new(Opcode::BFE_imm).dst(dst).a(value).int_imm20((bits << 8) as i32));
        Ok(())
    }

    fn prmt(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let selector = inst.field(32, 32);
        if inst.form() != 4 || inst.field(72, 24) != 0 || selector > 0xFFFF {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = self.reg(inst.reg(24))?;
        let c = self.reg(inst.reg(64))?;
        self.emit(Word::new(Opcode::PRMT_imm).dst(dst).a(a).set(20, 16, selector).c(c));
        Ok(())
    }

    fn f2fp(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.field(62, 2) != 0 || inst.field(72, 6) != 0 || inst.field(79, 13) != 0 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let low = self.slot32()?;
        let low = self.register(low.operand)?;
        let high = self.scratch(2)?;
        let to_half = |source: u8, dst: u8| Word::new(Opcode::F2F_reg).dst(dst).b(source).set(8, 2, 1).set(10, 2, 2);
        if inst.bit(78) {
            self.copy(high, self.reg(inst.reg(64))?);
        } else {
            self.emit(to_half(self.reg(inst.reg(24))?, high));
        }
        self.emit(to_half(low, high + 1));
        self.emit(
            Word::new(Opcode::LEA_lo_reg)
                .dst(dst)
                .a(high)
                .b(high + 1)
                .set(39, 5, 16)
                .set(48, 3, u64::from(PT)),
        );
        Ok(())
    }

    fn uldc(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let size = inst.field(73, 3);
        let indexed = inst.alu_opcode() == 0x0bb;
        if size > 5 || inst.bit(91) != indexed || inst.guard() != (PT, false) {
            return self.unsupported();
        }
        let dst = self.ureg(inst.dst())?;
        let index = if indexed { self.ureg(inst.reg(24))? } else { RZ };
        self.emit(
            Word::new(Opcode::LDC)
                .dst(dst)
                .a(index)
                .set(20, 16, inst.field(38, 16))
                .set(36, 5, u64::from(cbuf_bank(inst.field(54, 5) as u8)))
                .set(48, 3, size),
        );
        Ok(())
    }

    fn r2ur(&mut self) -> Result<(), Error> {
        let dst = self.ureg(self.inst.dst())?;
        let source = self.reg(self.inst.reg(24))?;
        self.copy(dst, source);
        Ok(())
    }

    fn s2ur(&mut self) -> Result<(), Error> {
        let dst = self.ureg(self.inst.dst())?;
        if dst != RZ {
            self.emit(Word::new(Opcode::S2R).dst(dst).set(20, 8, self.inst.field(72, 8)));
        }
        Ok(())
    }

    fn mov(&mut self) -> Result<(), Error> {
        let dst = self.reg(self.inst.dst())?;
        let source = self.slot32()?;
        if dst == RZ {
            return Ok(());
        }
        self.move_into(dst, source.operand)
    }

    fn r2p(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.form() != 4 || inst.field(72, 4) != 0 || inst.field(78, 2) != 0 {
            return self.unsupported();
        }
        let value = self.reg(inst.reg(24))?;
        let mask = inst.field(32, 32) as u32 & 0x7F;
        self.emit(
            Word::new(Opcode::R2P_imm)
                .a(value)
                .int_imm20(mask as i32)
                .set(41, 2, inst.field(76, 2)),
        );
        Ok(())
    }

    fn sel(&mut self, float: bool) -> Result<(), Error> {
        let inst = self.inst;
        let dst = self.reg(inst.dst())?;
        let a = inst.src0();
        let b = self.slot32()?;
        let a = if float {
            self.float_register(a)?
        } else {
            self.register(a.operand)?
        };
        let word = if float && b.modified() {
            Word::new(Opcode::SEL_reg).b(self.float_register(b)?)
        } else {
            self.with_b([Opcode::SEL_reg, Opcode::SEL_cbuf, Opcode::SEL_imm], b.operand, false)?
        };
        let (pred, negate) = inst.pred(87, 90);
        self.emit(word.dst(dst).a(a).set(39, 3, u64::from(pred)).flag(42, negate));
        Ok(())
    }

    fn fmnmx(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let dst = self.reg(inst.dst())?;
        let a = inst.src0();
        let b = self.slot32()?;
        let a_register = self.register(a.operand)?;
        let word = self.with_b([Opcode::FMNMX_reg, Opcode::FMNMX_cbuf, Opcode::FMNMX_imm], b.operand, true)?;
        let (pred, negate) = inst.pred(87, 90);
        self.emit(
            word.dst(dst)
                .a(a_register)
                .set(39, 3, u64::from(pred))
                .flag(42, negate)
                .flag(48, a.neg)
                .flag(46, a.abs)
                .flag(45, b.neg)
                .flag(49, b.abs)
                .flag(44, inst.bit(80)),
        );
        Ok(())
    }

    fn fsetp(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let a = inst.src0();
        let b = self.slot32()?;
        let a_register = self.register(a.operand)?;
        let word = self.with_b([Opcode::FSETP_reg, Opcode::FSETP_cbuf, Opcode::FSETP_imm], b.operand, true)?;
        let (pred, negate) = inst.pred(87, 90);
        self.emit(
            word.a(a_register)
                .set(3, 3, inst.field(81, 3))
                .set(0, 3, inst.field(84, 3))
                .set(39, 3, u64::from(pred))
                .flag(42, negate)
                .set(45, 2, inst.field(74, 2))
                .set(48, 4, inst.field(76, 4))
                .flag(43, a.neg)
                .flag(7, a.abs)
                .flag(6, b.neg)
                .flag(44, b.abs)
                .flag(47, inst.bit(80)),
        );
        Ok(())
    }

    fn fset(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.field(74, 2) == 3 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = inst.src0();
        let b = self.slot32()?;
        let a_register = self.register(a.operand)?;
        let word = self.with_b([Opcode::FSET_reg, Opcode::FSET_cbuf, Opcode::FSET_imm], b.operand, true)?;
        let (pred, negate) = inst.pred(87, 90);
        self.emit(
            word.dst(dst)
                .a(a_register)
                .set(39, 3, u64::from(pred))
                .flag(42, negate)
                .flag(43, a.neg)
                .flag(54, a.abs)
                .flag(53, b.neg)
                .flag(44, b.abs)
                .set(45, 2, inst.field(74, 2))
                .set(48, 4, inst.field(76, 4))
                .flag(52, true)
                .flag(55, inst.bit(80)),
        );
        Ok(())
    }

    fn isetp(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let b = self.slot32()?;
        if inst.bit(72) || b.modified() {
            return self.unsupported();
        }
        let a = self.reg(inst.reg(24))?;
        let word = self.with_b([Opcode::ISETP_reg, Opcode::ISETP_cbuf, Opcode::ISETP_imm], b.operand, false)?;
        let (pred, negate) = inst.pred(87, 90);
        self.emit(
            word.a(a)
                .set(3, 3, inst.field(81, 3))
                .set(0, 3, inst.field(84, 3))
                .set(39, 3, u64::from(pred))
                .flag(42, negate)
                .set(45, 2, inst.field(74, 2))
                .flag(48, inst.bit(73))
                .set(49, 3, inst.field(76, 3)),
        );
        Ok(())
    }

    fn add3(&mut self, dst: u8, a: Source, b: Source, c: Source) -> Result<(), Error> {
        let a_register = self.register(a.operand)?;
        let (wide, other) = if matches!(c.operand, Operand::Imm(_) | Operand::Cbuf { .. }) {
            (c, b)
        } else {
            (b, c)
        };
        let other_register = self.register(other.operand)?;
        let word = match wide.operand {
            Operand::Imm(value) => {
                let value = if wide.neg { value.wrapping_neg() } else { value };
                if int_fits_imm20(value) {
                    Word::new(Opcode::IADD3_imm).int_imm20(value as i32)
                } else {
                    Word::new(Opcode::IADD3_reg).b(self.register(Operand::Imm(value))?)
                }
            }
            Operand::Cbuf { bank, offset } => Word::new(Opcode::IADD3_cbuf)
                .cbuf(bank, self.cbuf(offset)?)
                .flag(50, wide.neg),
            operand => Word::new(Opcode::IADD3_reg)
                .b(self.register(operand)?)
                .flag(50, wide.neg),
        };
        self.emit(
            word.dst(dst)
                .a(a_register)
                .c(other_register)
                .flag(51, a.neg)
                .flag(49, other.neg),
        );
        Ok(())
    }

    fn iadd3(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let a = inst.src0();
        let (b, c) = self.sources()?;
        let c = Source { abs: false, ..c };
        if a.abs || b.abs {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let carry_in = [inst.pred(87, 90), inst.pred(77, 80)];
        let carry_out = [inst.field(81, 3) as u8, inst.field(84, 3) as u8];
        if inst.bit(74) {
            return self.iadd3_extended(dst, [a, b, c], carry_in, carry_out);
        }
        if carry_in != [(PT, true); 2] {
            return self.unsupported();
        }
        if carry_out == [PT, PT] {
            return self.add3(dst, a, b, c);
        }
        let third_zero = matches!(c.operand, Operand::Reg(RZ) | Operand::Imm(0));
        if carry_out[1] != PT || !third_zero || (a.neg && b.neg) {
            return self.unsupported();
        }
        let sum = self.scratch(1)?;
        self.add3(sum, a, b, c)?;
        let first = self.register(a.operand)?;
        let compare = match (a.neg, b.neg) {
            (false, false) => self.compare_into(carry_out[0], sum, Operand::Reg(first), COMPARE_LT),
            (false, true) => self.compare_into(carry_out[0], first, b.operand, COMPARE_GE),
            (true, _) => self.compare_into(carry_out[0], first, b.operand, COMPARE_LE),
        };
        compare?;
        self.copy(dst, sum);
        Ok(())
    }

    fn iadd3_extended(
        &mut self,
        dst: u8,
        sources: [Source; 3],
        carry_in: [(u8, bool); 2],
        carry_out: [u8; 2],
    ) -> Result<(), Error> {
        if carry_out != [PT, PT] {
            return self.unsupported();
        }
        let mut terms = [RZ; 3];
        for (index, source) in sources.into_iter().enumerate() {
            terms[index] = self.inverted(source)?;
        }
        let mut carries = [RZ; 2];
        for (index, carry) in carry_in.into_iter().enumerate() {
            if carry != (PT, true) {
                carries[index] = self.carry_mask(carry)?;
            }
        }
        let sum = self.scratch(1)?;
        self.emit(Word::new(Opcode::IADD3_reg).dst(sum).a(terms[0]).b(terms[1]).c(terms[2]));
        self.emit(
            Word::new(Opcode::IADD3_reg)
                .dst(dst)
                .a(sum)
                .b(carries[0])
                .c(carries[1])
                .flag(50, true)
                .flag(49, true),
        );
        Ok(())
    }

    fn inverted(&mut self, source: Source) -> Result<u8, Error> {
        if !source.neg {
            return self.register(source.operand);
        }
        if let Operand::Imm(value) = source.operand {
            return self.register(Operand::Imm(!value));
        }
        let value = self.register(source.operand)?;
        let inverted = self.scratch(1)?;
        self.logic(inverted, [value, RZ, RZ], !LUT_A, PT);
        Ok(inverted)
    }

    fn carry_mask(&mut self, (predicate, negate): (u8, bool)) -> Result<u8, Error> {
        if self.uniform_fields {
            self.uniform_predicate_mask(predicate, negate)
        } else {
            self.predicate_mask(predicate, negate)
        }
    }

    fn compare_into(&mut self, target: u8, a: u8, b: Operand, compare: u64) -> Result<(), Error> {
        if self.uniform_fields {
            let value = self.scratch(1)?;
            let word = self.with_b([Opcode::ISET_reg, Opcode::ISET_cbuf, Opcode::ISET_imm], b, false)?;
            self.emit(word.dst(value).a(a).set(39, 3, u64::from(PT)).set(49, 3, compare));
            self.set_uniform_predicate(target, value);
        } else {
            let word = self.with_b([Opcode::ISETP_reg, Opcode::ISETP_cbuf, Opcode::ISETP_imm], b, false)?;
            self.emit(
                word.a(a)
                    .set(3, 3, u64::from(target))
                    .set(0, 3, u64::from(PT))
                    .set(39, 3, u64::from(PT))
                    .set(49, 3, compare),
            );
        }
        Ok(())
    }

    fn lea(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let (b, _) = self.sources()?;
        let negated = matches!(inst.form(), 1 | 5 | 6) && b.neg;
        let carry_out = inst.field(81, 3) as u8;
        if inst.bit(80) {
            return self.lea_high(b, carry_out);
        }
        if inst.bit(74) || negated || (carry_out != PT && inst.bit(72)) {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = self.reg(inst.reg(24))?;
        let word = self.with_b([Opcode::LEA_lo_reg, Opcode::LEA_lo_cbuf, Opcode::LEA_lo_imm], b.operand, false)?;
        let word = word
            .a(a)
            .set(39, 5, inst.field(75, 5))
            .flag(45, inst.bit(72))
            .set(48, 3, u64::from(PT));
        if carry_out == PT {
            self.emit(word.dst(dst));
            return Ok(());
        }
        let sum = self.scratch(1)?;
        self.emit(word.dst(sum));
        self.compare_into(carry_out, sum, b.operand, COMPARE_LT)?;
        self.copy(dst, sum);
        Ok(())
    }

    fn lea_high(&mut self, b: Source, carry_out: u8) -> Result<(), Error> {
        let inst = self.inst;
        if carry_out != PT || inst.bit(72) || b.modified() {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let low = self.reg(inst.reg(24))?;
        let high = if inst.bit(73) {
            let sign = self.scratch(1)?;
            self.emit(Word::new(Opcode::SHR_imm).dst(sign).a(low).int_imm20(31).flag(48, true));
            sign
        } else {
            self.reg(inst.reg(64))?
        };
        let shift = inst.field(75, 5);
        let word = match b.operand {
            Operand::Cbuf { bank, offset } => Word::new(Opcode::LEA_hi_cbuf)
                .cbuf(bank, self.cbuf(offset)?)
                .set(51, 5, shift),
            operand => Word::new(Opcode::LEA_hi_reg).b(self.register(operand)?).set(28, 5, shift),
        };
        let word = word.a(low).c(high).set(48, 3, u64::from(PT));
        let carry = inst.pred(87, 90);
        if !inst.bit(74) || carry == (PT, true) {
            self.emit(word.dst(dst));
            return Ok(());
        }
        let sum = self.scratch(1)?;
        self.emit(word.dst(sum));
        let carry = self.carry_mask(carry)?;
        self.emit(Word::new(Opcode::IADD3_reg).dst(dst).a(sum).b(carry).c(RZ).flag(50, true));
        Ok(())
    }

    fn vote(&mut self, uniform: bool) -> Result<(), Error> {
        let inst = self.inst;
        let mode = inst.field(72, 2);
        if mode > 2 || inst.field(32, 32) != 0 || inst.field(64, 32) & !0x078e_0300 != 0 {
            return self.unsupported();
        }
        let dst = if uniform { self.ureg(inst.dst())? } else { self.reg(inst.dst())? };
        let (source, negate) = inst.pred(87, 90);
        let target = inst.field(81, 3) as u8;
        let vote = |target: u8| {
            Word::new(Opcode::VOTE)
                .dst(dst)
                .set(39, 3, u64::from(source))
                .flag(42, negate)
                .set(45, 3, u64::from(target))
                .set(48, 2, mode)
        };
        if !uniform || target == PT {
            self.emit(vote(target));
            return Ok(());
        }
        let spare = if inst.guard().0 == SPARE_PREDICATE { SPARE_PREDICATE - 1 } else { SPARE_PREDICATE };
        let saved = self.scratch(1)?;
        self.always(Word::new(Opcode::P2R_imm).dst(saved).a(RZ).int_imm20(1 << spare));
        self.emit(vote(spare));
        let mask = self.predicate_mask(spare, false)?;
        self.set_uniform_predicate(target, mask);
        self.always(Word::new(Opcode::R2P_imm).a(saved).int_imm20(1 << spare));
        Ok(())
    }

    fn lop3(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.bit(80) || inst.pred(87, 90) != (PT, true) {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let pred_dst = inst.field(81, 3) as u8;
        let a = self.reg(inst.reg(24))?;
        let (b, c) = self.sources()?;
        let mut lut = inst.field(72, 8) as u8;
        let (b, c) = if matches!(c.operand, Operand::Imm(_) | Operand::Cbuf { .. }) {
            lut = swap_lut_bc(lut);
            (c, b)
        } else {
            (b, c)
        };
        let c_register = self.register(c.operand)?;
        let word = match b.operand {
            Operand::Imm(value) if pred_dst == PT && int_fits_imm20(value) => {
                Word::new(Opcode::LOP3_imm).int_imm20(value as i32).set(48, 8, u64::from(lut))
            }
            Operand::Cbuf { bank, offset } if pred_dst == PT => Word::new(Opcode::LOP3_cbuf)
                .cbuf(bank, self.cbuf(offset)?)
                .set(48, 8, u64::from(lut)),
            operand => Word::new(Opcode::LOP3_reg)
                .b(self.register(operand)?)
                .set(28, 8, u64::from(lut))
                .set(36, 2, 3)
                .set(48, 3, u64::from(pred_dst)),
        };
        self.emit(word.dst(dst).a(a).c(c_register));
        Ok(())
    }

    fn imnmx(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let b = self.slot32()?;
        if b.modified() {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = self.reg(inst.reg(24))?;
        let word = self.with_b([Opcode::IMNMX_reg, Opcode::IMNMX_cbuf, Opcode::IMNMX_imm], b.operand, false)?;
        let (pred, negate) = inst.pred(87, 90);
        self.emit(
            word.dst(dst)
                .a(a)
                .flag(48, inst.bit(73))
                .set(39, 3, u64::from(pred))
                .flag(42, negate),
        );
        Ok(())
    }

    fn shf(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let (shift, high) = self.sources()?;
        let dst = self.reg(inst.dst())?;
        let low = self.reg(inst.reg(24))?;
        let kind = inst.field(73, 2);
        let signed = kind == 0 || kind == 2;
        let max_shift = match kind {
            0 => 3,
            1 => 2,
            _ => 0,
        };
        let word = match (inst.bit(76), inst.bit(80)) {
            (false, false) => self
                .with_b([Opcode::SHL_reg, Opcode::SHL_cbuf, Opcode::SHL_imm], shift.operand, false)?
                .a(low),
            (true, true) => {
                let high = self.register(high.operand)?;
                self.with_b([Opcode::SHR_reg, Opcode::SHR_cbuf, Opcode::SHR_imm], shift.operand, false)?
                    .a(high)
                    .flag(48, signed)
            }
            (right, _) => {
                let high = self.register(high.operand)?;
                let word = match shift.operand {
                    Operand::Imm(value) if value < 64 => Word::new(if right {
                        Opcode::SHF_r_imm
                    } else {
                        Opcode::SHF_l_imm
                    })
                    .int_imm20(value as i32),
                    operand => Word::new(if right {
                        Opcode::SHF_r_reg
                    } else {
                        Opcode::SHF_l_reg
                    })
                    .b(self.register(operand)?),
                };
                word.a(low)
                    .c(high)
                    .set(37, 2, max_shift)
                    .flag(50, inst.bit(75))
            }
        };
        self.emit(word.dst(dst));
        Ok(())
    }

    fn fmul(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let scale = match inst.field(84, 3) {
            0 => return self.unsupported(),
            scale => [0, 3, 2, 1, 0, 6, 5, 4][scale as usize],
        };
        let dst = self.reg(inst.dst())?;
        let mut a = inst.src0();
        let mut b = self.slot32()?;
        let negate = a.neg ^ b.neg;
        a.neg = false;
        b.neg = false;
        let a_register = self.float_register(a)?;
        let saturate = inst.bit(77);
        let word = match b.operand {
            Operand::Imm(value) => {
                let value = value ^ (u32::from(negate) << 31);
                if float_fits_imm20(value) {
                    Word::new(Opcode::FMUL_imm).float_imm20(value)
                } else if scale != 0 {
                    Word::new(Opcode::FMUL_reg).b(self.register(Operand::Imm(value))?)
                } else {
                    self.emit(
                        Word::new(Opcode::FMUL32I)
                            .dst(dst)
                            .a(a_register)
                            .imm32(value)
                            .flag(55, saturate)
                            .flag(53, inst.bit(80)),
                    );
                    return Ok(());
                }
            }
            Operand::Cbuf { bank, offset } if !b.abs => Word::new(Opcode::FMUL_cbuf)
                .cbuf(bank, self.cbuf(offset)?)
                .flag(48, negate),
            _ => Word::new(Opcode::FMUL_reg)
                .b(self.float_register(b)?)
                .flag(48, negate),
        };
        self.emit(
            word.dst(dst)
                .a(a_register)
                .flag(50, saturate)
                .set(39, 2, inst.field(78, 2))
                .set(41, 3, scale)
                .flag(44, inst.bit(80)),
        );
        Ok(())
    }

    fn fadd(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let dst = self.reg(inst.dst())?;
        let a = inst.src0();
        let b = self.slot32()?;
        let a_register = self.register(a.operand)?;
        let saturate = inst.bit(77);
        if let Operand::Imm(value) = b.operand {
            if !float_fits_imm20(value) && !saturate {
                self.emit(
                    Word::new(Opcode::FADD32I)
                        .dst(dst)
                        .a(a_register)
                        .imm32(value)
                        .flag(56, a.neg)
                        .flag(54, a.abs)
                        .flag(55, inst.bit(80)),
                );
                return Ok(());
            }
        }
        let word = self.with_b([Opcode::FADD_reg, Opcode::FADD_cbuf, Opcode::FADD_imm], b.operand, true)?;
        self.emit(
            word.dst(dst)
                .a(a_register)
                .flag(48, a.neg)
                .flag(46, a.abs)
                .flag(45, b.neg)
                .flag(49, b.abs)
                .flag(50, saturate)
                .set(39, 2, inst.field(78, 2))
                .flag(44, inst.bit(80)),
        );
        Ok(())
    }

    fn ffma(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let dst = self.reg(inst.dst())?;
        let mut a = inst.src0();
        let (mut b, mut c) = self.sources()?;
        let mut negate_product = a.neg ^ b.neg;
        let negate_c = c.neg;
        a.neg = false;
        b.neg = false;
        c.neg = false;
        if let Operand::Imm(value) = b.operand {
            b.operand = Operand::Imm(value ^ (u32::from(negate_product) << 31));
            negate_product = false;
        }
        let a_register = self.float_register(a)?;
        let saturate = inst.bit(77);
        if let (Operand::Imm(value), Operand::Reg(register)) = (b.operand, c.operand) {
            if !float_fits_imm20(value) && register == inst.dst() && !c.abs && !saturate {
                self.emit(
                    Word::new(Opcode::FFMA32I)
                        .dst(dst)
                        .a(a_register)
                        .imm32(value)
                        .flag(57, negate_c)
                        .flag(53, inst.bit(80)),
                );
                return Ok(());
            }
        }
        let word = match (b.operand, c.operand) {
            (Operand::Cbuf { bank, offset }, _) if !b.abs => Word::new(Opcode::FFMA_cr)
                .cbuf(bank, self.cbuf(offset)?)
                .c(self.float_register(c)?),
            (_, Operand::Cbuf { bank, offset }) if !c.abs => Word::new(Opcode::FFMA_rc)
                .c(self.float_register(b)?)
                .cbuf(bank, self.cbuf(offset)?),
            (Operand::Imm(value), _) if float_fits_imm20(value) => Word::new(Opcode::FFMA_imm)
                .float_imm20(value)
                .c(self.float_register(c)?),
            _ => Word::new(Opcode::FFMA_reg)
                .b(self.float_register(b)?)
                .c(self.float_register(c)?),
        };
        self.emit(
            word.dst(dst)
                .a(a_register)
                .flag(48, negate_product)
                .flag(49, negate_c)
                .flag(50, saturate)
                .set(51, 2, inst.field(78, 2))
                .flag(53, inst.bit(80)),
        );
        Ok(())
    }

    fn imad(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.field(81, 3) != u64::from(PT) {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let a = inst.src0();
        let (b, c) = self.sources()?;
        let zero = |source: &Source| matches!(source.operand, Operand::Reg(RZ) | Operand::Imm(0));
        if zero(&a) || zero(&b) {
            if c.neg {
                let zero = Source::plain(Operand::Reg(RZ));
                return self.add3(dst, zero, zero, c);
            }
            if dst == RZ {
                return Ok(());
            }
            return self.move_into(dst, c.operand);
        }
        if b.operand == Operand::Imm(1) {
            return self.add3(dst, a, Source::plain(Operand::Reg(RZ)), c);
        }
        let negate_product = a.neg ^ b.neg;
        let a_register = self.register(a.operand)?;
        if negate_product || c.neg {
            let product = self.scratch(1)?;
            let word = self.multiply(b.operand, Operand::Reg(RZ))?;
            self.emit(word.dst(product).a(a_register));
            let product = Source {
                operand: Operand::Reg(product),
                neg: negate_product,
                abs: false,
            };
            return self.add_scratch(dst, product, c);
        }
        let word = self.multiply(b.operand, c.operand)?;
        self.emit(word.dst(dst).a(a_register));
        Ok(())
    }

    fn add_scratch(&mut self, dst: u8, product: Source, c: Source) -> Result<(), Error> {
        let Operand::Reg(register) = product.operand else {
            return self.unsupported();
        };
        let c_register = self.register(c.operand)?;
        self.emit(
            Word::new(Opcode::IADD3_reg)
                .dst(dst)
                .a(register)
                .b(c_register)
                .c(RZ)
                .flag(51, product.neg)
                .flag(50, c.neg),
        );
        Ok(())
    }

    fn multiply(&mut self, b: Operand, c: Operand) -> Result<Word, Error> {
        Ok(match (b, c) {
            (Operand::Imm(value), _) if int_fits_imm20(value) && !matches!(c, Operand::Cbuf { .. }) => {
                Word::new(Opcode::IMAD_imm)
                    .int_imm20(value as i32)
                    .c(self.register(c)?)
            }
            (Operand::Cbuf { bank, offset }, _) => Word::new(Opcode::IMAD_cr)
                .cbuf(bank, self.cbuf(offset)?)
                .c(self.register(c)?),
            (_, Operand::Cbuf { bank, offset }) => Word::new(Opcode::IMAD_rc)
                .c(self.register(b)?)
                .cbuf(bank, self.cbuf(offset)?),
            _ => Word::new(Opcode::IMAD_reg)
                .b(self.register(b)?)
                .c(self.register(c)?),
        })
    }

    fn i2f(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let source = self.slot32()?;
        let size = inst.field(84, 2);
        let selector = if matches!(inst.form(), 2 | 4) { 0 } else { inst.field(60, 2) };
        let byte = match size {
            0 => selector,
            1 if selector < 2 => selector * 2,
            2 if selector == 0 => 0,
            _ => return self.unsupported(),
        };
        if inst.field(75, 2) != 2 || source.modified() {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let word = self.with_b([Opcode::I2F_reg, Opcode::I2F_cbuf, Opcode::I2F_imm], source.operand, false)?;
        self.emit(
            word.dst(dst)
                .set(8, 2, 2)
                .set(10, 2, size)
                .set(41, 2, byte)
                .flag(13, inst.bit(74))
                .set(39, 2, inst.field(78, 2)),
        );
        Ok(())
    }

    fn f2i(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let size = inst.field(75, 2);
        let narrow = size == 1 && !inst.bit(72) && inst.field(78, 2) == 0;
        if !(size == 2 || narrow) || inst.field(84, 2) != 2 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let source = self.slot32()?;
        let word = if source.modified() {
            Word::new(Opcode::F2I_reg).b(self.float_register(source)?)
        } else {
            self.with_b([Opcode::F2I_reg, Opcode::F2I_cbuf, Opcode::F2I_imm], source.operand, true)?
        };
        self.emit(
            word.dst(dst)
                .set(8, 2, size)
                .set(10, 2, 2)
                .flag(12, inst.bit(72))
                .set(39, 2, inst.field(78, 2))
                .flag(44, inst.bit(80)),
        );
        Ok(())
    }

    fn frnd(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.field(75, 2) != 2 || inst.field(84, 2) != 2 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let source = self.slot32()?;
        let word = self.with_b([Opcode::F2F_reg, Opcode::F2F_cbuf, Opcode::F2F_imm], source.operand, true)?;
        self.emit(
            word.dst(dst)
                .set(8, 2, 2)
                .set(10, 2, 2)
                .set(39, 4, 8 | inst.field(78, 2))
                .flag(44, inst.bit(80))
                .flag(45, source.neg)
                .flag(49, source.abs),
        );
        Ok(())
    }

    fn mufu(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let function = inst.field(74, 6);
        if inst.bit(73) || function > 8 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let source = self.slot32()?;
        let mut register = self.register(source.operand)?;
        let (mut negate, mut abs) = (source.neg, source.abs);
        if function <= 1 {
            let base = self.float_mods(register, false, abs)?;
            let scaled = self.scratch(1)?;
            self.emit(
                Word::new(Opcode::FMUL32I)
                    .dst(scaled)
                    .a(base)
                    .imm32(TWO_PI ^ (u32::from(negate) << 31)),
            );
            register = scaled;
            negate = false;
            abs = false;
        }
        self.emit(
            Word::new(Opcode::MUFU)
                .dst(dst)
                .a(register)
                .set(20, 4, function)
                .flag(48, negate)
                .flag(46, abs),
        );
        Ok(())
    }

    fn ald(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.bit(76) || inst.bit(77) || inst.bit(79) {
            return self.unsupported();
        }
        let count = inst.field(74, 2) as u8 + 1;
        let dst = self.regs(inst.dst(), count)?;
        let index = self.reg(inst.reg(24))?;
        let vertex = self.reg(inst.reg(32))?;
        self.emit(
            Word::new(Opcode::ALD)
                .dst(dst)
                .a(index)
                .set(20, 10, inst.field(40, 10))
                .c(vertex)
                .set(47, 2, u64::from(count - 1)),
        );
        Ok(())
    }

    fn ast(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.bit(76) || inst.bit(77) {
            return self.unsupported();
        }
        let count = inst.field(74, 2) as u8 + 1;
        let data = self.regs(inst.reg(32), count)?;
        let index = self.reg(inst.reg(24))?;
        let vertex = self.reg(inst.reg(64))?;
        self.emit(
            Word::new(Opcode::AST)
                .dst(data)
                .a(index)
                .set(20, 10, inst.field(40, 10))
                .c(vertex)
                .set(47, 2, u64::from(count - 1)),
        );
        Ok(())
    }

    fn ipa(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mode = match inst.field(78, 2) {
            0 => 0,
            1 => 2,
            _ => return self.unsupported(),
        };
        let location = inst.field(76, 2);
        if location == 3 || inst.field(81, 3) != u64::from(PT) || inst.reg(24) != 0 {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let offset = self.reg(inst.reg(32))?;
        self.emit(
            Word::new(Opcode::IPA)
                .dst(dst)
                .a(RZ)
                .b(RZ)
                .set(28, 10, inst.field(64, 8) * 4)
                .c(offset)
                .set(52, 2, location)
                .set(54, 2, mode),
        );
        Ok(())
    }

    fn ldc(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let size = inst.field(73, 3);
        let count = match size {
            0..=4 => 1,
            5 => 2,
            _ => return self.unsupported(),
        };
        if inst.bit(91) {
            return self.unsupported();
        }
        let dst = self.regs(inst.dst(), count)?;
        let index = self.reg(inst.reg(24))?;
        self.emit(
            Word::new(Opcode::LDC)
                .dst(dst)
                .a(index)
                .set(20, 16, inst.field(38, 16))
                .set(36, 5, u64::from(cbuf_bank(inst.field(54, 5) as u8)))
                .set(44, 2, inst.field(78, 2))
                .set(48, 3, size),
        );
        Ok(())
    }

    fn global_address(&mut self, uniform: Option<u8>) -> Result<u8, Error> {
        let inst = self.inst;
        if !inst.bit(72) {
            return self.unsupported();
        }
        let register = inst.reg(24);
        let base = match uniform {
            Some(index) => self.ureg(index)?,
            None => RZ,
        };
        if base == RZ {
            return self.regs(register, 2);
        }
        if register == RZ {
            return Ok(base);
        }
        let low = self.reg(register)?;
        let high = if inst.bit(90) { self.reg(register + 1)? } else { RZ };
        self.scratch += self.scratch & 1;
        let pair = self.scratch(2)?;
        self.emit(Word::new(Opcode::IADD_reg).dst(pair).a(base).b(low).flag(47, true));
        self.emit(Word::new(Opcode::IADD_reg).dst(pair + 1).a(base + 1).b(high).flag(43, true));
        Ok(pair)
    }

    fn global_count(&self) -> Result<u8, Error> {
        match self.inst.field(73, 3) {
            0..=4 => Ok(1),
            5 => Ok(2),
            6 => Ok(4),
            _ => self.unsupported(),
        }
    }

    fn ldg(&mut self, uniform: Option<u8>) -> Result<(), Error> {
        let inst = self.inst;
        if std::env::var_os("NEXIUM_AMPERE_NO_LDG").is_some() {
            return self.unsupported();
        }
        if (uniform.is_some() && inst.field(38, 2) != 0)
            || (uniform.is_none() && inst.reg(32) != 0)
            || inst.field(81, 3) != u64::from(PT)
        {
            return self.unsupported();
        }
        let dst = self.regs(inst.dst(), self.global_count()?)?;
        let address = self.global_address(uniform)?;
        self.emit(
            Word::new(Opcode::LDG)
                .dst(dst)
                .a(address)
                .offset24(inst.signed_offset(40, 24))
                .flag(45, true)
                .set(48, 3, inst.field(73, 3)),
        );
        Ok(())
    }

    fn stg(&mut self, uniform: Option<u8>) -> Result<(), Error> {
        let inst = self.inst;
        let data = self.regs(inst.reg(32), self.global_count()?)?;
        let address = self.global_address(uniform)?;
        self.emit(
            Word::new(Opcode::STG)
                .dst(data)
                .a(address)
                .offset24(inst.signed_offset(40, 24))
                .flag(45, true)
                .set(48, 3, inst.field(73, 3)),
        );
        Ok(())
    }

    fn s2r(&mut self) -> Result<(), Error> {
        let dst = self.reg(self.inst.dst())?;
        self.emit(Word::new(Opcode::S2R).dst(dst).set(20, 8, self.inst.field(72, 8)));
        Ok(())
    }

    fn cs2r(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let count = 1 + u8::from(inst.bit(80));
        let dst = self.regs(inst.dst(), count)?;
        if dst == RZ {
            return Ok(());
        }
        let register = inst.field(72, 8);
        for index in 0..count {
            let word = if register == 0xFF {
                Word::new(Opcode::MOV32I).imm32(0).set(12, 4, 0xF)
            } else {
                Word::new(Opcode::S2R).set(20, 8, register + u64::from(index))
            };
            self.emit(word.dst(dst + index));
        }
        Ok(())
    }

    fn branch(&mut self, modifiers: u64) -> Result<(), Error> {
        let inst = self.inst;
        let condition = inst.pred(87, 90);
        if (condition != (PT, false) && modifiers != 0) || inst.field(32, 2) != 0 || inst.field(82, 5) != modifiers {
            return self.unsupported();
        }
        let target = self.target_index(inst.branch_target(self.offset, 34, 48))?;
        let target = match self.redirects.get(&(self.offset / INSTRUCTION_SIZE)) {
            _ if modifiers != 0 => Target::Landing(target),
            Some(copy) => Target::Copy(*copy),
            None => Target::Instruction(target),
        };
        let (pred, negate) = self.second_condition(condition);
        self.items.push(Item::Branch {
            word: Word::new(Opcode::BRA).set(0, 5, FLOW_ALWAYS).guard(pred, negate),
            target,
        });
        Ok(())
    }

    fn second_condition(&mut self, condition: (u8, bool)) -> (u8, bool) {
        let (pred, negate) = self.inst.guard();
        if condition == (PT, false) {
            return (pred, negate);
        }
        if (pred, negate) != (PT, false) {
            self.items.push(Item::Branch {
                word: Word::new(Opcode::BRA).set(0, 5, FLOW_ALWAYS).guard(pred, !negate),
                target: Target::Instruction(self.offset / INSTRUCTION_SIZE + 1),
            });
        }
        condition
    }

    fn indirect(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        if inst.guard() != (PT, false)
            || inst.pred(87, 90) != (PT, false)
            || inst.field(32, 2) != 0
            || inst.field(82, 5) != 0
        {
            return self.unsupported();
        }
        let register = self.reg(inst.reg(24))?;
        self.items.push(Item::Indirect {
            word: Word::new(Opcode::BRX)
                .a(register)
                .set(0, 5, FLOW_ALWAYS)
                .guard(PT, false),
            base: inst.branch_target(self.offset, 34, 48),
        });
        Ok(())
    }

    fn ret(&mut self) -> Result<(), Error> {
        let Some(target) = self.return_to else {
            return self.unsupported();
        };
        if self.inst.field(82, 5) != 0x10 {
            return self.unsupported();
        }
        let (pred, negate) = self.second_condition(self.inst.pred(87, 90));
        self.items.push(Item::Branch {
            word: Word::new(Opcode::BRA).set(0, 5, FLOW_ALWAYS).guard(pred, negate),
            target: Target::Instruction(target),
        });
        Ok(())
    }

    fn exit(&mut self) -> Result<(), Error> {
        let (pred, negate) = self.second_condition(self.inst.pred(87, 90));
        self.items.push(Item::Word(Word::new(Opcode::EXIT).set(0, 5, FLOW_ALWAYS).guard(pred, negate)));
        Ok(())
    }

    fn kill(&mut self) -> Result<(), Error> {
        let (pred, negate) = self.second_condition(self.inst.pred(87, 90));
        self.items.push(Item::Word(Word::new(Opcode::KIL).set(0, 5, FLOW_ALWAYS).guard(pred, negate)));
        Ok(())
    }

    fn vector(&mut self, registers: &[u8]) -> Result<u8, Error> {
        if registers.is_empty() {
            return Ok(RZ);
        }
        if contiguous(registers) {
            return Ok(registers[0]);
        }
        let base = self.scratch(registers.len() as u8)?;
        for (index, register) in registers.iter().enumerate() {
            self.emit(
                Word::new(Opcode::MOV_reg)
                    .dst(base + index as u8)
                    .b(*register)
                    .set(39, 4, 0xF),
            );
        }
        Ok(base)
    }

    fn texture_components(&self, coordinates: u8, extras: u8) -> Result<Vec<u8>, Error> {
        let inst = self.inst;
        let total = coordinates + extras;
        let (first, second) = (inst.reg(24), inst.reg(32));
        let components: Vec<u8> = if inst.bit(60) {
            if total > 4 {
                return self.unsupported();
            }
            let split = total.div_ceil(2);
            (0..split)
                .map(|index| offset_register(first, index))
                .chain((0..total - split).map(|index| offset_register(second, index)))
                .collect()
        } else {
            (0..coordinates)
                .map(|index| offset_register(first, index))
                .chain((0..extras).map(|index| offset_register(second, index)))
                .collect()
        };
        for register in &components {
            self.reg(*register)?;
        }
        Ok(components)
    }

    fn texture_targets(&self, mask: u8) -> Result<Vec<u8>, Error> {
        let (dst0, dst1) = (self.inst.dst(), self.inst.reg(64));
        let targets: Vec<u8> = (0..mask.count_ones() as u8)
            .map(|index| {
                if index < 2 || dst1 == RZ {
                    offset_register(dst0, index)
                } else {
                    offset_register(dst1, index - 2)
                }
            })
            .collect();
        for register in &targets {
            self.reg(*register)?;
        }
        Ok(targets)
    }

    fn texture_handle(&mut self, extras: &[u8]) -> Result<u8, Error> {
        let inst = self.inst;
        let handle = self.scratch(1 + extras.len() as u8)?;
        self.emit(
            Word::new(Opcode::MOV_cbuf)
                .dst(handle)
                .cbuf(inst.field(54, 5) as u8, inst.field(40, 14) as u16 * 4)
                .set(39, 4, 0xF),
        );
        for (index, register) in extras.iter().enumerate() {
            self.copy(handle + 1 + index as u8, *register);
        }
        Ok(handle)
    }

    fn texture_result(&mut self, word: Word, targets: &[u8]) -> Result<(), Error> {
        let direct = contiguous(targets);
        let result = if direct { targets[0] } else { self.scratch(targets.len() as u8)? };
        self.emit(word.dst(result));
        if !direct {
            for (index, register) in targets.iter().enumerate() {
                self.copy(*register, result + index as u8);
            }
        }
        Ok(())
    }

    fn tmml(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mask = inst.field(72, 4) as u8;
        if inst.field(61, 3) != 1 || inst.field(76, 4) != 0 || inst.field(80, 10) != 0 || inst.bit(59) || mask == 0 {
            return self.unsupported();
        }
        let (first, second) = (inst.reg(24), inst.reg(32));
        let coordinates = [first, if second == RZ { offset_register(first, 1) } else { second }];
        for register in &coordinates {
            self.reg(*register)?;
        }
        let targets = self.texture_targets(mask)?;
        let coordinate_base = self.vector(&coordinates)?;
        let handle = self.texture_handle(&[])?;
        self.texture_result(
            Word::new(Opcode::TMML_b)
                .a(coordinate_base)
                .b(handle)
                .set(28, 3, 2)
                .set(31, 4, u64::from(mask)),
            &targets,
        )
    }

    fn tex(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let (tex_type, coordinates) = match inst.field(61, 3) {
            0 => (0, 1u8),
            1 => (2, 2),
            2 => (4, 3),
            3 => (6, 3),
            5 => (3, 3),
            7 => (7, 4),
            _ => return self.unsupported(),
        };
        let lod = inst.field(87, 3);
        let mask = inst.field(72, 4) as u8;
        if lod > 3 || inst.bit(59) || inst.bit(77) || inst.field(81, 3) != u64::from(PT) || mask == 0 {
            return self.unsupported();
        }
        let offsets = inst.bit(76);
        let compare = inst.bit(78);
        let extras = u8::from(lod >= 2) + u8::from(offsets) + u8::from(compare);
        let components = self.texture_components(coordinates, extras)?;
        let targets = self.texture_targets(mask)?;
        let coordinate_base = self.vector(&components[..usize::from(coordinates)])?;
        let handle = self.texture_handle(&components[usize::from(coordinates)..])?;
        self.texture_result(
            Word::new(Opcode::TEX_b)
                .a(coordinate_base)
                .b(handle)
                .set(28, 3, tex_type)
                .set(31, 4, u64::from(mask))
                .flag(36, offsets)
                .set(37, 3, lod)
                .flag(50, compare)
                .set(51, 3, u64::from(PT)),
            &targets,
        )
    }

    fn tld4(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mask = inst.field(72, 4) as u8;
        let offsets = inst.field(76, 2) as u8;
        if inst.field(61, 3) != 1
            || offsets == 3
            || inst.field(78, 2) != 0
            || inst.field(81, 3) != u64::from(PT)
            || inst.bit(59)
            || inst.bit(89)
            || mask == 0
        {
            return self.unsupported();
        }
        let components = self.texture_components(2, offsets)?;
        let targets = self.texture_targets(mask)?;
        let coordinate_base = self.vector(&components[..2])?;
        let handle = self.texture_handle(&components[2..])?;
        self.texture_result(
            Word::new(Opcode::TLD4_b)
                .a(coordinate_base)
                .b(handle)
                .set(28, 3, 2)
                .set(31, 4, u64::from(mask))
                .set(36, 2, u64::from(offsets))
                .set(38, 2, inst.field(87, 2))
                .set(51, 3, u64::from(PT)),
            &targets,
        )
    }

    fn tld(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let (tex_type, coordinates) = match inst.field(61, 3) {
            0 => (0, 1u8),
            1 => (2, 2),
            2 => (4, 3),
            5 => (3, 3),
            _ => return self.unsupported(),
        };
        let mask = inst.field(72, 4) as u8;
        if mask == 0 || inst.field(87, 3) != 1 || inst.field(76, 5) != 0 || inst.field(81, 3) != u64::from(PT) || inst.bit(59) {
            return self.unsupported();
        }
        let components = self.texture_components(coordinates, 0)?;
        let targets = self.texture_targets(mask)?;
        let coordinate_base = self.vector(&components)?;
        let handle = self.texture_handle(&[])?;
        self.texture_result(
            Word::new(Opcode::TLD_b)
                .a(coordinate_base)
                .b(handle)
                .set(28, 3, tex_type)
                .set(31, 4, u64::from(mask))
                .set(51, 3, u64::from(PT)),
            &targets,
        )
    }

    fn txq(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mask = inst.field(72, 4) as u8;
        if mask == 0 || inst.field(59, 5) != 0 || inst.field(76, 14) != 0 || inst.bit(91) || inst.reg(64) != RZ {
            return self.unsupported();
        }
        let level = self.reg(inst.reg(24))?;
        let targets = self.texture_targets(mask)?;
        let handle = self.texture_handle(&[level])?;
        self.texture_result(
            Word::new(Opcode::TXQ_b).a(handle).set(22, 3, 1).set(31, 4, u64::from(mask)),
            &targets,
        )
    }

    fn surface_coordinates(&self, dimension: u64) -> Result<u8, Error> {
        let count = match dimension {
            0 | 1 => 1,
            3 => 2,
            4 | 5 => 3,
            _ => return self.unsupported(),
        };
        self.regs(self.inst.reg(24), count)
    }

    fn suld(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let mask = inst.field(72, 4);
        let dimension = inst.field(61, 3);
        if mask == 0 || inst.field(59, 2) != 0 || inst.field(64, 32) & !0xF00 != 0x001e_0000 {
            return self.unsupported();
        }
        let coordinates = self.surface_coordinates(dimension)?;
        let dst = self.regs(inst.dst(), mask.count_ones() as u8)?;
        let handle = self.texture_handle(&[])?;
        self.emit(
            Word::new(Opcode::SULD)
                .dst(dst)
                .a(coordinates)
                .set(20, 4, mask)
                .set(33, 3, dimension)
                .c(handle),
        );
        Ok(())
    }

    fn sust(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let dimension = inst.field(61, 3);
        if inst.field(59, 2) != 0 || inst.field(64, 32) != 0x0010_0f00 {
            return self.unsupported();
        }
        let coordinates = self.surface_coordinates(dimension)?;
        let data = self.regs(inst.reg(32), 4)?;
        let handle = self.texture_handle(&[])?;
        self.emit(
            Word::new(Opcode::SUST)
                .dst(data)
                .a(coordinates)
                .set(20, 4, 0xF)
                .set(33, 3, dimension)
                .c(handle),
        );
        Ok(())
    }

    fn shared(&mut self, load: bool) -> Result<(), Error> {
        let inst = self.inst;
        let width = inst.field(73, 3);
        let stride = inst.field(78, 2);
        let uniform_base = load && inst.bit(91);
        let expected = (width << 9) | (stride << 14) | if uniform_base { 0x0800_0000 } else { 0 };
        if width > 6 || inst.field(64, 32) != expected || (load && !uniform_base && inst.reg(32) != 0) {
            return self.unsupported();
        }
        let count = match width {
            0..=4 => 1,
            5 => 2,
            _ => 4,
        };
        let data = self.regs(if load { inst.dst() } else { inst.reg(32) }, count)?;
        let mut address = self.reg(inst.reg(24))?;
        if stride != 0 && address != RZ {
            let scaled = self.scratch(1)?;
            self.emit(Word::new(Opcode::SHL_imm).dst(scaled).a(address).int_imm20(stride as i32 + 1));
            address = scaled;
        }
        let base = if uniform_base { self.ureg(inst.reg(32))? } else { RZ };
        if base != RZ {
            let sum = self.scratch(1)?;
            self.emit(Word::new(Opcode::IADD3_reg).dst(sum).a(address).b(base).c(RZ));
            address = sum;
        }
        self.emit(
            Word::new(if load { Opcode::LDS } else { Opcode::STS })
                .dst(data)
                .a(address)
                .set(20, 24, inst.field(40, 24))
                .set(48, 3, width),
        );
        Ok(())
    }

    fn shfl(&mut self) -> Result<(), Error> {
        let inst = self.inst;
        let form = inst.form();
        let lane_immediate = matches!(form, 4 | 7);
        let mask_immediate = matches!(form, 2 | 7);
        if !matches!(form, 1 | 2 | 4 | 7)
            || inst.field(60, 4) != 0
            || inst.field(72, 9) != 0
            || inst.field(84, 12) != 0
            || (lane_immediate && inst.reg(32) != 0)
        {
            return self.unsupported();
        }
        let dst = self.reg(inst.dst())?;
        let source = self.reg(inst.reg(24))?;
        let word = Word::new(Opcode::SHFL)
            .dst(dst)
            .a(source)
            .set(30, 2, inst.field(58, 2))
            .set(48, 3, inst.field(81, 3));
        let word = if lane_immediate {
            word.set(20, 5, inst.field(53, 5)).flag(28, true)
        } else {
            word.b(self.reg(inst.reg(32))?)
        };
        let word = if mask_immediate {
            word.set(34, 13, inst.field(40, 13)).flag(29, true)
        } else {
            word.c(self.reg(inst.reg(64))?)
        };
        self.emit(word);
        Ok(())
    }

    fn barrier(&mut self) -> Result<(), Error> {
        if self.inst.field(32, 64) != 0 {
            return self.unsupported();
        }
        self.emit(Word(WORKGROUP_BARRIER));
        Ok(())
    }

    fn memory_barrier(&mut self, error: bool) -> Result<(), Error> {
        let scope = if error { 1 } else { self.inst.field(76, 3) };
        let expected = if error { 0 } else { scope << 12 };
        if self.inst.field(32, 32) != 0 || self.inst.field(64, 32) != expected {
            return self.unsupported();
        }
        self.emit(Word::new(Opcode::MEMBAR).set(8, 2, u64::from(scope != 0)));
        Ok(())
    }

}

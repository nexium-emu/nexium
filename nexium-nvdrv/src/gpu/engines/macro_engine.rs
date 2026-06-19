use std::collections::{HashMap, HashSet};

const NUM_REGS: usize = 8;
pub const MACRO_REGISTERS_START: u32 = 0xE00;
const NUM_MACRO_POSITIONS: usize = 0x80;

fn macro_hash(code: &[u32]) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    let mut h: u64 = 0;
    for &v in code {
        let mut k = v as u64;
        k = k.wrapping_mul(M);
        k ^= k >> 47;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
        h = h.wrapping_add(0xe654_6b64);
    }
    h
}

const REG_GLOBAL_BASE_VERTEX: u32 = 0x50D;
const REG_GLOBAL_BASE_INSTANCE: u32 = 0x50E;
const REG_VERTEX_FIRST: u32 = 0x35D;
const REG_VERTEX_COUNT: u32 = 0x35E;
const REG_DRAW_BEGIN: u32 = 0x586;
const REG_INDEX_FIRST: u32 = 0x5F7;
const REG_INDEX_COUNT: u32 = 0x5F8;
const REG_CB_SIZE: u32 = 0x8E0;
const REG_CB_ADDR_HI: u32 = 0x8E1;
const REG_CB_ADDR_LO: u32 = 0x8E2;
const REG_CB_OFFSET: u32 = 0x8E3;
const REG_UPLOAD_LINE_LENGTH: u32 = 0x60;
const REG_UPLOAD_LINE_COUNT: u32 = 0x61;
const REG_UPLOAD_DST_HI: u32 = 0x62;
const REG_UPLOAD_DST_LO: u32 = 0x63;
const REG_LAUNCH_DMA: u32 = 0x6C;

fn hle_macro(hash: u64, params: &[u32]) -> Option<Vec<(u32, u32)>> {
    let p = |i: usize| params.get(i).copied().unwrap_or(0);
    let mut w: Vec<(u32, u32)> = Vec::new();
    match hash {
        0x0D61_FC9F_AAC9_FCAD | 0x8A4D_173E_B99A_8603 => {
            let topology = p(0) & 0xFFFF;
            let vertex_count = p(1);
            let vertex_first = p(3);
            if hash == 0x8A4D_173E_B99A_8603 {
                w.push((REG_GLOBAL_BASE_INSTANCE, p(4)));
            }
            w.push((REG_DRAW_BEGIN, topology));
            w.push((REG_VERTEX_FIRST, vertex_first));
            w.push((REG_VERTEX_COUNT, vertex_count));
        }
        0x771B_B18C_6244_4DA0 | 0x0217_9201_0048_8FF7 => {
            let topology = p(0) & 0xFFFF;
            let index_count = p(1);
            let index_first = p(3);
            let base_vertex = p(4);
            w.push((REG_GLOBAL_BASE_VERTEX, base_vertex));
            if hash == 0x0217_9201_0048_8FF7 {
                w.push((REG_GLOBAL_BASE_INSTANCE, p(5)));
            }
            w.push((REG_DRAW_BEGIN, topology));
            w.push((REG_INDEX_FIRST, index_first));
            w.push((REG_INDEX_COUNT, index_count));
        }
        0x6C97_861D_891E_DF7E | 0xD246_FDDF_3A61_73D7 => {
            let size = if hash == 0x6C97_861D_891E_DF7E { 0x5F00 } else { 0x7000 };
            w.push((REG_CB_SIZE, size));
            w.push((REG_CB_ADDR_HI, p(0)));
            w.push((REG_CB_ADDR_LO, p(1)));
            w.push((REG_CB_OFFSET, 0));
        }
        0xEE4D_0004_BEC8_ECF4 => {
            w.push((REG_UPLOAD_LINE_LENGTH, p(2)));
            w.push((REG_UPLOAD_LINE_COUNT, 1));
            w.push((REG_UPLOAD_DST_HI, p(0)));
            w.push((REG_UPLOAD_DST_LO, p(1)));
            w.push((REG_LAUNCH_DMA, 0x1011));
        }
        _ => return None,
    }
    Some(w)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    Alu,
    AddImmediate,
    ExtractInsert,
    ExtractShiftLeftImmediate,
    ExtractShiftLeftRegister,
    Read,
    Unused,
    Branch,
}

impl Operation {
    fn from_u32(v: u32) -> Self {
        match v & 0x7 {
            0 => Self::Alu,
            1 => Self::AddImmediate,
            2 => Self::ExtractInsert,
            3 => Self::ExtractShiftLeftImmediate,
            4 => Self::ExtractShiftLeftRegister,
            5 => Self::Read,
            6 => Self::Unused,
            _ => Self::Branch,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ResultOperation {
    IgnoreAndFetch,
    Move,
    MoveAndSetMethod,
    FetchAndSend,
    MoveAndSend,
    FetchAndSetMethod,
    MoveAndSetMethodFetchAndSend,
    MoveAndSetMethodSend,
}

impl ResultOperation {
    fn from_u32(v: u32) -> Self {
        match v & 0x7 {
            0 => Self::IgnoreAndFetch,
            1 => Self::Move,
            2 => Self::MoveAndSetMethod,
            3 => Self::FetchAndSend,
            4 => Self::MoveAndSend,
            5 => Self::FetchAndSetMethod,
            6 => Self::MoveAndSetMethodFetchAndSend,
            _ => Self::MoveAndSetMethodSend,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AluOp {
    Add, AddWithCarry, Subtract, SubtractWithBorrow,
    Xor, Or, And, AndNot, Nand, Unknown,
}

impl AluOp {
    fn from_u32(v: u32) -> Self {
        match v & 0x1F {
            0 => Self::Add, 1 => Self::AddWithCarry,
            2 => Self::Subtract, 3 => Self::SubtractWithBorrow,
            8 => Self::Xor, 9 => Self::Or,
            10 => Self::And, 11 => Self::AndNot, 12 => Self::Nand,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy)]
struct Opcode(u32);

impl Opcode {
    fn operation(self) -> Operation { Operation::from_u32(self.0 & 0x7) }
    fn result_operation(self) -> ResultOperation { ResultOperation::from_u32((self.0 >> 4) & 0x7) }
    fn branch_zero(self) -> bool { (self.0 >> 4) & 0x1 == 0 }
    fn branch_annul(self) -> bool { (self.0 >> 5) & 0x1 != 0 }
    fn is_exit(self) -> bool { (self.0 >> 7) & 0x1 != 0 }
    fn dst(self) -> u32 { (self.0 >> 8) & 0x7 }
    fn src_a(self) -> u32 { (self.0 >> 11) & 0x7 }
    fn src_b(self) -> u32 { (self.0 >> 14) & 0x7 }
    fn immediate(self) -> i32 {
        let raw = self.0 >> 14;
        if raw & 0x2_0000 != 0 { (raw | 0xFFFC_0000) as i32 } else { raw as i32 }
    }
    fn alu_op(self) -> AluOp { AluOp::from_u32((self.0 >> 17) & 0x1F) }
    fn bf_src_bit(self) -> u32 { (self.0 >> 17) & 0x1F }
    fn bf_size(self) -> u32 { (self.0 >> 22) & 0x1F }
    fn bf_dst_bit(self) -> u32 { (self.0 >> 27) & 0x1F }
    fn bitfield_mask(self) -> u32 {
        let size = self.bf_size();
        if size >= 32 { 0xFFFF_FFFF } else { (1u32 << size) - 1 }
    }
    fn branch_target(self) -> i32 { self.immediate().wrapping_mul(4) }
}

#[derive(Default)]
pub struct MacroOutput {
    pub writes: Vec<(u32, u32)>,
}

pub struct MacroEngine {
    uploaded_code: HashMap<u32, Vec<u32>>,
    compiled: HashMap<u32, Vec<u32>>,
    macro_positions: [u32; NUM_MACRO_POSITIONS],
    instruction_ptr: u32,
    start_address_ptr: u32,
    executing_macro: u32,
    pending_params: Vec<u32>,
    seen_hashes: HashSet<u64>,
}

impl MacroEngine {
    pub fn new() -> Self {
        Self {
            uploaded_code: HashMap::new(),
            compiled: HashMap::new(),
            macro_positions: [0; NUM_MACRO_POSITIONS],
            instruction_ptr: 0,
            start_address_ptr: 0,
            executing_macro: 0,
            pending_params: Vec::new(),
            seen_hashes: HashSet::new(),
        }
    }

    pub fn set_instruction_ptr(&mut self, value: u32) {
        self.instruction_ptr = value;
        self.uploaded_code.remove(&value);
        self.compiled.clear();
    }

    pub fn upload_instruction(&mut self, word: u32) {
        self.uploaded_code.entry(self.instruction_ptr).or_default().push(word);
    }

    pub fn set_start_address_ptr(&mut self, value: u32) {
        self.start_address_ptr = value & 0x7F;
    }

    pub fn bind_macro_entry(&mut self, offset: u32) {
        let slot = (self.start_address_ptr as usize) % NUM_MACRO_POSITIONS;
        self.macro_positions[slot] = offset;
        self.start_address_ptr = self.start_address_ptr.wrapping_add(1) & 0x7F;
        self.compiled.remove(&offset);
    }

    pub fn on_macro_method(
        &mut self,
        method: u32,
        arg: u32,
        is_last_call: bool,
        reg_reader: &dyn Fn(u32) -> u32,
    ) -> Option<MacroOutput> {
        if self.executing_macro == 0 {
            self.executing_macro = method & !1;
            self.pending_params.clear();
        }
        self.pending_params.push(arg);
        if !is_last_call {
            return None;
        }
        let trigger = self.executing_macro;
        self.executing_macro = 0;
        let entry = ((trigger - MACRO_REGISTERS_START) >> 1) as usize % NUM_MACRO_POSITIONS;
        let offset = self.macro_positions[entry];
        let params = std::mem::take(&mut self.pending_params);
        let code = self.resolve_code(offset);
        if code.is_empty() {
            log::trace!("MME: trigger {:#x} entry={} offset={} - no code", trigger, entry, offset);
            return Some(MacroOutput::default());
        }
        let hash = macro_hash(&code);
        let hle = hle_macro(hash, &params);
        if self.seen_hashes.insert(hash) {
            log::info!("MME: macro entry={} offset={} hash={:#018x} len={} params={} hle={} code={:08x?}",
                entry, offset, hash, code.len(), params.len(), hle.is_some(),
                &code[..code.len().min(28)]);
        }
        if let Some(writes) = hle {
            return Some(MacroOutput { writes });
        }
        let mut interp = Interpreter::new(&code, &params, reg_reader);
        interp.run();
        Some(MacroOutput { writes: interp.writes })
    }

    fn resolve_code(&mut self, offset: u32) -> Vec<u32> {
        if let Some(c) = self.compiled.get(&offset) {
            return c.clone();
        }
        if let Some(c) = self.uploaded_code.get(&offset) {
            let v = c.clone();
            self.compiled.insert(offset, v.clone());
            return v;
        }
        let mut found: Option<Vec<u32>> = None;
        for (&base, code) in self.uploaded_code.iter() {
            if offset >= base && (offset - base) < code.len() as u32 {
                let start = (offset - base) as usize;
                found = Some(code[start..].to_vec());
                break;
            }
        }
        let v = found.unwrap_or_default();
        if !v.is_empty() {
            self.compiled.insert(offset, v.clone());
        }
        v
    }
}

impl Default for MacroEngine {
    fn default() -> Self { Self::new() }
}

struct Interpreter<'a> {
    code: &'a [u32],
    params: &'a [u32],
    next_param: usize,
    registers: [u32; NUM_REGS],
    pc: usize,
    delayed_pc: Option<usize>,
    method_address: u32,
    carry: bool,
    writes: Vec<(u32, u32)>,
    written: HashMap<u32, u32>,
    reg_reader: &'a dyn Fn(u32) -> u32,
    steps_remaining: u32,
}

impl<'a> Interpreter<'a> {
    fn new(code: &'a [u32], params: &'a [u32], reg_reader: &'a dyn Fn(u32) -> u32) -> Self {
        let mut regs = [0u32; NUM_REGS];
        if !params.is_empty() {
            regs[1] = params[0];
        }
        Self {
            code, params, next_param: 1,
            registers: regs, pc: 0, delayed_pc: None,
            method_address: 0, carry: false,
            writes: Vec::new(), written: HashMap::new(), reg_reader,
            steps_remaining: 8192,
        }
    }

    fn run(&mut self) {
        let mut exited = false;
        while self.steps_remaining > 0 {
            self.steps_remaining -= 1;
            if !self.step(false) {
                exited = true;
                break;
            }
        }
        if !exited {
            log::warn!("MME: macro hit step cap (produced {} writes) — discarding as runaway", self.writes.len());
            self.writes.clear();
        }
    }

    fn fetch_opcode(&self) -> Opcode {
        Opcode(*self.code.get(self.pc / 4).unwrap_or(&0))
    }

    fn read_reg(&self, id: u32) -> u32 {
        if id == 0 { 0 } else { self.registers[id as usize & 7] }
    }

    fn write_reg(&mut self, id: u32, value: u32) {
        if id == 0 { return; }
        self.registers[id as usize & 7] = value;
    }

    fn send(&mut self, value: u32) {
        let address = self.method_address & 0xFFF;
        let increment = (self.method_address >> 12) & 0x3F;
        self.writes.push((address, value));
        self.written.insert(address, value);
        if address == 0x8C4 {
            self.written.insert(0xD00, 1);
        }
        let next = (address.wrapping_add(increment)) & 0xFFF;
        self.method_address = (self.method_address & !0xFFF) | next;
    }

    fn fetch_param(&mut self) -> u32 {
        if self.next_param >= self.params.len() {
            return 0;
        }
        let v = self.params[self.next_param];
        self.next_param += 1;
        v
    }

    fn alu(&mut self, op: AluOp, a: u32, b: u32) -> u32 {
        match op {
            AluOp::Add => {
                let r = (a as u64) + (b as u64);
                self.carry = r > 0xFFFF_FFFF;
                r as u32
            }
            AluOp::AddWithCarry => {
                let r = (a as u64) + (b as u64) + if self.carry { 1 } else { 0 };
                self.carry = r > 0xFFFF_FFFF;
                r as u32
            }
            AluOp::Subtract => {
                let r = (a as u64).wrapping_sub(b as u64);
                self.carry = r < 0x1_0000_0000_u64;
                r as u32
            }
            AluOp::SubtractWithBorrow => {
                let r = (a as u64).wrapping_sub(b as u64).wrapping_sub(if self.carry { 0 } else { 1 });
                self.carry = r < 0x1_0000_0000_u64;
                r as u32
            }
            AluOp::Xor => a ^ b,
            AluOp::Or => a | b,
            AluOp::And => a & b,
            AluOp::AndNot => a & !b,
            AluOp::Nand => !(a & b),
            AluOp::Unknown => 0,
        }
    }

    fn process_result(&mut self, op: ResultOperation, reg: u32, result: u32) {
        match op {
            ResultOperation::IgnoreAndFetch => {
                let p = self.fetch_param();
                self.write_reg(reg, p);
            }
            ResultOperation::Move => self.write_reg(reg, result),
            ResultOperation::MoveAndSetMethod => {
                self.write_reg(reg, result);
                self.method_address = result;
            }
            ResultOperation::FetchAndSend => {
                let p = self.fetch_param();
                self.write_reg(reg, p);
                self.send(result);
            }
            ResultOperation::MoveAndSend => {
                self.write_reg(reg, result);
                self.send(result);
            }
            ResultOperation::FetchAndSetMethod => {
                let p = self.fetch_param();
                self.write_reg(reg, p);
                self.method_address = result;
            }
            ResultOperation::MoveAndSetMethodFetchAndSend => {
                self.write_reg(reg, result);
                self.method_address = result;
                let p = self.fetch_param();
                self.send(p);
            }
            ResultOperation::MoveAndSetMethodSend => {
                self.write_reg(reg, result);
                self.method_address = result;
                self.send((result >> 12) & 0x3F);
            }
        }
    }

    fn step(&mut self, is_delay_slot: bool) -> bool {
        if self.pc >= self.code.len() * 4 {
            return false;
        }
        let base = self.pc;
        let op = self.fetch_opcode();
        self.pc += 4;
        if let Some(d) = self.delayed_pc.take() {
            let _ = is_delay_slot;
            self.pc = d;
        }
        match op.operation() {
            Operation::Alu => {
                let a = self.read_reg(op.src_a());
                let b = self.read_reg(op.src_b());
                let r = self.alu(op.alu_op(), a, b);
                self.process_result(op.result_operation(), op.dst(), r);
            }
            Operation::AddImmediate => {
                let a = self.read_reg(op.src_a());
                let r = a.wrapping_add(op.immediate() as u32);
                self.process_result(op.result_operation(), op.dst(), r);
            }
            Operation::ExtractInsert => {
                let mut dst = self.read_reg(op.src_a());
                let mut src = self.read_reg(op.src_b());
                let mask = op.bitfield_mask();
                let src_bit = op.bf_src_bit();
                let dst_bit = op.bf_dst_bit();
                src = (src >> src_bit) & mask;
                dst &= !(mask << dst_bit);
                dst |= src << dst_bit;
                self.process_result(op.result_operation(), op.dst(), dst);
            }
            Operation::ExtractShiftLeftImmediate => {
                let dst = self.read_reg(op.src_a());
                let src = self.read_reg(op.src_b());
                let mask = op.bitfield_mask();
                let r = ((src >> dst) & mask) << op.bf_dst_bit();
                self.process_result(op.result_operation(), op.dst(), r);
            }
            Operation::ExtractShiftLeftRegister => {
                let dst = self.read_reg(op.src_a());
                let src = self.read_reg(op.src_b());
                let mask = op.bitfield_mask();
                let r = ((src >> op.bf_src_bit()) & mask) << dst;
                self.process_result(op.result_operation(), op.dst(), r);
            }
            Operation::Read => {
                let addr = self.read_reg(op.src_a()).wrapping_add(op.immediate() as u32);
                let v = self.written.get(&addr).copied().unwrap_or_else(|| (self.reg_reader)(addr));
                self.process_result(op.result_operation(), op.dst(), v);
            }
            Operation::Branch => {
                let v = self.read_reg(op.src_a());
                let taken = if op.branch_zero() { v == 0 } else { v != 0 };
                if taken {
                    let target = (base as i64).wrapping_add(op.branch_target() as i64);
                    let target = if target < 0 { 0 } else { target as usize };
                    if op.branch_annul() {
                        self.pc = target;
                        return true;
                    }
                    self.delayed_pc = Some(target);
                    return self.step(true);
                }
            }
            Operation::Unused => {}
        }
        if op.is_exit() && !is_delay_slot {
            self.step(true);
            return false;
        }
        true
    }
}

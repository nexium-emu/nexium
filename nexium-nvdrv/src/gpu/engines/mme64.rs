use nexium_common::fast_hash::FastMap as HashMap;
use std::collections::VecDeque;

use crate::gpu::GuestMemoryAccess;

pub const DATA_RAM_WORDS: usize = 0xC00;
const CODE_WORDS: usize = 0x3000;
const START_ENTRIES: usize = 0x80;
const WORDS_PER_INSTRUCTION: usize = 3;
const STEP_LIMIT: u32 = 1 << 20;
const METHODS: usize = 0x1000;
const DMA_WORD_LIMIT: u32 = 0x1000;
const PAGE: u64 = 0x1000;

pub const METHOD_INSTRUCTION_POINTER: u32 = 0x45;
pub const METHOD_INSTRUCTION: u32 = 0x46;
pub const METHOD_START_POINTER: u32 = 0x47;
pub const METHOD_START: u32 = 0x48;
pub const METHOD_MEM_ADDRESS_A: u32 = 0x154;
pub const METHOD_MEM_ADDRESS_B: u32 = 0x155;
pub const METHOD_DATA_RAM_ADDRESS: u32 = 0x156;
pub const METHOD_DMA_READ: u32 = 0x157;
pub const METHOD_DMA_READ_FIFOED: u32 = 0x158;
pub const METHOD_DMA_SYNC: u32 = 0x15C;
pub const MACRO_METHODS_START: u32 = 0xE00;
pub const METHOD_SCRATCH: u32 = 0xD00;
pub const METHOD_FALCON_3D: u32 = 0x8C4;
pub const METHOD_FALCON_COMPUTE: u32 = 0x144;
const FIRMWARE_DONE: u32 = 1;

const REG_ZERO: u8 = 24;
const REG_IMM: u8 = 25;
const REG_IMMPAIR: u8 = 26;
const REG_IMM32: u8 = 27;
const REG_LOAD0: u8 = 28;
const REG_LOAD1: u8 = 29;

const OUT_NONE: u8 = 0;
const OUT_ALU0: u8 = 1;
const OUT_ALU1: u8 = 2;
const OUT_LOAD0: u8 = 3;
const OUT_LOAD1: u8 = 4;
const OUT_IMM0: u8 = 5;
const OUT_IMM1: u8 = 6;
const OUT_IMMHIGH0: u8 = 8;
const OUT_IMMHIGH1: u8 = 9;
const OUT_IMM32: u8 = 10;

const PREDICATES: [&[u8; 4]; 16] = [
    b"UUUU", b"TTTT", b"FFFF", b"TTUU", b"FFUU", b"TFUU", b"TUUU", b"FUUU", b"UUTT", b"UUTF", b"UUTU",
    b"UUFT", b"UUFF", b"UUFU", b"UUUT", b"UUUF",
];

const BARRIER_X: u32 = 0x1000;
const BARRIER_Y: u32 = 1;
const STOP_OFFSET: i32 = -0x1000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Op {
    #[default]
    Add,
    Addc,
    Sub,
    Subb,
    Mul,
    Mulh,
    Mulu,
    Extended,
    Clz,
    Sll,
    Srl,
    Sra,
    And,
    Nand,
    Or,
    Xor,
    Merge,
    Slt,
    Sltu,
    Sle,
    Sleu,
    Seq,
    State,
    Loop,
    Jal,
    Blt,
    Bltu,
    Ble,
    Bleu,
    Beq,
    Dread,
    Dwrite,
}

impl Op {
    fn from_bits(bits: u32) -> Self {
        const OPS: [Op; 32] = [
            Op::Add,
            Op::Addc,
            Op::Sub,
            Op::Subb,
            Op::Mul,
            Op::Mulh,
            Op::Mulu,
            Op::Extended,
            Op::Clz,
            Op::Sll,
            Op::Srl,
            Op::Sra,
            Op::And,
            Op::Nand,
            Op::Or,
            Op::Xor,
            Op::Merge,
            Op::Slt,
            Op::Sltu,
            Op::Sle,
            Op::Sleu,
            Op::Seq,
            Op::State,
            Op::Loop,
            Op::Jal,
            Op::Blt,
            Op::Bltu,
            Op::Ble,
            Op::Bleu,
            Op::Beq,
            Op::Dread,
            Op::Dwrite,
        ];
        OPS[(bits & 31) as usize]
    }

    fn compare(self, x: u32, y: u32) -> bool {
        match self {
            Op::Blt | Op::Slt => (x as i32) < (y as i32),
            Op::Bltu | Op::Sltu => x < y,
            Op::Ble | Op::Sle => (x as i32) <= (y as i32),
            Op::Bleu | Op::Sleu => x <= y,
            _ => x == y,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Alu {
    op: Op,
    dst: u8,
    src: [u8; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Out {
    method: u8,
    emit: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Instruction {
    end_next: bool,
    predicate_mode: u8,
    predicate: u8,
    imm: [u16; 2],
    alu: [Alu; 2],
    out: [Out; 2],
    loads: u8,
}

impl Instruction {
    fn decode(words: [u32; 3]) -> Self {
        let bits = u128::from(words[2]) | (u128::from(words[1]) << 32) | (u128::from(words[0]) << 64);
        let field = |low: u32, high: u32| ((bits >> low) & ((1u128 << (high - low + 1)) - 1)) as u32;
        let mut instruction = Self {
            end_next: field(0, 0) != 0,
            predicate_mode: field(1, 4) as u8,
            predicate: field(5, 9) as u8,
            imm: [field(30, 45) as u16, field(66, 81) as u16],
            alu: [
                Alu {
                    op: Op::from_bits(field(10, 14)),
                    dst: field(15, 19) as u8,
                    src: [field(20, 24) as u8, field(25, 29) as u8],
                },
                Alu {
                    op: Op::from_bits(field(46, 50)),
                    dst: field(51, 55) as u8,
                    src: [field(56, 60) as u8, field(61, 65) as u8],
                },
            ],
            out: [
                Out {
                    method: field(82, 84) as u8,
                    emit: field(85, 88) as u8,
                },
                Out {
                    method: field(89, 91) as u8,
                    emit: field(92, 95) as u8,
                },
            ],
            loads: 0,
        };
        instruction.loads = if instruction.reads(REG_LOAD1, OUT_LOAD1) {
            2
        } else {
            u8::from(instruction.reads(REG_LOAD0, OUT_LOAD0))
        };
        instruction
    }

    fn reads(&self, register: u8, out: u8) -> bool {
        self.predicate == register
            || self.alu.iter().any(|alu| alu.src.contains(&register))
            || self.out.iter().any(|slot| slot.method == out || slot.emit == out)
    }

    fn imm32(&self) -> u32 {
        (u32::from(self.imm[0]) << 16) | u32::from(self.imm[1])
    }
}

#[derive(Default)]
struct Dma {
    address_high: u32,
    address_low: u32,
    ram_address: u32,
    fifo_words: u32,
}

impl Dma {
    fn address(&self) -> u64 {
        (u64::from(self.address_high & 0x1FF_FFFF) << 32) | u64::from(self.address_low)
    }
}

struct Program {
    code: Vec<u32>,
    decoded: Vec<Instruction>,
    stale: Option<(usize, usize)>,
    upload: usize,
    starts: [u32; START_ENTRIES],
    start_pointer: usize,
}

impl Program {
    fn new() -> Self {
        Self {
            code: vec![0; CODE_WORDS],
            decoded: vec![Instruction::decode([0; 3]); CODE_WORDS / WORDS_PER_INSTRUCTION],
            stale: None,
            upload: 0,
            starts: [0; START_ENTRIES],
            start_pointer: 0,
        }
    }

    fn upload(&mut self, method: u32, value: u32) -> bool {
        match method {
            METHOD_INSTRUCTION_POINTER => self.upload = value as usize * WORDS_PER_INSTRUCTION,
            METHOD_INSTRUCTION => {
                if let Some(word) = self.code.get_mut(self.upload) {
                    *word = value;
                    let index = self.upload / WORDS_PER_INSTRUCTION;
                    self.stale = Some(self.stale.map_or((index, index), |(first, last)| (first.min(index), last.max(index))));
                }
                self.upload += 1;
            }
            METHOD_START_POINTER => self.start_pointer = value as usize % START_ENTRIES,
            METHOD_START => {
                self.starts[self.start_pointer] = value;
                self.start_pointer = (self.start_pointer + 1) % START_ENTRIES;
            }
            _ => return false,
        }
        true
    }

    fn refresh(&mut self) {
        let Some((first, last)) = self.stale.take() else {
            return;
        };
        for index in first..=last.min(self.decoded.len() - 1) {
            let base = index * WORDS_PER_INSTRUCTION;
            self.decoded[index] = Instruction::decode([self.code[base], self.code[base + 1], self.code[base + 2]]);
        }
    }
}

pub struct Mme64 {
    compute: Vec<u32>,
    graphics: Program,
    compute_programs: HashMap<u32, Program>,
    compute_active: bool,
    data_ram: HashMap<u32, Box<[u32]>>,
    channel: u32,
    dma: Dma,
    trigger: Option<u32>,
    params: Vec<u32>,
    memory: Option<GuestMemoryAccess>,
    runaways: u32,
    spare_writes: Vec<(u32, u32)>,
    overlay: Vec<(u32, u32)>,
    generation: u32,
    falcon: u32,
}

pub struct MacroRun {
    pub writes: Vec<(u32, u32)>,
    pub entry: u32,
    pub steps: u32,
    pub params: Vec<u32>,
}

impl Mme64 {
    pub(crate) fn new(memory: Option<GuestMemoryAccess>) -> Self {
        Self {
            compute: vec![0; MACRO_METHODS_START as usize],
            graphics: Program::new(),
            compute_programs: HashMap::default(),
            compute_active: false,
            data_ram: HashMap::default(),
            channel: 0,
            dma: Dma::default(),
            trigger: None,
            params: Vec::new(),
            memory,
            runaways: 0,
            spare_writes: Vec::new(),
            overlay: vec![(0, 0); METHODS],
            generation: 0,
            falcon: METHOD_FALCON_3D,
        }
    }

    fn overlay_read(&self, method: u32) -> Option<u32> {
        self.overlay
            .get(method as usize)
            .and_then(|&(generation, value)| (generation == self.generation).then_some(value))
    }

    fn overlay_write(&mut self, method: u32, value: u32) {
        let generation = self.generation;
        if let Some(slot) = self.overlay.get_mut(method as usize) {
            *slot = (generation, value);
        }
    }

    pub fn note_compute(&mut self, method: u32, value: u32) {
        if (METHOD_INSTRUCTION_POINTER..=METHOD_START).contains(&method) {
            self.compute_programs.entry(self.channel).or_insert_with(Program::new).upload(method, value);
        }
        if let Some(slot) = self.compute.get_mut(method as usize) {
            *slot = value;
        }
        if method == METHOD_FALCON_COMPUTE {
            self.compute[METHOD_SCRATCH as usize] = FIRMWARE_DONE;
        }
    }

    pub fn call_compute(&mut self, method: u32, value: u32, last: bool) -> Option<MacroRun> {
        let shadow = std::mem::take(&mut self.compute);
        self.falcon = METHOD_FALCON_COMPUTE;
        self.compute_active = self.compute_programs.contains_key(&self.channel);
        let run = self.call(method, std::slice::from_ref(&value), last, &|index| {
            shadow.get(index as usize).copied().unwrap_or(0)
        });
        self.compute_active = false;
        self.falcon = METHOD_FALCON_3D;
        self.compute = shadow;
        if let Some(run) = &run {
            for &(method, value) in &run.writes {
                self.note_compute(method, value);
            }
        }
        run
    }

    pub fn recycle(&mut self, mut writes: Vec<(u32, u32)>) {
        writes.clear();
        if writes.capacity() > self.spare_writes.capacity() {
            self.spare_writes = writes;
        }
    }

    fn program(&self) -> &Program {
        self.compute_programs
            .get(&self.channel)
            .filter(|_| self.compute_active)
            .unwrap_or(&self.graphics)
    }

    fn program_mut(&mut self) -> &mut Program {
        match self.compute_programs.get_mut(&self.channel) {
            Some(program) if self.compute_active => program,
            _ => &mut self.graphics,
        }
    }

    pub fn set_channel(&mut self, channel: u32) {
        self.channel = channel;
    }

    pub fn handles(method: u32) -> bool {
        matches!(
            method,
            METHOD_INSTRUCTION_POINTER
                | METHOD_INSTRUCTION
                | METHOD_START_POINTER
                | METHOD_START
                | METHOD_MEM_ADDRESS_A
                | METHOD_MEM_ADDRESS_B
                | METHOD_DATA_RAM_ADDRESS
                | METHOD_DMA_READ
                | METHOD_DMA_READ_FIFOED
                | METHOD_DMA_SYNC
        )
    }

    pub fn method(&mut self, method: u32, value: u32) {
        if !self.graphics.upload(method, value) {
            self.dma_method(method, value);
        }
    }

    fn data_ram(&mut self) -> &mut [u32] {
        self.data_ram
            .entry(self.channel)
            .or_insert_with(|| vec![0; DATA_RAM_WORDS].into_boxed_slice())
    }

    pub fn data_ram_word(&mut self, index: u32) -> u32 {
        self.data_ram().get(index as usize).copied().unwrap_or(0)
    }

    fn read_gpu(&self, gpu_va: u64, words: u32) -> Vec<u32> {
        let mut bytes = vec![0u8; words as usize * 4];
        if let Some(memory) = &self.memory {
            if let Some(reader) = memory.reader() {
                let mappings = memory.read_mappings();
                let mut done = 0usize;
                while done < bytes.len() {
                    let address = gpu_va + done as u64;
                    let chunk = ((PAGE - (address % PAGE)) as usize).min(bytes.len() - done);
                    if let Some(cpu) = mappings.cpu_address_for(address) {
                        reader(cpu, &mut bytes[done..done + chunk]);
                    }
                    done += chunk;
                }
            }
        }
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    fn dma_method(&mut self, method: u32, value: u32) -> bool {
        match method {
            METHOD_MEM_ADDRESS_A => self.dma.address_high = value,
            METHOD_MEM_ADDRESS_B => self.dma.address_low = value,
            METHOD_DATA_RAM_ADDRESS => self.dma.ram_address = value,
            METHOD_DMA_READ => {
                let words = self.read_gpu(self.dma.address(), value.min(DMA_WORD_LIMIT));
                let start = self.dma.ram_address as usize;
                let ram = self.data_ram();
                for (index, word) in words.into_iter().enumerate() {
                    if let Some(slot) = ram.get_mut(start + index) {
                        *slot = word;
                    }
                }
            }
            METHOD_DMA_READ_FIFOED => self.dma.fifo_words = value.min(DMA_WORD_LIMIT),
            METHOD_DMA_SYNC => {
                let index = self.dma.ram_address as usize;
                if let Some(slot) = self.data_ram().get_mut(index) {
                    *slot = value;
                }
            }
            _ => return false,
        }
        true
    }

    fn barrier(&mut self, fifo: &mut VecDeque<u32>) {
        if self.dma.fifo_words == 0 {
            return;
        }
        let words = self.read_gpu(self.dma.address(), self.dma.fifo_words);
        self.dma.fifo_words = 0;
        fifo.extend(words);
    }

    pub fn call(&mut self, method: u32, args: &[u32], last: bool, state: &dyn Fn(u32) -> u32) -> Option<MacroRun> {
        let trigger = *self.trigger.get_or_insert(method & !1);
        self.params.extend_from_slice(args);
        if !last {
            return None;
        }
        self.trigger = None;
        let entry = ((trigger - MACRO_METHODS_START) >> 1) % START_ENTRIES as u32;
        self.program_mut().refresh();
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.overlay.fill((0, 0));
            self.generation = 1;
        }
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let traced = if *TRACE.get_or_init(|| std::env::var_os("NEXIUM_MME_TRACE").is_some()) {
            self.params.clone()
        } else {
            Vec::new()
        };
        let params: VecDeque<u32> = std::mem::take(&mut self.params).into();
        let start = self.program().starts[entry as usize] as usize;
        let mut run = Run::new(start, params, std::mem::take(&mut self.spare_writes));
        run.execute(self, state);
        self.params = Vec::from(std::mem::take(&mut run.params));
        self.params.clear();
        if run.steps >= STEP_LIMIT {
            self.runaways += 1;
            if self.runaways <= 4 {
                log::warn!(
                    "[mme64] macro {entry} at {start:#x} ran past {STEP_LIMIT} steps with {} writes, ip={:#x}:\n{}",
                    run.writes.len(),
                    run.ip,
                    self.listing(entry, 96)
                );
            }
        }
        Some(MacroRun {
            writes: run.writes,
            entry,
            steps: run.steps,
            params: traced,
        })
    }

    pub fn listing(&mut self, entry: u32, limit: usize) -> String {
        self.program_mut().refresh();
        let program = self.program();
        let start = program.starts[entry as usize % START_ENTRIES] as usize;
        let mut lines = Vec::new();
        for ip in start..(start + limit).min(program.decoded.len()) {
            let instruction = program.decoded[ip];
            lines.push(format!(
                "{ip:#05x}: end={} pred={}:{} imm=[{:#x},{:#x}] alu0={:?} {}<-{},{} alu1={:?} {}<-{},{} out0={}/{} out1={}/{}",
                u8::from(instruction.end_next),
                instruction.predicate_mode,
                instruction.predicate,
                instruction.imm[0],
                instruction.imm[1],
                instruction.alu[0].op,
                instruction.alu[0].dst,
                instruction.alu[0].src[0],
                instruction.alu[0].src[1],
                instruction.alu[1].op,
                instruction.alu[1].dst,
                instruction.alu[1].src[0],
                instruction.alu[1].src[1],
                instruction.out[0].method,
                instruction.out[0].emit,
                instruction.out[1].method,
                instruction.out[1].emit,
            ));
            if instruction.end_next && lines.len() > 1 {
                if let Some(next) = program.decoded.get(ip + 1) {
                    lines.push(format!("{:#05x}: (delay) {:?}", ip + 1, next));
                }
                break;
            }
        }
        lines.join("\n")
    }

    fn instruction(&self, ip: usize) -> Instruction {
        self.program().decoded.get(ip).copied().unwrap_or(Instruction {
            end_next: true,
            predicate: REG_ZERO,
            ..Instruction::default()
        })
    }
}

struct Run {
    ip: usize,
    next_ip: usize,
    registers: [u32; 24],
    load: [u32; 2],
    results: [u32; 2],
    carry: u32,
    method: u32,
    increment: u32,
    loop_count: u32,
    loop_start: usize,
    loop_end: usize,
    stop: bool,
    params: VecDeque<u32>,
    writes: Vec<(u32, u32)>,
    steps: u32,
}

impl Run {
    fn new(start: usize, params: VecDeque<u32>, writes: Vec<(u32, u32)>) -> Self {
        Self {
            ip: start,
            next_ip: start,
            registers: [0; 24],
            load: [0; 2],
            results: [0; 2],
            carry: 0,
            method: 0,
            increment: 0,
            loop_count: 0,
            loop_start: 0,
            loop_end: 0,
            stop: false,
            params,
            writes,
            steps: 0,
        }
    }

    fn execute(&mut self, engine: &mut Mme64, state: &dyn Fn(u32) -> u32) {
        let mut end_next = false;
        while self.steps < STEP_LIMIT {
            self.steps += 1;
            let instruction = engine.instruction(self.ip);
            self.next_ip = self.ip + 1;
            if instruction.loads > 0 {
                self.load[0] = self.params.pop_front().unwrap_or(0);
                if instruction.loads > 1 {
                    self.load[1] = self.params.pop_front().unwrap_or(0);
                }
            }
            let mask = self.predicate_mask(&instruction);
            let mut stores = [None; 2];
            for alu in 0..2 {
                if mask & (1 << alu) != 0 {
                    stores[alu] = Some((instruction.alu[alu].dst, self.alu(engine, &instruction, alu, state)));
                }
            }
            for (register, value) in stores.into_iter().flatten() {
                self.store(register, value);
            }
            for out in 0..2 {
                if mask & (1 << (out + 2)) != 0 {
                    self.output(engine, &instruction, out);
                }
            }
            if end_next || self.stop {
                break;
            }
            if self.loop_count > 0 && self.ip == self.loop_end {
                self.loop_count -= 1;
                self.next_ip = self.loop_start + 1;
            }
            end_next = instruction.end_next && self.next_ip == self.ip + 1;
            self.ip = self.next_ip;
        }
    }

    fn predicate_mask(&self, instruction: &Instruction) -> u8 {
        if instruction.predicate_mode == 0 {
            return 0xF;
        }
        let value = self.register(instruction, None, instruction.predicate) != 0;
        let pattern = PREDICATES[instruction.predicate_mode as usize & 0xF];
        let mut mask = 0;
        for (slot, letter) in pattern.iter().enumerate() {
            if *letter != if value { b'T' } else { b'F' } {
                mask |= 1 << slot;
            }
        }
        mask
    }

    fn register(&self, instruction: &Instruction, alu: Option<usize>, register: u8) -> u32 {
        match register {
            0..=23 => self.registers[register as usize],
            REG_IMM => alu.map_or(0, |alu| instruction.imm[alu] as i16 as i32 as u32),
            REG_IMMPAIR => alu.map_or(0, |alu| instruction.imm[1 - alu] as i16 as i32 as u32),
            REG_IMM32 => instruction.imm32(),
            REG_LOAD0 => self.load[0],
            REG_LOAD1 => self.load[1],
            _ => 0,
        }
    }

    fn store(&mut self, register: u8, value: u32) {
        if let Some(slot) = self.registers.get_mut(register as usize) {
            *slot = value;
        }
    }

    fn alu(&mut self, engine: &mut Mme64, instruction: &Instruction, index: usize, state: &dyn Fn(u32) -> u32) -> u32 {
        let alu = instruction.alu[index];
        let x = self.register(instruction, Some(index), alu.src[0]);
        let y = self.register(instruction, Some(index), alu.src[1]);
        let imm = instruction.imm[index];
        let result = match alu.op {
            Op::Add => {
                let (sum, carry) = x.overflowing_add(y);
                self.carry = u32::from(carry);
                sum
            }
            Op::Addc => x.wrapping_add(y).wrapping_add(self.carry),
            Op::Sub => {
                let (difference, borrow) = x.overflowing_sub(y);
                self.carry = u32::from(borrow);
                difference
            }
            Op::Subb => x.wrapping_sub(y).wrapping_sub(self.carry),
            Op::Mul => {
                let product = i64::from(x as i32).wrapping_mul(i64::from(y as i32)) as u64;
                self.carry = (product >> 32) as u32;
                product as u32
            }
            Op::Mulh => self.carry,
            Op::Mulu => {
                let product = u64::from(x) * u64::from(y);
                self.carry = (product >> 32) as u32;
                product as u32
            }
            Op::Extended => {
                if (x, y) == (BARRIER_X, BARRIER_Y) {
                    engine.barrier(&mut self.params);
                }
                0
            }
            Op::Clz => x.leading_zeros(),
            Op::Sll => x << (y & 31),
            Op::Srl => x >> (y & 31),
            Op::Sra => ((x as i32) >> (y & 31)) as u32,
            Op::And => x & y,
            Op::Nand => !(x & y),
            Op::Or => x | y,
            Op::Xor => x ^ y,
            Op::Merge => {
                let destination = u32::from((imm >> 10) & 0x3F);
                let bits = u32::from((imm >> 5) & 0x1F);
                let source = u32::from(imm & 0x1F);
                let mask = (1u64 << bits) - 1;
                let cleared = u64::from(x) & !(mask << destination);
                (cleared | (((u64::from(y) >> source) & mask) << destination)) as u32
            }
            Op::Slt | Op::Sltu | Op::Sle | Op::Sleu | Op::Seq => {
                if alu.op.compare(x, y) {
                    u32::MAX
                } else {
                    0
                }
            }
            Op::State => {
                let method = x.wrapping_add(y) & 0xFFFF;
                engine.overlay_read(method).unwrap_or_else(|| state(method))
            }
            Op::Loop => {
                self.loop_count = x.max(1) - 1;
                self.loop_start = self.ip;
                self.loop_end = self.ip + usize::from(imm).saturating_sub(1);
                0
            }
            Op::Jal => {
                self.next_ip = self.ip + usize::from(imm & 0x7FFF);
                0
            }
            Op::Blt | Op::Bltu | Op::Ble | Op::Bleu | Op::Beq => {
                let expect = imm & 0x8000 != 0;
                if alu.op.compare(x, y) == expect {
                    let offset = (i32::from(imm as i16) << 19) >> 19;
                    if offset == STOP_OFFSET {
                        self.stop = true;
                    } else {
                        self.next_ip = (self.ip as i64 + i64::from(offset)).max(0) as usize;
                    }
                }
                0
            }
            Op::Dread => engine.data_ram_word(x),
            Op::Dwrite => {
                if let Some(slot) = engine.data_ram().get_mut(x as usize) {
                    *slot = y;
                }
                0
            }
        };
        self.results[index] = result;
        result
    }

    fn operand(&self, instruction: &Instruction, op: u8) -> u32 {
        match op {
            OUT_ALU0 => self.results[0],
            OUT_ALU1 => self.results[1],
            OUT_LOAD0 => self.load[0],
            OUT_LOAD1 => self.load[1],
            OUT_IMM0 => u32::from(instruction.imm[0]),
            OUT_IMM1 => u32::from(instruction.imm[1]),
            OUT_IMMHIGH0 => u32::from(instruction.imm[0] >> 12),
            OUT_IMMHIGH1 => u32::from(instruction.imm[1] >> 12),
            OUT_IMM32 => instruction.imm32(),
            _ => 0,
        }
    }

    fn output(&mut self, engine: &mut Mme64, instruction: &Instruction, index: usize) {
        let slot = instruction.out[index];
        if slot.method != OUT_NONE {
            let value = self.operand(instruction, slot.method);
            self.method = value & 0xFFF;
            self.increment = (value >> 12) & 0xF;
        }
        if slot.emit != OUT_NONE {
            let value = self.operand(instruction, slot.emit);
            let method = self.method;
            if !engine.dma_method(method, value) {
                engine.overlay_write(method, value);
                self.writes.push((method, value));
                if method == engine.falcon {
                    engine.overlay_write(METHOD_SCRATCH, FIRMWARE_DONE);
                    self.writes.push((METHOD_SCRATCH, FIRMWARE_DONE));
                }
            }
            self.method = (self.method + self.increment) & 0xFFF;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(instruction: &Instruction) -> [u32; 3] {
        let mut bits = 0u128;
        let mut put = |low: u32, high: u32, value: u32| {
            let mask = (1u128 << (high - low + 1)) - 1;
            bits |= (u128::from(value) & mask) << low;
        };
        put(0, 0, u32::from(instruction.end_next));
        put(1, 4, u32::from(instruction.predicate_mode));
        put(5, 9, u32::from(instruction.predicate));
        put(10, 14, instruction.alu[0].op as u32);
        put(15, 19, u32::from(instruction.alu[0].dst));
        put(20, 24, u32::from(instruction.alu[0].src[0]));
        put(25, 29, u32::from(instruction.alu[0].src[1]));
        put(30, 45, u32::from(instruction.imm[0]));
        put(46, 50, instruction.alu[1].op as u32);
        put(51, 55, u32::from(instruction.alu[1].dst));
        put(56, 60, u32::from(instruction.alu[1].src[0]));
        put(61, 65, u32::from(instruction.alu[1].src[1]));
        put(66, 81, u32::from(instruction.imm[1]));
        put(82, 84, u32::from(instruction.out[0].method));
        put(85, 88, u32::from(instruction.out[0].emit));
        put(89, 91, u32::from(instruction.out[1].method));
        put(92, 95, u32::from(instruction.out[1].emit));
        [(bits >> 64) as u32, (bits >> 32) as u32, bits as u32]
    }

    fn idle() -> Alu {
        Alu {
            op: Op::Add,
            dst: REG_ZERO,
            src: [REG_ZERO, REG_ZERO],
        }
    }

    fn nop() -> Instruction {
        Instruction {
            predicate: REG_ZERO,
            alu: [idle(), idle()],
            ..Instruction::default()
        }
    }

    fn engine_with(program: &[Instruction]) -> Mme64 {
        let mut engine = Mme64::new(None);
        engine.method(METHOD_INSTRUCTION_POINTER, 4);
        for instruction in program {
            for word in encode(instruction) {
                engine.method(METHOD_INSTRUCTION, word);
            }
        }
        engine.method(METHOD_START_POINTER, 3);
        engine.method(METHOD_START, 4);
        engine
    }

    fn run(engine: &mut Mme64, params: &[u32]) -> Vec<(u32, u32)> {
        engine.call(MACRO_METHODS_START + 6, params, true, &|method| method * 10).unwrap().writes
    }

    #[test]
    fn decodes_the_title_macro_words() {
        let first = Instruction::decode([0x0000_0003, 0x5c13_0000, 0x73c0_ab00]);
        assert_eq!(first.alu[0], Alu { op: Op::Srl, dst: 1, src: [REG_LOAD0, REG_IMM] });
        assert_eq!(first.alu[1], Alu { op: Op::And, dst: 2, src: [REG_LOAD0, REG_IMMPAIR] });
        assert_eq!(first.imm, [1, 0]);
        assert!(!first.end_next);
        for instruction in [first, nop()] {
            assert_eq!(Instruction::decode(encode(&instruction)), instruction);
        }
    }

    #[test]
    fn emits_methods_with_increment_and_ends_after_the_delay_slot() {
        let mut set = nop();
        set.alu[0] = Alu { op: Op::Add, dst: 0, src: [REG_LOAD0, REG_IMM] };
        set.imm = [5, 0x1000 | 0x40];
        set.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU0 };
        set.out[1] = Out { method: OUT_NONE, emit: OUT_LOAD0 };
        set.end_next = true;
        let mut tail = nop();
        tail.out[0] = Out { method: OUT_NONE, emit: OUT_IMM0 };
        tail.imm = [7, 0];
        let mut never = nop();
        never.out[0] = Out { method: OUT_NONE, emit: OUT_IMM0 };
        never.imm = [9, 0];
        let mut engine = engine_with(&[set, tail, never]);
        assert_eq!(run(&mut engine, &[100]), vec![(0x40, 105), (0x41, 100), (0x42, 7)]);
    }

    #[test]
    fn loops_repeat_the_body_and_state_sees_earlier_writes() {
        let mut start = nop();
        start.alu[0] = Alu { op: Op::Loop, dst: REG_ZERO, src: [REG_LOAD0, REG_ZERO] };
        start.imm = [2, 0];
        let mut body = nop();
        body.alu[0] = Alu { op: Op::Add, dst: 3, src: [3, REG_IMM] };
        body.imm = [1, 0x20];
        body.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU0 };
        let mut read = nop();
        read.alu[0] = Alu { op: Op::State, dst: 4, src: [REG_IMM, REG_ZERO] };
        read.alu[1] = Alu { op: Op::State, dst: 5, src: [REG_IMM, REG_ZERO] };
        read.imm = [0x20, 0x30];
        read.end_next = true;
        let mut emit = nop();
        emit.alu[0] = Alu { op: Op::Add, dst: 6, src: [4, 5] };
        emit.imm = [0, 0x50];
        emit.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU0 };
        let mut engine = engine_with(&[start, body, read, emit]);
        let writes = run(&mut engine, &[3]);
        assert_eq!(&writes[..3], &[(0x20, 1), (0x20, 2), (0x20, 3)]);
        assert_eq!(writes[3], (0x50, 3 + 0x30 * 10));
    }

    #[test]
    fn both_alus_read_registers_before_either_writes() {
        let mut load = nop();
        load.alu[0] = Alu { op: Op::Add, dst: 0, src: [REG_LOAD0, REG_ZERO] };
        let mut pair = nop();
        pair.alu[0] = Alu { op: Op::Add, dst: 0, src: [0, REG_IMM] };
        pair.alu[1] = Alu { op: Op::Add, dst: 1, src: [0, REG_ZERO] };
        pair.imm = [1, 0x80];
        pair.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU1 };
        pair.end_next = true;
        let mut engine = engine_with(&[load, pair, nop()]);
        assert_eq!(run(&mut engine, &[0x10]), vec![(0x80, 0x10)]);
    }

    #[test]
    fn branches_follow_the_expected_condition_and_can_stop() {
        let mut branch = nop();
        branch.alu[0] = Alu { op: Op::Beq, dst: REG_ZERO, src: [REG_LOAD0, REG_ZERO] };
        branch.imm = [0x8000 | 2, 0];
        let mut skipped = nop();
        skipped.out[0] = Out { method: OUT_IMM1, emit: OUT_IMM1 };
        skipped.imm = [0, 0x11];
        let mut stop = nop();
        stop.alu[0] = Alu { op: Op::Beq, dst: REG_ZERO, src: [REG_ZERO, REG_ZERO] };
        stop.imm = [0x8000 | 0x1000, 0];
        stop.out[0] = Out { method: OUT_IMM1, emit: OUT_IMM1 };
        stop.imm[1] = 0x22;
        let mut engine = engine_with(&[branch, skipped, stop, skipped]);
        assert_eq!(run(&mut engine, &[0]), vec![(0x22, 0x22)]);
        assert_eq!(run(&mut engine, &[1]), vec![(0x11, 0x11), (0x22, 0x22)]);
    }

    #[test]
    fn a_taken_branch_cancels_end_next() {
        let mut count = nop();
        count.alu[0] = Alu { op: Op::Add, dst: 5, src: [REG_LOAD0, REG_ZERO] };
        let mut body = nop();
        body.alu[0] = Alu { op: Op::Sub, dst: 5, src: [5, REG_IMM] };
        body.imm = [1, 0x30];
        body.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU0 };
        let mut back = nop();
        back.alu[0] = Alu { op: Op::Beq, dst: REG_ZERO, src: [5, REG_ZERO] };
        back.imm = [0x1FFF, 0];
        back.end_next = true;
        let mut engine = engine_with(&[count, body, back, nop()]);
        assert_eq!(run(&mut engine, &[3]), vec![(0x30, 2), (0x30, 1), (0x30, 0)]);
    }

    #[test]
    fn falcon_requests_are_acknowledged_through_scratch_zero() {
        let mut request = nop();
        request.imm = [0x0D00, 0x08C4];
        request.out[0] = Out { method: OUT_IMM0, emit: OUT_IMMHIGH1 };
        request.out[1] = Out { method: OUT_IMM1, emit: OUT_LOAD0 };
        let mut poll = nop();
        poll.alu[0] = Alu { op: Op::State, dst: 0, src: [REG_IMM, REG_ZERO] };
        poll.imm = [0x0D00, 0];
        let mut wait = nop();
        wait.alu[0] = Alu { op: Op::Beq, dst: REG_ZERO, src: [0, REG_IMMPAIR] };
        wait.imm = [0x1FFF, 1];
        wait.end_next = true;
        let mut engine = engine_with(&[request, poll, wait, nop()]);
        let writes = run(&mut engine, &[0x1234]);
        assert_eq!(writes, vec![(0xD00, 0), (0x8C4, 0x1234), (0xD00, 1)]);
    }

    #[test]
    fn predicates_gate_each_slot() {
        let mut gated = nop();
        gated.predicate_mode = 3;
        gated.predicate = REG_LOAD0;
        gated.alu[0] = Alu { op: Op::Add, dst: 0, src: [REG_IMM, REG_ZERO] };
        gated.alu[1] = Alu { op: Op::Add, dst: 1, src: [REG_IMM, REG_ZERO] };
        gated.imm = [3, 4];
        gated.end_next = true;
        let mut emit = nop();
        emit.alu[0] = Alu { op: Op::Add, dst: 2, src: [0, 1] };
        emit.imm = [0, 0x60];
        emit.out[0] = Out { method: OUT_IMM1, emit: OUT_ALU0 };
        let mut engine = engine_with(&[gated, emit]);
        assert_eq!(run(&mut engine, &[1]), vec![(0x60, 0)]);
        assert_eq!(run(&mut engine, &[0]), vec![(0x60, 7)]);
    }

    #[test]
    fn data_ram_and_dma_sync_are_per_channel() {
        let mut write = nop();
        write.alu[0] = Alu { op: Op::Dwrite, dst: REG_ZERO, src: [REG_IMM, REG_LOAD0] };
        write.imm = [0x27, 0];
        write.end_next = true;
        let mut engine = engine_with(&[write, nop()]);
        engine.set_channel(1);
        run(&mut engine, &[0xAB]);
        assert_eq!(engine.data_ram_word(0x27), 0xAB);
        engine.set_channel(2);
        assert_eq!(engine.data_ram_word(0x27), 0);
        engine.method(METHOD_DATA_RAM_ADDRESS, 5);
        engine.method(METHOD_DMA_SYNC, 0xCAFE);
        assert_eq!(engine.data_ram_word(5), 0xCAFE);
    }

    #[test]
    fn merge_and_multiply_match_the_hardware_rules() {
        let mut merge = nop();
        merge.alu[0] = Alu { op: Op::Merge, dst: 0, src: [REG_LOAD0, REG_LOAD1] };
        merge.imm = [(8 << 10) | (4 << 5) | 4, 0];
        merge.alu[1] = Alu { op: Op::Mul, dst: 1, src: [REG_LOAD0, REG_IMM] };
        merge.imm[1] = (-2i16) as u16;
        merge.end_next = true;
        let mut emit = nop();
        emit.out[0] = Out { method: OUT_IMM1, emit: OUT_IMM0 };
        emit.alu[0] = Alu { op: Op::Add, dst: 2, src: [0, 1] };
        emit.out[1] = Out { method: OUT_NONE, emit: OUT_ALU0 };
        emit.imm = [0, 0x1070];
        let mut engine = engine_with(&[merge, emit]);
        let writes = run(&mut engine, &[0xFFFF_0000, 0xAB0]);
        assert_eq!(writes, vec![(0x70, 0), (0x71, 0x0001_0B00)]);
    }
}

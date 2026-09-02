use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

const NUM_REGS: usize = 8;
pub const MACRO_REGISTERS_START: u32 = 0xE00;
const NUM_MACRO_POSITIONS: usize = 0x80;
const LLE_WRITE_SCRATCH_LIMIT: usize = 32 * 1024;
const LLE_PROFILE_SAMPLE_MASK: u64 = 63;
const LLE_PROFILE_REPORT_MASK: u64 = 1_048_575;
const LLE_PROFILE_HASH_LIMIT: usize = 128;

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| truthy(&value))
}

fn enabled_by_default(value: Option<&str>) -> bool {
    value.is_none_or(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

fn mme_fast_lle_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        let value = std::env::var("NEXIUM_MME_FAST_LLE").ok();
        enabled_by_default(value.as_deref())
    })
}

fn mme_lle_profile_enabled() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| env_truthy("NEXIUM_RENDER_PROFILE") || env_truthy("NEXIUM_MME_LLE_PROFILE"))
}

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
const REG_DRAW_END: u32 = 0x585;
const REG_INDEX_FIRST: u32 = 0x5F7;
const REG_INDEX_COUNT: u32 = 0x5F8;
const REG_DRAW_INSTANCE_COUNT: u32 = 0xD1B;
const REG_VERTEX_ID_BASE: u32 = 0x446;
const REG_VERTEX_BUFFER_INSTANCE: u32 = 0x573;
const REG_CB_SIZE: u32 = 0x8E0;
const REG_CB_ADDR_HI: u32 = 0x8E1;
const REG_CB_ADDR_LO: u32 = 0x8E2;
const REG_CB_OFFSET: u32 = 0x8E3;
const REG_CB_DATA: u32 = 0x8E4;
const REG_MME_SCRATCH_DRAW_TOPOLOGY: u32 = 0xD03;
const REG_MME_SCRATCH_DRAW_ARGUMENT: u32 = 0xD06;
const REG_MME_SCRATCH_VERTEX_BUFFER_MASK: u32 = 0xD07;
const REG_MME_SCRATCH_DRAW_BASE: u32 = 0xD1B;
const REG_MME_SCRATCH_TOPOLOGY: u32 = 0xD1C;
const REG_MME_SCRATCH_VERTEX_BUFFER: u32 = 0xD1D;
const REG_UPLOAD_LINE_LENGTH: u32 = 0x60;
const REG_UPLOAD_LINE_COUNT: u32 = 0x61;
const REG_UPLOAD_DST_HI: u32 = 0x62;
const REG_UPLOAD_DST_LO: u32 = 0x63;
const REG_LAUNCH_DMA: u32 = 0x6C;

#[derive(Clone, Copy)]
enum HleMacro {
    DrawArrays { base_instance: bool },
    DrawIndexed { base_instance: bool },
    DrawInstancedWithVbMask { indexed: bool },
    ConstantBuffer { size: u32 },
    Upload,
}

fn hle_macro_kind(hash: u64) -> Option<HleMacro> {
    match hash {
        0x0D61_FC9F_AAC9_FCAD => Some(HleMacro::DrawArrays {
            base_instance: false,
        }),
        0x8A4D_173E_B99A_8603 => Some(HleMacro::DrawArrays {
            base_instance: true,
        }),
        0x771B_B18C_6244_4DA0 => Some(HleMacro::DrawIndexed {
            base_instance: false,
        }),
        0x0217_9201_0048_8FF7 => Some(HleMacro::DrawIndexed {
            base_instance: true,
        }),
        0x62AB_88C4_D2DF_58E3 => Some(HleMacro::DrawInstancedWithVbMask { indexed: false }),
        0xD00E_2028_0475_8F38 => Some(HleMacro::DrawInstancedWithVbMask { indexed: true }),
        0x6C97_861D_891E_DF7E => Some(HleMacro::ConstantBuffer { size: 0x5F00 }),
        0xD246_FDDF_3A61_73D7 => Some(HleMacro::ConstantBuffer { size: 0x7000 }),
        0xEE4D_0004_BEC8_ECF4 => Some(HleMacro::Upload),
        _ => None,
    }
}

fn hle_macro(
    kind: HleMacro,
    params: &[u32],
    reg_reader: &dyn Fn(u32) -> u32,
) -> Option<MacroOutput> {
    let p = |i: usize| params.get(i).copied().unwrap_or(0);
    let macro_instance_count = || (reg_reader(REG_DRAW_INSTANCE_COUNT) & p(2)).max(1);
    let mut out = MacroOutput::default();
    match kind {
        HleMacro::DrawArrays { base_instance } => {
            let topology = p(0) & 0xFFFF;
            let vertex_count = p(1);
            let vertex_first = p(3);
            out.draw_instance_count = Some(macro_instance_count());
            if base_instance {
                out.writes.push((REG_GLOBAL_BASE_INSTANCE, p(4)));
            }
            out.writes.push((REG_DRAW_BEGIN, topology));
            out.writes.push((REG_VERTEX_FIRST, vertex_first));
            out.writes.push((REG_VERTEX_COUNT, vertex_count));
            if base_instance {
                out.writes.push((REG_GLOBAL_BASE_INSTANCE, 0));
            }
        }
        HleMacro::DrawIndexed { base_instance } => {
            let topology = p(0) & 0xFFFF;
            let index_count = p(1);
            let index_first = p(3);
            let base_vertex = p(4);
            out.draw_instance_count = Some(macro_instance_count());
            out.writes.push((REG_GLOBAL_BASE_VERTEX, base_vertex));
            if base_instance {
                out.writes.push((REG_GLOBAL_BASE_INSTANCE, p(5)));
            }
            out.writes.push((REG_DRAW_BEGIN, topology));
            out.writes.push((REG_INDEX_FIRST, index_first));
            out.writes.push((REG_INDEX_COUNT, index_count));
            out.writes.push((REG_GLOBAL_BASE_VERTEX, 0));
            if base_instance {
                out.writes.push((REG_GLOBAL_BASE_INSTANCE, 0));
            }
        }
        HleMacro::DrawInstancedWithVbMask { indexed } => {
            let vertex_buffer_mask = reg_reader(REG_MME_SCRATCH_VERTEX_BUFFER_MASK);
            let group_count = if vertex_buffer_mask == 0 {
                1
            } else {
                vertex_buffer_mask.count_ones()
            };
            let draw_count = u64::from(p(2)) * u64::from(group_count);

            if draw_count > 128 {
                return None;
            }

            let extra_group_writes = if vertex_buffer_mask == 0 {
                0
            } else {
                group_count as usize * 4
            };
            out.writes
                .reserve(15 + extra_group_writes + draw_count as usize * 4);

            let base_vertex = if indexed { p(4) } else { 0 };
            let base_instance = if indexed { p(5) } else { p(4) };
            let draw_base = if indexed { p(4) } else { p(3) };

            out.writes.push((REG_MME_SCRATCH_DRAW_BASE, draw_base));
            out.writes.push((REG_CB_OFFSET, 0));
            out.writes.push((REG_CB_DATA, draw_base));
            out.writes.push((REG_GLOBAL_BASE_INSTANCE, base_instance));
            out.writes.push((REG_CB_OFFSET, 4));
            out.writes.push((REG_CB_DATA, base_instance));
            out.writes.push((REG_MME_SCRATCH_TOPOLOGY, p(0)));
            out.writes.push((REG_CB_OFFSET, 8));
            out.writes.push((REG_CB_DATA, p(0)));
            out.writes.push((REG_MME_SCRATCH_VERTEX_BUFFER, 0));
            out.writes.push((REG_CB_OFFSET, 12));
            out.writes.push((REG_CB_DATA, 0));
            out.writes.push((REG_GLOBAL_BASE_VERTEX, base_vertex));
            out.writes.push((REG_VERTEX_ID_BASE, base_vertex));
            out.writes.push((REG_MME_SCRATCH_DRAW_ARGUMENT, p(0)));

            let draw_topology = reg_reader(REG_MME_SCRATCH_DRAW_TOPOLOGY);
            let push_draws = |out: &mut MacroOutput| {
                let mut topology = draw_topology;
                for _ in 0..p(2) {
                    out.writes.push((REG_DRAW_BEGIN, topology));
                    if indexed {
                        out.writes.push((REG_INDEX_FIRST, p(3)));
                        out.writes.push((REG_INDEX_COUNT, p(1)));
                    } else {
                        out.writes.push((REG_VERTEX_FIRST, p(3)));
                        out.writes.push((REG_VERTEX_COUNT, p(1)));
                    }
                    out.writes.push((REG_DRAW_END, 0));
                    topology = (topology & !(3 << 26)) | (1 << 26);
                }
            };

            if vertex_buffer_mask == 0 {
                push_draws(&mut out);
            } else {
                for vertex_buffer in 0..32 {
                    if vertex_buffer_mask & (1 << vertex_buffer) == 0 {
                        continue;
                    }
                    out.writes
                        .push((REG_MME_SCRATCH_VERTEX_BUFFER, vertex_buffer));
                    out.writes.push((REG_CB_OFFSET, 12));
                    out.writes.push((REG_CB_DATA, vertex_buffer));
                    out.writes.push((REG_VERTEX_BUFFER_INSTANCE, vertex_buffer));
                    push_draws(&mut out);
                }
            }
        }
        HleMacro::ConstantBuffer { size } => {
            out.writes.push((REG_CB_SIZE, size));
            out.writes.push((REG_CB_ADDR_HI, p(0)));
            out.writes.push((REG_CB_ADDR_LO, p(1)));
            out.writes.push((REG_CB_OFFSET, 0));
        }
        HleMacro::Upload => {
            out.writes.push((REG_UPLOAD_LINE_LENGTH, p(2)));
            out.writes.push((REG_UPLOAD_LINE_COUNT, 1));
            out.writes.push((REG_UPLOAD_DST_HI, p(0)));
            out.writes.push((REG_UPLOAD_DST_LO, p(1)));
            out.writes.push((REG_LAUNCH_DMA, 0x1011));
        }
    }
    Some(out)
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
    Add,
    AddWithCarry,
    Subtract,
    SubtractWithBorrow,
    Xor,
    Or,
    And,
    AndNot,
    Nand,
    Unknown,
}

impl AluOp {
    fn from_u32(v: u32) -> Self {
        match v & 0x1F {
            0 => Self::Add,
            1 => Self::AddWithCarry,
            2 => Self::Subtract,
            3 => Self::SubtractWithBorrow,
            8 => Self::Xor,
            9 => Self::Or,
            10 => Self::And,
            11 => Self::AndNot,
            12 => Self::Nand,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy)]
struct Opcode(u32);

impl Opcode {
    fn operation(self) -> Operation {
        Operation::from_u32(self.0 & 0x7)
    }
    fn result_operation(self) -> ResultOperation {
        ResultOperation::from_u32((self.0 >> 4) & 0x7)
    }
    fn branch_zero(self) -> bool {
        (self.0 >> 4) & 0x1 == 0
    }
    fn branch_annul(self) -> bool {
        (self.0 >> 5) & 0x1 != 0
    }
    fn is_exit(self) -> bool {
        (self.0 >> 7) & 0x1 != 0
    }
    fn dst(self) -> u32 {
        (self.0 >> 8) & 0x7
    }
    fn src_a(self) -> u32 {
        (self.0 >> 11) & 0x7
    }
    fn src_b(self) -> u32 {
        (self.0 >> 14) & 0x7
    }
    fn immediate(self) -> i32 {
        let raw = self.0 >> 14;
        if raw & 0x2_0000 != 0 {
            (raw | 0xFFFC_0000) as i32
        } else {
            raw as i32
        }
    }
    fn alu_op(self) -> AluOp {
        AluOp::from_u32((self.0 >> 17) & 0x1F)
    }
    fn bf_src_bit(self) -> u32 {
        (self.0 >> 17) & 0x1F
    }
    fn bf_size(self) -> u32 {
        (self.0 >> 22) & 0x1F
    }
    fn bf_dst_bit(self) -> u32 {
        (self.0 >> 27) & 0x1F
    }
    fn bitfield_mask(self) -> u32 {
        let size = self.bf_size();
        if size >= 32 {
            0xFFFF_FFFF
        } else {
            (1u32 << size) - 1
        }
    }
    fn branch_target(self) -> i32 {
        self.immediate().wrapping_mul(4)
    }
}

#[derive(Clone, Copy)]
struct DecodedOpcode {
    operation: Operation,
    result_operation: ResultOperation,
    is_exit: bool,
    dst: u32,
    src_a: u32,
    src_b: u32,
    immediate: i32,
    alu_op: AluOp,
    bf_src_bit: u32,
    bitfield_mask: u32,
    bf_dst_bit: u32,
    branch_zero: bool,
    branch_annul: bool,
    branch_target: i32,
}

impl From<Opcode> for DecodedOpcode {
    fn from(op: Opcode) -> Self {
        Self {
            operation: op.operation(),
            result_operation: op.result_operation(),
            is_exit: op.is_exit(),
            dst: op.dst(),
            src_a: op.src_a(),
            src_b: op.src_b(),
            immediate: op.immediate(),
            alu_op: op.alu_op(),
            bf_src_bit: op.bf_src_bit(),
            bitfield_mask: op.bitfield_mask(),
            bf_dst_bit: op.bf_dst_bit(),
            branch_zero: op.branch_zero(),
            branch_annul: op.branch_annul(),
            branch_target: op.branch_target(),
        }
    }
}

#[derive(Default)]
pub struct MacroOutput {
    pub writes: Vec<(u32, u32)>,
    pub draw_instance_count: Option<u32>,
    pub hash: u64,
    pub entry: u32,
    pub hle: bool,
    fast_lle: bool,
}

#[derive(Default)]
struct LleHashProfile {
    hash: u64,
    calls: u64,
    fast_calls: u64,
    steps: u64,
    writes: u64,
    elapsed: Duration,
}

#[derive(Default)]
struct LleProfile {
    calls: u64,
    fast_calls: u64,
    steps: u64,
    writes: u64,
    sampled_calls: u64,
    sampled_elapsed: Duration,
    hashes: Vec<LleHashProfile>,
}

impl LleProfile {
    fn sample_next(&self) -> bool {
        self.calls & LLE_PROFILE_SAMPLE_MASK == 0
    }

    fn record(&mut self, hash: u64, fast: bool, steps: u32, writes: usize, elapsed: Duration) {
        self.calls = self.calls.wrapping_add(1);
        self.fast_calls += u64::from(fast);
        self.steps = self.steps.wrapping_add(u64::from(steps));
        self.writes = self.writes.wrapping_add(writes as u64);
        if elapsed != Duration::ZERO {
            self.sampled_calls = self.sampled_calls.wrapping_add(1);
            self.sampled_elapsed += elapsed;
            if let Some(profile) = self.hashes.iter_mut().find(|profile| profile.hash == hash) {
                profile.calls = profile.calls.wrapping_add(1);
                profile.fast_calls += u64::from(fast);
                profile.steps = profile.steps.wrapping_add(u64::from(steps));
                profile.writes = profile.writes.wrapping_add(writes as u64);
                profile.elapsed += elapsed;
            } else if self.hashes.len() < LLE_PROFILE_HASH_LIMIT {
                self.hashes.push(LleHashProfile {
                    hash,
                    calls: 1,
                    fast_calls: u64::from(fast),
                    steps: u64::from(steps),
                    writes: writes as u64,
                    elapsed,
                });
            }
        }
        if self.calls & LLE_PROFILE_REPORT_MASK == 0 {
            log::info!(
                "[mme-lle] calls={} fast={} steps={} writes={} samples={} sample_ms={:.3}",
                self.calls,
                self.fast_calls,
                self.steps,
                self.writes,
                self.sampled_calls,
                self.sampled_elapsed.as_secs_f64() * 1000.0
            );
            for profile in &self.hashes {
                log::info!(
                    "[mme-lle] hash={:#018x} samples={} fast={} steps={} writes={} ns_call={:.0}",
                    profile.hash,
                    profile.calls,
                    profile.fast_calls,
                    profile.steps,
                    profile.writes,
                    profile.elapsed.as_nanos() as f64 / profile.calls.max(1) as f64
                );
            }
        }
    }
}

fn mme_forensics() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_FORENSICS").is_some())
}

fn mme_full_code() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_FULL_CODE").is_some())
}

fn is_hi_addr_reg(m: u32) -> bool {
    m == 0x582
        || m == 0x6c0
        || m == 0x8e1
        || m == 0x1c
        || m == 0x554
        || (0x200..0x280).contains(&m) && (m & 0xF) == 0
}

fn suspicious_write(m: u32, a: u32) -> bool {
    (is_hi_addr_reg(m) && a > 0xFF)
        || (matches!(m, 0x582 | 0x583 | 0x6c0 | 0x6c1 | 0x6c2) && matches!(a >> 24, 0x3E..=0x48))
}

fn forensic_report(out: &MacroOutput, params: &[u32]) {
    if !mme_forensics() {
        return;
    }
    if !out.writes.iter().any(|&(m, a)| suspicious_write(m, a)) {
        return;
    }
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    if N.fetch_add(1, Ordering::Relaxed) >= 96 {
        return;
    }
    log::warn!(
        "[mme-garbage] entry={} hash={:#018x} hle={} params={:08x?} writes={:08x?}",
        out.entry,
        out.hash,
        out.hle,
        &params[..params.len().min(12)],
        &out.writes[..out.writes.len().min(16)]
    );
}

pub struct MacroEngine {
    uploaded_code: HashMap<u32, Vec<u32>>,
    compiled: HashMap<u32, CompiledMacro>,
    macro_positions: [u32; NUM_MACRO_POSITIONS],
    instruction_ptr: u32,
    start_address_ptr: u32,
    executing_macro: u32,
    pending_params: Vec<u32>,
    seen_hashes: HashSet<u64>,
    fast_lle: bool,
    upload_serials: HashMap<u32, u64>,
    next_upload_serial: u64,
    lle_write_scratch: Vec<(u32, u32)>,
    lle_profile: Option<LleProfile>,
}

struct CompiledMacro {
    code: Vec<u32>,
    hash: u64,
    hle: Option<HleMacro>,
    decoded: Option<Box<[DecodedOpcode]>>,
}

impl CompiledMacro {
    fn new(code: Vec<u32>, fast_lle: bool) -> Self {
        let hash = macro_hash(&code);
        let decoded = fast_lle.then(|| {
            code.iter()
                .copied()
                .map(Opcode)
                .map(DecodedOpcode::from)
                .collect()
        });
        Self {
            code,
            hash,
            hle: hle_macro_kind(hash),
            decoded,
        }
    }
}

impl MacroEngine {
    pub fn new() -> Self {
        Self::new_with_options(mme_fast_lle_enabled(), mme_lle_profile_enabled())
    }

    fn new_with_options(fast_lle: bool, profile_lle: bool) -> Self {
        Self {
            uploaded_code: HashMap::new(),
            compiled: HashMap::new(),
            macro_positions: [0; NUM_MACRO_POSITIONS],
            instruction_ptr: 0,
            start_address_ptr: 0,
            executing_macro: 0,
            pending_params: Vec::new(),
            seen_hashes: HashSet::new(),
            fast_lle,
            upload_serials: HashMap::new(),
            next_upload_serial: 0,
            lle_write_scratch: Vec::new(),
            lle_profile: profile_lle.then(LleProfile::default),
        }
    }

    pub fn set_instruction_ptr(&mut self, value: u32) {
        self.instruction_ptr = value;
        self.uploaded_code.remove(&value);
        if self.fast_lle {
            self.upload_serials.remove(&value);
        }
        self.compiled.clear();
    }

    pub fn upload_instruction(&mut self, word: u32) {
        if self.fast_lle {
            self.compiled.clear();
            self.next_upload_serial = self.next_upload_serial.wrapping_add(1).max(1);
            self.upload_serials
                .insert(self.instruction_ptr, self.next_upload_serial);
        }
        self.uploaded_code
            .entry(self.instruction_ptr)
            .or_default()
            .push(word);
    }

    pub fn set_start_address_ptr(&mut self, value: u32) {
        self.start_address_ptr = value & 0x7F;
    }

    pub fn bind_macro_entry(&mut self, offset: u32) {
        let slot = (self.start_address_ptr as usize) % NUM_MACRO_POSITIONS;
        self.macro_positions[slot] = offset;
        self.start_address_ptr = self.start_address_ptr.wrapping_add(1) & 0x7F;
        if self.fast_lle {
            self.compiled.clear();
        } else {
            self.compiled.remove(&offset);
        }
    }

    pub fn recycle_output(&mut self, mut output: MacroOutput) {
        if !output.fast_lle {
            return;
        }
        output.writes.clear();
        if output.writes.capacity() <= LLE_WRITE_SCRATCH_LIMIT
            && output.writes.capacity() > self.lle_write_scratch.capacity()
        {
            self.lle_write_scratch = output.writes;
        }
    }

    pub fn on_macro_method(
        &mut self,
        method: u32,
        arg: u32,
        is_last_call: bool,
        reg_reader: &dyn Fn(u32) -> u32,
    ) -> Option<MacroOutput> {
        self.on_macro_methods(method, std::slice::from_ref(&arg), is_last_call, reg_reader)
    }

    pub fn on_macro_methods(
        &mut self,
        method: u32,
        args: &[u32],
        is_last_call: bool,
        reg_reader: &dyn Fn(u32) -> u32,
    ) -> Option<MacroOutput> {
        debug_assert!(!args.is_empty());
        if self.executing_macro == 0 {
            self.executing_macro = method & !1;
            self.pending_params.clear();
        }
        self.pending_params.extend_from_slice(args);
        if !is_last_call {
            return None;
        }
        let trigger = self.executing_macro;
        self.executing_macro = 0;
        let entry = ((trigger - MACRO_REGISTERS_START) >> 1) as usize % NUM_MACRO_POSITIONS;
        let offset = self.macro_positions[entry];
        self.resolve_code(offset);
        let Some(compiled) = self.compiled.get(&offset) else {
            log::trace!(
                "MME: trigger {:#x} entry={} offset={} - no code",
                trigger,
                entry,
                offset
            );
            self.pending_params.clear();
            return Some(MacroOutput::default());
        };
        let params = self.pending_params.as_slice();
        let code = compiled.code.as_slice();
        let hash = compiled.hash;
        let hle = compiled
            .hle
            .and_then(|kind| hle_macro(kind, params, reg_reader));
        if mme_forensics() && self.seen_hashes.insert(hash) {
            let logged_code = if mme_full_code() {
                code
            } else {
                &code[..code.len().min(28)]
            };
            log::info!(
                "MME: macro entry={} offset={} hash={:#018x} len={} params={} hle={} code={:08x?}",
                entry,
                offset,
                hash,
                code.len(),
                params.len(),
                hle.is_some(),
                logged_code
            );
        }
        let sample_lle = hle.is_none()
            && self
                .lle_profile
                .as_ref()
                .is_some_and(LleProfile::sample_next);
        let started = sample_lle.then(Instant::now);
        let mut lle_stats = None;
        let mut out = if let Some(mut out) = hle {
            out.hle = true;
            out
        } else if self.fast_lle {
            let decoded = compiled.decoded.as_deref().unwrap_or_default();
            let scratch = std::mem::take(&mut self.lle_write_scratch);
            let mut interp = FastInterpreter::new(decoded, params, reg_reader, scratch);
            interp.run();
            if self.lle_profile.is_some() {
                lle_stats = Some((true, 8192 - interp.steps_remaining, interp.writes.len()));
            }
            MacroOutput {
                writes: interp.writes,
                draw_instance_count: None,
                hash: 0,
                entry: 0,
                hle: false,
                fast_lle: true,
            }
        } else {
            let mut interp = Interpreter::new(&code, &params, reg_reader);
            interp.run();
            if self.lle_profile.is_some() {
                lle_stats = Some((false, 8192 - interp.steps_remaining, interp.writes.len()));
            }
            MacroOutput {
                writes: interp.writes,
                draw_instance_count: None,
                hash: 0,
                entry: 0,
                hle: false,
                fast_lle: false,
            }
        };
        out.hash = hash;
        out.entry = entry as u32;
        forensic_report(&out, params);
        if let (Some(profile), Some((fast, steps, writes))) = (self.lle_profile.as_mut(), lle_stats)
        {
            profile.record(
                hash,
                fast,
                steps,
                writes,
                started.map_or(Duration::ZERO, |started| started.elapsed()),
            );
        }
        self.pending_params.clear();
        Some(out)
    }

    fn resolve_code(&mut self, offset: u32) {
        if self.compiled.contains_key(&offset) {
            return;
        }
        if self.fast_lle {
            let code = self.resolve_fast_code(offset);
            if !code.is_empty() {
                self.compiled.insert(offset, CompiledMacro::new(code, true));
            }
            return;
        }
        if let Some(c) = self.uploaded_code.get(&offset) {
            self.compiled
                .insert(offset, CompiledMacro::new(c.clone(), self.fast_lle));
            return;
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
            self.compiled
                .insert(offset, CompiledMacro::new(v, self.fast_lle));
        }
    }

    fn resolve_fast_code(&self, offset: u32) -> Vec<u32> {
        let mut end = self
            .uploaded_code
            .iter()
            .filter_map(|(&base, code)| {
                let segment_end = base.saturating_add(code.len() as u32);
                (base <= offset && offset < segment_end).then_some(segment_end)
            })
            .max()
            .unwrap_or(offset);
        if end == offset {
            return Vec::new();
        }
        loop {
            let expanded = self
                .uploaded_code
                .iter()
                .filter_map(|(&base, code)| {
                    let segment_end = base.saturating_add(code.len() as u32);
                    (base < end && segment_end > offset).then_some(segment_end)
                })
                .max()
                .unwrap_or(end);
            if expanded <= end {
                break;
            }
            end = expanded;
        }
        let mut code = Vec::with_capacity((end - offset) as usize);
        for address in offset..end {
            let selected = self
                .uploaded_code
                .iter()
                .filter_map(|(&base, words)| {
                    let index = address.checked_sub(base)? as usize;
                    let word = words.get(index).copied()?;
                    let serial = self.upload_serials.get(&base).copied().unwrap_or(0);
                    Some((serial, base, word))
                })
                .max_by_key(|&(serial, base, _)| (serial, base));
            let Some((_, _, word)) = selected else {
                break;
            };
            code.push(word);
        }
        code
    }
}

impl Default for MacroEngine {
    fn default() -> Self {
        Self::new()
    }
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
            code,
            params,
            next_param: 1,
            registers: regs,
            pc: 0,
            delayed_pc: None,
            method_address: 0,
            carry: false,
            writes: Vec::new(),
            written: HashMap::new(),
            reg_reader,
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
            log::warn!(
                "MME: macro hit step cap (produced {} writes) — discarding as runaway",
                self.writes.len()
            );
            self.writes.clear();
        }
    }

    fn fetch_opcode(&self) -> Opcode {
        Opcode(*self.code.get(self.pc / 4).unwrap_or(&0))
    }

    fn read_reg(&self, id: u32) -> u32 {
        if id == 0 {
            0
        } else {
            self.registers[id as usize & 7]
        }
    }

    fn write_reg(&mut self, id: u32, value: u32) {
        if id == 0 {
            return;
        }
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
                let r = (a as u64)
                    .wrapping_sub(b as u64)
                    .wrapping_sub(if self.carry { 0 } else { 1 });
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
                let addr = self
                    .read_reg(op.src_a())
                    .wrapping_add(op.immediate() as u32);
                let v = self
                    .written
                    .get(&addr)
                    .copied()
                    .unwrap_or_else(|| (self.reg_reader)(addr));
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

struct FastInterpreter<'a> {
    code: &'a [DecodedOpcode],
    params: &'a [u32],
    next_param: usize,
    registers: [u32; NUM_REGS],
    pc: usize,
    delayed_pc: Option<usize>,
    method_address: u32,
    carry: bool,
    writes: Vec<(u32, u32)>,
    reg_reader: &'a dyn Fn(u32) -> u32,
    steps_remaining: u32,
}

impl<'a> FastInterpreter<'a> {
    fn new(
        code: &'a [DecodedOpcode],
        params: &'a [u32],
        reg_reader: &'a dyn Fn(u32) -> u32,
        writes: Vec<(u32, u32)>,
    ) -> Self {
        let mut registers = [0; NUM_REGS];
        if let Some(&first) = params.first() {
            registers[1] = first;
        }
        Self {
            code,
            params,
            next_param: 1,
            registers,
            pc: 0,
            delayed_pc: None,
            method_address: 0,
            carry: false,
            writes,
            reg_reader,
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
            log::warn!(
                "MME: macro hit step cap (produced {} writes) — discarding as runaway",
                self.writes.len()
            );
            self.writes.clear();
        }
    }

    fn read_reg(&self, id: u32) -> u32 {
        if id == 0 {
            0
        } else {
            self.registers[id as usize & 7]
        }
    }

    fn write_reg(&mut self, id: u32, value: u32) {
        if id != 0 {
            self.registers[id as usize & 7] = value;
        }
    }

    fn send(&mut self, value: u32) {
        let address = self.method_address & 0xFFF;
        let increment = (self.method_address >> 12) & 0x3F;
        self.writes.push((address, value));
        let next = address.wrapping_add(increment) & 0xFFF;
        self.method_address = (self.method_address & !0xFFF) | next;
    }

    fn fetch_param(&mut self) -> u32 {
        if self.next_param >= self.params.len() {
            return 0;
        }
        let value = self.params[self.next_param];
        self.next_param += 1;
        value
    }

    fn read_method(&self, address: u32) -> u32 {
        for &(method, value) in self.writes.iter().rev() {
            if method == address {
                return value;
            }
            if method == 0x8C4 && address == 0xD00 {
                return 1;
            }
        }
        (self.reg_reader)(address)
    }

    fn alu(&mut self, op: AluOp, a: u32, b: u32) -> u32 {
        match op {
            AluOp::Add => {
                let result = u64::from(a) + u64::from(b);
                self.carry = result > u64::from(u32::MAX);
                result as u32
            }
            AluOp::AddWithCarry => {
                let result = u64::from(a) + u64::from(b) + u64::from(self.carry);
                self.carry = result > u64::from(u32::MAX);
                result as u32
            }
            AluOp::Subtract => {
                let result = u64::from(a).wrapping_sub(u64::from(b));
                self.carry = result < 0x1_0000_0000;
                result as u32
            }
            AluOp::SubtractWithBorrow => {
                let result = u64::from(a)
                    .wrapping_sub(u64::from(b))
                    .wrapping_sub(u64::from(!self.carry));
                self.carry = result < 0x1_0000_0000;
                result as u32
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
                let param = self.fetch_param();
                self.write_reg(reg, param);
            }
            ResultOperation::Move => self.write_reg(reg, result),
            ResultOperation::MoveAndSetMethod => {
                self.write_reg(reg, result);
                self.method_address = result;
            }
            ResultOperation::FetchAndSend => {
                let param = self.fetch_param();
                self.write_reg(reg, param);
                self.send(result);
            }
            ResultOperation::MoveAndSend => {
                self.write_reg(reg, result);
                self.send(result);
            }
            ResultOperation::FetchAndSetMethod => {
                let param = self.fetch_param();
                self.write_reg(reg, param);
                self.method_address = result;
            }
            ResultOperation::MoveAndSetMethodFetchAndSend => {
                self.write_reg(reg, result);
                self.method_address = result;
                let param = self.fetch_param();
                self.send(param);
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
        let op = self.code[self.pc / 4];
        self.pc += 4;
        if let Some(delayed_pc) = self.delayed_pc.take() {
            let _ = is_delay_slot;
            self.pc = delayed_pc;
        }
        match op.operation {
            Operation::Alu => {
                let a = self.read_reg(op.src_a);
                let b = self.read_reg(op.src_b);
                let result = self.alu(op.alu_op, a, b);
                self.process_result(op.result_operation, op.dst, result);
            }
            Operation::AddImmediate => {
                let result = self.read_reg(op.src_a).wrapping_add(op.immediate as u32);
                self.process_result(op.result_operation, op.dst, result);
            }
            Operation::ExtractInsert => {
                let mut dst = self.read_reg(op.src_a);
                let mut src = self.read_reg(op.src_b);
                src = (src >> op.bf_src_bit) & op.bitfield_mask;
                dst &= !(op.bitfield_mask << op.bf_dst_bit);
                dst |= src << op.bf_dst_bit;
                self.process_result(op.result_operation, op.dst, dst);
            }
            Operation::ExtractShiftLeftImmediate => {
                let dst = self.read_reg(op.src_a);
                let src = self.read_reg(op.src_b);
                let result = ((src >> dst) & op.bitfield_mask) << op.bf_dst_bit;
                self.process_result(op.result_operation, op.dst, result);
            }
            Operation::ExtractShiftLeftRegister => {
                let dst = self.read_reg(op.src_a);
                let src = self.read_reg(op.src_b);
                let result = ((src >> op.bf_src_bit) & op.bitfield_mask) << dst;
                self.process_result(op.result_operation, op.dst, result);
            }
            Operation::Read => {
                let address = self.read_reg(op.src_a).wrapping_add(op.immediate as u32);
                let value = self.read_method(address);
                self.process_result(op.result_operation, op.dst, value);
            }
            Operation::Branch => {
                let value = self.read_reg(op.src_a);
                let taken = if op.branch_zero {
                    value == 0
                } else {
                    value != 0
                };
                if taken {
                    let target = (base as i64).wrapping_add(i64::from(op.branch_target));
                    let target = if target < 0 { 0 } else { target as usize };
                    if op.branch_annul {
                        self.pc = target;
                        return true;
                    }
                    self.delayed_pc = Some(target);
                    return self.step(true);
                }
            }
            Operation::Unused => {}
        }
        if op.is_exit && !is_delay_slot {
            self.step(true);
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAW_ARRAYS_INSTANCED_WITH_VB_MASK_CODE: [u32; 85] = [
        0x0000_0200,
        0x0000_0300,
        0x0000_0400,
        0x0000_0500,
        0x0746_C021,
        0x0000_2040,
        0x0638_C021,
        0x0000_0040,
        0x0639_0021,
        0x0000_2040,
        0x0543_8021,
        0x0000_2840,
        0x0638_C021,
        0x0001_0641,
        0x0639_0021,
        0x0000_2840,
        0x0747_0021,
        0x0000_0840,
        0x0638_C021,
        0x0002_0641,
        0x0639_0021,
        0x0000_0840,
        0x0747_4021,
        0x0000_0040,
        0x0638_C021,
        0x0003_0641,
        0x0639_0021,
        0x0000_0040,
        0x0543_4021,
        0x0000_0040,
        0x0511_8021,
        0x0000_0040,
        0x0741_8021,
        0x0000_0840,
        0x0341_C115,
        0x0003_C837,
        0x0340_C115,
        0x0000_1D10,
        0x0002_C027,
        0x0561_8021,
        0x0000_0840,
        0x04D7_4021,
        0x0000_2040,
        0x0000_1040,
        0x0561_4021,
        0x0000_0040,
        0x0000_4611,
        0xD081_8912,
        0xFFFF_ED11,
        0xFFFD_A837,
        0x0341_C115,
        0x0007_C827,
        0x0000_0110,
        0x0006_C027,
        0x0341_C515,
        0x0041_4E13,
        0x0005_F027,
        0x0747_4021,
        0x0000_0840,
        0x0638_C021,
        0x0003_0541,
        0x0639_0021,
        0x0000_0840,
        0x055C_C021,
        0x0000_0840,
        0x0340_C515,
        0x0000_1E10,
        0x0002_C027,
        0x0561_8021,
        0x0000_2840,
        0x04D7_4021,
        0x0000_2040,
        0x0000_1040,
        0x0561_4021,
        0x0000_0040,
        0x0000_4711,
        0xD081_ED12,
        0xFFFF_F611,
        0xFFFD_B037,
        0x0000_4911,
        0xFFF8_0D11,
        0xFFF9_6837,
        0x0341_8115,
        0x0000_0090,
        0x0000_0010,
    ];

    const DRAW_INDEXED_INSTANCED_WITH_VB_MASK_CODE: [u32; 86] = [
        0x0000_0200,
        0x0000_0300,
        0x0000_0400,
        0x0000_0500,
        0x0000_0600,
        0x0746_C021,
        0x0000_2840,
        0x0638_C021,
        0x0000_0040,
        0x0639_0021,
        0x0000_2840,
        0x0543_8021,
        0x0000_3040,
        0x0638_C021,
        0x0001_0741,
        0x0639_0021,
        0x0000_3040,
        0x0747_0021,
        0x0000_0840,
        0x0638_C021,
        0x0002_0741,
        0x0639_0021,
        0x0000_0840,
        0x0747_4021,
        0x0000_0040,
        0x0638_C021,
        0x0003_0741,
        0x0639_0021,
        0x0000_0040,
        0x0543_4021,
        0x0000_2840,
        0x0511_8021,
        0x0000_2840,
        0x0741_8021,
        0x0000_0840,
        0x0341_C115,
        0x0003_C837,
        0x0340_C115,
        0x0000_1D10,
        0x0002_C027,
        0x0561_8021,
        0x0000_0840,
        0x057D_C021,
        0x0000_2040,
        0x0000_1040,
        0x0561_4021,
        0x0000_0040,
        0x0000_4611,
        0xD081_8912,
        0xFFFF_ED11,
        0xFFFD_A837,
        0x0341_C115,
        0x0007_C827,
        0x0000_0110,
        0x0006_C027,
        0x0341_C515,
        0x0041_4E13,
        0x0005_F027,
        0x0747_4021,
        0x0000_0840,
        0x0638_C021,
        0x0003_0541,
        0x0639_0021,
        0x0000_0840,
        0x055C_C021,
        0x0000_0840,
        0x0340_C515,
        0x0000_1E10,
        0x0002_C027,
        0x0561_8021,
        0x0000_2840,
        0x057D_C021,
        0x0000_2040,
        0x0000_1040,
        0x0561_4021,
        0x0000_0040,
        0x0000_4711,
        0xD081_ED12,
        0xFFFF_F611,
        0xFFFD_B037,
        0x0000_4911,
        0xFFF8_0D11,
        0xFFF9_6837,
        0x0341_8115,
        0x0000_0090,
        0x0000_0010,
    ];

    fn upload_entry_11(engine: &mut MacroEngine, code: &[u32]) {
        engine.set_instruction_ptr(0);
        for &word in code {
            engine.upload_instruction(word);
        }
        engine.set_start_address_ptr(11);
        engine.bind_macro_entry(0);
    }

    fn invoke_entry_11(engine: &mut MacroEngine) -> MacroOutput {
        let mut output = None;
        for param in 0..6u32 {
            output = engine.on_macro_method(0xE17, param, param == 5, &|_| 0);
        }
        output.expect("the final parameter must execute the macro")
    }

    fn decoded(code: &[u32]) -> Vec<DecodedOpcode> {
        code.iter()
            .copied()
            .map(Opcode)
            .map(DecodedOpcode::from)
            .collect()
    }

    fn assert_fast_matches_legacy(
        code: &[u32],
        params: &[u32],
        reg_reader: &dyn Fn(u32) -> u32,
    ) -> (Vec<(u32, u32)>, u32) {
        let mut legacy = Interpreter::new(code, params, reg_reader);
        legacy.run();
        let decoded = decoded(code);
        let mut fast = FastInterpreter::new(&decoded, params, reg_reader, Vec::new());
        fast.run();
        assert_eq!(fast.writes, legacy.writes);
        assert_eq!(fast.steps_remaining, legacy.steps_remaining);
        (fast.writes, fast.steps_remaining)
    }

    fn add_immediate(result: u32, dst: u32, src_a: u32, immediate: u32) -> u32 {
        1 | (result << 4) | (dst << 8) | (src_a << 11) | (immediate << 14)
    }

    fn set_method(method: u32) -> u32 {
        add_immediate(2, 1, 0, method)
    }

    fn send_immediate(value: u32) -> u32 {
        add_immediate(4, 2, 0, value)
    }

    fn read_and_send(address: u32) -> u32 {
        5 | (4 << 4) | (2 << 8) | (address << 14)
    }

    #[test]
    fn compiled_macro_and_parameter_storage_are_reused() {
        let mut engine = MacroEngine::new();
        let code = [0x0000_0090, 0x0000_0010, 0xDEAD_BEEF];
        upload_entry_11(&mut engine, &code);

        let first = invoke_entry_11(&mut engine);
        let compiled = engine.compiled.get(&0).unwrap();
        let code_ptr = compiled.code.as_ptr();
        let params_capacity = engine.pending_params.capacity();
        assert_eq!(first.hash, macro_hash(&code));
        assert_eq!(compiled.hash, first.hash);
        assert!(compiled.hle.is_none());
        assert!(params_capacity >= 6);

        let second = invoke_entry_11(&mut engine);
        let compiled = engine.compiled.get(&0).unwrap();
        assert_eq!(compiled.code.as_ptr(), code_ptr);
        assert_eq!(second.hash, first.hash);
        assert_eq!(engine.pending_params.capacity(), params_capacity);
        assert!(engine.pending_params.is_empty());
    }

    #[test]
    fn sliced_and_scalar_parameter_delivery_match() {
        let code = [
            0x0480_0221,
            0x0000_0A30,
            0x0000_1330,
            0x0000_1BC0,
            0x0000_0010,
        ];
        let mut scalar = MacroEngine::new();
        let mut sliced = MacroEngine::new();
        upload_entry_11(&mut scalar, &code);
        upload_entry_11(&mut sliced, &code);

        assert!(scalar.on_macro_method(0xE17, 0x11, false, &|_| 0).is_none());
        assert!(scalar.on_macro_method(0xE17, 0x22, false, &|_| 0).is_none());
        let scalar = scalar.on_macro_method(0xE17, 0x33, true, &|_| 0).unwrap();

        assert!(sliced
            .on_macro_methods(0xE17, &[0x11, 0x22], false, &|_| 0)
            .is_none());
        let sliced = sliced.on_macro_method(0xE17, 0x33, true, &|_| 0).unwrap();

        assert_eq!(sliced.writes, scalar.writes);
        assert_eq!(sliced.draw_instance_count, scalar.draw_instance_count);
        assert_eq!(sliced.hash, scalar.hash);
        assert_eq!(sliced.entry, scalar.entry);
        assert_eq!(sliced.hle, scalar.hle);
    }

    #[test]
    fn replacing_uploaded_code_invalidates_compiled_metadata() {
        let mut engine = MacroEngine::new();
        let original = [0x0000_0090, 0x0000_0010];
        upload_entry_11(&mut engine, &original);
        let original_hash = invoke_entry_11(&mut engine).hash;

        let replacement = [0x0000_0090, 0x0000_0010, 0x1234_5678];
        upload_entry_11(&mut engine, &replacement);
        assert!(engine.compiled.is_empty());

        let replacement_hash = invoke_entry_11(&mut engine).hash;
        assert_eq!(replacement_hash, macro_hash(&replacement));
        assert_ne!(replacement_hash, original_hash);
    }

    #[test]
    fn fast_lle_is_enabled_by_default_with_explicit_rollback_values() {
        assert!(enabled_by_default(None));
        for value in ["1", "true", "TRUE", " yes ", "On", "enabled"] {
            assert!(enabled_by_default(Some(value)));
        }
        for value in ["", "0", "false", " no ", "off"] {
            assert!(!enabled_by_default(Some(value)));
        }
    }

    #[test]
    fn decoded_plans_are_only_created_for_fast_lle() {
        let code = [0x0000_0090, 0x0000_0010];
        let mut legacy = MacroEngine::new_with_options(false, false);
        upload_entry_11(&mut legacy, &code);
        let legacy_output = invoke_entry_11(&mut legacy);
        assert!(!legacy_output.fast_lle);
        assert!(legacy.compiled.get(&0).unwrap().decoded.is_none());

        let mut fast = MacroEngine::new_with_options(true, false);
        upload_entry_11(&mut fast, &code);
        let fast_output = invoke_entry_11(&mut fast);
        assert!(fast_output.fast_lle);
        assert_eq!(
            fast.compiled
                .get(&0)
                .unwrap()
                .decoded
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(fast_output.writes, legacy_output.writes);
    }

    #[test]
    fn fast_lle_preserves_branch_delay_and_exit_slots() {
        let non_annul = [
            7 | (3 << 14),
            send_immediate(0x11),
            send_immediate(0x22),
            0x0000_0090,
            send_immediate(0x33),
        ];
        let (writes, _) = assert_fast_matches_legacy(&non_annul, &[], &|_| 0);
        assert_eq!(writes, vec![(0, 0x11), (0, 0x33)]);

        let annul = [
            7 | (1 << 5) | (2 << 14),
            send_immediate(0x44),
            0x0000_0090,
            send_immediate(0x55),
        ];
        let (writes, _) = assert_fast_matches_legacy(&annul, &[], &|_| 0);
        assert_eq!(writes, vec![(0, 0x55)]);
    }

    #[test]
    fn fast_lle_preserves_step_cap() {
        let runaway = [7 | (1 << 5)];
        let (writes, steps_remaining) = assert_fast_matches_legacy(&runaway, &[], &|_| 0);
        assert!(writes.is_empty());
        assert_eq!(steps_remaining, 0);
    }

    #[test]
    fn fast_lle_latest_write_lookup_preserves_d00_special_order() {
        let code = [
            set_method(0xD00),
            send_immediate(7),
            set_method(0x8C4),
            send_immediate(9),
            set_method(0x700),
            read_and_send(0xD00),
            set_method(0xD00),
            send_immediate(3),
            set_method(0x701),
            read_and_send(0xD00),
            0x0000_0090,
            0x0000_0010,
        ];
        let (writes, _) = assert_fast_matches_legacy(&code, &[], &|_| 0xDEAD_BEEF);
        assert_eq!(
            writes,
            vec![(0xD00, 7), (0x8C4, 9), (0x700, 1), (0xD00, 3), (0x701, 3)]
        );
    }

    #[test]
    fn fast_lle_invalidates_plans_on_upload_bind_and_pointer_reset() {
        let code = [0x0000_0090, 0x0000_0010];
        let mut engine = MacroEngine::new_with_options(true, false);
        upload_entry_11(&mut engine, &code);
        let output = invoke_entry_11(&mut engine);
        engine.recycle_output(output);
        assert!(engine.compiled.contains_key(&0));

        engine.upload_instruction(0x1234_5678);
        assert!(engine.compiled.is_empty());
        let output = invoke_entry_11(&mut engine);
        engine.recycle_output(output);
        assert!(engine.compiled.contains_key(&0));

        engine.set_start_address_ptr(11);
        engine.bind_macro_entry(0);
        assert!(engine.compiled.is_empty());
        let output = invoke_entry_11(&mut engine);
        engine.recycle_output(output);
        assert!(engine.compiled.contains_key(&0));

        engine.set_instruction_ptr(0);
        assert!(engine.compiled.is_empty());
    }

    #[test]
    fn fast_lle_resolves_overlapping_uploads_by_latest_write() {
        let mut engine = MacroEngine::new_with_options(true, false);
        engine.set_instruction_ptr(0);
        for word in [0xA0, 0xB0, 0xC0, 0xD0] {
            engine.upload_instruction(word);
        }
        engine.set_instruction_ptr(2);
        for word in [0xC1, 0xD1] {
            engine.upload_instruction(word);
        }

        engine.resolve_code(1);
        assert_eq!(engine.compiled.get(&1).unwrap().code, [0xB0, 0xC1, 0xD1]);
        engine.resolve_code(3);
        assert_eq!(engine.compiled.get(&3).unwrap().code, [0xD1]);
    }

    #[test]
    fn fast_lle_write_scratch_is_reused_after_replay() {
        let code = [
            set_method(0x700),
            send_immediate(0x1234),
            0x0000_0090,
            0x0000_0010,
        ];
        let mut engine = MacroEngine::new_with_options(true, false);
        upload_entry_11(&mut engine, &code);

        let first = invoke_entry_11(&mut engine);
        let first_ptr = first.writes.as_ptr();
        assert!(first.fast_lle);
        assert_eq!(first.writes, vec![(0x700, 0x1234)]);
        engine.recycle_output(first);
        assert_eq!(engine.lle_write_scratch.as_ptr(), first_ptr);

        let second = invoke_entry_11(&mut engine);
        assert_eq!(second.writes.as_ptr(), first_ptr);
        assert_eq!(second.writes, vec![(0x700, 0x1234)]);
        engine.recycle_output(second);
    }

    fn assert_draw_instanced_with_vb_mask_hle_matches_interpreter(
        code: &[u32],
        expected_hash: u64,
        indexed: bool,
        param_count: usize,
        mut random: u32,
    ) {
        let hash = macro_hash(code);
        assert_eq!(hash, expected_hash);
        let kind =
            hle_macro_kind(hash).expect("the captured macro must have an HLE implementation");
        let HleMacro::DrawInstancedWithVbMask {
            indexed: actual_indexed,
        } = kind
        else {
            panic!("captured draw macro resolved to the wrong HLE kind");
        };
        assert_eq!(actual_indexed, indexed);

        let mut next_random = || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };

        for case in 0..256u32 {
            let params = [
                next_random(),
                next_random(),
                next_random() % 4,
                next_random(),
                next_random(),
                next_random(),
            ];
            let params = &params[..param_count];
            let vertex_buffer_mask = match case % 8 {
                0 => 0,
                1 => 1,
                2 => 1 << 31,
                3 => u32::MAX,
                _ => next_random(),
            };
            let draw_topology = next_random();
            let reg_reader = |reg| match reg {
                REG_MME_SCRATCH_DRAW_TOPOLOGY => draw_topology,
                REG_MME_SCRATCH_VERTEX_BUFFER_MASK => vertex_buffer_mask,
                _ => 0xC001_C0DE,
            };

            let mut interpreter = Interpreter::new(code, params, &reg_reader);
            interpreter.run();
            let decoded = decoded(code);
            let mut fast = FastInterpreter::new(&decoded, params, &reg_reader, Vec::new());
            fast.run();
            let hle = hle_macro(kind, params, &reg_reader)
                .expect("bounded randomized draws must use the HLE path");

            assert_eq!(fast.writes, interpreter.writes);
            assert_eq!(fast.steps_remaining, interpreter.steps_remaining);
            assert_eq!(
                hle.writes, interpreter.writes,
                "write mismatch for hash {hash:#018x}, case {case}, params={params:08x?}, mask={vertex_buffer_mask:#010x}, topology={draw_topology:#010x}"
            );
            assert_eq!(hle.draw_instance_count, None);
        }

        let oversized_params = [0, 1, 129, 0, 0, 0];
        assert!(hle_macro(kind, &oversized_params[..param_count], &|_| 0).is_none());
    }

    #[test]
    fn draw_arrays_instanced_with_vb_mask_hle_matches_interpreter() {
        assert_draw_instanced_with_vb_mask_hle_matches_interpreter(
            &DRAW_ARRAYS_INSTANCED_WITH_VB_MASK_CODE,
            0x62AB_88C4_D2DF_58E3,
            false,
            5,
            0x51A2_7E10,
        );
    }

    #[test]
    fn draw_indexed_instanced_with_vb_mask_hle_matches_interpreter() {
        assert_draw_instanced_with_vb_mask_hle_matches_interpreter(
            &DRAW_INDEXED_INSTANCED_WITH_VB_MASK_CODE,
            0xD00E_2028_0475_8F38,
            true,
            6,
            0xA5A5_1234,
        );
    }

    #[test]
    #[ignore = "microbenchmark; run explicitly with --ignored --nocapture"]
    fn benchmark_draw_instanced_with_vb_mask_hle() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        fn measure(
            code: &[u32],
            params: &[u32],
            kind: HleMacro,
            iterations: u32,
        ) -> (Duration, Duration) {
            let reg_reader = |reg| match reg {
                REG_MME_SCRATCH_DRAW_TOPOLOGY => 3,
                REG_MME_SCRATCH_VERTEX_BUFFER_MASK => 0,
                _ => 0,
            };

            let interpreter_start = Instant::now();
            for _ in 0..iterations {
                let mut interpreter =
                    Interpreter::new(black_box(code), black_box(params), &reg_reader);
                interpreter.run();
                black_box(interpreter.writes);
            }
            let interpreter_elapsed = interpreter_start.elapsed();

            let hle_start = Instant::now();
            for _ in 0..iterations {
                let output = hle_macro(kind, black_box(params), &reg_reader).unwrap();
                black_box(output.writes);
            }

            (interpreter_elapsed, hle_start.elapsed())
        }

        let iterations = 100_000;
        let arrays_params = [3, 6, 1, 0, 0];
        let indexed_params = [3, 6, 1, 0, 0, 0];
        for (name, code, params) in [
            (
                "arrays-instanced",
                DRAW_ARRAYS_INSTANCED_WITH_VB_MASK_CODE.as_slice(),
                arrays_params.as_slice(),
            ),
            (
                "indexed-instanced",
                DRAW_INDEXED_INSTANCED_WITH_VB_MASK_CODE.as_slice(),
                indexed_params.as_slice(),
            ),
        ] {
            let kind = hle_macro_kind(macro_hash(code)).unwrap();
            let (interpreter_elapsed, hle_elapsed) = measure(code, params, kind, iterations);
            eprintln!(
                "{name} macro: interpreter={:.1} ns/call, hle={:.1} ns/call, speedup={:.2}x",
                interpreter_elapsed.as_nanos() as f64 / iterations as f64,
                hle_elapsed.as_nanos() as f64 / iterations as f64,
                interpreter_elapsed.as_secs_f64() / hle_elapsed.as_secs_f64()
            );
        }
    }
}

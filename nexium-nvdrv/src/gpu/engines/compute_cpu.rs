use super::super::GpuMappings;
use std::collections::{HashMap, HashSet};

const RZ: u8 = 0xff;
const PT: u8 = 7;
const MAX_LANE_STEPS: u32 = 200_000;
const GPU_PAGE_SIZE: u64 = 0x1_0000;
const GPU_PAGE_MASK: u64 = !(GPU_PAGE_SIZE - 1);

fn compute_cpu_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_CPU_TRACE").is_some())
}

fn compute_cpu_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("NEXIUM_NO_COMPUTE_CPU").is_some())
}

fn tic_video_backing_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_TIC_VIDEO_BACKING_TRACE").is_some())
}

fn is_pps_b6300_buffer_sust(raw: u64) -> bool {
    matches!(
        raw,
        0xeb20_0682_00f7_0700
            | 0xeb20_0682_00f7_0c08
            | 0xeb20_0382_00f7_0208
            | 0xeb20_0582_00f7_0800
            | 0xeb20_0582_00f7_0904
            | 0xeb20_0602_00f7_0304
    )
}

fn is_pps_b6300_buffer_suatom(raw: u64) -> bool {
    matches!(raw, 0xea70_0282_0022_0701 | 0xea70_0282_0022_0705)
}

fn is_pps_5dbe_buffer_sust(raw: u64) -> bool {
    matches!(
        raw,
        0xeb20_0582_00f7_0800
            | 0xeb20_0582_00f7_0904
            | 0xeb20_0a82_00f7_0800
            | 0xeb20_0a82_00f7_0a04
    )
}

fn is_pps_5dbe_buffer_suatom(raw: u64) -> bool {
    matches!(
        raw,
        0xea70_0382_0020_0502 | 0xea70_0203_00a7_0d04 | 0xea70_0583_0097_0404
    )
}

fn is_pps_scene_buffer_sust(program: u32, raw: u64) -> bool {
    matches!(
        (program, raw),
        (0x5bc00, 0xeb20_0082_00f7_0004) | (0x5a800, 0xeb20_0302_00f7_0500)
    )
}

fn linear_lane_id(local: [u32; 3], block: [u32; 3]) -> u32 {
    local[0]
        .wrapping_add(block[0].wrapping_mul(local[1].wrapping_add(block[1].wrapping_mul(local[2]))))
        & 31
}

fn subgroup_mask(special_register: u32, lane_id: u32) -> u32 {
    let equal = 1u32 << lane_id;
    let less = equal.wrapping_sub(1);
    match special_register {
        0x38 => equal,
        0x39 => less,
        0x3a => less | equal,
        0x3b => !(less | equal),
        0x3c => !less,
        _ => 0,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SyncToken {
    origin: usize,
    target: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LaneRunState {
    Ready,
    WaitingBarrier { pc: usize, epoch: u32 },
    WaitingSync { token: SyncToken },
    WaitingShuffle { pc: usize, raw: u64 },
    WaitingVote { pc: usize, raw: u64 },
    Exited,
}

struct LaneState {
    group: [u32; 3],
    local: [u32; 3],
    regs: [u32; 256],
    preds: [bool; 8],
    sync_stack: Vec<SyncToken>,
    pc: usize,
    steps: u32,
    state: LaneRunState,
}

impl LaneState {
    fn new(group: [u32; 3], local: [u32; 3]) -> Self {
        let mut preds = [false; 8];
        preds[PT as usize] = true;
        Self {
            group,
            local,
            regs: [0; 256],
            preds,
            sync_stack: Vec::new(),
            pc: 0,
            steps: 0,
            state: LaneRunState::Ready,
        }
    }
}

enum LaneYield {
    Quantum,
    Barrier {
        pc: usize,
        raw: u64,
    },
    Shuffle {
        pc: usize,
        raw: u64,
    },
    Sync {
        pc: usize,
        token: SyncToken,
    },
    Vote {
        pc: usize,
        raw: u64,
    },
    Exit,
    Fault {
        pc: usize,
        raw: u64,
        opcode: nexium_shader::Opcode,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pps5dTldSample {
    pc: usize,
    raw: u64,
    coord: [u32; 2],
    handle: u32,
    mask: u8,
    word_count: u8,
    words: [u32; 4],
}

#[derive(Default)]
struct Pps5dTrace {
    tld_reach: [u64; 2],
    tld_pass: [u64; 2],
    cull_reach: [u64; 2],
    cull_taken: [u64; 2],
    vote_reach: [u64; 2],
    vote_pass: [u64; 2],
    exit_sites: Vec<(usize, u64)>,
    exit_overflow: u64,
    tld_samples: Vec<Pps5dTldSample>,
}

impl Pps5dTrace {
    fn pair_index(pc: usize, first: usize, second: usize) -> Option<usize> {
        match pc {
            value if value == first => Some(0),
            value if value == second => Some(1),
            _ => None,
        }
    }

    fn record_instruction(&mut self, pc: usize, active: bool) {
        if let Some(index) = Self::pair_index(pc, 0x8a8, 0x970) {
            if active {
                self.tld_reach[index] += 1;
            }
        }
        if let Some(index) = Self::pair_index(pc, 0x950, 0xb18) {
            self.cull_reach[index] += 1;
            if active {
                self.cull_taken[index] += 1;
            }
        }
        if let Some(index) = Self::pair_index(pc, 0xb28, 0xe08) {
            if active {
                self.vote_reach[index] += 1;
            }
        }
    }

    fn record_tld_pass(&mut self, mut sample: Pps5dTldSample, regs: &[u32; 256]) {
        let Some(index) = Self::pair_index(sample.pc, 0x8a8, 0x970) else {
            return;
        };
        self.tld_pass[index] += 1;
        if self
            .tld_samples
            .iter()
            .filter(|existing| existing.pc == sample.pc)
            .count()
            >= 4
        {
            return;
        }
        let dest = reg_dest(sample.raw);
        sample.word_count = sample.mask.count_ones().min(4) as u8;
        for index in 0..sample.word_count as usize {
            sample.words[index] = get_reg(regs, dest.wrapping_add(index as u8));
        }
        self.tld_samples.push(sample);
    }

    fn record_vote_pass(&mut self, pc: usize, lanes: u64) {
        if let Some(index) = Self::pair_index(pc, 0xb28, 0xe08) {
            self.vote_pass[index] += lanes;
        }
    }

    fn record_exit(&mut self, pc: usize) {
        if let Some((_, count)) = self.exit_sites.iter_mut().find(|(site, _)| *site == pc) {
            *count += 1;
        } else if self.exit_sites.len() < 8 {
            self.exit_sites.push((pc, 1));
        } else {
            self.exit_overflow += 1;
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ComputeTextureState {
    pub tic_pool_gpu_va: u64,
    pub tic_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_limit: u32,
    pub tex_cb_index: u32,
}

pub fn try_execute(
    qmd: &[u32; 0x40],
    code_base: u64,
    texture: ComputeTextureState,
    renderer: Option<&nexium_gpu::Renderer>,
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &dyn Fn(u64, &[u8]) -> bool,
    launch_count: u64,
) -> bool {
    if !is_candidate(qmd) {
        return false;
    }

    let grid_x = qmd[0x0c] & 0x7fff_ffff;
    let grid_y = qmd[0x0d] & 0xffff;
    let grid_z = qmd[0x0d] >> 16;
    let block_x = qmd[0x12] >> 16;
    let block_y = qmd[0x13] & 0xffff;
    let block_z = qmd[0x13] >> 16;
    let invocations = grid_x as u64
        * grid_y.max(1) as u64
        * grid_z.max(1) as u64
        * block_x.max(1) as u64
        * block_y.max(1) as u64
        * block_z.max(1) as u64;
    let code_gpu = code_base.wrapping_add(qmd[0x08] as u64);
    let Some(code) = read_gpu_vec(mappings, mem_read, code_gpu, 0x3000) else {
        return false;
    };
    let decoded_code = decode_code(&code);
    let cbuf_data = snapshot_compute_cbufs(qmd, mappings, mem_read);

    let mut exec = ComputeExec {
        qmd,
        code: &code,
        decoded_code,
        cbuf_data,
        mappings,
        mem_read,
        mem_write,
        renderer,
        texture,
        tex_cache: HashMap::new(),
        tsc_cache: HashMap::new(),
        sust_tic_cache: HashMap::new(),
        write_page_cache: HashMap::new(),
        invalidated_pages: HashSet::new(),
        tex_trace: compute_tex_trace_enabled(qmd[0x08]),
        pps_5d_trace: (qmd[0x08] == 0x5dbe00 && compute_cpu_trace_enabled())
        .then(Pps5dTrace::default),
        tex_logs: 0,
        sust_logs: 0,
        writes: 0,
        unsupported: None,
        map_cache: std::cell::Cell::new(None),
    };

    let shared_size = (qmd[0x11] & 0x3ffff) as usize;
    let cooperative = shared_size != 0 || code_uses_warp_cooperation(&code);
    'groups: for gz in 0..grid_z.max(1) {
        for gy in 0..grid_y.max(1) {
            for gx in 0..grid_x {
                let group = [gx, gy, gz];
                if cooperative {
                    exec.run_cta(
                        group,
                        [block_x.max(1), block_y.max(1), block_z.max(1)],
                        shared_size,
                    );
                } else {
                    for lz in 0..block_z.max(1) {
                        for ly in 0..block_y.max(1) {
                            for lx in 0..block_x.max(1) {
                                exec.run_lane(group, [lx, ly, lz]);
                                if exec.unsupported.is_some() {
                                    break 'groups;
                                }
                            }
                        }
                    }
                }
                if exec.unsupported.is_some() {
                    break 'groups;
                }
            }
        }
    }

    exec.log_pps_5d_trace(qmd[0x08], code_gpu, invocations);

    if let Some((pc, raw, opcode)) = exec.unsupported {
        if launch_count < 8 || compute_cpu_trace_enabled() {
            log::warn!(
                "KeplerCompute::cpu unsupported program={:#x} code={:#x} pc={:#x} raw={:#018x} opcode={:?}",
                qmd[0x08],
                code_gpu,
                pc,
                raw,
                opcode
            );
        }
        return false;
    }

    if exec.writes != 0 {
        if launch_count < 8 || compute_cpu_trace_enabled() {
            log::warn!(
                "KeplerCompute::cpu executed program={:#x} invocations={} writes={} dirty_pages={} sust_tics={} write_pages={}",
                code_gpu,
                invocations,
                exec.writes,
                exec.invalidated_pages.len(),
                exec.sust_tic_cache.len(),
                exec.write_page_cache.len(),
            );
        }
        true
    } else {
        false
    }
}

pub(super) fn is_candidate(qmd: &[u32; 0x40]) -> bool {
    if compute_cpu_disabled() {
        return false;
    }
    let grid_x = qmd[0x0c] & 0x7fff_ffff;
    let grid_y = (qmd[0x0d] & 0xffff).max(1);
    let grid_z = (qmd[0x0d] >> 16).max(1);
    let block_x = (qmd[0x12] >> 16).max(1);
    let block_y = (qmd[0x13] & 0xffff).max(1);
    let block_z = (qmd[0x13] >> 16).max(1);
    let invocations = grid_x as u64
        * grid_y as u64
        * grid_z as u64
        * block_x as u64
        * block_y as u64
        * block_z as u64;
    invocations != 0 && invocations <= compute_cpu_invocation_limit(qmd[0x08])
}

fn compute_cpu_invocation_limit(program: u32) -> u64 {
    static ALLOWED: std::sync::OnceLock<HashSet<u32>> = std::sync::OnceLock::new();
    let allowed = ALLOWED.get_or_init(|| {
        let Ok(allow) = std::env::var("NEXIUM_COMPUTE_CPU_ALLOW") else {
            return HashSet::new();
        };
        let list = if let Some(path) = allow.strip_prefix('@') {
            std::fs::read_to_string(path).unwrap_or_default()
        } else {
            allow
        };
        list.split([',', '\n', '\r'])
            .filter_map(|part| {
                let part = part.trim().trim_start_matches("0x");
                u32::from_str_radix(part, 16).ok()
            })
            .collect()
    });
    if allowed.contains(&program) {
        return compute_cpu_invocation_limit_for(program, None);
    }
    static CONFIGURED: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let configured = *CONFIGURED.get_or_init(|| {
        std::env::var("NEXIUM_COMPUTE_CPU_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|limit| *limit != 0)
    });
    configured.unwrap_or(4_096)
}

fn compute_tex_trace_enabled(program: u32) -> bool {
    static CONFIGURED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(configured) = CONFIGURED
        .get_or_init(|| std::env::var("NEXIUM_COMPUTE_TEX_TRACE").ok())
        .as_deref()
    else {
        return false;
    };
    let configured = configured.trim();
    if configured.is_empty() || configured == "0" {
        return false;
    }
    if configured == "1" || configured.eq_ignore_ascii_case("all") {
        return true;
    }
    configured
        .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
        .filter(|value| !value.is_empty())
        .any(|value| {
            let value = value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
                .unwrap_or(value);
            u32::from_str_radix(value, 16).ok() == Some(program)
        })
}

fn compute_cpu_invocation_limit_for(program: u32, configured: Option<u64>) -> u64 {
    configured
        .filter(|limit| *limit != 0)
        .unwrap_or_else(|| match program {
            0x5dbe00 | 0xb6300 => 32_768,
            0xd3f00 => 524_288,
            0xc7000 | 0x75000 => 1_500_000,
            _ => 4_096,
        })
}

struct ComputeExec<'a> {
    qmd: &'a [u32; 0x40],
    code: &'a [u8],
    decoded_code: Vec<Option<nexium_shader::Opcode>>,
    cbuf_data: [Vec<u8>; 8],
    mappings: &'a GpuMappings,
    mem_read: &'a dyn Fn(u64, &mut [u8]) -> bool,
    mem_write: &'a dyn Fn(u64, &[u8]) -> bool,
    renderer: Option<&'a nexium_gpu::Renderer>,
    texture: ComputeTextureState,
    tex_cache: HashMap<u32, TextureData>,
    tsc_cache: HashMap<u32, CachedTsc>,
    sust_tic_cache: HashMap<u32, CachedSustTic>,
    write_page_cache: HashMap<u64, u64>,
    invalidated_pages: HashSet<u64>,
    tex_trace: bool,
    pps_5d_trace: Option<Pps5dTrace>,
    tex_logs: u32,
    sust_logs: u32,
    writes: u64,
    unsupported: Option<(usize, u64, nexium_shader::Opcode)>,
    map_cache: std::cell::Cell<Option<(u64, u64, u64)>>,
}

struct TextureData {
    raw: [u8; 32],
    tic: nexium_gpu::texture::TicEntry,
    linear: Vec<u8>,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    layers: usize,
    layer_size: usize,
}

#[derive(Clone, Copy)]
struct CachedSustTic {
    raw: [u8; 32],
    entry: nexium_gpu::texture::TicEntry,
}

#[derive(Clone, Copy)]
struct CachedTsc {
    raw: [u8; 32],
    entry: nexium_gpu::texture::TscEntry,
}

impl ComputeExec<'_> {
    fn run_lane(&mut self, group: [u32; 3], local: [u32; 3]) {
        let mut lane = LaneState::new(group, local);
        let mut shared = [];
        loop {
            match self.resume_lane(&mut lane, &mut shared, false) {
                LaneYield::Quantum => {}
                LaneYield::Barrier { pc, raw } => {
                    self.unsupported = Some((pc, raw, nexium_shader::Opcode::BAR));
                    return;
                }
                LaneYield::Shuffle { pc, raw } => {
                    self.unsupported = Some((pc, raw, nexium_shader::Opcode::SHFL));
                    return;
                }
                LaneYield::Sync { pc, .. } => {
                    self.unsupported = Some((pc, 0, nexium_shader::Opcode::SYNC));
                    return;
                }
                LaneYield::Vote { pc, raw } => {
                    self.unsupported = Some((pc, raw, nexium_shader::Opcode::VOTE));
                    return;
                }
                LaneYield::Exit => return,
                LaneYield::Fault { pc, raw, opcode } => {
                    self.unsupported = Some((pc, raw, opcode));
                    return;
                }
            }
        }
    }

    fn run_cta(&mut self, group: [u32; 3], block: [u32; 3], shared_size: usize) {
        let lane_count = block[0] as usize * block[1] as usize * block[2] as usize;
        let mut lanes = Vec::with_capacity(lane_count);
        for lz in 0..block[2] {
            for ly in 0..block[1] {
                for lx in 0..block[0] {
                    lanes.push(LaneState::new(group, [lx, ly, lz]));
                }
            }
        }
        let mut shared = vec![0u8; shared_size];
        let mut epoch = 0u32;

        loop {
            for lane in &mut lanes {
                if lane.state != LaneRunState::Ready {
                    continue;
                }
                match self.resume_lane(lane, &mut shared, true) {
                    LaneYield::Quantum => {}
                    LaneYield::Barrier { pc, raw } => {
                        lane.state = LaneRunState::WaitingBarrier { pc, epoch };
                        if !is_supported_barrier(raw) {
                            self.unsupported = Some((pc, raw, nexium_shader::Opcode::BAR));
                            return;
                        }
                    }
                    LaneYield::Shuffle { pc, raw } => {
                        lane.state = LaneRunState::WaitingShuffle { pc, raw };
                    }
                    LaneYield::Sync { token, .. } => {
                        lane.state = LaneRunState::WaitingSync { token };
                    }
                    LaneYield::Vote { pc, raw } => {
                        lane.state = LaneRunState::WaitingVote { pc, raw };
                    }
                    LaneYield::Exit => lane.state = LaneRunState::Exited,
                    LaneYield::Fault { pc, raw, opcode } => {
                        self.unsupported = Some((pc, raw, opcode));
                        return;
                    }
                }
            }

            if lanes.iter().all(|lane| lane.state == LaneRunState::Exited) {
                return;
            }

            let mut resolved_warp = false;
            for warp in lanes.chunks_mut(32) {
                let mut vote_ticket = None;
                let mut vote_lanes = 0usize;
                let mut vote_ready = true;
                for lane in warp.iter() {
                    match lane.state {
                        LaneRunState::WaitingVote { pc, raw } => {
                            vote_lanes += 1;
                            let lane_ticket = (pc, raw);
                            if vote_ticket.is_none() {
                                vote_ticket = Some(lane_ticket);
                            } else if vote_ticket != Some(lane_ticket) {
                                vote_ready = false;
                            }
                        }
                        LaneRunState::WaitingSync { .. } | LaneRunState::Exited => {}
                        _ => vote_ready = false,
                    }
                }
                if vote_ready && vote_lanes != 0 {
                    let (pc, raw) = vote_ticket.unwrap();
                    if !apply_warp_vote(warp, pc, raw) {
                        self.unsupported = Some((pc, raw, nexium_shader::Opcode::VOTE));
                        return;
                    }
                    if let Some(trace) = &mut self.pps_5d_trace {
                        trace.record_vote_pass(pc, vote_lanes as u64);
                    }
                    for lane in warp.iter_mut() {
                        if matches!(lane.state, LaneRunState::WaitingVote { pc: lane_pc, raw: lane_raw } if lane_pc == pc && lane_raw == raw)
                        {
                            lane.state = LaneRunState::Ready;
                        }
                    }
                    resolved_warp = true;
                    continue;
                }

                let mut shuffle_ticket = None;
                let mut shuffle_lanes = 0usize;
                let mut shuffle_ready = true;
                for lane in warp.iter() {
                    match lane.state {
                        LaneRunState::WaitingShuffle { pc, raw } => {
                            shuffle_lanes += 1;
                            let lane_ticket = (pc, raw);
                            if shuffle_ticket.is_none() {
                                shuffle_ticket = Some(lane_ticket);
                            } else if shuffle_ticket != Some(lane_ticket) {
                                shuffle_ready = false;
                            }
                        }
                        LaneRunState::WaitingSync { .. } | LaneRunState::Exited => {}
                        _ => shuffle_ready = false,
                    }
                }
                if shuffle_ready && shuffle_lanes != 0 {
                    let (pc, raw) = shuffle_ticket.unwrap();
                    apply_warp_shuffle(warp, raw);
                    for lane in warp.iter_mut() {
                        if matches!(lane.state, LaneRunState::WaitingShuffle { pc: lane_pc, raw: lane_raw } if lane_pc == pc && lane_raw == raw)
                        {
                            lane.state = LaneRunState::Ready;
                        }
                    }
                    resolved_warp = true;
                    continue;
                }

                if release_warp_reconvergence(warp) {
                    resolved_warp = true;
                }
            }
            if resolved_warp {
                continue;
            }

            let mut barrier_ticket = None;
            let mut all_live_waiting_barrier = true;
            let mut any_ready = false;
            for lane in &lanes {
                match lane.state {
                    LaneRunState::Ready => {
                        any_ready = true;
                        all_live_waiting_barrier = false;
                    }
                    LaneRunState::WaitingBarrier {
                        pc,
                        epoch: lane_epoch,
                    } => {
                        let lane_ticket = (pc, lane_epoch);
                        if barrier_ticket.is_none() {
                            barrier_ticket = Some(lane_ticket);
                        } else if barrier_ticket != Some(lane_ticket) {
                            self.unsupported = Some((pc, 0, nexium_shader::Opcode::BAR));
                            return;
                        }
                    }
                    LaneRunState::WaitingSync { .. }
                    | LaneRunState::WaitingShuffle { .. }
                    | LaneRunState::WaitingVote { .. } => all_live_waiting_barrier = false,
                    LaneRunState::Exited => {}
                }
            }

            if all_live_waiting_barrier {
                let Some((_, ticket_epoch)) = barrier_ticket else {
                    return;
                };
                if ticket_epoch != epoch {
                    self.unsupported = Some((0, 0, nexium_shader::Opcode::BAR));
                    return;
                }
                epoch = epoch.wrapping_add(1);
                for lane in &mut lanes {
                    if matches!(lane.state, LaneRunState::WaitingBarrier { .. }) {
                        lane.state = LaneRunState::Ready;
                    }
                }
            } else if !any_ready {
                let (pc, opcode) = lanes
                    .iter()
                    .find_map(|lane| match lane.state {
                        LaneRunState::WaitingVote { pc, .. } => {
                            Some((pc, nexium_shader::Opcode::VOTE))
                        }
                        LaneRunState::WaitingShuffle { pc, .. } => {
                            Some((pc, nexium_shader::Opcode::SHFL))
                        }
                        LaneRunState::WaitingSync { token } => {
                            Some((token.target, nexium_shader::Opcode::SYNC))
                        }
                        LaneRunState::WaitingBarrier { pc, .. } => {
                            Some((pc, nexium_shader::Opcode::BAR))
                        }
                        LaneRunState::Ready | LaneRunState::Exited => None,
                    })
                    .unwrap_or((0, nexium_shader::Opcode::BAR));
                self.unsupported = Some((pc, 0, opcode));
                return;
            }
        }
    }

    fn resume_lane(
        &mut self,
        lane: &mut LaneState,
        shared: &mut [u8],
        cooperative: bool,
    ) -> LaneYield {
        let group = lane.group;
        let local = lane.local;
        let mut regs = &mut lane.regs;
        let mut preds = &mut lane.preds;
        let sync_stack = &mut lane.sync_stack;
        let mut pc = lane.pc;

        while pc + 8 <= self.code.len() {
            if lane.steps >= MAX_LANE_STEPS {
                return LaneYield::Fault {
                    pc,
                    raw: 0,
                    opcode: nexium_shader::Opcode::NOP,
                };
            }
            lane.steps += 1;
            if pc % 0x20 == 0 {
                pc += 8;
                lane.pc = pc;
                continue;
            }
            let raw = u64::from_le_bytes(self.code[pc..pc + 8].try_into().unwrap());
            if raw == 0 {
                pc += 8;
                lane.pc = pc;
                continue;
            }
            let decoded = match self.decoded_code.get(pc / 8) {
                Some(Some(opcode)) => Some(*opcode),
                Some(None) => None,
                None => nexium_shader::decode_one(raw).map(|decoded| decoded.opcode),
            };
            let Some(opcode) = decoded else {
                return LaneYield::Fault {
                    pc,
                    raw,
                    opcode: nexium_shader::Opcode::NOP,
                };
            };
            let active = opcode == nexium_shader::Opcode::SSY || pred_active(raw, &preds);
            if let Some(trace) = &mut self.pps_5d_trace {
                trace.record_instruction(pc, active);
            }
            if !active {
                pc += 8;
                lane.pc = pc;
                continue;
            }

            use nexium_shader::Opcode::*;
            match opcode {
                NOP | DEPBAR => {}
                SSY => sync_stack.push(SyncToken {
                    origin: pc,
                    target: bra_target(pc, raw),
                }),
                SYNC => {
                    let sync_pc = pc;
                    if let Some(token) = sync_stack.pop() {
                        pc = token.target;
                        lane.pc = pc;
                        if cooperative {
                            return LaneYield::Sync { pc: sync_pc, token };
                        }
                    } else {
                        pc += 8;
                        lane.pc = pc;
                    }
                    continue;
                }
                EXIT => {
                    if let Some(trace) = &mut self.pps_5d_trace {
                        trace.record_exit(pc);
                    }
                    return LaneYield::Exit;
                }
                BRA | JMP => {
                    pc = bra_target(pc, raw);
                    lane.pc = pc;
                    continue;
                }
                S2R => {
                    let block = [
                        (self.qmd[0x12] >> 16).max(1),
                        (self.qmd[0x13] & 0xffff).max(1),
                        (self.qmd[0x13] >> 16).max(1),
                    ];
                    let lane_id = linear_lane_id(local, block);
                    let val = match bits(raw, 20, 27) as u32 {
                        0 => lane_id,
                        0x20 => local[0] | (local[1] << 16) | (local[2] << 26),
                        0x21 => local[0],
                        0x22 => local[1],
                        0x23 => local[2],
                        0x25 => group[0],
                        0x26 => group[1],
                        0x27 => group[2],
                        0x28 => self.block_dim_word(),
                        0x38..=0x3c => subgroup_mask(bits(raw, 20, 27) as u32, lane_id),
                        _ => 0,
                    };
                    set_reg(&mut regs, reg_dest(raw), val);
                }
                MOV_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                MOV_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                MOV_imm => set_reg(&mut regs, reg_dest(raw), imm20(raw) as u32),
                MOV32I => set_reg(&mut regs, reg_dest(raw), imm32(raw)),
                FADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fadd(&mut regs, raw, b);
                }
                FADD_cbuf => self.fadd(&mut regs, raw, self.cbuf_u32(raw)),
                FADD_imm => self.fadd(&mut regs, raw, float_imm20(raw).to_bits()),
                FMUL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fmul(&mut regs, raw, b);
                }
                FMUL_cbuf => self.fmul(&mut regs, raw, self.cbuf_u32(raw)),
                FMUL_imm => self.fmul(&mut regs, raw, float_imm20(raw).to_bits()),
                FMUL32I => self.fmul32i(&mut regs, raw),
                FADD32I => self.fadd32i(&mut regs, raw),
                FFMA_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_cr => {
                    let b = self.cbuf_u32(raw);
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_rc => {
                    let b = get_reg(&regs, reg_c(raw));
                    let c = self.cbuf_u32(raw);
                    self.ffma(&mut regs, raw, b, c);
                }
                FFMA_imm => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.ffma(&mut regs, raw, float_imm20(raw).to_bits(), c);
                }
                FMNMX_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    if !self.fmnmx(&mut regs, &preds, raw, b) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FMNMX_reg,
                        };
                    }
                }
                FMNMX_cbuf => {
                    let b = self.cbuf_u32(raw);
                    if !self.fmnmx(&mut regs, &preds, raw, b) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FMNMX_cbuf,
                        };
                    }
                }
                FMNMX_imm => {
                    let b = float_imm20(raw).to_bits();
                    if !self.fmnmx(&mut regs, &preds, raw, b) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FMNMX_imm,
                        };
                    }
                }
                IADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iadd(&mut regs, raw, b);
                }
                IADD_cbuf => self.iadd(&mut regs, raw, self.cbuf_u32(raw)),
                IADD_imm => self.iadd(&mut regs, raw, imm20(raw) as u32),
                IADD32I => self.iadd32i(&mut regs, raw),
                IADD3_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, b, c, true);
                }
                IADD3_cbuf => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, self.cbuf_u32(raw), c, false);
                }
                IADD3_imm => {
                    let c = get_reg(&regs, reg_c(raw));
                    self.iadd3(&mut regs, raw, imm20(raw) as u32, c, false);
                }
                ISCADD_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iscadd(&mut regs, raw, b);
                }
                ISCADD_cbuf => self.iscadd(&mut regs, raw, self.cbuf_u32(raw)),
                ISCADD_imm => self.iscadd(&mut regs, raw, imm20(raw) as u32),
                XMAD_reg => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = get_reg(&regs, reg_b(raw));
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 52) as u8,
                        bit(raw, 35) as u8,
                        bit(raw, 36),
                        bit(raw, 37),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_rc => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = get_reg(&regs, reg_c(raw));
                    let c = self.cbuf_u32(raw);
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 51) as u8,
                        bit(raw, 52) as u8,
                        false,
                        false,
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_cr => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = self.cbuf_u32(raw);
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 51) as u8,
                        bit(raw, 52) as u8,
                        bit(raw, 55),
                        bit(raw, 56),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                XMAD_imm => {
                    let a = get_reg(&regs, reg_a(raw));
                    let b = bits(raw, 20, 35) as u32;
                    let c = get_reg(&regs, reg_c(raw));
                    let v = xmad_eval(
                        raw,
                        a,
                        b,
                        c,
                        bits(raw, 50, 52) as u8,
                        0,
                        bit(raw, 36),
                        bit(raw, 37),
                    );
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                SHL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.ishl(&mut regs, raw, b);
                }
                SHL_cbuf => self.ishl(&mut regs, raw, self.cbuf_u32(raw)),
                SHL_imm => self.ishl(&mut regs, raw, imm20(raw) as u32),
                SHF_l_reg => {
                    if !shf_l_reg(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: SHF_l_reg,
                        };
                    }
                }
                SHR_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.ishr(&mut regs, raw, b);
                }
                SHR_cbuf => self.ishr(&mut regs, raw, self.cbuf_u32(raw)),
                SHR_imm => self.ishr(&mut regs, raw, imm20(raw) as u32),
                LOP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.lop(
                        &mut regs,
                        raw,
                        b,
                        lop_op(raw),
                        lop_not_a(raw),
                        lop_not_b(raw),
                    );
                }
                LOP_cbuf => self.lop(
                    &mut regs,
                    raw,
                    self.cbuf_u32(raw),
                    lop_op(raw),
                    lop_not_a(raw),
                    lop_not_b(raw),
                ),
                LOP_imm => self.lop(&mut regs, raw, imm20(raw) as u32, lop_op(raw), false, false),
                LOP32I => self.lop(
                    &mut regs,
                    raw,
                    imm32(raw),
                    lop32i_op(raw),
                    lop32i_not_a(raw),
                    lop32i_not_b(raw),
                ),
                LOP3_imm => {
                    let Some(v) = lop3_imm_f8(&regs, raw) else {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: LOP3_imm,
                        };
                    };
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                BFE_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.bfe(&mut regs, raw, b);
                }
                BFE_cbuf => self.bfe(&mut regs, raw, self.cbuf_u32(raw)),
                BFE_imm => self.bfe(&mut regs, raw, imm20(raw) as u32),
                FLO_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    if !flo(&mut regs, raw, src) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FLO_reg,
                        };
                    }
                }
                FLO_cbuf => {
                    if !flo(&mut regs, raw, self.cbuf_u32(raw)) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FLO_cbuf,
                        };
                    }
                }
                FLO_imm => {
                    if !flo(&mut regs, raw, imm20(raw) as u32) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: FLO_imm,
                        };
                    }
                }
                POPC_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    popc(&mut regs, raw, src);
                }
                POPC_cbuf => {
                    let src = self.cbuf_u32(raw);
                    popc(&mut regs, raw, src);
                }
                POPC_imm => popc(&mut regs, raw, imm20(raw) as u32),
                F2F_reg => {
                    if !f2f_reg_f32_floor(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: F2F_reg,
                        };
                    }
                }
                F2I_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    self.f2i(&mut regs, raw, src);
                }
                F2I_cbuf => self.f2i(&mut regs, raw, self.cbuf_u32(raw)),
                F2I_imm => self.f2i(&mut regs, raw, float_imm20(raw).to_bits()),
                I2F_reg => {
                    let src = get_reg(&regs, reg_b(raw));
                    self.i2f(&mut regs, raw, src);
                }
                I2F_cbuf => self.i2f(&mut regs, raw, self.cbuf_u32(raw)),
                I2F_imm => self.i2f(&mut regs, raw, imm20(raw) as u32),
                ISET_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.iset(&mut regs, raw, b);
                }
                ISET_cbuf => self.iset(&mut regs, raw, self.cbuf_u32(raw)),
                ISET_imm => self.iset(&mut regs, raw, imm20(raw) as u32),
                IMNMX_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    if !self.imnmx(&mut regs, &preds, raw, b) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: IMNMX_reg,
                        };
                    }
                }
                IMNMX_cbuf => {
                    let b = self.cbuf_u32(raw);
                    if !self.imnmx(&mut regs, &preds, raw, b) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: IMNMX_cbuf,
                        };
                    }
                }
                IMNMX_imm => {
                    if !self.imnmx(&mut regs, &preds, raw, imm20(raw) as u32) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: IMNMX_imm,
                        };
                    }
                }
                ISETP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.isetp(&mut regs, &mut preds, raw, b);
                }
                ISETP_cbuf => self.isetp(&mut regs, &mut preds, raw, self.cbuf_u32(raw)),
                ISETP_imm => self.isetp(&mut regs, &mut preds, raw, imm20(raw) as u32),
                FSETP_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.fsetp(&mut regs, &mut preds, raw, b);
                }
                FSETP_cbuf => self.fsetp(&mut regs, &mut preds, raw, self.cbuf_u32(raw)),
                FSETP_imm => self.fsetp(&mut regs, &mut preds, raw, float_imm20(raw).to_bits()),
                HSETP2_reg => {
                    if !hsetp2_reg_pps(&regs, &mut preds, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: HSETP2_reg,
                        };
                    }
                }
                HADD2_imm => {
                    if !pps_hadd2_imm(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: HADD2_imm,
                        };
                    }
                }
                PSETP => self.psetp(&mut preds, raw),
                SEL_reg => {
                    let b = get_reg(&regs, reg_b(raw));
                    self.sel(&mut regs, &preds, raw, b);
                }
                SEL_cbuf => self.sel(&mut regs, &preds, raw, self.cbuf_u32(raw)),
                SEL_imm => self.sel(&mut regs, &preds, raw, imm20(raw) as u32),
                MUFU => self.mufu(&mut regs, raw),
                RRO_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                RRO_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_reg => {
                    let v = get_reg(&regs, reg_b(raw));
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_cbuf => {
                    let v = self.cbuf_u32(raw);
                    set_reg(&mut regs, reg_dest(raw), v);
                }
                I2I_imm => set_reg(&mut regs, reg_dest(raw), imm20(raw) as u32),
                LDS => {
                    if !shared_load(&mut regs, raw, shared) {
                        if compute_cpu_trace_enabled() {
                            log::warn!(
                                "KeplerCompute::shared-load-fault pc={:#x} group={:?} local={:?} base=R{}:{:#x} imm={:#x} addr={:#x} width={:?} shared={:#x} preds={:?}",
                                pc,
                                group,
                                local,
                                reg_a(raw),
                                get_reg(&regs, reg_a(raw)),
                                bits(raw, 20, 43),
                                shared_address(&regs, raw),
                                shared_access_width(raw),
                                shared.len(),
                                &preds[..7]
                            );
                        }
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: LDS,
                        };
                    }
                    pc += 8;
                    lane.pc = pc;
                    return LaneYield::Quantum;
                }
                STS => {
                    if let Some(next_pc) = pps_2513900_shared_clear_end(
                        self.qmd[0x08],
                        self.code,
                        pc,
                        raw,
                        shared.len(),
                    ) {
                        pc = next_pc;
                        lane.pc = pc;
                        continue;
                    }
                    if !shared_store(&regs, raw, shared) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: STS,
                        };
                    }
                    pc += 8;
                    lane.pc = pc;
                    return LaneYield::Quantum;
                }
                BAR => {
                    let barrier_pc = pc;
                    pc += 8;
                    lane.pc = pc;
                    return LaneYield::Barrier {
                        pc: barrier_pc,
                        raw,
                    };
                }
                SHFL => {
                    let shuffle_pc = pc;
                    pc += 8;
                    lane.pc = pc;
                    if cooperative {
                        return LaneYield::Shuffle {
                            pc: shuffle_pc,
                            raw,
                        };
                    }
                    return LaneYield::Fault {
                        pc: shuffle_pc,
                        raw,
                        opcode: SHFL,
                    };
                }
                VOTE => {
                    let vote_pc = pc;
                    pc += 8;
                    lane.pc = pc;
                    if cooperative && bits(raw, 48, 49) <= 2 {
                        return LaneYield::Vote { pc: vote_pc, raw };
                    }
                    return LaneYield::Fault {
                        pc: vote_pc,
                        raw,
                        opcode: VOTE,
                    };
                }
                LDC => {
                    if !self.ldc_pps_5dbe(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: LDC,
                        };
                    }
                }
                LDG => self.ldg(&mut regs, raw),
                STG => self.stg(&regs, raw),
                SUATOM => {
                    if !self.suatom(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: SUATOM,
                        };
                    }
                }
                SUST => {
                    if !self.sust(&regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: SUST,
                        };
                    }
                }
                TEXS | TLDS | TLD4S => self.texs(&mut regs, raw),
                TEX_b => {
                    if !self.tex_b(&mut regs, raw) {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: TEX_b,
                        };
                    }
                }
                TLD_b => {
                    let trace_sample = (self.pps_5d_trace.is_some() && matches!(pc, 0x8a8 | 0x970))
                        .then(|| {
                            let coord_reg = reg_a(raw);
                            Pps5dTldSample {
                                pc,
                                raw,
                                coord: [
                                    get_reg(&regs, coord_reg),
                                    if bits(raw, 28, 30) == 0 {
                                        0
                                    } else {
                                        get_reg(&regs, coord_reg.wrapping_add(1))
                                    },
                                ],
                                handle: get_reg(&regs, reg_b(raw)),
                                mask: bits(raw, 31, 34) as u8,
                                ..Pps5dTldSample::default()
                            }
                        });
                    let passed = self.tld(&mut regs, raw);
                    if passed {
                        if let (Some(trace), Some(sample)) = (&mut self.pps_5d_trace, trace_sample)
                        {
                            trace.record_tld_pass(sample, &regs);
                        }
                    } else {
                        return LaneYield::Fault {
                            pc,
                            raw,
                            opcode: TLD_b,
                        };
                    }
                }
                other => {
                    return LaneYield::Fault {
                        pc,
                        raw,
                        opcode: other,
                    };
                }
            }
            pc += 8;
            lane.pc = pc;
        }
        LaneYield::Exit
    }

    fn log_pps_5d_trace(&self, program: u32, code_gpu: u64, invocations: u64) {
        let Some(trace) = &self.pps_5d_trace else {
            return;
        };
        let exits = trace
            .exit_sites
            .iter()
            .map(|(pc, count)| format!("{pc:#x}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        let samples = trace
            .tld_samples
            .iter()
            .map(|sample| {
                format!(
                    "pc={:#x} raw={:#018x} coord=({:#x},{:#x}) handle={:#x} mask={:#x} words={:x?}",
                    sample.pc,
                    sample.raw,
                    sample.coord[0],
                    sample.coord[1],
                    sample.handle,
                    sample.mask,
                    &sample.words[..sample.word_count as usize],
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        log::warn!(
            "KeplerCompute::cpu-5dbe summary program={:#x} code={:#x} invocations={} writes={} unsupported={:?} tld[0x8a8]=reach:{} pass:{} tld[0x970]=reach:{} pass:{} cull_sync[0x950]=reach:{} taken:{} continue:{} cull_sync[0xb18]=reach:{} taken:{} continue:{} vote[0xb28]=reach:{} pass:{} vote[0xe08]=reach:{} pass:{} exits=[{}] exit_overflow={} tld_samples=[{}]",
            program,
            code_gpu,
            invocations,
            self.writes,
            self.unsupported,
            trace.tld_reach[0],
            trace.tld_pass[0],
            trace.tld_reach[1],
            trace.tld_pass[1],
            trace.cull_reach[0],
            trace.cull_taken[0],
            trace.cull_reach[0].saturating_sub(trace.cull_taken[0]),
            trace.cull_reach[1],
            trace.cull_taken[1],
            trace.cull_reach[1].saturating_sub(trace.cull_taken[1]),
            trace.vote_reach[0],
            trace.vote_pass[0],
            trace.vote_reach[1],
            trace.vote_pass[1],
            exits,
            trace.exit_overflow,
            samples,
        );
    }

    fn block_dim_word(&self) -> u32 {
        (self.qmd[0x12] >> 16) | ((self.qmd[0x13] & 0xffff) << 16)
    }

    fn cbuf_u32(&self, raw: u64) -> u32 {
        let c = nexium_shader::cbuf(raw);
        self.cbuf_slot_u32(c.binding, c.byte_offset)
    }

    fn cbuf_slot_u32(&self, slot: u8, offset: u32) -> u32 {
        if let Some(buffer) = self.cbuf_data.get(slot as usize) {
            let offset = offset as usize;
            if let Some(bytes) = buffer.get(offset..offset.saturating_add(4)) {
                return u32::from_le_bytes(bytes.try_into().unwrap());
            }
        }
        let base = 0x1d + slot as usize * 2;
        if base + 1 >= self.qmd.len() {
            return 0;
        }
        let lo = self.qmd[base] as u64;
        let hi_size = self.qmd[base + 1];
        let gpu = (((hi_size & 0xff) as u64) << 32) | lo;
        self.read_gpu_u32(gpu.wrapping_add(offset as u64))
    }

    fn resolve_gpu(&self, gpu: u64) -> Option<u64> {
        if let Some((start, end, cpu_base)) = self.map_cache.get() {
            if gpu >= start && gpu < end {
                return Some(cpu_base + (gpu - start));
            }
        }
        if let Some((start, size, cpu_base)) = self.mappings.mapping_at(gpu) {
            self.map_cache.set(Some((start, start + size, cpu_base)));
            return Some(cpu_base + (gpu - start));
        }
        self.mappings
            .cpu_address_for_any32(gpu)
            .map(|(_, cpu, _)| cpu)
    }

    fn read_gpu_u32(&self, gpu: u64) -> u32 {
        let Some(cpu) = self.resolve_gpu(gpu) else {
            return 0;
        };
        let mut b = [0u8; 4];
        if (self.mem_read)(cpu, &mut b) {
            u32::from_le_bytes(b)
        } else {
            0
        }
    }

    fn map_gpu_for_write(&mut self, gpu: u64) -> Option<u64> {
        let page = gpu & GPU_PAGE_MASK;
        let offset = gpu - page;
        if let Some(cpu_page) = self.write_page_cache.get(&page) {
            return cpu_page.checked_add(offset);
        }

        let cpu = map_gpu(self.mappings, gpu)?;
        let page_mapping = self.mappings.cpu_range_for(page).or_else(|| {
            self.mappings
                .cpu_address_for_any32(page)
                .map(|(_, cpu, remaining)| (cpu, remaining))
        });
        if let Some((cpu_page, remaining)) = page_mapping {
            if remaining >= GPU_PAGE_SIZE && cpu_page.checked_add(offset) == Some(cpu) {
                self.write_page_cache.insert(page, cpu_page);
            }
        }
        Some(cpu)
    }

    fn note_gpu_write(&mut self, gpu: u64, size: u64) {
        if size == 0 {
            return;
        }

        let tic_bytes = (u64::from(self.texture.tic_limit) + 1).saturating_mul(32);
        let tic_end = self.texture.tic_pool_gpu_va.saturating_add(tic_bytes);
        let write_end = gpu.saturating_add(size);
        if self.texture.tic_pool_gpu_va != 0
            && gpu < tic_end
            && self.texture.tic_pool_gpu_va < write_end
        {
            self.sust_tic_cache.clear();
        }

        let tsc_bytes = (u64::from(self.texture.tsc_limit) + 1).saturating_mul(32);
        let tsc_end = self.texture.tsc_pool_gpu_va.saturating_add(tsc_bytes);
        if self.texture.tsc_pool_gpu_va != 0
            && gpu < tsc_end
            && self.texture.tsc_pool_gpu_va < write_end
        {
            self.tsc_cache.clear();
        }

        let mut page = gpu & GPU_PAGE_MASK;
        let end = gpu.saturating_add(size).saturating_add(GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
        while page < end {
            if self.invalidated_pages.insert(page) {
                nexium_gpu::tex_invalidate::bump_region(page, 1);
            }
            let next = page.saturating_add(GPU_PAGE_SIZE);
            if next == page {
                break;
            }
            page = next;
        }
    }

    fn write_gpu_u32(&mut self, gpu: u64, value: u32) {
        let Some(cpu) = self.resolve_gpu(gpu) else {
            return;
        };
        if (self.mem_write)(cpu, &value.to_le_bytes()) {
            self.note_gpu_write(gpu, 4);
            self.writes += 1;
        }
    }

    fn ldg(&self, regs: &mut [u32; 256], raw: u64) {
        let addr_reg = reg_a(raw);
        let base = addr_pair(regs, addr_reg);
        let offset = ldg_offset(raw);
        let count = match ldg_size(raw) {
            5 => 2,
            6 | 7 => 4,
            _ => 1,
        };
        let dest = reg_dest(raw);
        for i in 0..count {
            let gpu = base
                .wrapping_add(offset as u64)
                .wrapping_add((i * 4) as u64);
            set_reg(regs, dest.wrapping_add(i as u8), self.read_gpu_u32(gpu));
        }
    }

    fn stg(&mut self, regs: &[u32; 256], raw: u64) {
        let base = addr_pair(regs, reg_a(raw));
        let gpu = base.wrapping_add(ldg_offset(raw) as u64);
        self.write_gpu_u32(gpu, get_reg(regs, reg_dest(raw)));
    }

    fn ldc_pps_5dbe(&self, regs: &mut [u32; 256], raw: u64) -> bool {
        if self.qmd[0x08] != 0x5dbe00
            || !matches!(raw, 0xef95_0040_0007_0104 | 0xef95_0040_0087_0102)
            || bits(raw, 36, 40) != 4
            || bits(raw, 44, 45) != 0
            || bits(raw, 48, 50) != 5
            || reg_dest(raw) & 1 != 0
        {
            return false;
        }
        let immediate = bits(raw, 20, 35) as u16 as i16 as i32 as u32;
        let offset = get_reg(regs, reg_a(raw)).wrapping_add(immediate);
        let dest = reg_dest(raw);
        set_reg(regs, dest, self.cbuf_slot_u32(4, offset));
        set_reg(
            regs,
            dest.wrapping_add(1),
            self.cbuf_slot_u32(4, offset.wrapping_add(4)),
        );
        true
    }

    fn suatom(&mut self, regs: &mut [u32; 256], raw: u64) -> bool {
        let operation = bits(raw, 29, 32);
        let exact_program_op =
            (self.qmd[0x08] == 0xb6300 && is_pps_b6300_buffer_suatom(raw) && operation == 0)
                || (self.qmd[0x08] == 0x5dbe00
                    && is_pps_5dbe_buffer_suatom(raw)
                    && match raw {
                        0xea70_0382_0020_0502 => operation == 0,
                        0xea70_0203_00a7_0d04 | 0xea70_0583_0097_0404 => operation == 8,
                        _ => false,
                    });
        let exact = exact_program_op
            && bits(raw, 33, 35) == 1
            && bits(raw, 49, 50) == 0
            && bits(raw, 51, 53) == 6
            && bit(raw, 54);
        if !exact {
            return false;
        }

        let handle = get_reg(regs, bits(raw, 39, 46) as u8);
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let tic_index = if linked_tsc {
            handle
        } else {
            handle & 0x000f_ffff
        };
        if tic_index > self.texture.tic_limit || self.texture.tic_pool_gpu_va == 0 {
            return false;
        }
        let tic_gpu = self
            .texture
            .tic_pool_gpu_va
            .wrapping_add((tic_index as u64).saturating_mul(32));
        let Some(tic_cpu) = map_gpu(self.mappings, tic_gpu) else {
            return false;
        };
        let mut tic_raw = [0u8; 32];
        if !(self.mem_read)(tic_cpu, &mut tic_raw) {
            return false;
        }
        let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) else {
            return false;
        };
        let supported = tic.format == nexium_gpu::texture::TicFormat::R32
            && tic.component_types[0] == nexium_gpu::texture::ComponentType::Uint
            && tic.is_buffer()
            && !tic.is_block_linear
            && !tic.normalized_coords
            && !tic.is_srgb
            && tic.height == 1
            && tic.depth == 1
            && tic.view_base_mip() == 0
            && tic.view_mip_levels() == 1;
        if !supported {
            return false;
        }

        let x = get_reg(regs, bits(raw, 8, 15) as u8);
        if x >= tic.width {
            return true;
        }
        let gpu = tic.gpu_va.wrapping_add((x as u64).saturating_mul(4));
        let Some(cpu) = self.map_gpu_for_write(gpu) else {
            return false;
        };
        let mut old_bytes = [0u8; 4];
        if !(self.mem_read)(cpu, &mut old_bytes) {
            return false;
        }
        let old = u32::from_le_bytes(old_bytes);
        let operand = get_reg(regs, bits(raw, 20, 27) as u8);
        let new = match operation {
            0 => old.wrapping_add(operand),
            8 => operand,
            _ => return false,
        };
        if !(self.mem_write)(cpu, &new.to_le_bytes()) {
            return false;
        }
        self.note_gpu_write(gpu, 4);
        set_reg(regs, reg_dest(raw), old);
        self.writes += 1;
        if self.tex_trace && self.sust_logs < 18 {
            log::warn!(
                "KeplerCompute::suatom-buffer handle={:#x} tic={} va={:#x} coord={} operand={:#x} old={:#x} new={:#x}",
                handle,
                tic_index,
                tic.gpu_va,
                x,
                operand,
                old,
                new,
            );
            self.sust_logs += 1;
        }
        true
    }

    fn trace_sust_pre_tic_failure(
        &mut self,
        stage: &str,
        raw: u64,
        handle: u32,
        linked_tsc: bool,
        tic_index: u32,
        tic_gpu: u64,
        tic_cpu: Option<u64>,
        tic_raw: Option<&[u8; 32]>,
    ) -> bool {
        if self.tex_trace && self.sust_logs < 18 {
            log::warn!(
                "KeplerCompute::sust-pre-tic stage={} program={:#x} raw={:#018x} handle={:#x} linked_tsc={} tic={} packed_tic={} packed_tsc={} pool={:#x}/{} tic_gpu={:#x} tic_cpu={:?} tic_raw={:02x?} {}",
                stage,
                self.qmd[0x08],
                raw,
                handle,
                linked_tsc,
                tic_index,
                handle & 0x000f_ffff,
                handle >> 20,
                self.texture.tic_pool_gpu_va,
                self.texture.tic_limit,
                tic_gpu,
                tic_cpu,
                tic_raw.map(|value| &value[..]).unwrap_or(&[]),
                self.mappings.bracket(tic_gpu),
            );
            self.sust_logs += 1;
        }
        false
    }

    fn sust(&mut self, regs: &[u32; 256], raw: u64) -> bool {
        const PPS_BUFFER_STORE: u64 = 0xeb20_0682_00f7_0700;
        let exact_pps_buffer_store = (self.qmd[0x08] == 0xb6300 && is_pps_b6300_buffer_sust(raw))
            || (self.qmd[0x08] == 0x5dbe00 && is_pps_5dbe_buffer_sust(raw))
            || is_pps_scene_buffer_sust(self.qmd[0x08], raw);
        let exact_pps_store = exact_pps_buffer_store
            || matches!(
                (self.qmd[0x08], raw),
                (0x74a00, 0xeb20_0306_00f7_0400)
                    | (0xc7000, 0xeb20_0306_00f7_0400)
                    | (0x2513900, 0xeb20_0006_00f7_0a04)
                    | (0x75000, 0xeb20_0806_00f0_0600)
                    | (0x75000, 0xeb20_0106_00f7_0408)
                    | (0xd3f00, 0xeb20_0486_00f7_0204)
                    | (0xd3f00, 0xeb20_0206_00f7_0208)
                    | (0xd3f00, 0xeb20_0086_00f7_0204)
                    | (0xd5600, 0xeb20_0586_00f7_0004)
                    | (0xd5600, 0xeb20_0506_00f7_0004)
            );
        let swizzle = bits(raw, 20, 23);
        let cache = bits(raw, 24, 25);
        let surface_type = bits(raw, 33, 35);
        let clamp = bits(raw, 49, 50);
        let is_bound = bit(raw, 51);
        let is_typed = bit(raw, 52);
        let guard_ok = exact_pps_store
            && swizzle == 0xf
            && cache <= 1
            && matches!(surface_type, 1 | 3)
            && clamp == 0
            && !is_bound
            && !is_typed;
        if !guard_ok {
            if self.tex_trace && self.sust_logs < 18 {
                log::warn!(
                    "KeplerCompute::sust-guard program={:#x} expected_program={:#x} raw={:#018x} expected_raw={:#018x} exact={} program_eq={} raw_eq={} swizzle={:#x}/0xf cache={}/0..1 type={}/1|3 clamp={} bound={} typed={}",
                    self.qmd[0x08],
                    0xb6300,
                    raw,
                    PPS_BUFFER_STORE,
                    exact_pps_store,
                    self.qmd[0x08] == 0xb6300,
                    raw == PPS_BUFFER_STORE,
                    swizzle,
                    cache,
                    surface_type,
                    clamp,
                    is_bound,
                    is_typed,
                );
                self.sust_logs += 1;
            }
            return false;
        }

        let handle = get_reg(regs, bits(raw, 39, 46) as u8);
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let tic_index = if linked_tsc {
            handle
        } else {
            handle & 0x000f_ffff
        };
        let tic_gpu = self
            .texture
            .tic_pool_gpu_va
            .wrapping_add((tic_index as u64).saturating_mul(32));
        if tic_index > self.texture.tic_limit || self.texture.tic_pool_gpu_va == 0 {
            return self.trace_sust_pre_tic_failure(
                "index-or-pool",
                raw,
                handle,
                linked_tsc,
                tic_index,
                tic_gpu,
                None,
                None,
            );
        }
        let (tic_raw, tic) = if let Some(cached) = self.sust_tic_cache.get(&tic_index).copied() {
            (cached.raw, cached.entry)
        } else {
            let Some(tic_cpu) = map_gpu(self.mappings, tic_gpu) else {
                return self.trace_sust_pre_tic_failure(
                    "map", raw, handle, linked_tsc, tic_index, tic_gpu, None, None,
                );
            };
            let mut tic_raw = [0u8; 32];
            if !(self.mem_read)(tic_cpu, &mut tic_raw) {
                return self.trace_sust_pre_tic_failure(
                    "read",
                    raw,
                    handle,
                    linked_tsc,
                    tic_index,
                    tic_gpu,
                    Some(tic_cpu),
                    None,
                );
            }
            let Some(tic) = nexium_gpu::texture::TicEntry::parse(&tic_raw) else {
                return self.trace_sust_pre_tic_failure(
                    "parse",
                    raw,
                    handle,
                    linked_tsc,
                    tic_index,
                    tic_gpu,
                    Some(tic_cpu),
                    Some(&tic_raw),
                );
            };
            self.sust_tic_cache.insert(
                tic_index,
                CachedSustTic {
                    raw: tic_raw,
                    entry: tic,
                },
            );
            (tic_raw, tic)
        };
        let coord = reg_a(raw);
        let x = get_reg(regs, coord);
        if bits(raw, 33, 35) == 1 {
            let component_type = tic.component_types[0];
            let buffer_layout = match tic.format {
                nexium_gpu::texture::TicFormat::R32G32B32A32 => Some((4usize, 4usize)),
                nexium_gpu::texture::TicFormat::R32 => Some((1usize, 4usize)),
                nexium_gpu::texture::TicFormat::R16
                    if component_type == nexium_gpu::texture::ComponentType::Uint =>
                {
                    Some((1usize, 2usize))
                }
                _ => None,
            };
            let supported = exact_pps_buffer_store
                && tic.is_buffer()
                && !tic.is_block_linear
                && !tic.normalized_coords
                && !tic.is_srgb
                && tic.height == 1
                && tic.depth == 1
                && tic.view_base_mip() == 0
                && tic.view_mip_levels() == 1
                && buffer_layout.is_some()
                && tic
                    .component_types
                    .iter()
                    .take(buffer_layout.map_or(0, |(components, _)| components))
                    .all(|ty| *ty == component_type)
                && matches!(
                    component_type,
                    nexium_gpu::texture::ComponentType::Float
                        | nexium_gpu::texture::ComponentType::Uint
                        | nexium_gpu::texture::ComponentType::Sint
                );
            if self.tex_trace && self.sust_logs < 18 {
                log::warn!(
                    "KeplerCompute::sust-buffer handle={:#x} tic={} raw={:02x?} va={:#x} fmt={:?} types={:?} swizzle={:?} width={} type={} coord={} data={:08x?} supported={}",
                    handle,
                    tic_index,
                    tic_raw,
                    tic.gpu_va,
                    tic.format,
                    tic.component_types,
                    tic.swizzle,
                    tic.width,
                    tic.texture_type,
                    x,
                    [
                        get_reg(regs, reg_dest(raw)),
                        get_reg(regs, reg_dest(raw).wrapping_add(1)),
                        get_reg(regs, reg_dest(raw).wrapping_add(2)),
                        get_reg(regs, reg_dest(raw).wrapping_add(3)),
                    ],
                    supported,
                );
                self.sust_logs += 1;
            }
            if !supported {
                return false;
            }
            if x >= tic.width {
                return true;
            }
            let (components, component_bytes) = buffer_layout.unwrap();
            let byte_count = components * component_bytes;
            let mut bytes = [0u8; 16];
            for component in 0..components {
                let start = component * component_bytes;
                let value = get_reg(regs, reg_dest(raw).wrapping_add(component as u8));
                if component_bytes == 2 {
                    bytes[start..start + 2].copy_from_slice(&(value as u16).to_le_bytes());
                } else {
                    bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            let gpu = tic
                .gpu_va
                .wrapping_add((x as u64).saturating_mul(byte_count as u64));
            let Some(cpu) = self.map_gpu_for_write(gpu) else {
                if self.tex_trace && self.sust_logs < 18 {
                    log::warn!(
                        "KeplerCompute::sust-buffer-write stage=map program={:#x} raw={:#018x} handle={:#x} tic={} gpu={:#x} bytes={} {}",
                        self.qmd[0x08],
                        raw,
                        handle,
                        tic_index,
                        gpu,
                        byte_count,
                        self.mappings.bracket(gpu),
                    );
                    self.sust_logs += 1;
                }
                return false;
            };
            if !(self.mem_write)(cpu, &bytes[..byte_count]) {
                if self.tex_trace && self.sust_logs < 18 {
                    log::warn!(
                        "KeplerCompute::sust-buffer-write stage=write program={:#x} raw={:#018x} handle={:#x} tic={} gpu={:#x} cpu={:#x} bytes={}",
                        self.qmd[0x08],
                        raw,
                        handle,
                        tic_index,
                        gpu,
                        cpu,
                        byte_count,
                    );
                    self.sust_logs += 1;
                }
                return false;
            }
            self.note_gpu_write(gpu, byte_count as u64);
            self.writes += 1;
            return true;
        }
        let y = get_reg(regs, coord.wrapping_add(1));
        let rgba16_float = tic.format == nexium_gpu::texture::TicFormat::R16G16B16A16
            && tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float)
            && tic.swizzle
                == [
                    nexium_gpu::texture::SwizzleSource::R,
                    nexium_gpu::texture::SwizzleSource::G,
                    nexium_gpu::texture::SwizzleSource::B,
                    nexium_gpu::texture::SwizzleSource::A,
                ];
        let b10g11r11_float = tic.format == nexium_gpu::texture::TicFormat::B10G11R11
            && tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float)
            && tic.swizzle
                == [
                    nexium_gpu::texture::SwizzleSource::R,
                    nexium_gpu::texture::SwizzleSource::G,
                    nexium_gpu::texture::SwizzleSource::B,
                    nexium_gpu::texture::SwizzleSource::One,
                ];
        let r16_float = tic.format == nexium_gpu::texture::TicFormat::R16
            && tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float)
            && tic.swizzle
                == [
                    nexium_gpu::texture::SwizzleSource::R,
                    nexium_gpu::texture::SwizzleSource::Zero,
                    nexium_gpu::texture::SwizzleSource::Zero,
                    nexium_gpu::texture::SwizzleSource::One,
                ];
        let mip_layout = nexium_gpu::texture::block_linear_mip_layout(&tic);
        let mip_level = mip_layout
            .as_ref()
            .and_then(|layout| layout.levels.get(tic.view_base_mip() as usize));
        let supported = (rgba16_float || b10g11r11_float || r16_float)
            && tic.is_block_linear
            && !tic.is_srgb
            && tic.texture_type == 1
            && tic.depth == 1
            && tic.block_width_log2 == 0
            && tic.block_depth_log2 == 0
            && tic.tile_width_spacing == 0
            && mip_level.is_some();
        if self.tex_trace && self.sust_logs < 18 {
            let view_extent = mip_level.map(|level| (level.width, level.height));
            log::warn!(
                "KeplerCompute::sust handle={:#x} tic={} raw={:02x?} va={:#x} fmt={:?} types={:?} swizzle={:?} {}x{}x{} view={:?} type={} block_linear={} bh={} srgb={} mip={}/{} coord=({},{}) data={:08x?} supported={}",
                handle,
                tic_index,
                tic_raw,
                tic.gpu_va,
                tic.format,
                tic.component_types,
                tic.swizzle,
                tic.width,
                tic.height,
                tic.depth,
                view_extent,
                tic.texture_type,
                tic.is_block_linear,
                tic.block_height_log2,
                tic.is_srgb,
                tic.view_base_mip(),
                tic.view_mip_levels(),
                x,
                y,
                [
                    get_reg(regs, reg_dest(raw)),
                    get_reg(regs, reg_dest(raw).wrapping_add(1)),
                    get_reg(regs, reg_dest(raw).wrapping_add(2)),
                    get_reg(regs, reg_dest(raw).wrapping_add(3)),
                ],
                supported,
            );
            self.sust_logs += 1;
        }
        if !supported {
            return false;
        }
        let mip_level = mip_level.unwrap();
        if x >= mip_level.width || y >= mip_level.height {
            return true;
        }

        let Some(offset) = block_linear_pixel_offset_strided(
            mip_level.storage_width,
            mip_level.storage_height,
            tic.format.src_bpp(),
            mip_level.block_height_log2,
            mip_level.stride_alignment_log2,
            x,
            y,
        ) else {
            return false;
        };
        let mut bytes = [0u8; 8];
        let byte_count = if r16_float {
            bytes[..2]
                .copy_from_slice(&f32_to_f16_bits(get_reg(regs, reg_dest(raw))).to_le_bytes());
            2
        } else if b10g11r11_float {
            bytes[..4].copy_from_slice(
                &pack_b10g11r11(
                    get_reg(regs, reg_dest(raw)),
                    get_reg(regs, reg_dest(raw).wrapping_add(1)),
                    get_reg(regs, reg_dest(raw).wrapping_add(2)),
                )
                .to_le_bytes(),
            );
            4
        } else {
            for component in 0..4 {
                let value = get_reg(regs, reg_dest(raw).wrapping_add(component as u8));
                let half = f32_to_f16_bits(value).to_le_bytes();
                bytes[component * 2..component * 2 + 2].copy_from_slice(&half);
            }
            8
        };
        let gpu = tic
            .gpu_va
            .wrapping_add(mip_level.guest_offset as u64)
            .wrapping_add(offset as u64);
        let Some(cpu) = self.map_gpu_for_write(gpu) else {
            return false;
        };
        if !(self.mem_write)(cpu, &bytes[..byte_count]) {
            return false;
        }
        self.note_gpu_write(gpu, byte_count as u64);
        self.writes += 1;
        true
    }

    fn fadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 46) {
            a = a.abs();
        }
        if bit(raw, 48) {
            a = -a;
        }
        if bit(raw, 49) {
            b = b.abs();
        }
        if bit(raw, 45) {
            b = -b;
        }
        set_reg(regs, reg_dest(raw), fsat(a + b, bit(raw, 50)).to_bits());
    }

    fn fmul(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 48) {
            b = -b;
        }
        set_reg(regs, reg_dest(raw), fsat(a * b, bit(raw, 50)).to_bits());
    }

    fn fmul32i(&self, regs: &mut [u32; 256], raw: u64) {
        let a = get_reg(regs, reg_a(raw));
        set_reg(regs, reg_dest(raw), fmul32i_value(raw, a));
    }

    fn fadd32i(&self, regs: &mut [u32; 256], raw: u64) {
        let a = get_reg(regs, reg_a(raw));
        set_reg(regs, reg_dest(raw), fadd32i_value(raw, a));
    }

    fn ffma(&self, regs: &mut [u32; 256], raw: u64, b: u32, c: u32) {
        let a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        let mut c = f32::from_bits(c);
        if bit(raw, 48) {
            b = -b;
        }
        if bit(raw, 49) {
            c = -c;
        }
        set_reg(
            regs,
            reg_dest(raw),
            fsat(a.mul_add(b, c), bit(raw, 50)).to_bits(),
        );
    }

    fn fmnmx(&self, regs: &mut [u32; 256], preds: &[bool; 8], raw: u64, b: u32) -> bool {
        let a = get_reg(regs, reg_a(raw));
        let Some(value) = fmnmx_value(a, b, raw, preds) else {
            return false;
        };
        set_reg(regs, reg_dest(raw), value);
        true
    }

    fn iadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if bit(raw, 49) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 48) {
            b = (!b).wrapping_add(1);
        }
        set_reg(regs, reg_dest(raw), a.wrapping_add(b));
    }

    fn iadd32i(&self, regs: &mut [u32; 256], raw: u64) {
        let a = get_reg(regs, reg_a(raw));
        set_reg(regs, reg_dest(raw), iadd32i_value(raw, a));
    }

    fn iadd3(&self, regs: &mut [u32; 256], raw: u64, b: u32, c: u32, use_halves: bool) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        let mut c = c;
        if use_halves {
            a = half(a, bits(raw, 35, 36) as u8);
            b = half(b, bits(raw, 33, 34) as u8);
            c = half(c, bits(raw, 31, 32) as u8);
        }
        if bit(raw, 51) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 50) {
            b = (!b).wrapping_add(1);
        }
        if bit(raw, 49) {
            c = (!c).wrapping_add(1);
        }
        let mut v = a.wrapping_add(b);
        match bits(raw, 37, 38) {
            1 => v >>= 16,
            2 => v <<= 16,
            _ => {}
        }
        set_reg(regs, reg_dest(raw), v.wrapping_add(c));
    }

    fn iscadd(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if bit(raw, 49) {
            a = (!a).wrapping_add(1);
        }
        if bit(raw, 48) {
            b = (!b).wrapping_add(1);
        }
        set_reg(
            regs,
            reg_dest(raw),
            (a << bits(raw, 39, 43)).wrapping_add(b),
        );
    }

    fn ishl(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        set_reg(regs, reg_dest(raw), get_reg(regs, reg_a(raw)) << (b & 31));
    }

    fn ishr(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let v = if bit(raw, 48) {
            ((a as i32) >> (b & 31)) as u32
        } else {
            a >> (b & 31)
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn bfe(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        set_reg(regs, reg_dest(raw), bfe_value(a, b, bit(raw, 48)));
    }

    fn lop(&self, regs: &mut [u32; 256], raw: u64, b: u32, op: u64, not_a: bool, not_b: bool) {
        let mut a = get_reg(regs, reg_a(raw));
        let mut b = b;
        if not_a {
            a = !a;
        }
        if not_b {
            b = !b;
        }
        let v = match op & 3 {
            0 => a & b,
            1 => a | b,
            2 => a ^ b,
            _ => b,
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn f2i(&self, regs: &mut [u32; 256], raw: u64, src: u32) {
        let f = f32::from_bits(src);
        let v = if bit(raw, 12) {
            f as i32 as u32
        } else {
            f as u32
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn i2f(&self, regs: &mut [u32; 256], raw: u64, src: u32) {
        let int_format = bits(raw, 10, 11) as u8;
        let selector = bits(raw, 41, 42) as u32;
        let extracted = match int_format {
            0 => (src >> (selector * 8)) & 0xff,
            1 => (src >> (selector * 8)) & 0xffff,
            _ => src,
        };
        let mut f = if bit(raw, 13) {
            sign_extend(
                extracted,
                match int_format {
                    0 => 8,
                    1 => 16,
                    _ => 32,
                },
            ) as f32
        } else {
            extracted as f32
        };
        if bit(raw, 49) {
            f = f.abs();
        }
        if bit(raw, 45) {
            f = -f;
        }
        set_reg(regs, reg_dest(raw), f.to_bits());
    }

    fn iset(&self, regs: &mut [u32; 256], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let pass = icmp(bits(raw, 49, 51), bit(raw, 48), a, b);
        let v = if bit(raw, 44) {
            if pass {
                1.0f32.to_bits()
            } else {
                0
            }
        } else if pass {
            u32::MAX
        } else {
            0
        };
        set_reg(regs, reg_dest(raw), v);
    }

    fn imnmx(&self, regs: &mut [u32; 256], preds: &[bool; 8], raw: u64, b: u32) -> bool {
        let a = get_reg(regs, reg_a(raw));
        let Some(value) = imnmx_value(a, b, preds, raw) else {
            return false;
        };
        set_reg(regs, reg_dest(raw), value);
        true
    }

    fn isetp(&self, regs: &mut [u32; 256], preds: &mut [bool; 8], raw: u64, b: u32) {
        let a = get_reg(regs, reg_a(raw));
        let cmp = icmp(bits(raw, 49, 51), bit(raw, 48), a, b);
        let src = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let yes = boolop(bits(raw, 45, 46), cmp, src);
        let no = boolop(bits(raw, 45, 46), !cmp, src);
        set_pred(preds, bits(raw, 3, 5) as u8, yes);
        set_pred(preds, bits(raw, 0, 2) as u8, no);
    }
    fn fsetp(&self, regs: &mut [u32; 256], preds: &mut [bool; 8], raw: u64, b: u32) {
        let mut a = f32::from_bits(get_reg(regs, reg_a(raw)));
        let mut b = f32::from_bits(b);
        if bit(raw, 7) {
            a = a.abs();
        }
        if bit(raw, 43) {
            a = -a;
        }
        if bit(raw, 44) {
            b = b.abs();
        }
        if bit(raw, 6) {
            b = -b;
        }
        let cmp = fcmp(bits(raw, 48, 51), a, b);
        let src = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let yes = boolop(bits(raw, 45, 46), cmp, src);
        let no = boolop(bits(raw, 45, 46), !cmp, src);
        set_pred(preds, bits(raw, 3, 5) as u8, yes);
        set_pred(preds, bits(raw, 0, 2) as u8, no);
    }

    fn psetp(&self, preds: &mut [bool; 8], raw: u64) {
        let pa = pred_value(preds, bits(raw, 12, 14) as u8, bit(raw, 15));
        let pb = pred_value(preds, bits(raw, 29, 31) as u8, bit(raw, 32));
        let pc = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let lhs_a = boolop(bits(raw, 24, 25), pa, pb);
        let lhs_b = boolop(bits(raw, 24, 25), !pa, pb);
        let result_a = boolop(bits(raw, 45, 46), lhs_a, pc);
        let result_b = boolop(bits(raw, 45, 46), lhs_b, pc);
        set_pred(preds, bits(raw, 3, 5) as u8, result_a);
        set_pred(preds, bits(raw, 0, 2) as u8, result_b);
    }

    fn sel(&self, regs: &mut [u32; 256], preds: &[bool; 8], raw: u64, b: u32) {
        let p = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
        let v = if p { get_reg(regs, reg_a(raw)) } else { b };
        set_reg(regs, reg_dest(raw), v);
    }

    fn mufu(&self, regs: &mut [u32; 256], raw: u64) {
        let x = f32::from_bits(get_reg(regs, reg_a(raw)));
        let y = match bits(raw, 20, 23) {
            0 => x.cos(),
            1 => x.sin(),
            2 => x.exp2(),
            3 => x.log2(),
            4 => 1.0 / x,
            5 => 1.0 / x.sqrt(),
            8 => x.sqrt(),
            _ => 0.0,
        };
        set_reg(regs, reg_dest(raw), y.to_bits());
    }

    fn texs(&mut self, regs: &mut [u32; 256], raw: u64) {
        let dest_a = reg_dest(raw);
        let dest_b = bits(raw, 28, 35) as u8;
        let swizzle = bits(raw, 50, 52) as usize;
        let mask = if dest_b == RZ {
            [1, 2, 4, 8, 3, 9, 10, 12][swizzle.min(7)]
        } else {
            [7, 11, 13, 14, 15].get(swizzle).copied().unwrap_or(1)
        };
        let sample = self.sample_tex(regs, raw).unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let mut store = 0;
        for component in 0..4 {
            if (mask >> component) & 1 == 0 {
                continue;
            }
            let dst = match store {
                0 => dest_a,
                1 => dest_a.wrapping_add(1),
                2 => dest_b,
                _ => dest_b.wrapping_add(1),
            };
            set_reg(regs, dst, sample[component].to_bits());
            store += 1;
        }
    }

    fn sample_tex(&mut self, regs: &[u32; 256], raw: u64) -> Option<[f32; 4]> {
        let handle_offset = (bits(raw, 36, 48) as u32).wrapping_mul(4);
        let handle = self.cbuf_slot_u32(self.texture.tex_cb_index as u8, handle_offset);
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let tic_index = if linked_tsc {
            handle
        } else {
            handle & 0x000f_ffff
        };
        if tic_index > self.texture.tic_limit || self.texture.tic_pool_gpu_va == 0 {
            return None;
        }
        if !self.tex_cache.contains_key(&tic_index) {
            let data = self.load_texture(tic_index)?;
            self.tex_cache.insert(tic_index, data);
        }
        let data = self.tex_cache.get(&tic_index)?;
        let encoding = bits(raw, 53, 56);
        let reg_a_id = reg_a(raw);
        let reg_b_id = reg_b(raw);
        let fx = |r| f32::from_bits(get_reg(regs, r));
        let (u, v, layer_f) = match encoding {
            7 | 8 | 9 => {
                let layer = (get_reg(regs, reg_a_id) & 0xffff) as f32;
                (fx(reg_a_id.wrapping_add(1)), fx(reg_b_id), layer)
            }
            10 | 11 | 12 | 13 => (fx(reg_a_id), fx(reg_a_id.wrapping_add(1)), fx(reg_b_id)),
            _ => (fx(reg_a_id), fx(reg_b_id), 0.0),
        };
        let x = coord_to_index(u, data.width, data.tic.normalized_coords);
        let y = coord_to_index(v, data.height, data.tic.normalized_coords);
        let layer = if data.layers <= 1 {
            0
        } else if data.tic.texture_type == 2 && data.tic.normalized_coords {
            coord_to_index(layer_f, data.layers as u32, true)
        } else {
            layer_f.floor().clamp(0.0, (data.layers - 1) as f32) as usize
        };
        let off = layer
            .saturating_mul(data.layer_size)
            .saturating_add((y * data.width as usize + x) * 4);
        if off + 4 > data.rgba.len() {
            return None;
        }
        let px = [
            data.rgba[off],
            data.rgba[off + 1],
            data.rgba[off + 2],
            data.rgba[off + 3],
        ];
        if self.tex_trace && self.tex_logs < 16 {
            log::warn!(
                "KeplerCompute::texs handle={:#x} tic={} fmt={:?} {}x{}x{} type={} coord=({:.4},{:.4},{:.4}) px={:?}",
                handle,
                tic_index,
                data.tic.format,
                data.width,
                data.height,
                data.layers,
                data.tic.texture_type,
                u,
                v,
                layer_f,
                px
            );
            self.tex_logs += 1;
        }
        let px = apply_swizzle(px, data.tic.swizzle);
        Some([
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
            px[3] as f32 / 255.0,
        ])
    }

    fn load_texture(&self, tic_index: u32) -> Option<TextureData> {
        let tic_gpu = self
            .texture
            .tic_pool_gpu_va
            .wrapping_add((tic_index as u64).saturating_mul(32));
        let tic_cpu = map_gpu(self.mappings, tic_gpu)?;
        let mut tic_raw = [0u8; 32];
        if !(self.mem_read)(tic_cpu, &mut tic_raw) {
            return None;
        }
        let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw)?;
        if tic_video_backing_trace_enabled() {
            let read_size = nexium_gpu::texture::texture_guest_size_bytes(&tic, 1)
                .unwrap_or_else(|| tic.format.linear_size(tic.width, tic.height));
            if let Some((target, cpu_va, nvmap_id)) =
                super::super::vk_dispatch::video_tic_cpu_target_match(
                    self.mappings,
                    tic.gpu_va,
                    read_size as u64,
                )
            {
                use std::sync::{Mutex, OnceLock};
                static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u64)>>> = OnceLock::new();
                if let Ok(mut seen) = SEEN.get_or_init(|| Mutex::new(HashSet::new())).lock() {
                    if seen.len() < 128 && seen.insert((self.qmd[0x08], tic_index, tic.gpu_va)) {
                        log::warn!(
                            "[compute-tic-cpu-target] program={:#x} tic={} target={:#x} gpu={:#x} cpu={:#x} nvmap={} fmt={:?} {}x{} type={} block_linear={}",
                            self.qmd[0x08],
                            tic_index,
                            target,
                            tic.gpu_va,
                            cpu_va,
                            nvmap_id,
                            tic.format,
                            tic.width,
                            tic.height,
                            tic.texture_type,
                            tic.is_block_linear,
                        );
                    }
                }
            }
        }
        let layers = texture_layer_count(&tic);
        let effective_block_linear =
            tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
        let selected_mip_layout = if effective_block_linear && tic.view_base_mip() != 0 {
            nexium_gpu::texture::block_linear_mip_layout(&tic)
        } else {
            None
        };
        let base_pitch_size = tic.format.linear_size(tic.width, tic.height);
        let layer_read_size = texture_layer_read_size(&tic, base_pitch_size);
        let wants_depth_readback = tic.format == nexium_gpu::texture::TicFormat::Z24S8
            && tic.view_base_mip() == 0
            && layers == 1;
        let wants_color_readback = self.qmd[0x08] == 0xc7000
            && tic.format != nexium_gpu::texture::TicFormat::Z24S8
            && tic.view_base_mip() == 0
            && layers == 1;
        let depth_readback = wants_depth_readback
            .then(|| {
                let renderer = self.renderer?;
                let nvmap_id = self.mappings.nvmap_id_for(tic.gpu_va)?;
                renderer.readback_depth_target_raw(nvmap_id, tic.width, tic.height, tic.gpu_va)
            })
            .flatten();
        let color_readback = wants_color_readback
            .then(|| {
                let renderer = self.renderer?;
                let nvmap_id = self.mappings.nvmap_id_for(tic.gpu_va)?;
                let (source_width, source_height, bpp, raw) =
                    renderer.readback_target_raw_content(nvmap_id, tic.gpu_va, tic.format.src_bpp())?;
                if bpp != tic.format.src_bpp() {
                    return None;
                }
                let linear = crop_linear_rows(
                    &raw,
                    source_width,
                    source_height,
                    tic.width,
                    tic.height,
                    bpp,
                )?;
                if self.tex_trace {
                    let raw_nonzero = linear.iter().filter(|byte| **byte != 0).count();
                    let first_nonzero = linear
                        .chunks_exact(4)
                        .enumerate()
                        .find(|(_, word)| word.iter().any(|byte| *byte != 0))
                        .map(|(index, word)| {
                            let raw = u32::from_le_bytes(word.try_into().unwrap());
                            (index as u32 % tic.width, index as u32 / tic.width, raw)
                        });
                    log::warn!(
                        "KeplerCompute::color-readback tic={} nvmap={} va={:#x} src={}x{} dst={}x{} bpp={} rawbnz={}/{} first={:?}",
                        tic_index,
                        nvmap_id,
                        tic.gpu_va,
                        source_width,
                        source_height,
                        tic.width,
                        tic.height,
                        bpp,
                        raw_nonzero,
                        linear.len(),
                        first_nonzero
                    );
                }
                Some((linear, tic.width, tic.height))
            })
            .flatten();
        if self.qmd[0x08] == 0x75000 && wants_depth_readback && depth_readback.is_none() {
            return None;
        }
        let (linear, rgba, width, height) = if let Some((width, height, 4, raw)) = depth_readback {
            if self.tex_trace {
                let mut min_depth = 0x00ff_ffffu32;
                let mut max_depth = 0u32;
                let mut nonzero = 0usize;
                for texel in raw.chunks_exact(4) {
                    let word = u32::from_le_bytes(texel.try_into().unwrap());
                    let depth = word & 0x00ff_ffff;
                    min_depth = min_depth.min(depth);
                    max_depth = max_depth.max(depth);
                    nonzero += usize::from(depth != 0);
                }
                log::warn!(
                    "KeplerCompute::depth-readback tic={} va={:#x} {}x{} depth={:#08x}..{:#08x} nonzero={}/{}",
                    tic_index,
                    tic.gpu_va,
                    width,
                    height,
                    min_depth,
                    max_depth,
                    nonzero,
                    raw.len() / 4
                );
            }
            let rgba = nexium_gpu::texture::decode_to_rgba8(
                &raw,
                width,
                height,
                nexium_gpu::texture::TicFormat::Z24S8,
            );
            (raw, rgba, width, height)
        } else if let Some((linear, width, height)) = color_readback {
            let rgba = nexium_gpu::texture::decode_to_rgba8(&linear, width, height, tic.format);
            (linear, rgba, width, height)
        } else {
            let read_size = selected_mip_layout
                .as_ref()
                .map(|layout| layout.guest_size_bytes(layers as u32))
                .unwrap_or_else(|| layer_read_size.saturating_mul(layers));
            let tex_cpu = map_gpu(self.mappings, tic.gpu_va)?;
            let mut raw = vec![0u8; read_size];
            if !(self.mem_read)(tex_cpu, &mut raw) {
                return None;
            }
            if let Some(layout) = selected_mip_layout.as_ref() {
                decode_texture_mip_layers(&raw, &tic, layout, tic.view_base_mip(), layers)?
            } else {
                let (linear, rgba) =
                    decode_texture_layers(&raw, &tic, base_pitch_size, layer_read_size, layers);
                (linear, rgba, tic.width, tic.height)
            }
        };
        let layer_size = width as usize * height as usize * 4;
        Some(TextureData {
            raw: tic_raw,
            tic,
            linear,
            rgba,
            width,
            height,
            layers,
            layer_size,
        })
    }

    fn tsc_entry(&mut self, tsc_index: u32) -> Option<CachedTsc> {
        if let Some(cached) = self.tsc_cache.get(&tsc_index).copied() {
            return Some(cached);
        }
        let tsc_cpu = map_gpu(
            self.mappings,
            self.texture
                .tsc_pool_gpu_va
                .wrapping_add((tsc_index as u64).saturating_mul(32)),
        )?;
        let mut raw = [0u8; 32];
        if !(self.mem_read)(tsc_cpu, &mut raw) {
            return None;
        }
        let entry = nexium_gpu::texture::TscEntry::parse(&raw)?;
        let cached = CachedTsc { raw, entry };
        self.tsc_cache.insert(tsc_index, cached);
        Some(cached)
    }

    fn tex_b(&mut self, regs: &mut [u32; 256], raw: u64) -> bool {
        let mask = bits(raw, 31, 34) as u8;
        let aoffi = bit(raw, 36);
        let extended_pps_sample = matches!(
            (self.qmd[0x08], raw),
            (0xc7000, 0xdeb8_0030_a0a7_0009)
                | (0xc7000, 0xdeb8_0021_a027_0002)
                | (0xc7000, 0xdeba_0027_a0e7_0c0c)
        );
        if bits(raw, 28, 30) != 2
            || mask == 0
            || (mask != 1 && !extended_pps_sample)
            || bit(raw, 35)
            || (aoffi && !extended_pps_sample)
            || bits(raw, 37, 39) != 1
            || bit(raw, 40)
            || bit(raw, 50)
            || bits(raw, 51, 53) != PT as u64
        {
            if self.tex_trace && self.tex_logs == 0 {
                log::warn!(
                    "KeplerCompute::tex_b reject=encoding program={:#x} raw={:#018x} type={} mask={:#x} aoffi={} blod={} lc={} dc={} sparse={}",
                    self.qmd[0x08],
                    raw,
                    bits(raw, 28, 30),
                    mask,
                    aoffi,
                    bits(raw, 37, 39),
                    bit(raw, 40),
                    bit(raw, 50),
                    bits(raw, 51, 53)
                );
                self.tex_logs += 1;
            }
            return false;
        }

        let handle = get_reg(regs, reg_b(raw));
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let (tic_index, tsc_index) = if linked_tsc {
            (handle, handle)
        } else {
            (handle & 0x000f_ffff, handle >> 20)
        };
        if tic_index > self.texture.tic_limit
            || tsc_index > self.texture.tsc_limit
            || self.texture.tic_pool_gpu_va == 0
            || self.texture.tsc_pool_gpu_va == 0
        {
            if self.tex_trace && self.tex_logs == 0 {
                log::warn!(
                    "KeplerCompute::tex_b reject=handle program={:#x} raw={:#018x} handle={:#x} tic={}/{} tsc={}/{}",
                    self.qmd[0x08],
                    raw,
                    handle,
                    tic_index,
                    self.texture.tic_limit,
                    tsc_index,
                    self.texture.tsc_limit
                );
                self.tex_logs += 1;
            }
            return false;
        }
        if !self.tex_cache.contains_key(&tic_index) {
            let Some(data) = self.load_texture(tic_index) else {
                if self.tex_trace && self.tex_logs == 0 {
                    log::warn!(
                        "KeplerCompute::tex_b reject=tic-load program={:#x} raw={:#018x} handle={:#x} tic={}",
                        self.qmd[0x08],
                        raw,
                        handle,
                        tic_index
                    );
                    self.tex_logs += 1;
                }
                return false;
            };
            self.tex_cache.insert(tic_index, data);
        }
        let Some(CachedTsc {
            raw: tsc_raw,
            entry: tsc,
        }) = self.tsc_entry(tsc_index)
        else {
            return false;
        };
        let data = self.tex_cache.get(&tic_index).unwrap();
        let r16_float = data.tic.format == nexium_gpu::texture::TicFormat::R16
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let rgba8_unorm = data.tic.format == nexium_gpu::texture::TicFormat::A8B8G8R8
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Unorm);
        let rgba16_float = data.tic.format == nexium_gpu::texture::TicFormat::R16G16B16A16
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let b10g11r11_float = data.tic.format == nexium_gpu::texture::TicFormat::B10G11R11
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let z24s8_depth = data.tic.format == nexium_gpu::texture::TicFormat::Z24S8
            && data.tic.component_types
                == [
                    nexium_gpu::texture::ComponentType::Uint,
                    nexium_gpu::texture::ComponentType::Unorm,
                    nexium_gpu::texture::ComponentType::Unorm,
                    nexium_gpu::texture::ComponentType::Unorm,
                ]
            && data.tic.swizzle == [nexium_gpu::texture::SwizzleSource::G; 4];
        let nearest_sample = matches!(tsc.mag_filter, nexium_gpu::texture::TexFilter::Nearest)
            && matches!(tsc.min_filter, nexium_gpu::texture::TexFilter::Nearest)
            && matches!(tsc.mip_filter, nexium_gpu::texture::TexFilter::Nearest);
        let linear_filter = matches!(tsc.mag_filter, nexium_gpu::texture::TexFilter::Linear)
            && matches!(tsc.min_filter, nexium_gpu::texture::TexFilter::Linear)
            && matches!(tsc.mip_filter, nexium_gpu::texture::TexFilter::Nearest);
        let linear_single_texel =
            rgba8_unorm && data.width == 1 && data.height == 1 && linear_filter;
        let pps_rgba16_linear = self.qmd[0x08] == 0xc7000
            && raw == 0xdeba_0027_a0e7_0c0c
            && rgba16_float
            && data.tic.texture_type == 1
            && linear_filter;
        let supported =
            (r16_float || rgba8_unorm || rgba16_float || b10g11r11_float || z24s8_depth)
                && data.tic.normalized_coords
                && !data.tic.is_srgb
                && data.tic.view_mip_levels() == 1
                && data.layers == 1
                && matches!(tsc.wrap_u, nexium_gpu::texture::WrapMode::ClampToEdge)
                && matches!(tsc.wrap_v, nexium_gpu::texture::WrapMode::ClampToEdge)
                && (nearest_sample || linear_single_texel || pps_rgba16_linear)
                && matches!(
                    tsc.reduction,
                    nexium_gpu::texture::SamplerReduction::WeightedAverage
                )
                && !tsc.depth_compare_enabled;
        let u = f32::from_bits(get_reg(regs, reg_a(raw)));
        let v = f32::from_bits(get_reg(regs, reg_a(raw).wrapping_add(1)));
        let offset_word = if aoffi {
            get_reg(regs, reg_b(raw).wrapping_add(1))
        } else {
            0
        };
        let offset_x = if aoffi {
            sign_extend_nibble(offset_word)
        } else {
            0
        };
        let offset_y = if aoffi {
            sign_extend_nibble(offset_word >> 4)
        } else {
            0
        };
        let x = coord_to_index_offset(u, data.width, true, offset_x);
        let y = coord_to_index_offset(v, data.height, true, offset_y);
        let byte_count = if rgba16_float {
            8
        } else if z24s8_depth || rgba8_unorm || b10g11r11_float {
            4
        } else {
            2
        };
        let offset =
            (y.saturating_mul(data.width as usize).saturating_add(x)).saturating_mul(byte_count);
        let linear_rgba16_values = pps_rgba16_linear.then(|| {
            sample_rgba16f_linear_clamp(
                &data.linear,
                data.width,
                data.height,
                u,
                v,
                offset_x,
                offset_y,
            )
        });
        let linear_rgba16_values = linear_rgba16_values.flatten();
        let sample_valid = if pps_rgba16_linear {
            linear_rgba16_values.is_some()
        } else {
            offset + byte_count <= data.linear.len()
        };
        let values = if let Some(physical) = linear_rgba16_values {
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else if z24s8_depth && sample_valid {
            let word = u32::from_le_bytes(data.linear[offset..offset + 4].try_into().unwrap());
            [z24s8_depth_bits(word); 4]
        } else if r16_float && sample_valid {
            let red = f16_to_f32_bits(u16::from_le_bytes([
                data.linear[offset],
                data.linear[offset + 1],
            ]));
            apply_swizzle_words([red, 0, 0, 0], data.tic.swizzle, 1.0f32.to_bits())
        } else if rgba8_unorm && sample_valid {
            let rgba_offset =
                (y.saturating_mul(data.width as usize).saturating_add(x)).saturating_mul(4);
            let physical = [
                (data.rgba[rgba_offset] as f32 / 255.0).to_bits(),
                (data.rgba[rgba_offset + 1] as f32 / 255.0).to_bits(),
                (data.rgba[rgba_offset + 2] as f32 / 255.0).to_bits(),
                (data.rgba[rgba_offset + 3] as f32 / 255.0).to_bits(),
            ];
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else if rgba16_float && sample_valid {
            let physical = std::array::from_fn(|component| {
                let component_offset = offset + component * 2;
                f16_to_f32_bits(u16::from_le_bytes([
                    data.linear[component_offset],
                    data.linear[component_offset + 1],
                ]))
            });
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else if b10g11r11_float && sample_valid {
            let packed = u32::from_le_bytes(data.linear[offset..offset + 4].try_into().unwrap());
            apply_swizzle_words(unpack_b10g11r11(packed), data.tic.swizzle, 1.0f32.to_bits())
        } else {
            [0; 4]
        };
        if self.tex_trace && self.tex_logs < 16 {
            log::warn!(
                "KeplerCompute::tex_b program={:#x} raw={:#018x} handle={:#x} tic={} tsc={} tic_raw={:02x?} tsc_raw={:02x?} va={:#x} fmt={:?} types={:?} swizzle={:?} {}x{} base={}x{} layers={} type={} norm={} block_linear={} srgb={} mip={}/{} wrap=({:?},{:?},{:?}) filter=({:?},{:?},{:?}) reduction={:?} coord=({:.6},{:.6}) offset=({},{}) texel=({},{}) mask={:#x} value={:?} supported={}",
                self.qmd[0x08],
                raw,
                handle,
                tic_index,
                tsc_index,
                data.raw,
                tsc_raw,
                data.tic.gpu_va,
                data.tic.format,
                data.tic.component_types,
                data.tic.swizzle,
                data.width,
                data.height,
                data.tic.width,
                data.tic.height,
                data.layers,
                data.tic.texture_type,
                data.tic.normalized_coords,
                data.tic.is_block_linear,
                data.tic.is_srgb,
                data.tic.view_base_mip(),
                data.tic.view_mip_levels(),
                tsc.wrap_u,
                tsc.wrap_v,
                tsc.wrap_p,
                tsc.mag_filter,
                tsc.min_filter,
                tsc.mip_filter,
                tsc.reduction,
                u,
                v,
                offset_x,
                offset_y,
                x,
                y,
                mask,
                f32::from_bits(values[0]),
                supported && u.is_finite() && v.is_finite() && sample_valid,
            );
            self.tex_logs += 1;
        }
        if !supported || !u.is_finite() || !v.is_finite() || !sample_valid {
            return false;
        }
        let mut dest = reg_dest(raw);
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            set_reg(regs, dest, values[component]);
            dest = dest.wrapping_add(1);
        }
        true
    }
}

fn texture_layer_count(tic: &nexium_gpu::texture::TicEntry) -> usize {
    match tic.texture_type {
        2 => tic.depth.max(1) as usize,
        5 => tic.base_layer.saturating_add(tic.depth).max(1) as usize,
        _ => 1,
    }
}

fn texture_layer_read_size(tic: &nexium_gpu::texture::TicEntry, pitch_size: usize) -> usize {
    if tic.is_block_linear {
        tic.format
            .block_linear_size(tic.width, tic.height, tic.block_height_log2)
            .max(pitch_size)
    } else {
        pitch_size
    }
}

fn decode_texture_layers(
    raw: &[u8],
    tic: &nexium_gpu::texture::TicEntry,
    pitch_size: usize,
    layer_read_size: usize,
    layers: usize,
) -> (Vec<u8>, Vec<u8>) {
    let effective_block_linear =
        tic.is_block_linear && !nexium_gpu::pitch_oracle::is_pitch_dst(tic.gpu_va);
    let layer_rgba_size = tic.width as usize * tic.height as usize * 4;
    let mut linear_out = Vec::with_capacity(pitch_size.saturating_mul(layers));
    let mut rgba_out = Vec::with_capacity(layer_rgba_size.saturating_mul(layers));
    for layer in 0..layers {
        let start = layer.saturating_mul(layer_read_size);
        if start >= raw.len() {
            linear_out.resize(linear_out.len() + pitch_size, 0);
            rgba_out.resize(rgba_out.len() + layer_rgba_size, 0);
            continue;
        }
        let end = (start + layer_read_size).min(raw.len());
        let layer_raw = &raw[start..end];
        let linear = if effective_block_linear {
            let (storage_width, storage_height, bpp) =
                tic.format.storage_extent(tic.width, tic.height);
            nexium_gpu::texture::unswizzle_block_linear(
                layer_raw,
                storage_width,
                storage_height,
                bpp,
                tic.block_height_log2,
            )
        } else if layer_raw.len() >= pitch_size {
            layer_raw[..pitch_size].to_vec()
        } else {
            layer_raw.to_vec()
        };
        let mut linear = linear;
        linear.resize(pitch_size, 0);
        let mut decoded =
            nexium_gpu::texture::decode_to_rgba8(&linear, tic.width, tic.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        linear_out.extend(linear);
        rgba_out.extend(decoded);
    }
    linear_out.resize(pitch_size.saturating_mul(layers), 0);
    rgba_out.resize(layer_rgba_size.saturating_mul(layers), 0);
    (linear_out, rgba_out)
}

fn decode_texture_mip_layers(
    raw: &[u8],
    tic: &nexium_gpu::texture::TicEntry,
    layout: &nexium_gpu::texture::BlockLinearMipLayout,
    mip: u32,
    layers: usize,
) -> Option<(Vec<u8>, Vec<u8>, u32, u32)> {
    let level = layout.levels.get(mip as usize)?;
    let layer_rgba_size = level.width as usize * level.height as usize * 4;
    let mut linear_out = Vec::with_capacity(level.linear_size.saturating_mul(layers));
    let mut rgba_out = Vec::with_capacity(layer_rgba_size.saturating_mul(layers));
    for layer in 0..layers {
        let start = layer
            .saturating_mul(layout.layer_stride)
            .saturating_add(level.guest_offset);
        let end = start.saturating_add(level.guest_size);
        if end > raw.len() {
            return None;
        }
        let mut linear = nexium_gpu::texture::unswizzle_block_linear_strided(
            &raw[start..end],
            level.storage_width,
            level.storage_height,
            tic.format.src_bpp(),
            level.block_height_log2,
            level.stride_alignment_log2,
        );
        linear.resize(level.linear_size, 0);
        let mut decoded =
            nexium_gpu::texture::decode_to_rgba8(&linear, level.width, level.height, tic.format);
        decoded.resize(layer_rgba_size, 0);
        linear_out.extend(linear);
        rgba_out.extend(decoded);
    }
    Some((linear_out, rgba_out, level.width, level.height))
}

fn coord_to_index(v: f32, size: u32, normalized: bool) -> usize {
    if size == 0 {
        return 0;
    }
    let max = (size - 1) as f32;
    let f = if normalized { v * size as f32 } else { v };
    f.floor().clamp(0.0, max) as usize
}

fn coord_to_index_offset(v: f32, size: u32, normalized: bool, offset: i32) -> usize {
    if size == 0 {
        return 0;
    }
    (coord_to_index(v, size, normalized) as i64 + offset as i64)
        .clamp(0, size.saturating_sub(1) as i64) as usize
}

fn linear_sample_axis(v: f32, size: u32, offset: i32) -> Option<(usize, usize, f32)> {
    if size == 0 || !v.is_finite() {
        return None;
    }
    let position = v * size as f32 - 0.5 + offset as f32;
    if !position.is_finite() {
        return None;
    }
    let base_f = position.floor();
    let fraction = position - base_f;
    let base = base_f as i64;
    let max = size.saturating_sub(1) as i64;
    Some((
        base.clamp(0, max) as usize,
        base.saturating_add(1).clamp(0, max) as usize,
        fraction,
    ))
}

fn sample_rgba16f_linear_clamp(
    data: &[u8],
    width: u32,
    height: u32,
    u: f32,
    v: f32,
    offset_x: i32,
    offset_y: i32,
) -> Option<[u32; 4]> {
    let (x0, x1, fx) = linear_sample_axis(u, width, offset_x)?;
    let (y0, y1, fy) = linear_sample_axis(v, height, offset_y)?;
    let read = |x: usize, y: usize| -> Option<[f32; 4]> {
        let texel = y.checked_mul(width as usize)?.checked_add(x)?;
        let start = texel.checked_mul(8)?;
        let bytes = data.get(start..start.checked_add(8)?)?;
        Some(std::array::from_fn(|component| {
            let component = component * 2;
            f32::from_bits(f16_to_f32_bits(u16::from_le_bytes([
                bytes[component],
                bytes[component + 1],
            ])))
        }))
    };
    let top_left = read(x0, y0)?;
    let top_right = read(x1, y0)?;
    let bottom_left = read(x0, y1)?;
    let bottom_right = read(x1, y1)?;
    Some(std::array::from_fn(|component| {
        let top = top_left[component] + (top_right[component] - top_left[component]) * fx;
        let bottom =
            bottom_left[component] + (bottom_right[component] - bottom_left[component]) * fx;
        (top + (bottom - top) * fy).to_bits()
    }))
}

fn sign_extend_nibble(value: u32) -> i32 {
    ((value & 0xf) as i32) << 28 >> 28
}

fn shf_l_reg(regs: &mut [u32; 256], raw: u64) -> bool {
    let max_shift_mode = bits(raw, 37, 38);
    if max_shift_mode == 1 || bit(raw, 47) || bits(raw, 48, 49) != 0 {
        return false;
    }
    let max_shift = if max_shift_mode == 0 { 32 } else { 63 };
    let shift = get_reg(regs, reg_b(raw));
    let safe_shift = if bit(raw, 50) {
        shift & (max_shift - 1)
    } else {
        shift.min(max_shift)
    };
    let packed = (get_reg(regs, reg_a(raw)) as u64) | ((get_reg(regs, reg_c(raw)) as u64) << 32);
    set_reg(
        regs,
        reg_dest(raw),
        (packed.wrapping_shl(safe_shift) >> 32) as u32,
    );
    true
}

fn apply_swizzle(src: [u8; 4], swizzle: [nexium_gpu::texture::SwizzleSource; 4]) -> [u8; 4] {
    fn one(src: [u8; 4], s: nexium_gpu::texture::SwizzleSource) -> u8 {
        match s {
            nexium_gpu::texture::SwizzleSource::Zero => 0,
            nexium_gpu::texture::SwizzleSource::R => src[0],
            nexium_gpu::texture::SwizzleSource::G => src[1],
            nexium_gpu::texture::SwizzleSource::B => src[2],
            nexium_gpu::texture::SwizzleSource::A => src[3],
            nexium_gpu::texture::SwizzleSource::One => 255,
            nexium_gpu::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    [
        one(src, swizzle[0]),
        one(src, swizzle[1]),
        one(src, swizzle[2]),
        one(src, swizzle[3]),
    ]
}

fn apply_swizzle_words(
    src: [u32; 4],
    swizzle: [nexium_gpu::texture::SwizzleSource; 4],
    one: u32,
) -> [u32; 4] {
    fn select(src: [u32; 4], source: nexium_gpu::texture::SwizzleSource, one: u32) -> u32 {
        match source {
            nexium_gpu::texture::SwizzleSource::Zero => 0,
            nexium_gpu::texture::SwizzleSource::R => src[0],
            nexium_gpu::texture::SwizzleSource::G => src[1],
            nexium_gpu::texture::SwizzleSource::B => src[2],
            nexium_gpu::texture::SwizzleSource::A => src[3],
            nexium_gpu::texture::SwizzleSource::One => one,
            nexium_gpu::texture::SwizzleSource::Unknown(_) => 0,
        }
    }
    [
        select(src, swizzle[0], one),
        select(src, swizzle[1], one),
        select(src, swizzle[2], one),
        select(src, swizzle[3], one),
    ]
}

fn z24s8_depth_bits(raw: u32) -> u32 {
    ((raw & 0x00ff_ffff) as f32 / 16_777_215.0).to_bits()
}

fn rgba32_float_texel_words(data: &[u8], width: u32, x: usize, y: usize) -> Option<[u32; 4]> {
    let texel = y.checked_mul(width as usize)?.checked_add(x)?;
    let start = texel.checked_mul(16)?;
    let bytes = data.get(start..start.checked_add(16)?)?;
    Some(std::array::from_fn(|component| {
        let component = component * 4;
        u32::from_le_bytes(bytes[component..component + 4].try_into().unwrap())
    }))
}

fn f16_to_f32_bits(value: u16) -> u32 {
    let sign = ((value as u32) & 0x8000) << 16;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let mut mantissa = (value & 0x03ff) as u32;
    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut unbiased = -14i32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                unbiased -= 1;
            }
            mantissa &= 0x3ff;
            sign | (((unbiased + 127) as u32) << 23) | (mantissa << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    }
}

fn f32_to_f16_bits(value: u32) -> u16 {
    let sign = ((value >> 16) & 0x8000) as u16;
    let exponent = (value >> 23) & 0xff;
    let mantissa = value & 0x007f_ffff;
    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) as u16).max(1);
        return sign | 0x7c00 | payload;
    }

    let half_exponent = exponent as i32 - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let rounded = round_shift_right_even(significand, (14 - half_exponent) as u32);
        return sign | rounded as u16;
    }

    let rounded = round_shift_right_even(mantissa, 13);
    let mut encoded_exponent = half_exponent as u16;
    let encoded_mantissa = if rounded == 0x400 {
        encoded_exponent += 1;
        0
    } else {
        rounded as u16
    };
    if encoded_exponent >= 31 {
        sign | 0x7c00
    } else {
        sign | (encoded_exponent << 10) | encoded_mantissa
    }
}

fn f32_to_ufloat_bits(value: u32, mantissa_bits: u32) -> u32 {
    let sign = value & 0x8000_0000;
    let mut magnitude = value & 0x7fff_ffff;
    let exponent_mask = 0x1f << mantissa_bits;
    let full_mask = (1 << (mantissa_bits + 5)) - 1;
    if magnitude & 0x7f80_0000 == 0x7f80_0000 {
        if magnitude & 0x007f_ffff != 0 {
            return full_mask;
        }
        return if sign == 0 { exponent_mask } else { 0 };
    }

    let (minimum, maximum, maximum_encoded) = match mantissa_bits {
        6 => (0x3580_0000, 0x477e_0000, 0x7bf),
        5 => (0x3600_0000, 0x477c_0000, 0x3df),
        _ => return 0,
    };
    if sign != 0 || magnitude < minimum {
        return 0;
    }
    if magnitude > maximum {
        return maximum_encoded;
    }
    if magnitude < 0x3880_0000 {
        let shift = 113 - (magnitude >> 23);
        magnitude = (0x0080_0000 | (magnitude & 0x007f_ffff)) >> shift;
    } else {
        magnitude = magnitude.wrapping_add(0xc800_0000);
    }
    let shift = 23 - mantissa_bits;
    let bias = (1 << (shift - 1)) - 1;
    ((magnitude + bias + ((magnitude >> shift) & 1)) >> shift) & full_mask
}

fn pack_b10g11r11(r: u32, g: u32, b: u32) -> u32 {
    f32_to_ufloat_bits(r, 6) | (f32_to_ufloat_bits(g, 6) << 11) | (f32_to_ufloat_bits(b, 5) << 22)
}

fn ufloat_to_f32_bits(value: u32, mantissa_bits: u32) -> u32 {
    let mantissa_mask = (1 << mantissa_bits) - 1;
    let mantissa = value & mantissa_mask;
    let exponent = (value >> mantissa_bits) & 0x1f;
    match exponent {
        0 if mantissa == 0 => 0,
        0 => {
            let leading = 31 - mantissa.leading_zeros();
            let ieee_exponent = (leading as i32 - 14 - mantissa_bits as i32 + 127) as u32;
            let ieee_mantissa = (mantissa ^ (1 << leading)) << (23 - leading);
            (ieee_exponent << 23) | ieee_mantissa
        }
        0x1f if mantissa == 0 => f32::INFINITY.to_bits(),
        0x1f => 0x7f80_0000 | (mantissa << (23 - mantissa_bits)),
        _ => ((exponent + 112) << 23) | (mantissa << (23 - mantissa_bits)),
    }
}

fn unpack_b10g11r11(value: u32) -> [u32; 4] {
    [
        ufloat_to_f32_bits(value & 0x7ff, 6),
        ufloat_to_f32_bits((value >> 11) & 0x7ff, 6),
        ufloat_to_f32_bits((value >> 22) & 0x3ff, 5),
        1.0f32.to_bits(),
    ]
}

fn round_shift_right_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let truncated = value >> shift;
    let remainder_mask = (1u32 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1u32 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

#[cfg(test)]
fn block_linear_pixel_offset(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    block_height_log2: u32,
    x: u32,
    y: u32,
) -> Option<usize> {
    let stride_alignment_log2 = 6u32.saturating_sub(bytes_per_pixel.checked_ilog2()?);
    block_linear_pixel_offset_strided(
        width,
        height,
        bytes_per_pixel,
        block_height_log2,
        stride_alignment_log2,
        x,
        y,
    )
}

fn block_linear_pixel_offset_strided(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    block_height_log2: u32,
    stride_alignment_log2: u32,
    x: u32,
    y: u32,
) -> Option<usize> {
    if width == 0 || height == 0 || x >= width || y >= height || !bytes_per_pixel.is_power_of_two()
    {
        return None;
    }
    let width = width as usize;
    let x = x as usize;
    let y = y as usize;
    let alignment = 1usize.checked_shl(stride_alignment_log2)?;
    let aligned_width = width.checked_add(alignment - 1)? & !(alignment - 1);
    let gobs_per_row = aligned_width
        .checked_mul(bytes_per_pixel)?
        .checked_add(63)?
        / 64;
    let block_height = 1usize.checked_shl(block_height_log2)?;
    let rows_per_block = block_height.checked_mul(8)?;
    let block_y = y / rows_per_block;
    let y_in_block = y - block_y * rows_per_block;
    let gob_row_in_block = y_in_block / 8;
    let y_in_gob = y_in_block & 7;
    let block_row_stride = gobs_per_row.checked_mul(block_height)?.checked_mul(512)?;
    let byte_x = x.checked_mul(bytes_per_pixel)?;
    let gob_col = byte_x / 64;
    let x_in_gob = byte_x & 63;
    let gob_offset = block_y
        .checked_mul(block_row_stride)?
        .checked_add(gob_col.checked_mul(block_height)?.checked_mul(512)?)?
        .checked_add(gob_row_in_block.checked_mul(512)?)?;
    let in_gob = ((x_in_gob >> 5) & 1) * 256
        + ((y_in_gob >> 1) & 3) * 64
        + ((x_in_gob >> 4) & 1) * 32
        + (y_in_gob & 1) * 16
        + (x_in_gob & 15);
    gob_offset.checked_add(in_gob)
}

fn h16_ord_eq(a: u16, b: u16) -> bool {
    let is_nan = |value: u16| value & 0x7c00 == 0x7c00 && value & 0x03ff != 0;
    !is_nan(a) && !is_nan(b) && (((a & 0x7fff) == 0 && (b & 0x7fff) == 0) || a == b)
}

fn hsetp2_reg_pps(regs: &[u32; 256], preds: &mut [bool; 8], raw: u64) -> bool {
    if bit(raw, 6)
        || bit(raw, 30)
        || bit(raw, 31)
        || bit(raw, 43)
        || bit(raw, 44)
        || bits(raw, 35, 38) != 2
        || bits(raw, 45, 46) != 0
        || bit(raw, 49)
        || bits(raw, 47, 48) != 2
        || bits(raw, 28, 29) != 2
    {
        return false;
    }

    let cmp = h16_ord_eq(
        get_reg(regs, reg_a(raw)) as u16,
        get_reg(regs, reg_b(raw)) as u16,
    );
    let src = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
    let lane = cmp & src;
    set_pred(preds, bits(raw, 3, 5) as u8, lane);
    set_pred(preds, bits(raw, 0, 2) as u8, lane);
    true
}

fn pps_hadd2_imm(regs: &mut [u32; 256], raw: u64) -> bool {
    const AT_P0: u64 = 0x7a05_083c_0f00_ff11;
    const AT_NOT_P0: u64 = 0x7a05_083c_0f08_ff11;
    if raw != AT_P0 && raw != AT_NOT_P0 {
        return false;
    }
    let dest = reg_dest(raw);
    let old = get_reg(regs, dest);
    set_reg(regs, dest, (old & 0xffff_0000) | 0x0000_3c00);
    true
}

fn empty_compute_cbufs() -> [Vec<u8>; 8] {
    std::array::from_fn(|_| Vec::new())
}

fn pps_2513900_shared_clear_end(
    program: u32,
    code: &[u8],
    pc: usize,
    raw: u64,
    shared_size: usize,
) -> Option<usize> {
    const FIRST_STORE: u64 = 0xef5c_0000_0007_03ff;
    const LAST_STORE: u64 = 0xef5c_0001_f807_03ff;
    const NEXT_INSTRUCTION: u64 = 0x5c10_0000_00c7_0d03;
    if program != 0x2513900 || pc != 0x70 || raw != FIRST_STORE || shared_size != 0x2000 {
        return None;
    }
    let word = |offset: usize| {
        code.get(offset..offset + 8)
            .and_then(|bytes| bytes.try_into().ok())
            .map(u64::from_le_bytes)
    };
    (word(0x328) == Some(LAST_STORE) && word(0x348) == Some(NEXT_INSTRUCTION)).then_some(0x348)
}

fn snapshot_compute_cbufs(
    qmd: &[u32; 0x40],
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
) -> [Vec<u8>; 8] {
    let mut buffers = empty_compute_cbufs();
    let mask = qmd[0x14] & 0xff;
    for (slot, buffer) in buffers.iter_mut().enumerate() {
        if mask & (1 << slot) == 0 {
            continue;
        }
        let base = 0x1d + slot * 2;
        let lo = qmd[base] as u64;
        let hi_size = qmd[base + 1];
        let gpu = (((hi_size & 0xff) as u64) << 32) | lo;
        let size = ((hi_size >> 15) & 0x1ffff) as usize;
        if gpu == 0 || size == 0 {
            continue;
        }
        let Some(cpu) = map_gpu(mappings, gpu) else {
            continue;
        };
        buffer.resize(size, 0);
        if !mem_read(cpu, buffer) {
            buffer.clear();
        }
    }
    buffers
}

fn read_gpu_vec(
    mappings: &GpuMappings,
    mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    gpu: u64,
    len: usize,
) -> Option<Vec<u8>> {
    let cpu = map_gpu(mappings, gpu)?;
    let mut bytes = vec![0u8; len];
    mem_read(cpu, &mut bytes).then_some(bytes)
}

fn map_gpu(mappings: &GpuMappings, gpu: u64) -> Option<u64> {
    mappings
        .cpu_address_for(gpu)
        .or_else(|| mappings.cpu_address_for_any32(gpu).map(|(_, cpu, _)| cpu))
}

fn get_reg(regs: &[u32; 256], r: u8) -> u32 {
    if r == RZ {
        0
    } else {
        regs[r as usize]
    }
}

fn decode_code(code: &[u8]) -> Vec<Option<nexium_shader::Opcode>> {
    code.chunks_exact(8)
        .enumerate()
        .map(|(index, bytes)| {
            let raw = u64::from_le_bytes(bytes.try_into().unwrap());
            if index % 4 == 0 || raw == 0 {
                None
            } else {
                nexium_shader::decode_one(raw).map(|decoded| decoded.opcode)
            }
        })
        .collect()
}

fn code_uses_warp_cooperation(code: &[u8]) -> bool {
    let mut pc = 0usize;
    while pc + 8 <= code.len() {
        if pc % 0x20 != 0 {
            let raw = u64::from_le_bytes(code[pc..pc + 8].try_into().unwrap());
            if let Some(decoded) = nexium_shader::decode_one(raw) {
                if matches!(
                    decoded.opcode,
                    nexium_shader::Opcode::BAR
                        | nexium_shader::Opcode::SHFL
                        | nexium_shader::Opcode::VOTE
                ) {
                    return true;
                }
            }
        }
        pc += 8;
    }
    false
}

fn apply_warp_vote(lanes: &mut [LaneState], pc: usize, raw: u64) -> bool {
    let source_predicate = bits(raw, 39, 41) as u8;
    let negate_source = bit(raw, 42);
    let destination_predicate = bits(raw, 45, 47) as u8;
    let vote_op = bits(raw, 48, 49);
    if vote_op > 2 {
        return false;
    }

    let mut ballot = 0u32;
    let mut values = Vec::new();
    for (lane_id, lane) in lanes.iter().enumerate() {
        if !matches!(lane.state, LaneRunState::WaitingVote { pc: lane_pc, raw: lane_raw } if lane_pc == pc && lane_raw == raw)
        {
            continue;
        }
        let value = pred_value(&lane.preds, source_predicate, negate_source);
        values.push(value);
        if value {
            ballot |= 1u32 << lane_id;
        }
    }
    if values.is_empty() {
        return false;
    }
    let result = match vote_op {
        0 => values.iter().all(|value| *value),
        1 => values.iter().any(|value| *value),
        2 => values.iter().all(|value| *value == values[0]),
        _ => unreachable!(),
    };
    for lane in lanes.iter_mut() {
        if !matches!(lane.state, LaneRunState::WaitingVote { pc: lane_pc, raw: lane_raw } if lane_pc == pc && lane_raw == raw)
        {
            continue;
        }
        set_reg(&mut lane.regs, reg_dest(raw), ballot);
        set_pred(&mut lane.preds, destination_predicate, result);
    }
    true
}

fn release_warp_reconvergence(lanes: &mut [LaneState]) -> bool {
    let mut tokens = Vec::new();
    for lane in lanes.iter() {
        let LaneRunState::WaitingSync { token } = lane.state else {
            continue;
        };
        if !tokens.contains(&token) {
            tokens.push(token);
        }
    }

    let mut released = false;
    for token in tokens {
        let all_arrived = lanes.iter().all(|lane| match lane.state {
            LaneRunState::WaitingSync { token: lane_token } if lane_token == token => true,
            LaneRunState::Exited => true,
            _ => !lane.sync_stack.contains(&token),
        });
        if !all_arrived {
            continue;
        }
        for lane in lanes.iter_mut() {
            if matches!(lane.state, LaneRunState::WaitingSync { token: lane_token } if lane_token == token)
            {
                lane.state = LaneRunState::Ready;
            }
        }
        released = true;
    }
    released
}

fn shuffle_source_lane(lane: u32, mode: u32, index: u32, mask: u32) -> (u32, bool) {
    let clamp = mask & 0x1f;
    let segmentation_mask = (mask >> 8) & 0x1f;
    let not_segmentation_mask = !segmentation_mask;
    let min_lane = lane & segmentation_mask;
    let max_lane = min_lane | (clamp & not_segmentation_mask);
    let source_lane = match mode {
        0 => (index & not_segmentation_mask) | min_lane,
        1 => lane.wrapping_sub(index),
        2 => lane.wrapping_add(index),
        3 => lane ^ index,
        _ => unreachable!(),
    };
    let in_range = match mode {
        1 => (source_lane as i32) >= (max_lane as i32),
        _ => (source_lane as i32) <= (max_lane as i32),
    };
    (source_lane, in_range)
}

fn apply_warp_shuffle(lanes: &mut [LaneState], raw: u64) {
    let source_reg = bits(raw, 8, 15) as u8;
    let destination_reg = reg_dest(raw);
    let destination_predicate = bits(raw, 48, 50) as usize;
    let mode = bits(raw, 30, 31) as u32;
    let source_values: Vec<u32> = lanes
        .iter()
        .map(|lane| get_reg(&lane.regs, source_reg))
        .collect();

    for (lane_id, lane) in lanes.iter_mut().enumerate() {
        if !matches!(lane.state, LaneRunState::WaitingShuffle { raw: value, .. } if value == raw) {
            continue;
        }
        let index = if bit(raw, 28) {
            bits(raw, 20, 24) as u32
        } else {
            get_reg(&lane.regs, bits(raw, 20, 27) as u8)
        };
        let mask = if bit(raw, 29) {
            bits(raw, 34, 46) as u32
        } else {
            get_reg(&lane.regs, bits(raw, 39, 46) as u8)
        };
        let (source_lane, mut in_range) = shuffle_source_lane(lane_id as u32, mode, index, mask);
        in_range &= (source_lane as usize) < source_values.len();
        let value = if in_range {
            source_values[source_lane as usize]
        } else {
            source_values[lane_id]
        };
        set_reg(&mut lane.regs, destination_reg, value);
        if destination_predicate != PT as usize {
            lane.preds[destination_predicate] = in_range;
        }
    }
}

fn set_reg(regs: &mut [u32; 256], r: u8, v: u32) {
    if r != RZ {
        regs[r as usize] = v;
    }
}

fn is_supported_barrier(raw: u64) -> bool {
    raw == 0xf0a8_1b80_0007_0000
}

fn shared_access_width(raw: u64) -> Option<usize> {
    match bits(raw, 48, 50) {
        4 => Some(4),
        6 => Some(16),
        _ => None,
    }
}

fn shared_address(regs: &[u32; 256], raw: u64) -> usize {
    let immediate = bits(raw, 20, 43) as u32;
    let address = if reg_a(raw) == RZ {
        immediate
    } else {
        get_reg(regs, reg_a(raw)).wrapping_add(sign_extend(immediate, 24) as u32)
    };
    address as usize
}

fn shared_load(regs: &mut [u32; 256], raw: u64, shared: &[u8]) -> bool {
    let Some(width) = shared_access_width(raw) else {
        return false;
    };
    let start = shared_address(regs, raw);
    let Some(end) = start.checked_add(width) else {
        return false;
    };
    let Some(bytes) = shared.get(start..end) else {
        return false;
    };
    let dest = reg_dest(raw);
    if dest.checked_add((width / 4 - 1) as u8).is_none() {
        return false;
    }
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let reg = dest + index as u8;
        set_reg(regs, reg, u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    true
}

fn shared_store(regs: &[u32; 256], raw: u64, shared: &mut [u8]) -> bool {
    let Some(width) = shared_access_width(raw) else {
        return false;
    };
    let start = shared_address(regs, raw);
    let Some(end) = start.checked_add(width) else {
        return false;
    };
    let Some(bytes) = shared.get_mut(start..end) else {
        return false;
    };
    let source = reg_dest(raw);
    if source.checked_add((width / 4 - 1) as u8).is_none() {
        return false;
    }
    for (index, chunk) in bytes.chunks_exact_mut(4).enumerate() {
        let reg = source + index as u8;
        chunk.copy_from_slice(&get_reg(regs, reg).to_le_bytes());
    }
    true
}

fn addr_pair(regs: &[u32; 256], r: u8) -> u64 {
    if r == RZ {
        0
    } else {
        get_reg(regs, r) as u64 | ((get_reg(regs, r.wrapping_add(1)) as u64) << 32)
    }
}

fn bits(insn: u64, lo: u32, hi: u32) -> u64 {
    (insn >> lo) & ((1u64 << (hi - lo + 1)) - 1)
}

fn bit(insn: u64, n: u32) -> bool {
    ((insn >> n) & 1) != 0
}

fn xmad_half(v: u32, half: u8, signed: bool) -> u32 {
    let x = (v >> (16 * half as u32)) & 0xffff;
    if signed {
        (((x as i32) << 16) >> 16) as u32
    } else {
        x
    }
}

fn xmad_eval(
    raw: u64,
    a_full: u32,
    src_b: u32,
    src_c: u32,
    select: u8,
    half_b: u8,
    psl: bool,
    mrg: bool,
) -> u32 {
    let a = xmad_half(a_full, bit(raw, 53) as u8, bit(raw, 48));
    let b = xmad_half(src_b, half_b, bit(raw, 49));
    let mut product = a.wrapping_mul(b);
    if psl {
        product <<= 16;
    }
    let c = match select {
        1 => src_c & 0xffff,
        2 => src_c >> 16,
        4 => (src_b << 16).wrapping_add(src_c),
        _ => src_c,
    };
    let mut result = product.wrapping_add(c);
    if mrg {
        result = (result & 0xffff) | (src_b << 16);
    }
    result
}

fn reg_dest(insn: u64) -> u8 {
    bits(insn, 0, 7) as u8
}

fn reg_a(insn: u64) -> u8 {
    bits(insn, 8, 15) as u8
}

fn reg_b(insn: u64) -> u8 {
    bits(insn, 20, 27) as u8
}

fn reg_c(insn: u64) -> u8 {
    bits(insn, 39, 46) as u8
}

fn imm20(insn: u64) -> i32 {
    let raw = bits(insn, 20, 38) as i32;
    if bit(insn, 56) {
        raw - (1 << 19)
    } else {
        raw
    }
}

fn imm32(insn: u64) -> u32 {
    bits(insn, 20, 51) as u32
}

fn float_imm20(insn: u64) -> f32 {
    let value = (bits(insn, 20, 38) as u32) << 12;
    let sign = if bit(insn, 56) { 1u32 << 31 } else { 0 };
    f32::from_bits(value | sign)
}

fn ldg_offset(insn: u64) -> i32 {
    let raw = bits(insn, 20, 43) as u32;
    if raw & (1 << 23) != 0 {
        (raw | 0xff00_0000) as i32
    } else {
        raw as i32
    }
}

fn ldg_size(insn: u64) -> u32 {
    bits(insn, 48, 50) as u32
}

fn lop_op(insn: u64) -> u64 {
    bits(insn, 41, 42)
}

fn lop_not_a(insn: u64) -> bool {
    bit(insn, 39)
}

fn lop_not_b(insn: u64) -> bool {
    bit(insn, 40)
}

fn lop32i_op(insn: u64) -> u64 {
    bits(insn, 53, 54)
}

fn lop32i_not_a(insn: u64) -> bool {
    bit(insn, 55)
}

fn lop32i_not_b(insn: u64) -> bool {
    bit(insn, 56)
}

fn lop3_imm_f8(regs: &[u32; 256], insn: u64) -> Option<u32> {
    if bit(insn, 47) || bits(insn, 48, 55) != 0xf8 {
        return None;
    }
    let a = get_reg(regs, reg_a(insn));
    let b = imm20(insn) as u32;
    let c = get_reg(regs, reg_c(insn));
    Some(a | (b & c))
}

fn pred_active(raw: u64, preds: &[bool; 8]) -> bool {
    pred_value(preds, bits(raw, 16, 18) as u8, bit(raw, 19))
}

fn pred_value(preds: &[bool; 8], pred: u8, neg: bool) -> bool {
    let v = if pred == PT {
        true
    } else {
        preds[pred as usize]
    };
    if neg {
        !v
    } else {
        v
    }
}

fn set_pred(preds: &mut [bool; 8], pred: u8, v: bool) {
    if pred != PT {
        preds[pred as usize] = v;
    }
}

fn boolop(op: u64, a: bool, b: bool) -> bool {
    match op & 3 {
        0 => a & b,
        1 => a | b,
        _ => a ^ b,
    }
}

fn icmp(cmp: u64, signed: bool, a: u32, b: u32) -> bool {
    match cmp & 7 {
        0 => false,
        1 => {
            if signed {
                (a as i32) < (b as i32)
            } else {
                a < b
            }
        }
        2 => a == b,
        3 => {
            if signed {
                (a as i32) <= (b as i32)
            } else {
                a <= b
            }
        }
        4 => {
            if signed {
                (a as i32) > (b as i32)
            } else {
                a > b
            }
        }
        5 => a != b,
        6 => {
            if signed {
                (a as i32) >= (b as i32)
            } else {
                a >= b
            }
        }
        _ => true,
    }
}

fn fcmp(cmp: u64, a: f32, b: f32) -> bool {
    let ord = !a.is_nan() && !b.is_nan();
    match cmp & 0xf {
        0 => false,
        1 => ord && a < b,
        2 => ord && a == b,
        3 => ord && a <= b,
        4 => ord && a > b,
        5 => ord && a != b,
        6 => ord && a >= b,
        7 => ord,
        8 => !ord,
        9 => !ord || a < b,
        10 => !ord || a == b,
        11 => !ord || a <= b,
        12 => !ord || a > b,
        13 => !ord || a != b,
        14 => !ord || a >= b,
        _ => true,
    }
}

fn fsat(v: f32, sat: bool) -> f32 {
    if sat {
        v.clamp(0.0, 1.0)
    } else {
        v
    }
}

fn crop_linear_rows(
    raw: &[u8],
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    bpp: usize,
) -> Option<Vec<u8>> {
    if bpp == 0 || source_width < target_width || source_height < target_height {
        return None;
    }
    let source_stride = (source_width as usize).checked_mul(bpp)?;
    let target_stride = (target_width as usize).checked_mul(bpp)?;
    let source_size = source_stride.checked_mul(source_height as usize)?;
    let target_size = target_stride.checked_mul(target_height as usize)?;
    if raw.len() < source_size {
        return None;
    }
    let mut cropped = vec![0u8; target_size];
    for y in 0..target_height as usize {
        let source_start = y.checked_mul(source_stride)?;
        let target_start = y.checked_mul(target_stride)?;
        cropped[target_start..target_start + target_stride]
            .copy_from_slice(&raw[source_start..source_start + target_stride]);
    }
    Some(cropped)
}

fn flush_subnormal(v: f32) -> f32 {
    if v.is_subnormal() {
        f32::from_bits(v.to_bits() & 0x8000_0000)
    } else {
        v
    }
}

fn fmul32i_value(raw: u64, a: u32) -> u32 {
    let fmz = bits(raw, 53, 54);
    let mut a = f32::from_bits(a);
    let mut b = f32::from_bits(imm32(raw));
    if fmz != 0 {
        a = flush_subnormal(a);
        b = flush_subnormal(b);
    }
    let mut value = flush_subnormal(a * b);
    if fmz == 2 && (a == 0.0 || b == 0.0) {
        value = 0.0;
    }
    fsat(value, bit(raw, 55)).to_bits()
}

fn fadd32i_value(raw: u64, a: u32) -> u32 {
    let mut a = f32::from_bits(a);
    let mut b = f32::from_bits(imm32(raw));
    if bit(raw, 54) {
        a = a.abs();
    }
    if bit(raw, 56) {
        a = -a;
    }
    if bit(raw, 57) {
        b = b.abs();
    }
    if bit(raw, 53) {
        b = -b;
    }
    if bit(raw, 55) {
        a = flush_subnormal(a);
        b = flush_subnormal(b);
        flush_subnormal(a + b).to_bits()
    } else {
        (a + b).to_bits()
    }
}

fn iadd32i_value(raw: u64, a: u32) -> u32 {
    let po = bits(raw, 55, 56) == 3;
    let mut a = a;
    if !po && bit(raw, 56) {
        a = (!a).wrapping_add(1);
    }
    let mut value = a.wrapping_add(imm32(raw));
    if po {
        value = value.wrapping_add(1);
    }
    value
}

fn half(v: u32, half: u8) -> u32 {
    match half {
        1 => v & 0xffff,
        2 => (v >> 16) & 0xffff,
        _ => v,
    }
}

fn sign_extend(v: u32, bits: u32) -> i32 {
    if bits >= 32 {
        v as i32
    } else {
        let shift = 32 - bits;
        ((v << shift) as i32) >> shift
    }
}

fn bfe_value(value: u32, control: u32, signed: bool) -> u32 {
    let position = control & 0xff;
    if position >= 32 {
        return 0;
    }
    let count = ((control >> 8) & 0xff).min(32 - position);
    if count == 0 {
        return 0;
    }
    let mask = if count == 32 {
        u32::MAX
    } else {
        (1u32 << count) - 1
    };
    let extracted = (value >> position) & mask;
    if signed {
        sign_extend(extracted, count) as u32
    } else {
        extracted
    }
}

fn fmnmx_value(a: u32, b: u32, raw: u64, preds: &[bool; 8]) -> Option<u32> {
    if bit(raw, 47) || bit(raw, 50) {
        return None;
    }

    let a = float_source_modifiers(a, bit(raw, 46), bit(raw, 48), bit(raw, 44));
    let b = float_source_modifiers(b, bit(raw, 49), bit(raw, 45), bit(raw, 44));
    let af = f32::from_bits(a);
    let bf = f32::from_bits(b);

    let both_zero = (a & 0x7fff_ffff) == 0 && (b & 0x7fff_ffff) == 0;
    let (min, max) = if af.is_nan() && bf.is_nan() {
        (a, a)
    } else if af.is_nan() {
        (b, b)
    } else if bf.is_nan() {
        (a, a)
    } else if both_zero {
        ((a | b) & 0x8000_0000, (a & b) & 0x8000_0000)
    } else if af < bf {
        (a, b)
    } else if bf < af {
        (b, a)
    } else {
        (a, a)
    };
    let select_min = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
    Some(if select_min { min } else { max })
}

fn imnmx_value(a: u32, b: u32, preds: &[bool; 8], raw: u64) -> Option<u32> {
    if bits(raw, 43, 44) != 0 || bit(raw, 47) {
        return None;
    }
    let (min, max) = if bit(raw, 48) {
        if (a as i32) <= (b as i32) {
            (a, b)
        } else {
            (b, a)
        }
    } else if a <= b {
        (a, b)
    } else {
        (b, a)
    };
    let select_min = pred_value(preds, bits(raw, 39, 41) as u8, bit(raw, 42));
    Some(if select_min { min } else { max })
}

fn float_source_modifiers(mut value: u32, abs: bool, neg: bool, ftz: bool) -> u32 {
    if abs {
        value &= 0x7fff_ffff;
    }
    if neg {
        value ^= 0x8000_0000;
    }
    if ftz && value & 0x7f80_0000 == 0 && value & 0x007f_ffff != 0 {
        value &= 0x8000_0000;
    }
    value
}

fn f2f_reg_f32_floor(regs: &mut [u32; 256], raw: u64) -> bool {
    let dst_size = bits(raw, 8, 9);
    let src_size = bits(raw, 10, 11);
    let rounding = bits(raw, 39, 42) & 0x0b;
    if dst_size != 2 || src_size != 2 || rounding != 9 || bit(raw, 47) {
        return false;
    }
    let source = float_source_modifiers(
        get_reg(regs, reg_b(raw)),
        bit(raw, 49),
        bit(raw, 45),
        bit(raw, 44),
    );
    set_reg(
        regs,
        reg_dest(raw),
        fsat(f32::from_bits(source).floor(), bit(raw, 50)).to_bits(),
    );
    true
}

fn flo(regs: &mut [u32; 256], raw: u64, source: u32) -> bool {
    if bit(raw, 47) {
        return false;
    }
    let source = if bit(raw, 40) { !source } else { source };
    let scan = if bit(raw, 48) && source & 0x8000_0000 != 0 {
        !source
    } else {
        source
    };
    let mut result = if scan == 0 {
        u32::MAX
    } else {
        31 - scan.leading_zeros()
    };
    if bit(raw, 41) && result != u32::MAX {
        result ^= 31;
    }
    set_reg(regs, reg_dest(raw), result);
    true
}

fn popc(regs: &mut [u32; 256], raw: u64, source: u32) {
    let source = if bit(raw, 40) { !source } else { source };
    set_reg(regs, reg_dest(raw), source.count_ones());
}

fn bra_target(pc: usize, raw: u64) -> usize {
    let raw_24 = bits(raw, 20, 43) as u32;
    let signed = if raw_24 & 0x0080_0000 != 0 {
        (raw_24 | 0xff00_0000) as i32
    } else {
        raw_24 as i32
    };
    (pc as i64 + signed as i64 + 8) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pps_shared_b32_round_trip_uses_base_plus_immediate() {
        let store = 0xef5c_0000_0007_010c;
        let load = 0xef4c_1000_1407_0105;
        assert_eq!(
            nexium_shader::decode_one(store).unwrap().opcode,
            nexium_shader::Opcode::STS
        );
        assert_eq!(
            nexium_shader::decode_one(load).unwrap().opcode,
            nexium_shader::Opcode::LDS
        );

        let mut regs = [0u32; 256];
        let mut shared = [0u8; 0x200];
        regs[1] = 0x180;
        regs[12] = 0x89ab_cdef;
        assert!(shared_store(&regs, store, &mut shared));
        assert_eq!(&shared[0x180..0x184], &0x89ab_cdefu32.to_le_bytes());

        regs[1] = 0x40;
        assert!(shared_load(&mut regs, load, &shared));
        assert_eq!(regs[5], 0x89ab_cdef);
    }

    #[test]
    fn pps_shared_b128_round_trip_allows_exact_end() {
        let store = 0xef5e_0000_0007_0d08;
        let load = 0xef4e_1000_8007_0d00;
        assert_eq!(shared_access_width(store), Some(16));
        assert_eq!(shared_access_width(load), Some(16));

        let mut regs = [0u32; 256];
        let mut shared = [0u8; 0x1000];
        regs[13] = 0xff0;
        regs[8..12].copy_from_slice(&[1, 2, 3, 4]);
        assert!(shared_store(&regs, store, &mut shared));
        assert_eq!(
            &shared[0xff0..0x1000],
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
        );

        regs[13] = 0x7f0;
        assert!(shared_load(&mut regs, load, &shared));
        assert_eq!(&regs[0..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn shared_accesses_fail_without_partial_changes() {
        let store = 0xef5e_0000_0007_0d08;
        let load = 0xef4e_1000_0007_0d00;
        let mut regs = [0u32; 256];
        let mut shared = [0xa5u8; 0x20];
        regs[13] = 0x18;
        regs[0..4].copy_from_slice(&[1, 2, 3, 4]);

        assert!(!shared_store(&regs, store, &mut shared));
        assert_eq!(shared, [0xa5; 0x20]);
        assert!(!shared_load(&mut regs, load, &shared));
        assert_eq!(&regs[0..4], &[1, 2, 3, 4]);

        let unsupported_size = store | (1u64 << 48);
        assert_eq!(shared_access_width(unsupported_size), None);
        assert!(!shared_store(&regs, unsupported_size, &mut shared));
        assert_eq!(shared, [0xa5; 0x20]);
    }

    #[test]
    fn pps_barrier_releases_full_cta() {
        let barrier = 0xf0a8_1b80_0007_0000;
        assert!(is_supported_barrier(barrier));
        assert!(!is_supported_barrier(barrier ^ (1 << 8)));

        let mut code = vec![0u8; 0x40];
        code[8..16].copy_from_slice(&barrier.to_le_bytes());
        code[16..24].copy_from_slice(&0xe300_0000_0007_000fu64.to_le_bytes());
        let qmd = [0u32; 0x40];
        let mappings = GpuMappings::new();
        let read = |_: u64, _: &mut [u8]| false;
        let write = |_: u64, _: &[u8]| false;
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &code,
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState::default(),
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        exec.run_cta([0, 0, 0], [2, 1, 1], 0x20);

        assert!(exec.unsupported.is_none());
    }

    #[test]
    fn pps_shfl_up_reads_the_previous_warp_lane() {
        let raw = 0xef11_7f80_5017_0001u64;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SHFL
        );
        let mut lanes: Vec<LaneState> = (0..4)
            .map(|x| {
                let mut lane = LaneState::new([0, 0, 0], [x, 0, 0]);
                lane.regs[0] = 10 + x * 10;
                lane.state = LaneRunState::WaitingShuffle { pc: 0x1a8, raw };
                lane
            })
            .collect();

        apply_warp_shuffle(&mut lanes, raw);

        assert_eq!(lanes[0].regs[1], 10);
        assert!(!lanes[0].preds[1]);
        assert_eq!(lanes[1].regs[1], 10);
        assert!(lanes[1].preds[1]);
        assert_eq!(lanes[2].regs[1], 20);
        assert!(lanes[2].preds[1]);
        assert_eq!(lanes[3].regs[1], 30);
        assert!(lanes[3].preds[1]);
    }

    #[test]
    fn pps_multidimensional_cta_uses_linear_lane_ids() {
        let block = [4, 4, 4];
        assert_eq!(linear_lane_id([0, 0, 0], block), 0);
        assert_eq!(linear_lane_id([3, 0, 0], block), 3);
        assert_eq!(linear_lane_id([0, 1, 0], block), 4);
        assert_eq!(linear_lane_id([3, 3, 1], block), 31);
        assert_eq!(linear_lane_id([0, 0, 2], block), 0);
        assert_eq!(linear_lane_id([3, 3, 3], block), 31);
    }

    #[test]
    fn shfl_opcode_selects_cooperative_cta_execution() {
        let raw = 0xef11_7f80_5017_0001u64;
        let mut code = vec![0u8; 0x20];
        code[8..16].copy_from_slice(&raw.to_le_bytes());
        assert!(code_uses_warp_cooperation(&code));
    }

    #[test]
    fn pps_ssy_pushes_reconvergence_target_independent_of_predicate() {
        let mut code = vec![0u8; 0x88];
        code[8..16].copy_from_slice(&0xe290_0000_0700_0000u64.to_le_bytes());
        code[16..24].copy_from_slice(&0xf0f8_0000_000e_000fu64.to_le_bytes());
        let qmd = [0u32; 0x40];
        let mappings = GpuMappings::new();
        let read = |_: u64, _: &mut [u8]| false;
        let write = |_: u64, _: &[u8]| false;
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &code,
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState::default(),
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };
        let mut lane = LaneState::new([0, 0, 0], [0, 0, 0]);
        let mut shared = [];

        assert!(matches!(
            exec.resume_lane(&mut lane, &mut shared, true),
            LaneYield::Sync {
                pc: 0x10,
                token: SyncToken {
                    origin: 0x8,
                    target: 0x80,
                },
            }
        ));
        assert_eq!(lane.pc, 0x80);
        assert!(lane.sync_stack.is_empty());
    }

    #[test]
    fn pps_5dbe_vote_flo_and_popc_match_visibility_compaction() {
        let vote = 0x50d8_e380_0007_0001;
        assert_eq!(
            nexium_shader::decode_one(vote).unwrap().opcode,
            nexium_shader::Opcode::VOTE
        );
        assert_eq!(reg_dest(vote), 1);
        assert_eq!(bits(vote, 39, 41), PT as u64);
        assert!(!bit(vote, 42));
        assert_eq!(bits(vote, 45, 47), PT as u64);
        assert_eq!(bits(vote, 48, 49), 0);

        let find_last = 0x5c30_0000_0017_0003;
        assert_eq!(
            nexium_shader::decode_one(find_last).unwrap().opcode,
            nexium_shader::Opcode::FLO_reg
        );
        assert_eq!((reg_dest(find_last), reg_b(find_last)), (3, 1));
        assert!(!bit(find_last, 40));
        assert!(!bit(find_last, 41));
        assert!(!bit(find_last, 47));
        assert!(!bit(find_last, 48));

        let count_active = 0x5c08_0000_0017_0002;
        let count_preceding = 0x5c08_0000_0047_0004;
        for (raw, dest, source) in [(count_active, 2, 1), (count_preceding, 4, 4)] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::POPC_reg
            );
            assert_eq!((reg_dest(raw), reg_b(raw)), (dest, source));
            assert!(!bit(raw, 40));
        }

        let mut regs = [0u32; 256];
        regs[1] = 0b0101;
        let ballot = regs[1];
        assert!(flo(&mut regs, find_last, ballot));
        assert_eq!(regs[3], 2);
        popc(&mut regs, count_active, ballot);
        assert_eq!(regs[2], 2);
        regs[4] = 0b0011;
        let preceding = regs[4];
        popc(&mut regs, count_preceding, preceding);
        assert_eq!(regs[4], 2);

        regs[1] = 0;
        assert!(flo(&mut regs, find_last, 0));
        assert_eq!(regs[3], u32::MAX);
    }

    #[test]
    fn pps_5dbe_s2r_lanemask_fields_match_maxwell() {
        let ltmask = 0xf0c8_0000_0397_0004;
        assert_eq!(
            nexium_shader::decode_one(ltmask).unwrap().opcode,
            nexium_shader::Opcode::S2R
        );
        assert_eq!(reg_dest(ltmask), 4);
        assert_eq!(bits(ltmask, 20, 27), 0x39);

        assert_eq!(subgroup_mask(0x38, 2), 0x0000_0004);
        assert_eq!(subgroup_mask(0x39, 2), 0x0000_0003);
        assert_eq!(subgroup_mask(0x3a, 2), 0x0000_0007);
        assert_eq!(subgroup_mask(0x3b, 2), 0xffff_fff8);
        assert_eq!(subgroup_mask(0x3c, 2), 0xffff_fffc);

        assert_eq!(subgroup_mask(0x39, 0), 0);
        assert_eq!(subgroup_mask(0x3c, 0), u32::MAX);
        assert_eq!(subgroup_mask(0x38, 31), 0x8000_0000);
        assert_eq!(subgroup_mask(0x3a, 31), u32::MAX);
        assert_eq!(subgroup_mask(0x3b, 31), 0);
    }

    #[test]
    fn pps_5dbe_divergent_vote_uses_only_active_lanes_then_reconverges() {
        let vote = 0x50d8_e380_0007_0001;
        let token = SyncToken {
            origin: 0x8b0,
            target: 0xc58,
        };
        let mut lanes: Vec<LaneState> = (0..4)
            .map(|lane_id| {
                let mut lane = LaneState::new([0, 0, 0], [lane_id, 0, 0]);
                lane.regs[1] = 0xdead_beef;
                lane.sync_stack.push(token);
                if lane_id & 1 == 0 {
                    lane.pc = 0xb30;
                    lane.state = LaneRunState::WaitingVote {
                        pc: 0xb28,
                        raw: vote,
                    };
                } else {
                    assert!(lane.sync_stack.pop() == Some(token));
                    lane.pc = token.target;
                    lane.state = LaneRunState::WaitingSync { token };
                }
                lane
            })
            .collect();

        assert!(!release_warp_reconvergence(&mut lanes));
        assert!(apply_warp_vote(&mut lanes, 0xb28, vote));
        assert_eq!(lanes[0].regs[1], 0b0101);
        assert_eq!(lanes[2].regs[1], 0b0101);
        assert_eq!(lanes[1].regs[1], 0xdead_beef);
        assert_eq!(lanes[3].regs[1], 0xdead_beef);

        for lane in lanes.iter_mut().step_by(2) {
            lane.state = LaneRunState::Ready;
            assert!(lane.sync_stack.pop() == Some(token));
            lane.pc = token.target;
            lane.state = LaneRunState::WaitingSync { token };
        }
        assert!(release_warp_reconvergence(&mut lanes));
        assert!(lanes
            .iter()
            .all(|lane| lane.state == LaneRunState::Ready && lane.pc == token.target));
    }

    #[test]
    fn vote_opcode_selects_cooperative_cta_execution() {
        let raw = 0x50d8_e380_0007_0001u64;
        let mut code = vec![0u8; 0x20];
        code[8..16].copy_from_slice(&raw.to_le_bytes());
        assert!(code_uses_warp_cooperation(&code));
    }

    #[test]
    fn pps_visibility_programs_get_targeted_invocation_limit() {
        assert_eq!(compute_cpu_invocation_limit_for(0x5dbe00, None), 32_768);
        assert_eq!(compute_cpu_invocation_limit_for(0xb6300, None), 32_768);
        assert_eq!(compute_cpu_invocation_limit_for(0xd3f00, None), 524_288);
        assert_eq!(compute_cpu_invocation_limit_for(0xc7000, None), 1_500_000);
        assert_eq!(compute_cpu_invocation_limit_for(0x75000, None), 1_500_000);
        assert_eq!(compute_cpu_invocation_limit_for(0x74a00, None), 4_096);
        assert_eq!(
            compute_cpu_invocation_limit_for(0x5dbe00, Some(12_345)),
            12_345
        );
        assert_eq!(
            compute_cpu_invocation_limit_for(0xc7000, Some(12_345)),
            12_345
        );
        assert_eq!(compute_cpu_invocation_limit_for(0x5dbe00, Some(0)), 32_768);
    }

    #[test]
    fn pps_5d_trace_aggregates_control_flow_and_bounds_samples() {
        let mut trace = Pps5dTrace::default();
        trace.record_instruction(0x8a8, true);
        trace.record_instruction(0x8a8, false);
        trace.record_instruction(0x970, true);
        trace.record_instruction(0x950, true);
        trace.record_instruction(0x950, false);
        trace.record_instruction(0xb18, false);
        trace.record_instruction(0xb28, true);
        trace.record_instruction(0xb28, false);
        trace.record_instruction(0xe08, true);
        trace.record_vote_pass(0xb28, 3);
        trace.record_vote_pass(0xe08, 2);

        let mut regs = [0u32; 256];
        regs[4] = 0x1122_3344;
        for coord in 0..6 {
            trace.record_tld_pass(
                Pps5dTldSample {
                    pc: 0x8a8,
                    raw: 4,
                    coord: [coord, 0],
                    mask: 1,
                    ..Pps5dTldSample::default()
                },
                &regs,
            );
        }
        trace.record_tld_pass(
            Pps5dTldSample {
                pc: 0x970,
                raw: 4,
                coord: [9, 0],
                mask: 1,
                ..Pps5dTldSample::default()
            },
            &regs,
        );
        trace.record_exit(0xf68);
        trace.record_exit(0xf68);
        trace.record_exit(0x1010);

        assert_eq!(trace.tld_reach, [1, 1]);
        assert_eq!(trace.tld_pass, [6, 1]);
        assert_eq!(trace.cull_reach, [2, 1]);
        assert_eq!(trace.cull_taken, [1, 0]);
        assert_eq!(trace.vote_reach, [1, 1]);
        assert_eq!(trace.vote_pass, [3, 2]);
        assert_eq!(trace.exit_sites, vec![(0xf68, 2), (0x1010, 1)]);
        assert_eq!(trace.tld_samples.len(), 5);
        assert_eq!(trace.tld_samples[0].words[0], 0x1122_3344);
    }

    #[test]
    fn pps_lop3_imm_f8_r2_aliases_c_source() {
        let raw = 0x3cf8_0100_0027_0302;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::LOP3_imm
        );
        let mut regs = [0u32; 256];
        regs[3] = 0x8123_4500;
        regs[2] = 0xffff_ffff;

        let result = lop3_imm_f8(&regs, raw).unwrap();

        assert_eq!(result, 0x8123_4502);
        assert_eq!(reg_dest(raw), 2);
        assert_eq!(reg_c(raw), 2);
    }

    #[test]
    fn pps_lop3_imm_f8_r0_aliases_c_source() {
        let raw = 0x3cf8_0000_0027_0100;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::LOP3_imm
        );
        let mut regs = [0u32; 256];
        regs[1] = 0x1020_3040;
        regs[0] = 0xffff_fffd;

        let result = lop3_imm_f8(&regs, raw).unwrap();

        assert_eq!(result, 0x1020_3040);
        assert_eq!(imm20(raw), 2);
        assert_eq!(bits(raw, 48, 55), 0xf8);
        assert!(!bit(raw, 47));
    }

    #[test]
    fn pps_bfe_imm_extracts_bit_four() {
        for raw in [0x3800_0000_1047_0003, 0x3800_0000_1047_0301] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::BFE_imm
            );
            assert_eq!(imm20(raw), 0x104);
            assert_eq!(bfe_value(0x10, imm20(raw) as u32, false), 1);
            assert_eq!(bfe_value(0x00, imm20(raw) as u32, false), 0);
        }
    }

    #[test]
    fn pps_composite_f2f_floors_f32_with_ftz() {
        for (raw, dest, source_reg) in [
            (0x5ca8_1480_0197_0a06, 6usize, 25usize),
            (0x5ca8_1480_0057_0a07, 7usize, 5usize),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::F2F_reg
            );
            assert_eq!(
                (reg_dest(raw) as usize, reg_b(raw) as usize),
                (dest, source_reg)
            );
            assert_eq!((bits(raw, 8, 9), bits(raw, 10, 11)), (2, 2));
            assert_eq!(bits(raw, 39, 42) & 0x0b, 9);
            assert!(bit(raw, 44));
            assert!(!bit(raw, 45));
            assert!(!bit(raw, 47));
            assert!(!bit(raw, 49));
            assert!(!bit(raw, 50));

            for (source, expected) in [
                (3.75f32.to_bits(), 3.0f32.to_bits()),
                ((-1.25f32).to_bits(), (-2.0f32).to_bits()),
                (0x8000_0001, (-0.0f32).to_bits()),
            ] {
                let mut regs = [0u32; 256];
                regs[source_reg] = source;
                assert!(f2f_reg_f32_floor(&mut regs, raw));
                assert_eq!(regs[dest], expected);
            }
        }
    }
}

impl ComputeExec<'_> {
    fn tld(&mut self, regs: &mut [u32; 256], raw: u64) -> bool {
        let tex_type = bits(raw, 28, 30) as u8;
        let mask = bits(raw, 31, 34) as u8;
        let sparse_pred = bits(raw, 51, 53) as u8;
        if !matches!(tex_type, 0 | 2)
            || mask == 0
            || bit(raw, 35)
            || bit(raw, 50)
            || bit(raw, 54)
            || bit(raw, 55)
            || sparse_pred != PT
        {
            return false;
        }

        let handle = get_reg(regs, reg_b(raw));
        let linked_tsc = (self.qmd[0x0b] & (1 << 30)) != 0;
        let tic_index = if linked_tsc {
            handle
        } else {
            handle & 0x000f_ffff
        };
        if tic_index > self.texture.tic_limit || self.texture.tic_pool_gpu_va == 0 {
            return false;
        }
        if !self.tex_cache.contains_key(&tic_index) {
            let Some(data) = self.load_texture(tic_index) else {
                return false;
            };
            self.tex_cache.insert(tic_index, data);
        }
        let data = self.tex_cache.get(&tic_index).unwrap();
        let rgba8_unorm = matches!(
            data.tic.format,
            nexium_gpu::texture::TicFormat::A8B8G8R8 | nexium_gpu::texture::TicFormat::R8G8B8A8
        ) && data.tic.component_types.iter().all(|ty| {
            matches!(
                ty,
                nexium_gpu::texture::ComponentType::Unorm
                    | nexium_gpu::texture::ComponentType::UnormForceFp16
            )
        });
        let rgba16_float = data.tic.format == nexium_gpu::texture::TicFormat::R16G16B16A16
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let rgba32_float = data.tic.format == nexium_gpu::texture::TicFormat::R32G32B32A32
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let pps_rgba32_tld = self.qmd[0x08] == 0xc7000
            && raw == 0xdd3a_0000_a227_1c1c
            && rgba32_float
            && data.tic.texture_type == 1;
        let b10g11r11_float = data.tic.format == nexium_gpu::texture::TicFormat::B10G11R11
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == nexium_gpu::texture::ComponentType::Float);
        let z24s8_depth = data.tic.format == nexium_gpu::texture::TicFormat::Z24S8
            && data.tic.component_types
                == [
                    nexium_gpu::texture::ComponentType::Uint,
                    nexium_gpu::texture::ComponentType::Unorm,
                    nexium_gpu::texture::ComponentType::Unorm,
                    nexium_gpu::texture::ComponentType::Unorm,
                ]
            && data.tic.swizzle == [nexium_gpu::texture::SwizzleSource::G; 4];
        let buffer_format = match data.tic.format {
            nexium_gpu::texture::TicFormat::R32G32B32A32 => Some(4usize),
            nexium_gpu::texture::TicFormat::R32 => Some(1usize),
            _ => None,
        };
        let buffer_component_type = data.tic.component_types[0];
        let buffer_raw32 = tex_type == 0
            && data.tic.is_buffer()
            && !data.tic.is_block_linear
            && !data.tic.normalized_coords
            && !data.tic.is_srgb
            && data.tic.height == 1
            && data.tic.depth == 1
            && data.tic.view_base_mip() == 0
            && data.tic.view_mip_levels() == 1
            && buffer_format.is_some()
            && data
                .tic
                .component_types
                .iter()
                .all(|ty| *ty == buffer_component_type)
            && matches!(
                buffer_component_type,
                nexium_gpu::texture::ComponentType::Float
                    | nexium_gpu::texture::ComponentType::Uint
                    | nexium_gpu::texture::ComponentType::Sint
            );
        let coord = reg_a(raw);
        let x = get_reg(regs, coord) as i32;
        let y = if tex_type == 0 {
            0
        } else {
            get_reg(regs, coord.wrapping_add(1)) as i32
        };
        let supported = if tex_type == 0 {
            buffer_raw32
        } else {
            (rgba8_unorm || rgba16_float || pps_rgba32_tld || b10g11r11_float || z24s8_depth)
                && !data.tic.is_srgb
                && data.tic.view_base_mip() == 0
                && (!z24s8_depth || mask == 1)
        };
        if self.tex_trace && self.tex_logs < 16 {
            log::warn!(
                "KeplerCompute::tld handle={:#x} tic={} raw={:02x?} va={:#x} fmt={:?} types={:?} swizzle={:?} {}x{}x{} type={} block_linear={} srgb={} mip={}/{} coord=({},{}) mask={:#x} supported={}",
                handle,
                tic_index,
                data.raw,
                data.tic.gpu_va,
                data.tic.format,
                data.tic.component_types,
                data.tic.swizzle,
                data.tic.width,
                data.tic.height,
                data.layers,
                data.tic.texture_type,
                data.tic.is_block_linear,
                data.tic.is_srgb,
                data.tic.view_base_mip(),
                data.tic.view_mip_levels(),
                x,
                y,
                mask,
                supported
            );
            self.tex_logs += 1;
        }
        if !supported {
            return false;
        }

        let in_bounds =
            x >= 0 && y >= 0 && (x as u32) < data.tic.width && (y as u32) < data.tic.height;
        let values = if buffer_raw32 {
            let components = buffer_format.unwrap();
            let stride = components * 4;
            let physical = if in_bounds {
                let off = x as usize * stride;
                if off + stride > data.linear.len() {
                    return false;
                }
                let mut words = [0u32; 4];
                for (component, word) in words.iter_mut().enumerate().take(components) {
                    let component_off = off + component * 4;
                    *word = u32::from_le_bytes(
                        data.linear[component_off..component_off + 4]
                            .try_into()
                            .unwrap(),
                    );
                }
                words
            } else {
                [0; 4]
            };
            let one = if buffer_component_type == nexium_gpu::texture::ComponentType::Float {
                1.0f32.to_bits()
            } else {
                1
            };
            apply_swizzle_words(physical, data.tic.swizzle, one)
        } else if z24s8_depth {
            let depth = if in_bounds {
                let off = (y as usize * data.tic.width as usize + x as usize) * 4;
                if off + 4 > data.linear.len() {
                    return false;
                }
                let raw = u32::from_le_bytes(data.linear[off..off + 4].try_into().unwrap());
                f32::from_bits(z24s8_depth_bits(raw))
            } else {
                0.0
            };
            [depth.to_bits(); 4]
        } else if pps_rgba32_tld {
            let physical = if in_bounds {
                let Some(words) =
                    rgba32_float_texel_words(&data.linear, data.tic.width, x as usize, y as usize)
                else {
                    return false;
                };
                words
            } else {
                [0; 4]
            };
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else if rgba16_float {
            let physical = if in_bounds {
                let off = (y as usize * data.tic.width as usize + x as usize) * 8;
                if off + 8 > data.linear.len() {
                    return false;
                }
                std::array::from_fn(|component| {
                    let component_off = off + component * 2;
                    f16_to_f32_bits(u16::from_le_bytes([
                        data.linear[component_off],
                        data.linear[component_off + 1],
                    ]))
                })
            } else {
                [0; 4]
            };
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else if b10g11r11_float {
            let physical = if in_bounds {
                let off = (y as usize * data.tic.width as usize + x as usize) * 4;
                if off + 4 > data.linear.len() {
                    return false;
                }
                unpack_b10g11r11(u32::from_le_bytes(
                    data.linear[off..off + 4].try_into().unwrap(),
                ))
            } else {
                [0, 0, 0, 1.0f32.to_bits()]
            };
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        } else {
            let physical = if in_bounds {
                let off = (y as usize * data.tic.width as usize + x as usize) * 4;
                if off + 4 > data.rgba.len() {
                    return false;
                }
                [
                    (data.rgba[off] as f32 / 255.0).to_bits(),
                    (data.rgba[off + 1] as f32 / 255.0).to_bits(),
                    (data.rgba[off + 2] as f32 / 255.0).to_bits(),
                    (data.rgba[off + 3] as f32 / 255.0).to_bits(),
                ]
            } else {
                [0; 4]
            };
            apply_swizzle_words(physical, data.tic.swizzle, 1.0f32.to_bits())
        };
        let mut dest = reg_dest(raw);
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            set_reg(regs, dest, values[component]);
            dest = dest.wrapping_add(1);
        }
        true
    }
}

#[cfg(test)]
mod compute_texture_tests {
    use super::*;

    fn r32_uint_buffer_tic(gpu_va: u32, width: u32) -> [u8; 32] {
        let format_word: u32 =
            0x0f | (4 << 7) | (4 << 10) | (4 << 13) | (4 << 16) | (2 << 19) | (6 << 28);
        let mut raw = [0u8; 32];
        for (index, word) in [
            format_word,
            gpu_va,
            0,
            0,
            width.saturating_sub(1) | (6 << 23),
            0,
            0,
            0,
        ]
        .into_iter()
        .enumerate()
        {
            raw[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        raw
    }

    fn r16_uint_buffer_tic(gpu_va: u32, width: u32) -> [u8; 32] {
        let format_word: u32 =
            0x1b | (4 << 7) | (4 << 10) | (4 << 13) | (4 << 16) | (2 << 19) | (6 << 28);
        let mut raw = [0u8; 32];
        for (index, word) in [
            format_word,
            gpu_va,
            0,
            0,
            width.saturating_sub(1) | (6 << 23),
            0,
            0,
            0,
        ]
        .into_iter()
        .enumerate()
        {
            raw[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        raw
    }

    #[test]
    fn predecoded_code_matches_shader_opcode_lookup() {
        let raw = 0xe300_0000_0007_000f_u64;
        let mut code = vec![0u8; 0x20];
        code[8..16].copy_from_slice(&raw.to_le_bytes());

        let decoded = decode_code(&code);

        assert_eq!(decoded.len(), code.len() / 8);
        assert_eq!(
            decoded[1],
            nexium_shader::decode_one(raw).map(|instruction| instruction.opcode)
        );
    }

    #[test]
    fn bfe_clamps_count_and_sign_extends() {
        assert_eq!(bfe_value(0x8000_0000, 0x081c, false), 8);
        assert_eq!(bfe_value(0x8000_0000, 0x081c, true), 0xffff_fff8);
        assert_eq!(bfe_value(u32::MAX, 0x2020, false), 0);
    }

    #[test]
    fn pps_funnel_shift_wraps_register_count() {
        let raw = 0x5bfc_0480_0087_ff04;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SHF_l_reg
        );
        assert_eq!(
            (reg_dest(raw), reg_a(raw), reg_b(raw), reg_c(raw)),
            (4, RZ, 8, 9)
        );
        let mut regs = [0u32; 256];
        regs[8] = 35;
        regs[9] = 0x1234_5678;
        assert!(shf_l_reg(&mut regs, raw));
        assert_eq!(regs[4], 0x91a2_b3c0);
    }

    #[test]
    fn pps_iadd32i_negative_one_uses_the_full_immediate() {
        let x_raw = 0x1c0f_ffff_fff7_0002;
        let y_raw = 0x1c0f_ffff_fff7_0100;
        assert_eq!(imm32(x_raw), u32::MAX);
        assert_eq!(imm32(y_raw), u32::MAX);
        assert_eq!(iadd32i_value(x_raw, 10), 9);
        assert_eq!(iadd32i_value(y_raw, 10), 9);
    }

    #[test]
    fn pps_fmul32i_dither_factor_stays_positive() {
        let raw = 0x1e23_7800_0017_1919;
        let factor = f32::from_bits(0x3780_0001);
        assert_eq!(
            fmul32i_value(raw, 2.0f32.to_bits()),
            (2.0 * factor).to_bits()
        );
    }

    #[test]
    fn live_rt_crop_preserves_native_rows_and_pixel_bytes() {
        let source: Vec<u8> = (0..16).collect();
        let cropped = crop_linear_rows(&source, 4, 2, 3, 2, 2).unwrap();
        assert_eq!(cropped, vec![0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]);
    }

    #[test]
    fn live_rt_crop_rejects_invalid_dimensions_and_storage() {
        assert!(crop_linear_rows(&[0; 16], 4, 2, 5, 2, 2).is_none());
        assert!(crop_linear_rows(&[0; 16], 4, 2, 3, 3, 2).is_none());
        assert!(crop_linear_rows(&[0; 15], 4, 2, 3, 2, 2).is_none());
        assert!(crop_linear_rows(&[0; 16], 4, 2, 3, 2, 0).is_none());
    }

    #[test]
    fn pps_fmnmx_cbuf_is_ftz_min() {
        let raw = 0x4c60_138c_0087_0606;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::FMNMX_cbuf
        );
        let cb = nexium_shader::cbuf(raw);
        assert_eq!((cb.binding, cb.byte_offset), (3, 0x20));
        assert_eq!((reg_dest(raw), reg_a(raw)), (6, 6));
        assert!(bit(raw, 44));
        assert!(!bit(raw, 47));

        let mut preds = [false; 8];
        preds[PT as usize] = true;
        assert_eq!(
            fmnmx_value(9.0f32.to_bits(), 4.0f32.to_bits(), raw, &preds),
            Some(4.0f32.to_bits())
        );
        assert_eq!(
            fmnmx_value(2.0f32.to_bits(), 4.0f32.to_bits(), raw, &preds),
            Some(2.0f32.to_bits())
        );
    }

    #[test]
    fn fmnmx_selects_max_and_preserves_signed_zero_under_ftz() {
        let min_raw = 0x4c60_138c_0087_0606;
        let max_raw = min_raw | (1 << 42);
        let mut preds = [false; 8];
        preds[PT as usize] = true;

        assert_eq!(
            fmnmx_value(2.0f32.to_bits(), 4.0f32.to_bits(), max_raw, &preds),
            Some(4.0f32.to_bits())
        );
        assert_eq!(
            fmnmx_value(0x0000_0000, 0x8000_0000, min_raw, &preds),
            Some(0x8000_0000)
        );
        assert_eq!(
            fmnmx_value(0x0000_0000, 0x8000_0000, max_raw, &preds),
            Some(0x0000_0000)
        );
        assert_eq!(
            fmnmx_value(0x8000_0001, 0x0000_0000, min_raw, &preds),
            Some(0x8000_0000)
        );
        assert_eq!(
            fmnmx_value(f32::NAN.to_bits(), 2.0f32.to_bits(), min_raw, &preds),
            Some(2.0f32.to_bits())
        );
        assert_eq!(
            fmnmx_value(2.0f32.to_bits(), f32::NAN.to_bits(), max_raw, &preds),
            Some(2.0f32.to_bits())
        );
        assert_eq!(
            fmnmx_value(
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                min_raw | (1 << 47),
                &preds
            ),
            None
        );
    }

    #[test]
    fn pps_tld_bindless_is_plain_rgba_2d_fetch() {
        let raw = 0xdd3a_0007_a067_0400;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::TLD_b
        );
        assert_eq!((reg_dest(raw), reg_a(raw), reg_b(raw)), (0, 4, 6));
        assert_eq!(bits(raw, 28, 30), 2);
        assert_eq!(bits(raw, 31, 34), 0xf);
        assert_eq!(bits(raw, 51, 53) as u8, PT);
        assert!(!bit(raw, 35));
        assert!(!bit(raw, 50));
        assert!(!bit(raw, 54));
        assert!(!bit(raw, 55));
        assert_eq!(0x1000_021f & 0x000f_ffff, 0x21f);

        let reflection_constant = 0xdd3a_0000_a227_1c1c;
        assert_eq!(
            nexium_shader::decode_one(reflection_constant)
                .unwrap()
                .opcode,
            nexium_shader::Opcode::TLD_b
        );
        assert_eq!(
            (
                reg_dest(reflection_constant),
                reg_a(reflection_constant),
                reg_b(reflection_constant),
            ),
            (28, 28, 34)
        );
        assert_eq!(bits(reflection_constant, 28, 30), 2);
        assert_eq!(bits(reflection_constant, 31, 34), 1);
        assert_eq!(bits(reflection_constant, 51, 53) as u8, PT);
    }

    #[test]
    fn pps_rgba32_float_tld_preserves_all_component_bits() {
        let words = [
            1.25f32.to_bits(),
            (-0.0f32).to_bits(),
            f32::NAN.to_bits(),
            f32::INFINITY.to_bits(),
        ];
        let mut data = vec![0u8; 16];
        for word in words {
            data.extend_from_slice(&word.to_le_bytes());
        }
        assert_eq!(rgba32_float_texel_words(&data, 2, 1, 0), Some(words));
        assert_eq!(rgba32_float_texel_words(&data[..31], 2, 1, 0), None);
    }

    #[test]
    fn pps_tld_bindless_1d_fetch_preserves_raw_words() {
        let raw = 0xdd38_0000_8017_0406;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::TLD_b
        );
        assert_eq!((reg_dest(raw), reg_a(raw), reg_b(raw)), (6, 4, 1));
        assert_eq!(bits(raw, 28, 30), 0);
        assert_eq!(bits(raw, 31, 34), 1);
        assert_eq!(bits(raw, 51, 53) as u8, PT);
        assert!(!bit(raw, 35));
        assert!(!bit(raw, 50));
        assert!(!bit(raw, 54));
        assert!(!bit(raw, 55));

        let words = [0x3f80_0000, 0x8000_0000, 0x7fc0_1234, 0xffff_ffff];
        assert_eq!(
            apply_swizzle_words(
                words,
                [
                    nexium_gpu::texture::SwizzleSource::R,
                    nexium_gpu::texture::SwizzleSource::G,
                    nexium_gpu::texture::SwizzleSource::B,
                    nexium_gpu::texture::SwizzleSource::A,
                ],
                1.0f32.to_bits(),
            ),
            words
        );
    }

    #[test]
    fn tld_fp16_conversion_preserves_values_and_signed_zero() {
        assert_eq!(f16_to_f32_bits(0x0000), 0.0f32.to_bits());
        assert_eq!(f16_to_f32_bits(0x8000), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32_bits(0x3c00), 1.0f32.to_bits());
        assert_eq!(f16_to_f32_bits(0xc000), (-2.0f32).to_bits());
        assert_eq!(f16_to_f32_bits(0x7c00), f32::INFINITY.to_bits());
        assert!(f32::from_bits(f16_to_f32_bits(0x7e00)).is_nan());
        assert_eq!(f16_to_f32_bits(0x0001), 2f32.powi(-24).to_bits());
    }

    #[test]
    fn pps_b10g11r11_store_matches_unsigned_float_layout() {
        assert_eq!(f32_to_ufloat_bits(0.0f32.to_bits(), 6), 0);
        assert_eq!(f32_to_ufloat_bits((-1.0f32).to_bits(), 6), 0);
        assert_eq!(f32_to_ufloat_bits(2f32.powi(-20).to_bits(), 6), 1);
        assert_eq!(f32_to_ufloat_bits(2f32.powi(-21).to_bits(), 6), 0);
        assert_eq!(f32_to_ufloat_bits(1.0f32.to_bits(), 6), 0x3c0);
        assert_eq!(f32_to_ufloat_bits(f32::INFINITY.to_bits(), 6), 0x7c0);
        assert_eq!(f32_to_ufloat_bits(f32::NAN.to_bits(), 6), 0x7ff);
        assert_eq!(f32_to_ufloat_bits(100_000_000.0f32.to_bits(), 6), 0x7bf);
        assert_eq!(f32_to_ufloat_bits(2f32.powi(-19).to_bits(), 5), 1);
        assert_eq!(f32_to_ufloat_bits(2f32.powi(-20).to_bits(), 5), 0);
        assert_eq!(f32_to_ufloat_bits(1.0f32.to_bits(), 5), 0x1e0);
        assert_eq!(f32_to_ufloat_bits(f32::INFINITY.to_bits(), 5), 0x3e0);
        assert_eq!(f32_to_ufloat_bits(f32::NAN.to_bits(), 5), 0x3ff);
        assert_eq!(f32_to_ufloat_bits(100_000_000.0f32.to_bits(), 5), 0x3df);
        assert_eq!(ufloat_to_f32_bits(1, 6), 2f32.powi(-20).to_bits());
        assert_eq!(ufloat_to_f32_bits(0x3c0, 6), 1.0f32.to_bits());
        assert_eq!(ufloat_to_f32_bits(0x1e0, 5), 1.0f32.to_bits());
        assert_eq!(ufloat_to_f32_bits(0x7c0, 6), f32::INFINITY.to_bits());
        assert!(f32::from_bits(ufloat_to_f32_bits(0x7ff, 6)).is_nan());
        assert_eq!(
            pack_b10g11r11(
                0.0f32.to_bits(),
                0.5f32.to_bits(),
                100_000_000.0f32.to_bits(),
            ),
            0xf7dc_0000
        );
        assert_eq!(
            unpack_b10g11r11(pack_b10g11r11(
                0.0f32.to_bits(),
                0.5f32.to_bits(),
                1.0f32.to_bits(),
            )),
            [
                0.0f32.to_bits(),
                0.5f32.to_bits(),
                1.0f32.to_bits(),
                1.0f32.to_bits(),
            ]
        );
    }

    #[test]
    fn tld_word_swizzle_uses_float_one() {
        use nexium_gpu::texture::SwizzleSource::{One, Zero, A, B};
        let src = [10, 20, 30, 40];
        assert_eq!(
            apply_swizzle_words(src, [A, Zero, B, One], 1.0f32.to_bits()),
            [40, 0, 30, 1.0f32.to_bits()]
        );
    }

    #[test]
    fn pps_z24s8_tld_reads_low_24_bit_normalized_depth() {
        assert_eq!(z24s8_depth_bits(0xff00_0000), 0.0f32.to_bits());
        assert_eq!(z24s8_depth_bits(0x00ff_ffff), 1.0f32.to_bits());
        assert_eq!(z24s8_depth_bits(0xffff_ffff), 1.0f32.to_bits());
        assert_eq!(
            z24s8_depth_bits(0x5a80_0000),
            (0x80_0000 as f32 / 16_777_215.0).to_bits()
        );
    }

    #[test]
    fn pps_hsetp2_tests_low_half_for_ordered_zero() {
        let raw = 0x5d21_0390_2ff7_1117;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::HSETP2_reg
        );
        assert_eq!(
            (
                bits(raw, 3, 5),
                bits(raw, 0, 2),
                reg_a(raw),
                reg_b(raw),
                bits(raw, 47, 48),
                bits(raw, 28, 29),
                bits(raw, 35, 38),
                bits(raw, 45, 46),
                bit(raw, 49),
                bit(raw, 6),
            ),
            (2, 7, 17, RZ, 2, 2, 2, 0, false, false)
        );

        for (value, expected) in [
            (0x3c00_0000, true),
            (0x0000_3c00, false),
            (0xbeef_8000, true),
            (0xbeef_0001, false),
            (0xbeef_7e00, false),
        ] {
            let mut regs = [0u32; 256];
            regs[17] = value;
            let mut preds = [false; 8];
            preds[PT as usize] = true;
            assert!(hsetp2_reg_pps(&regs, &mut preds, raw));
            assert_eq!(preds[2], expected);
            assert!(preds[PT as usize]);
        }
    }

    #[test]
    fn compute_tsc_cache_reads_sampler_once_per_dispatch() {
        let raw = [
            0x92, 0x60, 0x02, 0x00, 0xa2, 0x03, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let qmd = [0u32; 0x40];
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x1000, 0x9000, 1);
        let reads = std::cell::Cell::new(0u32);
        let read = |cpu: u64, out: &mut [u8]| {
            reads.set(reads.get() + 1);
            if cpu != 0x9000 || out.len() != raw.len() {
                return false;
            }
            out.copy_from_slice(&raw);
            true
        };
        let write = |_: u64, _: &[u8]| false;
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tsc_pool_gpu_va: 0x4000,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        let first = exec.tsc_entry(0).expect("first sampler lookup");
        let second = exec.tsc_entry(0).expect("cached sampler lookup");
        assert_eq!(first.raw, raw);
        assert_eq!(first.entry, second.entry);
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn pps_2513900_skips_only_the_exact_redundant_shared_clear() {
        let first = 0xef5c_0000_0007_03ffu64;
        let mut code = vec![0u8; 0x350];
        code[0x328..0x330].copy_from_slice(&0xef5c_0001_f807_03ffu64.to_le_bytes());
        code[0x348..0x350].copy_from_slice(&0x5c10_0000_00c7_0d03u64.to_le_bytes());
        assert_eq!(
            pps_2513900_shared_clear_end(0x2513900, &code, 0x70, first, 0x2000),
            Some(0x348)
        );
        assert_eq!(
            pps_2513900_shared_clear_end(0x2513901, &code, 0x70, first, 0x2000),
            None
        );
        code[0x328] ^= 1;
        assert_eq!(
            pps_2513900_shared_clear_end(0x2513900, &code, 0x70, first, 0x2000),
            None
        );
    }

    #[test]
    fn pps_tex_b_is_bindless_2d_lod_zero_red_sample() {
        for (raw, meta) in [(0xdeb8_0020_a0d7_0604, 13), (0xdeb8_0020_a0e7_0604, 14)] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::TEX_b
            );
            assert_eq!((reg_dest(raw), reg_a(raw), reg_b(raw)), (4, 6, meta));
            assert_eq!(bits(raw, 28, 30), 2);
            assert_eq!(bits(raw, 31, 34), 1);
            assert!(!bit(raw, 35));
            assert!(!bit(raw, 36));
            assert_eq!(bits(raw, 37, 39), 1);
            assert!(!bit(raw, 40));
            assert!(!bit(raw, 50));
            assert_eq!(bits(raw, 51, 53), PT as u64);
        }
        assert_eq!(0x1020_0217 & 0x000f_ffff, 535);
        assert_eq!(0x1020_0217 >> 20, 258);

        let composite = 0xdeba_0027_a0e7_0c0c;
        assert_eq!(
            nexium_shader::decode_one(composite).unwrap().opcode,
            nexium_shader::Opcode::TEX_b
        );
        assert_eq!(
            (reg_dest(composite), reg_a(composite), reg_b(composite)),
            (12, 12, 14)
        );
        assert_eq!(bits(composite, 28, 30), 2);
        assert_eq!(bits(composite, 31, 34), 0xf);
        assert!(!bit(composite, 36));
        assert_eq!(bits(composite, 37, 39), 1);
        assert!(bit(composite, 49));
        assert!(!bit(composite, 50));
        assert_eq!(bits(composite, 51, 53), PT as u64);
        let linear_tsc_raw = [
            0x92, 0x60, 0x02, 0x00, 0xa2, 0x03, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let linear_tsc = nexium_gpu::texture::TscEntry::parse(&linear_tsc_raw).unwrap();
        assert_eq!(
            linear_tsc.mag_filter,
            nexium_gpu::texture::TexFilter::Linear
        );
        assert_eq!(
            linear_tsc.min_filter,
            nexium_gpu::texture::TexFilter::Linear
        );
        assert_eq!(
            linear_tsc.mip_filter,
            nexium_gpu::texture::TexFilter::Nearest
        );

        let tic_raw = [
            0x9b, 0xff, 0x17, 0x70, 0x00, 0x00, 0x85, 0x26, 0x05, 0x00, 0x60, 0x00, 0x20, 0x00,
            0x07, 0x90, 0xff, 0x03, 0x80, 0xe8, 0xff, 0x01, 0x00, 0x80, 0x00, 0x00, 0x00, 0x03,
            0x33, 0x00, 0x00, 0x00,
        ];
        let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw).unwrap();
        let layout = nexium_gpu::texture::block_linear_mip_layout(&tic).unwrap();
        assert_eq!(
            (
                tic.width,
                tic.height,
                tic.view_base_mip(),
                tic.view_mip_levels(),
                tic.block_height_log2,
                layout.layer_size,
                layout.layer_stride,
            ),
            (1024, 512, 3, 1, 4, 0x155c00, 0x156000)
        );
        let mip3 = &layout.levels[3];
        assert_eq!(
            (
                mip3.width,
                mip3.height,
                mip3.block_height_log2,
                mip3.stride_alignment_log2,
                mip3.guest_offset,
                mip3.guest_size,
                mip3.linear_size,
            ),
            (128, 64, 3, 5, 0x150000, 0x4000, 0x4000)
        );
        assert_eq!(coord_to_index(0.009765625, mip3.width, true), 1);
        assert_eq!(coord_to_index(0.01953125, mip3.height, true), 1);

        let mip4 = &layout.levels[4];
        assert_eq!(
            (
                mip4.width,
                mip4.height,
                mip4.block_height_log2,
                mip4.stride_alignment_log2,
                mip4.guest_offset,
                mip4.guest_size,
                mip4.linear_size,
            ),
            (64, 32, 2, 5, 0x154000, 0x1000, 0x1000)
        );
        let mip8 = &layout.levels[8];
        assert_eq!(
            (
                mip8.width,
                mip8.height,
                mip8.block_height_log2,
                mip8.stride_alignment_log2,
                mip8.guest_offset,
                mip8.guest_size,
                mip8.linear_size,
            ),
            (4, 2, 0, 5, 0x155800, 0x200, 0x10)
        );
        assert_eq!(
            block_linear_pixel_offset_strided(64, 32, 2, 2, 5, 63, 31),
            Some(0xffe)
        );
        assert_eq!(
            block_linear_pixel_offset_strided(4, 2, 2, 0, 5, 3, 1),
            Some(0x16)
        );
    }

    #[test]
    fn pps_rgba16f_linear_sample_uses_normalized_centers_offsets_and_edge_clamp() {
        let texels = [
            [0.0f32, 10.0, 20.0, 30.0],
            [2.0f32, 12.0, 22.0, 32.0],
            [4.0f32, 14.0, 24.0, 34.0],
            [6.0f32, 16.0, 26.0, 36.0],
        ];
        let mut data = Vec::new();
        for texel in texels {
            for component in texel {
                data.extend_from_slice(&f32_to_f16_bits(component.to_bits()).to_le_bytes());
            }
        }
        let sample = |u, v, offset_x, offset_y| {
            sample_rgba16f_linear_clamp(&data, 2, 2, u, v, offset_x, offset_y)
                .unwrap()
                .map(f32::from_bits)
        };
        assert_eq!(sample(0.25, 0.25, 0, 0), texels[0]);
        assert_eq!(sample(0.75, 0.75, 0, 0), texels[3]);
        assert_eq!(sample(0.5, 0.5, 0, 0), [3.0, 13.0, 23.0, 33.0]);
        assert_eq!(sample(0.25, 0.25, 1, 0), texels[1]);
        assert_eq!(sample(0.0, 0.5, 0, 0), [2.0, 12.0, 22.0, 32.0]);
        assert!(sample_rgba16f_linear_clamp(&data, 2, 2, f32::NAN, 0.5, 0, 0).is_none());
    }

    #[test]
    fn pps_depth_pyramid_sust_variants_decode() {
        for (raw, data, coord, handle) in [
            (0xeb20_0486_00f7_0204, 4, 2, 9),
            (0xeb20_0206_00f7_0208, 8, 2, 4),
            (0xeb20_0086_00f7_0204, 4, 2, 1),
            (0xeb20_0586_00f7_0004, 4, 0, 11),
            (0xeb20_0506_00f7_0004, 4, 0, 10),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::SUST
            );
            assert_eq!((reg_dest(raw), reg_a(raw)), (data, coord));
            assert_eq!(bits(raw, 39, 46), handle);
            assert_eq!(bits(raw, 20, 23), 0xf);
            assert_eq!(bits(raw, 33, 35), 3);
        }
    }

    #[test]
    fn pps_compute_dump_sust_fields_match_yuzu() {
        for (pc, raw, data, coord, surface_type, handle) in [
            (0x02c8, 0xeb20_0682_00f7_0700, 0, 7, 1, 13),
            (0x02f0, 0xeb20_0682_00f7_0c08, 8, 12, 1, 13),
            (0x03b8, 0xeb20_0382_00f7_0208, 8, 2, 1, 7),
            (0x05b8, 0xeb20_0582_00f7_0800, 0, 8, 1, 11),
            (0x05f0, 0xeb20_0582_00f7_0904, 4, 9, 1, 11),
            (0x06b0, 0xeb20_0602_00f7_0304, 4, 3, 1, 12),
            (0x1770, 0xeb20_0306_00f7_1400, 0, 20, 3, 6),
            (0x1a68, 0xeb20_0506_00f7_0004, 4, 0, 3, 10),
            (0x1c78, 0xeb20_0186_00f7_0004, 4, 0, 3, 3),
            (0x1f88, 0xeb20_0406_00f7_1600, 0, 22, 3, 8),
            (0x21b0, 0xeb20_0306_00f7_0400, 0, 4, 3, 6),
            (0x23d0, 0xeb20_0306_00f7_0400, 0, 4, 3, 6),
            (0x2890, 0xeb20_0306_00f7_0400, 0, 4, 3, 6),
            (0x2ab8, 0xeb20_0306_00f7_0400, 0, 4, 3, 6),
            (0x2d90, 0xeb20_0506_00f7_1804, 4, 24, 3, 10),
            (0x2fb0, 0xeb20_0406_00f7_0204, 4, 2, 3, 8),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::SUST,
                "pc={pc:#x}"
            );
            assert_eq!((reg_dest(raw), reg_a(raw)), (data, coord), "pc={pc:#x}");
            assert_eq!(bits(raw, 39, 46), handle, "pc={pc:#x}");
            assert_eq!(bits(raw, 16, 18), PT as u64, "pc={pc:#x}");
            assert!(!bit(raw, 19), "pc={pc:#x}");
            assert_eq!(bits(raw, 20, 23), 0xf, "pc={pc:#x}");
            assert_eq!(bits(raw, 24, 25), 0, "pc={pc:#x}");
            assert_eq!(bits(raw, 33, 35), surface_type, "pc={pc:#x}");
            assert_eq!(bits(raw, 49, 50), 0, "pc={pc:#x}");
            assert!(!bit(raw, 51), "pc={pc:#x}");
            assert!(!bit(raw, 52), "pc={pc:#x}");
            assert_eq!(is_pps_b6300_buffer_sust(raw), surface_type == 1);
        }
    }

    #[test]
    fn pps_5dbe_ldc_b64_uses_dynamic_cbuf4_offsets() {
        let first = 0xef95_0040_0007_0104;
        let second = 0xef95_0040_0087_0102;
        for (raw, dest, immediate) in [(first, 4, 0), (second, 2, 8)] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::LDC
            );
            assert_eq!((reg_dest(raw), reg_a(raw)), (dest, 1));
            assert_eq!(bits(raw, 20, 35), immediate);
            assert_eq!(bits(raw, 36, 40), 4);
            assert_eq!(bits(raw, 44, 45), 0);
            assert_eq!(bits(raw, 48, 50), 5);
        }

        let words = [
            0x0102_0304u32,
            0x1112_1314,
            0x2122_2324,
            0x3132_3334,
            0x4142_4344,
        ];
        let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0x5dbe00;
        qmd[0x14] = 1 << 4;
        qmd[0x25] = 0x4000;
        qmd[0x26] = (bytes.len() as u32) << 15;
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x1000, 0x9000, 1);
        let reads = std::cell::Cell::new(0u32);
        let read = |cpu: u64, out: &mut [u8]| {
            reads.set(reads.get() + 1);
            let Some(offset) = cpu.checked_sub(0x9000).map(|value| value as usize) else {
                return false;
            };
            if offset + out.len() > bytes.len() {
                return false;
            }
            out.copy_from_slice(&bytes[offset..offset + out.len()]);
            true
        };
        let write = |_: u64, _: &[u8]| false;
        let cbuf_data = snapshot_compute_cbufs(&qmd, &mappings, &read);
        assert_eq!(reads.get(), 1);
        let exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data,
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState::default(),
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };
        let mut regs = [0u32; 256];
        regs[1] = 4;

        assert!(exec.ldc_pps_5dbe(&mut regs, first));
        assert_eq!((regs[4], regs[5]), (words[1], words[2]));
        assert!(exec.ldc_pps_5dbe(&mut regs, second));
        assert_eq!((regs[2], regs[3]), (words[3], words[4]));
        assert!(!exec.ldc_pps_5dbe(&mut regs, first ^ (1 << 48)));
        assert_eq!(reads.get(), 1);
    }

    #[test]
    fn pps_5dbe_suatom_add_and_exchange_return_old_values() {
        let add = 0xea70_0382_0020_0502;
        let exchange_handle_dest_alias = 0xea70_0203_00a7_0d04;
        let exchange_coord_dest_alias = 0xea70_0583_0097_0404;
        for (raw, operation, dest, coord, operand, handle) in [
            (add, 0, 2, 5, 2, 7),
            (exchange_handle_dest_alias, 8, 4, 13, 10, 4),
            (exchange_coord_dest_alias, 8, 4, 4, 9, 11),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::SUATOM
            );
            assert!(is_pps_5dbe_buffer_suatom(raw));
            assert_eq!(
                (
                    reg_dest(raw),
                    reg_a(raw),
                    bits(raw, 20, 27),
                    bits(raw, 39, 46)
                ),
                (dest, coord, operand, handle)
            );
            assert_eq!(bits(raw, 29, 32), operation);
            assert_eq!(bits(raw, 33, 35), 1);
            assert_eq!(bits(raw, 49, 50), 0);
            assert_eq!(bits(raw, 51, 53), 6);
            assert!(bit(raw, 54));
        }

        let tic_raw = r32_uint_buffer_tic(0x2000, 4);
        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0x5dbe00;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x2000, 0x1000, 0xa000, 2);
        let initial = [10u32, 20, 30, 40];
        let data = std::cell::RefCell::new(
            initial
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu == 0x9000 && out.len() == tic_raw.len() {
                out.copy_from_slice(&tic_raw);
                return true;
            }
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + out.len() > data.borrow().len() {
                return false;
            }
            out.copy_from_slice(&data.borrow()[offset..offset + out.len()]);
            true
        };
        let write = |cpu: u64, bytes: &[u8]| {
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + bytes.len() > data.borrow().len() {
                return false;
            }
            data.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            true
        };
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tic_pool_gpu_va: 0x1000,
                tic_limit: 0,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        let mut regs = [0u32; 256];
        regs[2] = 5;
        regs[5] = 0;
        regs[7] = 0;
        assert!(exec.suatom(&mut regs, add));
        assert_eq!(regs[2], 10);

        regs = [0; 256];
        regs[4] = 0;
        regs[10] = 0xaaaa_0001;
        regs[13] = 1;
        assert!(exec.suatom(&mut regs, exchange_handle_dest_alias));
        assert_eq!(regs[4], 20);

        regs = [0; 256];
        regs[4] = 2;
        regs[9] = 0xbbbb_0002;
        regs[11] = 0;
        assert!(exec.suatom(&mut regs, exchange_coord_dest_alias));
        assert_eq!(regs[4], 30);

        let data = data.borrow();
        let actual: Vec<u32> = data
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, vec![15, 0xaaaa_0001, 0xbbbb_0002, 40]);
        assert_eq!(exec.writes, 3);
    }

    #[test]
    fn pps_5dbe_sust_writes_all_four_compaction_words() {
        let stores = [
            (0xeb20_0582_00f7_0800, 0u8, 8u8, 11u8),
            (0xeb20_0582_00f7_0904, 4, 9, 11),
            (0xeb20_0a82_00f7_0800, 0, 8, 21),
            (0xeb20_0a82_00f7_0a04, 4, 10, 21),
        ];
        for (raw, data, coord, handle) in stores {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::SUST
            );
            assert!(is_pps_5dbe_buffer_sust(raw));
            assert_eq!((reg_dest(raw), reg_a(raw)), (data, coord));
            assert_eq!(bits(raw, 39, 46), handle as u64);
            assert_eq!(bits(raw, 20, 23), 0xf);
            assert_eq!(bits(raw, 24, 25), 0);
            assert_eq!(bits(raw, 33, 35), 1);
            assert_eq!(bits(raw, 49, 52), 0);
        }

        let tic_raw = r32_uint_buffer_tic(0x2000, 4);
        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0x5dbe00;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x2000, 0x1000, 0xa000, 2);
        let data = std::cell::RefCell::new(vec![0u8; 16]);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x9000 || out.len() != tic_raw.len() {
                return false;
            }
            out.copy_from_slice(&tic_raw);
            true
        };
        let write = |cpu: u64, bytes: &[u8]| {
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + bytes.len() > data.borrow().len() {
                return false;
            }
            data.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            true
        };
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tic_pool_gpu_va: 0x1000,
                tic_limit: 0,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        for (index, (raw, data_reg, coord_reg, handle_reg)) in stores.into_iter().enumerate() {
            let mut regs = [0u32; 256];
            regs[data_reg as usize] = 0x5100_0000 | index as u32;
            regs[coord_reg as usize] = index as u32;
            regs[handle_reg as usize] = 0;
            assert!(exec.sust(&regs, raw));
        }

        let data = data.borrow();
        let actual: Vec<u32> = data
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual,
            vec![0x5100_0000, 0x5100_0001, 0x5100_0002, 0x5100_0003]
        );
        assert_eq!(exec.writes, 4);
    }

    #[test]
    fn pps_scene_sust_writes_r32_buffer_texels() {
        let stores = [
            (0x5bc00, 0xeb20_0082_00f7_0004, 4usize, 0usize, 1usize),
            (0x5a800, 0xeb20_0302_00f7_0500, 0, 5, 6),
        ];
        let tic_raw = r32_uint_buffer_tic(0x2000, stores.len() as u32);
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x2000, 0x1000, 0xa000, 2);
        let data = std::cell::RefCell::new(vec![0u8; stores.len() * 4]);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x9000 || out.len() != tic_raw.len() {
                return false;
            }
            out.copy_from_slice(&tic_raw);
            true
        };
        let write = |cpu: u64, bytes: &[u8]| {
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + bytes.len() > data.borrow().len() {
                return false;
            }
            data.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            true
        };

        for (index, (program, raw, data_reg, coord_reg, handle_reg)) in
            stores.into_iter().enumerate()
        {
            assert!(is_pps_scene_buffer_sust(program, raw));
            let mut qmd = [0u32; 0x40];
            qmd[0x08] = program;
            let mut exec = ComputeExec {
                qmd: &qmd,
                code: &[],
                decoded_code: Vec::new(),
                cbuf_data: empty_compute_cbufs(),
                mappings: &mappings,
                mem_read: &read,
                mem_write: &write,
                renderer: None,
                texture: ComputeTextureState {
                    tic_pool_gpu_va: 0x1000,
                    tic_limit: 0,
                    ..ComputeTextureState::default()
                },
                tex_cache: HashMap::new(),
                tsc_cache: HashMap::new(),
                sust_tic_cache: HashMap::new(),
                write_page_cache: HashMap::new(),
                invalidated_pages: HashSet::new(),
                tex_trace: false,
                pps_5d_trace: None,
                tex_logs: 0,
                sust_logs: 0,
                writes: 0,
                unsupported: None,
                map_cache: std::cell::Cell::new(None),
            };
            let mut regs = [0u32; 256];
            regs[data_reg] = 0x5200_0000 | index as u32;
            regs[coord_reg] = index as u32;
            regs[handle_reg] = 0;
            assert!(
                exec.sust(&regs, raw),
                "program={program:#x} raw={raw:#018x}"
            );
            assert_eq!(exec.writes, 1);
        }

        let actual: Vec<u32> = data
            .borrow()
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, vec![0x5200_0000, 0x5200_0001]);
    }

    #[test]
    fn pps_cluster_sust_writes_rgba32_buffer_texel() {
        let format_word: u32 = 0x01
            | (4 << 7)
            | (4 << 10)
            | (4 << 13)
            | (4 << 16)
            | (2 << 19)
            | (3 << 22)
            | (4 << 25)
            | (5 << 28);
        let mut tic_raw = [0u8; 32];
        for (index, word) in [format_word, 0x1_0000, 0, 0, 3 | (6 << 23), 0, 0, 0]
            .into_iter()
            .enumerate()
        {
            tic_raw[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0xb6300;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x1_0000, 0x1_0000, 0xa000, 2);
        let descriptor_reads = std::cell::Cell::new(0u32);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x9000 || out.len() != tic_raw.len() {
                return false;
            }
            descriptor_reads.set(descriptor_reads.get() + 1);
            out.copy_from_slice(&tic_raw);
            true
        };
        let stores = std::cell::RefCell::new(Vec::<(u64, Vec<u8>)>::new());
        let write = |cpu: u64, bytes: &[u8]| {
            stores.borrow_mut().push((cpu, bytes.to_vec()));
            true
        };
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tic_pool_gpu_va: 0x1000,
                tic_limit: 0,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };
        let values = [0x0123_4567, 0x89ab_cdef, 0x1357_9bdf, 0x2468_ace0];
        for (raw, data, coord, handle) in [
            (0xeb20_0682_00f7_0700, 0usize, 7usize, 13usize),
            (0xeb20_0682_00f7_0c08, 8, 12, 13),
            (0xeb20_0382_00f7_0208, 8, 2, 7),
            (0xeb20_0582_00f7_0800, 0, 8, 11),
            (0xeb20_0582_00f7_0904, 4, 9, 11),
            (0xeb20_0602_00f7_0304, 4, 3, 12),
        ] {
            let mut regs = [0u32; 256];
            regs[data..data + 4].copy_from_slice(&values);
            regs[coord] = 2;
            regs[handle] = 0;
            assert!(exec.sust(&regs, raw), "raw={raw:#018x}");
        }
        assert_eq!(exec.writes, 6);
        let expected: Vec<u8> = values.into_iter().flat_map(u32::to_le_bytes).collect();
        assert_eq!(&*stores.borrow(), &vec![(0xa020, expected); 6]);
        assert_eq!(descriptor_reads.get(), 1);
        assert_eq!(exec.sust_tic_cache.len(), 1);
        assert_eq!(exec.write_page_cache.len(), 1);
        assert_eq!(exec.invalidated_pages.len(), 1);
    }

    #[test]
    fn pps_cluster_sust_writes_r16_uint_buffer_texel() {
        let tic_raw = r16_uint_buffer_tic(0x2000, 4);
        let tic = nexium_gpu::texture::TicEntry::parse(&tic_raw).unwrap();
        assert_eq!(
            u32::from_le_bytes(tic_raw[..4].try_into().unwrap()),
            0x6014_921b
        );
        assert_eq!(tic.format, nexium_gpu::texture::TicFormat::R16);
        assert_eq!(
            tic.component_types,
            [nexium_gpu::texture::ComponentType::Uint; 4]
        );
        assert_eq!(
            tic.swizzle,
            [
                nexium_gpu::texture::SwizzleSource::R,
                nexium_gpu::texture::SwizzleSource::Zero,
                nexium_gpu::texture::SwizzleSource::Zero,
                nexium_gpu::texture::SwizzleSource::One,
            ]
        );
        assert_eq!((tic.gpu_va, tic.width, tic.texture_type), (0x2000, 4, 6));

        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0xb6300;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x2000, 0x1000, 0xa000, 2);
        let data = std::cell::RefCell::new(vec![0x5a; 8]);
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu != 0x9000 || out.len() != tic_raw.len() {
                return false;
            }
            out.copy_from_slice(&tic_raw);
            true
        };
        let write = |cpu: u64, bytes: &[u8]| {
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + bytes.len() > data.borrow().len() {
                return false;
            }
            data.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            true
        };
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tic_pool_gpu_va: 0x1000,
                tic_limit: 0,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        let pc_3b8 = 0xeb20_0382_00f7_0208;
        let mut regs = [0u32; 256];
        regs[8] = 0xdead_1234;
        regs[9] = 0xffff_ffff;
        regs[2] = 1;
        regs[7] = 0;
        assert!(exec.sust(&regs, pc_3b8));

        let pc_6b0 = 0xeb20_0602_00f7_0304;
        regs = [0; 256];
        regs[4] = 0xbeef_abcd;
        regs[5] = 0x1111_2222;
        regs[3] = 3;
        regs[12] = 0;
        assert!(exec.sust(&regs, pc_6b0));

        regs[3] = 4;
        assert!(exec.sust(&regs, pc_6b0));
        assert_eq!(
            &*data.borrow(),
            &[0x5a, 0x5a, 0x34, 0x12, 0x5a, 0x5a, 0xcd, 0xab]
        );
        assert_eq!(exec.writes, 2);
    }

    #[test]
    fn pps_cluster_suatom_add_returns_old_buffer_value() {
        for raw in [0xea70_0282_0022_0701, 0xea70_0282_0022_0705] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::SUATOM
            );
            assert!(is_pps_b6300_buffer_suatom(raw));
            assert_eq!(bits(raw, 29, 32), 0);
            assert_eq!(bits(raw, 33, 35), 1);
            assert_eq!(bits(raw, 49, 50), 0);
            assert_eq!(bits(raw, 51, 53), 6);
            assert!(bit(raw, 54));
        }

        let format_word: u32 = 0x0f
            | (4 << 7)
            | (4 << 10)
            | (4 << 13)
            | (4 << 16)
            | (2 << 19)
            | (0 << 22)
            | (0 << 25)
            | (6 << 28);
        let mut tic_raw = [0u8; 32];
        for (index, word) in [format_word, 0x2000, 0, 0, 3 | (6 << 23), 0, 0, 0]
            .into_iter()
            .enumerate()
        {
            tic_raw[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut qmd = [0u32; 0x40];
        qmd[0x08] = 0xb6300;
        let mut mappings = GpuMappings::new();
        mappings.add(0x1000, 0x1000, 0x9000, 1);
        mappings.add(0x2000, 0x1000, 0xa000, 2);
        let data = std::cell::RefCell::new([0u8; 16]);
        data.borrow_mut()[8..12].copy_from_slice(&0xffff_fffeu32.to_le_bytes());
        let read = |cpu: u64, out: &mut [u8]| {
            if cpu == 0x9000 && out.len() == tic_raw.len() {
                out.copy_from_slice(&tic_raw);
                return true;
            }
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + out.len() > data.borrow().len() {
                return false;
            }
            out.copy_from_slice(&data.borrow()[offset..offset + out.len()]);
            true
        };
        let write = |cpu: u64, bytes: &[u8]| {
            let Some(offset) = cpu.checked_sub(0xa000).map(|value| value as usize) else {
                return false;
            };
            if offset + bytes.len() > data.borrow().len() {
                return false;
            }
            data.borrow_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            true
        };
        let mut exec = ComputeExec {
            qmd: &qmd,
            code: &[],
            decoded_code: Vec::new(),
            cbuf_data: empty_compute_cbufs(),
            mappings: &mappings,
            mem_read: &read,
            mem_write: &write,
            renderer: None,
            texture: ComputeTextureState {
                tic_pool_gpu_va: 0x1000,
                tic_limit: 0,
                ..ComputeTextureState::default()
            },
            tex_cache: HashMap::new(),
            tsc_cache: HashMap::new(),
            sust_tic_cache: HashMap::new(),
            write_page_cache: HashMap::new(),
            invalidated_pages: HashSet::new(),
            tex_trace: false,
            pps_5d_trace: None,
            tex_logs: 0,
            sust_logs: 0,
            writes: 0,
            unsupported: None,
            map_cache: std::cell::Cell::new(None),
        };

        let mut regs = [0u32; 256];
        regs[2] = 5;
        regs[5] = 0;
        regs[7] = 2;
        assert!(exec.suatom(&mut regs, 0xea70_0282_0022_0701));
        assert_eq!(regs[1], 0xffff_fffe);
        assert_eq!(
            u32::from_le_bytes(data.borrow()[8..12].try_into().unwrap()),
            3
        );

        regs[2] = 4;
        regs[5] = 0;
        assert!(exec.suatom(&mut regs, 0xea70_0282_0022_0705));
        assert_eq!(regs[5], 3);
        assert_eq!(
            u32::from_le_bytes(data.borrow()[8..12].try_into().unwrap()),
            7
        );
        assert_eq!(exec.writes, 2);
    }

    #[test]
    fn pps_hadd2_imm_merges_exact_one_into_low_half() {
        for (raw, inverted) in [
            (0x7a05_083c_0f00_ff11, false),
            (0x7a05_083c_0f08_ff11, true),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::HADD2_imm
            );
            assert_eq!((reg_dest(raw), reg_a(raw)), (17, RZ));
            assert_eq!((bits(raw, 16, 18), bit(raw, 19)), (0, inverted));
            assert_eq!((bits(raw, 20, 28), bit(raw, 29)), (0xf0, false));
            assert_eq!((bits(raw, 30, 38), bit(raw, 56)), (0xf0, false));
            assert_eq!(
                (bit(raw, 39), bit(raw, 43), bit(raw, 44)),
                (false, true, false)
            );
            assert_eq!(
                (bits(raw, 47, 48), bits(raw, 49, 50), bit(raw, 52)),
                (2, 2, false)
            );
            let imm = ((bits(raw, 20, 28) as u32) << 6)
                | ((bit(raw, 29) as u32) << 15)
                | ((bits(raw, 30, 38) as u32) << 22)
                | ((bit(raw, 56) as u32) << 31);
            assert_eq!(imm, 0x3c00_3c00);

            let mut regs = [0u32; 256];
            regs[17] = 0xabcd_7e00;
            assert!(pps_hadd2_imm(&mut regs, raw));
            assert_eq!(regs[17], 0xabcd_3c00);
        }

        let mut regs = [0u32; 256];
        assert!(!pps_hadd2_imm(&mut regs, 0x7a05_083c_0f00_ff11 ^ (1 << 49)));
    }

    #[test]
    fn pps_reduction_sust_is_bindless_full_rgba_2d() {
        let raw = 0xeb20_0306_00f7_0400;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SUST
        );
        assert_eq!((reg_dest(raw), reg_a(raw)), (0, 4));
        assert_eq!(bits(raw, 20, 23), 0xf);
        assert_eq!(bits(raw, 24, 25), 0);
        assert_eq!(bits(raw, 33, 35), 3);
        assert_eq!(bits(raw, 39, 46), 6);
        assert_eq!(bits(raw, 49, 50), 0);
        assert!(!bit(raw, 51));
        assert!(!bit(raw, 52));
    }

    #[test]
    fn pps_depth_reduction_sust_is_bindless_full_rgba_2d() {
        let raw = 0xeb20_0806_00f0_0600;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SUST
        );
        assert_eq!((reg_dest(raw), reg_a(raw)), (0, 6));
        assert_eq!(bits(raw, 20, 23), 0xf);
        assert_eq!(bits(raw, 24, 25), 0);
        assert_eq!(bits(raw, 33, 35), 3);
        assert_eq!(bits(raw, 39, 46), 16);
        assert_eq!(bits(raw, 49, 50), 0);
        assert!(!bit(raw, 51));
        assert!(!bit(raw, 52));
    }

    #[test]
    fn pps_depth_reduction_summary_sust_is_bindless_full_rgba_2d() {
        let raw = 0xeb20_0106_00f7_0408;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SUST
        );
        assert_eq!((reg_dest(raw), reg_a(raw)), (8, 4));
        assert_eq!(bits(raw, 20, 23), 0xf);
        assert_eq!(bits(raw, 24, 25), 0);
        assert_eq!(bits(raw, 33, 35), 3);
        assert_eq!(bits(raw, 39, 46), 2);
        assert_eq!(bits(raw, 49, 50), 0);
        assert!(!bit(raw, 51));
        assert!(!bit(raw, 52));
    }

    #[test]
    fn pps_composite_sust_is_bindless_full_rgba_2d() {
        let raw = 0xeb20_0006_00f7_0a04;
        assert_eq!(
            nexium_shader::decode_one(raw).unwrap().opcode,
            nexium_shader::Opcode::SUST
        );
        assert_eq!((reg_dest(raw), reg_a(raw)), (4, 10));
        assert_eq!(bits(raw, 20, 23), 0xf);
        assert_eq!(bits(raw, 24, 25), 0);
        assert_eq!(bits(raw, 33, 35), 3);
        assert_eq!(bits(raw, 39, 46), 0);
        assert_eq!(bits(raw, 49, 50), 0);
        assert!(!bit(raw, 51));
        assert!(!bit(raw, 52));
    }

    #[test]
    fn pps_reduction_ssy_targets_loop_reconvergence() {
        let outer = 0xe290_0000_3100_0000;
        let inner = 0xe290_0000_2b80_0000;
        assert_eq!(
            nexium_shader::decode_one(outer).unwrap().opcode,
            nexium_shader::Opcode::SSY
        );
        assert_eq!(bra_target(0x88, outer), 0x3a0);
        assert_eq!(bra_target(0xb8, inner), 0x378);
    }

    #[test]
    fn f32_to_f16_round_trips_every_non_nan_half() {
        for half in 0u16..=u16::MAX {
            let is_nan = half & 0x7c00 == 0x7c00 && half & 0x03ff != 0;
            let converted = f32_to_f16_bits(f16_to_f32_bits(half));
            if is_nan {
                assert_eq!(converted & 0x7c00, 0x7c00);
                assert_ne!(converted & 0x03ff, 0);
            } else {
                assert_eq!(converted, half);
            }
        }
        assert_eq!(f32_to_f16_bits(2f32.powi(-25).to_bits()), 0);
        assert_eq!(f32_to_f16_bits(2f32.powi(-24).to_bits()), 1);
        assert_eq!(f32_to_f16_bits(65504.0f32.to_bits()), 0x7bff);
    }

    #[test]
    fn block_linear_pixel_offset_matches_unswizzle() {
        let width = 128;
        let height = 64;
        let bytes_per_pixel = 8;
        let block_height_log2 = 3;
        let guest_size = nexium_gpu::texture::TicFormat::R16G16B16A16.block_linear_size(
            width,
            height,
            block_height_log2,
        );
        let mut guest = vec![0u8; guest_size];
        let marker = [1, 2, 3, 4, 5, 6, 7, 8];
        let offset =
            block_linear_pixel_offset(width, height, bytes_per_pixel, block_height_log2, 37, 29)
                .unwrap();
        guest[offset..offset + marker.len()].copy_from_slice(&marker);
        let linear = nexium_gpu::texture::unswizzle_block_linear(
            &guest,
            width,
            height,
            bytes_per_pixel,
            block_height_log2,
        );
        let linear_offset = (29 * width as usize + 37) * bytes_per_pixel;
        assert_eq!(
            &linear[linear_offset..linear_offset + marker.len()],
            &marker
        );
    }

    #[test]
    fn pps_imnmx_clamps_launch6_coordinates_to_unsigned_bounds() {
        for (raw, dest, a, b) in [
            (0x5c20_0380_0007_0606, 6, 6, 0),
            (0x5c20_0380_0087_0707, 7, 7, 8),
        ] {
            assert_eq!(
                nexium_shader::decode_one(raw).unwrap().opcode,
                nexium_shader::Opcode::IMNMX_reg
            );
            assert_eq!((reg_dest(raw), reg_a(raw), reg_b(raw)), (dest, a, b));
            assert_eq!(bits(raw, 39, 41), PT as u64);
            assert!(!bit(raw, 42));
            assert_eq!(bits(raw, 43, 44), 0);
            assert!(!bit(raw, 47));
            assert!(!bit(raw, 48));

            let mut preds = [false; 8];
            preds[PT as usize] = true;
            assert_eq!(imnmx_value(1200, 1067, &preds, raw), Some(1067));
            assert_eq!(imnmx_value(800, 1067, &preds, raw), Some(800));
        }
    }
}

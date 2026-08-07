use super::super::GpuMappings;

pub const KEPLER_COMPUTE_CLASS: u32 = 0xB1C0;

const NUM_REGS: usize = 0xCF8;
const M_LINE_LENGTH_IN: u32 = 0x60;
const M_LINE_COUNT: u32 = 0x61;
const M_OFFSET_OUT_UPPER: u32 = 0x62;
const M_OFFSET_OUT_LOWER: u32 = 0x63;
const M_PITCH_OUT: u32 = 0x64;
const M_DST_BLOCK_SIZE: u32 = 0x65;
const M_DST_WIDTH: u32 = 0x66;
const M_DST_HEIGHT: u32 = 0x67;
const M_DST_ORIGIN_BYTES_X: u32 = 0x6A;
const M_DST_ORIGIN_SAMPLES_Y: u32 = 0x6B;
const M_EXEC_UPLOAD: u32 = 0x6C;
const M_LOAD_INLINE_DATA: u32 = 0x6D;
const M_LAUNCH_DESC_LOC: u32 = 0xAD;
const M_LAUNCH: u32 = 0xAF;
const M_TSC_ADDRESS_HIGH: u32 = 0x557;
const M_TSC_ADDRESS_LOW: u32 = 0x558;
const M_TSC_LIMIT: u32 = 0x559;
const M_TIC_ADDRESS_HIGH: u32 = 0x55D;
const M_TIC_ADDRESS_LOW: u32 = 0x55E;
const M_TIC_LIMIT: u32 = 0x55F;
const M_CODE_LOC_UPPER: u32 = 0x582;
const M_CODE_LOC_LOWER: u32 = 0x583;
const M_TEX_CB_INDEX: u32 = 0x982;
const LAUNCH_WORDS: usize = 0x40;

#[derive(Clone, Copy, Default)]
pub(super) struct ComputeTextureState {
    pub tic_pool_gpu_va: u64,
    pub tic_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_limit: u32,
    pub tex_cb_index: u32,
}

pub struct KeplerCompute {
    regs: Vec<u32>,
    upload: ComputeUpload,
    launch_description: [u32; LAUNCH_WORDS],
    pub launch_count: u64,
    target_dumped: bool,
}

impl KeplerCompute {
    pub fn new() -> Self {
        Self {
            regs: vec![0; NUM_REGS],
            upload: ComputeUpload::default(),
            launch_description: [0; LAUNCH_WORDS],
            launch_count: 0,
            target_dumped: false,
        }
    }

    pub(crate) fn method_requires_hard_boundary(&self, method: u32, is_last_call: bool) -> bool {
        Self::method_is_launch(method) || self.method_writes_guest_memory(method, is_last_call)
    }

    pub(crate) fn method_is_launch(method: u32) -> bool {
        method == M_LAUNCH
    }

    pub(crate) fn method_writes_guest_memory(&self, method: u32, is_last_call: bool) -> bool {
        method == M_LOAD_INLINE_DATA && is_last_call && self.upload.copy_size > 0
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        is_last_call: bool,
        renderer: Option<&std::sync::Arc<nexium_gpu::Renderer>>,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        content_key: &dyn Fn(u64, usize) -> Option<u64>,
    ) -> super::KeplerMemoryWriteOutcome {
        let index = method as usize;
        if index >= self.regs.len() {
            log::trace!("KeplerCompute: invalid method {:#x} arg={:#x}", method, arg);
            return super::KeplerMemoryWriteOutcome::NoWrite;
        }

        self.regs[index] = arg;

        match method {
            M_EXEC_UPLOAD => {
                self.upload.exec(arg, &self.regs);
                super::KeplerMemoryWriteOutcome::NoWrite
            }
            M_LOAD_INLINE_DATA => {
                self.upload
                    .data(arg, is_last_call, mappings, mem_read, mem_write, &self.regs)
            }
            M_LAUNCH => {
                self.launch(renderer, mappings, mem_read, mem_write, content_key);
                super::KeplerMemoryWriteOutcome::NoWrite
            }
            _ => super::KeplerMemoryWriteOutcome::NoWrite,
        }
    }

    fn launch(
        &mut self,
        renderer: Option<&std::sync::Arc<nexium_gpu::Renderer>>,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        content_key: &dyn Fn(u64, usize) -> Option<u64>,
    ) {
        let kp = crate::gpu::pusher::kickprof::start();
        let launch_gpu = (self.reg(M_LAUNCH_DESC_LOC) as u64) << 8;
        let launch_cpu = mappings.cpu_address_for(launch_gpu).unwrap_or(launch_gpu);
        let mut bytes = [0u8; LAUNCH_WORDS * 4];

        if let (Some(renderer), Some((cpu_addr, available))) =
            (renderer, mappings.cpu_range_for(launch_gpu))
        {
            if super::maxwell_compute::has_pending_writebacks()
                && available >= bytes.len() as u64
                && super::maxwell_compute::pending_writeback_overlaps(
                    launch_gpu,
                    cpu_addr,
                    bytes.len(),
                )
            {
                super::maxwell_compute::resolve_pending_writebacks(renderer, mappings, mem_write);
            }
        }

        if mem_read(launch_cpu, &mut bytes) {
            for i in 0..LAUNCH_WORDS {
                let off = i * 4;
                self.launch_description[i] = u32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
            }
        } else {
            log::trace!(
                "KeplerCompute::launch: descriptor gpu={:#x} cpu={:#x} unreadable",
                launch_gpu,
                launch_cpu
            );
        }

        let code_base =
            ((self.reg(M_CODE_LOC_UPPER) as u64) << 32) | self.reg(M_CODE_LOC_LOWER) as u64;
        let texture = ComputeTextureState {
            tic_pool_gpu_va: ((self.reg(M_TIC_ADDRESS_HIGH) as u64) << 32)
                | self.reg(M_TIC_ADDRESS_LOW) as u64,
            tic_limit: self.reg(M_TIC_LIMIT),
            tsc_pool_gpu_va: ((self.reg(M_TSC_ADDRESS_HIGH) as u64) << 32)
                | self.reg(M_TSC_ADDRESS_LOW) as u64,
            tsc_limit: self.reg(M_TSC_LIMIT),
            tex_cb_index: self.reg(M_TEX_CB_INDEX),
        };
        let profile_started = std::env::var_os("NEXIUM_NVDRV_PROFILE")
            .is_some()
            .then(std::time::Instant::now);
        let mut backend = "unsupported";
        let mut handled = false;
        let mut executed = false;

        if renderer.is_some() {
            match super::maxwell_compute::try_execute(
                &self.launch_description,
                code_base,
                texture,
                renderer,
                mappings,
                mem_read,
                mem_write,
                content_key,
            ) {
                super::maxwell_compute::MaxwellComputeOutcome::Executed => {
                    backend = "vulkan-recompiler";
                    handled = true;
                    executed = true;
                    if compute_recompiler_trace_enabled() {
                        use std::sync::{Mutex, OnceLock};
                        static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> =
                            OnceLock::new();
                        let program = self.launch_description[0x08];
                        if SEEN
                            .get_or_init(|| Mutex::new(Default::default()))
                            .lock()
                            .unwrap()
                            .insert(program)
                        {
                            log::warn!(
                                "[compute-recompiler] program={:#x} backend=vulkan local={:?} groups={:?}",
                                program,
                                [
                                    self.launch_description[0x12] >> 16,
                                    self.launch_description[0x13] & 0xffff,
                                    self.launch_description[0x13] >> 16,
                                ],
                                [
                                    self.launch_description[0x0c] & 0x7fff_ffff,
                                    self.launch_description[0x0d] & 0xffff,
                                    self.launch_description[0x0d] >> 16,
                                ],
                            );
                        }
                    }
                }
                super::maxwell_compute::MaxwellComputeOutcome::Unsupported(reason) => {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    backend = "unsupported";
                    static RECOMPILER_SKIPS: AtomicU64 = AtomicU64::new(0);
                    let skipped = RECOMPILER_SKIPS.fetch_add(1, Ordering::Relaxed);
                    if skipped < 24 {
                        log::warn!(
                            "[compute-recompiler-skip #{}] program={:#x} reason={}",
                            skipped + 1,
                            self.launch_description[0x08],
                            reason
                        );
                    }
                }
                super::maxwell_compute::MaxwellComputeOutcome::SubmittedFailure(reason) => {
                    backend = "vulkan-recompiler-error";
                    handled = true;
                    log::error!(
                        "[compute-recompiler-error] program={:#x} reason={}",
                        self.launch_description[0x08],
                        reason
                    );
                }
            }
        }

        if let Some(profile_started) = profile_started {
            let elapsed = profile_started.elapsed();
            if elapsed >= std::time::Duration::from_millis(1) {
                let invocations = launch_invocations(&self.launch_description);
                log::warn!(
                    "[compute-profile] launch={} program={:#x} invocations={} backend={} executed={} elapsed_ms={:.3}",
                    self.launch_count,
                    self.launch_description[0x08],
                    invocations,
                    backend,
                    executed,
                    elapsed.as_secs_f64() * 1000.0,
                );
            }
        }

        let program_start = self.launch_description[0x08];
        if !handled {
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
            let mut seen = SEEN
                .get_or_init(|| Mutex::new(Default::default()))
                .lock()
                .unwrap();
            if seen.insert(program_start) {
                let invocations = launch_invocations(&self.launch_description);
                log::warn!(
                    "[compute-unsupported] program={:#x} invocations={}",
                    program_start,
                    invocations
                );
            }
        }
        let dump_invocations = compute_dump_invocations();
        let target_dump = !self.target_dumped
            && compute_dump_program().is_some_and(|program| program == program_start)
            && dump_invocations
                .is_none_or(|expected| expected == launch_invocations(&self.launch_description));
        let sampled_launch_diagnostics = self.launch_count < 16
            && dump_invocations.is_none()
            && (compute_recompiler_trace_enabled() || compute_launch_dump_enabled());
        if sampled_launch_diagnostics || target_dump {
            log::warn!(
                "KeplerCompute::launch {} gpu={:#x} code={:#x} program_start={:#x} grid=({}, {}, {}) block=({}, {}, {}) tic={:#x}/{} tsc={:#x}/{} tex_cb={}",
                backend,
                launch_gpu,
                code_base,
                self.launch_description[0x8],
                self.launch_description[0xC] & 0x7FFF_FFFF,
                self.launch_description[0xD] & 0xFFFF,
                self.launch_description[0xD] >> 16,
                self.launch_description[0x12] >> 16,
                self.launch_description[0x13] & 0xFFFF,
                self.launch_description[0x13] >> 16,
                texture.tic_pool_gpu_va,
                texture.tic_limit,
                texture.tsc_pool_gpu_va,
                texture.tsc_limit,
                texture.tex_cb_index
            );
            self.dump_launch(launch_gpu, code_base, mappings, mem_read);
            self.target_dumped |= target_dump;
        }
        self.launch_count = self.launch_count.wrapping_add(1);
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KC_LAUNCH, kp);
    }

    fn dump_launch(
        &self,
        launch_gpu: u64,
        code_base: u64,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) {
        let Some(dir) = std::env::var_os("NEXIUM_COMPUTE_LAUNCH_DUMP") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let idx = self.launch_count;

        let mut desc = Vec::with_capacity(LAUNCH_WORDS * 4);
        for w in &self.launch_description {
            desc.extend_from_slice(&w.to_le_bytes());
        }
        let _ = std::fs::write(dir.join(format!("compute_{idx:04}_desc.bin")), desc);

        let code_gpu = code_base + self.launch_description[0x8] as u64;
        let code_dump_len = std::env::var("NEXIUM_COMPUTE_CODE_DUMP_SIZE")
            .ok()
            .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0x10000);
        if let Some(code) = self.dump_range(
            &dir.join(format!("compute_{idx:04}_code.bin")),
            code_gpu,
            code_dump_len,
            mappings,
            mem_read,
        ) {
            let text = format_compute_sass(&code);
            let _ = std::fs::write(dir.join(format!("compute_{idx:04}_code.sass")), text);
        }

        let mask = self.launch_description[0x14] & 0xff;
        for slot in 0..8 {
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let base = 0x1d + slot * 2;
            let lo = self.launch_description[base];
            let hi_size = self.launch_description[base + 1];
            let gpu = (((hi_size & 0xff) as u64) << 32) | lo as u64;
            let size = ((hi_size >> 15) & 0x1ffff).max(1).min(0x4000);
            let _ = self.dump_range(
                &dir.join(format!("compute_{idx:04}_cb{slot}.bin")),
                gpu,
                size as usize,
                mappings,
                mem_read,
            );
        }

        log::warn!(
            "KeplerCompute::dump idx={} launch={:#x} code_base={:#x} code={:#x} cb_mask={:#x}",
            idx,
            launch_gpu,
            code_base,
            code_gpu,
            mask
        );
    }

    fn dump_range(
        &self,
        path: &std::path::Path,
        gpu: u64,
        len: usize,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
    ) -> Option<Vec<u8>> {
        let cpu = mappings
            .cpu_address_for(gpu)
            .or_else(|| mappings.cpu_address_for_any32(gpu).map(|(_, cpu, _)| cpu));
        let Some(cpu) = cpu else {
            return None;
        };
        let mut bytes = vec![0u8; len];
        if mem_read(cpu, &mut bytes) {
            let _ = std::fs::write(path, &bytes);
            Some(bytes)
        } else {
            None
        }
    }

    fn reg(&self, method: u32) -> u32 {
        self.regs.get(method as usize).copied().unwrap_or(0)
    }
}

fn compute_dump_program() -> Option<u32> {
    let value = std::env::var("NEXIUM_COMPUTE_DUMP_PROGRAM").ok()?;
    let value = value.trim();
    u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16).ok()
}

fn compute_recompiler_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_RECOMPILER_TRACE").is_some())
}

fn compute_launch_dump_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_COMPUTE_LAUNCH_DUMP").is_some())
}

fn compute_dump_invocations() -> Option<u64> {
    let value = std::env::var("NEXIUM_COMPUTE_DUMP_INVOCATIONS").ok()?;
    parse_u64(&value)
}

fn parse_u64(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

fn launch_invocations(desc: &[u32; LAUNCH_WORDS]) -> u64 {
    (desc[0x0c] & 0x7fff_ffff) as u64
        * (desc[0x0d] & 0xffff).max(1) as u64
        * (desc[0x0d] >> 16).max(1) as u64
        * (desc[0x12] >> 16).max(1) as u64
        * (desc[0x13] & 0xffff).max(1) as u64
        * (desc[0x13] >> 16).max(1) as u64
}

#[derive(Default)]
struct ComputeUpload {
    write_offset: usize,
    copy_size: usize,
    inline_buf: Vec<u8>,
    is_linear: bool,
}

impl ComputeUpload {
    fn exec(&mut self, flags: u32, regs: &[u32]) {
        self.write_offset = 0;
        self.copy_size = reg(regs, M_LINE_LENGTH_IN) as usize * reg(regs, M_LINE_COUNT) as usize;
        self.inline_buf.clear();
        self.inline_buf.resize(self.copy_size, 0);
        self.is_linear = (flags & 1) != 0;
    }

    fn data(
        &mut self,
        data: u32,
        is_last_call: bool,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        regs: &[u32],
    ) -> super::KeplerMemoryWriteOutcome {
        let bytes = data.to_le_bytes();
        let off = self.write_offset;
        let mut n = 0;
        if off < self.inline_buf.len() {
            n = (self.inline_buf.len() - off).min(4);
            self.inline_buf[off..off + n].copy_from_slice(&bytes[..n]);
        }
        self.write_offset += n;
        if is_last_call && self.copy_size > 0 {
            let outcome = self.flush(mappings, mem_read, mem_write, regs);
            self.copy_size = 0;
            return outcome;
        }
        super::KeplerMemoryWriteOutcome::NoWrite
    }

    fn flush(
        &mut self,
        mappings: &GpuMappings,
        _mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        regs: &[u32],
    ) -> super::KeplerMemoryWriteOutcome {
        let kp = crate::gpu::pusher::kickprof::start();
        let dst_gpu =
            ((reg(regs, M_OFFSET_OUT_UPPER) as u64) << 32) | reg(regs, M_OFFSET_OUT_LOWER) as u64;
        let line_length = reg(regs, M_LINE_LENGTH_IN) as usize;
        let line_count = reg(regs, M_LINE_COUNT).max(1) as usize;
        if dst_gpu == 0 || line_length == 0 {
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KCU_FLUSH, kp);
            return super::KeplerMemoryWriteOutcome::NoWrite;
        }

        if self.is_linear {
            let pitch = reg(regs, M_PITCH_OUT).max(line_length as u32) as u64;
            for line in 0..line_count {
                let src_off = line * line_length;
                if src_off >= self.inline_buf.len() {
                    break;
                }
                let src_end = (src_off + line_length).min(self.inline_buf.len());
                let line_gpu = dst_gpu + line as u64 * pitch;
                if let Some(line_cpu) = mappings.cpu_address_for(line_gpu) {
                    mem_write(line_cpu, &self.inline_buf[src_off..src_end]);
                }
            }
            let span_bytes = pitch.saturating_mul(line_count as u64);
            nexium_gpu::tex_invalidate::bump_region(dst_gpu, span_bytes);
            trace_upload(
                dst_gpu,
                line_length,
                line_count,
                pitch as usize,
                true,
                self.inline_buf.len(),
            );
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KCU_FLUSH, kp);
            return super::KeplerMemoryWriteOutcome::Exact(vec![(dst_gpu, span_bytes as usize)]);
        }

        let Some((dst_cpu, dst_limit)) = mappings.cpu_range_for(dst_gpu) else {
            log::trace!(
                "KeplerCompute::upload: dst gpu_va {:#x} not mapped",
                dst_gpu
            );
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KCU_FLUSH, kp);
            return super::KeplerMemoryWriteOutcome::Unknown;
        };

        let block_height_log2 = ((reg(regs, M_DST_BLOCK_SIZE) >> 4) & 0xF) as u32;
        let dst_width_bytes = if reg(regs, M_DST_WIDTH) != 0 {
            reg(regs, M_DST_WIDTH) as usize
        } else {
            line_length
        };
        let dst_height = if reg(regs, M_DST_HEIGHT) != 0 {
            reg(regs, M_DST_HEIGHT) as usize
        } else {
            line_count
        };
        let tiled = super::maxwell_dma::swizzle_block_linear(
            &self.inline_buf,
            line_length,
            line_count,
            line_length,
            dst_width_bytes,
            dst_height,
            block_height_log2,
            reg(regs, M_DST_ORIGIN_BYTES_X) as usize,
            reg(regs, M_DST_ORIGIN_SAMPLES_Y) as usize,
        );
        let n = tiled.len().min(dst_limit as usize);
        let written = mem_write(dst_cpu, &tiled[..n]);
        nexium_gpu::tex_invalidate::bump_region(dst_gpu, n as u64);
        trace_upload(dst_gpu, line_length, line_count, dst_width_bytes, false, n);
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KCU_FLUSH, kp);
        if written {
            super::KeplerMemoryWriteOutcome::Exact(vec![(dst_gpu, n)])
        } else {
            super::KeplerMemoryWriteOutcome::Unknown
        }
    }
}

fn format_compute_sass(bytes: &[u8]) -> String {
    let mut lines = Vec::new();
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        let offset = i * 8;
        let raw = u64::from_le_bytes(chunk.try_into().unwrap());
        if offset % 0x20 == 0 {
            lines.push(format!("  +{offset:04x}  {raw:016x}  ; sched"));
            continue;
        }
        let pred_idx = ((raw >> 16) & 7) as u8;
        let pred_neg = ((raw >> 19) & 1) != 0;
        let pred = if pred_idx == 7 {
            String::new()
        } else if pred_neg {
            format!("@!P{} ", pred_idx)
        } else {
            format!("@P{} ", pred_idx)
        };
        match nexium_shader::decode_one(raw) {
            Some(d) => {
                let mnemonic = d
                    .display
                    .split_once(' ')
                    .map(|(m, _)| m)
                    .unwrap_or(d.display);
                let operands = nexium_shader::pretty_operands(d.opcode, d.raw).unwrap_or_default();
                if operands.is_empty() {
                    lines.push(format!("  +{offset:04x}  {raw:016x}  {pred}{mnemonic}"));
                } else {
                    lines.push(format!(
                        "  +{offset:04x}  {raw:016x}  {pred}{mnemonic:<8} {operands}"
                    ));
                }
            }
            None => lines.push(format!("  +{offset:04x}  {raw:016x}  ?")),
        }
    }
    lines.join("\n")
}

fn reg(regs: &[u32], method: u32) -> u32 {
    regs.get(method as usize).copied().unwrap_or(0)
}

fn trace_upload(
    dst_gpu: u64,
    line_length: usize,
    line_count: usize,
    dst_stride: usize,
    linear: bool,
    bytes: usize,
) {
    if std::env::var_os("NEXIUM_COMPUTE_UPLOAD_TRACE").is_none() {
        return;
    }
    log::warn!(
        "KeplerCompute::upload dst={:#x} line_len={} line_count={} stride={} linear={} bytes={}",
        dst_gpu,
        line_length,
        line_count,
        dst_stride,
        linear,
        bytes
    );
}

#[cfg(test)]
mod tests {
    use super::super::super::GpuMappings;
    use super::super::KeplerMemoryWriteOutcome;
    use super::{
        launch_invocations, parse_u64, KeplerCompute, LAUNCH_WORDS, M_EXEC_UPLOAD, M_LAUNCH,
        M_LAUNCH_DESC_LOC, M_LINE_COUNT, M_LINE_LENGTH_IN, M_LOAD_INLINE_DATA, M_OFFSET_OUT_LOWER,
        M_PITCH_OUT,
    };

    #[test]
    fn terminal_inline_upload_reports_exact_written_span() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x1000, 0x1000, 1);
        let mut compute = KeplerCompute::new();
        let read = |_: u64, _: &mut [u8]| true;
        let write = |_: u64, _: &[u8]| true;
        let dispatch = |compute: &mut KeplerCompute, method: u32, arg: u32, is_last: bool| {
            compute.dispatch_method(
                method,
                arg,
                is_last,
                None,
                &mappings,
                &read,
                &write,
                &|_, _| None,
            )
        };

        dispatch(&mut compute, M_LINE_LENGTH_IN, 8, false);
        dispatch(&mut compute, M_LINE_COUNT, 1, false);
        dispatch(&mut compute, M_OFFSET_OUT_LOWER, 0x4000, false);
        dispatch(&mut compute, M_PITCH_OUT, 8, false);
        dispatch(&mut compute, M_EXEC_UPLOAD, 1, false);
        assert_eq!(
            dispatch(&mut compute, M_LOAD_INLINE_DATA, 0x1122_3344, false),
            KeplerMemoryWriteOutcome::NoWrite
        );
        assert_eq!(
            dispatch(&mut compute, M_LOAD_INLINE_DATA, 0x5566_7788, true),
            KeplerMemoryWriteOutcome::Exact(vec![(0x4000, 8)])
        );
    }

    #[test]
    fn launch_and_terminal_inline_data_require_hard_boundary() {
        let mut compute = KeplerCompute::new();
        assert!(KeplerCompute::method_is_launch(M_LAUNCH));
        assert!(!KeplerCompute::method_is_launch(M_LAUNCH_DESC_LOC));
        assert!(!KeplerCompute::method_is_launch(M_LOAD_INLINE_DATA));
        assert!(compute.method_requires_hard_boundary(M_LAUNCH, false));
        assert!(!compute.method_writes_guest_memory(M_LAUNCH, false));
        assert!(!compute.method_requires_hard_boundary(M_LAUNCH_DESC_LOC, true));
        assert!(!compute.method_requires_hard_boundary(M_EXEC_UPLOAD, true));
        assert!(!compute.method_requires_hard_boundary(M_LOAD_INLINE_DATA, true));
        assert!(!compute.method_writes_guest_memory(M_LOAD_INLINE_DATA, true));

        compute.upload.copy_size = 4;
        assert!(!compute.method_requires_hard_boundary(M_LOAD_INLINE_DATA, false));
        assert!(compute.method_requires_hard_boundary(M_LOAD_INLINE_DATA, true));
        assert!(compute.method_writes_guest_memory(M_LOAD_INLINE_DATA, true));
        assert!(!compute.method_writes_guest_memory(M_LAUNCH, true));
    }

    #[test]
    fn invocation_count_multiplies_grid_and_block_dimensions() {
        let mut desc = [0; LAUNCH_WORDS];
        desc[0x0c] = 27;
        desc[0x0d] = 2 | (3 << 16);
        desc[0x12] = 4 << 16;
        desc[0x13] = 2 | (2 << 16);
        assert_eq!(launch_invocations(&desc), 27 * 2 * 3 * 4 * 2 * 2);
    }

    #[test]
    fn pps_2fb200_dispatch_shape_is_100_by_57_workgroups_of_64_threads() {
        let mut desc = [0; LAUNCH_WORDS];
        desc[0x0c] = 100;
        desc[0x0d] = 57 | (1 << 16);
        desc[0x12] = 64 << 16;
        desc[0x13] = 1 | (1 << 16);
        assert_eq!(launch_invocations(&desc), 100 * 57 * 64);
        assert_eq!(launch_invocations(&desc), 364_800);
    }

    #[test]
    fn dump_invocation_filter_accepts_decimal_and_hex() {
        assert_eq!(parse_u64("3456"), Some(3456));
        assert_eq!(parse_u64(" 0xD80 "), Some(3456));
        assert_eq!(parse_u64("not-a-count"), None);
    }
}

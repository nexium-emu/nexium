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
const M_CODE_LOC_UPPER: u32 = 0x582;
const M_CODE_LOC_LOWER: u32 = 0x583;
const LAUNCH_WORDS: usize = 0x40;

pub struct KeplerCompute {
    regs: Vec<u32>,
    upload: ComputeUpload,
    launch_description: [u32; LAUNCH_WORDS],
    pub launch_count: u64,
}

impl KeplerCompute {
    pub fn new() -> Self {
        Self {
            regs: vec![0; NUM_REGS],
            upload: ComputeUpload::default(),
            launch_description: [0; LAUNCH_WORDS],
            launch_count: 0,
        }
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        is_last_call: bool,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let index = method as usize;
        if index >= self.regs.len() {
            log::trace!("KeplerCompute: invalid method {:#x} arg={:#x}", method, arg);
            return;
        }

        self.regs[index] = arg;

        match method {
            M_EXEC_UPLOAD => self.upload.exec(arg, &self.regs),
            M_LOAD_INLINE_DATA => {
                self.upload
                    .data(arg, is_last_call, mappings, mem_read, mem_write, &self.regs)
            }
            M_LAUNCH => self.launch(mappings, mem_read),
            _ => {}
        }
    }

    fn launch(&mut self, mappings: &GpuMappings, mem_read: &dyn Fn(u64, &mut [u8]) -> bool) {
        let launch_gpu = (self.reg(M_LAUNCH_DESC_LOC) as u64) << 8;
        let launch_cpu = mappings.cpu_address_for(launch_gpu).unwrap_or(launch_gpu);
        let mut bytes = [0u8; LAUNCH_WORDS * 4];

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

        if self.launch_count < 8 {
            let code_base = ((self.reg(M_CODE_LOC_UPPER) as u64) << 32)
                | self.reg(M_CODE_LOC_LOWER) as u64;
            log::warn!(
                "KeplerCompute::launch ignored gpu={:#x} code={:#x} program_start={:#x} grid=({}, {}, {}) block=({}, {}, {})",
                launch_gpu,
                code_base,
                self.launch_description[0x8],
                self.launch_description[0xC] & 0x7FFF_FFFF,
                self.launch_description[0xD] & 0xFFFF,
                self.launch_description[0xD] >> 16,
                self.launch_description[0x12] >> 16,
                self.launch_description[0x13] & 0xFFFF,
                self.launch_description[0x13] >> 16
            );
            self.dump_launch(launch_gpu, code_base, mappings, mem_read);
        }
        self.launch_count = self.launch_count.wrapping_add(1);
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
        if let Some(code) = self.dump_range(
            &dir.join(format!("compute_{idx:04}_code.bin")),
            code_gpu,
            0x1000,
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
    ) {
        let bytes = data.to_le_bytes();
        let off = self.write_offset;
        let mut n = 0;
        if off < self.inline_buf.len() {
            n = (self.inline_buf.len() - off).min(4);
            self.inline_buf[off..off + n].copy_from_slice(&bytes[..n]);
        }
        self.write_offset += n;
        if is_last_call && self.copy_size > 0 {
            self.flush(mappings, mem_read, mem_write, regs);
            self.copy_size = 0;
        }
    }

    fn flush(
        &mut self,
        mappings: &GpuMappings,
        _mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
        regs: &[u32],
    ) {
        let dst_gpu =
            ((reg(regs, M_OFFSET_OUT_UPPER) as u64) << 32) | reg(regs, M_OFFSET_OUT_LOWER) as u64;
        let line_length = reg(regs, M_LINE_LENGTH_IN) as usize;
        let line_count = reg(regs, M_LINE_COUNT).max(1) as usize;
        if dst_gpu == 0 || line_length == 0 {
            return;
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
            nexium_gpu::tex_invalidate::bump_region(dst_gpu, pitch * line_count as u64);
            trace_upload(
                dst_gpu,
                line_length,
                line_count,
                pitch as usize,
                true,
                self.inline_buf.len(),
            );
            return;
        }

        let Some((dst_cpu, dst_limit)) = mappings.cpu_range_for(dst_gpu) else {
            log::trace!(
                "KeplerCompute::upload: dst gpu_va {:#x} not mapped",
                dst_gpu
            );
            return;
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
        mem_write(dst_cpu, &tiled[..n]);
        nexium_gpu::tex_invalidate::bump_region(dst_gpu, n as u64);
        trace_upload(dst_gpu, line_length, line_count, dst_width_bytes, false, n);
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
                let mnemonic = d.display.split_once(' ').map(|(m, _)| m).unwrap_or(d.display);
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

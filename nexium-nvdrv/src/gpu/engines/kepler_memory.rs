use super::super::GpuMappings;

pub const KEPLER_MEMORY_CLASS: u32 = 0xA140;

const M_LINE_LENGTH_IN: u32 = 0x60;
const M_LINE_COUNT: u32 = 0x61;
const M_OFFSET_OUT_UPPER: u32 = 0x62;
const M_OFFSET_OUT_LOWER: u32 = 0x63;
const M_PITCH_OUT: u32 = 0x64;
const M_DST_BLOCK_SIZE: u32 = 0x65;
const M_DST_WIDTH: u32 = 0x66;
const M_DST_HEIGHT: u32 = 0x67;
const M_DST_DEPTH: u32 = 0x68;
const M_DST_LAYER: u32 = 0x69;
const M_DST_ORIGIN_BYTES_X: u32 = 0x6A;
const M_DST_ORIGIN_SAMPLES_Y: u32 = 0x6B;
const M_LAUNCH_DMA: u32 = 0x6C;
const M_LOAD_INLINE_DATA: u32 = 0x6D;

const MAX_INLINE_BYTES: usize = 16 * 1024 * 1024;

const LAUNCH_DST_LAYOUT_BIT: u32 = 0;
const LAYOUT_BLOCK_LINEAR: u32 = 0;
const LAYOUT_PITCH: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeplerMemoryWriteOutcome {
    NoWrite,
    Exact(Vec<(u64, usize)>),
    Unknown,
}

#[derive(Default)]
pub struct KeplerMemory {
    line_length_in: u32,
    line_count: u32,
    offset_out_upper: u32,
    offset_out_lower: u32,
    pitch_out: u32,
    dst_block_size: u32,
    dst_width: u32,
    dst_height: u32,
    dst_origin_x: u32,
    dst_origin_y: u32,
    launch_flags: u32,
    inline_buf: Vec<u8>,
    write_offset: usize,
    copy_size: usize,
    pub upload_count: u64,
}

impl KeplerMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn method_requires_hard_boundary(&self, method: u32) -> bool {
        method == M_LOAD_INLINE_DATA
            && self.copy_size > 0
            && self.write_offset.saturating_add(4) >= self.copy_size
    }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> KeplerMemoryWriteOutcome {
        match method {
            M_LINE_LENGTH_IN => {
                self.line_length_in = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_LINE_COUNT => {
                self.line_count = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_OFFSET_OUT_UPPER => {
                self.offset_out_upper = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_OFFSET_OUT_LOWER => {
                self.offset_out_lower = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_PITCH_OUT => {
                self.pitch_out = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_DST_BLOCK_SIZE => {
                self.dst_block_size = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_DST_WIDTH => {
                self.dst_width = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_DST_HEIGHT => {
                self.dst_height = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_DST_DEPTH | M_DST_LAYER => KeplerMemoryWriteOutcome::NoWrite,
            M_DST_ORIGIN_BYTES_X => {
                self.dst_origin_x = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_DST_ORIGIN_SAMPLES_Y => {
                self.dst_origin_y = arg;
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_LAUNCH_DMA => {
                self.launch(arg);
                KeplerMemoryWriteOutcome::NoWrite
            }
            M_LOAD_INLINE_DATA => self.load_inline_data(arg, mappings, mem_read, mem_write),
            _ => {
                log::trace!(
                    "KeplerMemory: unhandled method {:#x} arg={:#x}",
                    method,
                    arg
                );
                KeplerMemoryWriteOutcome::NoWrite
            }
        }
    }

    fn launch(&mut self, flags: u32) {
        self.launch_flags = flags;
        let line_length = self.line_length_in as usize;
        let line_count = self.line_count.max(1) as usize;
        let requested = line_length.saturating_mul(line_count);
        let copy_size = requested.min(MAX_INLINE_BYTES);
        if copy_size != requested {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CLAMPED: AtomicU64 = AtomicU64::new(0);
            let n = CLAMPED.fetch_add(1, Ordering::Relaxed);
            if n < 32 || n % 1024 == 0 {
                log::warn!(
                    "[i2m-clamp] #{} bogus inline upload line_len={} line_count={} requested={} → clamped to {} (flags={:#x})",
                    n,
                    line_length,
                    line_count,
                    requested,
                    copy_size,
                    flags
                );
            }
        }
        self.copy_size = copy_size;
        self.write_offset = 0;
        self.inline_buf.clear();
        self.inline_buf.resize(copy_size, 0);
    }

    fn load_inline_data(
        &mut self,
        data: u32,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> KeplerMemoryWriteOutcome {
        let bytes = data.to_le_bytes();
        let off = self.write_offset;
        if off + 4 <= self.inline_buf.len() {
            self.inline_buf[off..off + 4].copy_from_slice(&bytes);
        } else if off < self.inline_buf.len() {
            let n = self.inline_buf.len() - off;
            self.inline_buf[off..].copy_from_slice(&bytes[..n]);
        }
        self.write_offset += 4;
        if self.write_offset >= self.copy_size && self.copy_size > 0 {
            let outcome = self.flush(mappings, mem_read, mem_write);
            self.copy_size = 0;
            outcome
        } else {
            KeplerMemoryWriteOutcome::NoWrite
        }
    }

    fn flush(
        &mut self,
        mappings: &GpuMappings,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> KeplerMemoryWriteOutcome {
        let kp = crate::gpu::pusher::kickprof::start();
        let dst_gpu = ((self.offset_out_upper as u64) << 32) | self.offset_out_lower as u64;
        let Some((dst_cpu, dst_limit)) = mappings.cpu_range_for(dst_gpu) else {
            use std::sync::atomic::{AtomicU64, Ordering};
            static DROPPED: AtomicU64 = AtomicU64::new(0);
            let n = DROPPED.fetch_add(1, Ordering::Relaxed);
            if n < 32 || n % 1024 == 0 {
                log::warn!(
                    "[i2m-drop] #{} dst gpu_va={:#x} bytes={}",
                    n,
                    dst_gpu,
                    self.copy_size
                );
            }
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KM_FLUSH, kp);
            return KeplerMemoryWriteOutcome::NoWrite;
        };
        let dst_limit = dst_limit as usize;

        let line_length = self.line_length_in as usize;
        let line_count = self.line_count.max(1) as usize;
        if line_length == 0 {
            crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KM_FLUSH, kp);
            return KeplerMemoryWriteOutcome::NoWrite;
        }

        let dst_layout = (self.launch_flags >> LAUNCH_DST_LAYOUT_BIT) & 1;

        if self.upload_count < 32 {
            log::info!(
                "KeplerMemory::flush[{}]: dst_gpu={:#x} cpu={:#x} layout={} line_len={} line_count={} flags={:#x}",
                self.upload_count,
                dst_gpu,
                dst_cpu,
                dst_layout,
                line_length,
                line_count,
                self.launch_flags
            );
        }

        let outcome = match dst_layout {
            LAYOUT_PITCH => {
                let pitch_out = self.pitch_out.max(line_length as u32) as usize;
                let mut writes = Vec::new();
                let mut unknown = false;
                if line_count == 1 || pitch_out == line_length {
                    let n = (line_length * line_count).min(dst_limit);
                    let wrote = mem_write(dst_cpu, &self.inline_buf[..n]);
                    if n != 0 {
                        if wrote {
                            writes.push((dst_gpu, n));
                        } else {
                            unknown = true;
                        }
                    } else if !wrote {
                        unknown = true;
                    }
                } else {
                    for y in 0..line_count {
                        let dst_row_off = y * pitch_out;
                        if dst_row_off >= dst_limit {
                            break;
                        }
                        let n = line_length.min(dst_limit - dst_row_off);
                        let src_off = y * line_length;
                        let dst_off = dst_cpu + dst_row_off as u64;
                        if mem_write(dst_off, &self.inline_buf[src_off..src_off + n]) {
                            writes.push((dst_gpu.saturating_add(dst_row_off as u64), n));
                        } else {
                            unknown = true;
                        }
                    }
                }
                if unknown {
                    KeplerMemoryWriteOutcome::Unknown
                } else if writes.is_empty() {
                    KeplerMemoryWriteOutcome::NoWrite
                } else {
                    KeplerMemoryWriteOutcome::Exact(writes)
                }
            }
            LAYOUT_BLOCK_LINEAR => {
                let block_height_log2 = ((self.dst_block_size >> 4) & 0xF) as u32;
                let dst_width_bytes = if self.dst_width != 0 {
                    self.dst_width as usize
                } else {
                    line_length
                };
                let dst_height = if self.dst_height != 0 {
                    self.dst_height as usize
                } else {
                    line_count
                };
                let tiled_size = super::maxwell_dma::tiled_size_bytes(
                    dst_width_bytes,
                    dst_height,
                    block_height_log2,
                );
                if tiled_size > MAX_INLINE_BYTES {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static SKIPPED: AtomicU64 = AtomicU64::new(0);
                    let n = SKIPPED.fetch_add(1, Ordering::Relaxed);
                    if n < 32 || n % 1024 == 0 {
                        log::warn!(
                            "[i2m-skip] #{} block-linear dst too large: tiled={} dst_w={} dst_h={} bh_log2={} line_len={} line_count={} — skipping upload",
                            n,
                            tiled_size,
                            dst_width_bytes,
                            dst_height,
                            block_height_log2,
                            line_length,
                            line_count
                        );
                    }
                    crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KM_FLUSH, kp);
                    return KeplerMemoryWriteOutcome::NoWrite;
                }
                let mut tiled = vec![0u8; tiled_size];
                let n = tiled_size.min(dst_limit);
                mem_read(dst_cpu, &mut tiled[..n]);
                super::maxwell_dma::swizzle_block_linear_into(
                    &mut tiled,
                    &self.inline_buf,
                    line_length,
                    line_count,
                    line_length,
                    dst_width_bytes,
                    block_height_log2,
                    self.dst_origin_x as usize,
                    self.dst_origin_y as usize,
                );
                if tiled_size > dst_limit {
                    log::warn!(
                        "KeplerMemory::flush CLAMP: tiled={} > dst_limit={} (dst_cpu={:#x} dst_w={} dst_h={} line_len={} line_count={}) — truncating to avoid heap overrun",
                        tiled_size,
                        dst_limit,
                        dst_cpu,
                        self.dst_width,
                        self.dst_height,
                        line_length,
                        line_count,
                    );
                }
                if n == 0 {
                    if mem_write(dst_cpu, &tiled[..n]) {
                        KeplerMemoryWriteOutcome::NoWrite
                    } else {
                        KeplerMemoryWriteOutcome::Unknown
                    }
                } else if mem_write(dst_cpu, &tiled[..n]) {
                    KeplerMemoryWriteOutcome::Exact(vec![(dst_gpu, n)])
                } else {
                    KeplerMemoryWriteOutcome::Unknown
                }
            }
            _ => unreachable!(),
        };

        let bump_size = match dst_layout {
            LAYOUT_BLOCK_LINEAR => {
                let bh = ((self.dst_block_size >> 4) & 0xF) as u32;
                let dw = if self.dst_width != 0 {
                    self.dst_width as usize
                } else {
                    line_length
                };
                let dh = if self.dst_height != 0 {
                    self.dst_height as usize
                } else {
                    line_count
                };
                super::maxwell_dma::tiled_size_bytes(dw, dh, bh) as u64
            }
            _ => {
                let pitch_out = self.pitch_out.max(line_length as u32) as u64;
                pitch_out * line_count as u64
            }
        };
        nexium_gpu::tex_invalidate::bump_region(dst_gpu, bump_size);

        self.upload_count = self.upload_count.wrapping_add(1);
        crate::gpu::pusher::kickprof::add(crate::gpu::pusher::kickprof::KM_FLUSH, kp);
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuMappings;
    use std::cell::{Cell, RefCell};

    #[test]
    fn only_terminal_inline_data_requires_hard_boundary() {
        let mut memory = KeplerMemory::new();
        assert!(!memory.method_requires_hard_boundary(M_LINE_LENGTH_IN));
        assert!(!memory.method_requires_hard_boundary(M_LAUNCH_DMA));
        assert!(!memory.method_requires_hard_boundary(M_LOAD_INLINE_DATA));

        memory.copy_size = 8;
        assert!(!memory.method_requires_hard_boundary(M_LOAD_INLINE_DATA));
        memory.write_offset = 4;
        assert!(memory.method_requires_hard_boundary(M_LOAD_INLINE_DATA));
        assert!(!memory.method_requires_hard_boundary(M_LINE_LENGTH_IN));

        memory.copy_size = 0;
        assert!(!memory.method_requires_hard_boundary(M_LOAD_INLINE_DATA));
    }

    #[test]
    fn pitch_upload_reports_exact_contiguous_span() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut memory = KeplerMemory::new();
        let writes = RefCell::new(Vec::new());
        let read = |_: u64, _: &mut [u8]| true;
        let write = |cpu: u64, data: &[u8]| {
            writes.borrow_mut().push((cpu, data.to_vec()));
            true
        };

        memory.dispatch_method(M_LINE_LENGTH_IN, 8, &mappings, &read, &write);
        memory.dispatch_method(M_LINE_COUNT, 1, &mappings, &read, &write);
        memory.dispatch_method(M_OFFSET_OUT_LOWER, 0x4000, &mappings, &read, &write);
        memory.dispatch_method(M_PITCH_OUT, 8, &mappings, &read, &write);
        memory.dispatch_method(M_LAUNCH_DMA, LAYOUT_PITCH, &mappings, &read, &write);
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x1122_3344, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::NoWrite
        );
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x5566_7788, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::Exact(vec![(0x4000, 8)])
        );
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x99aa_bbcc, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::NoWrite
        );
        assert_eq!(
            *writes.borrow(),
            vec![(0x1000, vec![0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55])]
        );
    }

    #[test]
    fn pitch_upload_reports_exact_spans_per_row() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut memory = KeplerMemory::new();
        let read = |_: u64, _: &mut [u8]| true;
        let write = |_: u64, _: &[u8]| true;

        memory.dispatch_method(M_LINE_LENGTH_IN, 4, &mappings, &read, &write);
        memory.dispatch_method(M_LINE_COUNT, 3, &mappings, &read, &write);
        memory.dispatch_method(M_OFFSET_OUT_LOWER, 0x4000, &mappings, &read, &write);
        memory.dispatch_method(M_PITCH_OUT, 8, &mappings, &read, &write);
        memory.dispatch_method(M_LAUNCH_DMA, LAYOUT_PITCH, &mappings, &read, &write);
        memory.dispatch_method(M_LOAD_INLINE_DATA, 0x1122_3344, &mappings, &read, &write);
        memory.dispatch_method(M_LOAD_INLINE_DATA, 0x5566_7788, &mappings, &read, &write);
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x99aa_bbcc, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::Exact(vec![(0x4000, 4), (0x4008, 4), (0x4010, 4)])
        );
    }

    #[test]
    fn block_linear_upload_with_failed_read_reports_exact_write() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x1000, 0x1000, 1);
        let mut memory = KeplerMemory::new();
        let writes = RefCell::new(Vec::new());
        let read = |_: u64, _: &mut [u8]| false;
        let write = |cpu: u64, data: &[u8]| {
            writes.borrow_mut().push((cpu, data.len()));
            true
        };
        let tiled_size = super::super::maxwell_dma::tiled_size_bytes(4, 1, 0);

        memory.dispatch_method(M_LINE_LENGTH_IN, 4, &mappings, &read, &write);
        memory.dispatch_method(M_LINE_COUNT, 1, &mappings, &read, &write);
        memory.dispatch_method(M_OFFSET_OUT_LOWER, 0x4000, &mappings, &read, &write);
        memory.dispatch_method(M_DST_WIDTH, 4, &mappings, &read, &write);
        memory.dispatch_method(M_DST_HEIGHT, 1, &mappings, &read, &write);
        memory.dispatch_method(M_LAUNCH_DMA, LAYOUT_BLOCK_LINEAR, &mappings, &read, &write);
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x1122_3344, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::Exact(vec![(0x4000, tiled_size)])
        );
        assert_eq!(*writes.borrow(), vec![(0x1000, tiled_size)]);
    }

    #[test]
    fn unmapped_upload_reports_no_write() {
        let mappings = GpuMappings::new();
        let mut memory = KeplerMemory::new();
        let read = |_: u64, _: &mut [u8]| true;
        let write = |_: u64, _: &[u8]| panic!("unmapped upload must not write");

        memory.dispatch_method(M_LINE_LENGTH_IN, 4, &mappings, &read, &write);
        memory.dispatch_method(M_LINE_COUNT, 1, &mappings, &read, &write);
        memory.dispatch_method(M_OFFSET_OUT_LOWER, 0x4000, &mappings, &read, &write);
        memory.dispatch_method(M_LAUNCH_DMA, LAYOUT_PITCH, &mappings, &read, &write);
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x1122_3344, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::NoWrite
        );
    }

    #[test]
    fn failed_pitch_row_reports_unknown_after_preserving_all_writes() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut memory = KeplerMemory::new();
        let write_calls = Cell::new(0usize);
        let read = |_: u64, _: &mut [u8]| true;
        let write = |_: u64, _: &[u8]| {
            let call = write_calls.get() + 1;
            write_calls.set(call);
            call != 2
        };

        memory.dispatch_method(M_LINE_LENGTH_IN, 4, &mappings, &read, &write);
        memory.dispatch_method(M_LINE_COUNT, 3, &mappings, &read, &write);
        memory.dispatch_method(M_OFFSET_OUT_LOWER, 0x4000, &mappings, &read, &write);
        memory.dispatch_method(M_PITCH_OUT, 8, &mappings, &read, &write);
        memory.dispatch_method(M_LAUNCH_DMA, LAYOUT_PITCH, &mappings, &read, &write);
        memory.dispatch_method(M_LOAD_INLINE_DATA, 0x1122_3344, &mappings, &read, &write);
        memory.dispatch_method(M_LOAD_INLINE_DATA, 0x5566_7788, &mappings, &read, &write);
        assert_eq!(
            memory.dispatch_method(M_LOAD_INLINE_DATA, 0x99aa_bbcc, &mappings, &read, &write),
            KeplerMemoryWriteOutcome::Unknown
        );
        assert_eq!(write_calls.get(), 3);
    }
}

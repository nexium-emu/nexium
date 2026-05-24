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

const LAUNCH_DST_LAYOUT_BIT: u32 = 0;
const LAYOUT_BLOCK_LINEAR: u32 = 0;
const LAYOUT_PITCH: u32 = 1;

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
    pub fn new() -> Self { Self::default() }

    pub fn dispatch_method(
        &mut self,
        method: u32,
        arg: u32,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        match method {
            M_LINE_LENGTH_IN => self.line_length_in = arg,
            M_LINE_COUNT => self.line_count = arg,
            M_OFFSET_OUT_UPPER => self.offset_out_upper = arg,
            M_OFFSET_OUT_LOWER => self.offset_out_lower = arg,
            M_PITCH_OUT => self.pitch_out = arg,
            M_DST_BLOCK_SIZE => self.dst_block_size = arg,
            M_DST_WIDTH => self.dst_width = arg,
            M_DST_HEIGHT => self.dst_height = arg,
            M_DST_DEPTH | M_DST_LAYER => {}
            M_DST_ORIGIN_BYTES_X => self.dst_origin_x = arg,
            M_DST_ORIGIN_SAMPLES_Y => self.dst_origin_y = arg,
            M_LAUNCH_DMA => self.launch(arg),
            M_LOAD_INLINE_DATA => self.load_inline_data(arg, mappings, mem_write),
            _ => {
                log::trace!("KeplerMemory: unhandled method {:#x} arg={:#x}", method, arg);
            }
        }
    }

    fn launch(&mut self, flags: u32) {
        self.launch_flags = flags;
        let line_length = self.line_length_in as usize;
        let line_count = self.line_count.max(1) as usize;
        self.copy_size = line_length * line_count;
        self.write_offset = 0;
        self.inline_buf.clear();
        self.inline_buf.resize(self.copy_size, 0);
    }

    fn load_inline_data(
        &mut self,
        data: u32,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let bytes = data.to_le_bytes();
        let off = self.write_offset;
        if off + 4 <= self.inline_buf.len() {
            self.inline_buf[off..off + 4].copy_from_slice(&bytes);
        }
        self.write_offset += 4;
        if self.write_offset >= self.copy_size && self.copy_size > 0 {
            self.flush(mappings, mem_write);
            self.copy_size = 0;
        }
    }

    fn flush(
        &mut self,
        mappings: &GpuMappings,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) {
        let dst_gpu = ((self.offset_out_upper as u64) << 32) | self.offset_out_lower as u64;
        let Some(dst_cpu) = mappings.cpu_address_for(dst_gpu) else {
            log::trace!("KeplerMemory::flush: dst gpu_va {:#x} not mapped", dst_gpu);
            return;
        };

        let line_length = self.line_length_in as usize;
        let line_count = self.line_count.max(1) as usize;
        if line_length == 0 {
            return;
        }

        let dst_layout = (self.launch_flags >> LAUNCH_DST_LAYOUT_BIT) & 1;

        if self.upload_count < 32 {
            log::info!(
                "KeplerMemory::flush[{}]: dst_gpu={:#x} cpu={:#x} layout={} line_len={} line_count={} flags={:#x}",
                self.upload_count, dst_gpu, dst_cpu, dst_layout, line_length, line_count, self.launch_flags
            );
        }

        match dst_layout {
            LAYOUT_PITCH => {
                let pitch_out = self.pitch_out.max(line_length as u32) as usize;
                if line_count == 1 || pitch_out == line_length {
                    mem_write(dst_cpu, &self.inline_buf[..line_length * line_count]);
                } else {
                    for y in 0..line_count {
                        let src_off = y * line_length;
                        let dst_off = dst_cpu + (y * pitch_out) as u64;
                        mem_write(dst_off, &self.inline_buf[src_off..src_off + line_length]);
                    }
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
                let tiled = super::maxwell_dma::swizzle_block_linear(
                    &self.inline_buf,
                    line_length,
                    line_count,
                    line_length,
                    dst_width_bytes,
                    dst_height,
                    block_height_log2,
                    self.dst_origin_x as usize,
                    self.dst_origin_y as usize,
                );
                mem_write(dst_cpu, &tiled);
            }
            _ => unreachable!(),
        }

        self.upload_count = self.upload_count.wrapping_add(1);
    }
}

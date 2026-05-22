pub const MAXWELL3D_CLASS: u32 = 0xB197;

#[derive(Clone, Copy, Default, Debug)]
pub struct RenderTarget {
    pub address_lo: u32,
    pub address_hi: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub tile_mode: u32,
    pub depth: u32,
    pub layer_stride: u32,
    pub base_layer: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub depth_min: f32,
    pub depth_max: f32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ClearColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct VertexAttribute {
    pub buffer: u32,
    pub offset: u32,
    pub format: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct VertexBuffer {
    pub stride: u32,
    pub address_lo: u32,
    pub address_hi: u32,
    pub size: u32,
    pub end_lo: u32,
    pub end_hi: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ShaderProgram {
    pub address_lo: u32,
    pub address_hi: u32,
    pub gpr_count: u32,
    pub enabled: bool,
}

#[derive(Clone)]
pub struct Maxwell3DRegisters {
    pub rt: [RenderTarget; 8],
    pub viewport: Viewport,
    pub clear_color: ClearColor,
    pub clear_depth: f32,
    pub clear_stencil: u32,
    pub vertex_attribs: [VertexAttribute; 32],
    pub vertex_buffers: [VertexBuffer; 32],
    pub shader_programs: [ShaderProgram; 6],
    pub draw_vertex_count: u32,
    pub draw_first_vertex: u32,
    pub draw_topology: u32,
    pub index_buffer_lo: u32,
    pub index_buffer_hi: u32,
    pub index_buffer_end_lo: u32,
    pub index_buffer_end_hi: u32,
    pub index_format: u32,
    pub index_count: u32,
    pub depth_test_enable: bool,
    pub blend_enable: [bool; 8],
    pub draw_count: u64,
    pub clear_count: u64,

    pub tic_pool_va_lo: u32,
    pub tic_pool_va_hi: u32,

    pub program_region_va_hi: u32,
    pub program_region_va_lo: u32,
    pub tic_pool_limit: u32,

    pub draw_texture_dst_x: u32,
    pub draw_texture_dst_y: u32,
    pub draw_texture_dst_width: u32,
    pub draw_texture_dst_height: u32,
    pub draw_texture_count: u64,

    pub constbuf_selector_size: u32,
    pub constbuf_selector_addr_hi: u32,
    pub constbuf_selector_addr_lo: u32,
    pub constbuf_load_offset: u32,

    pub pending_constbuf_writes: Vec<(u64, u32)>,

    pub last_constbuf_addr: u64,
    pub last_constbuf_size: u32,

    pub cbuf_binds: [[(u64, u32); 16]; 5],
}

impl Default for Maxwell3DRegisters {
    fn default() -> Self {
        Self {
            rt: [RenderTarget::default(); 8],
            viewport: Viewport::default(),
            clear_color: ClearColor::default(),
            clear_depth: 1.0,
            clear_stencil: 0,
            vertex_attribs: [VertexAttribute::default(); 32],
            vertex_buffers: [VertexBuffer::default(); 32],
            shader_programs: [ShaderProgram::default(); 6],
            draw_vertex_count: 0,
            draw_first_vertex: 0,
            draw_topology: 0,
            index_buffer_lo: 0,
            index_buffer_hi: 0,
            index_buffer_end_lo: 0,
            index_buffer_end_hi: 0,
            index_format: 0,
            index_count: 0,
            depth_test_enable: false,
            blend_enable: [false; 8],
            draw_count: 0,
            clear_count: 0,
            tic_pool_va_lo: 0,
            tic_pool_va_hi: 0,
            program_region_va_hi: 0,
            program_region_va_lo: 0,
            tic_pool_limit: 0,
            draw_texture_dst_x: 0,
            draw_texture_dst_y: 0,
            draw_texture_dst_width: 0,
            draw_texture_dst_height: 0,
            draw_texture_count: 0,
            constbuf_selector_size: 0,
            constbuf_selector_addr_hi: 0,
            constbuf_selector_addr_lo: 0,
            constbuf_load_offset: 0,
            pending_constbuf_writes: Vec::new(),
            last_constbuf_addr: 0,
            last_constbuf_size: 0,
            cbuf_binds: [[(0, 0); 16]; 5],
        }
    }
}

#[derive(Clone, Debug)]
pub struct DrawCall {
    pub topology: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub indexed: bool,
    pub index_count: u32,

    pub rt: [RenderTarget; 8],
    pub vertex_buffers: [VertexBuffer; 32],
    pub vertex_attribs: [VertexAttribute; 32],
    pub viewport: Viewport,
    pub clear_color: ClearColor,
    pub is_clear: bool,

    pub draw_texture: Option<DrawTextureCall>,

    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,

    pub last_constbuf_addr: u64,
    pub last_constbuf_size: u32,

    pub fs_bindless_cb_addr: u64,
    pub fs_bindless_cb_size: u32,

    pub fs_shader_gpu_va: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct DrawTextureCall {
    pub dst_x: u32,
    pub dst_y: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub texture_id: u32,
    pub sampler_id: u32,
    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
}

pub struct Maxwell3D {
    pub regs: Maxwell3DRegisters,
    pub reg_file: Vec<u32>,
    pub macro_engine: super::MacroEngine,
    pub method_freq: std::collections::HashMap<u32, u64>,

    pub pending_draws: Vec<DrawCall>,

    pub macro_uploads_logged: u32,
    pub macro_invocations: u32,
    pub macro_writes_logged: u32,
}

const REG_LOAD_MME_INSTRUCTION_PTR: u32 = 0x45;
const REG_LOAD_MME_INSTRUCTION: u32 = 0x46;
const REG_LOAD_MME_START_ADDRESS_PTR: u32 = 0x47;
const REG_LOAD_MME_START_ADDRESS: u32 = 0x48;

impl Maxwell3D {
    pub fn new() -> Self {
        Self {
            regs: Maxwell3DRegisters::default(),
            reg_file: vec![0u32; 0xE00],
            macro_engine: super::MacroEngine::new(),
            method_freq: std::collections::HashMap::new(),
            pending_draws: Vec::new(),
            macro_uploads_logged: 0,
            macro_invocations: 0,
            macro_writes_logged: 0,
        }
    }

    pub fn record_method(&mut self, method: u32) {
        *self.method_freq.entry(method).or_insert(0) += 1;
    }

    pub fn take_top_methods(&mut self, n: usize) -> Vec<(u32, u64)> {
        let mut v: Vec<(u32, u64)> = self.method_freq.drain().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    pub fn dispatch_method(&mut self, method: u32, arg: u32, is_last: bool) {
        if method >= super::MACRO_REGISTERS_START {

            if self.macro_invocations < 24 {
                log::info!(
                    "maxwell3d: MME invoke method={:#x} arg={:#x} is_last={} (slot offset {:#x})",
                    method, arg, is_last, method - super::MACRO_REGISTERS_START
                );
                self.macro_invocations += 1;
            }
            let reg_file_ptr = &self.reg_file as *const Vec<u32>;
            let writes = self.macro_engine.on_macro_method(method, arg, is_last, &|idx: u32| {
                unsafe {
                    let rf = &*reg_file_ptr;
                    rf.get(idx as usize).copied().unwrap_or(0)
                }
            });
            if let Some(out) = writes {
                if self.macro_writes_logged < 24 {
                    log::info!("maxwell3d: MME produced {} writes: {:?}", out.writes.len(),
                        out.writes.iter().take(8).copied().collect::<Vec<_>>());
                    self.macro_writes_logged += 1;
                }
                for (m, a) in out.writes {
                    self.write_register(m, a);
                }
            }
            return;
        }

        match method {
            REG_LOAD_MME_INSTRUCTION_PTR => {
                if self.macro_uploads_logged < 4 {
                    log::info!("maxwell3d: MME set_instruction_ptr = {:#x}", arg);
                }
                self.macro_engine.set_instruction_ptr(arg);
                return;
            }
            REG_LOAD_MME_INSTRUCTION => {
                if self.macro_uploads_logged < 4 {
                    self.macro_uploads_logged += 1;
                    log::info!("maxwell3d: MME upload_instruction (first dword = {:#x})", arg);
                }
                self.macro_engine.upload_instruction(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS_PTR => {
                log::info!("maxwell3d: MME set_start_address_ptr = {:#x}", arg);
                self.macro_engine.set_start_address_ptr(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS => {
                log::info!("maxwell3d: MME bind_macro_entry = {:#x}", arg);
                self.macro_engine.bind_macro_entry(arg);
                return;
            }
            _ => {}
        }

        self.write_register(method, arg);
    }

    pub fn write_register(&mut self, method: u32, arg: u32) {
        if (method as usize) < self.reg_file.len() {
            self.reg_file[method as usize] = arg;
        }

        if method >= 0x200 && method < 0x300 {
            let rt_index = ((method - 0x200) / 0x10) as usize;
            let field = (method - 0x200) % 0x10;
            if rt_index < 8 {
                let rt = &mut self.regs.rt[rt_index];
                match field {
                    0 => rt.address_hi = arg,
                    1 => rt.address_lo = arg,
                    2 => rt.width = arg,
                    3 => rt.height = arg,
                    4 => rt.format = arg,
                    5 => rt.tile_mode = arg,
                    6 => rt.depth = arg,
                    7 => rt.layer_stride = arg,
                    8 => rt.base_layer = arg,
                    _ => {}
                }
            }
            return;
        }

        match method {
            0x360..=0x363 => {
                let idx = (method - 0x360) as usize;
                let f = f32::from_bits(arg);
                match idx {
                    0 => self.regs.clear_color.r = f,
                    1 => self.regs.clear_color.g = f,
                    2 => self.regs.clear_color.b = f,
                    3 => self.regs.clear_color.a = f,
                    _ => {}
                }
            }
            0x364 => self.regs.clear_depth = f32::from_bits(arg),
            0x368 => self.regs.clear_stencil = arg,
            0x674 => {
                self.regs.clear_count += 1;
                log::debug!("maxwell3d: CLEAR_SURFACE arg={:#x} color={:?}",
                    arg, self.regs.clear_color);
                self.pending_draws.push(DrawCall {
                    topology: 0,
                    first_vertex: 0,
                    vertex_count: 0,
                    indexed: false,
                    index_count: 0,
                    rt: self.regs.rt,
                    vertex_buffers: self.regs.vertex_buffers,
                    vertex_attribs: self.regs.vertex_attribs,
                    viewport: self.regs.viewport,
                    clear_color: self.regs.clear_color,
                    is_clear: true,
                    draw_texture: None,
                    tic_pool_gpu_va: ((self.regs.tic_pool_va_hi as u64) << 32) | self.regs.tic_pool_va_lo as u64,
                    tic_pool_limit: self.regs.tic_pool_limit,
                    last_constbuf_addr: self.regs.last_constbuf_addr,
                    last_constbuf_size: self.regs.last_constbuf_size,
                    fs_bindless_cb_addr: self.regs.cbuf_binds[4][15].0,
                    fs_bindless_cb_size: self.regs.cbuf_binds[4][15].1,
                    fs_shader_gpu_va: {
                        let fs = &self.regs.shader_programs[5];
                        let region = ((self.regs.program_region_va_hi as u64) << 32)
                            | self.regs.program_region_va_lo as u64;
                        if fs.address_lo != 0 { region.wrapping_add(fs.address_lo as u64) } else { 0 }
                    },
                });
            }
            0x280..=0x2A0 => {
                let idx = ((method - 0x280) / 4) as usize;
                if idx < 8 {
                    let f = f32::from_bits(arg);
                    let field = (method - 0x280) % 4;
                    match field {
                        0 => self.regs.viewport.width = f * 2.0,
                        1 => self.regs.viewport.height = f * 2.0,
                        2 => self.regs.viewport.depth_max = f,
                        3 => {},
                        _ => {}
                    }
                    let _ = idx;
                }
            }
            0x35D => {

                self.regs.draw_count += 1;
                let count = (arg >> 16) & 0xFFFF;
                let topology = (arg >> 28) & 0xF;
                log::debug!("maxwell3d: DRAW_VERTEX_ARRAY_BEGIN count={} topology={}", count, topology);
                self.push_draw(topology, self.regs.draw_first_vertex, count, false, 0);
            }
            0x35E => {

                self.regs.draw_vertex_count = arg;
                self.regs.draw_count += 1;
                log::debug!("maxwell3d: DrawArraysCount count={} topology={} first={}",
                    arg, self.regs.draw_topology, self.regs.draw_first_vertex);
                if arg > 0 {
                    self.push_draw(self.regs.draw_topology, self.regs.draw_first_vertex, arg, false, 0);
                }
            }
            0x35F => self.regs.draw_first_vertex = arg,
            0x485 | 0x486 => {

                self.regs.draw_count += 1;
                let count = (arg >> 16) & 0xFFF;
                let topology = (arg >> 28) & 0xF;
                log::debug!("maxwell3d: DRAW_VERTEX_ARRAY_BEGIN_END count={} topology={}", count, topology);
                self.push_draw(topology, self.regs.draw_first_vertex, count, false, 0);
            }

            0x586 => {
                self.regs.draw_topology = arg & 0xFFFF;
            }
            0x585 => {

            }
            0x5F8 => {
                self.regs.draw_count += 1;
                self.regs.index_count = arg;
                log::debug!("maxwell3d: DrawElementsCount count={} topology={}", arg, self.regs.draw_topology);
                self.push_draw(self.regs.draw_topology, 0, 0, true, arg);
            }

            0x55D => {
                log::info!("maxwell3d: SetTexHeaderPool[hi] = {:#x}", arg);
                self.regs.tic_pool_va_hi = arg;
            }
            0x55E => {
                log::info!("maxwell3d: SetTexHeaderPool[lo] = {:#x} → full {:#x}",
                    arg, ((self.regs.tic_pool_va_hi as u64) << 32) | arg as u64);
                self.regs.tic_pool_va_lo = arg;
            }
            0x55F => {
                log::info!("maxwell3d: SetTexHeaderPoolMaximumIndex = {:#x}", arg);
                self.regs.tic_pool_limit = arg;
            }

            0x8E0 => self.regs.constbuf_selector_size = arg,
            0x8E1 => self.regs.constbuf_selector_addr_hi = arg,
            0x8E2 => {
                self.regs.constbuf_selector_addr_lo = arg;
                self.regs.last_constbuf_addr =
                    ((self.regs.constbuf_selector_addr_hi as u64) << 32) | arg as u64;
                self.regs.last_constbuf_size = self.regs.constbuf_selector_size;
            }
            0x8E3 => self.regs.constbuf_load_offset = arg,
            0x8E4 => {
                let cb_addr = ((self.regs.constbuf_selector_addr_hi as u64) << 32)
                    | self.regs.constbuf_selector_addr_lo as u64;
                if cb_addr != 0 {
                    let target = cb_addr + self.regs.constbuf_load_offset as u64;
                    self.regs.pending_constbuf_writes.push((target, arg));
                }
                self.regs.constbuf_load_offset = self.regs.constbuf_load_offset.wrapping_add(4);
            }

            0x904 | 0x90C | 0x914 | 0x91C | 0x924 => {
                let stage = ((method - 0x904) / 8) as usize;
                let valid = (arg & 1) != 0;
                let slot = ((arg >> 4) & 0xF) as usize;
                if valid && stage < 5 && slot < 16 {
                    let cb_addr = ((self.regs.constbuf_selector_addr_hi as u64) << 32)
                        | self.regs.constbuf_selector_addr_lo as u64;
                    self.regs.cbuf_binds[stage][slot] = (cb_addr, self.regs.constbuf_selector_size);
                }
            }
            0x700..=0x77F => {
                let idx = ((method - 0x700) / 4) as usize;
                let field = (method - 0x700) % 4;
                if idx < 32 {
                    let vb = &mut self.regs.vertex_buffers[idx];
                    match field {
                        0 => vb.stride = arg & 0xFFF,
                        1 => vb.address_hi = arg,
                        2 => vb.address_lo = arg,
                        3 => vb.size = arg,
                        _ => {}
                    }
                }
            }
            0x458..=0x477 => {

                let idx = (method - 0x458) as usize;
                if idx < 32 {
                    let va = &mut self.regs.vertex_attribs[idx];
                    va.buffer = arg & 0x1F;
                    va.offset = (arg >> 7) & 0x3FFF;
                    let size = (arg >> 21) & 0x3F;
                    let r#type = (arg >> 27) & 0x7;
                    va.format = size | (r#type << 6);
                }
            }

            0x582 => {
                log::trace!("maxwell3d: SetProgramRegion[hi] = {:#x}", arg);
                self.regs.program_region_va_hi = arg;
            }
            0x583 => {
                log::trace!("maxwell3d: SetProgramRegion[lo] = {:#x} → full {:#x}",
                    arg, ((self.regs.program_region_va_hi as u64) << 32) | arg as u64);
                self.regs.program_region_va_lo = arg;
            }

            0x800..=0x85F => {
                let idx = ((method - 0x800) / 0x10) as usize;
                if idx < 6 {
                    let sp = &mut self.regs.shader_programs[idx];
                    let field = (method - 0x800) % 0x10;
                    log::trace!("maxwell3d: SetProgram[stage={}] field={} = {:#x}", idx, field, arg);
                    match field {
                        0 => sp.enabled = (arg & 1) != 0,

                        1 => sp.address_lo = arg,
                        3 => sp.gpr_count = arg,
                        _ => {}
                    }
                }
            }
            _ => {
                log::trace!("maxwell3d: write method {:#x} = {:#x}", method, arg);
            }
        }
    }

    fn push_draw(&mut self, topology: u32, first: u32, count: u32, indexed: bool, index_count: u32) {
        let tic_pool_gpu_va = ((self.regs.tic_pool_va_hi as u64) << 32) | self.regs.tic_pool_va_lo as u64;
        let (fs_bindless_cb_addr, fs_bindless_cb_size) = self.regs.cbuf_binds[4][15];
        let fs = &self.regs.shader_programs[5];

        let program_region = ((self.regs.program_region_va_hi as u64) << 32)
            | self.regs.program_region_va_lo as u64;
        let fs_shader_gpu_va = if fs.address_lo != 0 {
            program_region.wrapping_add(fs.address_lo as u64)
        } else { 0 };
        self.pending_draws.push(DrawCall {
            topology,
            first_vertex: first,
            vertex_count: count,
            indexed,
            index_count,
            rt: self.regs.rt,
            vertex_buffers: self.regs.vertex_buffers,
            vertex_attribs: self.regs.vertex_attribs,
            viewport: self.regs.viewport,
            clear_color: self.regs.clear_color,
            is_clear: false,
            draw_texture: None,
            tic_pool_gpu_va,
            tic_pool_limit: self.regs.tic_pool_limit,
            last_constbuf_addr: self.regs.last_constbuf_addr,
            last_constbuf_size: self.regs.last_constbuf_size,
            fs_bindless_cb_addr,
            fs_bindless_cb_size,
            fs_shader_gpu_va,
        });
    }

    fn push_draw_texture(&mut self, texture_id: u32, sampler_id: u32) {
        let tic_pool_gpu_va = ((self.regs.tic_pool_va_hi as u64) << 32) | self.regs.tic_pool_va_lo as u64;
        self.regs.draw_count += 1;
        self.regs.draw_texture_count += 1;
        self.pending_draws.push(DrawCall {
            topology: 0,
            first_vertex: 0,
            vertex_count: 0,
            indexed: false,
            index_count: 0,
            rt: self.regs.rt,
            vertex_buffers: self.regs.vertex_buffers,
            vertex_attribs: self.regs.vertex_attribs,
            viewport: self.regs.viewport,
            clear_color: self.regs.clear_color,
            is_clear: false,
            draw_texture: Some(DrawTextureCall {
                dst_x: self.regs.draw_texture_dst_x,
                dst_y: self.regs.draw_texture_dst_y,
                dst_width: self.regs.draw_texture_dst_width,
                dst_height: self.regs.draw_texture_dst_height,
                texture_id,
                sampler_id,
                tic_pool_gpu_va,
                tic_pool_limit: self.regs.tic_pool_limit,
            }),
            tic_pool_gpu_va,
            tic_pool_limit: self.regs.tic_pool_limit,
            last_constbuf_addr: self.regs.last_constbuf_addr,
            last_constbuf_size: self.regs.last_constbuf_size,
            fs_bindless_cb_addr: self.regs.cbuf_binds[4][15].0,
            fs_bindless_cb_size: self.regs.cbuf_binds[4][15].1,
            fs_shader_gpu_va: {
                let region = ((self.regs.program_region_va_hi as u64) << 32)
                    | self.regs.program_region_va_lo as u64;
                let fs = &self.regs.shader_programs[5];
                if fs.address_lo != 0 { region.wrapping_add(fs.address_lo as u64) } else { 0 }
            },
        });
    }

    pub fn render_target(&self, idx: usize) -> Option<&RenderTarget> {
        self.regs.rt.get(idx).filter(|rt| rt.width > 0 && rt.height > 0)
    }

    pub fn primary_rt_gpu_va(&self) -> Option<u64> {
        let rt = self.render_target(0)?;
        Some(((rt.address_hi as u64) << 32) | rt.address_lo as u64)
    }

    pub fn primary_rt_size(&self) -> Option<(u32, u32)> {
        let rt = self.render_target(0)?;
        Some((rt.width, rt.height))
    }

    pub fn draw_count(&self) -> u64 {
        self.regs.draw_count
    }

    pub fn clear_count(&self) -> u64 {
        self.regs.clear_count
    }
}

impl Default for Maxwell3D {
    fn default() -> Self {
        Self::new()
    }
}

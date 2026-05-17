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
        }
    }
}

pub struct Maxwell3D {
    pub regs: Maxwell3DRegisters,
    pub reg_file: Vec<u32>,
    pub macro_engine: super::MacroEngine,
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
        }
    }

    pub fn dispatch_method(&mut self, method: u32, arg: u32, is_last: bool) {
        if method >= super::MACRO_REGISTERS_START {
            let reg_file_ptr = &self.reg_file as *const Vec<u32>;
            let writes = self.macro_engine.on_macro_method(method, arg, is_last, &|idx: u32| {
                unsafe {
                    let rf = &*reg_file_ptr;
                    rf.get(idx as usize).copied().unwrap_or(0)
                }
            });
            if let Some(out) = writes {
                for (m, a) in out.writes {
                    self.write_register(m, a);
                }
            }
            return;
        }

        match method {
            REG_LOAD_MME_INSTRUCTION_PTR => {
                self.macro_engine.set_instruction_ptr(arg);
                return;
            }
            REG_LOAD_MME_INSTRUCTION => {
                self.macro_engine.upload_instruction(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS_PTR => {
                self.macro_engine.set_start_address_ptr(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS => {
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
                log::info!("maxwell3d: CLEAR_SURFACE arg={:#x} color={:?}",
                    arg, self.regs.clear_color);
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
                log::info!("maxwell3d: DRAW_VERTEX_ARRAY_BEGIN count={} topology={}",
                    count, topology);
            }
            0x485 | 0x486 => {
                self.regs.draw_count += 1;
                let count = (arg >> 16) & 0xFFF;
                let topology = (arg >> 28) & 0xF;
                log::info!("maxwell3d: DRAW_VERTEX_ARRAY_BEGIN_END count={} topology={}",
                    count, topology);
            }
            0x35E => self.regs.draw_vertex_count = arg,
            0x35F => self.regs.draw_first_vertex = arg,
            0x5F8 => {
                self.regs.draw_count += 1;
                self.regs.index_count = arg;
                log::info!("maxwell3d: DRAW_INDEX_BUFFER count={}", arg);
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
            0x900..=0x91F => {
                let idx = (method - 0x900) as usize;
                if idx < 32 {
                    let va = &mut self.regs.vertex_attribs[idx];
                    va.buffer = arg & 0x1F;
                    va.offset = (arg >> 7) & 0x3FFF;
                    va.format = (arg >> 21) & 0x3F;
                }
            }
            0x5C0..=0x5F0 => {
                let idx = ((method - 0x5C0) / 0x10) as usize;
                if idx < 6 {
                    let sp = &mut self.regs.shader_programs[idx];
                    let field = (method - 0x5C0) % 0x10;
                    match field {
                        0 => sp.enabled = (arg & 1) != 0,
                        1 => sp.address_lo = arg,
                        2 => sp.address_hi = arg,
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

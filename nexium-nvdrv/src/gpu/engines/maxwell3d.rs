#[cfg(test)]
use super::kepler_memory::{KeplerMemory, KeplerMemoryWriteOutcome};

pub const MAXWELL3D_CLASS: u32 = 0xB197;
pub const GRAPHICS_CBUF_SLOTS: usize = nexium_spirv::GFX_CBUF_STAGE_SLOTS as usize;
pub type GraphicsCbufBinds = [[(u64, u32); GRAPHICS_CBUF_SLOTS]; 5];

#[derive(Clone, Copy)]
pub struct GsDebugRegs {
    pub post_vtg_masks: [u32; 8],
    pub stage3_cbuf_binds: [(u64, u32); GRAPHICS_CBUF_SLOTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemaphoreWriteOrdering {
    RendererOrdered,
    PayloadFence,
    SyntheticCounter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingSemaphoreWrite {
    pub gpu_va: u64,
    pub payload: u32,
    pub long: bool,
    pub ordering: SemaphoreWriteOrdering,
}

impl PendingSemaphoreWrite {
    pub fn requires_renderer_completion(self) -> bool {
        matches!(
            self.ordering,
            SemaphoreWriteOrdering::RendererOrdered | SemaphoreWriteOrdering::PayloadFence
        )
    }

    pub fn can_complete_asynchronously(self) -> bool {
        self.ordering == SemaphoreWriteOrdering::PayloadFence && !self.long
    }
}

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
pub struct ZetaSurface {
    pub address_lo: u32,
    pub address_hi: u32,
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub block_size: u32,
    pub array_pitch: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFaceState {
    pub fail_op: u32,
    pub depth_fail_op: u32,
    pub depth_pass_op: u32,
    pub compare_op: u32,
    pub reference: u32,
    pub compare_mask: u32,
    pub write_mask: u32,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            fail_op: 1,
            depth_fail_op: 1,
            depth_pass_op: 1,
            compare_op: 0x207,
            reference: 0,
            compare_mask: u32::MAX,
            write_mask: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub depth_min: f32,
    pub depth_max: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub translate_x: f32,
    pub translate_y: f32,
    pub scale_z: f32,
    pub translate_z: f32,
    pub swizzle: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            depth_min: 0.0,
            depth_max: 0.0,
            scale_x: 0.0,
            scale_y: 0.0,
            translate_x: 0.0,
            translate_y: 0.0,
            scale_z: 0.0,
            translate_z: 0.0,
            swizzle: 0x6420,
        }
    }
}

impl Viewport {
    pub fn y_swizzle(self) -> u32 {
        (self.swizzle >> 4) & 0x7
    }

    pub fn y_negate(self) -> bool {
        self.y_swizzle() == 3
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct SurfaceClip {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl SurfaceClip {
    pub fn effective(self, rt_width: u32, rt_height: u32) -> Self {
        Self {
            x: self.x,
            y: self.y,
            width: if self.width != 0 {
                self.width
            } else {
                rt_width
            },
            height: if self.height != 0 {
                self.height
            } else {
                rt_height
            },
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ViewportClipControl {
    pub raw: u32,
}

impl ViewportClipControl {
    pub fn geometry_clip(self) -> u32 {
        (self.raw >> 11) & 0x7
    }

    pub fn depth_clamp_enabled(self) -> bool {
        !matches!(self.geometry_clip(), 1 | 3 | 5)
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct WindowOrigin {
    pub raw: u32,
}

impl WindowOrigin {
    pub fn lower_left(self) -> bool {
        self.raw & 1 != 0
    }

    pub fn triangle_rast_flip(self) -> bool {
        self.raw & 0x10 != 0
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ScissorTest {
    pub enabled: bool,
    pub min_x: u32,
    pub max_x: u32,
    pub min_y: u32,
    pub max_y: u32,
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
    pub constant: bool,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct VertexBuffer {
    pub stride: u32,
    pub enabled: bool,
    pub address_lo: u32,
    pub address_hi: u32,
    pub frequency: u32,
    pub end_lo: u32,
    pub end_hi: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorBlendState {
    pub blend_enable: [bool; 8],
    pub blend_eq_rgb: u32,
    pub blend_src_rgb: u32,
    pub blend_dst_rgb: u32,
    pub blend_eq_alpha: u32,
    pub blend_src_alpha: u32,
    pub blend_dst_alpha: u32,
    pub blend_per_target_enabled: bool,
    pub blend_pt_eq_rgb: [u32; 8],
    pub blend_pt_src_rgb: [u32; 8],
    pub blend_pt_dst_rgb: [u32; 8],
    pub blend_pt_eq_alpha: [u32; 8],
    pub blend_pt_src_alpha: [u32; 8],
    pub blend_pt_dst_alpha: [u32; 8],
    pub color_mask_common: bool,
    pub color_masks: [u32; 8],
    pub blend_constants: [f32; 4],
}

impl Default for ColorBlendState {
    fn default() -> Self {
        Self {
            blend_enable: [false; 8],
            blend_eq_rgb: 0x8006,
            blend_src_rgb: 0x4001,
            blend_dst_rgb: 0x4000,
            blend_eq_alpha: 0x8006,
            blend_src_alpha: 0x4001,
            blend_dst_alpha: 0x4000,
            blend_per_target_enabled: false,
            blend_pt_eq_rgb: [0x8006; 8],
            blend_pt_src_rgb: [0x4001; 8],
            blend_pt_dst_rgb: [0x4000; 8],
            blend_pt_eq_alpha: [0x8006; 8],
            blend_pt_src_alpha: [0x4001; 8],
            blend_pt_dst_alpha: [0x4000; 8],
            color_mask_common: false,
            color_masks: [0x1111; 8],
            blend_constants: [0.0; 4],
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ShaderProgram {
    pub address_lo: u32,
    pub address_hi: u32,
    pub gpr_count: u32,
    pub binding_group: Option<u32>,
    pub enabled: bool,
}

impl ShaderProgram {
    pub fn cbuf_group(self, fallback: usize) -> usize {
        self.binding_group
            .and_then(|group| (group < 5).then_some(group as usize))
            .unwrap_or(fallback)
    }
}

#[derive(Clone)]
pub struct Maxwell3DRegisters {
    pub rt: [RenderTarget; 8],
    pub rt_control: u32,
    pub viewport: Viewport,
    pub clear_color: ClearColor,
    pub clear_depth: f32,
    pub clear_stencil: u32,
    pub vertex_attribs: [VertexAttribute; 32],
    pub vertex_buffers: [VertexBuffer; 32],
    pub vertex_stream_instances: [u32; 32],
    pub shader_programs: [ShaderProgram; 6],
    pub draw_vertex_count: u32,
    pub draw_first_vertex: u32,
    pub draw_topology: u32,
    pub primitive_topology_control: u32,
    pub topology_override: u32,
    pub vertex_array_instance_count: u32,
    pub global_base_vertex_index: u32,
    pub global_base_instance_index: u32,
    pub viewport_transform_en: bool,
    pub viewport_clip_control: ViewportClipControl,
    pub surface_clip: SurfaceClip,
    pub window_origin: WindowOrigin,
    pub scissor: ScissorTest,
    pub clear_control: u32,
    pub index_buffer_lo: u32,
    pub index_buffer_hi: u32,
    pub index_buffer_end_lo: u32,
    pub index_buffer_end_hi: u32,
    pub index_format: u32,
    pub index_count: u32,
    pub index_first: u32,
    pub primitive_restart_enabled: bool,
    pub primitive_restart_index: u32,
    pub depth_mode: u32,
    pub depth_test_enable: bool,
    pub zeta: ZetaSurface,
    pub zeta_enable: bool,
    pub multisample_mode: u32,
    pub depth_write_enable: bool,
    pub depth_func: u32,
    pub stencil_enable: bool,
    pub stencil_two_side_enable: bool,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub cull_test_enable: bool,
    pub alpha_test_enabled: bool,
    pub alpha_test_ref: u32,
    pub alpha_test_func: u32,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_fill_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub blend_enable: [bool; 8],
    pub blend_eq_rgb: u32,
    pub blend_src_rgb: u32,
    pub blend_dst_rgb: u32,
    pub blend_eq_alpha: u32,
    pub blend_src_alpha: u32,
    pub blend_dst_alpha: u32,
    pub blend_per_target_enabled: bool,
    pub blend_pt_eq_rgb: [u32; 8],
    pub blend_pt_src_rgb: [u32; 8],
    pub blend_pt_dst_rgb: [u32; 8],
    pub blend_pt_eq_alpha: [u32; 8],
    pub blend_pt_src_alpha: [u32; 8],
    pub blend_pt_dst_alpha: [u32; 8],
    pub color_mask_common: bool,
    pub color_masks: [u32; 8],
    pub blend_constants: [f32; 4],
    pub draw_count: u64,
    pub clear_count: u64,

    pub tic_pool_va_lo: u32,
    pub tic_pool_va_hi: u32,
    pub tsc_pool_va_lo: u32,
    pub tsc_pool_va_hi: u32,
    pub tsc_pool_limit: u32,

    pub program_region_va_hi: u32,
    pub program_region_va_lo: u32,
    pub tic_pool_limit: u32,

    pub draw_texture_dst_x: u32,
    pub draw_texture_dst_y: u32,
    pub draw_texture_dst_width: u32,
    pub draw_texture_dst_height: u32,
    pub draw_texture_dx_du_lo: u32,
    pub draw_texture_dx_du_hi: u32,
    pub draw_texture_dy_dv_lo: u32,
    pub draw_texture_dy_dv_hi: u32,
    pub draw_texture_src_sampler: u32,
    pub draw_texture_src_texture: u32,
    pub draw_texture_src_x: u32,
    pub draw_texture_src_y: u32,
    pub draw_texture_count: u64,

    pub constbuf_selector_size: u32,
    pub constbuf_selector_addr_hi: u32,
    pub constbuf_selector_addr_lo: u32,
    pub constbuf_load_offset: u32,

    pub pending_constbuf_writes: Vec<(u64, u32)>,
    pub pending_semaphore_writes: Vec<PendingSemaphoreWrite>,
    pub pending_semaphore_acquires: Vec<(u64, u32, u32)>,
    pub pending_barrier_flushes: u32,
    pub pending_fragment_barriers: u32,
    pub pending_tiled_cache_barriers: u32,
    pub pending_texture_cache_invalidates: u32,
    pub sync_info: u32,
    pub clear_report_value: u32,
    pub zpass_pixel_count_enable: bool,

    pub render_enable_addr_hi: u32,
    pub render_enable_addr_lo: u32,
    pub render_enable_mode: u32,
    pub render_enable_override: u32,

    pub last_constbuf_addr: u64,
    pub last_constbuf_size: u32,

    pub cbuf_binds: GraphicsCbufBinds,
    pub tex_cb_index: u32,
    pub sampler_binding: u32,
    pub bindless_texture_const_buffer_slot: u32,
}

impl Default for Maxwell3DRegisters {
    fn default() -> Self {
        Self {
            rt: [RenderTarget::default(); 8],
            rt_control: 1,
            viewport: Viewport::default(),
            clear_color: ClearColor::default(),
            clear_depth: 0.0,
            clear_stencil: 0,
            vertex_attribs: [VertexAttribute::default(); 32],
            vertex_buffers: [VertexBuffer::default(); 32],
            vertex_stream_instances: [0; 32],
            shader_programs: [ShaderProgram::default(); 6],
            draw_vertex_count: 0,
            draw_first_vertex: 0,
            draw_topology: 0,
            primitive_topology_control: 0,
            topology_override: 0,
            vertex_array_instance_count: 0,
            global_base_vertex_index: 0,
            global_base_instance_index: 0,
            viewport_transform_en: true,
            viewport_clip_control: ViewportClipControl::default(),
            surface_clip: SurfaceClip::default(),
            window_origin: WindowOrigin::default(),
            scissor: ScissorTest::default(),
            clear_control: 0,
            index_buffer_lo: 0,
            index_buffer_hi: 0,
            index_buffer_end_lo: 0,
            index_buffer_end_hi: 0,
            index_format: 0,
            index_count: 0,
            index_first: 0,
            primitive_restart_enabled: false,
            primitive_restart_index: 0,
            depth_mode: 0,
            depth_test_enable: false,
            zeta: ZetaSurface::default(),
            zeta_enable: false,
            multisample_mode: 0,
            depth_write_enable: false,
            depth_func: 0x207,
            stencil_enable: false,
            stencil_two_side_enable: true,
            stencil_front: StencilFaceState::default(),
            stencil_back: StencilFaceState::default(),
            cull_test_enable: false,
            alpha_test_enabled: false,
            alpha_test_ref: 0,
            alpha_test_func: 7,
            cull_face: 0x405,
            front_face: 0x901,
            poly_offset_fill_enable: false,
            poly_offset_units: 0.0,
            poly_offset_factor: 0.0,
            blend_enable: [false; 8],
            blend_eq_rgb: 0x8006,
            blend_src_rgb: 0x4001,
            blend_dst_rgb: 0x4000,
            blend_eq_alpha: 0x8006,
            blend_src_alpha: 0x4001,
            blend_dst_alpha: 0x4000,
            blend_per_target_enabled: false,
            blend_pt_eq_rgb: [0x8006; 8],
            blend_pt_src_rgb: [0x4001; 8],
            blend_pt_dst_rgb: [0x4000; 8],
            blend_pt_eq_alpha: [0x8006; 8],
            blend_pt_src_alpha: [0x4001; 8],
            blend_pt_dst_alpha: [0x4000; 8],
            color_mask_common: false,
            color_masks: [0x1111; 8],
            blend_constants: [0.0; 4],
            draw_count: 0,
            clear_count: 0,
            tic_pool_va_lo: 0,
            tic_pool_va_hi: 0,
            tsc_pool_va_lo: 0,
            tsc_pool_va_hi: 0,
            tsc_pool_limit: 0,
            program_region_va_hi: 0,
            program_region_va_lo: 0,
            tic_pool_limit: 0,
            draw_texture_dst_x: 0,
            draw_texture_dst_y: 0,
            draw_texture_dst_width: 0,
            draw_texture_dst_height: 0,
            draw_texture_dx_du_lo: 0,
            draw_texture_dx_du_hi: 0,
            draw_texture_dy_dv_lo: 0,
            draw_texture_dy_dv_hi: 0,
            draw_texture_src_sampler: 0,
            draw_texture_src_texture: 0,
            draw_texture_src_x: 0,
            draw_texture_src_y: 0,
            draw_texture_count: 0,
            constbuf_selector_size: 0,
            constbuf_selector_addr_hi: 0,
            constbuf_selector_addr_lo: 0,
            constbuf_load_offset: 0,
            pending_constbuf_writes: Vec::new(),
            pending_semaphore_writes: Vec::new(),
            pending_semaphore_acquires: Vec::new(),
            pending_barrier_flushes: 0,
            pending_fragment_barriers: 0,
            pending_tiled_cache_barriers: 0,
            pending_texture_cache_invalidates: 0,
            sync_info: 0,
            clear_report_value: 0,
            zpass_pixel_count_enable: false,
            render_enable_addr_hi: 0,
            render_enable_addr_lo: 0,
            render_enable_mode: 1,
            render_enable_override: 0,
            last_constbuf_addr: 0,
            last_constbuf_size: 0,
            cbuf_binds: [[(0, 0); GRAPHICS_CBUF_SLOTS]; 5],
            tex_cb_index: 0,
            sampler_binding: 0,
            bindless_texture_const_buffer_slot: 0,
        }
    }
}

impl Maxwell3DRegisters {
    fn reset_register_image(&self) -> Vec<u32> {
        let mut image = vec![0u32; MAXWELL3D_REGISTER_COUNT];

        image[0x286] = self.viewport.swizzle;
        image[0x3D6] = self.stencil_back.write_mask;
        image[0x3D7] = self.stencil_back.compare_mask;
        image[0x487] = self.rt_control;
        image[0x4C3] = self.depth_func;
        image[0x4C5] = self.alpha_test_func;
        for (index, value) in self.blend_constants.iter().enumerate() {
            image[0x4C7 + index] = value.to_bits();
        }
        image[0x4D0] = self.blend_eq_rgb;
        image[0x4D1] = self.blend_src_rgb;
        image[0x4D2] = self.blend_dst_rgb;
        image[0x4D3] = self.blend_eq_alpha;
        image[0x4D4] = self.blend_src_alpha;
        image[0x4D6] = self.blend_dst_alpha;
        image[0x4E1] = self.stencil_front.fail_op;
        image[0x4E2] = self.stencil_front.depth_fail_op;
        image[0x4E3] = self.stencil_front.depth_pass_op;
        image[0x4E4] = self.stencil_front.compare_op;
        image[0x4E6] = self.stencil_front.compare_mask;
        image[0x4E7] = self.stencil_front.write_mask;
        image[0x556] = self.render_enable_mode;
        image[0x565] = u32::from(self.stencil_two_side_enable);
        image[0x566] = self.stencil_back.fail_op;
        image[0x567] = self.stencil_back.depth_fail_op;
        image[0x568] = self.stencil_back.depth_pass_op;
        image[0x569] = self.stencil_back.compare_op;
        image[0x647] = self.front_face;
        image[0x648] = self.cull_face;
        image[0x64B] = u32::from(self.viewport_transform_en);

        for (rt, &mask) in self.color_masks.iter().enumerate() {
            image[0x680 + rt] = mask;
        }
        for rt in 0..8 {
            let base = 0x780 + rt * 8;
            image[base + 1] = self.blend_pt_eq_rgb[rt];
            image[base + 2] = self.blend_pt_src_rgb[rt];
            image[base + 3] = self.blend_pt_dst_rgb[rt];
            image[base + 4] = self.blend_pt_eq_alpha[rt];
            image[base + 5] = self.blend_pt_src_alpha[rt];
            image[base + 6] = self.blend_pt_dst_alpha[rt];
        }

        image
    }

    fn color_blend_state(&self) -> ColorBlendState {
        ColorBlendState {
            blend_enable: self.blend_enable,
            blend_eq_rgb: self.blend_eq_rgb,
            blend_src_rgb: self.blend_src_rgb,
            blend_dst_rgb: self.blend_dst_rgb,
            blend_eq_alpha: self.blend_eq_alpha,
            blend_src_alpha: self.blend_src_alpha,
            blend_dst_alpha: self.blend_dst_alpha,
            blend_per_target_enabled: self.blend_per_target_enabled,
            blend_pt_eq_rgb: self.blend_pt_eq_rgb,
            blend_pt_src_rgb: self.blend_pt_src_rgb,
            blend_pt_dst_rgb: self.blend_pt_dst_rgb,
            blend_pt_eq_alpha: self.blend_pt_eq_alpha,
            blend_pt_src_alpha: self.blend_pt_src_alpha,
            blend_pt_dst_alpha: self.blend_pt_dst_alpha,
            color_mask_common: self.color_mask_common,
            color_masks: self.color_masks,
            blend_constants: self.blend_constants,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DrawCall {
    pub topology: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_instance: u32,
    pub indexed: bool,
    pub index_count: u32,
    pub index_gpu_va: u64,
    pub index_format: u32,
    pub index_first: u32,
    pub inline_indices: Vec<u32>,
    pub primitive_restart_enabled: bool,
    pub primitive_restart_index: u32,
    pub point_size: f32,

    pub rt: [RenderTarget; 8],
    pub rt_control: u32,
    pub vertex_buffers: [VertexBuffer; 32],
    pub vertex_stream_instances: [u32; 32],
    pub vertex_attribs: [VertexAttribute; 32],
    pub viewport: Viewport,
    pub depth_mode: u32,
    pub viewport_transform_en: bool,
    pub viewport_clip_control: ViewportClipControl,
    pub surface_clip: SurfaceClip,
    pub window_origin: WindowOrigin,
    pub scissor: ScissorTest,
    pub clear_control: u32,
    pub clear_color: ClearColor,
    pub color_blend: ColorBlendState,
    pub is_clear: bool,

    pub draw_texture: Option<DrawTextureCall>,

    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
    pub tsc_pool_gpu_va: u64,
    pub tsc_pool_limit: u32,

    pub shader_programs: [ShaderProgram; 6],
    pub program_region_gpu_va: u64,
    pub cbuf_binds: GraphicsCbufBinds,
    pub sampler_binding: u32,
    pub bindless_texture_const_buffer_slot: u32,
    pub tex_cb_index: u32,
    pub constbuf_write_count: usize,

    pub last_constbuf_addr: u64,
    pub last_constbuf_size: u32,

    pub fs_bindless_cb_addr: u64,
    pub fs_bindless_cb_size: u32,

    pub fs_shader_gpu_va: u64,

    pub render_enable_addr: u64,
    pub render_enable_mode: u32,
    pub render_enable_override: u32,

    pub cull_test_enable: bool,
    pub alpha_test_enabled: bool,
    pub alpha_test_ref: u32,
    pub alpha_test_func: u32,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_fill_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,

    pub zeta: ZetaSurface,
    pub zeta_enable: bool,
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_func: u32,
    pub stencil_enable: bool,
    pub stencil_two_side_enable: bool,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub multisample_mode: u32,
    pub clear_depth: f32,
    pub clear_stencil: u32,
    pub clear_mask: u32,
    pub state_clean_from_previous: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct DrawTextureCall {
    pub dst_x: f32,
    pub dst_y: f32,
    pub dst_width: f32,
    pub dst_height: f32,
    pub src_x: f32,
    pub src_y: f32,
    pub src_width: f32,
    pub src_height: f32,
    pub texture_id: u32,
    pub sampler_id: u32,
    pub tic_pool_gpu_va: u64,
    pub tic_pool_limit: u32,
}

pub struct Maxwell3D {
    pub regs: Maxwell3DRegisters,
    pub reg_file: Vec<u32>,
    reg_file_written: Vec<u8>,
    pub shadow_ram_control: u32,
    pub shadow_regs: Vec<u32>,
    pub macro_engine: super::MacroEngine,
    pub method_freq: std::collections::HashMap<u32, u64>,
    method_profile_enabled: bool,

    pub pending_draws: Vec<DrawCall>,

    pending_inline_upload_methods: Vec<(u32, u32)>,
    inline_indices: Vec<u32>,
    inline_u8_setup: Option<(usize, usize)>,
    inline_u16_setup: Option<(usize, usize)>,

    pub macro_uploads_logged: u32,
    pub macro_invocations: u32,
    pub macro_writes_logged: u32,
    macro_draw_instance_count: Option<u32>,
    mme_active: bool,
    mme_hash: u64,
    mme_entry: u32,
    legacy_draw_instance_id: u32,
    legacy_draw_begin_pending: bool,
    legacy_draw_vertex_pending: bool,
    legacy_draw_index_pending: bool,
    draw_state_dirty_since_last_draw: bool,
    last_draw_allows_continuation: bool,
}

const REG_LOAD_MME_INSTRUCTION_PTR: u32 = 0x45;
const REG_LOAD_MME_INSTRUCTION: u32 = 0x46;
const REG_LOAD_MME_START_ADDRESS_PTR: u32 = 0x47;
const REG_LOAD_MME_START_ADDRESS: u32 = 0x48;
const MAXWELL3D_REGISTER_COUNT: usize = 0xE00;

fn mme_forensics() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_FORENSICS").is_some())
}

fn mme_trace() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_MME_TRACE").is_some())
}

fn viewport_enable_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_VPEN_DBG").is_some())
}

fn cbuf_bind_trace() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_CBUF_BIND_TRACE").is_some())
}

fn raw_counter_reports() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_RAW_COUNTER_REPORTS").is_some())
}

fn wf_state_log() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("NEXIUM_WATER_FORENSICS").is_some())
}

fn synthetic_counter_value() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static C: AtomicU32 = AtomicU32::new(0x1000);
    C.fetch_add(0x1000, Ordering::Relaxed)
}

fn is_short_payload_fence(query: u32) -> bool {
    let operation = query & 0x3;
    let reduction_enabled = (query >> 3) & 1;
    let sub_report = (query >> 5) & 0x7;
    let dword_number = (query >> 21) & 1;
    let report = (query >> 23) & 0x1f;
    let short_query = (query >> 28) & 1;
    operation == 0
        && reduction_enabled == 0
        && sub_report == 0
        && dword_number == 0
        && report == 0
        && short_query == 1
}

fn report_semaphore_write_ordering(
    query: u32,
    raw_counter_reports: bool,
) -> Option<SemaphoreWriteOrdering> {
    let operation = query & 0x3;
    match operation {
        0 if is_short_payload_fence(query) => Some(SemaphoreWriteOrdering::PayloadFence),
        0 => Some(SemaphoreWriteOrdering::RendererOrdered),
        2 if !raw_counter_reports => Some(SemaphoreWriteOrdering::SyntheticCounter),
        2 => Some(SemaphoreWriteOrdering::RendererOrdered),
        _ => None,
    }
}

fn trace_sync_method(method: u32, arg: u32, pending: u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::OnceLock;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MW3D_SYNC_DBG").is_some()) {
        return;
    }
    static N: AtomicU32 = AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 256 {
        log::warn!(
            "[mw3d-sync] #{} method={:#x} arg={:#x} pending={}",
            n + 1,
            method,
            arg,
            pending
        );
    }
}

impl Maxwell3D {
    pub fn new() -> Self {
        let regs = Maxwell3DRegisters::default();
        let reg_file = regs.reset_register_image();
        let shadow_regs = reg_file.clone();
        Self {
            regs,
            reg_file,
            reg_file_written: vec![0; MAXWELL3D_REGISTER_COUNT],
            shadow_ram_control: 0,
            shadow_regs,
            macro_engine: super::MacroEngine::new(),
            method_freq: std::collections::HashMap::new(),
            method_profile_enabled: std::env::var("NEXIUM_GPU_METHOD_PROFILE")
                .map(|v| v != "0")
                .unwrap_or(false),
            pending_draws: Vec::new(),
            pending_inline_upload_methods: Vec::new(),
            inline_indices: Vec::new(),
            inline_u8_setup: None,
            inline_u16_setup: None,
            macro_uploads_logged: 0,
            macro_invocations: 0,
            macro_writes_logged: 0,
            macro_draw_instance_count: None,
            mme_active: false,
            mme_hash: 0,
            mme_entry: 0,
            legacy_draw_instance_id: 0,
            legacy_draw_begin_pending: false,
            legacy_draw_vertex_pending: false,
            legacy_draw_index_pending: false,
            draw_state_dirty_since_last_draw: true,
            last_draw_allows_continuation: false,
        }
    }

    pub fn record_method(&mut self, method: u32) {
        if !self.method_profile_enabled {
            return;
        }
        *self.method_freq.entry(method).or_insert(0) += 1;
    }

    pub(crate) fn is_pusher_passive_method(method: u32) -> bool {
        (0x40..super::MACRO_REGISTERS_START).contains(&method)
            && !matches!(method, 0x45..=0x49)
            && Self::repeat_is_idempotent(method)
    }

    pub(crate) fn has_pending_pusher_work(&self) -> bool {
        !self.pending_draws.is_empty()
            || !self.pending_inline_upload_methods.is_empty()
            || !self.regs.pending_constbuf_writes.is_empty()
            || !self.regs.pending_semaphore_acquires.is_empty()
            || !self.regs.pending_semaphore_writes.is_empty()
            || self.regs.pending_barrier_flushes != 0
            || self.regs.pending_fragment_barriers != 0
            || self.regs.pending_tiled_cache_barriers != 0
            || self.regs.pending_texture_cache_invalidates != 0
    }

    pub fn take_top_methods(&mut self, n: usize) -> Vec<(u32, u64)> {
        if !self.method_profile_enabled {
            return Vec::new();
        }
        let mut v: Vec<(u32, u64)> = self.method_freq.drain().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    pub(crate) fn inline_upload_launch_pending(&self) -> bool {
        self.pending_inline_upload_methods
            .iter()
            .any(|&(method, _)| method == 0x6C)
    }

    pub(crate) fn has_pending_inline_uploads(&self) -> bool {
        !self.pending_inline_upload_methods.is_empty()
    }

    pub(crate) fn take_pending_inline_uploads(&mut self) -> Vec<(u32, u32)> {
        std::mem::take(&mut self.pending_inline_upload_methods)
    }

    pub(crate) fn gs_debug_regs(&self) -> GsDebugRegs {
        GsDebugRegs {
            post_vtg_masks: std::array::from_fn(|index| {
                self.reg_file
                    .get(0x490 + index)
                    .copied()
                    .unwrap_or_default()
            }),
            stage3_cbuf_binds: self.regs.cbuf_binds[3],
        }
    }

    pub fn dispatch_method(&mut self, method: u32, arg: u32, is_last: bool) {
        if method >= super::MACRO_REGISTERS_START {
            if mme_trace() && self.macro_invocations < 1024 {
                log::info!(
                    "maxwell3d: MME invoke method={:#x} arg={:#x} is_last={} (slot offset {:#x})",
                    method,
                    arg,
                    is_last,
                    method - super::MACRO_REGISTERS_START
                );
                self.macro_invocations += 1;
            }
            let reg_file_ptr = &self.reg_file as *const Vec<u32>;
            let writes =
                self.macro_engine
                    .on_macro_method(method, arg, is_last, &|idx: u32| unsafe {
                        let rf = &*reg_file_ptr;
                        rf.get(idx as usize).copied().unwrap_or(0)
                    });
            if let Some(out) = writes {
                if mme_trace() && self.macro_writes_logged < 512 {
                    log::info!(
                        "maxwell3d: MME produced {} writes inst={:?}: {:?}",
                        out.writes.len(),
                        out.draw_instance_count,
                        out.writes.iter().take(8).copied().collect::<Vec<_>>()
                    );
                    self.macro_writes_logged += 1;
                }
                self.macro_draw_instance_count = out.draw_instance_count;
                self.mme_active = true;
                self.mme_hash = out.hash;
                self.mme_entry = out.entry;
                for (m, a) in out.writes {
                    self.write_register(m, a);
                }
                self.mme_active = false;
                self.macro_draw_instance_count = None;
            }
            return;
        }

        match method {
            REG_LOAD_MME_INSTRUCTION_PTR => {
                if mme_trace() && self.macro_uploads_logged < 4 {
                    log::info!("maxwell3d: MME set_instruction_ptr = {:#x}", arg);
                }
                self.macro_engine.set_instruction_ptr(arg);
                return;
            }
            REG_LOAD_MME_INSTRUCTION => {
                if mme_trace() && self.macro_uploads_logged < 4 {
                    self.macro_uploads_logged += 1;
                    log::info!(
                        "maxwell3d: MME upload_instruction (first dword = {:#x})",
                        arg
                    );
                }
                self.macro_engine.upload_instruction(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS_PTR => {
                if mme_trace() {
                    log::info!("maxwell3d: MME set_start_address_ptr = {:#x}", arg);
                }
                self.macro_engine.set_start_address_ptr(arg);
                return;
            }
            REG_LOAD_MME_START_ADDRESS => {
                if mme_trace() {
                    log::info!("maxwell3d: MME bind_macro_entry = {:#x}", arg);
                }
                self.macro_engine.bind_macro_entry(arg);
                return;
            }
            _ => {}
        }

        self.write_register(method, arg);
    }

    pub fn write_register(&mut self, method: u32, arg: u32) {
        let incoming_arg = arg;
        let arg = if method == 0x49 {
            self.shadow_ram_control = arg & 0x3;
            arg
        } else if method < 0x80 {
            arg
        } else {
            let m = method as usize;
            match self.shadow_ram_control {
                0 | 1 => {
                    if m < self.shadow_regs.len() {
                        self.shadow_regs[m] = arg;
                    }
                    arg
                }
                3 => {
                    if m < self.shadow_regs.len() {
                        self.shadow_regs[m]
                    } else {
                        arg
                    }
                }
                _ => arg,
            }
        };

        let m = method as usize;
        let register_changed =
            m >= self.reg_file.len() || self.reg_file_written[m] == 0 || self.reg_file[m] != arg;
        if (register_changed
            || (0x60..=0x6D).contains(&method)
            || (0x8E4..=0x8F3).contains(&method))
            && !matches!(
                method,
                0x35D
                    | 0x35E
                    | 0x4C0
                    | 0x4C1
                    | 0x47D
                    | 0x57A
                    | 0x57B
                    | 0x57C
                    | 0x585
                    | 0x586
                    | 0x5F7
                    | 0x5F8
                    | 0xB2
            )
            && !(0x5F9..=0x5FE).contains(&method)
        {
            self.draw_state_dirty_since_last_draw = true;
        }

        if matches!(method, 0x35F | 0x4B3 | 0x4BA | 0x4C3) && wf_state_log() {
            log::warn!(
                "[depth-method] method={:#x} incoming={:#010x} applied={:#010x} shadow={} mme={} hash={:#018x} entry={}",
                method,
                incoming_arg,
                arg,
                self.shadow_ram_control,
                self.mme_active,
                self.mme_hash,
                self.mme_entry
            );
        }

        let addr_hi_reg = method == 0x582
            || method == 0x6c0
            || method == 0x8e1
            || method == 0x554
            || (method >= 0x200 && method < 0x280 && (method & 0xF) == 0);
        if addr_hi_reg && arg > 0xFF {
            if mme_forensics() {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 48 {
                    log::warn!(
                        "[reg-reject] method={:#x} arg={:#010x} (impossible VA hi, keeping prior)",
                        method,
                        arg
                    );
                }
            }
            return;
        }

        if (method as usize) < self.reg_file.len() {
            self.reg_file[method as usize] = arg;
            self.reg_file_written[method as usize] = 1;
        }

        if !register_changed && Self::repeat_is_idempotent(method) {
            return;
        }

        if (0x60..=0x6D).contains(&method) {
            self.pending_inline_upload_methods.push((method, arg));
        }

        if mme_forensics() {
            let hi_reg = method == 0x582
                || method == 0x6c0
                || method == 0x8e1
                || method == 0x1c
                || method == 0x554
                || (method >= 0x200 && method < 0x280 && (method & 0xF) == 0);
            let floaty = matches!(method, 0x582 | 0x583 | 0x6c0 | 0x6c1 | 0x6c2)
                && matches!(arg >> 24, 0x3E..=0x48);
            if (hi_reg && arg > 0xFF) || floaty {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 96 {
                    log::warn!(
                        "[reg-garbage] method={:#x} arg={:#010x} mme={} hash={:#018x} entry={}",
                        method,
                        arg,
                        self.mme_active,
                        self.mme_hash,
                        self.mme_entry
                    );
                }
            }
        }

        if method >= 0x200 && method < 0x280 {
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
            0x44 => {
                self.regs.pending_barrier_flushes =
                    self.regs.pending_barrier_flushes.saturating_add(1);
                trace_sync_method(method, arg, self.regs.pending_barrier_flushes);
            }
            0x378 => {
                self.regs.pending_barrier_flushes =
                    self.regs.pending_barrier_flushes.saturating_add(1);
                self.regs.pending_fragment_barriers =
                    self.regs.pending_fragment_barriers.saturating_add(1);
                trace_sync_method(method, arg, self.regs.pending_barrier_flushes);
            }
            0x3df => {
                self.regs.pending_barrier_flushes =
                    self.regs.pending_barrier_flushes.saturating_add(1);
                self.regs.pending_tiled_cache_barriers =
                    self.regs.pending_tiled_cache_barriers.saturating_add(1);
                trace_sync_method(method, arg, self.regs.pending_barrier_flushes);
            }
            0xb2 => {
                self.regs.sync_info = arg;
                crate::gpu::record_engine_syncpt_increment(arg & 0xFFF);
            }
            0x3dd => {
                self.regs.pending_barrier_flushes =
                    self.regs.pending_barrier_flushes.saturating_add(1);
                self.regs.pending_texture_cache_invalidates = self
                    .regs
                    .pending_texture_cache_invalidates
                    .saturating_add(1);
                trace_sync_method(method, arg, self.regs.pending_barrier_flushes);
            }
            0x545 => self.regs.zpass_pixel_count_enable = (arg & 1) != 0,
            0x54c => self.regs.clear_report_value = arg,
            0x8c4 => {
                if 0xD00 < self.reg_file.len() {
                    self.reg_file[0xD00] = 1;
                }
            }
            0x6c3 => {
                let operation = arg & 0x3;
                let off_hi = self.reg_file.get(0x6c0).copied().unwrap_or(0);
                let off_lo = self.reg_file.get(0x6c1).copied().unwrap_or(0);
                let payload = self.reg_file.get(0x6c2).copied().unwrap_or(0);
                let gpu_va = ((off_hi as u64) << 32) | (off_lo as u64);
                if operation == 2 {
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static N: AtomicU32 = AtomicU32::new(0);
                    let n = N.fetch_add(1, Ordering::Relaxed);
                    if n < 8 || n % 4096 == 0 {
                        log::debug!(
                            "maxwell3d: REPORT_SEMAPHORE #{} arg={:#x} op={} gpu_va={:#x} payload={:#x}",
                            n,
                            arg,
                            operation,
                            gpu_va,
                            payload
                        );
                    }
                } else {
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static N: AtomicU32 = AtomicU32::new(0);
                    if operation != 0
                        || N.fetch_add(1, Ordering::Relaxed) < 12
                        || std::env::var_os("NEXIUM_SYNCPT_DEBUG").is_some()
                    {
                        log::info!(
                            "maxwell3d: REPORT_SEMAPHORE arg={:#x} op={} gpu_va={:#x} payload={:#x}",
                            arg,
                            operation,
                            gpu_va,
                            payload
                        );
                    }
                }
                if let Some(ordering) = report_semaphore_write_ordering(arg, raw_counter_reports())
                {
                    let long = ((arg >> 28) & 1) == 0;
                    let value = if ordering == SemaphoreWriteOrdering::SyntheticCounter {
                        synthetic_counter_value()
                    } else {
                        payload
                    };
                    self.regs
                        .pending_semaphore_writes
                        .push(PendingSemaphoreWrite {
                            gpu_va,
                            payload: value,
                            long,
                            ordering,
                        });
                } else if operation == 1 {
                    let acquire_mode = (arg >> 12) & 0x7;
                    self.regs
                        .pending_semaphore_acquires
                        .push((gpu_va, payload, acquire_mode));
                }
            }
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
            0x3d5 => self.regs.stencil_back.reference = arg,
            0x3d6 => self.regs.stencil_back.write_mask = arg,
            0x3d7 => self.regs.stencil_back.compare_mask = arg,
            0x380 => self.regs.scissor.enabled = (arg & 1) != 0,
            0x381 => {
                self.regs.scissor.min_x = arg & 0xFFFF;
                self.regs.scissor.max_x = arg >> 16;
            }
            0x382 => {
                self.regs.scissor.min_y = arg & 0xFFFF;
                self.regs.scissor.max_y = arg >> 16;
            }
            0x43E => self.regs.clear_control = arg,
            0x554 => self.regs.render_enable_addr_hi = arg,
            0x555 => self.regs.render_enable_addr_lo = arg,
            0x556 => self.regs.render_enable_mode = arg,
            0x651 => self.regs.render_enable_override = arg,
            0x652 => self.regs.primitive_topology_control = arg,
            0x65C => self.regs.topology_override = arg,
            0x420 => self.regs.draw_texture_dst_x = arg,
            0x421 => self.regs.draw_texture_dst_y = arg,
            0x422 => self.regs.draw_texture_dst_width = arg,
            0x423 => self.regs.draw_texture_dst_height = arg,
            0x424 => self.regs.draw_texture_dx_du_lo = arg,
            0x425 => self.regs.draw_texture_dx_du_hi = arg,
            0x426 => self.regs.draw_texture_dy_dv_lo = arg,
            0x427 => self.regs.draw_texture_dy_dv_hi = arg,
            0x428 => self.regs.draw_texture_src_sampler = arg,
            0x429 => self.regs.draw_texture_src_texture = arg,
            0x42A => self.regs.draw_texture_src_x = arg,
            0x42B => {
                self.regs.draw_texture_src_y = arg;
                self.push_draw_texture();
            }
            0x674 => {
                self.regs.clear_count += 1;
                log::trace!(
                    "maxwell3d: CLEAR_SURFACE arg={:#x} color={:?} clear_control={:#x} scissor={:?}",
                    arg,
                    self.regs.clear_color,
                    self.regs.clear_control,
                    self.regs.scissor
                );
                self.pending_draws.push(DrawCall {
                    state_clean_from_previous: false,
                    topology: 0,
                    first_vertex: 0,
                    vertex_count: 0,
                    instance_count: 1,
                    first_instance: 0,
                    indexed: false,
                    index_count: 0,
                    index_gpu_va: 0,
                    index_format: 0,
                    index_first: 0,
                    inline_indices: Vec::new(),
                    primitive_restart_enabled: self.regs.primitive_restart_enabled,
                    primitive_restart_index: self.regs.primitive_restart_index,
                    point_size: 1.0,
                    rt: self.regs.rt,
                    rt_control: self.regs.rt_control,
                    vertex_buffers: self.regs.vertex_buffers,
                    vertex_stream_instances: self.regs.vertex_stream_instances,
                    vertex_attribs: self.regs.vertex_attribs,
                    viewport: self.regs.viewport,
                    depth_mode: self.regs.depth_mode,
                    viewport_transform_en: self.regs.viewport_transform_en,
                    viewport_clip_control: self.regs.viewport_clip_control,
                    surface_clip: self.regs.surface_clip,
                    window_origin: self.regs.window_origin,
                    scissor: self.regs.scissor,
                    clear_control: self.regs.clear_control,
                    clear_color: self.regs.clear_color,
                    color_blend: self.regs.color_blend_state(),
                    is_clear: true,
                    draw_texture: None,
                    tic_pool_gpu_va: ((self.regs.tic_pool_va_hi as u64) << 32)
                        | self.regs.tic_pool_va_lo as u64,
                    tic_pool_limit: self.regs.tic_pool_limit,
                    tsc_pool_gpu_va: ((self.regs.tsc_pool_va_hi as u64) << 32)
                        | self.regs.tsc_pool_va_lo as u64,
                    tsc_pool_limit: self.regs.tsc_pool_limit,
                    shader_programs: self.regs.shader_programs,
                    program_region_gpu_va: ((self.regs.program_region_va_hi as u64) << 32)
                        | self.regs.program_region_va_lo as u64,
                    cbuf_binds: self.regs.cbuf_binds,
                    sampler_binding: self.regs.sampler_binding,
                    bindless_texture_const_buffer_slot: self
                        .regs
                        .bindless_texture_const_buffer_slot,
                    tex_cb_index: self.regs.tex_cb_index,
                    constbuf_write_count: self.regs.pending_constbuf_writes.len(),
                    last_constbuf_addr: self.regs.last_constbuf_addr,
                    last_constbuf_size: self.regs.last_constbuf_size,
                    fs_bindless_cb_addr: self.regs.cbuf_binds
                        [self.regs.shader_programs[5].cbuf_group(4)][15]
                        .0,
                    fs_bindless_cb_size: self.regs.cbuf_binds
                        [self.regs.shader_programs[5].cbuf_group(4)][15]
                        .1,
                    fs_shader_gpu_va: {
                        let fs = &self.regs.shader_programs[5];
                        let region = ((self.regs.program_region_va_hi as u64) << 32)
                            | self.regs.program_region_va_lo as u64;
                        if fs.address_lo != 0 {
                            region.wrapping_add(fs.address_lo as u64)
                        } else {
                            0
                        }
                    },
                    render_enable_addr: ((self.regs.render_enable_addr_hi as u64) << 32)
                        | self.regs.render_enable_addr_lo as u64,
                    render_enable_mode: self.regs.render_enable_mode,
                    render_enable_override: self.regs.render_enable_override,
                    cull_test_enable: self.regs.cull_test_enable,
                    alpha_test_enabled: self.regs.alpha_test_enabled,
                    alpha_test_ref: self.regs.alpha_test_ref,
                    alpha_test_func: self.regs.alpha_test_func,
                    cull_face: self.regs.cull_face,
                    front_face: self.regs.front_face,
                    poly_offset_fill_enable: self.regs.poly_offset_fill_enable,
                    poly_offset_units: self.regs.poly_offset_units,
                    poly_offset_factor: self.regs.poly_offset_factor,
                    zeta: self.regs.zeta,
                    zeta_enable: self.regs.zeta_enable,
                    multisample_mode: self.regs.multisample_mode,
                    depth_test_enable: self.regs.depth_test_enable,
                    depth_write_enable: self.regs.depth_write_enable,
                    depth_func: self.regs.depth_func,
                    stencil_enable: self.regs.stencil_enable,
                    stencil_two_side_enable: self.regs.stencil_two_side_enable,
                    stencil_front: self.regs.stencil_front,
                    stencil_back: self.regs.stencil_back,
                    clear_depth: self.regs.clear_depth,
                    clear_stencil: self.regs.clear_stencil,
                    clear_mask: arg,
                });
            }
            0x280 => {
                let v = f32::from_bits(arg);
                self.regs.viewport.scale_x = v;
                self.regs.viewport.width = v.abs() * 2.0;
            }
            0x281 => {
                let v = f32::from_bits(arg);
                self.regs.viewport.scale_y = v;
                self.regs.viewport.height = v.abs() * 2.0;
            }
            0x282 => self.regs.viewport.scale_z = f32::from_bits(arg),
            0x283 => self.regs.viewport.translate_x = f32::from_bits(arg),
            0x284 => self.regs.viewport.translate_y = f32::from_bits(arg),
            0x285 => self.regs.viewport.translate_z = f32::from_bits(arg),
            0x286 => self.regs.viewport.swizzle = arg,
            0x3FD => {
                self.regs.surface_clip.x = arg & 0xFFFF;
                self.regs.surface_clip.width = arg >> 16;
            }
            0x3FE => {
                self.regs.surface_clip.y = arg & 0xFFFF;
                self.regs.surface_clip.height = arg >> 16;
            }
            0x4EB => self.regs.window_origin.raw = arg,
            0x4e0 => self.regs.stencil_enable = (arg & 1) != 0,
            0x4e1 => self.regs.stencil_front.fail_op = arg,
            0x4e2 => self.regs.stencil_front.depth_fail_op = arg,
            0x4e3 => self.regs.stencil_front.depth_pass_op = arg,
            0x4e4 => self.regs.stencil_front.compare_op = arg,
            0x4e5 => self.regs.stencil_front.reference = arg,
            0x4e6 => self.regs.stencil_front.compare_mask = arg,
            0x4e7 => self.regs.stencil_front.write_mask = arg,
            0x565 => self.regs.stencil_two_side_enable = (arg & 1) != 0,
            0x566 => self.regs.stencil_back.fail_op = arg,
            0x567 => self.regs.stencil_back.depth_fail_op = arg,
            0x568 => self.regs.stencil_back.depth_pass_op = arg,
            0x569 => self.regs.stencil_back.compare_op = arg,
            0x64B => {
                self.regs.viewport_transform_en = arg & 1 != 0;
                if viewport_enable_debug() {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    let n = N.fetch_add(1, Ordering::Relaxed);
                    if n < 200 {
                        log::warn!(
                            "[vp-en] #{} arg={} draws={}",
                            n,
                            arg & 1,
                            self.regs.draw_count
                        );
                    }
                }
            }
            0x64F => self.regs.viewport_clip_control.raw = arg,
            0x50D => self.regs.global_base_vertex_index = arg,
            0x50E => self.regs.global_base_instance_index = arg,
            0x35D => self.regs.draw_first_vertex = arg,
            0x35E => {
                self.regs.draw_vertex_count = arg;
                if arg > 0 {
                    if self.legacy_draw_begin_pending && !self.mme_active {
                        self.legacy_draw_vertex_pending = true;
                    } else {
                        self.regs.draw_count += 1;
                        let legacy_instance_id = self.take_legacy_draw_instance_id();
                        self.push_draw(
                            self.regs.draw_topology,
                            self.regs.draw_first_vertex,
                            arg,
                            false,
                            0,
                            1,
                            self.regs.global_base_instance_index,
                            legacy_instance_id,
                            Vec::new(),
                        );
                    }
                }
            }
            0x35F => self.regs.depth_mode = arg & 1,
            0x485 | 0x486 => {
                self.regs.draw_count += 1;
                let first = arg & 0xFFFF;
                let count = (arg >> 16) & 0xFFF;
                let topology = (arg >> 28) & 0xF;
                if method == 0x485 || self.regs.vertex_array_instance_count == 0 {
                    self.regs.vertex_array_instance_count = 1;
                }
                let first_instance = self.regs.vertex_array_instance_count - 1;
                self.regs.vertex_array_instance_count =
                    self.regs.vertex_array_instance_count.wrapping_add(1);
                log::trace!(
                    "maxwell3d: DRAW_VERTEX_ARRAY_BEGIN_END first={} count={} topology={} first_instance={}",
                    first,
                    count,
                    topology,
                    first_instance
                );
                self.push_draw(
                    topology,
                    first,
                    count,
                    false,
                    0,
                    1,
                    first_instance,
                    None,
                    Vec::new(),
                );
            }

            0x586 => {
                self.regs.draw_topology = arg & 0xFFFF;
                self.legacy_draw_instance_id = (arg >> 26) & 0x3;
                self.legacy_draw_begin_pending = true;
                self.legacy_draw_vertex_pending = false;
                self.legacy_draw_index_pending = false;
                self.inline_indices.clear();
                self.inline_u8_setup = None;
                self.inline_u16_setup = None;
            }
            0x585 => {
                let inline_indices = std::mem::take(&mut self.inline_indices);
                let inline_index_count = inline_indices.len().min(u32::MAX as usize) as u32;
                let draw = if inline_index_count > 0 {
                    Some((
                        self.regs.global_base_vertex_index,
                        0,
                        true,
                        inline_index_count,
                        inline_indices,
                    ))
                } else if self.legacy_draw_index_pending && self.regs.index_count > 0 {
                    Some((
                        self.regs.global_base_vertex_index,
                        0,
                        true,
                        self.regs.index_count,
                        Vec::new(),
                    ))
                } else if self.legacy_draw_vertex_pending && self.regs.draw_vertex_count > 0 {
                    Some((
                        self.regs.draw_first_vertex,
                        self.regs.draw_vertex_count,
                        false,
                        0,
                        Vec::new(),
                    ))
                } else {
                    None
                };
                if let Some((first, count, indexed, index_count, inline_indices)) = draw {
                    self.regs.draw_count += 1;
                    let legacy_instance_id = self.take_legacy_draw_instance_id();
                    self.push_draw(
                        self.regs.draw_topology,
                        first,
                        count,
                        indexed,
                        index_count,
                        1,
                        self.regs.global_base_instance_index,
                        legacy_instance_id,
                        inline_indices,
                    );
                } else {
                    self.legacy_draw_begin_pending = false;
                }
                self.legacy_draw_vertex_pending = false;
                self.legacy_draw_index_pending = false;
                self.inline_u8_setup = None;
                self.inline_u16_setup = None;
            }
            0x4C0 => {
                let count = (arg & 0x3FFF_FFFF) as usize;
                self.inline_u8_setup = (count != 0).then_some((((arg >> 30) & 3) as usize, count));
            }
            0x4C1 => self.push_packed_inline_indices(arg, 8),
            0x57A => self.inline_indices.push(arg),
            0x57B => {
                let count = (arg & 0x7FFF_FFFF) as usize;
                self.inline_u16_setup = (count != 0).then_some((((arg >> 31) & 1) as usize, count));
            }
            0x57C => self.push_packed_inline_indices(arg, 16),
            0x591 => self.regs.primitive_restart_enabled = (arg & 1) != 0,
            0x592 => self.regs.primitive_restart_index = arg,
            0x5F2 => self.regs.index_buffer_hi = arg,
            0x5F3 => self.regs.index_buffer_lo = arg,
            0x5F4 => self.regs.index_buffer_end_hi = arg,
            0x5F5 => self.regs.index_buffer_end_lo = arg,
            0x5F6 => self.regs.index_format = arg,
            0x5F7 => self.regs.index_first = arg,
            0x5F8 => {
                self.regs.index_count = arg;
                log::trace!(
                    "maxwell3d: DrawElementsCount count={} topology={}",
                    arg,
                    self.regs.draw_topology
                );
                if arg > 0 {
                    if self.legacy_draw_begin_pending && !self.mme_active {
                        self.legacy_draw_index_pending = true;
                    } else {
                        self.regs.draw_count += 1;
                        let legacy_instance_id = self.take_legacy_draw_instance_id();
                        self.push_draw(
                            self.regs.draw_topology,
                            self.regs.global_base_vertex_index,
                            0,
                            true,
                            arg,
                            1,
                            self.regs.global_base_instance_index,
                            legacy_instance_id,
                            Vec::new(),
                        );
                    }
                }
            }
            0x5F9..=0x5FE => {
                let first_index = arg & 0xFFFF;
                let index_count = (arg >> 16) & 0xFFF;
                let topology = (arg >> 28) & 0xF;
                let instance_id = u32::from(method >= 0x5FC);
                let saved_index_first = self.regs.index_first;
                self.regs.index_first = first_index;
                self.regs.draw_count += 1;
                self.push_draw(
                    topology,
                    self.regs.global_base_vertex_index,
                    0,
                    true,
                    index_count,
                    1,
                    self.regs.global_base_instance_index,
                    Some(instance_id),
                    Vec::new(),
                );
                self.regs.index_first = saved_index_first;
            }

            0x557 => {
                log::trace!("maxwell3d: SetTexSamplerPool[hi] = {:#x}", arg);
                self.regs.tsc_pool_va_hi = arg;
            }
            0x558 => {
                log::trace!(
                    "maxwell3d: SetTexSamplerPool[lo] = {:#x} → full {:#x}",
                    arg,
                    ((self.regs.tsc_pool_va_hi as u64) << 32) | arg as u64
                );
                self.regs.tsc_pool_va_lo = arg;
            }
            0x559 => {
                log::trace!("maxwell3d: SetTexSamplerPoolMaximumIndex = {:#x}", arg);
                self.regs.tsc_pool_limit = arg;
            }

            0x55D => {
                log::trace!("maxwell3d: SetTexHeaderPool[hi] = {:#x}", arg);
                self.regs.tic_pool_va_hi = arg;
            }
            0x55E => {
                log::trace!(
                    "maxwell3d: SetTexHeaderPool[lo] = {:#x} → full {:#x}",
                    arg,
                    ((self.regs.tic_pool_va_hi as u64) << 32) | arg as u64
                );
                self.regs.tic_pool_va_lo = arg;
            }
            0x55F => {
                log::trace!("maxwell3d: SetTexHeaderPoolMaximumIndex = {:#x}", arg);
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
            0x8E4..=0x8F3 => {
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
                let slot = ((arg >> 4) & 0x1F) as usize;
                let cb_addr = ((self.regs.constbuf_selector_addr_hi as u64) << 32)
                    | self.regs.constbuf_selector_addr_lo as u64;
                let cb_size = self.regs.constbuf_selector_size;
                if cbuf_bind_trace() {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static N: AtomicU64 = AtomicU64::new(0);
                    static N1136: AtomicU64 = AtomicU64::new(0);
                    let n = N.fetch_add(1, Ordering::Relaxed);
                    let hot = cb_size == 1136 || cb_size == 80 || slot >= 16;
                    let hot_n = if hot {
                        N1136.fetch_add(1, Ordering::Relaxed)
                    } else {
                        0
                    };
                    if n < 2048 || (hot && hot_n < 512) || n % 65536 == 0 {
                        log::warn!(
                            "[cbuf-bind] #{} stage={} slot={} valid={} addr={:#x} size={} mme={}",
                            n,
                            stage,
                            slot,
                            valid as u8,
                            cb_addr,
                            cb_size,
                            self.mme_active
                        );
                    }
                }
                if stage < 5 && slot < GRAPHICS_CBUF_SLOTS {
                    self.regs.cbuf_binds[stage][slot] =
                        if valid { (cb_addr, cb_size) } else { (0, 0) };
                }
            }
            0x48D => self.regs.sampler_binding = arg,
            0x982 => {
                let slot = arg & 0x1F;
                self.regs.tex_cb_index = slot;
                self.regs.bindless_texture_const_buffer_slot = slot;
            }
            0x4BB => self.regs.alpha_test_enabled = (arg & 1) != 0,
            0x4C4 => self.regs.alpha_test_ref = arg,
            0x4C5 => self.regs.alpha_test_func = arg,
            0x574 => {
                if self.regs.multisample_mode != arg {
                    log::info!(
                        "maxwell3d: multisample mode {:#x} -> {:#x}",
                        self.regs.multisample_mode,
                        arg
                    );
                }
                self.regs.multisample_mode = arg;
            }
            0x646 => self.regs.cull_test_enable = (arg & 1) != 0,
            0x647 => self.regs.front_face = arg,
            0x648 => self.regs.cull_face = arg,

            0x372 => self.regs.poly_offset_fill_enable = (arg & 1) != 0,
            0x55B => self.regs.poly_offset_factor = f32::from_bits(arg),
            0x56F => self.regs.poly_offset_units = f32::from_bits(arg),
            0x3F8 => self.regs.zeta.address_hi = arg,
            0x3F9 => self.regs.zeta.address_lo = arg,
            0x3FA => self.regs.zeta.format = arg & 0x1F,
            0x3FB => self.regs.zeta.block_size = arg,
            0x3FC => self.regs.zeta.array_pitch = arg,
            0x48A => self.regs.zeta.width = arg & 0x0FFF_FFFF,
            0x48B => self.regs.zeta.height = arg & 0x0001_FFFF,
            0x54E => self.regs.zeta_enable = (arg & 1) != 0,
            0x4B3 => {
                let new = (arg & 1) != 0;
                if new != self.regs.depth_test_enable && wf_state_log() {
                    log::warn!(
                        "[wf-state] depth_test {} -> {}",
                        self.regs.depth_test_enable,
                        new
                    );
                }
                self.regs.depth_test_enable = new;
            }
            0x4BA => {
                let new = (arg & 1) != 0;
                if new != self.regs.depth_write_enable && wf_state_log() {
                    log::warn!(
                        "[wf-state] depth_write {} -> {}",
                        self.regs.depth_write_enable,
                        new
                    );
                }
                self.regs.depth_write_enable = new;
            }
            0x4C3 => {
                if arg != self.regs.depth_func && wf_state_log() {
                    log::warn!(
                        "[wf-state] depth_func {:#x} -> {:#x}",
                        self.regs.depth_func,
                        arg
                    );
                }
                self.regs.depth_func = arg;
            }
            0x4C7..=0x4CA => {
                self.regs.blend_constants[(method - 0x4C7) as usize] = f32::from_bits(arg);
            }
            0x4D0 => self.regs.blend_eq_rgb = arg,
            0x4D1 => self.regs.blend_src_rgb = arg,
            0x4D2 => self.regs.blend_dst_rgb = arg,
            0x4D3 => self.regs.blend_eq_alpha = arg,
            0x4D4 => self.regs.blend_src_alpha = arg,
            0x4D6 => self.regs.blend_dst_alpha = arg,
            0x487 => self.regs.rt_control = arg,
            0x4B9 => self.regs.blend_per_target_enabled = (arg & 1) != 0,
            0x3E4 => self.regs.color_mask_common = (arg & 1) != 0,
            0x680..=0x687 => {
                let rt = (method - 0x680) as usize;
                if rt < 8 {
                    self.regs.color_masks[rt] = arg;
                }
            }
            0x4D8..=0x4DF => {
                let rt = (method - 0x4D8) as usize;
                if rt < 8 {
                    self.regs.blend_enable[rt] = (arg & 1) != 0;
                }
            }
            0x780..=0x7BF => {
                let rt = ((method - 0x780) / 8) as usize;
                let field = (method - 0x780) % 8;
                if rt < 8 {
                    match field {
                        1 => self.regs.blend_pt_eq_rgb[rt] = arg,
                        2 => self.regs.blend_pt_src_rgb[rt] = arg,
                        3 => self.regs.blend_pt_dst_rgb[rt] = arg,
                        4 => self.regs.blend_pt_eq_alpha[rt] = arg,
                        5 => self.regs.blend_pt_src_alpha[rt] = arg,
                        6 => self.regs.blend_pt_dst_alpha[rt] = arg,
                        _ => {}
                    }
                }
            }
            0x620..=0x63F => {
                let idx = (method - 0x620) as usize;
                if idx < 32 {
                    self.regs.vertex_stream_instances[idx] = arg;
                }
            }
            0x700..=0x77F => {
                let idx = ((method - 0x700) / 4) as usize;
                let field = (method - 0x700) % 4;
                if idx < 32 {
                    let vb = &mut self.regs.vertex_buffers[idx];
                    match field {
                        0 => {
                            vb.stride = arg & 0xFFF;
                            vb.enabled = arg & 0x1000 != 0;
                        }
                        1 => vb.address_hi = arg,
                        2 => vb.address_lo = arg,
                        3 => vb.frequency = arg,
                        _ => {}
                    }
                }
            }
            0x7C0..=0x7FF => {
                let idx = ((method - 0x7C0) / 2) as usize;
                let field = (method - 0x7C0) % 2;
                if idx < 32 {
                    let vb = &mut self.regs.vertex_buffers[idx];
                    match field {
                        0 => vb.end_hi = arg,
                        1 => vb.end_lo = arg,
                        _ => {}
                    }
                }
            }
            0x458..=0x477 => {
                let idx = (method - 0x458) as usize;
                if idx < 32 {
                    let va = &mut self.regs.vertex_attribs[idx];
                    va.buffer = arg & 0x1F;
                    va.constant = (arg >> 6) & 1 != 0;
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
                log::trace!(
                    "maxwell3d: SetProgramRegion[lo] = {:#x} → full {:#x}",
                    arg,
                    ((self.regs.program_region_va_hi as u64) << 32) | arg as u64
                );
                self.regs.program_region_va_lo = arg;
            }

            0x800..=0x85F => {
                let idx = ((method - 0x800) / 0x10) as usize;
                if idx < 6 {
                    let sp = &mut self.regs.shader_programs[idx];
                    let field = (method - 0x800) % 0x10;
                    log::trace!(
                        "maxwell3d: SetProgram[stage={}] field={} = {:#x}",
                        idx,
                        field,
                        arg
                    );
                    match field {
                        0 => sp.enabled = (arg & 1) != 0,

                        1 => sp.address_lo = arg,
                        3 => sp.gpr_count = arg,
                        4 => sp.binding_group = Some(arg & 0x7),
                        _ => {}
                    }
                }
            }
            _ => {
                log::trace!("maxwell3d: write method {:#x} = {:#x}", method, arg);
            }
        }
    }

    fn repeat_is_idempotent(method: u32) -> bool {
        !matches!(
            method,
            0x44 | 0x378
                | 0x3DD
                | 0x3DF
                | 0x35D
                | 0x35E
                | 0x42B
                | 0x47D
                | 0x485
                | 0x486
                | 0x4C0
                | 0x4C1
                | 0x57A
                | 0x57B
                | 0x57C
                | 0x585
                | 0x586
                | 0x5F7
                | 0x5F8
                | 0x674
                | 0x6C3
                | 0x8C4
                | 0x8E2
                | 0x8E3
                | 0x904
                | 0x90C
                | 0x914
                | 0x91C
                | 0x924
                | 0xB2
        ) && !(0x60..=0x6D).contains(&method)
            && !(0x8E4..=0x8F3).contains(&method)
            && !(0x5F9..=0x5FE).contains(&method)
    }

    fn effective_topology(&self, draw_topology: u32) -> u32 {
        if self.regs.primitive_topology_control != 1 {
            return draw_topology;
        }
        match self.regs.topology_override {
            0 => draw_topology,
            1 | 0x1001 => 0,
            2 | 0x1002 | 0x100F | 0x1018 | 0x101B => 1,
            3 | 0x1010 | 0x1011 => 3,
            4 | 0x1003 | 0x1012 | 0x101A => 4,
            5 | 0x1013 | 0x1014 => 5,
            0x1015 | 0x1016 | 0x1017 => 6,
            topology @ (0xA..=0xE) => topology,
            _ => draw_topology,
        }
    }

    fn push_draw(
        &mut self,
        topology: u32,
        first: u32,
        count: u32,
        indexed: bool,
        index_count: u32,
        instance_count: u32,
        first_instance: u32,
        legacy_instance_id: Option<u32>,
        inline_indices: Vec<u32>,
    ) {
        let topology = self.effective_topology(topology);
        let instance_count = self
            .macro_draw_instance_count
            .take()
            .unwrap_or(instance_count)
            .max(1);
        let index_gpu_va =
            ((self.regs.index_buffer_hi as u64) << 32) | self.regs.index_buffer_lo as u64;

        let is_first_or_subsequent = matches!(legacy_instance_id, Some(0 | 1));
        let state_clean_from_previous = !self.draw_state_dirty_since_last_draw;
        let can_coalesce = legacy_instance_id == Some(1)
            && self.last_draw_allows_continuation
            && state_clean_from_previous;
        self.last_draw_allows_continuation = is_first_or_subsequent;
        self.draw_state_dirty_since_last_draw = false;

        if can_coalesce {
            if let Some(previous) = self.pending_draws.last_mut() {
                let same_draw = !previous.is_clear
                    && previous.draw_texture.is_none()
                    && previous.topology == topology
                    && previous.first_vertex == first
                    && previous.vertex_count == count
                    && previous.first_instance == first_instance
                    && previous.indexed == indexed
                    && previous.index_count == index_count
                    && previous.index_gpu_va == index_gpu_va
                    && previous.index_format == self.regs.index_format
                    && previous.index_first == self.regs.index_first
                    && previous.inline_indices == inline_indices;
                if same_draw {
                    previous.instance_count =
                        previous.instance_count.saturating_add(instance_count);
                    return;
                }
            }
        }

        let tic_pool_gpu_va =
            ((self.regs.tic_pool_va_hi as u64) << 32) | self.regs.tic_pool_va_lo as u64;
        let tsc_pool_gpu_va =
            ((self.regs.tsc_pool_va_hi as u64) << 32) | self.regs.tsc_pool_va_lo as u64;
        let fs = &self.regs.shader_programs[5];
        let (fs_bindless_cb_addr, fs_bindless_cb_size) = self.regs.cbuf_binds[fs.cbuf_group(4)][15];

        let program_region =
            ((self.regs.program_region_va_hi as u64) << 32) | self.regs.program_region_va_lo as u64;
        let fs_shader_gpu_va = if fs.address_lo != 0 {
            program_region.wrapping_add(fs.address_lo as u64)
        } else {
            0
        };
        let ps = f32::from_bits(self.reg_file.get(0x546).copied().unwrap_or(0));
        let point_size = if ps.is_finite() && ps > 0.0 { ps } else { 1.0 };
        self.pending_draws.push(DrawCall {
            topology,
            first_vertex: first,
            vertex_count: count,
            instance_count,
            first_instance,
            indexed,
            index_count,
            index_gpu_va,
            index_format: self.regs.index_format,
            index_first: self.regs.index_first,
            inline_indices,
            state_clean_from_previous,
            primitive_restart_enabled: self.regs.primitive_restart_enabled,
            primitive_restart_index: self.regs.primitive_restart_index,
            point_size,
            rt: self.regs.rt,
            rt_control: self.regs.rt_control,
            vertex_buffers: self.regs.vertex_buffers,
            vertex_stream_instances: self.regs.vertex_stream_instances,
            vertex_attribs: self.regs.vertex_attribs,
            viewport: self.regs.viewport,
            depth_mode: self.regs.depth_mode,
            viewport_transform_en: self.regs.viewport_transform_en,
            viewport_clip_control: self.regs.viewport_clip_control,
            surface_clip: self.regs.surface_clip,
            window_origin: self.regs.window_origin,
            scissor: self.regs.scissor,
            clear_control: self.regs.clear_control,
            clear_color: self.regs.clear_color,
            color_blend: self.regs.color_blend_state(),
            is_clear: false,
            draw_texture: None,
            tic_pool_gpu_va,
            tic_pool_limit: self.regs.tic_pool_limit,
            tsc_pool_gpu_va,
            tsc_pool_limit: self.regs.tsc_pool_limit,
            shader_programs: self.regs.shader_programs,
            program_region_gpu_va: program_region,
            cbuf_binds: self.regs.cbuf_binds,
            sampler_binding: self.regs.sampler_binding,
            bindless_texture_const_buffer_slot: self.regs.bindless_texture_const_buffer_slot,
            tex_cb_index: self.regs.tex_cb_index,
            constbuf_write_count: self.regs.pending_constbuf_writes.len(),
            last_constbuf_addr: self.regs.last_constbuf_addr,
            last_constbuf_size: self.regs.last_constbuf_size,
            fs_bindless_cb_addr,
            fs_bindless_cb_size,
            fs_shader_gpu_va,
            render_enable_addr: ((self.regs.render_enable_addr_hi as u64) << 32)
                | self.regs.render_enable_addr_lo as u64,
            render_enable_mode: self.regs.render_enable_mode,
            render_enable_override: self.regs.render_enable_override,
            cull_test_enable: self.regs.cull_test_enable,
            alpha_test_enabled: self.regs.alpha_test_enabled,
            alpha_test_ref: self.regs.alpha_test_ref,
            alpha_test_func: self.regs.alpha_test_func,
            cull_face: self.regs.cull_face,
            front_face: self.regs.front_face,
            poly_offset_fill_enable: self.regs.poly_offset_fill_enable,
            poly_offset_units: self.regs.poly_offset_units,
            poly_offset_factor: self.regs.poly_offset_factor,
            zeta: self.regs.zeta,
            zeta_enable: self.regs.zeta_enable,
            multisample_mode: self.regs.multisample_mode,
            depth_test_enable: self.regs.depth_test_enable,
            depth_write_enable: self.regs.depth_write_enable,
            depth_func: self.regs.depth_func,
            stencil_enable: self.regs.stencil_enable,
            stencil_two_side_enable: self.regs.stencil_two_side_enable,
            stencil_front: self.regs.stencil_front,
            stencil_back: self.regs.stencil_back,
            clear_depth: self.regs.clear_depth,
            clear_stencil: self.regs.clear_stencil,
            clear_mask: 0,
        });
    }

    fn take_legacy_draw_instance_id(&mut self) -> Option<u32> {
        if !std::mem::replace(&mut self.legacy_draw_begin_pending, false) {
            return None;
        }
        Some(self.legacy_draw_instance_id)
    }

    fn push_packed_inline_indices(&mut self, arg: u32, bits: u32) {
        let slots = (32 / bits) as usize;
        let mask = (1u32 << bits) - 1;
        let setup = if bits == 8 {
            &mut self.inline_u8_setup
        } else {
            &mut self.inline_u16_setup
        };
        let constrained = setup.is_some();
        let (skip, limit) = setup.as_ref().copied().unwrap_or((0, slots));
        let mut pushed = 0usize;
        for slot in skip.min(slots)..slots {
            if constrained && pushed >= limit {
                break;
            }
            self.inline_indices
                .push((arg >> (slot as u32 * bits)) & mask);
            pushed += 1;
        }
        if let Some((skip, remaining)) = setup.as_mut() {
            *skip = 0;
            *remaining = remaining.saturating_sub(pushed);
        }
    }

    fn push_draw_texture(&mut self) {
        let fixed_20_12 = |v: u32| (v as i32) as f32 / 4096.0;
        let fixed_32_32 = |lo: u32, hi: u32| {
            let raw = ((hi as u64) << 32) | lo as u64;
            (raw as i64) as f64 / 4_294_967_296.0
        };
        let dst_x = fixed_20_12(self.regs.draw_texture_dst_x);
        let mut dst_y = fixed_20_12(self.regs.draw_texture_dst_y);
        let dst_width = fixed_20_12(self.regs.draw_texture_dst_width);
        let dst_height = fixed_20_12(self.regs.draw_texture_dst_height);
        if self.regs.window_origin.lower_left() {
            let clip = self
                .regs
                .surface_clip
                .effective(self.regs.rt[0].width, self.regs.rt[0].height);
            dst_y = clip.height as f32 - dst_y;
        }
        let src_x = fixed_20_12(self.regs.draw_texture_src_x);
        let src_y = fixed_20_12(self.regs.draw_texture_src_y);
        let src_width = (fixed_32_32(
            self.regs.draw_texture_dx_du_lo,
            self.regs.draw_texture_dx_du_hi,
        ) as f32)
            * dst_width;
        let src_height = (fixed_32_32(
            self.regs.draw_texture_dy_dv_lo,
            self.regs.draw_texture_dy_dv_hi,
        ) as f32)
            * dst_height;
        self.regs.draw_texture_count = self.regs.draw_texture_count.wrapping_add(1);
        self.regs.draw_count = self.regs.draw_count.wrapping_add(1);
        if self.regs.draw_texture_count <= 8 {
            log::info!(
                "maxwell3d: DrawTexture[{}] dst=({},{} {}x{}) src=({},{} {}x{}) tex={} samp={}",
                self.regs.draw_texture_count - 1,
                dst_x,
                dst_y,
                dst_width,
                dst_height,
                src_x,
                src_y,
                src_width,
                src_height,
                self.regs.draw_texture_src_texture,
                self.regs.draw_texture_src_sampler
            );
        }
        self.pending_draws.push(DrawCall {
            state_clean_from_previous: false,
            topology: 0,
            first_vertex: 0,
            vertex_count: 0,
            instance_count: 1,
            first_instance: 0,
            indexed: false,
            index_count: 0,
            index_gpu_va: 0,
            index_format: 0,
            index_first: 0,
            inline_indices: Vec::new(),
            primitive_restart_enabled: self.regs.primitive_restart_enabled,
            primitive_restart_index: self.regs.primitive_restart_index,
            point_size: 1.0,
            rt: self.regs.rt,
            rt_control: self.regs.rt_control,
            vertex_buffers: self.regs.vertex_buffers,
            vertex_stream_instances: self.regs.vertex_stream_instances,
            vertex_attribs: self.regs.vertex_attribs,
            viewport: self.regs.viewport,
            depth_mode: self.regs.depth_mode,
            viewport_transform_en: self.regs.viewport_transform_en,
            viewport_clip_control: self.regs.viewport_clip_control,
            surface_clip: self.regs.surface_clip,
            window_origin: self.regs.window_origin,
            scissor: self.regs.scissor,
            clear_control: self.regs.clear_control,
            clear_color: self.regs.clear_color,
            color_blend: self.regs.color_blend_state(),
            is_clear: false,
            draw_texture: Some(DrawTextureCall {
                dst_x,
                dst_y,
                dst_width,
                dst_height,
                src_x,
                src_y,
                src_width,
                src_height,
                texture_id: self.regs.draw_texture_src_texture,
                sampler_id: self.regs.draw_texture_src_sampler,
                tic_pool_gpu_va: ((self.regs.tic_pool_va_hi as u64) << 32)
                    | self.regs.tic_pool_va_lo as u64,
                tic_pool_limit: self.regs.tic_pool_limit,
            }),
            tic_pool_gpu_va: ((self.regs.tic_pool_va_hi as u64) << 32)
                | self.regs.tic_pool_va_lo as u64,
            tic_pool_limit: self.regs.tic_pool_limit,
            tsc_pool_gpu_va: ((self.regs.tsc_pool_va_hi as u64) << 32)
                | self.regs.tsc_pool_va_lo as u64,
            tsc_pool_limit: self.regs.tsc_pool_limit,
            shader_programs: self.regs.shader_programs,
            program_region_gpu_va: ((self.regs.program_region_va_hi as u64) << 32)
                | self.regs.program_region_va_lo as u64,
            cbuf_binds: self.regs.cbuf_binds,
            sampler_binding: self.regs.sampler_binding,
            bindless_texture_const_buffer_slot: self.regs.bindless_texture_const_buffer_slot,
            tex_cb_index: self.regs.tex_cb_index,
            constbuf_write_count: self.regs.pending_constbuf_writes.len(),
            last_constbuf_addr: self.regs.last_constbuf_addr,
            last_constbuf_size: self.regs.last_constbuf_size,
            fs_bindless_cb_addr: self.regs.cbuf_binds[self.regs.shader_programs[5].cbuf_group(4)]
                [15]
            .0,
            fs_bindless_cb_size: self.regs.cbuf_binds[self.regs.shader_programs[5].cbuf_group(4)]
                [15]
            .1,
            fs_shader_gpu_va: {
                let fs = &self.regs.shader_programs[5];
                let region = ((self.regs.program_region_va_hi as u64) << 32)
                    | self.regs.program_region_va_lo as u64;
                if fs.address_lo != 0 {
                    region.wrapping_add(fs.address_lo as u64)
                } else {
                    0
                }
            },
            render_enable_addr: ((self.regs.render_enable_addr_hi as u64) << 32)
                | self.regs.render_enable_addr_lo as u64,
            render_enable_mode: self.regs.render_enable_mode,
            render_enable_override: self.regs.render_enable_override,
            cull_test_enable: self.regs.cull_test_enable,
            alpha_test_enabled: self.regs.alpha_test_enabled,
            alpha_test_ref: self.regs.alpha_test_ref,
            alpha_test_func: self.regs.alpha_test_func,
            cull_face: self.regs.cull_face,
            front_face: self.regs.front_face,
            poly_offset_fill_enable: self.regs.poly_offset_fill_enable,
            poly_offset_units: self.regs.poly_offset_units,
            poly_offset_factor: self.regs.poly_offset_factor,
            zeta: self.regs.zeta,
            zeta_enable: self.regs.zeta_enable,
            multisample_mode: self.regs.multisample_mode,
            depth_test_enable: self.regs.depth_test_enable,
            depth_write_enable: self.regs.depth_write_enable,
            depth_func: self.regs.depth_func,
            stencil_enable: self.regs.stencil_enable,
            stencil_two_side_enable: self.regs.stencil_two_side_enable,
            stencil_front: self.regs.stencil_front,
            stencil_back: self.regs.stencil_back,
            clear_depth: self.regs.clear_depth,
            clear_stencil: self.regs.clear_stencil,
            clear_mask: 0,
        });
    }

    pub fn render_target(&self, idx: usize) -> Option<&RenderTarget> {
        const MAX_RT_DIMENSION: u32 = 32768;
        let rt = self.regs.rt.get(idx)?;
        if rt.width == 0 || rt.height == 0 {
            return None;
        }
        if rt.width > MAX_RT_DIMENSION || rt.height > MAX_RT_DIMENSION {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n < 16 || n % 4096 == 0 {
                log::warn!(
                    "maxwell3d: rejecting render target {} with implausible size {}x{} (n={})",
                    idx,
                    rt.width,
                    rt.height,
                    n + 1
                );
            }
            return None;
        }
        Some(rt)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuMappings;

    #[test]
    fn synthetic_counter_report_does_not_require_renderer_completion() {
        let synthetic = report_semaphore_write_ordering(2, false).unwrap();
        let release = report_semaphore_write_ordering(0, false).unwrap();
        let raw_counter = report_semaphore_write_ordering(2, true).unwrap();

        assert_eq!(synthetic, SemaphoreWriteOrdering::SyntheticCounter);
        assert_eq!(release, SemaphoreWriteOrdering::RendererOrdered);
        assert_eq!(raw_counter, SemaphoreWriteOrdering::RendererOrdered);
        assert!(report_semaphore_write_ordering(1, false).is_none());

        let write = PendingSemaphoreWrite {
            gpu_va: 0x1234,
            payload: 0x5678,
            long: true,
            ordering: synthetic,
        };
        assert!(!write.requires_renderer_completion());
        assert!(PendingSemaphoreWrite {
            ordering: release,
            ..write
        }
        .requires_renderer_completion());
    }

    #[test]
    fn inline_upload_drain_reports_each_terminal_write_in_order() {
        let mut mappings = GpuMappings::new();
        mappings.add(0x4000, 0x100, 0x1000, 1);
        let mut engine = Maxwell3D::new();
        engine.pending_inline_upload_methods = vec![
            (0x60, 4),
            (0x61, 1),
            (0x63, 0x4000),
            (0x6c, 1),
            (0x6d, 0x1122_3344),
            (0x60, 4),
            (0x61, 1),
            (0x63, 0x4010),
            (0x6c, 1),
            (0x6d, 0x5566_7788),
        ];
        let read = |_: u64, _: &mut [u8]| true;
        let write = |_: u64, _: &[u8]| true;
        let mut outcomes = Vec::new();

        let methods = engine.take_pending_inline_uploads();
        assert!(engine.take_pending_inline_uploads().is_empty());
        let mut upload = KeplerMemory::new();
        for (method, arg) in methods {
            let outcome = upload.dispatch_method(method, arg, &mappings, &read, &write);
            if !matches!(outcome, KeplerMemoryWriteOutcome::NoWrite) {
                outcomes.push(outcome);
            }
        }

        assert_eq!(
            outcomes,
            vec![
                KeplerMemoryWriteOutcome::Exact(vec![(0x4000, 4)]),
                KeplerMemoryWriteOutcome::Exact(vec![(0x4010, 4)]),
            ]
        );
    }

    #[test]
    fn only_plain_short_payload_release_uses_async_fence_path() {
        const SMO_PAYLOAD_FENCE: u32 = 0x1000_f010;
        assert!(is_short_payload_fence(SMO_PAYLOAD_FENCE));
        assert_eq!(
            report_semaphore_write_ordering(SMO_PAYLOAD_FENCE, false),
            Some(SemaphoreWriteOrdering::PayloadFence)
        );
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE & !(1 << 28)));
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE | 2));
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE | (1 << 23)));
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE | (1 << 3)));
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE | (1 << 5)));
        assert!(!is_short_payload_fence(SMO_PAYLOAD_FENCE | (1 << 21)));
    }

    #[test]
    fn pusher_passive_method_filter_excludes_all_work_emitting_methods() {
        for method in [0x200, 0x280, 0x458, 0x700, 0x800, 0x8e0, 0x8e1] {
            assert!(Maxwell3D::is_pusher_passive_method(method), "{method:#x}");
        }

        for method in [
            0x3f,
            0x44,
            0x45,
            0x46,
            0x47,
            0x48,
            0x49,
            0x35d,
            0x35e,
            0x378,
            0x3dd,
            0x3df,
            0x42b,
            0x47d,
            0x485,
            0x486,
            0x4c0,
            0x4c1,
            0x57a,
            0x57b,
            0x57c,
            0x585,
            0x586,
            0x5f7,
            0x5f8,
            0x674,
            0x6c3,
            0x8c4,
            0x8e2,
            0x8e3,
            0x904,
            0x90c,
            0x914,
            0x91c,
            0x924,
            0xb2,
            crate::gpu::engines::MACRO_REGISTERS_START,
        ] {
            assert!(!Maxwell3D::is_pusher_passive_method(method), "{method:#x}");
        }
        for method in 0x60..=0x6d {
            assert!(!Maxwell3D::is_pusher_passive_method(method), "{method:#x}");
        }
        for method in 0x5f9..=0x5fe {
            assert!(!Maxwell3D::is_pusher_passive_method(method), "{method:#x}");
        }
        for method in 0x8e4..=0x8f3 {
            assert!(!Maxwell3D::is_pusher_passive_method(method), "{method:#x}");
        }
    }

    #[test]
    fn force_heavy_method_sync_is_state_only() {
        let mut engine = Maxwell3D::new();
        engine.draw_state_dirty_since_last_draw = false;
        engine.dispatch_method(0x47d, 1, true);
        assert_eq!(engine.regs.pending_barrier_flushes, 0);
        assert_eq!(engine.reg_file[0x47d], 1);
        assert!(!engine.draw_state_dirty_since_last_draw);

        engine.dispatch_method(0xb2, 0x110001, true);
        assert_eq!(engine.regs.sync_info, 0x110001);
        assert_eq!(engine.regs.pending_barrier_flushes, 0);
        assert!(!engine.draw_state_dirty_since_last_draw);

        engine.dispatch_method(0x378, 1, true);
        assert_eq!(engine.regs.pending_barrier_flushes, 1);
        assert_eq!(engine.regs.pending_fragment_barriers, 1);
        assert_eq!(engine.regs.pending_tiled_cache_barriers, 0);
        assert_eq!(engine.regs.pending_texture_cache_invalidates, 0);

        engine.dispatch_method(0x3df, 1, true);
        assert_eq!(engine.regs.pending_barrier_flushes, 2);
        assert_eq!(engine.regs.pending_fragment_barriers, 1);
        assert_eq!(engine.regs.pending_tiled_cache_barriers, 1);

        engine.dispatch_method(0x3dd, 1, true);
        assert_eq!(engine.regs.pending_barrier_flushes, 3);
        assert_eq!(engine.regs.pending_texture_cache_invalidates, 1);
        assert!(engine.draw_state_dirty_since_last_draw);
    }

    #[test]
    fn face_state_methods_decode_in_order() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x646, 1, true);
        engine.dispatch_method(0x647, 0x900, true);
        engine.dispatch_method(0x648, 0x405, true);
        assert!(engine.regs.cull_test_enable);
        assert_eq!(engine.regs.front_face, 0x900);
        assert_eq!(engine.regs.cull_face, 0x405);
    }

    #[test]
    fn depth_register_defaults_match_maxwell_reset_state() {
        let regs = Maxwell3DRegisters::default();
        assert!(!regs.depth_write_enable);
        assert_eq!(regs.clear_depth, 0.0);
        assert_eq!(regs.depth_mode, 0);
    }

    #[test]
    fn shadow_ram_starts_from_the_encoded_live_reset_state() {
        let engine = Maxwell3D::new();

        assert_eq!(engine.shadow_regs, engine.reg_file);
        assert_eq!(engine.reg_file[0x286], engine.regs.viewport.swizzle);
        assert_eq!(engine.reg_file[0x4c3], engine.regs.depth_func);
        assert_eq!(engine.reg_file[0x4d0], engine.regs.blend_eq_rgb);
        assert_eq!(
            engine.reg_file[0x4e6],
            engine.regs.stencil_front.compare_mask
        );
        assert_eq!(engine.reg_file[0x64b], 1);
        for rt in 0..8 {
            assert_eq!(engine.reg_file[0x680 + rt], engine.regs.color_masks[rt]);
            assert_eq!(
                engine.reg_file[0x780 + rt * 8 + 1],
                engine.regs.blend_pt_eq_rgb[rt]
            );
            assert_eq!(
                engine.reg_file[0x780 + rt * 8 + 6],
                engine.regs.blend_pt_dst_alpha[rt]
            );
        }
    }

    #[test]
    fn replay_before_track_preserves_viewport_and_fixed_function_defaults() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x49, 3, true);
        for method in [0x286, 0x4c3, 0x4d0, 0x4e6, 0x64b, 0x680, 0x781] {
            engine.dispatch_method(method, 0, true);
        }

        assert_eq!(engine.regs.viewport.swizzle, 0x6420);
        assert_eq!(engine.reg_file[0x286], 0x6420);
        assert_eq!(engine.regs.depth_func, 0x207);
        assert_eq!(engine.regs.blend_eq_rgb, 0x8006);
        assert_eq!(engine.regs.stencil_front.compare_mask, u32::MAX);
        assert!(engine.regs.viewport_transform_en);
        assert_eq!(engine.regs.color_masks[0], 0x1111);
        assert_eq!(engine.regs.blend_pt_eq_rgb[0], 0x8006);
    }

    #[test]
    fn low_upload_methods_bypass_shadow_tracking_and_replay() {
        let mut engine = Maxwell3D::new();
        engine.shadow_regs[0x60] = 0xdead_0060;
        engine.shadow_regs[0x6d] = 0xdead_006d;

        engine.dispatch_method(0x60, 0x1111_0060, true);
        engine.dispatch_method(0x6d, 0x1111_006d, true);
        assert_eq!(engine.shadow_regs[0x60], 0xdead_0060);
        assert_eq!(engine.shadow_regs[0x6d], 0xdead_006d);
        assert_eq!(
            engine.take_pending_inline_uploads(),
            vec![(0x60, 0x1111_0060), (0x6d, 0x1111_006d)]
        );

        engine.dispatch_method(0x49, 3, true);
        engine.dispatch_method(0x60, 0x2222_0060, true);
        engine.dispatch_method(0x6d, 0x2222_006d, true);
        assert_eq!(engine.reg_file[0x60], 0x2222_0060);
        assert_eq!(engine.reg_file[0x6d], 0x2222_006d);
        assert_eq!(
            engine.take_pending_inline_uploads(),
            vec![(0x60, 0x2222_0060), (0x6d, 0x2222_006d)]
        );
    }

    #[test]
    fn high_methods_replay_with_masked_shadow_control() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x4c3, 0x203, true);

        let raw_control = 0xffff_fff3;
        engine.dispatch_method(0x49, raw_control, true);
        assert_eq!(engine.shadow_ram_control, 3);
        assert_eq!(engine.reg_file[0x49], raw_control);

        engine.dispatch_method(0x4c3, 0x207, true);
        assert_eq!(engine.reg_file[0x4c3], 0x203);
        assert_eq!(engine.regs.depth_func, 0x203);
    }

    #[test]
    fn track_updates_live_and_shadow_register_images() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x286, 0x6430, true);
        engine.dispatch_method(0x4d0, 0x800a, true);
        engine.dispatch_method(0x680, 0x1011, true);

        for method in [0x286, 0x4d0, 0x680] {
            assert_eq!(
                engine.shadow_regs[method as usize],
                engine.reg_file[method as usize]
            );
        }
        assert_eq!(engine.regs.viewport.swizzle, 0x6430);
        assert_eq!(engine.regs.blend_eq_rgb, 0x800a);
        assert_eq!(engine.regs.color_masks[0], 0x1011);
    }

    #[test]
    fn constant_buffer_disable_clears_stale_binding() {
        let mut engine = Maxwell3D::new();
        let slot = 17usize;

        engine.dispatch_method(0x8e0, 0x630, true);
        engine.dispatch_method(0x8e1, 0x12, true);
        engine.dispatch_method(0x8e2, 0x3456_7000, true);
        engine.dispatch_method(0x904, ((slot as u32) << 4) | 1, true);
        assert_eq!(engine.regs.cbuf_binds[0][slot], (0x12_3456_7000, 0x630));

        engine.dispatch_method(0x904, (slot as u32) << 4, true);
        assert_eq!(engine.regs.cbuf_binds[0][slot], (0, 0));
    }

    #[test]
    fn shader_program_captures_binding_group() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x814, 0xffff_fffbu32, true);

        assert_eq!(engine.regs.shader_programs[1].binding_group, Some(3));
    }

    #[test]
    fn draws_snapshot_shader_and_constant_buffer_state() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x582, 1, true);
        engine.dispatch_method(0x583, 0x8000, true);
        engine.dispatch_method(0x810, 1, true);
        engine.dispatch_method(0x811, 0x40, true);
        engine.dispatch_method(0x8e0, 0x100, true);
        engine.dispatch_method(0x8e1, 0, true);
        engine.dispatch_method(0x8e2, 0x1000, true);
        engine.dispatch_method(0x904, 1, true);
        engine.dispatch_method(0x48d, 1, true);
        engine.dispatch_method(0x982, 3, true);
        engine.dispatch_method(0x8e3, 0, true);
        engine.dispatch_method(0x8e4, 0x1111_1111, true);
        engine.dispatch_method(0x35e, 3, true);

        engine.dispatch_method(0x583, 0x9000, true);
        engine.dispatch_method(0x811, 0x80, true);
        engine.dispatch_method(0x8e0, 0x200, true);
        engine.dispatch_method(0x8e2, 0x2000, true);
        engine.dispatch_method(0x904, 1, true);
        engine.dispatch_method(0x48d, 0, true);
        engine.dispatch_method(0x982, 4, true);
        engine.dispatch_method(0x8e3, 0, true);
        engine.dispatch_method(0x8e4, 0x2222_2222, true);
        engine.dispatch_method(0x35e, 3, true);

        let first = &engine.pending_draws[0];
        assert_eq!(first.shader_programs[1].address_lo, 0x40);
        assert_eq!(first.program_region_gpu_va, 0x1_0000_8000);
        assert_eq!(first.cbuf_binds[0][0], (0x1000, 0x100));
        assert_eq!(first.sampler_binding, 1);
        assert_eq!(first.bindless_texture_const_buffer_slot, 3);
        assert_eq!(first.tex_cb_index, 3);
        assert_eq!(first.constbuf_write_count, 1);

        let second = &engine.pending_draws[1];
        assert_eq!(second.shader_programs[1].address_lo, 0x80);
        assert_eq!(second.program_region_gpu_va, 0x1_0000_9000);
        assert_eq!(second.cbuf_binds[0][0], (0x2000, 0x200));
        assert_eq!(second.sampler_binding, 0);
        assert_eq!(second.bindless_texture_const_buffer_slot, 4);
        assert_eq!(second.tex_cb_index, 4);
        assert_eq!(second.constbuf_write_count, 2);
    }

    #[test]
    fn draws_snapshot_blend_and_color_mask_state() {
        let mut engine = Maxwell3D::new();
        let first_constants = [0x3e80_0001, 0x3f00_0002, 0x3f40_0003, 0x3f80_0004];
        let second_constants = [0x4000_0001, 0x4020_0002, 0x4040_0003, 0x4060_0004];

        engine.dispatch_method(0x4b9, 1, true);
        engine.dispatch_method(0x4d8, 1, true);
        engine.dispatch_method(0x782, 0x0302, true);
        engine.dispatch_method(0x783, 0x0303, true);
        engine.dispatch_method(0x680, 0x0111, true);
        for (index, bits) in first_constants.into_iter().enumerate() {
            engine.dispatch_method(0x4c7 + index as u32, bits, true);
        }
        engine.dispatch_method(0x35e, 3, true);

        engine.dispatch_method(0x4b9, 0, true);
        engine.dispatch_method(0x4d8, 0, true);
        engine.dispatch_method(0x4d1, 0x0304, true);
        engine.dispatch_method(0x4d2, 0x0305, true);
        engine.dispatch_method(0x680, 0x1110, true);
        for (index, bits) in second_constants.into_iter().enumerate() {
            engine.dispatch_method(0x4c7 + index as u32, bits, true);
        }
        engine.dispatch_method(0x35e, 3, true);

        let first = engine.pending_draws[0].color_blend;
        assert!(first.blend_per_target_enabled);
        assert!(first.blend_enable[0]);
        assert_eq!(first.blend_pt_src_rgb[0], 0x0302);
        assert_eq!(first.blend_pt_dst_rgb[0], 0x0303);
        assert_eq!(first.color_masks[0], 0x0111);
        assert_eq!(first.blend_constants.map(f32::to_bits), first_constants);

        let second = engine.pending_draws[1].color_blend;
        assert!(!second.blend_per_target_enabled);
        assert!(!second.blend_enable[0]);
        assert_eq!(second.blend_src_rgb, 0x0304);
        assert_eq!(second.blend_dst_rgb, 0x0305);
        assert_eq!(second.color_masks[0], 0x1110);
        assert_eq!(second.blend_constants.map(f32::to_bits), second_constants);
    }

    #[test]
    fn vertex_stream_format_preserves_enable_bit() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x700, 0x40, true);
        assert_eq!(engine.regs.vertex_buffers[0].stride, 0x40);
        assert!(!engine.regs.vertex_buffers[0].enabled);

        engine.dispatch_method(0x700, 0x1000 | 0x20, true);
        assert_eq!(engine.regs.vertex_buffers[0].stride, 0x20);
        assert!(engine.regs.vertex_buffers[0].enabled);
    }

    #[test]
    fn depth_mode_does_not_alias_first_vertex() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x35d, 17, true);
        engine.dispatch_method(0x35f, 1, true);
        assert_eq!(engine.regs.draw_first_vertex, 17);
        assert_eq!(engine.regs.depth_mode, 1);
    }

    #[test]
    fn topology_override_methods_match_maxwell_encodings() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x65c, 4, true);
        assert_eq!(engine.effective_topology(5), 5);

        engine.dispatch_method(0x652, 1, true);
        for (override_raw, expected) in [
            (0, 5),
            (1, 0),
            (2, 1),
            (3, 3),
            (4, 4),
            (5, 5),
            (0xa, 0xa),
            (0xe, 0xe),
            (0x1001, 0),
            (0x1002, 1),
            (0x1003, 4),
            (0x1010, 3),
            (0x1015, 6),
            (0x1017, 6),
            (0x1018, 1),
            (0x101a, 4),
            (0x101b, 1),
        ] {
            engine.dispatch_method(0x65c, override_raw, true);
            assert_eq!(engine.effective_topology(5), expected, "{override_raw:#x}");
        }
    }

    #[test]
    fn topology_override_is_snapshotted_by_all_draw_entry_paths() {
        let configure_override = |engine: &mut Maxwell3D| {
            engine.dispatch_method(0x652, 1, true);
            engine.dispatch_method(0x65c, 4, true);
        };

        let mut ordinary = Maxwell3D::new();
        configure_override(&mut ordinary);
        ordinary.dispatch_method(0x35e, 3, true);
        assert_eq!(ordinary.pending_draws[0].topology, 4);

        let mut legacy = Maxwell3D::new();
        configure_override(&mut legacy);
        legacy.dispatch_method(0x586, 5, true);
        legacy.dispatch_method(0x35d, 2, true);
        legacy.dispatch_method(0x35e, 3, true);
        legacy.dispatch_method(0x585, 0, true);
        assert_eq!(legacy.pending_draws[0].topology, 4);

        let mut packed_vertex = Maxwell3D::new();
        configure_override(&mut packed_vertex);
        packed_vertex.dispatch_method(0x485, (5 << 28) | (3 << 16) | 2, true);
        assert_eq!(packed_vertex.pending_draws[0].topology, 4);

        let mut packed_index = Maxwell3D::new();
        configure_override(&mut packed_index);
        packed_index.dispatch_method(0x5f3, 0x1234_0000, true);
        packed_index.dispatch_method(0x5f9, (5 << 28) | (3 << 16) | 2, true);
        assert_eq!(packed_index.pending_draws[0].topology, 4);

        let mut inline_index = Maxwell3D::new();
        configure_override(&mut inline_index);
        inline_index.dispatch_method(0x586, 5, true);
        inline_index.dispatch_method(0x57a, 0, true);
        inline_index.dispatch_method(0x57a, 1, true);
        inline_index.dispatch_method(0x57a, 2, true);
        inline_index.dispatch_method(0x585, 0, true);
        assert_eq!(inline_index.pending_draws[0].topology, 4);
    }

    #[test]
    fn primitive_restart_registers_are_snapshotted_by_draws() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x591, 1, true);
        engine.dispatch_method(0x592, 0x1234_5678, true);
        engine.dispatch_method(0x586, 5, true);
        engine.dispatch_method(0x5f8, 3, true);
        engine.dispatch_method(0x585, 0, true);

        assert!(engine.regs.primitive_restart_enabled);
        assert_eq!(engine.regs.primitive_restart_index, 0x1234_5678);
        let draw = engine.pending_draws.last().unwrap();
        assert!(draw.primitive_restart_enabled);
        assert_eq!(draw.primitive_restart_index, 0x1234_5678);
    }

    #[test]
    fn stencil_state_methods_decode_yuzu_register_layout() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x4e0, 1, true);
        engine.dispatch_method(0x4e1, 0x1e01, true);
        engine.dispatch_method(0x4e2, 0x1e02, true);
        engine.dispatch_method(0x4e3, 0x8507, true);
        engine.dispatch_method(0x4e4, 0x206, true);
        engine.dispatch_method(0x4e5, 0x44, true);
        engine.dispatch_method(0x4e6, 0x7f, true);
        engine.dispatch_method(0x4e7, 0x3f, true);
        engine.dispatch_method(0x565, 1, true);
        engine.dispatch_method(0x566, 3, true);
        engine.dispatch_method(0x567, 4, true);
        engine.dispatch_method(0x568, 5, true);
        engine.dispatch_method(0x569, 6, true);
        engine.dispatch_method(0x3d5, 0x55, true);
        engine.dispatch_method(0x3d6, 0xaa, true);
        engine.dispatch_method(0x3d7, 0xf0, true);

        assert!(engine.regs.stencil_enable);
        assert!(engine.regs.stencil_two_side_enable);
        assert_eq!(engine.regs.stencil_front.fail_op, 0x1e01);
        assert_eq!(engine.regs.stencil_front.depth_fail_op, 0x1e02);
        assert_eq!(engine.regs.stencil_front.depth_pass_op, 0x8507);
        assert_eq!(engine.regs.stencil_front.compare_op, 0x206);
        assert_eq!(engine.regs.stencil_front.reference, 0x44);
        assert_eq!(engine.regs.stencil_front.compare_mask, 0x7f);
        assert_eq!(engine.regs.stencil_front.write_mask, 0x3f);
        assert_eq!(engine.regs.stencil_back.fail_op, 3);
        assert_eq!(engine.regs.stencil_back.depth_fail_op, 4);
        assert_eq!(engine.regs.stencil_back.depth_pass_op, 5);
        assert_eq!(engine.regs.stencil_back.compare_op, 6);
        assert_eq!(engine.regs.stencil_back.reference, 0x55);
        assert_eq!(engine.regs.stencil_back.write_mask, 0xaa);
        assert_eq!(engine.regs.stencil_back.compare_mask, 0xf0);

        engine.dispatch_method(0x35e, 3, true);
        let draw = engine.pending_draws.last().unwrap();
        assert!(draw.stencil_enable);
        assert!(draw.stencil_two_side_enable);
        assert_eq!(draw.stencil_front, engine.regs.stencil_front);
        assert_eq!(draw.stencil_back, engine.regs.stencil_back);
    }

    #[test]
    fn clear_snapshots_independent_stencil_value_and_mask() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x368, 0x6d, true);
        engine.dispatch_method(0x674, 0x2, true);
        let clear = engine.pending_draws.last().unwrap();
        assert!(clear.is_clear);
        assert_eq!(clear.clear_stencil, 0x6d);
        assert_eq!(clear.clear_mask, 0x2);
    }

    #[test]
    fn normal_draws_remain_separate_without_instance_continuation() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x35e, 6, true);

        assert_eq!(engine.pending_draws.len(), 2);
        assert_eq!(engine.pending_draws[0].instance_count, 1);
        assert_eq!(engine.pending_draws[1].instance_count, 1);
    }

    #[test]
    fn inline_index_methods_emit_one_indexed_draw_at_end() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x586, 4, true);
        engine.dispatch_method(0x35d, 0, true);
        engine.dispatch_method(0x35e, 114, true);
        engine.dispatch_method(0x57a, 7, true);
        engine.dispatch_method(0x57c, (11 << 16) | 9, true);
        assert!(engine.pending_draws.is_empty());
        engine.dispatch_method(0x585, 0, true);

        assert_eq!(engine.pending_draws.len(), 1);
        let draw = &engine.pending_draws[0];
        assert!(draw.indexed);
        assert_eq!(draw.vertex_count, 0);
        assert_eq!(draw.index_count, 3);
        assert_eq!(draw.inline_indices, [7, 9, 11]);
    }

    #[test]
    fn packed_inline_index_setup_applies_offset_and_count() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x586, 4, true);
        engine.dispatch_method(0x57b, (1 << 31) | 3, true);
        engine.dispatch_method(0x57c, (2 << 16) | 1, true);
        engine.dispatch_method(0x57c, (4 << 16) | 3, true);
        engine.dispatch_method(0x57c, (6 << 16) | 5, true);
        engine.dispatch_method(0x57b, 0, true);
        engine.dispatch_method(0x57c, (8 << 16) | 7, true);
        engine.dispatch_method(0x585, 0, true);

        let draw = &engine.pending_draws[0];
        assert_eq!(draw.inline_indices, [2, 3, 4, 7, 8]);
    }

    #[test]
    fn legacy_nonindexed_continuations_accumulate_instances() {
        let mut engine = Maxwell3D::new();

        for instance_id in [0, 1, 1] {
            engine.dispatch_method(0x586, 5 | (instance_id << 26), true);
            engine.dispatch_method(0x35d, 4, true);
            engine.dispatch_method(0x35e, 6, true);
            engine.dispatch_method(0x585, 0, true);
        }

        assert_eq!(engine.pending_draws.len(), 1);
        let draw = &engine.pending_draws[0];
        assert!(!draw.indexed);
        assert_eq!(draw.topology, 5);
        assert_eq!(draw.first_vertex, 4);
        assert_eq!(draw.vertex_count, 6);
        assert_eq!(draw.instance_count, 3);
    }

    #[test]
    fn legacy_indexed_continuations_accumulate_instances() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x5f2, 0, true);
        engine.dispatch_method(0x5f3, 0x1234_0000, true);
        engine.dispatch_method(0x5f6, 2, true);

        for instance_id in [0, 1] {
            engine.dispatch_method(0x586, 4 | (instance_id << 26), true);
            engine.dispatch_method(0x5f7, 7, true);
            engine.dispatch_method(0x5f8, 12, true);
            engine.dispatch_method(0x585, 0, true);
        }

        assert_eq!(engine.pending_draws.len(), 1);
        let draw = &engine.pending_draws[0];
        assert!(draw.indexed);
        assert_eq!(draw.topology, 4);
        assert_eq!(draw.index_first, 7);
        assert_eq!(draw.index_count, 12);
        assert_eq!(draw.index_gpu_va, 0x1234_0000);
        assert_eq!(draw.index_format, 2);
        assert_eq!(draw.instance_count, 2);
    }

    #[test]
    fn packed_indexed_draw_methods_decode_current_index_state() {
        for method in 0x5f9..=0x5fe {
            let mut engine = Maxwell3D::new();
            engine.dispatch_method(0x5f2, 0x4, true);
            engine.dispatch_method(0x5f3, 0x1234_0000, true);
            engine.dispatch_method(0x5f6, 2, true);
            engine.dispatch_method(0x5f7, 9, true);
            engine.dispatch_method(0x50d, 0xffff_fff9, true);
            engine.dispatch_method(0x50e, 11, true);

            engine.dispatch_method(method, (5 << 28) | (0x345 << 16) | 0x4567, true);

            assert_eq!(engine.pending_draws.len(), 1, "method {method:#x}");
            let draw = &engine.pending_draws[0];
            assert!(draw.indexed, "method {method:#x}");
            assert_eq!(draw.topology, 5, "method {method:#x}");
            assert_eq!(draw.index_first, 0x4567, "method {method:#x}");
            assert_eq!(draw.index_count, 0x345, "method {method:#x}");
            assert_eq!(draw.index_gpu_va, 0x4_1234_0000, "method {method:#x}");
            assert_eq!(draw.index_format, 2, "method {method:#x}");
            assert_eq!(draw.first_vertex, 0xffff_fff9, "method {method:#x}");
            assert_eq!(draw.first_instance, 11, "method {method:#x}");
            assert_eq!(draw.instance_count, 1, "method {method:#x}");
            assert_eq!(engine.regs.index_first, 9, "method {method:#x}");
        }
    }

    #[test]
    fn packed_indexed_subsequent_methods_accumulate_instances() {
        let mut engine = Maxwell3D::new();
        engine.dispatch_method(0x5f3, 0x1234_0000, true);
        engine.dispatch_method(0x5f6, 2, true);
        engine.dispatch_method(0x50d, 7, true);
        engine.dispatch_method(0x50e, 3, true);
        let argument = (4 << 28) | (12 << 16) | 5;

        engine.dispatch_method(0x5fa, argument, true);
        engine.dispatch_method(0x5fd, argument, true);
        engine.dispatch_method(0x5fd, argument, true);

        assert_eq!(engine.pending_draws.len(), 1);
        let draw = &engine.pending_draws[0];
        assert!(draw.indexed);
        assert_eq!(draw.topology, 4);
        assert_eq!(draw.index_first, 5);
        assert_eq!(draw.index_count, 12);
        assert_eq!(draw.index_format, 2);
        assert_eq!(draw.first_vertex, 7);
        assert_eq!(draw.first_instance, 3);
        assert_eq!(draw.instance_count, 3);
    }

    #[test]
    fn legacy_continuation_stops_at_state_change() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x586, 5, true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        engine.dispatch_method(0x35f, 1, true);
        engine.dispatch_method(0x586, 5 | (1 << 26), true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        assert_eq!(engine.pending_draws.len(), 2);
        assert_eq!(engine.pending_draws[0].instance_count, 1);
        assert_eq!(engine.pending_draws[0].depth_mode, 0);
        assert_eq!(engine.pending_draws[1].instance_count, 1);
        assert_eq!(engine.pending_draws[1].depth_mode, 1);
    }

    #[test]
    fn legacy_continuation_survives_redundant_state_write() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x35f, 1, true);
        engine.dispatch_method(0x586, 5, true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        engine.dispatch_method(0x35f, 1, true);
        engine.dispatch_method(0x586, 5 | (1 << 26), true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        assert_eq!(engine.pending_draws.len(), 1);
        assert_eq!(engine.pending_draws[0].instance_count, 2);
        assert_eq!(engine.pending_draws[0].depth_mode, 1);
    }

    #[test]
    fn first_zero_register_write_after_draw_is_a_state_change() {
        let mut engine = Maxwell3D::new();

        engine.dispatch_method(0x586, 5, true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        engine.dispatch_method(0x4c3, 0, true);
        engine.dispatch_method(0x586, 5 | (1 << 26), true);
        engine.dispatch_method(0x35d, 4, true);
        engine.dispatch_method(0x35e, 6, true);
        engine.dispatch_method(0x585, 0, true);

        assert_eq!(engine.pending_draws.len(), 2);
        assert_eq!(engine.pending_draws[0].depth_func, 0x207);
        assert_eq!(engine.pending_draws[1].depth_func, 0);
    }
}

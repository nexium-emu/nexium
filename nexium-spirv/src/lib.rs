#![deny(unsafe_op_in_unsafe_fn)]

mod opt;

use std::collections::HashMap;

use nexium_shader::{
    BasicBlock, BlockId, BoolOp, BranchKind, Cfg, FComp, ICmp, IrInst, IrOp, IrValue, LogicOp,
    MufuFunc, ValueId,
};
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel, FunctionControl, ImageFormat,
    MemoryModel, StorageClass, Word,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
}

fn slot_is_gl_position(aligned_slot: u32) -> bool {
    matches!(aligned_slot, 0x70 | 0x1c0)
}

const UBO_VEC4S: u32 = 4096;
const CBUF_LOGICAL_SLOTS: u32 = 32;
const CBUF_SLOT_VEC4S: u32 = UBO_VEC4S / CBUF_LOGICAL_SLOTS;
const MAX_TEXTURE_DESCRIPTORS: u32 = 32;
const SSBO_BINDING_BASE: u32 = 3;
pub const MAX_SSBO: u32 = 8;

pub struct Emitter {
    b: rspirv::dr::Builder,
    stage: Stage,
    f32_t: Word,
    vec2_t: Word,
    vec3_t: Word,
    vec4_t: Word,
    u32_t: Word,
    i32_t: Word,
    ptr_uniform_f32: Word,
    ptr_input_vec4: Word,
    ptr_input_f32: Word,
    ptr_input_u32: Word,
    ptr_output_vec4: Word,
    ptr_output_f32: Word,
    ptr_image: Word,
    ptr_image_arrayed: Word,
    ptr_sampler: Word,
    ptr_image_array: Word,
    ptr_image_arrayed_array: Word,
    ptr_sampler_array: Word,
    image_3d_t: Word,
    sampled_image_3d_t: Word,
    ptr_image_3d: Word,
    ptr_image_3d_array: Word,
    image_3d_var: Option<Word>,
    ubo_var: Word,
    f32_zero: Word,
    f32_one: Word,
    glsl: Word,
    input_vars: HashMap<u32, AttrVar>,
    output_vars: HashMap<u32, AttrVar>,
    pos_var: Option<Word>,
    point_size_var: Option<Word>,
    frag_coord_var: Option<Word>,
    frag_color_vars: HashMap<u32, Word>,
    vertex_index_var: Option<Word>,
    instance_index_var: Option<Word>,
    image_var: Option<Word>,
    sampler_var: Option<Word>,
    image_t: Word,
    image_arrayed_t: Word,
    sampler_t: Word,
    sampled_image_t: Word,
    sampled_image_arrayed_t: Word,
    interface: Vec<Word>,
    value_to_word: HashMap<ValueId, Word>,
    block_labels: HashMap<BlockId, Word>,
    cond_merge: Option<Vec<u32>>,
    used_merge_blocks: std::collections::HashSet<Word>,
    shared_merge_headers: HashMap<u32, Vec<u32>>,
    header_merge_label: HashMap<u32, Word>,
    synth_merge_blocks: HashMap<u32, Vec<(Word, u32, Word)>>,
    synth_phi_results: HashMap<(Word, ValueId), Word>,
    synth_pred_phi_results: HashMap<(Word, ValueId), Word>,
    merge_redirects: HashMap<(u32, u32), Word>,
    block_end_labels: HashMap<BlockId, Word>,
    self_loops: HashMap<BlockId, (Word, Word, Word, BlockId)>,
    loop_carried: HashMap<BlockId, Vec<(ValueId, Word, Word)>>,
    current_block: Option<BlockId>,
    cbuf_bindings_used: u32,
    texs_ids_used: std::collections::BTreeSet<u32>,
    texture_slots: HashMap<u32, u32>,
    vertex_opts: VertexOptions,
    const_cache_f32: HashMap<u32, Word>,
    const_cache_u32: HashMap<u32, Word>,
    bool_t: Word,
    bool_true: Word,
    bool_false: Word,
    pred_regs: [Option<Word>; 7],
    ubo_vec4s: u32,
    ssbo_vars: Vec<Option<Word>>,
    ptr_storage_u32: Option<Word>,
    return_block: Option<Word>,
    sample_debug_slot: Option<u32>,
    sample_debug_component: Option<u32>,
    sample_debug_value: Option<Word>,
    texcoord_debug_slot: Option<u32>,
    tex_v_flip_slots: Vec<u32>,
    sampler_arrayed: bool,
    no_kil_shader: bool,
    fragment_color_outputs: u32,
    fragment_output_map: u32,
    ps_input_map: [u8; 32],
}

#[derive(Clone, Copy)]
struct AttrVar {
    var: Word,
    ptr_f32: Word,
    is_uint: bool,
    is_sint: bool,
}

impl Emitter {
    pub fn new(stage: Stage) -> Self {
        Self::new_sized(stage, UBO_VEC4S)
    }

    fn new_sized(stage: Stage, ubo_vec4s: u32) -> Self {
        let mut b = rspirv::dr::Builder::new();
        b.set_version(1, 0);
        b.capability(Capability::Shader);
        let glsl = b.ext_inst_import("GLSL.std.450");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let f32_t = b.type_float(32);
        let vec2_t = b.type_vector(f32_t, 2);
        let vec3_t = b.type_vector(f32_t, 3);
        let vec4_t = b.type_vector(f32_t, 4);
        let u32_t = b.type_int(32, 0);
        let i32_t = b.type_int(32, 1);
        let ptr_uniform_f32 = b.type_pointer(None, StorageClass::Uniform, f32_t);
        let ptr_input_vec4 = b.type_pointer(None, StorageClass::Input, vec4_t);
        let ptr_input_f32 = b.type_pointer(None, StorageClass::Input, f32_t);
        let ptr_input_u32 = b.type_pointer(None, StorageClass::Input, u32_t);
        let ptr_output_vec4 = b.type_pointer(None, StorageClass::Output, vec4_t);
        let ptr_output_f32 = b.type_pointer(None, StorageClass::Output, f32_t);

        let ubo_vec4s_const = b.constant_bit32(u32_t, ubo_vec4s);
        let vec4_arr = b.type_array(vec4_t, ubo_vec4s_const);
        b.decorate(
            vec4_arr,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(16)],
        );
        let ubo_struct = b.type_struct([vec4_arr]);
        b.decorate(ubo_struct, Decoration::Block, []);
        b.member_decorate(
            ubo_struct,
            0,
            Decoration::Offset,
            [Operand::LiteralBit32(0)],
        );
        let ptr_uniform_struct = b.type_pointer(None, StorageClass::Uniform, ubo_struct);
        let ubo_var = b.variable(ptr_uniform_struct, None, StorageClass::Uniform, None);
        b.decorate(
            ubo_var,
            Decoration::DescriptorSet,
            [Operand::LiteralBit32(0)],
        );
        b.decorate(ubo_var, Decoration::Binding, [Operand::LiteralBit32(0)]);

        let image_t = b.type_image(
            f32_t,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let image_arrayed_t = b.type_image(
            f32_t,
            rspirv::spirv::Dim::Dim2D,
            0,
            1,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let sampler_t = b.type_sampler();
        let sampled_image_t = b.type_sampled_image(image_t);
        let sampled_image_arrayed_t = b.type_sampled_image(image_arrayed_t);
        let ptr_image = b.type_pointer(None, StorageClass::UniformConstant, image_t);
        let ptr_image_arrayed =
            b.type_pointer(None, StorageClass::UniformConstant, image_arrayed_t);
        let ptr_sampler = b.type_pointer(None, StorageClass::UniformConstant, sampler_t);
        let texture_slots_const = b.constant_bit32(u32_t, MAX_TEXTURE_DESCRIPTORS);
        let image_array_t = b.type_array(image_t, texture_slots_const);
        let image_arrayed_array_t = b.type_array(image_arrayed_t, texture_slots_const);
        let sampler_array_t = b.type_array(sampler_t, texture_slots_const);
        let ptr_image_array = b.type_pointer(None, StorageClass::UniformConstant, image_array_t);
        let ptr_image_arrayed_array =
            b.type_pointer(None, StorageClass::UniformConstant, image_arrayed_array_t);
        let ptr_sampler_array =
            b.type_pointer(None, StorageClass::UniformConstant, sampler_array_t);
        let image_3d_t = b.type_image(
            f32_t,
            rspirv::spirv::Dim::Dim3D,
            0,
            0,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let sampled_image_3d_t = b.type_sampled_image(image_3d_t);
        let ptr_image_3d = b.type_pointer(None, StorageClass::UniformConstant, image_3d_t);
        let image_3d_array_t = b.type_array(image_3d_t, texture_slots_const);
        let ptr_image_3d_array =
            b.type_pointer(None, StorageClass::UniformConstant, image_3d_array_t);

        let f32_zero = b.constant_bit32(f32_t, 0.0f32.to_bits());
        let f32_one = b.constant_bit32(f32_t, 1.0f32.to_bits());

        let mut const_cache_f32: HashMap<u32, Word> = HashMap::new();
        let const_cache_u32: HashMap<u32, Word> = HashMap::new();
        const_cache_f32.insert(0.0f32.to_bits(), f32_zero);
        const_cache_f32.insert(1.0f32.to_bits(), f32_one);

        let bool_t = b.type_bool();
        let bool_true = b.constant_true(bool_t);
        let bool_false = b.constant_false(bool_t);

        Self {
            b,
            stage,
            f32_t,
            vec2_t,
            vec3_t,
            vec4_t,
            u32_t,
            i32_t,
            ptr_uniform_f32,
            ptr_input_vec4,
            ptr_input_f32,
            ptr_input_u32,
            ptr_output_vec4,
            ptr_output_f32,
            ptr_image,
            ptr_image_arrayed,
            ptr_sampler,
            ptr_image_array,
            ptr_image_arrayed_array,
            ptr_sampler_array,
            image_3d_t,
            sampled_image_3d_t,
            ptr_image_3d,
            ptr_image_3d_array,
            image_3d_var: None,
            ubo_var,
            f32_zero,
            f32_one,
            glsl,
            input_vars: HashMap::new(),
            output_vars: HashMap::new(),
            pos_var: None,
            point_size_var: None,
            frag_coord_var: None,
            frag_color_vars: HashMap::new(),
            vertex_index_var: None,
            instance_index_var: None,
            image_var: None,
            sampler_var: None,
            image_t,
            image_arrayed_t,
            sampler_t,
            sampled_image_t,
            sampled_image_arrayed_t,
            interface: Vec::new(),
            value_to_word: HashMap::new(),
            block_labels: HashMap::new(),
            cond_merge: None,
            used_merge_blocks: std::collections::HashSet::new(),
            shared_merge_headers: HashMap::new(),
            header_merge_label: HashMap::new(),
            synth_merge_blocks: HashMap::new(),
            synth_phi_results: HashMap::new(),
            synth_pred_phi_results: HashMap::new(),
            merge_redirects: HashMap::new(),
            block_end_labels: HashMap::new(),
            self_loops: HashMap::new(),
            loop_carried: HashMap::new(),
            current_block: None,
            cbuf_bindings_used: 0,
            texs_ids_used: std::collections::BTreeSet::new(),
            texture_slots: HashMap::new(),
            vertex_opts: VertexOptions::default(),
            const_cache_f32,
            const_cache_u32,
            bool_t,
            bool_true,
            bool_false,
            pred_regs: [None; 7],
            ubo_vec4s,
            ssbo_vars: vec![None; MAX_SSBO as usize],
            ptr_storage_u32: None,
            return_block: None,
            sample_debug_slot: std::env::var("NEXIUM_FS_SAMPLE_SLOT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok()),
            sample_debug_component: std::env::var("NEXIUM_FS_SAMPLE_COMPONENT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|v| v.min(3)),
            sample_debug_value: None,
            texcoord_debug_slot: std::env::var("NEXIUM_FS_TEXCOORD_SLOT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok()),
            tex_v_flip_slots: std::env::var("NEXIUM_TEX_V_FLIP_SLOTS")
                .ok()
                .map(|v| {
                    v.split(',')
                        .filter_map(|part| part.trim().parse::<u32>().ok())
                        .collect()
                })
                .unwrap_or_default(),
            sampler_arrayed: false,
            no_kil_shader: false,
            fragment_color_outputs: 1,
            fragment_output_map: 0,
            ps_input_map: [0; 32],
        }
    }

    fn setup_ssbos(&mut self, count: u32) {
        if count == 0 {
            return;
        }
        let u32_t = self.u32_t;
        let ptr_st = self.b.type_pointer(None, StorageClass::Uniform, u32_t);
        self.ptr_storage_u32 = Some(ptr_st);
        let n = count.min(MAX_SSBO);
        for i in 0..n {
            let rt = self.b.type_runtime_array(u32_t);
            self.b
                .decorate(rt, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
            let st = self.b.type_struct([rt]);
            self.b.decorate(st, Decoration::BufferBlock, []);
            self.b
                .member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
            self.b.member_decorate(st, 0, Decoration::NonWritable, []);
            let ptr_struct = self.b.type_pointer(None, StorageClass::Uniform, st);
            let var = self
                .b
                .variable(ptr_struct, None, StorageClass::Uniform, None);
            self.b
                .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            self.b.decorate(
                var,
                Decoration::Binding,
                [Operand::LiteralBit32(SSBO_BINDING_BASE + i)],
            );
            self.ssbo_vars[i as usize] = Some(var);
        }
    }

    pub fn new_with_vertex_opts(stage: Stage, vertex_opts: VertexOptions) -> Self {
        let mut e = Self::new(stage);
        e.vertex_opts = vertex_opts;
        e
    }

    fn new_with_vertex_opts_sized(
        stage: Stage,
        vertex_opts: VertexOptions,
        ubo_vec4s: u32,
    ) -> Self {
        let mut e = Self::new_sized(stage, ubo_vec4s);
        e.setup_ssbos(vertex_opts.num_ssbo);
        e.vertex_opts = vertex_opts;
        e
    }

    fn position_var(&mut self) -> Word {
        if let Some(v) = self.pos_var {
            return v;
        }
        let v = self
            .b
            .variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::Position)],
        );
        self.interface.push(v);
        self.pos_var = Some(v);
        v
    }

    fn point_size_var_id(&mut self) -> Word {
        if let Some(v) = self.point_size_var {
            return v;
        }
        let v = self
            .b
            .variable(self.ptr_output_f32, None, StorageClass::Output, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::PointSize)],
        );
        self.interface.push(v);
        self.point_size_var = Some(v);
        v
    }

    fn frag_coord_var(&mut self) -> Word {
        if let Some(v) = self.frag_coord_var {
            return v;
        }
        let v = self
            .b
            .variable(self.ptr_input_vec4, None, StorageClass::Input, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::FragCoord)],
        );
        self.interface.push(v);
        self.frag_coord_var = Some(v);
        v
    }

    fn frag_color_var_at(&mut self, location: u32) -> Word {
        if let Some(&v) = self.frag_color_vars.get(&location) {
            return v;
        }
        let attachment_location = self.fragment_output_attachment_location(location);
        let v = self
            .b
            .variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        self.b.decorate(
            v,
            Decoration::Location,
            [Operand::LiteralBit32(attachment_location)],
        );
        self.interface.push(v);
        self.frag_color_vars.insert(location, v);
        v
    }

    fn fragment_output_group(&self, location: u32) -> Option<(u8, u32)> {
        if self.fragment_output_map == 0 {
            return Some(((location * 4) as u8, 0xF));
        }
        if location >= 8 {
            return None;
        }
        let mask = (self.fragment_output_map >> (location * 4)) & 0xF;
        if mask == 0 {
            return None;
        }
        let mut base = 0u32;
        for rt in 0..location {
            if ((self.fragment_output_map >> (rt * 4)) & 0xF) != 0 {
                base += 4;
            }
        }
        Some((base as u8, mask))
    }

    fn fragment_output_mask(&self, location: u32) -> u32 {
        self.fragment_output_group(location)
            .map(|(_, mask)| mask)
            .unwrap_or(0)
    }

    fn fragment_output_locations(&self) -> Vec<u32> {
        let count = self.fragment_color_outputs.max(1).min(8);
        if self.fragment_output_map == 0 {
            return (0..count).collect();
        }
        (0..8)
            .filter(|location| self.fragment_output_mask(*location) != 0)
            .take(count as usize)
            .collect()
    }

    fn fragment_output_attachment_location(&self, location: u32) -> u32 {
        if self.fragment_output_map == 0 {
            return location;
        }
        (0..location)
            .filter(|rt| self.fragment_output_mask(*rt) != 0)
            .count() as u32
    }

    fn fragment_output_reg_base(&self, location: u32) -> u8 {
        self.fragment_output_group(location)
            .map(|(base, _)| base)
            .unwrap_or((location * 4) as u8)
    }

    fn fragment_output_vec(
        &mut self,
        es: Option<&HashMap<u8, nexium_shader::ir::Value>>,
        location: u32,
        defaults: [Word; 4],
    ) -> Word {
        let base = self.fragment_output_reg_base(location);
        let mask = self.fragment_output_mask(location);
        let chans: [Word; 4] = std::array::from_fn(|c| {
            if (mask & (1 << c)) == 0 {
                defaults[c]
            } else {
                es.and_then(|m| m.get(&(base + c as u8)))
                    .map(|v| self.lower_value(v))
                    .unwrap_or(defaults[c])
            }
        });
        self.b
            .composite_construct(self.vec4_t, None, chans)
            .unwrap()
    }

    fn store_fragment_output_vec(&mut self, location: u32, value: Word) {
        if self.fragment_output_map == 0 {
            let fc = self.frag_color_var_at(location);
            self.b.store(fc, value, None, []).unwrap();
            return;
        }
        let mask = self.fragment_output_mask(location);
        if mask == 0 {
            return;
        }
        let fc = self.frag_color_var_at(location);
        for component in 0..4 {
            if (mask & (1 << component)) == 0 {
                continue;
            }
            let idx = self.const_u32(component);
            let ptr = self
                .b
                .access_chain(self.ptr_output_f32, None, fc, [idx])
                .unwrap();
            let c = self
                .b
                .composite_extract(self.f32_t, None, value, [component])
                .unwrap();
            self.b.store(ptr, c, None, []).unwrap();
        }
    }

    fn apply_fragment_output_debug_overrides(&mut self, outputs: &mut [(u32, Word)]) {
        let Some(loc) = std::env::var("NEXIUM_FS_FORCE_OUTPUT_LOC")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        else {
            return;
        };
        let value = std::env::var("NEXIUM_FS_FORCE_OUTPUT_VALUE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(1.0);
        let v = self.const_f32(value.to_bits());
        let forced = self
            .b
            .composite_construct(self.vec4_t, None, [v, v, v, v])
            .unwrap();
        for (out_loc, out) in outputs {
            if *out_loc == loc {
                *out = forced;
            }
        }
    }

    fn vertex_index_var_id(&mut self) -> Word {
        if let Some(v) = self.vertex_index_var {
            return v;
        }
        let v = self
            .b
            .variable(self.ptr_input_u32, None, StorageClass::Input, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::VertexIndex)],
        );
        self.interface.push(v);
        self.vertex_index_var = Some(v);
        v
    }

    fn instance_index_var_id(&mut self) -> Word {
        if let Some(v) = self.instance_index_var {
            return v;
        }
        let v = self
            .b
            .variable(self.ptr_input_u32, None, StorageClass::Input, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::InstanceIndex)],
        );
        self.interface.push(v);
        self.instance_index_var = Some(v);
        v
    }

    fn system_attr_var_id(&mut self, slot: u32) -> Option<Word> {
        if !matches!(self.stage, Stage::Vertex) {
            return None;
        }
        match slot {
            0x2f8 => Some(self.instance_index_var_id()),
            0x2fc => Some(self.vertex_index_var_id()),
            _ => None,
        }
    }

    fn load_system_attr_bits(&mut self, slot: u32) -> Option<Word> {
        let var = self.system_attr_var_id(slot)?;
        let raw = self.b.load(self.u32_t, None, var, None, []).unwrap();
        Some(self.b.bitcast(self.f32_t, None, raw).unwrap())
    }

    fn input_var(&mut self, slot: u32) -> AttrVar {
        if let Some(av) = self.input_vars.get(&slot) {
            return *av;
        }
        let location = if slot >= 0x80 { (slot - 0x80) / 16 } else { 0 };
        if location >= 32 {
            let av = AttrVar {
                var: 0,
                ptr_f32: self.ptr_input_f32,
                is_uint: false,
                is_sint: false,
            };
            self.input_vars.insert(slot, av);
            return av;
        }
        let is_uint = matches!(self.stage, Stage::Vertex)
            && (self.vertex_opts.uint_attr_mask >> location) & 1 == 1;
        let is_sint = matches!(self.stage, Stage::Vertex)
            && (self.vertex_opts.sint_attr_mask >> location) & 1 == 1;
        let av = if is_uint {
            let uvec4_t = self.b.type_vector(self.u32_t, 4);
            let ptr_uvec4 = self.b.type_pointer(None, StorageClass::Input, uvec4_t);
            let ptr_u32 = self.b.type_pointer(None, StorageClass::Input, self.u32_t);
            let var = self.b.variable(ptr_uvec4, None, StorageClass::Input, None);
            self.b
                .decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
            AttrVar {
                var,
                ptr_f32: ptr_u32,
                is_uint: true,
                is_sint: false,
            }
        } else if is_sint {
            let ivec4_t = self.b.type_vector(self.i32_t, 4);
            let ptr_ivec4 = self.b.type_pointer(None, StorageClass::Input, ivec4_t);
            let ptr_i32 = self.b.type_pointer(None, StorageClass::Input, self.i32_t);
            let var = self.b.variable(ptr_ivec4, None, StorageClass::Input, None);
            self.b
                .decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
            AttrVar {
                var,
                ptr_f32: ptr_i32,
                is_uint: false,
                is_sint: true,
            }
        } else {
            let var = self
                .b
                .variable(self.ptr_input_vec4, None, StorageClass::Input, None);
            self.b
                .decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
            self.decorate_fs_input_interpolation(var, location);
            AttrVar {
                var,
                ptr_f32: self.ptr_input_f32,
                is_uint: false,
                is_sint: false,
            }
        };
        self.input_vars.insert(slot, av);
        self.interface.push(av.var);
        av
    }

    fn decorate_fs_input_interpolation(&mut self, var: Word, location: u32) {
        if !matches!(self.stage, Stage::Fragment) || location >= 32 {
            return;
        }
        let raw = self.ps_input_map[location as usize];
        let mut seen = None;
        for component in 0..4 {
            let mode = (raw >> (component * 2)) & 0x3;
            if mode == 0 {
                continue;
            }
            if let Some(prev) = seen {
                if prev != mode {
                    return;
                }
            } else {
                seen = Some(mode);
            }
        }
        match seen {
            Some(1) => {
                self.b.decorate(var, Decoration::Flat, []);
            }
            Some(3) => {
                self.b.decorate(var, Decoration::NoPerspective, []);
            }
            _ => {}
        }
    }

    fn ps_input_component_mode(&self, location: u32, component: u32) -> u8 {
        if !matches!(self.stage, Stage::Fragment) || location >= 32 || component >= 4 {
            return 0;
        }
        (self.ps_input_map[location as usize] >> (component * 2)) & 0x3
    }

    fn output_var(&mut self, slot: u32) -> AttrVar {
        if let Some(av) = self.output_vars.get(&slot) {
            return *av;
        }
        let location = if slot >= 0x80 { (slot - 0x80) / 16 } else { 0 };
        if location >= 32 {
            let av = AttrVar {
                var: 0,
                ptr_f32: self.ptr_output_f32,
                is_uint: false,
                is_sint: false,
            };
            self.output_vars.insert(slot, av);
            return av;
        }
        let var = self
            .b
            .variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        self.b
            .decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
        let av = AttrVar {
            var,
            ptr_f32: self.ptr_output_f32,
            is_uint: false,
            is_sint: false,
        };
        self.output_vars.insert(slot, av);
        self.interface.push(var);
        av
    }

    fn ensure_sampler_array(&mut self) -> (Word, Word) {
        if let (Some(img), Some(samp)) = (self.image_var, self.sampler_var) {
            return (img, samp);
        }
        let ptr_image_array = if self.sampler_arrayed {
            self.ptr_image_arrayed_array
        } else {
            self.ptr_image_array
        };
        let img = self
            .b
            .variable(ptr_image_array, None, StorageClass::UniformConstant, None);
        self.b
            .decorate(img, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b
            .decorate(img, Decoration::Binding, [Operand::LiteralBit32(1)]);
        let samp = self.b.variable(
            self.ptr_sampler_array,
            None,
            StorageClass::UniformConstant,
            None,
        );
        self.b
            .decorate(samp, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b
            .decorate(samp, Decoration::Binding, [Operand::LiteralBit32(2)]);
        self.image_var = Some(img);
        self.sampler_var = Some(samp);
        (img, samp)
    }

    fn ensure_image_3d_array(&mut self) -> Word {
        if let Some(v) = self.image_3d_var {
            return v;
        }
        let var = self.b.variable(
            self.ptr_image_3d_array,
            None,
            StorageClass::UniformConstant,
            None,
        );
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b
            .decorate(var, Decoration::Binding, [Operand::LiteralBit32(11)]);
        self.image_3d_var = Some(var);
        var
    }

    fn sampler_3d_at(&mut self, tex_id: u32) -> (Word, Word) {
        let (_, samp_array) = self.ensure_sampler_array();
        let img_array = self.ensure_image_3d_array();
        let idx = self
            .texture_slots
            .get(&tex_id)
            .copied()
            .unwrap_or(0)
            .min(MAX_TEXTURE_DESCRIPTORS - 1);
        let idx = self.const_u32(idx);
        let img = self
            .b
            .access_chain(self.ptr_image_3d, None, img_array, [idx])
            .unwrap();
        let samp = self
            .b
            .access_chain(self.ptr_sampler, None, samp_array, [idx])
            .unwrap();
        (img, samp)
    }

    fn sampler_at(&mut self, tex_id: u32) -> (Word, Word) {
        let (img_array, samp_array) = self.ensure_sampler_array();
        let idx = self
            .texture_slots
            .get(&tex_id)
            .copied()
            .unwrap_or(0)
            .min(MAX_TEXTURE_DESCRIPTORS - 1);
        let idx = self.const_u32(idx);
        let ptr_image = if self.sampler_arrayed {
            self.ptr_image_arrayed
        } else {
            self.ptr_image
        };
        let img = self
            .b
            .access_chain(ptr_image, None, img_array, [idx])
            .unwrap();
        let samp = self
            .b
            .access_chain(self.ptr_sampler, None, samp_array, [idx])
            .unwrap();
        (img, samp)
    }

    fn const_f32(&mut self, bits: u32) -> Word {
        if let Some(&w) = self.const_cache_f32.get(&bits) {
            return w;
        }
        let f32_t = self.f32_t;
        let w = self.b.constant_bit32(f32_t, bits);
        self.const_cache_f32.insert(bits, w);
        w
    }

    fn const_u32(&mut self, value: u32) -> Word {
        if let Some(&w) = self.const_cache_u32.get(&value) {
            return w;
        }
        let u32_t = self.u32_t;
        let w = self.b.constant_bit32(u32_t, value);
        self.const_cache_u32.insert(value, w);
        w
    }

    fn f32_undef_id(&mut self) -> Word {
        self.f32_zero
    }

    fn apply_neg_abs(&mut self, v: Word, neg: bool, abs: bool) -> Word {
        let mut r = v;
        if abs {
            r = self
                .b
                .ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(r)])
                .unwrap();
        }
        if neg {
            let neg_one = self.const_f32((-1.0f32).to_bits());
            r = self.b.f_mul(self.f32_t, None, r, neg_one).unwrap();
        }
        r
    }

    fn apply_sat(&mut self, v: Word, sat: bool) -> Word {
        if !sat {
            return v;
        }
        let zero = self.f32_zero;
        let one = self.f32_one;
        self.b
            .ext_inst(
                self.f32_t,
                None,
                self.glsl,
                43,
                [Operand::IdRef(v), Operand::IdRef(zero), Operand::IdRef(one)],
            )
            .unwrap()
    }

    fn resolve_pred(&mut self, idx: u8, negate: bool) -> Word {
        let raw = if idx == 7 {
            self.bool_true
        } else if let Some(w) = self.pred_regs.get(idx as usize).copied().flatten() {
            w
        } else {
            self.bool_false
        };
        if negate {
            let bt = self.bool_t;
            self.b.logical_not(bt, None, raw).unwrap()
        } else {
            raw
        }
    }

    fn write_pred_reg(&mut self, idx: u8, value: Word, guard: Option<Word>) {
        if idx >= 7 {
            return;
        }
        let value = if let Some(cond) = guard {
            let old = self.pred_regs[idx as usize].unwrap_or(self.bool_false);
            self.b.select(self.bool_t, None, cond, value, old).unwrap()
        } else {
            value
        };
        self.pred_regs[idx as usize] = Some(value);
    }

    fn lower_value(&mut self, v: &IrValue) -> Word {
        match v {
            IrValue::Inst(id) => match self.value_to_word.get(id).copied() {
                Some(w) => w,
                None => self.f32_undef_id(),
            },
            IrValue::Zero => self.f32_zero,
            IrValue::ImmF32(x) => self.const_f32(x.to_bits()),
            IrValue::ImmU32(u) => {
                let f = f32::from_bits(*u);
                self.const_f32(f.to_bits())
            }
            IrValue::GprIn(_) => self.f32_undef_id(),
        }
    }

    fn read_attr_component(&mut self, av: AttrVar, component: u32) -> Word {
        if av.var == 0 {
            return self.f32_zero;
        }
        let idx = self.const_u32(component);
        let ac = self
            .b
            .access_chain(av.ptr_f32, None, av.var, [idx])
            .unwrap();
        if av.is_uint {
            let raw = self.b.load(self.u32_t, None, ac, None, []).unwrap();
            self.b.bitcast(self.f32_t, None, raw).unwrap()
        } else if av.is_sint {
            let raw = self.b.load(self.i32_t, None, ac, None, []).unwrap();
            self.b.bitcast(self.f32_t, None, raw).unwrap()
        } else {
            self.b.load(self.f32_t, None, ac, None, []).unwrap()
        }
    }

    fn write_attr_component(&mut self, av: AttrVar, component: u32, val: Word) {
        if av.var == 0 {
            return;
        }
        let idx = self.const_u32(component);
        let ac = self
            .b
            .access_chain(av.ptr_f32, None, av.var, [idx])
            .unwrap();
        self.b.store(ac, val, None, []).unwrap();
    }

    fn select_guarded(&mut self, val: Word, guard: Option<Word>, old: Word) -> Word {
        if let Some(cond) = guard {
            self.b.select(self.f32_t, None, cond, val, old).unwrap()
        } else {
            val
        }
    }

    fn lower_store_attr(&mut self, slot: u32, src: &IrValue, guard: Option<Word>) {
        let mut val = self.lower_value(src);
        let component = (slot & 0xC) >> 2;
        let aligned_slot = slot & !0xF;
        if slot_is_gl_position(aligned_slot) {
            val = match (self.vertex_opts.window_ndc, component) {
                (Some((sx, _)), 0) if matches!(self.stage, Stage::Vertex) => {
                    let s = self.const_f32(sx.to_bits());
                    let m = self.b.f_mul(self.f32_t, None, val, s).unwrap();
                    self.b.f_sub(self.f32_t, None, m, self.f32_one).unwrap()
                }
                (Some((_, sy)), 1) if matches!(self.stage, Stage::Vertex) => {
                    let s = self.const_f32(sy.to_bits());
                    let m = self.b.f_mul(self.f32_t, None, val, s).unwrap();
                    self.b.f_sub(self.f32_t, None, m, self.f32_one).unwrap()
                }
                _ => val,
            };
            let pos = self.position_var();
            let idx = self.const_u32(component);
            let ac = self
                .b
                .access_chain(self.ptr_output_f32, None, pos, [idx])
                .unwrap();
            if guard.is_some() {
                let old = self.b.load(self.f32_t, None, ac, None, []).unwrap();
                val = self.select_guarded(val, guard, old);
            }
            self.b.store(ac, val, None, []).unwrap();
        } else if slot == 0x6C && matches!(self.stage, Stage::Vertex) {
            let v = self.point_size_var_id();
            if guard.is_some() {
                let old = self.b.load(self.f32_t, None, v, None, []).unwrap();
                val = self.select_guarded(val, guard, old);
            }
            self.b.store(v, val, None, []).unwrap();
        } else if slot < 0x80 {
            let _ = (val, component);
        } else {
            let av = self.output_var(aligned_slot);
            if guard.is_some() {
                let old = self.read_attr_component(av, component);
                val = self.select_guarded(val, guard, old);
            }
            self.write_attr_component(av, component, val);
        }
    }

    fn lower_fcompare(&mut self, cmp: &FComp, va: Word, vb: Word) -> Word {
        match cmp {
            FComp::F => self.bool_false,
            FComp::T => self.bool_true,
            FComp::Lt => self.b.f_ord_less_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Eq => self.b.f_ord_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Le => self
                .b
                .f_ord_less_than_equal(self.bool_t, None, va, vb)
                .unwrap(),
            FComp::Gt => self
                .b
                .f_ord_greater_than(self.bool_t, None, va, vb)
                .unwrap(),
            FComp::Ne => self.b.f_ord_not_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Ge => self
                .b
                .f_ord_greater_than_equal(self.bool_t, None, va, vb)
                .unwrap(),
            FComp::Num => self.b.ordered(self.bool_t, None, va, vb).unwrap(),
            FComp::Nan => self.b.unordered(self.bool_t, None, va, vb).unwrap(),
            FComp::Ltu => self.b.f_unord_less_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Equ => self.b.f_unord_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Leu => self
                .b
                .f_unord_less_than_equal(self.bool_t, None, va, vb)
                .unwrap(),
            FComp::Gtu => self
                .b
                .f_unord_greater_than(self.bool_t, None, va, vb)
                .unwrap(),
            FComp::Neu => self.b.f_unord_not_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Geu => self
                .b
                .f_unord_greater_than_equal(self.bool_t, None, va, vb)
                .unwrap(),
        }
    }

    fn as_i32(&mut self, w: Word) -> Word {
        self.b.bitcast(self.i32_t, None, w).unwrap()
    }

    fn as_u32(&mut self, w: Word) -> Word {
        self.b.bitcast(self.u32_t, None, w).unwrap()
    }

    fn store_bits(&mut self, w: Word) -> Word {
        self.b.bitcast(self.f32_t, None, w).unwrap()
    }

    fn lower_icompare(&mut self, cmp: &ICmp, signed: bool, a_u: Word, b_u: Word) -> Word {
        match cmp {
            ICmp::F => self.bool_false,
            ICmp::T => self.bool_true,
            ICmp::Eq => self.b.i_equal(self.bool_t, None, a_u, b_u).unwrap(),
            ICmp::Ne => self.b.i_not_equal(self.bool_t, None, a_u, b_u).unwrap(),
            ICmp::Lt => {
                if signed {
                    let (a, b) = (self.as_i32(a_u), self.as_i32(b_u));
                    self.b.s_less_than(self.bool_t, None, a, b).unwrap()
                } else {
                    self.b.u_less_than(self.bool_t, None, a_u, b_u).unwrap()
                }
            }
            ICmp::Le => {
                if signed {
                    let (a, b) = (self.as_i32(a_u), self.as_i32(b_u));
                    self.b.s_less_than_equal(self.bool_t, None, a, b).unwrap()
                } else {
                    self.b
                        .u_less_than_equal(self.bool_t, None, a_u, b_u)
                        .unwrap()
                }
            }
            ICmp::Gt => {
                if signed {
                    let (a, b) = (self.as_i32(a_u), self.as_i32(b_u));
                    self.b.s_greater_than(self.bool_t, None, a, b).unwrap()
                } else {
                    self.b.u_greater_than(self.bool_t, None, a_u, b_u).unwrap()
                }
            }
            ICmp::Ge => {
                if signed {
                    let (a, b) = (self.as_i32(a_u), self.as_i32(b_u));
                    self.b
                        .s_greater_than_equal(self.bool_t, None, a, b)
                        .unwrap()
                } else {
                    self.b
                        .u_greater_than_equal(self.bool_t, None, a_u, b_u)
                        .unwrap()
                }
            }
        }
    }

    fn lower_boolop(&mut self, op: &BoolOp, a: Word, b: Word) -> Word {
        match op {
            BoolOp::And => self.b.logical_and(self.bool_t, None, a, b).unwrap(),
            BoolOp::Or => self.b.logical_or(self.bool_t, None, a, b).unwrap(),
            BoolOp::Xor => self.b.logical_not_equal(self.bool_t, None, a, b).unwrap(),
        }
    }

    fn lower_flow_test(&mut self, flow_test: u8) -> Word {
        match flow_test {
            0 | 1 | 2 | 3 | 8 | 9 | 10 | 11 | 20 | 21 | 22 | 23 | 30 => self.bool_false,
            4 | 5 | 6 | 7 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 31 => self.bool_true,
            _ => self.bool_false,
        }
    }

    fn lower_op(&mut self, inst: &IrInst) {
        let result = inst.result;
        let word = match &inst.op {
            IrOp::Mov(src) => Some(self.lower_value(src)),
            IrOp::FMul { a, b, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let mut av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                if mods.scale != 0 {
                    let factor: f32 = match mods.scale {
                        1 => 0.5,
                        2 => 0.25,
                        3 => 0.125,
                        4 => 8.0,
                        5 => 4.0,
                        6 => 2.0,
                        _ => 1.0,
                    };
                    let fc = self.const_f32(factor.to_bits());
                    av = self.b.f_mul(self.f32_t, None, av, fc).unwrap();
                }
                let r = self.b.f_mul(self.f32_t, None, av, bv).unwrap();
                Some(self.apply_sat(r, mods.sat))
            }
            IrOp::FAdd { a, b, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let r = self.b.f_add(self.f32_t, None, av, bv).unwrap();
                Some(self.apply_sat(r, mods.sat))
            }
            IrOp::FFma { a, b, c, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let cv = self.lower_value(c);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let cv = self.apply_neg_abs(cv, mods.neg_c, false);
                let glsl = self.glsl;
                let f32_t = self.f32_t;
                let r = self
                    .b
                    .ext_inst(
                        f32_t,
                        None,
                        glsl,
                        50,
                        [Operand::IdRef(av), Operand::IdRef(bv), Operand::IdRef(cv)],
                    )
                    .unwrap();
                Some(self.apply_sat(r, mods.sat))
            }
            IrOp::FMin { a, b, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let (glsl, f32_t) = (self.glsl, self.f32_t);
                Some(
                    self.b
                        .ext_inst(
                            f32_t,
                            None,
                            glsl,
                            37,
                            [Operand::IdRef(av), Operand::IdRef(bv)],
                        )
                        .unwrap(),
                )
            }
            IrOp::FMax { a, b, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let (glsl, f32_t) = (self.glsl, self.f32_t);
                Some(
                    self.b
                        .ext_inst(
                            f32_t,
                            None,
                            glsl,
                            40,
                            [Operand::IdRef(av), Operand::IdRef(bv)],
                        )
                        .unwrap(),
                )
            }
            IrOp::FMinMaxPred {
                a,
                b,
                mods,
                pred,
                neg_pred,
            } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let f32_t = self.f32_t;
                let min = self
                    .b
                    .ext_inst(
                        f32_t,
                        None,
                        self.glsl,
                        37,
                        [Operand::IdRef(av), Operand::IdRef(bv)],
                    )
                    .unwrap();
                let max = self
                    .b
                    .ext_inst(
                        f32_t,
                        None,
                        self.glsl,
                        40,
                        [Operand::IdRef(av), Operand::IdRef(bv)],
                    )
                    .unwrap();
                let cond = self.resolve_pred(*pred, *neg_pred);
                Some(self.b.select(f32_t, None, cond, min, max).unwrap())
            }
            IrOp::SelectPred {
                pred,
                if_true,
                if_false,
            } => {
                let cond = self.resolve_pred(pred.idx, pred.negate);
                let t = self.lower_value(if_true);
                let f = self.lower_value(if_false);
                Some(self.b.select(self.f32_t, None, cond, t, f).unwrap())
            }
            IrOp::MultiFunc { src, func } => {
                let s = self.lower_value(src);
                let glsl = self.glsl;
                let f32_t = self.f32_t;
                let f32_one = self.f32_one;
                let one_arg = [Operand::IdRef(s)];
                Some(match func {
                    MufuFunc::Sin => self.b.ext_inst(f32_t, None, glsl, 13, one_arg).unwrap(),
                    MufuFunc::Cos => self.b.ext_inst(f32_t, None, glsl, 14, one_arg).unwrap(),
                    MufuFunc::Ex2 => self.b.ext_inst(f32_t, None, glsl, 29, one_arg).unwrap(),
                    MufuFunc::Lg2 => self.b.ext_inst(f32_t, None, glsl, 30, one_arg).unwrap(),
                    MufuFunc::Sqrt => self.b.ext_inst(f32_t, None, glsl, 31, one_arg).unwrap(),
                    MufuFunc::Rsq | MufuFunc::Rsq64h => {
                        self.b.ext_inst(f32_t, None, glsl, 32, one_arg).unwrap()
                    }
                    MufuFunc::Rcp | MufuFunc::Rcp64h => {
                        self.b.f_div(f32_t, None, f32_one, s).unwrap()
                    }
                    MufuFunc::Unknown(_) => self.b.undef(f32_t, None),
                })
            }
            IrOp::LoadCbuf {
                binding,
                byte_offset,
            } => {
                let logical_binding = match self.stage {
                    Stage::Vertex => (*binding as u32) & 0xF,
                    Stage::Fragment => 16 + ((*binding as u32) & 0xF),
                };
                self.cbuf_bindings_used |= 1u32 << logical_binding;
                let local_vec4 = (byte_offset / 16).min(CBUF_SLOT_VEC4S.saturating_sub(1));
                let vec4_index = (logical_binding * CBUF_SLOT_VEC4S + local_vec4) % self.ubo_vec4s;
                let component = (byte_offset / 4) & 0x3;
                let v_idx = self.const_u32(vec4_index);
                let c_idx = self.const_u32(component);
                let zero_u32 = self.const_u32(0);
                let ubo_var = self.ubo_var;
                let ptr_uniform_f32 = self.ptr_uniform_f32;
                let ac = self
                    .b
                    .access_chain(ptr_uniform_f32, None, ubo_var, [zero_u32, v_idx, c_idx])
                    .unwrap();
                Some(self.b.load(self.f32_t, None, ac, None, []).unwrap())
            }
            IrOp::LoadCbufIndexed {
                binding,
                byte_offset,
                index,
            } => {
                let logical_binding = match self.stage {
                    Stage::Vertex => (*binding as u32) & 0xF,
                    Stage::Fragment => 16 + ((*binding as u32) & 0xF),
                };
                self.cbuf_bindings_used |= 1u32 << logical_binding;
                let idx_f32 = self.lower_value(index);
                let u32_t = self.u32_t;
                let idx_u32 = self.b.bitcast(u32_t, None, idx_f32).unwrap();
                let imm = self.const_u32(*byte_offset);
                let eff = self.b.i_add(u32_t, None, imm, idx_u32).unwrap();
                let sh4 = self.const_u32(4);
                let slot_vec4s = self.const_u32(CBUF_SLOT_VEC4S);
                let binding_base = self.const_u32(logical_binding * CBUF_SLOT_VEC4S);
                let ubo_vec4s_c = self.const_u32(self.ubo_vec4s);
                let local = self.b.shift_right_logical(u32_t, None, eff, sh4).unwrap();
                let local_w = self.b.u_mod(u32_t, None, local, slot_vec4s).unwrap();
                let global = self.b.i_add(u32_t, None, binding_base, local_w).unwrap();
                let v_idx = self.b.u_mod(u32_t, None, global, ubo_vec4s_c).unwrap();
                let sh2 = self.const_u32(2);
                let three = self.const_u32(3);
                let comp_sh = self.b.shift_right_logical(u32_t, None, eff, sh2).unwrap();
                let c_idx = self.b.bitwise_and(u32_t, None, comp_sh, three).unwrap();
                let zero_u32 = self.const_u32(0);
                let ubo_var = self.ubo_var;
                let vec4_t = self.vec4_t;
                let ptr_vec4 = self.b.type_pointer(None, StorageClass::Uniform, vec4_t);
                let ac = self
                    .b
                    .access_chain(ptr_vec4, None, ubo_var, [zero_u32, v_idx])
                    .unwrap();
                let vec = self.b.load(vec4_t, None, ac, None, []).unwrap();
                Some(
                    self.b
                        .vector_extract_dynamic(self.f32_t, None, vec, c_idx)
                        .unwrap(),
                )
            }
            IrOp::LoadGlobal { .. } => Some(self.f32_zero),
            IrOp::LoadStorage {
                buffer_index,
                addr_lo,
                imm,
                cbuf_binding,
                cbuf_offset,
                align,
            } => {
                let bi = *buffer_index as usize;
                let ssbo = if bi < self.ssbo_vars.len() {
                    self.ssbo_vars[bi]
                } else {
                    None
                };
                match (ssbo, self.ptr_storage_u32) {
                    (Some(ssbo), Some(ptr_u)) => {
                        let u32_t = self.u32_t;
                        let addr_f = self.lower_value(addr_lo);
                        let addr_u = self.b.bitcast(u32_t, None, addr_f).unwrap();
                        let eff = if *imm != 0 {
                            let immc = self.const_u32(*imm as u32);
                            self.b.i_add(u32_t, None, addr_u, immc).unwrap()
                        } else {
                            addr_u
                        };
                        let logical_binding = match self.stage {
                            Stage::Vertex => (*cbuf_binding as u32) & 0xF,
                            Stage::Fragment => 16 + ((*cbuf_binding as u32) & 0xF),
                        };
                        self.cbuf_bindings_used |= 1u32 << logical_binding;
                        let local_vec4 = (cbuf_offset / 16).min(CBUF_SLOT_VEC4S.saturating_sub(1));
                        let vec4_index =
                            (logical_binding * CBUF_SLOT_VEC4S + local_vec4) % self.ubo_vec4s;
                        let component = (cbuf_offset / 4) & 0x3;
                        let v_idx = self.const_u32(vec4_index);
                        let c_idx = self.const_u32(component);
                        let zero_u32 = self.const_u32(0);
                        let ubo_var = self.ubo_var;
                        let ptr_uniform_f32 = self.ptr_uniform_f32;
                        let base_ac = self
                            .b
                            .access_chain(ptr_uniform_f32, None, ubo_var, [zero_u32, v_idx, c_idx])
                            .unwrap();
                        let base_f = self.b.load(self.f32_t, None, base_ac, None, []).unwrap();
                        let base_u = self.b.bitcast(u32_t, None, base_f).unwrap();
                        let mask = self.const_u32(!(align.saturating_sub(1)));
                        let base_a = self.b.bitwise_and(u32_t, None, base_u, mask).unwrap();
                        let offset = self.b.i_sub(u32_t, None, eff, base_a).unwrap();
                        let two = self.const_u32(2);
                        let word = self
                            .b
                            .shift_right_logical(u32_t, None, offset, two)
                            .unwrap();
                        let zero2 = self.const_u32(0);
                        let dac = self
                            .b
                            .access_chain(ptr_u, None, ssbo, [zero2, word])
                            .unwrap();
                        let val = self.b.load(u32_t, None, dac, None, []).unwrap();
                        Some(self.b.bitcast(self.f32_t, None, val).unwrap())
                    }
                    _ => Some(self.f32_zero),
                }
            }
            IrOp::LoadAttr { slot } => {
                if let Some(val) = self.load_system_attr_bits(*slot) {
                    Some(val)
                } else {
                    let component = (slot & 0xC) >> 2;
                    let aligned_slot = slot & !0xF;
                    let av = self.input_var(aligned_slot);
                    Some(self.read_attr_component(av, component))
                }
            }
            IrOp::InterpAttr {
                slot,
                perspective,
                mode,
                sat,
            } => {
                let component = (slot & 0xC) >> 2;
                let aligned_slot = slot & !0xF;
                let mut val = if aligned_slot == 0x70 {
                    let fc = self.frag_coord_var();
                    let idx = self.const_u32(component);
                    let ac = self
                        .b
                        .access_chain(self.ptr_input_f32, None, fc, [idx])
                        .unwrap();
                    self.b.load(self.f32_t, None, ac, None, []).unwrap()
                } else {
                    let av = self.input_var(aligned_slot);
                    let mut val = self.read_attr_component(av, component);
                    if aligned_slot >= 0x80 {
                        let loc = (aligned_slot - 0x80) / 16;
                        if self.ps_input_component_mode(loc, component) == 2 {
                            let fc = self.frag_coord_var();
                            let idx = self.const_u32(3);
                            let ac = self
                                .b
                                .access_chain(self.ptr_input_f32, None, fc, [idx])
                                .unwrap();
                            let w = self.b.load(self.f32_t, None, ac, None, []).unwrap();
                            val = self.b.f_mul(self.f32_t, None, val, w).unwrap();
                        }
                    }
                    val
                };
                if *mode == 1 {
                    let p = self.lower_value(perspective);
                    val = self.b.f_mul(self.f32_t, None, val, p).unwrap();
                }
                val = self.apply_sat(val, *sat);
                Some(val)
            }
            IrOp::StoreAttr { slot, src } => {
                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.lower_store_attr(*slot, src, guard);
                None
            }
            IrOp::SampleTex {
                tex_id,
                u,
                v,
                array,
                volume,
                component,
            } => 'sample_tex: {
                self.texs_ids_used.insert(*tex_id);
                if let Some(w) = volume {
                    let tex_slot = self
                        .texture_slots
                        .get(tex_id)
                        .copied()
                        .unwrap_or(0)
                        .min(MAX_TEXTURE_DESCRIPTORS - 1);
                    let uv0 = self.lower_value(u);
                    let uv1 = self.lower_value(v);
                    let uv2 = self.lower_value(w);
                    let coords = self
                        .b
                        .composite_construct(self.vec3_t, None, [uv0, uv1, uv2])
                        .unwrap();
                    if matches!(self.stage, Stage::Fragment)
                        && self.texcoord_debug_slot == Some(tex_slot)
                        && self.sample_debug_value.is_none()
                    {
                        self.sample_debug_value = Some(
                            self.b
                                .composite_construct(
                                    self.vec4_t,
                                    None,
                                    [uv0, uv1, uv2, self.f32_one],
                                )
                                .unwrap(),
                        );
                    }
                    let (img_var, samp_var) = self.sampler_3d_at(*tex_id);
                    let img = self
                        .b
                        .load(self.image_3d_t, None, img_var, None, [])
                        .unwrap();
                    let samp = self
                        .b
                        .load(self.sampler_t, None, samp_var, None, [])
                        .unwrap();
                    let sampled_img = self
                        .b
                        .sampled_image(self.sampled_image_3d_t, None, img, samp)
                        .unwrap();
                    let lod_zero = self.f32_zero;
                    let sampled = self
                        .b
                        .image_sample_explicit_lod(
                            self.vec4_t,
                            None,
                            sampled_img,
                            coords,
                            rspirv::spirv::ImageOperands::LOD,
                            [Operand::IdRef(lod_zero)],
                        )
                        .unwrap();
                    if matches!(self.stage, Stage::Fragment)
                        && self.sample_debug_slot == Some(tex_slot)
                        && self.sample_debug_value.is_none()
                    {
                        let debug_value = if let Some(component) = self.sample_debug_component {
                            let c = self
                                .b
                                .composite_extract(self.f32_t, None, sampled, [component])
                                .unwrap_or(self.f32_zero);
                            self.b
                                .composite_construct(self.vec4_t, None, [c, c, c, self.f32_one])
                                .unwrap()
                        } else {
                            sampled
                        };
                        self.sample_debug_value = Some(debug_value);
                    }
                    let c = self
                        .b
                        .composite_extract(self.f32_t, None, sampled, [*component as u32])
                        .unwrap_or(self.f32_zero);
                    break 'sample_tex Some(c);
                }
                let tex_slot = self
                    .texture_slots
                    .get(tex_id)
                    .copied()
                    .unwrap_or(0)
                    .min(MAX_TEXTURE_DESCRIPTORS - 1);
                let uv_override = std::env::var("NEXIUM_TEX_UV_OVERRIDE").ok().and_then(|s| {
                    let mut parts = s.split(',');
                    let u = parts.next()?.trim().parse::<f32>().ok()?;
                    let v = parts.next()?.trim().parse::<f32>().ok()?;
                    Some((u, v))
                });
                let (uv0, mut uv1) = if let Some((u, v)) = uv_override {
                    (self.const_f32(u.to_bits()), self.const_f32(v.to_bits()))
                } else {
                    (self.lower_value(u), self.lower_value(v))
                };
                if matches!(self.stage, Stage::Fragment)
                    && self.tex_v_flip_slots.contains(&tex_slot)
                {
                    uv1 = self.b.f_sub(self.f32_t, None, self.f32_one, uv1).unwrap();
                }
                let coords = if self.sampler_arrayed {
                    let layer = if let Some(override_layer) =
                        std::env::var("NEXIUM_TEX_LAYER_OVERRIDE")
                            .ok()
                            .and_then(|v| v.parse::<f32>().ok())
                    {
                        self.const_f32(override_layer.to_bits())
                    } else if let Some(array) = array {
                        let layer_value = self.lower_value(array);
                        let raw = self.as_u32(layer_value);
                        let mask = self.const_u32(0xFFFF);
                        let layer_u = self.b.bitwise_and(self.u32_t, None, raw, mask).unwrap();
                        self.b.convert_u_to_f(self.f32_t, None, layer_u).unwrap()
                    } else {
                        self.f32_zero
                    };
                    self.b
                        .composite_construct(self.vec3_t, None, [uv0, uv1, layer])
                        .unwrap()
                } else {
                    self.b
                        .composite_construct(self.vec2_t, None, [uv0, uv1])
                        .unwrap()
                };
                if matches!(self.stage, Stage::Fragment)
                    && self.texcoord_debug_slot == Some(tex_slot)
                    && self.sample_debug_value.is_none()
                {
                    let layer = if self.sampler_arrayed {
                        self.b
                            .composite_extract(self.f32_t, None, coords, [2])
                            .unwrap_or(self.f32_zero)
                    } else {
                        self.f32_zero
                    };
                    self.sample_debug_value = Some(
                        self.b
                            .composite_construct(self.vec4_t, None, [uv0, uv1, layer, self.f32_one])
                            .unwrap(),
                    );
                }
                let (img_var, samp_var) = self.sampler_at(*tex_id);
                let image_t = if self.sampler_arrayed {
                    self.image_arrayed_t
                } else {
                    self.image_t
                };
                let sampled_image_t = if self.sampler_arrayed {
                    self.sampled_image_arrayed_t
                } else {
                    self.sampled_image_t
                };
                let img = self.b.load(image_t, None, img_var, None, []).unwrap();
                let samp = self
                    .b
                    .load(self.sampler_t, None, samp_var, None, [])
                    .unwrap();
                let sampled_img = self
                    .b
                    .sampled_image(sampled_image_t, None, img, samp)
                    .unwrap();
                let lod_zero = self.f32_zero;
                let sampled = self
                    .b
                    .image_sample_explicit_lod(
                        self.vec4_t,
                        None,
                        sampled_img,
                        coords,
                        rspirv::spirv::ImageOperands::LOD,
                        [Operand::IdRef(lod_zero)],
                    )
                    .unwrap();
                if matches!(self.stage, Stage::Fragment)
                    && self.sample_debug_slot == Some(tex_slot)
                    && self.sample_debug_value.is_none()
                {
                    let debug_value = if let Some(component) = self.sample_debug_component {
                        let c = self
                            .b
                            .composite_extract(self.f32_t, None, sampled, [component])
                            .unwrap_or(self.f32_zero);
                        self.b
                            .composite_construct(self.vec4_t, None, [c, c, c, self.f32_one])
                            .unwrap()
                    } else {
                        sampled
                    };
                    self.sample_debug_value = Some(debug_value);
                }
                let c = self
                    .b
                    .composite_extract(self.f32_t, None, sampled, [*component as u32])
                    .unwrap_or(self.f32_zero);
                if std::env::var("NEXIUM_TEX_2X").is_ok() {
                    let two = self.const_f32(2.0f32.to_bits());
                    Some(self.b.f_mul(self.f32_t, None, c, two).unwrap())
                } else {
                    Some(c)
                }
            }
            IrOp::GatherTex {
                tex_id,
                u,
                v,
                gather_component,
                lane,
            } => {
                self.texs_ids_used.insert(*tex_id);
                let tex_slot = self
                    .texture_slots
                    .get(tex_id)
                    .copied()
                    .unwrap_or(0)
                    .min(MAX_TEXTURE_DESCRIPTORS - 1);
                let uv0 = self.lower_value(u);
                let mut uv1 = self.lower_value(v);
                if matches!(self.stage, Stage::Fragment)
                    && self.tex_v_flip_slots.contains(&tex_slot)
                {
                    uv1 = self.b.f_sub(self.f32_t, None, self.f32_one, uv1).unwrap();
                }
                let coords = if self.sampler_arrayed {
                    self.b
                        .composite_construct(self.vec3_t, None, [uv0, uv1, self.f32_zero])
                        .unwrap()
                } else {
                    self.b
                        .composite_construct(self.vec2_t, None, [uv0, uv1])
                        .unwrap()
                };
                if matches!(self.stage, Stage::Fragment)
                    && self.texcoord_debug_slot == Some(tex_slot)
                    && self.sample_debug_value.is_none()
                {
                    self.sample_debug_value = Some(
                        self.b
                            .composite_construct(
                                self.vec4_t,
                                None,
                                [uv0, uv1, self.f32_zero, self.f32_one],
                            )
                            .unwrap(),
                    );
                }
                let (img_var, samp_var) = self.sampler_at(*tex_id);
                let image_t = if self.sampler_arrayed {
                    self.image_arrayed_t
                } else {
                    self.image_t
                };
                let sampled_image_t = if self.sampler_arrayed {
                    self.sampled_image_arrayed_t
                } else {
                    self.sampled_image_t
                };
                let img = self.b.load(image_t, None, img_var, None, []).unwrap();
                let samp = self
                    .b
                    .load(self.sampler_t, None, samp_var, None, [])
                    .unwrap();
                let sampled_img = self
                    .b
                    .sampled_image(sampled_image_t, None, img, samp)
                    .unwrap();
                let component = self.const_u32((*gather_component).min(3) as u32);
                let gathered = self
                    .b
                    .image_gather(self.vec4_t, None, sampled_img, coords, component, None, [])
                    .unwrap();
                if matches!(self.stage, Stage::Fragment)
                    && self.sample_debug_slot == Some(tex_slot)
                    && self.sample_debug_value.is_none()
                {
                    self.sample_debug_value = Some(gathered);
                }
                Some(
                    self.b
                        .composite_extract(self.f32_t, None, gathered, [(*lane).min(3) as u32])
                        .unwrap_or(self.f32_zero),
                )
            }
            IrOp::FSetPred {
                cmp,
                bop,
                src_a,
                src_b,
                neg_a,
                abs_a,
                neg_b,
                abs_b,
                src_pred,
                src_pred_inv,
                dest_p,
                dest_np,
            } => {
                let mut va = self.lower_value(src_a);
                let mut vb = self.lower_value(src_b);
                if *abs_a {
                    va = self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(va)])
                        .unwrap();
                }
                if *neg_a {
                    let neg_one = self.const_f32((-1.0f32).to_bits());
                    va = self.b.f_mul(self.f32_t, None, va, neg_one).unwrap();
                }
                if *abs_b {
                    vb = self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(vb)])
                        .unwrap();
                }
                if *neg_b {
                    let neg_one = self.const_f32((-1.0f32).to_bits());
                    vb = self.b.f_mul(self.f32_t, None, vb, neg_one).unwrap();
                }

                let cmp_result = match cmp {
                    FComp::F => self.bool_false,
                    FComp::T => self.bool_true,
                    FComp::Lt => self.b.f_ord_less_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Eq => self.b.f_ord_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Le => self
                        .b
                        .f_ord_less_than_equal(self.bool_t, None, va, vb)
                        .unwrap(),
                    FComp::Gt => self
                        .b
                        .f_ord_greater_than(self.bool_t, None, va, vb)
                        .unwrap(),
                    FComp::Ne => self.b.f_ord_not_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Ge => self
                        .b
                        .f_ord_greater_than_equal(self.bool_t, None, va, vb)
                        .unwrap(),
                    FComp::Num => self.b.ordered(self.bool_t, None, va, vb).unwrap(),
                    FComp::Nan => self.b.unordered(self.bool_t, None, va, vb).unwrap(),
                    FComp::Ltu => self.b.f_unord_less_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Equ => self.b.f_unord_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Leu => self
                        .b
                        .f_unord_less_than_equal(self.bool_t, None, va, vb)
                        .unwrap(),
                    FComp::Gtu => self
                        .b
                        .f_unord_greater_than(self.bool_t, None, va, vb)
                        .unwrap(),
                    FComp::Neu => self.b.f_unord_not_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Geu => self
                        .b
                        .f_unord_greater_than_equal(self.bool_t, None, va, vb)
                        .unwrap(),
                };

                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self
                        .b
                        .logical_and(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Or => self
                        .b
                        .logical_or(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Xor => self
                        .b
                        .logical_not_equal(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                };

                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.write_pred_reg(*dest_p, combined, guard);
                if *dest_np < 7 {
                    let not_cmp = self.b.logical_not(self.bool_t, None, cmp_result).unwrap();
                    let combined_np = match bop {
                        BoolOp::And => self
                            .b
                            .logical_and(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                        BoolOp::Or => self
                            .b
                            .logical_or(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                        BoolOp::Xor => self
                            .b
                            .logical_not_equal(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                    };
                    self.write_pred_reg(*dest_np, combined_np, guard);
                }
                Some(combined)
            }

            IrOp::F2F {
                src,
                neg,
                abs,
                sat,
                round,
            } => {
                let v = self.lower_value(src);
                let v = self.apply_neg_abs(v, *neg, *abs);
                let v = match *round {
                    1 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 2, [Operand::IdRef(v)])
                        .unwrap(),
                    2 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 8, [Operand::IdRef(v)])
                        .unwrap(),
                    3 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 9, [Operand::IdRef(v)])
                        .unwrap(),
                    4 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 3, [Operand::IdRef(v)])
                        .unwrap(),
                    _ => v,
                };
                Some(self.apply_sat(v, *sat))
            }
            IrOp::IAdd { a, b, neg_a, neg_b } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let mut au = self.as_u32(av);
                let mut bu = self.as_u32(bv);
                if *neg_a {
                    au = self.b.s_negate(self.u32_t, None, au).unwrap();
                }
                if *neg_b {
                    bu = self.b.s_negate(self.u32_t, None, bu).unwrap();
                }
                let r = self.b.i_add(self.u32_t, None, au, bu).unwrap();
                Some(self.store_bits(r))
            }
            IrOp::IMul { a, b } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let r = self.b.i_mul(self.u32_t, None, au, bu).unwrap();
                Some(self.store_bits(r))
            }
            IrOp::IMinMaxPred {
                a,
                b,
                signed,
                pred,
                neg_pred,
            } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let (min_op, max_op) = if *signed { (39, 42) } else { (38, 41) };
                let min = self
                    .b
                    .ext_inst(
                        self.u32_t,
                        None,
                        self.glsl,
                        min_op,
                        [Operand::IdRef(au), Operand::IdRef(bu)],
                    )
                    .unwrap();
                let max = self
                    .b
                    .ext_inst(
                        self.u32_t,
                        None,
                        self.glsl,
                        max_op,
                        [Operand::IdRef(au), Operand::IdRef(bu)],
                    )
                    .unwrap();
                let cond = self.resolve_pred(*pred, *neg_pred);
                let r = self.b.select(self.u32_t, None, cond, min, max).unwrap();
                Some(self.store_bits(r))
            }
            IrOp::IScAdd {
                a,
                b,
                shift,
                neg_a,
                neg_b,
            } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let mut au = self.as_u32(av);
                let mut bu = self.as_u32(bv);
                if *neg_a {
                    au = self.b.s_negate(self.u32_t, None, au).unwrap();
                }
                if *neg_b {
                    bu = self.b.s_negate(self.u32_t, None, bu).unwrap();
                }
                let sh = self.const_u32(*shift as u32);
                let shifted = self.b.shift_left_logical(self.u32_t, None, au, sh).unwrap();
                let r = self.b.i_add(self.u32_t, None, shifted, bu).unwrap();
                Some(self.store_bits(r))
            }
            IrOp::ILop {
                a,
                b,
                op,
                not_a,
                not_b,
            } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let mut au = self.as_u32(av);
                let mut bu = self.as_u32(bv);
                if *not_a {
                    au = self.b.not(self.u32_t, None, au).unwrap();
                }
                if *not_b {
                    bu = self.b.not(self.u32_t, None, bu).unwrap();
                }
                let r = match op {
                    LogicOp::And => self.b.bitwise_and(self.u32_t, None, au, bu).unwrap(),
                    LogicOp::Or => self.b.bitwise_or(self.u32_t, None, au, bu).unwrap(),
                    LogicOp::Xor => self.b.bitwise_xor(self.u32_t, None, au, bu).unwrap(),
                    LogicOp::PassB => bu,
                };
                Some(self.store_bits(r))
            }
            IrOp::IShl { a, b } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let r = self.b.shift_left_logical(self.u32_t, None, au, bu).unwrap();
                Some(self.store_bits(r))
            }
            IrOp::IShr { a, b, signed } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let r = if *signed {
                    let ai = self.as_i32(au);
                    let ri = self
                        .b
                        .shift_right_arithmetic(self.i32_t, None, ai, bu)
                        .unwrap();
                    self.b.bitcast(self.u32_t, None, ri).unwrap()
                } else {
                    self.b
                        .shift_right_logical(self.u32_t, None, au, bu)
                        .unwrap()
                };
                Some(self.store_bits(r))
            }
            IrOp::F2I { src, signed, round } => {
                let f = self.lower_value(src);
                let f = match *round {
                    0 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 2, [Operand::IdRef(f)])
                        .unwrap(),
                    1 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 8, [Operand::IdRef(f)])
                        .unwrap(),
                    2 => self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 9, [Operand::IdRef(f)])
                        .unwrap(),
                    _ => f,
                };
                let r = if *signed {
                    let i = self.b.convert_f_to_s(self.i32_t, None, f).unwrap();
                    self.b.bitcast(self.u32_t, None, i).unwrap()
                } else {
                    self.b.convert_f_to_u(self.u32_t, None, f).unwrap()
                };
                Some(self.store_bits(r))
            }
            IrOp::Bfe { a, b, signed } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let mask = self.const_u32(0xff);
                let pos = self.b.bitwise_and(self.u32_t, None, bu, mask).unwrap();
                let eight = self.const_u32(8);
                let size_raw = self
                    .b
                    .shift_right_logical(self.u32_t, None, bu, eight)
                    .unwrap();
                let cnt = self
                    .b
                    .bitwise_and(self.u32_t, None, size_raw, mask)
                    .unwrap();
                let r = if *signed {
                    self.b
                        .bit_field_s_extract(self.u32_t, None, au, pos, cnt)
                        .unwrap()
                } else {
                    self.b
                        .bit_field_u_extract(self.u32_t, None, au, pos, cnt)
                        .unwrap()
                };
                Some(self.store_bits(r))
            }
            IrOp::ISet {
                cmp,
                signed,
                a,
                b,
                bool_float,
            } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let cmp_result = self.lower_icompare(cmp, *signed, au, bu);
                if *bool_float {
                    let one = self.f32_one;
                    let zero = self.f32_zero;
                    Some(
                        self.b
                            .select(self.f32_t, None, cmp_result, one, zero)
                            .unwrap(),
                    )
                } else {
                    let ones = self.const_u32(0xFFFF_FFFF);
                    let zeros = self.const_u32(0);
                    let sel = self
                        .b
                        .select(self.u32_t, None, cmp_result, ones, zeros)
                        .unwrap();
                    Some(self.store_bits(sel))
                }
            }
            IrOp::I2F {
                src,
                signed,
                neg,
                abs,
                int_format,
                selector,
            } => {
                let f = self.lower_value(src);
                let bits_u = self.b.bitcast(self.u32_t, None, f).unwrap();
                let extracted = match *int_format {
                    0 => {
                        let off = self.const_u32((*selector as u32) * 8);
                        let cnt = self.const_u32(8);
                        if *signed {
                            self.b
                                .bit_field_s_extract(self.u32_t, None, bits_u, off, cnt)
                                .unwrap()
                        } else {
                            self.b
                                .bit_field_u_extract(self.u32_t, None, bits_u, off, cnt)
                                .unwrap()
                        }
                    }
                    1 => {
                        let off = self.const_u32((*selector as u32) * 8);
                        let cnt = self.const_u32(16);
                        if *signed {
                            self.b
                                .bit_field_s_extract(self.u32_t, None, bits_u, off, cnt)
                                .unwrap()
                        } else {
                            self.b
                                .bit_field_u_extract(self.u32_t, None, bits_u, off, cnt)
                                .unwrap()
                        }
                    }
                    _ => bits_u,
                };
                let mut val = if *signed {
                    let as_i = self.b.bitcast(self.i32_t, None, extracted).unwrap();
                    self.b.convert_s_to_f(self.f32_t, None, as_i).unwrap()
                } else {
                    self.b.convert_u_to_f(self.f32_t, None, extracted).unwrap()
                };
                if *abs {
                    val = self
                        .b
                        .ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(val)])
                        .unwrap();
                }
                if *neg {
                    let n = self.const_f32((-1.0f32).to_bits());
                    val = self.b.f_mul(self.f32_t, None, val, n).unwrap();
                }
                Some(val)
            }
            IrOp::FSet {
                cmp,
                bop,
                src_a,
                src_b,
                neg_a,
                abs_a,
                neg_b,
                abs_b,
                bf,
                src_pred,
                src_pred_inv,
            } => {
                let va = self.lower_value(src_a);
                let vb = self.lower_value(src_b);
                let va = self.apply_neg_abs(va, *neg_a, *abs_a);
                let vb = self.apply_neg_abs(vb, *neg_b, *abs_b);
                let cmp_result = self.lower_fcompare(cmp, va, vb);
                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self
                        .b
                        .logical_and(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Or => self
                        .b
                        .logical_or(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Xor => self
                        .b
                        .logical_not_equal(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                };
                if *bf {
                    let one = self.f32_one;
                    let zero = self.f32_zero;
                    Some(
                        self.b
                            .select(self.f32_t, None, combined, one, zero)
                            .unwrap(),
                    )
                } else {
                    let one_mask = self.const_u32(u32::MAX);
                    let zero_mask = self.const_u32(0);
                    let bits = self
                        .b
                        .select(self.u32_t, None, combined, one_mask, zero_mask)
                        .unwrap();
                    Some(self.b.bitcast(self.f32_t, None, bits).unwrap())
                }
            }
            IrOp::ISetPred {
                cmp,
                signed,
                bop,
                src_a,
                src_b,
                src_pred,
                src_pred_inv,
                dest_p,
                dest_np,
            } => {
                let fa = self.lower_value(src_a);
                let fb = self.lower_value(src_b);
                let a_u = self.b.bitcast(self.u32_t, None, fa).unwrap();
                let b_u = self.b.bitcast(self.u32_t, None, fb).unwrap();
                let cmp_result = self.lower_icompare(cmp, *signed, a_u, b_u);
                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self
                        .b
                        .logical_and(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Or => self
                        .b
                        .logical_or(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                    BoolOp::Xor => self
                        .b
                        .logical_not_equal(self.bool_t, None, cmp_result, src_p_word)
                        .unwrap(),
                };
                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.write_pred_reg(*dest_p, combined, guard);
                if *dest_np < 7 {
                    let not_cmp = self.b.logical_not(self.bool_t, None, cmp_result).unwrap();
                    let combined_np = match bop {
                        BoolOp::And => self
                            .b
                            .logical_and(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                        BoolOp::Or => self
                            .b
                            .logical_or(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                        BoolOp::Xor => self
                            .b
                            .logical_not_equal(self.bool_t, None, not_cmp, src_p_word)
                            .unwrap(),
                    };
                    self.write_pred_reg(*dest_np, combined_np, guard);
                }
                Some(combined)
            }
            IrOp::PSetPred {
                dest_p,
                dest_np,
                pred_a,
                neg_pred_a,
                pred_b,
                neg_pred_b,
                pred_c,
                neg_pred_c,
                bop_1,
                bop_2,
            } => {
                let pa = self.resolve_pred(*pred_a, *neg_pred_a);
                let pb = self.resolve_pred(*pred_b, *neg_pred_b);
                let pc = self.resolve_pred(*pred_c, *neg_pred_c);
                let lhs_a = self.lower_boolop(bop_1, pa, pb);
                let not_pa = self.b.logical_not(self.bool_t, None, pa).unwrap();
                let lhs_b = self.lower_boolop(bop_1, not_pa, pb);
                let result_a = self.lower_boolop(bop_2, lhs_a, pc);
                let result_b = self.lower_boolop(bop_2, lhs_b, pc);
                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.write_pred_reg(*dest_p, result_a, guard);
                self.write_pred_reg(*dest_np, result_b, guard);
                Some(result_a)
            }

            IrOp::CSetPred {
                dest_p,
                dest_np,
                flow_test,
                bop_pred,
                neg_bop_pred,
                bop,
            } => {
                let cc_result = self.lower_flow_test(*flow_test);
                let bop_pred = self.resolve_pred(*bop_pred, *neg_bop_pred);
                let result_a = self.lower_boolop(bop, cc_result, bop_pred);
                let not_cc = self.b.logical_not(self.bool_t, None, cc_result).unwrap();
                let result_b = self.lower_boolop(bop, not_cc, bop_pred);
                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.write_pred_reg(*dest_p, result_a, guard);
                self.write_pred_reg(*dest_np, result_b, guard);
                Some(result_a)
            }

            IrOp::PSet {
                pred_a,
                neg_pred_a,
                pred_b,
                neg_pred_b,
                pred_c,
                neg_pred_c,
                bop_1,
                bop_2,
                bool_float,
            } => {
                let pa = self.resolve_pred(*pred_a, *neg_pred_a);
                let pb = self.resolve_pred(*pred_b, *neg_pred_b);
                let pc = self.resolve_pred(*pred_c, *neg_pred_c);
                let lhs = self.lower_boolop(bop_1, pa, pb);
                let result = self.lower_boolop(bop_2, lhs, pc);
                let true_value = if *bool_float {
                    self.const_u32(0x3f80_0000)
                } else {
                    self.const_u32(0xffff_ffff)
                };
                let zero = self.const_u32(0);
                let value = self
                    .b
                    .select(self.u32_t, None, result, true_value, zero)
                    .unwrap();
                Some(self.store_bits(value))
            }

            IrOp::Kill => {
                if std::env::var("NEXIUM_NO_KIL").ok().as_deref() == Some("1") || self.no_kil_shader
                {
                    return;
                }
                let cond = if let Some(pred_guard) = &inst.pred {
                    self.resolve_pred(pred_guard.idx, pred_guard.negate)
                } else {
                    self.bool_true
                };

                let kill_block = self.b.id();
                let merge_block = self.b.id();
                self.b
                    .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
                    .unwrap();
                self.b
                    .branch_conditional(cond, kill_block, merge_block, [])
                    .unwrap();
                self.b.begin_block(Some(kill_block)).unwrap();
                self.b.kill().unwrap();
                self.b.begin_block(Some(merge_block)).unwrap();
                if let Some(block) = self.current_block {
                    self.block_end_labels.insert(block, merge_block);
                }
                None
            }

            IrOp::Exit => None,
            IrOp::Phi { .. } => Some(self.f32_undef_id()),
            IrOp::Unimplemented { .. } => Some(self.f32_undef_id()),
        };
        if let (Some(id), Some(w)) = (result, word) {
            self.value_to_word.insert(id, w);
        }
    }

    fn compute_shared_merges(&mut self, cfg: &Cfg) {
        let Some(ipd) = self.cond_merge.clone() else {
            return;
        };
        let dom = dominators(cfg);
        let mut by_merge: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for block in &cfg.blocks {
            if let BranchKind::Conditional { target, .. } = block.branch {
                let next = block.id + 1;
                if self.block_labels.contains_key(&next) && target != next {
                    let m = ipd[block.id as usize];
                    by_merge.entry(m).or_default().push(block.id);
                }
            }
        }
        for (m, mut headers) in by_merge {
            if headers.len() < 2 {
                continue;
            }
            headers.sort_unstable();
            let m_lbl = self.block_labels[&m];
            self.header_merge_label.insert(headers[0], m_lbl);
            let mut synths: Vec<(Word, u32, Word)> = Vec::new();
            for (idx, &h) in headers.iter().enumerate().skip(1) {
                let s = self.b.id();
                self.header_merge_label.insert(h, s);
                let parent = headers[..idx]
                    .iter()
                    .rev()
                    .find(|&&p| dom[h as usize][p as usize])
                    .copied();
                let target = parent
                    .and_then(|p| self.header_merge_label.get(&p).copied())
                    .unwrap_or(m_lbl);
                synths.push((s, h, target));
            }
            for b in 0..m {
                let target = headers
                    .iter()
                    .rev()
                    .find(|&&h| dom[b as usize][h as usize])
                    .and_then(|h| self.header_merge_label.get(h).copied())
                    .unwrap_or(m_lbl);
                self.merge_redirects.insert((m, b), target);
            }
            synths.reverse();
            self.synth_merge_blocks.insert(m, synths);
            self.shared_merge_headers.insert(m, headers);
        }
    }

    fn merge_redirect(&self, from_block: u32, to_block: u32) -> Word {
        if self.shared_merge_headers.contains_key(&to_block) {
            if let Some(&label) = self.merge_redirects.get(&(to_block, from_block)) {
                return label;
            }
        }
        self.block_labels[&to_block]
    }

    fn phi_pred_label(&self, block: BlockId) -> Word {
        self.block_end_labels
            .get(&block)
            .copied()
            .unwrap_or_else(|| self.block_labels[&block])
    }

    fn emit_synth_merge_blocks(&mut self, m_block: &BasicBlock) {
        let m = m_block.id;
        let Some(synths) = self.synth_merge_blocks.get(&m).cloned() else {
            return;
        };
        let phis: Vec<(ValueId, Vec<(BlockId, IrValue)>)> = m_block
            .program
            .instructions
            .iter()
            .filter_map(|inst| match &inst.op {
                IrOp::Phi { sources } => inst.result.map(|rid| (rid, sources.clone())),
                _ => None,
            })
            .collect();
        for (s_i, _h_i, target) in synths {
            self.b.begin_block(Some(s_i)).unwrap();
            for (rid, sources) in &phis {
                let mut pairs: Vec<(Word, Word)> = Vec::new();
                for (pred, val) in sources {
                    if self.merge_redirect(*pred, m) == s_i {
                        let v = self.lower_value(val);
                        let lbl = self.phi_pred_label(*pred);
                        pairs.push((v, lbl));
                    }
                }
                let child_pairs: Vec<(Word, Word)> = self
                    .synth_merge_blocks
                    .get(&m)
                    .into_iter()
                    .flatten()
                    .filter(|(child, _, child_target)| *child_target == s_i && *child != s_i)
                    .filter_map(|(child, _, _)| {
                        self.synth_phi_results
                            .get(&(*child, *rid))
                            .copied()
                            .map(|v| (v, *child))
                    })
                    .collect();
                pairs.extend(child_pairs);
                if pairs.is_empty() {
                    continue;
                }
                let pid = self.b.phi(self.f32_t, None, pairs).unwrap();
                self.synth_phi_results.insert((s_i, *rid), pid);
            }
            for phi in &m_block.pred_phis {
                let mut pairs: Vec<(Word, Word)> = Vec::new();
                for (pred, val) in &phi.sources {
                    if self.merge_redirect(*pred, m) == s_i {
                        let v = val
                            .and_then(|id| self.value_to_word.get(&id).copied())
                            .unwrap_or(self.bool_false);
                        let lbl = self.phi_pred_label(*pred);
                        pairs.push((v, lbl));
                    }
                }
                let child_pairs: Vec<(Word, Word)> = self
                    .synth_merge_blocks
                    .get(&m)
                    .into_iter()
                    .flatten()
                    .filter(|(child, _, child_target)| *child_target == s_i && *child != s_i)
                    .filter_map(|(child, _, _)| {
                        self.synth_pred_phi_results
                            .get(&(*child, phi.result))
                            .copied()
                            .map(|v| (v, *child))
                    })
                    .collect();
                pairs.extend(child_pairs);
                if pairs.is_empty() {
                    continue;
                }
                let pid = self.b.phi(self.bool_t, None, pairs).unwrap();
                self.synth_pred_phi_results.insert((s_i, phi.result), pid);
            }
            self.b.branch(target).unwrap();
        }
    }

    fn phi_cfg_lines(&self, cfg: &Cfg) -> Vec<String> {
        let mut lines = Vec::new();
        for block in &cfg.blocks {
            let has_phi = block
                .program
                .instructions
                .iter()
                .any(|inst| matches!(inst.op, IrOp::Phi { .. }));
            if !has_phi && block.pred_phis.is_empty() && !self.self_loops.contains_key(&block.id) {
                continue;
            }
            lines.push(format!(
                "[spirv-phi-cfg] block={} label={} end={} branch={:?}",
                block.id,
                self.block_labels.get(&block.id).copied().unwrap_or(0),
                self.phi_pred_label(block.id),
                block.branch
            ));
            for inst in &block.program.instructions {
                if let IrOp::Phi { sources } = &inst.op {
                    let mut parts = Vec::new();
                    for (pred, val) in sources {
                        parts.push(format!(
                            "{}:lbl{}:end{}:redir{}:{:?}",
                            pred,
                            self.block_labels.get(pred).copied().unwrap_or(0),
                            self.phi_pred_label(*pred),
                            self.merge_redirect(*pred, block.id),
                            val
                        ));
                    }
                    lines.push(format!(
                        "[spirv-phi-cfg] block={} phi={:?} sources={}",
                        block.id,
                        inst.result,
                        parts.join(",")
                    ));
                }
            }
            for phi in &block.pred_phis {
                let mut parts = Vec::new();
                for (pred, val) in &phi.sources {
                    parts.push(format!(
                        "{}:lbl{}:end{}:redir{}:{:?}",
                        pred,
                        self.block_labels.get(pred).copied().unwrap_or(0),
                        self.phi_pred_label(*pred),
                        self.merge_redirect(*pred, block.id),
                        val
                    ));
                }
                lines.push(format!(
                    "[spirv-phi-cfg] block={} pred_phi={} result={:?} sources={}",
                    block.id,
                    phi.pred,
                    phi.result,
                    parts.join(",")
                ));
            }
        }
        for (m, headers) in &self.shared_merge_headers {
            lines.push(format!(
                "[spirv-phi-merge] merge={} label={} headers={:?} synths={:?}",
                m,
                self.block_labels.get(m).copied().unwrap_or(0),
                headers,
                self.synth_merge_blocks.get(m)
            ));
        }
        lines
    }

    fn lower_cfg(&mut self, cfg: &Cfg) {
        let single_block = cfg.blocks.len() <= 1;
        for (idx, block) in cfg.blocks.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            let back_target = match block.branch {
                BranchKind::Conditional { target, .. } if target <= block.id => Some(target),
                BranchKind::Unconditional { target } if target <= block.id => Some(target),
                _ => None,
            };
            if let Some(header) = back_target {
                if header == 0 {
                    continue;
                }
                let body = self.b.id();
                let cont = self.b.id();
                let merge = self.b.id();
                self.self_loops
                    .insert(header, (body, cont, merge, block.id));
            }
        }
        for (idx, block) in cfg.blocks.iter().enumerate() {
            let is_first = idx == 0;
            if !is_first {
                self.emit_synth_merge_blocks(block);
                let label = self.block_labels[&block.id];
                self.b.begin_block(Some(label)).unwrap();
            }
            self.current_block = Some(block.id);
            self.lower_pred_phis(block);
            self.lower_phis(block);
            if let Some(&(body, cont, merge, _)) = self.self_loops.get(&block.id) {
                self.b
                    .loop_merge(merge, cont, rspirv::spirv::LoopControl::NONE, [])
                    .unwrap();
                self.b.branch(body).unwrap();
                self.b.begin_block(Some(body)).unwrap();
                self.block_end_labels.insert(block.id, body);
            }
            for inst in &block.program.instructions {
                if matches!(inst.op, IrOp::Phi { .. }) {
                    continue;
                }
                self.lower_op(inst);
            }
            if !single_block {
                self.emit_terminator(block);
            }
            self.current_block = None;
        }
    }

    fn lower_pred_phis(&mut self, block: &BasicBlock) {
        let m = block.id;
        let shared = self.shared_merge_headers.contains_key(&m);
        for phi in &block.pred_phis {
            let mut pairs: Vec<(Word, Word)> = Vec::new();
            if shared {
                let m_lbl = self.block_labels[&m];
                for (pred, val) in &phi.sources {
                    if self.merge_redirect(*pred, m) == m_lbl {
                        let v = val
                            .and_then(|id| self.value_to_word.get(&id).copied())
                            .unwrap_or(self.bool_false);
                        let label = self.phi_pred_label(*pred);
                        pairs.push((v, label));
                    }
                }
                let synths = self.synth_merge_blocks.get(&m).cloned().unwrap_or_default();
                for (s_i, _h, target) in synths {
                    if target != m_lbl {
                        continue;
                    }
                    let v = self
                        .synth_pred_phi_results
                        .get(&(s_i, phi.result))
                        .copied()
                        .unwrap_or(self.bool_false);
                    pairs.push((v, s_i));
                }
            } else {
                for (pred, val) in &phi.sources {
                    if let Some(&(_, cont, _, latch)) = self.self_loops.get(&m) {
                        if *pred == latch {
                            let v = match val {
                                Some(id) if !self.value_to_word.contains_key(id) => {
                                    let w = self.b.id();
                                    let bt = self.bool_t;
                                    self.loop_carried.entry(m).or_default().push((*id, w, bt));
                                    w
                                }
                                Some(id) => self.value_to_word[id],
                                None => self.bool_false,
                            };
                            pairs.push((v, cont));
                            continue;
                        }
                    }
                    let v = val
                        .and_then(|id| self.value_to_word.get(&id).copied())
                        .unwrap_or(self.bool_false);
                    let label = self.phi_pred_label(*pred);
                    pairs.push((v, label));
                }
            }
            let id = self.b.phi(self.bool_t, None, pairs).unwrap();
            self.value_to_word.insert(phi.result, id);
            if phi.pred < 7 {
                self.pred_regs[phi.pred as usize] = Some(id);
            }
        }
    }

    fn lower_phis(&mut self, block: &BasicBlock) {
        let m = block.id;
        let shared = self.shared_merge_headers.contains_key(&m);
        for inst in &block.program.instructions {
            let IrOp::Phi { sources } = &inst.op else {
                continue;
            };
            let f32_t = self.f32_t;
            let mut pairs: Vec<(Word, Word)> = Vec::with_capacity(sources.len());
            if shared {
                let m_lbl = self.block_labels[&m];
                for (pred_id, val) in sources {
                    if self.merge_redirect(*pred_id, m) == m_lbl {
                        let v = self.lower_value(val);
                        let label = self.phi_pred_label(*pred_id);
                        pairs.push((v, label));
                    }
                }
                if let Some(rid) = inst.result {
                    let synths = self.synth_merge_blocks.get(&m).cloned().unwrap_or_default();
                    for (s_i, _h, target) in synths {
                        if target != m_lbl {
                            continue;
                        }
                        let v = match self.synth_phi_results.get(&(s_i, rid)).copied() {
                            Some(w) => w,
                            None => self.f32_undef_id(),
                        };
                        pairs.push((v, s_i));
                    }
                }
            } else {
                for (pred_id, val) in sources {
                    if let Some(&(_, cont, _, latch)) = self.self_loops.get(&m) {
                        if *pred_id == latch {
                            let v = match val {
                                IrValue::Inst(id) if !self.value_to_word.contains_key(id) => {
                                    let w = self.b.id();
                                    self.loop_carried
                                        .entry(m)
                                        .or_default()
                                        .push((*id, w, f32_t));
                                    w
                                }
                                _ => self.lower_value(val),
                            };
                            pairs.push((v, cont));
                            continue;
                        }
                    }
                    let v = self.lower_value(val);
                    let label = self.phi_pred_label(*pred_id);
                    pairs.push((v, label));
                }
            }
            let id = self.b.phi(f32_t, None, pairs).unwrap();
            if let Some(rid) = inst.result {
                self.value_to_word.insert(rid, id);
            }
        }
    }

    fn emit_self_loop_close(&mut self, header_id: BlockId, latch_id: BlockId, cond: Option<Word>) {
        let (_, cont, merge, _) = self.self_loops[&header_id];
        match cond {
            Some(c) => self.b.branch_conditional(c, cont, merge, []).unwrap(),
            None => self.b.branch(cont).unwrap(),
        }
        self.b.begin_block(Some(cont)).unwrap();
        if let Some(carried) = self.loop_carried.get(&header_id).cloned() {
            for (vid, reserved, ty) in carried {
                let actual = match self.value_to_word.get(&vid).copied() {
                    Some(a) => a,
                    None => {
                        if ty == self.bool_t {
                            self.bool_false
                        } else {
                            self.f32_undef_id()
                        }
                    }
                };
                self.b.copy_object(ty, Some(reserved), actual).unwrap();
            }
        }
        let header = self.block_labels[&header_id];
        self.b.branch(header).unwrap();
        self.b.begin_block(Some(merge)).unwrap();
        let next = latch_id + 1;
        if let Some(&nl) = self.block_labels.get(&next) {
            self.b.branch(nl).unwrap();
        } else if let Some(rb) = self.return_block {
            self.b.branch(rb).unwrap();
        }
        self.block_end_labels.insert(latch_id, merge);
    }

    fn emit_terminator(&mut self, block: &BasicBlock) {
        match block.branch {
            BranchKind::Exit => {
                if let Some(rb) = self.return_block {
                    if matches!(self.stage, Stage::Fragment) {
                        let f0 = self.f32_zero;
                        let f1 = self.f32_one;
                        let es = block.program.exit_reg_state.as_ref();
                        let defaults = [f0, f0, f0, f1];
                        let mut outputs = self
                            .fragment_output_locations()
                            .into_iter()
                            .map(|loc| (loc, self.fragment_output_vec(es, loc, defaults)))
                            .collect::<Vec<_>>();
                        if let Ok(loc) = std::env::var("NEXIUM_FS_COLOR_LOC")
                            .ok()
                            .and_then(|v| v.parse::<u32>().ok())
                            .ok_or(())
                        {
                            let color = self.input_var(0x80 + loc.saturating_mul(16));
                            let forced = [
                                self.read_attr_component(color, 0),
                                self.read_attr_component(color, 1),
                                self.read_attr_component(color, 2),
                                self.read_attr_component(color, 3),
                            ];
                            if let Some((_, out)) = outputs.get_mut(0) {
                                *out = self
                                    .b
                                    .composite_construct(self.vec4_t, None, forced)
                                    .unwrap();
                            }
                        }
                        if let Ok(scale) = std::env::var("NEXIUM_FS_COLOR_SCALE")
                            .ok()
                            .and_then(|v| v.parse::<f32>().ok())
                            .ok_or(())
                        {
                            let s = self.const_f32(scale.to_bits());
                            let sv = self
                                .b
                                .composite_construct(self.vec4_t, None, [s, s, s, s])
                                .unwrap();
                            for (_, v) in &mut outputs {
                                *v = self.b.f_mul(self.vec4_t, None, *v, sv).unwrap();
                            }
                        }
                        if std::env::var("NEXIUM_FS_COLOR_ALPHA_ONE").ok().as_deref() == Some("1") {
                            for (_, v) in &mut outputs {
                                let r =
                                    self.b.composite_extract(self.f32_t, None, *v, [0]).unwrap();
                                let g =
                                    self.b.composite_extract(self.f32_t, None, *v, [1]).unwrap();
                                let b =
                                    self.b.composite_extract(self.f32_t, None, *v, [2]).unwrap();
                                *v = self
                                    .b
                                    .composite_construct(self.vec4_t, None, [r, g, b, self.f32_one])
                                    .unwrap();
                            }
                        }
                        if let Some(sample) = self.sample_debug_value {
                            if let Some((_, out)) = outputs.get_mut(0) {
                                *out = sample;
                            }
                        }
                        self.apply_fragment_output_debug_overrides(&mut outputs);
                        for (loc, v) in outputs {
                            self.store_fragment_output_vec(loc, v);
                        }
                    }
                    self.b.branch(rb).unwrap();
                }
            }
            BranchKind::Unconditional { target } => {
                if self
                    .self_loops
                    .get(&target)
                    .map_or(false, |&(_, _, _, latch)| latch == block.id)
                {
                    self.emit_self_loop_close(target, block.id, None);
                    return;
                }
                let lbl = self.merge_redirect(block.id, target);
                self.b.branch(lbl).unwrap();
            }
            BranchKind::Conditional { target, pred } => {
                if self
                    .self_loops
                    .get(&target)
                    .map_or(false, |&(_, _, _, latch)| latch == block.id)
                {
                    let cond = self.resolve_pred(pred.idx, pred.negate);
                    self.emit_self_loop_close(target, block.id, Some(cond));
                    return;
                }
                let next = block.id + 1;
                if self.block_labels.contains_key(&next) {
                    let cond = self.resolve_pred(pred.idx, pred.negate);
                    let true_lbl = self.merge_redirect(block.id, target);
                    let false_lbl = self.merge_redirect(block.id, next);
                    if target == next {
                        self.b.branch(true_lbl).unwrap();
                    } else {
                        let merge_lbl = match self.header_merge_label.get(&block.id).copied() {
                            Some(l) => l,
                            None => {
                                let merge_id = match &self.cond_merge {
                                    Some(ipd) => ipd[block.id as usize],
                                    None => next,
                                };
                                self.block_labels
                                    .get(&merge_id)
                                    .copied()
                                    .or(self.return_block)
                                    .unwrap_or(false_lbl)
                            }
                        };
                        assert!(
                            self.used_merge_blocks.insert(merge_lbl),
                            "nexium-spirv: shared selection merge block"
                        );
                        self.b
                            .selection_merge(merge_lbl, rspirv::spirv::SelectionControl::NONE)
                            .unwrap();
                        self.b
                            .branch_conditional(cond, true_lbl, false_lbl, [])
                            .unwrap();
                    }
                } else {
                    let true_lbl = self.block_labels[&target];
                    self.b.branch(true_lbl).unwrap();
                }
            }
            BranchKind::FallThrough => {
                let next = block.id + 1;
                if self.block_labels.contains_key(&next) {
                    let lbl = self.merge_redirect(block.id, next);
                    self.b.branch(lbl).unwrap();
                } else if let Some(rb) = self.return_block {
                    self.b.branch(rb).unwrap();
                }
            }
        }
    }

    fn emit_entry_inits(
        &mut self,
        required_outputs: &[(u32, AttrVar)],
        ps_inject: Option<(Word, u32)>,
    ) {
        if matches!(self.stage, Stage::Vertex) {
            for (_loc, av) in required_outputs {
                let defaults = [self.f32_zero, self.f32_zero, self.f32_zero, self.f32_one];
                for (c, value) in defaults.into_iter().enumerate() {
                    self.write_attr_component(*av, c as u32, value);
                }
            }
        }
        if let Some((v, bits)) = ps_inject {
            let c = self.const_f32(bits);
            self.b.store(v, c, None, []).unwrap();
        }
    }

    fn preallocate_resources(&mut self, cfg: &Cfg) {
        let mut needs_sampler = false;
        let mut needs_arrayed_sampler = false;
        let mut needs_3d_sampler = false;
        let mut tex_ids = std::collections::BTreeSet::new();
        for block in &cfg.blocks {
            for inst in &block.program.instructions {
                match &inst.op {
                    IrOp::LoadAttr { slot } => {
                        if self.system_attr_var_id(*slot).is_some() {
                            continue;
                        }
                        let aligned = slot & !0xF;
                        self.input_var(aligned);
                    }
                    IrOp::InterpAttr { slot, .. } => {
                        let aligned = slot & !0xF;
                        if aligned == 0x70 {
                            if matches!(self.stage, Stage::Fragment) {
                                self.frag_coord_var();
                            } else {
                                self.position_var();
                            }
                        } else {
                            self.input_var(aligned);
                        }
                    }
                    IrOp::StoreAttr { slot, .. } => {
                        let aligned = slot & !0xF;
                        if slot_is_gl_position(aligned) {
                            self.position_var();
                        } else if *slot == 0x6C && matches!(self.stage, Stage::Vertex) {
                            self.point_size_var_id();
                        } else if *slot >= 0x80 {
                            self.output_var(aligned);
                        }
                    }
                    IrOp::SampleTex {
                        tex_id,
                        array,
                        volume,
                        ..
                    } => {
                        needs_sampler = true;
                        needs_arrayed_sampler |= array.is_some();
                        needs_3d_sampler |= volume.is_some();
                        tex_ids.insert(*tex_id);
                    }
                    IrOp::GatherTex { tex_id, .. } => {
                        needs_sampler = true;
                        tex_ids.insert(*tex_id);
                    }
                    _ => {}
                }
            }
        }
        match self.stage {
            Stage::Vertex => {
                self.position_var();
            }
            Stage::Fragment => {
                for loc in self.fragment_output_locations() {
                    self.frag_color_var_at(loc);
                }
            }
        }
        if needs_sampler {
            self.sampler_arrayed = needs_arrayed_sampler;
            self.texs_ids_used.extend(tex_ids.iter().copied());
            if let Some(id) = std::env::var("NEXIUM_NO_KIL_TEX_ID")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
            {
                self.no_kil_shader = tex_ids.contains(&id);
            }
            self.texture_slots.clear();
            let tex_slot_base = self.vertex_opts.tex_slot_base;
            let tex_budget =
                (MAX_TEXTURE_DESCRIPTORS as usize).saturating_sub(tex_slot_base as usize);
            for (slot, tex_id) in tex_ids.iter().copied().take(tex_budget).enumerate() {
                self.texture_slots
                    .insert(tex_id, slot as u32 + tex_slot_base);
            }
            self.ensure_sampler_array();
            if needs_3d_sampler {
                self.ensure_image_3d_array();
            }
        }
        for block in &cfg.blocks {
            let id = self.b.id();
            self.block_labels.insert(block.id, id);
            self.block_end_labels.insert(block.id, id);
        }
    }

    pub fn finish(self, cfg: &Cfg) -> Vec<u32> {
        self.finish_with_required_outputs(cfg, &[])
    }

    pub fn finish_with_required_outputs_and_bindings(
        self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> (Vec<u32>, u32) {
        let (words, mask, _ids, _) = self.finish_inner(cfg, required_output_locations);
        (words, mask)
    }

    pub fn finish_full(
        self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> (Vec<u32>, u32, Vec<u32>) {
        let (words, mask, tex_ids, _) = self.finish_inner(cfg, required_output_locations);
        (words, mask, tex_ids)
    }

    pub fn finish_full_meta(
        self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> (Vec<u32>, u32, Vec<u32>, bool) {
        self.finish_inner(cfg, required_output_locations)
    }

    pub fn finish_with_required_outputs(
        self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> Vec<u32> {
        self.finish_inner(cfg, required_output_locations).0
    }

    fn finish_inner(
        mut self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> (Vec<u32>, u32, Vec<u32>, bool) {
        self.preallocate_resources(cfg);
        let sampler_arrayed = self.sampler_arrayed;

        let ps_inject: Option<(Word, u32)> = if matches!(self.stage, Stage::Vertex) {
            self.vertex_opts
                .point_size
                .map(|ps| (self.point_size_var_id(), ps.to_bits()))
        } else {
            None
        };

        let mut required_outputs: Vec<(u32, AttrVar)> = Vec::new();
        if matches!(self.stage, Stage::Vertex) {
            for &loc in required_output_locations {
                let slot = loc * 16 + 0x80;
                let av = self.output_var(slot);
                required_outputs.push((loc, av));
            }
        }

        let multi_exit = cfg
            .blocks
            .iter()
            .enumerate()
            .any(|(i, b)| matches!(b.branch, BranchKind::Exit) && i + 1 != cfg.blocks.len());

        let void_t = self.b.type_void();
        let main_t = self.b.type_function(void_t, vec![]);
        let main_id = self
            .b
            .begin_function(void_t, None, FunctionControl::NONE, main_t)
            .unwrap();

        if multi_exit && std::env::var_os("NEXIUM_NO_STRUCT_EXIT").is_none() {
            self.return_block = Some(self.b.id());
        }

        let entry_label = cfg.blocks.first().map(|b| self.block_labels[&b.id]);
        self.b.begin_block(entry_label).unwrap();
        self.emit_entry_inits(&required_outputs, ps_inject);

        self.cond_merge = structurizer_cond_merges(cfg);
        self.compute_shared_merges(cfg);
        self.lower_cfg(cfg);

        if let Some(rb) = self.return_block {
            self.b.begin_block(Some(rb)).unwrap();
        }

        match self.stage {
            Stage::Vertex => {
                if self.pos_var.is_none() {
                    let pos = self.position_var();
                    let z = self.f32_zero;
                    let o = self.f32_one;
                    let v = self
                        .b
                        .composite_construct(self.vec4_t, None, [z, z, z, o])
                        .unwrap();
                    self.b.store(pos, v, None, []).unwrap();
                }
                if self.vertex_opts.inject_ubo_matrix {
                    let pos_var = self.position_var();
                    let p = self.b.load(self.vec4_t, None, pos_var, None, []).unwrap();
                    let zero_u32 = self.const_u32(0);
                    let mut m = [[0u32; 4]; 4];
                    for col in 0u32..4 {
                        for row in 0u32..4 {
                            let vec4_idx = self.const_u32(col);
                            let comp_idx = self.const_u32(row);
                            let ac = self
                                .b
                                .access_chain(
                                    self.ptr_uniform_f32,
                                    None,
                                    self.ubo_var,
                                    [zero_u32, vec4_idx, comp_idx],
                                )
                                .unwrap();
                            m[col as usize][row as usize] =
                                self.b.load(self.f32_t, None, ac, None, []).unwrap();
                        }
                    }
                    let px = self.b.composite_extract(self.f32_t, None, p, [0]).unwrap();
                    let py = self.b.composite_extract(self.f32_t, None, p, [1]).unwrap();
                    let pz = self.b.composite_extract(self.f32_t, None, p, [2]).unwrap();
                    let pw = self.b.composite_extract(self.f32_t, None, p, [3]).unwrap();
                    let comps = [px, py, pz, pw];
                    let mut clip = [0u32; 4];
                    for row in 0..4 {
                        let mut acc = self.b.f_mul(self.f32_t, None, m[0][row], comps[0]).unwrap();
                        for col in 1..4 {
                            let term = self
                                .b
                                .f_mul(self.f32_t, None, m[col][row], comps[col])
                                .unwrap();
                            acc = self.b.f_add(self.f32_t, None, acc, term).unwrap();
                        }
                        clip[row] = acc;
                    }
                    let new_pos = self.b.composite_construct(self.vec4_t, None, clip).unwrap();
                    self.b.store(pos_var, new_pos, None, []).unwrap();
                    self.cbuf_bindings_used |= 1;
                }
                let do_z_remap = if std::env::var("NEXIUM_VS_Z_REMAP").ok().as_deref() == Some("1")
                {
                    true
                } else {
                    self.vertex_opts.apply_z_remap
                };
                let pos = self.position_var();
                let cur = self.b.load(self.vec4_t, None, pos, None, []).unwrap();
                let mut new_pos = cur;
                if do_z_remap {
                    let cur_z = self
                        .b
                        .composite_extract(self.f32_t, None, cur, [2])
                        .unwrap();
                    let cur_w = self
                        .b
                        .composite_extract(self.f32_t, None, cur, [3])
                        .unwrap();
                    let sum = self.b.f_add(self.f32_t, None, cur_z, cur_w).unwrap();
                    let half = self.const_f32(0.5f32.to_bits());
                    let new_z = self.b.f_mul(self.f32_t, None, sum, half).unwrap();
                    new_pos = self
                        .b
                        .composite_insert(self.vec4_t, None, new_z, cur, [2])
                        .unwrap();
                }
                let vs = self.vertex_opts.vptx_scale_z;
                let vt = self.vertex_opts.vptx_translate_z;
                if (vs - 1.0).abs() > 1e-6 || vt.abs() > 1e-6 {
                    let cur_z = self
                        .b
                        .composite_extract(self.f32_t, None, new_pos, [2])
                        .unwrap();
                    let cur_w = self
                        .b
                        .composite_extract(self.f32_t, None, new_pos, [3])
                        .unwrap();
                    let scale = self.const_f32(vs.to_bits());
                    let trans = self.const_f32(vt.to_bits());
                    let sz = self.b.f_mul(self.f32_t, None, scale, cur_z).unwrap();
                    let tw = self.b.f_mul(self.f32_t, None, trans, cur_w).unwrap();
                    let remapped_z = self.b.f_add(self.f32_t, None, sz, tw).unwrap();
                    new_pos = self
                        .b
                        .composite_insert(self.vec4_t, None, remapped_z, new_pos, [2])
                        .unwrap();
                }
                self.b.store(pos, new_pos, None, []).unwrap();
            }
            Stage::Fragment => {
                if self.return_block.is_none() {
                    let degenerate = cfg.blocks.is_empty()
                        || cfg.blocks.iter().all(|b| b.program.instructions.is_empty());
                    if degenerate {
                        log::warn!(
                            "spirv: degenerate Fragment CFG (blocks={}, all-empty) — \
                         emitting [0,0,0,1] fallback. If this is hot, the SASS \
                         decoder (nexium-shader::decode) is likely missing opcodes.",
                            cfg.blocks.len()
                        );
                    }
                    let exit_state = cfg
                        .blocks
                        .iter()
                        .find_map(|b| b.program.exit_reg_state.as_ref());
                    let alpha_recovery: Option<Word> =
                        exit_state.filter(|m| !m.contains_key(&3u8)).and_then(|m| {
                            for v in m.values() {
                                if let nexium_shader::ir::Value::Inst(vid) = v {
                                    let is_alpha = cfg.blocks.iter().any(|b| {
                                        b.program.instructions.iter().any(|inst| {
                                            inst.result == Some(*vid)
                                                && matches!(
                                                    inst.op,
                                                    nexium_shader::ir::Op::SampleTex {
                                                        component: 3,
                                                        ..
                                                    }
                                                )
                                        })
                                    });
                                    if is_alpha {
                                        return Some(self.lower_value(v));
                                    }
                                }
                            }
                            None
                        });
                    let f0 = self.f32_zero;
                    let f1 = self.f32_one;
                    let defaults: [Word; 4] = [f0, f0, f0, alpha_recovery.unwrap_or(f1)];
                    let mut outputs = self
                        .fragment_output_locations()
                        .into_iter()
                        .map(|loc| (loc, self.fragment_output_vec(exit_state, loc, defaults)))
                        .collect::<Vec<_>>();
                    if std::env::var("NEXIUM_FS_COLOR_ATTR").ok().as_deref() == Some("1") {
                        let color = self.input_var(0x80);
                        let forced = [
                            self.read_attr_component(color, 0),
                            self.read_attr_component(color, 1),
                            self.read_attr_component(color, 2),
                            self.read_attr_component(color, 3),
                        ];
                        if let Some((_, out)) = outputs.get_mut(0) {
                            *out = self
                                .b
                                .composite_construct(self.vec4_t, None, forced)
                                .unwrap();
                        }
                    }
                    if let Ok(loc) = std::env::var("NEXIUM_FS_COLOR_LOC")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .ok_or(())
                    {
                        let color = self.input_var(0x80 + loc.saturating_mul(16));
                        let forced = [
                            self.read_attr_component(color, 0),
                            self.read_attr_component(color, 1),
                            self.read_attr_component(color, 2),
                            self.read_attr_component(color, 3),
                        ];
                        if let Some((_, out)) = outputs.get_mut(0) {
                            *out = self
                                .b
                                .composite_construct(self.vec4_t, None, forced)
                                .unwrap();
                        }
                    }
                    if let Ok(scale) = std::env::var("NEXIUM_FS_COLOR_SCALE")
                        .ok()
                        .and_then(|v| v.parse::<f32>().ok())
                        .ok_or(())
                    {
                        let s = self.const_f32(scale.to_bits());
                        let sv = self
                            .b
                            .composite_construct(self.vec4_t, None, [s, s, s, s])
                            .unwrap();
                        for (_, v) in &mut outputs {
                            *v = self.b.f_mul(self.vec4_t, None, *v, sv).unwrap();
                        }
                    }
                    if std::env::var("NEXIUM_FS_COLOR_ALPHA_ONE").ok().as_deref() == Some("1") {
                        for (_, v) in &mut outputs {
                            let r = self.b.composite_extract(self.f32_t, None, *v, [0]).unwrap();
                            let g = self.b.composite_extract(self.f32_t, None, *v, [1]).unwrap();
                            let b = self.b.composite_extract(self.f32_t, None, *v, [2]).unwrap();
                            *v = self
                                .b
                                .composite_construct(self.vec4_t, None, [r, g, b, self.f32_one])
                                .unwrap();
                        }
                    }
                    if let Some(sample) = self.sample_debug_value {
                        if let Some((_, out)) = outputs.get_mut(0) {
                            *out = sample;
                        }
                    }
                    if std::env::var("NEXIUM_FRAG_2X").is_ok() {
                        let two = self.const_f32(2.0f32.to_bits());
                        let two_vec = self
                            .b
                            .composite_construct(self.vec4_t, None, [two, two, two, two])
                            .unwrap();
                        for (_, v) in &mut outputs {
                            *v = self.b.f_mul(self.vec4_t, None, *v, two_vec).unwrap();
                        }
                    }
                    self.apply_fragment_output_debug_overrides(&mut outputs);
                    for (loc, v) in outputs {
                        self.store_fragment_output_vec(loc, v);
                    }
                }
            }
        }

        self.b.ret().unwrap();
        self.b.end_function().unwrap();

        let exec_model = match self.stage {
            Stage::Vertex => ExecutionModel::Vertex,
            Stage::Fragment => ExecutionModel::Fragment,
        };
        self.b
            .entry_point(exec_model, main_id, "main", self.interface.clone());
        if self.stage == Stage::Fragment {
            self.b
                .execution_mode(main_id, rspirv::spirv::ExecutionMode::OriginUpperLeft, []);
        }

        let bindings = self.cbuf_bindings_used;
        let tex_ids: Vec<u32> = self.texs_ids_used.iter().copied().collect();
        let phi_cfg_lines = if std::env::var_os("NEXIUM_SPIRV_PHI_DBG").is_some() {
            self.phi_cfg_lines(cfg)
        } else {
            Vec::new()
        };
        let words = opt::dedup_constants(self.b.module().assemble());
        if !phi_preds_consistent(&words) {
            for line in phi_cfg_lines {
                log::warn!("{}", line);
            }
            panic!("nexium-spirv: invalid phi predecessors");
        }
        if std::env::var_os("NEXIUM_EXIT_GUARD").is_some() && !selection_exits_structured(&words) {
            panic!("nexium-spirv: unstructured selection exit");
        }
        if let Some(dir) = std::env::var_os("NEXIUM_DUMP_SPIRV") {
            use std::io::Write;
            let stage = match self.stage {
                Stage::Vertex => "vs",
                Stage::Fragment => "fs",
            };
            let mut h: u64 = 1469598103934665603;
            for w in &words {
                h ^= *w as u64;
                h = h.wrapping_mul(1099511628211);
            }
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("{}_{:016x}_me{}.spv", stage, h, multi_exit as u8));
            if let Ok(mut f) = std::fs::File::create(&path) {
                let mut bytes = Vec::with_capacity(words.len() * 4);
                for w in &words {
                    bytes.extend_from_slice(&w.to_le_bytes());
                }
                let _ = f.write_all(&bytes);
            }
        }
        (words, bindings, tex_ids, sampler_arrayed)
    }
}

fn phi_preds_consistent(words: &[u32]) -> bool {
    use rspirv::spirv::Op;
    let module = match rspirv::dr::load_words(words) {
        Ok(m) => m,
        Err(_) => return false,
    };
    for func in &module.functions {
        let mut succ: HashMap<Word, Vec<Word>> = HashMap::new();
        for block in &func.blocks {
            let Some(label) = block.label.as_ref().and_then(|l| l.result_id) else {
                continue;
            };
            let mut targets: Vec<Word> = Vec::new();
            if let Some(term) = block.instructions.last() {
                match term.class.opcode {
                    Op::Branch => {
                        if let Some(Operand::IdRef(t)) = term.operands.first() {
                            targets.push(*t);
                        }
                    }
                    Op::BranchConditional => {
                        for op in term.operands.iter().skip(1).take(2) {
                            if let Operand::IdRef(t) = op {
                                targets.push(*t);
                            }
                        }
                    }
                    Op::Switch => {
                        for op in term.operands.iter().skip(1) {
                            if let Operand::IdRef(t) = op {
                                targets.push(*t);
                            }
                        }
                    }
                    _ => {}
                }
            }
            succ.insert(label, targets);
        }
        let mut preds: HashMap<Word, std::collections::HashSet<Word>> = HashMap::new();
        for (b, ts) in &succ {
            for t in ts {
                preds.entry(*t).or_default().insert(*b);
            }
        }
        let empty = std::collections::HashSet::new();
        for block in &func.blocks {
            let Some(label) = block.label.as_ref().and_then(|l| l.result_id) else {
                continue;
            };
            let bpreds = preds.get(&label).unwrap_or(&empty);
            for inst in &block.instructions {
                if inst.class.opcode == Op::Phi {
                    let mut i = 1;
                    while i < inst.operands.len() {
                        if let Operand::IdRef(parent) = inst.operands[i] {
                            if !bpreds.contains(&parent) {
                                if std::env::var_os("NEXIUM_SPIRV_PHI_DBG").is_some() {
                                    let mut have: Vec<Word> = bpreds.iter().copied().collect();
                                    have.sort_unstable();
                                    log::warn!(
                                        "[spirv-phi] block={} parent={} preds={:?}",
                                        label,
                                        parent,
                                        have
                                    );
                                }
                                return false;
                            }
                        }
                        i += 2;
                    }
                }
            }
        }
    }
    true
}

fn selection_exits_structured(words: &[u32]) -> bool {
    use rspirv::spirv::Op;
    let module = match rspirv::dr::load_words(words) {
        Ok(m) => m,
        Err(_) => return false,
    };
    for func in &module.functions {
        let has_loop = func.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|i| i.class.opcode == Op::LoopMerge)
        });
        if has_loop {
            continue;
        }
        let mut index: HashMap<Word, usize> = HashMap::new();
        for (i, b) in func.blocks.iter().enumerate() {
            if let Some(l) = b.label.as_ref().and_then(|l| l.result_id) {
                index.insert(l, i);
            }
        }
        let mut constructs: Vec<(usize, usize)> = Vec::new();
        for (i, b) in func.blocks.iter().enumerate() {
            for inst in &b.instructions {
                if inst.class.opcode == Op::SelectionMerge {
                    if let Some(Operand::IdRef(m)) = inst.operands.first() {
                        if let Some(&mi) = index.get(m) {
                            constructs.push((i, mi));
                        }
                    }
                }
            }
        }
        if constructs.is_empty() {
            continue;
        }
        for (i, b) in func.blocks.iter().enumerate() {
            let mut targets: Vec<usize> = Vec::new();
            if let Some(term) = b.instructions.last() {
                let ops = match term.class.opcode {
                    Op::Branch => term.operands.iter().take(1).collect::<Vec<_>>(),
                    Op::BranchConditional => {
                        term.operands.iter().skip(1).take(2).collect::<Vec<_>>()
                    }
                    Op::Switch => term.operands.iter().skip(1).collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                for op in ops {
                    if let Operand::IdRef(t) = op {
                        if let Some(&ti) = index.get(t) {
                            targets.push(ti);
                        }
                    }
                }
            }
            for &(h, m) in &constructs {
                if i > h && i < m {
                    for &t in &targets {
                        if t <= h || t > m {
                            return false;
                        }
                    }
                }
            }
        }
    }
    true
}

fn dominators(cfg: &Cfg) -> Vec<Vec<bool>> {
    let n = cfg.blocks.len();
    let preds = cfg.predecessors();
    let mut dom = vec![vec![true; n]; n];
    if n == 0 {
        return dom;
    }
    dom[0].fill(false);
    dom[0][0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..n {
            let mut next = vec![true; n];
            if preds[b].is_empty() {
                next.fill(false);
            } else {
                for &p in &preds[b] {
                    for d in 0..n {
                        next[d] &= dom[p as usize][d];
                    }
                }
            }
            next[b] = true;
            if next != dom[b] {
                dom[b] = next;
                changed = true;
            }
        }
    }
    dom
}

fn structurizer_cond_merges(cfg: &Cfg) -> Option<Vec<u32>> {
    let n = cfg.blocks.len();
    if n <= 1 {
        return None;
    }
    let virt = n as u32;
    let total = n + 1;
    let mut succs = vec![Vec::<u32>::new(); total];
    for (i, block) in cfg.blocks.iter().enumerate() {
        succs[i] = if matches!(block.branch, BranchKind::Exit) {
            vec![virt]
        } else {
            cfg.successors(block.id)
        };
        if succs[i].is_empty() {
            succs[i].push(virt);
        }
    }
    succs[n].push(virt);

    let mut pdom = vec![vec![true; total]; total];
    pdom[n].fill(false);
    pdom[n][n] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut next = vec![true; total];
            for &s in &succs[b] {
                for d in 0..total {
                    next[d] &= pdom[s as usize][d];
                }
            }
            next[b] = true;
            if next != pdom[b] {
                pdom[b] = next;
                changed = true;
            }
        }
    }

    let mut ipdom = vec![virt; total];
    ipdom[n] = virt;
    for b in 0..n {
        let strict = (0..total)
            .filter(|&c| c != b && pdom[b][c])
            .collect::<Vec<_>>();
        if strict.is_empty() {
            return None;
        }
        ipdom[b] = strict
            .iter()
            .copied()
            .find(|&c| strict.iter().all(|&d| d == c || pdom[c][d]))
            .map(|c| c as u32)
            .unwrap_or(virt);
    }
    Some(ipdom)
}

pub fn emit_vertex(cfg: &Cfg) -> Vec<u32> {
    Emitter::new(Stage::Vertex).finish(cfg)
}

pub fn emit_vertex_with_required_outputs(cfg: &Cfg, required_outputs: &[u32]) -> Vec<u32> {
    Emitter::new(Stage::Vertex).finish_with_required_outputs(cfg, required_outputs)
}

pub fn emit_fragment(cfg: &Cfg) -> Vec<u32> {
    Emitter::new(Stage::Fragment).finish(cfg)
}

pub fn emit_vertex_with_bindings(cfg: &Cfg, required_outputs: &[u32]) -> (Vec<u32>, u32) {
    Emitter::new(Stage::Vertex).finish_with_required_outputs_and_bindings(cfg, required_outputs)
}

#[derive(Clone, Copy, Debug)]
pub struct VertexOptions {
    pub apply_z_remap: bool,
    pub vptx_scale_z: f32,
    pub vptx_translate_z: f32,
    pub inject_ubo_matrix: bool,
    pub point_size: Option<f32>,
    pub window_ndc: Option<(f32, f32)>,
    pub num_ssbo: u32,
    pub uint_attr_mask: u32,
    pub sint_attr_mask: u32,
    pub tex_slot_base: u32,
}

impl Default for VertexOptions {
    fn default() -> Self {
        Self {
            apply_z_remap: false,
            vptx_scale_z: 1.0,
            vptx_translate_z: 0.0,
            inject_ubo_matrix: false,
            point_size: None,
            window_ndc: None,
            num_ssbo: 0,
            uint_attr_mask: 0,
            sint_attr_mask: 0,
            tex_slot_base: 0,
        }
    }
}

fn cbuf_vec4s(cfg: &Cfg, floor_vec4s: u32) -> u32 {
    let mut max_byte = 0u32;
    for block in &cfg.blocks {
        for inst in &block.program.instructions {
            if let IrOp::LoadCbuf { byte_offset, .. } = &inst.op {
                max_byte = max_byte.max(byte_offset.saturating_add(4));
            }
        }
    }
    let vec4s = ((max_byte + 15) / 16).max(floor_vec4s).max(16);
    ((vec4s + 15) & !15).min(UBO_VEC4S)
}

pub fn emit_vertex_with_bindings_opts(
    cfg: &Cfg,
    required_outputs: &[u32],
    opts: VertexOptions,
) -> (Vec<u32>, u32, u32) {
    let vec4s = cbuf_vec4s(cfg, if opts.inject_ubo_matrix { 4 } else { 1 }).max(UBO_VEC4S);
    let (words, mask) = Emitter::new_with_vertex_opts_sized(Stage::Vertex, opts, vec4s)
        .finish_with_required_outputs_and_bindings(cfg, required_outputs);
    (words, mask, vec4s * 16)
}

pub fn emit_fragment_with_bindings(cfg: &Cfg) -> (Vec<u32>, u32) {
    Emitter::new(Stage::Fragment).finish_with_required_outputs_and_bindings(cfg, &[])
}

pub fn emit_fragment_full(cfg: &Cfg) -> (Vec<u32>, u32, Vec<u32>, u32) {
    emit_fragment_full_with_input_map(cfg, [0; 32])
}

pub fn emit_fragment_full_with_input_map(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
) -> (Vec<u32>, u32, Vec<u32>, u32) {
    let (words, mask, tex_ids, cbuf_size, _) =
        emit_fragment_full_with_input_map_meta(cfg, ps_input_map);
    (words, mask, tex_ids, cbuf_size)
}

pub fn emit_fragment_full_with_input_map_meta(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
) -> (Vec<u32>, u32, Vec<u32>, u32, bool) {
    emit_fragment_full_with_input_map_meta_outputs(cfg, ps_input_map, 1, 0)
}

pub fn emit_fragment_full_with_input_map_meta_outputs(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
    color_outputs: u32,
    output_map: u32,
) -> (Vec<u32>, u32, Vec<u32>, u32, bool) {
    emit_fragment_full_with_input_map_meta_outputs_debug(
        cfg,
        ps_input_map,
        color_outputs,
        output_map,
        true,
    )
}

pub fn emit_fragment_full_with_input_map_meta_outputs_debug(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
    color_outputs: u32,
    output_map: u32,
    debug_active: bool,
) -> (Vec<u32>, u32, Vec<u32>, u32, bool) {
    let vec4s = cbuf_vec4s(cfg, 1).max(UBO_VEC4S);
    let mut emitter = Emitter::new_sized(Stage::Fragment, vec4s);
    if !debug_active {
        emitter.sample_debug_slot = None;
        emitter.texcoord_debug_slot = None;
        emitter.tex_v_flip_slots.clear();
    }
    emitter.ps_input_map = ps_input_map;
    emitter.fragment_color_outputs = color_outputs.max(1).min(8);
    emitter.fragment_output_map = output_map;
    let (words, mask, tex_ids, sampler_arrayed) = emitter.finish_full_meta(cfg, &[]);
    (words, mask, tex_ids, vec4s * 16, sampler_arrayed)
}

pub fn scan_input_locations(words: &[u32]) -> Vec<u32> {
    use std::collections::{HashMap, HashSet};
    if words.len() < 5 || words[0] != 0x07230203 {
        return Vec::new();
    }
    let mut input_var_ids: HashSet<u32> = HashSet::new();
    let mut id_to_location: HashMap<u32, u32> = HashMap::new();
    let mut i = 5;
    while i < words.len() {
        let w0 = words[i];
        let word_count = (w0 >> 16) as usize;
        let opcode = w0 & 0xFFFF;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            71 => {
                if word_count >= 4 {
                    let target = words[i + 1];
                    let decoration = words[i + 2];
                    if decoration == 30 && word_count >= 4 {
                        id_to_location.insert(target, words[i + 3]);
                    }
                }
            }
            59 => {
                if word_count >= 4 {
                    let result_id = words[i + 2];
                    let storage_class = words[i + 3];
                    if storage_class == 1 {
                        input_var_ids.insert(result_id);
                    }
                }
            }
            _ => {}
        }
        i += word_count;
    }
    let mut locations: Vec<u32> = input_var_ids
        .iter()
        .filter_map(|id| id_to_location.get(id).copied())
        .collect();
    locations.sort_unstable();
    locations.dedup();
    locations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_fmul_reg(rd: u8, ra: u8, rb: u8) -> u64 {
        0x5C68_1000_0000_0000u64
            | ((rb as u64) << 20)
            | ((ra as u64) << 8)
            | (rd as u64)
            | 0x0007_0000
    }
    fn enc_exit() -> u64 {
        0xE300_0000_0007_000Fu64
    }
    fn build_test_program(words: &[u64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in words.chunks(3) {
            bytes.extend_from_slice(&[0u8; 8]);
            for w in chunk {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        bytes
    }

    fn validates_with_naga(words: &[u32]) -> naga::Module {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = naga::front::spv::parse_u8_slice(&bytes, &naga::front::spv::Options::default())
            .unwrap_or_else(|e| panic!("naga parse failed: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap_or_else(|e| panic!("naga validate failed: {e:?}"));
        m
    }

    #[test]
    fn empty_vertex_passes_naga_validation() {
        let bytes = build_test_program(&[enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_vertex(&cfg);
        let m = validates_with_naga(&words);
        assert_eq!(m.entry_points.len(), 1);
        assert_eq!(m.entry_points[0].stage, naga::ShaderStage::Vertex);
    }

    #[test]
    fn empty_fragment_passes_naga_validation() {
        let bytes = build_test_program(&[enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_fragment(&cfg);
        let m = validates_with_naga(&words);
        assert_eq!(m.entry_points.len(), 1);
        assert_eq!(m.entry_points[0].stage, naga::ShaderStage::Fragment);
    }

    #[test]
    fn vertex_with_alu_passes_naga_validation() {
        let bytes = build_test_program(&[enc_fmul_reg(2, 0, 1), enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_vertex(&cfg);
        validates_with_naga(&words);
    }

    #[test]
    fn scan_input_locations_recovers_emitted_slots() {
        let bytes = build_test_program(&[0xEFD8_FF80_0807_FF00u64, enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_vertex(&cfg);
        let locs = scan_input_locations(&words);
        assert_eq!(locs, vec![0]);
    }

    #[test]
    fn scan_input_locations_rejects_garbage_input() {
        assert!(scan_input_locations(&[]).is_empty());
        assert!(scan_input_locations(&[0xDEAD_BEEF, 0, 0, 0, 0]).is_empty());
    }
}

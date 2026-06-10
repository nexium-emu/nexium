#![deny(unsafe_op_in_unsafe_fn)]

mod opt;

use std::collections::HashMap;

use nexium_shader::{
    BasicBlock, BlockId, BoolOp, BranchKind, Cfg, FComp, ICmp, IrInst, IrOp, IrValue, MufuFunc, ValueId,
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

pub struct Emitter {
    b: rspirv::dr::Builder,
    stage: Stage,
    f32_t: Word,
    vec2_t: Word,
    vec4_t: Word,
    u32_t: Word,
    i32_t: Word,
    ptr_uniform_f32: Word,
    ptr_input_vec4: Word,
    ptr_input_f32: Word,
    ptr_output_vec4: Word,
    ptr_output_f32: Word,
    ptr_image: Word,
    ptr_sampler: Word,
    ubo_var: Word,
    f32_zero: Word,
    f32_one: Word,
    glsl: Word,
    input_vars: HashMap<u32, AttrVar>,
    output_vars: HashMap<u32, AttrVar>,
    pos_var: Option<Word>,
    point_size_var: Option<Word>,
    frag_coord_var: Option<Word>,
    frag_color_var: Option<Word>,
    image_var: Option<Word>,
    sampler_var: Option<Word>,
    image_t: Word,
    sampler_t: Word,
    sampled_image_t: Word,
    interface: Vec<Word>,
    value_to_word: HashMap<ValueId, Word>,
    block_labels: HashMap<BlockId, Word>,
    cbuf_bindings_used: u32,
    texs_ids_used: std::collections::BTreeSet<u32>,
    vertex_opts: VertexOptions,
    const_cache_f32: HashMap<u32, Word>,
    const_cache_u32: HashMap<u32, Word>,
    bool_t: Word,
    bool_true: Word,
    bool_false: Word,
    pred_regs: [Option<Word>; 7],
    ubo_vec4s: u32,
}

#[derive(Clone, Copy)]
struct AttrVar {
    var: Word,
    ptr_f32: Word,
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
        let vec4_t = b.type_vector(f32_t, 4);
        let u32_t = b.type_int(32, 0);
        let i32_t = b.type_int(32, 1);
        let ptr_uniform_f32 = b.type_pointer(None, StorageClass::Uniform, f32_t);
        let ptr_input_vec4 = b.type_pointer(None, StorageClass::Input, vec4_t);
        let ptr_input_f32 = b.type_pointer(None, StorageClass::Input, f32_t);
        let ptr_output_vec4 = b.type_pointer(None, StorageClass::Output, vec4_t);
        let ptr_output_f32 = b.type_pointer(None, StorageClass::Output, f32_t);

        let ubo_vec4s_const = b.constant_bit32(u32_t, ubo_vec4s);
        let vec4_arr = b.type_array(vec4_t, ubo_vec4s_const);
        b.decorate(vec4_arr, Decoration::ArrayStride, [Operand::LiteralBit32(16)]);
        let ubo_struct = b.type_struct([vec4_arr]);
        b.decorate(ubo_struct, Decoration::Block, []);
        b.member_decorate(ubo_struct, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        let ptr_uniform_struct = b.type_pointer(None, StorageClass::Uniform, ubo_struct);
        let ubo_var = b.variable(ptr_uniform_struct, None, StorageClass::Uniform, None);
        b.decorate(ubo_var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
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
        let sampler_t = b.type_sampler();
        let sampled_image_t = b.type_sampled_image(image_t);
        let ptr_image = b.type_pointer(None, StorageClass::UniformConstant, image_t);
        let ptr_sampler = b.type_pointer(None, StorageClass::UniformConstant, sampler_t);

        let f32_zero = b.constant_bit32(f32_t, 0.0f32.to_bits());
        let f32_one = b.constant_bit32(f32_t, 1.0f32.to_bits());

        let mut const_cache_f32: HashMap<u32, Word> = HashMap::new();
        let const_cache_u32: HashMap<u32, Word> = HashMap::new();
        const_cache_f32.insert(0.0f32.to_bits(), f32_zero);
        const_cache_f32.insert(1.0f32.to_bits(), f32_one);

        let bool_t = b.type_bool();
        let bool_true  = b.constant_true(bool_t);
        let bool_false = b.constant_false(bool_t);

        Self {
            b,
            stage,
            f32_t,
            vec2_t,
            vec4_t,
            u32_t,
            i32_t,
            ptr_uniform_f32,
            ptr_input_vec4,
            ptr_input_f32,
            ptr_output_vec4,
            ptr_output_f32,
            ptr_image,
            ptr_sampler,
            ubo_var,
            f32_zero,
            f32_one,
            glsl,
            input_vars: HashMap::new(),
            output_vars: HashMap::new(),
            pos_var: None,
            point_size_var: None,
            frag_coord_var: None,
            frag_color_var: None,
            image_var: None,
            sampler_var: None,
            image_t,
            sampler_t,
            sampled_image_t,
            interface: vec![ubo_var],
            value_to_word: HashMap::new(),
            block_labels: HashMap::new(),
            cbuf_bindings_used: 0,
            texs_ids_used: std::collections::BTreeSet::new(),
            vertex_opts: VertexOptions::default(),
            const_cache_f32,
            const_cache_u32,
            bool_t,
            bool_true,
            bool_false,
            pred_regs: [None; 7],
            ubo_vec4s,
        }
    }

    pub fn new_with_vertex_opts(stage: Stage, vertex_opts: VertexOptions) -> Self {
        let mut e = Self::new(stage);
        e.vertex_opts = vertex_opts;
        e
    }

    fn new_with_vertex_opts_sized(stage: Stage, vertex_opts: VertexOptions, ubo_vec4s: u32) -> Self {
        let mut e = Self::new_sized(stage, ubo_vec4s);
        e.vertex_opts = vertex_opts;
        e
    }

    fn position_var(&mut self) -> Word {
        if let Some(v) = self.pos_var {
            return v;
        }
        let v = self.b.variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::Position)]);
        self.interface.push(v);
        self.pos_var = Some(v);
        v
    }

    fn point_size_var_id(&mut self) -> Word {
        if let Some(v) = self.point_size_var {
            return v;
        }
        let v = self.b.variable(self.ptr_output_f32, None, StorageClass::Output, None);
        self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::PointSize)]);
        self.interface.push(v);
        self.point_size_var = Some(v);
        v
    }

    fn frag_coord_var(&mut self) -> Word {
        if let Some(v) = self.frag_coord_var {
            return v;
        }
        let v = self.b.variable(self.ptr_input_vec4, None, StorageClass::Input, None);
        self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::FragCoord)]);
        self.interface.push(v);
        self.frag_coord_var = Some(v);
        v
    }

    fn frag_color_var_id(&mut self) -> Word {
        if let Some(v) = self.frag_color_var {
            return v;
        }
        let v = self.b.variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(0)]);
        self.interface.push(v);
        self.frag_color_var = Some(v);
        v
    }

    fn input_var(&mut self, slot: u32) -> AttrVar {
        if let Some(av) = self.input_vars.get(&slot) {
            return *av;
        }
        let var = self.b.variable(self.ptr_input_vec4, None, StorageClass::Input, None);
        let location = if slot >= 0x80 { (slot - 0x80) / 16 } else { 0 };
        self.b.decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
        let av = AttrVar { var, ptr_f32: self.ptr_input_f32 };
        self.input_vars.insert(slot, av);
        self.interface.push(var);
        av
    }

    fn output_var(&mut self, slot: u32) -> AttrVar {
        if let Some(av) = self.output_vars.get(&slot) {
            return *av;
        }
        let var = self.b.variable(self.ptr_output_vec4, None, StorageClass::Output, None);
        let location = if slot >= 0x80 { (slot - 0x80) / 16 } else { 0 };
        self.b.decorate(var, Decoration::Location, [Operand::LiteralBit32(location)]);
        let av = AttrVar { var, ptr_f32: self.ptr_output_f32 };
        self.output_vars.insert(slot, av);
        self.interface.push(var);
        av
    }

    fn sampler(&mut self) -> (Word, Word) {
        if let (Some(img), Some(samp)) = (self.image_var, self.sampler_var) {
            return (img, samp);
        }
        let img = self
            .b
            .variable(self.ptr_image, None, StorageClass::UniformConstant, None);
        self.b
            .decorate(img, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b
            .decorate(img, Decoration::Binding, [Operand::LiteralBit32(1)]);
        let samp = self
            .b
            .variable(self.ptr_sampler, None, StorageClass::UniformConstant, None);
        self.b
            .decorate(samp, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b
            .decorate(samp, Decoration::Binding, [Operand::LiteralBit32(2)]);
        self.image_var = Some(img);
        self.sampler_var = Some(samp);
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
            r = self.b.ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(r)]).unwrap();
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
        let idx = self.const_u32(component);
        let ac = self.b.access_chain(av.ptr_f32, None, av.var, [idx]).unwrap();
        self.b.load(self.f32_t, None, ac, None, []).unwrap()
    }

    fn write_attr_component(&mut self, av: AttrVar, component: u32, val: Word) {
        let idx = self.const_u32(component);
        let ac = self.b.access_chain(av.ptr_f32, None, av.var, [idx]).unwrap();
        self.b.store(ac, val, None, []).unwrap();
    }

    fn lower_fcompare(&mut self, cmp: &FComp, va: Word, vb: Word) -> Word {
        match cmp {
            FComp::F   => self.bool_false,
            FComp::T   => self.bool_true,
            FComp::Lt  => self.b.f_ord_less_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Eq  => self.b.f_ord_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Le  => self.b.f_ord_less_than_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Gt  => self.b.f_ord_greater_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Ne  => self.b.f_ord_not_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Ge  => self.b.f_ord_greater_than_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Num => self.b.ordered(self.bool_t, None, va, vb).unwrap(),
            FComp::Nan => self.b.unordered(self.bool_t, None, va, vb).unwrap(),
            FComp::Ltu => self.b.f_unord_less_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Equ => self.b.f_unord_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Leu => self.b.f_unord_less_than_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Gtu => self.b.f_unord_greater_than(self.bool_t, None, va, vb).unwrap(),
            FComp::Neu => self.b.f_unord_not_equal(self.bool_t, None, va, vb).unwrap(),
            FComp::Geu => self.b.f_unord_greater_than_equal(self.bool_t, None, va, vb).unwrap(),
        }
    }

    fn as_i32(&mut self, w: Word) -> Word {
        self.b.bitcast(self.i32_t, None, w).unwrap()
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
                    self.b.u_less_than_equal(self.bool_t, None, a_u, b_u).unwrap()
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
                    self.b.s_greater_than_equal(self.bool_t, None, a, b).unwrap()
                } else {
                    self.b.u_greater_than_equal(self.bool_t, None, a_u, b_u).unwrap()
                }
            }
        }
    }

    fn lower_op(&mut self, inst: &IrInst) {
        let result = inst.result;
        let word = match &inst.op {
            IrOp::Mov(src) => Some(self.lower_value(src)),
            IrOp::FMul { a, b, mods } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let av = self.apply_neg_abs(av, mods.neg_a, mods.abs_a);
                let bv = self.apply_neg_abs(bv, mods.neg_b, mods.abs_b);
                let mut r = self.b.f_mul(self.f32_t, None, av, bv).unwrap();
                if mods.scale != 0 {
                    let exp = if mods.scale < 4 { mods.scale as i32 } else { mods.scale as i32 - 8 };
                    let factor = 2.0f32.powi(exp);
                    static FMUL_SCALE_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if FMUL_SCALE_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                        log::warn!("[fmul-scale] applying field={} factor={}", mods.scale, factor);
                    }
                    let fc = self.const_f32(factor.to_bits());
                    r = self.b.f_mul(self.f32_t, None, r, fc).unwrap();
                }
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
                        .ext_inst(f32_t, None, glsl, 37, [Operand::IdRef(av), Operand::IdRef(bv)])
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
                        .ext_inst(f32_t, None, glsl, 40, [Operand::IdRef(av), Operand::IdRef(bv)])
                        .unwrap(),
                )
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
            IrOp::LoadCbuf { binding, byte_offset } => {
                self.cbuf_bindings_used |= 1u32 << (binding & 0x1F);
                let vec4_index = (byte_offset / 16) % self.ubo_vec4s;
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
            IrOp::LoadAttr { slot } => {
                let component = (slot & 0xC) >> 2;
                let aligned_slot = slot & !0xF;
                let av = self.input_var(aligned_slot);
                Some(self.read_attr_component(av, component))
            }
            IrOp::InterpAttr { slot, perspective: _ } => {
                let component = (slot & 0xC) >> 2;
                let aligned_slot = slot & !0xF;
                if aligned_slot == 0x70 {
                    let fc = self.frag_coord_var();
                    let idx = self.const_u32(component);
                    let ac = self.b.access_chain(self.ptr_input_f32, None, fc, [idx]).unwrap();
                    let val = self.b.load(self.f32_t, None, ac, None, []).unwrap();
                    Some(val)
                } else {
                    let av = self.input_var(aligned_slot);
                    Some(self.read_attr_component(av, component))
                }
            }
            IrOp::StoreAttr { slot, src } => {
                let val = self.lower_value(src);
                let component = (slot & 0xC) >> 2;
                let aligned_slot = slot & !0xF;
                if slot_is_gl_position(aligned_slot) {
                    let val = match (self.vertex_opts.window_ndc, component) {
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
                    let ac = self.b.access_chain(self.ptr_output_f32, None, pos, [idx]).unwrap();
                    self.b.store(ac, val, None, []).unwrap();
                } else if *slot == 0x6C && matches!(self.stage, Stage::Vertex) {
                    let v = self.point_size_var_id();
                    self.b.store(v, val, None, []).unwrap();
                } else if *slot < 0x80 {
                    let _ = (val, component);
                } else {
                    let av = self.output_var(aligned_slot);
                    self.write_attr_component(av, component, val);
                }
                None
            }
            IrOp::SampleTex { tex_id, u, v, component } => {
                self.texs_ids_used.insert(*tex_id);
                let uv0 = self.lower_value(u);
                let uv1 = self.lower_value(v);
                let coords = self
                    .b
                    .composite_construct(self.vec2_t, None, [uv0, uv1])
                    .unwrap();
                let (img_var, samp_var) = self.sampler();
                let img = self.b.load(self.image_t, None, img_var, None, []).unwrap();
                let samp = self.b.load(self.sampler_t, None, samp_var, None, []).unwrap();
                let sampled_img = self
                    .b
                    .sampled_image(self.sampled_image_t, None, img, samp)
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
            IrOp::FSetPred {
                cmp, bop, src_a, src_b,
                neg_a, abs_a, neg_b, abs_b,
                src_pred, src_pred_inv, dest_p, dest_np,
            } => {
                let mut va = self.lower_value(src_a);
                let mut vb = self.lower_value(src_b);
                if *abs_a {
                    va = self.b.ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(va)]).unwrap();
                }
                if *neg_a {
                    let neg_one = self.const_f32((-1.0f32).to_bits());
                    va = self.b.f_mul(self.f32_t, None, va, neg_one).unwrap();
                }
                if *abs_b {
                    vb = self.b.ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(vb)]).unwrap();
                }
                if *neg_b {
                    let neg_one = self.const_f32((-1.0f32).to_bits());
                    vb = self.b.f_mul(self.f32_t, None, vb, neg_one).unwrap();
                }

                let cmp_result = match cmp {
                    FComp::F   => self.bool_false,
                    FComp::T   => self.bool_true,
                    FComp::Lt  => self.b.f_ord_less_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Eq  => self.b.f_ord_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Le  => self.b.f_ord_less_than_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Gt  => self.b.f_ord_greater_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Ne  => self.b.f_ord_not_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Ge  => self.b.f_ord_greater_than_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Num => self.b.ordered(self.bool_t, None, va, vb).unwrap(),
                    FComp::Nan => self.b.unordered(self.bool_t, None, va, vb).unwrap(),
                    FComp::Ltu => self.b.f_unord_less_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Equ => self.b.f_unord_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Leu => self.b.f_unord_less_than_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Gtu => self.b.f_unord_greater_than(self.bool_t, None, va, vb).unwrap(),
                    FComp::Neu => self.b.f_unord_not_equal(self.bool_t, None, va, vb).unwrap(),
                    FComp::Geu => self.b.f_unord_greater_than_equal(self.bool_t, None, va, vb).unwrap(),
                };

                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self.b.logical_and(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Or  => self.b.logical_or(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Xor => self.b.logical_not_equal(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                };

                if *dest_p < 7 {
                    self.pred_regs[*dest_p as usize] = Some(combined);
                }
                if *dest_np < 7 {
                    let bt = self.bool_t;
                    let inv = self.b.logical_not(bt, None, combined).unwrap();
                    self.pred_regs[*dest_np as usize] = Some(inv);
                }
                Some(combined)
            }

            IrOp::F2F { src, neg, abs, sat, round } => {
                let v = self.lower_value(src);
                let v = self.apply_neg_abs(v, *neg, *abs);
                let v = match *round {
                    1 => self.b.ext_inst(self.f32_t, None, self.glsl, 2, [Operand::IdRef(v)]).unwrap(),
                    2 => self.b.ext_inst(self.f32_t, None, self.glsl, 8, [Operand::IdRef(v)]).unwrap(),
                    3 => self.b.ext_inst(self.f32_t, None, self.glsl, 9, [Operand::IdRef(v)]).unwrap(),
                    4 => self.b.ext_inst(self.f32_t, None, self.glsl, 3, [Operand::IdRef(v)]).unwrap(),
                    _ => v,
                };
                Some(self.apply_sat(v, *sat))
            }
            IrOp::I2F { src, signed, neg, abs, int_format, selector } => {
                let f = self.lower_value(src);
                let bits_u = self.b.bitcast(self.u32_t, None, f).unwrap();
                let extracted = match *int_format {
                    0 => {
                        let off = self.const_u32((*selector as u32) * 8);
                        let cnt = self.const_u32(8);
                        if *signed {
                            self.b.bit_field_s_extract(self.u32_t, None, bits_u, off, cnt).unwrap()
                        } else {
                            self.b.bit_field_u_extract(self.u32_t, None, bits_u, off, cnt).unwrap()
                        }
                    }
                    1 => {
                        let off = self.const_u32((*selector as u32) * 8);
                        let cnt = self.const_u32(16);
                        if *signed {
                            self.b.bit_field_s_extract(self.u32_t, None, bits_u, off, cnt).unwrap()
                        } else {
                            self.b.bit_field_u_extract(self.u32_t, None, bits_u, off, cnt).unwrap()
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
                    val = self.b.ext_inst(self.f32_t, None, self.glsl, 4, [Operand::IdRef(val)]).unwrap();
                }
                if *neg {
                    let n = self.const_f32((-1.0f32).to_bits());
                    val = self.b.f_mul(self.f32_t, None, val, n).unwrap();
                }
                Some(val)
            }
            IrOp::FSet {
                cmp, bop, src_a, src_b,
                neg_a, abs_a, neg_b, abs_b,
                src_pred, src_pred_inv,
            } => {
                let va = self.lower_value(src_a);
                let vb = self.lower_value(src_b);
                let va = self.apply_neg_abs(va, *neg_a, *abs_a);
                let vb = self.apply_neg_abs(vb, *neg_b, *abs_b);
                let cmp_result = self.lower_fcompare(cmp, va, vb);
                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self.b.logical_and(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Or  => self.b.logical_or(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Xor => self.b.logical_not_equal(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                };
                let one = self.f32_one;
                let zero = self.f32_zero;
                Some(self.b.select(self.f32_t, None, combined, one, zero).unwrap())
            }
            IrOp::ISetPred {
                cmp, signed, bop, src_a, src_b,
                src_pred, src_pred_inv, dest_p, dest_np,
            } => {
                let fa = self.lower_value(src_a);
                let fb = self.lower_value(src_b);
                let a_u = self.b.bitcast(self.u32_t, None, fa).unwrap();
                let b_u = self.b.bitcast(self.u32_t, None, fb).unwrap();
                let cmp_result = self.lower_icompare(cmp, *signed, a_u, b_u);
                let src_p_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let combined = match bop {
                    BoolOp::And => self.b.logical_and(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Or  => self.b.logical_or(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                    BoolOp::Xor => self.b.logical_not_equal(self.bool_t, None, cmp_result, src_p_word).unwrap(),
                };
                if *dest_p < 7 {
                    self.pred_regs[*dest_p as usize] = Some(combined);
                }
                if *dest_np < 7 {
                    let not_cmp = self.b.logical_not(self.bool_t, None, cmp_result).unwrap();
                    let combined_np = match bop {
                        BoolOp::And => self.b.logical_and(self.bool_t, None, not_cmp, src_p_word).unwrap(),
                        BoolOp::Or  => self.b.logical_or(self.bool_t, None, not_cmp, src_p_word).unwrap(),
                        BoolOp::Xor => self.b.logical_not_equal(self.bool_t, None, not_cmp, src_p_word).unwrap(),
                    };
                    self.pred_regs[*dest_np as usize] = Some(combined_np);
                }
                Some(combined)
            }

            IrOp::Kill => {
                let kill_block = self.b.id();
                let merge_block = self.b.id();

                let cond = if let Some(pred_guard) = &inst.pred {
                    self.resolve_pred(pred_guard.idx, pred_guard.negate)
                } else {
                    self.bool_true
                };

                self.b.selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE).unwrap();
                self.b.branch_conditional(cond, kill_block, merge_block, []).unwrap();
                self.b.begin_block(Some(kill_block)).unwrap();
                self.b.kill().unwrap();
                self.b.begin_block(Some(merge_block)).unwrap();
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

    fn lower_cfg(&mut self, cfg: &Cfg) {
        let single_block = cfg.blocks.len() <= 1;
        for (idx, block) in cfg.blocks.iter().enumerate() {
            let is_first = idx == 0;
            if !is_first {
                let label = self.block_labels[&block.id];
                self.b.begin_block(Some(label)).unwrap();
            }
            self.lower_phis(block);
            for inst in &block.program.instructions {
                if matches!(inst.op, IrOp::Phi { .. }) {
                    continue;
                }
                self.lower_op(inst);
            }
            if !single_block {
                self.emit_terminator(block);
            }
        }
    }

    fn lower_phis(&mut self, block: &BasicBlock) {
        for inst in &block.program.instructions {
            let IrOp::Phi { sources } = &inst.op else { continue };
            let f32_t = self.f32_t;
            let mut pairs: Vec<(Word, Word)> = Vec::with_capacity(sources.len());
            for (pred_id, val) in sources {
                let v = self.lower_value(val);
                let label = self.block_labels.get(pred_id).copied().unwrap_or(0);
                pairs.push((v, label));
            }
            let id = self.b.phi(f32_t, None, pairs).unwrap();
            if let Some(rid) = inst.result {
                self.value_to_word.insert(rid, id);
            }
        }
    }

    fn emit_terminator(&mut self, block: &BasicBlock) {
        match block.branch {
            BranchKind::Exit => {}
            BranchKind::Unconditional { target } => {
                let lbl = self.block_labels[&target];
                self.b.branch(lbl).unwrap();
            }
            BranchKind::Conditional { target, .. } => {
                let lbl = self.block_labels[&target];
                self.b.branch(lbl).unwrap();
            }
            BranchKind::FallThrough => {
                let next = block.id + 1;
                if let Some(&lbl) = self.block_labels.get(&next) {
                    self.b.branch(lbl).unwrap();
                }
            }
        }
    }

    fn preallocate_resources(&mut self, cfg: &Cfg) {
        let mut needs_sampler = false;
        for block in &cfg.blocks {
            for inst in &block.program.instructions {
                match &inst.op {
                    IrOp::LoadAttr { slot } | IrOp::InterpAttr { slot, .. } => {
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
                    IrOp::SampleTex { .. } => needs_sampler = true,
                    _ => {}
                }
            }
        }
        match self.stage {
            Stage::Vertex => {
                self.position_var();
            }
            Stage::Fragment => {
                self.frag_color_var_id();
            }
        }
        if needs_sampler {
            self.sampler();
        }
        for block in &cfg.blocks {
            let id = self.b.id();
            self.block_labels.insert(block.id, id);
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
        let (words, mask, _ids) = self.finish_inner(cfg, required_output_locations);
        (words, mask)
    }

    pub fn finish_full(
        self,
        cfg: &Cfg,
        required_output_locations: &[u32],
    ) -> (Vec<u32>, u32, Vec<u32>) {
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
    ) -> (Vec<u32>, u32, Vec<u32>) {
        self.preallocate_resources(cfg);

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

        let void_t = self.b.type_void();
        let main_t = self.b.type_function(void_t, vec![]);
        let main_id = self
            .b
            .begin_function(void_t, None, FunctionControl::NONE, main_t)
            .unwrap();
        let entry_label = cfg.blocks.first().map(|b| self.block_labels[&b.id]);
        self.b.begin_block(entry_label).unwrap();

        if matches!(self.stage, Stage::Vertex) {
            for (_loc, av) in &required_outputs {
                let o = self.f32_one;
                for c in 0..4 {
                    self.write_attr_component(*av, c, o);
                }
            }
        }

        if let Some((v, bits)) = ps_inject {
            let c = self.const_f32(bits);
            self.b.store(v, c, None, []).unwrap();
        }

        self.lower_cfg(cfg);

        match self.stage {
            Stage::Vertex => {
                if self.pos_var.is_none() {
                    let pos = self.position_var();
                    let z = self.f32_zero;
                    let o = self.f32_one;
                    let v = self.b.composite_construct(self.vec4_t, None, [z, z, z, o]).unwrap();
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
                            let ac = self.b.access_chain(
                                self.ptr_uniform_f32, None, self.ubo_var,
                                [zero_u32, vec4_idx, comp_idx],
                            ).unwrap();
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
                            let term = self.b.f_mul(self.f32_t, None, m[col][row], comps[col]).unwrap();
                            acc = self.b.f_add(self.f32_t, None, acc, term).unwrap();
                        }
                        clip[row] = acc;
                    }
                    let new_pos = self.b.composite_construct(
                        self.vec4_t, None, clip,
                    ).unwrap();
                    self.b.store(pos_var, new_pos, None, []).unwrap();
                    self.cbuf_bindings_used |= 1;
                }
                let do_z_remap = if std::env::var("NEXIUM_VS_Z_REMAP").ok().as_deref() == Some("1") {
                    true
                } else {
                    self.vertex_opts.apply_z_remap
                };
                let pos = self.position_var();
                let cur = self.b.load(self.vec4_t, None, pos, None, []).unwrap();
                let mut new_pos = cur;
                if do_z_remap {
                    let cur_z = self.b.composite_extract(self.f32_t, None, cur, [2]).unwrap();
                    let cur_w = self.b.composite_extract(self.f32_t, None, cur, [3]).unwrap();
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
                    let cur_z = self.b.composite_extract(self.f32_t, None, new_pos, [2]).unwrap();
                    let cur_w = self.b.composite_extract(self.f32_t, None, new_pos, [3]).unwrap();
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
                let degenerate = cfg.blocks.is_empty()
                    || cfg
                        .blocks
                        .iter()
                        .all(|b| b.program.instructions.is_empty());
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
                let alpha_recovery: Option<Word> = exit_state
                    .filter(|m| !m.contains_key(&3u8))
                    .and_then(|m| {
                        for v in m.values() {
                            if let nexium_shader::ir::Value::Inst(vid) = v {
                                let is_alpha = cfg.blocks.iter().any(|b| {
                                    b.program.instructions.iter().any(|inst| {
                                        inst.result == Some(*vid)
                                            && matches!(
                                                inst.op,
                                                nexium_shader::ir::Op::SampleTex { component: 3, .. }
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
                let chans: [Word; 4] = std::array::from_fn(|r| {
                    match exit_state.and_then(|m| m.get(&(r as u8))) {
                        Some(v) => self.lower_value(v),
                        None => defaults[r],
                    }
                });
                let mut v = self
                    .b
                    .composite_construct(self.vec4_t, None, chans)
                    .unwrap();
                if std::env::var("NEXIUM_FRAG_2X").is_ok() {
                    let two = self.const_f32(2.0f32.to_bits());
                    let two_vec = self
                        .b
                        .composite_construct(self.vec4_t, None, [two, two, two, two])
                        .unwrap();
                    v = self.b.f_mul(self.vec4_t, None, v, two_vec).unwrap();
                }
                let fc = self.frag_color_var_id();
                self.b.store(fc, v, None, []).unwrap();
            }
        }

        self.b.ret().unwrap();
        self.b.end_function().unwrap();

        let exec_model = match self.stage {
            Stage::Vertex => ExecutionModel::Vertex,
            Stage::Fragment => ExecutionModel::Fragment,
        };
        self.b.entry_point(exec_model, main_id, "main", self.interface.clone());
        if self.stage == Stage::Fragment {
            self.b
                .execution_mode(main_id, rspirv::spirv::ExecutionMode::OriginUpperLeft, []);
        }

        let bindings = self.cbuf_bindings_used;
        let tex_ids: Vec<u32> = self.texs_ids_used.iter().copied().collect();
        let words = opt::dedup_constants(self.b.module().assemble());
        (words, bindings, tex_ids)
    }
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

pub fn emit_vertex_with_bindings(
    cfg: &Cfg,
    required_outputs: &[u32],
) -> (Vec<u32>, u32) {
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
    let vec4s = cbuf_vec4s(cfg, if opts.inject_ubo_matrix { 4 } else { 1 });
    let (words, mask) = Emitter::new_with_vertex_opts_sized(Stage::Vertex, opts, vec4s)
        .finish_with_required_outputs_and_bindings(cfg, required_outputs);
    (words, mask, vec4s * 16)
}

pub fn emit_fragment_with_bindings(cfg: &Cfg) -> (Vec<u32>, u32) {
    Emitter::new(Stage::Fragment).finish_with_required_outputs_and_bindings(cfg, &[])
}

pub fn emit_fragment_full(cfg: &Cfg) -> (Vec<u32>, u32, Vec<u32>, u32) {
    let vec4s = cbuf_vec4s(cfg, 1);
    let (words, mask, tex_ids) = Emitter::new_sized(Stage::Fragment, vec4s).finish_full(cfg, &[]);
    (words, mask, tex_ids, vec4s * 16)
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
        let bytes = build_test_program(&[
            enc_fmul_reg(2, 0, 1),
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_vertex(&cfg);
        validates_with_naga(&words);
    }

    #[test]
    fn scan_input_locations_recovers_emitted_slots() {
        let bytes = build_test_program(&[
            0xEFD8_FF80_0807_FF00u64,
            enc_exit(),
        ]);
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

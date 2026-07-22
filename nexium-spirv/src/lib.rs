#![deny(unsafe_op_in_unsafe_fn)]

mod opt;

use std::collections::HashMap;

use nexium_shader::ir::{
    ImageAtomicOp, ImageAtomicType, ImageDimension, MemoryBarrierScope, TextureHandleOrigin,
};
use nexium_shader::{
    BasicBlock, BlockId, BoolOp, BranchKind, CbufAddressMode, Cfg, FComp, HalfMerge, HalfPrecision,
    HalfSwizzle, ICmp, IrInst, IrOp, IrValue, LogicOp, MufuFunc, SubgroupMask, ValueId, VoteMode,
};
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel, FunctionControl, GLOp,
    ImageFormat, MemoryModel, MemorySemantics, Scope, StorageClass, Word,
};

fn lop3_anf_coefficients(lut: u8) -> u8 {
    let mut coefficients = lut;
    for variable in 0..3u8 {
        let variable_bit = 1u8 << variable;
        for mask in 0..8u8 {
            if mask & variable_bit != 0 && ((coefficients >> (mask ^ variable_bit)) & 1) != 0 {
                coefficients ^= 1u8 << mask;
            }
        }
    }
    coefficients
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureNumericType {
    #[default]
    Float,
    Uint,
    Sint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeTexelFormat {
    R16Float,
    R16Uint,
    R16Sint,
    R32Float,
    R32Uint,
    R32Sint,
    Rgba32Float,
    Rgba32Uint,
    Rgba32Sint,
}

impl ComputeTexelFormat {
    pub const fn numeric_type(self) -> TextureNumericType {
        match self {
            Self::R16Float | Self::R32Float | Self::Rgba32Float => TextureNumericType::Float,
            Self::R16Uint | Self::R32Uint | Self::Rgba32Uint => TextureNumericType::Uint,
            Self::R16Sint | Self::R32Sint | Self::Rgba32Sint => TextureNumericType::Sint,
        }
    }

    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::R16Float | Self::R16Uint | Self::R16Sint => 2,
            Self::R32Float | Self::R32Uint | Self::R32Sint => 4,
            Self::Rgba32Float | Self::Rgba32Uint | Self::Rgba32Sint => 16,
        }
    }

    pub const fn requires_storage_image_extended_formats(self) -> bool {
        matches!(self, Self::R16Float | Self::R16Uint | Self::R16Sint)
    }

    pub const fn supports_storage_atomics(self) -> bool {
        matches!(self, Self::R32Uint)
    }

    fn storage_image_format(self) -> ImageFormat {
        match self {
            Self::R16Float => ImageFormat::R16f,
            Self::R16Uint => ImageFormat::R16ui,
            Self::R16Sint => ImageFormat::R16i,
            Self::R32Float => ImageFormat::R32f,
            Self::R32Uint => ImageFormat::R32ui,
            Self::R32Sint => ImageFormat::R32i,
            Self::Rgba32Float => ImageFormat::Rgba32f,
            Self::Rgba32Uint => ImageFormat::Rgba32ui,
            Self::Rgba32Sint => ImageFormat::Rgba32i,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeResourceKind {
    CombinedSampledImage,
    SampledImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    StorageImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeImageResource {
    pub handle: TextureHandleOrigin,
    pub binding: u32,
    pub kind: ComputeResourceKind,
    pub dimension: ImageDimension,
    pub numeric_type: TextureNumericType,
    pub texel_format: Option<ComputeTexelFormat>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeOptions {
    pub local_size: [u32; 3],
    pub local_memory_low_size: u32,
    pub local_memory_high_size: u32,
    pub local_memory_crs_size: u32,
    pub shared_memory_size: u32,
    pub texture_bound_cbuf: u8,
    pub cbuf_sizes: [u32; COMPUTE_CBUF_SLOTS],
    pub resources: Vec<ComputeImageResource>,
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            local_size: [1, 1, 1],
            local_memory_low_size: 0,
            local_memory_high_size: 0,
            local_memory_crs_size: 0,
            shared_memory_size: 0,
            texture_bound_cbuf: 0,
            cbuf_sizes: [COMPUTE_CBUF_MAX_SIZE; COMPUTE_CBUF_SLOTS],
            resources: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeDescriptorKind {
    UniformBuffer,
    CombinedSampledImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    SampledImage,
    StorageImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeDescriptor {
    pub binding: u32,
    pub kind: ComputeDescriptorKind,
    pub handle: Option<TextureHandleOrigin>,
    pub dimension: Option<ImageDimension>,
    pub numeric_type: Option<TextureNumericType>,
    pub texel_format: Option<ComputeTexelFormat>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeModule {
    pub words: Vec<u32>,
    pub cbuf_bindings: u32,
    pub cbuf_size: u32,
    pub cbuf_required_sizes: [u32; COMPUTE_CBUF_SLOTS],
    pub texture_bound_cbuf: u8,
    pub descriptors: Vec<ComputeDescriptor>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ComputeEmitError {
    #[error("compute local size components must all be non-zero (got {0:?})")]
    InvalidLocalSize([u32; 3]),
    #[error("compute shared-memory size {0:#x} exceeds the supported {MAX_COMPUTE_SHARED_MEMORY_SIZE:#x} bytes")]
    InvalidSharedMemorySize(u32),
    #[error("compute shared-memory IR requires a non-zero shared-memory allocation")]
    MissingSharedMemory,
    #[error("compute local-memory IR requires a non-zero QMD local-memory allocation")]
    MissingLocalMemory,
    #[error("compute local-memory size {0:#x} exceeds the supported {MAX_COMPUTE_LOCAL_MEMORY_SIZE:#x} bytes per thread")]
    InvalidLocalMemorySize(u32),
    #[error("compute QMD local-memory allocation overflows: low={low:#x}, high={high:#x}")]
    InvalidLocalMemoryAllocation { low: u32, high: u32 },
    #[error("compute texture-bound cbuf index {0} is outside Maxwell's 16 cbuf slots")]
    InvalidTextureBoundCbuf(u8),
    #[error("compute image binding {0} is reserved for a QMD cbuf descriptor")]
    ReservedCbufBinding(u32),
    #[error("compute cbuf {binding} read ending at {end:#x} exceeds its QMD size {available:#x}")]
    CbufOutOfBounds {
        binding: u8,
        end: u32,
        available: u32,
    },
    #[error("compute indexed cbuf {binding} size {size:#x} exceeds the supported {COMPUTE_CBUF_MAX_SIZE:#x} bytes")]
    IndexedCbufTooLarge { binding: u8, size: u32 },
    #[error("compute IR references cbuf slot {0} outside the eight-entry QMD table")]
    InvalidCbufBinding(u8),
    #[error("compute image binding zero is reserved for QMD cbuf zero")]
    ReservedImageBinding,
    #[error("duplicate compute descriptor binding {0}")]
    DuplicateBinding(u32),
    #[error("duplicate compute metadata for {handle:?} as {kind:?}")]
    DuplicateResource {
        handle: TextureHandleOrigin,
        kind: ComputeResourceKind,
    },
    #[error("{kind:?} resource {handle:?} cannot use image dimension {dimension:?}")]
    InvalidResourceDimension {
        handle: TextureHandleOrigin,
        kind: ComputeResourceKind,
        dimension: ImageDimension,
    },
    #[error("missing {kind:?} compute resource metadata for {handle:?}")]
    MissingResource {
        handle: TextureHandleOrigin,
        kind: ComputeResourceKind,
    },
    #[error(
        "compute resource {handle:?} has runtime dimension {actual:?}, incompatible with instruction dimension {instruction:?}"
    )]
    DimensionMismatch {
        handle: TextureHandleOrigin,
        instruction: ImageDimension,
        actual: ImageDimension,
    },
    #[error("compute CFG still contains {0} untranslated instruction(s)")]
    UnimplementedIr(u32),
    #[error("compute SPIR-V backend does not support IR operation {0}")]
    UnsupportedOperation(String),
    #[error("{dimension:?} image operation for {handle:?} is missing coordinate {coordinate}")]
    MissingCoordinate {
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        coordinate: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpirvEmitError {
    #[error("SPIR-V backend does not support indexed cbuf address mode {0:?}")]
    UnsupportedCbufAddressMode(CbufAddressMode),
    #[error("graphics shader references cbuf bank {0}, outside Maxwell's 16 stage-local banks")]
    InvalidGraphicsCbufBinding(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphicsImageKind {
    #[default]
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
    Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphicsTextureResource {
    pub shader_id: u32,
    pub descriptor_slot: u32,
    pub numeric_type: TextureNumericType,
    pub image_kind: GraphicsImageKind,
}

impl GraphicsTextureResource {
    pub const fn new(
        shader_id: u32,
        descriptor_slot: u32,
        numeric_type: TextureNumericType,
    ) -> Self {
        Self {
            shader_id,
            descriptor_slot,
            numeric_type,
            image_kind: GraphicsImageKind::D2,
        }
    }

    pub const fn with_image_kind(mut self, image_kind: GraphicsImageKind) -> Self {
        self.image_kind = image_kind;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphicsTextureManifestError {
    #[error("graphics texture descriptor slot {0} is outside the 32-entry ABI")]
    InvalidSlot(u32),
    #[error(
        "graphics texture descriptor slot {slot} is assigned to both shader texture IDs {first_shader_id:#x} and {second_shader_id:#x}"
    )]
    SlotConflict {
        slot: u32,
        first_shader_id: u32,
        second_shader_id: u32,
    },
    #[error(
        "shader texture ID {shader_id:#x} at descriptor slot {slot} requires incompatible {first:?} and {second:?} numeric families"
    )]
    NumericConflict {
        shader_id: u32,
        slot: u32,
        first: TextureNumericType,
        second: TextureNumericType,
    },
    #[error(
        "shader texture ID {shader_id:#x} at descriptor slot {slot} requires incompatible {first:?} and {second:?} image families"
    )]
    ImageKindConflict {
        shader_id: u32,
        slot: u32,
        first: GraphicsImageKind,
        second: GraphicsImageKind,
    },
    #[error(
        "graphics texture descriptor slot {slot} should contain shader texture ID {expected_shader_id:#x}, but the manifest contains {actual_shader_id:#x}"
    )]
    ShaderIdMismatch {
        slot: u32,
        expected_shader_id: u32,
        actual_shader_id: u32,
    },
    #[error(
        "graphics texture descriptor slot {slot} for shader texture ID {shader_id:#x} is declared as {actual:?}, but the shader operation requires {expected:?}"
    )]
    ImageKindMismatch {
        slot: u32,
        shader_id: u32,
        expected: GraphicsImageKind,
        actual: GraphicsImageKind,
    },
    #[error(
        "graphics texture descriptor slot {slot} for shader texture ID {shader_id:#x} is missing from the manifest"
    )]
    MissingResource { slot: u32, shader_id: u32 },
    #[error(
        "graphics texture manifest contains unused descriptor slot {slot} for shader texture ID {shader_id:#x}"
    )]
    UnexpectedResource { slot: u32, shader_id: u32 },
    #[error(
        "graphics shader needs {count} texture descriptors starting at slot {base}, exceeding the 32-entry ABI"
    )]
    TooManyResources { base: u32, count: usize },
}

pub fn normalize_graphics_texture_manifest(
    mut resources: Vec<GraphicsTextureResource>,
) -> Result<Vec<GraphicsTextureResource>, GraphicsTextureManifestError> {
    resources.sort_unstable_by_key(|resource| {
        (
            resource.descriptor_slot,
            resource.shader_id,
            resource.numeric_type,
            resource.image_kind,
        )
    });

    let mut normalized: Vec<GraphicsTextureResource> = Vec::with_capacity(resources.len());
    for resource in resources {
        if resource.descriptor_slot >= MAX_TEXTURE_DESCRIPTORS {
            return Err(GraphicsTextureManifestError::InvalidSlot(
                resource.descriptor_slot,
            ));
        }
        if let Some(previous) = normalized.last().copied() {
            if previous.descriptor_slot == resource.descriptor_slot {
                if previous.shader_id != resource.shader_id {
                    return Err(GraphicsTextureManifestError::SlotConflict {
                        slot: resource.descriptor_slot,
                        first_shader_id: previous.shader_id,
                        second_shader_id: resource.shader_id,
                    });
                }
                if previous.numeric_type != resource.numeric_type {
                    return Err(GraphicsTextureManifestError::NumericConflict {
                        shader_id: resource.shader_id,
                        slot: resource.descriptor_slot,
                        first: previous.numeric_type,
                        second: resource.numeric_type,
                    });
                }
                if previous.image_kind != resource.image_kind {
                    return Err(GraphicsTextureManifestError::ImageKindConflict {
                        shader_id: resource.shader_id,
                        slot: resource.descriptor_slot,
                        first: previous.image_kind,
                        second: resource.image_kind,
                    });
                }
                continue;
            }
        }
        normalized.push(resource);
    }
    Ok(normalized)
}

fn record_graphics_texture_image_kind(
    kinds: &mut std::collections::BTreeMap<u32, GraphicsImageKind>,
    shader_id: u32,
    image_kind: GraphicsImageKind,
) {
    if let Some(previous) = kinds.insert(shader_id, image_kind) {
        assert_eq!(
            previous, image_kind,
            "graphics shader texture {shader_id:#x} is used with incompatible {previous:?} and {image_kind:?} image families"
        );
    }
}

fn validate_spirv_ir(cfg: &Cfg, stage: Stage) -> Result<(), SpirvEmitError> {
    for instruction in cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
    {
        if let IrOp::LoadCbufIndexed { address_mode, .. } = &instruction.op {
            if *address_mode != CbufAddressMode::Default {
                return Err(SpirvEmitError::UnsupportedCbufAddressMode(*address_mode));
            }
        }
        if stage != Stage::Compute {
            let binding = match &instruction.op {
                IrOp::LoadCbuf { binding, .. } | IrOp::LoadCbufIndexed { binding, .. } => {
                    Some(*binding)
                }
                IrOp::LoadStorage { cbuf_binding, .. } => Some(*cbuf_binding),
                _ => None,
            };
            if let Some(binding) = binding.filter(|binding| *binding >= 16) {
                return Err(SpirvEmitError::InvalidGraphicsCbufBinding(binding));
            }
        }
    }
    Ok(())
}

fn slot_is_gl_position(aligned_slot: u32) -> bool {
    matches!(aligned_slot, 0x70 | 0x1c0)
}

const UBO_VEC4S: u32 = 4096;
const CBUF_LOGICAL_SLOTS: u32 = 32;
const CBUF_SLOT_VEC4S: u32 = UBO_VEC4S / CBUF_LOGICAL_SLOTS;

pub const GFX_CBUF_SLOTS: u32 = CBUF_LOGICAL_SLOTS;
pub const GFX_CBUF_DIRECTORY_WORDS: u32 = GFX_CBUF_SLOTS * 2;
pub const GFX_CBUF_ZERO_WORD: u32 = GFX_CBUF_DIRECTORY_WORDS;
pub const GFX_CBUF_PAYLOAD_WORD: u32 = 68;
pub const GFX_CBUF_MAX_SIZE: u32 = 64 * 1024;
pub const GFX_CBUF_MIN_SIZE: u32 = GFX_CBUF_PAYLOAD_WORD * 4;
pub const COMPUTE_CBUF_SLOT_STRIDE: u32 = CBUF_SLOT_VEC4S * 16;
pub const COMPUTE_CBUF_SIZE: u32 = UBO_VEC4S * 16;
pub const COMPUTE_CBUF_SLOTS: usize = 8;
pub const COMPUTE_CBUF_MAX_SIZE: u32 = UBO_VEC4S * 16;
const COMPUTE_CBUF_DESCRIPTOR_BASE: u32 = 0x1000;

pub fn compute_cbuf_descriptor_binding(slot: u8) -> Option<u32> {
    (usize::from(slot) < COMPUTE_CBUF_SLOTS).then_some(if slot == 0 {
        0
    } else {
        COMPUTE_CBUF_DESCRIPTOR_BASE + u32::from(slot)
    })
}

pub fn compute_cbuf_slot_for_descriptor_binding(binding: u32) -> Option<u8> {
    if binding == 0 {
        Some(0)
    } else {
        let slot = binding.checked_sub(COMPUTE_CBUF_DESCRIPTOR_BASE)?;
        (slot > 0 && slot < COMPUTE_CBUF_SLOTS as u32).then_some(slot as u8)
    }
}
const MAX_TEXTURE_DESCRIPTORS: u32 = 32;
const LEGACY_GRAPHICS_LOCAL_MEMORY_SIZE: u32 = 4 * 1024;
pub const GFX_BINDING_CBUF: u32 = 0;
pub const GFX_BINDING_FLOAT_2D: u32 = 1;
pub const GFX_BINDING_SAMPLERS: u32 = 2;
pub const GFX_BINDING_SSBO_BASE: u32 = 3;
pub const GFX_BINDING_FLOAT_3D: u32 = 11;
pub const GFX_BINDING_FLOAT_CUBE: u32 = 12;
pub const GFX_BINDING_FLOAT_CUBE_ARRAY: u32 = 13;
pub const GFX_BINDING_FLOAT_TEXEL_BUFFER: u32 = 14;
pub const GFX_BINDING_UINT_2D: u32 = 15;
pub const GFX_BINDING_UINT_3D: u32 = 16;
pub const GFX_BINDING_UINT_CUBE: u32 = 17;
pub const GFX_BINDING_UINT_CUBE_ARRAY: u32 = 18;
pub const GFX_BINDING_UINT_TEXEL_BUFFER: u32 = 19;
pub const GFX_BINDING_SINT_2D: u32 = 20;
pub const GFX_BINDING_SINT_3D: u32 = 21;
pub const GFX_BINDING_SINT_CUBE: u32 = 22;
pub const GFX_BINDING_SINT_CUBE_ARRAY: u32 = 23;
pub const GFX_BINDING_SINT_TEXEL_BUFFER: u32 = 24;

pub const fn graphics_image_binding(
    numeric_type: TextureNumericType,
    kind: GraphicsImageKind,
) -> u32 {
    match (numeric_type, kind) {
        (TextureNumericType::Float, GraphicsImageKind::D2 | GraphicsImageKind::D2Array) => {
            GFX_BINDING_FLOAT_2D
        }
        (TextureNumericType::Float, GraphicsImageKind::D3) => GFX_BINDING_FLOAT_3D,
        (TextureNumericType::Float, GraphicsImageKind::Cube) => GFX_BINDING_FLOAT_CUBE,
        (TextureNumericType::Float, GraphicsImageKind::CubeArray) => {
            GFX_BINDING_FLOAT_CUBE_ARRAY
        }
        (TextureNumericType::Float, GraphicsImageKind::Buffer) => {
            GFX_BINDING_FLOAT_TEXEL_BUFFER
        }
        (TextureNumericType::Uint, GraphicsImageKind::D2 | GraphicsImageKind::D2Array) => {
            GFX_BINDING_UINT_2D
        }
        (TextureNumericType::Uint, GraphicsImageKind::D3) => GFX_BINDING_UINT_3D,
        (TextureNumericType::Uint, GraphicsImageKind::Cube) => GFX_BINDING_UINT_CUBE,
        (TextureNumericType::Uint, GraphicsImageKind::CubeArray) => {
            GFX_BINDING_UINT_CUBE_ARRAY
        }
        (TextureNumericType::Uint, GraphicsImageKind::Buffer) => GFX_BINDING_UINT_TEXEL_BUFFER,
        (TextureNumericType::Sint, GraphicsImageKind::D2 | GraphicsImageKind::D2Array) => {
            GFX_BINDING_SINT_2D
        }
        (TextureNumericType::Sint, GraphicsImageKind::D3) => GFX_BINDING_SINT_3D,
        (TextureNumericType::Sint, GraphicsImageKind::Cube) => GFX_BINDING_SINT_CUBE,
        (TextureNumericType::Sint, GraphicsImageKind::CubeArray) => {
            GFX_BINDING_SINT_CUBE_ARRAY
        }
        (TextureNumericType::Sint, GraphicsImageKind::Buffer) => GFX_BINDING_SINT_TEXEL_BUFFER,
    }
}
const SSBO_BINDING_BASE: u32 = GFX_BINDING_SSBO_BASE;
pub const MAX_SSBO: u32 = 8;
const SHADER_LOOP_SAFETY_LIMIT: u32 = 0x2000;
pub const MAX_COMPUTE_SHARED_MEMORY_SIZE: u32 = 64 * 1024;
pub const MAX_COMPUTE_LOCAL_MEMORY_SIZE: u32 = 512 * 1024;

pub struct Emitter {
    b: rspirv::dr::Builder,
    stage: Stage,
    f32_t: Word,
    vec2_t: Word,
    vec3_t: Word,
    vec4_t: Word,
    u32_t: Word,
    i32_t: Word,
    uvec2_t: Word,
    uvec3_t: Word,
    uvec4_t: Word,
    ivec4_t: Word,
    ivec2_t: Word,
    ivec3_t: Word,
    ptr_uniform_f32: Word,
    ptr_uniform_u32: Word,
    ptr_input_vec4: Word,
    ptr_input_f32: Word,
    ptr_input_u32: Word,
    ptr_input_uvec3: Word,
    ptr_output_vec4: Word,
    ptr_output_f32: Word,
    ptr_output_uvec4: Word,
    ptr_output_ivec4: Word,
    ptr_output_u32: Word,
    ptr_output_i32: Word,
    ptr_uniform_struct: Word,
    ptr_image: Word,
    ptr_image_arrayed: Word,
    ptr_sampler: Word,
    ptr_image_array: Word,
    ptr_image_arrayed_array: Word,
    ptr_sampler_array: Word,
    image_buffer_t: Option<Word>,
    ptr_image_buffer: Option<Word>,
    image_buffer_var: Option<Word>,
    image_3d_t: Word,
    sampled_image_3d_t: Word,
    ptr_image_3d: Word,
    ptr_image_3d_array: Word,
    image_3d_var: Option<Word>,
    image_cube_t: Word,
    sampled_image_cube_t: Word,
    ptr_image_cube: Word,
    ptr_image_cube_array: Word,
    image_cube_var: Option<Word>,
    image_cube_arrayed_t: Word,
    sampled_image_cube_arrayed_t: Word,
    ptr_image_cube_arrayed: Word,
    ptr_image_cube_arrayed_array: Word,
    image_cube_arrayed_var: Option<Word>,
    ubo_var: Word,
    compute_cbuf_vars: [Option<Word>; COMPUTE_CBUF_SLOTS],
    local_mem_var: Option<Word>,
    shared_mem_var: Option<Word>,
    ptr_private_u32: Word,
    ptr_workgroup_u32: Word,
    ptr_image_u32: Option<Word>,
    f32_zero: Word,
    f32_one: Word,
    i32_zero: Word,
    glsl: Word,
    input_vars: HashMap<u32, AttrVar>,
    output_vars: HashMap<u32, AttrVar>,
    pos_var: Option<Word>,
    point_size_var: Option<Word>,
    subgroup_id_var: Option<Word>,
    local_invocation_id_var: Option<Word>,
    workgroup_id_var: Option<Word>,
    fswzadd_lut_a: Option<Word>,
    fswzadd_lut_b: Option<Word>,
    layer_var: Option<Word>,
    frag_coord_var: Option<Word>,
    frag_color_vars: HashMap<u32, Word>,
    vertex_index_var: Option<Word>,
    instance_index_var: Option<Word>,
    base_instance_var: Option<Word>,
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
    indirect_default_blocks: Vec<Word>,
    indirect_structural_cases: HashMap<BlockId, Vec<IndirectStructuralCaseGroup>>,
    self_loops: HashMap<BlockId, LoopInfo>,
    loop_break_merges: HashMap<BlockId, (BlockId, Word)>,
    loop_carried: HashMap<BlockId, Vec<(ValueId, Word, Word, Option<u8>)>>,
    loop_safety_vars: HashMap<BlockId, Word>,
    current_block: Option<BlockId>,
    cbuf_bindings_used: u32,
    texs_ids_used: std::collections::BTreeSet<u32>,
    texture_slots: HashMap<u32, u32>,
    texture_numeric_manifest: Vec<GraphicsTextureResource>,
    typed_image_decls: HashMap<(TextureNumericType, GraphicsImageKind), GraphicsImageDecl>,
    vertex_opts: VertexOptions,
    const_cache_f32: HashMap<u32, Word>,
    const_cache_u32: HashMap<u32, Word>,
    bool_t: Word,
    bool_true: Word,
    bool_false: Word,
    pred_regs: [Option<Word>; 7],
    pred_value_to_word: HashMap<(ValueId, u8), Word>,
    block_pred_exits: HashMap<BlockId, [Option<Word>; 7]>,
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
    fragment_uint_output_mask: u32,
    fragment_sint_output_mask: u32,
    texel_buffer_mask: u32,
    ps_input_map: [u8; 32],
    alpha_test_func: u32,
    alpha_test_ref: u32,
    y_negate: bool,
    compute_options: Option<ComputeOptions>,
    compute_resource_vars: Vec<ComputeResourceVar>,
    ir_constant_facts: nexium_shader::IrConstantFacts,
}

#[derive(Clone, Copy)]
struct AttrVar {
    var: Word,
    ptr_f32: Word,
    is_uint: bool,
    is_sint: bool,
}

#[derive(Clone, Copy)]
struct ComputeResourceVar {
    resource: ComputeImageResource,
    image_t: Word,
    sampled_image_t: Option<Word>,
    scalar_t: Word,
    vec4_t: Word,
    var: Word,
}

#[derive(Clone, Copy)]
struct GraphicsImageDecl {
    image_t: Word,
    ptr_image: Word,
    var: Word,
}

#[derive(Clone)]
struct IndirectStructuralCaseGroup {
    header: Word,
    merge: Word,
    fallback: Word,
    cases: Vec<(u32, Word)>,
}

impl Emitter {
    pub fn new(stage: Stage) -> Self {
        Self::new_sized(stage, UBO_VEC4S)
    }

    fn new_sized(stage: Stage, ubo_vec4s: u32) -> Self {
        let mut b = rspirv::dr::Builder::new();
        b.set_version(1, 3);
        b.capability(Capability::Shader);
        b.capability(Capability::DerivativeControl);
        let glsl = b.ext_inst_import("GLSL.std.450");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let f32_t = b.type_float(32);
        let vec2_t = b.type_vector(f32_t, 2);
        let vec3_t = b.type_vector(f32_t, 3);
        let vec4_t = b.type_vector(f32_t, 4);
        let u32_t = b.type_int(32, 0);
        let i32_t = b.type_int(32, 1);
        let uvec2_t = b.type_vector(u32_t, 2);
        let uvec3_t = b.type_vector(u32_t, 3);
        let uvec4_t = b.type_vector(u32_t, 4);
        let ivec4_t = b.type_vector(i32_t, 4);
        let ivec2_t = b.type_vector(i32_t, 2);
        let ivec3_t = b.type_vector(i32_t, 3);
        let ptr_uniform_f32 = b.type_pointer(None, StorageClass::Uniform, f32_t);
        let ptr_uniform_u32 = b.type_pointer(None, StorageClass::Uniform, u32_t);
        let ptr_input_vec4 = b.type_pointer(None, StorageClass::Input, vec4_t);
        let ptr_input_f32 = b.type_pointer(None, StorageClass::Input, f32_t);
        let ptr_input_u32 = b.type_pointer(None, StorageClass::Input, u32_t);
        let ptr_input_uvec3 = b.type_pointer(None, StorageClass::Input, uvec3_t);
        let ptr_output_vec4 = b.type_pointer(None, StorageClass::Output, vec4_t);
        let ptr_output_f32 = b.type_pointer(None, StorageClass::Output, f32_t);
        let ptr_output_uvec4 = b.type_pointer(None, StorageClass::Output, uvec4_t);
        let ptr_output_ivec4 = b.type_pointer(None, StorageClass::Output, ivec4_t);
        let ptr_output_u32 = b.type_pointer(None, StorageClass::Output, u32_t);
        let ptr_output_i32 = b.type_pointer(None, StorageClass::Output, i32_t);

        let ubo_struct = if stage == Stage::Compute {
            let ubo_vec4s_const = b.constant_bit32(u32_t, ubo_vec4s);
            let vec4_arr = b.type_array(vec4_t, ubo_vec4s_const);
            b.decorate(
                vec4_arr,
                Decoration::ArrayStride,
                [Operand::LiteralBit32(16)],
            );
            let structure = b.type_struct([vec4_arr]);
            b.decorate(structure, Decoration::Block, []);
            structure
        } else {
            let words = b.type_runtime_array(u32_t);
            b.decorate(words, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
            let structure = b.type_struct([words]);
            b.decorate(structure, Decoration::BufferBlock, []);
            b.member_decorate(structure, 0, Decoration::NonWritable, []);
            structure
        };
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

        let ptr_private_u32 = b.type_pointer(None, StorageClass::Private, u32_t);
        let ptr_workgroup_u32 = b.type_pointer(None, StorageClass::Workgroup, u32_t);

        let texture_scalar_t = f32_t;
        let image_t = b.type_image(
            texture_scalar_t,
            rspirv::spirv::Dim::Dim2D,
            0,
            0,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let image_arrayed_t = b.type_image(
            texture_scalar_t,
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
            texture_scalar_t,
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
        let image_cube_t = b.type_image(
            texture_scalar_t,
            rspirv::spirv::Dim::DimCube,
            0,
            0,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let sampled_image_cube_t = b.type_sampled_image(image_cube_t);
        let ptr_image_cube = b.type_pointer(None, StorageClass::UniformConstant, image_cube_t);
        let image_cube_array_t = b.type_array(image_cube_t, texture_slots_const);
        let ptr_image_cube_array =
            b.type_pointer(None, StorageClass::UniformConstant, image_cube_array_t);
        let image_cube_arrayed_t = b.type_image(
            texture_scalar_t,
            rspirv::spirv::Dim::DimCube,
            0,
            1,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let sampled_image_cube_arrayed_t = b.type_sampled_image(image_cube_arrayed_t);
        let ptr_image_cube_arrayed =
            b.type_pointer(None, StorageClass::UniformConstant, image_cube_arrayed_t);
        let image_cube_arrayed_array_t = b.type_array(image_cube_arrayed_t, texture_slots_const);
        let ptr_image_cube_arrayed_array = b.type_pointer(
            None,
            StorageClass::UniformConstant,
            image_cube_arrayed_array_t,
        );

        let f32_zero = b.constant_bit32(f32_t, 0.0f32.to_bits());
        let f32_one = b.constant_bit32(f32_t, 1.0f32.to_bits());
        let i32_zero = b.constant_bit32(i32_t, 0);

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
            uvec2_t,
            uvec3_t,
            uvec4_t,
            ivec4_t,
            ivec2_t,
            ivec3_t,
            ptr_uniform_f32,
            ptr_uniform_u32,
            ptr_input_vec4,
            ptr_input_f32,
            ptr_input_u32,
            ptr_input_uvec3,
            ptr_output_vec4,
            ptr_output_f32,
            ptr_output_uvec4,
            ptr_output_ivec4,
            ptr_output_u32,
            ptr_output_i32,
            ptr_uniform_struct,
            ptr_image,
            ptr_image_arrayed,
            ptr_sampler,
            ptr_image_array,
            ptr_image_arrayed_array,
            ptr_sampler_array,
            image_buffer_t: None,
            ptr_image_buffer: None,
            image_buffer_var: None,
            image_3d_t,
            sampled_image_3d_t,
            ptr_image_3d,
            ptr_image_3d_array,
            image_3d_var: None,
            image_cube_t,
            sampled_image_cube_t,
            ptr_image_cube,
            ptr_image_cube_array,
            image_cube_var: None,
            image_cube_arrayed_t,
            sampled_image_cube_arrayed_t,
            ptr_image_cube_arrayed,
            ptr_image_cube_arrayed_array,
            image_cube_arrayed_var: None,
            ubo_var,
            compute_cbuf_vars: std::array::from_fn(|slot| (slot == 0).then_some(ubo_var)),
            local_mem_var: None,
            shared_mem_var: None,
            ptr_private_u32,
            ptr_workgroup_u32,
            ptr_image_u32: None,
            f32_zero,
            f32_one,
            i32_zero,
            glsl,
            input_vars: HashMap::new(),
            output_vars: HashMap::new(),
            pos_var: None,
            point_size_var: None,
            subgroup_id_var: None,
            local_invocation_id_var: None,
            workgroup_id_var: None,
            fswzadd_lut_a: None,
            fswzadd_lut_b: None,
            layer_var: None,
            frag_coord_var: None,
            frag_color_vars: HashMap::new(),
            vertex_index_var: None,
            instance_index_var: None,
            base_instance_var: None,
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
            indirect_default_blocks: Vec::new(),
            indirect_structural_cases: HashMap::new(),
            self_loops: HashMap::new(),
            loop_break_merges: HashMap::new(),
            loop_carried: HashMap::new(),
            loop_safety_vars: HashMap::new(),
            current_block: None,
            cbuf_bindings_used: 0,
            texs_ids_used: std::collections::BTreeSet::new(),
            texture_slots: HashMap::new(),
            texture_numeric_manifest: Vec::new(),
            typed_image_decls: HashMap::new(),
            vertex_opts: VertexOptions::default(),
            const_cache_f32,
            const_cache_u32,
            bool_t,
            bool_true,
            bool_false,
            pred_regs: [None; 7],
            pred_value_to_word: HashMap::new(),
            block_pred_exits: HashMap::new(),
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
            fragment_uint_output_mask: 0,
            fragment_sint_output_mask: 0,
            texel_buffer_mask: 0,
            ps_input_map: [0; 32],
            alpha_test_func: 0,
            alpha_test_ref: 0,
            y_negate: false,
            compute_options: (stage == Stage::Compute).then(ComputeOptions::default),
            compute_resource_vars: Vec::new(),
            ir_constant_facts: nexium_shader::IrConstantFacts::default(),
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

    fn ensure_local_mem_var(&mut self) -> Word {
        if let Some(var) = self.local_mem_var {
            return var;
        }
        let byte_size = if self.stage == Stage::Compute {
            let options = self
                .compute_options
                .as_ref()
                .expect("compute options are required for local memory");
            options
                .local_memory_low_size
                .checked_add(options.local_memory_high_size)
                .expect("local-memory allocation must be validated before emission")
        } else {
            LEGACY_GRAPHICS_LOCAL_MEMORY_SIZE
        };
        assert!(byte_size != 0, "local-memory allocation must be non-zero");
        let words = self.b.constant_bit32(self.u32_t, byte_size.div_ceil(4));
        let array_t = self.b.type_array(self.u32_t, words);
        let ptr_t = self.b.type_pointer(None, StorageClass::Private, array_t);
        let var = self.b.variable(ptr_t, None, StorageClass::Private, None);
        self.local_mem_var = Some(var);
        var
    }

    fn local_word_pointer(&mut self, addr: &IrValue) -> (Word, Word) {
        let byte_size = if self.stage == Stage::Compute {
            let options = self
                .compute_options
                .as_ref()
                .expect("compute options are required for local memory");
            options
                .local_memory_low_size
                .checked_add(options.local_memory_high_size)
                .expect("local-memory allocation must be validated before emission")
        } else {
            LEGACY_GRAPHICS_LOCAL_MEMORY_SIZE
        };
        let addr_v = self.lower_value(addr);
        let addr_u = self.as_u32(addr_v);
        let byte_size = self.const_u32(byte_size);
        let in_bounds = self
            .b
            .u_less_than(self.bool_t, None, addr_u, byte_size)
            .unwrap();
        let two = self.const_u32(2);
        let word = self
            .b
            .shift_right_logical(self.u32_t, None, addr_u, two)
            .unwrap();
        let zero = self.const_u32(0);
        let safe_word = self
            .b
            .select(self.u32_t, None, in_bounds, word, zero)
            .unwrap();
        let local_mem_var = self
            .local_mem_var
            .expect("local memory must be preallocated");
        let pointer = self
            .b
            .access_chain(self.ptr_private_u32, None, local_mem_var, [safe_word])
            .unwrap();
        (pointer, in_bounds)
    }

    fn ensure_shared_mem_var(&mut self) -> Word {
        if let Some(var) = self.shared_mem_var {
            return var;
        }
        assert_eq!(self.stage, Stage::Compute);
        let byte_size = self
            .compute_options
            .as_ref()
            .expect("compute options are required for shared memory")
            .shared_memory_size;
        assert!(byte_size != 0, "shared-memory allocation must be non-zero");
        let word_count = byte_size.div_ceil(4);
        let words = self.b.constant_bit32(self.u32_t, word_count);
        let array_t = self.b.type_array(self.u32_t, words);
        let ptr_t = self.b.type_pointer(None, StorageClass::Workgroup, array_t);
        let var = self.b.variable(ptr_t, None, StorageClass::Workgroup, None);
        self.shared_mem_var = Some(var);
        var
    }

    fn compute_cbuf_var(&mut self, binding: u8) -> Word {
        assert_eq!(self.stage, Stage::Compute);
        let index = usize::from(binding);
        if let Some(var) = self.compute_cbuf_vars[index] {
            return var;
        }
        let descriptor_binding = compute_cbuf_descriptor_binding(binding)
            .expect("compute cbuf binding must be validated before emission");
        let var = self
            .b
            .variable(self.ptr_uniform_struct, None, StorageClass::Uniform, None);
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(descriptor_binding)],
        );
        self.compute_cbuf_vars[index] = Some(var);
        var
    }

    fn graphics_cbuf_load_word(&mut self, logical_binding: u32, effective_byte: Word) -> Word {
        assert!(self.stage != Stage::Compute);
        assert!(logical_binding < GFX_CBUF_SLOTS);

        let zero = self.const_u32(0);
        let two = self.const_u32(2);
        let word_offset = self
            .b
            .shift_right_logical(self.u32_t, None, effective_byte, two)
            .unwrap();
        let base_index = self.const_u32(logical_binding * 2);
        let count_index = self.const_u32(logical_binding * 2 + 1);
        let base_pointer = self
            .b
            .access_chain(
                self.ptr_uniform_u32,
                None,
                self.ubo_var,
                [zero, base_index],
            )
            .unwrap();
        let count_pointer = self
            .b
            .access_chain(
                self.ptr_uniform_u32,
                None,
                self.ubo_var,
                [zero, count_index],
            )
            .unwrap();
        let base = self
            .b
            .load(self.u32_t, None, base_pointer, None, [])
            .unwrap();
        let count = self
            .b
            .load(self.u32_t, None, count_pointer, None, [])
            .unwrap();
        let in_bounds = self
            .b
            .u_less_than(self.bool_t, None, word_offset, count)
            .unwrap();
        let payload_index = self
            .b
            .i_add(self.u32_t, None, base, word_offset)
            .unwrap();
        let sentinel = self.const_u32(GFX_CBUF_ZERO_WORD);
        let safe_index = self
            .b
            .select(self.u32_t, None, in_bounds, payload_index, sentinel)
            .unwrap();
        let pointer = self
            .b
            .access_chain(
                self.ptr_uniform_u32,
                None,
                self.ubo_var,
                [zero, safe_index],
            )
            .unwrap();
        self.b
            .load(self.u32_t, None, pointer, None, [])
            .unwrap()
    }

    pub fn new_with_vertex_opts(stage: Stage, vertex_opts: VertexOptions) -> Self {
        let mut e = Self::new_sized(stage, UBO_VEC4S);
        e.texel_buffer_mask = vertex_opts.texel_buffer_mask;
        e.texture_numeric_manifest = normalize_graphics_texture_manifest(
            vertex_opts.texture_numeric_manifest.clone(),
        )
        .unwrap_or_else(|error| panic!("nexium-spirv: invalid graphics texture manifest: {error}"));
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
        e.texel_buffer_mask = vertex_opts.texel_buffer_mask;
        e.texture_numeric_manifest = normalize_graphics_texture_manifest(
            vertex_opts.texture_numeric_manifest.clone(),
        )
        .unwrap_or_else(|error| panic!("nexium-spirv: invalid graphics texture manifest: {error}"));
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

    fn subgroup_id_var(&mut self) -> Word {
        if let Some(v) = self.subgroup_id_var {
            return v;
        }
        self.b.capability(Capability::GroupNonUniform);
        let v = self
            .b
            .variable(self.ptr_input_u32, None, StorageClass::Input, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)],
        );
        self.b.decorate(v, Decoration::Flat, []);
        self.interface.push(v);
        self.subgroup_id_var = Some(v);
        v
    }

    fn subgroup_lane_id(&mut self) -> Word {
        let v = self.subgroup_id_var();
        self.b.load(self.u32_t, None, v, None, []).unwrap()
    }

    fn subgroup_mask(&mut self, kind: SubgroupMask) -> Word {
        let lane = self.subgroup_lane_id();
        let one = self.const_u32(1);
        let equal = self
            .b
            .shift_left_logical(self.u32_t, None, one, lane)
            .unwrap();
        let less = self.b.i_sub(self.u32_t, None, equal, one).unwrap();
        match kind {
            SubgroupMask::Eq => equal,
            SubgroupMask::Lt => less,
            SubgroupMask::Le => self.b.bitwise_or(self.u32_t, None, less, equal).unwrap(),
            SubgroupMask::Gt => {
                let less_equal = self.b.bitwise_or(self.u32_t, None, less, equal).unwrap();
                self.b.not(self.u32_t, None, less_equal).unwrap()
            }
            SubgroupMask::Ge => self.b.not(self.u32_t, None, less).unwrap(),
        }
    }

    fn enable_subgroup_vote(&mut self) {
        self.b.capability(Capability::GroupNonUniform);
        self.b.capability(Capability::GroupNonUniformBallot);
        self.b.capability(Capability::GroupNonUniformVote);
    }

    fn local_invocation_id_var(&mut self) -> Word {
        assert_eq!(self.stage, Stage::Compute);
        if let Some(var) = self.local_invocation_id_var {
            return var;
        }
        let var = self
            .b
            .variable(self.ptr_input_uvec3, None, StorageClass::Input, None);
        self.b.decorate(
            var,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::LocalInvocationId)],
        );
        self.interface.push(var);
        self.local_invocation_id_var = Some(var);
        var
    }

    fn workgroup_id_var(&mut self) -> Word {
        assert_eq!(self.stage, Stage::Compute);
        if let Some(var) = self.workgroup_id_var {
            return var;
        }
        let var = self
            .b
            .variable(self.ptr_input_uvec3, None, StorageClass::Input, None);
        self.b.decorate(
            var,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::WorkgroupId)],
        );
        self.interface.push(var);
        self.workgroup_id_var = Some(var);
        var
    }

    fn compute_builtin_component(&mut self, local: bool, component: u8) -> Word {
        let var = if local {
            self.local_invocation_id_var()
        } else {
            self.workgroup_id_var()
        };
        let vector = self.b.load(self.uvec3_t, None, var, None, []).unwrap();
        let component = self
            .b
            .composite_extract(self.u32_t, None, vector, [(component.min(2)) as u32])
            .unwrap();
        self.store_bits(component)
    }

    fn fswzadd_luts(&mut self) -> (Word, Word) {
        if let (Some(a), Some(b)) = (self.fswzadd_lut_a, self.fswzadd_lut_b) {
            return (a, b);
        }
        let n1 = self.const_f32((-1.0f32).to_bits());
        let p1 = self.const_f32((1.0f32).to_bits());
        let z = self.const_f32((0.0f32).to_bits());
        let a = self.b.constant_composite(self.vec4_t, [n1, p1, n1, z]);
        let b = self.b.constant_composite(self.vec4_t, [n1, n1, p1, n1]);
        self.fswzadd_lut_a = Some(a);
        self.fswzadd_lut_b = Some(b);
        (a, b)
    }

    fn shfl_target(&mut self, mode: u8, index: Word, mask: Word) -> (Word, Word) {
        self.b.capability(Capability::GroupNonUniformShuffle);
        let u32_t = self.u32_t;
        let bool_t = self.bool_t;
        let c0 = self.const_u32(0);
        let c5 = self.const_u32(5);
        let c8 = self.const_u32(8);
        let clamp = self
            .b
            .bit_field_u_extract(u32_t, None, mask, c0, c5)
            .unwrap();
        let seg_mask = self
            .b
            .bit_field_u_extract(u32_t, None, mask, c8, c5)
            .unwrap();
        let tid = self.subgroup_lane_id();
        let not_seg = self.b.not(u32_t, None, seg_mask).unwrap();
        let min_tid = self.b.bitwise_and(u32_t, None, tid, seg_mask).unwrap();
        let clamp_notseg = self.b.bitwise_and(u32_t, None, clamp, not_seg).unwrap();
        let max_tid = self
            .b
            .bitwise_or(u32_t, None, min_tid, clamp_notseg)
            .unwrap();
        match mode {
            0 => {
                let lhs = self.b.bitwise_and(u32_t, None, index, not_seg).unwrap();
                let src = self.b.bitwise_or(u32_t, None, lhs, min_tid).unwrap();
                let in_range = self
                    .b
                    .s_less_than_equal(bool_t, None, src, max_tid)
                    .unwrap();
                (src, in_range)
            }
            1 => {
                let src = self.b.i_sub(u32_t, None, tid, index).unwrap();
                let in_range = self
                    .b
                    .s_greater_than_equal(bool_t, None, src, min_tid)
                    .unwrap();
                (src, in_range)
            }
            2 => {
                let src = self.b.i_add(u32_t, None, tid, index).unwrap();
                let in_range = self
                    .b
                    .s_less_than_equal(bool_t, None, src, max_tid)
                    .unwrap();
                (src, in_range)
            }
            _ => {
                let src = self.b.bitwise_xor(u32_t, None, tid, index).unwrap();
                let in_range = self
                    .b
                    .s_less_than_equal(bool_t, None, src, max_tid)
                    .unwrap();
                (src, in_range)
            }
        }
    }

    fn layer_var_id(&mut self) -> Word {
        if let Some(v) = self.layer_var {
            return v;
        }
        self.b.capability(Capability::ShaderViewportIndexLayerEXT);
        self.b.extension("SPV_EXT_shader_viewport_index_layer");
        let v = self
            .b
            .variable(self.ptr_output_u32, None, StorageClass::Output, None);
        self.b
            .decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::Layer)]);
        self.interface.push(v);
        self.layer_var = Some(v);
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
        let ptr_type = match self.fragment_output_numeric_type(location) {
            TextureNumericType::Float => self.ptr_output_vec4,
            TextureNumericType::Uint => self.ptr_output_uvec4,
            TextureNumericType::Sint => self.ptr_output_ivec4,
        };
        let v = self.b.variable(ptr_type, None, StorageClass::Output, None);
        self.b.decorate(
            v,
            Decoration::Location,
            [Operand::LiteralBit32(attachment_location)],
        );
        self.interface.push(v);
        self.frag_color_vars.insert(location, v);
        v
    }

    fn fragment_output_numeric_type(&self, location: u32) -> TextureNumericType {
        let bit = 1u32.checked_shl(location).unwrap_or(0);
        if self.fragment_uint_output_mask & bit != 0 {
            TextureNumericType::Uint
        } else if self.fragment_sint_output_mask & bit != 0 {
            TextureNumericType::Sint
        } else {
            TextureNumericType::Float
        }
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
        let value = {
            use std::sync::OnceLock;
            static SOLID: OnceLock<bool> = OnceLock::new();
            let on = *SOLID.get_or_init(|| std::env::var_os("NEXIUM_FS_SOLID").is_some());
            if on {
                let one = self.const_f32(1.0f32.to_bits());
                let zero = self.const_f32(0.0f32.to_bits());
                self.b
                    .composite_construct(self.vec4_t, None, [one, zero, one, one])
                    .unwrap()
            } else {
                value
            }
        };
        let numeric_type = self.fragment_output_numeric_type(location);
        if self.fragment_output_map == 0 {
            let fc = self.frag_color_var_at(location);
            let stored = match numeric_type {
                TextureNumericType::Float => value,
                TextureNumericType::Uint => self.b.bitcast(self.uvec4_t, None, value).unwrap(),
                TextureNumericType::Sint => self.b.bitcast(self.ivec4_t, None, value).unwrap(),
            };
            self.b.store(fc, stored, None, []).unwrap();
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
            let (ptr_type, scalar_type) = match numeric_type {
                TextureNumericType::Float => (self.ptr_output_f32, self.f32_t),
                TextureNumericType::Uint => (self.ptr_output_u32, self.u32_t),
                TextureNumericType::Sint => (self.ptr_output_i32, self.i32_t),
            };
            let ptr = self.b.access_chain(ptr_type, None, fc, [idx]).unwrap();
            let float_component = self
                .b
                .composite_extract(self.f32_t, None, value, [component])
                .unwrap();
            let c = if numeric_type == TextureNumericType::Float {
                float_component
            } else {
                self.b.bitcast(scalar_type, None, float_component).unwrap()
            };
            self.b.store(ptr, c, None, []).unwrap();
        }
    }

    fn emit_alpha_test(&mut self, outputs: &[(u32, Word)]) {
        let func = self.alpha_test_func;
        if func == 0 || func == 0x207 || func == 8 {
            return;
        }
        let Some((_, value)) = outputs.iter().find(|(loc, _)| *loc == 0).copied() else {
            return;
        };
        let alpha = self
            .b
            .composite_extract(self.f32_t, None, value, [3])
            .unwrap();
        let reference = self.const_f32(self.alpha_test_ref);
        let bool_t = self.bool_t;
        let pass = match func {
            0x200 | 1 => self.bool_false,
            0x201 | 2 => self
                .b
                .f_ord_less_than(bool_t, None, alpha, reference)
                .unwrap(),
            0x202 | 3 => self.b.f_ord_equal(bool_t, None, alpha, reference).unwrap(),
            0x203 | 4 => self
                .b
                .f_ord_less_than_equal(bool_t, None, alpha, reference)
                .unwrap(),
            0x204 | 5 => self
                .b
                .f_ord_greater_than(bool_t, None, alpha, reference)
                .unwrap(),
            0x205 | 6 => self
                .b
                .f_ord_not_equal(bool_t, None, alpha, reference)
                .unwrap(),
            0x206 | 7 => self
                .b
                .f_ord_greater_than_equal(bool_t, None, alpha, reference)
                .unwrap(),
            _ => return,
        };
        let fail = self.b.logical_not(bool_t, None, pass).unwrap();
        let kill_block = self.b.id();
        let merge_block = self.b.id();
        self.b
            .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
            .unwrap();
        self.b
            .branch_conditional(fail, kill_block, merge_block, [])
            .unwrap();
        self.b.begin_block(Some(kill_block)).unwrap();
        self.b.kill().unwrap();
        self.b.begin_block(Some(merge_block)).unwrap();
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

    fn base_instance_var_id(&mut self) -> Word {
        if let Some(v) = self.base_instance_var {
            return v;
        }
        self.b.extension("SPV_KHR_shader_draw_parameters");
        self.b.capability(Capability::DrawParameters);
        let v = self
            .b
            .variable(self.ptr_input_u32, None, StorageClass::Input, None);
        self.b.decorate(
            v,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::BaseInstance)],
        );
        self.interface.push(v);
        self.base_instance_var = Some(v);
        v
    }

    fn system_attr_var_id(&mut self, slot: u32) -> Option<Word> {
        if !matches!(self.stage, Stage::Vertex) {
            return None;
        }
        match slot {
            0x2f8 => {
                self.base_instance_var_id();
                Some(self.instance_index_var_id())
            }
            0x2fc => Some(self.vertex_index_var_id()),
            _ => None,
        }
    }

    fn load_system_attr_bits(&mut self, slot: u32) -> Option<Word> {
        let var = self.system_attr_var_id(slot)?;
        let raw = self.b.load(self.u32_t, None, var, None, []).unwrap();
        let raw = if slot == 0x2f8 {
            let base_var = self.base_instance_var_id();
            let base = self.b.load(self.u32_t, None, base_var, None, []).unwrap();
            self.b.i_sub(self.u32_t, None, raw, base).unwrap()
        } else {
            raw
        };
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

    fn ensure_image_array(&mut self) -> Word {
        if let Some(img) = self.image_var {
            return img;
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
        self.b.decorate(
            img,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_FLOAT_2D)],
        );
        self.image_var = Some(img);
        img
    }

    fn ensure_sampler_array(&mut self) -> (Word, Word) {
        let img = self.ensure_image_array();
        let samp = self.ensure_sampler_only();
        (img, samp)
    }

    fn ensure_sampler_only(&mut self) -> Word {
        if let Some(samp) = self.sampler_var {
            return samp;
        }
        let samp = self.b.variable(
            self.ptr_sampler_array,
            None,
            StorageClass::UniformConstant,
            None,
        );
        self.b
            .decorate(samp, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            samp,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_SAMPLERS)],
        );
        self.sampler_var = Some(samp);
        samp
    }

    fn ensure_image_buffer_array(&mut self) -> Word {
        if let Some(var) = self.image_buffer_var {
            return var;
        }
        self.b.capability(Capability::SampledBuffer);
        let image_t = self.b.type_image(
            self.f32_t,
            rspirv::spirv::Dim::DimBuffer,
            0,
            0,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let ptr_image = self
            .b
            .type_pointer(None, StorageClass::UniformConstant, image_t);
        let texture_slots = self.b.constant_bit32(self.u32_t, MAX_TEXTURE_DESCRIPTORS);
        let image_array_t = self.b.type_array(image_t, texture_slots);
        let ptr_image_array =
            self.b
                .type_pointer(None, StorageClass::UniformConstant, image_array_t);
        let var = self
            .b
            .variable(ptr_image_array, None, StorageClass::UniformConstant, None);
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_FLOAT_TEXEL_BUFFER)],
        );
        self.image_buffer_t = Some(image_t);
        self.ptr_image_buffer = Some(ptr_image);
        self.image_buffer_var = Some(var);
        var
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
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_FLOAT_3D)],
        );
        self.image_3d_var = Some(var);
        var
    }

    fn ensure_image_cube_array(&mut self) -> Word {
        if let Some(var) = self.image_cube_var {
            return var;
        }
        let var = self.b.variable(
            self.ptr_image_cube_array,
            None,
            StorageClass::UniformConstant,
            None,
        );
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_FLOAT_CUBE)],
        );
        self.image_cube_var = Some(var);
        var
    }

    fn ensure_image_cube_arrayed_array(&mut self) -> Word {
        if let Some(var) = self.image_cube_arrayed_var {
            return var;
        }
        self.b.capability(Capability::ImageCubeArray);
        let var = self.b.variable(
            self.ptr_image_cube_arrayed_array,
            None,
            StorageClass::UniformConstant,
            None,
        );
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(GFX_BINDING_FLOAT_CUBE_ARRAY)],
        );
        self.image_cube_arrayed_var = Some(var);
        var
    }

    fn graphics_numeric_scalar_type(&self, numeric_type: TextureNumericType) -> Word {
        match numeric_type {
            TextureNumericType::Float => self.f32_t,
            TextureNumericType::Uint => self.u32_t,
            TextureNumericType::Sint => self.i32_t,
        }
    }

    fn graphics_numeric_vec4_type(&self, numeric_type: TextureNumericType) -> Word {
        match numeric_type {
            TextureNumericType::Float => self.vec4_t,
            TextureNumericType::Uint => self.uvec4_t,
            TextureNumericType::Sint => self.ivec4_t,
        }
    }

    fn graphics_image_binding(numeric_type: TextureNumericType, kind: GraphicsImageKind) -> u32 {
        graphics_image_binding(numeric_type, kind)
    }

    fn ensure_typed_image_array(
        &mut self,
        numeric_type: TextureNumericType,
        kind: GraphicsImageKind,
    ) -> GraphicsImageDecl {
        if numeric_type == TextureNumericType::Float {
            return match kind {
                GraphicsImageKind::D2 | GraphicsImageKind::D2Array => {
                    let var = self.ensure_image_array();
                    if self.sampler_arrayed {
                        GraphicsImageDecl {
                            image_t: self.image_arrayed_t,
                            ptr_image: self.ptr_image_arrayed,
                            var,
                        }
                    } else {
                        GraphicsImageDecl {
                            image_t: self.image_t,
                            ptr_image: self.ptr_image,
                            var,
                        }
                    }
                }
                GraphicsImageKind::D3 => GraphicsImageDecl {
                    image_t: self.image_3d_t,
                    ptr_image: self.ptr_image_3d,
                    var: self.ensure_image_3d_array(),
                },
                GraphicsImageKind::Cube => GraphicsImageDecl {
                    image_t: self.image_cube_t,
                    ptr_image: self.ptr_image_cube,
                    var: self.ensure_image_cube_array(),
                },
                GraphicsImageKind::CubeArray => GraphicsImageDecl {
                    image_t: self.image_cube_arrayed_t,
                    ptr_image: self.ptr_image_cube_arrayed,
                    var: self.ensure_image_cube_arrayed_array(),
                },
                GraphicsImageKind::Buffer => {
                    let var = self.ensure_image_buffer_array();
                    GraphicsImageDecl {
                        image_t: self.image_buffer_t.expect("buffer image type"),
                        ptr_image: self.ptr_image_buffer.expect("buffer image pointer type"),
                        var,
                    }
                }
            };
        }

        if let Some(decl) = self.typed_image_decls.get(&(numeric_type, kind)).copied() {
            return decl;
        }

        let (dimension, arrayed) = match kind {
            GraphicsImageKind::D2 => (rspirv::spirv::Dim::Dim2D, 0),
            GraphicsImageKind::D2Array => (rspirv::spirv::Dim::Dim2D, 1),
            GraphicsImageKind::D3 => (rspirv::spirv::Dim::Dim3D, 0),
            GraphicsImageKind::Cube => (rspirv::spirv::Dim::DimCube, 0),
            GraphicsImageKind::CubeArray => {
                self.b.capability(Capability::ImageCubeArray);
                (rspirv::spirv::Dim::DimCube, 1)
            }
            GraphicsImageKind::Buffer => {
                self.b.capability(Capability::SampledBuffer);
                (rspirv::spirv::Dim::DimBuffer, 0)
            }
        };
        let scalar_t = self.graphics_numeric_scalar_type(numeric_type);
        let image_t = self.b.type_image(
            scalar_t,
            dimension,
            0,
            arrayed,
            0,
            1,
            ImageFormat::Unknown,
            None,
        );
        let ptr_image = self
            .b
            .type_pointer(None, StorageClass::UniformConstant, image_t);
        let texture_slots = self.b.constant_bit32(self.u32_t, MAX_TEXTURE_DESCRIPTORS);
        let image_array_t = self.b.type_array(image_t, texture_slots);
        let ptr_image_array =
            self.b
                .type_pointer(None, StorageClass::UniformConstant, image_array_t);
        let var = self
            .b
            .variable(ptr_image_array, None, StorageClass::UniformConstant, None);
        self.b
            .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(
            var,
            Decoration::Binding,
            [Operand::LiteralBit32(Self::graphics_image_binding(
                numeric_type,
                kind,
            ))],
        );
        let decl = GraphicsImageDecl {
            image_t,
            ptr_image,
            var,
        };
        self.typed_image_decls.insert((numeric_type, kind), decl);
        decl
    }

    fn texture_slot(&self, tex_id: u32) -> u32 {
        self.texture_slots
            .get(&tex_id)
            .copied()
            .unwrap_or(0)
            .min(MAX_TEXTURE_DESCRIPTORS - 1)
    }

    fn texture_numeric_type_at(&self, tex_id: u32) -> TextureNumericType {
        if self.texture_numeric_manifest.is_empty() {
            return TextureNumericType::Float;
        }
        let slot = self.texture_slot(tex_id);
        self.texture_numeric_manifest
            .binary_search_by_key(&slot, |resource| resource.descriptor_slot)
            .ok()
            .map(|index| self.texture_numeric_manifest[index].numeric_type)
            .unwrap_or(TextureNumericType::Float)
    }

    fn validate_graphics_texture_manifest(
        &self,
        texture_kinds: &std::collections::BTreeMap<u32, GraphicsImageKind>,
    ) -> Result<(), GraphicsTextureManifestError> {
        if self.texture_numeric_manifest.is_empty() {
            return Ok(());
        }

        let base = self.vertex_opts.tex_slot_base;
        if base as usize + texture_kinds.len() > MAX_TEXTURE_DESCRIPTORS as usize {
            return Err(GraphicsTextureManifestError::TooManyResources {
                base,
                count: texture_kinds.len(),
            });
        }

        let expected = texture_kinds
            .iter()
            .enumerate()
            .map(|(index, (&shader_id, &image_kind))| {
                (base + index as u32, (shader_id, image_kind))
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        for (&slot, &(shader_id, image_kind)) in &expected {
            let resource = self
                .texture_numeric_manifest
                .binary_search_by_key(&slot, |resource| resource.descriptor_slot)
                .ok()
                .map(|index| self.texture_numeric_manifest[index])
                .ok_or(GraphicsTextureManifestError::MissingResource { slot, shader_id })?;
            if resource.shader_id != shader_id {
                return Err(GraphicsTextureManifestError::ShaderIdMismatch {
                    slot,
                    expected_shader_id: shader_id,
                    actual_shader_id: resource.shader_id,
                });
            }
            if resource.image_kind != image_kind {
                return Err(GraphicsTextureManifestError::ImageKindMismatch {
                    slot,
                    shader_id,
                    expected: image_kind,
                    actual: resource.image_kind,
                });
            }
        }
        for resource in &self.texture_numeric_manifest {
            if !expected.contains_key(&resource.descriptor_slot) {
                return Err(GraphicsTextureManifestError::UnexpectedResource {
                    slot: resource.descriptor_slot,
                    shader_id: resource.shader_id,
                });
            }
        }
        Ok(())
    }

    fn typed_image_at(
        &mut self,
        tex_id: u32,
        numeric_type: TextureNumericType,
        kind: GraphicsImageKind,
    ) -> (GraphicsImageDecl, Word) {
        let decl = self.ensure_typed_image_array(numeric_type, kind);
        let idx = self.const_u32(self.texture_slot(tex_id));
        let pointer = self
            .b
            .access_chain(decl.ptr_image, None, decl.var, [idx])
            .unwrap();
        (decl, pointer)
    }

    fn typed_fetch_image_at(
        &mut self,
        tex_id: u32,
        numeric_type: TextureNumericType,
        kind: GraphicsImageKind,
    ) -> (GraphicsImageDecl, Word) {
        assert!(
            !matches!(kind, GraphicsImageKind::Cube | GraphicsImageKind::CubeArray),
            "graphics cube texel fetch is unsupported: the current IR does not retain cube face semantics"
        );
        self.typed_image_at(tex_id, numeric_type, kind)
    }

    fn graphics_fetch_component_as_f32(
        &mut self,
        fetched: Word,
        numeric_type: TextureNumericType,
        component: u8,
    ) -> Word {
        let scalar_t = self.graphics_numeric_scalar_type(numeric_type);
        let component = self
            .b
            .composite_extract(scalar_t, None, fetched, [component.min(3) as u32])
            .unwrap();
        match numeric_type {
            TextureNumericType::Float => component,
            TextureNumericType::Uint | TextureNumericType::Sint => {
                self.b.bitcast(self.f32_t, None, component).unwrap()
            }
        }
    }

    fn setup_compute_resources(&mut self) {
        if !self.compute_resource_vars.is_empty() {
            return;
        }
        let resources = self
            .compute_options
            .as_ref()
            .map(|options| options.resources.clone())
            .unwrap_or_default();
        for resource in resources {
            let scalar_t = match resource.numeric_type {
                TextureNumericType::Float => self.f32_t,
                TextureNumericType::Uint => self.u32_t,
                TextureNumericType::Sint => self.i32_t,
            };
            let vec4_t = match resource.numeric_type {
                TextureNumericType::Float => self.vec4_t,
                TextureNumericType::Uint => self.uvec4_t,
                TextureNumericType::Sint => self.ivec4_t,
            };
            let dimension = match resource.dimension {
                ImageDimension::D1 => rspirv::spirv::Dim::Dim1D,
                ImageDimension::Buffer => rspirv::spirv::Dim::DimBuffer,
                ImageDimension::D2 => rspirv::spirv::Dim::Dim2D,
                ImageDimension::D3 => rspirv::spirv::Dim::Dim3D,
            };
            match resource.kind {
                ComputeResourceKind::UniformTexelBuffer => {
                    self.b.capability(Capability::SampledBuffer);
                }
                ComputeResourceKind::StorageTexelBuffer => {
                    self.b.capability(Capability::ImageBuffer);
                    let format = resource
                        .texel_format
                        .expect("validated storage texel resource format");
                    if format.requires_storage_image_extended_formats() {
                        self.b.capability(Capability::StorageImageExtendedFormats);
                    }
                    if format.supports_storage_atomics() {
                        self.ptr_image_u32.get_or_insert_with(|| {
                            self.b.type_pointer(None, StorageClass::Image, self.u32_t)
                        });
                    }
                }
                ComputeResourceKind::StorageImage => {
                    self.b
                        .capability(Capability::StorageImageWriteWithoutFormat);
                }
                ComputeResourceKind::CombinedSampledImage | ComputeResourceKind::SampledImage => {}
            }
            let sampled = if matches!(
                resource.kind,
                ComputeResourceKind::StorageImage | ComputeResourceKind::StorageTexelBuffer
            ) {
                2
            } else {
                1
            };
            let image_format = match resource.kind {
                ComputeResourceKind::StorageTexelBuffer => resource
                    .texel_format
                    .expect("validated storage texel resource format")
                    .storage_image_format(),
                _ => ImageFormat::Unknown,
            };
            let image_t =
                self.b
                    .type_image(scalar_t, dimension, 0, 0, 0, sampled, image_format, None);
            let sampled_image_t = (resource.kind == ComputeResourceKind::CombinedSampledImage)
                .then(|| self.b.type_sampled_image(image_t));
            let descriptor_t = sampled_image_t.unwrap_or(image_t);
            let ptr_t = self
                .b
                .type_pointer(None, StorageClass::UniformConstant, descriptor_t);
            let var = self
                .b
                .variable(ptr_t, None, StorageClass::UniformConstant, None);
            self.b
                .decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            self.b.decorate(
                var,
                Decoration::Binding,
                [Operand::LiteralBit32(resource.binding)],
            );
            self.compute_resource_vars.push(ComputeResourceVar {
                resource,
                image_t,
                sampled_image_t,
                scalar_t,
                vec4_t,
                var,
            });
        }
    }

    fn compute_sampled_resource(&self, handle: TextureHandleOrigin) -> ComputeResourceVar {
        self.compute_resource_vars
            .iter()
            .copied()
            .find(|resource| {
                resource.resource.handle == handle
                    && matches!(
                        resource.resource.kind,
                        ComputeResourceKind::SampledImage | ComputeResourceKind::UniformTexelBuffer
                    )
            })
            .expect("compute resource metadata was validated before emission")
    }

    fn compute_storage_resource(
        &self,
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
    ) -> ComputeResourceVar {
        self.compute_resource_vars
            .iter()
            .copied()
            .find(|resource| {
                resource.resource.handle == handle
                    && resource.resource.dimension == dimension
                    && matches!(
                        resource.resource.kind,
                        ComputeResourceKind::StorageImage | ComputeResourceKind::StorageTexelBuffer
                    )
            })
            .expect("compute resource metadata was validated before emission")
    }

    fn compute_storage_texel_resource(&self, handle: TextureHandleOrigin) -> ComputeResourceVar {
        self.compute_resource_vars
            .iter()
            .copied()
            .find(|resource| {
                resource.resource.handle == handle
                    && resource.resource.kind == ComputeResourceKind::StorageTexelBuffer
            })
            .expect("compute resource metadata was validated before emission")
    }

    fn compute_filtered_resource(&self, handle: TextureHandleOrigin) -> ComputeResourceVar {
        self.compute_resource_vars
            .iter()
            .copied()
            .find(|resource| {
                resource.resource.handle == handle
                    && resource.resource.kind == ComputeResourceKind::CombinedSampledImage
            })
            .expect("compute resource metadata was validated before emission")
    }

    fn load_compute_image(&mut self, resource: ComputeResourceVar) -> Word {
        self.b
            .load(resource.image_t, None, resource.var, None, [])
            .unwrap()
    }

    fn load_compute_sampled_image(&mut self, resource: ComputeResourceVar) -> Word {
        self.b
            .load(
                resource
                    .sampled_image_t
                    .expect("filtered compute resource must have a sampled-image type"),
                None,
                resource.var,
                None,
                [],
            )
            .unwrap()
    }

    fn compute_image_coords(
        &mut self,
        dimension: ImageDimension,
        x: &IrValue,
        y: Option<&IrValue>,
        z: Option<&IrValue>,
    ) -> Word {
        let x_value = self.lower_value(x);
        let x = self.as_i32(x_value);
        match dimension {
            ImageDimension::D1 | ImageDimension::Buffer => x,
            ImageDimension::D2 => {
                let y_value = self.lower_value(y.expect("2D image operation requires Y"));
                let y = self.as_i32(y_value);
                self.b
                    .composite_construct(self.ivec2_t, None, [x, y])
                    .unwrap()
            }
            ImageDimension::D3 => {
                let y_value = self.lower_value(y.expect("3D image operation requires Y"));
                let y = self.as_i32(y_value);
                let z_value = self.lower_value(z.expect("3D image operation requires Z"));
                let z = self.as_i32(z_value);
                self.b
                    .composite_construct(self.ivec3_t, None, [x, y, z])
                    .unwrap()
            }
        }
    }

    fn lower_compute_texel_fetch(
        &mut self,
        handle: TextureHandleOrigin,
        x: &IrValue,
        y: Option<&IrValue>,
        z: Option<&IrValue>,
        component: u8,
    ) -> Word {
        let resource = self.compute_sampled_resource(handle);
        let image = self.load_compute_image(resource);
        let coords = self.compute_image_coords(resource.resource.dimension, x, y, z);
        let (operands, params) = if resource.resource.dimension == ImageDimension::Buffer {
            (None, Vec::new())
        } else {
            (
                Some(rspirv::spirv::ImageOperands::LOD),
                vec![Operand::IdRef(self.i32_zero)],
            )
        };
        let fetched = self
            .b
            .image_fetch(resource.vec4_t, None, image, coords, operands, params)
            .unwrap();
        let component = self
            .b
            .composite_extract(
                resource.scalar_t,
                None,
                fetched,
                [(component.min(3)) as u32],
            )
            .unwrap();
        match resource.resource.numeric_type {
            TextureNumericType::Float => component,
            TextureNumericType::Uint | TextureNumericType::Sint => {
                self.b.bitcast(self.f32_t, None, component).unwrap()
            }
        }
    }

    fn lower_compute_texture_sample(
        &mut self,
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        u: &IrValue,
        v: Option<&IrValue>,
        w: Option<&IrValue>,
        implicit_lod: bool,
        explicit_lod: Option<&IrValue>,
        texel_offset: Option<&(IrValue, IrValue)>,
        component: u8,
    ) -> Word {
        use rspirv::spirv::ImageOperands;

        let resource = self.compute_filtered_resource(handle);
        debug_assert_eq!(resource.resource.dimension, dimension);
        let sampled_image = self.load_compute_sampled_image(resource);
        let u = self.lower_value(u);
        let coords = match dimension {
            ImageDimension::D1 => u,
            ImageDimension::D2 => {
                let v = self.lower_value(v.expect("2D filtered sample requires V"));
                self.b
                    .composite_construct(self.vec2_t, None, [u, v])
                    .unwrap()
            }
            ImageDimension::D3 => {
                let v = self.lower_value(v.expect("3D filtered sample requires V"));
                let w = self.lower_value(w.expect("3D filtered sample requires W"));
                self.b
                    .composite_construct(self.vec3_t, None, [u, v, w])
                    .unwrap()
            }
            ImageDimension::Buffer => unreachable!("buffer images cannot be filtered"),
        };
        let lod = if implicit_lod {
            self.f32_zero
        } else {
            explicit_lod
                .map(|lod| self.lower_value(lod))
                .unwrap_or(self.f32_zero)
        };
        let offset = texel_offset.map(|(x, y)| {
            let constant = |emitter: &mut Self, value: &IrValue| {
                let bits = match value {
                    IrValue::Zero => 0,
                    IrValue::ImmU32(bits) => *bits,
                    IrValue::ImmF32(value) => value.to_bits(),
                    IrValue::GprIn(_) | IrValue::Inst(_) => {
                        unreachable!("compute sample offset was validated as constant")
                    }
                };
                emitter.b.constant_bit32(emitter.i32_t, bits)
            };
            let x = constant(self, x);
            match dimension {
                ImageDimension::D1 => x,
                ImageDimension::D2 => {
                    let y = constant(self, y);
                    self.b.constant_composite(self.ivec2_t, [x, y])
                }
                ImageDimension::D3 | ImageDimension::Buffer => {
                    unreachable!("validated compute sample offset dimension")
                }
            }
        });
        let sampled = if let Some(offset) = offset {
            self.b
                .image_sample_explicit_lod(
                    resource.vec4_t,
                    None,
                    sampled_image,
                    coords,
                    ImageOperands::LOD | ImageOperands::CONST_OFFSET,
                    [Operand::IdRef(lod), Operand::IdRef(offset)],
                )
                .unwrap()
        } else {
            self.b
                .image_sample_explicit_lod(
                    resource.vec4_t,
                    None,
                    sampled_image,
                    coords,
                    ImageOperands::LOD,
                    [Operand::IdRef(lod)],
                )
                .unwrap()
        };
        let component = self
            .b
            .composite_extract(resource.scalar_t, None, sampled, [component.min(3) as u32])
            .unwrap();
        match resource.resource.numeric_type {
            TextureNumericType::Float => component,
            TextureNumericType::Uint | TextureNumericType::Sint => {
                self.b.bitcast(self.f32_t, None, component).unwrap()
            }
        }
    }

    fn lower_compute_texture_query(
        &mut self,
        handle: TextureHandleOrigin,
        lod: &IrValue,
        component: u8,
    ) -> Word {
        let resource = self.compute_sampled_resource(handle);
        let image = self.load_compute_image(resource);
        let result = if component == 3 {
            if resource.resource.dimension == ImageDimension::Buffer {
                self.const_u32(0)
            } else {
                self.b.image_query_levels(self.u32_t, None, image).unwrap()
            }
        } else {
            let component_count = match resource.resource.dimension {
                ImageDimension::D1 | ImageDimension::Buffer => 1,
                ImageDimension::D2 => 2,
                ImageDimension::D3 => 3,
            };
            if component as u32 >= component_count {
                self.const_u32(0)
            } else {
                let result_t = match resource.resource.dimension {
                    ImageDimension::D1 | ImageDimension::Buffer => self.u32_t,
                    ImageDimension::D2 => self.uvec2_t,
                    ImageDimension::D3 => self.uvec3_t,
                };
                let dimensions = if resource.resource.dimension == ImageDimension::Buffer {
                    self.b.image_query_size(result_t, None, image).unwrap()
                } else {
                    let lod_value = self.lower_value(lod);
                    let lod = self.as_i32(lod_value);
                    self.b
                        .image_query_size_lod(result_t, None, image, lod)
                        .unwrap()
                };
                if component_count == 1 {
                    dimensions
                } else {
                    self.b
                        .composite_extract(self.u32_t, None, dimensions, [component as u32])
                        .unwrap()
                }
            }
        };
        self.store_bits(result)
    }

    fn shared_word_index(&mut self, addr: &IrValue) -> Word {
        let addr_value = self.lower_value(addr);
        let addr_u32 = self.as_u32(addr_value);
        let shift = self.const_u32(2);
        self
            .b
            .shift_right_logical(self.u32_t, None, addr_u32, shift)
            .unwrap()
    }

    fn shared_word_in_bounds(&mut self, word_index: Word) -> Word {
        let word_count = self
            .compute_options
            .as_ref()
            .expect("compute options are required for shared memory")
            .shared_memory_size
            .div_ceil(4);
        let word_count = self.const_u32(word_count);
        let in_bounds = self
            .b
            .u_less_than(self.bool_t, None, word_index, word_count)
            .unwrap();
        in_bounds
    }

    fn shared_word_pointer_from_index(&mut self, word_index: Word) -> Word {
        let shared = self
            .shared_mem_var
            .expect("shared memory must be preallocated");
        self.b
            .access_chain(self.ptr_workgroup_u32, None, shared, [word_index])
            .unwrap()
    }

    fn shared_word_pointer(&mut self, addr: &IrValue) -> Word {
        let word_index = self.shared_word_index(addr);
        self.shared_word_pointer_from_index(word_index)
    }

    fn lower_shared_store(&mut self, inst: &IrInst, addr: &IrValue, value: &IrValue) {
        let pointer = self.shared_word_pointer(addr);
        let value_word = self.lower_value(value);
        let value_u32 = self.as_u32(value_word);
        let guard = inst
            .pred
            .map(|predicate| self.resolve_pred(predicate.idx, predicate.negate));
        if let Some(guard) = guard {
            let store_block = self.b.id();
            let merge_block = self.b.id();
            self.b
                .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
                .unwrap();
            self.b
                .branch_conditional(guard, store_block, merge_block, [])
                .unwrap();
            self.b.begin_block(Some(store_block)).unwrap();
            self.b.store(pointer, value_u32, None, []).unwrap();
            self.b.branch(merge_block).unwrap();
            self.b.begin_block(Some(merge_block)).unwrap();
            if let Some(block) = self.current_block {
                self.block_end_labels.insert(block, merge_block);
            }
        } else {
            self.b.store(pointer, value_u32, None, []).unwrap();
        }
    }

    fn lower_shared_atomic(
        &mut self,
        inst: &IrInst,
        addr: &IrValue,
        value: &IrValue,
        op: ImageAtomicOp,
    ) -> Word {
        let word_index = self.shared_word_index(addr);
        let in_bounds = self.shared_word_in_bounds(word_index);
        let value_word = self.lower_value(value);
        let value = self.as_u32(value_word);
        let guard = if let Some(predicate) = inst.pred {
            let predicate = self.resolve_pred(predicate.idx, predicate.negate);
            self.b
                .logical_and(self.bool_t, None, in_bounds, predicate)
                .unwrap()
        } else {
            in_bounds
        };

        let current_block = self
            .current_block
            .expect("shared atomic must be emitted inside a CFG block");
        let header_block = self
            .block_end_labels
            .get(&current_block)
            .copied()
            .unwrap_or(self.block_labels[&current_block]);
        let atomic_block = self.b.id();
        let merge_block = self.b.id();
        let false_value = self.const_u32(0);
        self.b
            .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
            .unwrap();
        self.b
            .branch_conditional(guard, atomic_block, merge_block, [])
            .unwrap();

        self.b.begin_block(Some(atomic_block)).unwrap();
        let pointer = self.shared_word_pointer_from_index(word_index);
        let scope = self.const_u32(Scope::Workgroup as u32);
        let semantics = self.const_u32(MemorySemantics::NONE.bits());
        let (atomic, atomic_end) = self.emit_compute_image_atomic(
            op,
            ImageAtomicType::U32,
            pointer,
            scope,
            semantics,
            value,
            atomic_block,
        );
        self.b.branch(merge_block).unwrap();

        self.b.begin_block(Some(merge_block)).unwrap();
        let result = self
            .b
            .phi(
                self.u32_t,
                None,
                [(atomic, atomic_end), (false_value, header_block)],
            )
            .unwrap();
        self.block_end_labels.insert(current_block, merge_block);
        self.store_bits(result)
    }

    fn lower_workgroup_barrier(&mut self) {
        let workgroup = self.const_u32(Scope::Workgroup as u32);
        let semantics = self.const_u32(
            (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
        );
        self.b
            .control_barrier(workgroup, workgroup, semantics)
            .unwrap();
    }

    fn lower_memory_barrier(&mut self, scope: MemoryBarrierScope) {
        let scope = self.const_u32(match scope {
            MemoryBarrierScope::Workgroup => Scope::Workgroup as u32,
            MemoryBarrierScope::Device => Scope::Device as u32,
        });
        let semantics = self.const_u32(
            (MemorySemantics::ACQUIRE_RELEASE
                | MemorySemantics::UNIFORM_MEMORY
                | MemorySemantics::WORKGROUP_MEMORY
                | MemorySemantics::ATOMIC_COUNTER_MEMORY
                | MemorySemantics::IMAGE_MEMORY)
                .bits(),
        );
        self.b.memory_barrier(scope, semantics).unwrap();
    }

    fn lower_compute_image_write(
        &mut self,
        inst: &IrInst,
        handle: TextureHandleOrigin,
        dimension: ImageDimension,
        x: &IrValue,
        y: Option<&IrValue>,
        z: Option<&IrValue>,
        values: &[IrValue; 4],
    ) {
        let resource = self.compute_storage_resource(handle, dimension);
        let image = self.load_compute_image(resource);
        let coords = self.compute_image_coords(resource.resource.dimension, x, y, z);
        let raw = values.map(|value| self.lower_value(&value));
        let texel = match resource.resource.numeric_type {
            TextureNumericType::Float => self
                .b
                .composite_construct(resource.vec4_t, None, raw)
                .unwrap(),
            TextureNumericType::Uint => {
                let converted = raw.map(|value| self.as_u32(value));
                self.b
                    .composite_construct(resource.vec4_t, None, converted)
                    .unwrap()
            }
            TextureNumericType::Sint => {
                let converted = raw.map(|value| self.as_i32(value));
                self.b
                    .composite_construct(resource.vec4_t, None, converted)
                    .unwrap()
            }
        };
        let guard = inst
            .pred
            .map(|predicate| self.resolve_pred(predicate.idx, predicate.negate));
        if let Some(guard) = guard {
            let write_block = self.b.id();
            let merge_block = self.b.id();
            self.b
                .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
                .unwrap();
            self.b
                .branch_conditional(guard, write_block, merge_block, [])
                .unwrap();
            self.b.begin_block(Some(write_block)).unwrap();
            self.b.image_write(image, coords, texel, None, []).unwrap();
            self.b.branch(merge_block).unwrap();
            self.b.begin_block(Some(merge_block)).unwrap();
            if let Some(block) = self.current_block {
                self.block_end_labels.insert(block, merge_block);
            }
        } else {
            self.b.image_write(image, coords, texel, None, []).unwrap();
        }
    }

    fn emit_compute_image_atomic(
        &mut self,
        op: ImageAtomicOp,
        data_type: ImageAtomicType,
        pointer: Word,
        scope: Word,
        semantics: Word,
        value: Word,
        entry_label: Word,
    ) -> (Word, Word) {
        match op {
            ImageAtomicOp::Add => (
                self.b
                    .atomic_i_add(self.u32_t, None, pointer, scope, semantics, value)
                    .unwrap(),
                entry_label,
            ),
            ImageAtomicOp::Min => (
                if data_type.is_signed() {
                    self.b
                        .atomic_s_min(self.u32_t, None, pointer, scope, semantics, value)
                        .unwrap()
                } else {
                    self.b
                        .atomic_u_min(self.u32_t, None, pointer, scope, semantics, value)
                        .unwrap()
                },
                entry_label,
            ),
            ImageAtomicOp::Max => (
                if data_type.is_signed() {
                    self.b
                        .atomic_s_max(self.u32_t, None, pointer, scope, semantics, value)
                        .unwrap()
                } else {
                    self.b
                        .atomic_u_max(self.u32_t, None, pointer, scope, semantics, value)
                        .unwrap()
                },
                entry_label,
            ),
            ImageAtomicOp::Increment | ImageAtomicOp::Decrement => self
                .emit_compute_image_atomic_wrap(op, pointer, scope, semantics, value, entry_label),
            ImageAtomicOp::And => (
                self.b
                    .atomic_and(self.u32_t, None, pointer, scope, semantics, value)
                    .unwrap(),
                entry_label,
            ),
            ImageAtomicOp::Or => (
                self.b
                    .atomic_or(self.u32_t, None, pointer, scope, semantics, value)
                    .unwrap(),
                entry_label,
            ),
            ImageAtomicOp::Xor => (
                self.b
                    .atomic_xor(self.u32_t, None, pointer, scope, semantics, value)
                    .unwrap(),
                entry_label,
            ),
            ImageAtomicOp::Exchange => (
                self.b
                    .atomic_exchange(self.u32_t, None, pointer, scope, semantics, value)
                    .unwrap(),
                entry_label,
            ),
        }
    }

    fn emit_compute_image_atomic_wrap(
        &mut self,
        op: ImageAtomicOp,
        pointer: Word,
        scope: Word,
        semantics: Word,
        limit: Word,
        entry_label: Word,
    ) -> (Word, Word) {
        debug_assert!(matches!(
            op,
            ImageAtomicOp::Increment | ImageAtomicOp::Decrement
        ));
        let initial = self
            .b
            .atomic_load(self.u32_t, None, pointer, scope, semantics)
            .unwrap();
        let header = self.b.id();
        let body = self.b.id();
        let cont = self.b.id();
        let merge = self.b.id();
        let expected = self.b.id();
        let observed = self.b.id();
        self.b.branch(header).unwrap();

        self.b.begin_block(Some(header)).unwrap();
        self.b
            .phi(
                self.u32_t,
                Some(expected),
                [(initial, entry_label), (observed, cont)],
            )
            .unwrap();
        self.b
            .loop_merge(merge, cont, rspirv::spirv::LoopControl::NONE, [])
            .unwrap();
        self.b.branch(body).unwrap();

        self.b.begin_block(Some(body)).unwrap();
        let zero = self.const_u32(0);
        let one = self.const_u32(1);
        let replacement = match op {
            ImageAtomicOp::Increment => {
                let wrapped = self
                    .b
                    .u_greater_than_equal(self.bool_t, None, expected, limit)
                    .unwrap();
                let incremented = self.b.i_add(self.u32_t, None, expected, one).unwrap();
                self.b
                    .select(self.u32_t, None, wrapped, zero, incremented)
                    .unwrap()
            }
            ImageAtomicOp::Decrement => {
                let is_zero = self.b.i_equal(self.bool_t, None, expected, zero).unwrap();
                let above_limit = self
                    .b
                    .u_greater_than(self.bool_t, None, expected, limit)
                    .unwrap();
                let wrapped = self
                    .b
                    .logical_or(self.bool_t, None, is_zero, above_limit)
                    .unwrap();
                let decremented = self.b.i_sub(self.u32_t, None, expected, one).unwrap();
                self.b
                    .select(self.u32_t, None, wrapped, limit, decremented)
                    .unwrap()
            }
            _ => unreachable!(),
        };
        self.b
            .atomic_compare_exchange(
                self.u32_t,
                Some(observed),
                pointer,
                scope,
                semantics,
                semantics,
                replacement,
                expected,
            )
            .unwrap();
        let succeeded = self
            .b
            .i_equal(self.bool_t, None, observed, expected)
            .unwrap();
        self.b
            .branch_conditional(succeeded, merge, cont, [])
            .unwrap();

        self.b.begin_block(Some(cont)).unwrap();
        self.b.branch(header).unwrap();
        self.b.begin_block(Some(merge)).unwrap();
        (observed, merge)
    }

    fn lower_compute_image_atomic(
        &mut self,
        inst: &IrInst,
        handle: TextureHandleOrigin,
        x: &IrValue,
        value: &IrValue,
        op: ImageAtomicOp,
        data_type: ImageAtomicType,
    ) -> Word {
        let resource = self.compute_storage_texel_resource(handle);
        let coords = self.compute_image_coords(ImageDimension::Buffer, x, None, None);
        let pointer = self
            .b
            .image_texel_pointer(
                self.ptr_image_u32
                    .expect("storage texel resource must declare an image pointer"),
                None,
                resource.var,
                coords,
                self.i32_zero,
            )
            .unwrap();
        let value_word = self.lower_value(value);
        let value = self.as_u32(value_word);
        let scope = self.const_u32(Scope::Device as u32);
        let semantics = self.const_u32(MemorySemantics::NONE.bits());

        let result = if let Some(predicate) = inst.pred {
            let guard = self.resolve_pred(predicate.idx, predicate.negate);
            let current_block = self
                .current_block
                .expect("compute atomic must be emitted inside a CFG block");
            let header_block = self
                .block_end_labels
                .get(&current_block)
                .copied()
                .unwrap_or(self.block_labels[&current_block]);
            let atomic_block = self.b.id();
            let merge_block = self.b.id();
            let false_value = self.const_u32(0);
            self.b
                .selection_merge(merge_block, rspirv::spirv::SelectionControl::NONE)
                .unwrap();
            self.b
                .branch_conditional(guard, atomic_block, merge_block, [])
                .unwrap();
            self.b.begin_block(Some(atomic_block)).unwrap();
            let (atomic, atomic_end) = self.emit_compute_image_atomic(
                op,
                data_type,
                pointer,
                scope,
                semantics,
                value,
                atomic_block,
            );
            self.b.branch(merge_block).unwrap();
            self.b.begin_block(Some(merge_block)).unwrap();
            let result = self
                .b
                .phi(
                    self.u32_t,
                    None,
                    [(atomic, atomic_end), (false_value, header_block)],
                )
                .unwrap();
            self.block_end_labels.insert(current_block, merge_block);
            result
        } else {
            let current_block = self
                .current_block
                .expect("compute atomic must be emitted inside a CFG block");
            let entry_label = self
                .block_end_labels
                .get(&current_block)
                .copied()
                .unwrap_or(self.block_labels[&current_block]);
            let (result, end_label) = self.emit_compute_image_atomic(
                op,
                data_type,
                pointer,
                scope,
                semantics,
                value,
                entry_label,
            );
            if end_label != entry_label {
                self.block_end_labels.insert(current_block, end_label);
            }
            result
        };
        self.store_bits(result)
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

    fn sampler_cube_at(&mut self, tex_id: u32) -> (Word, Word) {
        let samp_array = self.ensure_sampler_only();
        let img_array = self.ensure_image_cube_array();
        let idx = self
            .texture_slots
            .get(&tex_id)
            .copied()
            .unwrap_or(0)
            .min(MAX_TEXTURE_DESCRIPTORS - 1);
        let idx = self.const_u32(idx);
        let img = self
            .b
            .access_chain(self.ptr_image_cube, None, img_array, [idx])
            .unwrap();
        let samp = self
            .b
            .access_chain(self.ptr_sampler, None, samp_array, [idx])
            .unwrap();
        (img, samp)
    }

    fn sampler_cube_arrayed_at(&mut self, tex_id: u32) -> (Word, Word) {
        let samp_array = self.ensure_sampler_only();
        let img_array = self.ensure_image_cube_arrayed_array();
        let idx = self
            .texture_slots
            .get(&tex_id)
            .copied()
            .unwrap_or(0)
            .min(MAX_TEXTURE_DESCRIPTORS - 1);
        let idx = self.const_u32(idx);
        let img = self
            .b
            .access_chain(self.ptr_image_cube_arrayed, None, img_array, [idx])
            .unwrap();
        let samp = self
            .b
            .access_chain(self.ptr_sampler, None, samp_array, [idx])
            .unwrap();
        (img, samp)
    }

    fn texel_buffer_slot_enabled(&self, tex_id: u32) -> bool {
        self.texture_slots
            .get(&tex_id)
            .copied()
            .filter(|slot| *slot < MAX_TEXTURE_DESCRIPTORS)
            .is_some_and(|slot| self.texel_buffer_mask & (1u32 << slot) != 0)
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
            let layer_src = (matches!(self.stage, Stage::Vertex)
                && self.vertex_opts.layer_output_slot == Some(slot))
            .then_some(val);
            let av = self.output_var(aligned_slot);
            if guard.is_some() {
                let old = self.read_attr_component(av, component);
                val = self.select_guarded(val, guard, old);
            }
            self.write_attr_component(av, component, val);
            if let Some(layer_src) = layer_src {
                let layer = self.layer_var.expect("layer output must be preallocated");
                let mut layer_value = self.b.bitcast(self.u32_t, None, layer_src).unwrap();
                if let Some(cond) = guard {
                    let old = self.b.load(self.u32_t, None, layer, None, []).unwrap();
                    layer_value = self
                        .b
                        .select(self.u32_t, None, cond, layer_value, old)
                        .unwrap();
                }
                self.b.store(layer, layer_value, None, []).unwrap();
            }
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

    fn lower_half_pair(&mut self, value: &IrValue, swizzle: HalfSwizzle) -> [Word; 2] {
        let raw = self.lower_value(value);
        match swizzle {
            HalfSwizzle::F32 => [raw, raw],
            HalfSwizzle::H1H0 | HalfSwizzle::H0H0 | HalfSwizzle::H1H1 => {
                let bits = self.as_u32(raw);
                let vector = self
                    .b
                    .ext_inst(self.vec2_t, None, self.glsl, 62, [Operand::IdRef(bits)])
                    .unwrap();
                let low = self
                    .b
                    .composite_extract(self.f32_t, None, vector, [0])
                    .unwrap();
                let high = self
                    .b
                    .composite_extract(self.f32_t, None, vector, [1])
                    .unwrap();
                match swizzle {
                    HalfSwizzle::H1H0 => [low, high],
                    HalfSwizzle::H0H0 => [low, low],
                    HalfSwizzle::H1H1 => [high, high],
                    HalfSwizzle::F32 => unreachable!(),
                }
            }
        }
    }

    fn lower_half_pack(&mut self, lhs: Word, rhs: Word, old: &IrValue, merge: HalfMerge) -> Word {
        if matches!(merge, HalfMerge::F32) {
            return self.round_half_pair(lhs, rhs)[0];
        }
        let vector = match merge {
            HalfMerge::H1H0 => self
                .b
                .composite_construct(self.vec2_t, None, [lhs, rhs])
                .unwrap(),
            HalfMerge::MrgH0 | HalfMerge::MrgH1 => {
                let old_raw = self.lower_value(old);
                let old_bits = self.as_u32(old_raw);
                let old_vector = self
                    .b
                    .ext_inst(self.vec2_t, None, self.glsl, 62, [Operand::IdRef(old_bits)])
                    .unwrap();
                let index = if matches!(merge, HalfMerge::MrgH0) {
                    0
                } else {
                    1
                };
                let value = if index == 0 { lhs } else { rhs };
                self.b
                    .composite_insert(self.vec2_t, None, value, old_vector, [index])
                    .unwrap()
            }
            HalfMerge::F32 => unreachable!(),
        };
        let packed = self
            .b
            .ext_inst(self.u32_t, None, self.glsl, 58, [Operand::IdRef(vector)])
            .unwrap();
        self.store_bits(packed)
    }

    fn round_half_pair(&mut self, lhs: Word, rhs: Word) -> [Word; 2] {
        let vector = self
            .b
            .composite_construct(self.vec2_t, None, [lhs, rhs])
            .unwrap();
        let packed = self
            .b
            .ext_inst(self.u32_t, None, self.glsl, 58, [Operand::IdRef(vector)])
            .unwrap();
        let unpacked = self
            .b
            .ext_inst(self.vec2_t, None, self.glsl, 62, [Operand::IdRef(packed)])
            .unwrap();
        [
            self.b
                .composite_extract(self.f32_t, None, unpacked, [0])
                .unwrap(),
            self.b
                .composite_extract(self.f32_t, None, unpacked, [1])
                .unwrap(),
        ]
    }

    fn half_pair_promotes(
        swizzle_a: HalfSwizzle,
        swizzle_b: HalfSwizzle,
        swizzle_c: Option<HalfSwizzle>,
    ) -> bool {
        if !matches!(swizzle_a, HalfSwizzle::F32) {
            return true;
        }
        if !matches!(swizzle_b, HalfSwizzle::F32) {
            return true;
        }
        swizzle_c.map_or(false, |swizzle| !matches!(swizzle, HalfSwizzle::F32))
    }

    fn lower_half_abs_neg(&mut self, value: Word, abs: bool, neg: bool) -> Word {
        self.apply_neg_abs(value, neg, abs)
    }

    fn sample_image(
        &mut self,
        sampled_image: Word,
        coords: Word,
        implicit_lod: bool,
        lod_bias: Option<Word>,
        explicit_lod: Option<Word>,
        texel_offset: Option<Word>,
    ) -> Word {
        use rspirv::spirv::ImageOperands;

        if implicit_lod && matches!(self.stage, Stage::Fragment) {
            return match (lod_bias, texel_offset) {
                (Some(bias), Some(offset)) => self
                    .b
                    .image_sample_implicit_lod(
                        self.vec4_t,
                        None,
                        sampled_image,
                        coords,
                        Some(ImageOperands::BIAS | ImageOperands::CONST_OFFSET),
                        [Operand::IdRef(bias), Operand::IdRef(offset)],
                    )
                    .unwrap(),
                (Some(bias), None) => self
                    .b
                    .image_sample_implicit_lod(
                        self.vec4_t,
                        None,
                        sampled_image,
                        coords,
                        Some(ImageOperands::BIAS),
                        [Operand::IdRef(bias)],
                    )
                    .unwrap(),
                (None, Some(offset)) => self
                    .b
                    .image_sample_implicit_lod(
                        self.vec4_t,
                        None,
                        sampled_image,
                        coords,
                        Some(ImageOperands::CONST_OFFSET),
                        [Operand::IdRef(offset)],
                    )
                    .unwrap(),
                (None, None) => self
                    .b
                    .image_sample_implicit_lod(self.vec4_t, None, sampled_image, coords, None, [])
                    .unwrap(),
            };
        }

        let lod = explicit_lod.unwrap_or(self.f32_zero);
        if let Some(offset) = texel_offset {
            self.b
                .image_sample_explicit_lod(
                    self.vec4_t,
                    None,
                    sampled_image,
                    coords,
                    ImageOperands::LOD | ImageOperands::CONST_OFFSET,
                    [Operand::IdRef(lod), Operand::IdRef(offset)],
                )
                .unwrap()
        } else {
            self.b
                .image_sample_explicit_lod(
                    self.vec4_t,
                    None,
                    sampled_image,
                    coords,
                    ImageOperands::LOD,
                    [Operand::IdRef(lod)],
                )
                .unwrap()
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
            IrOp::DpdxFine { src } => {
                let value = self.lower_value(src);
                Some(self.b.d_pdx_fine(self.f32_t, None, value).unwrap())
            }
            IrOp::DpdyFine { src } => {
                let value = self.lower_value(src);
                Some(self.b.d_pdy_fine(self.f32_t, None, value).unwrap())
            }
            IrOp::YDirection => {
                Some(self.const_f32(if self.y_negate { -1.0f32 } else { 1.0f32 }.to_bits()))
            }
            IrOp::HAdd {
                a,
                b,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                abs_a,
                neg_a,
                abs_b,
                neg_b,
                sat,
                ..
            } => {
                let [a0, a1] = self.lower_half_pair(a, *swizzle_a);
                let [b0, b1] = self.lower_half_pair(b, *swizzle_b);
                let a0 = self.lower_half_abs_neg(a0, *abs_a, *neg_a);
                let a1 = self.lower_half_abs_neg(a1, *abs_a, *neg_a);
                let b0 = self.lower_half_abs_neg(b0, *abs_b, *neg_b);
                let b1 = self.lower_half_abs_neg(b1, *abs_b, *neg_b);
                let lhs = self.b.f_add(self.f32_t, None, a0, b0).unwrap();
                let rhs = self.b.f_add(self.f32_t, None, a1, b1).unwrap();
                let [lhs, rhs] = if Self::half_pair_promotes(*swizzle_a, *swizzle_b, None) {
                    self.round_half_pair(lhs, rhs)
                } else {
                    [lhs, rhs]
                };
                let lhs = self.apply_sat(lhs, *sat);
                let rhs = self.apply_sat(rhs, *sat);
                let packed = self.lower_half_pack(lhs, rhs, old, *merge);
                Some(packed)
            }
            IrOp::HMul {
                a,
                b,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                abs_a,
                neg_a,
                abs_b,
                neg_b,
                sat,
                precision,
                ..
            } => {
                let [a0, a1] = self.lower_half_pair(a, *swizzle_a);
                let [b0, b1] = self.lower_half_pair(b, *swizzle_b);
                let a0 = self.lower_half_abs_neg(a0, *abs_a, *neg_a);
                let a1 = self.lower_half_abs_neg(a1, *abs_a, *neg_a);
                let b0 = self.lower_half_abs_neg(b0, *abs_b, *neg_b);
                let b1 = self.lower_half_abs_neg(b1, *abs_b, *neg_b);
                let mut lhs = self.b.f_mul(self.f32_t, None, a0, b0).unwrap();
                let mut rhs = self.b.f_mul(self.f32_t, None, a1, b1).unwrap();
                if matches!(precision, HalfPrecision::FMZ) && !*sat {
                    let z = self.f32_zero;
                    let az0 = self.b.f_ord_equal(self.bool_t, None, a0, z).unwrap();
                    let bz0 = self.b.f_ord_equal(self.bool_t, None, b0, z).unwrap();
                    let az1 = self.b.f_ord_equal(self.bool_t, None, a1, z).unwrap();
                    let bz1 = self.b.f_ord_equal(self.bool_t, None, b1, z).unwrap();
                    let z0 = self.b.logical_or(self.bool_t, None, az0, bz0).unwrap();
                    let z1 = self.b.logical_or(self.bool_t, None, az1, bz1).unwrap();
                    lhs = self.b.select(self.f32_t, None, z0, z, lhs).unwrap();
                    rhs = self.b.select(self.f32_t, None, z1, z, rhs).unwrap();
                }
                let [lhs, rhs] = if Self::half_pair_promotes(*swizzle_a, *swizzle_b, None) {
                    self.round_half_pair(lhs, rhs)
                } else {
                    [lhs, rhs]
                };
                let lhs = self.apply_sat(lhs, *sat);
                let rhs = self.apply_sat(rhs, *sat);
                let packed = self.lower_half_pack(lhs, rhs, old, *merge);
                Some(packed)
            }
            IrOp::HFma {
                a,
                b,
                c,
                old,
                merge,
                swizzle_a,
                swizzle_b,
                swizzle_c,
                neg_b,
                neg_c,
                sat,
                precision,
                ..
            } => {
                let [a0, a1] = self.lower_half_pair(a, *swizzle_a);
                let [b0, b1] = self.lower_half_pair(b, *swizzle_b);
                let [c0, c1] = self.lower_half_pair(c, *swizzle_c);
                let b0 = self.lower_half_abs_neg(b0, false, *neg_b);
                let b1 = self.lower_half_abs_neg(b1, false, *neg_b);
                let c0 = self.lower_half_abs_neg(c0, false, *neg_c);
                let c1 = self.lower_half_abs_neg(c1, false, *neg_c);
                let mut lhs = self
                    .b
                    .ext_inst(
                        self.f32_t,
                        None,
                        self.glsl,
                        50,
                        [Operand::IdRef(a0), Operand::IdRef(b0), Operand::IdRef(c0)],
                    )
                    .unwrap();
                let mut rhs = self
                    .b
                    .ext_inst(
                        self.f32_t,
                        None,
                        self.glsl,
                        50,
                        [Operand::IdRef(a1), Operand::IdRef(b1), Operand::IdRef(c1)],
                    )
                    .unwrap();
                if matches!(precision, HalfPrecision::FMZ) && !*sat {
                    let z = self.f32_zero;
                    let az0 = self.b.f_ord_equal(self.bool_t, None, a0, z).unwrap();
                    let bz0 = self.b.f_ord_equal(self.bool_t, None, b0, z).unwrap();
                    let az1 = self.b.f_ord_equal(self.bool_t, None, a1, z).unwrap();
                    let bz1 = self.b.f_ord_equal(self.bool_t, None, b1, z).unwrap();
                    let z0 = self.b.logical_or(self.bool_t, None, az0, bz0).unwrap();
                    let z1 = self.b.logical_or(self.bool_t, None, az1, bz1).unwrap();
                    lhs = self.b.select(self.f32_t, None, z0, c0, lhs).unwrap();
                    rhs = self.b.select(self.f32_t, None, z1, c1, rhs).unwrap();
                }
                let [lhs, rhs] =
                    if Self::half_pair_promotes(*swizzle_a, *swizzle_b, Some(*swizzle_c)) {
                        self.round_half_pair(lhs, rhs)
                    } else {
                        [lhs, rhs]
                    };
                let lhs = self.apply_sat(lhs, *sat);
                let rhs = self.apply_sat(rhs, *sat);
                let packed = self.lower_half_pack(lhs, rhs, old, *merge);
                Some(packed)
            }
            IrOp::PackHalf2 { lo, hi } => {
                let l = self.lower_value(lo);
                let h = self.lower_value(hi);
                let vector = self
                    .b
                    .composite_construct(self.vec2_t, None, [l, h])
                    .unwrap();
                let packed = self
                    .b
                    .ext_inst(self.u32_t, None, self.glsl, 58, [Operand::IdRef(vector)])
                    .unwrap();
                Some(self.store_bits(packed))
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
                    Stage::Vertex => u32::from(*binding),
                    Stage::Fragment => 16 + u32::from(*binding),
                    Stage::Compute => u32::from(*binding),
                };
                self.cbuf_bindings_used |= 1u32 << logical_binding;
                if self.stage == Stage::Compute {
                    let ubo_var = self.compute_cbuf_var(*binding);
                    let vec4_index = byte_offset / 16;
                    let component = (byte_offset / 4) & 0x3;
                    let v_idx = self.const_u32(vec4_index);
                    let c_idx = self.const_u32(component);
                    let zero_u32 = self.const_u32(0);
                    let ac = self
                        .b
                        .access_chain(
                            self.ptr_uniform_f32,
                            None,
                            ubo_var,
                            [zero_u32, v_idx, c_idx],
                        )
                        .unwrap();
                    Some(self.b.load(self.f32_t, None, ac, None, []).unwrap())
                } else {
                    let effective = self.const_u32(*byte_offset);
                    let word = self.graphics_cbuf_load_word(logical_binding, effective);
                    Some(self.b.bitcast(self.f32_t, None, word).unwrap())
                }
            }
            IrOp::LoadCbufIndexed {
                binding,
                byte_offset,
                index,
                address_mode,
            } => {
                assert_eq!(
                    *address_mode,
                    CbufAddressMode::Default,
                    "segmented cbuf addressing must be rejected before SPIR-V lowering",
                );
                let logical_binding = match self.stage {
                    Stage::Vertex => u32::from(*binding),
                    Stage::Fragment => 16 + u32::from(*binding),
                    Stage::Compute => u32::from(*binding),
                };
                self.cbuf_bindings_used |= 1u32 << logical_binding;
                let idx_f32 = self.lower_value(index);
                let u32_t = self.u32_t;
                let idx_u32 = self.b.bitcast(u32_t, None, idx_f32).unwrap();
                let imm = self.const_u32(*byte_offset);
                let eff = self.b.i_add(u32_t, None, imm, idx_u32).unwrap();
                if self.stage == Stage::Compute {
                    let ubo_var = self.compute_cbuf_var(*binding);
                    let sh4 = self.const_u32(4);
                    let v_idx = self
                        .b
                        .shift_right_logical(u32_t, None, eff, sh4)
                        .unwrap();
                    let sh2 = self.const_u32(2);
                    let three = self.const_u32(3);
                    let comp_sh = self
                        .b
                        .shift_right_logical(u32_t, None, eff, sh2)
                        .unwrap();
                    let c_idx = self
                        .b
                        .bitwise_and(u32_t, None, comp_sh, three)
                        .unwrap();
                    let zero_u32 = self.const_u32(0);
                    let ptr_vec4 = self.b.type_pointer(None, StorageClass::Uniform, self.vec4_t);
                    let ac = self
                        .b
                        .access_chain(ptr_vec4, None, ubo_var, [zero_u32, v_idx])
                        .unwrap();
                    let vec = self.b.load(self.vec4_t, None, ac, None, []).unwrap();
                    Some(
                        self.b
                            .vector_extract_dynamic(self.f32_t, None, vec, c_idx)
                            .unwrap(),
                    )
                } else {
                    let word = self.graphics_cbuf_load_word(logical_binding, eff);
                    Some(self.b.bitcast(self.f32_t, None, word).unwrap())
                }
            }
            IrOp::LoadGlobal { .. } => Some(self.f32_zero),
            IrOp::LoadLocal { addr } => {
                let (pointer, in_bounds) = self.local_word_pointer(addr);
                let loaded = self.b.load(self.u32_t, None, pointer, None, []).unwrap();
                let zero = self.const_u32(0);
                let value = self
                    .b
                    .select(self.u32_t, None, in_bounds, loaded, zero)
                    .unwrap();
                Some(self.store_bits(value))
            }
            IrOp::StoreLocal { addr, value } => {
                let predicate_guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                let val_v = self.lower_value(value);
                let val_u = self.as_u32(val_v);
                let (pointer, in_bounds) = self.local_word_pointer(addr);
                let guard = if let Some(predicate_guard) = predicate_guard {
                    self.b
                        .logical_and(self.bool_t, None, in_bounds, predicate_guard)
                        .unwrap()
                } else {
                    in_bounds
                };
                let old = self.b.load(self.u32_t, None, pointer, None, []).unwrap();
                let stored = self.b.select(self.u32_t, None, guard, val_u, old).unwrap();
                self.b.store(pointer, stored, None, []).unwrap();
                None
            }
            IrOp::LoadShared { addr } => {
                let pointer = self.shared_word_pointer(addr);
                let value = self.b.load(self.u32_t, None, pointer, None, []).unwrap();
                Some(self.store_bits(value))
            }
            IrOp::StoreShared { addr, value } => {
                self.lower_shared_store(inst, addr, value);
                None
            }
            IrOp::SharedAtomic { addr, value, op } => {
                Some(self.lower_shared_atomic(inst, addr, value, *op))
            }
            IrOp::WorkgroupBarrier => {
                self.lower_workgroup_barrier();
                None
            }
            IrOp::MemoryBarrier { scope } => {
                self.lower_memory_barrier(*scope);
                None
            }
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
                            Stage::Vertex => u32::from(*cbuf_binding),
                            Stage::Fragment => 16 + u32::from(*cbuf_binding),
                            Stage::Compute => (*cbuf_binding as u32) & 0xF,
                        };
                        self.cbuf_bindings_used |= 1u32 << logical_binding;
                        let base_u = if self.stage == Stage::Compute {
                            let slot_base = logical_binding * CBUF_SLOT_VEC4S;
                            let local_vec4 =
                                (cbuf_offset / 16).min(CBUF_SLOT_VEC4S.saturating_sub(1));
                            let vec4_index = (slot_base + local_vec4) % self.ubo_vec4s;
                            let component = (cbuf_offset / 4) & 0x3;
                            let v_idx = self.const_u32(vec4_index);
                            let c_idx = self.const_u32(component);
                            let zero_u32 = self.const_u32(0);
                            let base_ac = self
                                .b
                                .access_chain(
                                    self.ptr_uniform_f32,
                                    None,
                                    self.ubo_var,
                                    [zero_u32, v_idx, c_idx],
                                )
                                .unwrap();
                            let base_f = self
                                .b
                                .load(self.f32_t, None, base_ac, None, [])
                                .unwrap();
                            self.b.bitcast(u32_t, None, base_f).unwrap()
                        } else {
                            let effective = self.const_u32(*cbuf_offset);
                            self.graphics_cbuf_load_word(logical_binding, effective)
                        };
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
            IrOp::TexelFetch {
                cbuf_binding,
                cbuf_word_offset,
                cbuf_secondary_word_offset,
                x,
                y,
                z,
                component,
            } => {
                let tex_id = nexium_shader::bindless_texture_id_pair(
                    *cbuf_binding,
                    *cbuf_word_offset,
                    *cbuf_secondary_word_offset,
                );
                self.texs_ids_used.insert(tex_id);
                let x_value = self.lower_value(x);
                let x_coord = self.as_i32(x_value);
                let numeric_type = self.texture_numeric_type_at(tex_id);
                let vec4_t = self.graphics_numeric_vec4_type(numeric_type);
                if self
                    .ir_constant_facts
                    .texel_fetch_buffer_coordinates_compatible(y.as_ref(), z.as_ref())
                    && self.texel_buffer_slot_enabled(tex_id)
                {
                    let (decl, image_var) =
                        self.typed_fetch_image_at(tex_id, numeric_type, GraphicsImageKind::Buffer);
                    let image = self
                        .b
                        .load(decl.image_t, None, image_var, None, [])
                        .unwrap();
                    let fetched = self
                        .b
                        .image_fetch(vec4_t, None, image, x_coord, None, [])
                        .unwrap();
                    Some(self.graphics_fetch_component_as_f32(fetched, numeric_type, *component))
                } else {
                    let (decl, image_var, coords) = if let Some(z) = z {
                        let y = y.as_ref().expect("3D texel fetch requires Y coordinate");
                        let y_value = self.lower_value(y);
                        let y_coord = self.as_i32(y_value);
                        let z_value = self.lower_value(z);
                        let z_coord = self.as_i32(z_value);
                        let coords = self
                            .b
                            .composite_construct(self.ivec3_t, None, [x_coord, y_coord, z_coord])
                            .unwrap();
                        let (decl, image_var) = self.typed_fetch_image_at(
                            tex_id,
                            numeric_type,
                            GraphicsImageKind::D3,
                        );
                        (decl, image_var, coords)
                    } else {
                        let y_coord = if let Some(y) = y {
                            let y_value = self.lower_value(y);
                            self.as_i32(y_value)
                        } else {
                            self.i32_zero
                        };
                        let kind = if self.sampler_arrayed {
                            GraphicsImageKind::D2Array
                        } else {
                            GraphicsImageKind::D2
                        };
                        let (decl, image_var) =
                            self.typed_fetch_image_at(tex_id, numeric_type, kind);
                        let coords = if self.sampler_arrayed {
                            self.b
                                .composite_construct(
                                    self.ivec3_t,
                                    None,
                                    [x_coord, y_coord, self.i32_zero],
                                )
                                .unwrap()
                        } else {
                            self.b
                                .composite_construct(self.ivec2_t, None, [x_coord, y_coord])
                                .unwrap()
                        };
                        (decl, image_var, coords)
                    };
                    let image = self
                        .b
                        .load(decl.image_t, None, image_var, None, [])
                        .unwrap();
                    let fetched = self
                        .b
                        .image_fetch(
                            vec4_t,
                            None,
                            image,
                            coords,
                            Some(rspirv::spirv::ImageOperands::LOD),
                            [Operand::IdRef(self.i32_zero)],
                        )
                        .unwrap();
                    Some(self.graphics_fetch_component_as_f32(fetched, numeric_type, *component))
                }
            }
            IrOp::TexelFetchHandle {
                handle,
                dimension: _,
                x,
                y,
                z,
                component,
            } => {
                Some(self.lower_compute_texel_fetch(*handle, x, y.as_ref(), z.as_ref(), *component))
            }
            IrOp::SampleTexHandle {
                handle,
                dimension,
                u,
                v,
                w,
                implicit_lod,
                explicit_lod,
                texel_offset,
                component,
                ..
            } => Some(self.lower_compute_texture_sample(
                *handle,
                *dimension,
                u,
                v.as_ref(),
                w.as_ref(),
                *implicit_lod,
                explicit_lod.as_ref(),
                texel_offset.as_ref(),
                *component,
            )),
            IrOp::TextureQueryDimension {
                handle,
                lod,
                component,
            } => Some(self.lower_compute_texture_query(*handle, lod, *component)),
            IrOp::LocalInvocationId { component } => {
                Some(self.compute_builtin_component(true, *component))
            }
            IrOp::WorkgroupId { component } => {
                Some(self.compute_builtin_component(false, *component))
            }
            IrOp::SubgroupLaneId => {
                let lane = self.subgroup_lane_id();
                Some(self.store_bits(lane))
            }
            IrOp::SubgroupMask { kind } => {
                let mask = self.subgroup_mask(*kind);
                Some(self.store_bits(mask))
            }
            IrOp::ImageWrite {
                handle,
                dimension,
                x,
                y,
                z,
                values,
            } => {
                self.lower_compute_image_write(
                    inst,
                    *handle,
                    *dimension,
                    x,
                    y.as_ref(),
                    z.as_ref(),
                    values,
                );
                None
            }
            IrOp::ImageAtomic {
                handle,
                dimension: _,
                x,
                value,
                op,
                data_type,
                ..
            } => Some(self.lower_compute_image_atomic(inst, *handle, x, value, *op, *data_type)),
            IrOp::SampleTex {
                tex_id,
                u,
                v,
                array,
                volume,
                cube,
                dref,
                implicit_lod,
                lod_bias,
                explicit_lod,
                texel_offset,
                component,
            } => 'sample_tex: {
                self.texs_ids_used.insert(*tex_id);
                let lod_bias = lod_bias.as_ref().map(|bias| self.lower_value(bias));
                let explicit_lod = explicit_lod.as_ref().map(|lod| self.lower_value(lod));
                let dref = dref.as_ref().map(|reference| self.lower_value(reference));
                let texel_offset = texel_offset.as_ref().map(|(x, y)| {
                    let immediate_bits = |value: &IrValue| match value {
                        IrValue::Zero => Some(0),
                        IrValue::ImmU32(bits) => Some(*bits),
                        IrValue::ImmF32(value) => Some(value.to_bits()),
                        IrValue::GprIn(_) | IrValue::Inst(_) => None,
                    };
                    let x = immediate_bits(x).expect("sample offset must be constant");
                    let y = immediate_bits(y).expect("sample offset must be constant");
                    let x = self.b.constant_bit32(self.i32_t, x);
                    let y = self.b.constant_bit32(self.i32_t, y);
                    self.b.constant_composite(self.ivec2_t, [x, y])
                });
                if let Some(w) = cube {
                    let tex_slot = self
                        .texture_slots
                        .get(tex_id)
                        .copied()
                        .unwrap_or(0)
                        .min(MAX_TEXTURE_DESCRIPTORS - 1);
                    let uv0 = self.lower_value(u);
                    let uv1 = self.lower_value(v);
                    let uv2 = self.lower_value(w);
                    let (coords, debug_coords, img_var, samp_var, image_t, sampled_image_t) =
                        if let Some(array) = array {
                            let layer = if let Some(override_layer) =
                                std::env::var("NEXIUM_TEX_LAYER_OVERRIDE")
                                    .ok()
                                    .and_then(|value| value.parse::<f32>().ok())
                            {
                                self.const_f32(override_layer.to_bits())
                            } else {
                                let layer_value = self.lower_value(array);
                                let raw = self.as_u32(layer_value);
                                let mask = self.const_u32(0xFFFF);
                                let layer_u =
                                    self.b.bitwise_and(self.u32_t, None, raw, mask).unwrap();
                                self.b.convert_u_to_f(self.f32_t, None, layer_u).unwrap()
                            };
                            let coords = self
                                .b
                                .composite_construct(self.vec4_t, None, [uv0, uv1, uv2, layer])
                                .unwrap();
                            let (img_var, samp_var) = self.sampler_cube_arrayed_at(*tex_id);
                            (
                                coords,
                                coords,
                                img_var,
                                samp_var,
                                self.image_cube_arrayed_t,
                                self.sampled_image_cube_arrayed_t,
                            )
                        } else {
                            let coords = self
                                .b
                                .composite_construct(self.vec3_t, None, [uv0, uv1, uv2])
                                .unwrap();
                            let debug_coords = self
                                .b
                                .composite_construct(
                                    self.vec4_t,
                                    None,
                                    [uv0, uv1, uv2, self.f32_one],
                                )
                                .unwrap();
                            let (img_var, samp_var) = self.sampler_cube_at(*tex_id);
                            (
                                coords,
                                debug_coords,
                                img_var,
                                samp_var,
                                self.image_cube_t,
                                self.sampled_image_cube_t,
                            )
                        };
                    if matches!(self.stage, Stage::Fragment)
                        && self.texcoord_debug_slot == Some(tex_slot)
                        && self.sample_debug_value.is_none()
                    {
                        self.sample_debug_value = Some(debug_coords);
                    }
                    let img = self.b.load(image_t, None, img_var, None, []).unwrap();
                    let samp = self
                        .b
                        .load(self.sampler_t, None, samp_var, None, [])
                        .unwrap();
                    let sampled_img = self
                        .b
                        .sampled_image(sampled_image_t, None, img, samp)
                        .unwrap();
                    let sampled = if let Some(dref) = dref {
                        self.sample_image_dref(
                            sampled_img,
                            coords,
                            dref,
                            *implicit_lod,
                            lod_bias,
                            explicit_lod,
                            texel_offset,
                        )
                    } else {
                        self.sample_image(
                            sampled_img,
                            coords,
                            *implicit_lod,
                            lod_bias,
                            explicit_lod,
                            texel_offset,
                        )
                    };
                    if matches!(self.stage, Stage::Fragment)
                        && self.sample_debug_slot == Some(tex_slot)
                        && self.sample_debug_value.is_none()
                    {
                        let debug_value = if dref.is_some() {
                            self.b
                                .composite_construct(
                                    self.vec4_t,
                                    None,
                                    [sampled, sampled, sampled, self.f32_one],
                                )
                                .unwrap()
                        } else if let Some(component) = self.sample_debug_component {
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
                    let c = if dref.is_some() {
                        sampled
                    } else {
                        self.b
                            .composite_extract(self.f32_t, None, sampled, [*component as u32])
                            .unwrap_or(self.f32_zero)
                    };
                    break 'sample_tex Some(c);
                }
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
                    let sampled = if let Some(dref) = dref {
                        self.sample_image_dref(
                            sampled_img,
                            coords,
                            dref,
                            *implicit_lod,
                            lod_bias,
                            explicit_lod,
                            texel_offset,
                        )
                    } else {
                        self.sample_image(
                            sampled_img,
                            coords,
                            *implicit_lod,
                            lod_bias,
                            explicit_lod,
                            texel_offset,
                        )
                    };
                    if matches!(self.stage, Stage::Fragment)
                        && self.sample_debug_slot == Some(tex_slot)
                        && self.sample_debug_value.is_none()
                    {
                        let debug_value = if dref.is_some() {
                            self.b
                                .composite_construct(
                                    self.vec4_t,
                                    None,
                                    [sampled, sampled, sampled, self.f32_one],
                                )
                                .unwrap()
                        } else if let Some(component) = self.sample_debug_component {
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
                    let c = if dref.is_some() {
                        sampled
                    } else {
                        self.b
                            .composite_extract(self.f32_t, None, sampled, [*component as u32])
                            .unwrap_or(self.f32_zero)
                    };
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
                let sampled = if let Some(dref) = dref {
                    self.sample_image_dref(
                        sampled_img,
                        coords,
                        dref,
                        *implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                    )
                } else {
                    self.sample_image(
                        sampled_img,
                        coords,
                        *implicit_lod,
                        lod_bias,
                        explicit_lod,
                        texel_offset,
                    )
                };
                if matches!(self.stage, Stage::Fragment)
                    && self.sample_debug_slot == Some(tex_slot)
                    && self.sample_debug_value.is_none()
                {
                    let debug_value = if dref.is_some() {
                        self.b
                            .composite_construct(
                                self.vec4_t,
                                None,
                                [sampled, sampled, sampled, self.f32_one],
                            )
                            .unwrap()
                    } else if let Some(component) = self.sample_debug_component {
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
                let c = if dref.is_some() {
                    sampled
                } else {
                    self.b
                        .composite_extract(self.f32_t, None, sampled, [*component as u32])
                        .unwrap_or(self.f32_zero)
                };
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
            IrOp::HSetPred {
                cmp,
                bop,
                src_a,
                src_b,
                swizzle_a,
                swizzle_b,
                neg_a,
                abs_a,
                neg_b,
                abs_b,
                src_pred,
                src_pred_inv,
                dest_p,
                dest_np,
                h_and,
                ..
            } => {
                let [a0, a1] = self.lower_half_pair(src_a, *swizzle_a);
                let [b0, b1] = self.lower_half_pair(src_b, *swizzle_b);
                let a0 = self.lower_half_abs_neg(a0, *abs_a, *neg_a);
                let a1 = self.lower_half_abs_neg(a1, *abs_a, *neg_a);
                let b0 = self.lower_half_abs_neg(b0, *abs_b, *neg_b);
                let b1 = self.lower_half_abs_neg(b1, *abs_b, *neg_b);
                let src_pred_word = self.resolve_pred(*src_pred, *src_pred_inv);
                let cmp0 = self.lower_fcompare(cmp, a0, b0);
                let cmp1 = self.lower_fcompare(cmp, a1, b1);
                let result0 = self.lower_boolop(bop, cmp0, src_pred_word);
                let result1 = self.lower_boolop(bop, cmp1, src_pred_word);
                let (result_p, result_np) = if *h_and {
                    let both = self
                        .b
                        .logical_and(self.bool_t, None, result0, result1)
                        .unwrap();
                    (both, self.b.logical_not(self.bool_t, None, both).unwrap())
                } else {
                    (result0, result1)
                };
                let guard = inst
                    .pred
                    .map(|pred| self.resolve_pred(pred.idx, pred.negate));
                self.write_pred_reg(*dest_p, result_p, guard);
                self.write_pred_reg(*dest_np, result_np, guard);
                Some(result_p)
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
            IrOp::ILop3 { a, b, c, lut } => {
                let av = self.lower_value(a);
                let bv = self.lower_value(b);
                let cv = self.lower_value(c);
                let au = self.as_u32(av);
                let bu = self.as_u32(bv);
                let cu = self.as_u32(cv);

                let r = match *lut {
                    0x00 => self.const_u32(0),
                    0xff => self.const_u32(u32::MAX),
                    0xf0 => au,
                    0x0f => self.b.not(self.u32_t, None, au).unwrap(),
                    0xcc => bu,
                    0x33 => self.b.not(self.u32_t, None, bu).unwrap(),
                    0xaa => cu,
                    0x55 => self.b.not(self.u32_t, None, cu).unwrap(),
                    0xf8 => {
                        let bc = self.b.bitwise_and(self.u32_t, None, bu, cu).unwrap();
                        self.b.bitwise_or(self.u32_t, None, au, bc).unwrap()
                    }
                    0xf4 => {
                        let not_c = self.b.not(self.u32_t, None, cu).unwrap();
                        let b_not_c = self.b.bitwise_and(self.u32_t, None, bu, not_c).unwrap();
                        self.b.bitwise_or(self.u32_t, None, au, b_not_c).unwrap()
                    }
                    _ => {
                        let coefficients = lop3_anf_coefficients(*lut);
                        let mut bc = None;
                        let mut ac = None;
                        let mut ab = None;
                        let mut result = (coefficients & 1 != 0).then(|| self.const_u32(u32::MAX));

                        for mask in 1..8u8 {
                            if coefficients & (1u8 << mask) == 0 {
                                continue;
                            }
                            let term = match mask {
                                1 => cu,
                                2 => bu,
                                3 => {
                                    let value =
                                        self.b.bitwise_and(self.u32_t, None, bu, cu).unwrap();
                                    bc = Some(value);
                                    value
                                }
                                4 => au,
                                5 => {
                                    let value =
                                        self.b.bitwise_and(self.u32_t, None, au, cu).unwrap();
                                    ac = Some(value);
                                    value
                                }
                                6 => {
                                    let value =
                                        self.b.bitwise_and(self.u32_t, None, au, bu).unwrap();
                                    ab = Some(value);
                                    value
                                }
                                7 => {
                                    let pair = if let Some(value) = bc.or(ac).or(ab) {
                                        value
                                    } else {
                                        let value =
                                            self.b.bitwise_and(self.u32_t, None, bu, cu).unwrap();
                                        bc = Some(value);
                                        value
                                    };
                                    let remaining = if bc == Some(pair) {
                                        au
                                    } else if ac == Some(pair) {
                                        bu
                                    } else {
                                        cu
                                    };
                                    self.b
                                        .bitwise_and(self.u32_t, None, pair, remaining)
                                        .unwrap()
                                }
                                _ => unreachable!(),
                            };
                            result = Some(if let Some(accumulator) = result {
                                self.b
                                    .bitwise_xor(self.u32_t, None, accumulator, term)
                                    .unwrap()
                            } else {
                                term
                            });
                        }
                        result.unwrap_or_else(|| self.const_u32(0))
                    }
                };
                Some(self.store_bits(r))
            }
            IrOp::FindUMsb { value } => {
                let value = self.lower_value(value);
                let value = self.as_u32(value);
                let r = self
                    .b
                    .ext_inst(
                        self.u32_t,
                        None,
                        self.glsl,
                        GLOp::FindUMsb as u32,
                        [Operand::IdRef(value)],
                    )
                    .unwrap();
                Some(self.store_bits(r))
            }
            IrOp::BitCount { value } => {
                let value = self.lower_value(value);
                let value = self.as_u32(value);
                let r = self.b.bit_count(self.u32_t, None, value).unwrap();
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
            IrOp::Bfi {
                base,
                insert,
                control,
            } => {
                let basev = self.lower_value(base);
                let insertv = self.lower_value(insert);
                let controlv = self.lower_value(control);
                let baseu = self.as_u32(basev);
                let insertu = self.as_u32(insertv);
                let controlu = self.as_u32(controlv);
                let mask = self.const_u32(0xff);
                let pos = self
                    .b
                    .bitwise_and(self.u32_t, None, controlu, mask)
                    .unwrap();
                let eight = self.const_u32(8);
                let size_raw = self
                    .b
                    .shift_right_logical(self.u32_t, None, controlu, eight)
                    .unwrap();
                let cnt = self
                    .b
                    .bitwise_and(self.u32_t, None, size_raw, mask)
                    .unwrap();
                let r = self
                    .b
                    .bit_field_insert(self.u32_t, None, baseu, insertu, pos, cnt)
                    .unwrap();
                Some(self.store_bits(r))
            }
            IrOp::Shfl {
                value,
                index,
                mask,
                mode,
                pred_dest,
            } => {
                let valv = self.lower_value(value);
                let val = self.as_u32(valv);
                let idxv = self.lower_value(index);
                let idx = self.as_u32(idxv);
                let maskv = self.lower_value(mask);
                let mask_u = self.as_u32(maskv);
                let (src_tid, in_range) = self.shfl_target(*mode, idx, mask_u);
                let scope = self.const_u32(3);
                let shuffled = self
                    .b
                    .group_non_uniform_shuffle(self.u32_t, None, scope, val, src_tid)
                    .unwrap();
                let sel = self
                    .b
                    .select(self.u32_t, None, in_range, shuffled, val)
                    .unwrap();
                if *pred_dest < 7 {
                    let guard = inst.pred.map(|p| self.resolve_pred(p.idx, p.negate));
                    self.write_pred_reg(*pred_dest, in_range, guard);
                }
                Some(self.store_bits(sel))
            }
            IrOp::SubgroupVote {
                source_pred,
                mode,
                pred_dest,
                old,
            } => {
                self.enable_subgroup_vote();
                let source = self.resolve_pred(source_pred.idx, source_pred.negate);
                let guard = inst
                    .pred
                    .map(|predicate| self.resolve_pred(predicate.idx, predicate.negate));
                let ballot_predicate = match guard {
                    Some(guard) => self
                        .b
                        .logical_and(self.bool_t, None, guard, source)
                        .unwrap(),
                    None => source,
                };
                let vote_predicate = match (mode, guard) {
                    (VoteMode::All, Some(guard)) => {
                        let disabled = self.b.logical_not(self.bool_t, None, guard).unwrap();
                        self.b
                            .logical_or(self.bool_t, None, disabled, source)
                            .unwrap()
                    }
                    (VoteMode::All, None) => source,
                    (VoteMode::Any, _) => ballot_predicate,
                    (VoteMode::Equal, _) => ballot_predicate,
                };
                let scope = self.const_u32(3);
                let ballot = self
                    .b
                    .group_non_uniform_ballot(self.uvec4_t, None, scope, ballot_predicate)
                    .unwrap();
                let ballot_x = self
                    .b
                    .composite_extract(self.u32_t, None, ballot, [0])
                    .unwrap();
                let scalar = match mode {
                    VoteMode::All => self
                        .b
                        .group_non_uniform_all(self.bool_t, None, scope, vote_predicate)
                        .unwrap(),
                    VoteMode::Any => self
                        .b
                        .group_non_uniform_any(self.bool_t, None, scope, vote_predicate)
                        .unwrap(),
                    VoteMode::Equal => {
                        let participant_predicate = guard.unwrap_or(self.bool_true);
                        let participants = self
                            .b
                            .group_non_uniform_ballot(
                                self.uvec4_t,
                                None,
                                scope,
                                participant_predicate,
                            )
                            .unwrap();
                        let participants_x = self
                            .b
                            .composite_extract(self.u32_t, None, participants, [0])
                            .unwrap();
                        let zero = self.const_u32(0);
                        let all_false = self.b.i_equal(self.bool_t, None, ballot_x, zero).unwrap();
                        let all_true = self
                            .b
                            .i_equal(self.bool_t, None, ballot_x, participants_x)
                            .unwrap();
                        self.b
                            .logical_or(self.bool_t, None, all_false, all_true)
                            .unwrap()
                    }
                };
                let result = match guard {
                    Some(guard) => {
                        let old = self.lower_value(old);
                        let old = self.as_u32(old);
                        self.b
                            .select(self.u32_t, None, guard, ballot_x, old)
                            .unwrap()
                    }
                    None => ballot_x,
                };
                self.write_pred_reg(*pred_dest, scalar, guard);
                Some(self.store_bits(result))
            }
            IrOp::FSwzAdd { a, b, swizzle } => {
                let af = self.lower_value(a);
                let bf = self.lower_value(b);
                let lane = self.subgroup_lane_id();
                let three = self.const_u32(3);
                let one = self.const_u32(1);
                let laneq = self.b.bitwise_and(self.u32_t, None, lane, three).unwrap();
                let sh = self
                    .b
                    .shift_left_logical(self.u32_t, None, laneq, one)
                    .unwrap();
                let sw = self.const_u32(*swizzle);
                let shifted = self
                    .b
                    .shift_right_logical(self.u32_t, None, sw, sh)
                    .unwrap();
                let sel = self
                    .b
                    .bitwise_and(self.u32_t, None, shifted, three)
                    .unwrap();
                let (lut_a, lut_b) = self.fswzadd_luts();
                let mod_a = self
                    .b
                    .vector_extract_dynamic(self.f32_t, None, lut_a, sel)
                    .unwrap();
                let mod_b = self
                    .b
                    .vector_extract_dynamic(self.f32_t, None, lut_b, sel)
                    .unwrap();
                let ra = self.b.f_mul(self.f32_t, None, af, mod_a).unwrap();
                let rb = self.b.f_mul(self.f32_t, None, bf, mod_b).unwrap();
                Some(self.b.f_add(self.f32_t, None, ra, rb).unwrap())
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
            match &inst.op {
                IrOp::HSetPred {
                    dest_p, dest_np, ..
                }
                | IrOp::FSetPred {
                    dest_p, dest_np, ..
                }
                | IrOp::ISetPred {
                    dest_p, dest_np, ..
                }
                | IrOp::PSetPred {
                    dest_p, dest_np, ..
                }
                | IrOp::CSetPred {
                    dest_p, dest_np, ..
                } => {
                    for pred in [*dest_p, *dest_np] {
                        if let Some(value) = self.pred_regs.get(pred as usize).copied().flatten() {
                            self.pred_value_to_word.insert((id, pred), value);
                        }
                    }
                }
                IrOp::SubgroupVote { pred_dest, .. } => {
                    if let Some(value) = self.pred_regs.get(*pred_dest as usize).copied().flatten()
                    {
                        self.pred_value_to_word.insert((id, *pred_dest), value);
                    }
                }
                _ => {}
            }
        }
    }

    fn sample_image_dref(
        &mut self,
        sampled_image: Word,
        coords: Word,
        dref: Word,
        implicit_lod: bool,
        lod_bias: Option<Word>,
        explicit_lod: Option<Word>,
        texel_offset: Option<Word>,
    ) -> Word {
        use rspirv::spirv::ImageOperands;

        if implicit_lod && matches!(self.stage, Stage::Fragment) {
            return match (lod_bias, texel_offset) {
                (Some(bias), Some(offset)) => self
                    .b
                    .image_sample_dref_implicit_lod(
                        self.f32_t,
                        None,
                        sampled_image,
                        coords,
                        dref,
                        Some(ImageOperands::BIAS | ImageOperands::CONST_OFFSET),
                        [Operand::IdRef(bias), Operand::IdRef(offset)],
                    )
                    .unwrap(),
                (Some(bias), None) => self
                    .b
                    .image_sample_dref_implicit_lod(
                        self.f32_t,
                        None,
                        sampled_image,
                        coords,
                        dref,
                        Some(ImageOperands::BIAS),
                        [Operand::IdRef(bias)],
                    )
                    .unwrap(),
                (None, Some(offset)) => self
                    .b
                    .image_sample_dref_implicit_lod(
                        self.f32_t,
                        None,
                        sampled_image,
                        coords,
                        dref,
                        Some(ImageOperands::CONST_OFFSET),
                        [Operand::IdRef(offset)],
                    )
                    .unwrap(),
                (None, None) => self
                    .b
                    .image_sample_dref_implicit_lod(
                        self.f32_t,
                        None,
                        sampled_image,
                        coords,
                        dref,
                        None,
                        [],
                    )
                    .unwrap(),
            };
        }

        let lod = explicit_lod.unwrap_or(self.f32_zero);
        if let Some(offset) = texel_offset {
            self.b
                .image_sample_dref_explicit_lod(
                    self.f32_t,
                    None,
                    sampled_image,
                    coords,
                    dref,
                    ImageOperands::LOD | ImageOperands::CONST_OFFSET,
                    [Operand::IdRef(lod), Operand::IdRef(offset)],
                )
                .unwrap()
        } else {
            self.b
                .image_sample_dref_explicit_lod(
                    self.f32_t,
                    None,
                    sampled_image,
                    coords,
                    dref,
                    ImageOperands::LOD,
                    [Operand::IdRef(lod)],
                )
                .unwrap()
        }
    }

    fn compute_shared_merges(&mut self, cfg: &Cfg) {
        let Some(ipd) = self.cond_merge.clone() else {
            return;
        };
        let dom = dominators(cfg);
        let reachable = cfg_reachable(cfg);
        let loop_latches: std::collections::HashSet<BlockId> =
            self.self_loops.values().map(|info| info.latch).collect();
        let mut by_merge: std::collections::BTreeMap<u32, Vec<u32>> =
            std::collections::BTreeMap::new();
        for block in &cfg.blocks {
            if !reachable[block.id as usize] {
                continue;
            }
            if self.loop_break_merge(block).is_some() || loop_latches.contains(&block.id) {
                continue;
            }
            let owns_selection = match block.branch {
                BranchKind::Conditional { target, .. } => {
                    let next = block.id + 1;
                    self.block_labels.contains_key(&next) && target != next && target > block.id
                }
                BranchKind::Indirect { .. } => true,
                _ => false,
            };
            if owns_selection {
                let m = ipd[block.id as usize];
                by_merge.entry(m).or_default().push(block.id);
            }
        }
        for (m, mut headers) in by_merge {
            let mut loop_headers: Vec<BlockId> = self
                .self_loops
                .iter()
                .filter_map(|(&header, info)| (info.real_merge == Some(m)).then_some(header))
                .collect();
            if headers.len() + loop_headers.len() < 2 {
                continue;
            }
            headers.sort_unstable();
            loop_headers.sort_unstable();
            let m_lbl = self
                .block_labels
                .get(&m)
                .copied()
                .or_else(|| {
                    if m == cfg.blocks.len() as BlockId {
                        self.return_block
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| panic!("nexium-spirv: missing shared merge block {m}"));
            let mut owners: Vec<(BlockId, bool)> = headers
                .iter()
                .copied()
                .map(|header| (header, false))
                .chain(loop_headers.iter().copied().map(|header| (header, true)))
                .collect();
            owners.sort_unstable_by_key(|(header, is_loop)| {
                (*header, if *is_loop { 0u8 } else { 1u8 })
            });

            let mut synths: Vec<(Word, u32, Word)> = Vec::new();
            let mut owner_labels: Vec<(BlockId, bool, Word)> = Vec::with_capacity(owners.len());
            for (idx, &(header, is_loop)) in owners.iter().enumerate() {
                let label = if idx == 0 { m_lbl } else { self.b.id() };
                if is_loop {
                    self.self_loops.get_mut(&header).unwrap().merge = label;
                } else {
                    self.header_merge_label.insert(header, label);
                }
                if idx != 0 {
                    let target = owner_labels
                        .iter()
                        .rev()
                        .find(|(parent, _, _)| dom[header as usize][*parent as usize])
                        .map(|(_, _, label)| *label)
                        .unwrap_or(m_lbl);
                    synths.push((label, header, target));
                }
                owner_labels.push((header, is_loop, label));
            }
            for b in 0..m {
                let target = owner_labels
                    .iter()
                    .rev()
                    .find(|(header, _, _)| dom[b as usize][*header as usize])
                    .map(|(_, _, label)| *label)
                    .unwrap_or(m_lbl);
                self.merge_redirects.insert((m, b), target);
            }
            synths.reverse();
            self.synth_merge_blocks.insert(m, synths);
            self.shared_merge_headers
                .insert(m, owners.into_iter().map(|(header, _)| header).collect());
        }
        self.rebuild_loop_break_merges(cfg, &reachable, &dom);
    }

    fn merge_redirect(&self, from_block: u32, to_block: u32) -> Word {
        if let Some(&(merge, label)) = self.loop_break_merges.get(&from_block) {
            if merge == to_block {
                return label;
            }
        }
        if self.shared_merge_headers.contains_key(&to_block) {
            if let Some(&label) = self.merge_redirects.get(&(to_block, from_block)) {
                return label;
            }
        }
        self.block_labels
            .get(&to_block)
            .copied()
            .or_else(|| {
                (to_block == self.block_labels.len() as BlockId)
                    .then_some(self.return_block)
                    .flatten()
            })
            .unwrap_or_else(|| panic!("nexium-spirv: missing branch target block {to_block}"))
    }

    fn prepare_indirect_structural_cases(&mut self, cfg: &Cfg) {
        let Some(ipd) = self.cond_merge.clone() else {
            return;
        };
        let dominators = dominators(cfg);
        let predecessors = cfg.predecessors();
        let reachable = cfg_reachable(cfg);

        for block in &cfg.blocks {
            let BranchKind::Indirect { count, targets, .. } = block.branch else {
                continue;
            };
            let merge = ipd[block.id as usize];
            let mut case_headers = targets
                .iter()
                .take(count as usize)
                .map(|target| target.target)
                .collect::<std::collections::HashSet<_>>();
            let mut structural_blocks = Vec::new();

            loop {
                let mut added = false;
                for candidate in &cfg.blocks {
                    let candidate_id = candidate.id;
                    if candidate_id == block.id
                        || candidate_id == merge
                        || !reachable[candidate_id as usize]
                        || !dominators[candidate_id as usize][block.id as usize]
                        || case_headers
                            .iter()
                            .any(|&header| dominators[candidate_id as usize][header as usize])
                    {
                        continue;
                    }
                    let enters_from_case =
                        predecessors[candidate_id as usize].iter().any(|&pred| {
                            case_headers
                                .iter()
                                .any(|&header| dominators[pred as usize][header as usize])
                        });
                    if enters_from_case && case_headers.insert(candidate_id) {
                        structural_blocks.push(candidate_id);
                        added = true;
                    }
                }
                if !added {
                    break;
                }
            }

            let mut groups = Vec::new();
            let mut grouped_selectors = std::collections::HashSet::new();
            for structural_block in structural_blocks {
                let structural_predecessors = &predecessors[structural_block as usize];
                let mut cases = Vec::new();
                for entry in targets.iter().take(count as usize) {
                    if grouped_selectors.contains(&entry.selector)
                        || !structural_predecessors
                            .iter()
                            .any(|&pred| dominators[pred as usize][entry.target as usize])
                    {
                        continue;
                    }
                    cases.push((entry.selector, self.merge_redirect(block.id, entry.target)));
                }
                if cases.len() < 2
                    || !structural_predecessors.iter().all(|&pred| {
                        cases.iter().any(|&(selector, _)| {
                            targets
                                .iter()
                                .take(count as usize)
                                .find(|entry| entry.selector == selector)
                                .is_some_and(|entry| {
                                    dominators[pred as usize][entry.target as usize]
                                })
                        })
                    })
                {
                    continue;
                }
                let merge_labels = structural_predecessors
                    .iter()
                    .map(|&pred| self.merge_redirect(pred, structural_block))
                    .collect::<std::collections::HashSet<_>>();
                if merge_labels.len() != 1 {
                    continue;
                }
                grouped_selectors.extend(cases.iter().map(|&(selector, _)| selector));
                groups.push(IndirectStructuralCaseGroup {
                    header: self.b.id(),
                    merge: *merge_labels.iter().next().unwrap(),
                    fallback: self.b.id(),
                    cases,
                });
            }
            if !groups.is_empty() {
                self.indirect_structural_cases.insert(block.id, groups);
            }
        }
    }

    fn phi_pred_label(&self, block: BlockId) -> Word {
        self.block_end_labels
            .get(&block)
            .copied()
            .unwrap_or_else(|| self.block_labels[&block])
    }

    fn loop_phi_parent(&self, header: BlockId, pred: BlockId) -> Option<Word> {
        self.self_loops
            .get(&header)
            .and_then(|info| (pred == info.latch).then_some(info.cont))
    }

    fn lower_pred_phi_source(
        &mut self,
        header: BlockId,
        pred: BlockId,
        pred_index: u8,
        value: Option<ValueId>,
    ) -> (Word, Word) {
        let emitted = self
            .block_pred_exits
            .get(&pred)
            .and_then(|state| state.get(pred_index as usize))
            .copied()
            .flatten();
        if let Some(cont) = self.loop_phi_parent(header, pred) {
            let value = match value {
                Some(_) if emitted.is_some() => emitted.unwrap(),
                Some(id) if !self.pred_value_to_word.contains_key(&(id, pred_index)) => {
                    let reserved = self.b.id();
                    let ty = self.bool_t;
                    self.loop_carried.entry(header).or_default().push((
                        id,
                        reserved,
                        ty,
                        Some(pred_index),
                    ));
                    reserved
                }
                Some(id) => self.pred_value_to_word[&(id, pred_index)],
                None => self.bool_false,
            };
            return (value, cont);
        }
        let value = emitted
            .or_else(|| {
                value.and_then(|id| self.pred_value_to_word.get(&(id, pred_index)).copied())
            })
            .unwrap_or(self.bool_false);
        (value, self.phi_pred_label(pred))
    }

    fn lower_phi_source(
        &mut self,
        header: BlockId,
        pred: BlockId,
        value: &IrValue,
    ) -> (Word, Word) {
        if let Some(cont) = self.loop_phi_parent(header, pred) {
            let value = match value {
                IrValue::Inst(id) if !self.value_to_word.contains_key(id) => {
                    let reserved = self.b.id();
                    let ty = self.f32_t;
                    self.loop_carried
                        .entry(header)
                        .or_default()
                        .push((*id, reserved, ty, None));
                    reserved
                }
                _ => self.lower_value(value),
            };
            return (value, cont);
        }
        (self.lower_value(value), self.phi_pred_label(pred))
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
                        let v = self
                            .block_pred_exits
                            .get(pred)
                            .and_then(|state| state.get(phi.pred as usize))
                            .copied()
                            .flatten()
                            .or_else(|| {
                                val.and_then(|id| {
                                    self.pred_value_to_word.get(&(id, phi.pred)).copied()
                                })
                            })
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

    fn emit_virtual_synth_merge_blocks(&mut self, merge: BlockId) {
        let Some(synths) = self.synth_merge_blocks.get(&merge).cloned() else {
            return;
        };
        for (label, _, target) in synths {
            self.b.begin_block(Some(label)).unwrap();
            self.b.branch(target).unwrap();
        }
    }

    fn redirect_stale_phi_parents(&mut self) {
        use rspirv::spirv::Op;

        let block_by_initial_label: HashMap<Word, BlockId> = self
            .block_labels
            .iter()
            .map(|(&block, &label)| (label, block))
            .collect();
        let loop_parents: HashMap<(Word, Word), Word> = self
            .self_loops
            .iter()
            .map(|(&header, info)| {
                (
                    (self.block_labels[&header], self.block_labels[&info.latch]),
                    info.cont,
                )
            })
            .collect();
        let block_end_labels = self.block_end_labels.clone();

        for func in &mut self.b.module_mut().functions {
            let mut predecessors: HashMap<Word, std::collections::HashSet<Word>> = HashMap::new();
            for block in &func.blocks {
                let Some(label) = block.label.as_ref().and_then(|inst| inst.result_id) else {
                    continue;
                };
                let Some(term) = block.instructions.last() else {
                    continue;
                };
                let targets: Vec<Word> = match term.class.opcode {
                    Op::Branch => term
                        .operands
                        .iter()
                        .take(1)
                        .filter_map(|operand| match operand {
                            Operand::IdRef(target) => Some(*target),
                            _ => None,
                        })
                        .collect(),
                    Op::BranchConditional => term
                        .operands
                        .iter()
                        .skip(1)
                        .take(2)
                        .filter_map(|operand| match operand {
                            Operand::IdRef(target) => Some(*target),
                            _ => None,
                        })
                        .collect(),
                    Op::Switch => term
                        .operands
                        .iter()
                        .skip(1)
                        .filter_map(|operand| match operand {
                            Operand::IdRef(target) => Some(*target),
                            _ => None,
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                for target in targets {
                    predecessors.entry(target).or_default().insert(label);
                }
            }

            for block in &mut func.blocks {
                let Some(label) = block.label.as_ref().and_then(|inst| inst.result_id) else {
                    continue;
                };
                let actual = predecessors.get(&label);
                for inst in &mut block.instructions {
                    if inst.class.opcode != Op::Phi {
                        continue;
                    }
                    let mut index = 1;
                    while index < inst.operands.len() {
                        let Operand::IdRef(parent) = inst.operands[index] else {
                            index += 2;
                            continue;
                        };
                        if actual.is_some_and(|preds| preds.contains(&parent)) {
                            index += 2;
                            continue;
                        }

                        let redirected = loop_parents
                            .get(&(label, parent))
                            .copied()
                            .filter(|candidate| {
                                actual.is_some_and(|preds| preds.contains(candidate))
                            })
                            .or_else(|| {
                                let block_id = block_by_initial_label.get(&parent)?;
                                let candidate = block_end_labels.get(block_id).copied()?;
                                actual
                                    .is_some_and(|preds| preds.contains(&candidate))
                                    .then_some(candidate)
                            });
                        if let Some(parent) = redirected {
                            inst.operands[index] = Operand::IdRef(parent);
                        }
                        index += 2;
                    }
                }
            }
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

    fn prepare_loops(&mut self, cfg: &Cfg) {
        let Some(ipd) = self.cond_merge.clone() else {
            return;
        };
        let reachable = cfg_reachable(cfg);
        let predecessors = cfg.predecessors();
        let dom = dominators(cfg);
        for block in &cfg.blocks {
            if !reachable[block.id as usize] {
                continue;
            }
            let (header, conditional) = match block.branch {
                BranchKind::Conditional { target, .. } if target <= block.id && target != 0 => {
                    (target, true)
                }
                BranchKind::Unconditional { target } if target <= block.id && target != 0 => {
                    (target, false)
                }
                _ => continue,
            };
            let body = self.b.id();
            let cont = self.b.id();
            let (merge, real_merge) = if conditional {
                let loop_blocks = natural_loop_blocks(header, block.id, &predecessors, &dom);
                let mut merge = ipd[header as usize];
                let mut seen = std::collections::HashSet::new();
                while (merge as usize) < cfg.blocks.len() && loop_blocks.contains(&merge) {
                    assert!(
                        seen.insert(merge),
                        "nexium-spirv: cyclic loop postdominator chain"
                    );
                    merge = ipd[merge as usize];
                }
                assert!(
                    (merge as usize) < cfg.blocks.len(),
                    "nexium-spirv: conditional loop has no real merge block"
                );
                (self.block_labels[&merge], Some(merge))
            } else {
                (self.b.id(), None)
            };
            self.self_loops.insert(
                header,
                LoopInfo {
                    body,
                    cont,
                    merge,
                    latch: block.id,
                    real_merge,
                },
            );
        }
        self.rebuild_loop_break_merges(cfg, &reachable, &dom);
    }

    fn rebuild_loop_break_merges(&mut self, cfg: &Cfg, reachable: &[bool], dom: &[Vec<bool>]) {
        self.loop_break_merges.clear();
        for block in &cfg.blocks {
            if !reachable[block.id as usize] {
                continue;
            }
            let BranchKind::Conditional { target, .. } = block.branch else {
                continue;
            };
            let next = block.id + 1;
            let candidate = self
                .self_loops
                .iter()
                .filter_map(|(&header, info)| {
                    let merge = info.real_merge?;
                    (dom[block.id as usize][header as usize] && (target == merge || next == merge))
                        .then_some((header, merge, info.merge))
                })
                .max_by_key(|(header, _, _)| *header);
            if let Some((_, merge, label)) = candidate {
                self.loop_break_merges.insert(block.id, (merge, label));
            }
        }
    }

    fn loop_break_merge(&self, block: &BasicBlock) -> Option<(BlockId, Word)> {
        self.loop_break_merges.get(&block.id).copied()
    }

    fn lower_cfg(&mut self, cfg: &Cfg) {
        let single_block = cfg.blocks.len() <= 1;
        let predecessors = cfg.predecessors();
        let reachable = cfg_reachable(cfg);
        for (idx, block) in cfg.blocks.iter().enumerate() {
            let is_first = idx == 0;
            if !is_first {
                if reachable[block.id as usize] {
                    self.emit_synth_merge_blocks(block);
                }
                let label = self.block_labels[&block.id];
                self.b.begin_block(Some(label)).unwrap();
            }
            if !reachable[block.id as usize] {
                self.b.unreachable().unwrap();
                continue;
            }
            self.current_block = Some(block.id);
            self.restore_pred_regs(cfg, &predecessors, block);
            self.lower_pred_phis(block);
            self.lower_phis(block);
            if let Some(info) = self.self_loops.get(&block.id).copied() {
                self.b
                    .loop_merge(info.merge, info.cont, rspirv::spirv::LoopControl::NONE, [])
                    .unwrap();
                self.b.branch(info.body).unwrap();
                self.b.begin_block(Some(info.body)).unwrap();
                self.block_end_labels.insert(block.id, info.body);
            }
            for inst in &block.program.instructions {
                if matches!(inst.op, IrOp::Phi { .. }) {
                    continue;
                }
                self.lower_op(inst);
            }
            self.block_pred_exits.insert(block.id, self.pred_regs);
            if !single_block {
                self.emit_terminator(block);
            }
            self.current_block = None;
        }
    }

    fn restore_pred_regs(&mut self, cfg: &Cfg, predecessors: &[Vec<BlockId>], block: &BasicBlock) {
        self.pred_regs = [None; 7];
        let Some(preds) = predecessors.get(block.id as usize) else {
            return;
        };
        for pred_index in 0..7u8 {
            if block.pred_phis.iter().any(|phi| phi.pred == pred_index) {
                continue;
            }
            let mut incoming = preds
                .iter()
                .map(|pred| cfg.block(*pred).pred_exit.get(&pred_index).copied());
            let Some(first) = incoming.next() else {
                continue;
            };
            if !incoming.all(|value| value == first) {
                continue;
            }
            let Some(value_id) = first else {
                continue;
            };
            self.pred_regs[pred_index as usize] = self
                .pred_value_to_word
                .get(&(value_id, pred_index))
                .copied()
                .or_else(|| {
                    preds.iter().find_map(|pred| {
                        self.block_pred_exits
                            .get(pred)
                            .and_then(|state| state[pred_index as usize])
                    })
                });
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
                        let (v, label) = self.lower_pred_phi_source(m, *pred, phi.pred, *val);
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
                    let (v, label) = self.lower_pred_phi_source(m, *pred, phi.pred, *val);
                    pairs.push((v, label));
                }
            }
            let id = self.b.phi(self.bool_t, None, pairs).unwrap();
            self.value_to_word.insert(phi.result, id);
            self.pred_value_to_word.insert((phi.result, phi.pred), id);
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
                        let (v, label) = self.lower_phi_source(m, *pred_id, val);
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
                    let (v, label) = self.lower_phi_source(m, *pred_id, val);
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
        let info = self.self_loops[&header_id];
        let unconditional = cond.is_none();
        let counter = self.loop_safety_vars[&header_id];
        let old_counter = self.b.load(self.u32_t, None, counter, None, []).unwrap();
        let one = self.const_u32(1);
        let new_counter = self.b.i_sub(self.u32_t, None, old_counter, one).unwrap();
        self.b.store(counter, new_counter, None, []).unwrap();
        let zero = self.const_u32(0);
        let safety_cond = self
            .b
            .s_greater_than_equal(self.bool_t, None, new_counter, zero)
            .unwrap();
        let repeat_cond = match cond {
            Some(c) => self
                .b
                .logical_and(self.bool_t, None, c, safety_cond)
                .unwrap(),
            None => safety_cond,
        };
        let exit_target = if unconditional {
            info.merge
        } else {
            let next = latch_id + 1;
            self.block_labels
                .get(&next)
                .map(|_| self.merge_redirect(latch_id, next))
                .unwrap_or(info.merge)
        };
        self.b
            .branch_conditional(repeat_cond, info.cont, exit_target, [])
            .unwrap();
        self.b.begin_block(Some(info.cont)).unwrap();
        if let Some(carried) = self.loop_carried.get(&header_id).cloned() {
            for (vid, reserved, ty, pred_index) in carried {
                let actual = match pred_index
                    .and_then(|pred| self.pred_value_to_word.get(&(vid, pred)).copied())
                    .or_else(|| pred_index.and_then(|pred| self.pred_regs[pred as usize]))
                    .or_else(|| self.value_to_word.get(&vid).copied())
                {
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
        if unconditional {
            self.b.begin_block(Some(info.merge)).unwrap();
            let virtual_merge = self.block_labels.len() as BlockId;
            let target = if self.shared_merge_headers.contains_key(&virtual_merge) {
                self.merge_redirect(latch_id, virtual_merge)
            } else {
                self.return_block.unwrap()
            };
            self.b.branch(target).unwrap();
            self.block_end_labels.insert(latch_id, info.merge);
        }
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
                        self.emit_alpha_test(&outputs);
                        for (loc, v) in outputs {
                            self.store_fragment_output_vec(loc, v);
                        }
                    }
                    let virtual_merge = self.block_labels.len() as BlockId;
                    let target = if self.shared_merge_headers.contains_key(&virtual_merge) {
                        self.merge_redirect(block.id, virtual_merge)
                    } else {
                        rb
                    };
                    self.b.branch(target).unwrap();
                }
            }
            BranchKind::Unconditional { target } => {
                if self
                    .self_loops
                    .get(&target)
                    .map_or(false, |info| info.latch == block.id)
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
                    .map_or(false, |info| info.latch == block.id)
                {
                    let cond = self.resolve_pred(pred.idx, pred.negate);
                    self.emit_self_loop_close(target, block.id, Some(cond));
                    return;
                }
                let next = block.id + 1;
                if self.block_labels.contains_key(&next) {
                    let cond = if std::env::var("NEXIUM_FORCE_GATE").is_ok()
                        && matches!(self.stage, Stage::Fragment)
                    {
                        self.bool_false
                    } else {
                        self.resolve_pred(pred.idx, pred.negate)
                    };
                    let loop_break = self.loop_break_merge(block);
                    let true_lbl = loop_break
                        .filter(|(merge, _)| *merge == target)
                        .map(|(_, label)| label)
                        .unwrap_or_else(|| self.merge_redirect(block.id, target));
                    let false_lbl = loop_break
                        .filter(|(merge, _)| *merge == next)
                        .map(|(_, label)| label)
                        .unwrap_or_else(|| self.merge_redirect(block.id, next));
                    if target == next {
                        self.b.branch(true_lbl).unwrap();
                    } else if loop_break.is_some() {
                        self.b
                            .branch_conditional(cond, true_lbl, false_lbl, [])
                            .unwrap();
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
            BranchKind::Indirect {
                register,
                base,
                count,
                targets,
                ..
            } => {
                let raw = block
                    .reg_exit
                    .get(&register)
                    .copied()
                    .unwrap_or(IrValue::GprIn(register));
                let raw_word = self.lower_value(&raw);
                let mut selector = self.as_u32(raw_word);
                if base != 0 {
                    let offset = self.const_u32(base);
                    selector = self.b.i_add(self.u32_t, None, selector, offset).unwrap();
                }
                let fallback = self.b.id();
                self.indirect_default_blocks.push(fallback);
                let merge_lbl = match self.header_merge_label.get(&block.id).copied() {
                    Some(label) => label,
                    None => {
                        let merge_id = match &self.cond_merge {
                            Some(ipd) => ipd[block.id as usize],
                            None => block.id + 1,
                        };
                        self.block_labels
                            .get(&merge_id)
                            .copied()
                            .or(self.return_block)
                            .expect("nexium-spirv: indirect branch has no merge block")
                    }
                };
                assert!(
                    self.used_merge_blocks.insert(merge_lbl),
                    "nexium-spirv: shared selection merge block"
                );
                self.b
                    .selection_merge(merge_lbl, rspirv::spirv::SelectionControl::NONE)
                    .unwrap();
                let structural_groups = self
                    .indirect_structural_cases
                    .get(&block.id)
                    .cloned()
                    .unwrap_or_default();
                let cases = targets
                    .iter()
                    .take(count as usize)
                    .map(|entry| {
                        let target = structural_groups
                            .iter()
                            .find(|group| {
                                group
                                    .cases
                                    .iter()
                                    .any(|&(selector, _)| selector == entry.selector)
                            })
                            .map(|group| group.header)
                            .unwrap_or_else(|| self.merge_redirect(block.id, entry.target));
                        (Operand::LiteralBit32(entry.selector), target)
                    })
                    .collect::<Vec<_>>();
                self.b.switch(selector, fallback, cases).unwrap();
                for group in structural_groups {
                    self.b.begin_block(Some(group.header)).unwrap();
                    assert!(
                        self.used_merge_blocks.insert(group.merge),
                        "nexium-spirv: shared structural case merge block"
                    );
                    self.b
                        .selection_merge(group.merge, rspirv::spirv::SelectionControl::NONE)
                        .unwrap();
                    let cases = group
                        .cases
                        .into_iter()
                        .map(|(literal, target)| (Operand::LiteralBit32(literal), target))
                        .collect::<Vec<_>>();
                    self.b.switch(selector, group.fallback, cases).unwrap();
                    self.b.begin_block(Some(group.fallback)).unwrap();
                    self.b.unreachable().unwrap();
                }
            }
            BranchKind::FallThrough => {
                let next = block.id + 1;
                if self.block_labels.contains_key(&next) {
                    let lbl = self.merge_redirect(block.id, next);
                    self.b.branch(lbl).unwrap();
                } else if let Some(rb) = self.return_block {
                    let virtual_merge = self.block_labels.len() as BlockId;
                    let target = if self.shared_merge_headers.contains_key(&virtual_merge) {
                        self.merge_redirect(block.id, virtual_merge)
                    } else {
                        rb
                    };
                    self.b.branch(target).unwrap();
                }
            }
        }
    }

    fn emit_indirect_defaults(&mut self) {
        for label in std::mem::take(&mut self.indirect_default_blocks) {
            self.b.begin_block(Some(label)).unwrap();
            self.b.unreachable().unwrap();
        }
    }

    fn emit_entry_inits(
        &mut self,
        required_outputs: &[(u32, AttrVar)],
        ps_inject: Option<(Word, u32)>,
    ) {
        if matches!(self.stage, Stage::Vertex) {
            let position = self.position_var();
            let default_position = self
                .b
                .composite_construct(
                    self.vec4_t,
                    None,
                    [self.f32_zero, self.f32_zero, self.f32_zero, self.f32_one],
                )
                .unwrap();
            self.b.store(position, default_position, None, []).unwrap();
            if let Some(layer) = self.layer_var {
                let zero = self.const_u32(0);
                self.b.store(layer, zero, None, []).unwrap();
            }
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

    fn set_image_type_depth(&mut self, image_type: Word) {
        let instruction = self
            .b
            .module_mut()
            .types_global_values
            .iter_mut()
            .find(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::TypeImage
                    && instruction.result_id == Some(image_type)
            })
            .expect("image type must exist");
        instruction.operands[2] = Operand::LiteralBit32(1);
    }

    fn configure_depth_image_types(&mut self, cfg: &Cfg) {
        let mut image_2d = [false; 2];
        let mut image_2d_arrayed = [false; 2];
        let mut image_cube = [false; 2];
        let mut image_cube_arrayed = [false; 2];
        let has_arrayed_2d = cfg.blocks.iter().any(|block| {
            block.program.instructions.iter().any(|instruction| {
                matches!(
                    &instruction.op,
                    IrOp::SampleTex {
                        array: Some(_),
                        volume: None,
                        cube: None,
                        ..
                    }
                )
            })
        });

        for block in &cfg.blocks {
            for instruction in &block.program.instructions {
                match &instruction.op {
                    IrOp::SampleTex {
                        array,
                        volume,
                        cube,
                        dref,
                        ..
                    } => {
                        let depth = usize::from(dref.is_some());
                        if volume.is_some() {
                            assert!(dref.is_none(), "3D depth-compare sampling is unsupported");
                        } else if cube.is_some() {
                            if array.is_some() {
                                image_cube_arrayed[depth] = true;
                            } else {
                                image_cube[depth] = true;
                            }
                        } else if has_arrayed_2d {
                            image_2d_arrayed[depth] = true;
                        } else {
                            image_2d[depth] = true;
                        }
                    }
                    IrOp::GatherTex { .. } => {
                        if has_arrayed_2d {
                            image_2d_arrayed[0] = true;
                        } else {
                            image_2d[0] = true;
                        }
                    }
                    IrOp::TexelFetch { y, z, .. } if z.is_none() && y.is_some() => {
                        if has_arrayed_2d {
                            image_2d_arrayed[0] = true;
                        } else {
                            image_2d[0] = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        for (usage, image_type, label) in [
            (image_2d, self.image_t, "2D"),
            (image_2d_arrayed, self.image_arrayed_t, "2D array"),
            (image_cube, self.image_cube_t, "cube"),
            (image_cube_arrayed, self.image_cube_arrayed_t, "cube array"),
        ] {
            if usage[1] {
                assert!(
                    !usage[0],
                    "mixed color and depth-compare {label} sampling needs separate descriptor arrays"
                );
                self.set_image_type_depth(image_type);
            }
        }
    }

    fn preallocate_resources(&mut self, cfg: &Cfg) {
        if self.stage == Stage::Compute {
            self.setup_compute_resources();
            if self
                .compute_options
                .as_ref()
                .is_some_and(|options| options.shared_memory_size != 0)
            {
                self.ensure_shared_mem_var();
            }
        }
        if matches!(self.stage, Stage::Vertex) && self.vertex_opts.layer_output_slot.is_some() {
            self.layer_var_id();
        }
        let mut needs_image = false;
        let has_arrayed_2d = cfg.blocks.iter().any(|block| {
            block.program.instructions.iter().any(|instruction| {
                matches!(
                    &instruction.op,
                    IrOp::SampleTex {
                        array: Some(_),
                        volume: None,
                        cube: None,
                        ..
                    }
                )
            })
        });
        let mut needs_sampler = false;
        let mut needs_arrayed_sampler = false;
        let mut needs_3d_image = false;
        let mut needs_cube_image = false;
        let mut needs_cube_arrayed_image = false;
        let mut tex_ids = std::collections::BTreeSet::new();
        let mut texture_kinds = std::collections::BTreeMap::new();
        let mut filtered_tex_ids = std::collections::BTreeSet::new();
        let mut texel_fetches = Vec::new();
        for block in &cfg.blocks {
            for inst in &block.program.instructions {
                match &inst.op {
                    IrOp::LoadCbuf { binding, .. } | IrOp::LoadCbufIndexed { binding, .. }
                        if self.stage == Stage::Compute =>
                    {
                        self.compute_cbuf_var(*binding);
                    }
                    IrOp::LoadLocal { .. } | IrOp::StoreLocal { .. } => {
                        self.ensure_local_mem_var();
                    }
                    IrOp::SubgroupLaneId | IrOp::SubgroupMask { .. } | IrOp::FSwzAdd { .. } => {
                        self.subgroup_id_var();
                    }
                    IrOp::Shfl { .. } => {
                        self.b.capability(Capability::GroupNonUniformShuffle);
                        self.subgroup_id_var();
                    }
                    IrOp::SubgroupVote { .. } => {
                        self.enable_subgroup_vote();
                    }
                    IrOp::LocalInvocationId { .. } => {
                        self.local_invocation_id_var();
                    }
                    IrOp::WorkgroupId { .. } => {
                        self.workgroup_id_var();
                    }
                    IrOp::TextureQueryDimension { .. } => {
                        self.b.capability(Capability::ImageQuery);
                    }
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
                        cube,
                        ..
                    } => {
                        needs_image = true;
                        needs_sampler = true;
                        needs_arrayed_sampler |= array.is_some() && cube.is_none();
                        needs_3d_image |= volume.is_some();
                        needs_cube_image |= cube.is_some() && array.is_none();
                        needs_cube_arrayed_image |= cube.is_some() && array.is_some();
                        tex_ids.insert(*tex_id);
                        filtered_tex_ids.insert(*tex_id);
                        let image_kind = if cube.is_some() {
                            if array.is_some() {
                                GraphicsImageKind::CubeArray
                            } else {
                                GraphicsImageKind::Cube
                            }
                        } else if volume.is_some() {
                            GraphicsImageKind::D3
                        } else if has_arrayed_2d {
                            GraphicsImageKind::D2Array
                        } else {
                            GraphicsImageKind::D2
                        };
                        record_graphics_texture_image_kind(
                            &mut texture_kinds,
                            *tex_id,
                            image_kind,
                        );
                    }
                    IrOp::TexelFetch {
                        cbuf_binding,
                        cbuf_word_offset,
                        cbuf_secondary_word_offset,
                        y,
                        z,
                        ..
                    } => {
                        needs_image = true;
                        let tex_id = nexium_shader::bindless_texture_id_pair(
                            *cbuf_binding,
                            *cbuf_word_offset,
                            *cbuf_secondary_word_offset,
                        );
                        tex_ids.insert(tex_id);
                        texel_fetches.push((
                            tex_id,
                            self.ir_constant_facts
                                .texel_fetch_buffer_coordinates_compatible(
                                    y.as_ref(),
                                    z.as_ref(),
                                ),
                            z.is_some(),
                        ));
                    }
                    IrOp::GatherTex { tex_id, .. } => {
                        needs_image = true;
                        needs_sampler = true;
                        tex_ids.insert(*tex_id);
                        filtered_tex_ids.insert(*tex_id);
                        record_graphics_texture_image_kind(
                            &mut texture_kinds,
                            *tex_id,
                            if has_arrayed_2d {
                                GraphicsImageKind::D2Array
                            } else {
                                GraphicsImageKind::D2
                            },
                        );
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
            Stage::Compute => {}
        }
        if needs_image {
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

            for tex_id in filtered_tex_ids {
                let slot = self.texture_slot(tex_id);
                let numeric_type = self.texture_numeric_type_at(tex_id);
                assert_eq!(
                    numeric_type,
                    TextureNumericType::Float,
                    "graphics texture {tex_id:#x} at descriptor slot {slot} is used by Sample/Gather but was assigned the incompatible {numeric_type:?} descriptor family"
                );
            }
            for (tex_id, buffer_candidate, is_3d) in texel_fetches {
                let kind = if buffer_candidate && self.texel_buffer_slot_enabled(tex_id) {
                    GraphicsImageKind::Buffer
                } else if is_3d {
                    GraphicsImageKind::D3
                } else if self.sampler_arrayed {
                    GraphicsImageKind::D2Array
                } else {
                    GraphicsImageKind::D2
                };
                record_graphics_texture_image_kind(&mut texture_kinds, tex_id, kind);
                let numeric_type = self.texture_numeric_type_at(tex_id);
                self.ensure_typed_image_array(numeric_type, kind);
            }
            if needs_sampler {
                self.ensure_sampler_array();
            }
            if needs_3d_image {
                self.ensure_image_3d_array();
            }
            if needs_cube_image {
                self.ensure_image_cube_array();
            }
            if needs_cube_arrayed_image {
                self.ensure_image_cube_arrayed_array();
            }
        }
        self.validate_graphics_texture_manifest(&texture_kinds)
            .unwrap_or_else(|error| {
                panic!("nexium-spirv: graphics texture manifest does not match shader resources: {error}")
            });
        for block in &cfg.blocks {
            let id = self.b.id();
            self.block_labels.insert(block.id, id);
            self.block_end_labels.insert(block.id, id);
        }
        let loop_headers = cfg
            .blocks
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(_, block)| match block.branch {
                BranchKind::Conditional { target, .. } if target <= block.id && target != 0 => {
                    Some(target)
                }
                BranchKind::Unconditional { target } if target <= block.id && target != 0 => {
                    Some(target)
                }
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        if !loop_headers.is_empty() {
            let ptr_private_u32 = self.b.type_pointer(None, StorageClass::Private, self.u32_t);
            let initial = self.const_u32(SHADER_LOOP_SAFETY_LIMIT);
            for header in loop_headers {
                let var =
                    self.b
                        .variable(ptr_private_u32, None, StorageClass::Private, Some(initial));
                self.loop_safety_vars.insert(header, var);
            }
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
        validate_spirv_ir(cfg, self.stage)
            .unwrap_or_else(|error| panic!("nexium-spirv: refusing unsupported IR: {error}"));
        self.ir_constant_facts = nexium_shader::IrConstantFacts::analyze(cfg);
        self.configure_depth_image_types(cfg);
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
        let bounded_unconditional_loop = cfg.blocks.iter().enumerate().skip(1).any(|(_, block)| {
            matches!(
                block.branch,
                BranchKind::Unconditional { target }
                    if target <= block.id && target != 0
            )
        });
        let has_indirect_branch = cfg
            .blocks
            .iter()
            .any(|block| matches!(block.branch, BranchKind::Indirect { .. }));

        let void_t = self.b.type_void();
        let main_t = self.b.type_function(void_t, vec![]);
        let main_id = self
            .b
            .begin_function(void_t, None, FunctionControl::NONE, main_t)
            .unwrap();

        if has_indirect_branch
            || bounded_unconditional_loop
            || (multi_exit && std::env::var_os("NEXIUM_NO_STRUCT_EXIT").is_none())
        {
            self.return_block = Some(self.b.id());
        }

        let entry_label = cfg.blocks.first().map(|b| self.block_labels[&b.id]);
        self.b.begin_block(entry_label).unwrap();
        self.emit_entry_inits(&required_outputs, ps_inject);

        self.cond_merge = structurizer_cond_merges(cfg);
        self.prepare_loops(cfg);
        self.compute_shared_merges(cfg);
        self.prepare_indirect_structural_cases(cfg);
        self.lower_cfg(cfg);
        self.emit_indirect_defaults();
        self.emit_virtual_synth_merge_blocks(cfg.blocks.len() as BlockId);

        if let Some(rb) = self.return_block {
            self.b.begin_block(Some(rb)).unwrap();
        }

        match self.stage {
            Stage::Vertex => {
                if self.vertex_opts.inject_ubo_matrix {
                    let pos_var = self.position_var();
                    let p = self.b.load(self.vec4_t, None, pos_var, None, []).unwrap();
                    let mut m = [[0u32; 4]; 4];
                    for col in 0u32..4 {
                        for row in 0u32..4 {
                            let effective = self.const_u32(col * 16 + row * 4);
                            let word = self.graphics_cbuf_load_word(0, effective);
                            m[col as usize][row as usize] =
                                self.b.bitcast(self.f32_t, None, word).unwrap();
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
                    self.emit_alpha_test(&outputs);
                    for (loc, v) in outputs {
                        self.store_fragment_output_vec(loc, v);
                    }
                }
            }
            Stage::Compute => {}
        }

        self.b.ret().unwrap();
        self.b.end_function().unwrap();

        let exec_model = match self.stage {
            Stage::Vertex => ExecutionModel::Vertex,
            Stage::Fragment => ExecutionModel::Fragment,
            Stage::Compute => ExecutionModel::GLCompute,
        };
        self.b
            .entry_point(exec_model, main_id, "main", self.interface.clone());
        if self.stage == Stage::Fragment {
            self.b
                .execution_mode(main_id, rspirv::spirv::ExecutionMode::OriginUpperLeft, []);
        } else if self.stage == Stage::Compute {
            self.b.extension("SPV_KHR_float_controls");
            self.b.capability(Capability::SignedZeroInfNanPreserve);
            self.b.execution_mode(
                main_id,
                rspirv::spirv::ExecutionMode::SignedZeroInfNanPreserve,
                [32],
            );
            let local_size = self
                .compute_options
                .as_ref()
                .expect("compute options are required for a compute emitter")
                .local_size;
            self.b
                .execution_mode(main_id, rspirv::spirv::ExecutionMode::LocalSize, local_size);
        }

        self.redirect_stale_phi_parents();
        let bindings = self.cbuf_bindings_used;
        let tex_ids: Vec<u32> = self.texs_ids_used.iter().copied().collect();
        let phi_cfg_lines = if std::env::var_os("NEXIUM_SPIRV_PHI_DBG").is_some() {
            self.phi_cfg_lines(cfg)
        } else {
            Vec::new()
        };
        let words = opt::dedup_constants(self.b.module().assemble());
        dump_spirv_words(&words, self.stage, multi_exit);
        if !phi_preds_consistent(&words) {
            for line in phi_cfg_lines {
                log::warn!("{}", line);
            }
            panic!("nexium-spirv: invalid phi predecessors");
        }
        if let Err(error) = validate_structured_cfg(&words) {
            panic!("nexium-spirv: invalid structured CFG: {error}");
        }
        if std::env::var_os("NEXIUM_EXIT_GUARD").is_some() && !selection_exits_structured(&words) {
            panic!("nexium-spirv: unstructured selection exit");
        }
        (words, bindings, tex_ids, sampler_arrayed)
    }
}

fn dump_spirv_words(words: &[u32], stage: Stage, multi_exit: bool) {
    use std::io::Write;

    let Some(dir) = std::env::var_os("NEXIUM_DUMP_SPIRV") else {
        return;
    };
    let stage = match stage {
        Stage::Vertex => "vs",
        Stage::Fragment => "fs",
        Stage::Compute => "cs",
    };
    let mut hash: u64 = 1469598103934665603;
    for word in words {
        hash ^= *word as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!(
        "{}_{:016x}_me{}.spv",
        stage, hash, multi_exit as u8
    ));
    if let Ok(mut file) = std::fs::File::create(path) {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let _ = file.write_all(&bytes);
    }
}

pub fn phi_preds_consistent(words: &[u32]) -> bool {
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
                    let debug = std::env::var_os("NEXIUM_SPIRV_PHI_DBG").is_some();
                    if inst.operands.len() % 2 != 0 {
                        if debug {
                            log::warn!(
                                "[spirv-phi] block={} malformed_operands={}",
                                label,
                                inst.operands.len()
                            );
                        }
                        return false;
                    }
                    let pair_count = inst.operands.len() / 2;
                    if pair_count == 0 || pair_count != bpreds.len() {
                        if debug {
                            let mut have: Vec<Word> = bpreds.iter().copied().collect();
                            have.sort_unstable();
                            log::warn!(
                                "[spirv-phi] block={} pairs={} predecessor_count={} preds={:?}",
                                label,
                                pair_count,
                                bpreds.len(),
                                have
                            );
                        }
                        return false;
                    }
                    let mut parents = std::collections::HashSet::new();
                    for pair in inst.operands.chunks_exact(2) {
                        if !matches!(pair[0], Operand::IdRef(_)) {
                            if debug {
                                log::warn!("[spirv-phi] block={} non_id_value", label);
                            }
                            return false;
                        }
                        let Operand::IdRef(parent) = pair[1] else {
                            if debug {
                                log::warn!("[spirv-phi] block={} non_id_parent", label);
                            }
                            return false;
                        };
                        if !parents.insert(parent) {
                            if debug {
                                log::warn!(
                                    "[spirv-phi] block={} duplicate_parent={}",
                                    label,
                                    parent
                                );
                            }
                            return false;
                        }
                    }
                    if parents != *bpreds {
                        if debug {
                            let mut missing: Vec<Word> =
                                bpreds.difference(&parents).copied().collect();
                            let mut extra: Vec<Word> =
                                parents.difference(bpreds).copied().collect();
                            missing.sort_unstable();
                            extra.sort_unstable();
                            log::warn!(
                                "[spirv-phi] block={} missing={:?} extra={:?}",
                                label,
                                missing,
                                extra
                            );
                        }
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn validate_structured_cfg(words: &[u32]) -> Result<(), String> {
    use rspirv::spirv::Op;

    let module = rspirv::dr::load_words(words).map_err(|error| error.to_string())?;
    for function in &module.functions {
        if function.blocks.is_empty() {
            continue;
        }
        let labels = function
            .blocks
            .iter()
            .map(|block| {
                block
                    .label
                    .as_ref()
                    .and_then(|label| label.result_id)
                    .ok_or_else(|| "function block without a label".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let label_index = labels
            .iter()
            .enumerate()
            .map(|(index, &label)| (label, index))
            .collect::<HashMap<_, _>>();
        let mut successors = vec![Vec::<usize>::new(); function.blocks.len()];
        for (index, block) in function.blocks.iter().enumerate() {
            let Some(terminator) = block.instructions.last() else {
                return Err(format!("block {} has no terminator", labels[index]));
            };
            for target in structured_branch_targets(terminator) {
                let Some(&target_index) = label_index.get(&target) else {
                    return Err(format!(
                        "block {} branches to unknown label {target}",
                        labels[index]
                    ));
                };
                if !successors[index].contains(&target_index) {
                    successors[index].push(target_index);
                }
            }
        }

        let mut reachable = vec![false; function.blocks.len()];
        let mut pending = vec![0usize];
        while let Some(block) = pending.pop() {
            if reachable[block] {
                continue;
            }
            reachable[block] = true;
            pending.extend(successors[block].iter().copied());
        }
        let mut predecessors = vec![Vec::<usize>::new(); function.blocks.len()];
        for (source, targets) in successors.iter().enumerate() {
            if !reachable[source] {
                continue;
            }
            for &target in targets {
                if reachable[target] {
                    predecessors[target].push(source);
                }
            }
        }
        let dominators = structured_dominators(&reachable, &predecessors);
        let postdominators = structured_postdominators(&reachable, &successors);
        let mut merge_owners = HashMap::<usize, (usize, Op)>::new();
        let mut loops = HashMap::<usize, (usize, usize)>::new();
        let mut selections = Vec::<(usize, usize)>::new();
        for (header, block) in function.blocks.iter().enumerate() {
            if !reachable[header] {
                continue;
            }
            for instruction in &block.instructions {
                let (merge_label, cont_label) = match instruction.class.opcode {
                    Op::LoopMerge => {
                        let (Some(Operand::IdRef(merge)), Some(Operand::IdRef(cont))) =
                            (instruction.operands.first(), instruction.operands.get(1))
                        else {
                            return Err(format!(
                                "malformed loop merge in block {}",
                                labels[header]
                            ));
                        };
                        (*merge, Some(*cont))
                    }
                    Op::SelectionMerge => {
                        let Some(Operand::IdRef(merge)) = instruction.operands.first() else {
                            return Err(format!(
                                "malformed selection merge in block {}",
                                labels[header]
                            ));
                        };
                        (*merge, None)
                    }
                    _ => continue,
                };
                let Some(&merge) = label_index.get(&merge_label) else {
                    return Err(format!("unknown merge label {merge_label}"));
                };
                if let Some((owner, opcode)) =
                    merge_owners.insert(merge, (header, instruction.class.opcode))
                {
                    return Err(format!(
                        "merge block {} is shared by {:?} header {} and {:?} header {}",
                        labels[merge],
                        opcode,
                        labels[owner],
                        instruction.class.opcode,
                        labels[header]
                    ));
                }
                if let Some(cont_label) = cont_label {
                    let Some(&cont) = label_index.get(&cont_label) else {
                        return Err(format!("unknown continue label {cont_label}"));
                    };
                    loops.insert(header, (merge, cont));
                } else {
                    selections.push((header, merge));
                }
            }
        }

        let terminal_blocks = function
            .blocks
            .iter()
            .map(|block| {
                block.instructions.last().is_some_and(|terminator| {
                    rspirv::grammar::reflect::is_return_or_abort(terminator.class.opcode)
                })
            })
            .collect::<Vec<_>>();
        for &(header, merge) in &selections {
            if !dominators[merge][header] {
                return Err(format!(
                    "selection header {} does not dominate merge {}",
                    labels[header], labels[merge]
                ));
            }
            let mut visit_state = vec![0u8; function.blocks.len()];
            let mut visit_result = vec![false; function.blocks.len()];
            let exits_through_merge_or_termination = successors[header].iter().all(|&target| {
                selection_path_reaches_merge_or_termination(
                    target,
                    merge,
                    &successors,
                    &postdominators,
                    &terminal_blocks,
                    &mut visit_state,
                    &mut visit_result,
                )
            });
            if !postdominators[header][merge] && !exits_through_merge_or_termination {
                return Err(format!(
                    "selection merge {} does not postdominate header {}",
                    labels[merge], labels[header]
                ));
            }
        }

        let mut backedges = HashMap::<usize, Vec<usize>>::new();
        for (source, targets) in successors.iter().enumerate() {
            if !reachable[source] {
                continue;
            }
            for &target in targets {
                if target <= source {
                    if !loops.contains_key(&target) {
                        return Err(format!(
                            "backedge {} -> {} does not target a loop header",
                            labels[source], labels[target]
                        ));
                    }
                    if !dominators[source][target] {
                        return Err(format!(
                            "loop header {} does not dominate backedge source {}",
                            labels[target], labels[source]
                        ));
                    }
                    backedges.entry(target).or_default().push(source);
                }
            }
        }

        for (&header, &(merge, cont)) in &loops {
            if !reachable[merge] || !reachable[cont] {
                return Err(format!(
                    "loop header {} has unreachable targets",
                    labels[header]
                ));
            }
            if !dominators[cont][header] {
                return Err(format!(
                    "loop header {} does not dominate continue target {}",
                    labels[header], labels[cont]
                ));
            }
            let Some(latches) = backedges.get(&header) else {
                return Err(format!("loop header {} has no backedge", labels[header]));
            };
            let mut region = std::collections::HashSet::from([header]);
            let mut reverse = latches.clone();
            region.extend(latches.iter().copied());
            while let Some(block) = reverse.pop() {
                if block == header {
                    continue;
                }
                for &pred in &predecessors[block] {
                    if dominators[pred][header] && region.insert(pred) {
                        reverse.push(pred);
                    }
                }
            }
            let mut forward = region.iter().copied().collect::<Vec<_>>();
            while let Some(block) = forward.pop() {
                for &target in &successors[block] {
                    if target == merge || region.contains(&target) {
                        continue;
                    }
                    if !dominators[target][header] {
                        return Err(format!(
                            "loop {} exits through {} instead of merge {}",
                            labels[header], labels[target], labels[merge]
                        ));
                    }
                    region.insert(target);
                    forward.push(target);
                }
            }
            if region.contains(&merge) {
                return Err(format!(
                    "loop merge {} is inside loop {}",
                    labels[merge], labels[header]
                ));
            }
            for &block in &region {
                for &target in &successors[block] {
                    if !region.contains(&target) && target != merge {
                        return Err(format!(
                            "loop {} exits through {} instead of merge {}",
                            labels[header], labels[target], labels[merge]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn selection_path_reaches_merge_or_termination(
    block: usize,
    merge: usize,
    successors: &[Vec<usize>],
    postdominators: &[Vec<bool>],
    terminal_blocks: &[bool],
    visit_state: &mut [u8],
    visit_result: &mut [bool],
) -> bool {
    if block == merge || postdominators[block][merge] || terminal_blocks[block] {
        return true;
    }
    match visit_state[block] {
        1 => return true,
        2 => return visit_result[block],
        _ => {}
    }
    visit_state[block] = 1;
    let result = !successors[block].is_empty()
        && successors[block].iter().all(|&target| {
            selection_path_reaches_merge_or_termination(
                target,
                merge,
                successors,
                postdominators,
                terminal_blocks,
                visit_state,
                visit_result,
            )
        });
    visit_state[block] = 2;
    visit_result[block] = result;
    result
}

fn structured_branch_targets(instruction: &rspirv::dr::Instruction) -> Vec<Word> {
    use rspirv::spirv::Op;

    let operands: Box<dyn Iterator<Item = &Operand> + '_> = match instruction.class.opcode {
        Op::Branch => Box::new(instruction.operands.iter().take(1)),
        Op::BranchConditional => Box::new(instruction.operands.iter().skip(1).take(2)),
        Op::Switch => Box::new(instruction.operands.iter().skip(1)),
        _ => Box::new(std::iter::empty()),
    };
    operands
        .filter_map(|operand| match operand {
            Operand::IdRef(target) => Some(*target),
            _ => None,
        })
        .collect()
}

fn structured_dominators(reachable: &[bool], predecessors: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = reachable.len();
    let mut dominators = vec![vec![false; n]; n];
    for block in 0..n {
        if reachable[block] {
            for candidate in 0..n {
                dominators[block][candidate] = reachable[candidate];
            }
        }
    }
    if n == 0 {
        return dominators;
    }
    dominators[0].fill(false);
    dominators[0][0] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for block in 1..n {
            if !reachable[block] {
                continue;
            }
            let mut next = vec![false; n];
            for candidate in 0..n {
                next[candidate] = reachable[candidate]
                    && predecessors[block]
                        .iter()
                        .filter(|&&pred| reachable[pred])
                        .all(|&pred| dominators[pred][candidate]);
            }
            next[block] = true;
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
    }
    dominators
}

fn structured_postdominators(reachable: &[bool], successors: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let n = reachable.len();
    let virt = n;
    let mut postdominators = vec![vec![false; n + 1]; n + 1];
    for block in 0..n {
        if reachable[block] {
            for candidate in 0..=n {
                postdominators[block][candidate] = candidate == virt || reachable[candidate];
            }
        }
    }
    postdominators[virt][virt] = true;
    let mut changed = true;
    while changed {
        changed = false;
        for block in (0..n).rev() {
            if !reachable[block] {
                continue;
            }
            let targets = successors[block]
                .iter()
                .copied()
                .filter(|&target| reachable[target])
                .collect::<Vec<_>>();
            let mut next = vec![false; n + 1];
            for candidate in 0..=n {
                next[candidate] = if targets.is_empty() {
                    postdominators[virt][candidate]
                } else {
                    targets
                        .iter()
                        .all(|&target| postdominators[target][candidate])
                };
            }
            next[block] = true;
            if next != postdominators[block] {
                postdominators[block] = next;
                changed = true;
            }
        }
    }
    postdominators
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

fn natural_loop_blocks(
    header: BlockId,
    latch: BlockId,
    predecessors: &[Vec<BlockId>],
    dominators: &[Vec<bool>],
) -> std::collections::HashSet<BlockId> {
    let mut blocks = std::collections::HashSet::from([header, latch]);
    let mut pending = vec![latch];
    while let Some(block) = pending.pop() {
        if block == header {
            continue;
        }
        for &pred in &predecessors[block as usize] {
            if dominators[pred as usize][header as usize] && blocks.insert(pred) {
                pending.push(pred);
            }
        }
    }
    blocks
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
        if is_bounded_unconditional_backedge(block) {
            succs[i].push(virt);
        }
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

#[derive(Clone, Copy)]
struct LoopInfo {
    body: Word,
    cont: Word,
    merge: Word,
    latch: BlockId,
    real_merge: Option<BlockId>,
}

fn is_bounded_unconditional_backedge(block: &BasicBlock) -> bool {
    matches!(
        block.branch,
        BranchKind::Unconditional { target } if target <= block.id && target != 0
    )
}

fn cfg_reachable(cfg: &Cfg) -> Vec<bool> {
    let mut reachable = vec![false; cfg.blocks.len()];
    if reachable.is_empty() {
        return reachable;
    }
    let mut pending = vec![0u32];
    while let Some(block) = pending.pop() {
        let Some(seen) = reachable.get_mut(block as usize) else {
            continue;
        };
        if *seen {
            continue;
        }
        *seen = true;
        pending.extend(cfg.successors(block));
    }
    reachable
}

pub fn emit_vertex(cfg: &Cfg) -> Vec<u32> {
    Emitter::new(Stage::Vertex).finish(cfg)
}

pub fn try_emit_vertex(cfg: &Cfg) -> Result<Vec<u32>, SpirvEmitError> {
    validate_spirv_ir(cfg, Stage::Vertex)?;
    Ok(Emitter::new(Stage::Vertex).finish(cfg))
}

pub fn emit_vertex_with_required_outputs(cfg: &Cfg, required_outputs: &[u32]) -> Vec<u32> {
    Emitter::new(Stage::Vertex).finish_with_required_outputs(cfg, required_outputs)
}

pub fn emit_fragment(cfg: &Cfg) -> Vec<u32> {
    Emitter::new(Stage::Fragment).finish(cfg)
}

pub fn try_emit_fragment(cfg: &Cfg) -> Result<Vec<u32>, SpirvEmitError> {
    validate_spirv_ir(cfg, Stage::Fragment)?;
    Ok(Emitter::new(Stage::Fragment).finish(cfg))
}

fn compute_resource_dimension_matches(instruction: ImageDimension, actual: ImageDimension) -> bool {
    instruction == actual || (instruction == ImageDimension::D1 && actual == ImageDimension::Buffer)
}

fn validate_compute_coordinates(
    handle: TextureHandleOrigin,
    dimension: ImageDimension,
    y: Option<&IrValue>,
    z: Option<&IrValue>,
) -> Result<(), ComputeEmitError> {
    if matches!(dimension, ImageDimension::D2 | ImageDimension::D3) && y.is_none() {
        return Err(ComputeEmitError::MissingCoordinate {
            handle,
            dimension,
            coordinate: "Y",
        });
    }
    if dimension == ImageDimension::D3 && z.is_none() {
        return Err(ComputeEmitError::MissingCoordinate {
            handle,
            dimension,
            coordinate: "Z",
        });
    }
    Ok(())
}

fn validate_compute_options(
    cfg: &Cfg,
    options: &ComputeOptions,
) -> Result<[u32; COMPUTE_CBUF_SLOTS], ComputeEmitError> {
    validate_spirv_ir(cfg, Stage::Compute)
        .map_err(|error| ComputeEmitError::UnsupportedOperation(error.to_string()))?;
    if options.local_size.contains(&0) {
        return Err(ComputeEmitError::InvalidLocalSize(options.local_size));
    }
    if options.shared_memory_size > MAX_COMPUTE_SHARED_MEMORY_SIZE {
        return Err(ComputeEmitError::InvalidSharedMemorySize(
            options.shared_memory_size,
        ));
    }
    let uses_local_memory = cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
        .any(|instruction| {
            matches!(
                &instruction.op,
                IrOp::LoadLocal { .. } | IrOp::StoreLocal { .. }
            )
        });
    if uses_local_memory {
        let local_memory_size = options
            .local_memory_low_size
            .checked_add(options.local_memory_high_size)
            .ok_or(ComputeEmitError::InvalidLocalMemoryAllocation {
                low: options.local_memory_low_size,
                high: options.local_memory_high_size,
            })?;
        if local_memory_size == 0 {
            return Err(ComputeEmitError::MissingLocalMemory);
        }
        if local_memory_size > MAX_COMPUTE_LOCAL_MEMORY_SIZE {
            return Err(ComputeEmitError::InvalidLocalMemorySize(local_memory_size));
        }
    }
    if usize::from(options.texture_bound_cbuf) >= COMPUTE_CBUF_SLOTS {
        return Err(ComputeEmitError::InvalidTextureBoundCbuf(
            options.texture_bound_cbuf,
        ));
    }
    if cfg.unimplemented != 0 {
        return Err(ComputeEmitError::UnimplementedIr(cfg.unimplemented));
    }

    let mut cbuf_required_sizes = [0u32; COMPUTE_CBUF_SLOTS];
    for instruction in cfg
        .blocks
        .iter()
        .flat_map(|block| &block.program.instructions)
    {
        let (binding, static_end, indexed) = match &instruction.op {
            IrOp::LoadCbuf {
                binding,
                byte_offset,
            } => (*binding, byte_offset.checked_add(4), false),
            IrOp::LoadCbufIndexed {
                binding,
                byte_offset,
                ..
            } => (*binding, byte_offset.checked_add(4), true),
            _ => continue,
        };
        let slot = usize::from(binding);
        if slot >= COMPUTE_CBUF_SLOTS {
            return Err(ComputeEmitError::InvalidCbufBinding(binding));
        }
        let available = options.cbuf_sizes[slot];
        let required = if indexed {
            if available > COMPUTE_CBUF_MAX_SIZE {
                return Err(ComputeEmitError::IndexedCbufTooLarge {
                    binding,
                    size: available,
                });
            }
            available
        } else {
            static_end.unwrap_or(u32::MAX)
        };
        if required == 0 || static_end.is_none_or(|end| end > available) {
            return Err(ComputeEmitError::CbufOutOfBounds {
                binding,
                end: static_end.unwrap_or(u32::MAX),
                available,
            });
        }
        cbuf_required_sizes[slot] = cbuf_required_sizes[slot].max(required);
    }

    let mut bindings = std::collections::HashSet::new();
    let mut filtered_handles = std::collections::HashSet::new();
    let mut sampled_handles = std::collections::HashSet::new();
    let mut storage_texel_handles = std::collections::HashSet::new();
    let mut storage_handles = std::collections::HashSet::new();
    for resource in &options.resources {
        if compute_cbuf_slot_for_descriptor_binding(resource.binding).is_some() {
            return Err(if resource.binding == 0 {
                ComputeEmitError::ReservedImageBinding
            } else {
                ComputeEmitError::ReservedCbufBinding(resource.binding)
            });
        }
        if !bindings.insert(resource.binding) {
            return Err(ComputeEmitError::DuplicateBinding(resource.binding));
        }
        let valid_dimension = match resource.kind {
            ComputeResourceKind::CombinedSampledImage => {
                resource.dimension != ImageDimension::Buffer
            }
            ComputeResourceKind::SampledImage => resource.dimension != ImageDimension::Buffer,
            ComputeResourceKind::UniformTexelBuffer => resource.dimension == ImageDimension::Buffer,
            ComputeResourceKind::StorageTexelBuffer => resource.dimension == ImageDimension::Buffer,
            ComputeResourceKind::StorageImage => resource.dimension != ImageDimension::Buffer,
        };
        if !valid_dimension {
            return Err(ComputeEmitError::InvalidResourceDimension {
                handle: resource.handle,
                kind: resource.kind,
                dimension: resource.dimension,
            });
        }
        let is_texel_buffer = matches!(
            resource.kind,
            ComputeResourceKind::UniformTexelBuffer | ComputeResourceKind::StorageTexelBuffer
        );
        match (is_texel_buffer, resource.texel_format) {
            (true, Some(format)) if format.numeric_type() == resource.numeric_type => {}
            (true, Some(format)) => {
                return Err(ComputeEmitError::UnsupportedOperation(format!(
                    "texel buffer {:?} declares {format:?} with {:?} numeric type",
                    resource.handle, resource.numeric_type
                )));
            }
            (true, None) => {
                return Err(ComputeEmitError::UnsupportedOperation(format!(
                    "texel buffer {:?} has no exact element format",
                    resource.handle
                )));
            }
            (false, Some(format)) => {
                return Err(ComputeEmitError::UnsupportedOperation(format!(
                    "non-texel resource {:?} unexpectedly declares {format:?}",
                    resource.handle
                )));
            }
            (false, None) => {}
        }
        let inserted = match resource.kind {
            ComputeResourceKind::CombinedSampledImage => filtered_handles.insert(resource.handle),
            ComputeResourceKind::SampledImage | ComputeResourceKind::UniformTexelBuffer => {
                sampled_handles.insert(resource.handle)
            }
            ComputeResourceKind::StorageTexelBuffer => {
                storage_texel_handles.insert(resource.handle)
            }
            ComputeResourceKind::StorageImage => storage_handles.insert(resource.handle),
        };
        if !inserted {
            return Err(ComputeEmitError::DuplicateResource {
                handle: resource.handle,
                kind: resource.kind,
            });
        }
    }

    let filtered_resource = |handle: TextureHandleOrigin| {
        options.resources.iter().find(|resource| {
            resource.handle == handle && resource.kind == ComputeResourceKind::CombinedSampledImage
        })
    };
    let sampled_resource = |handle: TextureHandleOrigin| {
        options.resources.iter().find(|resource| {
            resource.handle == handle
                && matches!(
                    resource.kind,
                    ComputeResourceKind::SampledImage | ComputeResourceKind::UniformTexelBuffer
                )
        })
    };
    let storage_resource = |handle: TextureHandleOrigin| {
        options.resources.iter().find(|resource| {
            resource.handle == handle
                && matches!(
                    resource.kind,
                    ComputeResourceKind::StorageImage | ComputeResourceKind::StorageTexelBuffer
                )
        })
    };
    let atomic_resource = |handle: TextureHandleOrigin| {
        options.resources.iter().find(|resource| {
            resource.handle == handle && resource.kind == ComputeResourceKind::StorageTexelBuffer
        })
    };

    for block in &cfg.blocks {
        for instruction in &block.program.instructions {
            match &instruction.op {
                IrOp::SampleTexHandle {
                    handle,
                    dimension,
                    v,
                    w,
                    dref,
                    texel_offset,
                    component,
                    ..
                } => {
                    if *component > 3 {
                        return Err(ComputeEmitError::UnsupportedOperation(format!(
                            "SampleTexHandle component {component}"
                        )));
                    }
                    if dref.is_some() {
                        return Err(ComputeEmitError::UnsupportedOperation(
                            "compute depth-compare sample".to_owned(),
                        ));
                    }
                    let resource =
                        filtered_resource(*handle).ok_or(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: ComputeResourceKind::CombinedSampledImage,
                        })?;
                    if *dimension != resource.dimension {
                        return Err(ComputeEmitError::DimensionMismatch {
                            handle: *handle,
                            instruction: *dimension,
                            actual: resource.dimension,
                        });
                    }
                    validate_compute_coordinates(
                        *handle,
                        resource.dimension,
                        v.as_ref(),
                        w.as_ref(),
                    )?;
                    if let Some((x, y)) = texel_offset {
                        if !matches!(dimension, ImageDimension::D1 | ImageDimension::D2) {
                            return Err(ComputeEmitError::UnsupportedOperation(format!(
                                "filtered sample offset for {dimension:?}"
                            )));
                        }
                        let is_constant = |value: &IrValue| {
                            matches!(
                                value,
                                IrValue::Zero | IrValue::ImmU32(_) | IrValue::ImmF32(_)
                            )
                        };
                        if !is_constant(x) || !is_constant(y) {
                            return Err(ComputeEmitError::UnsupportedOperation(
                                "dynamic filtered sample offset".to_owned(),
                            ));
                        }
                    }
                }
                IrOp::TexelFetchHandle {
                    handle,
                    dimension,
                    y,
                    z,
                    ..
                } => {
                    let resource =
                        sampled_resource(*handle).ok_or(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: ComputeResourceKind::SampledImage,
                        })?;
                    if !compute_resource_dimension_matches(*dimension, resource.dimension) {
                        return Err(ComputeEmitError::DimensionMismatch {
                            handle: *handle,
                            instruction: *dimension,
                            actual: resource.dimension,
                        });
                    }
                    validate_compute_coordinates(
                        *handle,
                        resource.dimension,
                        y.as_ref(),
                        z.as_ref(),
                    )?;
                }
                IrOp::TextureQueryDimension {
                    handle, component, ..
                } => {
                    if *component > 3 {
                        return Err(ComputeEmitError::UnsupportedOperation(format!(
                            "TextureQueryDimension component {component}"
                        )));
                    }
                    if sampled_resource(*handle).is_none() {
                        return Err(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: ComputeResourceKind::SampledImage,
                        });
                    }
                }
                IrOp::LocalInvocationId { component } | IrOp::WorkgroupId { component }
                    if *component > 2 =>
                {
                    return Err(ComputeEmitError::UnsupportedOperation(format!(
                        "compute builtin component {component}"
                    )));
                }
                IrOp::LoadShared { .. }
                | IrOp::StoreShared { .. }
                | IrOp::SharedAtomic { .. }
                    if options.shared_memory_size == 0 =>
                {
                    return Err(ComputeEmitError::MissingSharedMemory);
                }
                IrOp::LoadShared { .. } if instruction.pred.is_some() => {
                    return Err(ComputeEmitError::UnsupportedOperation(
                        "predicated LoadShared".to_owned(),
                    ));
                }
                IrOp::SharedAtomic { op, .. } if *op != ImageAtomicOp::Or => {
                    return Err(ComputeEmitError::UnsupportedOperation(format!(
                        "unsupported shared atomic operation {op:?}"
                    )));
                }
                IrOp::WorkgroupBarrier | IrOp::MemoryBarrier { .. }
                    if instruction.pred.is_some() =>
                {
                    return Err(ComputeEmitError::UnsupportedOperation(
                        "predicated compute barrier".to_owned(),
                    ));
                }
                IrOp::ImageWrite {
                    handle,
                    dimension,
                    y,
                    z,
                    ..
                } => {
                    let expected_kind = if *dimension == ImageDimension::Buffer {
                        ComputeResourceKind::StorageTexelBuffer
                    } else {
                        ComputeResourceKind::StorageImage
                    };
                    let resource =
                        storage_resource(*handle).ok_or(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: expected_kind,
                        })?;
                    if resource.kind != expected_kind {
                        return Err(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: expected_kind,
                        });
                    }
                    if *dimension != resource.dimension {
                        return Err(ComputeEmitError::DimensionMismatch {
                            handle: *handle,
                            instruction: *dimension,
                            actual: resource.dimension,
                        });
                    }
                    validate_compute_coordinates(
                        *handle,
                        resource.dimension,
                        y.as_ref(),
                        z.as_ref(),
                    )?;
                }
                IrOp::ImageAtomic {
                    handle,
                    dimension,
                    y,
                    z,
                    ..
                } => {
                    let resource =
                        atomic_resource(*handle).ok_or(ComputeEmitError::MissingResource {
                            handle: *handle,
                            kind: ComputeResourceKind::StorageTexelBuffer,
                        })?;
                    if resource.numeric_type != TextureNumericType::Uint {
                        return Err(ComputeEmitError::UnsupportedOperation(format!(
                            "image atomic storage texel buffer {:?} uses {:?} instead of Uint",
                            resource.handle, resource.numeric_type
                        )));
                    }
                    if !resource
                        .texel_format
                        .is_some_and(ComputeTexelFormat::supports_storage_atomics)
                    {
                        return Err(ComputeEmitError::UnsupportedOperation(format!(
                            "image atomic storage texel buffer {:?} is not R32Uint",
                            resource.handle
                        )));
                    }
                    if *dimension != resource.dimension {
                        return Err(ComputeEmitError::DimensionMismatch {
                            handle: *handle,
                            instruction: *dimension,
                            actual: resource.dimension,
                        });
                    }
                    if y.is_some() || z.is_some() {
                        return Err(ComputeEmitError::UnsupportedOperation(
                            "buffer ImageAtomic with non-X coordinates".to_owned(),
                        ));
                    }
                }
                IrOp::Unimplemented { .. } => {
                    return Err(ComputeEmitError::UnimplementedIr(1));
                }
                IrOp::SampleTex { .. }
                | IrOp::GatherTex { .. }
                | IrOp::TexelFetch { .. }
                | IrOp::LoadGlobal { .. }
                | IrOp::LoadStorage { .. }
                | IrOp::LoadAttr { .. }
                | IrOp::StoreAttr { .. }
                | IrOp::InterpAttr { .. }
                | IrOp::Kill => {
                    return Err(ComputeEmitError::UnsupportedOperation(
                        match &instruction.op {
                            IrOp::SampleTex { .. } => "SampleTex",
                            IrOp::GatherTex { .. } => "GatherTex",
                            IrOp::TexelFetch { .. } => "legacy TexelFetch",
                            IrOp::LoadGlobal { .. } => "LoadGlobal",
                            IrOp::LoadStorage { .. } => "LoadStorage",
                            IrOp::LoadAttr { .. } => "LoadAttr",
                            IrOp::StoreAttr { .. } => "StoreAttr",
                            IrOp::InterpAttr { .. } => "InterpAttr",
                            IrOp::Kill => "Kill",
                            _ => unreachable!(),
                        }
                        .to_owned(),
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(cbuf_required_sizes)
}

pub fn emit_compute(
    cfg: &Cfg,
    options: &ComputeOptions,
) -> Result<ComputeModule, ComputeEmitError> {
    let cbuf_required_sizes = validate_compute_options(cfg, options)?;

    let mut emitter = Emitter::new(Stage::Compute);
    emitter.compute_options = Some(options.clone());
    let (words, spirv_cbuf_bindings, _, _) = emitter.finish_inner(cfg, &[]);

    let mut descriptors =
        Vec::with_capacity(options.resources.len() + spirv_cbuf_bindings.count_ones() as usize);
    for slot in 0..COMPUTE_CBUF_SLOTS as u8 {
        if spirv_cbuf_bindings & (1 << slot) == 0 {
            continue;
        }
        descriptors.push(ComputeDescriptor {
            binding: compute_cbuf_descriptor_binding(slot).unwrap(),
            kind: ComputeDescriptorKind::UniformBuffer,
            handle: None,
            dimension: None,
            numeric_type: None,
            texel_format: None,
        });
    }
    descriptors.extend(options.resources.iter().map(|resource| ComputeDescriptor {
        binding: resource.binding,
        kind: match resource.kind {
            ComputeResourceKind::CombinedSampledImage => {
                ComputeDescriptorKind::CombinedSampledImage
            }
            ComputeResourceKind::SampledImage => ComputeDescriptorKind::SampledImage,
            ComputeResourceKind::UniformTexelBuffer => ComputeDescriptorKind::UniformTexelBuffer,
            ComputeResourceKind::StorageTexelBuffer => ComputeDescriptorKind::StorageTexelBuffer,
            ComputeResourceKind::StorageImage => ComputeDescriptorKind::StorageImage,
        },
        handle: Some(resource.handle),
        dimension: Some(resource.dimension),
        numeric_type: Some(resource.numeric_type),
        texel_format: resource.texel_format,
    }));
    descriptors.sort_unstable_by_key(|descriptor| descriptor.binding);

    let mut cbuf_bindings = spirv_cbuf_bindings;
    for resource in &options.resources {
        let binding = match resource.handle {
            TextureHandleOrigin::Bound { .. } => options.texture_bound_cbuf,
            TextureHandleOrigin::Bindless { cbuf_binding, .. } => cbuf_binding,
        };
        cbuf_bindings |= 1u32 << binding;
    }

    Ok(ComputeModule {
        words,
        cbuf_bindings,
        cbuf_size: COMPUTE_CBUF_SIZE,
        cbuf_required_sizes,
        texture_bound_cbuf: options.texture_bound_cbuf,
        descriptors,
    })
}

pub fn emit_vertex_with_bindings(cfg: &Cfg, required_outputs: &[u32]) -> (Vec<u32>, u32) {
    Emitter::new(Stage::Vertex).finish_with_required_outputs_and_bindings(cfg, required_outputs)
}

#[derive(Clone, Debug)]
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
    pub texture_numeric_manifest: Vec<GraphicsTextureResource>,
    pub texel_buffer_mask: u32,
    pub layer_output_slot: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FragmentOptions {
    pub uint_output_mask: u32,
    pub sint_output_mask: u32,
    pub texture_numeric_manifest: Vec<GraphicsTextureResource>,
    pub texel_buffer_mask: u32,
    pub y_negate: bool,
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
            texture_numeric_manifest: Vec::new(),
            texel_buffer_mask: 0,
            layer_output_slot: None,
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
    emit_fragment_full_with_alpha_test(
        cfg,
        ps_input_map,
        color_outputs,
        output_map,
        debug_active,
        0,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_fragment_full_with_alpha_test(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
    color_outputs: u32,
    output_map: u32,
    debug_active: bool,
    alpha_test_func: u32,
    alpha_test_ref: u32,
) -> (Vec<u32>, u32, Vec<u32>, u32, bool) {
    emit_fragment_full_with_options(
        cfg,
        ps_input_map,
        color_outputs,
        output_map,
        debug_active,
        alpha_test_func,
        alpha_test_ref,
        FragmentOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_fragment_full_with_options(
    cfg: &Cfg,
    ps_input_map: [u8; 32],
    color_outputs: u32,
    output_map: u32,
    debug_active: bool,
    alpha_test_func: u32,
    alpha_test_ref: u32,
    options: FragmentOptions,
) -> (Vec<u32>, u32, Vec<u32>, u32, bool) {
    let vec4s = cbuf_vec4s(cfg, 1).max(UBO_VEC4S);
    let mut emitter = Emitter::new_sized(Stage::Fragment, vec4s);
    emitter.texture_numeric_manifest = normalize_graphics_texture_manifest(
        options.texture_numeric_manifest,
    )
    .unwrap_or_else(|error| panic!("nexium-spirv: invalid graphics texture manifest: {error}"));
    if !debug_active {
        emitter.sample_debug_slot = None;
        emitter.texcoord_debug_slot = None;
        emitter.tex_v_flip_slots.clear();
    }
    emitter.ps_input_map = ps_input_map;
    emitter.fragment_color_outputs = color_outputs.max(1).min(8);
    emitter.fragment_output_map = output_map;
    emitter.fragment_uint_output_mask = options.uint_output_mask;
    emitter.fragment_sint_output_mask = options.sint_output_mask & !options.uint_output_mask;
    emitter.texel_buffer_mask = options.texel_buffer_mask;
    emitter.alpha_test_func = alpha_test_func;
    emitter.alpha_test_ref = alpha_test_ref;
    emitter.y_negate = options.y_negate;
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
    fn enc_static_ldc(dest: u8, binding: u8, byte_offset: u32) -> u64 {
        (0xEF94u64 << 48)
            | (u64::from(binding) << 36)
            | (u64::from(byte_offset & 0xFFFF) << 20)
            | (7u64 << 16)
            | (0xFFu64 << 8)
            | u64::from(dest)
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

    fn indirect_test_program() -> (Vec<u8>, Vec<u32>) {
        let targets = vec![0x30u32, 0x38, 0x48, 0x50, 0x58];
        let mut bytes = vec![0u8; 0x60];
        let words = [
            (0x08usize, 0x3820_0380_0047_0000u64),
            (0x10, 0x3848_0000_0027_0000u64),
            (0x18, 0xEF94_0010_0007_0000u64),
            (0x28, 0xE250_0FFF_FD07_000Fu64),
        ];
        for (offset, word) in words {
            bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
        }
        for &target in &targets {
            bytes[target as usize..target as usize + 8].copy_from_slice(&enc_exit().to_le_bytes());
        }
        (bytes, targets)
    }

    fn cfg_block(id: BlockId, branch: BranchKind, program: nexium_shader::IrProgram) -> BasicBlock {
        BasicBlock {
            id,
            start_offset: id as usize * 8,
            end_offset: id as usize * 8 + 8,
            branch,
            program,
            reg_exit: std::collections::HashMap::new(),
            pred_phis: Vec::new(),
            pred_exit: std::collections::HashMap::new(),
        }
    }

    fn empty_cfg_block(id: BlockId, branch: BranchKind) -> BasicBlock {
        cfg_block(id, branch, nexium_shader::IrProgram::new())
    }

    fn always_pred() -> nexium_shader::Predicate {
        nexium_shader::Predicate {
            idx: 7,
            negate: false,
        }
    }

    fn unconditional_loop_before_phi_cfg() -> Cfg {
        let mut phi_program = nexium_shader::IrProgram::new();
        phi_program.emit(
            IrOp::Phi {
                sources: vec![(0, IrValue::ImmF32(1.0)), (2, IrValue::ImmF32(2.0))],
            },
            Some(0),
        );
        Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Unconditional { target: 1 }),
                cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                    phi_program,
                ),
                empty_cfg_block(3, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn single_conditional_loop_cfg() -> Cfg {
        Cfg {
            blocks: vec![
                empty_cfg_block(0, BranchKind::FallThrough),
                empty_cfg_block(
                    1,
                    BranchKind::Conditional {
                        target: 1,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(2, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn psetp_program(
        offset: u32,
        dest_p: u8,
        dest_np: u8,
        guard: Option<nexium_shader::Predicate>,
    ) -> (nexium_shader::IrProgram, ValueId) {
        let mut program = nexium_shader::IrProgram::with_offset(offset);
        let result = program.emit_pred(
            IrOp::PSetPred {
                dest_p,
                dest_np,
                pred_a: 7,
                neg_pred_a: false,
                pred_b: 7,
                neg_pred_b: false,
                pred_c: 7,
                neg_pred_c: false,
                bop_1: BoolOp::And,
                bop_2: BoolOp::And,
            },
            None,
            guard,
        );
        (program, result)
    }

    fn predicate_state_diamond_cfg() -> Cfg {
        let (entry_program, entry_pred) = psetp_program(0, 0, 7, None);
        let (first_arm_program, first_arm_pred) = psetp_program(1, 0, 7, None);
        let (second_arm_program, second_arm_pred) = psetp_program(2, 0, 7, Some(always_pred()));
        let merge_pred = ValueId(3);
        let mut entry = cfg_block(
            0,
            BranchKind::Conditional {
                target: 2,
                pred: always_pred(),
            },
            entry_program,
        );
        entry.pred_exit.insert(0, entry_pred);
        let mut first_arm = cfg_block(
            1,
            BranchKind::Unconditional { target: 3 },
            first_arm_program,
        );
        first_arm.pred_exit.insert(0, first_arm_pred);
        let mut second_arm = cfg_block(2, BranchKind::FallThrough, second_arm_program);
        second_arm.pred_exit.insert(0, second_arm_pred);
        let mut merge = empty_cfg_block(3, BranchKind::Exit);
        merge.pred_phis.push(nexium_shader::cfg::PredPhi {
            pred: 0,
            result: merge_pred,
            sources: vec![(1, Some(first_arm_pred)), (2, Some(second_arm_pred))],
        });
        merge.pred_exit.insert(0, merge_pred);
        Cfg {
            blocks: vec![entry, first_arm, second_arm, merge],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn inverse_predicate_loop_cfg() -> Cfg {
        let (preheader_program, preheader_pred) = psetp_program(0, 7, 1, None);
        let header_pred = ValueId(1);
        let (latch_program, latch_pred) = psetp_program(2, 7, 1, None);
        let mut preheader = cfg_block(0, BranchKind::FallThrough, preheader_program);
        preheader.pred_exit.insert(1, preheader_pred);
        let mut header = cfg_block(
            1,
            BranchKind::FallThrough,
            nexium_shader::IrProgram::with_offset(2),
        );
        header.pred_phis.push(nexium_shader::cfg::PredPhi {
            pred: 1,
            result: header_pred,
            sources: vec![(0, Some(preheader_pred)), (2, Some(latch_pred))],
        });
        header.pred_exit.insert(1, header_pred);
        let mut latch = cfg_block(
            2,
            BranchKind::Conditional {
                target: 1,
                pred: always_pred(),
            },
            latch_program,
        );
        latch.pred_exit.insert(1, latch_pred);
        let mut exit = empty_cfg_block(3, BranchKind::Exit);
        exit.pred_exit.insert(1, latch_pred);
        Cfg {
            blocks: vec![preheader, header, latch, exit],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn unconditional_loop_with_inner_selection_cfg() -> Cfg {
        Cfg {
            blocks: vec![
                empty_cfg_block(0, BranchKind::FallThrough),
                empty_cfg_block(
                    1,
                    BranchKind::Conditional {
                        target: 3,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(2, BranchKind::Unconditional { target: 4 }),
                empty_cfg_block(3, BranchKind::FallThrough),
                empty_cfg_block(4, BranchKind::Unconditional { target: 1 }),
                empty_cfg_block(5, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn conditional_loop_with_break_and_cleanup_cfg() -> Cfg {
        let mut merge_program = nexium_shader::IrProgram::new();
        merge_program.emit(
            IrOp::Phi {
                sources: vec![(1, IrValue::ImmF32(1.0)), (4, IrValue::ImmF32(2.0))],
            },
            Some(0),
        );
        Cfg {
            blocks: vec![
                empty_cfg_block(0, BranchKind::FallThrough),
                empty_cfg_block(
                    1,
                    BranchKind::Conditional {
                        target: 5,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 1,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::FallThrough),
                empty_cfg_block(4, BranchKind::FallThrough),
                cfg_block(5, BranchKind::Exit, merge_program),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn shared_selection_loop_break_phi_cfg() -> Cfg {
        let mut merge_program = nexium_shader::IrProgram::new();
        merge_program.emit(
            IrOp::Phi {
                sources: vec![
                    (0, IrValue::ImmF32(1.0)),
                    (2, IrValue::ImmF32(2.0)),
                    (4, IrValue::ImmF32(3.0)),
                ],
            },
            Some(0),
        );
        Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 5,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::FallThrough),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 5,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(
                    3,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(4, BranchKind::FallThrough),
                cfg_block(5, BranchKind::Exit, merge_program),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
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

    fn validates_with_spirv_val_if_available(words: &[u32]) {
        static NEXT_VALIDATION_FILE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let serial = NEXT_VALIDATION_FILE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "nexium-spirv-validation-{}-{serial}.spv",
            std::process::id()
        ));
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(&path, bytes).expect("write temporary SPIR-V");
        let output = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.1"])
            .arg(&path)
            .output();
        let _ = std::fs::remove_file(&path);
        match output {
            Ok(output) => assert!(
                output.status.success(),
                "spirv-val failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("failed to run spirv-val: {error}"),
        }
    }

    fn assert_graphics_cbuf_sample_operand(
        module: &rspirv::dr::Module,
        sample_operand: Word,
        logical_binding: u32,
        byte_offset: u32,
    ) {
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let definitions = module
            .types_global_values
            .iter()
            .chain(instructions.iter().copied())
            .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
            .collect::<HashMap<_, _>>();
        let constant = |id: Word| {
            definitions.get(&id).and_then(|instruction| {
                (instruction.class.opcode == rspirv::spirv::Op::Constant)
                    .then(|| match instruction.operands.as_slice() {
                        [Operand::LiteralBit32(value)] => Some(*value),
                        _ => None,
                    })
                    .flatten()
            })
        };
        let id_operand = |instruction: &rspirv::dr::Instruction, index: usize| {
            match instruction.operands.get(index) {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            }
        };
        let cbuf_var = module
            .annotations
            .iter()
            .find_map(|instruction| match instruction.operands.as_slice() {
                [
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(binding),
                ] if *binding == GFX_BINDING_CBUF => Some(*target),
                _ => None,
            })
            .expect("graphics cbuf descriptor variable");

        let access_index = |pointer: Word| {
            let access = definitions
                .get(&pointer)
                .copied()
                .expect("cbuf load pointer definition");
            assert_eq!(access.class.opcode, rspirv::spirv::Op::AccessChain);
            assert_eq!(
                id_operand(access, 0),
                Some(cbuf_var),
                "cbuf access must originate at graphics descriptor binding zero"
            );
            let zero = id_operand(access, 1).expect("cbuf struct member index");
            assert_eq!(constant(zero), Some(0));
            id_operand(access, 2).expect("cbuf word index")
        };
        let directory_load = |value: Word, expected_index: u32| {
            let load = definitions
                .get(&value)
                .copied()
                .expect("cbuf directory load definition");
            assert_eq!(load.class.opcode, rspirv::spirv::Op::Load);
            let pointer = id_operand(load, 0).expect("cbuf directory load pointer");
            let index = access_index(pointer);
            assert_eq!(
                constant(index),
                Some(expected_index),
                "cbuf directory entry"
            );
        };

        let mut value = sample_operand;
        let mut visited = std::collections::HashSet::new();
        let payload_load = loop {
            assert!(visited.insert(value), "cycle while tracing sample operand");
            let definition = definitions
                .get(&value)
                .copied()
                .expect("sample operand definition");
            match definition.class.opcode {
                rspirv::spirv::Op::Bitcast | rspirv::spirv::Op::CopyObject => {
                    value = id_operand(definition, 0).expect("value-preserving source")
                }
                rspirv::spirv::Op::Load => break definition,
                opcode => panic!("unexpected sample operand producer {opcode:?}"),
            }
        };

        let payload_pointer = id_operand(payload_load, 0).expect("cbuf payload load pointer");
        let safe_index = access_index(payload_pointer);
        let select = definitions
            .get(&safe_index)
            .copied()
            .expect("cbuf bounds-selected index definition");
        assert_eq!(select.class.opcode, rspirv::spirv::Op::Select);
        let condition = id_operand(select, 0).expect("cbuf bounds condition");
        let payload_index = id_operand(select, 1).expect("cbuf payload index");
        let sentinel = id_operand(select, 2).expect("cbuf sentinel index");
        assert_eq!(constant(sentinel), Some(GFX_CBUF_ZERO_WORD));

        let payload_add = definitions
            .get(&payload_index)
            .copied()
            .expect("cbuf payload address definition");
        assert_eq!(payload_add.class.opcode, rspirv::spirv::Op::IAdd);
        let base = id_operand(payload_add, 0).expect("cbuf payload base");
        let word_offset = id_operand(payload_add, 1).expect("cbuf payload word offset");
        directory_load(base, logical_binding * 2);

        let shift = definitions
            .get(&word_offset)
            .copied()
            .expect("cbuf byte-to-word conversion");
        assert_eq!(
            shift.class.opcode,
            rspirv::spirv::Op::ShiftRightLogical
        );
        let effective_byte = id_operand(shift, 0).expect("cbuf effective byte address");
        let shift_amount = id_operand(shift, 1).expect("cbuf word shift amount");
        assert_eq!(constant(effective_byte), Some(byte_offset));
        assert_eq!(constant(shift_amount), Some(2));

        let bounds = definitions
            .get(&condition)
            .copied()
            .expect("cbuf bounds-check definition");
        assert_eq!(bounds.class.opcode, rspirv::spirv::Op::ULessThan);
        assert_eq!(id_operand(bounds, 0), Some(word_offset));
        let count = id_operand(bounds, 1).expect("cbuf directory word count");
        directory_load(count, logical_binding * 2 + 1);
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
    fn partial_vertex_position_starts_from_zero_zero_zero_one() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit_void(IrOp::StoreAttr {
            slot: 0x70,
            src: IrValue::ImmF32(0.25),
        });
        program.emit_void(IrOp::StoreAttr {
            slot: 0x74,
            src: IrValue::ImmF32(-0.5),
        });
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let words = emit_vertex(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let position = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(id),
                    Operand::Decoration(Decoration::BuiltIn),
                    Operand::BuiltIn(BuiltIn::Position),
                ] => Some(*id),
                _ => None,
            })
            .expect("position output");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let initial_store = instructions
            .iter()
            .copied()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Store
                    && inst.operands.first() == Some(&Operand::IdRef(position))
            })
            .expect("direct position initialization");
        let Operand::IdRef(initial_value) = initial_store.operands[1] else {
            panic!("position initialization value");
        };
        let composite = instructions
            .iter()
            .copied()
            .find(|inst| inst.result_id == Some(initial_value))
            .expect("position initialization composite");
        assert_eq!(
            composite.class.opcode,
            rspirv::spirv::Op::CompositeConstruct
        );
        let component_bits = composite
            .operands
            .iter()
            .map(|operand| {
                let Operand::IdRef(id) = operand else {
                    panic!("position component id");
                };
                module
                    .types_global_values
                    .iter()
                    .find_map(|inst| {
                        (inst.result_id == Some(*id)
                            && inst.class.opcode == rspirv::spirv::Op::Constant)
                            .then(|| match inst.operands.as_slice() {
                                [Operand::LiteralBit32(bits)] => *bits,
                                _ => panic!("f32 constant bits"),
                            })
                    })
                    .expect("position component constant")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            component_bits,
            vec![
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits()
            ]
        );
    }

    #[test]
    fn vertex_instance_id_subtracts_base_instance() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(IrOp::LoadAttr { slot: 0x2f8 }, Some(0));
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let words = emit_vertex(&cfg);
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");

        assert!(module
            .capabilities
            .iter()
            .any(|inst| { inst.operands == [Operand::Capability(Capability::DrawParameters)] }));
        assert!(module.extensions.iter().any(|inst| {
            inst.operands
                == [Operand::LiteralString(
                    "SPV_KHR_shader_draw_parameters".to_string(),
                )]
        }));

        let builtin_var = |builtin| {
            module.annotations.iter().find_map(|inst| {
                match inst.operands.as_slice() {
                    [
                        Operand::IdRef(var),
                        Operand::Decoration(Decoration::BuiltIn),
                        Operand::BuiltIn(found),
                    ] if *found == builtin => Some(*var),
                    _ => None,
                }
            })
        };
        let instance_var = builtin_var(BuiltIn::InstanceIndex).expect("InstanceIndex input");
        let base_var = builtin_var(BuiltIn::BaseInstance).expect("BaseInstance input");
        let entry = &module.entry_points[0];
        assert!(entry.operands[3..].contains(&Operand::IdRef(instance_var)));
        assert!(entry.operands[3..].contains(&Operand::IdRef(base_var)));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let loaded_from = |var| {
            instructions.iter().find_map(|inst| {
                if inst.class.opcode == rspirv::spirv::Op::Load
                    && inst.operands.first() == Some(&Operand::IdRef(var))
                {
                    inst.result_id
                } else {
                    None
                }
            })
        };
        let instance = loaded_from(instance_var).expect("InstanceIndex load");
        let base = loaded_from(base_var).expect("BaseInstance load");
        let difference = instructions
            .iter()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::ISub
                    && inst.operands == [Operand::IdRef(instance), Operand::IdRef(base)]
            })
            .and_then(|inst| inst.result_id)
            .expect("InstanceIndex - BaseInstance");
        assert!(instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Bitcast
                && inst.operands == [Operand::IdRef(difference)]
        }));
    }

    #[test]
    fn vertex_layer_output_mirrors_selected_generic_bits() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit_void(IrOp::StoreAttr {
            slot: 0x90,
            src: IrValue::ImmU32(7),
        });
        program.emit_void(IrOp::StoreAttr {
            slot: 0x94,
            src: IrValue::ImmU32(11),
        });
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _) = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                layer_output_slot: Some(0x90),
                ..VertexOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");

        assert!(module.capabilities.iter().any(|inst| {
            inst.operands == [Operand::Capability(Capability::ShaderViewportIndexLayerEXT)]
        }));
        assert!(module.extensions.iter().any(|inst| {
            inst.operands
                == [Operand::LiteralString(
                    "SPV_EXT_shader_viewport_index_layer".to_string(),
                )]
        }));

        let layer = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(var),
                    Operand::Decoration(Decoration::BuiltIn),
                    Operand::BuiltIn(BuiltIn::Layer),
                ] => Some(*var),
                _ => None,
            })
            .expect("Layer output");
        let generic = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(var),
                    Operand::Decoration(Decoration::Location),
                    Operand::LiteralBit32(1),
                ] => Some(*var),
                _ => None,
            })
            .expect("generic output at location 1");
        let entry = &module.entry_points[0];
        assert!(entry.operands[3..].contains(&Operand::IdRef(layer)));
        assert!(entry.operands[3..].contains(&Operand::IdRef(generic)));

        let u32_type = module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::TypeInt
                    && inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)])
                .then_some(inst.result_id)
                .flatten()
            })
            .expect("u32 type");
        let f32_type = module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::TypeFloat
                    && inst.operands == [Operand::LiteralBit32(32)])
                .then_some(inst.result_id)
                .flatten()
            })
            .expect("f32 type");
        let seven_bits = module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::Constant
                    && inst.result_type == Some(f32_type)
                    && inst.operands == [Operand::LiteralBit32(7)])
                .then_some(inst.result_id)
                .flatten()
            })
            .expect("source bits");
        let zero = module
            .types_global_values
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::Constant
                    && inst.result_type == Some(u32_type)
                    && inst.operands == [Operand::LiteralBit32(0)])
                .then_some(inst.result_id)
                .flatten()
            })
            .expect("u32 zero");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let layer_bits = instructions
            .iter()
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::Bitcast
                    && inst.result_type == Some(u32_type)
                    && inst.operands == [Operand::IdRef(seven_bits)])
                .then_some(inst.result_id)
                .flatten()
            })
            .expect("layer bitcast");
        assert!(instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Store
                && inst.operands == [Operand::IdRef(layer), Operand::IdRef(zero)]
        }));
        assert!(instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Store
                && inst.operands == [Operand::IdRef(layer), Operand::IdRef(layer_bits)]
        }));
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
    fn indirect_branch_switch_passes_validation() {
        let (bytes, targets) = indirect_test_program();
        let cfg = nexium_shader::build_cfg_with_cbuf(&bytes, |binding, offset| {
            (binding == 1)
                .then(|| targets.get((offset / 4) as usize).copied())
                .flatten()
        });
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        assert!(validate_structured_cfg(&words).is_ok());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let opcodes = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|instruction| instruction.class.opcode)
            .collect::<Vec<_>>();
        assert!(opcodes.contains(&rspirv::spirv::Op::SelectionMerge));
        assert!(opcodes.contains(&rspirv::spirv::Op::Switch));
        assert!(opcodes.contains(&rspirv::spirv::Op::Unreachable));
    }

    #[test]
    fn indirect_branch_structures_shared_case_join_and_nested_termination() {
        let mut targets = [nexium_shader::IndirectBranchTarget::default();
            nexium_shader::MAX_INDIRECT_BRANCH_TARGETS];
        for (entry, (selector, target)) in
            targets
                .iter_mut()
                .zip([(10, 2), (20, 3), (30, 4), (40, 5), (50, 6)])
        {
            *entry = nexium_shader::IndirectBranchTarget { selector, target };
        }
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 8,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(
                    1,
                    BranchKind::Indirect {
                        register: 0,
                        base: 0,
                        cbuf_binding: 1,
                        cbuf_offset: 0,
                        table_entries: 5,
                        count: 5,
                        targets,
                    },
                ),
                empty_cfg_block(2, BranchKind::Unconditional { target: 8 }),
                empty_cfg_block(3, BranchKind::Unconditional { target: 8 }),
                empty_cfg_block(4, BranchKind::Unconditional { target: 8 }),
                empty_cfg_block(5, BranchKind::Unconditional { target: 7 }),
                empty_cfg_block(6, BranchKind::Unconditional { target: 7 }),
                empty_cfg_block(7, BranchKind::Unconditional { target: 8 }),
                empty_cfg_block(8, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let switches = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::Switch)
            .collect::<Vec<_>>();
        assert_eq!(switches.len(), 2);
        let mut operand_counts = switches
            .iter()
            .map(|instruction| instruction.operands.len())
            .collect::<Vec<_>>();
        operand_counts.sort_unstable();
        assert_eq!(operand_counts, [6, 12]);
        let outer = switches
            .iter()
            .find(|instruction| instruction.operands.len() == 12)
            .expect("outer indirect switch");
        assert_eq!(outer.operands[9], outer.operands[11]);
        assert!(module.functions[0].blocks.iter().any(|block| {
            block.instructions.len() == 1
                && block.instructions[0].class.opcode == rspirv::spirv::Op::Unreachable
        }));
    }

    #[test]
    fn vertex_with_alu_passes_naga_validation() {
        let bytes = build_test_program(&[enc_fmul_reg(2, 0, 1), enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let words = emit_vertex(&cfg);
        validates_with_naga(&words);
    }

    #[test]
    fn pps_b64_local_pair_uses_two_private_words() {
        let bytes = build_test_program(&[0xef55_0000_0007_ff24, 0xef45_1000_0007_ff24, enc_exit()]);
        let cfg = nexium_shader::build_fragment_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let private_words = module
            .types_global_values
            .iter()
            .filter_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::Variable
                    && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Private)))
                .then_some(inst.result_id.unwrap())
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(private_words.len(), 1);

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let variable = *private_words.first().expect("private local-memory array");
        let local_ptrs = instructions
            .iter()
            .filter_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::AccessChain
                    && inst.operands.first() == Some(&Operand::IdRef(variable)))
                .then(|| inst.result_id.expect("access-chain result"))
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            local_ptrs.len() >= 4,
            "two word stores and loads need four accesses"
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::Store
                        && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if local_ptrs.contains(id))
                })
                .count(),
            2
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::Load
                        && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if local_ptrs.contains(id))
                })
                .count(),
            4
        );
    }

    #[test]
    fn sibling_predicate_state_is_restored_before_guarded_write() {
        let words = emit_fragment(&predicate_state_diamond_cfg());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let guarded_select = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Select
                    && inst.result_type
                        == Some(
                            module
                                .types_global_values
                                .iter()
                                .find_map(|ty| {
                                    (ty.class.opcode == rspirv::spirv::Op::TypeBool)
                                        .then_some(ty.result_id?)
                                })
                                .unwrap(),
                        )
            })
            .expect("guarded predicate write");
        let Operand::IdRef(old_predicate) = guarded_select.operands[2] else {
            panic!("guarded predicate write must select an old predicate id");
        };
        let defining_block = function
            .blocks
            .iter()
            .position(|block| {
                block
                    .instructions
                    .iter()
                    .any(|inst| inst.result_id == Some(old_predicate))
            })
            .expect("old predicate definition");
        assert_eq!(
            defining_block, 0,
            "the second arm must inherit P0 from the entry, not its sibling"
        );
    }

    #[test]
    fn loop_carried_inverse_predicate_uses_inverse_result() {
        let words = emit_fragment(&inverse_predicate_loop_cfg());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let continue_label = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .and_then(|inst| match inst.operands.get(1) {
                Some(Operand::IdRef(label)) => Some(*label),
                _ => None,
            })
            .expect("loop continue label");
        let carried_value = function
            .blocks
            .iter()
            .find(|block| {
                block.label.as_ref().and_then(|label| label.result_id) == Some(continue_label)
            })
            .and_then(|block| {
                block.instructions.iter().find_map(|inst| {
                    (inst.class.opcode == rspirv::spirv::Op::CopyObject)
                        .then(|| match inst.operands.first() {
                            Some(Operand::IdRef(value)) => Some(*value),
                            _ => None,
                        })
                        .flatten()
                })
            })
            .expect("loop-carried predicate value");
        let definitions = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| inst.result_id.map(|id| (id, inst)))
            .collect::<HashMap<_, _>>();
        fn depends_on_logical_not(
            id: Word,
            definitions: &HashMap<Word, &rspirv::dr::Instruction>,
            visited: &mut std::collections::HashSet<Word>,
        ) -> bool {
            if !visited.insert(id) {
                return false;
            }
            let Some(inst) = definitions.get(&id) else {
                return false;
            };
            inst.class.opcode == rspirv::spirv::Op::LogicalNot
                || inst.operands.iter().any(|operand| match operand {
                    Operand::IdRef(input) => depends_on_logical_not(*input, definitions, visited),
                    _ => false,
                })
        }
        assert!(
            depends_on_logical_not(
                carried_value,
                &definitions,
                &mut std::collections::HashSet::new()
            ),
            "the loop carry must copy PSetPred's inverse output"
        );
    }

    #[test]
    fn structured_cfg_verifier_rejects_backedge_without_loop_header() {
        let words = emit_fragment(&predicate_state_diamond_cfg());
        assert!(validate_structured_cfg(&words).is_ok());
        let mut module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let entry = module.functions[0].blocks[0]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
            .unwrap();
        let branch = module.functions[0]
            .blocks
            .iter_mut()
            .skip(1)
            .flat_map(|block| &mut block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::Branch)
            .expect("forward branch");
        branch.operands[0] = Operand::IdRef(entry);
        let error = validate_structured_cfg(&module.assemble()).expect_err("invalid backedge");
        assert!(error.contains("does not target a loop header"), "{error}");
    }

    #[test]
    fn alpha_test_kill_exit_is_a_valid_selection_termination() {
        let cfg = Cfg {
            blocks: vec![empty_cfg_block(0, BranchKind::Exit)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _, _, _) =
            emit_fragment_full_with_alpha_test(&cfg, [0; 32], 1, 0, false, 0x204, 0);
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let header = function
            .blocks
            .iter()
            .find(|block| {
                block
                    .instructions
                    .iter()
                    .any(|inst| inst.class.opcode == rspirv::spirv::Op::SelectionMerge)
            })
            .expect("alpha-test selection header");
        let branch = header.instructions.last().expect("selection branch");
        assert_eq!(branch.class.opcode, rspirv::spirv::Op::BranchConditional);
        let targets = structured_branch_targets(branch);
        assert!(targets.iter().any(|target| {
            function.blocks.iter().any(|block| {
                block.label.as_ref().and_then(|label| label.result_id) == Some(*target)
                    && block.instructions.last().is_some_and(|terminator| {
                        terminator.class.opcode == rspirv::spirv::Op::Kill
                    })
            })
        }));
    }

    #[test]
    fn shared_virtual_exit_merges_are_unique() {
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Exit),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::Exit),
                empty_cfg_block(4, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let merges = structurizer_cond_merges(&cfg).expect("conditional merge map");
        assert_eq!(merges[0], 5);
        assert_eq!(merges[2], 5);

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let merges = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter_map(|inst| match (&inst.class.opcode, inst.operands.first()) {
                (rspirv::spirv::Op::SelectionMerge, Some(Operand::IdRef(merge))) => Some(*merge),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(merges.len(), 2);
        assert_ne!(merges[0], merges[1]);
    }

    #[test]
    fn shared_virtual_exit_final_fallthrough_uses_private_merge() {
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Exit),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::Exit),
                empty_cfg_block(4, BranchKind::FallThrough),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let merges = structurizer_cond_merges(&cfg).expect("conditional merge map");
        assert_eq!(merges[0], 5);
        assert_eq!(merges[2], 5);

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let final_label = function.blocks[4]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
            .expect("final CFG block label");
        let inner_merge = function
            .blocks
            .iter()
            .find_map(|block| {
                let branch = block.instructions.last()?;
                (branch.class.opcode == rspirv::spirv::Op::BranchConditional
                    && structured_branch_targets(branch).contains(&final_label))
                .then(|| {
                    block.instructions.iter().find_map(|inst| {
                        match (inst.class.opcode, inst.operands.first()) {
                            (rspirv::spirv::Op::SelectionMerge, Some(Operand::IdRef(merge))) => {
                                Some(*merge)
                            }
                            _ => None,
                        }
                    })
                })
                .flatten()
            })
            .expect("inner private merge");
        let final_branch = function.blocks[4]
            .instructions
            .last()
            .expect("final CFG terminator");
        assert_eq!(
            (final_branch.class.opcode, final_branch.operands.as_slice()),
            (
                rspirv::spirv::Op::Branch,
                [Operand::IdRef(inner_merge)].as_slice(),
            )
        );
    }

    #[test]
    fn bounded_unconditional_loop_gives_inner_selection_a_forward_postdominator() {
        let cfg = unconditional_loop_with_inner_selection_cfg();
        let merges = structurizer_cond_merges(&cfg).expect("conditional merge map");
        assert_eq!(merges[1], 4, "the inner diamond must merge at its latch");

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let loop_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .expect("loop merge");
        let (Operand::IdRef(loop_exit), Operand::IdRef(loop_continue)) =
            (&loop_merge.operands[0], &loop_merge.operands[1])
        else {
            panic!("loop targets must be ids");
        };
        let latch_label = function
            .blocks
            .iter()
            .find_map(|block| {
                let term = block.instructions.last()?;
                (term.class.opcode == rspirv::spirv::Op::BranchConditional
                    && term.operands.get(1) == Some(&Operand::IdRef(*loop_continue))
                    && term.operands.get(2) == Some(&Operand::IdRef(*loop_exit)))
                .then(|| block.label.as_ref()?.result_id)
                .flatten()
            })
            .expect("bounded latch");
        let selection_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::SelectionMerge)
            .expect("inner selection merge");
        assert_eq!(
            selection_merge.operands.first(),
            Some(&Operand::IdRef(latch_label))
        );
        assert!(function.blocks.iter().any(|block| {
            block.instructions.len() == 1
                && block.instructions[0].class.opcode == rspirv::spirv::Op::Unreachable
        }));
    }

    #[test]
    fn conditional_loop_break_uses_real_merge_and_keeps_cleanup_path() {
        let words = emit_fragment(&conditional_loop_with_break_and_cleanup_cfg());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let loop_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .expect("loop merge");
        let (Operand::IdRef(merge), Operand::IdRef(cont)) =
            (&loop_merge.operands[0], &loop_merge.operands[1])
        else {
            panic!("loop targets must be ids");
        };
        let merge_block = function
            .blocks
            .iter()
            .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(*merge))
            .expect("real merge block");
        assert!(merge_block
            .instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Phi));
        assert!(!function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::SelectionMerge));
        assert!(function.blocks.iter().any(|block| {
            block.instructions.last().is_some_and(|term| {
                term.class.opcode == rspirv::spirv::Op::BranchConditional
                    && term.operands.get(1) == Some(&Operand::IdRef(*merge))
            })
        }));
        let latch_exit = function
            .blocks
            .iter()
            .filter_map(|block| block.instructions.last())
            .find(|term| {
                term.class.opcode == rspirv::spirv::Op::BranchConditional
                    && term.operands.get(1) == Some(&Operand::IdRef(*cont))
            })
            .expect("loop latch");
        assert_ne!(latch_exit.operands.get(2), Some(&Operand::IdRef(*merge)));
    }

    #[test]
    fn shared_merge_phi_uses_emitted_predecessors() {
        let mut merge_program = nexium_shader::IrProgram::new();
        merge_program.emit(
            IrOp::Phi {
                sources: vec![
                    (1, IrValue::ImmF32(1.0)),
                    (2, IrValue::ImmF32(2.0)),
                    (3, IrValue::ImmF32(3.0)),
                ],
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Unconditional { target: 4 }),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::Unconditional { target: 4 }),
                cfg_block(4, BranchKind::Exit, merge_program),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let phi_count = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::Phi)
            .count();
        assert_eq!(phi_count, 2);
    }

    #[test]
    fn shared_selection_and_loop_exit_uses_private_loop_merge() {
        let mut merge_program = nexium_shader::IrProgram::new();
        merge_program.emit(
            IrOp::Phi {
                sources: vec![(0, IrValue::ImmF32(1.0)), (2, IrValue::ImmF32(2.0))],
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 3,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::FallThrough),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                cfg_block(3, BranchKind::Exit, merge_program),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let selection_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::SelectionMerge).then(|| {
                    match inst.operands[0] {
                        Operand::IdRef(label) => label,
                        _ => panic!("selection merge target must be an id"),
                    }
                })
            })
            .expect("selection merge");
        let (loop_merge, loop_continue) = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find_map(|inst| {
                (inst.class.opcode == rspirv::spirv::Op::LoopMerge).then(|| {
                    let Operand::IdRef(merge) = inst.operands[0] else {
                        panic!("loop merge target must be an id");
                    };
                    let Operand::IdRef(cont) = inst.operands[1] else {
                        panic!("loop continue target must be an id");
                    };
                    (merge, cont)
                })
            })
            .expect("loop merge");
        assert_ne!(selection_merge, loop_merge);
        let loop_merge_block = function
            .blocks
            .iter()
            .find(|block| {
                block.label.as_ref().and_then(|label| label.result_id) == Some(loop_merge)
            })
            .expect("private loop merge block");
        assert!(matches!(
            loop_merge_block.instructions.last().map(|inst| (&inst.class.opcode, &inst.operands[..])),
            Some((rspirv::spirv::Op::Branch, [Operand::IdRef(target)])) if *target == selection_merge
        ));
        assert!(function.blocks.iter().any(|block| {
            block.instructions.last().is_some_and(|inst| {
                if inst.class.opcode != rspirv::spirv::Op::BranchConditional {
                    return false;
                }
                let [_, Operand::IdRef(a), Operand::IdRef(b), ..] = &inst.operands[..] else {
                    return false;
                };
                (*a == loop_continue && *b == loop_merge)
                    || (*a == loop_merge && *b == loop_continue)
            })
        }));
    }

    #[test]
    fn shared_virtual_exit_unconditional_loop_uses_private_selection_merge() {
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Exit),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::Exit),
                empty_cfg_block(4, BranchKind::Unconditional { target: 4 }),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let merges = structurizer_cond_merges(&cfg).expect("conditional merge map");
        assert_eq!(merges[0], cfg.blocks.len() as BlockId);
        assert_eq!(merges[2], cfg.blocks.len() as BlockId);

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let (loop_header, loop_merge) = function
            .blocks
            .iter()
            .find_map(|block| {
                let loop_merge = block
                    .instructions
                    .iter()
                    .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)?;
                let Operand::IdRef(merge) = loop_merge.operands[0] else {
                    panic!("loop merge target must be an id");
                };
                Some((block.label.as_ref()?.result_id?, merge))
            })
            .expect("bounded unconditional loop");
        let selection_merge = function
            .blocks
            .iter()
            .find_map(|block| {
                let term = block.instructions.last()?;
                if term.class.opcode != rspirv::spirv::Op::BranchConditional
                    || !structured_branch_targets(term).contains(&loop_header)
                {
                    return None;
                }
                block.instructions.iter().find_map(|inst| {
                    match (inst.class.opcode, inst.operands.first()) {
                        (rspirv::spirv::Op::SelectionMerge, Some(Operand::IdRef(merge))) => {
                            Some(*merge)
                        }
                        _ => None,
                    }
                })
            })
            .expect("private selection merge");
        assert_ne!(loop_merge, selection_merge);

        let loop_merge_block = function
            .blocks
            .iter()
            .find(|block| {
                block.label.as_ref().and_then(|label| label.result_id) == Some(loop_merge)
            })
            .expect("private loop merge block");
        assert!(matches!(
            loop_merge_block
                .instructions
                .last()
                .map(|inst| (inst.class.opcode, inst.operands.as_slice())),
            Some((rspirv::spirv::Op::Branch, [Operand::IdRef(target)]))
                if *target == selection_merge
        ));
    }

    #[test]
    fn shared_selection_loop_break_phi_uses_emitted_predecessors() {
        let words = emit_fragment(&shared_selection_loop_break_phi_cfg());
        assert!(phi_preds_consistent(&words));
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_naga(&words);

        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let function = &module.functions[0];
        let selection_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find_map(|inst| match (&inst.class.opcode, inst.operands.first()) {
                (rspirv::spirv::Op::SelectionMerge, Some(Operand::IdRef(label))) => Some(*label),
                _ => None,
            })
            .expect("selection merge");
        let loop_merge = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find_map(|inst| match (&inst.class.opcode, inst.operands.first()) {
                (rspirv::spirv::Op::LoopMerge, Some(Operand::IdRef(label))) => Some(*label),
                _ => None,
            })
            .expect("loop merge");
        assert_ne!(selection_merge, loop_merge);
        for label in [loop_merge, selection_merge] {
            let block = function
                .blocks
                .iter()
                .find(|block| block.label.as_ref().and_then(|inst| inst.result_id) == Some(label))
                .expect("merge block");
            let phi = block
                .instructions
                .iter()
                .find(|inst| inst.class.opcode == rspirv::spirv::Op::Phi)
                .expect("merge phi");
            assert_eq!(phi.operands.len(), 4);
        }
    }

    #[test]
    fn unconditional_self_loop_merge_exits_through_safety_guard() {
        let words = emit_fragment(&unconditional_loop_before_phi_cfg());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let first_loop_merge = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .expect("first loop merge");
        let Operand::IdRef(merge_label) = first_loop_merge.operands[0] else {
            panic!("loop merge target must be an id");
        };
        let merge_block = module.functions[0]
            .blocks
            .iter()
            .find(|block| block.label.as_ref().and_then(|inst| inst.result_id) == Some(merge_label))
            .expect("unconditional loop merge block");
        assert_eq!(
            merge_block
                .instructions
                .last()
                .map(|inst| inst.class.opcode),
            Some(rspirv::spirv::Op::Branch)
        );
    }

    #[test]
    fn structured_loops_have_private_iteration_guards() {
        let words = emit_fragment(&single_conditional_loop_cfg());
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let private_vars = module
            .types_global_values
            .iter()
            .filter(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Variable
                    && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Private))
            })
            .collect::<Vec<_>>();
        assert_eq!(private_vars.len(), 1);
        let initializers = private_vars
            .iter()
            .filter_map(|inst| match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        assert!(!initializers.is_empty());
        assert!(module.types_global_values.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Constant
                && inst.result_id.is_some_and(|id| initializers.contains(&id))
                && inst.operands.first() == Some(&Operand::LiteralBit32(SHADER_LOOP_SAFETY_LIMIT))
        }));

        let function = &module.functions[0];
        let loop_merges = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .collect::<Vec<_>>();
        assert_eq!(loop_merges.len(), 1);
        for loop_merge in loop_merges {
            let (Operand::IdRef(merge), Operand::IdRef(cont)) =
                (&loop_merge.operands[0], &loop_merge.operands[1])
            else {
                panic!("loop targets must be ids");
            };
            assert!(function.blocks.iter().any(|block| {
                block.instructions.last().is_some_and(|term| {
                    term.class.opcode == rspirv::spirv::Op::BranchConditional
                        && term.operands.get(1) == Some(&Operand::IdRef(*cont))
                        && term.operands.get(2) == Some(&Operand::IdRef(*merge))
                })
            }));
        }
        let ops = function
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| inst.class.opcode)
            .collect::<Vec<_>>();
        assert_eq!(
            ops.iter()
                .filter(|op| **op == rspirv::spirv::Op::ISub)
                .count(),
            1
        );
        assert_eq!(
            ops.iter()
                .filter(|op| **op == rspirv::spirv::Op::SGreaterThanEqual)
                .count(),
            1
        );
    }

    #[test]
    fn loop_guard_is_not_emitted_for_acyclic_cfg() {
        let words = emit_fragment(&Cfg {
            blocks: vec![empty_cfg_block(0, BranchKind::Exit)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        });
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(!module.types_global_values.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Private))
        }));
        assert!(!module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::SGreaterThanEqual));
    }

    #[test]
    fn phi_validator_rejects_incomplete_and_duplicate_parent_sets() {
        fn mutate_phi(words: &[u32], mutate: impl FnOnce(&mut Vec<Operand>)) -> Vec<u32> {
            let mut module = rspirv::dr::load_words(words).expect("valid SPIR-V");
            let phi = module.functions[0]
                .blocks
                .iter_mut()
                .flat_map(|block| &mut block.instructions)
                .find(|inst| {
                    inst.class.opcode == rspirv::spirv::Op::Phi && inst.operands.len() >= 4
                })
                .expect("multi-parent phi");
            mutate(&mut phi.operands);
            module.assemble()
        }

        let words = emit_fragment(&unconditional_loop_before_phi_cfg());
        let missing = mutate_phi(&words, |operands| {
            operands.truncate(operands.len() - 2);
        });
        assert!(!phi_preds_consistent(&missing));

        let duplicate = mutate_phi(&words, |operands| {
            operands[3] = operands[1].clone();
        });
        assert!(!phi_preds_consistent(&duplicate));

        let malformed = mutate_phi(&words, |operands| {
            operands.pop();
        });
        assert!(!phi_preds_consistent(&malformed));
    }

    #[test]
    fn shared_merge_loop_phi_uses_continue_predecessor() {
        let mut merge_program = nexium_shader::IrProgram::new();
        let phi_result = merge_program.emit(
            IrOp::Phi {
                sources: vec![
                    (1, IrValue::ImmF32(1.0)),
                    (2, IrValue::ImmF32(2.0)),
                    (3, IrValue::ImmF32(3.0)),
                    (5, IrValue::Inst(ValueId(1))),
                ],
            },
            Some(0),
        );
        assert_eq!(phi_result, ValueId(0));
        let mut latch_program = nexium_shader::IrProgram::with_offset(1);
        let latch_value = latch_program.emit(IrOp::Mov(IrValue::ImmF32(4.0)), Some(0));
        assert_eq!(latch_value, ValueId(1));
        let cfg = Cfg {
            blocks: vec![
                empty_cfg_block(
                    0,
                    BranchKind::Conditional {
                        target: 2,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(1, BranchKind::Unconditional { target: 4 }),
                empty_cfg_block(
                    2,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                ),
                empty_cfg_block(3, BranchKind::Unconditional { target: 4 }),
                cfg_block(4, BranchKind::FallThrough, merge_program),
                cfg_block(
                    5,
                    BranchKind::Conditional {
                        target: 4,
                        pred: always_pred(),
                    },
                    latch_program,
                ),
                empty_cfg_block(6, BranchKind::Exit),
            ],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let words = emit_fragment(&cfg);
        assert!(phi_preds_consistent(&words));
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let header = module.functions[0]
            .blocks
            .iter()
            .find(|block| {
                block
                    .instructions
                    .iter()
                    .any(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            })
            .expect("loop header");
        let loop_merge = header
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::LoopMerge)
            .unwrap();
        let Operand::IdRef(continue_label) = loop_merge.operands[1] else {
            panic!("continue target must be an id");
        };
        let phi = header
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::Phi)
            .expect("loop phi");
        let continue_value = phi
            .operands
            .chunks_exact(2)
            .find_map(|pair| match (&pair[0], &pair[1]) {
                (Operand::IdRef(value), Operand::IdRef(parent)) if *parent == continue_label => {
                    Some(*value)
                }
                _ => None,
            })
            .expect("loop phi continue input");
        let continue_block = module.functions[0]
            .blocks
            .iter()
            .find(|block| {
                block.label.as_ref().and_then(|inst| inst.result_id) == Some(continue_label)
            })
            .expect("continue block");
        assert!(continue_block.instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::CopyObject
                && inst.result_id == Some(continue_value)
        }));
    }

    #[test]
    fn fragment_fine_derivatives_pass_naga_validation() {
        let bytes = build_test_program(&[
            0xEF17_700C_F017_0103u64,
            0x50F8_0009_9017_0303u64,
            0xef17_700c_f027_1106u64,
            0x50f8_000a_5117_0606u64,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_fragment_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
    }

    #[test]
    fn fragment_y_direction_uses_pipeline_orientation_sign() {
        let bytes = build_test_program(&[0xf0c8_0000_0127_0003, enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);

        for (y_negate, expected_bits) in [(false, 1.0f32.to_bits()), (true, (-1.0f32).to_bits())] {
            let (words, _, _, _, _) = emit_fragment_full_with_options(
                &cfg,
                [0; 32],
                1,
                0,
                false,
                0,
                0,
                FragmentOptions {
                    y_negate,
                    ..FragmentOptions::default()
                },
            );
            validates_with_naga(&words);
            let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
            assert!(module.types_global_values.iter().any(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Constant
                    && inst.operands == [Operand::LiteralBit32(expected_bits)]
            }));
        }
    }

    #[test]
    fn fragment_tld_b_1d_emits_signed_2d_image_fetch_without_sampler_use() {
        let bytes = build_test_program(&[
            0x4c98_0788_06c7_0004,
            0x4c47_0208_1627_0404,
            0xdd38_0000_8047_1515,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        assert!(nexium_shader::shader_uses_texel_fetch(&cfg));
        assert_eq!(
            nexium_shader::texture_ids(&cfg),
            vec![nexium_shader::bindless_texture_id_pair(
                2,
                0x6c,
                Some(0x162)
            )]
        );

        let (words, _, tex_ids, _) = emit_fragment_full(&cfg);
        assert!(validate_structured_cfg(&words).is_ok());
        assert_eq!(
            tex_ids,
            vec![nexium_shader::bindless_texture_id_pair(
                2,
                0x6c,
                Some(0x162)
            )]
        );
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let fetches = instructions
            .iter()
            .copied()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch)
            .collect::<Vec<_>>();
        assert_eq!(fetches.len(), 1);
        assert!(!instructions.iter().any(|inst| matches!(
            inst.class.opcode,
            rspirv::spirv::Op::SampledImage
                | rspirv::spirv::Op::ImageSampleImplicitLod
                | rspirv::spirv::Op::ImageSampleExplicitLod
        )));
        assert!(!module.annotations.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Decorate
                && matches!(
                    inst.operands.as_slice(),
                    [
                        Operand::IdRef(_),
                        Operand::Decoration(Decoration::Binding),
                        Operand::LiteralBit32(2)
                    ]
                )
        }));

        let Operand::IdRef(coord_id) = fetches[0].operands[1] else {
            panic!("image fetch coordinate must be an id");
        };
        let coord = instructions
            .iter()
            .copied()
            .find(|inst| inst.result_id == Some(coord_id))
            .expect("coordinate constructor");
        assert_eq!(coord.class.opcode, rspirv::spirv::Op::CompositeConstruct);
        let coord_type = coord.result_type.expect("coordinate type");
        let vector_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(coord_type))
            .expect("coordinate vector type");
        assert_eq!(vector_type.class.opcode, rspirv::spirv::Op::TypeVector);
        assert_eq!(vector_type.operands[1], Operand::LiteralBit32(2));
        let Operand::IdRef(component_type) = vector_type.operands[0] else {
            panic!("vector component type must be an id");
        };
        let scalar_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(component_type))
            .expect("coordinate scalar type");
        assert_eq!(scalar_type.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar_type.operands[1], Operand::LiteralBit32(1));
    }

    #[test]
    fn fragment_tld_b_buffer_slot_emits_uint_buffer_fetch_on_binding_19() {
        let bytes = build_test_program(&[
            0x4c98_0788_06c7_0004,
            0x4c47_0208_1627_0404,
            0xdd38_0000_8047_1515,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);

        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    nexium_shader::bindless_texture_id_pair(2, 0x6c, Some(0x162)),
                    0,
                    TextureNumericType::Uint,
                )
                .with_image_kind(GraphicsImageKind::Buffer)],
                texel_buffer_mask: 1,
                ..FragmentOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module
            .capabilities
            .iter()
            .any(|inst| { inst.operands == [Operand::Capability(Capability::SampledBuffer)] }));
        let buffer_var = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(GFX_BINDING_UINT_TEXEL_BUFFER),
                ] => Some(*target),
                _ => None,
            })
            .expect("binding 14 buffer image array");
        let variable = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Variable
                    && inst.result_id == Some(buffer_var)
            })
            .expect("buffer image array variable");
        let pointer_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == variable.result_type)
            .expect("buffer image array pointer type");
        let Operand::IdRef(array_type_id) = pointer_type.operands[1] else {
            panic!("buffer descriptor must point to an array");
        };
        let array_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(array_type_id))
            .expect("buffer image array type");
        let Operand::IdRef(image_type_id) = array_type.operands[0] else {
            panic!("buffer descriptor array must contain images");
        };
        let image_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(image_type_id))
            .expect("buffer image type");
        assert_eq!(image_type.class.opcode, rspirv::spirv::Op::TypeImage);
        assert_eq!(
            image_type.operands[1],
            Operand::Dim(rspirv::spirv::Dim::DimBuffer)
        );
        let Operand::IdRef(sampled_type_id) = image_type.operands[0] else {
            panic!("buffer image sampled type must be an id");
        };
        let sampled_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(sampled_type_id))
            .expect("buffer image sampled type");
        assert_eq!(sampled_type.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(sampled_type.operands[0], Operand::LiteralBit32(32));
        assert_eq!(sampled_type.operands[1], Operand::LiteralBit32(0));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let fetch = instructions
            .iter()
            .copied()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch)
            .expect("buffer image fetch");
        assert_eq!(fetch.operands.len(), 2, "buffer fetch must not carry Lod");
        let Operand::IdRef(coord_id) = fetch.operands[1] else {
            panic!("buffer fetch coordinate must be an id");
        };
        let coord = module
            .types_global_values
            .iter()
            .chain(instructions.iter().copied())
            .find(|inst| inst.result_id == Some(coord_id))
            .expect("buffer fetch coordinate");
        let coord_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == coord.result_type)
            .expect("buffer coordinate type");
        assert_eq!(coord_type.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(coord_type.operands[0], Operand::LiteralBit32(32));
        assert_eq!(coord_type.operands[1], Operand::LiteralBit32(1));

        let fetched_id = fetch.result_id.expect("buffer fetch result");
        let extracted = instructions
            .iter()
            .copied()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::CompositeExtract
                    && inst.operands.first() == Some(&Operand::IdRef(fetched_id))
            })
            .and_then(|inst| inst.result_id)
            .expect("uint buffer component extraction");
        assert!(instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Bitcast
                && inst.operands == [Operand::IdRef(extracted)]
        }));
    }

    #[test]
    fn vertex_tld_b_buffer_slot_emits_float_buffer_fetch_on_binding_14() {
        let bytes = build_test_program(&[
            0x4c98_0788_06c7_0004,
            0x4c47_0208_1627_0404,
            0xdd38_0000_8047_1515,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let (words, _, _) = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                texel_buffer_mask: 1,
                ..VertexOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.annotations.iter().any(|inst| {
            matches!(
                inst.operands.as_slice(),
                [
                    Operand::IdRef(_),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(14)
                ]
            )
        }));
        let buffer_image = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::TypeImage
                    && inst.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimBuffer))
            })
            .expect("buffer image type");
        let Operand::IdRef(sampled_type_id) = buffer_image.operands[0] else {
            panic!("buffer image sampled type must be an id");
        };
        let sampled_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(sampled_type_id))
            .expect("buffer image sampled type");
        assert_eq!(sampled_type.class.opcode, rspirv::spirv::Op::TypeFloat);
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let fetch = instructions
            .iter()
            .copied()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch)
            .expect("buffer image fetch");
        assert_eq!(fetch.operands.len(), 2);
        let fetched_id = fetch.result_id.expect("buffer fetch result");
        let extracted = instructions
            .iter()
            .copied()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::CompositeExtract
                    && inst.operands.first() == Some(&Operand::IdRef(fetched_id))
            })
            .and_then(|inst| inst.result_id)
            .expect("float buffer component extraction");
        assert!(!instructions.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Bitcast
                && inst.operands == [Operand::IdRef(extracted)]
        }));
    }

    #[test]
    fn pps_slot10_d2_tld_with_copied_zero_y_emits_scalar_buffer_fetch() {
        let shader_id = nexium_shader::bindless_texture_id_pair(2, 0x5e, Some(0x15a));
        let mut program = nexium_shader::IrProgram::new();
        let zero_y = program.emit(IrOp::Mov(IrValue::Zero), Some(9));
        program.emit(
            IrOp::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x5e,
                cbuf_secondary_word_offset: Some(0x15a),
                x: IrValue::GprIn(8),
                y: Some(IrValue::Inst(zero_y)),
                z: None,
                component: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _) = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    shader_id,
                    0,
                    TextureNumericType::Float,
                )
                .with_image_kind(GraphicsImageKind::Buffer)],
                texel_buffer_mask: 1,
                ..VertexOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let image_type = image_type_for_binding(&module, GFX_BINDING_FLOAT_TEXEL_BUFFER);
        assert_eq!(
            image_type.operands[1],
            Operand::Dim(rspirv::spirv::Dim::DimBuffer)
        );

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let fetch = instructions
            .iter()
            .copied()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch)
            .expect("buffer image fetch");
        assert_eq!(fetch.operands.len(), 2, "buffer fetch must not carry Lod");
        let Operand::IdRef(coord_id) = fetch.operands[1] else {
            panic!("buffer fetch coordinate must be an id");
        };
        let coord = module
            .types_global_values
            .iter()
            .chain(instructions.iter().copied())
            .find(|inst| inst.result_id == Some(coord_id))
            .expect("buffer fetch coordinate");
        let coord_type = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == coord.result_type)
            .expect("buffer coordinate type");
        assert_eq!(coord_type.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(coord_type.operands[0], Operand::LiteralBit32(32));
        assert_eq!(coord_type.operands[1], Operand::LiteralBit32(1));
        assert!(!module.annotations.iter().any(|inst| {
            matches!(
                inst.operands.as_slice(),
                [
                    Operand::IdRef(_),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(GFX_BINDING_FLOAT_2D)
                ]
            )
        }));
    }

    #[test]
    #[should_panic(expected = "graphics texture manifest does not match shader resources")]
    fn d2_tld_with_dynamic_y_rejects_buffer_manifest() {
        let shader_id = nexium_shader::bindless_texture_id_pair(2, 0x5e, Some(0x15a));
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x5e,
                cbuf_secondary_word_offset: Some(0x15a),
                x: IrValue::GprIn(8),
                y: Some(IrValue::GprIn(9)),
                z: None,
                component: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let _ = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    shader_id,
                    0,
                    TextureNumericType::Float,
                )
                .with_image_kind(GraphicsImageKind::Buffer)],
                texel_buffer_mask: 1,
                ..VertexOptions::default()
            },
        );
    }

    fn interface_scalar_type(
        module: &rspirv::dr::Module,
        storage_class: StorageClass,
        location: u32,
    ) -> &rspirv::dr::Instruction {
        let variable_id = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Location),
                    Operand::LiteralBit32(found),
                ] if *found == location => Some(*target),
                _ => None,
            })
            .and_then(|target| {
                module.types_global_values.iter().find_map(|inst| {
                    (inst.class.opcode == rspirv::spirv::Op::Variable
                        && inst.result_id == Some(target)
                        && inst.operands.first()
                            == Some(&Operand::StorageClass(storage_class)))
                    .then_some(inst.result_type?)
                })
            })
            .expect("located interface variable");
        let vector_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(variable_id))
            .and_then(|inst| match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .expect("interface pointer target");
        let scalar_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(vector_id))
            .and_then(|inst| match inst.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .expect("interface vector scalar");
        module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(scalar_id))
            .expect("interface scalar type")
    }

    fn image_type_for_binding(
        module: &rspirv::dr::Module,
        binding: u32,
    ) -> &rspirv::dr::Instruction {
        let variable_id = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(found),
                ] if *found == binding => Some(*target),
                _ => None,
            })
            .expect("image descriptor binding");
        let pointer_type_id = module
            .types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Variable
                    && inst.result_id == Some(variable_id)
            })
            .and_then(|inst| inst.result_type)
            .expect("image descriptor pointer type");
        let array_type_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(pointer_type_id))
            .and_then(|inst| match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .expect("image descriptor array type");
        let image_type_id = module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(array_type_id))
            .and_then(|inst| match inst.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .expect("image descriptor element type");
        module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(image_type_id))
            .expect("image type")
    }

    fn image_scalar_type_for_binding(
        module: &rspirv::dr::Module,
        binding: u32,
    ) -> &rspirv::dr::Instruction {
        let image_type = image_type_for_binding(module, binding);
        let scalar_id = match image_type.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => panic!("image scalar type must be an id"),
        };
        module
            .types_global_values
            .iter()
            .find(|inst| inst.result_id == Some(scalar_id))
            .expect("image scalar type")
    }

    #[test]
    fn fragment_uint_output_declares_uint_and_bitcasts_at_store() {
        let cfg = Cfg {
            blocks: vec![empty_cfg_block(0, BranchKind::Exit)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                uint_output_mask: 1,
                ..FragmentOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let scalar = interface_scalar_type(&module, StorageClass::Output, 0);
        assert_eq!(scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar.operands[1], Operand::LiteralBit32(0));
        assert!(module.functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block
                    .instructions
                    .iter()
                    .any(|inst| inst.class.opcode == rspirv::spirv::Op::Bitcast)
            })
        }));
    }

    #[test]
    fn fragment_sint_output_uses_guest_location_before_attachment_remap() {
        let cfg = Cfg {
            blocks: vec![empty_cfg_block(0, BranchKind::Exit)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0xF << 8,
            false,
            0,
            0,
            FragmentOptions {
                sint_output_mask: 1 << 2,
                ..FragmentOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let scalar = interface_scalar_type(&module, StorageClass::Output, 0);
        assert_eq!(scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar.operands[1], Operand::LiteralBit32(1));
    }

    #[test]
    fn fragment_uint_tld_uses_uint_image_fetch_then_restores_bit_container() {
        let bytes = build_test_program(&[
            0x4c98_0788_06c7_0004,
            0x4c47_0208_1627_0404,
            0xdd38_0000_8047_1515,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    nexium_shader::bindless_texture_id_pair(2, 0x6c, Some(0x162)),
                    0,
                    TextureNumericType::Uint,
                )],
                ..FragmentOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let scalar = image_scalar_type_for_binding(&module, GFX_BINDING_UINT_2D);
        assert_eq!(scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar.operands[1], Operand::LiteralBit32(0));
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Bitcast));
    }

    #[test]
    fn mixed_sample_and_buffer_tld_keep_independent_numeric_types() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::TexelFetch {
                cbuf_binding: 2,
                cbuf_word_offset: 0x68,
                cbuf_secondary_word_offset: Some(0x162),
                x: IrValue::ImmF32(0.0),
                y: None,
                z: None,
                component: 0,
            },
            Some(0),
        );
        program.emit(
            IrOp::SampleTex {
                tex_id: 0,
                u: IrValue::ImmF32(0.0),
                v: IrValue::ImmF32(0.0),
                array: None,
                volume: None,
                cube: None,
                dref: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                component: 0,
            },
            Some(1),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                texture_numeric_manifest: vec![
                    GraphicsTextureResource::new(0, 0, TextureNumericType::Float),
                    GraphicsTextureResource::new(
                        nexium_shader::bindless_texture_id_pair(2, 0x68, Some(0x162)),
                        1,
                        TextureNumericType::Uint,
                    )
                    .with_image_kind(GraphicsImageKind::Buffer),
                ],
                texel_buffer_mask: 1 << 1,
                ..FragmentOptions::default()
            },
        );
        assert!(validate_structured_cfg(&words).is_ok());
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let sampled_image = image_type_for_binding(&module, GFX_BINDING_FLOAT_2D);
        assert_eq!(
            sampled_image.operands[1],
            Operand::Dim(rspirv::spirv::Dim::Dim2D)
        );
        let sampled_scalar = image_scalar_type_for_binding(&module, GFX_BINDING_FLOAT_2D);
        assert_eq!(sampled_scalar.class.opcode, rspirv::spirv::Op::TypeFloat);

        let fetched_image = image_type_for_binding(&module, GFX_BINDING_UINT_TEXEL_BUFFER);
        assert_eq!(
            fetched_image.operands[1],
            Operand::Dim(rspirv::spirv::Dim::DimBuffer)
        );
        let fetched_scalar = image_scalar_type_for_binding(&module, GFX_BINDING_UINT_TEXEL_BUFFER);
        assert_eq!(fetched_scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(fetched_scalar.operands[1], Operand::LiteralBit32(0));
        assert!(!module.annotations.iter().any(|inst| {
            matches!(
                inst.operands.as_slice(),
                [
                    Operand::IdRef(_),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(GFX_BINDING_FLOAT_TEXEL_BUFFER)
                ]
            )
        }));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Bitcast));
    }

    #[test]
    fn fragment_sint_tld_uses_sint_binding_and_restores_bit_container() {
        let bytes = build_test_program(&[
            0x4c98_0788_06c7_0004,
            0x4c47_0208_1627_0404,
            0xdd38_0000_8047_1515,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let (words, _, _, _, _) = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    nexium_shader::bindless_texture_id_pair(2, 0x6c, Some(0x162)),
                    0,
                    TextureNumericType::Sint,
                )],
                ..FragmentOptions::default()
            },
        );
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let scalar = image_scalar_type_for_binding(&module, GFX_BINDING_SINT_2D);
        assert_eq!(scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar.operands[1], Operand::LiteralBit32(1));
        assert!(module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::Bitcast));
    }

    #[test]
    #[should_panic(expected = "used by Sample/Gather but was assigned the incompatible Uint")]
    fn filtered_sampling_rejects_integer_descriptor_family_before_emission() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::SampleTex {
                tex_id: 0x44,
                u: IrValue::ImmF32(0.0),
                v: IrValue::ImmF32(0.0),
                array: None,
                volume: None,
                cube: None,
                dref: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                component: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let _ = emit_fragment_full_with_options(
            &cfg,
            [0; 32],
            1,
            0,
            false,
            0,
            0,
            FragmentOptions {
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    0x44,
                    0,
                    TextureNumericType::Uint,
                )],
                ..FragmentOptions::default()
            },
        );
    }

    #[test]
    fn graphics_image_binding_selects_every_numeric_family_and_image_kind() {
        let expected = [
            (
                GraphicsImageKind::D2,
                [
                    GFX_BINDING_FLOAT_2D,
                    GFX_BINDING_UINT_2D,
                    GFX_BINDING_SINT_2D,
                ],
            ),
            (
                GraphicsImageKind::D3,
                [
                    GFX_BINDING_FLOAT_3D,
                    GFX_BINDING_UINT_3D,
                    GFX_BINDING_SINT_3D,
                ],
            ),
            (
                GraphicsImageKind::Cube,
                [
                    GFX_BINDING_FLOAT_CUBE,
                    GFX_BINDING_UINT_CUBE,
                    GFX_BINDING_SINT_CUBE,
                ],
            ),
            (
                GraphicsImageKind::CubeArray,
                [
                    GFX_BINDING_FLOAT_CUBE_ARRAY,
                    GFX_BINDING_UINT_CUBE_ARRAY,
                    GFX_BINDING_SINT_CUBE_ARRAY,
                ],
            ),
            (
                GraphicsImageKind::Buffer,
                [
                    GFX_BINDING_FLOAT_TEXEL_BUFFER,
                    GFX_BINDING_UINT_TEXEL_BUFFER,
                    GFX_BINDING_SINT_TEXEL_BUFFER,
                ],
            ),
        ];
        let numeric_types = [
            TextureNumericType::Float,
            TextureNumericType::Uint,
            TextureNumericType::Sint,
        ];

        for (kind, bindings) in expected {
            for (numeric_type, binding) in numeric_types.into_iter().zip(bindings) {
                assert_eq!(Emitter::graphics_image_binding(numeric_type, kind), binding);
            }
        }

        for (numeric_type, binding) in numeric_types.into_iter().zip([
            GFX_BINDING_FLOAT_2D,
            GFX_BINDING_UINT_2D,
            GFX_BINDING_SINT_2D,
        ]) {
            assert_eq!(
                Emitter::graphics_image_binding(numeric_type, GraphicsImageKind::D2Array),
                binding
            );
        }
    }

    #[test]
    fn graphics_typed_image_declarations_preserve_dimension_arraying_and_scalar_family() {
        let image_kinds = [
            (GraphicsImageKind::D2, rspirv::spirv::Dim::Dim2D, 0),
            (GraphicsImageKind::D3, rspirv::spirv::Dim::Dim3D, 0),
            (GraphicsImageKind::Cube, rspirv::spirv::Dim::DimCube, 0),
            (
                GraphicsImageKind::CubeArray,
                rspirv::spirv::Dim::DimCube,
                1,
            ),
            (
                GraphicsImageKind::Buffer,
                rspirv::spirv::Dim::DimBuffer,
                0,
            ),
        ];

        for numeric_type in [
            TextureNumericType::Float,
            TextureNumericType::Uint,
            TextureNumericType::Sint,
        ] {
            for (kind, expected_dimension, expected_arrayed) in image_kinds {
                let mut emitter = Emitter::new(Stage::Fragment);
                let decl = emitter.ensure_typed_image_array(numeric_type, kind);
                let expected_binding = Emitter::graphics_image_binding(numeric_type, kind);
                let module = emitter.b.module();

                assert!(module.annotations.iter().any(|instruction| {
                    matches!(
                        instruction.operands.as_slice(),
                        [
                            Operand::IdRef(target),
                            Operand::Decoration(Decoration::Binding),
                            Operand::LiteralBit32(binding),
                        ] if *target == decl.var && *binding == expected_binding
                    )
                }));

                let image_type = module
                    .types_global_values
                    .iter()
                    .find(|instruction| instruction.result_id == Some(decl.image_t))
                    .expect("typed image declaration");
                assert_eq!(image_type.class.opcode, rspirv::spirv::Op::TypeImage);
                assert_eq!(
                    image_type.operands[1],
                    Operand::Dim(expected_dimension),
                    "unexpected dimension for {numeric_type:?} {kind:?}"
                );
                assert_eq!(
                    image_type.operands[3],
                    Operand::LiteralBit32(expected_arrayed),
                    "unexpected array flag for {numeric_type:?} {kind:?}"
                );

                let scalar_id = match image_type.operands[0] {
                    Operand::IdRef(id) => id,
                    _ => panic!("typed image scalar must be an id"),
                };
                let scalar_type = module
                    .types_global_values
                    .iter()
                    .find(|instruction| instruction.result_id == Some(scalar_id))
                    .expect("typed image scalar declaration");
                match numeric_type {
                    TextureNumericType::Float => {
                        assert_eq!(scalar_type.class.opcode, rspirv::spirv::Op::TypeFloat);
                    }
                    TextureNumericType::Uint => {
                        assert_eq!(scalar_type.class.opcode, rspirv::spirv::Op::TypeInt);
                        assert_eq!(scalar_type.operands[1], Operand::LiteralBit32(0));
                    }
                    TextureNumericType::Sint => {
                        assert_eq!(scalar_type.class.opcode, rspirv::spirv::Op::TypeInt);
                        assert_eq!(scalar_type.operands[1], Operand::LiteralBit32(1));
                    }
                }
            }
        }
    }

    #[test]
    fn graphics_cube_texel_fetch_fails_closed_without_face_semantics() {
        for kind in [GraphicsImageKind::Cube, GraphicsImageKind::CubeArray] {
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut emitter = Emitter::new(Stage::Fragment);
                emitter.typed_fetch_image_at(0, TextureNumericType::Uint, kind);
            }))
            .expect_err("cube texel fetch must not be lowered as a 3D fetch");
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                message.contains(
                    "graphics cube texel fetch is unsupported: the current IR does not retain cube face semantics"
                ),
                "unexpected panic for {kind:?}: {message}"
            );
        }
    }

    #[test]
    fn graphics_descriptor_binding_constants_match_the_runtime_abi() {
        assert_eq!(GFX_BINDING_CBUF, 0);
        assert_eq!(GFX_BINDING_FLOAT_2D, 1);
        assert_eq!(GFX_BINDING_SAMPLERS, 2);
        assert_eq!(GFX_BINDING_SSBO_BASE, 3);
        assert_eq!(GFX_BINDING_SSBO_BASE + MAX_SSBO - 1, 10);
        assert_eq!(
            [
                GFX_BINDING_FLOAT_3D,
                GFX_BINDING_FLOAT_CUBE,
                GFX_BINDING_FLOAT_CUBE_ARRAY,
                GFX_BINDING_FLOAT_TEXEL_BUFFER,
                GFX_BINDING_UINT_2D,
                GFX_BINDING_UINT_3D,
                GFX_BINDING_UINT_CUBE,
                GFX_BINDING_UINT_CUBE_ARRAY,
                GFX_BINDING_UINT_TEXEL_BUFFER,
                GFX_BINDING_SINT_2D,
                GFX_BINDING_SINT_3D,
                GFX_BINDING_SINT_CUBE,
                GFX_BINDING_SINT_CUBE_ARRAY,
                GFX_BINDING_SINT_TEXEL_BUFFER,
            ],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        );
    }

    #[test]
    fn graphics_texture_manifest_normalization_is_deterministic() {
        let first = GraphicsTextureResource::new(0x10, 1, TextureNumericType::Float);
        let second = GraphicsTextureResource::new(0x44, 7, TextureNumericType::Uint);
        let forward =
            normalize_graphics_texture_manifest(vec![first, second, second]).unwrap();
        let reverse =
            normalize_graphics_texture_manifest(vec![second, first, second]).unwrap();
        assert_eq!(forward, vec![first, second]);
        assert_eq!(forward, reverse);

        let cube = GraphicsTextureResource::new(0x80, 4, TextureNumericType::Float)
            .with_image_kind(GraphicsImageKind::Cube);
        let cube_array = GraphicsTextureResource::new(0x80, 4, TextureNumericType::Float)
            .with_image_kind(GraphicsImageKind::CubeArray);
        assert!(matches!(
            normalize_graphics_texture_manifest(vec![cube, cube_array]),
            Err(GraphicsTextureManifestError::ImageKindConflict { slot: 4, .. })
        ));
    }

    #[test]
    fn graphics_texture_manifest_rejects_shader_id_mismatch() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::SampleTex {
                tex_id: 0x44,
                u: IrValue::ImmF32(0.0),
                v: IrValue::ImmF32(0.0),
                array: None,
                volume: None,
                cube: None,
                dref: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                component: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            emit_fragment_full_with_options(
                &cfg,
                [0; 32],
                1,
                0,
                false,
                0,
                0,
                FragmentOptions {
                    texture_numeric_manifest: vec![GraphicsTextureResource::new(
                        0x45,
                        0,
                        TextureNumericType::Float,
                    )],
                    ..FragmentOptions::default()
                },
            )
        }))
        .expect_err("mismatched shader texture ID must fail closed");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            message.contains(
                "descriptor slot 0 should contain shader texture ID 0x44, but the manifest contains 0x45"
            ),
            "unexpected panic: {message}"
        );
    }

    #[test]
    fn vertex_texture_manifest_includes_descriptor_slot_base() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::SampleTex {
                tex_id: 0x22,
                u: IrValue::ImmF32(0.0),
                v: IrValue::ImmF32(0.0),
                array: None,
                volume: None,
                cube: None,
                dref: None,
                implicit_lod: true,
                lod_bias: None,
                explicit_lod: None,
                texel_offset: None,
                component: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let (words, _, _) = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                tex_slot_base: 5,
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    0x22,
                    5,
                    TextureNumericType::Float,
                )],
                ..VertexOptions::default()
            },
        );
        validates_with_spirv_val_if_available(&words);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            emit_vertex_with_bindings_opts(
                &cfg,
                &[],
                VertexOptions {
                    tex_slot_base: 5,
                    texture_numeric_manifest: vec![GraphicsTextureResource::new(
                        0x22,
                        0,
                        TextureNumericType::Float,
                    )],
                    ..VertexOptions::default()
                },
            )
        }));
        assert!(panic.is_err(), "vertex manifest must honor tex_slot_base");
    }

    #[test]
    fn vertex_tld_b_3d_emits_image_fetch_on_binding_11() {
        let bytes = build_test_program(&[
            enc_static_ldc(39, 3, 0x90),
            0xdd38_0007_c277_0000,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let (words, _, tex_ids) = Emitter::new(Stage::Vertex).finish_full(&cfg, &[]);
        assert!(validate_structured_cfg(&words).is_ok());
        assert_eq!(tex_ids, vec![nexium_shader::bindless_texture_id(3, 0x24)]);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|inst| inst.class.opcode == rspirv::spirv::Op::ImageFetch)
                .count(),
            4
        );
        assert!(!instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::SampledImage));
        assert!(!module.annotations.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Decorate
                && matches!(
                    inst.operands.as_slice(),
                    [
                        Operand::IdRef(_),
                        Operand::Decoration(Decoration::Binding),
                        Operand::LiteralBit32(2)
                    ]
                )
        }));
        assert!(module.annotations.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::Decorate
                && matches!(
                    inst.operands.as_slice(),
                    [
                        Operand::IdRef(_),
                        Operand::Decoration(Decoration::Binding),
                        Operand::LiteralBit32(11)
                    ]
                )
        }));
    }

    #[test]
    fn fragment_texs_2d_emits_implicit_lod() {
        let bytes = build_test_program(&[0xD822_00A0_5087_0500u64, enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block.instructions.iter().any(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod
                })
            })
        }));
        assert!(!module.functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block.instructions.iter().any(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::ImageSampleExplicitLod
                })
            })
        }));
    }

    #[test]
    fn fragment_tex_b_ll_emits_register_explicit_lod() {
        let bytes = build_test_program(&[
            enc_static_ldc(4, 2, 0x1a0),
            enc_static_ldc(5, 0, 0x40),
            0xdeb8_0060_a047_0a04,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(
            nexium_shader::texture_ids(&cfg),
            vec![nexium_shader::bindless_texture_id_pair(2, 0x68, None)]
        );
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let samples = instructions
            .iter()
            .copied()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleExplicitLod)
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 1);
        assert!(!instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod));
        assert_eq!(
            samples[0].operands.get(2),
            Some(&Operand::ImageOperands(rspirv::spirv::ImageOperands::LOD))
        );
        let Some(Operand::IdRef(lod_id)) = samples[0].operands.get(3) else {
            panic!("explicit LOD operand must be an id");
        };
        assert_graphics_cbuf_sample_operand(&module, *lod_id, 16, 0x40);
    }

    #[test]
    fn fragment_tex_b_lb_emits_register_implicit_lod_bias() {
        let bytes = build_test_program(&[
            enc_static_ldc(8, 2, 0x1a0),
            enc_static_ldc(9, 0, 0x40),
            0xdeba_0044_2087_0600,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        assert_eq!(
            nexium_shader::texture_ids(&cfg),
            vec![nexium_shader::bindless_texture_id_pair(2, 0x68, None)]
        );
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let samples = instructions
            .iter()
            .copied()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod)
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 1);
        assert_eq!(
            samples[0].operands.get(2),
            Some(&Operand::ImageOperands(rspirv::spirv::ImageOperands::BIAS))
        );
        let Some(Operand::IdRef(bias_id)) = samples[0].operands.get(3) else {
            panic!("LOD bias operand must be an id");
        };
        assert_graphics_cbuf_sample_operand(&module, *bias_id, 16, 0x40);
    }

    #[test]
    fn fragment_tex_b_cube_uses_cube_binding_and_xyz_direction() {
        let raw = 0xdeba_0003_e037_0404u64;
        let handle_reg = ((raw >> 20) & 0xff) as u8;
        let bytes = build_test_program(&[enc_static_ldc(handle_reg, 2, 0x1a0), raw, enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::TypeImage
                && inst.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimCube))
        }));
        for binding in [2, 12] {
            assert!(module.annotations.iter().any(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Decorate
                    && matches!(
                        inst.operands.as_slice(),
                        [
                            Operand::IdRef(_),
                            Operand::Decoration(Decoration::Binding),
                            Operand::LiteralBit32(found)
                        ] if *found == binding
                    )
            }));
        }
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let sample = instructions
            .iter()
            .copied()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod)
            .expect("cube sample");
        let Some(Operand::IdRef(coords_id)) = sample.operands.get(1) else {
            panic!("cube coordinate must be an id");
        };
        let coords = instructions
            .iter()
            .copied()
            .find(|inst| inst.result_id == Some(*coords_id))
            .expect("cube coordinate constructor");
        assert_eq!(coords.class.opcode, rspirv::spirv::Op::CompositeConstruct);
        assert_eq!(coords.operands.len(), 3);
        assert!(!instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::FSub));
    }

    #[test]
    fn fragment_tex_b_cube_depth_compare_emits_dref_sample() {
        let bytes = build_test_program(&[
            enc_static_ldc(14, 2, 0x570),
            0xdebe_0000_e0e7_080a,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        assert!(validate_structured_cfg(&words).is_ok());
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypeImage
                && instruction.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimCube))
                && instruction.operands.get(2) == Some(&Operand::LiteralBit32(1))
                && instruction.operands.get(3) == Some(&Operand::LiteralBit32(0))
        }));
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::ImageSampleDrefImplicitLod
                })
                .count(),
            1
        );
        assert!(!instructions.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::ImageSampleImplicitLod
        }));
    }

    #[test]
    fn fragment_tex_b_cube_array_uses_arrayed_cube_binding_layer_and_ll_lod() {
        let raw = 0xdeb8_0067_f147_1010u64;
        let handle_reg = ((raw >> 20) & 0xff) as u8;
        let bytes = build_test_program(&[enc_static_ldc(handle_reg, 2, 0x1a0), raw, enc_exit()]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|inst| {
            inst.class.opcode == rspirv::spirv::Op::TypeImage
                && inst.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimCube))
                && inst.operands.get(3) == Some(&Operand::LiteralBit32(1))
        }));
        for binding in [2, 13] {
            assert!(module.annotations.iter().any(|inst| {
                inst.class.opcode == rspirv::spirv::Op::Decorate
                    && matches!(
                        inst.operands.as_slice(),
                        [
                            Operand::IdRef(_),
                            Operand::Decoration(Decoration::Binding),
                            Operand::LiteralBit32(found)
                        ] if *found == binding
                    )
            }));
        }
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let sample = instructions
            .iter()
            .copied()
            .find(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleExplicitLod)
            .expect("cube-array LL sample");
        assert_eq!(
            sample.operands.get(2),
            Some(&Operand::ImageOperands(rspirv::spirv::ImageOperands::LOD))
        );
        let Some(Operand::IdRef(coords_id)) = sample.operands.get(1) else {
            panic!("cube-array coordinate must be an id");
        };
        let coords = instructions
            .iter()
            .copied()
            .find(|inst| inst.result_id == Some(*coords_id))
            .expect("cube-array coordinate constructor");
        assert_eq!(coords.class.opcode, rspirv::spirv::Op::CompositeConstruct);
        assert_eq!(coords.operands.len(), 4);
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::BitwiseAnd));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == rspirv::spirv::Op::ConvertUToF));
    }

    #[test]
    fn fragment_tex_b_aoffi_emits_constant_offset() {
        let bytes = build_test_program(&[
            enc_static_ldc(30, 2, 0x1a0),
            0x0100_0000_0017_f01f,
            0xdeba_0033_a1e7_180c,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let words = emit_fragment(&cfg);
        validates_with_naga(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let samples = instructions
            .iter()
            .copied()
            .filter(|inst| inst.class.opcode == rspirv::spirv::Op::ImageSampleExplicitLod)
            .collect::<Vec<_>>();
        assert_eq!(samples.len(), 3);
        let expected =
            rspirv::spirv::ImageOperands::LOD | rspirv::spirv::ImageOperands::CONST_OFFSET;
        assert!(samples
            .iter()
            .all(|sample| { sample.operands.get(2) == Some(&Operand::ImageOperands(expected)) }));
        let Some(Operand::IdRef(offset_id)) = samples[0].operands.get(4) else {
            panic!("constant texel offset operand must be an id");
        };
        assert!(module.types_global_values.iter().any(|inst| {
            inst.result_id == Some(*offset_id)
                && inst.class.opcode == rspirv::spirv::Op::ConstantComposite
        }));
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

    fn compute_test_cfg() -> Cfg {
        let buffer = TextureHandleOrigin::Bindless {
            cbuf_binding: 2,
            cbuf_word_offset: 0x168 / 4,
            cbuf_secondary_word_offset: Some(0x568 / 4),
        };
        let sampled_3d = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x140 / 4,
        };
        let storage_3d = TextureHandleOrigin::Bindless {
            cbuf_binding: 2,
            cbuf_word_offset: 0x120 / 4,
            cbuf_secondary_word_offset: None,
        };
        let mut program = nexium_shader::IrProgram::new();
        let local_x = program.emit(IrOp::LocalInvocationId { component: 0 }, Some(0));
        let workgroup_x = program.emit(IrOp::WorkgroupId { component: 0 }, Some(1));
        let cbuf_value = program.emit(
            IrOp::LoadCbuf {
                binding: 3,
                byte_offset: 0x10,
            },
            Some(2),
        );
        let buffer_value = program.emit(
            IrOp::TexelFetchHandle {
                handle: buffer,
                dimension: ImageDimension::D1,
                x: IrValue::Inst(local_x),
                y: None,
                z: None,
                component: 0,
            },
            Some(3),
        );
        let sampled_value = program.emit(
            IrOp::TexelFetchHandle {
                handle: sampled_3d,
                dimension: ImageDimension::D3,
                x: IrValue::Inst(local_x),
                y: Some(IrValue::Zero),
                z: Some(IrValue::Zero),
                component: 1,
            },
            Some(4),
        );
        let mip_count = program.emit(
            IrOp::TextureQueryDimension {
                handle: sampled_3d,
                lod: IrValue::Zero,
                component: 3,
            },
            Some(5),
        );
        let width = program.emit(
            IrOp::TextureQueryDimension {
                handle: sampled_3d,
                lod: IrValue::Zero,
                component: 0,
            },
            Some(6),
        );
        program.emit_void(IrOp::ImageWrite {
            handle: storage_3d,
            dimension: ImageDimension::D3,
            x: IrValue::Inst(local_x),
            y: Some(IrValue::Inst(workgroup_x)),
            z: Some(IrValue::Zero),
            values: [
                IrValue::Inst(buffer_value),
                IrValue::Inst(sampled_value),
                IrValue::Inst(width),
                IrValue::Inst(mip_count),
            ],
        });
        program.emit(IrOp::Mov(IrValue::Inst(cbuf_value)), Some(7));
        Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        }
    }

    fn compute_test_options(storage_numeric_type: TextureNumericType) -> ComputeOptions {
        ComputeOptions {
            local_size: [64, 1, 1],
            local_memory_low_size: 0,
            local_memory_high_size: 0,
            local_memory_crs_size: 0,
            shared_memory_size: 0,
            texture_bound_cbuf: 2,
            cbuf_sizes: [COMPUTE_CBUF_MAX_SIZE; COMPUTE_CBUF_SLOTS],
            resources: vec![
                ComputeImageResource {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x168 / 4,
                        cbuf_secondary_word_offset: Some(0x568 / 4),
                    },
                    binding: 1,
                    kind: ComputeResourceKind::UniformTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: Some(ComputeTexelFormat::R32Uint),
                },
                ComputeImageResource {
                    handle: TextureHandleOrigin::Bound {
                        cbuf_word_offset: 0x140 / 4,
                    },
                    binding: 2,
                    kind: ComputeResourceKind::SampledImage,
                    dimension: ImageDimension::D3,
                    numeric_type: TextureNumericType::Float,
                    texel_format: None,
                },
                ComputeImageResource {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x120 / 4,
                        cbuf_secondary_word_offset: None,
                    },
                    binding: 3,
                    kind: ComputeResourceKind::StorageImage,
                    dimension: ImageDimension::D3,
                    numeric_type: storage_numeric_type,
                    texel_format: None,
                },
            ],
        }
    }

    #[test]
    fn lop3_anf_matches_every_maxwell_truth_table() {
        for lut in 0..=u8::MAX {
            let coefficients = lop3_anf_coefficients(lut);
            for inputs in 0..8u8 {
                let mut actual = false;
                for monomial in 0..8u8 {
                    if coefficients & (1u8 << monomial) != 0 && monomial & inputs == monomial {
                        actual = !actual;
                    }
                }
                let expected = lut & (1u8 << inputs) != 0;
                assert_eq!(actual, expected, "lut={lut:#04x} inputs={inputs:#05b}");
            }
        }
    }

    #[test]
    fn compute_integer_bit_ops_emit_valid_spirv() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::FindUMsb {
                value: IrValue::ImmU32(0x8000_0000),
            },
            Some(0),
        );
        program.emit(
            IrOp::BitCount {
                value: IrValue::ImmU32(0xf0f0_1234),
            },
            Some(1),
        );
        program.emit(
            IrOp::ILop3 {
                a: IrValue::ImmU32(0xaaaa_aaaa),
                b: IrValue::ImmU32(0xcccc_cccc),
                c: IrValue::ImmU32(0xf0f0_f0f0),
                lut: 0xf8,
            },
            Some(2),
        );
        program.emit(
            IrOp::ILop3 {
                a: IrValue::ImmU32(0xaaaa_aaaa),
                b: IrValue::ImmU32(0xcccc_cccc),
                c: IrValue::ImmU32(0xf0f0_f0f0),
                lut: 0x96,
            },
            Some(3),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let emitted =
            emit_compute(&cfg, &ComputeOptions::default()).expect("integer compute module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();

        assert!(instructions.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::ExtInst
                && instruction.operands.get(1)
                    == Some(&Operand::LiteralExtInstInteger(GLOp::FindUMsb as u32))
        }));
        for opcode in [
            rspirv::spirv::Op::BitCount,
            rspirv::spirv::Op::BitwiseAnd,
            rspirv::spirv::Op::BitwiseOr,
            rspirv::spirv::Op::BitwiseXor,
        ] {
            assert!(
                instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == opcode),
                "missing {opcode:?}"
            );
        }
    }

    #[test]
    fn compute_subgroup_ir_emits_lane_masks_ballot_all_any_and_equal() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(IrOp::SubgroupLaneId, Some(0));
        for (register, kind) in [
            (1, SubgroupMask::Eq),
            (2, SubgroupMask::Lt),
            (3, SubgroupMask::Le),
            (4, SubgroupMask::Gt),
            (5, SubgroupMask::Ge),
        ] {
            program.emit(IrOp::SubgroupMask { kind }, Some(register));
        }
        program.emit_pred(
            IrOp::SubgroupVote {
                source_pred: nexium_shader::Predicate {
                    idx: 7,
                    negate: false,
                },
                mode: VoteMode::All,
                pred_dest: 0,
                old: IrValue::Zero,
            },
            Some(6),
            None,
        );
        program.emit_pred(
            IrOp::SubgroupVote {
                source_pred: nexium_shader::Predicate {
                    idx: 0,
                    negate: false,
                },
                mode: VoteMode::Any,
                pred_dest: 7,
                old: IrValue::Zero,
            },
            Some(7),
            None,
        );
        program.emit_pred(
            IrOp::SubgroupVote {
                source_pred: nexium_shader::Predicate {
                    idx: 0,
                    negate: false,
                },
                mode: VoteMode::Equal,
                pred_dest: 7,
                old: IrValue::Zero,
            },
            Some(8),
            None,
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [32, 1, 1],
                ..ComputeOptions::default()
            },
        )
        .expect("subgroup compute module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");

        for capability in [
            Capability::GroupNonUniform,
            Capability::GroupNonUniformBallot,
            Capability::GroupNonUniformVote,
        ] {
            assert!(module
                .capabilities
                .iter()
                .any(|instruction| { instruction.operands == [Operand::Capability(capability)] }));
        }
        assert!(module.annotations.iter().any(|annotation| {
            annotation.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
                && annotation.operands.get(2)
                    == Some(&Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId))
        }));
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        for opcode in [
            rspirv::spirv::Op::ShiftLeftLogical,
            rspirv::spirv::Op::ISub,
            rspirv::spirv::Op::BitwiseOr,
            rspirv::spirv::Op::Not,
            rspirv::spirv::Op::GroupNonUniformBallot,
            rspirv::spirv::Op::GroupNonUniformAll,
            rspirv::spirv::Op::GroupNonUniformAny,
            rspirv::spirv::Op::IEqual,
        ] {
            assert!(
                instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == opcode),
                "missing {opcode:?}"
            );
        }
    }

    #[test]
    fn compute_filtered_sample_uses_combined_sampler_and_explicit_lod_zero() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x80 / 4,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::SampleTexHandle {
                handle,
                dimension: ImageDimension::D2,
                u: IrValue::ImmF32(0.25),
                v: Some(IrValue::ImmF32(0.75)),
                w: None,
                implicit_lod: true,
                lod_bias: Some(IrValue::ImmF32(1.0)),
                explicit_lod: None,
                texel_offset: Some((IrValue::ImmU32(u32::MAX), IrValue::ImmU32(2))),
                dref: None,
                component: 2,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let options = ComputeOptions {
            texture_bound_cbuf: 3,
            resources: vec![ComputeImageResource {
                handle,
                binding: 1,
                kind: ComputeResourceKind::CombinedSampledImage,
                dimension: ImageDimension::D2,
                numeric_type: TextureNumericType::Float,
                texel_format: None,
            }],
            ..ComputeOptions::default()
        };

        let emitted = emit_compute(&cfg, &options).expect("filtered compute module");
        assert_eq!(
            emitted
                .descriptors
                .iter()
                .map(|descriptor| (descriptor.binding, descriptor.kind))
                .collect::<Vec<_>>(),
            vec![(1, ComputeDescriptorKind::CombinedSampledImage)]
        );
        assert_ne!(emitted.cbuf_bindings & (1 << 3), 0);
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypeSampledImage
        }));
        let sample = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .find(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::ImageSampleExplicitLod
            })
            .expect("explicit-LOD filtered sample");
        assert_eq!(
            sample.operands.get(2),
            Some(&Operand::ImageOperands(
                rspirv::spirv::ImageOperands::LOD | rspirv::spirv::ImageOperands::CONST_OFFSET
            ))
        );
        let Some(Operand::IdRef(lod_id)) = sample.operands.get(3) else {
            panic!("sample LOD must be an id");
        };
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.result_id == Some(*lod_id)
                && instruction.class.opcode == rspirv::spirv::Op::Constant
                && instruction.operands.last() == Some(&Operand::LiteralBit32(0))
        }));
    }

    #[test]
    fn compute_emits_vulkan_builtins_queries_and_typed_images() {
        let cfg = compute_test_cfg();
        for storage_numeric_type in [TextureNumericType::Float, TextureNumericType::Uint] {
            let emitted = emit_compute(&cfg, &compute_test_options(storage_numeric_type))
                .expect("compute module");
            assert_eq!(emitted.cbuf_size, COMPUTE_CBUF_SIZE);
            assert_eq!(
                emitted.cbuf_bindings & ((1 << 2) | (1 << 3)),
                (1 << 2) | (1 << 3)
            );
            assert_eq!(
                emitted
                    .descriptors
                    .iter()
                    .map(|descriptor| (descriptor.binding, descriptor.kind))
                    .collect::<Vec<_>>(),
                vec![
                    (1, ComputeDescriptorKind::UniformTexelBuffer),
                    (2, ComputeDescriptorKind::SampledImage),
                    (3, ComputeDescriptorKind::StorageImage),
                    (
                        compute_cbuf_descriptor_binding(3).unwrap(),
                        ComputeDescriptorKind::UniformBuffer,
                    ),
                ]
            );
            validates_with_spirv_val_if_available(&emitted.words);

            let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
            assert!(module.entry_points.iter().any(|entry| {
                entry.operands.first() == Some(&Operand::ExecutionModel(ExecutionModel::GLCompute))
            }));
            assert!(module.extensions.iter().any(|extension| {
                extension.operands == [Operand::LiteralString("SPV_KHR_float_controls".to_string())]
            }));
            assert!(module.capabilities.iter().any(|capability| {
                capability.operands == [Operand::Capability(Capability::SignedZeroInfNanPreserve)]
            }));
            assert!(module.execution_modes.iter().any(|mode| {
                matches!(
                    mode.operands.as_slice(),
                    [
                        Operand::IdRef(_),
                        Operand::ExecutionMode(
                            rspirv::spirv::ExecutionMode::SignedZeroInfNanPreserve
                        ),
                        Operand::LiteralBit32(32)
                    ]
                )
            }));
            assert!(module.execution_modes.iter().any(|mode| {
                matches!(
                    mode.operands.as_slice(),
                    [
                        Operand::IdRef(_),
                        Operand::ExecutionMode(rspirv::spirv::ExecutionMode::LocalSize),
                        Operand::LiteralBit32(64),
                        Operand::LiteralBit32(1),
                        Operand::LiteralBit32(1)
                    ]
                )
            }));
            for builtin in [BuiltIn::LocalInvocationId, BuiltIn::WorkgroupId] {
                assert!(module.annotations.iter().any(|annotation| {
                    annotation.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
                        && annotation.operands.get(2) == Some(&Operand::BuiltIn(builtin))
                }));
            }
            let instructions = module
                .functions
                .iter()
                .flat_map(|function| &function.blocks)
                .flat_map(|block| &block.instructions)
                .collect::<Vec<_>>();
            for opcode in [
                rspirv::spirv::Op::ImageFetch,
                rspirv::spirv::Op::ImageQuerySizeLod,
                rspirv::spirv::Op::ImageQueryLevels,
                rspirv::spirv::Op::ImageWrite,
            ] {
                assert!(instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == opcode));
            }
            assert!(module.types_global_values.iter().any(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::TypeImage
                    && instruction.operands.get(1)
                        == Some(&Operand::Dim(rspirv::spirv::Dim::DimBuffer))
            }));
            assert!(module.types_global_values.iter().any(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::TypeImage
                    && instruction.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::Dim3D))
            }));
        }
    }

    #[test]
    fn compute_sust_buffer_emits_exact_storage_texel_formats() {
        let resources = [
            (
                TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x20,
                },
                TextureNumericType::Float,
                ComputeTexelFormat::R32Float,
                ImageFormat::R32f,
            ),
            (
                TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x24,
                },
                TextureNumericType::Uint,
                ComputeTexelFormat::R32Uint,
                ImageFormat::R32ui,
            ),
            (
                TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x28,
                },
                TextureNumericType::Sint,
                ComputeTexelFormat::R32Sint,
                ImageFormat::R32i,
            ),
            (
                TextureHandleOrigin::Bound {
                    cbuf_word_offset: 0x2c,
                },
                TextureNumericType::Uint,
                ComputeTexelFormat::R16Uint,
                ImageFormat::R16ui,
            ),
        ];
        let mut program = nexium_shader::IrProgram::new();
        for (index, (handle, _, _, _)) in resources.iter().copied().enumerate() {
            program.emit_void(IrOp::ImageWrite {
                handle,
                dimension: ImageDimension::Buffer,
                x: IrValue::ImmU32(index as u32),
                y: None,
                z: None,
                values: [
                    IrValue::ImmU32(0x3f80_0000 + index as u32),
                    IrValue::Zero,
                    IrValue::Zero,
                    IrValue::Zero,
                ],
            });
        }
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let options = ComputeOptions {
            local_size: [8, 1, 1],
            resources: resources
                .iter()
                .enumerate()
                .map(
                    |(index, (handle, numeric_type, texel_format, _))| ComputeImageResource {
                        handle: *handle,
                        binding: index as u32 + 1,
                        kind: ComputeResourceKind::StorageTexelBuffer,
                        dimension: ImageDimension::Buffer,
                        numeric_type: *numeric_type,
                        texel_format: Some(*texel_format),
                    },
                )
                .collect(),
            ..ComputeOptions::default()
        };
        let emitted = emit_compute(&cfg, &options).expect("typed SUST buffer module");
        assert_eq!(
            emitted
                .descriptors
                .iter()
                .map(|descriptor| descriptor.kind)
                .collect::<Vec<_>>(),
            vec![ComputeDescriptorKind::StorageTexelBuffer; 4]
        );
        assert_eq!(
            emitted
                .descriptors
                .iter()
                .map(|descriptor| descriptor.texel_format)
                .collect::<Vec<_>>(),
            resources
                .iter()
                .map(|(_, _, texel_format, _)| Some(*texel_format))
                .collect::<Vec<_>>()
        );
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.capabilities.iter().any(|instruction| {
            instruction.operands == [Operand::Capability(Capability::StorageImageExtendedFormats)]
        }));
        for (_, numeric_type, _, format) in resources {
            let scalar_is_expected = |id: Word| {
                module.types_global_values.iter().any(|instruction| {
                    instruction.result_id == Some(id)
                        && match numeric_type {
                            TextureNumericType::Float => {
                                instruction.class.opcode == rspirv::spirv::Op::TypeFloat
                            }
                            TextureNumericType::Uint => {
                                instruction.class.opcode == rspirv::spirv::Op::TypeInt
                                    && instruction.operands.get(1)
                                        == Some(&Operand::LiteralBit32(0))
                            }
                            TextureNumericType::Sint => {
                                instruction.class.opcode == rspirv::spirv::Op::TypeInt
                                    && instruction.operands.get(1)
                                        == Some(&Operand::LiteralBit32(1))
                            }
                        }
                })
            };
            assert!(module.types_global_values.iter().any(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::TypeImage
                    && instruction.operands.get(1)
                        == Some(&Operand::Dim(rspirv::spirv::Dim::DimBuffer))
                    && instruction.operands.get(5) == Some(&Operand::LiteralBit32(2))
                    && instruction.operands.get(6) == Some(&Operand::ImageFormat(format))
                    && matches!(instruction.operands.first(), Some(Operand::IdRef(id)) if scalar_is_expected(*id))
            }));
        }
        let image_writes = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ImageWrite)
            .count();
        assert_eq!(image_writes, 4);
    }

    #[test]
    fn compute_uniform_rgba32_float_texel_buffer_emits_sampled_buffer() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x30,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::TexelFetchHandle {
                handle,
                dimension: ImageDimension::D1,
                x: IrValue::ImmU32(7),
                y: None,
                z: None,
                component: 3,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                resources: vec![ComputeImageResource {
                    handle,
                    binding: 1,
                    kind: ComputeResourceKind::UniformTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Float,
                    texel_format: Some(ComputeTexelFormat::Rgba32Float),
                }],
                ..ComputeOptions::default()
            },
        )
        .expect("RGBA32 float uniform texel-buffer module");
        assert_eq!(
            emitted.descriptors[0].texel_format,
            Some(ComputeTexelFormat::Rgba32Float)
        );
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.capabilities.iter().any(|instruction| {
            instruction.operands == [Operand::Capability(Capability::SampledBuffer)]
        }));
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypeImage
                && instruction.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimBuffer))
                && instruction.operands.get(5) == Some(&Operand::LiteralBit32(1))
                && instruction.operands.get(6) == Some(&Operand::ImageFormat(ImageFormat::Unknown))
        }));
        assert!(module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::ImageFetch));
    }

    #[test]
    fn compute_storage_texel_atomics_emit_valid_vulkan_spirv() {
        let handle = TextureHandleOrigin::Bindless {
            cbuf_binding: 2,
            cbuf_word_offset: 0x120 / 4,
            cbuf_secondary_word_offset: None,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit_void(IrOp::ImageWrite {
            handle,
            dimension: ImageDimension::Buffer,
            x: IrValue::GprIn(6),
            y: None,
            z: None,
            values: [
                IrValue::GprIn(3),
                IrValue::Zero,
                IrValue::Zero,
                IrValue::Zero,
            ],
        });
        program.emit(
            IrOp::ImageAtomic {
                handle,
                dimension: ImageDimension::Buffer,
                x: IrValue::GprIn(7),
                y: None,
                z: None,
                value: IrValue::GprIn(2),
                op: ImageAtomicOp::Add,
                data_type: ImageAtomicType::Sd32,
            },
            Some(1),
        );
        program.emit_pred(
            IrOp::ImageAtomic {
                handle,
                dimension: ImageDimension::Buffer,
                x: IrValue::GprIn(4),
                y: None,
                z: None,
                value: IrValue::GprIn(9),
                op: ImageAtomicOp::Exchange,
                data_type: ImageAtomicType::Sd32,
            },
            None,
            Some(nexium_shader::Predicate {
                idx: 0,
                negate: false,
            }),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let options = ComputeOptions {
            local_size: [32, 1, 1],
            texture_bound_cbuf: 2,
            resources: vec![ComputeImageResource {
                handle,
                binding: 1,
                kind: ComputeResourceKind::StorageTexelBuffer,
                dimension: ImageDimension::Buffer,
                numeric_type: TextureNumericType::Uint,
                texel_format: Some(ComputeTexelFormat::R32Uint),
            }],
            ..ComputeOptions::default()
        };

        let emitted = emit_compute(&cfg, &options).expect("storage texel atomic module");
        assert_eq!(
            emitted
                .descriptors
                .iter()
                .map(|descriptor| (descriptor.binding, descriptor.kind))
                .collect::<Vec<_>>(),
            vec![(1, ComputeDescriptorKind::StorageTexelBuffer)]
        );
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.capabilities.iter().any(|instruction| {
            instruction.operands.first() == Some(&Operand::Capability(Capability::ImageBuffer))
        }));
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypeImage
                && instruction.operands.get(1) == Some(&Operand::Dim(rspirv::spirv::Dim::DimBuffer))
                && instruction.operands.get(5) == Some(&Operand::LiteralBit32(2))
                && instruction.operands.get(6) == Some(&Operand::ImageFormat(ImageFormat::R32ui))
        }));
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypePointer
                && instruction.operands.first() == Some(&Operand::StorageClass(StorageClass::Image))
        }));

        let constants = module
            .types_global_values
            .iter()
            .filter_map(|instruction| {
                match (
                    instruction.class.opcode,
                    instruction.result_id,
                    instruction.operands.as_slice(),
                ) {
                    (rspirv::spirv::Op::Constant, Some(id), [Operand::LiteralBit32(value)]) => {
                        Some((id, *value))
                    }
                    _ => None,
                }
            })
            .collect::<std::collections::HashMap<_, _>>();
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        for opcode in [
            rspirv::spirv::Op::AtomicIAdd,
            rspirv::spirv::Op::AtomicExchange,
        ] {
            let atomic = instructions
                .iter()
                .find(|instruction| instruction.class.opcode == opcode)
                .unwrap_or_else(|| panic!("missing {opcode:?}"));
            let [_, Operand::IdScope(scope), Operand::IdMemorySemantics(semantics), _] =
                atomic.operands.as_slice()
            else {
                panic!("unexpected {opcode:?} operands: {:?}", atomic.operands);
            };
            assert_eq!(constants[scope], Scope::Device as u32);
            assert_eq!(constants[semantics], MemorySemantics::NONE.bits());
        }
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::ImageTexelPointer
                })
                .count(),
            2
        );
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::SelectionMerge));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::Phi));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::ImageWrite));
    }

    #[test]
    fn compute_storage_texel_atomic_matrix_preserves_types_and_wrap_semantics() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x24,
        };
        let cases = [
            (ImageAtomicOp::Add, ImageAtomicType::U32),
            (ImageAtomicOp::Min, ImageAtomicType::U32),
            (ImageAtomicOp::Min, ImageAtomicType::S32),
            (ImageAtomicOp::Min, ImageAtomicType::Sd32),
            (ImageAtomicOp::Max, ImageAtomicType::U32),
            (ImageAtomicOp::Max, ImageAtomicType::S32),
            (ImageAtomicOp::Max, ImageAtomicType::Sd32),
            (ImageAtomicOp::Increment, ImageAtomicType::U32),
            (ImageAtomicOp::Decrement, ImageAtomicType::S32),
            (ImageAtomicOp::And, ImageAtomicType::U32),
            (ImageAtomicOp::Or, ImageAtomicType::S32),
            (ImageAtomicOp::Xor, ImageAtomicType::Sd32),
            (ImageAtomicOp::Exchange, ImageAtomicType::U32),
        ];
        let mut program = nexium_shader::IrProgram::new();
        for (index, (op, data_type)) in cases.into_iter().enumerate() {
            program.emit(
                IrOp::ImageAtomic {
                    handle,
                    dimension: ImageDimension::Buffer,
                    x: IrValue::ImmU32(index as u32),
                    y: None,
                    z: None,
                    value: IrValue::ImmU32(7),
                    op,
                    data_type,
                },
                Some(index as u8),
            );
        }
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [32, 1, 1],
                resources: vec![ComputeImageResource {
                    handle,
                    binding: 1,
                    kind: ComputeResourceKind::StorageTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: Some(ComputeTexelFormat::R32Uint),
                }],
                ..ComputeOptions::default()
            },
        )
        .expect("complete 32-bit surface-atomic matrix");
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let count = |opcode| {
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == opcode)
                .count()
        };
        assert_eq!(count(rspirv::spirv::Op::AtomicIAdd), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicUMin), 2);
        assert_eq!(count(rspirv::spirv::Op::AtomicSMin), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicUMax), 2);
        assert_eq!(count(rspirv::spirv::Op::AtomicSMax), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicAnd), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicOr), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicXor), 1);
        assert_eq!(count(rspirv::spirv::Op::AtomicExchange), 1);

        assert_eq!(count(rspirv::spirv::Op::AtomicLoad), 2);
        assert_eq!(count(rspirv::spirv::Op::AtomicCompareExchange), 2);
        assert_eq!(count(rspirv::spirv::Op::LoopMerge), 2);
        assert_eq!(count(rspirv::spirv::Op::UGreaterThanEqual), 1);
        assert_eq!(count(rspirv::spirv::Op::UGreaterThan), 1);
        assert_eq!(count(rspirv::spirv::Op::IEqual), 3);
        assert_eq!(count(rspirv::spirv::Op::LogicalOr), 1);
        assert_eq!(count(rspirv::spirv::Op::IAdd), 1);
        assert_eq!(count(rspirv::spirv::Op::ISub), 1);
    }

    #[test]
    fn predicated_increment_suppresses_the_entire_compare_exchange_loop() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x24,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit_pred(
            IrOp::ImageAtomic {
                handle,
                dimension: ImageDimension::Buffer,
                x: IrValue::GprIn(4),
                y: None,
                z: None,
                value: IrValue::GprIn(5),
                op: ImageAtomicOp::Increment,
                data_type: ImageAtomicType::U32,
            },
            None,
            Some(nexium_shader::Predicate {
                idx: 0,
                negate: true,
            }),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                resources: vec![ComputeImageResource {
                    handle,
                    binding: 1,
                    kind: ComputeResourceKind::StorageTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: Some(ComputeTexelFormat::R32Uint),
                }],
                ..ComputeOptions::default()
            },
        )
        .expect("predicated IWRAP module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::AtomicCompareExchange
                })
                .count(),
            1
        );
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::SelectionMerge));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::LoopMerge));
    }

    #[test]
    fn compute_storage_texel_atomic_metadata_fails_closed() {
        let handle = TextureHandleOrigin::Bound {
            cbuf_word_offset: 0x24,
        };
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::ImageAtomic {
                handle,
                dimension: ImageDimension::Buffer,
                x: IrValue::Zero,
                y: None,
                z: None,
                value: IrValue::ImmU32(1),
                op: ImageAtomicOp::Add,
                data_type: ImageAtomicType::U32,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let resource = ComputeImageResource {
            handle,
            binding: 1,
            kind: ComputeResourceKind::StorageTexelBuffer,
            dimension: ImageDimension::Buffer,
            numeric_type: TextureNumericType::Uint,
            texel_format: Some(ComputeTexelFormat::R32Uint),
        };

        assert!(matches!(
            emit_compute(&cfg, &ComputeOptions::default()),
            Err(ComputeEmitError::MissingResource {
                kind: ComputeResourceKind::StorageTexelBuffer,
                ..
            })
        ));
        let mut missing_format = resource;
        missing_format.texel_format = None;
        assert!(matches!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    resources: vec![missing_format],
                    ..ComputeOptions::default()
                }
            ),
            Err(ComputeEmitError::UnsupportedOperation(_))
        ));
        let mut non_atomic_format = resource;
        non_atomic_format.texel_format = Some(ComputeTexelFormat::R16Uint);
        assert!(matches!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    resources: vec![non_atomic_format],
                    ..ComputeOptions::default()
                }
            ),
            Err(ComputeEmitError::UnsupportedOperation(_))
        ));
        let mut wrong_type = resource;
        wrong_type.numeric_type = TextureNumericType::Sint;
        assert!(matches!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    resources: vec![wrong_type],
                    ..ComputeOptions::default()
                }
            ),
            Err(ComputeEmitError::UnsupportedOperation(_))
        ));
        let mut wrong_dimension = resource;
        wrong_dimension.dimension = ImageDimension::D2;
        assert!(matches!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    resources: vec![wrong_dimension],
                    ..ComputeOptions::default()
                }
            ),
            Err(ComputeEmitError::InvalidResourceDimension {
                kind: ComputeResourceKind::StorageTexelBuffer,
                ..
            })
        ));
    }

    #[test]
    fn captured_suatom_instructions_translate_through_generic_compute_spirv() {
        let static_ldc = |dest: u8| {
            (0xEF94u64 << 48)
                | (2u64 << 36)
                | (0x120u64 << 20)
                | (7u64 << 16)
                | (0xffu64 << 8)
                | u64::from(dest)
        };
        let bytes = build_test_program(&[
            static_ldc(7),
            0xea70_0382_0020_0502,
            static_ldc(4),
            0xea70_0203_00a7_0d04,
            static_ldc(11),
            0xea70_0583_0097_0404,
            static_ldc(5),
            0xea70_0282_0022_0701,
            static_ldc(5),
            0xea70_0282_0022_0705,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_compute_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let handle = TextureHandleOrigin::Bindless {
            cbuf_binding: 2,
            cbuf_word_offset: 0x120 / 4,
            cbuf_secondary_word_offset: None,
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [32, 1, 1],
                texture_bound_cbuf: 2,
                resources: vec![ComputeImageResource {
                    handle,
                    binding: 1,
                    kind: ComputeResourceKind::StorageTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: Some(ComputeTexelFormat::R32Uint),
                }],
                ..ComputeOptions::default()
            },
        )
        .expect("captured SUATOM module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::AtomicIAdd)
                .count(),
            3
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::AtomicExchange
                })
                .count(),
            2
        );
    }

    #[test]
    fn captured_pps_77900_lut_kernel_emits_generic_compute_spirv() {
        let bytes = build_test_program(&[
            0xf0c8_0000_0257_0000,
            0xf0c8_0000_0217_0001,
            0x5c18_0300_0017_0000,
            0x4b62_038c_0007_0007,
            0xe300_0000_0008_000f,
            0x0100_0000_0017_f003,
            0x3600_7f80_0037_0001,
            0x4c98_0788_05a7_0007,
            0x3600_0180_0037_0005,
            0x3620_0090_0037_0004,
            0x4c47_0208_15a7_0707,
            0x3620_0290_0037_0005,
            0xdd3a_0000_8077_0404,
            0x1c00_0000_0017_0506,
            0xdd3a_0000_8077_0505,
            0xdd3a_0000_8077_0606,
            0x4c10_000c_0047_0000,
            0x4c98_0788_0487_000b,
            0x3828_0000_0087_0001,
            0x3828_0000_0107_0002,
            0x0400_0000_0ff7_0000,
            0x0400_0000_0ff7_0101,
            0xeb20_058a_00f7_0400,
            0xe300_0000_0007_000f,
        ]);
        let cfg = nexium_shader::build_compute_cfg(&bytes);
        assert_eq!(
            cfg.unimplemented, 0,
            "captured LUT kernel must fully translate"
        );
        let options = ComputeOptions {
            local_size: [64, 1, 1],
            local_memory_low_size: 0,
            local_memory_high_size: 0,
            local_memory_crs_size: 0,
            shared_memory_size: 0,
            texture_bound_cbuf: 2,
            cbuf_sizes: [COMPUTE_CBUF_MAX_SIZE; COMPUTE_CBUF_SLOTS],
            resources: vec![
                ComputeImageResource {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x168 / 4,
                        cbuf_secondary_word_offset: Some(0x568 / 4),
                    },
                    binding: 1,
                    kind: ComputeResourceKind::UniformTexelBuffer,
                    dimension: ImageDimension::Buffer,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: Some(ComputeTexelFormat::R32Uint),
                },
                ComputeImageResource {
                    handle: TextureHandleOrigin::Bindless {
                        cbuf_binding: 2,
                        cbuf_word_offset: 0x120 / 4,
                        cbuf_secondary_word_offset: None,
                    },
                    binding: 2,
                    kind: ComputeResourceKind::StorageImage,
                    dimension: ImageDimension::D3,
                    numeric_type: TextureNumericType::Uint,
                    texel_format: None,
                },
            ],
        };
        let emitted = emit_compute(&cfg, &options).expect("captured LUT compute module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ImageFetch)
                .count(),
            3
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ImageWrite)
                .count(),
            1
        );
    }

    #[test]
    fn compute_shared_memory_and_barriers_emit_valid_vulkan_spirv() {
        let mut program = nexium_shader::IrProgram::new();
        let value = program.emit(
            IrOp::LoadShared {
                addr: IrValue::ImmU32(0x10),
            },
            Some(0),
        );
        program.emit_void(IrOp::StoreShared {
            addr: IrValue::ImmU32(0x14),
            value: IrValue::Inst(value),
        });
        program.emit_void(IrOp::WorkgroupBarrier);
        program.emit_void(IrOp::MemoryBarrier {
            scope: MemoryBarrierScope::Workgroup,
        });
        program.emit_void(IrOp::MemoryBarrier {
            scope: MemoryBarrierScope::Device,
        });
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let options = ComputeOptions {
            local_size: [8, 4, 1],
            shared_memory_size: 0x201,
            ..ComputeOptions::default()
        };

        let emitted = emit_compute(&cfg, &options).expect("shared-memory compute module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::Variable
                && instruction.operands.first()
                    == Some(&Operand::StorageClass(StorageClass::Workgroup))
        }));

        let constants = module
            .types_global_values
            .iter()
            .filter_map(|instruction| {
                match (
                    instruction.class.opcode,
                    instruction.result_id,
                    instruction.operands.as_slice(),
                ) {
                    (rspirv::spirv::Op::Constant, Some(id), [Operand::LiteralBit32(value)]) => {
                        Some((id, *value))
                    }
                    _ => None,
                }
            })
            .collect::<std::collections::HashMap<_, _>>();
        assert!(constants
            .values()
            .any(|value| *value == 0x201u32.div_ceil(4)));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ControlBarrier)
                .count(),
            1
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::MemoryBarrier)
                .count(),
            2
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::Store)
                .count(),
            1
        );

        let control = instructions
            .iter()
            .find(|instruction| instruction.class.opcode == rspirv::spirv::Op::ControlBarrier)
            .expect("workgroup control barrier");
        let control_values = control
            .operands
            .iter()
            .map(|operand| match operand {
                Operand::IdScope(id) | Operand::IdMemorySemantics(id) => constants[id],
                other => panic!("unexpected barrier operand {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(control_values[0], Scope::Workgroup as u32);
        assert_eq!(control_values[1], Scope::Workgroup as u32);
        assert_eq!(
            control_values[2],
            (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits()
        );

        let memory_scopes = instructions
            .iter()
            .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::MemoryBarrier)
            .map(|instruction| match instruction.operands.first() {
                Some(Operand::IdScope(id)) => constants[id],
                other => panic!("unexpected memory-barrier scope {other:?}"),
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            memory_scopes,
            [Scope::Device as u32, Scope::Workgroup as u32]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn vertex_uint_tld_b_3d_uses_per_slot_binding_16() {
        let bytes = build_test_program(&[
            enc_static_ldc(39, 3, 0x90),
            0xdd38_0007_c277_0000,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_cfg(&bytes);
        let (words, _, _) = emit_vertex_with_bindings_opts(
            &cfg,
            &[],
            VertexOptions {
                tex_slot_base: 4,
                texture_numeric_manifest: vec![GraphicsTextureResource::new(
                    nexium_shader::bindless_texture_id(3, 0x24),
                    4,
                    TextureNumericType::Uint,
                )
                .with_image_kind(GraphicsImageKind::D3)],
                ..VertexOptions::default()
            },
        );
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let image = image_type_for_binding(&module, GFX_BINDING_UINT_3D);
        assert_eq!(
            image.operands[1],
            Operand::Dim(rspirv::spirv::Dim::Dim3D)
        );
        let scalar = image_scalar_type_for_binding(&module, GFX_BINDING_UINT_3D);
        assert_eq!(scalar.class.opcode, rspirv::spirv::Op::TypeInt);
        assert_eq!(scalar.operands[1], Operand::LiteralBit32(0));
    }

    #[test]
    fn shared_atomic_or_is_predicated_bounded_and_uses_workgroup_scope() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit_pred(
            IrOp::SharedAtomic {
                addr: IrValue::GprIn(4),
                value: IrValue::GprIn(8),
                op: ImageAtomicOp::Or,
            },
            Some(0),
            Some(nexium_shader::Predicate {
                idx: 1,
                negate: false,
            }),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [64, 1, 1],
                shared_memory_size: 0x100,
                ..ComputeOptions::default()
            },
        )
        .expect("bounded shared atomic module");
        validates_with_spirv_val_if_available(&emitted.words);

        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let constants = module
            .types_global_values
            .iter()
            .filter_map(|instruction| match (
                instruction.class.opcode,
                instruction.result_id,
                instruction.operands.as_slice(),
            ) {
                (rspirv::spirv::Op::Constant, Some(id), [Operand::LiteralBit32(value)]) => {
                    Some((id, *value))
                }
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        let blocks = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .collect::<Vec<_>>();
        let atomic_block = blocks
            .iter()
            .copied()
            .find(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::AtomicOr)
            })
            .expect("shared atomic block");
        assert!(atomic_block.instructions.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::AccessChain
        }));
        let atomic = atomic_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == rspirv::spirv::Op::AtomicOr)
            .expect("shared AtomicOr");
        let [_, Operand::IdScope(scope), Operand::IdMemorySemantics(semantics), _] =
            atomic.operands.as_slice()
        else {
            panic!("unexpected AtomicOr operands: {:?}", atomic.operands);
        };
        assert_eq!(constants[scope], Scope::Workgroup as u32);
        assert_eq!(constants[semantics], MemorySemantics::NONE.bits());

        let instructions = blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::ULessThan));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::SelectionMerge));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::Phi));
    }

    #[test]
    fn captured_atoms_or_u32_translates_through_compute_spirv() {
        let bytes = build_test_program(&[0xec60_0000_0081_ffff, enc_exit()]);
        let cfg = nexium_shader::build_compute_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [64, 1, 1],
                shared_memory_size: 0x1000,
                ..ComputeOptions::default()
            },
        )
        .expect("captured ATOMS module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert_eq!(
            module
                .functions
                .iter()
                .flat_map(|function| &function.blocks)
                .flat_map(|block| &block.instructions)
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::AtomicOr)
                .count(),
            1
        );
    }

    #[test]
    fn captured_shared_sass_translates_through_generic_compute_spirv() {
        let bytes = build_test_program(&[
            0xef5e_0000_0007_0d08,
            0xf0a8_1b80_0007_0000,
            0xef4e_1000_8007_0d00,
            0xef98_0000_0007_0000,
            enc_exit(),
        ]);
        let cfg = nexium_shader::build_compute_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_size: [16, 16, 1],
                shared_memory_size: 0x1000,
                ..ComputeOptions::default()
            },
        )
        .expect("captured shared-memory module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ControlBarrier)
                .count(),
            1
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::MemoryBarrier)
                .count(),
            1
        );
        assert_eq!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::Store)
                .count(),
            4
        );
    }

    #[test]
    fn compute_local_memory_uses_qmd_size_and_guards_oob_without_wrapping() {
        let mut program = nexium_shader::IrProgram::new();
        let loaded = program.emit(
            IrOp::LoadLocal {
                addr: IrValue::GprIn(0),
            },
            Some(0),
        );
        program.emit_void(IrOp::StoreLocal {
            addr: IrValue::GprIn(1),
            value: IrValue::Inst(loaded),
        });
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(
            &cfg,
            &ComputeOptions {
                local_memory_low_size: 0x14,
                local_memory_high_size: 0x10,
                ..ComputeOptions::default()
            },
        )
        .expect("QMD-sized local-memory module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let constants = module
            .types_global_values
            .iter()
            .filter_map(|instruction| {
                match (
                    instruction.class.opcode,
                    instruction.result_id,
                    instruction.operands.as_slice(),
                ) {
                    (rspirv::spirv::Op::Constant, Some(id), [Operand::LiteralBit32(value)]) => {
                        Some((id, *value))
                    }
                    _ => None,
                }
            })
            .collect::<std::collections::HashMap<_, _>>();
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::TypeArray
                && matches!(
                    instruction.operands.get(1),
                    Some(Operand::IdRef(length)) if constants.get(length) == Some(&9)
                )
        }));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::ULessThan)
                .count()
                >= 2
        );
        assert!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::Select)
                .count()
                >= 4
        );
        assert!(!instructions.iter().any(|instruction| {
            matches!(
                instruction.class.opcode,
                rspirv::spirv::Op::UMod | rspirv::spirv::Op::BitwiseAnd
            )
        }));
    }

    #[test]
    fn compute_local_memory_metadata_fails_closed() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::LoadLocal {
                addr: IrValue::Zero,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert_eq!(
            emit_compute(&cfg, &ComputeOptions::default()),
            Err(ComputeEmitError::MissingLocalMemory)
        );
        assert_eq!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    local_memory_low_size: u32::MAX,
                    local_memory_high_size: 1,
                    ..ComputeOptions::default()
                },
            ),
            Err(ComputeEmitError::InvalidLocalMemoryAllocation {
                low: u32::MAX,
                high: 1,
            })
        );
        assert_eq!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    local_memory_low_size: MAX_COMPUTE_LOCAL_MEMORY_SIZE,
                    local_memory_high_size: 4,
                    ..ComputeOptions::default()
                },
            ),
            Err(ComputeEmitError::InvalidLocalMemorySize(
                MAX_COMPUTE_LOCAL_MEMORY_SIZE + 4
            ))
        );
    }

    #[test]
    fn compute_shared_memory_metadata_fails_closed() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::LoadShared {
                addr: IrValue::Zero,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert_eq!(
            emit_compute(&cfg, &ComputeOptions::default()),
            Err(ComputeEmitError::MissingSharedMemory)
        );

        let mut atomic = nexium_shader::IrProgram::new();
        atomic.emit(
            IrOp::SharedAtomic {
                addr: IrValue::Zero,
                value: IrValue::ImmU32(1),
                op: ImageAtomicOp::Or,
            },
            Some(0),
        );
        let atomic_cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, atomic)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert_eq!(
            emit_compute(&atomic_cfg, &ComputeOptions::default()),
            Err(ComputeEmitError::MissingSharedMemory)
        );

        let mut unsupported_atomic = nexium_shader::IrProgram::new();
        unsupported_atomic.emit(
            IrOp::SharedAtomic {
                addr: IrValue::Zero,
                value: IrValue::ImmU32(1),
                op: ImageAtomicOp::Add,
            },
            Some(0),
        );
        let unsupported_atomic_cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, unsupported_atomic)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert!(matches!(
            emit_compute(
                &unsupported_atomic_cfg,
                &ComputeOptions {
                    shared_memory_size: 4,
                    ..ComputeOptions::default()
                }
            ),
            Err(ComputeEmitError::UnsupportedOperation(_))
        ));
        assert_eq!(
            emit_compute(
                &cfg,
                &ComputeOptions {
                    shared_memory_size: MAX_COMPUTE_SHARED_MEMORY_SIZE + 1,
                    ..ComputeOptions::default()
                },
            ),
            Err(ComputeEmitError::InvalidSharedMemorySize(
                MAX_COMPUTE_SHARED_MEMORY_SIZE + 1
            ))
        );

        let mut predicated = nexium_shader::IrProgram::new();
        predicated.emit_void_pred(
            IrOp::WorkgroupBarrier,
            Some(nexium_shader::Predicate {
                idx: 0,
                negate: false,
            }),
        );
        let predicated_cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, predicated)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert!(matches!(
            emit_compute(&predicated_cfg, &ComputeOptions::default()),
            Err(ComputeEmitError::UnsupportedOperation(_))
        ));
    }

    #[test]
    fn compute_cbufs_use_independent_non_wrapping_descriptors() {
        let mut static_program = nexium_shader::IrProgram::new();
        static_program.emit(
            IrOp::LoadCbuf {
                binding: 2,
                byte_offset: 0x900,
            },
            Some(0),
        );
        let static_cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, static_program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let mut static_options = ComputeOptions::default();
        static_options.cbuf_sizes[2] = 0x1000;
        let emitted = emit_compute(&static_cfg, &static_options).expect("large static cbuf offset");
        assert_eq!(emitted.cbuf_required_sizes[2], 0x904);
        assert!(emitted.descriptors.iter().any(|descriptor| {
            descriptor.binding == compute_cbuf_descriptor_binding(2).unwrap()
                && descriptor.kind == ComputeDescriptorKind::UniformBuffer
        }));
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        assert!(module.types_global_values.iter().any(|instruction| {
            instruction.class.opcode == rspirv::spirv::Op::Constant
                && instruction.operands == [Operand::LiteralBit32(0x90)]
        }));

        static_options.cbuf_sizes[2] = 0x900;
        assert_eq!(
            emit_compute(&static_cfg, &static_options),
            Err(ComputeEmitError::CbufOutOfBounds {
                binding: 2,
                end: 0x904,
                available: 0x900,
            })
        );

        let mut indexed_program = nexium_shader::IrProgram::new();
        indexed_program.emit(
            IrOp::LoadCbufIndexed {
                binding: 3,
                byte_offset: 0x10,
                index: IrValue::GprIn(4),
                address_mode: CbufAddressMode::Default,
            },
            Some(0),
        );
        let indexed_cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, indexed_program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let mut indexed_options = ComputeOptions::default();
        indexed_options.cbuf_sizes[3] = 0x2800;
        let indexed = emit_compute(&indexed_cfg, &indexed_options).expect("indexed cbuf range");
        assert_eq!(indexed.cbuf_required_sizes[3], 0x2800);
        validates_with_spirv_val_if_available(&indexed.words);
        let indexed_module = rspirv::dr::load_words(&indexed.words).expect("valid SPIR-V");
        assert!(!indexed_module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::UMod));
    }

    #[test]
    fn graphics_cbufs_use_bounds_checked_directory_storage_buffer() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::LoadCbuf {
                binding: 6,
                byte_offset: 0x2aa0,
            },
            Some(0),
        );
        program.emit(
            IrOp::LoadCbuf {
                binding: 6,
                byte_offset: 0xffff_fffc,
            },
            Some(1),
        );
        program.emit(
            IrOp::LoadCbufIndexed {
                binding: 6,
                byte_offset: 0xffff_fff0,
                index: IrValue::ImmU32(0x10),
                address_mode: CbufAddressMode::Default,
            },
            Some(2),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };

        let words = try_emit_fragment(&cfg).expect("graphics directory cbuf module");
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        assert!(module.annotations.iter().any(|annotation| {
            annotation.operands.get(1) == Some(&Operand::Decoration(Decoration::BufferBlock))
        }));
        assert!(!module.annotations.iter().any(|annotation| {
            annotation.operands.get(1) == Some(&Operand::Decoration(Decoration::Block))
        }));

        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(!instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::UMod));
        assert!(instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::ULessThan));

        let safe_select = instructions
            .iter()
            .enumerate()
            .filter_map(|(index, instruction)| {
                (instruction.class.opcode == rspirv::spirv::Op::Select)
                    .then_some((index, instruction.result_id?))
            })
            .find(|(select_index, result)| {
                instructions.iter().skip(select_index + 1).any(|instruction| {
                    instruction.class.opcode == rspirv::spirv::Op::AccessChain
                        && instruction.operands.contains(&Operand::IdRef(*result))
                })
            });
        assert!(
            safe_select.is_some(),
            "the bounds-selected index must feed a later OpAccessChain"
        );

        for literal in [44, 45, GFX_CBUF_ZERO_WORD, 0x2aa0, 0xffff_fffc] {
            assert!(module.types_global_values.iter().any(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::Constant
                    && instruction.operands == [Operand::LiteralBit32(literal)]
            }), "missing cbuf ABI/address constant {literal:#x}");
        }
    }

    #[test]
    fn captured_b64_cbuf_halves_each_use_an_independent_safe_index() {
        let bytes = build_test_program(&[0xef95_0062_aa07_2500, enc_exit()]);
        let cfg = nexium_shader::build_fragment_cfg(&bytes);
        assert_eq!(cfg.unimplemented, 0);

        let words = try_emit_fragment(&cfg).expect("captured B64 graphics cbuf module");
        validates_with_spirv_val_if_available(&words);
        let module = rspirv::dr::load_words(&words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        assert!(!instructions
            .iter()
            .any(|instruction| instruction.class.opcode == rspirv::spirv::Op::UMod));

        let safe_indices = instructions
            .iter()
            .filter_map(|instruction| {
                (instruction.class.opcode == rspirv::spirv::Op::Select)
                    .then_some(instruction.result_id?)
            })
            .collect::<std::collections::HashSet<_>>();
        let accessed_safe_indices = instructions
            .iter()
            .filter(|instruction| instruction.class.opcode == rspirv::spirv::Op::AccessChain)
            .flat_map(|instruction| &instruction.operands)
            .filter_map(|operand| match operand {
                Operand::IdRef(id) if safe_indices.contains(id) => Some(*id),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            accessed_safe_indices.len(),
            2,
            "each B64 half must select its own payload-or-sentinel index"
        );
        assert!(
            instructions
                .iter()
                .filter(|instruction| instruction.class.opcode
                    == rspirv::spirv::Op::ULessThan)
                .count()
                >= 2
        );
        for offset in [0x2aa0, 0x2aa4] {
            assert!(module.types_global_values.iter().any(|instruction| {
                instruction.class.opcode == rspirv::spirv::Op::Constant
                    && instruction.operands == [Operand::LiteralBit32(offset)]
            }));
        }
    }

    #[test]
    fn graphics_cbuf_bank_sixteen_fails_closed() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::LoadCbuf {
                binding: 16,
                byte_offset: 0,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        assert_eq!(
            try_emit_vertex(&cfg),
            Err(SpirvEmitError::InvalidGraphicsCbufBinding(16))
        );
        assert_eq!(
            try_emit_fragment(&cfg),
            Err(SpirvEmitError::InvalidGraphicsCbufBinding(16))
        );
    }

    #[test]
    fn segmented_cbuf_addressing_fails_all_spirv_stages_closed() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::LoadCbufIndexed {
                binding: 3,
                byte_offset: 0xffff_fff0,
                index: IrValue::GprIn(4),
                address_mode: CbufAddressMode::Segmented,
            },
            Some(0),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let expected = SpirvEmitError::UnsupportedCbufAddressMode(CbufAddressMode::Segmented);
        assert_eq!(try_emit_vertex(&cfg), Err(expected));
        assert_eq!(try_emit_fragment(&cfg), Err(expected));

        let compute_error = emit_compute(&cfg, &ComputeOptions::default())
            .expect_err("segmented compute cbuf addressing must fail before lowering");
        match compute_error {
            ComputeEmitError::UnsupportedOperation(message) => {
                assert!(message.contains("Segmented"), "{message}");
            }
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }

        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emit_vertex(&cfg))).is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emit_fragment(&cfg))).is_err()
        );
    }

    #[test]
    fn compute_shuffle_up_checks_the_partition_minimum_lane() {
        let mut program = nexium_shader::IrProgram::new();
        program.emit(
            IrOp::Shfl {
                value: IrValue::GprIn(0),
                index: IrValue::ImmU32(1),
                mask: IrValue::ImmU32(0x1f),
                mode: 1,
                pred_dest: 7,
            },
            Some(1),
        );
        let cfg = Cfg {
            blocks: vec![cfg_block(0, BranchKind::Exit, program)],
            unimplemented: 0,
            bindless_or_partners: Default::default(),
        };
        let emitted = emit_compute(&cfg, &ComputeOptions::default()).expect("SHFL.UP module");
        validates_with_spirv_val_if_available(&emitted.words);
        let module = rspirv::dr::load_words(&emitted.words).expect("valid SPIR-V");
        let instructions = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .collect::<Vec<_>>();
        let comparison = instructions
            .iter()
            .find(|instruction| instruction.class.opcode == rspirv::spirv::Op::SGreaterThanEqual)
            .expect("SHFL.UP range comparison");
        let Operand::IdRef(min_lane) = comparison.operands[1] else {
            panic!("SHFL.UP comparison RHS is not an id")
        };
        let definition = instructions
            .iter()
            .find(|instruction| instruction.result_id == Some(min_lane))
            .expect("minimum-lane definition");
        assert_eq!(definition.class.opcode, rspirv::spirv::Op::BitwiseAnd);
    }

    #[test]
    fn compute_metadata_errors_fail_closed() {
        let cfg = compute_test_cfg();
        let missing = ComputeOptions {
            resources: Vec::new(),
            ..ComputeOptions::default()
        };
        assert!(matches!(
            emit_compute(&cfg, &missing),
            Err(ComputeEmitError::MissingResource { .. })
        ));

        let mut reserved = compute_test_options(TextureNumericType::Uint);
        reserved.resources[0].binding = 0;
        assert_eq!(
            emit_compute(&cfg, &reserved),
            Err(ComputeEmitError::ReservedImageBinding)
        );
    }
}

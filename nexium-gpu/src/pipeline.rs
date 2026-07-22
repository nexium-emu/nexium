use ash::vk;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn cache_path(device_tag: &str) -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")?;
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        PathBuf::from(base)
            .join("NeXium")
            .join("shader_cache")
            .join(device_tag)
            .join(format!("{}.bin", title)),
    )
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct PipelineKey {
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub topology: u32,
    pub color_format: u32,
    pub color_formats: [u32; 8],
    pub color_attachment_count: u32,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub vertex_layout_hash: u64,
    pub blend_signature: u64,
    pub raster_state_packed: u32,
    pub depth_state_packed: u32,
    pub depth_format: i32,
    pub depth_aspects: u32,
    pub stencil_enabled: bool,
    pub stencil_front: [u32; 7],
    pub stencil_back: [u32; 7],
    pub depth_clamp_enabled: bool,
    pub poly_offset_packed: u64,
    pub color_write_mask: u32,
}

const SPEC_VERSION: u32 = 30;
const KNOWN_DRIVER_HOSTILE_PIPELINES: &[(u64, u64)] =
    &[(0x59b9_0e74_4b2a_7537, 0xe505_d075_601e_e633)];

pub(crate) fn known_driver_hostile_pipeline(vs_hash: u64, fs_hash: u64) -> bool {
    KNOWN_DRIVER_HOSTILE_PIPELINES.contains(&(vs_hash, fs_hash))
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineSpec {
    pub key: PipelineKey,
    pub vs_spirv: Vec<u32>,
    pub fs_spirv: Vec<u32>,
    pub bindings: Vec<(u32, u32, u32)>,
    pub attrs: Vec<(u32, u32, i32, u32)>,
    pub topology: i32,
    pub color_format: i32,
    pub color_formats: Vec<i32>,
    pub color_attachment_count: u32,
    pub depth_format: i32,
    pub has_depth: bool,
    pub depth_aspects: u32,
    pub blend: (bool, i32, i32, i32, i32, i32, i32),
    pub blend_attachments: Vec<(bool, i32, i32, i32, i32, i32, i32, u32)>,
    pub color_write_mask: u32,
    pub depth: (bool, bool, i32),
    pub stencil_enabled: bool,
    pub stencil_front: (i32, i32, i32, i32, u32, u32, u32),
    pub stencil_back: (i32, i32, i32, i32, u32, u32, u32),
    pub depth_clamp_enabled: bool,
    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub depth_clip_control_enabled: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SpecFile {
    version: u32,
    specs: Vec<PipelineSpec>,
}

fn specs_path(device_tag: &str) -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")?;
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        PathBuf::from(base)
            .join("NeXium")
            .join("shader_cache")
            .join(device_tag)
            .join(format!("{}.specs", title)),
    )
}

pub fn spec_to_request(
    spec: &PipelineSpec,
    vs_mod: vk::ShaderModule,
    fs_mod: vk::ShaderModule,
    use_binding_divisors: bool,
) -> PipelineBuildRequest {
    let bindings = spec
        .bindings
        .iter()
        .map(|(b, s, d)| vk::VertexInputBindingDescription {
            binding: *b,
            stride: *s,
            input_rate: if *d != 0 {
                vk::VertexInputRate::INSTANCE
            } else {
                vk::VertexInputRate::VERTEX
            },
        })
        .collect();
    let binding_divisors = spec
        .bindings
        .iter()
        .filter_map(|(binding, _, divisor)| {
            (use_binding_divisors && *divisor > 1).then_some(
                vk::VertexInputBindingDivisorDescriptionKHR {
                    binding: *binding,
                    divisor: *divisor,
                },
            )
        })
        .collect();
    let attrs = spec
        .attrs
        .iter()
        .map(|(l, b, f, o)| vk::VertexInputAttributeDescription {
            location: *l,
            binding: *b,
            format: vk::Format::from_raw(*f),
            offset: *o,
        })
        .collect();
    let first_attachment = crate::draw::BlendAttachmentState {
        enabled: spec.blend.0,
        src_factor: vk::BlendFactor::from_raw(spec.blend.1),
        dst_factor: vk::BlendFactor::from_raw(spec.blend.2),
        op: vk::BlendOp::from_raw(spec.blend.3),
        src_alpha_factor: vk::BlendFactor::from_raw(spec.blend.4),
        dst_alpha_factor: vk::BlendFactor::from_raw(spec.blend.5),
        alpha_op: vk::BlendOp::from_raw(spec.blend.6),
        color_write_mask: vk::ColorComponentFlags::from_raw(spec.color_write_mask),
    };
    let mut attachments = [first_attachment; 8];
    for (idx, att) in spec.blend_attachments.iter().take(8).enumerate() {
        attachments[idx] = crate::draw::BlendAttachmentState {
            enabled: att.0,
            src_factor: vk::BlendFactor::from_raw(att.1),
            dst_factor: vk::BlendFactor::from_raw(att.2),
            op: vk::BlendOp::from_raw(att.3),
            src_alpha_factor: vk::BlendFactor::from_raw(att.4),
            dst_alpha_factor: vk::BlendFactor::from_raw(att.5),
            alpha_op: vk::BlendOp::from_raw(att.6),
            color_write_mask: vk::ColorComponentFlags::from_raw(att.7),
        };
    }
    let stencil_face = |face: (i32, i32, i32, i32, u32, u32, u32)| crate::draw::StencilFaceState {
        fail_op: vk::StencilOp::from_raw(face.0),
        pass_op: vk::StencilOp::from_raw(face.1),
        depth_fail_op: vk::StencilOp::from_raw(face.2),
        compare_op: vk::CompareOp::from_raw(face.3),
        compare_mask: face.4,
        write_mask: face.5,
        reference: face.6,
    };
    PipelineBuildRequest {
        key: spec.key,
        vs_mod,
        fs_mod,
        bindings,
        binding_divisors,
        attrs,
        topology: vk::PrimitiveTopology::from_raw(spec.topology),
        color_formats: spec_color_formats(spec),
        depth_format: vk::Format::from_raw(spec.depth_format),
        has_depth: spec.has_depth,
        depth_aspects: vk::ImageAspectFlags::from_raw(spec.depth_aspects),
        blend: crate::draw::BlendState {
            enabled: attachments[0].enabled,
            src_factor: attachments[0].src_factor,
            dst_factor: attachments[0].dst_factor,
            op: attachments[0].op,
            src_alpha_factor: attachments[0].src_alpha_factor,
            dst_alpha_factor: attachments[0].dst_alpha_factor,
            alpha_op: attachments[0].alpha_op,
            color_write_mask: attachments[0].color_write_mask,
            attachments,
        },
        depth: crate::draw::DepthState {
            test_enabled: spec.depth.0,
            write_enabled: spec.depth.1,
            compare_op: vk::CompareOp::from_raw(spec.depth.2),
        },
        stencil: crate::draw::StencilState {
            enabled: spec.stencil_enabled,
            front: stencil_face(spec.stencil_front),
            back: stencil_face(spec.stencil_back),
        },
        depth_clamp_enabled: spec.depth_clamp_enabled,
        cull_test_enable: spec.cull_test_enable,
        cull_face: spec.cull_face,
        front_face: spec.front_face,
        poly_offset_enable: spec.poly_offset_enable,
        poly_offset_units: spec.poly_offset_units,
        poly_offset_factor: spec.poly_offset_factor,
        depth_clip_control_enabled: spec.depth_clip_control_enabled,
    }
}

pub struct PipelineBuildRequest {
    pub key: PipelineKey,
    pub vs_mod: vk::ShaderModule,
    pub fs_mod: vk::ShaderModule,
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub binding_divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR>,
    pub attrs: Vec<vk::VertexInputAttributeDescription>,
    pub topology: vk::PrimitiveTopology,
    pub color_formats: Vec<vk::Format>,
    pub depth_format: vk::Format,
    pub has_depth: bool,
    pub depth_aspects: vk::ImageAspectFlags,
    pub blend: crate::draw::BlendState,
    pub depth: crate::draw::DepthState,
    pub stencil: crate::draw::StencilState,
    pub depth_clamp_enabled: bool,
    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub depth_clip_control_enabled: bool,
}

pub fn normalized_color_formats(formats: &[vk::Format]) -> Vec<vk::Format> {
    formats.iter().copied().take(8).collect()
}

pub fn color_format_key(formats: &[vk::Format]) -> (u32, [u32; 8], u32) {
    let formats = normalized_color_formats(formats);
    let mut key = [0u32; 8];
    for (idx, format) in formats.iter().enumerate() {
        key[idx] = format.as_raw() as u32;
    }
    (key[0], key, formats.len() as u32)
}

fn spec_color_formats(spec: &PipelineSpec) -> Vec<vk::Format> {
    if spec.color_attachment_count == 0 {
        Vec::new()
    } else if spec.color_formats.is_empty() {
        normalized_color_formats(&[vk::Format::from_raw(spec.color_format)])
    } else {
        normalized_color_formats(
            &spec
                .color_formats
                .iter()
                .map(|f| vk::Format::from_raw(*f))
                .collect::<Vec<_>>(),
        )
    }
}

pub fn build_graphics_pipeline(
    device: &ash::Device,
    vk_cache: vk::PipelineCache,
    pipeline_layout: vk::PipelineLayout,
    cache_lock: &std::sync::RwLock<()>,
    req: &PipelineBuildRequest,
) -> Result<vk::Pipeline, String> {
    if known_driver_hostile_pipeline(req.key.vs_hash, req.key.fs_hash) {
        return Err(format!(
            "rejected driver-hostile shader pair vs_hash={:016x} fs_hash={:016x}",
            req.key.vs_hash, req.key.fs_hash
        ));
    }
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::VERTEX,
            module: req.vs_mod,
            p_name: entry.as_ptr(),
            p_specialization_info: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: req.fs_mod,
            p_name: entry.as_ptr(),
            p_specialization_info: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        },
    ];

    let divisor_state = vk::PipelineVertexInputDivisorStateCreateInfoKHR {
        vertex_binding_divisor_count: req.binding_divisors.len() as u32,
        p_vertex_binding_divisors: if req.binding_divisors.is_empty() {
            std::ptr::null()
        } else {
            req.binding_divisors.as_ptr()
        },
        ..Default::default()
    };
    let vi_state = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        vertex_binding_description_count: req.bindings.len() as u32,
        p_vertex_binding_descriptions: req.bindings.as_ptr(),
        vertex_attribute_description_count: req.attrs.len() as u32,
        p_vertex_attribute_descriptions: req.attrs.as_ptr(),
        p_next: if req.binding_divisors.is_empty() {
            std::ptr::null()
        } else {
            &divisor_state as *const _ as *const std::ffi::c_void
        },
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let ia_state = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: req.topology,
        primitive_restart_enable: vk::FALSE,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let dcc_vp = vk::PipelineViewportDepthClipControlCreateInfoEXT {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        negative_one_to_one: vk::TRUE,
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };
    let vp_pnext: *const std::ffi::c_void = if req.depth_clip_control_enabled {
        &dcc_vp as *const _ as *const std::ffi::c_void
    } else {
        std::ptr::null()
    };
    let vp_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewport_count: 1,
        p_viewports: std::ptr::null(),
        scissor_count: 1,
        p_scissors: std::ptr::null(),
        p_next: vp_pnext,
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let host_front_face = match req.front_face {
        0x0900 => vk::FrontFace::CLOCKWISE,
        0x0901 => vk::FrontFace::COUNTER_CLOCKWISE,
        _ => vk::FrontFace::COUNTER_CLOCKWISE,
    };
    let no_cull = {
        use std::sync::OnceLock;
        static NC: OnceLock<bool> = OnceLock::new();
        *NC.get_or_init(|| std::env::var_os("NEXIUM_NO_CULL").is_some())
    };
    let host_cull = if !req.cull_test_enable || no_cull {
        vk::CullModeFlags::NONE
    } else {
        match req.cull_face {
            0x0404 | 0x0001 => vk::CullModeFlags::FRONT,
            0x0405 | 0x0002 => vk::CullModeFlags::BACK,
            0x0408 | 0x0003 => vk::CullModeFlags::FRONT_AND_BACK,
            _ => vk::CullModeFlags::NONE,
        }
    };

    let rs_state = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: host_cull,
        front_face: host_front_face,
        line_width: 1.0,
        depth_clamp_enable: if req.depth_clamp_enabled {
            vk::TRUE
        } else {
            vk::FALSE
        },
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_enable: if req.poly_offset_enable {
            vk::TRUE
        } else {
            vk::FALSE
        },
        depth_bias_constant_factor: req.poly_offset_units / 2.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: req.poly_offset_factor,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let ms_state = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: std::ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let color_formats = normalized_color_formats(&req.color_formats);
    let color_attachment_count = color_formats.len() as u32;
    let cb_attachments = req
        .blend
        .attachments
        .iter()
        .take(color_attachment_count as usize)
        .map(|att| vk::PipelineColorBlendAttachmentState {
            blend_enable: if att.enabled { vk::TRUE } else { vk::FALSE },
            src_color_blend_factor: att.src_factor,
            dst_color_blend_factor: att.dst_factor,
            color_blend_op: att.op,
            src_alpha_blend_factor: att.src_alpha_factor,
            dst_alpha_blend_factor: att.dst_alpha_factor,
            alpha_blend_op: att.alpha_op,
            color_write_mask: att.color_write_mask,
        })
        .collect::<Vec<_>>();
    let cb_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_attachment_count,
        p_attachments: cb_attachments.as_ptr(),
        blend_constants: [0.0; 4],
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let dyn_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_WRITE_MASK,
    ];
    let dyn_state = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        dynamic_state_count: dyn_states.len() as u32,
        p_dynamic_states: dyn_states.as_ptr(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let stencil_face = |face: crate::draw::StencilFaceState| vk::StencilOpState {
        fail_op: face.fail_op,
        pass_op: face.pass_op,
        depth_fail_op: face.depth_fail_op,
        compare_op: face.compare_op,
        compare_mask: face.compare_mask,
        write_mask: face.write_mask,
        reference: face.reference,
    };
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        depth_test_enable: if req.depth.test_enabled {
            vk::TRUE
        } else {
            vk::FALSE
        },
        depth_write_enable: if req.depth.write_enabled {
            vk::TRUE
        } else {
            vk::FALSE
        },
        depth_compare_op: req.depth.compare_op,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: if req.stencil.enabled {
            vk::TRUE
        } else {
            vk::FALSE
        },
        front: stencil_face(req.stencil.front),
        back: stencil_face(req.stencil.back),
        min_depth_bounds: 0.0,
        max_depth_bounds: 1.0,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let p_depth_stencil_state: *const vk::PipelineDepthStencilStateCreateInfo =
        if !req.depth_aspects.is_empty() {
            &depth_stencil_state
        } else {
            std::ptr::null()
        };

    let mut rendering_info = vk::PipelineRenderingCreateInfo {
        s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
        view_mask: 0,
        color_attachment_count,
        p_color_attachment_formats: if color_formats.is_empty() {
            std::ptr::null()
        } else {
            color_formats.as_ptr()
        },
        depth_attachment_format: if req.depth_aspects.contains(vk::ImageAspectFlags::DEPTH) {
            req.depth_format
        } else {
            vk::Format::UNDEFINED
        },
        stencil_attachment_format: if req.depth_aspects.contains(vk::ImageAspectFlags::STENCIL) {
            req.depth_format
        } else {
            vk::Format::UNDEFINED
        },
        p_next: std::ptr::null(),
        _marker: std::marker::PhantomData,
    };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vi_state,
        p_input_assembly_state: &ia_state,
        p_tessellation_state: std::ptr::null(),
        p_viewport_state: &vp_state,
        p_rasterization_state: &rs_state,
        p_multisample_state: &ms_state,
        p_depth_stencil_state,
        p_color_blend_state: &cb_state,
        p_dynamic_state: &dyn_state,
        layout: pipeline_layout,
        render_pass: vk::RenderPass::null(),
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
        p_next: &mut rendering_info as *mut _ as *mut std::ffi::c_void,
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let pipelines = {
        let _guard = cache_lock.read().unwrap_or_else(|e| e.into_inner());
        let pipeline_debug = std::env::var_os("NEXIUM_PIPELINE_DBG").is_some();
        if pipeline_debug {
            log::warn!(
                "[pipeline-build] begin vs_hash={:016x} fs_hash={:016x} topology={} color_attachments={} depth_format={} has_depth={}",
                req.key.vs_hash,
                req.key.fs_hash,
                req.topology.as_raw(),
                color_attachment_count,
                req.depth_format.as_raw(),
                req.has_depth
            );
        }
        let result = unsafe {
            device
                .create_graphics_pipelines(vk_cache, &[pipeline_info], None)
                .map_err(|(_, e)| format!("create_graphics_pipelines: {:?}", e))?
        };
        if pipeline_debug {
            log::warn!(
                "[pipeline-build] end vs_hash={:016x} fs_hash={:016x}",
                req.key.vs_hash,
                req.key.fs_hash
            );
        }
        result
    };
    Ok(pipelines[0])
}

struct CompileWorker {
    req_tx: std::sync::mpsc::Sender<PipelineBuildRequest>,
    res_rx: std::sync::mpsc::Receiver<(PipelineKey, vk::Pipeline)>,
    in_flight: std::collections::HashMap<PipelineKey, u32>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub vk_cache: vk::PipelineCache,
    dirty: bool,
    last_save: std::time::Instant,
    last_saved_len: usize,
    save_tx: Option<std::sync::mpsc::Sender<Vec<u8>>>,
    worker: Option<CompileWorker>,
    failed: HashSet<PipelineKey>,
    cache_lock: std::sync::Arc<std::sync::RwLock<()>>,
    specs: HashMap<PipelineKey, PipelineSpec>,
    specs_dirty: bool,
    specs_saved_count: usize,
    specs_tx: Option<std::sync::mpsc::Sender<Vec<u8>>>,
}

impl PipelineCache {
    pub fn new(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        device_tag: &str,
    ) -> Result<Self, String> {
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };

        let layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("create_pipeline_layout: {:?}", e))?
        };

        let path = cache_path(device_tag);
        let initial = path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .unwrap_or_default();
        let cache_info = vk::PipelineCacheCreateInfo {
            s_type: vk::StructureType::PIPELINE_CACHE_CREATE_INFO,
            initial_data_size: initial.len(),
            p_initial_data: if initial.is_empty() {
                std::ptr::null()
            } else {
                initial.as_ptr() as *const std::ffi::c_void
            },
            p_next: std::ptr::null(),
            flags: Default::default(),
            _marker: std::marker::PhantomData,
        };
        let vk_cache = unsafe {
            device
                .create_pipeline_cache(&cache_info, None)
                .map_err(|e| format!("create_pipeline_cache: {:?}", e))?
        };
        log::info!(
            "VkPipelineCache initialized ({} bytes from disk)",
            initial.len()
        );

        let writer_path = path.clone();
        let (save_tx, save_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        std::thread::Builder::new()
            .name("nexium-pipecache".to_string())
            .spawn(move || {
                while let Ok(data) = save_rx.recv() {
                    let Some(path) = writer_path.as_ref() else {
                        continue;
                    };
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let tmp = path.with_extension("tmp");
                    if std::fs::write(&tmp, &data).is_ok() {
                        let _ = std::fs::rename(&tmp, path);
                        log::info!("VkPipelineCache saved ({} bytes)", data.len());
                    }
                }
            })
            .ok();

        let specs_disk_path = specs_path(device_tag);
        let specs: HashMap<PipelineKey, PipelineSpec> = specs_disk_path
            .as_ref()
            .and_then(|p| {
                let meta = std::fs::metadata(p).ok()?;
                if meta.len() > 512 * 1024 * 1024 {
                    log::warn!(
                        "shader specs file too large ({} bytes), ignoring",
                        meta.len()
                    );
                    return None;
                }
                match std::fs::read(p) {
                    Ok(bytes) => Some(bytes),
                    Err(e) => {
                        log::warn!("shader specs read failed: {:?}", e);
                        None
                    }
                }
            })
            .and_then(|bytes| match bincode::deserialize::<SpecFile>(&bytes) {
                Ok(f) if f.version == SPEC_VERSION => Some(f),
                Ok(f) => {
                    log::warn!(
                        "shader specs version {} != {}, ignoring",
                        f.version,
                        SPEC_VERSION
                    );
                    None
                }
                Err(e) => {
                    log::warn!("shader specs deserialize failed: {:?}", e);
                    None
                }
            })
            .map(|f| f.specs.into_iter().map(|s| (s.key, s)).collect())
            .unwrap_or_default();
        let specs_count = specs.len();
        log::info!("shader specs loaded: {}", specs_count);

        let (specs_tx, specs_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let specs_writer_path = specs_disk_path.clone();
        std::thread::Builder::new()
            .name("nexium-speccache".to_string())
            .spawn(move || {
                while let Ok(data) = specs_rx.recv() {
                    let Some(path) = specs_writer_path.as_ref() else {
                        continue;
                    };
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let tmp = path.with_extension("specs.tmp");
                    if std::fs::write(&tmp, &data).is_ok() {
                        let _ = std::fs::rename(&tmp, path);
                        log::info!("shader specs saved ({} bytes)", data.len());
                    }
                }
            })
            .ok();

        let cache_lock = std::sync::Arc::new(std::sync::RwLock::new(()));
        let worker = {
            let (req_tx, req_rx) = std::sync::mpsc::channel::<PipelineBuildRequest>();
            let (res_tx, res_rx) = std::sync::mpsc::channel::<(PipelineKey, vk::Pipeline)>();
            let req_rx = std::sync::Arc::new(std::sync::Mutex::new(req_rx));
            let num_workers = std::thread::available_parallelism()
                .map(|n| n.get().saturating_sub(2))
                .unwrap_or(2)
                .clamp(1, 8);
            let mut handles = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                let dev = device.clone();
                let wcache = vk_cache;
                let wlayout = layout;
                let wlock = cache_lock.clone();
                let rx = req_rx.clone();
                let tx = res_tx.clone();
                let h = std::thread::Builder::new()
                    .name("nexium-pipecompile".to_string())
                    .spawn(move || loop {
                        let req = {
                            let guard = rx.lock().unwrap_or_else(|e| e.into_inner());
                            match guard.recv() {
                                Ok(r) => r,
                                Err(_) => break,
                            }
                        };
                        let pipe =
                            match build_graphics_pipeline(&dev, wcache, wlayout, &wlock, &req) {
                                Ok(p) => p,
                                Err(e) => {
                                    log::warn!("async pipeline build failed: {}", e);
                                    vk::Pipeline::null()
                                }
                            };
                        if tx.send((req.key, pipe)).is_err() {
                            break;
                        }
                    })
                    .ok();
                if let Some(h) = h {
                    handles.push(h);
                }
            }
            log::info!("async pipeline workers: {}", handles.len());
            Some(CompileWorker {
                req_tx,
                res_rx,
                in_flight: std::collections::HashMap::new(),
                handles,
            })
        };

        Ok(Self {
            pipelines: HashMap::new(),
            layout,
            vk_cache,
            dirty: false,
            last_save: std::time::Instant::now(),
            last_saved_len: initial.len(),
            save_tx: Some(save_tx),
            worker,
            failed: HashSet::new(),
            cache_lock,
            specs,
            specs_dirty: false,
            specs_saved_count: specs_count,
            specs_tx: Some(specs_tx),
        })
    }

    pub fn register_spec(&mut self, spec: PipelineSpec) {
        if self.specs.insert(spec.key, spec).is_none() {
            self.specs_dirty = true;
        }
    }

    pub fn queue_build(&mut self, req: PipelineBuildRequest) {
        if self.pipelines.contains_key(&req.key) {
            return;
        }
        if let Some(w) = self.worker.as_mut() {
            if w.in_flight.contains_key(&req.key) {
                return;
            }
            let key = req.key;
            if w.req_tx.send(req).is_ok() {
                w.in_flight.insert(key, 0);
                nexium_common::shader_progress::begin();
            }
        }
    }

    pub fn prewarm_specs(&self) -> Vec<PipelineSpec> {
        self.specs.values().cloned().collect()
    }

    fn save_specs(&mut self) {
        if !self.specs_dirty || self.specs.len() == self.specs_saved_count {
            return;
        }
        let file = SpecFile {
            version: SPEC_VERSION,
            specs: self.specs.values().cloned().collect(),
        };
        if let Ok(bytes) = bincode::serialize(&file) {
            self.specs_saved_count = self.specs.len();
            if let Some(tx) = &self.specs_tx {
                let _ = tx.send(bytes);
            }
        }
        self.specs_dirty = false;
    }

    pub fn drain_completed(&mut self, device: &ash::Device) {
        let mut done: Vec<(PipelineKey, vk::Pipeline)> = Vec::new();
        if let Some(w) = self.worker.as_mut() {
            while let Ok(r) = w.res_rx.try_recv() {
                w.in_flight.remove(&r.0);
                nexium_common::shader_progress::end();
                done.push(r);
            }
        }
        for (key, pipe) in done {
            if pipe == vk::Pipeline::null() {
                self.failed.insert(key);
                continue;
            }
            if self.pipelines.contains_key(&key) {
                unsafe {
                    device.destroy_pipeline(pipe, None);
                }
            } else {
                self.pipelines.insert(key, pipe);
                self.dirty = true;
            }
        }
    }

    pub fn try_async_skip(&mut self, req: PipelineBuildRequest) -> Option<PipelineBuildRequest> {
        let Some(w) = self.worker.as_mut() else {
            return Some(req);
        };
        if self.failed.contains(&req.key) {
            return Some(req);
        }
        if w.in_flight.contains_key(&req.key) {
            return None;
        }
        let key = req.key;
        match w.req_tx.send(req) {
            Ok(()) => {
                w.in_flight.insert(key, 0);
                nexium_common::shader_progress::begin();
                None
            }
            Err(e) => Some(e.0),
        }
    }

    pub fn maybe_save(&mut self, device: &ash::Device) {
        if (self.dirty || self.specs_dirty)
            && self.last_save.elapsed() >= std::time::Duration::from_secs(4)
        {
            self.save(device);
            self.save_specs();
            self.dirty = false;
            self.last_save = std::time::Instant::now();
        }
    }

    pub fn save(&mut self, device: &ash::Device) {
        if self.vk_cache == vk::PipelineCache::null() {
            return;
        }
        let data = {
            let _guard = self.cache_lock.write().unwrap_or_else(|e| e.into_inner());
            match unsafe { device.get_pipeline_cache_data(self.vk_cache) } {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("get_pipeline_cache_data failed: {:?}", e);
                    return;
                }
            }
        };
        if data.len() == self.last_saved_len {
            return;
        }
        self.last_saved_len = data.len();
        if let Some(tx) = &self.save_tx {
            let _ = tx.send(data);
        }
    }

    pub fn build(
        &self,
        device: &ash::Device,
        req: &PipelineBuildRequest,
    ) -> Result<vk::Pipeline, String> {
        build_graphics_pipeline(device, self.vk_cache, self.layout, &self.cache_lock, req)
    }

    pub fn get(&self, key: &PipelineKey) -> Option<vk::Pipeline> {
        self.pipelines.get(key).copied()
    }

    pub fn insert(&mut self, key: PipelineKey, pipeline: vk::Pipeline) {
        self.pipelines.insert(key, pipeline);
        self.dirty = true;
    }

    pub fn clear(&mut self, device: &ash::Device) {
        if let Some(w) = self.worker.take() {
            let CompileWorker {
                req_tx,
                res_rx,
                handles,
                in_flight: _,
            } = w;
            drop(req_tx);
            for h in handles {
                let _ = h.join();
            }
            while let Ok((key, pipe)) = res_rx.try_recv() {
                nexium_common::shader_progress::end();
                if pipe != vk::Pipeline::null() {
                    self.pipelines.insert(key, pipe);
                } else {
                    self.failed.insert(key);
                }
            }
        }
        self.save(device);
        self.save_specs();
        for (_, pipeline) in self.pipelines.drain() {
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        if self.vk_cache != vk::PipelineCache::null() {
            unsafe {
                device.destroy_pipeline_cache(self.vk_cache, None);
            }
            self.vk_cache = vk::PipelineCache::null();
        }
        if self.layout != vk::PipelineLayout::null() {
            unsafe {
                device.destroy_pipeline_layout(self.layout, None);
            }
            self.layout = vk::PipelineLayout::null();
        }
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if !self.pipelines.is_empty() || self.layout != vk::PipelineLayout::null() {
            log::warn!("PipelineCache dropped without explicit cleanup");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::known_driver_hostile_pipeline;

    #[test]
    fn driver_hostile_pipeline_guard_is_pair_specific() {
        assert!(known_driver_hostile_pipeline(
            0x59b9_0e74_4b2a_7537,
            0xe505_d075_601e_e633
        ));
        assert!(!known_driver_hostile_pipeline(
            0x59b9_0e74_4b2a_7537,
            0xe505_d075_601e_e632
        ));
        assert!(!known_driver_hostile_pipeline(
            0x59b9_0e74_4b2a_7536,
            0xe505_d075_601e_e633
        ));
    }
}

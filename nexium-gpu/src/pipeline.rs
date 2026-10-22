use ash::vk;
use nexium_common::fast_hash::{FastMap as HashMap, FastSet as HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::io::Write;

fn cache_path(device_tag: &str) -> Option<PathBuf> {
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        nexium_common::paths::root()
            .join("shader_cache")
            .join(device_tag)
            .join(format!("{}.bin", title)),
    )
}

#[derive(
    Default, Hash, Eq, PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct PipelineKey {
    pub vs_hash: u64,
    pub gs_hash: u64,
    pub fs_hash: u64,
    pub topology: u32,
    pub primitive_restart_enable: bool,
    pub color_format: u32,
    pub color_formats: [u32; 8],
    pub color_attachment_count: u32,
    pub vs_cbuf_mask: u64,
    pub fs_cbuf_mask: u64,
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

#[derive(Default)]
struct CurrentPipeline {
    entry: Option<(PipelineKey, vk::Pipeline)>,
}

impl CurrentPipeline {
    fn get(&self, key: &PipelineKey) -> Option<vk::Pipeline> {
        self.entry
            .filter(|(current_key, _)| current_key == key)
            .map(|(_, pipeline)| pipeline)
    }

    fn set(&mut self, key: PipelineKey, pipeline: vk::Pipeline) {
        self.entry = Some((key, pipeline));
    }

    fn clear(&mut self) {
        self.entry = None;
    }
}

const SPEC_VERSION: u32 = 63;
const LEGACY_SPEC_VERSION: u32 = 62;
const NO_SPEC_MODULE: u32 = u32::MAX;
const MAX_SPEC_CACHE_BYTES: u64 = 512 * 1024 * 1024;
const CACHE_SAVE_IDLE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
const SPECS_SAVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(20);
const KNOWN_DRIVER_HOSTILE_PIPELINES: &[(u64, u64)] =
    &[(0x59b9_0e74_4b2a_7537, 0xe505_d075_601e_e633)];

pub(crate) fn known_driver_hostile_pipeline(vs_hash: u64, fs_hash: u64) -> bool {
    KNOWN_DRIVER_HOSTILE_PIPELINES.contains(&(vs_hash, fs_hash))
}

fn cache_save_due(
    dirty: bool,
    specs_dirty: bool,
    idle: std::time::Duration,
    builds_in_flight: bool,
) -> bool {
    !builds_in_flight && (dirty || specs_dirty) && idle >= CACHE_SAVE_IDLE_INTERVAL
}

fn specs_save_due(specs_dirty: bool, since_last_save: std::time::Duration) -> bool {
    specs_dirty && since_last_save >= SPECS_SAVE_INTERVAL
}

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PipelineSpec {
    #[serde(skip)]
    pub key: PipelineKey,
    #[serde(skip)]
    pub vs_spirv: Arc<Vec<u32>>,
    #[serde(skip)]
    pub gs_spirv: Arc<Vec<u32>>,
    #[serde(skip)]
    pub fs_spirv: Arc<Vec<u32>>,
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

#[derive(serde::Deserialize)]
struct SpecFile {
    version: u32,
    modules: Vec<Vec<u32>>,
    specs: Vec<StoredSpec>,
}

#[derive(serde::Deserialize)]
struct StoredSpec {
    key: PipelineKey,
    modules: [u32; 3],
    spec: PipelineSpec,
}

#[derive(serde::Serialize)]
struct BorrowedSpecFile<'a> {
    version: u32,
    modules: Vec<&'a [u32]>,
    specs: Vec<BorrowedStoredSpec<'a>>,
}

#[derive(serde::Serialize)]
struct BorrowedStoredSpec<'a> {
    key: PipelineKey,
    modules: [u32; 3],
    spec: &'a PipelineSpec,
}

#[derive(serde::Deserialize)]
struct LegacySpecFile {
    version: u32,
    specs: Vec<LegacySpec>,
}

#[derive(serde::Deserialize)]
struct LegacySpec {
    key: PipelineKey,
    vs_spirv: Vec<u32>,
    gs_spirv: Vec<u32>,
    fs_spirv: Vec<u32>,
    spec: PipelineSpec,
}

#[derive(Default)]
struct SpecModuleTable<'a> {
    modules: Vec<&'a [u32]>,
    by_ptr: HashMap<*const Vec<u32>, u32>,
    by_hash: HashMap<u64, Vec<u32>>,
}

impl<'a> SpecModuleTable<'a> {
    fn intern(&mut self, module: &'a Arc<Vec<u32>>) -> u32 {
        if module.is_empty() {
            return NO_SPEC_MODULE;
        }
        if let Some(&index) = self.by_ptr.get(&Arc::as_ptr(module)) {
            return index;
        }
        let candidates = self
            .by_hash
            .entry(crate::renderer::hash_spirv(module))
            .or_default();
        let index = match candidates
            .iter()
            .copied()
            .find(|&index| self.modules[index as usize] == module.as_slice())
        {
            Some(index) => index,
            None => {
                let index = self.modules.len() as u32;
                self.modules.push(module.as_slice());
                candidates.push(index);
                index
            }
        };
        self.by_ptr.insert(Arc::as_ptr(module), index);
        index
    }
}

fn encode_spec_file<W: Write>(specs: &[Arc<PipelineSpec>], writer: W) -> postcard::Result<W> {
    let mut table = SpecModuleTable::default();
    let stored: Vec<BorrowedStoredSpec> = specs
        .iter()
        .map(|spec| BorrowedStoredSpec {
            key: spec.key,
            modules: [
                table.intern(&spec.vs_spirv),
                table.intern(&spec.gs_spirv),
                table.intern(&spec.fs_spirv),
            ],
            spec: spec.as_ref(),
        })
        .collect();
    let file = BorrowedSpecFile {
        version: SPEC_VERSION,
        modules: table.modules,
        specs: stored,
    };
    postcard::to_io(&file, writer)
}

type DecodedSpecs = (HashMap<PipelineKey, Arc<PipelineSpec>>, bool);

fn decode_spec_file(bytes: &[u8]) -> Result<DecodedSpecs, String> {
    let (version, _) = postcard::take_from_bytes::<u32>(bytes)
        .map_err(|error| format!("header unreadable: {error:?}"))?;
    if version == LEGACY_SPEC_VERSION {
        let file: LegacySpecFile = postcard::from_bytes(bytes)
            .map_err(|error| format!("version {version} deserialize failed: {error:?}"))?;
        let mut interned: HashMap<u64, Vec<Arc<Vec<u32>>>> = HashMap::default();
        let mut intern = |words: Vec<u32>| -> Arc<Vec<u32>> {
            if words.is_empty() {
                return Arc::default();
            }
            let candidates = interned
                .entry(crate::renderer::hash_spirv(&words))
                .or_default();
            if let Some(existing) = candidates.iter().find(|module| ***module == words) {
                return Arc::clone(existing);
            }
            let module = Arc::new(words);
            candidates.push(Arc::clone(&module));
            module
        };
        let mut specs = HashMap::with_capacity_and_hasher(file.specs.len(), Default::default());
        for legacy in file.specs {
            let mut spec = legacy.spec;
            spec.key = legacy.key;
            spec.vs_spirv = intern(legacy.vs_spirv);
            spec.gs_spirv = intern(legacy.gs_spirv);
            spec.fs_spirv = intern(legacy.fs_spirv);
            specs.insert(spec.key, Arc::new(spec));
        }
        return Ok((specs, true));
    }
    if version != SPEC_VERSION {
        return Err(format!("version {version} != {SPEC_VERSION}"));
    }
    let file: SpecFile = postcard::from_bytes(bytes)
        .map_err(|error| format!("version {version} deserialize failed: {error:?}"))?;
    let modules: Vec<Arc<Vec<u32>>> = file.modules.into_iter().map(Arc::new).collect();
    let module = |index: u32| -> Result<Arc<Vec<u32>>, String> {
        if index == NO_SPEC_MODULE {
            return Ok(Arc::default());
        }
        modules
            .get(index as usize)
            .cloned()
            .ok_or_else(|| format!("module {index} out of range"))
    };
    let mut specs = HashMap::with_capacity_and_hasher(file.specs.len(), Default::default());
    for stored in file.specs {
        let mut spec = stored.spec;
        spec.key = stored.key;
        spec.vs_spirv = module(stored.modules[0])?;
        spec.gs_spirv = module(stored.modules[1])?;
        spec.fs_spirv = module(stored.modules[2])?;
        specs.insert(spec.key, Arc::new(spec));
    }
    Ok((specs, false))
}

struct LimitedSpecWriter<W> {
    writer: W,
    remaining: u64,
}

impl<W: Write> Write for LimitedSpecWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(std::io::Error::other("shader specs cache size limit"));
        }
        let written = self.writer.write(bytes)?;
        self.remaining -= written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

fn write_spec_snapshot(path: &std::path::Path, specs: &[Arc<PipelineSpec>]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("specs.tmp");
    let output = match std::fs::File::create(&tmp) {
        Ok(output) => output,
        Err(error) => {
            log::warn!("shader specs temp create failed: {:?}", error);
            return;
        }
    };
    let mut writer = LimitedSpecWriter {
        writer: std::io::BufWriter::new(output),
        remaining: MAX_SPEC_CACHE_BYTES,
    };
    let serialized = encode_spec_file(specs, &mut writer);
    if serialized.is_err() || writer.flush().is_err() {
        log::warn!("shader specs save failed or exceeded {} bytes; preserving previous cache", MAX_SPEC_CACHE_BYTES);
        drop(writer);
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    let bytes = MAX_SPEC_CACHE_BYTES - writer.remaining;
    drop(writer);
    if std::fs::rename(&tmp, path).is_ok() {
        log::info!("shader specs saved ({} bytes)", bytes);
    }
}

fn specs_path(device_tag: &str) -> Option<PathBuf> {
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        nexium_common::paths::root()
            .join("shader_cache")
            .join(device_tag)
            .join(format!("{}.specs", title)),
    )
}

pub fn spec_to_request(
    spec: &PipelineSpec,
    vs_mod: vk::ShaderModule,
    gs_mod: vk::ShaderModule,
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
        gs_mod,
        fs_mod,
        bindings,
        binding_divisors,
        attrs,
        topology: vk::PrimitiveTopology::from_raw(spec.topology),
        primitive_restart_enable: spec.key.primitive_restart_enable,
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
            constants: [0.0; 4],
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
    pub gs_mod: vk::ShaderModule,
    pub fs_mod: vk::ShaderModule,
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub binding_divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR>,
    pub attrs: Vec<vk::VertexInputAttributeDescription>,
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart_enable: bool,
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
    let formats = &formats[..formats.len().min(8)];
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

/// Shared-handle guard for a `PipelineCache`: builds hold it for reading for the
/// whole `vkCreate*Pipelines` call; `get_pipeline_cache_data` and teardown take
/// it for writing. The flag is false once the VkPipelineCache and pipeline
/// layout have been retired, so a builder that outlives them fails cleanly
/// instead of using destroyed handles.
pub type PipelineCacheLock = std::sync::RwLock<bool>;

const PIPELINE_CACHE_RETIRED: &str = "pipeline cache already torn down";

/// Flags for the shared VkPipelineCache. It is used concurrently by the compile
/// workers and by draw-time builds running outside the renderer lock, which is
/// only legal because it is *not* created with
/// `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT` (Vulkan then
/// synchronises access to it internally).
fn shared_pipeline_cache_create_flags() -> vk::PipelineCacheCreateFlags {
    vk::PipelineCacheCreateFlags::empty()
}

pub fn build_graphics_pipeline(
    device: &ash::Device,
    vk_cache: vk::PipelineCache,
    pipeline_layout: vk::PipelineLayout,
    cache_lock: &PipelineCacheLock,
    req: &PipelineBuildRequest,
) -> Result<vk::Pipeline, String> {
    let alive = cache_lock.read().unwrap_or_else(|e| e.into_inner());
    if !*alive {
        return Err(PIPELINE_CACHE_RETIRED.to_string());
    }
    create_graphics_pipeline(device, vk_cache, pipeline_layout, req)
}

/// The `vkCreateGraphicsPipelines` call itself. Callers must hold the cache's
/// `PipelineCacheLock` for reading and have checked it is still alive.
fn create_graphics_pipeline(
    device: &ash::Device,
    vk_cache: vk::PipelineCache,
    pipeline_layout: vk::PipelineLayout,
    req: &PipelineBuildRequest,
) -> Result<vk::Pipeline, String> {
    if known_driver_hostile_pipeline(req.key.vs_hash, req.key.fs_hash) {
        return Err(format!(
            "rejected driver-hostile shader pair vs_hash={:016x} fs_hash={:016x}",
            req.key.vs_hash, req.key.fs_hash
        ));
    }
    let entry = c"main";
    let mut stages = vec![
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

    if req.gs_mod != vk::ShaderModule::null() {
        stages.insert(1, vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::GEOMETRY).module(req.gs_mod).name(entry));
    }

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
        primitive_restart_enable: if req.primitive_restart_enable {
            vk::TRUE
        } else {
            vk::FALSE
        },
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
        vk::DynamicState::BLEND_CONSTANTS,
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
        nexium_common::async_compile::note_shader_built();
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InFlightPoll {
    Ready(vk::Pipeline),
    Pending,
    Gone,
}

/// What a `PipelineCache` needs from a device to dispose of pipelines; a seam
/// so the cache's concurrency protocol can be tested without a GPU.
pub trait PipelineDestroyer {
    fn destroy_unused_pipeline(&self, pipeline: vk::Pipeline);
}

impl PipelineDestroyer for ash::Device {
    fn destroy_unused_pipeline(&self, pipeline: vk::Pipeline) {
        // Only ever called for a pipeline no command buffer has recorded: a
        // duplicate that lost a build race.
        unsafe { self.destroy_pipeline(pipeline, None) };
    }
}

pub struct DetachedPipelineBuilder {
    vk_cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    cache_lock: std::sync::Arc<PipelineCacheLock>,
    external_dirty: std::sync::Arc<std::sync::atomic::AtomicBool>,
    active_builds: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

struct ActiveBuildGuard<'a>(&'a std::sync::atomic::AtomicUsize);

impl Drop for ActiveBuildGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl DetachedPipelineBuilder {
    /// Runs `create` with the shared VkPipelineCache and pipeline layout while
    /// holding the cache lock for reading, so the handles cannot be retired
    /// (or the cache data serialised) underneath it. Counted in
    /// `active_builds` so `maybe_save` does not block on it. Fails if the cache
    /// was already torn down.
    fn with_live_handles<T>(
        &self,
        create: impl FnOnce(vk::PipelineCache, vk::PipelineLayout) -> Result<T, String>,
    ) -> Result<T, String> {
        self.active_builds
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let _active = ActiveBuildGuard(&self.active_builds);
        let alive = self.cache_lock.read().unwrap_or_else(|e| e.into_inner());
        if !*alive {
            return Err(PIPELINE_CACHE_RETIRED.to_string());
        }
        create(self.vk_cache, self.layout)
    }

    /// Builds a graphics pipeline against the shared VkPipelineCache. Safe to
    /// call without the renderer lock: the VkPipelineCache is internally
    /// synchronised (see `shared_pipeline_cache_create_flags`), the layout and
    /// shader modules are only read, and the read guard keeps teardown and
    /// `get_pipeline_cache_data` out meanwhile.
    pub fn build(
        &self,
        device: &ash::Device,
        req: &PipelineBuildRequest,
    ) -> Result<vk::Pipeline, String> {
        self.with_live_handles(|vk_cache, layout| {
            create_graphics_pipeline(device, vk_cache, layout, req)
        })
    }

    /// Creates a compute pipeline through the shared VkPipelineCache so it is
    /// served from (and persisted to) the on-disk cache like graphics pipelines.
    pub fn create_compute_pipeline(
        &self,
        device: &ash::Device,
        info: &vk::ComputePipelineCreateInfo,
    ) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)> {
        let alive = self.cache_lock.read().unwrap_or_else(|e| e.into_inner());
        if !*alive {
            return Err((Vec::new(), vk::Result::ERROR_INITIALIZATION_FAILED));
        }
        let created = unsafe {
            device.create_compute_pipelines(self.vk_cache, std::slice::from_ref(info), None)
        };
        drop(alive);
        if created.is_ok() {
            self.external_dirty
                .store(true, std::sync::atomic::Ordering::Release);
        }
        created
    }
}

/// Waits, with the renderer lock released between polls, for a compile worker
/// that is already building `key`. `None` when the build failed or the worker
/// went away; the caller then builds it itself.
pub(crate) fn wait_for_in_flight_unlocked<G, D>(
    guard: &mut parking_lot::MutexGuard<'_, G>,
    project: fn(&mut G) -> (&D, &mut PipelineCache),
    key: &PipelineKey,
) -> Option<vk::Pipeline>
where
    D: PipelineDestroyer + ?Sized,
{
    let started = std::time::Instant::now();
    let mut spins = 0u32;
    loop {
        let poll = {
            let (device, cache) = project(&mut **guard);
            cache.poll_in_flight(device, key)
        };
        match poll {
            InFlightPoll::Ready(pipeline) => {
                let elapsed = started.elapsed();
                if elapsed >= std::time::Duration::from_millis(10) {
                    log::warn!(
                        "[pipeline-sync-wait] vs_hash={:016x} fs_hash={:016x} wait_ms={:.3} outcome=ready unlocked=true",
                        key.vs_hash,
                        key.fs_hash,
                        elapsed.as_secs_f64() * 1000.0,
                    );
                }
                return Some(pipeline);
            }
            InFlightPoll::Gone => return None,
            InFlightPoll::Pending => {}
        }
        spins = spins.saturating_add(1);
        parking_lot::MutexGuard::unlocked(guard, || {
            if spins < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        });
    }
}

/// Builds `key` with the renderer lock released and publishes it. If another
/// thread (a concurrent miss or a compile worker) published the same key in
/// the meantime, that pipeline is adopted and this build's duplicate is
/// destroyed, so exactly one pipeline object per key survives.
pub(crate) fn build_unlocked_and_insert<G, D>(
    guard: &mut parking_lot::MutexGuard<'_, G>,
    project: fn(&mut G) -> (&D, &mut PipelineCache),
    key: PipelineKey,
    build: impl FnOnce(&DetachedPipelineBuilder) -> Result<vk::Pipeline, String>,
) -> Result<vk::Pipeline, String>
where
    D: PipelineDestroyer + ?Sized,
{
    let builder = project(&mut **guard).1.detached_builder();
    let built = parking_lot::MutexGuard::unlocked(guard, || build(&builder))?;
    let (device, cache) = project(&mut **guard);
    Ok(cache.insert_or_adopt(device, key, built))
}

struct CompileWorker {
    req_tx: std::sync::mpsc::Sender<PipelineBuildRequest>,
    res_rx: std::sync::mpsc::Receiver<(PipelineKey, vk::Pipeline)>,
    in_flight: std::collections::HashMap<PipelineKey, u32>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    current_pipeline: CurrentPipeline,
    pub layout: vk::PipelineLayout,
    pub vk_cache: vk::PipelineCache,
    dirty: bool,
    last_change: std::time::Instant,
    last_saved_len: usize,
    save_tx: Option<std::sync::mpsc::Sender<Vec<u8>>>,
    worker: Option<CompileWorker>,
    failed: HashSet<PipelineKey>,
    cache_lock: std::sync::Arc<PipelineCacheLock>,
    external_dirty: std::sync::Arc<std::sync::atomic::AtomicBool>,
    active_detached_builds: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    specs: HashMap<PipelineKey, Arc<PipelineSpec>>,
    specs_dirty: bool,
    specs_saved_count: usize,
    specs_saved_at: std::time::Instant,
    specs_tx: Option<std::sync::mpsc::SyncSender<Vec<Arc<PipelineSpec>>>>,
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
            flags: shared_pipeline_cache_create_flags(),
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
        let (specs, specs_migrated): DecodedSpecs = specs_disk_path
            .as_ref()
            .and_then(|p| {
                let meta = std::fs::metadata(p).ok()?;
                if meta.len() > MAX_SPEC_CACHE_BYTES {
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
            .and_then(|bytes| match decode_spec_file(&bytes) {
                Ok(decoded) => Some(decoded),
                Err(error) => {
                    log::warn!("shader specs ignored: {}", error);
                    None
                }
            })
            .unwrap_or_default();
        let specs_count = specs.len();
        if specs_migrated {
            log::info!("shader specs loaded: {} (migrating to the deduplicated format)", specs_count);
        } else {
            log::info!("shader specs loaded: {}", specs_count);
        }

        let (specs_tx, specs_rx) = std::sync::mpsc::sync_channel::<Vec<Arc<PipelineSpec>>>(1);
        let specs_writer_path = specs_disk_path.clone();
        std::thread::Builder::new()
            .name("nexium-speccache".to_string())
            .spawn(move || {
                while let Ok(specs) = specs_rx.recv() {
                    if let Some(path) = specs_writer_path.as_ref() {
                        write_spec_snapshot(path, &specs);
                    }
                }
            })
            .ok();

        let cache_lock = std::sync::Arc::new(PipelineCacheLock::new(true));
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
                        let key = req.key;
                        let pipe = match std::panic::catch_unwind(
                            std::panic::AssertUnwindSafe(|| {
                                build_graphics_pipeline(&dev, wcache, wlayout, &wlock, &req)
                            }),
                        ) {
                            Ok(Ok(p)) => p,
                            Ok(Err(e)) => {
                                log::warn!("async pipeline build failed: {}", e);
                                vk::Pipeline::null()
                            }
                            Err(_) => {
                                log::warn!(
                                    "async pipeline worker panicked for vs_hash={:016x} fs_hash={:016x}",
                                    key.vs_hash,
                                    key.fs_hash
                                );
                                vk::Pipeline::null()
                            }
                        };
                        if tx.send((key, pipe)).is_err() {
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
                in_flight: std::collections::HashMap::default(),
                handles,
            })
        };

        Ok(Self {
            pipelines: HashMap::default(),
            current_pipeline: CurrentPipeline::default(),
            layout,
            vk_cache,
            dirty: false,
            last_change: std::time::Instant::now(),
            last_saved_len: initial.len(),
            save_tx: Some(save_tx),
            worker,
            failed: HashSet::default(),
            cache_lock,
            external_dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            active_detached_builds: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            specs,
            specs_dirty: specs_migrated,
            specs_saved_count: if specs_migrated { 0 } else { specs_count },
            specs_saved_at: std::time::Instant::now(),
            specs_tx: Some(specs_tx),
        })
    }

    pub fn register_spec(&mut self, spec: PipelineSpec) {
        if self.specs.insert(spec.key, Arc::new(spec)).is_none() {
            self.specs_dirty = true;
            self.last_change = std::time::Instant::now();
        }
    }

    pub fn can_queue_build(&self, key: &PipelineKey) -> bool {
        self.worker.is_some() && !self.failed.contains(key)
    }

    pub fn queue_build(&mut self, req: PipelineBuildRequest) -> bool {
        if self.pipelines.contains_key(&req.key) || self.failed.contains(&req.key) {
            return false;
        }
        if let Some(w) = self.worker.as_mut() {
            if w.in_flight.contains_key(&req.key) {
                return false;
            }
            let key = req.key;
            if w.req_tx.send(req).is_ok() {
                w.in_flight.insert(key, 0);
                nexium_common::shader_progress::begin();
                return true;
            }
        }
        false
    }

    pub fn prewarm_specs(&self) -> Vec<PipelineSpec> {
        self.specs.values().map(|spec| spec.as_ref().clone()).collect()
    }

    fn save_specs(&mut self) {
        if !self.specs_dirty || self.specs.len() == self.specs_saved_count {
            return;
        }
        let Some(tx) = &self.specs_tx else { return; };
        let snapshot = self.specs.values().cloned().collect();
        if tx.try_send(snapshot).is_ok() {
            self.specs_saved_count = self.specs.len();
            self.specs_dirty = false;
            self.specs_saved_at = std::time::Instant::now();
        }
    }

    fn integrate_completed<D: PipelineDestroyer + ?Sized>(
        &mut self,
        device: &D,
        key: PipelineKey,
        pipe: vk::Pipeline,
    ) {
        let was_in_flight = self
            .worker
            .as_mut()
            .is_some_and(|w| w.in_flight.remove(&key).is_some());
        if was_in_flight {
            nexium_common::shader_progress::end();
        }
        if pipe == vk::Pipeline::null() {
            self.failed.insert(key);
            return;
        }
        if self.pipelines.contains_key(&key) {
            device.destroy_unused_pipeline(pipe);
        } else {
            self.pipelines.insert(key, pipe);
            self.failed.remove(&key);
            self.dirty = true;
            self.last_change = std::time::Instant::now();
        }
    }

    pub fn drain_completed<D: PipelineDestroyer + ?Sized>(&mut self, device: &D) {
        let mut done: Vec<(PipelineKey, vk::Pipeline)> = Vec::new();
        if let Some(w) = self.worker.as_ref() {
            while let Ok(r) = w.res_rx.try_recv() {
                done.push(r);
            }
        }
        for (key, pipe) in done {
            self.integrate_completed(device, key, pipe);
        }
    }

    pub fn has_in_flight(&self, key: &PipelineKey) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|w| w.in_flight.contains_key(key))
    }

    pub fn has_any_in_flight(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|w| !w.in_flight.is_empty())
    }

    pub fn wait_for_in_flight<D: PipelineDestroyer + ?Sized>(
        &mut self,
        device: &D,
        target: &PipelineKey,
    ) -> Option<vk::Pipeline> {
        if !self.has_in_flight(target) {
            return None;
        }

        let started = std::time::Instant::now();
        let mut drained = 0usize;
        let mut outcome = "worker-disconnected";
        let pipeline = loop {
            let result = match self.worker.as_ref() {
                Some(w) => w.res_rx.recv(),
                None => break None,
            };
            let (key, pipe) = match result {
                Ok(result) => result,
                Err(_) => {
                    let abandoned = self
                        .worker
                        .as_mut()
                        .map(|w| w.in_flight.drain().map(|(key, _)| key).collect::<Vec<_>>())
                        .unwrap_or_default();
                    for key in abandoned {
                        nexium_common::shader_progress::end();
                        self.failed.insert(key);
                    }
                    break None;
                }
            };
            drained += 1;
            self.integrate_completed(device, key, pipe);
            if key == *target {
                let ready = self.get(target);
                outcome = if ready.is_some() {
                    "ready"
                } else {
                    "worker-failed"
                };
                break ready;
            }
        };

        let elapsed = started.elapsed();
        if elapsed >= std::time::Duration::from_millis(10) {
            log::warn!(
                "[pipeline-sync-wait] vs_hash={:016x} fs_hash={:016x} wait_ms={:.3} drained={} outcome={}",
                target.vs_hash,
                target.fs_hash,
                elapsed.as_secs_f64() * 1000.0,
                drained,
                outcome
            );
        }
        pipeline
    }

    /// Non-blocking counterpart of `wait_for_in_flight`: integrates whatever the
    /// workers have finished and reports whether `target` is ready, still being
    /// built, or will never arrive. Lets the caller wait without holding the
    /// renderer lock.
    pub fn poll_in_flight<D: PipelineDestroyer + ?Sized>(
        &mut self,
        device: &D,
        target: &PipelineKey,
    ) -> InFlightPoll {
        let mut disconnected = false;
        let mut done: Vec<(PipelineKey, vk::Pipeline)> = Vec::new();
        if let Some(w) = self.worker.as_ref() {
            loop {
                match w.res_rx.try_recv() {
                    Ok(result) => done.push(result),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        for (key, pipe) in done {
            self.integrate_completed(device, key, pipe);
        }
        if disconnected {
            let abandoned = self
                .worker
                .as_mut()
                .map(|w| w.in_flight.drain().map(|(key, _)| key).collect::<Vec<_>>())
                .unwrap_or_default();
            for key in abandoned {
                nexium_common::shader_progress::end();
                self.failed.insert(key);
            }
        }
        if let Some(pipeline) = self.get(target) {
            return InFlightPoll::Ready(pipeline);
        }
        if self.has_in_flight(target) {
            InFlightPoll::Pending
        } else {
            InFlightPoll::Gone
        }
    }

    /// Everything needed to build a pipeline without borrowing the cache, so a
    /// synchronous miss can compile outside the renderer lock.
    pub fn detached_builder(&self) -> DetachedPipelineBuilder {
        DetachedPipelineBuilder {
            vk_cache: self.vk_cache,
            layout: self.layout,
            cache_lock: self.cache_lock.clone(),
            external_dirty: self.external_dirty.clone(),
            active_builds: self.active_detached_builds.clone(),
        }
    }

    /// Inserts a pipeline built by a `DetachedPipelineBuilder`. If another
    /// builder won the race the duplicate is destroyed and the cached handle is
    /// returned.
    pub fn insert_or_adopt<D: PipelineDestroyer + ?Sized>(
        &mut self,
        device: &D,
        key: PipelineKey,
        pipeline: vk::Pipeline,
    ) -> vk::Pipeline {
        if let Some(existing) = self.get(&key) {
            if existing != pipeline {
                device.destroy_unused_pipeline(pipeline);
            }
            return existing;
        }
        self.insert(key, pipeline);
        pipeline
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
        if self
            .external_dirty
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            self.dirty = true;
            self.last_change = std::time::Instant::now();
        }
        // A build running outside the renderer lock holds the cache read guard;
        // saving now would block the caller (under the renderer lock) on it.
        let builds_in_flight = self
            .worker
            .as_ref()
            .is_some_and(|worker| !worker.in_flight.is_empty())
            || self
                .active_detached_builds
                .load(std::sync::atomic::Ordering::Acquire)
                != 0;
        if cache_save_due(
            self.dirty,
            self.specs_dirty,
            self.last_change.elapsed(),
            builds_in_flight,
        ) {
            // A detached build may have started since the check above; never
            // wait for it here (the caller holds the renderer lock). Retry on
            // a later call instead.
            if !self.save_with(device, false) {
                return;
            }
            self.save_specs();
            self.dirty = false;
        } else if specs_save_due(self.specs_dirty, self.specs_saved_at.elapsed()) {
            self.save_specs();
        }
    }

    pub fn save(&mut self, device: &ash::Device) {
        self.save_with(device, true);
    }

    /// Serialises the VkPipelineCache for the disk writer. With `block` false it
    /// gives up (returning false) instead of waiting for in-flight builds.
    fn save_with(&mut self, device: &ash::Device, block: bool) -> bool {
        if self.vk_cache == vk::PipelineCache::null() {
            return true;
        }
        let data = {
            let _guard = if block {
                self.cache_lock.write().unwrap_or_else(|e| e.into_inner())
            } else {
                match self.cache_lock.try_write() {
                    Ok(guard) => guard,
                    Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => return false,
                }
            };
            match unsafe { device.get_pipeline_cache_data(self.vk_cache) } {
                Ok(d) => d,
                Err(e) => {
                    log::warn!("get_pipeline_cache_data failed: {:?}", e);
                    return true;
                }
            }
        };
        if data.len() == self.last_saved_len {
            return true;
        }
        self.last_saved_len = data.len();
        if let Some(tx) = &self.save_tx {
            let _ = tx.send(data);
        }
        true
    }

    pub fn build(
        &self,
        device: &ash::Device,
        req: &PipelineBuildRequest,
    ) -> Result<vk::Pipeline, String> {
        build_graphics_pipeline(device, self.vk_cache, self.layout, &self.cache_lock, req)
    }

    pub fn get(&mut self, key: &PipelineKey) -> Option<vk::Pipeline> {
        if let Some(pipeline) = self.get_current(key) {
            return Some(pipeline);
        }
        let pipeline = self.pipelines.get(key).copied()?;
        self.current_pipeline.set(*key, pipeline);
        Some(pipeline)
    }

    pub fn get_current(&self, key: &PipelineKey) -> Option<vk::Pipeline> {
        self.current_pipeline.get(key)
    }

    pub fn insert(&mut self, key: PipelineKey, pipeline: vk::Pipeline) {
        self.pipelines.insert(key, pipeline);
        self.current_pipeline.set(key, pipeline);
        self.failed.remove(&key);
        self.dirty = true;
        self.last_change = std::time::Instant::now();
    }

    pub fn clear(&mut self, device: &ash::Device) {
        self.current_pipeline.clear();
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
        let (vk_cache, layout) = self.retire_shared_handles();
        if vk_cache != vk::PipelineCache::null() {
            unsafe {
                device.destroy_pipeline_cache(vk_cache, None);
            }
        }
        if layout != vk::PipelineLayout::null() {
            unsafe {
                device.destroy_pipeline_layout(layout, None);
            }
        }
    }

    /// Detaches the VkPipelineCache and pipeline layout for destruction. Takes
    /// the cache lock for writing, so it waits for every build still running
    /// against them (compile workers, detached draw-time builds, compute
    /// pipeline creation), and marks the lock retired so any builder that
    /// starts afterwards fails instead of touching destroyed handles.
    fn retire_shared_handles(&mut self) -> (vk::PipelineCache, vk::PipelineLayout) {
        let mut alive = self.cache_lock.write().unwrap_or_else(|e| e.into_inner());
        *alive = false;
        drop(alive);
        (
            std::mem::replace(&mut self.vk_cache, vk::PipelineCache::null()),
            std::mem::replace(&mut self.layout, vk::PipelineLayout::null()),
        )
    }
}

#[cfg(test)]
impl PipelineCache {
    /// A cache with no device behind it: fake handles, no disk I/O, and a
    /// compile "worker" without threads whose results the test injects
    /// through the returned sender.
    fn for_tests(
        vk_cache: vk::PipelineCache,
        layout: vk::PipelineLayout,
    ) -> (Self, std::sync::mpsc::Sender<(PipelineKey, vk::Pipeline)>) {
        let (req_tx, _req_rx) = std::sync::mpsc::channel::<PipelineBuildRequest>();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        let cache = Self {
            pipelines: HashMap::default(),
            current_pipeline: CurrentPipeline::default(),
            layout,
            vk_cache,
            dirty: false,
            last_change: std::time::Instant::now(),
            last_saved_len: 0,
            save_tx: None,
            worker: Some(CompileWorker {
                req_tx,
                res_rx,
                in_flight: std::collections::HashMap::default(),
                handles: Vec::new(),
            }),
            failed: HashSet::default(),
            cache_lock: std::sync::Arc::new(PipelineCacheLock::new(true)),
            external_dirty: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            active_detached_builds: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            specs: HashMap::default(),
            specs_dirty: false,
            specs_saved_count: 0,
            specs_saved_at: std::time::Instant::now(),
            specs_tx: None,
        };
        (cache, res_tx)
    }

    /// What `queue_build` records after handing a request to a worker.
    fn mark_queued_for_tests(&mut self, key: PipelineKey) {
        self.worker.as_mut().unwrap().in_flight.insert(key, 0);
        nexium_common::shader_progress::begin();
    }

    /// `clear` without a device: hands every cached pipeline to `destroyer`
    /// and retires the shared handles.
    fn teardown_for_tests<D: PipelineDestroyer + ?Sized>(&mut self, destroyer: &D) {
        self.current_pipeline.clear();
        if let Some(worker) = self.worker.take() {
            for _ in worker.in_flight {
                nexium_common::shader_progress::end();
            }
        }
        for (_, pipeline) in self.pipelines.drain() {
            destroyer.destroy_unused_pipeline(pipeline);
        }
        let _ = self.retire_shared_handles();
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
    use ash::vk;
    use ash::vk::Handle;

    use super::{
        cache_save_due, known_driver_hostile_pipeline, specs_save_due, CurrentPipeline,
        PipelineKey, CACHE_SAVE_IDLE_INTERVAL, SPECS_SAVE_INTERVAL,
    };
    use std::sync::Arc;

    /// Records every pipeline the cache asks to destroy.
    #[derive(Default)]
    struct RecordingDestroyer {
        destroyed: std::sync::Mutex<Vec<vk::Pipeline>>,
    }

    impl super::PipelineDestroyer for RecordingDestroyer {
        fn destroy_unused_pipeline(&self, pipeline: vk::Pipeline) {
            self.destroyed.lock().unwrap().push(pipeline);
        }
    }

    impl RecordingDestroyer {
        fn destroyed(&self) -> Vec<vk::Pipeline> {
            self.destroyed.lock().unwrap().clone()
        }
    }

    /// Stand-in for `RendererInner`: the state the renderer lock protects.
    struct FakeRenderer {
        destroyer: RecordingDestroyer,
        cache: super::PipelineCache,
    }

    fn fake_renderer_parts(
        inner: &mut FakeRenderer,
    ) -> (&RecordingDestroyer, &mut super::PipelineCache) {
        (&inner.destroyer, &mut inner.cache)
    }

    fn fake_renderer() -> (
        std::sync::Arc<parking_lot::Mutex<FakeRenderer>>,
        std::sync::mpsc::Sender<(PipelineKey, vk::Pipeline)>,
    ) {
        let (cache, completions) = super::PipelineCache::for_tests(
            vk::PipelineCache::from_raw(0xcac4e),
            vk::PipelineLayout::from_raw(0x1a70),
        );
        (
            std::sync::Arc::new(parking_lot::Mutex::new(FakeRenderer {
                destroyer: RecordingDestroyer::default(),
                cache,
            })),
            completions,
        )
    }

    fn next_fake_pipeline() -> vk::Pipeline {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x10_0000);
        vk::Pipeline::from_raw(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    /// The renderer's draw-time miss path (`Renderer::compile_pipeline` with the
    /// lock released), with `build` standing in for `vkCreateGraphicsPipelines`.
    fn resolve_miss(
        renderer: &parking_lot::Mutex<FakeRenderer>,
        key: PipelineKey,
        build: impl FnOnce() -> vk::Pipeline,
    ) -> Result<vk::Pipeline, String> {
        let mut inner = renderer.lock();
        {
            let FakeRenderer { destroyer, cache } = &mut *inner;
            cache.drain_completed(destroyer);
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline);
            }
        }
        if inner.cache.has_in_flight(&key) {
            if let Some(pipeline) =
                super::wait_for_in_flight_unlocked(&mut inner, fake_renderer_parts, &key)
            {
                return Ok(pipeline);
            }
        }
        super::build_unlocked_and_insert(&mut inner, fake_renderer_parts, key, |builder| {
            builder.with_live_handles(|vk_cache, layout| {
                assert_eq!(vk_cache, vk::PipelineCache::from_raw(0xcac4e));
                assert_eq!(layout, vk::PipelineLayout::from_raw(0x1a70));
                Ok(build())
            })
        })
    }

    /// Blocks until `count` threads have arrived, failing instead of hanging if
    /// they never do (e.g. because a build were still serialised by the lock).
    fn rendezvous(arrived: &std::sync::atomic::AtomicUsize, count: usize) {
        arrived.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while arrived.load(std::sync::atomic::Ordering::Acquire) < count {
            assert!(
                std::time::Instant::now() < deadline,
                "builds did not overlap: the renderer lock is held while building"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn pipeline_cache_is_internally_synchronised() {
        // Detached draw-time builds and the compile workers share it unlocked.
        assert!(!super::shared_pipeline_cache_create_flags()
            .contains(vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED));
    }

    #[test]
    fn concurrent_misses_on_one_key_keep_exactly_one_pipeline() {
        let (renderer, _completions) = fake_renderer();
        let key = PipelineKey { vs_hash: 0xabc, ..Default::default() };
        let arrived = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let built = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let renderer = renderer.clone();
                let arrived = arrived.clone();
                let built = built.clone();
                std::thread::spawn(move || {
                    resolve_miss(&renderer, key, || {
                        // Both builds are in progress at once: the lock is free.
                        rendezvous(&arrived, 2);
                        let pipeline = next_fake_pipeline();
                        built.lock().unwrap().push(pipeline);
                        pipeline
                    })
                    .unwrap()
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|thread| thread.join().unwrap()).collect();
        let built = built.lock().unwrap().clone();
        assert_eq!(built.len(), 2);
        assert_eq!(results[0], results[1], "both callers must use the same pipeline");
        let survivor = results[0];
        let loser = *built.iter().find(|&&pipeline| pipeline != survivor).unwrap();
        let mut inner = renderer.lock();
        assert_eq!(inner.destroyer.destroyed(), vec![loser], "loser destroyed exactly once");
        assert_eq!(inner.cache.pipelines.len(), 1);
        assert_eq!(inner.cache.get(&key), Some(survivor));
        assert_eq!(
            inner.cache.active_detached_builds.load(std::sync::atomic::Ordering::Acquire),
            0
        );
        let FakeRenderer { destroyer, cache } = &mut *inner;
        cache.teardown_for_tests(destroyer);
    }

    #[test]
    fn miss_waits_unlocked_for_a_queued_prewarm_build() {
        let (renderer, completions) = fake_renderer();
        let key = PipelineKey { fs_hash: 0x51, ..Default::default() };
        renderer.lock().cache.mark_queued_for_tests(key);
        let waiter = {
            let renderer = renderer.clone();
            std::thread::spawn(move || {
                resolve_miss(&renderer, key, || panic!("a queued build must be waited for"))
            })
        };
        // The waiter polls with the lock released: this thread can take it.
        for _ in 0..20 {
            assert!(renderer.lock().cache.has_in_flight(&key));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let worker_pipeline = next_fake_pipeline();
        completions.send((key, worker_pipeline)).unwrap();
        assert_eq!(waiter.join().unwrap().unwrap(), worker_pipeline);
        let mut inner = renderer.lock();
        assert!(inner.destroyer.destroyed().is_empty());
        assert!(!inner.cache.has_in_flight(&key));
        let FakeRenderer { destroyer, cache } = &mut *inner;
        cache.teardown_for_tests(destroyer);
    }

    #[test]
    fn failed_prewarm_build_falls_back_to_a_detached_build() {
        let (renderer, completions) = fake_renderer();
        let key = PipelineKey { fs_hash: 0x52, ..Default::default() };
        renderer.lock().cache.mark_queued_for_tests(key);
        completions.send((key, vk::Pipeline::null())).unwrap();
        let own = next_fake_pipeline();
        assert_eq!(resolve_miss(&renderer, key, || own).unwrap(), own);
        let mut inner = renderer.lock();
        assert!(inner.destroyer.destroyed().is_empty());
        assert_eq!(inner.cache.get(&key), Some(own));
        let FakeRenderer { destroyer, cache } = &mut *inner;
        cache.teardown_for_tests(destroyer);
    }

    #[test]
    fn detached_build_racing_prewarm_completion_keeps_one_pipeline_either_order() {
        for worker_lands_first in [true, false] {
            let (renderer, completions) = fake_renderer();
            let key = PipelineKey { vs_hash: 0x77, ..Default::default() };
            let detached = next_fake_pipeline();
            let worker = next_fake_pipeline();
            let (entered_tx, entered_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
            let miss = {
                let renderer = renderer.clone();
                std::thread::spawn(move || {
                    resolve_miss(&renderer, key, || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        detached
                    })
                    .unwrap()
                })
            };
            entered_rx.recv().unwrap();
            // Prewarm queued the same key and its worker finishes while the
            // draw-time build is still running without the lock.
            {
                let mut inner = renderer.lock();
                inner.cache.mark_queued_for_tests(key);
                completions.send((key, worker)).unwrap();
                if worker_lands_first {
                    let FakeRenderer { destroyer, cache } = &mut *inner;
                    cache.drain_completed(destroyer);
                }
            }
            release_tx.send(()).unwrap();
            let used = miss.join().unwrap();
            let mut inner = renderer.lock();
            {
                let FakeRenderer { destroyer, cache } = &mut *inner;
                cache.drain_completed(destroyer);
            }
            let (survivor, loser) =
                if worker_lands_first { (worker, detached) } else { (detached, worker) };
            assert_eq!(used, survivor, "worker_lands_first={worker_lands_first}");
            assert_eq!(inner.cache.get(&key), Some(survivor));
            assert_eq!(inner.destroyer.destroyed(), vec![loser]);
            assert!(!inner.cache.has_in_flight(&key));
            let FakeRenderer { destroyer, cache } = &mut *inner;
            cache.teardown_for_tests(destroyer);
        }
    }

    #[test]
    fn concurrent_misses_and_completions_never_leak_or_double_destroy() {
        let (renderer, completions) = fake_renderer();
        let keys: Vec<_> = (0..24u64)
            .map(|index| PipelineKey { vs_hash: index, ..Default::default() })
            .collect();
        for key in &keys[..3] {
            renderer.lock().cache.mark_queued_for_tests(*key);
        }
        let created = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let workers = {
            let created = created.clone();
            let keys = keys.clone();
            std::thread::spawn(move || {
                for key in &keys[..3] {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    let pipeline = next_fake_pipeline();
                    created.lock().unwrap().push(pipeline);
                    completions.send((*key, pipeline)).unwrap();
                }
            })
        };
        let threads: Vec<_> = (0..4)
            .map(|thread_index| {
                let renderer = renderer.clone();
                let created = created.clone();
                let keys = keys.clone();
                std::thread::spawn(move || {
                    let mut used = Vec::new();
                    for round in 0..48 {
                        let key = keys[(thread_index * 7 + round * 5) % keys.len()];
                        let pipeline = resolve_miss(&renderer, key, || {
                            std::thread::sleep(std::time::Duration::from_micros(300));
                            let pipeline = next_fake_pipeline();
                            created.lock().unwrap().push(pipeline);
                            pipeline
                        })
                        .unwrap();
                        used.push((key, pipeline));
                    }
                    used
                })
            })
            .collect();
        let used: Vec<_> = threads.into_iter().flat_map(|thread| thread.join().unwrap()).collect();
        workers.join().unwrap();
        let mut inner = renderer.lock();
        {
            let FakeRenderer { destroyer, cache } = &mut *inner;
            cache.drain_completed(destroyer);
        }
        let destroyed = inner.destroyer.destroyed();
        let created = created.lock().unwrap().clone();
        let survivors: Vec<_> = keys.iter().filter_map(|key| inner.cache.get(key)).collect();
        assert_eq!(survivors.len(), keys.len());
        for (key, pipeline) in &used {
            assert_eq!(inner.cache.get(key), Some(*pipeline), "caller used a non-surviving pipeline");
        }
        let mut accounted: Vec<_> = destroyed.iter().chain(&survivors).copied().collect();
        accounted.sort_by_key(|pipeline| pipeline.as_raw());
        let mut all_created = created.clone();
        all_created.sort_by_key(|pipeline| pipeline.as_raw());
        assert_eq!(accounted, all_created, "every pipeline is either cached or destroyed once");
        let FakeRenderer { destroyer, cache } = &mut *inner;
        cache.teardown_for_tests(destroyer);
    }

    #[test]
    fn teardown_waits_for_an_in_flight_build_and_rejects_later_ones() {
        let (cache, _completions) = super::PipelineCache::for_tests(
            vk::PipelineCache::from_raw(0xcac4e),
            vk::PipelineLayout::from_raw(0x1a70),
        );
        let builder = cache.detached_builder();
        let cache = std::sync::Arc::new(std::sync::Mutex::new(cache));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let build = std::thread::spawn(move || {
            let result = builder.with_live_handles(|vk_cache, layout| {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok((vk_cache, layout))
            });
            (result, builder)
        });
        entered_rx.recv().unwrap();
        let retired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let teardown = {
            let cache = cache.clone();
            let retired = retired.clone();
            std::thread::spawn(move || {
                let handles = cache.lock().unwrap().retire_shared_handles();
                retired.store(true, std::sync::atomic::Ordering::Release);
                handles
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            !retired.load(std::sync::atomic::Ordering::Acquire),
            "handles retired while a build was still using them"
        );
        release_tx.send(()).unwrap();
        let (result, builder) = build.join().unwrap();
        assert_eq!(
            result.unwrap(),
            (vk::PipelineCache::from_raw(0xcac4e), vk::PipelineLayout::from_raw(0x1a70))
        );
        assert_eq!(
            teardown.join().unwrap(),
            (vk::PipelineCache::from_raw(0xcac4e), vk::PipelineLayout::from_raw(0x1a70))
        );
        let late = builder.with_live_handles(|_, _| -> Result<(), String> {
            panic!("a build after teardown must not see the retired handles")
        });
        assert!(late.is_err());
        let mut cache = cache.lock().unwrap();
        assert_eq!(cache.vk_cache, vk::PipelineCache::null());
        assert_eq!(cache.layout, vk::PipelineLayout::null());
        assert_eq!(
            cache.active_detached_builds.load(std::sync::atomic::Ordering::Acquire),
            0
        );
        cache.teardown_for_tests(&RecordingDestroyer::default());
    }

    #[test]
    fn spec_file_round_trips_with_shared_modules() {
        let vs = Arc::new(vec![0x07230203, 0x00010000, 0, u32::MAX]);
        let first = Arc::new(super::PipelineSpec {
            key: PipelineKey { vs_hash: 1, ..Default::default() },
            vs_spirv: Arc::clone(&vs),
            fs_spirv: Arc::new(vec![7, 3, 1]),
            bindings: vec![(0, 16, 1)],
            attrs: vec![(0, 0, 109, 4)],
            ..Default::default()
        });
        let second = Arc::new(super::PipelineSpec {
            key: PipelineKey { vs_hash: 2, ..Default::default() },
            vs_spirv: Arc::new(vs.as_ref().clone()),
            fs_spirv: Arc::new(vec![9]),
            topology: 4,
            ..Default::default()
        });
        let bytes = super::encode_spec_file(&[first.clone(), second.clone()], Vec::new()).unwrap();
        let (specs, migrated) = super::decode_spec_file(&bytes).unwrap();
        assert!(!migrated);
        let a = &specs[&first.key];
        let b = &specs[&second.key];
        assert_eq!(*a.vs_spirv, *first.vs_spirv);
        assert_eq!(*a.fs_spirv, *first.fs_spirv);
        assert!(a.gs_spirv.is_empty());
        assert_eq!(a.bindings, first.bindings);
        assert_eq!(a.attrs, first.attrs);
        assert!(Arc::ptr_eq(&a.vs_spirv, &b.vs_spirv));
        assert_eq!(*b.fs_spirv, vec![9]);
        assert_eq!(b.topology, 4);
    }

    #[test]
    fn legacy_spec_file_is_migrated_with_shared_modules() {
        #[derive(serde::Serialize)]
        struct LegacyOut<'a> {
            key: PipelineKey,
            vs: &'a [u32],
            gs: &'a [u32],
            fs: &'a [u32],
            spec: &'a super::PipelineSpec,
        }
        let spec = super::PipelineSpec {
            bindings: vec![(1, 32, 0)],
            topology: 3,
            ..Default::default()
        };
        let key = PipelineKey { fs_hash: 5, ..Default::default() };
        let words = [0x07230203u32, 1, 2];
        let legacy = (
            super::LEGACY_SPEC_VERSION,
            vec![LegacyOut { key, vs: &words, gs: &[], fs: &words, spec: &spec }],
        );
        let bytes = postcard::to_allocvec(&legacy).unwrap();
        let (specs, migrated) = super::decode_spec_file(&bytes).unwrap();
        assert!(migrated);
        let loaded = &specs[&key];
        assert_eq!(loaded.bindings, spec.bindings);
        assert_eq!(loaded.topology, 3);
        assert_eq!(*loaded.vs_spirv, words.to_vec());
        assert!(Arc::ptr_eq(&loaded.vs_spirv, &loaded.fs_spirv));
        assert!(loaded.gs_spirv.is_empty());
    }

    #[test]
    fn spec_writer_never_exceeds_reader_size_limit() {
        use std::io::Write;
        let mut writer = super::LimitedSpecWriter { writer: Vec::new(), remaining: 5 };
        writer.write_all(&[1, 2, 3]).unwrap();
        assert!(writer.write_all(&[4, 5, 6]).is_err());
        assert_eq!(writer.writer, [1, 2, 3]);
        writer.write_all(&[4, 5]).unwrap();
        assert!(writer.write_all(&[6]).is_err());
        assert_eq!(writer.writer.len(), 5);
    }

    #[test]
    fn current_pipeline_requires_exact_key_and_resets() {
        let key = PipelineKey::default();
        let other_key = PipelineKey { fs_hash: 1, ..key };
        let pipeline = vk::Pipeline::from_raw(7);
        let other_pipeline = vk::Pipeline::from_raw(9);
        let mut current = CurrentPipeline::default();

        assert_eq!(current.get(&key), None);
        current.set(key, pipeline);
        assert_eq!(current.get(&key), Some(pipeline));
        assert_eq!(current.get(&other_key), None);
        current.set(other_key, other_pipeline);
        assert_eq!(current.get(&key), None);
        assert_eq!(current.get(&other_key), Some(other_pipeline));
        current.clear();
        assert_eq!(current.get(&other_key), None);
    }

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

    #[test]
    fn cache_save_waits_for_idle_interval() {
        assert!(!cache_save_due(
            true,
            false,
            CACHE_SAVE_IDLE_INTERVAL - std::time::Duration::from_nanos(1),
            false,
        ));
        assert!(cache_save_due(true, false, CACHE_SAVE_IDLE_INTERVAL, false));
        assert!(cache_save_due(false, true, CACHE_SAVE_IDLE_INTERVAL, false));
        assert!(!cache_save_due(
            false,
            false,
            CACHE_SAVE_IDLE_INTERVAL,
            false
        ));
        assert!(!cache_save_due(true, true, CACHE_SAVE_IDLE_INTERVAL, true));
    }

    #[test]
    fn new_specs_save_on_an_interval_while_pipelines_keep_arriving() {
        assert!(!specs_save_due(
            true,
            SPECS_SAVE_INTERVAL - std::time::Duration::from_nanos(1)
        ));
        assert!(specs_save_due(true, SPECS_SAVE_INTERVAL));
        assert!(!specs_save_due(false, SPECS_SAVE_INTERVAL * 10));
    }
}

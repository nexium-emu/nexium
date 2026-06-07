use ash::vk;
use std::collections::HashMap;
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

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct PipelineKey {
    pub vs_hash: u64,
    pub fs_hash: u64,
    pub topology: u32,
    pub color_format: u32,
    pub vs_cbuf_mask: u32,
    pub fs_cbuf_mask: u32,
    pub vertex_layout_hash: u64,
    pub blend_signature: u32,
    pub raster_state_packed: u32,
    pub depth_state_packed: u32,
    pub poly_offset_packed: u64,
}

pub struct PipelineBuildRequest {
    pub key: PipelineKey,
    pub vs_mod: vk::ShaderModule,
    pub fs_mod: vk::ShaderModule,
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attrs: Vec<vk::VertexInputAttributeDescription>,
    pub topology: vk::PrimitiveTopology,
    pub color_format: vk::Format,
    pub depth_format: vk::Format,
    pub has_depth: bool,
    pub blend: crate::draw::BlendState,
    pub depth: crate::draw::DepthState,
    pub cull_test_enable: bool,
    pub cull_face: u32,
    pub front_face: u32,
    pub poly_offset_enable: bool,
    pub poly_offset_units: f32,
    pub poly_offset_factor: f32,
    pub depth_clip_control_enabled: bool,
}

pub fn build_graphics_pipeline(
    device: &ash::Device,
    vk_cache: vk::PipelineCache,
    pipeline_layout: vk::PipelineLayout,
    cache_lock: &std::sync::RwLock<()>,
    req: &PipelineBuildRequest,
) -> Result<vk::Pipeline, String> {
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

    let vi_state = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        vertex_binding_description_count: req.bindings.len() as u32,
        p_vertex_binding_descriptions: req.bindings.as_ptr(),
        vertex_attribute_description_count: req.attrs.len() as u32,
        p_vertex_attribute_descriptions: req.attrs.as_ptr(),
        p_next: std::ptr::null(),
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
    let host_cull = if !req.cull_test_enable {
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
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_enable: if req.poly_offset_enable { vk::TRUE } else { vk::FALSE },
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

    let cb_attachment = vk::PipelineColorBlendAttachmentState {
        blend_enable: if req.blend.enabled { vk::TRUE } else { vk::FALSE },
        src_color_blend_factor: req.blend.src_factor,
        dst_color_blend_factor: req.blend.dst_factor,
        color_blend_op: req.blend.op,
        src_alpha_blend_factor: req.blend.src_factor,
        dst_alpha_blend_factor: req.blend.dst_factor,
        alpha_blend_op: req.blend.op,
        color_write_mask: vk::ColorComponentFlags::RGBA,
    };
    let cb_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: 1,
        p_attachments: &cb_attachment,
        blend_constants: [0.0; 4],
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dyn_state = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        dynamic_state_count: dyn_states.len() as u32,
        p_dynamic_states: dyn_states.as_ptr(),
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        depth_test_enable: if req.depth.test_enabled { vk::TRUE } else { vk::FALSE },
        depth_write_enable: if req.depth.write_enabled { vk::TRUE } else { vk::FALSE },
        depth_compare_op: req.depth.compare_op,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: vk::StencilOpState::default(),
        back: vk::StencilOpState::default(),
        min_depth_bounds: 0.0,
        max_depth_bounds: 1.0,
        p_next: std::ptr::null(),
        flags: Default::default(),
        _marker: std::marker::PhantomData,
    };
    let p_depth_stencil_state: *const vk::PipelineDepthStencilStateCreateInfo =
        if req.has_depth { &depth_stencil_state } else { std::ptr::null() };

    let color_formats = [req.color_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo {
        s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
        view_mask: 0,
        color_attachment_count: 1,
        p_color_attachment_formats: color_formats.as_ptr(),
        depth_attachment_format: req.depth_format,
        stencil_attachment_format: vk::Format::UNDEFINED,
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
        unsafe {
            device
                .create_graphics_pipelines(vk_cache, &[pipeline_info], None)
                .map_err(|(_, e)| format!("create_graphics_pipelines: {:?}", e))?
        }
    };
    Ok(pipelines[0])
}

struct CompileWorker {
    req_tx: std::sync::mpsc::Sender<PipelineBuildRequest>,
    res_rx: std::sync::mpsc::Receiver<(PipelineKey, vk::Pipeline)>,
    in_flight: std::collections::HashSet<PipelineKey>,
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
    cache_lock: std::sync::Arc<std::sync::RwLock<()>>,
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
            device.create_pipeline_layout(&pipeline_layout_info, None)
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
            device.create_pipeline_cache(&cache_info, None)
                .map_err(|e| format!("create_pipeline_cache: {:?}", e))?
        };
        log::info!("VkPipelineCache initialized ({} bytes from disk)", initial.len());

        let writer_path = path.clone();
        let (save_tx, save_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        std::thread::Builder::new()
            .name("nexium-pipecache".to_string())
            .spawn(move || {
                while let Ok(data) = save_rx.recv() {
                    let Some(path) = writer_path.as_ref() else { continue; };
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

        let cache_lock = std::sync::Arc::new(std::sync::RwLock::new(()));
        let worker = {
            let (req_tx, req_rx) = std::sync::mpsc::channel::<PipelineBuildRequest>();
            let (res_tx, res_rx) =
                std::sync::mpsc::channel::<(PipelineKey, vk::Pipeline)>();
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
                        let pipe = match build_graphics_pipeline(&dev, wcache, wlayout, &wlock, &req) {
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
                in_flight: std::collections::HashSet::new(),
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
            cache_lock,
        })
    }

    pub fn drain_completed(&mut self) {
        let mut done: Vec<(PipelineKey, vk::Pipeline)> = Vec::new();
        if let Some(w) = self.worker.as_mut() {
            while let Ok(r) = w.res_rx.try_recv() {
                w.in_flight.remove(&r.0);
                done.push(r);
            }
        }
        for (key, pipe) in done {
            if pipe != vk::Pipeline::null() {
                self.pipelines.insert(key, pipe);
                self.dirty = true;
            }
        }
    }

    pub fn request_async(&mut self, req: PipelineBuildRequest) {
        if let Some(w) = self.worker.as_mut() {
            if w.in_flight.contains(&req.key) {
                return;
            }
            let key = req.key;
            if w.req_tx.send(req).is_ok() {
                w.in_flight.insert(key);
            }
        }
    }

    pub fn maybe_save(&mut self, device: &ash::Device) {
        if self.dirty && self.last_save.elapsed() >= std::time::Duration::from_secs(4) {
            self.save(device);
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
            let CompileWorker { req_tx, res_rx, handles, in_flight: _ } = w;
            drop(req_tx);
            for h in handles {
                let _ = h.join();
            }
            while let Ok((key, pipe)) = res_rx.try_recv() {
                if pipe != vk::Pipeline::null() {
                    self.pipelines.insert(key, pipe);
                }
            }
        }
        self.save(device);
        for (_, pipeline) in self.pipelines.drain() {
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        if self.vk_cache != vk::PipelineCache::null() {
            unsafe { device.destroy_pipeline_cache(self.vk_cache, None); }
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

use ash::vk;
use std::collections::HashMap;
use std::path::PathBuf;

fn pipeline_cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")?;
    let title = nexium_common::title::title_key().unwrap_or_else(|| "default".to_string());
    Some(
        PathBuf::from(base)
            .join("NeXium")
            .join("pipeline_cache")
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

pub struct PipelineCache {
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub vk_cache: vk::PipelineCache,
    dirty: bool,
    last_save: std::time::Instant,
    last_saved_len: usize,
    save_tx: Option<std::sync::mpsc::Sender<Vec<u8>>>,
}

impl PipelineCache {
    pub fn new(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
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

        let initial = pipeline_cache_path()
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

        let (save_tx, save_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        std::thread::Builder::new()
            .name("nexium-pipecache".to_string())
            .spawn(move || {
                while let Ok(data) = save_rx.recv() {
                    let Some(path) = pipeline_cache_path() else { continue; };
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let tmp = path.with_extension("tmp");
                    if std::fs::write(&tmp, &data).is_ok() {
                        let _ = std::fs::rename(&tmp, &path);
                        log::info!("VkPipelineCache saved ({} bytes)", data.len());
                    }
                }
            })
            .ok();

        Ok(Self {
            pipelines: HashMap::new(),
            layout,
            vk_cache,
            dirty: false,
            last_save: std::time::Instant::now(),
            last_saved_len: initial.len(),
            save_tx: Some(save_tx),
        })
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
        let data = match unsafe { device.get_pipeline_cache_data(self.vk_cache) } {
            Ok(d) => d,
            Err(e) => {
                log::warn!("get_pipeline_cache_data failed: {:?}", e);
                return;
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

    pub fn get(&self, key: &PipelineKey) -> Option<vk::Pipeline> {
        self.pipelines.get(key).copied()
    }

    pub fn insert(&mut self, key: PipelineKey, pipeline: vk::Pipeline) {
        self.pipelines.insert(key, pipeline);
        self.dirty = true;
    }

    pub fn clear(&mut self, device: &ash::Device) {
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

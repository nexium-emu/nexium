use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod bufferqueue;
pub mod gpu;
pub use bufferqueue::{BufferQueue, GraphicBuffer, QueuedFrame};
pub use gpu::GpuContext;

#[derive(Default)]
pub struct PipelineStats {
    pub gpfifo_submits: AtomicU64,
    pub gpfifo_entries: AtomicU64,
    pub methods_dispatched: AtomicU64,
    pub maxwell3d_draws: AtomicU64,
    pub maxwell3d_clears: AtomicU64,
    pub fermi_2d_blits: AtomicU64,
    pub maxwell_dma_blits: AtomicU64,
    pub nvmap_creates: AtomicU64,
    pub nvmap_allocs: AtomicU64,
    pub queue_buffer_calls: AtomicU64,
    pub dequeue_buffer_calls: AtomicU64,
    pub request_buffer_calls: AtomicU64,
    pub vsync_signals: AtomicU64,
    pub frames_submitted: AtomicU64,
    pub frames_drained: AtomicU64,
    pub fence_releases: AtomicU64,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct PipelineStatsSnapshot {
    pub gpfifo_submits: u64,
    pub gpfifo_entries: u64,
    pub methods_dispatched: u64,
    pub maxwell3d_draws: u64,
    pub maxwell3d_clears: u64,
    pub fermi_2d_blits: u64,
    pub maxwell_dma_blits: u64,
    pub nvmap_creates: u64,
    pub nvmap_allocs: u64,
    pub queue_buffer_calls: u64,
    pub dequeue_buffer_calls: u64,
    pub request_buffer_calls: u64,
    pub vsync_signals: u64,
    pub frames_submitted: u64,
    pub frames_drained: u64,
    pub fence_releases: u64,
}

impl PipelineStats {
    pub fn snapshot(&self) -> PipelineStatsSnapshot {
        PipelineStatsSnapshot {
            gpfifo_submits: self.gpfifo_submits.load(Ordering::Relaxed),
            gpfifo_entries: self.gpfifo_entries.load(Ordering::Relaxed),
            methods_dispatched: self.methods_dispatched.load(Ordering::Relaxed),
            maxwell3d_draws: self.maxwell3d_draws.load(Ordering::Relaxed),
            maxwell3d_clears: self.maxwell3d_clears.load(Ordering::Relaxed),
            fermi_2d_blits: self.fermi_2d_blits.load(Ordering::Relaxed),
            maxwell_dma_blits: self.maxwell_dma_blits.load(Ordering::Relaxed),
            nvmap_creates: self.nvmap_creates.load(Ordering::Relaxed),
            nvmap_allocs: self.nvmap_allocs.load(Ordering::Relaxed),
            queue_buffer_calls: self.queue_buffer_calls.load(Ordering::Relaxed),
            dequeue_buffer_calls: self.dequeue_buffer_calls.load(Ordering::Relaxed),
            request_buffer_calls: self.request_buffer_calls.load(Ordering::Relaxed),
            vsync_signals: self.vsync_signals.load(Ordering::Relaxed),
            frames_submitted: self.frames_submitted.load(Ordering::Relaxed),
            frames_drained: self.frames_drained.load(Ordering::Relaxed),
            fence_releases: self.fence_releases.load(Ordering::Relaxed),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NvDevice {
    Nvmap,
    NvhostCtrl,
    NvhostCtrlGpu,
    NvhostAsGpu,
    NvhostGpu,
    NvhostDbgGpu,
    NvhostProfGpu,
    NvhostNvdec,
    NvhostNvjpg,
    NvhostVic,
    NvhostNvenc,
    Other,
}

impl NvDevice {
    pub fn from_path(path: &str) -> Option<Self> {
        let trimmed = path.trim_end_matches('\0').trim_end_matches('/');
        Some(match trimmed {
            "/dev/nvmap" => NvDevice::Nvmap,
            "/dev/nvhost-ctrl" => NvDevice::NvhostCtrl,
            "/dev/nvhost-ctrl-gpu" => NvDevice::NvhostCtrlGpu,
            "/dev/nvhost-as-gpu" => NvDevice::NvhostAsGpu,
            "/dev/nvhost-gpu" => NvDevice::NvhostGpu,
            "/dev/nvhost-dbg-gpu" => NvDevice::NvhostDbgGpu,
            "/dev/nvhost-prof-gpu" => NvDevice::NvhostProfGpu,
            "/dev/nvhost-nvdec" => NvDevice::NvhostNvdec,
            "/dev/nvhost-nvjpg" => NvDevice::NvhostNvjpg,
            "/dev/nvhost-vic" => NvDevice::NvhostVic,
            "/dev/nvhost-nvenc" => NvDevice::NvhostNvenc,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct NvFile {
    pub device: NvDevice,
}

pub struct NvmapHandle {
    pub id: u32,
    pub size: u32,
    pub address: u64,
    pub kind: u32,
}

pub struct IoctlRequest {
    pub fd: u32,
    pub ioctl_id: u32,
    pub in_data: Vec<u8>,
    pub out_size: usize,
}

pub struct IoctlOutcome {
    pub result: u32,
    pub data: Vec<u8>,
}

impl IoctlOutcome {
    pub fn ok(data: Vec<u8>) -> Self {
        Self { result: 0, data }
    }

    pub fn error(result: u32) -> Self {
        Self { result, data: Vec::new() }
    }
}

pub struct Nvdrv {
    pub files: HashMap<u32, NvFile>,
    pub next_fd: u32,
    pub nvmap_handles: HashMap<u32, NvmapHandle>,
    pub next_nvmap_id: u32,
    pub bufferqueues: Arc<Mutex<HashMap<u32, BufferQueue>>>,
    pub frame_queue: Arc<Mutex<Vec<QueuedFrame>>>,
    pub next_event_id: u32,
    pub gpu: Arc<GpuContext>,
    pub last_swap_return: Arc<Mutex<Option<std::time::Instant>>>,
    pub queue_buffer_active: Arc<std::sync::atomic::AtomicBool>,
    pub stats: Arc<PipelineStats>,
    pub channel_client_data: u64,
    pub legacy_gfx: std::sync::atomic::AtomicBool,
    pub renderer: std::sync::OnceLock<Option<Arc<nexium_gpu::Renderer>>>,
}

impl Nvdrv {
    pub fn new() -> Self {
        let stats = Arc::new(PipelineStats::default());
        Self {
            files: HashMap::new(),
            next_fd: 1,
            nvmap_handles: HashMap::new(),
            next_nvmap_id: 1,
            bufferqueues: Arc::new(Mutex::new(HashMap::new())),
            frame_queue: Arc::new(Mutex::new(Vec::new())),
            next_event_id: 1,
            gpu: Arc::new(GpuContext::with_stats(stats.clone())),
            last_swap_return: Arc::new(Mutex::new(None)),
            queue_buffer_active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats,
            channel_client_data: 0,
            legacy_gfx: std::sync::atomic::AtomicBool::new(false),
            renderer: std::sync::OnceLock::new(),
        }
    }

    pub fn frame_queue_depth(&self) -> usize {
        self.frame_queue.lock().len()
    }

    pub fn renderer(&self) -> Option<&Arc<nexium_gpu::Renderer>> {
        let slot = self.renderer.get_or_init(|| {
            match nexium_gpu::Renderer::new() {
                Ok(r) => {
                    log::info!("nexium-nvdrv: Vulkan Renderer initialized");
                    Some(r)
                }
                Err(e) => {
                    log::warn!("nexium-nvdrv: Vulkan Renderer init failed: {} (falling back to CPU)", e);
                    None
                }
            }
        });
        slot.as_ref()
    }

    pub fn pace_swap(&self, swap_interval: i32) {
        const VSYNC_NS: u64 = 16_666_667;
        let n = swap_interval.clamp(1, 4) as u64;
        let target = std::time::Duration::from_nanos(VSYNC_NS.saturating_mul(n));
        let mut slot = self.last_swap_return.lock();
        if let Some(prev) = *slot {
            let elapsed = prev.elapsed();
            if elapsed < target {
                std::thread::sleep(target - elapsed);
            }
        }
        *slot = Some(std::time::Instant::now());
    }

    pub fn open(&mut self, path: &str) -> Result<u32, ()> {
        let device = NvDevice::from_path(path).ok_or(())?;
        let fd = self.next_fd;
        self.next_fd = self.next_fd.wrapping_add(1);
        self.files.insert(fd, NvFile { device });
        log::debug!("nvdrv:Open '{}' → fd={}", path, fd);
        Ok(fd)
    }

    pub fn close(&mut self, fd: u32) {
        self.files.remove(&fd);
        log::debug!("nvdrv:Close fd={}", fd);
    }

    pub fn dispatch_ioctl(&mut self, req: IoctlRequest) -> IoctlOutcome {
        self.dispatch_ioctl_with_mem(req, &|_, _| false, &|_, _| false)
    }

    pub fn dispatch_ioctl_with_mem(
        &mut self,
        req: IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
        mem_write: &dyn Fn(u64, &[u8]) -> bool,
    ) -> IoctlOutcome {
        let device = match self.files.get(&req.fd) {
            Some(f) => f.device,
            None => {
                log::warn!("nvdrv:Ioctl on invalid fd={}", req.fd);
                return IoctlOutcome::error(0xCE01);
            }
        };
        let cmd = (req.ioctl_id & 0xFFFF) as u16;
        log::debug!("nvdrv:Ioctl fd={} device={:?} ioctl={:#010x} cmd={:#06x} in_size={} out_size={}",
            req.fd, device, req.ioctl_id, cmd, req.in_data.len(), req.out_size);

        match device {
            NvDevice::Nvmap => self.nvmap_ioctl(cmd, &req),
            NvDevice::NvhostCtrlGpu => self.nvhost_ctrl_gpu_ioctl(cmd, &req),
            NvDevice::NvhostAsGpu => self.nvhost_as_gpu_ioctl(cmd, &req),
            NvDevice::NvhostGpu => self.nvhost_gpu_ioctl_with_mem(cmd, &req, mem_read, mem_write),
            NvDevice::NvhostCtrl => Self::nvhost_ctrl_ioctl(cmd, &req),
            _ => IoctlOutcome::ok(vec![0u8; req.out_size]),
        }
    }

    fn nvmap_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x0101 => {
                let size = if req.in_data.len() >= 4 {
                    u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]])
                } else { 0 };
                let id = self.next_nvmap_id;
                self.next_nvmap_id = self.next_nvmap_id.wrapping_add(1);
                self.nvmap_handles.insert(id, NvmapHandle {
                    id, size, address: 0, kind: 0,
                });
                self.stats.nvmap_creates.fetch_add(1, Ordering::Relaxed);
                if out.len() < 8 { out.resize(8, 0); }
                out[0..4].copy_from_slice(&size.to_le_bytes());
                out[4..8].copy_from_slice(&id.to_le_bytes());
                log::debug!("nvmap:Create in_data={:02x?} → size={} id={}", &req.in_data[..req.in_data.len().min(16)], size, id);
            }
            0x0103 => {
                if req.in_data.len() >= 4 && out.len() >= 8 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    out[4..8].copy_from_slice(&id.to_le_bytes());
                    log::debug!("nvmap:FromId id={} → handle={}", id, id);
                }
            }
            0x0104 => {
                if req.in_data.len() >= 32 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let address = u64::from_le_bytes([
                        req.in_data[24], req.in_data[25], req.in_data[26], req.in_data[27],
                        req.in_data[28], req.in_data[29], req.in_data[30], req.in_data[31],
                    ]);
                    if let Some(h) = self.nvmap_handles.get_mut(&id) {
                        h.address = address;
                    }
                    self.stats.nvmap_allocs.fetch_add(1, Ordering::Relaxed);
                    log::debug!("nvmap:Alloc id={} addr={:#x}", id, address);
                }
            }
            0x0105 => {
                if req.in_data.len() >= 4 && out.len() >= 24 {
                    let handle = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let size = self.nvmap_handles.get(&handle).map(|h| h.size).unwrap_or(0);
                    self.nvmap_handles.remove(&handle);
                    out[8..16].copy_from_slice(&0u64.to_le_bytes());
                    out[16..20].copy_from_slice(&size.to_le_bytes());
                    out[20..24].copy_from_slice(&0u32.to_le_bytes());
                    log::debug!("nvmap:Free handle={} size={}", handle, size);
                }
            }
            0x0109 => {
                if req.in_data.len() >= 8 && out.len() >= 12 {
                    let handle = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let param = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    let result = match param {
                        1 => self.nvmap_handles.get(&handle).map(|h| h.size).unwrap_or(0),
                        2 => 0x10000,
                        3 => 0,
                        4 => self.nvmap_handles.get(&handle).map(|h| h.kind as u32).unwrap_or(0),
                        5 => 0,
                        _ => 0,
                    };
                    out[8..12].copy_from_slice(&result.to_le_bytes());
                    log::debug!("nvmap:Param handle={} param={} → {}", handle, param, result);
                }
            }
            0x010E => {
                if req.in_data.len() >= 8 && out.len() >= 4 {
                    let handle = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    out[0..4].copy_from_slice(&handle.to_le_bytes());
                    log::debug!("nvmap:GetId handle={} → id={}", handle, handle);
                }
            }
            other => {
                log::debug!("nvmap: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_ctrl_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4701 => {
                if out.len() < 4 { out.resize(4, 0); }
                out[0..4].copy_from_slice(&1u32.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:ZCullGetCtxSize → 1");
            }
            0x4702 => {
                if out.len() < 40 { out.resize(40, 0); }
                let words: [u32; 10] = [
                    0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10,
                ];
                for (i, w) in words.iter().enumerate() {
                    out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                }
                log::debug!("nvhost-ctrl-gpu:ZCullGetInfo");
            }
            0x4705 => {
                if out.len() < 0xB0 { out.resize(0xB0, 0); }
                out[0..8].copy_from_slice(&0xA0u64.to_le_bytes());
                out[8..16].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
                let gc_off = 16usize;
                let gc: &[(usize, u32)] = &[
                    (0x00, 0x120),
                    (0x04, 0x0b),
                    (0x08, 0xa1),
                    (0x0c, 0x01),
                    (0x10, 0x40000),
                    (0x14, 0x0),
                    (0x18, 0),
                    (0x1c, 0),
                    (0x20, 0x02),
                    (0x24, 0x20),
                    (0x28, 0x20000),
                    (0x2c, 0x20000),
                    (0x30, 0x1b),
                    (0x34, 0x30000),
                    (0x38, 0x01),
                    (0x3c, 0x503),
                    (0x40, 0x503),
                    (0x44, 0x80),
                    (0x48, 0x28),
                    (0x4c, 0x0),
                    (0x50, 0x55),
                    (0x54, 0x0),
                    (0x58, 0x902d),
                    (0x5c, 0xb197),
                    (0x60, 0xb1c0),
                    (0x64, 0xb06f),
                    (0x68, 0xa140),
                    (0x6c, 0xb0b5),
                    (0x70, 0x01),
                    (0x74, 0x0),
                    (0x78, 0x02),
                    (0x7c, 0x01),
                    (0x80, 0x0),
                    (0x84, 0x01),
                    (0x88, 0x21d70),
                    (0x8c, 0x0),
                ];
                for (off, val) in gc {
                    let pos = gc_off + off;
                    out[pos..pos + 4].copy_from_slice(&val.to_le_bytes());
                }
                let chipname: u64 = 0x6230326d67;
                out[gc_off + 0x90..gc_off + 0x98].copy_from_slice(&chipname.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:GetCharacteristics → GM20B");
            }
            0x4703 => {
                log::debug!("nvhost-ctrl-gpu:ZbcSetTable (ack)");
            }
            0x4704 => {
                log::debug!("nvhost-ctrl-gpu:ZbcQueryTable");
            }
            0x4706 => {
                if out.len() < 24 { out.resize(24, 0); }
                if req.in_data.len() >= 4 {
                    let mask_buf_size = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    if mask_buf_size != 0 {
                        out[16..20].copy_from_slice(&3u32.to_le_bytes());
                    }
                    out[0..4].copy_from_slice(&mask_buf_size.to_le_bytes());
                }
                log::debug!("nvhost-ctrl-gpu:GetTpcMasks → 3");
            }
            0x4714 => {
                if out.len() >= 8 {
                    out[0..4].copy_from_slice(&0x07u32.to_le_bytes());
                    out[4..8].copy_from_slice(&0x01u32.to_le_bytes());
                }
                self.legacy_gfx.store(true, Ordering::Relaxed);
                log::debug!("nvhost-ctrl-gpu:GetActiveSlotMask → slot=7 mask=1 (legacy_gfx detected)");
            }
            0x471c => {
                if out.len() < 16 { out.resize(16, 0); }
                let ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0);
                out[0..8].copy_from_slice(&ns.to_le_bytes());
                log::debug!("nvhost-ctrl-gpu:GetGpuTime → {}ns", ns);
            }
            other => {
                log::debug!("nvhost-ctrl-gpu: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_as_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4101 => { log::debug!("nvhost-as-gpu:BindChannel"); }
            0x4102 => {
                let pages = if req.in_data.len() >= 4 {
                    u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]])
                } else { 0 };
                let page_size = if req.in_data.len() >= 8 {
                    u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]])
                } else { 0x1000 };
                let total_size = (pages as u64) * (page_size as u64);
                let offset_in: u64 = if req.in_data.len() >= 24 {
                    u64::from_le_bytes([
                        req.in_data[16], req.in_data[17], req.in_data[18], req.in_data[19],
                        req.in_data[20], req.in_data[21], req.in_data[22], req.in_data[23],
                    ])
                } else { 0 };
                let alloc = if offset_in != 0 { offset_in } else { self.gpu.alloc_gpu_va(total_size.max(0x1000)) };
                log::debug!("nvhost-as-gpu:AllocSpace pages={} page_size={:#x} → gpu_va={:#x}",
                    pages, page_size, alloc);
                if out.len() >= 24 {
                    out[16..24].copy_from_slice(&alloc.to_le_bytes());
                }
            }
            0x4105 => {
                if req.in_data.len() >= 8 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    log::debug!("nvhost-as-gpu:UnmapBuffer gpu_va={:#x}", gpu_va);
                }
            }
            0x4106 => {
                if req.in_data.len() >= 40 {
                    let flags = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let _kind = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    let nvmap_id = u32::from_le_bytes([req.in_data[8], req.in_data[9], req.in_data[10], req.in_data[11]]);
                    let _page_size = u32::from_le_bytes([req.in_data[12], req.in_data[13], req.in_data[14], req.in_data[15]]);
                    let buffer_offset = u64::from_le_bytes([
                        req.in_data[16], req.in_data[17], req.in_data[18], req.in_data[19],
                        req.in_data[20], req.in_data[21], req.in_data[22], req.in_data[23],
                    ]);
                    let mapping_size_in = u64::from_le_bytes([
                        req.in_data[24], req.in_data[25], req.in_data[26], req.in_data[27],
                        req.in_data[28], req.in_data[29], req.in_data[30], req.in_data[31],
                    ]);
                    let requested_offset: u64 = u64::from_le_bytes([
                        req.in_data[32], req.in_data[33], req.in_data[34], req.in_data[35],
                        req.in_data[36], req.in_data[37], req.in_data[38], req.in_data[39],
                    ]);

                    let mapping_size = if mapping_size_in == 0 {
                        self.nvmap_handles.get(&nvmap_id)
                            .map(|h| (h.size as u64).saturating_sub(buffer_offset))
                            .unwrap_or(0x1000)
                    } else {
                        mapping_size_in
                    };
                    let gpu_va = if (flags & 0x1) != 0 && requested_offset != 0 {
                        requested_offset
                    } else {
                        self.gpu.alloc_gpu_va(mapping_size.max(0x10000))
                    };
                    let cpu_addr = self.nvmap_handles.get(&nvmap_id)
                        .map(|h| h.address.wrapping_add(buffer_offset))
                        .unwrap_or(0);
                    log::debug!("nvhost-as-gpu:MapBufferEx flags={:#x} nvmap_id={} cpu_addr={:#x} size={:#x} → gpu_va={:#x}",
                        flags, nvmap_id, cpu_addr, mapping_size, gpu_va);

                    self.gpu.mappings.lock().add(gpu_va, mapping_size, cpu_addr, nvmap_id);

                    if out.len() >= 40 {
                        out[32..40].copy_from_slice(&gpu_va.to_le_bytes());
                    }
                }
            }
            0x4108 => {
                let small_offset: u64 = 0x4_0000;
                let small_page: u32 = 0x1000;
                let small_pages: u64 = ((1u64 << 34) - small_offset) / small_page as u64;
                let big_offset: u64 = 1u64 << 34;
                let big_page: u32 = 0x10000;
                let big_pages: u64 = ((1u64 << 38) - big_offset) / big_page as u64;
                if out.len() < 64 {
                    out.resize(64, 0);
                }
                out[16..24].copy_from_slice(&small_offset.to_le_bytes());
                out[24..28].copy_from_slice(&small_page.to_le_bytes());
                out[32..40].copy_from_slice(&small_pages.to_le_bytes());
                out[40..48].copy_from_slice(&big_offset.to_le_bytes());
                out[48..52].copy_from_slice(&big_page.to_le_bytes());
                out[56..64].copy_from_slice(&big_pages.to_le_bytes());
                log::debug!("nvhost-as-gpu:GetVARegions small_pages={} big_pages={}", small_pages, big_pages);
            }
            0x4109 => { log::debug!("nvhost-as-gpu:AllocAsEx (InitializeEx)"); }
            0x4103 => {
                if req.in_data.len() >= 16 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    log::debug!("nvhost-as-gpu:FreeSpace gpu_va={:#x}", gpu_va);
                }
            }
            0x4114 => {
                let num_entries = req.in_data.len() / 20;
                for i in 0..num_entries {
                    let off = i * 20;
                    if req.in_data.len() < off + 20 { break; }
                    let _flags = u16::from_le_bytes([req.in_data[off], req.in_data[off + 1]]);
                    let _kind = u16::from_le_bytes([req.in_data[off + 2], req.in_data[off + 3]]);
                    let nvmap_handle = u32::from_le_bytes([
                        req.in_data[off + 4], req.in_data[off + 5],
                        req.in_data[off + 6], req.in_data[off + 7],
                    ]);
                    let handle_offset_big_pages = u32::from_le_bytes([
                        req.in_data[off + 8], req.in_data[off + 9],
                        req.in_data[off + 10], req.in_data[off + 11],
                    ]);
                    let as_offset_big_pages = u32::from_le_bytes([
                        req.in_data[off + 12], req.in_data[off + 13],
                        req.in_data[off + 14], req.in_data[off + 15],
                    ]);
                    let big_pages = u32::from_le_bytes([
                        req.in_data[off + 16], req.in_data[off + 17],
                        req.in_data[off + 18], req.in_data[off + 19],
                    ]);
                    let big_page_size: u64 = 0x10000;
                    let gpu_va = (as_offset_big_pages as u64) * big_page_size;
                    let size = (big_pages as u64) * big_page_size;
                    let handle_off = (handle_offset_big_pages as u64) * big_page_size;
                    let cpu_addr = self.nvmap_handles.get(&nvmap_handle)
                        .map(|h| h.address.wrapping_add(handle_off))
                        .unwrap_or(0);
                    log::debug!(
                        "nvhost-as-gpu:Remap[{}/{}] nvmap_id={} cpu={:#x} → gpu_va={:#x} size={:#x}",
                        i, num_entries, nvmap_handle, cpu_addr, gpu_va, size
                    );
                    if cpu_addr != 0 {
                        self.gpu.mappings.lock().add(gpu_va, size, cpu_addr, nvmap_handle);
                    }
                }
            }
            other => {
                log::debug!("nvhost-as-gpu: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_gpu_ioctl_with_mem(&mut self, cmd: u16, req: &IoctlRequest, mem_read: &dyn Fn(u64, &mut [u8]) -> bool, mem_write: &dyn Fn(u64, &[u8]) -> bool) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4801 => { log::debug!("nvhost-gpu:SetNvmapFd"); }
            0x4803 => { log::debug!("nvhost-gpu:ChannelSetTimeout"); }
            0x4808 | 0x481b => {
                if req.in_data.len() >= 16 {
                    let address = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    let num_entries = u32::from_le_bytes([
                        req.in_data[8], req.in_data[9], req.in_data[10], req.in_data[11],
                    ]);
                    log::debug!("nvhost-gpu:SubmitGPFIFO addr={:#x} entries={}", address, num_entries);

                    if cmd == 0x4808 && req.in_data.len() >= 16 + (num_entries as usize) * 8 {
                        let entries: Vec<gpu::CommandListHeader> = (0..num_entries as usize).map(|i| {
                            let off = 16 + i * 8;
                            gpu::CommandListHeader {
                                address_lo: u32::from_le_bytes([
                                    req.in_data[off], req.in_data[off + 1],
                                    req.in_data[off + 2], req.in_data[off + 3],
                                ]),
                                address_hi_and_count: u32::from_le_bytes([
                                    req.in_data[off + 4], req.in_data[off + 5],
                                    req.in_data[off + 6], req.in_data[off + 7],
                                ]),
                            }
                        }).collect();
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats.gpfifo_entries.fetch_add(entries.len() as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) = self.gpu.process_inline_gpfifo(&entries, mem_read, mem_write);
                        log::debug!("nvhost-gpu:SubmitGPFIFO processed {} entries (draws={}, clears={})",
                            entries.len(), self.gpu.maxwell3d.lock().draw_count(),
                            self.gpu.maxwell3d.lock().clear_count());
                        if out.len() >= 24 {
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else if cmd == 0x481b && address != 0 {
                        self.stats.gpfifo_submits.fetch_add(1, Ordering::Relaxed);
                        self.stats.gpfifo_entries.fetch_add(num_entries as u64, Ordering::Relaxed);
                        let (syncpt_id, syncpt_value) = self.gpu.submit_gpfifo(address, num_entries, mem_read, mem_write);
                        log::debug!("nvhost-gpu:SubmitGPFIFO (kickoff) addr={:#x} entries={} draws={}",
                            address, num_entries, self.gpu.maxwell3d.lock().draw_count());
                        if out.len() >= 24 {
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else {
                        let syncpt_id: u32 = 0;
                        let syncpt_value: u32 = 1;
                        if out.len() >= 24 {
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    }
                }
            }
            0x4809 => {
                if req.in_data.len() >= 8 && out.len() >= 16 {
                    let class_num = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    out[8..16].copy_from_slice(&0xDEAD_BEEFu64.to_le_bytes());
                    log::debug!("nvhost-gpu:AllocObjCtx class={:#x} → obj_id=0xDEADBEEF", class_num);
                }
            }
            0x480b => {
                if req.in_data.len() >= 12 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    let mode = u32::from_le_bytes([req.in_data[8], req.in_data[9], req.in_data[10], req.in_data[11]]);
                    log::debug!("nvhost-gpu:ZCullBind gpu_va={:#x} mode={}", gpu_va, mode);
                }
            }
            0x480c => {
                if req.in_data.len() >= 20 {
                    let enable = u32::from_le_bytes([req.in_data[16], req.in_data[17], req.in_data[18], req.in_data[19]]);
                    log::debug!("nvhost-gpu:SetErrorNotifier enable={}", enable);
                }
            }
            0x480d => {
                if req.in_data.len() >= 4 {
                    let prio = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvhost-gpu:SetPriority prio={:#x}", prio);
                }
            }
            0x4816 => {
                log::debug!("nvhost-gpu:GetErrorInfo");
            }
            0x4817 => {
                log::debug!("nvhost-gpu:GetErrorNotification");
            }
            0x481a => {
                if req.in_data.len() >= 28 && out.len() >= 28 {
                    let num_entries = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let flags = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    let syncpt_id = self.next_event_id;
                    self.next_event_id = self.next_event_id.wrapping_add(1);
                    out[12..16].copy_from_slice(&syncpt_id.to_le_bytes());
                    out[16..20].copy_from_slice(&0u32.to_le_bytes());
                    log::debug!("nvhost-gpu:AllocGpfifoEx2 num_entries={} flags={:#x} → fence_id={}", num_entries, flags, syncpt_id);
                }
            }
            0x4714 => {
                if req.in_data.len() >= 8 {
                    let data = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    self.channel_client_data = data;
                    log::debug!("nvhost-gpu:SetClientData data={:#x}", data);
                }
            }
            0x4715 => {
                if out.len() >= 8 {
                    out[0..8].copy_from_slice(&self.channel_client_data.to_le_bytes());
                }
                log::debug!("nvhost-gpu:GetClientData → {:#x}", self.channel_client_data);
            }
            0x481d => { log::debug!("nvhost-gpu:ChannelSetTimeslice"); }
            other => {
                log::debug!("nvhost-gpu: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_ctrl_ioctl(cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x0014 => {
                if req.in_data.len() >= 4 && out.len() >= 8 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    out[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
                    log::debug!("nvhost-ctrl:SyncptRead syncpt_id={} → {:#x}", id, u32::MAX);
                }
            }
            0x0015 => {
                if req.in_data.len() >= 4 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvhost-ctrl:SyncptIncr syncpt_id={}", id);
                }
            }
            0x0016 => {
                if req.in_data.len() >= 12 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let threshold = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    log::debug!("nvhost-ctrl:SyncptWait syncpt_id={} threshold={:#x}", id, threshold);
                }
            }
            0x001c => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvhost-ctrl:EventSignal event_id={}", event_id);
                }
            }
            0x001d => {
                if req.in_data.len() >= 16 && out.len() >= 16 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let threshold = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    out[12..16].copy_from_slice(&threshold.to_le_bytes());
                    log::debug!("nvhost-ctrl:EventWait syncpt={} threshold={:#x} (ack as signaled)", id, threshold);
                }
            }
            0x001e => {
                if req.in_data.len() >= 16 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    let threshold = u32::from_le_bytes([req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7]]);
                    let event_id = u32::from_le_bytes([req.in_data[12], req.in_data[13], req.in_data[14], req.in_data[15]]);
                    log::debug!("nvhost-ctrl:EventWaitAsync syncpt={} threshold={:#x} event_id={}", id, threshold, event_id);
                }
            }
            0x001f => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvhost-ctrl:EventRegister event_id={}", event_id);
                }
            }
            0x0020 => {
                if req.in_data.len() >= 4 {
                    let event_id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvhost-ctrl:EventUnregister event_id={}", event_id);
                }
            }
            other => {
                log::debug!("nvhost-ctrl: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    pub fn with_bufferqueue<R>(&self, binder_id: u32, f: impl FnOnce(&mut BufferQueue) -> R) -> R {
        let mut bqs = self.bufferqueues.lock();
        let bq = bqs.entry(binder_id).or_insert_with(|| BufferQueue::new(binder_id));
        f(bq)
    }

    pub fn drain_frames(&self) -> Vec<QueuedFrame> {
        let frames = std::mem::take(&mut *self.frame_queue.lock());
        self.stats.frames_drained.fetch_add(frames.len() as u64, Ordering::Relaxed);
        frames
    }

    pub fn submit_frame(&self, frame: QueuedFrame) {
        self.queue_buffer_active.store(true, std::sync::atomic::Ordering::Relaxed);
        self.frame_queue.lock().push(frame);
        self.stats.frames_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn capture_gpu_frame(&self, mem_read: impl Fn(u64, &mut [u8]) -> bool) -> Option<QueuedFrame> {
        let maxwell = self.gpu.maxwell3d.lock();
        let (w, h) = maxwell.primary_rt_size()?;
        let gpu_va = maxwell.primary_rt_gpu_va()?;
        drop(maxwell);

        let mappings = self.gpu.mappings.lock();
        let cpu = mappings.cpu_address_for(gpu_va)?;
        drop(mappings);

        let size = (w as usize) * (h as usize) * 4;
        let mut pixels = vec![0u8; size];
        if mem_read(cpu, &mut pixels) {
            Some(QueuedFrame { width: w, height: h, pixels })
        } else {
            None
        }
    }

    pub fn try_capture_sdl_surface(&self, mem_read: impl Fn(u64, &mut [u8]) -> bool) -> Option<QueuedFrame> {
        const CANDIDATES: &[(u32, u32, u32)] = &[
            (1280, 720, 1280),
            (1280, 768, 1280),
            (1920, 1080, 1920),
            (640, 360, 640),
            (854, 480, 854),
            (427, 240, 427),
            (480, 270, 480),
        ];

        let mut best: Option<(u32, u32, u32, u64, u32, Vec<u8>)> = None;
        for handle in self.nvmap_handles.values() {
            if handle.address == 0 || handle.size == 0 {
                continue;
            }
            let Some(&(w, h, stride)) = CANDIDATES.iter().find(|(_w, hh, stride)| (*stride as u32) * (*hh as u32) * 4 == handle.size as u32) else {
                continue;
            };

            let mut linear = vec![0u8; handle.size as usize];
            if !mem_read(handle.address, &mut linear) {
                continue;
            }

            let nz = linear.iter().filter(|b| **b != 0).count();
            if nz < 256 {
                continue;
            }

            for px in linear.chunks_exact_mut(4) {
                px[3] = 0xFF;
            }

            if best.as_ref().map(|b| nz > b.4 as usize).unwrap_or(true) {
                best = Some((w, h, stride, handle.address, nz as u32, linear));
            }
        }

        let (w, h, stride, addr, nz, linear) = best?;
        log::debug!("captured SDL surface addr={:#x} {}x{} stride={} (nz={})", addr, w, h, stride, nz);

        let dst_w = 1280u32;
        let dst_h = 720u32;
        let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
        for dy in 0..dst_h {
            let sy = dy * h / dst_h;
            for dx in 0..dst_w {
                let sx = dx * w / dst_w;
                let s = ((sy * stride + sx) * 4) as usize;
                let d = ((dy * dst_w + dx) * 4) as usize;
                if s + 4 <= linear.len() {
                    out[d..d + 4].copy_from_slice(&linear[s..s + 4]);
                }
            }
        }
        Some(QueuedFrame { width: dst_w, height: dst_h, pixels: out })
    }

    pub fn gpu_draw_count(&self) -> u64 {
        self.gpu.maxwell3d.lock().draw_count()
    }

    pub fn last_clear_color(&self) -> [f32; 4] {
        let m = self.gpu.maxwell3d.lock();
        let c = m.regs.clear_color;
        [c.r, c.g, c.b, c.a]
    }

    pub fn last_clear_count(&self) -> u64 {
        self.gpu.maxwell3d.lock().clear_count()
    }

    pub fn drain_fermi2d_frame(&self) -> Option<QueuedFrame> {
        let f2d = self.gpu.fermi_2d.lock();
        let mut q = f2d.captured_frames.lock().unwrap();
        q.pop()
    }
}

impl Default for Nvdrv {
    fn default() -> Self {
        Self::new()
    }
}

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub mod bufferqueue;
pub use bufferqueue::{BufferQueue, GraphicBuffer, QueuedFrame};

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
}

impl Nvdrv {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            next_fd: 1,
            nvmap_handles: HashMap::new(),
            next_nvmap_id: 1,
            bufferqueues: Arc::new(Mutex::new(HashMap::new())),
            frame_queue: Arc::new(Mutex::new(Vec::new())),
            next_event_id: 1,
        }
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
            NvDevice::NvhostGpu => self.nvhost_gpu_ioctl(cmd, &req),
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
                if out.len() >= 8 {
                    out[4..8].copy_from_slice(&id.to_le_bytes());
                }
                log::debug!("nvmap:Create size={} → id={}", size, id);
            }
            0x0103 => {
                if req.in_data.len() >= 4 {
                    let id = u32::from_le_bytes([req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3]]);
                    log::debug!("nvmap:FromId id={}", id);
                    if out.len() >= 4 {
                        out[0..4].copy_from_slice(&id.to_le_bytes());
                    }
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
                    log::debug!("nvmap:Alloc id={} addr={:#x}", id, address);
                }
            }
            0x0105 => {
                log::debug!("nvmap:Free");
            }
            0x0109 => {
                log::debug!("nvmap:Param");
            }
            0x010E => {
                log::debug!("nvmap:GetId");
                if out.len() >= 4 && req.in_data.len() >= 4 {
                    out[0..4].copy_from_slice(&req.in_data[0..4]);
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
                log::debug!("nvhost-ctrl-gpu:ZCullGetCtxSize");
                if out.len() >= 4 { out[0..4].copy_from_slice(&0u32.to_le_bytes()); }
            }
            0x4702 => {
                log::debug!("nvhost-ctrl-gpu:ZCullGetInfo");
            }
            0x4705 => {
                log::debug!("nvhost-ctrl-gpu:GetCharacteristics");
                if out.len() >= 0xA0 {
                    out[0..8].copy_from_slice(&0xA0u64.to_le_bytes());
                    out[8..16].copy_from_slice(&0xA0u64.to_le_bytes());
                    out[16..20].copy_from_slice(&0x00000021u32.to_le_bytes());
                    out[20..24].copy_from_slice(&0x00000000u32.to_le_bytes());
                    out[24..28].copy_from_slice(&0x000000b3u32.to_le_bytes());
                    out[28..32].copy_from_slice(&0x00000001u32.to_le_bytes());
                    out[32..36].copy_from_slice(&0x00000003u32.to_le_bytes());
                    out[36..40].copy_from_slice(&0x00010000u32.to_le_bytes());
                    out[64..68].copy_from_slice(&2u32.to_le_bytes());
                    out[68..72].copy_from_slice(&8u32.to_le_bytes());
                    out[72..76].copy_from_slice(&512u32.to_le_bytes());
                    out[76..80].copy_from_slice(&4u32.to_le_bytes());
                    out[80..84].copy_from_slice(&1024u32.to_le_bytes());
                    out[84..88].copy_from_slice(&0u32.to_le_bytes());
                    out[88..92].copy_from_slice(&192u32.to_le_bytes());
                    out[92..96].copy_from_slice(&192u32.to_le_bytes());
                    out[96..100].copy_from_slice(&192u32.to_le_bytes());
                    out[100..104].copy_from_slice(&65536u32.to_le_bytes());
                    out[104..108].copy_from_slice(&0u32.to_le_bytes());
                    out[108..112].copy_from_slice(&0u32.to_le_bytes());
                }
            }
            0x4706 => { log::debug!("nvhost-ctrl-gpu:GetTpcMasks"); }
            0x4714 => { log::debug!("nvhost-ctrl-gpu:GetActiveSlotMask"); }
            0x4718 => {
                log::debug!("nvhost-ctrl-gpu:GetGpuTime");
                if out.len() >= 8 { out[0..8].copy_from_slice(&0u64.to_le_bytes()); }
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
                log::debug!("nvhost-as-gpu:AllocSpace");
                if req.in_data.len() >= 24 && out.len() >= 24 {
                    let offset_in: u64 = u64::from_le_bytes([
                        req.in_data[16], req.in_data[17], req.in_data[18], req.in_data[19],
                        req.in_data[20], req.in_data[21], req.in_data[22], req.in_data[23],
                    ]);
                    let alloc = if offset_in != 0 { offset_in } else { 0x10000_0000u64 };
                    out[16..24].copy_from_slice(&alloc.to_le_bytes());
                }
            }
            0x4106 => { log::debug!("nvhost-as-gpu:FreeSpace"); }
            0x4114 => {
                log::debug!("nvhost-as-gpu:MapBufferEx");
                if req.in_data.len() >= 40 && out.len() >= 40 {
                    let offset_in: u64 = u64::from_le_bytes([
                        req.in_data[32], req.in_data[33], req.in_data[34], req.in_data[35],
                        req.in_data[36], req.in_data[37], req.in_data[38], req.in_data[39],
                    ]);
                    let mapped = if offset_in != 0 { offset_in } else { 0x10000_0000u64 };
                    out[32..40].copy_from_slice(&mapped.to_le_bytes());
                }
            }
            0x4105 => { log::debug!("nvhost-as-gpu:UnmapBuffer"); }
            0x4108 => {
                log::debug!("nvhost-as-gpu:GetVaRegions");
            }
            0x4109 => { log::debug!("nvhost-as-gpu:InitializeEx"); }
            other => {
                log::debug!("nvhost-as-gpu: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        let mut out = vec![0u8; req.out_size];
        let n = req.in_data.len().min(out.len());
        out[..n].copy_from_slice(&req.in_data[..n]);

        match cmd {
            0x4801 => { log::debug!("nvhost-gpu:SetNvmapFd"); }
            0x4803 => { log::debug!("nvhost-gpu:ChannelSetTimeout"); }
            0x4808 | 0x481b => {
                log::debug!("nvhost-gpu:SubmitGPFIFO");
                if out.len() >= 24 {
                    out[16..20].copy_from_slice(&1u32.to_le_bytes());
                    out[20..24].copy_from_slice(&1u32.to_le_bytes());
                }
            }
            0x4809 => { log::debug!("nvhost-gpu:AllocateObjectContext"); }
            0x480b => { log::debug!("nvhost-gpu:ZCullBind"); }
            0x480c => { log::debug!("nvhost-gpu:SetErrorNotifier"); }
            0x480d => { log::debug!("nvhost-gpu:SetChannelPriority"); }
            0x481a => {
                log::debug!("nvhost-gpu:AllocGPFIFOEx2");
                if out.len() >= 24 {
                    let event_id = self.next_event_id;
                    self.next_event_id = self.next_event_id.wrapping_add(1);
                    out[20..24].copy_from_slice(&event_id.to_le_bytes());
                }
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
            0x001b => { log::debug!("nvhost-ctrl:SyncptRead"); }
            0x001c => { log::debug!("nvhost-ctrl:SyncptIncr"); }
            0x001d => { log::debug!("nvhost-ctrl:SyncptWait"); }
            0x001e => { log::debug!("nvhost-ctrl:GetConfig"); }
            0x001f => { log::debug!("nvhost-ctrl:EventSignal"); }
            0x0020 => { log::debug!("nvhost-ctrl:EventWait"); }
            0x0021 => { log::debug!("nvhost-ctrl:EventWaitAsync"); }
            0x0022 => { log::debug!("nvhost-ctrl:EventRegister"); }
            0x0023 => { log::debug!("nvhost-ctrl:EventUnregister"); }
            0x0033 => { log::debug!("nvhost-ctrl:GetGpuCharacteristics"); }
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
        std::mem::take(&mut *self.frame_queue.lock())
    }

    pub fn submit_frame(&self, frame: QueuedFrame) {
        self.frame_queue.lock().push(frame);
    }
}

impl Default for Nvdrv {
    fn default() -> Self {
        Self::new()
    }
}

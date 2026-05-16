use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub mod bufferqueue;
pub mod gpu;
pub use bufferqueue::{BufferQueue, GraphicBuffer, QueuedFrame};
pub use gpu::GpuContext;

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
            gpu: Arc::new(GpuContext::new()),
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
        self.dispatch_ioctl_with_mem(req, &|_, _| false)
    }

    pub fn dispatch_ioctl_with_mem(
        &mut self,
        req: IoctlRequest,
        mem_read: &dyn Fn(u64, &mut [u8]) -> bool,
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
            NvDevice::NvhostGpu => self.nvhost_gpu_ioctl_with_mem(cmd, &req, mem_read),
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
                if out.len() < 8 { out.resize(8, 0); }
                out[0..4].copy_from_slice(&size.to_le_bytes());
                out[4..8].copy_from_slice(&id.to_le_bytes());
                log::debug!("nvmap:Create in_data={:02x?} → size={} id={}", &req.in_data[..req.in_data.len().min(16)], size, id);
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
            0x4714 => { log::debug!("nvhost-ctrl-gpu:GetL2State (legacy gfx)"); }
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
                log::debug!("nvhost-as-gpu:GetVaRegions small={} big={}", small_pages, big_pages);
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
                if req.in_data.len() >= 8 {
                    let gpu_va = u64::from_le_bytes([
                        req.in_data[0], req.in_data[1], req.in_data[2], req.in_data[3],
                        req.in_data[4], req.in_data[5], req.in_data[6], req.in_data[7],
                    ]);
                    log::debug!("nvhost-as-gpu:UnmapBuffer gpu_va={:#x}", gpu_va);
                }
            }
            0x4109 => { log::debug!("nvhost-as-gpu:AllocAsEx (InitializeEx)"); }
            other => {
                log::debug!("nvhost-as-gpu: unknown ioctl cmd={:#x}", other);
            }
        }
        IoctlOutcome::ok(out)
    }

    fn nvhost_gpu_ioctl(&mut self, cmd: u16, req: &IoctlRequest) -> IoctlOutcome {
        self.nvhost_gpu_ioctl_with_mem(cmd, req, &|_, _| false)
    }

    fn nvhost_gpu_ioctl_with_mem(&mut self, cmd: u16, req: &IoctlRequest, mem_read: &dyn Fn(u64, &mut [u8]) -> bool) -> IoctlOutcome {
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
                        let (syncpt_id, syncpt_value) = self.gpu.process_inline_gpfifo(&entries, mem_read);
                        log::debug!("nvhost-gpu:SubmitGPFIFO processed {} entries (draws={}, clears={})",
                            entries.len(), self.gpu.maxwell3d.lock().draw_count(),
                            self.gpu.maxwell3d.lock().clear_count());
                        if out.len() >= 24 {
                            out[16..20].copy_from_slice(&syncpt_id.to_le_bytes());
                            out[20..24].copy_from_slice(&syncpt_value.to_le_bytes());
                        }
                    } else if cmd == 0x481b && address != 0 {
                        let (syncpt_id, syncpt_value) = self.gpu.submit_gpfifo(address, num_entries, mem_read);
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

    pub fn gpu_draw_count(&self) -> u64 {
        self.gpu.maxwell3d.lock().draw_count()
    }
}

impl Default for Nvdrv {
    fn default() -> Self {
        Self::new()
    }
}

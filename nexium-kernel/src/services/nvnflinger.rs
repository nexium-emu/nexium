use crate::services::{FrameOut, IpcCtx};
use nexium_common::result::SUCCESS;

const QUEUE_BUFFER: u32 = 7;
const DEQUEUE_BUFFER: u32 = 3;
const REQUEST_BUFFER: u32 = 1;
const CONNECT: u32 = 10;
const DISCONNECT: u32 = 11;
const SET_PREALLOCATED_BUFFER: u32 = 14;

pub struct BufferQueueService {
    pub width: u32,
    pub height: u32,
    pub connected: bool,
}

impl BufferQueueService {
    pub fn new() -> Self {
        Self {
            width: 1280,
            height: 720,
            connected: false,
        }
    }

    pub fn dispatch(&mut self, cmd_id: u32, ctx: &mut IpcCtx) -> u32 {
        log::info!("nvnflinger/dispdrv cmd: {}", cmd_id);
        match cmd_id {
            0 => self.transact_parcel(ctx),
            1 => {
                log::info!("AdjustRefcount");
                SUCCESS
            }
            2 => {
                log::info!("GetNativeHandle");
                SUCCESS
            }
            3 => self.transact_parcel(ctx),
            _ => {
                log::info!("nvnflinger stub command: {}", cmd_id);
                SUCCESS
            }
        }
    }

    fn transact_parcel(&mut self, ctx: &mut IpcCtx) -> u32 {
        if ctx.tls_buf.len() < 28 {
            return SUCCESS;
        }

        let transaction_code = u32::from_le_bytes([
            ctx.tls_buf[20],
            ctx.tls_buf[21],
            ctx.tls_buf[22],
            ctx.tls_buf[23],
        ]);

        log::trace!("IGBP transaction code: {}", transaction_code);

        match transaction_code {
            CONNECT => {
                self.connected = true;
                log::debug!("IGBP::Connect");
                SUCCESS
            }
            DISCONNECT => {
                self.connected = false;
                log::debug!("IGBP::Disconnect");
                SUCCESS
            }
            SET_PREALLOCATED_BUFFER => {
                let w = u32::from_le_bytes([
                    ctx.tls_buf.get(24).copied().unwrap_or(0),
                    ctx.tls_buf.get(25).copied().unwrap_or(0),
                    ctx.tls_buf.get(26).copied().unwrap_or(0),
                    ctx.tls_buf.get(27).copied().unwrap_or(0),
                ]);
                let h = u32::from_le_bytes([
                    ctx.tls_buf.get(28).copied().unwrap_or(0),
                    ctx.tls_buf.get(29).copied().unwrap_or(0),
                    ctx.tls_buf.get(30).copied().unwrap_or(0),
                    ctx.tls_buf.get(31).copied().unwrap_or(0),
                ]);
                if w > 0 {
                    self.width = w;
                }
                if h > 0 {
                    self.height = h;
                }
                log::debug!("IGBP::SetPreallocatedBuffer {}x{}", self.width, self.height);
                SUCCESS
            }
            QUEUE_BUFFER => {
                log::debug!("IGBP::QueueBuffer {}x{}", self.width, self.height);
                let frame = self.compose_frame();
                if ctx.pending_frames.len() >= 2 {
                    ctx.pending_frames.remove(0);
                }
                ctx.pending_frames.push(frame);
                SUCCESS
            }
            REQUEST_BUFFER | DEQUEUE_BUFFER => {
                log::trace!("IGBP buffer op {}", transaction_code);
                SUCCESS
            }
            _ => {
                log::trace!("IGBP unknown transaction: {}", transaction_code);
                SUCCESS
            }
        }
    }

    fn compose_frame(&self) -> FrameOut {
        if let Some((w, h, px)) = nexium_common::frame_present::peek_last_presented_clone() {
            log::trace!(
                "nvnflinger present: forwarding GPU frame {}x{} ({} bytes)",
                w,
                h,
                px.len()
            );
            return FrameOut {
                width: w,
                height: h,
                pixels: px,
                depth: None,
            };
        }
        let pixel_count = (self.width * self.height) as usize;
        FrameOut {
            width: self.width,
            height: self.height,
            pixels: vec![0x10u8; pixel_count * 4],
            depth: None,
        }
    }
}

impl Default for BufferQueueService {
    fn default() -> Self {
        Self::new()
    }
}

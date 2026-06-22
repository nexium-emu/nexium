pub mod cmif;
pub mod dispatch;
pub mod hipc;
pub mod parcel;
pub mod request;

pub use cmif::{CmifDomainInHeader, CmifDomainOutHeader, CmifInHeader, CmifOutHeader};
pub use cmif::{CMIF_DOMAIN_REQ_CLOSE, CMIF_DOMAIN_REQ_SEND, CMIF_IN_MAGIC, CMIF_OUT_MAGIC};
pub use hipc::{HipcCommandType, HipcHeader, HipcSpecialHeader};
pub use hipc::{TLS_BUFFER_SIZE, TLS_REQUEST_OFFSET};

#[derive(Debug)]
pub enum IpcError {
    BufferTooSmall { have: usize, need: usize },
    BadCmifMagic { got: u32 },
    BadCommandType { got: u16 },
    ShortInput { have: usize, need: usize },
}

pub type IpcResult<T> = Result<T, IpcError>;

#[derive(Copy, Clone, Debug)]
pub struct IpcBuffer {
    pub addr: u64,
    pub size: u64,
    pub mode: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct DomainIn {
    pub kind: u8,
    pub object_id: u32,
    pub num_in_objects: u8,
    pub data_size: u16,
}

#[derive(Copy, Clone, Debug)]
pub struct TipcInfo {
    pub raw_request_id: u16,
}

#[derive(Clone)]
pub struct IpcCtx {
    pub buf: Vec<u8>,
    pub hipc: HipcHeader,
    pub special: Option<HipcSpecialHeader>,
    pub send_pid: Option<u64>,
    pub copy_handles: Vec<u32>,
    pub move_handles: Vec<u32>,

    pub send_buffers: Vec<IpcBuffer>,
    pub recv_buffers: Vec<IpcBuffer>,
    pub exch_buffers: Vec<IpcBuffer>,
    pub send_statics: Vec<IpcBuffer>,
    pub recv_statics: Vec<IpcBuffer>,

    pub domain: Option<DomainIn>,

    pub cmif_in: CmifInHeader,
    pub cmif_in_data_off: usize,
    pub cmif_in_data_len: usize,

    pub in_objects: Vec<u32>,

    pub tipc: Option<TipcInfo>,
}

impl IpcCtx {
    pub fn parse(mut buf: Vec<u8>, is_domain: bool) -> IpcResult<Self> {
        if buf.len() < 8 {
            return Err(IpcError::BufferTooSmall {
                have: buf.len(),
                need: 8,
            });
        }

        let mut hdr_bytes = [0u8; 8];
        hdr_bytes.copy_from_slice(&buf[..8]);
        let hipc = HipcHeader::from_le_bytes(hdr_bytes);
        let cmd_type = hipc.command_type();

        if cmd_type >= 16 {
            return Self::parse_tipc(buf, hipc, cmd_type);
        }

        let mut cursor = 8usize;

        let mut special = None;
        let mut send_pid_value: Option<u64> = None;
        let mut copy_handles = Vec::new();
        let mut move_handles = Vec::new();

        if hipc.has_special_header() {
            if buf.len() < cursor + 4 {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 4,
                });
            }
            let mut sh_bytes = [0u8; 4];
            sh_bytes.copy_from_slice(&buf[cursor..cursor + 4]);
            let sh = HipcSpecialHeader::from_u32(u32::from_le_bytes(sh_bytes));
            cursor += 4;

            if sh.send_pid() {
                if buf.len() < cursor + 8 {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 8,
                    });
                }
                let p = u64::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                    buf[cursor + 4],
                    buf[cursor + 5],
                    buf[cursor + 6],
                    buf[cursor + 7],
                ]);
                send_pid_value = Some(p);
                cursor += 8;
            }

            for _ in 0..sh.num_copy_handles() {
                if cursor + 4 > buf.len() {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 4,
                    });
                }
                let h = u32::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                ]);
                copy_handles.push(h);
                cursor += 4;
            }

            for _ in 0..sh.num_move_handles() {
                if cursor + 4 > buf.len() {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 4,
                    });
                }
                let h = u32::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                ]);
                move_handles.push(h);
                cursor += 4;
            }

            special = Some(sh);
        }

        let mut send_statics = Vec::new();
        for _ in 0..hipc.num_send_statics() {
            if cursor + 8 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 8,
                });
            }
            let word0 = u32::from_le_bytes([
                buf[cursor],
                buf[cursor + 1],
                buf[cursor + 2],
                buf[cursor + 3],
            ]);
            let addr_low = u32::from_le_bytes([
                buf[cursor + 4],
                buf[cursor + 5],
                buf[cursor + 6],
                buf[cursor + 7],
            ]) as u64;
            let index = (word0 & 0x3F) as u64;
            let addr_high = ((word0 >> 6) & 0x3F) as u64;
            let addr_mid = ((word0 >> 12) & 0xF) as u64;
            let size = ((word0 >> 16) & 0xFFFF) as u64;
            let addr = addr_low | (addr_mid << 32) | (addr_high << 36);
            send_statics.push(IpcBuffer {
                addr,
                size,
                mode: index as u32,
            });
            cursor += 8;
        }

        let mut send_buffers = Vec::new();
        for _ in 0..hipc.num_send_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            send_buffers.push(parse_abc_descriptor(&buf, cursor));
            cursor += 12;
        }

        let mut recv_buffers = Vec::new();
        for _ in 0..hipc.num_recv_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            recv_buffers.push(parse_abc_descriptor(&buf, cursor));
            cursor += 12;
        }

        let mut exch_buffers = Vec::new();
        for _ in 0..hipc.num_exch_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            exch_buffers.push(parse_abc_descriptor(&buf, cursor));
            cursor += 12;
        }

        let data_words_start = cursor;
        cursor = (cursor + 15) & !15;
        let _raw_data_off = cursor;

        let recv_list_count = match hipc.recv_static_mode() {
            0 => 0,
            2 => 1,
            n => (n as usize).saturating_sub(2),
        };
        let mut recv_statics = Vec::new();
        if recv_list_count > 0 {
            let raw_end = data_words_start + (hipc.num_data_words() as usize) * 4;
            let list_start = if hipc.recv_list_offset() != 0 {
                (hipc.recv_list_offset() as usize) * 4
            } else {
                raw_end
            };
            for i in 0..recv_list_count {
                let off = list_start + i * 8;
                if off + 8 > buf.len() {
                    break;
                }
                let lo =
                    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) as u64;
                let hi =
                    u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]])
                        as u64;
                let packed = lo | (hi << 32);
                let addr = packed & 0x0000_FFFF_FFFF_FFFF;
                let size = (packed >> 48) & 0xFFFF;
                recv_statics.push(IpcBuffer {
                    addr,
                    size,
                    mode: 0,
                });
            }
        }

        let mut domain = None;
        let mut in_objects = Vec::new();

        if is_domain {
            if buf.len() < cursor + 16 {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 16,
                });
            }
            let kind = buf[cursor];
            let num_in_objects = buf[cursor + 1];
            let data_size = u16::from_le_bytes([buf[cursor + 2], buf[cursor + 3]]);
            let object_id = u32::from_le_bytes([
                buf[cursor + 4],
                buf[cursor + 5],
                buf[cursor + 6],
                buf[cursor + 7],
            ]);
            cursor += 16;

            domain = Some(DomainIn {
                kind,
                object_id,
                num_in_objects,
                data_size,
            });

            if kind == CMIF_DOMAIN_REQ_CLOSE {
                let zero = CmifInHeader {
                    magic: 0,
                    version: 0,
                    cmd_id: 0,
                    token: 0,
                };
                return Ok(IpcCtx {
                    buf,
                    hipc,
                    special,
                    send_pid: send_pid_value,
                    copy_handles,
                    move_handles,
                    send_buffers,
                    recv_buffers,
                    exch_buffers,
                    send_statics,
                    recv_statics,
                    domain,
                    cmif_in: zero,
                    cmif_in_data_off: cursor,
                    cmif_in_data_len: 0,
                    in_objects,
                    tipc: None,
                });
            }
        }

        if buf.len() < cursor + 16 {
            return Err(IpcError::BufferTooSmall {
                have: buf.len(),
                need: cursor + 16,
            });
        }
        let magic = u32::from_le_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
        ]);
        if magic != CMIF_IN_MAGIC {
            return Err(IpcError::BadCmifMagic { got: magic });
        }
        let version = u32::from_le_bytes([
            buf[cursor + 4],
            buf[cursor + 5],
            buf[cursor + 6],
            buf[cursor + 7],
        ]);
        let cmd_id = u32::from_le_bytes([
            buf[cursor + 8],
            buf[cursor + 9],
            buf[cursor + 10],
            buf[cursor + 11],
        ]);
        let token = u32::from_le_bytes([
            buf[cursor + 12],
            buf[cursor + 13],
            buf[cursor + 14],
            buf[cursor + 15],
        ]);

        let cmif_in = CmifInHeader {
            magic,
            version,
            cmd_id,
            token,
        };
        let cmif_in_data_off = cursor + 16;

        let raw_size = (hipc.num_data_words() as usize) * 4;
        let cmif_in_data_len = if let Some(d) = domain {
            (d.data_size as usize).saturating_sub(16)
        } else {
            raw_size.saturating_sub(16)
        };

        let needed = cmif_in_data_off + cmif_in_data_len;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }

        if let Some(d) = domain {
            let obj_off = cmif_in_data_off + cmif_in_data_len;
            for i in 0..d.num_in_objects as usize {
                let off = obj_off + i * 4;
                if off + 4 > buf.len() {
                    break;
                }
                let obj_id =
                    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                in_objects.push(obj_id);
            }
        }

        Ok(IpcCtx {
            buf,
            hipc,
            special,
            send_pid: send_pid_value,
            copy_handles,
            move_handles,
            send_buffers,
            recv_buffers,
            exch_buffers,
            send_statics,
            recv_statics,
            domain,
            cmif_in,
            cmif_in_data_off,
            cmif_in_data_len,
            in_objects,
            tipc: None,
        })
    }

    fn parse_tipc(mut buf: Vec<u8>, hipc: HipcHeader, cmd_type: u16) -> IpcResult<Self> {
        let mut cursor = 8usize;
        let mut special = None;
        let mut send_pid_value: Option<u64> = None;
        let mut copy_handles = Vec::new();
        let mut move_handles = Vec::new();

        if hipc.has_special_header() {
            if buf.len() < cursor + 4 {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 4,
                });
            }
            let sh = HipcSpecialHeader::from_u32(u32::from_le_bytes([
                buf[cursor],
                buf[cursor + 1],
                buf[cursor + 2],
                buf[cursor + 3],
            ]));
            cursor += 4;
            if sh.send_pid() {
                if buf.len() < cursor + 8 {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 8,
                    });
                }
                send_pid_value = Some(u64::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                    buf[cursor + 4],
                    buf[cursor + 5],
                    buf[cursor + 6],
                    buf[cursor + 7],
                ]));
                cursor += 8;
            }
            for _ in 0..sh.num_copy_handles() {
                if cursor + 4 > buf.len() {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 4,
                    });
                }
                copy_handles.push(u32::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                ]));
                cursor += 4;
            }
            for _ in 0..sh.num_move_handles() {
                if cursor + 4 > buf.len() {
                    return Err(IpcError::BufferTooSmall {
                        have: buf.len(),
                        need: cursor + 4,
                    });
                }
                move_handles.push(u32::from_le_bytes([
                    buf[cursor],
                    buf[cursor + 1],
                    buf[cursor + 2],
                    buf[cursor + 3],
                ]));
                cursor += 4;
            }
            special = Some(sh);
        }

        for _ in 0..hipc.num_send_statics() {
            if cursor + 8 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 8,
                });
            }
            cursor += 8;
        }
        for _ in 0..hipc.num_send_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            cursor += 12;
        }
        for _ in 0..hipc.num_recv_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            cursor += 12;
        }
        for _ in 0..hipc.num_exch_buffers() {
            if cursor + 12 > buf.len() {
                return Err(IpcError::BufferTooSmall {
                    have: buf.len(),
                    need: cursor + 12,
                });
            }
            cursor += 12;
        }

        let _data_words_start = cursor;
        cursor = (cursor + 15) & !15;
        let raw_data_off = cursor;
        let raw_data_len = (hipc.num_data_words() as usize) * 4;

        let needed = raw_data_off + raw_data_len;
        if buf.len() < needed {
            buf.resize(needed, 0);
        }

        let logical_cmd_id = (cmd_type as u32).saturating_sub(16);
        let synth_cmif = CmifInHeader {
            magic: 0,
            version: 0,
            cmd_id: logical_cmd_id,
            token: 0,
        };

        Ok(IpcCtx {
            buf,
            hipc,
            special,
            send_pid: send_pid_value,
            copy_handles,
            move_handles,
            send_buffers: Vec::new(),
            recv_buffers: Vec::new(),
            exch_buffers: Vec::new(),
            send_statics: Vec::new(),
            recv_statics: Vec::new(),
            domain: None,
            cmif_in: synth_cmif,
            cmif_in_data_off: raw_data_off,
            cmif_in_data_len: raw_data_len,
            in_objects: Vec::new(),
            tipc: Some(TipcInfo {
                raw_request_id: cmd_type,
            }),
        })
    }

    pub fn input_data(&self) -> &[u8] {
        &self.buf[self.cmif_in_data_off..self.cmif_in_data_off + self.cmif_in_data_len]
    }
}

fn parse_abc_descriptor(buf: &[u8], off: usize) -> IpcBuffer {
    let size_low = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    let addr_low = u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]);
    let packed = u32::from_le_bytes([buf[off + 8], buf[off + 9], buf[off + 10], buf[off + 11]]);
    let mode = packed & 0x3;
    let addr_high = (packed >> 2) & 0x003F_FFFF;
    let size_high = (packed >> 24) & 0xF;
    let addr_mid = (packed >> 28) & 0xF;
    let addr = (addr_low as u64) | ((addr_mid as u64) << 32) | ((addr_high as u64) << 36);
    let size = (size_low as u64) | ((size_high as u64) << 32);
    IpcBuffer { addr, size, mode }
}

impl IpcCtx {
    fn _placeholder() {}

    pub fn build_response(&self, result: u32, out_data: &[u8]) -> Vec<u8> {
        let mut response = vec![0u8; 16 + out_data.len()];
        let cmif_out = if result == 0 {
            CmifOutHeader::success(self.cmif_in.token)
        } else {
            CmifOutHeader::error(result, self.cmif_in.token)
        };
        let bytes = bytemuck::bytes_of(&cmif_out);
        response[..16].copy_from_slice(bytes);
        if !out_data.is_empty() {
            response[16..].copy_from_slice(out_data);
        }
        response
    }
}

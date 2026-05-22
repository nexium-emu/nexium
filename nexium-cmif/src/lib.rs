pub use nexium_cmif_macros::{service, command};

pub trait Memory {
    fn read(&self, addr: u64, dst: &mut [u8]) -> bool;
    fn write(&self, addr: u64, src: &[u8]) -> bool;
}

#[derive(Copy, Clone, Debug)]
pub struct CmifBuffer {
    pub addr: u64,
    pub size: u64,
}

pub struct DispatchCtx<'a> {
    pub input_data: &'a [u8],
    pub recv_buffers: &'a [CmifBuffer],
    pub recv_statics: &'a [CmifBuffer],
    pub send_buffers: &'a [CmifBuffer],
    pub send_statics: &'a [CmifBuffer],
    pub mem: &'a dyn Memory,
}

pub struct DispatchOutcome {
    pub result: u32,
    pub inline_out: Vec<u8>,
}

impl DispatchOutcome {
    pub fn ok() -> Self {
        Self { result: 0, inline_out: Vec::new() }
    }

    pub fn ok_with(data: Vec<u8>) -> Self {
        Self { result: 0, inline_out: data }
    }

    pub fn err(rc: u32) -> Self {
        Self { result: rc, inline_out: Vec::new() }
    }
}

pub trait CmifReadable: Sized {
    const CMIF_SIZE: usize;
    fn read_le(bytes: &[u8]) -> Self;
}

impl CmifReadable for u8 {
    const CMIF_SIZE: usize = 1;
    fn read_le(b: &[u8]) -> Self { b[0] }
}

impl CmifReadable for i8 {
    const CMIF_SIZE: usize = 1;
    fn read_le(b: &[u8]) -> Self { b[0] as i8 }
}

impl CmifReadable for u16 {
    const CMIF_SIZE: usize = 2;
    fn read_le(b: &[u8]) -> Self { u16::from_le_bytes([b[0], b[1]]) }
}

impl CmifReadable for i16 {
    const CMIF_SIZE: usize = 2;
    fn read_le(b: &[u8]) -> Self { i16::from_le_bytes([b[0], b[1]]) }
}

impl CmifReadable for u32 {
    const CMIF_SIZE: usize = 4;
    fn read_le(b: &[u8]) -> Self { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
}

impl CmifReadable for i32 {
    const CMIF_SIZE: usize = 4;
    fn read_le(b: &[u8]) -> Self { i32::from_le_bytes([b[0], b[1], b[2], b[3]]) }
}

impl CmifReadable for u64 {
    const CMIF_SIZE: usize = 8;
    fn read_le(b: &[u8]) -> Self {
        u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
}

impl CmifReadable for i64 {
    const CMIF_SIZE: usize = 8;
    fn read_le(b: &[u8]) -> Self {
        i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }
}

pub trait CmifWritable {
    fn write_le(&self, out: &mut Vec<u8>);
}

impl CmifWritable for () {
    fn write_le(&self, _out: &mut Vec<u8>) {}
}

impl CmifWritable for u8 {
    fn write_le(&self, out: &mut Vec<u8>) { out.push(*self); }
}

impl CmifWritable for i8 {
    fn write_le(&self, out: &mut Vec<u8>) { out.push(*self as u8); }
}

impl CmifWritable for u16 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

impl CmifWritable for i16 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

impl CmifWritable for u32 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

impl CmifWritable for i32 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

impl CmifWritable for u64 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

impl CmifWritable for i64 {
    fn write_le(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
}

pub struct RecvBuffer<'a> {
    pub addr: u64,
    pub size: u64,
    pub mem: &'a dyn Memory,
}

impl<'a> RecvBuffer<'a> {
    pub fn from_ctx(ctx: &DispatchCtx<'a>) -> Option<Self> {
        let buf = ctx.recv_buffers.iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.recv_statics.iter().find(|b| b.size > 0 && b.addr != 0))?;
        Some(Self { addr: buf.addr, size: buf.size, mem: ctx.mem })
    }

    pub fn write_all(&self, data: &[u8]) -> usize {
        let len = data.len().min(self.size as usize);
        if !self.mem.write(self.addr, &data[..len]) {
            log::warn!("RecvBuffer::write_all: failed to write {} bytes at {:#x}", len, self.addr);
            return 0;
        }
        len
    }

    pub fn fill_zero(&self) {
        let chunk = vec![0u8; self.size.min(0x1000) as usize];
        let mut remaining = self.size as usize;
        let mut off = 0u64;
        while remaining > 0 {
            let n = remaining.min(chunk.len());
            let _ = self.mem.write(self.addr + off, &chunk[..n]);
            remaining -= n;
            off += n as u64;
        }
    }
}

pub struct SendBuffer<'a> {
    pub addr: u64,
    pub size: u64,
    pub mem: &'a dyn Memory,
}

impl<'a> SendBuffer<'a> {
    pub fn from_ctx(ctx: &DispatchCtx<'a>) -> Option<Self> {
        let buf = ctx.send_buffers.iter()
            .find(|b| b.size > 0 && b.addr != 0)
            .or_else(|| ctx.send_statics.iter().find(|b| b.size > 0 && b.addr != 0))?;
        Some(Self { addr: buf.addr, size: buf.size, mem: ctx.mem })
    }

    pub fn read_all(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.size as usize];
        if !self.mem.read(self.addr, &mut out) {
            log::warn!("SendBuffer::read_all: failed to read {} bytes at {:#x}", self.size, self.addr);
            out.clear();
        }
        out
    }
}

pub fn read_in_arg<T: CmifReadable>(input: &[u8], off: usize) -> T {
    if off + T::CMIF_SIZE > input.len() {
        log::warn!(
            "cmif: read_in_arg overflow off={} size={} input_len={}",
            off, T::CMIF_SIZE, input.len()
        );
        return T::read_le(&[0u8; 16][..T::CMIF_SIZE]);
    }
    T::read_le(&input[off..off + T::CMIF_SIZE])
}

pub fn finish<T: CmifWritable>(result: Result<T, u32>) -> DispatchOutcome {
    match result {
        Ok(v) => {
            let mut buf = Vec::new();
            v.write_le(&mut buf);
            DispatchOutcome { result: 0, inline_out: buf }
        }
        Err(rc) => DispatchOutcome::err(rc),
    }
}

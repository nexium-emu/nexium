use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

pub fn get_size(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 { 0 }

pub fn write(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _offset: u64, _data: &[u8]) {}

pub fn read(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _offset: u64, _out: &mut Vec<u8>) {}

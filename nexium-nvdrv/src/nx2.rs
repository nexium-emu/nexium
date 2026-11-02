use crate::Nvdrv;

pub const CHARACTERISTICS_SIZE: usize = 0xD0;
pub const THREED_CLASS: u32 = 0xC797;
pub const COMPUTE_CLASS: u32 = 0xC7C0;
pub const GPFIFO_CLASS: u32 = 0xC56F;
pub const DMA_COPY_CLASS: u32 = 0xC7B5;
pub const SUBMIT_INCREMENT_VALUE: u32 = 1 << 8;
pub const USER_VA_START: u64 = 0x800_0000;
pub const USER_VA_END: u64 = 0xC0_0000_0000;
pub const SYNCPOINT_SHIM_VA: u64 = USER_VA_END + (1 << 32) - 0x1000;
pub const SYNCPOINT_SHIM_STRIDE: u32 = 4;
pub const SYNCPOINT_SHIM_COUNT: u32 = 192;

const ARCH_AMPERE: u32 = 0x170;
const IMPL_T239: u32 = 0xE;
const TPC_PER_GPC: u32 = 6;
const SM_VERSION: u32 = 0x808;
const CHIPNAME: u64 = u64::from_le_bytes(*b"ga10f\0\0\0");
const FLAG_TSG: u64 = 1 << 8;
const FLAG_TSG_SUBCONTEXTS: u64 = 1 << 22;
const FLAG_SYNCPOINT_ADDRESS: u64 = 1 << 26;
const FLAG_MAPPING_MODIFY: u64 = 1 << 34;
const INTERFACE_VERSIONS: std::ops::RangeInclusive<usize> = 0x70..=0x80;
const INTERFACE_VERSION: i16 = 0x40;
const MAX_SUBCONTEXTS_OFFSET: usize = 0xB4;
const MAX_SUBCONTEXTS: u32 = 64;
const ENGINE_GR: u32 = 0;
const ENGINE_ASYNC_COPY: u32 = 2;
const ENGINES: [[u32; 4]; 2] = [[ENGINE_GR, 0, 0, 0], [ENGINE_ASYNC_COPY, 0, 1, 0]];

fn put_u32(block: &mut [u8], offset: usize, value: u32) {
    if let Some(field) = block.get_mut(offset..offset + 4) {
        field.copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u64(block: &mut [u8], offset: usize, value: u64) {
    if let Some(field) = block.get_mut(offset..offset + 8) {
        field.copy_from_slice(&value.to_le_bytes());
    }
}

pub fn apply_characteristics(gc: &mut [u8]) {
    for (offset, value) in [
        (0x00, ARCH_AMPERE),
        (0x04, IMPL_T239),
        (0x20, TPC_PER_GPC),
        (0x3C, SM_VERSION),
        (0x40, SM_VERSION),
        (0x5C, THREED_CLASS),
        (0x60, COMPUTE_CLASS),
        (0x64, GPFIFO_CLASS),
        (0x6C, DMA_COPY_CLASS),
        (MAX_SUBCONTEXTS_OFFSET, MAX_SUBCONTEXTS),
    ] {
        put_u32(gc, offset, value);
    }
    let flags = gc
        .get(0x50..0x58)
        .map_or(0, |field| u64::from_le_bytes(field.try_into().unwrap()));
    put_u64(
        gc,
        0x50,
        flags | FLAG_TSG | FLAG_TSG_SUBCONTEXTS | FLAG_SYNCPOINT_ADDRESS | FLAG_MAPPING_MODIFY,
    );
    put_u64(gc, 0x90, CHIPNAME);
    for offset in INTERFACE_VERSIONS.step_by(2) {
        if let Some(field) = gc.get_mut(offset..offset + 2) {
            field.copy_from_slice(&INTERFACE_VERSION.to_le_bytes());
        }
    }
}

const MAPPING_OWNER: u32 = u32::MAX;
const MAPPING_WAIT: std::time::Duration = std::time::Duration::from_millis(2);

impl Nvdrv {
    pub fn enable_switch2(&mut self) {
        self.switch2 = true;
        let memory = self.gpu.guest_memory();
        self.gpu.maxwell3d.lock().enable_mme64(memory);
        self.gpu.kepler_compute.lock().qmd3 = Some(Default::default());
        nexium_shader::ampere::set_enabled(true);
        nexium_gpu::texture::set_swapped_gob_sectors(true);
        nexium_gpu::texture::set_ampere_texture_headers(true);
    }

    fn nx2_mappings(&self, wait: bool) -> Option<parking_lot::RwLockWriteGuard<'_, crate::gpu::GpuMappings>> {
        if wait {
            Some(self.gpu.mappings.write())
        } else {
            self.gpu.mappings.try_write_for(MAPPING_WAIT)
        }
    }

    pub fn nx2_map(&mut self, gpu_va: u64, size: u64, cpu_addr: u64, wait: bool) -> bool {
        let Some(update) = self.nx2_mappings(wait).map(|mut mappings| {
            mappings.add_as_gpu_mapping(MAPPING_OWNER, gpu_va, size, cpu_addr, 0, None, None, true)
        }) else {
            return false;
        };
        self.invalidate_texture_mapping_update(&update);
        true
    }

    pub fn nx2_map_sparse(&mut self, gpu_va: u64, size: u64, wait: bool) -> bool {
        let Some(update) = self
            .nx2_mappings(wait)
            .map(|mut mappings| mappings.add_sparse_as_gpu_with_metadata(MAPPING_OWNER, gpu_va, size))
        else {
            return false;
        };
        self.invalidate_texture_mapping_update(&update);
        true
    }

    pub fn nx2_unmap(&mut self, gpu_va: u64, size: u64, wait: bool) -> bool {
        let Some(removed) = self
            .nx2_mappings(wait)
            .map(|mut mappings| mappings.remove_all_contained_with_metadata(MAPPING_OWNER, gpu_va, size))
        else {
            return false;
        };
        if let Ok(removed) = removed {
            self.invalidate_texture_mapping_update(&removed.update);
        }
        true
    }

    pub fn nx2_allocate_syncpoint(&mut self) -> u32 {
        let id = self.next_syncpoint_id;
        self.next_syncpoint_id = self.next_syncpoint_id.wrapping_add(1).max(1);
        id
    }

    pub fn nx2_increment_syncpoint(&mut self, id: u32) -> u32 {
        self.increment_syncpoint(id, 1)
    }

    pub fn nx2_syncpoint_value(&self, id: u32) -> u32 {
        self.syncpoint_value(id)
    }

    pub fn nx2_syncpoint_values(&self, values: &mut [u32]) {
        self.poll_gpu_completions();
        values.fill(0);
        for (&id, &(min, _)) in self.retired_syncpts.lock().iter() {
            if let Some(value) = values.get_mut(id as usize) {
                *value = min;
            }
        }
        for channel in self.gpu.channels.lock().values() {
            if let Some(value) = values.get_mut(channel.syncpt_id as usize) {
                *value = channel.syncpt_min;
            }
        }
    }
}

pub fn is_user_partition(va_start: u64, va_end: u64, va_split: u64) -> bool {
    va_start == USER_VA_START && va_end == USER_VA_END && va_split == 0
}

pub fn is_stub_device(path: &str) -> bool {
    path.trim_end_matches('\0').starts_with("/dev/nvhost-")
}

pub fn echo(in_data: &[u8], out_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_size];
    let copied = in_data.len().min(out_size);
    out[..copied].copy_from_slice(&in_data[..copied]);
    out
}

pub fn syncpoint_shim(out_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; out_size.max(16)];
    out[0..8].copy_from_slice(&SYNCPOINT_SHIM_VA.to_le_bytes());
    out[8..12].copy_from_slice(&SYNCPOINT_SHIM_STRIDE.to_le_bytes());
    out[12..16].copy_from_slice(&SYNCPOINT_SHIM_COUNT.to_le_bytes());
    out
}

pub fn engine_info(in_data: &[u8], out: &mut Vec<u8>) {
    let needed = (ENGINES.len() * 16) as u32;
    let capacity = in_data
        .get(0..4)
        .map_or(0, |field| u32::from_le_bytes(field.try_into().unwrap()));
    if out.len() < 16 {
        out.resize(16, 0);
    }
    out[0..4].copy_from_slice(&needed.to_le_bytes());
    out.truncate(16);
    if capacity >= needed {
        out.extend(ENGINES.iter().flatten().flat_map(|word| word.to_le_bytes()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(block: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(block[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn characteristics_describe_the_ampere_gpu() {
        let mut gc = vec![0u8; CHARACTERISTICS_SIZE];
        gc[0x50] = 0x55;
        apply_characteristics(&mut gc);
        assert_eq!(read_u32(&gc, 0x00), 0x170);
        assert_eq!(read_u32(&gc, 0x04), 0xE);
        assert_eq!(read_u32(&gc, 0x3C), 0x808);
        assert_eq!(read_u32(&gc, 0x5C), 0xC797);
        assert_eq!(read_u32(&gc, 0x64), 0xC56F);
        assert_eq!(read_u32(&gc, 0xB4), 64);
        assert_eq!(&gc[0x90..0x95], b"ga10f");
        assert_eq!(u64::from_le_bytes(gc[0x50..0x58].try_into().unwrap()), 0x55 | 0x100 | 1 << 22 | 1 << 26 | 1 << 34);
        assert!(gc[0x70..0x82].chunks(2).all(|version| version == [0x40, 0]));
    }

    #[test]
    fn engine_info_reports_size_then_engines() {
        let mut out = vec![0u8; 16];
        engine_info(&0u32.to_le_bytes(), &mut out);
        assert_eq!((read_u32(&out, 0), out.len()), (32, 16));
        engine_info(&32u32.to_le_bytes(), &mut out);
        assert_eq!(out.len(), 48);
        assert_eq!([read_u32(&out, 16), read_u32(&out, 24), read_u32(&out, 32), read_u32(&out, 40)], [0, 0, 2, 1]);
    }

    #[test]
    fn short_blocks_are_left_partially_written() {
        let mut gc = vec![0u8; 0xA0];
        apply_characteristics(&mut gc);
        assert_eq!(read_u32(&gc, 0x00), 0x170);
    }
}

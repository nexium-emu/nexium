use crate::bin_read::{u32at, u64at, u8at};

pub const META_MAGIC: u32 = 0x4154454D;
pub const ACID_MAGIC: u32 = 0x44494341;
pub const ACI0_MAGIC: u32 = 0x30494341;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddressSpaceType {
    Is32Bit,
    Is36Bit,
    Is32BitNoMap,
    Is39Bit,
    Is42Bit,
}

impl AddressSpaceType {
    fn from_bits(v: u8) -> Self {
        match v & 0x7 {
            0 => AddressSpaceType::Is32Bit,
            1 => AddressSpaceType::Is36Bit,
            2 => AddressSpaceType::Is32BitNoMap,
            4 => AddressSpaceType::Is42Bit,
            _ => AddressSpaceType::Is39Bit,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AcidLayout {
    Switch1,
    Switch2,
}

impl AcidLayout {
    pub const fn signature_area_size(self) -> usize {
        match self {
            AcidLayout::Switch1 => 0x200,
            AcidLayout::Switch2 => 0x300,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Acid {
    pub layout: AcidLayout,
    pub flags: u32,
    pub program_id_min: u64,
    pub program_id_max: u64,
    pub fs_permissions: u64,
    pub kernel_caps: KernelCaps,
}

impl Acid {
    pub fn parse(block: &[u8]) -> Result<Self, String> {
        let layout = [AcidLayout::Switch1, AcidLayout::Switch2]
            .into_iter()
            .find(|layout| u32at(block, layout.signature_area_size()).ok() == Some(ACID_MAGIC))
            .ok_or("ACID magic not found after a Switch 1 or Switch 2 signature area")?;
        let header = layout.signature_area_size();
        let fac = section(block, u32at(block, header + 0x20)?, u32at(block, header + 0x24)?)?;
        let kc = section(block, u32at(block, header + 0x30)?, u32at(block, header + 0x34)?)?;
        Ok(Self {
            layout,
            flags: u32at(block, header + 0x0C)?,
            program_id_min: u64at(block, header + 0x10)?,
            program_id_max: u64at(block, header + 0x18)?,
            fs_permissions: if fac.is_empty() { 0 } else { u64at(fac, 4)? },
            kernel_caps: KernelCaps::parse_bytes(kc),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Aci {
    pub program_id: u64,
    pub fs_permissions: u64,
    pub kernel_caps: KernelCaps,
}

impl Aci {
    pub fn parse(block: &[u8]) -> Result<Self, String> {
        if u32at(block, 0)? != ACI0_MAGIC {
            return Err("ACI0 magic not found".into());
        }
        let fah = section(block, u32at(block, 0x20)?, u32at(block, 0x24)?)?;
        let kc = section(block, u32at(block, 0x30)?, u32at(block, 0x34)?)?;
        Ok(Self {
            program_id: u64at(block, 0x10)?,
            fs_permissions: if fah.is_empty() { 0 } else { u64at(fah, 4)? },
            kernel_caps: KernelCaps::parse_bytes(kc),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ThreadInfo {
    pub highest_priority: u8,
    pub lowest_priority: u8,
    pub min_core: u8,
    pub max_core: u8,
}

impl ThreadInfo {
    pub fn core_mask(&self) -> u64 {
        bit_range(self.min_core, self.max_core)
    }

    pub fn priority_mask(&self) -> u64 {
        bit_range(self.highest_priority, self.lowest_priority)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KernelVersion {
    pub major: u16,
    pub minor: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct KernelCaps {
    pub thread_info: Option<ThreadInfo>,
    pub svc_mask: [u64; 3],
    pub kernel_version: Option<KernelVersion>,
    pub handle_table_size: Option<u16>,
    pub program_type: Option<u8>,
    pub debug_flags: Option<u8>,
}

impl KernelCaps {
    pub fn parse(words: &[u32]) -> Self {
        let mut caps = Self::default();
        let mut index = 0;
        while let Some(&word) = words.get(index) {
            index += 1;
            match (!word & word.wrapping_add(1)).trailing_zeros() {
                3 => {
                    caps.thread_info = Some(ThreadInfo {
                        lowest_priority: (word >> 4 & 0x3F) as u8,
                        highest_priority: (word >> 10 & 0x3F) as u8,
                        min_core: (word >> 16 & 0xFF) as u8,
                        max_core: (word >> 24) as u8,
                    });
                }
                4 => {
                    let base = (word >> 29) as usize * 24;
                    let mask = word >> 5 & 0xFF_FFFF;
                    for bit in (0..24).filter(|bit| mask >> bit & 1 != 0) {
                        let id = base + bit;
                        caps.svc_mask[id / 64] |= 1 << (id % 64);
                    }
                }
                6 => index += 1,
                13 => caps.program_type = Some((word >> 14 & 0x7) as u8),
                14 => {
                    caps.kernel_version = Some(KernelVersion {
                        major: (word >> 19) as u16,
                        minor: (word >> 15 & 0xF) as u8,
                    });
                }
                15 => caps.handle_table_size = Some((word >> 16 & 0x3FF) as u16),
                16 => caps.debug_flags = Some((word >> 17 & 0x7) as u8),
                _ => {}
            }
        }
        caps
    }

    pub fn parse_bytes(bytes: &[u8]) -> Self {
        let words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect::<Vec<_>>();
        Self::parse(&words)
    }

    pub fn allows_svc(&self, id: u32) -> bool {
        let id = id as usize;
        self.svc_mask
            .get(id / 64)
            .is_some_and(|word| word >> (id % 64) & 1 != 0)
    }

    pub fn svc_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.svc_mask.len() as u32 * 64).filter(|id| self.allows_svc(*id))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Npdm {
    pub is_64bit: bool,
    pub address_space: AddressSpaceType,
    pub main_thread_priority: u8,
    pub main_thread_core: u8,
    pub system_resource_size: u32,
    pub main_stack_size: u32,
    pub title_id: u64,
    pub acid: Option<Acid>,
    pub aci: Option<Aci>,
}

impl Npdm {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let magic = u32at(bytes, 0)?;
        if magic != META_MAGIC {
            return Err(format!("NPDM magic {:#010x} is not META", magic));
        }
        let flags = u8at(bytes, 0x0C)?;
        let is_64bit = flags & 1 != 0;
        let address_space = AddressSpaceType::from_bits(flags >> 1);
        let main_thread_priority = u8at(bytes, 0x0E)?;
        let main_thread_core = u8at(bytes, 0x0F)?;
        let system_resource_size = u32at(bytes, 0x14)?;
        let main_stack_size = u32at(bytes, 0x1C)?;

        let aci_offset = u32at(bytes, 0x70)? as usize;
        let title_id = u64at(bytes, aci_offset + 0x10).unwrap_or(0);
        let block = |offset: usize| -> Result<&[u8], String> {
            section(bytes, u32at(bytes, offset)?, u32at(bytes, offset + 4)?)
        };
        let aci = block(0x70).and_then(Aci::parse)
            .inspect_err(|error| log::debug!("NPDM ACI0 unavailable: {error}"))
            .ok();
        let acid = block(0x78).and_then(Acid::parse)
            .inspect_err(|error| log::debug!("NPDM ACID unavailable: {error}"))
            .ok();

        Ok(Self {
            is_64bit,
            address_space,
            main_thread_priority,
            main_thread_core,
            system_resource_size,
            main_stack_size,
            title_id,
            acid,
            aci,
        })
    }

    pub fn default_for_homebrew() -> Self {
        Self {
            is_64bit: true,
            address_space: AddressSpaceType::Is39Bit,
            main_thread_priority: 0x2C,
            main_thread_core: 0,
            system_resource_size: 0,
            main_stack_size: 0x100000,
            title_id: 0,
            acid: None,
            aci: None,
        }
    }

    pub fn acid_layout(&self) -> Option<AcidLayout> {
        self.acid.map(|acid| acid.layout)
    }

    pub fn is_switch2(&self) -> bool {
        self.acid_layout() == Some(AcidLayout::Switch2)
    }

    pub fn kernel_caps(&self) -> KernelCaps {
        self.aci.map(|aci| aci.kernel_caps).unwrap_or_default()
    }

    pub fn fs_permissions(&self) -> u64 {
        self.aci.map_or(u64::MAX, |aci| aci.fs_permissions)
    }
}

fn section(bytes: &[u8], offset: u32, size: u32) -> Result<&[u8], String> {
    let start = offset as usize;
    bytes
        .get(start..start.saturating_add(size as usize))
        .ok_or_else(|| format!("NPDM section {offset:#x}+{size:#x} exceeds {:#x} bytes", bytes.len()))
}

fn bit_range(low: u8, high: u8) -> u64 {
    if low > high || high >= 64 {
        return 0;
    }
    (u64::MAX >> (63 - high)) & (u64::MAX << low)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWITCH2_CAPS: [u32; 9] = [
        0x0500_73B7, 0x1FFF_FFCF, 0x267F_FFEF, 0x4000_0F8F, 0xA000_200F,
        0x0000_5FFF, 0x0048_BFFF, 0x0200_7FFF, 0x0002_FFFF,
    ];

    fn put(bytes: &mut [u8], offset: usize, value: &[u8]) {
        bytes[offset..offset + value.len()].copy_from_slice(value);
    }

    fn caps_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn npdm(layout: AcidLayout, program_id: u64, caps: &[u32]) -> Vec<u8> {
        let kc = caps_bytes(caps);
        let header = layout.signature_area_size();
        let acid_offset = 0x80;
        let fac_offset = header + 0x40;
        let acid_kc_offset = fac_offset + 0x2C + 4;
        let acid_size = acid_kc_offset + kc.len();
        let aci_offset = (acid_offset + acid_size + 0xF) & !0xF;
        let aci_size = 0x40 + 0x1C + kc.len();
        let mut bytes = vec![0u8; aci_offset + aci_size];
        put(&mut bytes, 0, &META_MAGIC.to_le_bytes());
        bytes[0x0C] = 0x37;
        bytes[0x0E] = 0x2C;
        put(&mut bytes, 0x14, &0x0400_0000u32.to_le_bytes());
        put(&mut bytes, 0x1C, &0x0010_0000u32.to_le_bytes());
        for (offset, value) in [(0x70, aci_offset), (0x74, aci_size), (0x78, acid_offset), (0x7C, acid_size)] {
            put(&mut bytes, offset, &(value as u32).to_le_bytes());
        }

        let acid = acid_offset;
        put(&mut bytes, acid + header, &ACID_MAGIC.to_le_bytes());
        put(&mut bytes, acid + header + 0x0C, &3u32.to_le_bytes());
        put(&mut bytes, acid + header + 0x10, &0x0400_0000_0001_0000u64.to_le_bytes());
        put(&mut bytes, acid + header + 0x18, &0x04FF_FFFF_FFFF_FFFFu64.to_le_bytes());
        for (offset, value) in [(0x20, fac_offset), (0x24, 0x2C), (0x30, acid_kc_offset), (0x34, kc.len())] {
            put(&mut bytes, acid + header + offset, &(value as u32).to_le_bytes());
        }
        bytes[acid + fac_offset] = 1;
        put(&mut bytes, acid + fac_offset + 4, &0x8000_0000_0000_0001u64.to_le_bytes());
        put(&mut bytes, acid + acid_kc_offset, &kc);

        let aci = aci_offset;
        put(&mut bytes, aci, &ACI0_MAGIC.to_le_bytes());
        put(&mut bytes, aci + 0x10, &program_id.to_le_bytes());
        for (offset, value) in [(0x20, 0x40), (0x24, 0x1C), (0x30, 0x5C), (0x34, kc.len())] {
            put(&mut bytes, aci + offset, &(value as u32).to_le_bytes());
        }
        bytes[aci + 0x40] = 1;
        put(&mut bytes, aci + 0x44, &0x4000_0000_0000_0000u64.to_le_bytes());
        put(&mut bytes, aci + 0x5C, &kc);
        bytes
    }

    #[test]
    fn switch2_acid_layout_is_detected_after_the_larger_signature_area() {
        let parsed = Npdm::parse(&npdm(AcidLayout::Switch2, 0x0400_C3F0_0006_E000, &SWITCH2_CAPS)).unwrap();
        assert!(parsed.is_switch2());
        assert_eq!(parsed.title_id, 0x0400_C3F0_0006_E000);
        assert_eq!(parsed.system_resource_size, 0x0400_0000);
        assert_eq!(parsed.address_space, AddressSpaceType::Is39Bit);
        let acid = parsed.acid.unwrap();
        assert_eq!(acid.flags, 3);
        assert_eq!(acid.program_id_min, 0x0400_0000_0001_0000);
        assert_eq!(acid.program_id_max, 0x04FF_FFFF_FFFF_FFFF);
        assert_eq!(acid.fs_permissions, 0x8000_0000_0000_0001);
        assert_eq!(parsed.fs_permissions(), 0x4000_0000_0000_0000);
        assert_eq!(acid.kernel_caps, parsed.kernel_caps());
    }

    #[test]
    fn switch1_acid_layout_still_parses() {
        let parsed = Npdm::parse(&npdm(AcidLayout::Switch1, 0x0100_1520_0002_2000, &[0x0300_43F7])).unwrap();
        assert!(!parsed.is_switch2());
        assert_eq!(parsed.acid_layout(), Some(AcidLayout::Switch1));
        assert_eq!(parsed.title_id, 0x0100_1520_0002_2000);
        assert_eq!(parsed.kernel_caps().thread_info.unwrap().core_mask(), 0xF);
        assert_eq!(parsed.fs_permissions(), 0x4000_0000_0000_0000);
    }

    #[test]
    fn switch2_kernel_capabilities_decode() {
        let caps = KernelCaps::parse(&SWITCH2_CAPS);
        let thread = caps.thread_info.unwrap();
        assert_eq!((thread.highest_priority, thread.lowest_priority), (28, 59));
        assert_eq!((thread.min_core, thread.max_core), (0, 5));
        assert_eq!(thread.core_mask(), 0x3F);
        assert_eq!(thread.priority_mask(), 0x0FFF_FFFF_F000_0000);
        let svcs = caps.svc_ids().collect::<Vec<_>>();
        assert_eq!(svcs.first(), Some(&0x01));
        assert!(caps.allows_svc(0x2D) && caps.allows_svc(0x36) && caps.allows_svc(0x80));
        assert!(!caps.allows_svc(0x2A) && !caps.allows_svc(0x37) && !caps.allows_svc(0x7F));
        assert_eq!(caps.kernel_version, Some(KernelVersion { major: 9, minor: 1 }));
        assert_eq!(caps.handle_table_size, Some(512));
        assert_eq!(caps.program_type, Some(1));
        assert_eq!(caps.debug_flags, Some(1));
    }

    #[test]
    fn address_space_42_bit_is_recognised() {
        let mut bytes = npdm(AcidLayout::Switch2, 1, &SWITCH2_CAPS);
        bytes[0x0C] = 1 | (4 << 1);
        assert_eq!(Npdm::parse(&bytes).unwrap().address_space, AddressSpaceType::Is42Bit);
    }

    #[test]
    fn map_range_pairs_and_padding_are_skipped() {
        let caps = KernelCaps::parse(&[0x0000_003F, 0x1FFF_FFCF, 0xFFFF_FFFF, 0x0300_43F7]);
        assert_eq!(caps.thread_info.unwrap().core_mask(), 0xF);
        assert_eq!(caps.svc_mask, [0; 3]);
    }

    #[test]
    fn missing_access_control_blocks_fall_back_to_the_meta_header() {
        let mut bytes = npdm(AcidLayout::Switch2, 0x0400_C3F0_0006_E000, &SWITCH2_CAPS);
        bytes[0x80 + 0x300] = 0;
        let parsed = Npdm::parse(&bytes).unwrap();
        assert!(parsed.acid.is_none());
        assert!(!parsed.is_switch2());
        assert_eq!(parsed.title_id, 0x0400_C3F0_0006_E000);
        assert!(parsed.aci.is_some());
    }
}

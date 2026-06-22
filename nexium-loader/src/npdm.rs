use crate::bin_read::{u32at, u64at, u8at};

pub const META_MAGIC: u32 = 0x4154454D;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddressSpaceType {
    Is32Bit,
    Is36Bit,
    Is32BitNoMap,
    Is39Bit,
}

impl AddressSpaceType {
    fn from_bits(v: u8) -> Self {
        match v & 0x7 {
            0 => AddressSpaceType::Is32Bit,
            1 => AddressSpaceType::Is36Bit,
            2 => AddressSpaceType::Is32BitNoMap,
            _ => AddressSpaceType::Is39Bit,
        }
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

        Ok(Self {
            is_64bit,
            address_space,
            main_thread_priority,
            main_thread_core,
            system_resource_size,
            main_stack_size,
            title_id,
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
        }
    }
}

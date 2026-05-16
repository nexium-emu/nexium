use byteorder::{LittleEndian, WriteBytesExt};

#[derive(Debug, Clone, Copy)]
pub enum ConfigEntry {
    MainThreadStackSize(u32),
    ProcessCategory(u32),
    Reserved(u32),
}

impl ConfigEntry {
    pub fn key(&self) -> u32 {
        match self {
            ConfigEntry::MainThreadStackSize(_) => 1,
            ConfigEntry::ProcessCategory(_) => 6,
            ConfigEntry::Reserved(_) => 0xFFFF,
        }
    }

    pub fn value(&self) -> u32 {
        match self {
            ConfigEntry::MainThreadStackSize(v) => *v,
            ConfigEntry::ProcessCategory(v) => *v,
            ConfigEntry::Reserved(v) => *v,
        }
    }
}

pub struct EnvBlock {
    main_thread_stack_size: u32,
    process_category: u32,
}

impl EnvBlock {
    pub fn new() -> Self {
        Self {
            main_thread_stack_size: 1024 * 1024,
            process_category: 0,
        }
    }

    pub fn with_stack_size(mut self, size: u32) -> Self {
        self.main_thread_stack_size = size;
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        let entries = vec![
            ConfigEntry::MainThreadStackSize(self.main_thread_stack_size),
            ConfigEntry::ProcessCategory(self.process_category),
        ];

        for entry in &entries {
            buf.write_u32::<LittleEndian>(entry.key()).unwrap();
            buf.write_u32::<LittleEndian>(entry.value()).unwrap();
        }

        let terminator_key = 0u32;
        let terminator_val = 0u32;
        buf.write_u32::<LittleEndian>(terminator_key).unwrap();
        buf.write_u32::<LittleEndian>(terminator_val).unwrap();

        log::debug!("Built env block: {} bytes", buf.len());
        buf
    }
}

impl Default for EnvBlock {
    fn default() -> Self {
        Self::new()
    }
}

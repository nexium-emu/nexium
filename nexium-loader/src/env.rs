use nexium_memory::AddressSpace;

#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EntryType {
    EndOfList = 0,
    MainThreadHandle = 1,
    NextLoadPath = 2,
    OverrideHeap = 3,
    OverrideService = 4,
    Argv = 5,
    SyscallAvailableHint = 6,
    AppletType = 7,
    AppletWorkaround = 8,
    ProcessHandle = 10,
    RandomSeed = 14,
    HosVersion = 16,
    SyscallAvailableHint2 = 17,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ConfigEntry {
    pub key: u32,
    pub flags: u32,
    pub value: [u64; 2],
}

const _: () = assert!(std::mem::size_of::<ConfigEntry>() == 24);

impl ConfigEntry {
    pub fn new(key: EntryType, flags: u32, v0: u64, v1: u64) -> Self {
        Self {
            key: key as u32,
            flags,
            value: [v0, v1],
        }
    }

    pub fn as_bytes(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&self.key.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.flags.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.value[0].to_le_bytes());
        bytes[16..24].copy_from_slice(&self.value[1].to_le_bytes());
        bytes
    }
}

pub struct EnvBlockBuilder {
    main_thread_handle: u32,
    process_handle: u32,
    heap_base: u64,
    heap_size: u64,
    applet_type: u64,
    hos_version: u64,
    syscall_hint: (u64, u64),
    syscall_hint2: (u64, u64),
    argv_string: Option<String>,
    next_load_path: Option<String>,
}

impl Default for EnvBlockBuilder {
    fn default() -> Self {
        Self {
            main_thread_handle: 0,
            process_handle: 0,
            heap_base: 0,
            heap_size: 0,
            applet_type: 0,
            hos_version: 0x000F_0000,
            syscall_hint: (u64::MAX, u64::MAX),
            syscall_hint2: (u64::MAX, 0),
            argv_string: None,
            next_load_path: None,
        }
    }
}

impl EnvBlockBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_handles(mut self, main_thread_handle: u32, process_handle: u32) -> Self {
        self.main_thread_handle = main_thread_handle;
        self.process_handle = process_handle;
        self
    }

    pub fn with_heap(mut self, base: u64, size: u64) -> Self {
        self.heap_base = base;
        self.heap_size = size;
        self
    }

    pub fn with_argv(mut self, argv: impl Into<String>) -> Self {
        self.argv_string = Some(argv.into());
        self
    }

    pub fn with_next_load_path(mut self, path: impl Into<String>) -> Self {
        self.next_load_path = Some(path.into());
        self
    }

    pub fn build_into(&self, mem: &AddressSpace, va: u64) -> Result<(), String> {
        const STRINGS_REGION_OFFSET: u64 = 0x800;
        const ARGV_STRING_OFFSET: u64 = STRINGS_REGION_OFFSET + 0x000;
        const NEXTLOAD_PATH_OFFSET: u64 = STRINGS_REGION_OFFSET + 0x200;
        const NEXTLOAD_ARGV_OFFSET: u64 = STRINGS_REGION_OFFSET + 0x400;

        let argv_va = va + ARGV_STRING_OFFSET;
        let nextload_path_va = va + NEXTLOAD_PATH_OFFSET;
        let nextload_argv_va = va + NEXTLOAD_ARGV_OFFSET;

        if let Some(s) = &self.argv_string {
            let mut bytes = s.as_bytes().to_vec();
            bytes.push(0);
            mem.write(argv_va, &bytes)
                .map_err(|e| format!("Failed to write argv string: {:?}", e))?;
        }
        if let Some(s) = &self.next_load_path {
            let mut bytes = s.as_bytes().to_vec();
            bytes.push(0);
            mem.write(nextload_path_va, &bytes)
                .map_err(|e| format!("Failed to write next_load_path: {:?}", e))?;
            mem.write(nextload_argv_va, &[0u8; 1])
                .map_err(|e| format!("Failed to write next_load_argv: {:?}", e))?;
        }

        let mut entries: Vec<ConfigEntry> = vec![
            ConfigEntry::new(
                EntryType::MainThreadHandle,
                0,
                self.main_thread_handle as u64,
                0,
            ),
            ConfigEntry::new(EntryType::ProcessHandle, 0, self.process_handle as u64, 0),
            ConfigEntry::new(EntryType::AppletType, 0, self.applet_type, 0),
            ConfigEntry::new(EntryType::OverrideHeap, 1, self.heap_base, self.heap_size),
            ConfigEntry::new(
                EntryType::SyscallAvailableHint,
                0,
                self.syscall_hint.0,
                self.syscall_hint.1,
            ),
            ConfigEntry::new(
                EntryType::SyscallAvailableHint2,
                0,
                self.syscall_hint2.0,
                self.syscall_hint2.1,
            ),
            ConfigEntry::new(
                EntryType::RandomSeed,
                0,
                0xDEAD_BEEF_CAFE_BABE,
                0x1234_5678_9ABC_DEF0,
            ),
            ConfigEntry::new(
                EntryType::HosVersion,
                0,
                self.hos_version,
                0x4154_4D4F_5350_4852,
            ),
        ];

        if self.argv_string.is_some() {
            entries.push(ConfigEntry::new(EntryType::Argv, 0, 0, argv_va));
        }
        if self.next_load_path.is_some() {
            entries.push(ConfigEntry::new(
                EntryType::NextLoadPath,
                0,
                nextload_path_va,
                nextload_argv_va,
            ));
        }

        entries.push(ConfigEntry::new(EntryType::EndOfList, 0, 0, 0));

        let mut offset = 0u64;
        for entry in entries {
            mem.write(va + offset, &entry.as_bytes())
                .map_err(|e| format!("Failed to write config entry: {:?}", e))?;
            offset += 24;
        }

        log::info!(
            "Built env block at {:#x}: {} entries, argv={:?} next_load_path={:?}",
            va,
            offset / 24,
            self.argv_string,
            self.next_load_path
        );
        Ok(())
    }
}

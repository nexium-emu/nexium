use std::collections::HashMap;

use super::super::GpuMappings;

pub const METHOD_PCAS_A: u32 = 0xAD;
pub const METHOD_PCAS_B: u32 = 0xAE;
pub const METHOD_SIGNALING_PCAS_B: u32 = 0xAF;
pub const METHOD_SIGNALING_PCAS2_B: u32 = 0xB0;
pub const METHOD_INLINE_CONTROL: u32 = 0xA7;
pub const METHOD_INLINE_ADDRESS_HIGH: u32 = 0xC6;
pub const METHOD_INLINE_ADDRESS_LOW: u32 = 0xC7;
pub const METHOD_INLINE_FIRST: u32 = 0xC8;
pub const METHOD_INLINE_LAUNCH: u32 = METHOD_INLINE_FIRST + QMD_WORDS as u32 - 1;
const INLINE_SCHEDULE: u32 = 1 << 24;
const INLINE_HELD: u32 = 1 << 12;

pub const QMD_WORDS: usize = 64;
const VERSION_WORD: usize = 18;
const VERSION: u32 = 0x30;
const DEPENDENCY_WORD: usize = 4;
const DEPENDENT_ADDRESS_WORD: usize = 15;
const DEPENDENT_WORD: usize = 16;
const RELEASE_SLOTS: [usize; 3] = [24, 28, 52];
const GRID_WORDS: [usize; 3] = [12, 13, 14];
const RELEASE_ENABLE: u32 = 1 << 23;
const RELEASE_MAX: u32 = 1 << 19;
const RELEASE_ADD: u32 = 1 << 26;
const RELEASE_UNSUPPORTED: u32 = (1 << 24) | (1 << 25) | (1 << 27) | (1 << 28) | (1 << 29);
const RELEASE_KIND_MASK: u32 = 0xC000_0000;
const RELEASE_KIND: u32 = 0x4000_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Schedule,
    Reload,
    Decrement(u32),
    Increment(u32),
}

impl Action {
    fn from_pcas2(value: u32) -> Option<Self> {
        let count = ((value >> 14) & 0x3FF) + 1;
        match value & 0xF {
            2 => Some(Self::Schedule),
            3 => Some(Self::Reload),
            7 => Some(Self::Decrement(count)),
            12 => Some(Self::Increment(count)),
            _ => None,
        }
    }

    fn from_dependent(value: u32) -> Option<Self> {
        match (value >> 1) & 0x7 {
            1 => Some(Self::Schedule),
            3 => Some(Self::Reload),
            4 => Some(Self::Decrement(1)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    loaded: bool,
    scheduled: bool,
    remaining: i64,
}

#[derive(Default)]
pub struct Qmd3Scheduler {
    address: u64,
    table: HashMap<u64, Entry>,
    pub completed: u64,
    pub releases: u64,
    pub launches: Vec<[u32; QMD_WORDS]>,
}

pub struct Memory<'a> {
    pub mappings: &'a GpuMappings,
    pub read: &'a dyn Fn(u64, &mut [u8]) -> bool,
    pub write: &'a dyn Fn(u64, &[u8]) -> bool,
}

impl Memory<'_> {
    fn read_words(&self, gpu_va: u64, words: &mut [u32]) -> bool {
        let Some(cpu) = self.mappings.cpu_address_for(gpu_va) else {
            return false;
        };
        let mut bytes = vec![0u8; words.len() * 4];
        if !(self.read)(cpu, &mut bytes) {
            return false;
        }
        for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
            *word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        true
    }

    fn read_word(&self, gpu_va: u64) -> Option<u32> {
        let mut word = [0u32];
        self.read_words(gpu_va, &mut word).then_some(word[0])
    }

    fn write_word(&self, gpu_va: u64, value: u32) -> bool {
        self.mappings
            .cpu_address_for(gpu_va)
            .is_some_and(|cpu| (self.write)(cpu, &value.to_le_bytes()))
    }
}

impl Qmd3Scheduler {
    pub fn handles(method: u32) -> bool {
        matches!(
            method,
            METHOD_PCAS_A | METHOD_PCAS_B | METHOD_SIGNALING_PCAS_B | METHOD_SIGNALING_PCAS2_B
        )
    }

    pub fn method(&mut self, method: u32, value: u32, memory: &Memory) {
        match method {
            METHOD_PCAS_A => self.address = u64::from(value) << 8,
            METHOD_SIGNALING_PCAS_B => match value & 0xF {
                2 => self.apply(self.address, Action::Schedule, memory),
                3 => self.apply(self.address, Action::Reload, memory),
                _ => {}
            },
            METHOD_SIGNALING_PCAS2_B => {
                if let Some(action) = Action::from_pcas2(value) {
                    self.apply(self.address, action, memory);
                }
            }
            _ => {}
        }
    }

    pub fn launch_inline(&mut self, address: u64, qmd: &[u32], control: u32, memory: &Memory) {
        if std::env::var_os("NEXIUM_QMD_TRACE").is_some() {
            log::warn!("[qmd-trace] inline {address:#x} control={control:#x} w4={:#x} words={}", qmd[DEPENDENCY_WORD], qmd.len());
        }
        for (index, word) in qmd.iter().enumerate() {
            memory.write_word(address + index as u64 * 4, *word);
        }
        if control & INLINE_SCHEDULE != 0 || qmd[DEPENDENCY_WORD] & INLINE_HELD == 0 {
            self.apply(address, Action::Reload, memory);
        }
    }

    fn load(entry: &mut Entry, address: u64, memory: &Memory) {
        let dependencies = memory
            .read_word(address + (DEPENDENCY_WORD as u64) * 4)
            .map_or(0, |word| (word >> 14) & 0xFFFF);
        let pending = if entry.loaded { 0 } else { entry.remaining };
        entry.remaining = i64::from(dependencies) + pending;
        entry.loaded = true;
    }

    fn apply(&mut self, address: u64, action: Action, memory: &Memory) {
        let mut queue = vec![(address, action)];
        while let Some((address, action)) = queue.pop() {
            let entry = self.table.entry(address).or_default();
            match action {
                Action::Schedule => {
                    if !entry.loaded {
                        Self::load(entry, address, memory);
                    }
                    entry.scheduled = true;
                }
                Action::Reload => {
                    entry.loaded = false;
                    entry.remaining = entry.remaining.min(0);
                    Self::load(entry, address, memory);
                    entry.scheduled = true;
                }
                Action::Decrement(count) => entry.remaining -= i64::from(count),
                Action::Increment(count) => entry.remaining += i64::from(count),
            }
            let ready = entry.loaded && entry.scheduled && entry.remaining <= 0;
            if std::env::var_os("NEXIUM_QMD_TRACE").is_some() {
                let mut qmd = [0u32; QMD_WORDS];
                let words = if matches!(action, Action::Reload | Action::Schedule) && memory.read_words(address, &mut qmd) {
                    format!(
                        " w4={:#x} w15={:#x} w16={:#x} r24={:x?} r28={:x?} r52={:x?}",
                        qmd[4], qmd[15], qmd[16], &qmd[24..27], &qmd[28..31], &qmd[52..55]
                    )
                } else {
                    String::new()
                };
                let entry = self.table.get(&address).copied().unwrap_or_default();
                log::warn!(
                    "[qmd-trace] {address:#x} {action:?} loaded={} scheduled={} remaining={} ready={ready}{words}",
                    entry.loaded,
                    entry.scheduled,
                    entry.remaining
                );
            }
            if ready {
                self.table.remove(&address);
                if let Some(dependent) = self.complete(address, memory) {
                    queue.push(dependent);
                }
            }
        }
    }

    fn complete(&mut self, address: u64, memory: &Memory) -> Option<(u64, Action)> {
        let mut qmd = [0u32; QMD_WORDS];
        if !memory.read_words(address, &mut qmd) {
            log::warn!("[qmd3] unreadable QMD at {address:#x}");
            return None;
        }
        self.completed += 1;
        if qmd[VERSION_WORD] & 0xFF != VERSION && self.completed <= 4 {
            log::warn!("[qmd3] QMD at {address:#x} has version {:#x}", qmd[VERSION_WORD] & 0xFF);
        }
        if self.completed <= 16 {
            log::debug!("[qmd3] complete {address:#x} words={qmd:08x?}");
        }
        if qmd[GRID_WORDS[0]] != 0 && qmd[GRID_WORDS[1]] & 0xFFFF != 0 && qmd[GRID_WORDS[2]] & 0xFFFF != 0 {
            self.launches.push(qmd);
        }
        for slot in RELEASE_SLOTS {
            self.release(&qmd[slot..slot + 3], memory);
        }
        let dependent = qmd[DEPENDENT_WORD];
        if dependent & 1 == 0 {
            return None;
        }
        let target = u64::from(qmd[DEPENDENT_ADDRESS_WORD]) << 8;
        Action::from_dependent(dependent).map(|action| (target, action))
    }

    fn release(&mut self, slot: &[u32], memory: &Memory) {
        let (low, flags, payload) = (slot[0], slot[1], slot[2]);
        if flags & RELEASE_ENABLE == 0 {
            return;
        }
        if flags & RELEASE_UNSUPPORTED != 0 || flags & RELEASE_KIND_MASK != RELEASE_KIND {
            log::warn!("[qmd3] unsupported release flags={flags:#x}");
            return;
        }
        let address = (u64::from(flags & 0xFF) << 32) | u64::from(low);
        let value = if flags & RELEASE_ADD != 0 {
            memory.read_word(address).unwrap_or(0).wrapping_add(payload)
        } else if flags & RELEASE_MAX != 0 {
            memory.read_word(address).unwrap_or(0).max(payload)
        } else {
            payload
        };
        if crate::gpu::pusher::semaphore_watch().is_some_and(|range| range.contains(&address)) {
            log::warn!("[sem-watch] qmd release gpu_va={address:#x} flags={flags:#x} payload={payload:#x} value={value:#x}");
        }
        if memory.write_word(address, value) {
            self.releases += 1;
        } else {
            log::warn!("[qmd3] release to unmapped {address:#x}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const BASE: u64 = 0x1000_0000;

    fn memory_with(mappings: &mut GpuMappings) {
        mappings.add(BASE, 0x10000, BASE, 0);
    }

    fn put(ram: &RefCell<Vec<u8>>, address: u64, value: u32) {
        let offset = (address - BASE) as usize;
        ram.borrow_mut()[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn get(ram: &RefCell<Vec<u8>>, address: u64) -> u32 {
        let offset = (address - BASE) as usize;
        u32::from_le_bytes(ram.borrow()[offset..offset + 4].try_into().unwrap())
    }

    fn qmd(ram: &RefCell<Vec<u8>>, at: u64, dependencies: u32, release: Option<(u64, u32, u32)>) {
        put(ram, at + 18 * 4, VERSION);
        put(ram, at + 4 * 4, dependencies << 14);
        if let Some((target, flags, payload)) = release {
            put(ram, at + 24 * 4, target as u32);
            put(ram, at + 25 * 4, RELEASE_KIND | RELEASE_ENABLE | flags | (target >> 32) as u32);
            put(ram, at + 26 * 4, payload);
        }
    }

    fn run(ram: &RefCell<Vec<u8>>, steps: &[(u32, u32)]) -> Qmd3Scheduler {
        let mut mappings = GpuMappings::new();
        memory_with(&mut mappings);
        let read = |cpu: u64, out: &mut [u8]| {
            let offset = (cpu - BASE) as usize;
            out.copy_from_slice(&ram.borrow()[offset..offset + out.len()]);
            true
        };
        let write = |cpu: u64, data: &[u8]| {
            let offset = (cpu - BASE) as usize;
            ram.borrow_mut()[offset..offset + data.len()].copy_from_slice(data);
            true
        };
        let memory = Memory { mappings: &mappings, read: &read, write: &write };
        let mut scheduler = Qmd3Scheduler::default();
        for &(method, value) in steps {
            scheduler.method(method, value, &memory);
        }
        scheduler
    }

    #[test]
    fn scheduled_qmd_without_dependencies_fires_its_release() {
        let ram = RefCell::new(vec![0u8; 0x10000]);
        let target = BASE + 0x8000;
        qmd(&ram, BASE + 0x100, 0, Some((target, 0, 7)));
        let scheduler = run(&ram, &[(METHOD_PCAS_A, ((BASE + 0x100) >> 8) as u32), (METHOD_SIGNALING_PCAS_B, 2)]);
        assert_eq!(get(&ram, target), 7);
        assert_eq!(scheduler.completed, 1);
    }

    #[test]
    fn dependencies_hold_the_qmd_until_decremented_even_before_loading() {
        let ram = RefCell::new(vec![0u8; 0x10000]);
        let target = BASE + 0x8000;
        let at = (((BASE + 0x200) >> 8) as u32, BASE + 0x200);
        qmd(&ram, at.1, 2, Some((target, 0, 9)));
        run(&ram, &[(METHOD_PCAS_A, at.0), (METHOD_SIGNALING_PCAS2_B, 7), (METHOD_SIGNALING_PCAS2_B, 2)]);
        assert_eq!(get(&ram, target), 0);
        let ram2 = RefCell::new(ram.borrow().clone());
        run(&ram2, &[(METHOD_PCAS_A, at.0), (METHOD_SIGNALING_PCAS2_B, 7 | (1 << 14)), (METHOD_SIGNALING_PCAS2_B, 2)]);
        assert_eq!(get(&ram2, target), 9);
    }

    #[test]
    fn add_and_max_releases_read_the_old_value() {
        let ram = RefCell::new(vec![0u8; 0x10000]);
        let target = BASE + 0x8000;
        put(&ram, target, 10);
        qmd(&ram, BASE + 0x300, 0, Some((target, RELEASE_ADD, 5)));
        run(&ram, &[(METHOD_PCAS_A, ((BASE + 0x300) >> 8) as u32), (METHOD_SIGNALING_PCAS_B, 3)]);
        assert_eq!(get(&ram, target), 15);
        qmd(&ram, BASE + 0x400, 0, Some((target, RELEASE_MAX, 12)));
        run(&ram, &[(METHOD_PCAS_A, ((BASE + 0x400) >> 8) as u32), (METHOD_SIGNALING_PCAS_B, 2)]);
        assert_eq!(get(&ram, target), 15);
    }

    #[test]
    fn completion_runs_the_dependent_qmd() {
        let ram = RefCell::new(vec![0u8; 0x10000]);
        let target = BASE + 0x8000;
        let (first, second) = (BASE + 0x500, BASE + 0x600);
        qmd(&ram, first, 0, None);
        put(&ram, first + 15 * 4, (second >> 8) as u32);
        put(&ram, first + 16 * 4, 1 | (4 << 1));
        qmd(&ram, second, 1, Some((target, 0, 3)));
        run(
            &ram,
            &[
                (METHOD_PCAS_A, (second >> 8) as u32),
                (METHOD_SIGNALING_PCAS_B, 2),
                (METHOD_PCAS_A, (first >> 8) as u32),
                (METHOD_SIGNALING_PCAS_B, 2),
            ],
        );
        assert_eq!(get(&ram, target), 3);
    }
}

use super::engines::Maxwell3D;
use super::GpuMappings;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CommandListHeader {
    pub address_lo: u32,
    pub address_hi_and_count: u32,
}

impl CommandListHeader {
    pub fn address(&self) -> u64 {
        ((self.address_hi_and_count as u64 & 0xFF) << 32) | self.address_lo as u64
    }

    pub fn entry_count(&self) -> u32 {
        (self.address_hi_and_count >> 10) & 0xFFFFF
    }

    pub fn no_prefetch(&self) -> bool {
        (self.address_hi_and_count & 0x8000_0000) != 0
    }

    pub fn not_main(&self) -> bool {
        (self.address_hi_and_count & 0x4000_0000) != 0
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mode {
    IncreasingOld,
    Increasing,
    NonIncreasingOld,
    NonIncreasing,
    Inline,
    IncreaseOnce,
}

impl Mode {
    fn from_bits(v: u32) -> Option<Mode> {
        Some(match v {
            0 => Mode::IncreasingOld,
            1 => Mode::Increasing,
            2 => Mode::NonIncreasingOld,
            3 => Mode::NonIncreasing,
            4 => Mode::Inline,
            5 => Mode::IncreaseOnce,
            _ => return None,
        })
    }
}

const METHOD_BIND_OBJECT: u32 = 0x00;
const METHOD_SEMAPHORE_ADDR_HIGH: u32 = 0x04;
const METHOD_SEMAPHORE_ADDR_LOW: u32 = 0x05;
const METHOD_SEMAPHORE_PAYLOAD: u32 = 0x06;
const METHOD_SEMAPHORE_OPERATION: u32 = 0x07;
const METHOD_SYNCPOINT_PAYLOAD: u32 = 0x1C;
const METHOD_SYNCPOINT_OPERATION: u32 = 0x1D;
const NON_PULLER_METHODS: u32 = 0x40;

#[derive(Default)]
struct DmaState {
    method: u32,
    subchannel: u32,
    method_count: u32,
    non_incrementing: bool,
    increment_once: bool,
}

#[derive(Default)]
struct PullerState {
    semaphore_addr_high: u32,
    semaphore_addr_low: u32,
    semaphore_payload: u32,
    syncpoint_payload: u32,
}

pub struct Pusher {
    pub syncpt_value: u32,
    bound_classes: [u32; 8],
    state: DmaState,
    puller: PullerState,
}

impl Pusher {
    pub fn new() -> Self {
        Self {
            syncpt_value: 0,
            bound_classes: [0; 8],
            state: DmaState::default(),
            puller: PullerState::default(),
        }
    }

    pub fn process_gpfifo(
        &mut self,
        address: u64,
        num_entries: u32,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        mem_read: &impl Fn(u64, &mut [u8]) -> bool,
    ) {
        let cpu_addr = mappings.cpu_address_for(address).unwrap_or(address);

        let bytes_needed = (num_entries as usize) * 8;
        let mut buf = vec![0u8; bytes_needed];
        if !mem_read(cpu_addr, &mut buf) {
            log::debug!("pusher: failed to read GPFIFO entries at cpu {:#x} (input addr {:#x})",
                cpu_addr, address);
            return;
        }

        for i in 0..num_entries as usize {
            let off = i * 8;
            let entry = CommandListHeader {
                address_lo: u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]),
                address_hi_and_count: u32::from_le_bytes([buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]]),
            };
            self.process_entry(&entry, mappings, maxwell, mem_read);
        }
    }

    pub fn process_entry(
        &mut self,
        entry: &CommandListHeader,
        mappings: &GpuMappings,
        maxwell: &mut Maxwell3D,
        mem_read: &impl Fn(u64, &mut [u8]) -> bool,
    ) {
        let address = entry.address();
        let word_count = entry.entry_count();
        if word_count == 0 || word_count > 0x100000 {
            return;
        }

        let cpu_addr = mappings.cpu_address_for(address).unwrap_or(address);

        let bytes_needed = (word_count as usize) * 4;
        let mut buf = vec![0u8; bytes_needed];
        if !mem_read(cpu_addr, &mut buf) {
            log::debug!("pusher: failed to read cmd buffer at cpu {:#x}", cpu_addr);
            return;
        }

        let mut words: Vec<u32> = Vec::with_capacity(word_count as usize);
        for i in 0..word_count as usize {
            let off = i * 4;
            words.push(u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]));
        }

        self.process_commands(&words, maxwell);
    }

    fn process_commands(&mut self, commands: &[u32], maxwell: &mut Maxwell3D) {
        let mut i = 0;
        while i < commands.len() {
            let header = commands[i];

            if self.state.method_count > 0 {
                self.dispatch_method(header, maxwell);
                if !self.state.non_incrementing {
                    self.state.method = self.state.method.wrapping_add(1);
                }
                if self.state.increment_once {
                    self.state.non_incrementing = true;
                }
                self.state.method_count -= 1;
                i += 1;
                continue;
            }

            let method = header & 0x1FFF;
            let subchannel = (header >> 13) & 0x7;
            let arg_count = (header >> 16) & 0x1FFF;
            let mode_bits = (header >> 29) & 0x7;
            let Some(mode) = Mode::from_bits(mode_bits) else {
                log::trace!("pusher: unknown mode {} in header {:#010x}", mode_bits, header);
                i += 1;
                continue;
            };

            self.state.method = method;
            self.state.subchannel = subchannel;
            self.state.method_count = arg_count;

            match mode {
                Mode::Increasing | Mode::IncreasingOld => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = false;
                }
                Mode::NonIncreasing | Mode::NonIncreasingOld => {
                    self.state.non_incrementing = true;
                    self.state.increment_once = false;
                }
                Mode::IncreaseOnce => {
                    self.state.non_incrementing = false;
                    self.state.increment_once = true;
                }
                Mode::Inline => {
                    self.state.method_count = 0;
                    self.dispatch_method(arg_count, maxwell);
                }
            }
            i += 1;
        }
    }

    fn dispatch_method(&mut self, arg: u32, maxwell: &mut Maxwell3D) {
        let method = self.state.method;
        let subchannel = self.state.subchannel as usize;

        if method < NON_PULLER_METHODS {
            self.handle_puller_method(method, arg, subchannel);
            return;
        }

        let bound_class = self.bound_classes[subchannel & 7];
        if bound_class == 0xB197 {
            let is_last = self.state.method_count <= 1;
            maxwell.dispatch_method(method, arg, is_last);
        } else {
            log::trace!("pusher: subch={} class={:#x} method={:#x} arg={:#x} (not 3D, ignored)",
                subchannel, bound_class, method, arg);
        }
    }

    fn handle_puller_method(&mut self, method: u32, arg: u32, subchannel: usize) {
        match method {
            METHOD_BIND_OBJECT => {
                self.bound_classes[subchannel & 7] = arg & 0xFFFF;
                log::debug!("puller: BindObject subch={} class={:#x}", subchannel, arg & 0xFFFF);
            }
            METHOD_SEMAPHORE_ADDR_HIGH => self.puller.semaphore_addr_high = arg,
            METHOD_SEMAPHORE_ADDR_LOW => self.puller.semaphore_addr_low = arg,
            METHOD_SEMAPHORE_PAYLOAD => self.puller.semaphore_payload = arg,
            METHOD_SEMAPHORE_OPERATION => {
                log::trace!("puller: SemaphoreOp op={:#x} payload={}",
                    arg, self.puller.semaphore_payload);
            }
            METHOD_SYNCPOINT_PAYLOAD => self.puller.syncpoint_payload = arg,
            METHOD_SYNCPOINT_OPERATION => {
                let op = arg & 0xFF;
                if op == 1 {
                    self.syncpt_value = self.syncpt_value.wrapping_add(1);
                    log::trace!("puller: SyncpointIncrement → {}", self.syncpt_value);
                }
            }
            _ => {
                log::trace!("puller: method {:#x} arg={:#x}", method, arg);
            }
        }
    }
}

impl Default for Pusher {
    fn default() -> Self {
        Self::new()
    }
}

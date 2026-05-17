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

const SUBCH_3D: u32 = 0;
const SUBCH_COMPUTE: u32 = 1;
const SUBCH_INLINE2MEMORY: u32 = 2;
const SUBCH_2D: u32 = 3;
const SUBCH_DMA: u32 = 4;

pub struct Pusher {
    pub syncpt_value: u32,
    bound_classes: [u32; 8],
}

impl Pusher {
    pub fn new() -> Self {
        Self {
            syncpt_value: 0,
            bound_classes: [0; 8],
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
        log::info!("pusher: reading {} GPFIFO entries from cpu_addr={:#x}", num_entries, cpu_addr);

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
        log::info!("pusher: entry addr={:#x} word_count={}", address, word_count);
        if word_count == 0 || word_count > 0x100000 {
            return;
        }

        let cpu_addr = match mappings.cpu_address_for(address) {
            Some(a) => a,
            None => {
                log::debug!("pusher: no mapping for cmd buffer {:#x} - trying CPU direct", address);
                address
            }
        };

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

        self.process_cmd_stream(&words, maxwell);
    }

    fn process_cmd_stream(&mut self, words: &[u32], maxwell: &mut Maxwell3D) {
        let mut i = 0;
        while i < words.len() {
            let header = words[i];
            i += 1;

            let method_offset = header & 0x1FFF;
            let subchannel = (header >> 13) & 0x7;
            let arg_count = (header >> 16) & 0x1FFF;
            let secop = (header >> 29) & 0x7;

            match secop {
                0 => {
                    if arg_count == 0 {
                        if let Some(arg) = words.get(i) {
                            self.dispatch_method(subchannel, method_offset, *arg, maxwell);
                            i += 1;
                        }
                    } else {
                        for n in 0..arg_count as usize {
                            if let Some(arg) = words.get(i + n) {
                                self.dispatch_method(subchannel, method_offset, *arg, maxwell);
                            }
                        }
                        i += arg_count as usize;
                    }
                }
                1 => {
                    let end = i + arg_count as usize;
                    while i < end && i < words.len() {
                        let arg = words[i];
                        self.dispatch_method(subchannel, method_offset, arg, maxwell);
                        i += 1;
                    }
                }
                3 => {
                    let end = i + arg_count as usize;
                    let mut off = method_offset;
                    while i < end && i < words.len() {
                        let arg = words[i];
                        self.dispatch_method(subchannel, off, arg, maxwell);
                        off += 1;
                        i += 1;
                    }
                }
                4 => {
                    if arg_count <= 0x1FFF {
                        let inline_data = arg_count;
                        self.dispatch_method(subchannel, method_offset, inline_data, maxwell);
                    }
                }
                _ => {
                    i += arg_count as usize;
                }
            }
        }
    }

    fn dispatch_method(&mut self, subchannel: u32, method: u32, arg: u32, maxwell: &mut Maxwell3D) {
        if method == 0 {
            self.bound_classes[subchannel as usize & 7] = arg;
            log::info!("pusher: BIND subch={} class={:#x}", subchannel, arg);
            return;
        }

        let bound_class = self.bound_classes[subchannel as usize & 7];
        log::debug!("pusher: subch={} class={:#x} method={:#x} arg={:#x}",
            subchannel, bound_class, method, arg);

        if subchannel == SUBCH_3D as u32 || bound_class == 0xB197 {
            maxwell.write_register(method, arg);
        }

        if method == 0x44 || method == 0x45 {
            self.syncpt_value = self.syncpt_value.wrapping_add(1);
        }
    }
}

impl Default for Pusher {
    fn default() -> Self {
        Self::new()
    }
}

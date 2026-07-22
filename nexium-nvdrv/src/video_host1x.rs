pub const NVDEC_CLASS_ID: u32 = 0xf0;
pub const VIC_CLASS_ID: u32 = 0x5d;
pub const THI_SET_METHOD_0: u32 = 0x10;
pub const THI_SET_METHOD_1: u32 = 0x11;
pub const ENGINE_EXECUTE_METHOD: u32 = 0xc0;
pub const ENGINE_REGISTER_COUNT: usize = 0x200;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VideoEngine {
    Nvdec,
    Vic,
}

impl VideoEngine {
    pub const fn from_class_id(class_id: u32) -> Option<Self> {
        match class_id {
            NVDEC_CLASS_ID => Some(Self::Nvdec),
            VIC_CLASS_ID => Some(Self::Vic),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineRegisterWrite {
    pub stream_word_index: u64,
    pub class_id: u32,
    pub engine: VideoEngine,
    pub method: u32,
    pub argument: u32,
}

impl EngineRegisterWrite {
    pub const fn is_execute(self) -> bool {
        self.method == ENGINE_EXECUTE_METHOD
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PacketState {
    method_offset: u32,
    mask: u32,
    count: u32,
    incrementing: bool,
}

pub struct VideoHost1xParser {
    current_class: u32,
    selected_engine_method: u32,
    packet: PacketState,
    stream_word_index: u64,
    nvdec_registers: [u32; ENGINE_REGISTER_COUNT],
    vic_registers: [u32; ENGINE_REGISTER_COUNT],
}

impl Default for VideoHost1xParser {
    fn default() -> Self {
        Self::new(0)
    }
}

impl VideoHost1xParser {
    pub fn new(initial_class: u32) -> Self {
        Self {
            current_class: initial_class,
            selected_engine_method: 0,
            packet: PacketState::default(),
            stream_word_index: 0,
            nvdec_registers: [0; ENGINE_REGISTER_COUNT],
            vic_registers: [0; ENGINE_REGISTER_COUNT],
        }
    }

    pub fn current_class(&self) -> u32 {
        self.current_class
    }

    pub fn selected_engine_method(&self) -> u32 {
        self.selected_engine_method
    }

    pub fn stream_word_index(&self) -> u64 {
        self.stream_word_index
    }

    pub fn registers(&self, engine: VideoEngine) -> &[u32; ENGINE_REGISTER_COUNT] {
        match engine {
            VideoEngine::Nvdec => &self.nvdec_registers,
            VideoEngine::Vic => &self.vic_registers,
        }
    }

    pub fn register(&self, engine: VideoEngine, method: u32) -> Option<u32> {
        self.registers(engine)
            .get(usize::try_from(method).ok()?)
            .copied()
    }

    pub fn feed(&mut self, words: &[u32]) -> Vec<EngineRegisterWrite> {
        let mut writes = Vec::new();
        for &raw in words {
            let word_index = self.stream_word_index;
            self.stream_word_index = self.stream_word_index.wrapping_add(1);

            if self.packet.mask != 0 {
                let bit = self.packet.mask.trailing_zeros();
                self.packet.mask &= !(1u32 << bit);
                self.execute_host_method(
                    self.packet.method_offset.wrapping_add(bit),
                    raw,
                    word_index,
                    &mut writes,
                );
                continue;
            }

            if self.packet.count != 0 {
                self.packet.count -= 1;
                let method = self.packet.method_offset;
                if self.packet.incrementing {
                    self.packet.method_offset = self.packet.method_offset.wrapping_add(1);
                }
                self.execute_host_method(method, raw, word_index, &mut writes);
                continue;
            }

            let value = raw & 0xffff;
            let method_offset = (raw >> 16) & 0x0fff;
            match raw >> 28 {
                0 => {
                    self.packet.method_offset = method_offset;
                    self.packet.mask = value & 0x3f;
                    self.current_class = (value >> 6) & 0x03ff;
                }
                1 | 2 => {
                    self.packet.method_offset = method_offset;
                    self.packet.count = value;
                    self.packet.incrementing = raw >> 28 == 1;
                }
                3 => {
                    self.packet.method_offset = method_offset;
                    self.packet.mask = value;
                }
                4 => {
                    self.execute_host_method(
                        method_offset,
                        value & 0x0fff,
                        word_index,
                        &mut writes,
                    );
                }
                _ => {}
            }
        }
        writes
    }

    fn execute_host_method(
        &mut self,
        method: u32,
        argument: u32,
        stream_word_index: u64,
        writes: &mut Vec<EngineRegisterWrite>,
    ) {
        let Some(engine) = VideoEngine::from_class_id(self.current_class) else {
            return;
        };

        match method {
            THI_SET_METHOD_0 => self.selected_engine_method = argument,
            THI_SET_METHOD_1 => {
                let engine_method = self.selected_engine_method;
                if let Ok(index) = usize::try_from(engine_method) {
                    if index < ENGINE_REGISTER_COUNT {
                        match engine {
                            VideoEngine::Nvdec => self.nvdec_registers[index] = argument,
                            VideoEngine::Vic => self.vic_registers[index] = argument,
                        }
                    }
                }
                writes.push(EngineRegisterWrite {
                    stream_word_index,
                    class_id: self.current_class,
                    engine,
                    method: engine_method,
                    argument,
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(mode: u32, method: u32, value: u32) -> u32 {
        (mode << 28) | ((method & 0x0fff) << 16) | (value & 0xffff)
    }

    fn set_class(method: u32, class_id: u32, mask: u32) -> u32 {
        header(0, method, ((class_id & 0x03ff) << 6) | (mask & 0x3f))
    }

    #[test]
    fn set_class_mask_writes_nvdec_register() {
        let mut parser = VideoHost1xParser::default();
        let writes = parser.feed(&[
            set_class(THI_SET_METHOD_0, NVDEC_CLASS_ID, 0b11),
            0x102,
            0x1234_5678,
        ]);

        assert_eq!(parser.current_class(), NVDEC_CLASS_ID);
        assert_eq!(parser.selected_engine_method(), 0x102);
        assert_eq!(
            parser.register(VideoEngine::Nvdec, 0x102),
            Some(0x1234_5678)
        );
        assert_eq!(
            writes,
            vec![EngineRegisterWrite {
                stream_word_index: 2,
                class_id: NVDEC_CLASS_ID,
                engine: VideoEngine::Nvdec,
                method: 0x102,
                argument: 0x1234_5678,
            }]
        );
    }

    #[test]
    fn packet_and_thi_state_persist_across_command_buffers() {
        let mut parser = VideoHost1xParser::default();
        assert!(parser
            .feed(&[
                set_class(THI_SET_METHOD_0, NVDEC_CLASS_ID, 0b11),
                ENGINE_EXECUTE_METHOD,
            ])
            .is_empty());

        let writes = parser.feed(&[1]);
        assert_eq!(writes.len(), 1);
        assert!(writes[0].is_execute());
        assert_eq!(writes[0].stream_word_index, 2);

        let writes = parser.feed(&[header(4, THI_SET_METHOD_1, 7)]);
        assert_eq!(writes.len(), 1);
        assert!(writes[0].is_execute());
        assert_eq!(writes[0].argument, 7);
        assert_eq!(writes[0].stream_word_index, 3);
    }

    #[test]
    fn incrementing_nonincrementing_mask_and_immediate_are_decoded() {
        let mut parser = VideoHost1xParser::new(VIC_CLASS_ID);
        let writes = parser.feed(&[
            header(1, THI_SET_METHOD_0, 2),
            0x100,
            0xaaaa_0001,
            header(2, THI_SET_METHOD_1, 2),
            0xbbbb_0002,
            0xcccc_0003,
            header(3, THI_SET_METHOD_0, 0b11),
            0x101,
            0xdddd_0004,
            header(4, THI_SET_METHOD_0, ENGINE_EXECUTE_METHOD),
            header(4, THI_SET_METHOD_1, 5),
        ]);

        assert_eq!(writes.len(), 5);
        assert_eq!(writes[0].method, 0x100);
        assert_eq!(writes[0].argument, 0xaaaa_0001);
        assert_eq!(writes[1].method, 0x100);
        assert_eq!(writes[1].argument, 0xbbbb_0002);
        assert_eq!(writes[2].method, 0x100);
        assert_eq!(writes[2].argument, 0xcccc_0003);
        assert_eq!(writes[3].method, 0x101);
        assert_eq!(writes[3].argument, 0xdddd_0004);
        assert!(writes[4].is_execute());
        assert_eq!(parser.register(VideoEngine::Vic, 0x100), Some(0xcccc_0003));
        assert_eq!(parser.register(VideoEngine::Vic, 0x101), Some(0xdddd_0004));
        assert_eq!(parser.register(VideoEngine::Nvdec, 0x100), Some(0));
    }

    #[test]
    fn engine_register_banks_are_independent() {
        let mut parser = VideoHost1xParser::default();
        parser.feed(&[
            set_class(THI_SET_METHOD_0, NVDEC_CLASS_ID, 0b11),
            0x10c,
            0x1111,
            set_class(THI_SET_METHOD_0, VIC_CLASS_ID, 0b11),
            0x10c,
            0x2222,
        ]);

        assert_eq!(parser.register(VideoEngine::Nvdec, 0x10c), Some(0x1111));
        assert_eq!(parser.register(VideoEngine::Vic, 0x10c), Some(0x2222));
    }

    #[test]
    fn out_of_bank_register_write_is_emitted_without_panicking() {
        let mut parser = VideoHost1xParser::new(NVDEC_CLASS_ID);
        let writes = parser.feed(&[
            header(4, THI_SET_METHOD_0, 0x2ff),
            header(4, THI_SET_METHOD_1, 0xabc),
        ]);

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].method, 0x2ff);
        assert_eq!(parser.register(VideoEngine::Nvdec, 0x2ff), None);
    }
}

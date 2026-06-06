use nexium_common::result::SUCCESS;

pub struct AudioService;

impl AudioService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!("audio cmd: {} (legacy dispatch)", cmd_id);
        match cmd_id {
            0 => self.cmd_open_audio_out(),
            1 => self.cmd_open_audio_renderer(),
            2 => self.cmd_query_audio_device_list(),
            _ => {
                log::debug!("unknown audio command: {}", cmd_id);
                0
            }
        }
    }

    fn cmd_open_audio_out(&self) -> u32 {
        log::debug!("Audio::OpenAudioOut");
        SUCCESS
    }

    fn cmd_open_audio_renderer(&self) -> u32 {
        log::debug!("Audio::OpenAudioRenderer");
        SUCCESS
    }

    fn cmd_query_audio_device_list(&self) -> u32 {
        log::debug!("Audio::QueryAudioDeviceList");
        SUCCESS
    }
}

impl Default for AudioService {
    fn default() -> Self {
        Self::new()
    }
}

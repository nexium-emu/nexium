pub mod sm;
pub mod hid;
pub mod time;
pub mod set;
pub mod am;
pub mod vi;
pub mod audio;
pub mod fs;
pub mod nifm;
pub mod nvnflinger;

pub struct FrameOut {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct Services {
    pub sm: sm::ServiceManager,
    pub hid: hid::HidService,
    pub time: time::TimeService,
    pub set: set::SettingsService,
    pub am: am::AppletService,
    pub vi: vi::DisplayService,
    pub audio: audio::AudioService,
    pub fs: fs::FileSystemService,
    pub nifm: nifm::NetworkService,
    pub nvnflinger: nvnflinger::BufferQueueService,
}

impl Services {
    pub fn new() -> Self {
        Self {
            sm: sm::ServiceManager::new(),
            hid: hid::HidService::new(),
            time: time::TimeService::new(),
            set: set::SettingsService::new(),
            am: am::AppletService::new(),
            vi: vi::DisplayService::new(),
            audio: audio::AudioService::new(),
            fs: fs::FileSystemService::new(),
            nifm: nifm::NetworkService::new(),
            nvnflinger: nvnflinger::BufferQueueService::new(),
        }
    }

    pub fn dispatch_service(&self, port_name: &str, cmd_id: u32) -> u32 {
        log::trace!("dispatch_service: port={} cmd_id={}", port_name, cmd_id);
        match port_name {
            "sm:" => self.sm.dispatch(cmd_id),
            "hid" => self.hid.dispatch(cmd_id),
            "time:s" | "time:a" | "time:r" => self.time.dispatch(cmd_id),
            "set" | "set:sys" => self.set.dispatch(cmd_id),
            "am" | "appletOE" => self.am.dispatch(cmd_id),
            "vi:m" | "vi:s" => self.vi.dispatch(cmd_id),
            "audio" | "audout:u" => self.audio.dispatch(cmd_id),
            "fsp-srv" => self.fs.dispatch(cmd_id),
            "nifm:u" | "nifm:a" => self.nifm.dispatch(cmd_id),
            "nvnflinger" => self.nvnflinger.dispatch(cmd_id),
            _ => {
                log::warn!("unknown service: {}", port_name);
                1
            }
        }
    }
}

impl Default for Services {
    fn default() -> Self {
        Self::new()
    }
}

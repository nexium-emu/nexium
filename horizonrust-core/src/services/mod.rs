pub mod base;
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
pub mod ns;
pub mod acc;
pub mod pctl;
pub mod ssl;
pub mod spl;
pub mod pl;
pub mod psc;
pub mod smc;
pub mod fatal;
pub mod apm;
pub mod bsd;
pub mod caps;
pub mod misc;
pub mod pdm;
pub mod prepo;
pub mod psm;
pub mod ldr;
pub mod lr;
pub mod ldn;
pub mod friends;
pub mod fsp;
pub mod btm;
pub mod mii;
pub mod irs;
pub mod hwopus;
pub mod grc;
pub mod gpio;
pub mod pcielm;
pub mod rtc;
pub mod jit;
pub mod omm;
pub mod nim;
pub mod olsc;
pub mod mount;
pub mod tc;
pub mod gm;
pub mod pm;
pub mod usb;
pub mod sys;
pub mod gpio2;

pub struct FrameOut {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct IpcCtx<'a> {
    pub tls_buf: &'a [u8],
    pub pending_frames: &'a mut Vec<FrameOut>,
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
    pub ns: ns::ContentService,
    pub acc: acc::AccountService,
    pub pctl: pctl::ParentalControlService,
    pub ssl: ssl::SslService,
    pub spl: spl::SplService,
    pub pl: pl::PlService,
    pub psc: psc::PscService,
    pub smc: smc::SmcService,
    pub fatal: fatal::FatalService,
    pub apm: apm::ApmService,
    pub bsd: bsd::BsdService,
    pub caps: caps::CapsService,
    pub misc: misc::MiscService,
    pub pdm: pdm::PdmService,
    pub prepo: prepo::PrepoService,
    pub psm: psm::PsmService,
    pub ldr: ldr::LdrService,
    pub lr: lr::LrService,
    pub ldn: ldn::LdnService,
    pub friends: friends::FriendsService,
    pub fsp: fsp::FspService,
    pub btm: btm::BluetoothService,
    pub mii: mii::MiiService,
    pub irs: irs::InfraredService,
    pub hwopus: hwopus::HwOpusService,
    pub grc: grc::GameRecordingService,
    pub gpio: gpio::GpioService,
    pub pcielm: pcielm::PcieLmService,
    pub rtc: rtc::RtcService,
    pub jit: jit::JitService,
    pub omm: omm::OmmService,
    pub nim: nim::NimService,
    pub olsc: olsc::OlscService,
    pub mount: mount::MountService,
    pub tc: tc::TcService,
    pub gm: gm::GmService,
    pub pm: pm::PmService,
    pub usb: usb::UsbService,
    pub sys: sys::SysService,
    pub gpio2: gpio2::Gpio2Service,
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
            ns: ns::ContentService::new(),
            acc: acc::AccountService::new(),
            pctl: pctl::ParentalControlService::new(),
            ssl: ssl::SslService::new(),
            spl: spl::SplService::new(),
            pl: pl::PlService::new(),
            psc: psc::PscService::new(),
            smc: smc::SmcService::new(),
            fatal: fatal::FatalService::new(),
            apm: apm::ApmService::new(),
            bsd: bsd::BsdService::new(),
            caps: caps::CapsService::new(),
            misc: misc::MiscService::new(),
            pdm: pdm::PdmService::new(),
            prepo: prepo::PrepoService::new(),
            psm: psm::PsmService::new(),
            ldr: ldr::LdrService::new(),
            lr: lr::LrService::new(),
            ldn: ldn::LdnService::new(),
            friends: friends::FriendsService::new(),
            fsp: fsp::FspService::new(),
            btm: btm::BluetoothService::new(),
            mii: mii::MiiService::new(),
            irs: irs::InfraredService::new(),
            hwopus: hwopus::HwOpusService::new(),
            grc: grc::GameRecordingService::new(),
            gpio: gpio::GpioService::new(),
            pcielm: pcielm::PcieLmService::new(),
            rtc: rtc::RtcService::new(),
            jit: jit::JitService::new(),
            omm: omm::OmmService::new(),
            nim: nim::NimService::new(),
            olsc: olsc::OlscService::new(),
            mount: mount::MountService::new(),
            tc: tc::TcService::new(),
            gm: gm::GmService::new(),
            pm: pm::PmService::new(),
            usb: usb::UsbService::new(),
            sys: sys::SysService::new(),
            gpio2: gpio2::Gpio2Service::new(),
        }
    }

    pub fn dispatch_service(&mut self, port_name: &str, cmd_id: u32, ctx: &mut IpcCtx) -> u32 {
        log::trace!("dispatch_service: port={} cmd_id={}", port_name, cmd_id);
        match port_name {
            "sm:" => self.sm.dispatch(cmd_id),
            "hid" => self.hid.dispatch(cmd_id),
            "time:s" | "time:a" | "time:r" => self.time.dispatch(cmd_id),
            "set" | "set:sys" => self.set.dispatch(cmd_id),
            "am" | "appletOE" | "appletAE" => self.am.dispatch(cmd_id),
            "vi:m" | "vi:s" | "vi:u" => self.vi.dispatch(cmd_id),
            "audio" | "audout:u" => self.audio.dispatch(cmd_id),
            "fsp-srv" => self.fs.dispatch(cmd_id),
            "nifm:u" | "nifm:a" => self.nifm.dispatch(cmd_id),
            "nvnflinger" | "dispdrv" => self.nvnflinger.dispatch(cmd_id, ctx),
            "ns:am2" | "ns:am" | "ns" => self.ns.dispatch(cmd_id),
            "acc:u0" | "acc:u1" | "acc:aa" => self.acc.dispatch(cmd_id),
            "pctl:a" | "pctl:r" | "pctl:s" => self.pctl.dispatch(cmd_id),
            "ssl" => self.ssl.dispatch(cmd_id),
            "spl:" => self.spl.dispatch(cmd_id),
            "pl:u" | "pl:s" => self.pl.dispatch(cmd_id),
            "psc:m" => self.psc.dispatch(cmd_id),
            "smc:" => self.smc.dispatch(cmd_id),
            "fatal:u" | "fatal:p" => self.fatal.dispatch(cmd_id),
            "apm" | "apm:am" => self.apm.dispatch(cmd_id),
            "bsd:u" | "bsd:s" => self.bsd.dispatch(cmd_id),
            "caps:a" | "caps:c" | "caps:su" | "caps:sc" => self.caps.dispatch(cmd_id),
            "lm" => self.misc.dispatch(cmd_id),
            "pdm:ntfy" | "pdm:qry" => self.pdm.dispatch(cmd_id),
            "prepo:a" | "prepo:m" | "prepo:u" => self.prepo.dispatch(cmd_id),
            "psm" => self.psm.dispatch(cmd_id),
            "ldr:ro" => self.ldr.dispatch(cmd_id),
            "lr" => self.lr.dispatch(cmd_id),
            "ldn:u" | "ldn:s" | "ldn:m" => self.ldn.dispatch(cmd_id),
            "friend:u" | "friend:a" | "friend:s" | "friend:v" => self.friends.dispatch(cmd_id),
            "fsp:pr" | "fsp:pc" => self.fsp.dispatch(cmd_id),
            "btm" | "btm:u" | "btm:dbg" => self.btm.dispatch(cmd_id),
            "mii:u" | "mii:e" => self.mii.dispatch(cmd_id),
            "irs:u" | "irs:o" => self.irs.dispatch(cmd_id),
            "hwopus" => self.hwopus.dispatch(cmd_id),
            "grc:u" | "grc:d" => self.grc.dispatch(cmd_id),
            "gpio" => self.gpio.dispatch(cmd_id),
            "pcielm" => self.pcielm.dispatch(cmd_id),
            "rtc" => self.rtc.dispatch(cmd_id),
            "jit:u" => self.jit.dispatch(cmd_id),
            "omm" => self.omm.dispatch(cmd_id),
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

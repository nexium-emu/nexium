use once_cell::sync::Lazy;
use parking_lot::Mutex;

pub const OUT_STRING_BYTES: usize = 0x7D4;
pub const OUT_DATA_BYTES: usize = 4 + OUT_STRING_BYTES;
pub const CONFIG_COMMON_SIZE: usize = 0x3D4;
pub const LIBAPPLET_ARGS_SIZE: usize = 0x20;
pub const DEFAULT_MAX_LEN: u32 = 500;

#[derive(Clone, Debug, Default)]
pub struct SwkbdConfig {
    pub keyboard_mode: u32,
    pub ok_text: String,
    pub header_text: String,
    pub sub_text: String,
    pub guide_text: String,
    pub max_len: u32,
    pub min_len: u32,
    pub password: bool,
    pub initial_string_offset: u32,
    pub initial_string_units: u32,
    pub text_check: bool,
}

#[derive(Clone, Debug)]
pub struct SwkbdRequest {
    pub generation: u64,
    pub header_text: String,
    pub sub_text: String,
    pub guide_text: String,
    pub ok_text: String,
    pub initial_text: String,
    pub max_len: u32,
    pub min_len: u32,
    pub password: bool,
}

#[derive(Clone, Debug)]
pub struct SwkbdResponse {
    pub generation: u64,
    pub accepted: bool,
    pub text: String,
}

#[derive(Default)]
struct SwkbdShared {
    generation: u64,
    la_version: u32,
    config: Option<SwkbdConfig>,
    request: Option<SwkbdRequest>,
    response: Option<SwkbdResponse>,
    out_data: Vec<u8>,
    completed: bool,
}

static SWKBD: Lazy<Mutex<SwkbdShared>> = Lazy::new(|| Mutex::new(SwkbdShared::default()));

pub fn begin_applet() -> u64 {
    let mut s = SWKBD.lock();
    s.generation += 1;
    s.la_version = 0;
    s.config = None;
    s.request = None;
    s.response = None;
    s.out_data.clear();
    s.completed = false;
    s.generation
}

fn read_utf16_field(buf: &[u8], offset: usize, max_units: usize) -> String {
    let mut units = Vec::with_capacity(max_units);
    for i in 0..max_units {
        let at = offset + i * 2;
        if at + 2 > buf.len() {
            break;
        }
        let u = u16::from_le_bytes([buf[at], buf[at + 1]]);
        if u == 0 {
            break;
        }
        units.push(u);
    }
    String::from_utf16_lossy(&units)
}

fn u32_at(buf: &[u8], offset: usize) -> u32 {
    if offset + 4 > buf.len() {
        return 0;
    }
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

pub fn capture_storage_write(bytes: &[u8]) {
    let mut s = SWKBD.lock();
    if bytes.len() == LIBAPPLET_ARGS_SIZE {
        s.la_version = u32_at(bytes, 8);
        log::debug!(
            "swkbd: captured LibAppletArgs la_version={:#x}",
            s.la_version
        );
        return;
    }
    if bytes.len() >= CONFIG_COMMON_SIZE {
        let cfg = SwkbdConfig {
            keyboard_mode: u32_at(bytes, 0x00),
            ok_text: read_utf16_field(bytes, 0x04, 8),
            header_text: read_utf16_field(bytes, 0x24, 64),
            sub_text: read_utf16_field(bytes, 0xA6, 128),
            guide_text: read_utf16_field(bytes, 0x1A8, 256),
            max_len: u32_at(bytes, 0x3AC),
            min_len: u32_at(bytes, 0x3B0),
            password: u32_at(bytes, 0x3B4) != 0,
            initial_string_offset: u32_at(bytes, 0x3C0),
            initial_string_units: u32_at(bytes, 0x3C4),
            text_check: bytes.len() > 0x3D0 && bytes[0x3D0] != 0,
        };
        log::info!(
            "swkbd: captured config size={:#x} mode={} header='{}' max={} min={} password={} initial=+{:#x}x{} text_check={}",
            bytes.len(),
            cfg.keyboard_mode,
            cfg.header_text,
            cfg.max_len,
            cfg.min_len,
            cfg.password,
            cfg.initial_string_offset,
            cfg.initial_string_units,
            cfg.text_check
        );
        s.config = Some(cfg);
        return;
    }
    log::debug!(
        "swkbd: ignoring storage write of {} bytes (not args/config)",
        bytes.len()
    );
}

pub fn config_initial_span() -> Option<(u32, u32)> {
    let s = SWKBD.lock();
    let cfg = s.config.as_ref()?;
    if cfg.initial_string_units == 0 {
        return None;
    }
    Some((cfg.initial_string_offset, cfg.initial_string_units))
}

pub fn start(initial_text: String) -> u64 {
    let mut s = SWKBD.lock();
    let cfg = s.config.clone().unwrap_or_default();
    let max_len = if cfg.max_len == 0 {
        DEFAULT_MAX_LEN
    } else {
        cfg.max_len
    };
    let generation = s.generation;
    let req = SwkbdRequest {
        generation,
        header_text: cfg.header_text,
        sub_text: cfg.sub_text,
        guide_text: cfg.guide_text,
        ok_text: cfg.ok_text,
        initial_text,
        max_len,
        min_len: cfg.min_len,
        password: cfg.password,
    };
    log::info!(
        "swkbd: request published gen={} header='{}' initial='{}' max={}",
        req.generation,
        req.header_text,
        req.initial_text,
        req.max_len
    );
    s.request = Some(req);
    generation
}

pub fn take_request() -> Option<SwkbdRequest> {
    SWKBD.lock().request.take()
}

pub fn post_response(response: SwkbdResponse) {
    let mut s = SWKBD.lock();
    if response.generation != s.generation {
        log::warn!(
            "swkbd: dropping stale response gen={} (current {})",
            response.generation,
            s.generation
        );
        return;
    }
    s.response = Some(response);
}

pub fn take_response() -> Option<SwkbdResponse> {
    let mut s = SWKBD.lock();
    let resp = s.response.take()?;
    if resp.generation != s.generation {
        return None;
    }
    Some(resp)
}

pub fn build_out_data(accepted: bool, text: &str) -> Vec<u8> {
    let mut out = vec![0u8; OUT_DATA_BYTES];
    let close_result: u32 = if accepted { 0 } else { 1 };
    out[0..4].copy_from_slice(&close_result.to_le_bytes());
    let mut at = 4;
    for unit in text.encode_utf16() {
        if at + 2 > 4 + OUT_STRING_BYTES - 2 {
            break;
        }
        out[at..at + 2].copy_from_slice(&unit.to_le_bytes());
        at += 2;
    }
    out
}

pub fn set_completed(out_data: Vec<u8>) {
    let mut s = SWKBD.lock();
    s.out_data = out_data;
    s.completed = true;
}

pub fn is_completed() -> bool {
    SWKBD.lock().completed
}

pub fn out_data() -> Vec<u8> {
    SWKBD.lock().out_data.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn config_blob(header: &str, max_len: u32, initial_off: u32, initial_units: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 0x3E8];
        let mut at = 0x24;
        for unit in header.encode_utf16() {
            buf[at..at + 2].copy_from_slice(&unit.to_le_bytes());
            at += 2;
        }
        buf[0x3AC..0x3B0].copy_from_slice(&max_len.to_le_bytes());
        buf[0x3C0..0x3C4].copy_from_slice(&initial_off.to_le_bytes());
        buf[0x3C4..0x3C8].copy_from_slice(&initial_units.to_le_bytes());
        buf
    }

    #[test]
    fn capture_discriminates_args_and_config() {
        let _guard = TEST_LOCK.lock();
        begin_applet();
        let mut args = vec![0u8; LIBAPPLET_ARGS_SIZE];
        args[8..12].copy_from_slice(&0x8000Du32.to_le_bytes());
        capture_storage_write(&args);
        capture_storage_write(&config_blob("Server", 128, 0x14, 3));
        let span = config_initial_span().expect("initial span");
        assert_eq!(span, (0x14, 3));
        let generation = start("abc".to_string());
        let req = take_request().expect("request");
        assert_eq!(req.generation, generation);
        assert_eq!(req.header_text, "Server");
        assert_eq!(req.max_len, 128);
        assert_eq!(req.initial_text, "abc");
    }

    #[test]
    fn zero_max_len_defaults() {
        let _guard = TEST_LOCK.lock();
        begin_applet();
        capture_storage_write(&config_blob("", 0, 0, 0));
        start(String::new());
        let req = take_request().expect("request");
        assert_eq!(req.max_len, DEFAULT_MAX_LEN);
        assert!(config_initial_span().is_none());
    }

    #[test]
    fn out_data_layout_matches_libnx_expectation() {
        let ok = build_out_data(true, "ab");
        assert_eq!(ok.len(), OUT_DATA_BYTES);
        assert_eq!(&ok[0..4], &0u32.to_le_bytes());
        assert_eq!(u16::from_le_bytes([ok[4], ok[5]]), 'a' as u16);
        assert_eq!(u16::from_le_bytes([ok[6], ok[7]]), 'b' as u16);
        assert_eq!(u16::from_le_bytes([ok[8], ok[9]]), 0);
        let cancel = build_out_data(false, "");
        assert_eq!(&cancel[0..4], &1u32.to_le_bytes());
    }

    #[test]
    fn response_generation_gating() {
        let _guard = TEST_LOCK.lock();
        let generation = begin_applet();
        post_response(SwkbdResponse {
            generation: generation.wrapping_sub(1),
            accepted: true,
            text: "stale".into(),
        });
        assert!(take_response().is_none());
        post_response(SwkbdResponse {
            generation,
            accepted: true,
            text: "fresh".into(),
        });
        let resp = take_response().expect("fresh response");
        assert_eq!(resp.text, "fresh");
        assert!(take_response().is_none());
    }

    #[test]
    fn long_text_truncates_inside_string_buffer() {
        let text: String = std::iter::repeat('x').take(3000).collect();
        let out = build_out_data(true, &text);
        assert_eq!(out.len(), OUT_DATA_BYTES);
        assert_eq!(
            u16::from_le_bytes([out[OUT_DATA_BYTES - 2], out[OUT_DATA_BYTES - 1]]),
            0
        );
    }
}

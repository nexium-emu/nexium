use std::ffi::{c_int, c_void};

unsafe extern "C" {
    fn sceUserServiceInitialize(params: *const c_void) -> c_int;
    fn sceUserServiceGetInitialUser(user: *mut i32) -> c_int;
    fn scePadInit() -> c_int;
    fn scePadOpen(user: i32, port_type: i32, index: i32, params: *const c_void) -> c_int;
    fn scePadRead(handle: c_int, samples: *mut c_void, capacity: c_int) -> c_int;
    fn scePadClose(handle: c_int) -> c_int;
}

pub const L3: u32 = 0x2;
pub const R3: u32 = 0x4;
pub const OPTIONS: u32 = 0x8;
pub const UP: u32 = 0x10;
pub const RIGHT: u32 = 0x20;
pub const DOWN: u32 = 0x40;
pub const LEFT: u32 = 0x80;
pub const L2: u32 = 0x100;
pub const R2: u32 = 0x200;
pub const L1: u32 = 0x400;
pub const R1: u32 = 0x800;
pub const TRIANGLE: u32 = 0x1000;
pub const CIRCLE: u32 = 0x2000;
pub const CROSS: u32 = 0x4000;
pub const SQUARE: u32 = 0x8000;
pub const TOUCH_PAD: u32 = 0x100000;
pub const SHELL_OWNS_PAD: u32 = 0x8000_0000;

const SAMPLE_BYTES: usize = 120;
const MAX_SAMPLES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadState {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
    pub connected: bool,
    pub timestamp_us: u64,
}

impl Default for PadState {
    fn default() -> Self {
        Self { buttons: 0, lx: 128, ly: 128, rx: 128, ry: 128, l2: 0, r2: 0, connected: false, timestamp_us: 0 }
    }
}

impl PadState {
    pub fn stick(v: u8) -> f32 {
        ((v as f32 - 128.0) / 127.0).clamp(-1.0, 1.0)
    }
}

pub struct Pad {
    handle: c_int,
    pub user: i32,
    last: PadState,
    buffer: Vec<u8>,
}

impl Pad {
    pub fn open() -> Result<Self, String> {
        unsafe {
            let init = sceUserServiceInitialize(std::ptr::null());
            let mut user = -1;
            let rc = sceUserServiceGetInitialUser(&mut user);
            if rc != 0 {
                return Err(format!("sceUserServiceGetInitialUser {rc:#x} (init {init:#x})"));
            }
            let rc = scePadInit();
            if rc != 0 {
                return Err(format!("scePadInit {rc:#x}"));
            }
            let mut last = 0;
            for _ in 0..10 {
                last = scePadOpen(user, 0, 0, std::ptr::null());
                if last >= 0 {
                    return Ok(Self {
                        handle: last,
                        user,
                        last: PadState::default(),
                        buffer: vec![0u8; SAMPLE_BYTES * MAX_SAMPLES],
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(format!("scePadOpen {last:#x} for user {user}"))
        }
    }

    pub fn poll(&mut self) -> PadState {
        let count = unsafe { scePadRead(self.handle, self.buffer.as_mut_ptr().cast(), MAX_SAMPLES as c_int) };
        if count <= 0 {
            return self.last;
        }
        let sample = &self.buffer[(count as usize - 1) * SAMPLE_BYTES..count as usize * SAMPLE_BYTES];
        let buttons = u32::from_le_bytes(sample[0..4].try_into().unwrap());
        if buttons & SHELL_OWNS_PAD != 0 {
            self.last.buttons = 0;
            return self.last;
        }
        self.last = PadState {
            buttons,
            lx: sample[4],
            ly: sample[5],
            rx: sample[6],
            ry: sample[7],
            l2: sample[8],
            r2: sample[9],
            connected: i32::from_le_bytes(sample[0x4c..0x50].try_into().unwrap()) != 0,
            timestamp_us: u64::from_le_bytes(sample[0x50..0x58].try_into().unwrap()),
        };
        self.last
    }
}

impl Drop for Pad {
    fn drop(&mut self) {
        unsafe {
            scePadClose(self.handle);
        }
    }
}

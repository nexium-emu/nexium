use parking_lot::Mutex;
use std::sync::Arc;

pub const HID_SHMEM_SIZE: usize = 0x40000;

const TOUCH_OFFSET: usize = 0x400;
const MOUSE_OFFSET: usize = 0x3400;
const KEYBOARD_OFFSET: usize = 0x3800;
const DEBUGPAD_OFFSET: usize = 0x0;

const NPAD_OFFSET: usize = 0x9A00;
const NPAD_ENTRY_SIZE: usize = 0x5000;
const NPAD_ENTRY_HANDHELD: usize = 8;
const NPAD_ENTRY_PLAYER1: usize = 0;

const NPAD_STYLE_TAG_OFFSET: usize = 0x00;
const NPAD_JOY_ASSIGN_OFFSET: usize = 0x04;

const LAYOUT_BASE_OFFSET: usize = 0x28;
const LAYOUT_STRIDE: usize = 0x350;
const LAYOUT_COUNT: usize = 7;

const LIFO_HEADER_SIZE: usize = 0x20;
const LIFO_STORAGE_ELEM_SIZE: usize = 0x30;
const LIFO_STORAGE_COUNT: usize = 17;

pub const STYLE_FULLKEY: u32 = 1 << 0;
pub const STYLE_HANDHELD: u32 = 1 << 1;

pub const ATTR_IS_CONNECTED: u32 = 1 << 0;
pub const ATTR_IS_WIRED: u32 = 1 << 1;

pub const NPAD_BUTTON_A: u64 = 1 << 0;
pub const NPAD_BUTTON_B: u64 = 1 << 1;
pub const NPAD_BUTTON_X: u64 = 1 << 2;
pub const NPAD_BUTTON_Y: u64 = 1 << 3;
pub const NPAD_BUTTON_STICK_L: u64 = 1 << 4;
pub const NPAD_BUTTON_STICK_R: u64 = 1 << 5;
pub const NPAD_BUTTON_L: u64 = 1 << 6;
pub const NPAD_BUTTON_R: u64 = 1 << 7;
pub const NPAD_BUTTON_ZL: u64 = 1 << 8;
pub const NPAD_BUTTON_ZR: u64 = 1 << 9;
pub const NPAD_BUTTON_PLUS: u64 = 1 << 10;
pub const NPAD_BUTTON_MINUS: u64 = 1 << 11;
pub const NPAD_BUTTON_LEFT: u64 = 1 << 12;
pub const NPAD_BUTTON_UP: u64 = 1 << 13;
pub const NPAD_BUTTON_RIGHT: u64 = 1 << 14;
pub const NPAD_BUTTON_DOWN: u64 = 1 << 15;

#[derive(Default, Clone, Copy)]
pub struct ControllerInput {
    pub buttons: u64,
    pub stick_l_x: i32,
    pub stick_l_y: i32,
    pub stick_r_x: i32,
    pub stick_r_y: i32,
}

pub struct HidState {
    buf: Box<[u8; HID_SHMEM_SIZE]>,
    pub input: ControllerInput,
    pub sampling_number: u64,
    pub shmem_va: Option<u64>,
    last_logged_buttons: u64,
    last_tick: Option<std::time::Instant>,
    dumped_shmem: bool,
}

impl HidState {
    pub fn new() -> Self {
        let mut s = Self {
            buf: Box::new([0u8; HID_SHMEM_SIZE]),
            input: ControllerInput::default(),
            sampling_number: 0,
            shmem_va: None,
            last_logged_buttons: 0,
            last_tick: None,
            dumped_shmem: false,
        };
        s.init_metadata();
        s.tick(ControllerInput::default());
        s
    }

    pub fn maybe_tick(&mut self, input: ControllerInput) {
        const VSYNC: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let now = std::time::Instant::now();
        let force = input.buttons != self.input.buttons
            || input.stick_l_x != self.input.stick_l_x
            || input.stick_l_y != self.input.stick_l_y
            || input.stick_r_x != self.input.stick_r_x
            || input.stick_r_y != self.input.stick_r_y;
        let elapsed = self
            .last_tick
            .map(|t| now.duration_since(t))
            .unwrap_or(VSYNC);
        if force || elapsed >= VSYNC {
            self.last_tick = Some(now);
            self.tick(input);
        }
    }

    pub fn host_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    pub fn size(&self) -> usize {
        HID_SHMEM_SIZE
    }

    pub fn update_input(&mut self, input: ControllerInput) {
        self.input = input;
    }

    pub fn tick(&mut self, input: ControllerInput) {
        if input.buttons != self.last_logged_buttons {
            log::info!(
                "hid:tick buttons={:#x} (was {:#x})",
                input.buttons,
                self.last_logged_buttons
            );
            self.last_logged_buttons = input.buttons;
        }
        self.sampling_number = self.sampling_number.wrapping_add(1);
        let configs = [(NPAD_ENTRY_PLAYER1, false), (NPAD_ENTRY_HANDHELD, true)];
        let sampling = self.sampling_number;
        for (entry_idx, _is_handheld) in configs {
            for layout in 0..LAYOUT_COUNT {
                let lifo_off = LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
                Self::write_npad_entry(&mut self.buf[..], entry_idx, lifo_off, &input, sampling);
            }
        }

        if !self.dumped_shmem && input.buttons != 0 {
            self.dumped_shmem = true;
            log::info!(
                "hid:shmem-dump @va={:?} sampling={}",
                self.shmem_va,
                sampling
            );
            for (entry_idx, name) in [
                (NPAD_ENTRY_PLAYER1, "Player1"),
                (NPAD_ENTRY_HANDHELD, "Handheld"),
            ] {
                let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
                let style =
                    u32::from_le_bytes(self.buf[base..base + 4].try_into().unwrap_or([0; 4]));
                log::info!(
                    "hid:shmem {} (entry={}) style_tag={:#x}",
                    name,
                    entry_idx,
                    style
                );
                for (layout_idx, layout_name) in
                    [(0usize, "FullKey"), (1, "Handheld"), (2, "JoyDual")]
                {
                    let lifo = base + LAYOUT_BASE_OFFSET + layout_idx * LAYOUT_STRIDE;
                    let hdr0 =
                        u64::from_le_bytes(self.buf[lifo..lifo + 8].try_into().unwrap_or([0; 8]));
                    let state = lifo + LIFO_HEADER_SIZE + 8;
                    let st_sample =
                        u64::from_le_bytes(self.buf[state..state + 8].try_into().unwrap_or([0; 8]));
                    let st_btn = u64::from_le_bytes(
                        self.buf[state + 8..state + 16].try_into().unwrap_or([0; 8]),
                    );
                    let st_attr = u32::from_le_bytes(
                        self.buf[state + 0x20..state + 0x24]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                    log::info!(
                        "  layout[{}={}] lifo_off=+{:#x} latest_sample={} state.sample={} state.buttons={:#x} state.attr={:#x}",
                        layout_idx, layout_name, lifo - base, hdr0, st_sample, st_btn, st_attr
                    );
                }
            }
        }
    }

    fn init_metadata(&mut self) {
        let style = STYLE_FULLKEY | STYLE_HANDHELD;
        for &idx in &[NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD] {
            let base = NPAD_OFFSET + idx * NPAD_ENTRY_SIZE;
            write_u32(&mut *self.buf, base + NPAD_STYLE_TAG_OFFSET, style);
            write_u32(&mut *self.buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
        }
        for &off in &[DEBUGPAD_OFFSET, TOUCH_OFFSET, MOUSE_OFFSET, KEYBOARD_OFFSET] {
            Self::write_empty_lifo(&mut self.buf[..], off);
        }
    }

    fn write_empty_lifo(buf: &mut [u8], lifo: usize) {
        write_u64(buf, lifo + 0x00, 1);
        write_u64(buf, lifo + 0x08, LIFO_STORAGE_COUNT as u64);
        write_u64(buf, lifo + 0x10, 0);
        write_u64(buf, lifo + 0x18, 1);
        let e0 = lifo + LIFO_HEADER_SIZE;
        write_u64(buf, e0 + 0x00, 1);
        write_u64(buf, e0 + 0x08, 1);
        write_u64(buf, e0 + 0x10, 0);
    }

    fn write_npad_entry(
        buf: &mut [u8],
        entry_idx: usize,
        lifo_offset_in_entry: usize,
        input: &ControllerInput,
        sampling: u64,
    ) {
        let entry_base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        let lifo = entry_base + lifo_offset_in_entry;

        write_u64(buf, lifo + 0x00, sampling);
        write_u64(buf, lifo + 0x08, sampling);
        write_u64(buf, lifo + 0x10, 0);
        write_u64(buf, lifo + 0x18, 1);

        let storage0 = lifo + LIFO_HEADER_SIZE;
        write_u64(buf, storage0, sampling.wrapping_mul(2));
        let state = storage0 + 8;
        write_u64(buf, state + 0x00, sampling);
        write_u64(buf, state + 0x08, input.buttons);
        write_i32(buf, state + 0x10, input.stick_l_x);
        write_i32(buf, state + 0x14, input.stick_l_y);
        write_i32(buf, state + 0x18, input.stick_r_x);
        write_i32(buf, state + 0x1C, input.stick_r_y);
        write_u32(buf, state + 0x20, ATTR_IS_CONNECTED | ATTR_IS_WIRED);
        write_u32(buf, state + 0x24, 0);

        for i in 1..LIFO_STORAGE_COUNT {
            let storage_i = lifo + LIFO_HEADER_SIZE + i * LIFO_STORAGE_ELEM_SIZE;
            write_u64(buf, storage_i, 0);
        }
    }
}

fn write_u64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn write_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn write_i32(buf: &mut [u8], off: usize, v: i32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub static HID_STATE: once_cell::sync::OnceCell<Arc<Mutex<HidState>>> =
    once_cell::sync::OnceCell::new();

pub fn get_hid_state() -> Arc<Mutex<HidState>> {
    HID_STATE
        .get_or_init(|| Arc::new(Mutex::new(HidState::new())))
        .clone()
}

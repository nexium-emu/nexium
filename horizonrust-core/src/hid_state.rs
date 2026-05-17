use parking_lot::Mutex;
use std::sync::Arc;

pub const HID_SHMEM_SIZE: usize = 0x40000;

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
    pub input: ControllerInput,
    pub sampling_number: u64,
    pub shmem_va: Option<u64>,
}

impl HidState {
    pub fn new() -> Self {
        Self {
            input: ControllerInput::default(),
            sampling_number: 0,
            shmem_va: None,
        }
    }

    pub fn build_initial_shmem(&mut self) -> Vec<u8> {
        let mut buf = vec![0u8; HID_SHMEM_SIZE];
        self.init_metadata(&mut buf);
        self.write_all_entries(&mut buf);
        buf
    }

    pub fn update_input(&mut self, input: ControllerInput) {
        self.input = input;
        self.sampling_number = self.sampling_number.wrapping_add(1);
    }

    fn init_metadata(&self, buf: &mut [u8]) {
        for &entry_idx in &[NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD] {
            let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
            let style = STYLE_FULLKEY | STYLE_HANDHELD;
            write_u32(buf, base + NPAD_STYLE_TAG_OFFSET, style);
            write_u32(buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
        }
    }

    fn write_all_entries(&self, buf: &mut [u8]) {
        for &entry_idx in &[NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD] {
            for layout in 0..LAYOUT_COUNT {
                let lifo_off = LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
                self.write_npad_entry(buf, entry_idx, lifo_off);
            }
        }
    }

    fn write_npad_entry(&self, buf: &mut [u8], entry_idx: usize, lifo_offset_in_entry: usize) {
        let entry_base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        let lifo = entry_base + lifo_offset_in_entry;

        let total = self.sampling_number;
        let tail: u64 = 0;
        let count: u64 = 1;

        write_u64(buf, lifo + 0x00, self.sampling_number);
        write_u64(buf, lifo + 0x08, total);
        write_u64(buf, lifo + 0x10, tail);
        write_u64(buf, lifo + 0x18, count);

        let storage0 = lifo + LIFO_HEADER_SIZE;
        write_u64(buf, storage0, self.sampling_number.wrapping_mul(2));
        let state = storage0 + 8;
        write_u64(buf, state + 0x00, self.sampling_number);
        write_u64(buf, state + 0x08, self.input.buttons);
        write_i32(buf, state + 0x10, self.input.stick_l_x);
        write_i32(buf, state + 0x14, self.input.stick_l_y);
        write_i32(buf, state + 0x18, self.input.stick_r_x);
        write_i32(buf, state + 0x1C, self.input.stick_r_y);
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

pub static HID_STATE: once_cell::sync::OnceCell<Arc<Mutex<HidState>>> = once_cell::sync::OnceCell::new();

pub fn get_hid_state() -> Arc<Mutex<HidState>> {
    HID_STATE
        .get_or_init(|| Arc::new(Mutex::new(HidState::new())))
        .clone()
}

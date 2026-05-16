use parking_lot::Mutex;
use std::sync::Arc;

pub const HID_SHMEM_SIZE: usize = 0x40000;

pub const NPAD_OFFSET: usize = 0x9A00;
pub const NPAD_ENTRY_SIZE: usize = 0x5000;

pub const STYLE_TAG_NPAD_FULL_KEY: u32 = 1 << 0;
pub const STYLE_TAG_NPAD_HANDHELD: u32 = 1 << 1;

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

pub const NPAD_DEVICE_TYPE_FULL_KEY: u32 = 1 << 0;
pub const NPAD_DEVICE_TYPE_HANDHELD: u32 = 1 << 1;

#[derive(Default, Clone)]
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

    pub fn build_initial_shmem(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HID_SHMEM_SIZE];
        write_npad_entry(&mut buf, 0, &self.input, self.sampling_number);
        for id in 0..10 {
            let offset = NPAD_OFFSET + id * NPAD_ENTRY_SIZE;
            buf[offset..offset + 4].copy_from_slice(&STYLE_TAG_NPAD_FULL_KEY.to_le_bytes());
            buf[offset + 0x6028..offset + 0x602C].copy_from_slice(&NPAD_DEVICE_TYPE_FULL_KEY.to_le_bytes());
        }
        write_npad_entry(&mut buf, 0, &self.input, 1);
        buf
    }

    pub fn update_input(&mut self, input: ControllerInput) {
        self.input = input;
        self.sampling_number = self.sampling_number.wrapping_add(1);
    }
}

fn write_npad_entry(buf: &mut [u8], npad_id: usize, input: &ControllerInput, sampling: u64) {
    let base = NPAD_OFFSET + npad_id * NPAD_ENTRY_SIZE;
    if base + NPAD_ENTRY_SIZE > buf.len() {
        return;
    }

    buf[base..base + 4].copy_from_slice(&STYLE_TAG_NPAD_FULL_KEY.to_le_bytes());
    buf[base + 4..base + 8].copy_from_slice(&0u32.to_le_bytes());
    buf[base + 8..base + 12].copy_from_slice(&1u32.to_le_bytes());
    buf[base + 12..base + 16].copy_from_slice(&0xFF323232u32.to_le_bytes());
    buf[base + 16..base + 20].copy_from_slice(&1u32.to_le_bytes());
    buf[base + 20..base + 24].copy_from_slice(&0xFF323232u32.to_le_bytes());
    buf[base + 24..base + 28].copy_from_slice(&0xFF323232u32.to_le_bytes());

    let lifo_offset = base + 0x18;
    write_npad_lifo(&mut buf[lifo_offset..], input, sampling);

    let handheld_lifo_offset = base + 0x350;
    write_npad_lifo(&mut buf[handheld_lifo_offset..], input, sampling);

    let device_type_offset = base + 0x6028;
    buf[device_type_offset..device_type_offset + 4]
        .copy_from_slice(&NPAD_DEVICE_TYPE_FULL_KEY.to_le_bytes());

    let system_properties_offset = base + 0x6030;
    buf[system_properties_offset..system_properties_offset + 8]
        .copy_from_slice(&0u64.to_le_bytes());

    let battery_offset = base + 0x6044;
    buf[battery_offset..battery_offset + 4].copy_from_slice(&4u32.to_le_bytes());
    buf[battery_offset + 4..battery_offset + 8].copy_from_slice(&4u32.to_le_bytes());
    buf[battery_offset + 8..battery_offset + 12].copy_from_slice(&4u32.to_le_bytes());
}

fn write_npad_lifo(buf: &mut [u8], input: &ControllerInput, sampling: u64) {
    if buf.len() < 0x338 {
        return;
    }
    buf[0..8].copy_from_slice(&sampling.to_le_bytes());
    buf[8..16].copy_from_slice(&17u64.to_le_bytes());
    buf[16..24].copy_from_slice(&0u64.to_le_bytes());
    buf[24..32].copy_from_slice(&17u64.to_le_bytes());

    let entry_base = 32;
    let entry_size = 48;
    let entry = build_npad_state_entry(input, sampling);
    for i in 0..17 {
        let off = entry_base + i * entry_size;
        if off + entry.len() > buf.len() {
            break;
        }
        buf[off..off + entry.len()].copy_from_slice(&entry);
    }
}

fn build_npad_state_entry(input: &ControllerInput, sampling: u64) -> Vec<u8> {
    let mut entry = Vec::with_capacity(48);
    entry.extend_from_slice(&sampling.to_le_bytes());
    entry.extend_from_slice(&sampling.to_le_bytes());
    entry.extend_from_slice(&input.buttons.to_le_bytes());
    entry.extend_from_slice(&input.stick_l_x.to_le_bytes());
    entry.extend_from_slice(&input.stick_l_y.to_le_bytes());
    entry.extend_from_slice(&input.stick_r_x.to_le_bytes());
    entry.extend_from_slice(&input.stick_r_y.to_le_bytes());
    entry.extend_from_slice(&0u32.to_le_bytes());
    entry.extend_from_slice(&0u32.to_le_bytes());
    entry
}

pub fn map_keyboard_to_controller(keys: &[bool; 256]) -> ControllerInput {
    let mut input = ControllerInput::default();

    if keys[b'Z' as usize] { input.buttons |= NPAD_BUTTON_A; }
    if keys[b'X' as usize] { input.buttons |= NPAD_BUTTON_B; }
    if keys[b'A' as usize] { input.buttons |= NPAD_BUTTON_X; }
    if keys[b'S' as usize] { input.buttons |= NPAD_BUTTON_Y; }
    if keys[b'Q' as usize] { input.buttons |= NPAD_BUTTON_L; }
    if keys[b'W' as usize] { input.buttons |= NPAD_BUTTON_R; }
    if keys[b'1' as usize] { input.buttons |= NPAD_BUTTON_ZL; }
    if keys[b'2' as usize] { input.buttons |= NPAD_BUTTON_ZR; }
    if keys[b'\r' as usize] { input.buttons |= NPAD_BUTTON_PLUS; }
    if keys[b'\t' as usize] { input.buttons |= NPAD_BUTTON_MINUS; }
    if keys[37] { input.buttons |= NPAD_BUTTON_LEFT; input.stick_l_x = -30000; }
    if keys[38] { input.buttons |= NPAD_BUTTON_UP; input.stick_l_y = 30000; }
    if keys[39] { input.buttons |= NPAD_BUTTON_RIGHT; input.stick_l_x = 30000; }
    if keys[40] { input.buttons |= NPAD_BUTTON_DOWN; input.stick_l_y = -30000; }

    input
}

pub static HID_STATE: once_cell::sync::OnceCell<Arc<Mutex<HidState>>> = once_cell::sync::OnceCell::new();

pub fn get_hid_state() -> Arc<Mutex<HidState>> {
    HID_STATE
        .get_or_init(|| Arc::new(Mutex::new(HidState::new())))
        .clone()
}

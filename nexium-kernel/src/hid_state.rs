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
const NPAD_ENTRY_OTHER: usize = 9;
const NPAD_ENTRY_PLAYER1: usize = 0;

const NPAD_STYLE_TAG_OFFSET: usize = 0x00;
const NPAD_JOY_ASSIGN_OFFSET: usize = 0x04;
const NPAD_DEVICE_TYPE_OFFSET: usize = 0x4188;
const NPAD_SYSTEM_PROPERTIES_OFFSET: usize = 0x4190;
const NPAD_APPLET_FOOTER_OFFSET: usize = 0x41AC;

const DEVICE_TYPE_FULLKEY: u32 = 1 << 0;
const DEVICE_TYPE_HANDHELD_LEFT: u32 = 1 << 2;
const DEVICE_TYPE_HANDHELD_RIGHT: u32 = 1 << 3;

const SYSPROP_IS_VERTICAL: u64 = 1 << 11;
const SYSPROP_USE_PLUS: u64 = 1 << 13;
const SYSPROP_USE_MINUS: u64 = 1 << 14;
const SYSPROP_USE_DIRECTIONAL: u64 = 1 << 15;

const FOOTER_SWITCH_PRO: u8 = 12;
const FOOTER_HANDHELD: u8 = 4;

const LAYOUT_BASE_OFFSET: usize = 0x28;
const LAYOUT_STRIDE: usize = 0x350;
const LAYOUT_COUNT: usize = 7;

const LIFO_HEADER_SIZE: usize = 0x20;
const LIFO_STORAGE_ELEM_SIZE: usize = 0x30;
const LIFO_STORAGE_COUNT: usize = 17;

pub const STYLE_FULLKEY: u32 = 1 << 0;
pub const STYLE_HANDHELD: u32 = 1 << 1;
pub const STYLE_JOY_DUAL: u32 = 1 << 2;
pub const STYLE_JOY_LEFT: u32 = 1 << 3;
pub const STYLE_JOY_RIGHT: u32 = 1 << 4;
pub const STYLE_SYSTEM_EXT: u32 = 1 << 29;

pub const ATTR_IS_CONNECTED: u32 = 1 << 0;
pub const ATTR_IS_WIRED: u32 = 1 << 1;
pub const ATTR_LEFT_CONNECTED: u32 = 1 << 2;
pub const ATTR_LEFT_WIRED: u32 = 1 << 3;
pub const ATTR_RIGHT_CONNECTED: u32 = 1 << 4;
pub const ATTR_RIGHT_WIRED: u32 = 1 << 5;

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
pub const NPAD_BUTTON_STICK_L_LEFT: u64 = 1 << 16;
pub const NPAD_BUTTON_STICK_L_UP: u64 = 1 << 17;
pub const NPAD_BUTTON_STICK_L_RIGHT: u64 = 1 << 18;
pub const NPAD_BUTTON_STICK_L_DOWN: u64 = 1 << 19;
pub const NPAD_BUTTON_STICK_R_LEFT: u64 = 1 << 20;
pub const NPAD_BUTTON_STICK_R_UP: u64 = 1 << 21;
pub const NPAD_BUTTON_STICK_R_RIGHT: u64 = 1 << 22;
pub const NPAD_BUTTON_STICK_R_DOWN: u64 = 1 << 23;
pub const NPAD_BUTTON_LEFT_SL: u64 = 1 << 24;
pub const NPAD_BUTTON_LEFT_SR: u64 = 1 << 25;
pub const NPAD_BUTTON_RIGHT_SL: u64 = 1 << 26;
pub const NPAD_BUTTON_RIGHT_SR: u64 = 1 << 27;

#[derive(Default, Clone, Copy)]
pub struct ControllerInput {
    pub buttons: u64,
    pub stick_l_x: i32,
    pub stick_l_y: i32,
    pub stick_r_x: i32,
    pub stick_r_y: i32,
}

pub const MOUSE_BUTTON_LEFT: u32 = 1 << 0;
pub const MOUSE_BUTTON_RIGHT: u32 = 1 << 1;
pub const MOUSE_BUTTON_MIDDLE: u32 = 1 << 2;
pub const MOUSE_BUTTON_FORWARD: u32 = 1 << 3;
pub const MOUSE_BUTTON_BACK: u32 = 1 << 4;

const MOUSE_ATTR_IS_CONNECTED: u32 = 1 << 1;
const KEYBOARD_ATTR_IS_CONNECTED: u32 = 1 << 0;

pub const KEYBOARD_MOD_CONTROL: u32 = 1 << 0;
pub const KEYBOARD_MOD_SHIFT: u32 = 1 << 1;
pub const KEYBOARD_MOD_LEFT_ALT: u32 = 1 << 2;
pub const KEYBOARD_MOD_RIGHT_ALT: u32 = 1 << 3;

const MOUSE_ELEM_SIZE: usize = 0x30;
const KEYBOARD_ELEM_SIZE: usize = 0x38;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseInput {
    pub x: i32,
    pub y: i32,
    pub wheel_x: i32,
    pub wheel_y: i32,
    pub buttons: u32,
    pub connected: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct KeyboardInput {
    pub modifiers: u32,
    pub keys: [u8; 32],
    pub connected: bool,
}

impl Default for KeyboardInput {
    fn default() -> Self {
        Self {
            modifiers: 0,
            keys: [0; 32],
            connected: false,
        }
    }
}

pub struct HidState {
    buf: Box<[u8; HID_SHMEM_SIZE]>,
    mapped_host_ptr: usize,
    pub input: ControllerInput,
    pub mouse: MouseInput,
    pub keyboard: KeyboardInput,
    last_written_mouse: MouseInput,
    pub sampling_number: u64,
    pub shmem_va: Option<u64>,
    last_tick: Option<std::time::Instant>,
}

impl HidState {
    pub fn new() -> Self {
        let mut s = Self {
            buf: Box::new([0u8; HID_SHMEM_SIZE]),
            mapped_host_ptr: 0,
            input: ControllerInput::default(),
            mouse: MouseInput::default(),
            keyboard: KeyboardInput::default(),
            last_written_mouse: MouseInput::default(),
            sampling_number: 0,
            shmem_va: None,
            last_tick: None,
        };
        s.init_metadata();
        s.tick(ControllerInput::default());
        s
    }

    pub fn update_devices(&mut self, mouse: MouseInput, keyboard: KeyboardInput) {
        let force = mouse.buttons != self.mouse.buttons
            || mouse.connected != self.mouse.connected
            || keyboard != self.keyboard;
        self.mouse = mouse;
        self.keyboard = keyboard;
        if force {
            self.last_tick = Some(std::time::Instant::now());
            let input = self.input;
            self.tick(input);
        }
    }

    pub fn maybe_tick(&mut self, input: ControllerInput) {
        const VSYNC: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let now = std::time::Instant::now();
        let force = input.buttons != self.input.buttons
            || input.stick_l_x != self.input.stick_l_x
            || input.stick_l_y != self.input.stick_l_y
            || input.stick_r_x != self.input.stick_r_x
            || input.stick_r_y != self.input.stick_r_y;
        self.input = input;
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

    pub unsafe fn bind_mapped_host(&mut self, ptr: *mut u8) {
        self.mapped_host_ptr = ptr as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(self.buf.as_ptr(), ptr, HID_SHMEM_SIZE);
        }
    }

    pub fn unbind_mapped_host(&mut self, ptr: usize) -> bool {
        if self.mapped_host_ptr != ptr {
            return false;
        }
        self.mapped_host_ptr = 0;
        self.shmem_va = None;
        true
    }

    pub fn size(&self) -> usize {
        HID_SHMEM_SIZE
    }

    pub fn update_input(&mut self, input: ControllerInput) {
        self.maybe_tick(input);
    }

    pub fn tick(&mut self, input: ControllerInput) {
        self.input = input;
        self.sampling_number = self.sampling_number.wrapping_add(1);
        let sampling = self.sampling_number;
        let docked = CONSOLE_DOCKED.load(std::sync::atomic::Ordering::Relaxed);
        let player1_joy_dual = PLAYER1_JOY_DUAL.load(std::sync::atomic::Ordering::Relaxed);
        let active_entry = if docked || player1_joy_dual {
            NPAD_ENTRY_PLAYER1
        } else {
            NPAD_ENTRY_HANDHELD
        };
        for entry_idx in 0..=NPAD_ENTRY_OTHER {
            if entry_idx != active_entry {
                Self::write_entry_style(&mut self.buf[..], entry_idx, 0);
            }
        }
        if player1_joy_dual {
            let attr = ATTR_IS_CONNECTED
                | ATTR_IS_WIRED
                | ATTR_LEFT_CONNECTED
                | ATTR_LEFT_WIRED
                | ATTR_RIGHT_CONNECTED
                | ATTR_RIGHT_WIRED;
            Self::setup_joy_dual(&mut self.buf[..], NPAD_ENTRY_PLAYER1);
            Self::write_standard_npad_lifos(
                &mut self.buf[..],
                NPAD_ENTRY_PLAYER1,
                &input,
                sampling,
                attr,
            );
        } else if docked {
            Self::setup_fullkey(&mut self.buf[..], NPAD_ENTRY_PLAYER1);
            Self::write_standard_npad_lifos(
                &mut self.buf[..],
                NPAD_ENTRY_PLAYER1,
                &input,
                sampling,
                ATTR_IS_CONNECTED | ATTR_IS_WIRED,
            );
        } else {
            let attr = ATTR_IS_CONNECTED
                | ATTR_IS_WIRED
                | ATTR_LEFT_CONNECTED
                | ATTR_LEFT_WIRED
                | ATTR_RIGHT_CONNECTED
                | ATTR_RIGHT_WIRED;
            Self::setup_handheld(&mut self.buf[..], NPAD_ENTRY_HANDHELD);
            Self::write_standard_npad_lifos(
                &mut self.buf[..],
                NPAD_ENTRY_HANDHELD,
                &input,
                sampling,
                attr,
            );
        }

        let mouse = self.mouse;
        let previous_mouse = self.last_written_mouse;
        Self::write_mouse_lifo(&mut self.buf[..], &mouse, &previous_mouse, sampling);
        self.last_written_mouse = mouse;
        let keyboard = self.keyboard;
        Self::write_keyboard_lifo(&mut self.buf[..], &keyboard, sampling);

        if self.mapped_host_ptr != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf.as_ptr(),
                    self.mapped_host_ptr as *mut u8,
                    HID_SHMEM_SIZE,
                );
            }
        }
    }

    fn write_device_lifo_header(buf: &mut [u8], lifo: usize, sampling: u64) -> usize {
        let tail = (sampling % LIFO_STORAGE_COUNT as u64) as usize;
        let count = sampling.min(LIFO_STORAGE_COUNT as u64 - 1);
        write_u64(buf, lifo + 0x00, sampling);
        write_u64(buf, lifo + 0x08, LIFO_STORAGE_COUNT as u64);
        write_u64(buf, lifo + 0x10, tail as u64);
        write_u64(buf, lifo + 0x18, count);
        tail
    }

    fn write_mouse_lifo(buf: &mut [u8], mouse: &MouseInput, previous: &MouseInput, sampling: u64) {
        let tail = Self::write_device_lifo_header(buf, MOUSE_OFFSET, sampling);
        let storage = MOUSE_OFFSET + LIFO_HEADER_SIZE + tail * MOUSE_ELEM_SIZE;
        write_u64(buf, storage, sampling);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        write_i32(buf, state + 0x08, mouse.x);
        write_i32(buf, state + 0x0C, mouse.y);
        write_i32(buf, state + 0x10, mouse.x.wrapping_sub(previous.x));
        write_i32(buf, state + 0x14, mouse.y.wrapping_sub(previous.y));
        write_i32(
            buf,
            state + 0x18,
            mouse.wheel_y.wrapping_sub(previous.wheel_y),
        );
        write_i32(
            buf,
            state + 0x1C,
            mouse.wheel_x.wrapping_sub(previous.wheel_x),
        );
        write_u32(buf, state + 0x20, mouse.buttons);
        write_u32(
            buf,
            state + 0x24,
            if mouse.connected {
                MOUSE_ATTR_IS_CONNECTED
            } else {
                0
            },
        );
    }

    fn write_keyboard_lifo(buf: &mut [u8], keyboard: &KeyboardInput, sampling: u64) {
        let tail = Self::write_device_lifo_header(buf, KEYBOARD_OFFSET, sampling);
        let storage = KEYBOARD_OFFSET + LIFO_HEADER_SIZE + tail * KEYBOARD_ELEM_SIZE;
        write_u64(buf, storage, sampling);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        write_u32(buf, state + 0x08, keyboard.modifiers);
        write_u32(
            buf,
            state + 0x0C,
            if keyboard.connected {
                KEYBOARD_ATTR_IS_CONNECTED
            } else {
                0
            },
        );
        buf[state + 0x10..state + 0x30].copy_from_slice(&keyboard.keys);
    }

    fn init_metadata(&mut self) {
        for idx in 0..=NPAD_ENTRY_OTHER {
            let base = NPAD_OFFSET + idx * NPAD_ENTRY_SIZE;
            let style = match idx {
                NPAD_ENTRY_PLAYER1 => STYLE_FULLKEY,
                NPAD_ENTRY_HANDHELD => STYLE_HANDHELD,
                _ => 0,
            };
            write_u32(&mut *self.buf, base + NPAD_STYLE_TAG_OFFSET, style);
            write_u32(&mut *self.buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
            for layout in 0..LAYOUT_COUNT {
                let lifo = base + LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
                Self::write_empty_lifo(&mut self.buf[..], lifo);
            }
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

    fn write_entry_style(buf: &mut [u8], entry_idx: usize, style: u32) {
        let entry_base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        write_u32(buf, entry_base + NPAD_STYLE_TAG_OFFSET, style);
    }

    fn setup_fullkey(buf: &mut [u8], entry_idx: usize) {
        let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        write_u32(buf, base + NPAD_STYLE_TAG_OFFSET, STYLE_FULLKEY);
        write_u32(buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
        write_u32(buf, base + NPAD_DEVICE_TYPE_OFFSET, DEVICE_TYPE_FULLKEY);
        write_u64(
            buf,
            base + NPAD_SYSTEM_PROPERTIES_OFFSET,
            SYSPROP_IS_VERTICAL | SYSPROP_USE_PLUS | SYSPROP_USE_MINUS,
        );
        buf[base + NPAD_APPLET_FOOTER_OFFSET] = FOOTER_SWITCH_PRO;
    }

    fn setup_handheld(buf: &mut [u8], entry_idx: usize) {
        let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        write_u32(buf, base + NPAD_STYLE_TAG_OFFSET, STYLE_HANDHELD);
        write_u32(buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
        write_u32(
            buf,
            base + NPAD_DEVICE_TYPE_OFFSET,
            DEVICE_TYPE_HANDHELD_LEFT | DEVICE_TYPE_HANDHELD_RIGHT,
        );
        write_u64(
            buf,
            base + NPAD_SYSTEM_PROPERTIES_OFFSET,
            SYSPROP_IS_VERTICAL | SYSPROP_USE_PLUS | SYSPROP_USE_MINUS | SYSPROP_USE_DIRECTIONAL,
        );
        buf[base + NPAD_APPLET_FOOTER_OFFSET] = FOOTER_HANDHELD;
    }

    fn setup_joy_dual(buf: &mut [u8], entry_idx: usize) {
        let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        write_u32(buf, base + NPAD_STYLE_TAG_OFFSET, STYLE_JOY_DUAL);
        write_u32(buf, base + NPAD_JOY_ASSIGN_OFFSET, 0);
        write_u32(buf, base + NPAD_DEVICE_TYPE_OFFSET, DEVICE_TYPE_FULLKEY);
        write_u64(
            buf,
            base + NPAD_SYSTEM_PROPERTIES_OFFSET,
            SYSPROP_IS_VERTICAL | SYSPROP_USE_PLUS | SYSPROP_USE_MINUS | SYSPROP_USE_DIRECTIONAL,
        );
        buf[base + NPAD_APPLET_FOOTER_OFFSET] = FOOTER_SWITCH_PRO;
    }

    fn write_npad_lifo(
        buf: &mut [u8],
        entry_idx: usize,
        layout: usize,
        input: &ControllerInput,
        sampling: u64,
        attr: u32,
    ) {
        let entry_base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        let lifo = entry_base + LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
        let previous_sampling = read_u64(buf, lifo + 0x00);
        let previous_total = read_u64(buf, lifo + 0x08);
        let previous_tail = read_u64(buf, lifo + 0x10);
        let tail = (sampling % LIFO_STORAGE_COUNT as u64) as usize;
        let count = sampling.min(LIFO_STORAGE_COUNT as u64);

        write_u64(buf, lifo + 0x00, sampling);
        write_u64(buf, lifo + 0x08, LIFO_STORAGE_COUNT as u64);
        write_u64(buf, lifo + 0x10, tail as u64);
        write_u64(buf, lifo + 0x18, count);

        let needs_seed = previous_total != LIFO_STORAGE_COUNT as u64
            || previous_tail >= LIFO_STORAGE_COUNT as u64
            || previous_sampling <= 1
            || sampling.saturating_sub(previous_sampling) > 1;

        if needs_seed {
            for i in 0..LIFO_STORAGE_COUNT {
                let age = (tail + LIFO_STORAGE_COUNT - i) % LIFO_STORAGE_COUNT;
                let entry_sampling = sampling.saturating_sub(age as u64).max(1);
                Self::write_npad_lifo_entry(buf, lifo, i, input, entry_sampling, attr);
            }
        } else {
            Self::write_npad_lifo_entry(buf, lifo, tail, input, sampling, attr);
        }
    }

    fn write_standard_npad_lifos(
        buf: &mut [u8],
        entry_idx: usize,
        input: &ControllerInput,
        sampling: u64,
        attr: u32,
    ) {
        for layout in 0..LAYOUT_COUNT {
            Self::write_npad_lifo(buf, entry_idx, layout, input, sampling, attr);
        }
    }

    fn write_npad_lifo_entry(
        buf: &mut [u8],
        lifo: usize,
        index: usize,
        input: &ControllerInput,
        sampling: u64,
        attr: u32,
    ) {
        let storage = lifo + LIFO_HEADER_SIZE + index * LIFO_STORAGE_ELEM_SIZE;
        write_u64(buf, storage, sampling);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        write_u64(buf, state + 0x08, input.buttons);
        write_i32(buf, state + 0x10, input.stick_l_x);
        write_i32(buf, state + 0x14, input.stick_l_y);
        write_i32(buf, state + 0x18, input.stick_r_x);
        write_i32(buf, state + 0x1C, input.stick_r_y);
        write_u32(buf, state + 0x20, attr);
        write_u32(buf, state + 0x24, 0);
    }
}

fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap_or([0; 8]))
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

pub static CONSOLE_DOCKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static CONSOLE_MODE_DIRTY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static PLAYER1_JOY_DUAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn is_docked() -> bool {
    CONSOLE_DOCKED.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_docked(value: bool) {
    let prev = CONSOLE_DOCKED.swap(value, std::sync::atomic::Ordering::Relaxed);
    if prev != value {
        CONSOLE_MODE_DIRTY.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn take_console_mode_dirty() -> bool {
    CONSOLE_MODE_DIRTY.swap(false, std::sync::atomic::Ordering::Relaxed)
}

pub fn apply_controller_applet_style(style_set: u32) -> u32 {
    let docked = is_docked();
    let (selected_id, joy_dual) = if !docked && style_set & STYLE_HANDHELD != 0 {
        (0x20, false)
    } else if docked && style_set & STYLE_FULLKEY != 0 {
        (0, false)
    } else if style_set & STYLE_JOY_DUAL != 0 {
        (0, true)
    } else if style_set & (STYLE_FULLKEY | STYLE_JOY_LEFT | STYLE_JOY_RIGHT) != 0 {
        (0, false)
    } else if style_set & STYLE_HANDHELD != 0 {
        (0x20, false)
    } else {
        (0, false)
    };
    set_player1_joy_dual(joy_dual);
    let state = get_hid_state();
    let mut hid = state.lock();
    let cur = hid.input;
    hid.tick(cur);
    selected_id
}

pub fn set_player1_joy_dual(value: bool) {
    PLAYER1_JOY_DUAL.store(value, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_npad_lifos_expose_the_latest_sample() {
        let mut hid = HidState::new();
        let input = ControllerInput {
            buttons: NPAD_BUTTON_PLUS,
            stick_l_x: 1234,
            ..ControllerInput::default()
        };
        for _ in 0..20 {
            hid.tick(input);
        }

        let base = NPAD_OFFSET + NPAD_ENTRY_PLAYER1 * NPAD_ENTRY_SIZE;
        for layout in 0..LAYOUT_COUNT {
            let lifo = base + LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
            assert_eq!(read_u64(&hid.buf[..], lifo + 0x08), 17);
            assert_eq!(read_u64(&hid.buf[..], lifo + 0x18), 17);
            let tail = read_u64(&hid.buf[..], lifo + 0x10) as usize;
            let state = lifo + LIFO_HEADER_SIZE + tail * LIFO_STORAGE_ELEM_SIZE + 8;
            assert_eq!(read_u64(&hid.buf[..], state), hid.sampling_number);
            assert_eq!(read_u64(&hid.buf[..], state + 0x08), NPAD_BUTTON_PLUS);
            assert_eq!(
                i32::from_le_bytes(hid.buf[state + 0x10..state + 0x14].try_into().unwrap()),
                1234
            );
        }
    }

    #[test]
    fn mouse_lifo_publishes_position_deltas_and_connection() {
        let mut hid = HidState::new();
        hid.mouse = MouseInput {
            x: 100,
            y: 200,
            wheel_x: 0,
            wheel_y: 0,
            buttons: MOUSE_BUTTON_LEFT,
            connected: true,
        };
        hid.tick(ControllerInput::default());
        hid.mouse = MouseInput {
            x: 140,
            y: 190,
            wheel_x: 0,
            wheel_y: 240,
            buttons: 0,
            connected: true,
        };
        hid.tick(ControllerInput::default());

        let tail = read_u64(&hid.buf[..], MOUSE_OFFSET + 0x10) as usize;
        assert_eq!(tail, (hid.sampling_number % 17) as usize);
        let state = MOUSE_OFFSET + LIFO_HEADER_SIZE + tail * MOUSE_ELEM_SIZE + 8;
        assert_eq!(read_u64(&hid.buf[..], state), hid.sampling_number);
        let read_i32 = |off: usize| {
            i32::from_le_bytes(hid.buf[state + off..state + off + 4].try_into().unwrap())
        };
        assert_eq!(read_i32(0x08), 140);
        assert_eq!(read_i32(0x0C), 190);
        assert_eq!(read_i32(0x10), 40);
        assert_eq!(read_i32(0x14), -10);
        assert_eq!(read_i32(0x18), 240);
        assert_eq!(read_i32(0x1C), 0);
        assert_eq!(read_i32(0x20), 0);
        assert_eq!(read_i32(0x24) as u32, 1 << 1);
    }

    #[test]
    fn keyboard_lifo_publishes_key_bitmap_and_modifiers() {
        let mut hid = HidState::new();
        let mut keys = [0u8; 32];
        keys[0] = 1 << 4;
        hid.keyboard = KeyboardInput {
            modifiers: KEYBOARD_MOD_SHIFT,
            keys,
            connected: true,
        };
        hid.tick(ControllerInput::default());

        let tail = read_u64(&hid.buf[..], KEYBOARD_OFFSET + 0x10) as usize;
        let state = KEYBOARD_OFFSET + LIFO_HEADER_SIZE + tail * KEYBOARD_ELEM_SIZE + 8;
        assert_eq!(read_u64(&hid.buf[..], state), hid.sampling_number);
        let read_u32_at = |off: usize| {
            u32::from_le_bytes(hid.buf[state + off..state + off + 4].try_into().unwrap())
        };
        assert_eq!(read_u32_at(0x08), KEYBOARD_MOD_SHIFT);
        assert_eq!(read_u32_at(0x0C), 1);
        assert_eq!(hid.buf[state + 0x10], 1 << 4);
        assert_eq!(&hid.buf[state + 0x11..state + 0x30], &[0u8; 31][..]);
    }

    #[test]
    fn mapped_host_binding_requires_its_owner_to_unbind() {
        let mut hid = HidState::new();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        let ptr = mapped.as_mut_ptr() as usize;
        unsafe {
            hid.bind_mapped_host(ptr as *mut u8);
        }
        hid.shmem_va = Some(0x1000);

        assert!(!hid.unbind_mapped_host(ptr.wrapping_add(1)));
        assert_eq!(hid.mapped_host_ptr, ptr);
        assert_eq!(hid.shmem_va, Some(0x1000));
        assert!(hid.unbind_mapped_host(ptr));
        assert_eq!(hid.mapped_host_ptr, 0);
        assert!(hid.shmem_va.is_none());
    }

    #[test]
    fn supported_style_selection_republishes_a_compatible_npad_without_changing_console_mode() {
        set_docked(true);
        set_player1_joy_dual(false);

        let state = get_hid_state();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        let mapped_ptr = mapped.as_mut_ptr() as usize;
        {
            let mut hid = state.lock();
            let input = hid.input;
            hid.tick(input);
            unsafe {
                hid.bind_mapped_host(mapped_ptr as *mut u8);
            }
        }

        let player_style_offset = NPAD_OFFSET + NPAD_ENTRY_PLAYER1 * NPAD_ENTRY_SIZE;
        let handheld_style_offset = NPAD_OFFSET + NPAD_ENTRY_HANDHELD * NPAD_ENTRY_SIZE;
        let read_player_style = |buf: &[u8]| {
            u32::from_le_bytes(
                buf[player_style_offset..player_style_offset + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let read_handheld_style = |buf: &[u8]| {
            u32::from_le_bytes(
                buf[handheld_style_offset..handheld_style_offset + 4]
                    .try_into()
                    .unwrap(),
            )
        };
        let initial_style = read_player_style(&mapped[..]);

        let initial_selection = apply_controller_applet_style(0x1f);
        let docked_style = read_player_style(&mapped[..]);
        let remains_docked = is_docked();

        set_docked(false);
        let handheld_selection = apply_controller_applet_style(0x1f);
        let handheld_style = read_handheld_style(&mapped[..]);
        let remains_handheld = !is_docked();

        let joy_dual_selection = apply_controller_applet_style(STYLE_HANDHELD | STYLE_JOY_DUAL);
        let joy_dual_style = read_handheld_style(&mapped[..]);
        let joy_dual_remains_handheld = !is_docked();

        let unbound = {
            let mut hid = state.lock();
            hid.unbind_mapped_host(mapped_ptr)
        };
        set_docked(true);
        set_player1_joy_dual(false);
        {
            let mut hid = state.lock();
            let input = hid.input;
            hid.tick(input);
        }

        assert_eq!(initial_selection, 0);
        assert_eq!(initial_style, STYLE_FULLKEY);
        assert_eq!(docked_style, STYLE_FULLKEY);
        assert!(remains_docked);
        assert_eq!(handheld_selection, 0x20);
        assert_eq!(handheld_style, STYLE_HANDHELD);
        assert!(remains_handheld);
        assert_eq!(joy_dual_selection, 0x20);
        assert_eq!(joy_dual_style, STYLE_HANDHELD);
        assert!(joy_dual_remains_handheld);
        assert!(unbound);
    }
}

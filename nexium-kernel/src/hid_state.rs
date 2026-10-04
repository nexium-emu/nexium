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
const DEVICE_TYPE_JOYCON_LEFT: u32 = 1 << 4;
const DEVICE_TYPE_JOYCON_RIGHT: u32 = 1 << 5;
const NPAD_JOY_ASSIGNMENT_SINGLE: u32 = 1;

const SYSPROP_IS_VERTICAL: u64 = 1 << 11;
const SYSPROP_IS_HORIZONTAL: u64 = 1 << 12;
const SYSPROP_USE_PLUS: u64 = 1 << 13;
const SYSPROP_USE_MINUS: u64 = 1 << 14;
const SYSPROP_USE_DIRECTIONAL: u64 = 1 << 15;

const FOOTER_SWITCH_PRO: u8 = 12;
const FOOTER_HANDHELD: u8 = 4;
const FOOTER_JOY_LEFT_HORIZONTAL: u8 = 8;
const FOOTER_JOY_LEFT_VERTICAL: u8 = 9;
const FOOTER_JOY_RIGHT_HORIZONTAL: u8 = 10;
const FOOTER_JOY_RIGHT_VERTICAL: u8 = 11;

const LAYOUT_BASE_OFFSET: usize = 0x28;
const LAYOUT_STRIDE: usize = 0x350;
const LAYOUT_COUNT: usize = 7;

const LIFO_HEADER_SIZE: usize = 0x20;
const LIFO_STORAGE_ELEM_SIZE: usize = 0x30;
const LIFO_STORAGE_COUNT: usize = 17;

const SIXAXIS_BASE_OFFSET: usize = LAYOUT_BASE_OFFSET + LAYOUT_COUNT * LAYOUT_STRIDE;
const SIXAXIS_STRIDE: usize = 0x708;
const SIXAXIS_COUNT: usize = 6;
const SIXAXIS_ELEM_SIZE: usize = 0x68;
const SIXAXIS_ATTR_IS_CONNECTED: u32 = 1 << 0;
const SIXAXIS_MAX_COUNT: u64 = 16;
const SIXAXIS_MIN_DELTA_NS: u64 = 1_000;
const SIXAXIS_SYNTH_DUE: std::time::Duration = std::time::Duration::from_millis(2);
const SIXAXIS_SYNTH_MAX_DELTA: std::time::Duration = std::time::Duration::from_millis(100);
const SIXAXIS_SYNTH_FIRST_DELTA_NS: u64 = 16_666_667;
const NPAD_SIXAXIS_PROPERTIES_OFFSET: usize = 0x43F0;
const SIXAXIS_PROPERTY_NEWLY_ASSIGNED: u8 = 1 << 0;
const NEWLY_ASSIGNED_SETTLE_SAMPLES: u64 = 16;

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

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
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

pub const KEYBOARD_MOD_CONTROL: u32 = 1 << 0;
pub const KEYBOARD_MOD_SHIFT: u32 = 1 << 1;
pub const KEYBOARD_MOD_LEFT_ALT: u32 = 1 << 2;
pub const KEYBOARD_MOD_RIGHT_ALT: u32 = 1 << 3;

const MOUSE_ELEM_SIZE: usize = 0x30;
const KEYBOARD_ELEM_SIZE: usize = 0x38;
const TOUCH_ELEM_SIZE: usize = 0x298;

const TOUCH_ATTR_START: u32 = 1 << 0;
const TOUCH_ATTR_END: u32 = 1 << 1;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseInput {
    pub x: i32,
    pub y: i32,
    pub wheel_x: i32,
    pub wheel_y: i32,
    pub buttons: u32,
    pub connected: bool,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct TouchInput {
    pub x: u32,
    pub y: u32,
    pub pressed: bool,
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
    pub touch: TouchInput,
    last_written_mouse: MouseInput,
    last_written_touch: TouchInput,
    pub sampling_number: u64,
    pub shmem_va: Option<u64>,
    last_tick: Option<std::time::Instant>,
    last_trace: Option<HidTraceSample>,
    motion_scratch: [Vec<crate::hid_motion::SixAxisFrame>; 3],
    sixaxis_last_write: [[Option<std::time::Instant>; SIXAXIS_COUNT]; 2],
    sixaxis_generation_seen: u64,
    injected_motion_at: Option<std::time::Instant>,
    injected_motion_time_ns: u64,
    injected_sources: Option<[bool; 3]>,
    injected_recenter_seen: Option<u32>,
    sixaxis_passthrough: Vec<u32>,
    newly_assigned_pending: Vec<(u32, u64)>,
    motion_trace_at: Option<std::time::Instant>,
    motion_trace_frames: [u32; 3],
    motion_trace_entries: u32,
    motion_trace_last: Option<crate::hid_motion::SixAxisFrame>,
}

#[derive(PartialEq, Eq)]
struct HidTraceSample {
    entry: usize,
    mapped_host_ptr: usize,
    requested_buttons: u64,
    injected: Option<ControllerInput>,
    npad_buttons: [u64; LAYOUT_COUNT],
    mouse_buttons: u32,
    mouse_x: i32,
    mouse_y: i32,
    mouse_wheel_x: i32,
    mouse_wheel_y: i32,
    keyboard_modifiers: u32,
    keyboard_keys: [u8; 32],
    touch_count: u32,
}

fn exclusive_input_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_HID_INJECT_EXCLUSIVE").as_deref() == Ok("1"))
}

fn hid_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_HID_TRACE").as_deref() == Ok("1"))
}

fn hid_motion_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_HID_MOTION_TRACE").as_deref() == Ok("1"))
}

fn selected_controller_input(
    host: ControllerInput,
    injected: Option<ControllerInput>,
    exclusive: bool,
) -> ControllerInput {
    injected.unwrap_or_else(|| if exclusive { ControllerInput::default() } else { host })
}

fn selected_auxiliary_input(
    mouse: MouseInput,
    keyboard: KeyboardInput,
    touch: TouchInput,
    exclusive: bool,
) -> (MouseInput, KeyboardInput, TouchInput) {
    if exclusive {
        (MouseInput::default(), KeyboardInput::default(), TouchInput::default())
    } else {
        (mouse, keyboard, touch)
    }
}

fn injected_input() -> Option<ControllerInput> {
    use std::sync::{Mutex, OnceLock};
    static STATE: OnceLock<Option<(std::path::PathBuf, Mutex<(std::time::Instant, Option<ControllerInput>, &'static str)>)>> = OnceLock::new();
    let (path, cache) = STATE
        .get_or_init(|| {
            std::env::var_os("NEXIUM_HID_INJECT").map(|value| {
                let stale = std::time::Instant::now() - std::time::Duration::from_secs(1);
                (std::path::PathBuf::from(value), Mutex::new((stale, None, "unread")))
            })
        })
        .as_ref()?;
    let mut cache = cache.lock().ok()?;
    if cache.0.elapsed() >= std::time::Duration::from_millis(25) {
        cache.0 = std::time::Instant::now();
        let (input, status) = match std::fs::read_to_string(path) {
            Ok(text) => match parse_injected_input(&text) {
                Some(input) => (Some(input), "active"),
                None => (None, "neutral-or-invalid"),
            },
            Err(_) => (None, "unreadable"),
        };
        if hid_trace_enabled() && (cache.1 != input || cache.2 != status) {
            log::info!("[hid-inject] status={} input={:?} exclusive={}", status, input, exclusive_input_enabled());
        }
        cache.1 = input;
        cache.2 = status;
    }
    cache.1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InjectedMotion {
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
    pub sources: [bool; 3],
}

fn injected_motion() -> (Option<InjectedMotion>, Option<u32>) {
    use std::sync::{Mutex, OnceLock};
    type MotionCache = Mutex<(std::time::Instant, (Option<InjectedMotion>, Option<u32>))>;
    static STATE: OnceLock<Option<(std::path::PathBuf, MotionCache)>> = OnceLock::new();
    let Some((path, cache)) = STATE
        .get_or_init(|| {
            std::env::var_os("NEXIUM_HID_INJECT").map(|value| {
                let stale = std::time::Instant::now() - std::time::Duration::from_secs(1);
                (std::path::PathBuf::from(value), Mutex::new((stale, (None, None))))
            })
        })
        .as_ref()
    else {
        return (None, None);
    };
    let Ok(mut cache) = cache.lock() else {
        return (None, None);
    };
    if cache.0.elapsed() >= std::time::Duration::from_millis(25) {
        cache.0 = std::time::Instant::now();
        cache.1 = std::fs::read_to_string(path).map(|text| parse_injected_motion(&text)).unwrap_or((None, None));
    }
    cache.1
}

fn parse_injected_motion(text: &str) -> (Option<InjectedMotion>, Option<u32>) {
    let mut motion = InjectedMotion { accel: [0.0, 0.0, -1.0], gyro: [0.0; 3], sources: [true; 3] };
    let mut present = false;
    let mut off = false;
    let mut invalid = false;
    let mut recenter = None;
    for token in text.split_whitespace() {
        let Some((key, value)) = token.split_once('=') else {
            continue;
        };
        let slot = match key {
            "ax" => &mut motion.accel[0],
            "ay" => &mut motion.accel[1],
            "az" => &mut motion.accel[2],
            "gx" => &mut motion.gyro[0],
            "gy" => &mut motion.gyro[1],
            "gz" => &mut motion.gyro[2],
            "motion" => {
                match value {
                    "rest" => present = true,
                    "off" => off = true,
                    _ => {}
                }
                continue;
            }
            "msrc" => {
                match value {
                    "all" => motion.sources = [true, true, true],
                    "p" => motion.sources = [true, false, false],
                    "l" => motion.sources = [false, true, false],
                    "r" => motion.sources = [false, false, true],
                    "lr" => motion.sources = [false, true, true],
                    _ => invalid = true,
                }
                continue;
            }
            "recenter" => {
                recenter = value.parse::<u32>().ok();
                continue;
            }
            _ => continue,
        };
        match value.parse::<f32>() {
            Ok(parsed) if parsed.is_finite() => {
                *slot = parsed;
                present = true;
            }
            _ => invalid = true,
        }
    }
    ((present && !off && !invalid).then_some(motion), recenter)
}

fn parse_injected_input(text: &str) -> Option<ControllerInput> {
    let mut input = ControllerInput::default();
    let mut any = false;
    for token in text.split_whitespace() {
        let (key, value) = token.split_once('=')?;
        any = true;
        let axis = |value: &str| value.parse::<f32>().ok().map(|v| (v.clamp(-1.0, 1.0) * 32767.0) as i32);
        match key {
            "lx" => input.stick_l_x = axis(value)?,
            "ly" => input.stick_l_y = axis(value)?,
            "rx" => input.stick_r_x = axis(value)?,
            "ry" => input.stick_r_y = axis(value)?,
            "buttons" => {
                for name in value.split(',').filter(|name| !name.is_empty()) {
                    input.buttons |= match name.to_ascii_uppercase().as_str() {
                        "A" => NPAD_BUTTON_A,
                        "B" => NPAD_BUTTON_B,
                        "X" => NPAD_BUTTON_X,
                        "Y" => NPAD_BUTTON_Y,
                        "L" => NPAD_BUTTON_L,
                        "R" => NPAD_BUTTON_R,
                        "ZL" => NPAD_BUTTON_ZL,
                        "ZR" => NPAD_BUTTON_ZR,
                        "PLUS" => NPAD_BUTTON_PLUS,
                        "MINUS" => NPAD_BUTTON_MINUS,
                        "DUP" => NPAD_BUTTON_UP,
                        "DDOWN" => NPAD_BUTTON_DOWN,
                        "DLEFT" => NPAD_BUTTON_LEFT,
                        "DRIGHT" => NPAD_BUTTON_RIGHT,
                        "LS" => NPAD_BUTTON_STICK_L,
                        "RS" => NPAD_BUTTON_STICK_R,
                        _ => 0,
                    };
                }
            }
            _ => {}
        }
    }
    let active = input.buttons != 0
        || input.stick_l_x != 0
        || input.stick_l_y != 0
        || input.stick_r_x != 0
        || input.stick_r_y != 0;
    (any && active).then_some(input)
}

impl HidState {
    pub fn new() -> Self {
        let mut s = Self {
            buf: Box::new([0u8; HID_SHMEM_SIZE]),
            mapped_host_ptr: 0,
            input: ControllerInput::default(),
            mouse: MouseInput::default(),
            keyboard: KeyboardInput::default(),
            touch: TouchInput::default(),
            last_written_mouse: MouseInput::default(),
            last_written_touch: TouchInput::default(),
            sampling_number: 0,
            shmem_va: None,
            last_tick: None,
            last_trace: None,
            motion_scratch: Default::default(),
            sixaxis_last_write: [[None; SIXAXIS_COUNT]; 2],
            sixaxis_generation_seen: 0,
            injected_motion_at: None,
            injected_motion_time_ns: 0,
            injected_sources: None,
            injected_recenter_seen: None,
            sixaxis_passthrough: Vec::new(),
            newly_assigned_pending: Vec::new(),
            motion_trace_at: None,
            motion_trace_frames: [0; 3],
            motion_trace_entries: 0,
            motion_trace_last: None,
        };
        s.init_metadata();
        s.tick(ControllerInput::default());
        s
    }

    pub fn update_devices(
        &mut self,
        mouse: MouseInput,
        keyboard: KeyboardInput,
        touch: TouchInput,
    ) {
        let (mouse, keyboard, touch) = selected_auxiliary_input(
            mouse, keyboard, touch, exclusive_input_enabled(),
        );
        let force = mouse.buttons != self.mouse.buttons
            || mouse.connected != self.mouse.connected
            || keyboard != self.keyboard
            || touch.pressed != self.touch.pressed;
        self.mouse = mouse;
        self.keyboard = keyboard;
        self.touch = touch;
        if force {
            self.last_tick = Some(std::time::Instant::now());
            let input = self.input;
            self.tick(input);
        }
    }

    pub fn maybe_tick(&mut self, input: ControllerInput) {
        const VSYNC: std::time::Duration = std::time::Duration::from_nanos(16_666_667);
        let now = std::time::Instant::now();
        let input = selected_controller_input(input, injected_input(), exclusive_input_enabled());
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
        self.sixaxis_passthrough.clear();
        self.newly_assigned_pending.clear();
        for entry in 0..=NPAD_ENTRY_OTHER {
            let properties = NPAD_OFFSET + entry * NPAD_ENTRY_SIZE + NPAD_SIXAXIS_PROPERTIES_OFFSET;
            self.buf[properties..properties + SIXAXIS_COUNT].fill(0);
        }
        true
    }

    pub fn size(&self) -> usize {
        HID_SHMEM_SIZE
    }

    pub fn set_sixaxis_passthrough(&mut self, handle: u32, enabled: bool) {
        self.sixaxis_passthrough.retain(|existing| *existing != handle);
        if enabled {
            self.sixaxis_passthrough.push(handle);
        }
    }

    pub fn request_sixaxis_newly_assigned(&mut self) {
        for &handle in &self.sixaxis_passthrough {
            let Some((entry, index)) = sixaxis_handle_slot(handle) else {
                continue;
            };
            let issued = read_u64(&self.buf[..], sixaxis_lifo_offset(entry, index));
            let target = issued.saturating_add(NEWLY_ASSIGNED_SETTLE_SAMPLES);
            self.newly_assigned_pending.retain(|(pending, _)| *pending != handle);
            self.newly_assigned_pending.push((handle, target));
        }
    }

    pub fn reset_sixaxis_newly_assigned(&mut self, handle: u32) -> bool {
        let Some((offset, _)) = sixaxis_property_offset(handle) else {
            return false;
        };
        let was_set = self.buf[offset] & SIXAXIS_PROPERTY_NEWLY_ASSIGNED != 0;
        self.buf[offset] &= !SIXAXIS_PROPERTY_NEWLY_ASSIGNED;
        if self.mapped_host_ptr != 0 {
            unsafe {
                (self.mapped_host_ptr as *mut u8).add(offset).write(self.buf[offset]);
            }
        }
        was_set
    }

    pub fn update_input(&mut self, input: ControllerInput) {
        self.maybe_tick(input);
    }

    pub fn publish_motion(&mut self) {
        let entry = p1_active_entry();
        let now = std::time::Instant::now();
        let touched = self.write_sixaxis_lifos(entry, now, false, exclusive_input_enabled());
        self.publish_newly_assigned();
        self.copy_sixaxis_to_mapped(entry, touched);
    }

    fn copy_sixaxis_to_mapped(&self, entry: usize, touched: u8) {
        if self.mapped_host_ptr == 0 {
            return;
        }
        let mapped = self.mapped_host_ptr as *mut u8;
        for index in 0..SIXAXIS_COUNT {
            if touched & (1 << index) == 0 {
                continue;
            }
            let lifo = sixaxis_lifo_offset(entry, index);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf.as_ptr().add(lifo + LIFO_HEADER_SIZE),
                    mapped.add(lifo + LIFO_HEADER_SIZE),
                    SIXAXIS_STRIDE - LIFO_HEADER_SIZE,
                );
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            unsafe {
                std::ptr::copy_nonoverlapping(self.buf.as_ptr().add(lifo), mapped.add(lifo), LIFO_HEADER_SIZE);
            }
        }
        let properties = sixaxis_property_at(entry, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(self.buf.as_ptr().add(properties), mapped.add(properties), SIXAXIS_COUNT);
        }
    }

    pub fn tick(&mut self, input: ControllerInput) {
        self.tick_with_source(input, injected_input(), exclusive_input_enabled());
    }

    fn tick_with_source(
        &mut self,
        requested: ControllerInput,
        injected: Option<ControllerInput>,
        exclusive: bool,
    ) {
        let input = selected_controller_input(requested, injected, exclusive);
        self.input = input;
        self.sampling_number = self.sampling_number.wrapping_add(1);
        let sampling = self.sampling_number;
        let docked = CONSOLE_DOCKED.load(std::sync::atomic::Ordering::Relaxed);
        let player1_joy_dual = PLAYER1_JOY_DUAL.load(std::sync::atomic::Ordering::Relaxed);
        let player1_joy_single = player1_joy_single();
        let active_entry = active_entry_for(docked, player1_joy_dual, player1_joy_single);
        let presented = presented_sticks(
            input,
            current_p1_presentation(docked, player1_joy_dual, player1_joy_single),
            crate::hid_motion::input_kind(),
        );
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
                &presented,
                sampling,
                attr,
            );
        } else if let Some(right) = player1_joy_single {
            let attr = if right {
                ATTR_IS_CONNECTED | ATTR_RIGHT_CONNECTED
            } else {
                ATTR_IS_CONNECTED | ATTR_LEFT_CONNECTED
            };
            Self::setup_joy_single(&mut self.buf[..], NPAD_ENTRY_PLAYER1, right);
            Self::write_standard_npad_lifos(
                &mut self.buf[..],
                NPAD_ENTRY_PLAYER1,
                &presented,
                sampling,
                attr,
            );
        } else if docked {
            Self::setup_fullkey(&mut self.buf[..], NPAD_ENTRY_PLAYER1);
            Self::write_standard_npad_lifos(
                &mut self.buf[..],
                NPAD_ENTRY_PLAYER1,
                &presented,
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
                &presented,
                sampling,
                attr,
            );
        }
        self.write_sixaxis_lifos(active_entry, std::time::Instant::now(), true, exclusive);
        self.publish_newly_assigned();

        let (mouse, keyboard, touch) = selected_auxiliary_input(self.mouse, self.keyboard, self.touch, exclusive);
        let previous_mouse = if exclusive { MouseInput::default() } else { self.last_written_mouse };
        Self::write_mouse_lifo(&mut self.buf[..], &mouse, &previous_mouse, sampling);
        self.last_written_mouse = mouse;
        Self::write_keyboard_lifo(&mut self.buf[..], &keyboard, sampling);
        let previous_touch = if exclusive { TouchInput::default() } else { self.last_written_touch };
        Self::write_touch_lifo(&mut self.buf[..], &touch, &previous_touch, sampling);
        self.last_written_touch = touch;

        if self.mapped_host_ptr != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buf.as_ptr(),
                    self.mapped_host_ptr as *mut u8,
                    HID_SHMEM_SIZE,
                );
            }
        }
        if hid_trace_enabled() {
            self.trace_published_input(active_entry, requested, injected, exclusive);
        }
    }

    fn trace_published_input(
        &mut self,
        entry: usize,
        requested: ControllerInput,
        injected: Option<ControllerInput>,
        exclusive: bool,
    ) {
        let state_at = |lifo: usize, stride: usize| {
            lifo + LIFO_HEADER_SIZE + read_u64(&self.buf[..], lifo + 0x10) as usize * stride + 8
        };
        let read_word = |offset: usize| u32::from_le_bytes(self.buf[offset..offset + 4].try_into().unwrap());
        let mouse = state_at(MOUSE_OFFSET, MOUSE_ELEM_SIZE);
        let keyboard = state_at(KEYBOARD_OFFSET, KEYBOARD_ELEM_SIZE);
        let touch = state_at(TOUCH_OFFSET, TOUCH_ELEM_SIZE);
        let sample = HidTraceSample {
            entry,
            mapped_host_ptr: self.mapped_host_ptr,
            requested_buttons: requested.buttons,
            injected,
            npad_buttons: std::array::from_fn(|layout| {
                let lifo = NPAD_OFFSET + entry * NPAD_ENTRY_SIZE + LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
                read_u64(&self.buf[..], state_at(lifo, LIFO_STORAGE_ELEM_SIZE) + 8)
            }),
            mouse_buttons: read_word(mouse + 0x20),
            mouse_x: read_word(mouse + 0x08) as i32,
            mouse_y: read_word(mouse + 0x0C) as i32,
            mouse_wheel_x: read_word(mouse + 0x18) as i32,
            mouse_wheel_y: read_word(mouse + 0x1C) as i32,
            keyboard_modifiers: read_word(keyboard + 0x08),
            keyboard_keys: self.buf[keyboard + 0x10..keyboard + 0x30].try_into().unwrap(),
            touch_count: read_word(touch + 0x08),
        };
        if self.last_trace.as_ref() == Some(&sample) {
            return;
        }
        log::info!(
            "[hid-publish] sample={} entry={} requested={:#x} injected={:?} npad={:x?} mouse={:#x} mouse_x={} mouse_y={} wheel_x={} wheel_y={} keyboard_mods={:#x} keyboard={:02x?} touches={} exclusive={} mapped={:#x}",
            self.sampling_number, entry, sample.requested_buttons, sample.injected,
            sample.npad_buttons, sample.mouse_buttons, sample.mouse_x, sample.mouse_y,
            sample.mouse_wheel_x, sample.mouse_wheel_y, sample.keyboard_modifiers,
            sample.keyboard_keys, sample.touch_count, exclusive, sample.mapped_host_ptr,
        );
        self.last_trace = Some(sample);
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
        write_u64(buf, storage, sampling << 1);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        write_i32(buf, state + 0x08, mouse.x);
        write_i32(buf, state + 0x0C, mouse.y);
        let delta = |current: i32, previous_value: i32| {
            if mouse.connected && previous.connected { current.wrapping_sub(previous_value) } else { 0 }
        };
        write_i32(buf, state + 0x10, delta(mouse.x, previous.x));
        write_i32(buf, state + 0x14, delta(mouse.y, previous.y));
        write_i32(buf, state + 0x18, delta(mouse.wheel_x, previous.wheel_x));
        write_i32(buf, state + 0x1C, delta(mouse.wheel_y, previous.wheel_y));
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

    fn write_touch_lifo(buf: &mut [u8], touch: &TouchInput, previous: &TouchInput, sampling: u64) {
        let tail = Self::write_device_lifo_header(buf, TOUCH_OFFSET, sampling);
        let storage = TOUCH_OFFSET + LIFO_HEADER_SIZE + tail * TOUCH_ELEM_SIZE;
        write_u64(buf, storage, sampling << 1);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        let ending = !touch.pressed && previous.pressed;
        let entry_count = if touch.pressed || ending { 1 } else { 0 };
        write_i32(buf, state + 0x08, entry_count);
        write_u32(buf, state + 0x0C, 0);
        let finger = state + 0x10;
        if entry_count == 0 {
            buf[finger..finger + 0x28].fill(0);
            return;
        }
        let attribute = if ending {
            TOUCH_ATTR_END
        } else if !previous.pressed {
            TOUCH_ATTR_START
        } else {
            0
        };
        let (x, y) = if ending {
            (previous.x, previous.y)
        } else {
            (touch.x, touch.y)
        };
        write_u64(buf, finger + 0x00, sampling);
        write_u32(buf, finger + 0x08, attribute);
        write_u32(buf, finger + 0x0C, 0);
        write_u32(buf, finger + 0x10, x.min(1279));
        write_u32(buf, finger + 0x14, y.min(719));
        write_u32(buf, finger + 0x18, 15);
        write_u32(buf, finger + 0x1C, 15);
        write_u32(buf, finger + 0x20, 0);
        write_u32(buf, finger + 0x24, 0);
    }

    fn write_keyboard_lifo(buf: &mut [u8], keyboard: &KeyboardInput, sampling: u64) {
        let tail = Self::write_device_lifo_header(buf, KEYBOARD_OFFSET, sampling);
        let storage = KEYBOARD_OFFSET + LIFO_HEADER_SIZE + tail * KEYBOARD_ELEM_SIZE;
        write_u64(buf, storage, sampling << 1);
        let state = storage + 8;
        write_u64(buf, state + 0x00, sampling);
        write_u64(buf, state + 0x08, u64::from(keyboard.modifiers));
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

    fn setup_joy_single(buf: &mut [u8], entry_idx: usize, right: bool) {
        let base = NPAD_OFFSET + entry_idx * NPAD_ENTRY_SIZE;
        let horizontal = NPAD_JOY_HOLD_HORIZONTAL.load(std::sync::atomic::Ordering::Relaxed);
        let (style, device, footer, button) = match (right, horizontal) {
            (true, true) => (STYLE_JOY_RIGHT, DEVICE_TYPE_JOYCON_RIGHT, FOOTER_JOY_RIGHT_HORIZONTAL, SYSPROP_USE_PLUS),
            (true, false) => (STYLE_JOY_RIGHT, DEVICE_TYPE_JOYCON_RIGHT, FOOTER_JOY_RIGHT_VERTICAL, SYSPROP_USE_PLUS),
            (false, true) => (STYLE_JOY_LEFT, DEVICE_TYPE_JOYCON_LEFT, FOOTER_JOY_LEFT_HORIZONTAL, SYSPROP_USE_MINUS),
            (false, false) => (STYLE_JOY_LEFT, DEVICE_TYPE_JOYCON_LEFT, FOOTER_JOY_LEFT_VERTICAL, SYSPROP_USE_MINUS),
        };
        let hold = if horizontal { SYSPROP_IS_HORIZONTAL } else { SYSPROP_IS_VERTICAL };
        write_u32(buf, base + NPAD_STYLE_TAG_OFFSET, style);
        write_u32(buf, base + NPAD_JOY_ASSIGN_OFFSET, NPAD_JOY_ASSIGNMENT_SINGLE);
        write_u32(buf, base + NPAD_DEVICE_TYPE_OFFSET, device);
        write_u64(buf, base + NPAD_SYSTEM_PROPERTIES_OFFSET, hold | button);
        buf[base + NPAD_APPLET_FOOTER_OFFSET] = footer;
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

    fn inject_motion_samples(&mut self, now: std::time::Instant) {
        use crate::hid_motion::{MotionSample, MotionSource, DEFAULT_PERIOD_NS};
        let (motion, recenter) = injected_motion();
        if let Some(request) = recenter {
            if self.injected_recenter_seen != Some(request) {
                self.injected_recenter_seen = Some(request);
                crate::hid_motion::recenter_all();
                self.request_sixaxis_newly_assigned();
            }
        }
        let Some(motion) = motion else {
            if self.injected_sources.take().is_some() {
                crate::hid_motion::set_injection(None);
            }
            self.injected_motion_at = None;
            return;
        };
        if self.injected_sources != Some(motion.sources) {
            crate::hid_motion::set_injection(Some(motion.sources));
            self.injected_sources = Some(motion.sources);
        }
        let period = std::time::Duration::from_nanos(DEFAULT_PERIOD_NS);
        let last = *self.injected_motion_at.get_or_insert(now.checked_sub(period).unwrap_or(now));
        let steps = (now.saturating_duration_since(last).as_nanos() / period.as_nanos()).min(64) as u32;
        if steps == 0 {
            return;
        }
        self.injected_motion_at = Some(last + period * steps);
        let samples: Vec<MotionSample> = (0..steps)
            .map(|_| {
                self.injected_motion_time_ns += DEFAULT_PERIOD_NS;
                MotionSample { sensor_time_ns: self.injected_motion_time_ns, accel: motion.accel, gyro: motion.gyro }
            })
            .collect();
        for source in MotionSource::ALL {
            if motion.sources[source.index()] {
                crate::hid_motion::push_injected_samples(source, &samples);
            }
        }
    }

    fn write_sixaxis_lifos(
        &mut self,
        entry: usize,
        now: std::time::Instant,
        synthesize: bool,
        exclusive: bool,
    ) -> u8 {
        self.inject_motion_samples(now);
        let mut frames = std::mem::take(&mut self.motion_scratch);
        let mut snapshot = crate::hid_motion::drain_frames(&mut frames);
        if exclusive && !snapshot.injecting {
            snapshot = crate::hid_motion::MotionSnapshot::default();
            for source_frames in frames.iter_mut() {
                source_frames.clear();
            }
        }
        self.note_motion_generation(snapshot.generation);
        let touched = self.write_sixaxis_from(entry, now, synthesize, &snapshot, &frames);
        if hid_motion_trace_enabled() {
            self.trace_motion(entry, now, &snapshot, &frames);
        }
        self.motion_scratch = frames;
        touched
    }

    fn note_motion_generation(&mut self, generation: u64) {
        if generation != self.sixaxis_generation_seen {
            self.sixaxis_generation_seen = generation;
            self.request_sixaxis_newly_assigned();
        }
    }

    fn write_sixaxis_from(
        &mut self,
        entry: usize,
        now: std::time::Instant,
        synthesize: bool,
        snapshot: &crate::hid_motion::MotionSnapshot,
        frames: &[Vec<crate::hid_motion::SixAxisFrame>; 3],
    ) -> u8 {
        use crate::hid_motion::SixAxisFrame;
        let slot = usize::from(entry != NPAD_ENTRY_PLAYER1);
        let mut touched = 0u8;
        for (index, route) in sixaxis_route(snapshot.connected).into_iter().enumerate() {
            let lifo = sixaxis_lifo_offset(entry, index);
            let last_write = self.sixaxis_last_write[slot][index];
            let live = route.map(|source| frames[source.index()].as_slice()).filter(|live| !live.is_empty());
            if let Some(live) = live {
                for frame in compact_frames(live, SIXAXIS_MAX_COUNT as usize).iter() {
                    append_sixaxis_entry(&mut self.buf[..], lifo, frame);
                    self.motion_trace_entries = self.motion_trace_entries.wrapping_add(1);
                    self.motion_trace_last = Some(*frame);
                }
                self.sixaxis_last_write[slot][index] = Some(now);
                touched |= 1 << index;
                continue;
            }
            let due = last_write.map_or(true, |at| now.saturating_duration_since(at) >= SIXAXIS_SYNTH_DUE);
            if !synthesize || !due {
                continue;
            }
            let base = match route {
                Some(source) if snapshot.stale[source.index()] => {
                    snapshot.last[source.index()].map_or(SixAxisFrame::REST, |frame| frame.hold())
                }
                Some(_) => continue,
                None => SixAxisFrame::REST,
            };
            let delta_time_ns = last_write.map_or(SIXAXIS_SYNTH_FIRST_DELTA_NS, |at| {
                let elapsed = now.saturating_duration_since(at);
                elapsed.clamp(SIXAXIS_SYNTH_DUE, SIXAXIS_SYNTH_MAX_DELTA).as_nanos() as u64
            });
            let frame = SixAxisFrame { delta_time_ns, ..base };
            append_sixaxis_entry(&mut self.buf[..], lifo, &frame);
            self.motion_trace_entries = self.motion_trace_entries.wrapping_add(1);
            self.motion_trace_last = Some(frame);
            self.sixaxis_last_write[slot][index] = Some(now);
            touched |= 1 << index;
        }
        touched
    }

    fn trace_motion(
        &mut self,
        entry: usize,
        now: std::time::Instant,
        snapshot: &crate::hid_motion::MotionSnapshot,
        frames: &[Vec<crate::hid_motion::SixAxisFrame>; 3],
    ) {
        for (count, source_frames) in self.motion_trace_frames.iter_mut().zip(frames.iter()) {
            *count = count.wrapping_add(source_frames.len() as u32);
        }
        let started = *self.motion_trace_at.get_or_insert(now);
        let elapsed = now.saturating_duration_since(started);
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let seconds = elapsed.as_secs_f32();
        let rates = self.motion_trace_frames.map(|count| (count as f32 / seconds).round() as u32);
        let entries = (self.motion_trace_entries as f32 / seconds).round() as u32;
        let rings: [u64; SIXAXIS_COUNT] =
            std::array::from_fn(|index| read_u64(&self.buf[..], sixaxis_lifo_offset(entry, index)));
        let (dt, gyro, accel, dir_y) = self.motion_trace_last.map_or((0, [0.0; 3], [0.0; 3], [0.0; 3]), |last| {
            (last.delta_time_ns, last.gyro, last.accel, last.direction[1])
        });
        log::info!(
            "[hid-motion] connected={:?} injecting={} calib={:?} frames/s={:?} entries/s={} entry={} rings={:?} last dt={} gyro={:?} accel={:?} dir_y={:?}",
            snapshot.connected, snapshot.injecting, crate::hid_motion::calibration_states(), rates, entries,
            entry, rings, dt, gyro, accel, dir_y,
        );
        self.motion_trace_at = Some(now);
        self.motion_trace_frames = [0; 3];
        self.motion_trace_entries = 0;
    }

    fn publish_newly_assigned(&mut self) {
        let buf = &mut self.buf;
        self.newly_assigned_pending.retain(|&(handle, target)| {
            let Some((entry, index)) = sixaxis_handle_slot(handle) else {
                return false;
            };
            if read_u64(&buf[..], sixaxis_lifo_offset(entry, index)) < target {
                return true;
            }
            buf[sixaxis_property_at(entry, index)] |= SIXAXIS_PROPERTY_NEWLY_ASSIGNED;
            log::info!("[hid] six-axis device newly assigned handle={:#x}", handle);
            false
        });
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
        write_u64(buf, storage, sampling << 1);
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

pub(crate) fn sixaxis_lifo_offset(entry: usize, index: usize) -> usize {
    NPAD_OFFSET + entry * NPAD_ENTRY_SIZE + SIXAXIS_BASE_OFFSET + index * SIXAXIS_STRIDE
}

fn sixaxis_property_at(entry: usize, index: usize) -> usize {
    NPAD_OFFSET + entry * NPAD_ENTRY_SIZE + NPAD_SIXAXIS_PROPERTIES_OFFSET + index
}

pub fn sixaxis_handle_slot(handle: u32) -> Option<(usize, usize)> {
    let [style, npad_id, device, _] = handle.to_le_bytes();
    let entry = match npad_id {
        0..=7 => usize::from(npad_id),
        0x10 => NPAD_ENTRY_OTHER,
        0x20 => NPAD_ENTRY_HANDHELD,
        _ => return None,
    };
    let index = match (style, device) {
        (3 | 8, _) => 0,
        (4, _) => 1,
        (5, 0) => 2,
        (5, 1) => 3,
        (6, _) => 4,
        (7, _) => 5,
        _ => return None,
    };
    Some((entry, index))
}

fn sixaxis_property_offset(handle: u32) -> Option<(usize, usize)> {
    let (entry, index) = sixaxis_handle_slot(handle)?;
    Some((sixaxis_property_at(entry, index), index))
}

pub fn sixaxis_route(connected: [bool; 3]) -> [Option<crate::hid_motion::MotionSource>; SIXAXIS_COUNT] {
    use crate::hid_motion::MotionSource::{self, Left, Primary, Right};
    const PREFERENCE: [[MotionSource; 3]; SIXAXIS_COUNT] = [
        [Primary, Right, Left],
        [Primary, Right, Left],
        [Left, Primary, Right],
        [Right, Primary, Left],
        [Left, Primary, Right],
        [Right, Primary, Left],
    ];
    PREFERENCE.map(|order| order.into_iter().find(|source| connected[source.index()]))
}

fn append_sixaxis_entry(buf: &mut [u8], lifo: usize, frame: &crate::hid_motion::SixAxisFrame) -> u64 {
    let total = read_u64(buf, lifo + 0x08);
    let mut tail = read_u64(buf, lifo + 0x10);
    let mut count = read_u64(buf, lifo + 0x18);
    if !(total == LIFO_STORAGE_COUNT as u64 && tail < LIFO_STORAGE_COUNT as u64 && count <= SIXAXIS_MAX_COUNT) {
        write_u64(buf, lifo + 0x08, LIFO_STORAGE_COUNT as u64);
        write_u64(buf, lifo + 0x10, 0);
        write_u64(buf, lifo + 0x18, 0);
        tail = 0;
        count = 0;
    }
    let previous = if count > 0 {
        read_u64(buf, lifo + LIFO_HEADER_SIZE + tail as usize * SIXAXIS_ELEM_SIZE + 0x10)
    } else {
        read_u64(buf, lifo + 0x00)
    };
    let sampling = previous.wrapping_add(1).max(1);
    let slot = (tail as usize + 1) % LIFO_STORAGE_COUNT;
    let mut state = frame.sanitized();
    state.delta_time_ns = state.delta_time_ns.max(SIXAXIS_MIN_DELTA_NS);
    let storage = lifo + LIFO_HEADER_SIZE + slot * SIXAXIS_ELEM_SIZE;
    write_u64(buf, storage, sampling << 1);
    let at = storage + 8;
    write_u64(buf, at + 0x00, state.delta_time_ns);
    write_u64(buf, at + 0x08, sampling);
    let vectors = [state.accel, state.gyro, state.angle, state.direction[0], state.direction[1], state.direction[2]];
    for (vector_index, vector) in vectors.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            write_f32(buf, at + 0x10 + vector_index * 0x0C + axis * 4, *value);
        }
    }
    write_u32(buf, at + 0x58, SIXAXIS_ATTR_IS_CONNECTED);
    write_u32(buf, at + 0x5C, 0);
    write_u64(buf, lifo + 0x00, sampling);
    write_u64(buf, lifo + 0x10, slot as u64);
    write_u64(buf, lifo + 0x18, (count + 1).min(SIXAXIS_MAX_COUNT));
    sampling
}

fn compact_frames(
    frames: &[crate::hid_motion::SixAxisFrame],
    max: usize,
) -> std::borrow::Cow<'_, [crate::hid_motion::SixAxisFrame]> {
    if max == 0 || frames.len() <= max {
        return std::borrow::Cow::Borrowed(frames);
    }
    let base = frames.len() / max;
    let extra = frames.len() % max;
    let mut start = 0;
    let groups = (0..max)
        .map(|group| {
            let len = base + usize::from(group < extra);
            let merged = crate::hid_motion::SixAxisFrame::merge(&frames[start..start + len]);
            start += len;
            merged
        })
        .collect();
    std::borrow::Cow::Owned(groups)
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

fn write_f32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub static HID_STATE: once_cell::sync::OnceCell<Arc<Mutex<HidState>>> =
    once_cell::sync::OnceCell::new();

pub fn get_hid_state() -> Arc<Mutex<HidState>> {
    HID_STATE
        .get_or_init(|| Arc::new(Mutex::new(HidState::new())))
        .clone()
}

pub fn recenter_motion() {
    crate::hid_motion::recenter_all();
    get_hid_state().lock().request_sixaxis_newly_assigned();
}

pub static CONSOLE_DOCKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static CONSOLE_MODE_DIRTY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static PLAYER1_JOY_DUAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PLAYER1_JOY_SINGLE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static NPAD_JOY_HOLD_HORIZONTAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn player1_joy_single() -> Option<bool> {
    match PLAYER1_JOY_SINGLE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

pub fn set_player1_joy_single(right: Option<bool>) {
    let value = match right {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    PLAYER1_JOY_SINGLE.store(value, std::sync::atomic::Ordering::Relaxed);
}

pub fn set_npad_joy_hold_horizontal(horizontal: bool) {
    NPAD_JOY_HOLD_HORIZONTAL.store(horizontal, std::sync::atomic::Ordering::Relaxed);
}

pub fn is_docked() -> bool {
    CONSOLE_DOCKED.load(std::sync::atomic::Ordering::Relaxed)
}

fn active_entry_for(docked: bool, joy_dual: bool, joy_single: Option<bool>) -> usize {
    if docked || joy_dual || joy_single.is_some() {
        NPAD_ENTRY_PLAYER1
    } else {
        NPAD_ENTRY_HANDHELD
    }
}

pub fn p1_active_entry() -> usize {
    active_entry_for(
        is_docked(),
        PLAYER1_JOY_DUAL.load(std::sync::atomic::Ordering::Relaxed),
        player1_joy_single(),
    )
}

pub fn player1_npad_id() -> u8 {
    if p1_active_entry() == NPAD_ENTRY_HANDHELD {
        0x20
    } else {
        0
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum P1Presentation {
    FullKey,
    Handheld,
    JoyDual,
    JoyLeft,
    JoyRight,
}

pub fn select_p1_presentation(
    style_set: u32,
    docked: bool,
    host: crate::hid_motion::HostKind,
    input: crate::hid_motion::HostKind,
    motion_connected: bool,
) -> P1Presentation {
    use crate::hid_motion::HostKind;
    let has = |styles: u32| style_set & styles != 0;
    if input == HostKind::JoyConPair && has(STYLE_JOY_DUAL) {
        return P1Presentation::JoyDual;
    }
    let joycon_host = matches!(host, HostKind::JoyConLeft | HostKind::JoyConRight | HostKind::JoyConPair);
    let detached = (joycon_host || motion_connected)
        && !has(STYLE_FULLKEY)
        && has(STYLE_JOY_DUAL | STYLE_JOY_LEFT | STYLE_JOY_RIGHT);
    if !docked && has(STYLE_HANDHELD) && !detached {
        return P1Presentation::Handheld;
    }
    if docked && has(STYLE_FULLKEY) {
        return P1Presentation::FullKey;
    }
    if !has(STYLE_FULLKEY) {
        if host == HostKind::JoyConLeft && has(STYLE_JOY_LEFT) {
            return P1Presentation::JoyLeft;
        }
        if host == HostKind::JoyConRight && has(STYLE_JOY_RIGHT) {
            return P1Presentation::JoyRight;
        }
    }
    if has(STYLE_JOY_DUAL) {
        return P1Presentation::JoyDual;
    }
    if has(STYLE_FULLKEY) {
        return P1Presentation::FullKey;
    }
    if has(STYLE_JOY_LEFT | STYLE_JOY_RIGHT) {
        return if host == HostKind::JoyConLeft && has(STYLE_JOY_LEFT) {
            P1Presentation::JoyLeft
        } else if has(STYLE_JOY_RIGHT) {
            P1Presentation::JoyRight
        } else {
            P1Presentation::JoyLeft
        };
    }
    if has(STYLE_HANDHELD) {
        return P1Presentation::Handheld;
    }
    P1Presentation::FullKey
}

pub fn current_p1_presentation(docked: bool, joy_dual: bool, joy_single: Option<bool>) -> P1Presentation {
    match (joy_dual, joy_single) {
        (true, _) => P1Presentation::JoyDual,
        (false, Some(true)) => P1Presentation::JoyRight,
        (false, Some(false)) => P1Presentation::JoyLeft,
        (false, None) if docked => P1Presentation::FullKey,
        (false, None) => P1Presentation::Handheld,
    }
}

pub fn presented_sticks(
    input: ControllerInput,
    presentation: P1Presentation,
    host: crate::hid_motion::HostKind,
) -> ControllerInput {
    fn stronger(first: (i32, i32), second: (i32, i32)) -> (i32, i32) {
        let magnitude = |(x, y): (i32, i32)| u64::from(x.unsigned_abs()).pow(2) + u64::from(y.unsigned_abs()).pow(2);
        if magnitude(second) > magnitude(first) {
            second
        } else {
            first
        }
    }
    let left = (input.stick_l_x, input.stick_l_y);
    let right = (input.stick_r_x, input.stick_r_y);
    let (left, right) = match presentation {
        P1Presentation::JoyRight => ((0, 0), stronger(right, left)),
        P1Presentation::JoyLeft => (stronger(left, right), (0, 0)),
        P1Presentation::FullKey | P1Presentation::Handheld
            if host == crate::hid_motion::HostKind::JoyConRight =>
        {
            (stronger(left, right), (0, 0))
        }
        _ => return input,
    };
    ControllerInput { stick_l_x: left.0, stick_l_y: left.1, stick_r_x: right.0, stick_r_y: right.1, ..input }
}

pub fn p1_presentation_flags(presentation: P1Presentation, assigned_joy_dual: Option<bool>) -> (bool, Option<bool>) {
    let single = match presentation {
        P1Presentation::JoyLeft => Some(false),
        P1Presentation::JoyRight => Some(true),
        _ => None,
    };
    match assigned_joy_dual {
        Some(true) => (true, None),
        Some(false) => (false, single),
        None => (presentation == P1Presentation::JoyDual, single),
    }
}

pub fn apply_p1_presentation(style_set: u32, assigned_joy_dual: Option<bool>) -> P1Presentation {
    let presentation = select_p1_presentation(
        style_set,
        is_docked(),
        crate::hid_motion::host_kind(),
        crate::hid_motion::input_kind(),
        crate::hid_motion::any_source_connected(),
    );
    let (joy_dual, single) = p1_presentation_flags(presentation, assigned_joy_dual);
    let state = get_hid_state();
    let mut hid = state.lock();
    set_player1_joy_dual(joy_dual);
    set_player1_joy_single(single);
    let cur = hid.input;
    hid.tick(cur);
    presentation
}

pub fn apply_controller_applet_style(style_set: u32) -> u32 {
    match apply_p1_presentation(style_set, None) {
        P1Presentation::Handheld => 0x20,
        _ => 0,
    }
}

pub fn set_player1_joy_dual(value: bool) {
    PLAYER1_JOY_DUAL.store(value, std::sync::atomic::Ordering::Relaxed);
    if value {
        PLAYER1_JOY_SINGLE.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hid_motion::{HostKind, MotionSnapshot, SixAxisFrame};
    use std::time::{Duration, Instant};

    fn read_u32_at(buf: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn read_f32_at(buf: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn read_vec3(buf: &[u8], offset: usize) -> [f32; 3] {
        std::array::from_fn(|axis| read_f32_at(buf, offset + axis * 4))
    }

    fn sixaxis_state(buf: &[u8], lifo: usize, age: usize) -> usize {
        let tail = read_u64(buf, lifo + 0x10) as usize;
        let slot = (tail + LIFO_STORAGE_COUNT - age) % LIFO_STORAGE_COUNT;
        lifo + LIFO_HEADER_SIZE + slot * SIXAXIS_ELEM_SIZE + 8
    }

    fn clear_sixaxis(hid: &mut HidState) {
        hid.sixaxis_last_write = [[None; SIXAXIS_COUNT]; 2];
        for entry in [NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD] {
            hid.buf[sixaxis_lifo_offset(entry, 0)..sixaxis_lifo_offset(entry, SIXAXIS_COUNT)].fill(0);
        }
    }

    #[test]
    fn exclusive_injection_blocks_host_fallback_and_preserves_live_input_by_default() {
        let host = ControllerInput {
            buttons: NPAD_BUTTON_PLUS | NPAD_BUTTON_DOWN,
            stick_l_y: -32767,
            ..ControllerInput::default()
        };
        let injected = parse_injected_input("buttons=A").unwrap();
        assert_eq!(injected.buttons, NPAD_BUTTON_A);
        assert_eq!(selected_controller_input(host, Some(injected), true), injected);
        assert_eq!(selected_controller_input(host, Some(injected), false), injected);
        for text in ["", "buttons=", "buttons= lx=0 ly=0 rx=0 ry=0", "invalid"] {
            let input = parse_injected_input(text);
            assert!(input.is_none());
            assert_eq!(selected_controller_input(host, input, true), ControllerInput::default());
            assert_eq!(selected_controller_input(host, input, false), host);
        }
    }

    #[test]
    fn exclusive_injection_publishes_only_selected_buttons_and_no_auxiliary_input() {
        let mut hid = HidState::new();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        unsafe { hid.bind_mapped_host(mapped.as_mut_ptr()); }
        hid.mouse = MouseInput { buttons: MOUSE_BUTTON_LEFT, connected: true, ..MouseInput::default() };
        hid.keyboard = KeyboardInput { modifiers: KEYBOARD_MOD_CONTROL, keys: [u8::MAX; 32], connected: true };
        hid.touch = TouchInput { x: 640, y: 360, pressed: true };
        hid.last_written_mouse = hid.mouse;
        hid.last_written_touch = hid.touch;
        let host = ControllerInput { buttons: NPAD_BUTTON_PLUS | NPAD_BUTTON_DOWN, ..ControllerInput::default() };
        let injected = ControllerInput { buttons: NPAD_BUTTON_A, ..ControllerInput::default() };
        for _ in 0..LIFO_STORAGE_COUNT + 1 {
            hid.tick_with_source(host, Some(injected), true);
        }
        assert_eq!(&mapped[..], &hid.buf[..]);
        let active = [NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD].into_iter().find(|entry| {
            let offset = NPAD_OFFSET + *entry * NPAD_ENTRY_SIZE + NPAD_STYLE_TAG_OFFSET;
            mapped[offset..offset + 4].iter().any(|byte| *byte != 0)
        }).unwrap();
        for layout in 0..LAYOUT_COUNT {
            let lifo = NPAD_OFFSET + active * NPAD_ENTRY_SIZE + LAYOUT_BASE_OFFSET + layout * LAYOUT_STRIDE;
            for index in 0..LIFO_STORAGE_COUNT {
                let buttons = lifo + LIFO_HEADER_SIZE + index * LIFO_STORAGE_ELEM_SIZE + 0x10;
                assert_eq!(read_u64(&mapped[..], buttons), NPAD_BUTTON_A);
            }
        }
        let state_at = |lifo: usize, stride: usize| lifo + LIFO_HEADER_SIZE
            + read_u64(&mapped[..], lifo + 0x10) as usize * stride + 8;
        let mouse = state_at(MOUSE_OFFSET, MOUSE_ELEM_SIZE);
        assert!(mapped[mouse + 0x08..mouse + 0x28].iter().all(|byte| *byte == 0));
        let keyboard = state_at(KEYBOARD_OFFSET, KEYBOARD_ELEM_SIZE);
        assert!(mapped[keyboard + 0x08..keyboard + 0x30].iter().all(|byte| *byte == 0));
        let touch = state_at(TOUCH_OFFSET, TOUCH_ELEM_SIZE);
        assert_eq!(read_u64(&mapped[..], touch + 0x08), 0);
        hid.tick_with_source(host, None, true);
        assert_eq!(hid.input, ControllerInput::default());
        assert!(hid.unbind_mapped_host(mapped.as_mut_ptr() as usize));
    }

    #[test]
    fn auxiliary_input_selection_preserves_devices_unless_exclusive() {
        let mouse = MouseInput { buttons: MOUSE_BUTTON_RIGHT, connected: true, ..MouseInput::default() };
        let keyboard = KeyboardInput { modifiers: KEYBOARD_MOD_SHIFT, keys: [1; 32], connected: true };
        let touch = TouchInput { x: 10, y: 20, pressed: true };
        assert!(selected_auxiliary_input(mouse, keyboard, touch, false) == (mouse, keyboard, touch));
        assert!(selected_auxiliary_input(mouse, keyboard, touch, true)
            == (MouseInput::default(), KeyboardInput::default(), TouchInput::default()));
    }

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
            wheel_x: -120,
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
        assert_eq!(read_i32(0x18), -120);
        assert_eq!(read_i32(0x1C), 240);
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
        assert_eq!(read_u64(&hid.buf[..], state + 0x08), u64::from(KEYBOARD_MOD_SHIFT));
        assert_eq!(read_u32_at(0x0C), 0);
        assert_eq!(hid.buf[state + 0x10], 1 << 4);
        assert_eq!(&hid.buf[state + 0x11..state + 0x30], &[0u8; 31][..]);
    }

    #[test]
    fn keyboard_f12_press_and_release_reach_guest_key_word() {
        let mut hid = HidState::new();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        unsafe { hid.bind_mapped_host(mapped.as_mut_ptr()); }
        let state_at = |buf: &[u8]| KEYBOARD_OFFSET + LIFO_HEADER_SIZE
            + read_u64(buf, KEYBOARD_OFFSET + 0x10) as usize * KEYBOARD_ELEM_SIZE + 8;
        hid.keyboard.connected = true;
        hid.keyboard.modifiers = KEYBOARD_MOD_CONTROL | KEYBOARD_MOD_SHIFT;
        hid.keyboard.keys[69 / 8] = 1 << (69 % 8);
        hid.tick_with_source(ControllerInput::default(), None, false);
        let pressed = state_at(&mapped[..]);
        assert_eq!(read_u64(&mapped[..], pressed), hid.sampling_number);
        assert_eq!(read_u64(&mapped[..], pressed + 0x08), u64::from(KEYBOARD_MOD_CONTROL | KEYBOARD_MOD_SHIFT));
        assert_eq!(read_u64(&mapped[..], pressed + 0x10), 0);
        assert_eq!(read_u64(&mapped[..], pressed + 0x18), 1u64 << (69 % 64));
        hid.keyboard.keys.fill(0);
        hid.keyboard.modifiers = 0;
        hid.tick_with_source(ControllerInput::default(), None, false);
        let released = state_at(&mapped[..]);
        assert!(mapped[released + 0x08..released + 0x30].iter().all(|byte| *byte == 0));
        assert_eq!(read_u64(&mapped[..], pressed + 0x18), 1u64 << (69 % 64));
    }

    #[test]
    fn repeated_guest_polls_preserve_the_current_mouse_sample() {
        let mut hid = HidState::new();
        hid.mouse = MouseInput { connected: true, ..MouseInput::default() };
        hid.tick(ControllerInput::default());
        hid.mouse.wheel_y = -1;
        hid.last_tick = None;
        hid.maybe_tick(ControllerInput::default());
        let sampling = hid.sampling_number;
        let snapshot = hid.buf.clone();
        for _ in 0..100 {
            hid.maybe_tick(ControllerInput::default());
        }
        assert_eq!(hid.sampling_number, sampling);
        assert_eq!(hid.buf.as_ref(), snapshot.as_ref());
        hid.last_tick = Some(std::time::Instant::now() - std::time::Duration::from_millis(20));
        hid.maybe_tick(ControllerInput::default());
        assert_eq!(hid.sampling_number, sampling + 1);
        assert_ne!(hid.buf.as_ref(), snapshot.as_ref());
    }

    #[test]
    fn mouse_wheel_totals_emit_one_delta_on_each_axis() {
        let mut hid = HidState::new();
        let state_at = |buf: &[u8]| MOUSE_OFFSET + LIFO_HEADER_SIZE
            + read_u64(buf, MOUSE_OFFSET + 0x10) as usize * MOUSE_ELEM_SIZE + 8;
        hid.mouse.connected = true;
        hid.tick_with_source(ControllerInput::default(), None, false);
        hid.mouse = MouseInput { wheel_x: 2, wheel_y: -3, connected: true, ..MouseInput::default() };
        hid.tick_with_source(ControllerInput::default(), None, false);
        let state = state_at(&hid.buf[..]);
        assert_eq!(i32::from_le_bytes(hid.buf[state + 0x18..state + 0x1C].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(hid.buf[state + 0x1C..state + 0x20].try_into().unwrap()), -3);
        hid.tick_with_source(ControllerInput::default(), None, false);
        let state = state_at(&hid.buf[..]);
        assert!(hid.buf[state + 0x18..state + 0x20].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn mouse_connection_changes_do_not_replay_motion_or_old_wheel_totals() {
        let mut hid = HidState::new();
        let deltas = |hid: &HidState| {
            let state = MOUSE_OFFSET + LIFO_HEADER_SIZE
                + read_u64(&hid.buf[..], MOUSE_OFFSET + 0x10) as usize * MOUSE_ELEM_SIZE + 8;
            std::array::from_fn::<_, 4, _>(|index| {
                let offset = state + 0x10 + index * 4;
                i32::from_le_bytes(hid.buf[offset..offset + 4].try_into().unwrap())
            })
        };
        hid.mouse = MouseInput { x: 300, y: 200, wheel_x: 8, wheel_y: -12, connected: true, ..MouseInput::default() };
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(deltas(&hid), [0; 4]);
        hid.mouse.x += 3;
        hid.mouse.y -= 4;
        hid.mouse.wheel_x += 1;
        hid.mouse.wheel_y -= 2;
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(deltas(&hid), [3, -4, 1, -2]);
        hid.mouse = MouseInput::default();
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(deltas(&hid), [0; 4]);
        hid.mouse = MouseInput { x: 900, y: 600, wheel_x: 9, wheel_y: -14, connected: true, ..MouseInput::default() };
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(deltas(&hid), [0; 4]);
        hid.mouse.x -= 2;
        hid.mouse.y += 1;
        hid.mouse.wheel_x -= 1;
        hid.mouse.wheel_y += 1;
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(deltas(&hid), [-2, 1, -1, 1]);
    }

    #[test]
    fn touch_lifo_tracks_press_hold_release_lifecycle() {
        let mut hid = HidState::new();
        let touch_state = |hid: &HidState| {
            let tail = read_u64(&hid.buf[..], TOUCH_OFFSET + 0x10) as usize;
            TOUCH_OFFSET + LIFO_HEADER_SIZE + tail * TOUCH_ELEM_SIZE + 8
        };
        let read_u32_at = |hid: &HidState, off: usize| {
            let state = touch_state(hid);
            u32::from_le_bytes(hid.buf[state + off..state + off + 4].try_into().unwrap())
        };

        hid.touch = TouchInput {
            x: 640,
            y: 360,
            pressed: true,
        };
        hid.tick(ControllerInput::default());
        assert_eq!(read_u32_at(&hid, 0x08), 1);
        assert_eq!(read_u32_at(&hid, 0x18), TOUCH_ATTR_START);
        assert_eq!(read_u32_at(&hid, 0x20), 640);
        assert_eq!(read_u32_at(&hid, 0x24), 360);

        hid.touch.x = 700;
        hid.tick(ControllerInput::default());
        assert_eq!(read_u32_at(&hid, 0x08), 1);
        assert_eq!(read_u32_at(&hid, 0x18), 0);
        assert_eq!(read_u32_at(&hid, 0x20), 700);

        hid.touch.pressed = false;
        hid.tick(ControllerInput::default());
        assert_eq!(read_u32_at(&hid, 0x08), 1);
        assert_eq!(read_u32_at(&hid, 0x18), TOUCH_ATTR_END);
        assert_eq!(read_u32_at(&hid, 0x20), 700);

        hid.tick(ControllerInput::default());
        assert_eq!(read_u32_at(&hid, 0x08), 0);
        assert_eq!(read_u32_at(&hid, 0x18), 0);
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

    #[test]
    fn sixaxis_property_offsets_follow_the_sdk_handle_layout() {
        let p1 = NPAD_OFFSET + NPAD_SIXAXIS_PROPERTIES_OFFSET;
        let other = NPAD_OFFSET + NPAD_ENTRY_OTHER * NPAD_ENTRY_SIZE + NPAD_SIXAXIS_PROPERTIES_OFFSET;
        assert_eq!(p1, 0xDDF0);
        assert_eq!(sixaxis_property_offset(0x0000_0003), Some((p1, 0)));
        assert_eq!(sixaxis_property_offset(0x0000_0005), Some((p1 + 2, 2)));
        assert_eq!(sixaxis_property_offset(0x0001_0005), Some((p1 + 3, 3)));
        assert_eq!(sixaxis_property_offset(0x0000_0006), Some((p1 + 4, 4)));
        assert_eq!(sixaxis_property_offset(0x0001_0007), Some((p1 + 5, 5)));
        assert_eq!(sixaxis_property_offset(0x0000_2004), Some((0x35DF1, 1)));
        assert_eq!(sixaxis_property_offset(0x0001_1006), Some((other + 4, 4)));
        for handle in [0x0000_0000, 0x0000_0802, 0x0002_0005, 0x0000_0807] {
            assert_eq!(sixaxis_property_offset(handle), None);
            assert_eq!(sixaxis_handle_slot(handle), None);
        }
        assert_eq!(sixaxis_handle_slot(0x0000_0005), Some((0, 2)));
        assert_eq!(sixaxis_handle_slot(0x0001_0005), Some((0, 3)));
        assert_eq!(sixaxis_handle_slot(0x0002_0003), Some((0, 0)));
        assert_eq!(sixaxis_handle_slot(0x0001_0007), Some((0, 5)));
        assert_eq!(sixaxis_handle_slot(0x0000_0006), Some((0, 4)));
        assert_eq!(sixaxis_handle_slot(0x0000_2004), Some((8, 1)));
        assert_eq!(sixaxis_handle_slot(0x0001_1006), Some((NPAD_ENTRY_OTHER, 4)));
        let lifos = [0xB158, 0xB860, 0xBF68, 0xC670, 0xCD78, 0xD480];
        for (index, lifo) in lifos.into_iter().enumerate() {
            assert_eq!(sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, index), lifo);
        }
    }

    #[test]
    fn newly_assigned_flag_waits_for_fresh_samples_and_clears_on_reset() {
        let mut hid = HidState::new();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        let ptr = mapped.as_mut_ptr();
        unsafe { hid.bind_mapped_host(ptr); }
        let p1 = NPAD_OFFSET + NPAD_SIXAXIS_PROPERTIES_OFFSET;
        let handle = 0x0001_0005;
        let flag = p1 + 3;
        let ring = sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, 3);
        let snapshot = MotionSnapshot::default();
        let frames: [Vec<SixAxisFrame>; 3] = Default::default();
        let t0 = Instant::now() + Duration::from_secs(60);
        let mut step = 0u32;
        let mut write = |hid: &mut HidState| {
            step += 1;
            let now = t0 + Duration::from_millis(5) * step;
            assert_ne!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, now, true, &snapshot, &frames) & 0b1000, 0);
            hid.publish_newly_assigned();
        };
        hid.request_sixaxis_newly_assigned();
        for _ in 0..=NEWLY_ASSIGNED_SETTLE_SAMPLES {
            write(&mut hid);
        }
        assert_eq!(&hid.buf[p1..p1 + SIXAXIS_COUNT], &[0u8; SIXAXIS_COUNT]);

        hid.set_sixaxis_passthrough(handle, true);
        hid.request_sixaxis_newly_assigned();
        let issued = read_u64(&hid.buf[..], ring);
        for _ in 1..NEWLY_ASSIGNED_SETTLE_SAMPLES {
            write(&mut hid);
        }
        assert_eq!(read_u64(&hid.buf[..], ring), issued + NEWLY_ASSIGNED_SETTLE_SAMPLES - 1);
        assert_eq!(hid.buf[flag], 0);
        write(&mut hid);
        assert_eq!(hid.buf[flag], SIXAXIS_PROPERTY_NEWLY_ASSIGNED);
        assert_eq!(mapped[flag], 0);
        hid.tick_with_source(ControllerInput::default(), None, false);
        assert_eq!(mapped[flag], SIXAXIS_PROPERTY_NEWLY_ASSIGNED);
        assert!(hid.reset_sixaxis_newly_assigned(handle));
        assert_eq!(mapped[flag], 0);
        assert_eq!(hid.buf[flag], 0);
        assert!(!hid.reset_sixaxis_newly_assigned(handle));
        assert!(!hid.reset_sixaxis_newly_assigned(0x0000_0802));

        let generation = hid.sixaxis_generation_seen.wrapping_add(1);
        hid.note_motion_generation(generation);
        let issued = read_u64(&hid.buf[..], ring);
        assert_eq!(hid.newly_assigned_pending, vec![(handle, issued + NEWLY_ASSIGNED_SETTLE_SAMPLES)]);
        hid.note_motion_generation(generation);
        assert_eq!(hid.newly_assigned_pending.len(), 1);
        for _ in 1..NEWLY_ASSIGNED_SETTLE_SAMPLES {
            write(&mut hid);
        }
        assert_eq!(hid.buf[flag], 0);
        write(&mut hid);
        assert_eq!(hid.buf[flag], SIXAXIS_PROPERTY_NEWLY_ASSIGNED);
        assert_eq!(&hid.buf[p1..p1 + 3], &[0u8; 3]);
        assert!(hid.buf[p1..p1 + SIXAXIS_COUNT].iter().all(|byte| byte & 0b10 == 0));
        assert!(mapped[p1..p1 + SIXAXIS_COUNT].iter().all(|byte| byte & 0b10 == 0));
        assert!(hid.reset_sixaxis_newly_assigned(handle));

        hid.set_sixaxis_passthrough(handle, true);
        hid.request_sixaxis_newly_assigned();
        assert!(hid.unbind_mapped_host(ptr as usize));
        for _ in 0..=NEWLY_ASSIGNED_SETTLE_SAMPLES {
            write(&mut hid);
        }
        assert_eq!(&hid.buf[p1..p1 + SIXAXIS_COUNT], &[0u8; SIXAXIS_COUNT]);
    }

    #[test]
    fn sixaxis_entries_consecutive_with_delta_time() {
        let mut buf = vec![0u8; SIXAXIS_STRIDE];
        for i in 0..40u64 {
            let frame = SixAxisFrame { delta_time_ns: 5_000_000 + i, ..SixAxisFrame::REST };
            assert_eq!(append_sixaxis_entry(&mut buf, 0, &frame), i + 1);
            assert_eq!(read_u64(&buf, 0x00), i + 1);
            assert_eq!(read_u64(&buf, 0x08), LIFO_STORAGE_COUNT as u64);
            let tail = read_u64(&buf, 0x10) as usize;
            assert_eq!(tail, (i as usize + 1) % LIFO_STORAGE_COUNT);
            assert_eq!(read_u64(&buf, 0x18), (i + 1).min(16));
            let storage = LIFO_HEADER_SIZE + tail * SIXAXIS_ELEM_SIZE;
            assert_eq!(read_u64(&buf, storage), (i + 1) << 1);
            assert_eq!(read_u64(&buf, storage + 8), 5_000_000 + i);
            assert_eq!(read_u64(&buf, storage + 0x10), i + 1);
            assert_eq!(read_u32_at(&buf, storage + 8 + 0x58), SIXAXIS_ATTR_IS_CONNECTED);
            assert_eq!(read_u32_at(&buf, storage + 8 + 0x5C), 0);
        }
        let short = SixAxisFrame { delta_time_ns: 0, ..SixAxisFrame::REST };
        append_sixaxis_entry(&mut buf, 0, &short);
        assert_eq!(read_u64(&buf, sixaxis_state(&buf, 0, 0)), SIXAXIS_MIN_DELTA_NS);
    }

    #[test]
    fn wraparound_keeps_window_consecutive() {
        let mut buf = vec![0u8; SIXAXIS_STRIDE];
        for _ in 0..50 {
            append_sixaxis_entry(&mut buf, 0, &SixAxisFrame::REST);
        }
        let count = read_u64(&buf, 0x18) as usize;
        assert_eq!(count, 16);
        let sampling = |age: usize| read_u64(&buf, sixaxis_state(&buf, 0, age) + 0x08);
        assert_eq!(sampling(0), 50);
        for age in 1..count {
            assert_eq!(sampling(age - 1) - sampling(age), 1);
        }
        write_u64(&mut buf, 0x18, 40);
        assert_eq!(append_sixaxis_entry(&mut buf, 0, &SixAxisFrame::REST), 51);
        assert_eq!(read_u64(&buf, 0x18), 1);
        assert_eq!(read_u64(&buf, 0x10), 1);
    }

    #[test]
    fn joy_dual_routes_left_right() {
        let mut hid = HidState::new();
        clear_sixaxis(&mut hid);
        let left = SixAxisFrame { accel: [0.25, 0.0, -1.0], ..SixAxisFrame::REST };
        let right = SixAxisFrame { accel: [0.5, 0.0, -1.0], ..SixAxisFrame::REST };
        let snapshot = MotionSnapshot { connected: [false, true, true], ..MotionSnapshot::default() };
        let frames = [Vec::new(), vec![left; 3], vec![right; 2]];
        let touched = hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, Instant::now(), true, &snapshot, &frames);
        assert_eq!(touched, 0b11_1111);
        let expected = [(0xB158, 0.5, 2), (0xB860, 0.5, 2), (0xBF68, 0.25, 3), (0xC670, 0.5, 2), (0xCD78, 0.25, 3), (0xD480, 0.5, 2)];
        for (lifo, accel_x, count) in expected {
            assert_eq!(read_u64(&hid.buf[..], lifo + 0x18), count, "{:#x}", lifo);
            assert_eq!(read_vec3(&hid.buf[..], sixaxis_state(&hid.buf[..], lifo, 0) + 0x10)[0], accel_x, "{:#x}", lifo);
        }
    }

    #[test]
    fn primary_feeds_all_lifos() {
        use crate::hid_motion::MotionSource::{Left, Primary, Right};
        let mut hid = HidState::new();
        clear_sixaxis(&mut hid);
        assert_eq!(sixaxis_route([false; 3]), [None; SIXAXIS_COUNT]);
        assert_eq!(sixaxis_route([true; 3]), [Some(Primary), Some(Primary), Some(Left), Some(Right), Some(Left), Some(Right)]);
        assert_eq!(sixaxis_route([false, false, true]), [Some(Right); SIXAXIS_COUNT]);
        assert_eq!(sixaxis_route([false, true, false]), [Some(Left); SIXAXIS_COUNT]);
        let primary = SixAxisFrame { accel: [0.75, 0.0, -1.0], ..SixAxisFrame::REST };
        let snapshot = MotionSnapshot { connected: [true, false, false], ..MotionSnapshot::default() };
        let frames = [vec![primary; 4], Vec::new(), Vec::new()];
        for entry in [NPAD_ENTRY_PLAYER1, NPAD_ENTRY_HANDHELD] {
            assert_eq!(hid.write_sixaxis_from(entry, Instant::now(), false, &snapshot, &frames), 0b11_1111);
            for index in 0..SIXAXIS_COUNT {
                let lifo = sixaxis_lifo_offset(entry, index);
                assert_eq!(read_u64(&hid.buf[..], lifo + 0x18), 4);
                assert_eq!(read_vec3(&hid.buf[..], sixaxis_state(&hid.buf[..], lifo, 0) + 0x10)[0], 0.75);
            }
        }
    }

    #[test]
    fn rest_and_hold_policy() {
        let mut hid = HidState::new();
        clear_sixaxis(&mut hid);
        let t0 = Instant::now() + Duration::from_secs(60);
        let at = |millis: u64| t0 + Duration::from_millis(millis);
        let none = MotionSnapshot::default();
        let empty: [Vec<SixAxisFrame>; 3] = Default::default();
        let ring = sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, 3);
        let latest = |hid: &HidState| sixaxis_state(&hid.buf[..], ring, 0);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(0), true, &none, &empty), 0b11_1111);
        assert_eq!(read_u64(&hid.buf[..], latest(&hid)), SIXAXIS_SYNTH_FIRST_DELTA_NS);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(1), true, &none, &empty), 0);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(10), false, &none, &empty), 0);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(10), true, &none, &empty), 0b11_1111);
        assert_eq!(read_u64(&hid.buf[..], latest(&hid)), 10_000_000);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(12), true, &none, &empty), 0b11_1111);
        assert_eq!(read_u64(&hid.buf[..], latest(&hid)), 2_000_000);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(600), true, &none, &empty), 0b11_1111);
        assert_eq!(read_u64(&hid.buf[..], latest(&hid)), 100_000_000);
        assert_eq!(read_vec3(&hid.buf[..], latest(&hid) + 0x10), [0.0, 0.0, -1.0]);
        assert_eq!(read_vec3(&hid.buf[..], latest(&hid) + 0x1C), [0.0, 0.0, 1.0e-6]);
        assert_eq!(read_u64(&hid.buf[..], sixaxis_lifo_offset(NPAD_ENTRY_HANDHELD, 3) + 0x18), 0);

        let live = MotionSnapshot { connected: [false, true, false], ..MotionSnapshot::default() };
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(700), true, &live, &empty), 0);

        let held = SixAxisFrame {
            accel: [0.2, -3.5, -1.0],
            gyro: [0.3, 0.0, 0.0],
            angle: [0.1, 0.2, 0.3],
            direction: [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            ..SixAxisFrame::REST
        };
        let stale = MotionSnapshot {
            connected: [false, true, false],
            stale: [false, true, false],
            last: [None, Some(held), None],
            ..MotionSnapshot::default()
        };
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(750), true, &stale, &empty), 0b11_1111);
        let hold = latest(&hid);
        assert_eq!(read_u64(&hid.buf[..], hold), 100_000_000);
        assert_eq!(read_vec3(&hid.buf[..], hold + 0x10), [0.0, -1.0, 0.0]);
        assert_eq!(read_vec3(&hid.buf[..], hold + 0x1C), [0.0, 0.0, 1.0e-6]);
        assert_eq!(read_vec3(&hid.buf[..], hold + 0x28), held.angle);
        assert_eq!(read_vec3(&hid.buf[..], hold + 0x40), held.direction[1]);
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(751), true, &stale, &empty), 0);

        let stale_unknown = MotionSnapshot { last: [None; 3], ..stale };
        assert_eq!(hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, at(760), true, &stale_unknown, &empty), 0b11_1111);
        assert_eq!(read_u64(&hid.buf[..], latest(&hid)), 10_000_000);
        assert_eq!(read_vec3(&hid.buf[..], latest(&hid) + 0x1C), [0.0, 0.0, 1.0e-6]);

        let identity = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for index in 0..SIXAXIS_COUNT {
            let lifo = sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, index);
            let count = read_u64(&hid.buf[..], lifo + 0x18) as usize;
            assert_eq!(count, 6);
            for age in 0..count {
                let state = sixaxis_state(&hid.buf[..], lifo, age);
                let direction: [f32; 9] = std::array::from_fn(|i| read_f32_at(&hid.buf[..], state + 0x34 + i * 4));
                assert_ne!(read_vec3(&hid.buf[..], state + 0x1C), [0.0; 3]);
                assert_ne!(read_vec3(&hid.buf[..], state + 0x10), [0.0; 3]);
                assert_ne!(direction, identity);
                assert!(read_u64(&hid.buf[..], state) >= SIXAXIS_MIN_DELTA_NS);
                assert_ne!(read_u64(&hid.buf[..], state + 0x08), 0);
                assert_eq!(read_u32_at(&hid.buf[..], state + 0x58), SIXAXIS_ATTR_IS_CONNECTED);
            }
        }
    }

    #[test]
    fn burst_is_compacted_to_sixteen_entries() {
        let mut hid = HidState::new();
        clear_sixaxis(&mut hid);
        let turned = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let burst: Vec<SixAxisFrame> = (0..60)
            .map(|i| SixAxisFrame {
                delta_time_ns: 5_000_000,
                gyro: [0.0, 0.0, i as f32],
                angle: [i as f32, 0.0, 0.0],
                direction: if i == 59 { turned } else { SixAxisFrame::REST.direction },
                ..SixAxisFrame::REST
            })
            .collect();
        let snapshot = MotionSnapshot { connected: [false, false, true], ..MotionSnapshot::default() };
        let frames = [Vec::new(), Vec::new(), burst];
        hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, Instant::now(), true, &snapshot, &frames);
        let ring = sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, 3);
        assert_eq!(read_u64(&hid.buf[..], ring + 0x18), 16);
        assert_eq!(read_u64(&hid.buf[..], ring), 16);
        let oldest_first: Vec<usize> = (0..16).rev().map(|age| sixaxis_state(&hid.buf[..], ring, age)).collect();
        let deltas: Vec<u64> = oldest_first.iter().map(|&state| read_u64(&hid.buf[..], state)).collect();
        assert_eq!(&deltas[..12], &[20_000_000; 12]);
        assert_eq!(&deltas[12..], &[15_000_000; 4]);
        assert_eq!(deltas.iter().sum::<u64>(), 300_000_000);
        for (position, &state) in oldest_first.iter().enumerate() {
            assert_eq!(read_u64(&hid.buf[..], state + 0x08), position as u64 + 1);
        }
        assert!((read_vec3(&hid.buf[..], oldest_first[0] + 0x1C)[2] - 1.5).abs() < 1e-6);
        let newest = oldest_first[15];
        assert_eq!(read_vec3(&hid.buf[..], newest + 0x28), [59.0, 0.0, 0.0]);
        assert_eq!(read_vec3(&hid.buf[..], newest + 0x34), turned[0]);
        assert_eq!(read_vec3(&hid.buf[..], newest + 0x40), turned[1]);
        let few = [SixAxisFrame::REST; 3];
        assert!(matches!(compact_frames(&few, 16), std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn publish_copies_touched_rings_and_properties() {
        let mut hid = HidState::new();
        let mut mapped = Box::new([0u8; HID_SHMEM_SIZE]);
        let ptr = mapped.as_mut_ptr();
        unsafe { hid.bind_mapped_host(ptr); }
        clear_sixaxis(&mut hid);
        let before = mapped.clone();
        let snapshot = MotionSnapshot { connected: [false, false, true], ..MotionSnapshot::default() };
        let frames = [Vec::new(), Vec::new(), vec![SixAxisFrame::REST; 2]];
        let touched = hid.write_sixaxis_from(NPAD_ENTRY_PLAYER1, Instant::now(), false, &snapshot, &frames);
        assert_eq!(touched, 0b11_1111);
        hid.buf[sixaxis_property_at(NPAD_ENTRY_PLAYER1, 3)] = SIXAXIS_PROPERTY_NEWLY_ASSIGNED;
        hid.copy_sixaxis_to_mapped(NPAD_ENTRY_PLAYER1, 0b1000);
        let ring = |index: usize| {
            sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, index)..sixaxis_lifo_offset(NPAD_ENTRY_PLAYER1, index + 1)
        };
        assert_eq!(&mapped[ring(3)], &hid.buf[ring(3)]);
        assert_eq!(&mapped[ring(2)], &before[ring(2)]);
        assert_ne!(&mapped[ring(2)], &hid.buf[ring(2)]);
        let properties = sixaxis_property_at(NPAD_ENTRY_PLAYER1, 0);
        assert_eq!(&mapped[properties..properties + SIXAXIS_COUNT], &hid.buf[properties..properties + SIXAXIS_COUNT]);
        assert_eq!(mapped[properties + 3], SIXAXIS_PROPERTY_NEWLY_ASSIGNED);
        assert!(hid.unbind_mapped_host(ptr as usize));
        hid.copy_sixaxis_to_mapped(NPAD_ENTRY_PLAYER1, 0b11_1111);
    }

    #[test]
    fn injected_motion_tokens() {
        let all = [true; 3];
        let flat = [0.0, 0.0, -1.0];
        assert_eq!(
            parse_injected_motion("gz=1"),
            (Some(InjectedMotion { accel: flat, gyro: [0.0, 0.0, 1.0], sources: all }), None)
        );
        assert_eq!(
            parse_injected_motion("msrc=r ax=0 ay=-1 az=0"),
            (Some(InjectedMotion { accel: [0.0, -1.0, 0.0], gyro: [0.0; 3], sources: [false, false, true] }), None)
        );
        assert_eq!(
            parse_injected_motion("buttons=ZR motion=rest"),
            (Some(InjectedMotion { accel: flat, gyro: [0.0; 3], sources: all }), None)
        );
        assert_eq!(parse_injected_motion("motion=off gz=1"), (None, None));
        assert_eq!(parse_injected_motion("buttons=A"), (None, None));
        assert_eq!(parse_injected_motion("gx=abc"), (None, None));
        assert_eq!(parse_injected_motion("msrc=x"), (None, None));
        assert_eq!(parse_injected_motion("msrc=x gz=1"), (None, None));
        assert_eq!(parse_injected_motion("gx=inf"), (None, None));
        assert_eq!(parse_injected_motion("recenter=3"), (None, Some(3)));
        assert_eq!(parse_injected_motion(""), (None, None));
        let (spec, recenter) = parse_injected_motion("buttons=ZR msrc=lr gx=1.6 ay=-3.0 az=-0.5 recenter=7");
        assert_eq!(spec, Some(InjectedMotion { accel: [0.0, -3.0, -0.5], gyro: [1.6, 0.0, 0.0], sources: [false, true, true] }));
        assert_eq!(recenter, Some(7));
        assert_eq!(parse_injected_motion("msrc=p motion=rest").0.map(|motion| motion.sources), Some([true, false, false]));
        assert_eq!(parse_injected_motion("msrc=l gy=0.5").0.map(|motion| motion.sources), Some([false, true, false]));
        assert_eq!(parse_injected_input("gz=1 buttons=A").map(|input| input.buttons), Some(NPAD_BUTTON_A));
    }

    #[test]
    fn select_p1_presentation_matrix() {
        use P1Presentation::{FullKey, Handheld, JoyDual, JoyLeft, JoyRight};
        let hosts = [HostKind::None, HostKind::Other, HostKind::JoyConLeft, HostKind::JoyConRight, HostKind::JoyConPair];
        for host in hosts {
            for (input, motion) in [(HostKind::None, false), (HostKind::None, true), (host, false), (host, true)] {
                let joycon = matches!(host, HostKind::JoyConLeft | HostKind::JoyConRight | HostKind::JoyConPair);
                let detached = joycon || motion;
                let pair = input == HostKind::JoyConPair;
                assert_eq!(select_p1_presentation(0x1F, true, host, input, motion), if pair { JoyDual } else { FullKey });
                assert_eq!(select_p1_presentation(0x1F, false, host, input, motion), if pair { JoyDual } else { Handheld });
                assert_eq!(select_p1_presentation(0x06, true, host, input, motion), JoyDual);
                let undocked_dual = if detached { JoyDual } else { Handheld };
                let selected = select_p1_presentation(0x06, false, host, input, motion);
                assert_eq!(selected, undocked_dual, "{:?} {:?} {}", host, input, motion);
                for docked in [true, false] {
                    let expected = if host == HostKind::JoyConLeft {
                        JoyLeft
                    } else if docked || detached {
                        JoyRight
                    } else {
                        Handheld
                    };
                    let selected = select_p1_presentation(0x1a, docked, host, input, motion);
                    assert_eq!(selected, expected, "{:?} {:?} {} {}", host, input, docked, motion);
                }
                assert_eq!(select_p1_presentation(0x02, true, host, input, motion), Handheld);
                assert_eq!(select_p1_presentation(0x00, true, host, input, motion), FullKey);
                assert_eq!(select_p1_presentation(0x00, false, host, input, motion), FullKey);
            }
        }
        for docked in [true, false] {
            for style_set in [0x1F, 0x07, 0x05] {
                assert_eq!(select_p1_presentation(style_set, docked, HostKind::JoyConPair, HostKind::JoyConPair, true), JoyDual);
                assert_eq!(select_p1_presentation(style_set, docked, HostKind::Other, HostKind::JoyConPair, false), JoyDual);
            }
            let pro_or_handheld = if docked { FullKey } else { Handheld };
            assert_eq!(select_p1_presentation(0x1F, docked, HostKind::JoyConPair, HostKind::None, true), pro_or_handheld);
            assert_eq!(select_p1_presentation(0x1F, docked, HostKind::JoyConPair, HostKind::Other, true), pro_or_handheld);
            assert_eq!(select_p1_presentation(0x03, docked, HostKind::JoyConPair, HostKind::JoyConPair, true), pro_or_handheld);
        }
        assert_eq!(select_p1_presentation(0x1C, true, HostKind::JoyConLeft, HostKind::JoyConLeft, false), JoyLeft);
        assert_eq!(select_p1_presentation(0x1C, true, HostKind::JoyConLeft, HostKind::JoyConLeft, true), JoyLeft);
        assert_eq!(select_p1_presentation(0x18, true, HostKind::None, HostKind::None, false), JoyRight);
        let legacy = |style_set: u32, docked: bool| {
            if !docked && style_set & STYLE_HANDHELD != 0 {
                Handheld
            } else if docked && style_set & STYLE_FULLKEY != 0 {
                FullKey
            } else if style_set & STYLE_JOY_DUAL != 0 {
                JoyDual
            } else if style_set & STYLE_FULLKEY != 0 {
                FullKey
            } else if style_set & (STYLE_JOY_LEFT | STYLE_JOY_RIGHT) != 0 {
                if style_set & STYLE_JOY_RIGHT != 0 {
                    JoyRight
                } else {
                    JoyLeft
                }
            } else if style_set & STYLE_HANDHELD != 0 {
                Handheld
            } else {
                FullKey
            }
        };
        for style_set in 0..0x20 {
            for docked in [true, false] {
                for host in [HostKind::None, HostKind::Other] {
                    let selected = select_p1_presentation(style_set, docked, host, host, false);
                    assert_eq!(selected, legacy(style_set, docked), "{:#x} {} {:?}", style_set, docked, host);
                }
            }
        }
        for style_set in [0x1F, 0x06, 0x1a, 0x03] {
            assert_eq!(select_p1_presentation(style_set, false, HostKind::Other, HostKind::Other, false), legacy(style_set, false));
        }
    }

    #[test]
    fn presented_sticks_follow_the_presented_side() {
        let input = ControllerInput { buttons: NPAD_BUTTON_A, stick_l_x: 100, stick_l_y: 0, stick_r_x: 0, stick_r_y: 50 };
        let sticks = |input: ControllerInput| (input.stick_l_x, input.stick_l_y, input.stick_r_x, input.stick_r_y);
        let right = presented_sticks(input, P1Presentation::JoyRight, HostKind::Other);
        assert_eq!(sticks(right), (0, 0, 100, 0));
        assert_eq!(right.buttons, NPAD_BUTTON_A);
        let mirrored = ControllerInput { stick_l_x: 0, stick_l_y: 50, stick_r_x: 100, stick_r_y: 0, ..input };
        assert_eq!(sticks(presented_sticks(mirrored, P1Presentation::JoyLeft, HostKind::Other)), (100, 0, 0, 0));
        let joycon_right = ControllerInput { stick_l_x: 0, stick_l_y: 0, stick_r_x: 0, stick_r_y: 20000, ..input };
        assert_eq!(sticks(presented_sticks(joycon_right, P1Presentation::FullKey, HostKind::JoyConRight)), (0, 20000, 0, 0));
        assert_eq!(sticks(presented_sticks(joycon_right, P1Presentation::Handheld, HostKind::JoyConRight)), (0, 20000, 0, 0));
        assert_eq!(presented_sticks(joycon_right, P1Presentation::FullKey, HostKind::Other), joycon_right);
        assert_eq!(presented_sticks(input, P1Presentation::JoyDual, HostKind::JoyConRight), input);
        let tie = ControllerInput { stick_l_x: 30, stick_l_y: 40, stick_r_x: -50, stick_r_y: 0, ..input };
        assert_eq!(sticks(presented_sticks(tie, P1Presentation::JoyRight, HostKind::None)), (0, 0, -50, 0));
        assert_eq!(sticks(presented_sticks(tie, P1Presentation::JoyLeft, HostKind::None)), (30, 40, 0, 0));
        let extreme = ControllerInput { stick_l_x: i32::MIN, stick_l_y: i32::MIN, stick_r_x: 1, stick_r_y: 1, ..input };
        assert_eq!(sticks(presented_sticks(extreme, P1Presentation::JoyRight, HostKind::None)), (0, 0, i32::MIN, i32::MIN));
        assert_eq!(current_p1_presentation(true, false, None), P1Presentation::FullKey);
        assert_eq!(current_p1_presentation(false, false, None), P1Presentation::Handheld);
        assert_eq!(current_p1_presentation(false, true, Some(true)), P1Presentation::JoyDual);
        assert_eq!(current_p1_presentation(true, false, Some(true)), P1Presentation::JoyRight);
        assert_eq!(current_p1_presentation(false, false, Some(false)), P1Presentation::JoyLeft);
        assert_eq!(active_entry_for(false, false, None), NPAD_ENTRY_HANDHELD);
        assert_eq!(active_entry_for(false, false, Some(true)), NPAD_ENTRY_PLAYER1);
        assert_eq!(active_entry_for(false, true, None), NPAD_ENTRY_PLAYER1);
        assert_eq!(active_entry_for(true, false, None), NPAD_ENTRY_PLAYER1);
    }

    #[test]
    fn joy_assignment_survives_presentation_refresh() {
        use P1Presentation::{FullKey, Handheld, JoyDual, JoyLeft, JoyRight};
        for presentation in [FullKey, Handheld, JoyDual, JoyLeft, JoyRight] {
            assert_eq!(p1_presentation_flags(presentation, Some(true)), (true, None), "{:?}", presentation);
        }
        assert_eq!(p1_presentation_flags(FullKey, None), (false, None));
        assert_eq!(p1_presentation_flags(Handheld, None), (false, None));
        assert_eq!(p1_presentation_flags(JoyDual, None), (true, None));
        assert_eq!(p1_presentation_flags(JoyLeft, None), (false, Some(false)));
        assert_eq!(p1_presentation_flags(JoyRight, None), (false, Some(true)));
        assert_eq!(p1_presentation_flags(JoyDual, Some(false)), (false, None));
        assert_eq!(p1_presentation_flags(JoyLeft, Some(false)), (false, Some(false)));
        assert_eq!(p1_presentation_flags(JoyRight, Some(false)), (false, Some(true)));
        let docked = select_p1_presentation(0x1F, true, HostKind::JoyConPair, HostKind::None, true);
        let (joy_dual, single) = p1_presentation_flags(docked, None);
        assert_eq!(current_p1_presentation(true, joy_dual, single), FullKey);
        let (joy_dual, single) = p1_presentation_flags(docked, Some(true));
        assert_eq!(current_p1_presentation(true, joy_dual, single), JoyDual);
        let undocked = select_p1_presentation(0x1F, false, HostKind::JoyConPair, HostKind::None, false);
        let (joy_dual, single) = p1_presentation_flags(undocked, Some(true));
        assert_eq!(active_entry_for(false, joy_dual, single), NPAD_ENTRY_PLAYER1);
        assert_eq!(current_p1_presentation(false, joy_dual, single), JoyDual);
        let paired = select_p1_presentation(0x1F, false, HostKind::JoyConPair, HostKind::JoyConPair, true);
        let (joy_dual, single) = p1_presentation_flags(paired, None);
        assert_eq!(active_entry_for(false, joy_dual, single), NPAD_ENTRY_PLAYER1);
        assert_eq!(current_p1_presentation(false, joy_dual, single), JoyDual);
        let (joy_dual, single) = p1_presentation_flags(paired, Some(false));
        assert_eq!(current_p1_presentation(true, joy_dual, single), FullKey);
    }
}

use parking_lot::Mutex;
use std::time::{Duration, Instant};

const SLOT_COUNT: usize = 20;
const NPAD_SLOT_COUNT: usize = 10;
const TRACE_HANDLES: usize = 8;
const STYLE_GC: u8 = 8;
const STYLE_N64: u8 = 13;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VibrationValue {
    pub amp_low: f32,
    pub freq_low: f32,
    pub amp_high: f32,
    pub freq_high: f32,
}

impl VibrationValue {
    pub const DEFAULT: Self = Self { amp_low: 0.0, freq_low: 160.0, amp_high: 0.0, freq_high: 320.0 };
    pub const FULL: Self = Self { amp_low: 1.0, freq_low: 160.0, amp_high: 1.0, freq_high: 320.0 };

    pub fn from_words(w: [u32; 4]) -> Self {
        Self {
            amp_low: f32::from_bits(w[0]),
            freq_low: f32::from_bits(w[1]),
            amp_high: f32::from_bits(w[2]),
            freq_high: f32::from_bits(w[3]),
        }
    }

    pub fn from_le_bytes(b: &[u8]) -> Self {
        if b.len() < 16 {
            return Self::DEFAULT;
        }
        let word = |at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        Self::from_words([word(0), word(4), word(8), word(12)])
    }

    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        let words = [self.amp_low, self.freq_low, self.amp_high, self.freq_high];
        for (index, value) in words.iter().enumerate() {
            out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }

    pub fn sanitized(self) -> Self {
        let amplitude = |value: f32| if value.is_finite() && value > 0.0 { value.min(1.0) } else { 0.0 };
        let frequency = |value: f32, fallback: f32| if value.is_finite() && value > 0.0 { value } else { fallback };
        Self {
            amp_low: amplitude(self.amp_low),
            freq_low: frequency(self.freq_low, Self::DEFAULT.freq_low),
            amp_high: amplitude(self.amp_high),
            freq_high: frequency(self.freq_high, Self::DEFAULT.freq_high),
        }
    }

    pub fn is_silent(&self) -> bool {
        !(self.amp_low > 0.0) && !(self.amp_high > 0.0)
    }

    pub fn scaled(self, s: f32) -> Self {
        Self { amp_low: self.amp_low * s, amp_high: self.amp_high * s, ..self }
    }

    pub fn louder_bands(self, o: Self) -> Self {
        let (amp_low, freq_low) =
            if o.amp_low > self.amp_low { (o.amp_low, o.freq_low) } else { (self.amp_low, self.freq_low) };
        let (amp_high, freq_high) =
            if o.amp_high > self.amp_high { (o.amp_high, o.freq_high) } else { (self.amp_high, self.freq_high) };
        Self { amp_low, freq_low, amp_high, freq_high }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceHandle {
    pub style: u8,
    pub npad: u8,
    pub index: u8,
}

pub fn decode(raw: u32) -> Option<DeviceHandle> {
    let [style, npad, index, _] = raw.to_le_bytes();
    let style_valid = matches!(style, 3..=8 | 13 | 0x20 | 0x21);
    (style_valid && npad_slot(npad).is_some() && index <= 1).then_some(DeviceHandle { style, npad, index })
}

fn npad_slot(npad: u8) -> Option<usize> {
    match npad {
        0..=7 => Some(usize::from(npad)),
        0x10 => Some(8),
        0x20 => Some(9),
        _ => None,
    }
}

fn slot_index(handle: DeviceHandle, index: u8) -> Option<usize> {
    npad_slot(handle.npad).map(|npad| npad * 2 + usize::from(index))
}

pub fn device_info(raw: u32) -> (u32, u32) {
    match decode(raw) {
        Some(handle) if (3..=7).contains(&handle.style) => (1, u32::from(handle.index) + 1),
        Some(handle) if handle.style == STYLE_GC => (2, 0),
        Some(handle) if handle.style == STYLE_N64 => (3, 0),
        _ => (0, 0),
    }
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    latest: VibrationValue,
    peak: VibrationValue,
    last_write: Option<Instant>,
    interval_ms: f32,
}

impl Slot {
    const EMPTY: Slot =
        Slot { latest: VibrationValue::DEFAULT, peak: VibrationValue::DEFAULT, last_write: None, interval_ms: 250.0 };

    fn stale_after(&self) -> Duration {
        let millis = (8.0 * self.interval_ms).clamp(150.0, 2000.0);
        Duration::from_micros((millis * 1000.0).round() as u64)
    }

    fn fresh(&self, now: Instant) -> bool {
        self.last_write.is_some_and(|at| now.saturating_duration_since(at) <= self.stale_after())
    }

    fn write(&mut self, v: VibrationValue, now: Instant) {
        let fresh = self.fresh(now);
        match self.last_write {
            None => self.interval_ms = 250.0,
            Some(at) => {
                let dt = now.saturating_duration_since(at).as_secs_f32() * 1000.0;
                self.interval_ms += (dt.min(2000.0) - self.interval_ms) * 0.1;
            }
        }
        self.peak = if fresh { self.peak.louder_bands(v) } else { v };
        self.latest = v;
        self.last_write = Some(now);
    }

    fn take(&mut self, now: Instant) -> VibrationValue {
        let out = if self.fresh(now) { self.peak } else { VibrationValue::DEFAULT };
        self.peak = self.latest;
        out
    }
}

#[derive(Clone, Copy, Debug)]
struct TraceState {
    window: Option<Instant>,
    batches: u32,
    singles: u32,
    handles: [u32; TRACE_HANDLES],
    handle_count: usize,
    peak: [f32; 2],
}

impl TraceState {
    const EMPTY: TraceState =
        TraceState { window: None, batches: 0, singles: 0, handles: [0; TRACE_HANDLES], handle_count: 0, peak: [0.0; 2] };

    fn record(&mut self, raw: u32, index: u8, value: VibrationValue) {
        if !self.handles[..self.handle_count].contains(&raw) && self.handle_count < TRACE_HANDLES {
            self.handles[self.handle_count] = raw;
            self.handle_count += 1;
        }
        let side = usize::from(index.min(1));
        self.peak[side] = self.peak[side].max(value.amp_low).max(value.amp_high);
    }

    fn line(&mut self, batch: bool, now: Instant) -> Option<String> {
        if batch {
            self.batches += 1;
        } else {
            self.singles += 1;
        }
        let started = *self.window.get_or_insert(now);
        let elapsed = now.saturating_duration_since(started);
        if elapsed < Duration::from_secs(1) {
            return None;
        }
        let seconds = elapsed.as_secs_f32();
        let handles: Vec<String> =
            self.handles[..self.handle_count].iter().map(|handle| format!("{:#x}", handle)).collect();
        let line = format!(
            "vibration: {:.0} batches/s, {:.0} single/s, handles {}, peak L {:.2} R {:.2}",
            self.batches as f32 / seconds,
            self.singles as f32 / seconds,
            handles.join(" "),
            self.peak[0],
            self.peak[1],
        );
        *self = TraceState { window: Some(now), ..TraceState::EMPTY };
        Some(line)
    }
}

fn trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NEXIUM_VIBRATION_TRACE").as_deref() == Ok("1"))
}

pub struct Mailbox {
    slots: [Slot; SLOT_COUNT],
    erm: [u64; NPAD_SLOT_COUNT],
    trace: TraceState,
}

impl Mailbox {
    pub const fn new() -> Self {
        Self { slots: [Slot::EMPTY; SLOT_COUNT], erm: [0; NPAD_SLOT_COUNT], trace: TraceState::EMPTY }
    }

    fn store(&mut self, raw: u32, handle: DeviceHandle, index: u8, value: VibrationValue, now: Instant) {
        let Some(slot) = slot_index(handle, index) else {
            return;
        };
        self.slots[slot].write(value, now);
        if trace_enabled() {
            self.trace.record(raw, index, value);
        }
    }

    pub fn submit(&mut self, raw: u32, v: VibrationValue, allowed: bool, now: Instant) {
        let Some(handle) = decode(raw) else {
            return;
        };
        let value = if allowed { v.sanitized() } else { VibrationValue::DEFAULT };
        self.store(raw, handle, handle.index, value, now);
    }

    pub fn submit_batch(&mut self, handles: &[u8], values: &[u8], allowed: bool, now: Instant) {
        let count = (handles.len() / 4).min(values.len() / 16);
        for pair in 0..count {
            let at = pair * 4;
            let raw = u32::from_le_bytes([handles[at], handles[at + 1], handles[at + 2], handles[at + 3]]);
            let value = VibrationValue::from_le_bytes(&values[pair * 16..pair * 16 + 16]);
            self.submit(raw, value, allowed, now);
        }
    }

    pub fn submit_erm(&mut self, raw: u32, command: u64, allowed: bool, now: Instant) {
        let Some(handle) = decode(raw).filter(|handle| handle.style == STYLE_GC) else {
            return;
        };
        if let Some(npad) = npad_slot(handle.npad) {
            self.erm[npad] = command;
        }
        let value = if allowed && command == 1 { VibrationValue::FULL } else { VibrationValue::DEFAULT };
        self.store(raw, handle, 0, value, now);
        self.store(raw, handle, 1, value, now);
    }

    pub fn submit_bool(&mut self, raw: u32, on: bool, allowed: bool, now: Instant) {
        let Some(handle) = decode(raw).filter(|handle| handle.style == STYLE_N64) else {
            return;
        };
        let value = if allowed && on { VibrationValue::FULL } else { VibrationValue::DEFAULT };
        self.store(raw, handle, 0, value, now);
        self.store(raw, handle, 1, value, now);
    }

    pub fn actual(&self, raw: u32) -> VibrationValue {
        decode(raw)
            .and_then(|handle| slot_index(handle, handle.index))
            .map(|slot| &self.slots[slot])
            .filter(|slot| slot.last_write.is_some())
            .map_or(VibrationValue::DEFAULT, |slot| slot.latest)
    }

    pub fn erm_command(&self, raw: u32) -> u64 {
        decode(raw)
            .filter(|handle| handle.style == STYLE_GC)
            .and_then(|handle| npad_slot(handle.npad))
            .map_or(0, |npad| self.erm[npad])
    }

    pub fn silence(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.latest = VibrationValue::DEFAULT;
            slot.peak = VibrationValue::DEFAULT;
        }
    }

    pub fn take(&mut self, npad: u8, now: Instant) -> [VibrationValue; 2] {
        let taken = match npad_slot(npad) {
            Some(npad) => [self.slots[npad * 2].take(now), self.slots[npad * 2 + 1].take(now)],
            None => [VibrationValue::DEFAULT; 2],
        };
        for slot in self.slots.iter_mut() {
            slot.peak = slot.latest;
        }
        taken
    }

    pub fn reset(&mut self) {
        *self = Mailbox::new();
    }

    fn trace_line(&mut self, batch: bool, now: Instant) -> Option<String> {
        if !trace_enabled() {
            return None;
        }
        self.trace.line(batch, now)
    }
}

impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}

static MAILBOX: Mutex<Mailbox> = Mutex::new(Mailbox::new());

fn log_trace(line: Option<String>) {
    if let Some(line) = line {
        log::info!("{}", line);
    }
}

pub fn submit(raw: u32, v: VibrationValue, allowed: bool) {
    let now = Instant::now();
    let line = {
        let mut mailbox = MAILBOX.lock();
        mailbox.submit(raw, v, allowed, now);
        mailbox.trace_line(false, now)
    };
    log_trace(line);
}

pub fn submit_batch(handles: &[u8], values: &[u8], allowed: bool) {
    let now = Instant::now();
    let line = {
        let mut mailbox = MAILBOX.lock();
        mailbox.submit_batch(handles, values, allowed, now);
        mailbox.trace_line(true, now)
    };
    log_trace(line);
}

pub fn submit_erm(raw: u32, command: u64, allowed: bool) {
    let now = Instant::now();
    let line = {
        let mut mailbox = MAILBOX.lock();
        mailbox.submit_erm(raw, command, allowed, now);
        mailbox.trace_line(false, now)
    };
    log_trace(line);
}

pub fn submit_bool(raw: u32, on: bool, allowed: bool) {
    let now = Instant::now();
    let line = {
        let mut mailbox = MAILBOX.lock();
        mailbox.submit_bool(raw, on, allowed, now);
        mailbox.trace_line(false, now)
    };
    log_trace(line);
}

pub fn actual(raw: u32) -> VibrationValue {
    MAILBOX.lock().actual(raw)
}

pub fn erm_command(raw: u32) -> u64 {
    MAILBOX.lock().erm_command(raw)
}

pub fn silence() {
    MAILBOX.lock().silence();
}

pub fn reset() {
    MAILBOX.lock().reset();
}

pub fn take_player1() -> [VibrationValue; 2] {
    let npad = crate::hid_state::player1_npad_id();
    let now = Instant::now();
    MAILBOX.lock().take(npad, now)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn amp(amp_low: f32) -> VibrationValue {
        VibrationValue { amp_low, ..VibrationValue::DEFAULT }
    }

    fn ms(t0: Instant, millis: u64) -> Instant {
        t0 + Duration::from_millis(millis)
    }

    fn batch(entries: &[(u32, VibrationValue)]) -> (Vec<u8>, Vec<u8>) {
        let handles = entries.iter().flat_map(|(raw, _)| raw.to_le_bytes()).collect();
        let values = entries.iter().flat_map(|(_, value)| value.to_le_bytes()).collect();
        (handles, values)
    }

    #[test]
    fn decode_packs_style_npad_and_index() {
        assert_eq!(decode(0x0001_0007), Some(DeviceHandle { style: 7, npad: 0, index: 1 }));
        assert_eq!(decode(0x0000_2004), Some(DeviceHandle { style: 4, npad: 0x20, index: 0 }));
        assert_eq!(decode(0x0000_1003), Some(DeviceHandle { style: 3, npad: 0x10, index: 0 }));
        assert_eq!(decode(0x0000_0021), Some(DeviceHandle { style: 0x21, npad: 0, index: 0 }));
        for raw in [0x0000_0803, 0x0002_0003, 0x0000_0000, 0x0000_0009] {
            assert_eq!(decode(raw), None, "{:#x}", raw);
        }
    }

    #[test]
    fn device_info_classes() {
        assert_eq!(device_info(0x3), (1, 1));
        assert_eq!(device_info(0x1_0007), (1, 2));
        assert_eq!(device_info(0x8), (2, 0));
        assert_eq!(device_info(0xD), (3, 0));
        assert_eq!(device_info(0x20), (0, 0));
        assert_eq!(device_info(0x803), (0, 0));
    }

    #[test]
    fn batch_uses_shorter_list() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        let (mut handles, values) = batch(&[(0x5, amp(0.4)), (0x1_0005, amp(0.6))]);
        handles.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        mailbox.submit_batch(&handles, &values, true, t0);
        assert_eq!(mailbox.actual(0x5), amp(0.4));
        assert_eq!(mailbox.actual(0x1_0005), amp(0.6));

        let mut mailbox = Mailbox::new();
        let (handles, _) = batch(&[(0x5, amp(0.4)), (0x1_0005, amp(0.6)), (0x106, amp(0.8))]);
        let (_, values) = batch(&[(0x5, amp(0.4)), (0x1_0005, amp(0.6))]);
        mailbox.submit_batch(&handles, &values, true, t0);
        assert_eq!(mailbox.actual(0x5), amp(0.4));
        assert_eq!(mailbox.actual(0x1_0005), amp(0.6));
        assert_eq!(mailbox.actual(0x106), VibrationValue::DEFAULT);
        assert_eq!(mailbox.take(1, t0), [VibrationValue::DEFAULT; 2]);
    }

    #[test]
    fn peak_survives_until_taken() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.8), true, t0);
        mailbox.submit(0x5, amp(0.0), true, ms(t0, 5));
        assert_eq!(mailbox.take(0, ms(t0, 10))[0].amp_low, 0.8);
        assert_eq!(mailbox.take(0, ms(t0, 15))[0].amp_low, 0.0);
    }

    #[test]
    fn stale_reads_default() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.8), true, t0);
        assert_eq!(mailbox.take(0, ms(t0, 3000)), [VibrationValue::DEFAULT; 2]);
    }

    #[test]
    fn stale_peak_not_replayed() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.9), true, t0);
        mailbox.submit(0x5, amp(0.2), true, ms(t0, 3000));
        assert_eq!(mailbox.take(0, ms(t0, 3001))[0].amp_low, 0.2);
    }

    #[test]
    fn undrained_npad_peak_is_not_replayed() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x2004, amp(1.0), true, t0);
        for i in 1..=200 {
            mailbox.submit(0x2004, amp(0.0), true, ms(t0, i * 5));
            assert_eq!(mailbox.take(0, ms(t0, i * 5)), [VibrationValue::DEFAULT; 2]);
        }
        assert_eq!(mailbox.take(0x20, ms(t0, 1001))[0].amp_low, 0.0);
    }

    #[test]
    fn other_npads_are_invisible() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x2004, amp(0.7), true, t0);
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::DEFAULT; 2]);
        assert_eq!(mailbox.take(0x20, ms(t0, 1))[0].amp_low, 0.7);
        assert_eq!(mailbox.take(8, ms(t0, 1)), [VibrationValue::DEFAULT; 2]);
    }

    #[test]
    fn disallowed_stores_default() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.7), false, t0);
        assert_eq!(mailbox.actual(0x5), VibrationValue::DEFAULT);
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::DEFAULT; 2]);
        let (handles, values) = batch(&[(0x1_0005, amp(0.7))]);
        mailbox.submit_batch(&handles, &values, false, ms(t0, 2));
        assert_eq!(mailbox.actual(0x1_0005), VibrationValue::DEFAULT);
    }

    #[test]
    fn le_bytes_round_trip() {
        let value = VibrationValue { amp_low: 0.25, freq_low: 180.5, amp_high: 0.75, freq_high: 330.0 };
        assert_eq!(VibrationValue::from_le_bytes(&value.to_le_bytes()), value);
        assert_eq!(VibrationValue::from_le_bytes(&[0; 8]), VibrationValue::DEFAULT);
        let words = [0.5f32.to_bits(), 100.0f32.to_bits(), 0.0f32.to_bits(), 320.0f32.to_bits()];
        let from_words = VibrationValue::from_words(words);
        assert_eq!(from_words, VibrationValue { amp_low: 0.5, freq_low: 100.0, amp_high: 0.0, freq_high: 320.0 });
        assert_eq!(&from_words.to_le_bytes()[0..4], &0.5f32.to_le_bytes());
    }

    #[test]
    fn sanitize() {
        let dirty = VibrationValue { amp_low: f32::NAN, freq_low: -5.0, amp_high: 1.7, freq_high: -5.0 };
        assert_eq!(dirty.sanitized(), VibrationValue { amp_low: 0.0, freq_low: 160.0, amp_high: 1.0, freq_high: 320.0 });
        let infinite = VibrationValue { amp_low: -1.0, freq_low: f32::INFINITY, amp_high: 0.3, freq_high: f32::INFINITY };
        assert_eq!(infinite.sanitized(), VibrationValue { amp_low: 0.0, freq_low: 160.0, amp_high: 0.3, freq_high: 320.0 });
        assert!(VibrationValue::DEFAULT.is_silent());
        assert!(VibrationValue { amp_low: f32::NAN, ..VibrationValue::DEFAULT }.is_silent());
        assert!(!amp(0.1).is_silent());
        assert_eq!(VibrationValue::FULL.scaled(0.5), VibrationValue { amp_low: 0.5, amp_high: 0.5, ..VibrationValue::FULL });
        let loud_low = VibrationValue { amp_low: 0.9, freq_low: 100.0, amp_high: 0.1, freq_high: 400.0 };
        let loud_high = VibrationValue { amp_low: 0.2, freq_low: 150.0, amp_high: 0.8, freq_high: 500.0 };
        assert_eq!(
            loud_low.louder_bands(loud_high),
            VibrationValue { amp_low: 0.9, freq_low: 100.0, amp_high: 0.8, freq_high: 500.0 }
        );
    }

    #[test]
    fn silence_mutes() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.7), true, t0);
        mailbox.silence();
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::DEFAULT; 2]);
        assert_eq!(mailbox.actual(0x5), VibrationValue::DEFAULT);
    }

    #[test]
    fn reset_clears_slots_and_erm() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit(0x5, amp(0.7), true, t0);
        mailbox.submit_erm(0x8, 1, true, t0);
        mailbox.reset();
        assert_eq!(mailbox.actual(0x5), VibrationValue::DEFAULT);
        assert_eq!(mailbox.erm_command(0x8), 0);
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::DEFAULT; 2]);
        assert!(mailbox.slots.iter().all(|slot| slot.last_write.is_none() && slot.interval_ms == 250.0));
    }

    #[test]
    fn erm() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit_erm(0x8, 1, true, t0);
        assert_eq!(mailbox.actual(0x8), VibrationValue::FULL);
        assert_eq!(mailbox.erm_command(0x8), 1);
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::FULL; 2]);
        mailbox.submit_erm(0x8, 2, true, ms(t0, 2));
        assert_eq!(mailbox.actual(0x8), VibrationValue::DEFAULT);
        assert_eq!(mailbox.erm_command(0x8), 2);
        mailbox.submit_erm(0x8, 1, false, ms(t0, 3));
        assert_eq!(mailbox.actual(0x8), VibrationValue::DEFAULT);

        let mut mailbox = Mailbox::new();
        mailbox.submit_erm(0x3, 1, true, t0);
        assert_eq!(mailbox.erm_command(0x3), 0);
        assert_eq!(mailbox.actual(0x3), VibrationValue::DEFAULT);
    }

    #[test]
    fn bool_n64_only() {
        let t0 = Instant::now();
        let mut mailbox = Mailbox::new();
        mailbox.submit_bool(0xD, true, true, t0);
        assert_eq!(mailbox.actual(0xD), VibrationValue::FULL);
        assert_eq!(mailbox.take(0, ms(t0, 1)), [VibrationValue::FULL; 2]);
        mailbox.submit_bool(0xD, false, true, ms(t0, 2));
        assert_eq!(mailbox.actual(0xD), VibrationValue::DEFAULT);

        let mut mailbox = Mailbox::new();
        mailbox.submit_bool(0x3, true, true, t0);
        assert_eq!(mailbox.actual(0x3), VibrationValue::DEFAULT);
    }

    #[test]
    fn stale_window_adapts() {
        let t0 = Instant::now();
        let mut slot = Slot::EMPTY;
        slot.write(amp(0.5), t0);
        assert_eq!(slot.stale_after(), Duration::from_millis(2000));
        let mut slot = Slot::EMPTY;
        for i in 0..40 {
            slot.write(amp(0.5), ms(t0, i * 5));
        }
        assert_eq!(slot.stale_after(), Duration::from_millis(150));
    }
}

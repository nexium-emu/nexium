use nexium_core::hid_vibration::VibrationValue;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const NEUTRAL: [u8; 4] = [0x00, 0x01, 0x40, 0x40];
pub const N64_NEUTRAL: [u8; 4] = [0x00, 0x00, 0x01, 0x40];
pub const SDL_FIXED_HZ: f32 = 149.933_41;
pub const PREVIEW: VibrationValue = VibrationValue {
    amp_low: 0.5,
    freq_low: 160.0,
    amp_high: 0.5,
    freq_high: 320.0,
};

pub fn neutral_for(product: u16) -> [u8; 4] {
    if product == 0x2019 {
        N64_NEUTRAL
    } else {
        NEUTRAL
    }
}

fn levels() -> &'static [f32; 101] {
    static LEVELS: OnceLock<[f32; 101]> = OnceLock::new();
    LEVELS.get_or_init(|| {
        let mut table = [0.0f32; 101];
        table[1] = 2.0 / 255.0;
        for (index, level) in table.iter_mut().enumerate().skip(2) {
            let i = index as f64;
            let exponent = if index <= 15 {
                (i - 15.125) / 4.0
            } else if index <= 31 {
                (i - 15.5) / 16.0
            } else {
                i / 32.0
            };
            *level = (exponent.exp2() / 8.7) as f32;
        }
        table
    })
}

pub fn amp_index(a: f32) -> u32 {
    if !(a > 0.0) {
        return 0;
    }
    let table = levels();
    (1..=100).find(|&i| table[i] >= a).unwrap_or(100) as u32
}

fn freq_code(f: f32) -> i32 {
    (32.0 * (f / 10.0).log2()).round() as i32
}

pub fn encode_side(v: VibrationValue, neutral: [u8; 4]) -> [u8; 4] {
    let v = v.sanitized();
    let high = amp_index(v.amp_high);
    let low = amp_index(v.amp_low);
    if high == 0 && low == 0 {
        return neutral;
    }
    let hf = (4 * (freq_code(v.freq_high).clamp(97, 223) - 96)) as u32;
    let lf = (freq_code(v.freq_low).clamp(65, 191) - 64) as u32;
    [
        (hf & 0xFF) as u8,
        ((2 * high) | ((hf >> 8) & 1)) as u8,
        (((low & 1) << 7) | lf) as u8,
        (0x40 + (low >> 1)) as u8,
    ]
}

pub fn protect(v: VibrationValue) -> VibrationValue {
    limit(v, 1.0)
}

pub fn limit(v: VibrationValue, cap: f32) -> VibrationValue {
    let sum = v.amp_low + v.amp_high;
    if sum > cap {
        v.scaled(cap / sum)
    } else {
        v
    }
}

pub fn report(counter: u8, left: [u8; 4], right: [u8; 4]) -> [u8; 10] {
    [
        0x10,
        counter & 0x0F,
        left[0],
        left[1],
        left[2],
        left[3],
        right[0],
        right[1],
        right[2],
        right[3],
    ]
}

fn unit(level: u16) -> f32 {
    f32::from(level) / 65535.0
}

fn level_u16(x: f32) -> u16 {
    (x.clamp(0.0, 1.0) * 65535.0).round() as u16
}

pub fn pulse_value(low: u16, high: u16) -> VibrationValue {
    VibrationValue {
        amp_low: unit(low),
        freq_low: SDL_FIXED_HZ,
        amp_high: unit(high),
        freq_high: SDL_FIXED_HZ,
    }
}

pub fn pulse_pair(low: u16, high: u16) -> [VibrationValue; 2] {
    [
        VibrationValue {
            amp_low: unit(low),
            freq_low: SDL_FIXED_HZ,
            amp_high: 0.0,
            freq_high: SDL_FIXED_HZ,
        },
        VibrationValue {
            amp_low: 0.0,
            freq_low: SDL_FIXED_HZ,
            amp_high: unit(high),
            freq_high: SDL_FIXED_HZ,
        },
    ]
}

pub fn pair_levels(l: VibrationValue, r: VibrationValue) -> (u16, u16) {
    (
        level_u16((l.amp_low + l.amp_high).min(1.0)),
        level_u16((r.amp_low + r.amp_high).min(1.0)),
    )
}

pub fn band_levels(l: VibrationValue, r: VibrationValue, curve: bool) -> (u16, u16) {
    let shape = |x: f32| if curve { (x + x.powf(0.35)) / 2.0 } else { x };
    let low_weight = |f: f32| {
        if f <= 140.0 {
            1.0
        } else {
            (1.0 - (f - 140.0) / 400.0).max(0.3)
        }
    };
    let high_weight = |f: f32| {
        if f <= 200.0 {
            1.0
        } else {
            (1.0 - (f - 200.0) / 700.0).max(0.3)
        }
    };
    let low = (shape(l.amp_low) * low_weight(l.freq_low)).max(shape(r.amp_low) * low_weight(r.freq_low));
    let high =
        (shape(l.amp_high) * high_weight(l.freq_high)).max(shape(r.amp_high) * high_weight(r.freq_high));
    (level_u16(low), level_u16(high))
}

pub trait Frame: Copy + PartialEq {
    const SILENT: Self;
    fn merge(self, other: Self) -> Self;
    fn is_silent(&self) -> bool;
}

impl Frame for VibrationValue {
    const SILENT: Self = VibrationValue::DEFAULT;

    fn merge(self, other: Self) -> Self {
        self.louder_bands(other)
    }

    fn is_silent(&self) -> bool {
        VibrationValue::is_silent(self)
    }
}

impl Frame for [VibrationValue; 2] {
    const SILENT: Self = [VibrationValue::DEFAULT; 2];

    fn merge(self, other: Self) -> Self {
        [self[0].louder_bands(other[0]), self[1].louder_bands(other[1])]
    }

    fn is_silent(&self) -> bool {
        VibrationValue::is_silent(&self[0]) && VibrationValue::is_silent(&self[1])
    }
}

impl Frame for (u16, u16) {
    const SILENT: Self = (0, 0);

    fn merge(self, other: Self) -> Self {
        (self.0.max(other.0), self.1.max(other.1))
    }

    fn is_silent(&self) -> bool {
        self.0 == 0 && self.1 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pace {
    pub spacing: Duration,
    pub keepalive: Duration,
    pub stop_repeat: Option<Duration>,
}

pub const HID_PACE: Pace = Pace {
    spacing: Duration::from_millis(30),
    keepalive: Duration::from_millis(50),
    stop_repeat: Some(Duration::from_millis(50)),
};

pub const SDL_HD_PACE: Pace = Pace {
    spacing: Duration::from_millis(50),
    keepalive: Duration::from_millis(50),
    stop_repeat: Some(Duration::from_millis(60)),
};

pub const SDL_PACE: Pace = Pace {
    spacing: Duration::from_millis(50),
    keepalive: Duration::from_millis(100),
    stop_repeat: None,
};

pub const SDL_QUIET_PACE: Pace = Pace {
    spacing: Duration::from_millis(30),
    keepalive: Duration::from_millis(100),
    stop_repeat: None,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step<T> {
    Idle,
    Write(T),
    Stop,
}

pub struct Pacer<T: Frame> {
    pace: Pace,
    window: T,
    last_at: Option<Instant>,
    sent: Option<T>,
    repeat_at: Option<Instant>,
}

impl<T: Frame> Pacer<T> {
    pub fn new(pace: Pace) -> Self {
        Self {
            pace,
            window: T::SILENT,
            last_at: None,
            sent: None,
            repeat_at: None,
        }
    }

    pub fn step(&mut self, now: Instant, frame: T) -> Step<T> {
        self.window = self.window.merge(frame);
        if self
            .last_at
            .is_some_and(|at| now.saturating_duration_since(at) < self.pace.spacing)
        {
            return Step::Idle;
        }
        let window = std::mem::replace(&mut self.window, T::SILENT);
        if !window.is_silent() {
            let due = self.sent != Some(window)
                || self
                    .last_at
                    .map_or(true, |at| now.saturating_duration_since(at) >= self.pace.keepalive);
            if !due {
                return Step::Idle;
            }
            self.sent = Some(window);
            self.last_at = Some(now);
            self.repeat_at = None;
            return Step::Write(window);
        }
        if self.sent.take().is_some() {
            self.last_at = Some(now);
            self.repeat_at = self.pace.stop_repeat.map(|delay| now + delay);
            return Step::Stop;
        }
        if self.repeat_at.is_some_and(|at| now >= at) {
            self.repeat_at = None;
            self.last_at = Some(now);
            return Step::Stop;
        }
        Step::Idle
    }

    pub fn busy(&self) -> bool {
        self.sent.is_some() || self.repeat_at.is_some()
    }

    pub fn cancel_repeat(&mut self) {
        self.repeat_at = None;
    }

    pub fn set_pace(&mut self, pace: Pace) {
        self.pace = pace;
    }

    pub fn owe_stop(&mut self) {
        self.sent = Some(T::SILENT);
    }

    pub fn last_at(&self) -> Option<Instant> {
        self.last_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(amp_low: f32, freq_low: f32, amp_high: f32, freq_high: f32) -> VibrationValue {
        VibrationValue {
            amp_low,
            freq_low,
            amp_high,
            freq_high,
        }
    }

    fn enc(v: VibrationValue) -> [u8; 4] {
        encode_side(v, NEUTRAL)
    }

    #[test]
    fn encoder_matches_the_reference_vectors() {
        let cases = [
            (VibrationValue::DEFAULT, [0x00, 0x01, 0x40, 0x40]),
            (value(0.0, 5.0, 0.0, 5000.0), [0x00, 0x01, 0x40, 0x40]),
            (value(0.5, 160.0, 0.5, 320.0), [0x00, 0x89, 0x40, 0x62]),
            (value(0.3, 100.0, 0.0, 320.0), [0x00, 0x01, 0xAA, 0x56]),
            (value(1.0, 149.93, 1.0, 149.93), [0x74, 0xC8, 0x3D, 0x72]),
            (value(0.0, 160.0, 0.8, 1000.0), [0xD4, 0xB5, 0x40, 0x40]),
            (value(0.5, 10.0, 0.5, 20.0), [0x04, 0x88, 0x01, 0x62]),
            (value(0.5, 5000.0, 0.5, 5000.0), [0xFC, 0x89, 0x7F, 0x62]),
            (value(f32::NAN, f32::NAN, -1.0, f32::INFINITY), [0x00, 0x01, 0x40, 0x40]),
            (value(7.0, 160.0, 0.0, 320.0), [0x00, 0x01, 0x40, 0x72]),
        ];
        for (input, bytes) in cases {
            assert_eq!(enc(input), bytes, "{input:?}");
        }
    }

    #[test]
    fn limit_comes_before_strength() {
        assert_eq!(enc(protect(VibrationValue::FULL)), [0x00, 0x89, 0x40, 0x62]);
        assert_eq!(enc(protect(value(1.0, 149.93, 1.0, 149.93))), [0x74, 0x88, 0x3D, 0x62]);
        assert_eq!(enc(protect(VibrationValue::FULL).scaled(0.8)), [0x00, 0x75, 0x40, 0x5D]);
        assert_eq!(enc(protect(VibrationValue::FULL).scaled(0.5)), [0x00, 0x49, 0x40, 0x52]);
        assert_eq!(enc(protect(VibrationValue::FULL).scaled(0.3)), [0x00, 0x2D, 0x40, 0x4B]);
        assert_eq!(enc(value(0.8, 160.0, 0.0, 320.0).scaled(0.5)), [0x00, 0x01, 0x40, 0x5D]);
        assert_eq!(enc(PREVIEW.scaled(0.4)), [0x00, 0x3B, 0xC0, 0x4E]);
        let window = value(1.0, 160.0, 0.0, 320.0).louder_bands(value(0.0, 160.0, 1.0, 320.0));
        assert_eq!(enc(protect(window)), [0x00, 0x89, 0x40, 0x62]);
        assert_eq!(enc(limit(window, 0.5)), [0x00, 0x49, 0x40, 0x52]);
        let merged = value(0.8, 160.0, 0.0, 320.0).louder_bands(value(0.0, 160.0, 0.8, 320.0));
        assert_eq!(enc(limit(merged, 0.8)), [0x00, 0x75, 0x40, 0x5D]);
        assert_eq!(limit(PREVIEW.scaled(0.4), 0.4), PREVIEW.scaled(0.4));
    }

    #[test]
    fn pulses_match_sdl_packets() {
        assert_eq!(enc(pulse_value(20000, 20000)), [0x74, 0x5C, 0x3D, 0x57]);
        assert_eq!(enc(pulse_value(28000, 28000)), [0x74, 0x7A, 0xBD, 0x5E]);
        assert_eq!(enc(pulse_value(10000, 10000)), [0x74, 0x2E, 0xBD, 0x4B]);
        let pair = pulse_pair(20000, 20000);
        assert_eq!(enc(pair[0]), [0x74, 0x00, 0x3D, 0x57]);
        assert_eq!(enc(pair[1]), [0x74, 0x5C, 0x3D, 0x40]);
    }

    #[test]
    fn neutral_and_report_layout() {
        assert_eq!(encode_side(VibrationValue::DEFAULT, N64_NEUTRAL), [0x00, 0x00, 0x01, 0x40]);
        assert_eq!(neutral_for(0x2019), N64_NEUTRAL);
        assert_eq!(neutral_for(0x2009), NEUTRAL);
        assert_eq!(
            report(0x13, enc(value(0.5, 160.0, 0.5, 320.0)), NEUTRAL),
            [0x10, 0x03, 0x00, 0x89, 0x40, 0x62, 0x00, 0x01, 0x40, 0x40]
        );
    }

    #[test]
    fn amplitude_levels_follow_the_closed_form() {
        let cases = [
            (0.0, 0),
            (2.0 / 255.0, 1),
            (1e-6, 1),
            (0.0112, 2),
            (0.0119, 3),
            (0.3, 45),
            (0.5, 68),
            (0.8, 90),
            (1.0, 100),
        ];
        for (amplitude, index) in cases {
            assert_eq!(amp_index(amplitude), index, "{amplitude}");
        }
        let table = levels();
        assert!(table.windows(2).all(|pair| pair[0] < pair[1]));
        for (index, sdl) in [
            (2, 775.0),
            (15, 7372.0),
            (16, 7698.0),
            (31, 14744.0),
            (32, 15067.0),
            (46, 20405.0),
            (64, 30134.0),
            (99, 64315.0),
        ] {
            assert!((table[index] - sdl / 65535.0).abs() < 2e-4, "{index}");
        }
    }

    #[test]
    fn standard_rumble_mappings() {
        let close = |actual: (u16, u16), expected: (u16, u16)| {
            assert!(
                actual.0.abs_diff(expected.0) <= 1 && actual.1.abs_diff(expected.1) <= 1,
                "{actual:?} vs {expected:?}"
            );
        };
        let left = value(1.0, 140.0, 0.0, 320.0);
        let right = value(0.0, 160.0, 0.5, 200.0);
        close(band_levels(left, right, true), (65535, 42093));
        close(band_levels(left, right, false), (65535, 32768));
        close(band_levels(VibrationValue::DEFAULT, value(0.0, 160.0, 0.5, 550.0), true), (0, 21046));
        close(band_levels(value(0.5, 160.0, 0.0, 320.0), VibrationValue::DEFAULT, true), (39988, 0));
        close(pair_levels(value(0.5, 160.0, 0.25, 320.0), VibrationValue::DEFAULT), (49151, 0));
        close(pair_levels(protect(VibrationValue::FULL).scaled(0.5), VibrationValue::DEFAULT), (32768, 0));
        close(pair_levels(protect(VibrationValue::FULL).scaled(0.7), VibrationValue::DEFAULT), (45875, 0));
    }

    fn run<T: Frame + std::fmt::Debug>(
        pace: Pace,
        frames: impl Fn(u64) -> T,
        until: u64,
        cancel_at: Option<u64>,
    ) -> Vec<(u64, Step<T>)> {
        let start = Instant::now();
        let mut pacer = Pacer::new(pace);
        let mut out = Vec::new();
        for t in (0..=until).step_by(5) {
            if cancel_at == Some(t) {
                pacer.cancel_repeat();
            }
            let step = pacer.step(start + Duration::from_millis(t), frames(t));
            if step != Step::Idle {
                out.push((t, step));
            }
        }
        out
    }

    fn amp(level: f32) -> VibrationValue {
        value(level, 160.0, level, 320.0)
    }

    #[test]
    fn hid_pace_writes_and_stops() {
        let a = amp(0.5);
        let silent = VibrationValue::DEFAULT;
        assert_eq!(run(HID_PACE, |_| a, 0, None), [(0, Step::Write(a))]);
        assert_eq!(
            run(HID_PACE, |t| if t < 10 { a } else { amp(0.7) }, 60, None),
            [(0, Step::Write(a)), (30, Step::Write(amp(0.7)))]
        );
        assert_eq!(
            run(HID_PACE, |t| if t == 15 { amp(0.9) } else { a }, 60, None),
            [(0, Step::Write(a)), (30, Step::Write(amp(0.9))), (60, Step::Write(a))]
        );
        assert_eq!(
            run(HID_PACE, |t| if t == 0 { a } else { silent }, 200, None),
            [(0, Step::Write(a)), (30, Step::Stop), (80, Step::Stop)]
        );
        assert_eq!(
            run(HID_PACE, |_| a, 160, None),
            [(0, Step::Write(a)), (50, Step::Write(a)), (100, Step::Write(a)), (150, Step::Write(a))]
        );
        assert_eq!(
            run(HID_PACE, |t| if t < 35 { a } else { silent }, 150, None),
            [(0, Step::Write(a)), (35, Step::Stop), (85, Step::Stop)]
        );
        assert_eq!(
            run(HID_PACE, |t| if t < 45 { a } else { silent }, 150, None),
            [(0, Step::Write(a)), (45, Step::Stop), (95, Step::Stop)]
        );
        assert_eq!(
            run(HID_PACE, |t| if t < 60 { a } else { silent }, 200, None),
            [(0, Step::Write(a)), (50, Step::Write(a)), (85, Step::Stop), (135, Step::Stop)]
        );
        assert_eq!(
            run(HID_PACE, |t| if t == 0 { a } else { silent }, 200, Some(40)),
            [(0, Step::Write(a)), (30, Step::Stop)]
        );
    }

    #[test]
    fn sdl_paces_write_and_stop() {
        let a = [amp(0.5), VibrationValue::DEFAULT];
        assert_eq!(
            run(SDL_HD_PACE, |t| if t == 0 { a } else { [VibrationValue::DEFAULT; 2] }, 200, None),
            [(0, Step::Write(a)), (50, Step::Stop), (110, Step::Stop)]
        );
        let level = (20000u16, 20000u16);
        assert_eq!(
            run(SDL_PACE, |_| level, 260, None),
            [(0, Step::Write(level)), (100, Step::Write(level)), (200, Step::Write(level))]
        );
        assert_eq!(
            run(SDL_PACE, |t| if t < 35 { level } else { (0, 0) }, 150, None),
            [(0, Step::Write(level)), (55, Step::Stop)]
        );
        assert_eq!(
            run(SDL_QUIET_PACE, |t| if t < 35 { level } else { (0, 0) }, 150, None),
            [(0, Step::Write(level)), (35, Step::Stop)]
        );
    }

    #[test]
    fn pacer_goes_idle_after_the_repeat_stop() {
        let start = Instant::now();
        let mut pacer = Pacer::new(HID_PACE);
        let at = |ms: u64| start + Duration::from_millis(ms);
        assert_eq!(pacer.step(at(0), amp(0.5)), Step::Write(amp(0.5)));
        assert!(pacer.busy());
        assert_eq!(pacer.step(at(30), VibrationValue::DEFAULT), Step::Stop);
        assert!(pacer.busy());
        assert_eq!(pacer.step(at(80), VibrationValue::DEFAULT), Step::Stop);
        assert!(!pacer.busy());
        assert_eq!(pacer.last_at(), Some(at(80)));
    }

    #[test]
    fn owed_stop_goes_out_on_the_first_silent_step() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let silent = [VibrationValue::DEFAULT; 2];
        let mut pacer = Pacer::new(SDL_HD_PACE);
        pacer.owe_stop();
        assert!(pacer.busy());
        assert_eq!(pacer.step(at(0), silent), Step::Stop);
        assert_eq!(pacer.step(at(60), silent), Step::Stop);
        assert!(!pacer.busy());
        let a = [amp(0.5), VibrationValue::DEFAULT];
        let mut pacer = Pacer::new(SDL_HD_PACE);
        pacer.owe_stop();
        assert_eq!(pacer.step(at(0), a), Step::Write(a));
    }
}

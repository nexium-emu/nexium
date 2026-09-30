use nexium_core::hid_motion::{
    self, AxisFrame, BiasMode, HostKind, ImuPairer, MotionSample, MotionSource, RawImuSample,
    DEFAULT_PERIOD_NS,
};
use sdl3::sys::events::{
    SDL_Event, SDL_FlushEvents, SDL_PeepEvents, SDL_EVENT_FIRST, SDL_EVENT_GAMEPAD_SENSOR_UPDATE,
    SDL_EVENT_LAST, SDL_GETEVENT,
};
use sdl3::sys::gamepad::{
    SDL_Gamepad, SDL_GamepadHasSensor, SDL_GamepadType, SDL_GetGamepadID, SDL_GetGamepadProduct,
    SDL_GetGamepadSensorDataRate, SDL_GetGamepadType, SDL_GetGamepadTypeForID,
    SDL_GetGamepadVendor, SDL_SetGamepadSensorEnabled, SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT,
    SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR, SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT,
    SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, SDL_GAMEPAD_TYPE_PS4, SDL_GAMEPAD_TYPE_PS5,
    SDL_GAMEPAD_TYPE_STANDARD,
};
use sdl3::sys::joystick::SDL_JoystickID;
use sdl3::sys::sensor::{
    SDL_SensorType, SDL_SENSOR_ACCEL, SDL_SENSOR_ACCEL_L, SDL_SENSOR_ACCEL_R, SDL_SENSOR_GYRO,
    SDL_SENSOR_GYRO_L, SDL_SENSOR_GYRO_R,
};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
struct SensorClock {
    last: Option<u64>,
    step: u64,
    accepted: u64,
    out: u64,
    holding: u8,
    seen_at: Option<Instant>,
}

impl SensorClock {
    fn hold(&mut self) {
        if self.last.is_some() {
            self.holding = 2;
        }
    }

    fn seen(&mut self, now: Instant) {
        if self
            .seen_at
            .is_some_and(|at| now.saturating_duration_since(at) >= hid_motion::STALE_AFTER)
        {
            self.hold();
        }
        self.seen_at = Some(now);
    }

    fn retime(&mut self, ts: u64) -> u64 {
        let Some(previous) = self.last.filter(|previous| ts >= *previous) else {
            self.last = Some(ts);
            self.out = ts;
            return ts;
        };
        let delta = ts - previous;
        if delta == 0 {
            return self.out;
        }
        if self.step == 0 || delta % self.step != 0 {
            self.step = delta;
            if self.holding > 0 && self.accepted != 0 {
                self.holding -= 1;
            } else {
                self.accepted = delta;
            }
        }
        self.last = Some(ts);
        let scaled = u128::from(delta) * u128::from(self.accepted) / u128::from(self.step);
        self.out = self.out.saturating_add(scaled as u64);
        self.out
    }
}

struct Stream {
    which: u32,
    accel: SDL_SensorType,
    gyro: SDL_SensorType,
    source: MotionSource,
    frame: AxisFrame,
    bias: BiasMode,
    period_ns: u64,
    pairer: ImuPairer,
    clock: Option<SensorClock>,
}

pub struct MotionCapture {
    streams: Vec<Stream>,
    samples: [Vec<MotionSample>; 3],
    raw: Vec<RawImuSample>,
    events: Vec<SDL_Event>,
    attached: Option<u32>,
}

fn is_switch2(vendor: u16, product: u16) -> bool {
    vendor == 0x057E && (0x2066..=0x2069).contains(&product)
}

fn synthetic_clock(kind: SDL_GamepadType, switch2: bool) -> bool {
    !switch2
        && (kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR
            || kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT
            || kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT
            || kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StreamPlan {
    accel: SDL_SensorType,
    gyro: SDL_SensorType,
    source: MotionSource,
    frame: AxisFrame,
    bias: BiasMode,
}

fn stream_plan(kind: SDL_GamepadType, switch2: bool, vertical: bool) -> Vec<StreamPlan> {
    let nintendo_bias = if switch2 {
        BiasMode::FactoryCalibrated
    } else {
        BiasMode::Learn
    };
    let single = |source: MotionSource, sideways: AxisFrame| StreamPlan {
        accel: SDL_SENSOR_ACCEL,
        gyro: SDL_SENSOR_GYRO,
        source,
        frame: if switch2 || !vertical {
            sideways
        } else {
            AxisFrame::Body
        },
        bias: nintendo_bias,
    };
    let primary = |bias: BiasMode| StreamPlan {
        accel: SDL_SENSOR_ACCEL,
        gyro: SDL_SENSOR_GYRO,
        source: MotionSource::Primary,
        frame: AxisFrame::Body,
        bias,
    };
    if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR {
        vec![
            StreamPlan {
                accel: SDL_SENSOR_ACCEL_L,
                gyro: SDL_SENSOR_GYRO_L,
                source: MotionSource::Left,
                frame: AxisFrame::Body,
                bias: nintendo_bias,
            },
            StreamPlan {
                accel: SDL_SENSOR_ACCEL_R,
                gyro: SDL_SENSOR_GYRO_R,
                source: MotionSource::Right,
                frame: AxisFrame::Body,
                bias: nintendo_bias,
            },
        ]
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT {
        vec![single(MotionSource::Left, AxisFrame::SidewaysLeft)]
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT {
        vec![single(MotionSource::Right, AxisFrame::SidewaysRight)]
    } else if kind == SDL_GAMEPAD_TYPE_PS4 || kind == SDL_GAMEPAD_TYPE_PS5 {
        vec![primary(BiasMode::FactoryCalibrated)]
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO {
        vec![primary(nintendo_bias)]
    } else {
        vec![primary(BiasMode::Learn)]
    }
}

pub fn motion_rank(kind: SDL_GamepadType) -> Option<u8> {
    if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR {
        Some(0)
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT
        || kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT
    {
        Some(1)
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO {
        Some(2)
    } else if kind == SDL_GAMEPAD_TYPE_PS5 {
        Some(3)
    } else if kind == SDL_GAMEPAD_TYPE_PS4 {
        Some(4)
    } else if kind == SDL_GAMEPAD_TYPE_STANDARD {
        Some(5)
    } else {
        None
    }
}

pub fn motion_rank_for_id(id: u32) -> Option<u8> {
    motion_rank(unsafe { SDL_GetGamepadTypeForID(SDL_JoystickID(id)) })
}

pub fn host_kind_for(kind: SDL_GamepadType) -> HostKind {
    if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT {
        HostKind::JoyConLeft
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT {
        HostKind::JoyConRight
    } else if kind == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR {
        HostKind::JoyConPair
    } else {
        HostKind::Other
    }
}

pub fn host_kind_for_id(id: u32) -> HostKind {
    host_kind_for(unsafe { SDL_GetGamepadTypeForID(SDL_JoystickID(id)) })
}

impl MotionCapture {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            samples: [Vec::new(), Vec::new(), Vec::new()],
            raw: Vec::new(),
            events: (0..256).map(|_| unsafe { std::mem::zeroed() }).collect(),
            attached: None,
        }
    }

    pub fn attached(&self) -> Option<u32> {
        self.attached
    }

    pub fn attach(&mut self, pad: *mut SDL_Gamepad, name: &str) -> bool {
        self.detach();
        if pad.is_null() {
            return false;
        }
        let which = unsafe { SDL_GetGamepadID(pad) }.0;
        let kind = unsafe { SDL_GetGamepadType(pad) };
        let switch2 = is_switch2(unsafe { SDL_GetGamepadVendor(pad) }, unsafe {
            SDL_GetGamepadProduct(pad)
        });
        let vertical =
            sdl3::hint::get("SDL_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS").as_deref() == Some("1");
        let synthetic = synthetic_clock(kind, switch2);
        for plan in stream_plan(kind, switch2, vertical) {
            let enabled = unsafe {
                SDL_GamepadHasSensor(pad, plan.accel)
                    && SDL_GamepadHasSensor(pad, plan.gyro)
                    && SDL_SetGamepadSensorEnabled(pad, plan.accel, true)
                    && SDL_SetGamepadSensorEnabled(pad, plan.gyro, true)
            };
            if !enabled {
                continue;
            }
            let rate = unsafe { SDL_GetGamepadSensorDataRate(pad, plan.gyro) };
            let period_ns = if rate.is_finite() && rate > 0.0 {
                ((1e9 / f64::from(rate)).round() as u64).max(1)
            } else {
                DEFAULT_PERIOD_NS
            };
            self.streams.push(Stream {
                which,
                accel: plan.accel,
                gyro: plan.gyro,
                source: plan.source,
                frame: plan.frame,
                bias: plan.bias,
                period_ns,
                pairer: ImuPairer::default(),
                clock: synthetic.then(SensorClock::default),
            });
            hid_motion::connect_source(plan.source, period_ns, plan.bias);
        }
        if self.streams.is_empty() {
            return false;
        }
        log::info!(
            "motion: using {} (type {}) streams=[{}]",
            name,
            kind.0,
            self.streams
                .iter()
                .map(|stream| format!(
                    "{:?} {:.0} Hz {:?} {:?}",
                    stream.source,
                    1e9 / stream.period_ns as f64,
                    stream.frame,
                    stream.bias
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.attached = Some(which);
        true
    }

    pub fn detach(&mut self) {
        for stream in self.streams.drain(..) {
            hid_motion::disconnect_source(stream.source);
        }
        self.attached = None;
    }

    pub fn release(&mut self, pad: *mut SDL_Gamepad) {
        if !pad.is_null() && self.attached == Some(unsafe { SDL_GetGamepadID(pad) }.0) {
            for stream in &self.streams {
                unsafe {
                    SDL_SetGamepadSensorEnabled(pad, stream.accel, false);
                    SDL_SetGamepadSensorEnabled(pad, stream.gyro, false);
                }
            }
        }
        self.detach();
    }

    pub fn pump(&mut self) {
        let now = Instant::now();
        let mut raw = std::mem::take(&mut self.raw);
        loop {
            let count = unsafe {
                SDL_PeepEvents(
                    self.events.as_mut_ptr(),
                    self.events.len() as i32,
                    SDL_GETEVENT,
                    SDL_EVENT_GAMEPAD_SENSOR_UPDATE.0,
                    SDL_EVENT_GAMEPAD_SENSOR_UPDATE.0,
                )
            };
            if count <= 0 {
                break;
            }
            for index in 0..count as usize {
                let event = unsafe { self.events[index].gsensor };
                if event.sensor_timestamp == 0 {
                    continue;
                }
                let Some(stream) = self.streams.iter_mut().find(|stream| {
                    stream.which == event.which.0
                        && (stream.accel.0 == event.sensor || stream.gyro.0 == event.sensor)
                }) else {
                    continue;
                };
                if let Some(clock) = stream.clock.as_mut() {
                    clock.seen(now);
                }
                raw.clear();
                if stream.gyro.0 == event.sensor {
                    stream.pairer.gyro(event.sensor_timestamp, event.data, &mut raw);
                } else {
                    stream.pairer.accel(event.sensor_timestamp, event.data, &mut raw);
                }
                for sample in &raw {
                    let (accel, gyro) = hid_motion::sdl_to_switch(stream.frame, sample.accel, sample.gyro);
                    let sensor_time_ns = stream
                        .clock
                        .as_mut()
                        .map_or(sample.sensor_time_ns, |clock| clock.retime(sample.sensor_time_ns));
                    self.samples[stream.source.index()].push(MotionSample {
                        sensor_time_ns,
                        accel,
                        gyro,
                    });
                }
            }
            if (count as usize) < self.events.len() {
                break;
            }
        }
        raw.clear();
        self.raw = raw;
        unsafe { SDL_FlushEvents(SDL_EVENT_FIRST.0, SDL_EVENT_LAST.0) };
        for (index, samples) in self.samples.iter_mut().enumerate() {
            if !samples.is_empty() {
                hid_motion::push_samples(MotionSource::ALL[index], samples);
                samples.clear();
            }
        }
    }
}

impl Default for MotionCapture {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdl3::sys::gamepad::SDL_GAMEPAD_TYPE_XBOX360;

    fn rows(plan: Vec<StreamPlan>) -> Vec<(i32, i32, MotionSource, AxisFrame, BiasMode)> {
        plan.into_iter()
            .map(|stream| (stream.accel.0, stream.gyro.0, stream.source, stream.frame, stream.bias))
            .collect()
    }

    #[test]
    fn pair_uses_only_the_side_sensors() {
        for (switch2, bias) in [(false, BiasMode::Learn), (true, BiasMode::FactoryCalibrated)] {
            for vertical in [false, true] {
                assert_eq!(
                    rows(stream_plan(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR, switch2, vertical)),
                    [
                        (SDL_SENSOR_ACCEL_L.0, SDL_SENSOR_GYRO_L.0, MotionSource::Left, AxisFrame::Body, bias),
                        (SDL_SENSOR_ACCEL_R.0, SDL_SENSOR_GYRO_R.0, MotionSource::Right, AxisFrame::Body, bias),
                    ]
                );
            }
        }
    }

    #[test]
    fn single_joycons_undo_the_sideways_rotation_only_when_needed() {
        let single = |kind, switch2, vertical| rows(stream_plan(kind, switch2, vertical));
        let left = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT;
        let right = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT;
        let row = |source, frame, bias| [(SDL_SENSOR_ACCEL.0, SDL_SENSOR_GYRO.0, source, frame, bias)];
        assert_eq!(single(left, false, true), row(MotionSource::Left, AxisFrame::Body, BiasMode::Learn));
        assert_eq!(single(right, false, true), row(MotionSource::Right, AxisFrame::Body, BiasMode::Learn));
        assert_eq!(
            single(left, false, false),
            row(MotionSource::Left, AxisFrame::SidewaysLeft, BiasMode::Learn)
        );
        assert_eq!(
            single(right, false, false),
            row(MotionSource::Right, AxisFrame::SidewaysRight, BiasMode::Learn)
        );
        assert_eq!(
            single(left, true, true),
            row(MotionSource::Left, AxisFrame::SidewaysLeft, BiasMode::FactoryCalibrated)
        );
        assert_eq!(
            single(right, true, true),
            row(MotionSource::Right, AxisFrame::SidewaysRight, BiasMode::FactoryCalibrated)
        );
    }

    #[test]
    fn other_pads_feed_the_primary_source() {
        let primary = |bias| [(SDL_SENSOR_ACCEL.0, SDL_SENSOR_GYRO.0, MotionSource::Primary, AxisFrame::Body, bias)];
        for kind in [SDL_GAMEPAD_TYPE_PS4, SDL_GAMEPAD_TYPE_PS5] {
            assert_eq!(rows(stream_plan(kind, false, true)), primary(BiasMode::FactoryCalibrated));
        }
        assert_eq!(
            rows(stream_plan(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, false, true)),
            primary(BiasMode::Learn)
        );
        assert_eq!(
            rows(stream_plan(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO, true, true)),
            primary(BiasMode::FactoryCalibrated)
        );
        assert_eq!(rows(stream_plan(SDL_GAMEPAD_TYPE_STANDARD, false, false)), primary(BiasMode::Learn));
    }

    #[test]
    fn switch2_ids_and_ranking() {
        for product in 0x2066..=0x2069 {
            assert!(is_switch2(0x057E, product));
        }
        for product in [0x2006, 0x2007, 0x2008, 0x2009, 0x2019] {
            assert!(!is_switch2(0x057E, product));
        }
        assert!(!is_switch2(0x054C, 0x2066));
        let ranks: Vec<Option<u8>> = [
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO,
            SDL_GAMEPAD_TYPE_PS5,
            SDL_GAMEPAD_TYPE_PS4,
            SDL_GAMEPAD_TYPE_STANDARD,
            SDL_GAMEPAD_TYPE_XBOX360,
        ]
        .into_iter()
        .map(motion_rank)
        .collect();
        assert_eq!(ranks, [Some(0), Some(1), Some(1), Some(2), Some(3), Some(4), Some(5), None]);
    }

    #[test]
    fn only_switch_one_pads_use_the_synthetic_clock() {
        for kind in [
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT,
            SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO,
        ] {
            assert!(synthetic_clock(kind, false));
            assert!(!synthetic_clock(kind, true));
        }
        for kind in [SDL_GAMEPAD_TYPE_PS4, SDL_GAMEPAD_TYPE_PS5, SDL_GAMEPAD_TYPE_STANDARD] {
            assert!(!synthetic_clock(kind, false));
        }
    }

    #[test]
    fn retime_follows_sdl_steps_without_a_pause() {
        let mut clock = SensorClock::default();
        clock.hold();
        let times: Vec<u64> = [5_000_000, 10_000_000, 10_000_000, 20_000_000, 25_020_000, 30_040_000]
            .into_iter()
            .map(|ts| clock.retime(ts))
            .collect();
        assert_eq!(times, [5_000_000, 10_000_000, 10_000_000, 20_000_000, 25_020_000, 30_040_000]);
        assert_eq!(clock.retime(3_000_000), 3_000_000);
        assert_eq!(clock.retime(8_020_000), 8_020_000);
    }

    #[test]
    fn retime_keeps_the_step_from_before_a_pause_for_two_windows() {
        let mut clock = SensorClock::default();
        assert_eq!(clock.retime(5_000_000), 5_000_000);
        assert_eq!(clock.retime(10_000_000), 10_000_000);
        clock.hold();
        let times: Vec<u64> = [
            24_000_000, 38_000_000, 66_000_000, 70_000_000, 74_000_000, 79_010_000, 84_020_000,
        ]
        .into_iter()
        .map(|ts| clock.retime(ts))
        .collect();
        assert_eq!(
            times,
            [15_000_000, 20_000_000, 30_000_000, 35_000_000, 40_000_000, 45_010_000, 50_020_000]
        );
    }

    #[test]
    fn retime_holds_the_step_when_a_stream_goes_quiet() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + std::time::Duration::from_millis(ms);
        let feed = |clock: &mut SensorClock, now: Instant, times: &[u64]| -> Vec<u64> {
            clock.seen(now);
            times.iter().map(|ts| clock.retime(*ts)).collect()
        };

        let mut clock = SensorClock::default();
        assert_eq!(feed(&mut clock, at(0), &[5_000_000, 10_000_000]), [5_000_000, 10_000_000]);
        assert_eq!(feed(&mut clock, at(200), &[15_000_000]), [15_000_000]);
        assert_eq!(feed(&mut clock, at(400), &[21_000_000]), [21_000_000]);

        let mut clock = SensorClock::default();
        assert_eq!(feed(&mut clock, at(0), &[5_000_000, 10_000_000]), [5_000_000, 10_000_000]);
        assert_eq!(feed(&mut clock, at(400), &[16_000_000, 22_000_000]), [15_000_000, 20_000_000]);
        assert_eq!(feed(&mut clock, at(415), &[27_010_000, 32_020_000]), [25_000_000, 30_000_000]);
        assert_eq!(feed(&mut clock, at(430), &[37_040_000]), [35_020_000]);
    }

    #[test]
    fn host_kind_follows_the_pad_type() {
        assert_eq!(host_kind_for(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT), HostKind::JoyConLeft);
        assert_eq!(host_kind_for(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT), HostKind::JoyConRight);
        assert_eq!(host_kind_for(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR), HostKind::JoyConPair);
        assert_eq!(host_kind_for(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO), HostKind::Other);
        assert_eq!(host_kind_for(SDL_GAMEPAD_TYPE_XBOX360), HostKind::Other);
    }
}

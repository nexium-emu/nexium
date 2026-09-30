use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

pub const DEFAULT_PERIOD_NS: u64 = 5_000_000;
pub const STANDARD_GRAVITY: f32 = 9.80665;
pub const MIN_OUTPUT_PERIOD_NS: u64 = 3_500_000;
pub const STALE_AFTER: Duration = Duration::from_millis(250);
pub const MAX_QUEUED_FRAMES: usize = 256;
const MIN_DT_NS: u64 = 1_000;
const MAX_DT_NS: u64 = 50_000_000;
const INIT_NORM_TOL_G: f32 = 0.2;
const GRAVITY_KP: f32 = 1.0;
const GRAVITY_WINDOW_G: f32 = 0.10;
const STILL_TAU_S: f32 = 0.2;
const STILL_GYRO_DEV: f32 = 0.010;
const STILL_ACCEL_DEV: f32 = 0.008;
const STILL_NORM_TOL_G: f32 = 0.05;
const STILL_MAX_OFFSET: f32 = 0.21;
const STILL_SETTLE_S: f64 = 0.5;
const CAPTURE_S: f64 = 1.0;
const BIAS_TAU_S: f32 = 2.0;
const REST_GYRO: f32 = 1.0e-6;
const IDENTITY_DIRECTION: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const DIRECTION_NUDGE: [[f32; 3]; 3] = [[0.9999995, 0.001, 0.0], [-0.001, 0.9999995, 0.0], [0.0, 0.0, 1.0]];
const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MotionSource {
    Primary = 0,
    Left = 1,
    Right = 2,
}

impl MotionSource {
    pub const ALL: [MotionSource; 3] = [MotionSource::Primary, MotionSource::Left, MotionSource::Right];

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum HostKind {
    #[default]
    None = 0,
    Other = 1,
    JoyConLeft = 2,
    JoyConRight = 3,
    JoyConPair = 4,
}

impl HostKind {
    fn from_raw(raw: u8) -> Self {
        match raw {
            1 => HostKind::Other,
            2 => HostKind::JoyConLeft,
            3 => HostKind::JoyConRight,
            4 => HostKind::JoyConPair,
            _ => HostKind::None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisFrame {
    Body,
    SidewaysLeft,
    SidewaysRight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasMode {
    Learn,
    FactoryCalibrated,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationState {
    Uncalibrated,
    Capturing,
    Calibrated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceSet {
    Host,
    Injected,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSample {
    pub sensor_time_ns: u64,
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawImuSample {
    pub sensor_time_ns: u64,
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SixAxisFrame {
    pub delta_time_ns: u64,
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
    pub angle: [f32; 3],
    pub direction: [[f32; 3]; 3],
    pub at_rest: bool,
}

impl SixAxisFrame {
    pub const REST: Self = Self {
        delta_time_ns: DEFAULT_PERIOD_NS,
        accel: [0.0, 0.0, -1.0],
        gyro: [0.0, 0.0, REST_GYRO],
        angle: [0.0; 3],
        direction: DIRECTION_NUDGE,
        at_rest: true,
    };

    pub fn hold(&self) -> Self {
        let down = [-self.direction[0][2], -self.direction[1][2], -self.direction[2][2]];
        Self { accel: down, gyro: [0.0, 0.0, REST_GYRO], ..*self }
    }

    pub fn sanitized(self) -> Self {
        let finite = self
            .accel
            .iter()
            .chain(&self.gyro)
            .chain(&self.angle)
            .chain(self.direction.iter().flatten())
            .all(|value| value.is_finite());
        if !finite {
            return Self { delta_time_ns: self.delta_time_ns, ..Self::REST };
        }
        let mut out = self;
        if out.gyro == [0.0; 3] {
            out.gyro[2] = REST_GYRO;
        }
        if out.accel == [0.0; 3] {
            out.accel = [0.0, 0.0, -1.0];
        }
        if out.direction == IDENTITY_DIRECTION {
            out.direction = DIRECTION_NUDGE;
        }
        out
    }

    pub fn merge(frames: &[SixAxisFrame]) -> Self {
        let Some(last) = frames.last() else {
            return Self::REST;
        };
        let total: u64 = frames.iter().map(|frame| frame.delta_time_ns).sum();
        if total == 0 {
            return *last;
        }
        let mut accel = [0.0f64; 3];
        let mut gyro = [0.0f64; 3];
        for frame in frames {
            let weight = frame.delta_time_ns as f64;
            for axis in 0..3 {
                accel[axis] += f64::from(frame.accel[axis]) * weight;
                gyro[axis] += f64::from(frame.gyro[axis]) * weight;
            }
        }
        let total_f = total as f64;
        Self {
            delta_time_ns: total,
            accel: accel.map(|value| (value / total_f) as f32),
            gyro: gyro.map(|value| (value / total_f) as f32),
            angle: last.angle,
            direction: last.direction,
            at_rest: frames.iter().all(|frame| frame.at_rest),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PendingHalf {
    Gyro(u64, [f32; 3]),
    Accel(u64, [f32; 3]),
}

#[derive(Clone, Debug, Default)]
pub struct ImuPairer {
    pending: Option<PendingHalf>,
    last_accel: Option<[f32; 3]>,
    last_gyro: Option<[f32; 3]>,
}

impl ImuPairer {
    pub fn gyro(&mut self, ts: u64, gyro_rad_s: [f32; 3], out: &mut Vec<RawImuSample>) {
        if let Some(PendingHalf::Accel(pending_ts, accel)) = self.pending {
            if pending_ts == ts {
                self.pending = None;
                self.emit(ts, accel, gyro_rad_s, out);
                return;
            }
        }
        self.flush(out);
        self.pending = Some(PendingHalf::Gyro(ts, gyro_rad_s));
    }

    pub fn accel(&mut self, ts: u64, accel_ms2: [f32; 3], out: &mut Vec<RawImuSample>) {
        if accel_ms2 == [0.0; 3] {
            return;
        }
        if let Some(PendingHalf::Gyro(pending_ts, gyro)) = self.pending {
            if pending_ts == ts {
                self.pending = None;
                self.emit(ts, accel_ms2, gyro, out);
                return;
            }
        }
        self.flush(out);
        self.pending = Some(PendingHalf::Accel(ts, accel_ms2));
    }

    fn flush(&mut self, out: &mut Vec<RawImuSample>) {
        match self.pending.take() {
            Some(PendingHalf::Gyro(ts, gyro)) => {
                if let Some(accel) = self.last_accel {
                    self.emit(ts, accel, gyro, out);
                }
            }
            Some(PendingHalf::Accel(ts, accel)) => {
                let gyro = self.last_gyro.unwrap_or([0.0; 3]);
                self.emit(ts, accel, gyro, out);
            }
            None => {}
        }
    }

    fn emit(&mut self, ts: u64, accel: [f32; 3], gyro: [f32; 3], out: &mut Vec<RawImuSample>) {
        self.last_accel = Some(accel);
        self.last_gyro = Some(gyro);
        out.push(RawImuSample { sensor_time_ns: ts, accel, gyro });
    }
}

pub fn sdl_to_switch(frame: AxisFrame, accel_ms2: [f32; 3], gyro_rad_s: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let (a, w) = (accel_ms2, gyro_rad_s);
    let (accel, gyro) = match frame {
        AxisFrame::Body => ([-a[0], a[2], -a[1]], [w[0], -w[2], w[1]]),
        AxisFrame::SidewaysLeft => ([a[2], a[0], -a[1]], [-w[2], -w[0], w[1]]),
        AxisFrame::SidewaysRight => ([-a[2], -a[0], -a[1]], [w[2], w[0], w[1]]),
    };
    (accel.map(|value| value / STANDARD_GRAVITY), gyro.map(|value| value / TAU))
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Quat {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl Quat {
    const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    fn mul(self, o: Self) -> Self {
        Self {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    fn normalized(self) -> Self {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n <= f32::EPSILON || !n.is_finite() {
            return Self::IDENTITY;
        }
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }

    fn from_rotation_vector(v: [f32; 3]) -> Self {
        let angle = norm(v);
        if angle <= 1e-9 {
            return Self { w: 1.0, x: v[0] * 0.5, y: v[1] * 0.5, z: v[2] * 0.5 }.normalized();
        }
        let half = angle * 0.5;
        let s = half.sin() / angle;
        Self { w: half.cos(), x: v[0] * s, y: v[1] * s, z: v[2] * s }
    }

    fn conjugate(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    fn rotate(self, v: [f32; 3]) -> [f32; 3] {
        let p = Self { w: 0.0, x: v[0], y: v[1], z: v[2] };
        let r = self.mul(p).mul(self.conjugate());
        [r.x, r.y, r.z]
    }

    fn matrix(self) -> [[f32; 3]; 3] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    }

    fn aligning(from: [f32; 3], to: [f32; 3]) -> Self {
        let a = normalized(from);
        let b = normalized(to);
        let d = dot(a, b).clamp(-1.0, 1.0);
        if d > 1.0 - 1e-6 {
            return Self::IDENTITY;
        }
        if d < -1.0 + 1e-6 {
            let helper = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let axis = normalized(cross(a, helper));
            return Self { w: 0.0, x: axis[0], y: axis[1], z: axis[2] };
        }
        let axis = cross(a, b);
        Self { w: 1.0 + d, x: axis[0], y: axis[1], z: axis[2] }.normalized()
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

fn normalized(v: [f32; 3]) -> [f32; 3] {
    let n = norm(v);
    if n <= f32::EPSILON {
        return [0.0, 0.0, -1.0];
    }
    [v[0] / n, v[1] / n, v[2] / n]
}

fn max_abs_diff(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

const WORLD_DOWN: [f32; 3] = [0.0, 0.0, -1.0];

fn direction_from_orientation(q: Quat) -> [[f32; 3]; 3] {
    let m = q.matrix();
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Calibration {
    Uncalibrated,
    Capturing { sum: [f64; 3], time: f64 },
    Calibrated,
}

#[derive(Clone, Debug)]
pub struct Fusion {
    q: Quat,
    bias: [f32; 3],
    angle: [f64; 3],
    initialized: bool,
    last_ts: Option<u64>,
    nominal_ns: u64,
    mode: BiasMode,
    calibration: Calibration,
    mean_g: [f32; 3],
    dev_g: f32,
    mean_a: [f32; 3],
    dev_a: f32,
    still_time: f64,
}

impl Fusion {
    pub const fn new(nominal_period_ns: u64, bias: BiasMode) -> Self {
        let nominal_ns = if nominal_period_ns == 0 {
            DEFAULT_PERIOD_NS
        } else if nominal_period_ns < MIN_DT_NS {
            MIN_DT_NS
        } else if nominal_period_ns > MAX_DT_NS {
            MAX_DT_NS
        } else {
            nominal_period_ns
        };
        let calibration = match bias {
            BiasMode::Learn => Calibration::Uncalibrated,
            BiasMode::FactoryCalibrated | BiasMode::Fixed => Calibration::Calibrated,
        };
        Self {
            q: Quat::IDENTITY,
            bias: [0.0; 3],
            angle: [0.0; 3],
            initialized: false,
            last_ts: None,
            nominal_ns,
            mode: bias,
            calibration,
            mean_g: [0.0; 3],
            dev_g: 0.0,
            mean_a: [0.0; 3],
            dev_a: 0.0,
            still_time: 0.0,
        }
    }

    pub fn step(&mut self, s: &MotionSample) -> Option<SixAxisFrame> {
        if !s.accel.iter().chain(&s.gyro).all(|value| value.is_finite()) {
            return None;
        }
        let (dt_ns, first) = match self.last_ts {
            None => (self.nominal_ns, true),
            Some(previous) if s.sensor_time_ns == previous => return None,
            Some(previous) if s.sensor_time_ns < previous => (self.nominal_ns, false),
            Some(previous) => ((s.sensor_time_ns - previous).clamp(MIN_DT_NS, MAX_DT_NS), false),
        };
        self.last_ts = Some(s.sensor_time_ns);
        let dt64 = dt_ns as f64 * 1e-9;
        let dt = dt64 as f32;
        let a = s.accel;
        let an = norm(a);
        let w_raw = s.gyro.map(|value| value * TAU);
        if first {
            self.mean_g = w_raw;
            self.dev_g = 0.0;
            self.mean_a = a;
            self.dev_a = 0.0;
        } else {
            let alpha = (dt / STILL_TAU_S).min(1.0);
            for axis in 0..3 {
                self.mean_g[axis] += alpha * (w_raw[axis] - self.mean_g[axis]);
                self.mean_a[axis] += alpha * (a[axis] - self.mean_a[axis]);
            }
            self.dev_g += alpha * (max_abs_diff(w_raw, self.mean_g) - self.dev_g);
            self.dev_a += alpha * (max_abs_diff(a, self.mean_a) - self.dev_a);
        }
        let still = self.dev_g < STILL_GYRO_DEV
            && self.dev_a < STILL_ACCEL_DEV
            && (an - 1.0).abs() < STILL_NORM_TOL_G
            && max_abs_diff(self.mean_g, self.bias) < STILL_MAX_OFFSET;
        self.still_time = if still { self.still_time + dt64 } else { 0.0 };
        self.update_bias(still, w_raw, dt, dt64);
        let w = [w_raw[0] - self.bias[0], w_raw[1] - self.bias[1], w_raw[2] - self.bias[2]];
        let mut integrate = !first;
        if !self.initialized && (an - 1.0).abs() < INIT_NORM_TOL_G {
            self.q = Quat::aligning(a, WORLD_DOWN);
            self.initialized = true;
            integrate = false;
        }
        if integrate {
            self.q = self.q.mul(Quat::from_rotation_vector(w.map(|value| value * dt))).normalized();
            if self.initialized && (an - 1.0).abs() <= GRAVITY_WINDOW_G {
                let up_measured = [-a[0] / an, -a[1] / an, -a[2] / an];
                let up_estimated = self.q.matrix()[2];
                let error = cross(up_measured, up_estimated);
                let correction = error.map(|value| value * GRAVITY_KP * dt);
                self.q = self.q.mul(Quat::from_rotation_vector(correction)).normalized();
            }
            for axis in 0..3 {
                self.angle[axis] += f64::from(w[axis] / TAU) * dt64;
            }
        }
        Some(SixAxisFrame {
            delta_time_ns: dt_ns,
            accel: a,
            gyro: w.map(|value| value / TAU),
            angle: self.angle.map(|value| value as f32),
            direction: direction_from_orientation(self.q),
            at_rest: self.still_time >= STILL_SETTLE_S,
        })
    }

    fn update_bias(&mut self, still: bool, w_raw: [f32; 3], dt: f32, dt64: f64) {
        if self.mode == BiasMode::Fixed {
            return;
        }
        let settled = self.still_time >= STILL_SETTLE_S;
        let current = self.calibration;
        self.calibration = match current {
            Calibration::Uncalibrated | Calibration::Capturing { .. } if !still => Calibration::Uncalibrated,
            Calibration::Uncalibrated if settled => Calibration::Capturing { sum: [0.0; 3], time: 0.0 },
            Calibration::Uncalibrated => Calibration::Uncalibrated,
            Calibration::Capturing { mut sum, time } => {
                for axis in 0..3 {
                    sum[axis] += f64::from(w_raw[axis]) * dt64;
                }
                let time = time + dt64;
                if time >= CAPTURE_S {
                    self.bias = sum.map(|value| (value / time) as f32);
                    Calibration::Calibrated
                } else {
                    Calibration::Capturing { sum, time }
                }
            }
            Calibration::Calibrated => {
                if settled {
                    let k = (dt / BIAS_TAU_S).min(1.0);
                    for axis in 0..3 {
                        self.bias[axis] += k * (self.mean_g[axis] - self.bias[axis]);
                    }
                }
                Calibration::Calibrated
            }
        };
    }

    pub fn recenter_heading(&mut self) {
        let direction = direction_from_orientation(self.q);
        let (right, forward) = (direction[0], direction[1]);
        let heading = if forward[0].hypot(forward[1]) >= 0.3 {
            (-forward[0]).atan2(forward[1])
        } else {
            right[1].atan2(right[0])
        };
        self.q = Quat::from_rotation_vector([0.0, 0.0, -heading]).mul(self.q).normalized();
    }

    pub fn request_calibration(&mut self) {
        if self.mode != BiasMode::Fixed {
            self.calibration = Calibration::Uncalibrated;
        }
    }

    pub fn calibration(&self) -> CalibrationState {
        match self.calibration {
            Calibration::Uncalibrated => CalibrationState::Uncalibrated,
            Calibration::Capturing { .. } => CalibrationState::Capturing,
            Calibration::Calibrated => CalibrationState::Calibrated,
        }
    }

    pub fn at_rest(&self) -> bool {
        self.still_time >= STILL_SETTLE_S
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MotionSnapshot {
    pub connected: [bool; 3],
    pub stale: [bool; 3],
    pub last: [Option<SixAxisFrame>; 3],
    pub generation: u64,
    pub injecting: bool,
}

struct SourceState {
    connected: bool,
    fusion: Fusion,
    merge_buf: Vec<SixAxisFrame>,
    merge_dt: u64,
    queue: VecDeque<SixAxisFrame>,
    last: Option<SixAxisFrame>,
    last_push: Option<Instant>,
    connected_at: Option<Instant>,
}

impl SourceState {
    const fn new() -> Self {
        Self {
            connected: false,
            fusion: Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed),
            merge_buf: Vec::new(),
            merge_dt: 0,
            queue: VecDeque::new(),
            last: None,
            last_push: None,
            connected_at: None,
        }
    }

    fn reset(&mut self, connected: bool, period_ns: u64, bias: BiasMode) {
        *self = Self { connected, fusion: Fusion::new(period_ns, bias), ..Self::new() };
    }

    fn drop_pending(&mut self) {
        self.queue.clear();
        self.merge_buf.clear();
        self.merge_dt = 0;
    }

    fn stale(&self, now: Instant) -> bool {
        self.connected
            && self.last_push.map_or(true, |at| now.saturating_duration_since(at) > STALE_AFTER)
    }
}

pub struct MotionHub {
    host: [SourceState; 3],
    injected: [SourceState; 3],
    injecting: bool,
    generation: u64,
}

impl MotionHub {
    pub const fn new() -> Self {
        Self {
            host: [SourceState::new(), SourceState::new(), SourceState::new()],
            injected: [SourceState::new(), SourceState::new(), SourceState::new()],
            injecting: false,
            generation: 0,
        }
    }

    fn sources_mut(&mut self, set: SourceSet) -> &mut [SourceState; 3] {
        match set {
            SourceSet::Host => &mut self.host,
            SourceSet::Injected => &mut self.injected,
        }
    }

    fn active(&self) -> &[SourceState; 3] {
        if self.injecting {
            &self.injected
        } else {
            &self.host
        }
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn connect(
        &mut self,
        set: SourceSet,
        src: MotionSource,
        period_ns: u64,
        bias: BiasMode,
        now: Instant,
    ) -> bool {
        let before = self.any_connected();
        let state = &mut self.sources_mut(set)[src.index()];
        state.reset(true, period_ns, bias);
        state.connected_at = Some(now);
        self.bump_generation();
        before != self.any_connected()
    }

    pub fn disconnect(&mut self, set: SourceSet, src: MotionSource) -> bool {
        let before = self.any_connected();
        let state = &mut self.sources_mut(set)[src.index()];
        let was_connected = state.connected;
        state.reset(false, DEFAULT_PERIOD_NS, BiasMode::Fixed);
        if was_connected {
            self.bump_generation();
        }
        before != self.any_connected()
    }

    pub fn push(&mut self, set: SourceSet, src: MotionSource, samples: &[MotionSample], now: Instant) {
        let state = &mut self.sources_mut(set)[src.index()];
        if !state.connected || samples.is_empty() {
            return;
        }
        for sample in samples {
            let Some(frame) = state.fusion.step(sample) else {
                continue;
            };
            state.last = Some(frame);
            state.merge_dt = state.merge_dt.saturating_add(frame.delta_time_ns);
            state.merge_buf.push(frame);
            if state.merge_dt >= MIN_OUTPUT_PERIOD_NS {
                let merged = SixAxisFrame::merge(&state.merge_buf);
                state.merge_buf.clear();
                state.merge_dt = 0;
                if state.queue.len() >= MAX_QUEUED_FRAMES {
                    state.queue.pop_front();
                }
                state.queue.push_back(merged);
            }
        }
        state.last_push = Some(now);
    }

    pub fn set_injection(&mut self, sources: Option<[bool; 3]>) -> bool {
        let before = self.any_connected();
        let mask = sources.unwrap_or([false; 3]);
        let mut changed = self.injecting != sources.is_some();
        self.injecting = sources.is_some();
        for src in MotionSource::ALL {
            let state = &mut self.injected[src.index()];
            let wanted = mask[src.index()];
            if wanted != state.connected {
                state.reset(wanted, DEFAULT_PERIOD_NS, BiasMode::Fixed);
                changed = true;
            }
        }
        if changed {
            self.bump_generation();
        }
        before != self.any_connected()
    }

    pub fn recenter_all(&mut self) {
        for state in self.host.iter_mut().chain(self.injected.iter_mut()) {
            state.fusion.recenter_heading();
            state.drop_pending();
            let direction = direction_from_orientation(state.fusion.q);
            if let Some(last) = state.last.as_mut() {
                last.direction = direction;
            }
        }
    }

    pub fn request_calibration(&mut self) {
        for state in self.host.iter_mut().chain(self.injected.iter_mut()) {
            state.fusion.request_calibration();
        }
    }

    pub fn drain(&mut self, out: &mut [Vec<SixAxisFrame>; 3], now: Instant) -> MotionSnapshot {
        let injecting = self.injecting;
        let mut snapshot =
            MotionSnapshot { generation: self.generation, injecting, ..MotionSnapshot::default() };
        let (active, inactive) = if injecting {
            (&mut self.injected, &mut self.host)
        } else {
            (&mut self.host, &mut self.injected)
        };
        for (index, state) in active.iter_mut().enumerate() {
            out[index].clear();
            out[index].extend(state.queue.drain(..));
            snapshot.connected[index] = state.connected;
            snapshot.stale[index] = state.stale(now);
            snapshot.last[index] = state.last;
        }
        for state in inactive.iter_mut() {
            state.queue.clear();
        }
        snapshot
    }

    pub fn any_connected(&self) -> bool {
        self.active().iter().any(|state| state.connected)
    }

    pub fn connected(&self) -> [bool; 3] {
        let active = self.active();
        [active[0].connected, active[1].connected, active[2].connected]
    }

    pub fn at_rest(&self, src: MotionSource) -> bool {
        let state = &self.active()[src.index()];
        !state.connected || state.last.map_or(true, |frame| frame.at_rest)
    }

    pub fn calibration(&self, src: MotionSource) -> Option<CalibrationState> {
        let state = &self.active()[src.index()];
        state.connected.then(|| state.fusion.calibration())
    }

    pub fn stale(&self, src: MotionSource, now: Instant) -> bool {
        let state = &self.active()[src.index()];
        state.connected
            && state
                .last_push
                .or(state.connected_at)
                .map_or(true, |at| now.saturating_duration_since(at) > STALE_AFTER)
    }
}

static MOTION: Mutex<MotionHub> = Mutex::new(MotionHub::new());
static HOST_KIND: AtomicU8 = AtomicU8::new(HostKind::None as u8);
static INPUT_KIND: AtomicU8 = AtomicU8::new(HostKind::None as u8);
static ANY_CONNECTED: AtomicBool = AtomicBool::new(false);
static PRESENTATION_DIRTY: AtomicBool = AtomicBool::new(false);

fn update_hub<R>(update: impl FnOnce(&mut MotionHub) -> R) -> R {
    let mut hub = MOTION.lock();
    let result = update(&mut hub);
    let connected = hub.any_connected();
    if ANY_CONNECTED.swap(connected, Ordering::Relaxed) != connected {
        PRESENTATION_DIRTY.store(true, Ordering::Relaxed);
    }
    result
}

pub fn connect_source(src: MotionSource, period_ns: u64, bias: BiasMode) {
    update_hub(|hub| hub.connect(SourceSet::Host, src, period_ns, bias, Instant::now()));
}

pub fn disconnect_source(src: MotionSource) {
    update_hub(|hub| hub.disconnect(SourceSet::Host, src));
}

pub fn push_samples(src: MotionSource, samples: &[MotionSample]) {
    if samples.is_empty() {
        return;
    }
    let now = Instant::now();
    MOTION.lock().push(SourceSet::Host, src, samples, now);
}

pub fn set_injection(sources: Option<[bool; 3]>) {
    update_hub(|hub| hub.set_injection(sources));
}

pub fn push_injected_samples(src: MotionSource, samples: &[MotionSample]) {
    if samples.is_empty() {
        return;
    }
    let now = Instant::now();
    MOTION.lock().push(SourceSet::Injected, src, samples, now);
}

pub fn recenter_all() {
    MOTION.lock().recenter_all();
}

pub fn request_calibration() {
    MOTION.lock().request_calibration();
}

pub fn calibration_states() -> [Option<CalibrationState>; 3] {
    let hub = MOTION.lock();
    MotionSource::ALL.map(|src| hub.calibration(src))
}

pub fn stale_sources() -> [bool; 3] {
    let now = Instant::now();
    let hub = MOTION.lock();
    MotionSource::ALL.map(|src| hub.stale(src, now))
}

pub fn connected_sources() -> [bool; 3] {
    MOTION.lock().connected()
}

pub fn drain_frames(out: &mut [Vec<SixAxisFrame>; 3]) -> MotionSnapshot {
    let now = Instant::now();
    MOTION.lock().drain(out, now)
}

pub fn source_at_rest(src: MotionSource) -> bool {
    MOTION.lock().at_rest(src)
}

pub fn any_source_connected() -> bool {
    ANY_CONNECTED.load(Ordering::Relaxed)
}

pub fn set_host_kind(kind: HostKind) {
    if HOST_KIND.swap(kind as u8, Ordering::Relaxed) != kind as u8 {
        PRESENTATION_DIRTY.store(true, Ordering::Relaxed);
    }
}

pub fn host_kind() -> HostKind {
    HostKind::from_raw(HOST_KIND.load(Ordering::Relaxed))
}

pub fn set_input_kind(kind: HostKind) {
    if INPUT_KIND.swap(kind as u8, Ordering::Relaxed) != kind as u8 {
        PRESENTATION_DIRTY.store(true, Ordering::Relaxed);
    }
}

pub fn input_kind() -> HostKind {
    HostKind::from_raw(INPUT_KIND.load(Ordering::Relaxed))
}

pub fn take_presentation_dirty() -> bool {
    PRESENTATION_DIRTY.swap(false, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT: [f32; 3] = [0.0, 0.0, -1.0];

    fn sample(sensor_time_ns: u64, accel: [f32; 3], gyro: [f32; 3]) -> MotionSample {
        MotionSample { sensor_time_ns, accel, gyro }
    }

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance
    }

    fn close3(a: [f32; 3], b: [f32; 3], tolerance: f32) -> bool {
        (0..3).all(|axis| close(a[axis], b[axis], tolerance))
    }

    fn close_direction(a: [[f32; 3]; 3], b: [[f32; 3]; 3], tolerance: f32) -> bool {
        (0..3).all(|column| close3(a[column], b[column], tolerance))
    }

    fn rest_samples(start_ns: u64, count: u64, period_ns: u64) -> Vec<MotionSample> {
        (0..count).map(|i| sample(start_ns + i * period_ns, FLAT, [0.0; 3])).collect()
    }

    fn yaw_run() -> (Fusion, SixAxisFrame) {
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        let mut frame = fusion.step(&sample(5_000_000, FLAT, [0.0; 3])).unwrap();
        for i in 1..=50u64 {
            frame = fusion.step(&sample((i + 1) * 5_000_000, FLAT, [0.0, 0.0, 1.0])).unwrap();
        }
        (fusion, frame)
    }

    fn qz(degrees: f32) -> Quat {
        Quat::from_rotation_vector([0.0, 0.0, degrees.to_radians()])
    }

    fn qx(degrees: f32) -> Quat {
        Quat::from_rotation_vector([degrees.to_radians(), 0.0, 0.0])
    }

    #[test]
    fn sdl_gravity_flat_maps_to_switch_down_for_every_frame() {
        for frame in [AxisFrame::Body, AxisFrame::SidewaysLeft, AxisFrame::SidewaysRight] {
            let (accel, _) = sdl_to_switch(frame, [0.0, STANDARD_GRAVITY, 0.0], [0.0; 3]);
            assert!(close3(accel, [0.0, 0.0, -1.0], 1e-6), "{:?} {:?}", frame, accel);
        }
    }

    #[test]
    fn remote_pose_maps_to_negative_y() {
        let g = STANDARD_GRAVITY;
        let cases = [
            (AxisFrame::Body, [0.0, 0.0, -g], [0.0, -1.0, 0.0]),
            (AxisFrame::SidewaysRight, [g, 0.0, 0.0], [0.0, -1.0, 0.0]),
            (AxisFrame::SidewaysLeft, [-g, 0.0, 0.0], [0.0, -1.0, 0.0]),
            (AxisFrame::Body, [-g, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        for (frame, input, expected) in cases {
            let (accel, _) = sdl_to_switch(frame, input, [0.0; 3]);
            assert!(close3(accel, expected, 1e-6), "{:?} {:?}", frame, accel);
        }
    }

    #[test]
    fn gyro_signs() {
        let cases = [
            (AxisFrame::Body, [0.0, TAU, 0.0], [0.0, 0.0, 1.0]),
            (AxisFrame::Body, [TAU, 0.0, 0.0], [1.0, 0.0, 0.0]),
            (AxisFrame::Body, [0.0, 0.0, TAU], [0.0, -1.0, 0.0]),
            (AxisFrame::SidewaysRight, [0.0, TAU, 0.0], [0.0, 0.0, 1.0]),
        ];
        for (frame, input, expected) in cases {
            let (_, gyro) = sdl_to_switch(frame, [0.0, STANDARD_GRAVITY, 0.0], input);
            assert!(close3(gyro, expected, 1e-6), "{:?} {:?}", frame, gyro);
        }
    }

    #[test]
    fn rest_gives_identity_direction_and_calibrates() {
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Learn);
        assert_eq!(fusion.calibration(), CalibrationState::Uncalibrated);
        let mut frame = None;
        for i in 0..400u64 {
            frame = fusion.step(&sample((i + 1) * 5_000_000, FLAT, [0.0; 3]));
        }
        let frame = frame.unwrap();
        assert!(close_direction(frame.direction, IDENTITY_DIRECTION, 1e-4), "{:?}", frame.direction);
        assert!(frame.at_rest);
        assert!(fusion.at_rest());
        assert_eq!(fusion.calibration(), CalibrationState::Calibrated);
    }

    #[test]
    fn yaw_quarter_turn_in_quarter_second() {
        let (_, frame) = yaw_run();
        let expected = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(close_direction(frame.direction, expected, 1e-3), "{:?}", frame.direction);
        assert!(close(frame.angle[2], 0.25, 1e-4), "{:?}", frame.angle);
        assert!(close3(frame.gyro, [0.0, 0.0, 1.0], 1e-6), "{:?}", frame.gyro);
    }

    #[test]
    fn pitch_quarter_turn_with_consistent_gravity() {
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        let mut frame = fusion.step(&sample(5_000_000, FLAT, [0.0; 3])).unwrap();
        for i in 1..=50u64 {
            let theta = TAU * (i as f32 * 0.005);
            let accel = [0.0, -theta.sin(), -theta.cos()];
            frame = fusion.step(&sample((i + 1) * 5_000_000, accel, [1.0, 0.0, 0.0])).unwrap();
        }
        assert!(close3(frame.direction[1], [0.0, 0.0, 1.0], 1e-3), "{:?}", frame.direction);
        assert!(close3(frame.direction[2], [0.0, -1.0, 0.0], 1e-3), "{:?}", frame.direction);
    }

    #[test]
    fn gravity_correction_recovers_tilt() {
        let tilt = 20f32.to_radians();
        let down_in_body = [0.0, -tilt.sin(), -tilt.cos()];
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        fusion.step(&sample(0, FLAT, [0.0; 3]));
        for i in 0..2000u64 {
            fusion.step(&sample((i + 1) * 5_000_000, down_in_body, [0.0; 3]));
        }
        let predicted = fusion.q.conjugate().rotate(WORLD_DOWN);
        assert!(close3(predicted, down_in_body, 2e-3), "{:?}", predicted);
    }

    #[test]
    fn bias_captured_when_still() {
        let bias = [0.01f32, -0.005, 0.008];
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Learn);
        let mut outputs = Vec::new();
        for i in 0..1000u64 {
            let noise = if i % 2 == 0 { 0.002f32 } else { -0.002 };
            let gyro = bias.map(|value| value + noise / TAU);
            let accel = [noise, noise, -1.0 + noise];
            outputs.push(fusion.step(&sample((i + 1) * 5_000_000, accel, gyro)).unwrap().gyro);
        }
        assert_eq!(fusion.calibration(), CalibrationState::Calibrated);
        for axis in 0..3 {
            let mean = outputs[800..].iter().map(|gyro| f64::from(gyro[axis])).sum::<f64>() / 200.0;
            assert!(mean.abs() < 1e-4, "axis {} mean {}", axis, mean);
        }
    }

    #[test]
    fn hand_tremor_does_not_calibrate() {
        let amplitude = 1.5f32.to_radians() / TAU;
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Learn);
        for i in 0..2000u64 {
            let t = (i + 1) as f32 * 0.005;
            let gyro = [amplitude * (TAU * 8.0 * t).sin(), 0.0, 0.0];
            fusion.step(&sample((i + 1) * 5_000_000, FLAT, gyro)).unwrap();
            assert_eq!(fusion.calibration(), CalibrationState::Uncalibrated, "sample {}", i);
        }
    }

    #[test]
    fn timestamps() {
        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        let first = fusion.step(&sample(1_000_000, FLAT, [0.0, 0.0, 1.0])).unwrap();
        assert_eq!(first.delta_time_ns, DEFAULT_PERIOD_NS);
        assert_eq!(first.direction, IDENTITY_DIRECTION);
        assert_eq!(first.angle, [0.0; 3]);
        assert!(fusion.step(&sample(1_000_000, FLAT, [0.0, 0.0, 1.0])).is_none());
        let backwards = fusion.step(&sample(500_000, FLAT, [0.0; 3])).unwrap();
        assert_eq!(backwards.delta_time_ns, DEFAULT_PERIOD_NS);
        let gap = fusion.step(&sample(1_000_500_000, FLAT, [0.0; 3])).unwrap();
        assert_eq!(gap.delta_time_ns, MAX_DT_NS);
        let short = fusion.step(&sample(1_000_600_000, FLAT, [0.0; 3])).unwrap();
        assert_eq!(short.delta_time_ns, 100_000);
        let tiny = fusion.step(&sample(1_000_600_200, FLAT, [0.0; 3])).unwrap();
        assert_eq!(tiny.delta_time_ns, MIN_DT_NS);
        assert!(fusion.step(&sample(1_000_700_000, [f32::NAN, 0.0, -1.0], [0.0; 3])).is_none());
    }

    #[test]
    fn recenter_removes_heading_keeps_tilt_and_angle() {
        let (mut fusion, _) = yaw_run();
        fusion.recenter_heading();
        assert!(close_direction(direction_from_orientation(fusion.q), IDENTITY_DIRECTION, 1e-3));
        assert!((fusion.angle[2] - 0.25).abs() < 1e-4, "{:?}", fusion.angle);

        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        fusion.q = qz(60.0).mul(qx(30.0));
        fusion.recenter_heading();
        let direction = direction_from_orientation(fusion.q);
        assert!(close3(direction[0], [1.0, 0.0, 0.0], 1e-3), "{:?}", direction);
        assert!(close3(direction[1], [0.0, 0.866_025, 0.5], 1e-3), "{:?}", direction);

        let mut fusion = Fusion::new(DEFAULT_PERIOD_NS, BiasMode::Fixed);
        fusion.q = qz(40.0).mul(qx(85.0));
        fusion.recenter_heading();
        let direction = direction_from_orientation(fusion.q);
        assert!(close3(direction[0], [1.0, 0.0, 0.0], 1e-3), "{:?}", direction);
    }

    #[test]
    fn pairer() {
        let a1 = [0.0, 9.8, 0.0];
        let a2 = [0.1, 9.7, 0.0];
        let g1 = [0.1, 0.0, 0.0];
        let g2 = [0.2, 0.0, 0.0];

        let mut pairer = ImuPairer::default();
        let mut out = Vec::new();
        pairer.gyro(10, g1, &mut out);
        pairer.accel(10, a1, &mut out);
        pairer.gyro(20, g2, &mut out);
        pairer.accel(20, a2, &mut out);
        assert_eq!(
            out,
            vec![
                RawImuSample { sensor_time_ns: 10, accel: a1, gyro: g1 },
                RawImuSample { sensor_time_ns: 20, accel: a2, gyro: g2 },
            ]
        );

        let mut pairer = ImuPairer::default();
        let mut out = Vec::new();
        pairer.accel(10, a1, &mut out);
        pairer.gyro(10, g1, &mut out);
        assert_eq!(out, vec![RawImuSample { sensor_time_ns: 10, accel: a1, gyro: g1 }]);

        pairer.gyro(20, g2, &mut out);
        pairer.gyro(30, g1, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], RawImuSample { sensor_time_ns: 20, accel: a1, gyro: g2 });

        let mut pairer = ImuPairer::default();
        let mut out = Vec::new();
        pairer.gyro(5, g1, &mut out);
        pairer.accel(5, a1, &mut out);
        pairer.accel(10, a2, &mut out);
        pairer.gyro(20, g2, &mut out);
        pairer.accel(20, a1, &mut out);
        assert_eq!(
            out,
            vec![
                RawImuSample { sensor_time_ns: 5, accel: a1, gyro: g1 },
                RawImuSample { sensor_time_ns: 10, accel: a2, gyro: g1 },
                RawImuSample { sensor_time_ns: 20, accel: a1, gyro: g2 },
            ]
        );
        assert!(out.windows(2).all(|pair| pair[0].sensor_time_ns < pair[1].sensor_time_ns));

        let mut pairer = ImuPairer::default();
        let mut out = Vec::new();
        pairer.accel(10, a1, &mut out);
        pairer.gyro(20, g1, &mut out);
        assert_eq!(out, vec![RawImuSample { sensor_time_ns: 10, accel: a1, gyro: [0.0; 3] }]);

        let mut pairer = ImuPairer::default();
        let mut out = Vec::new();
        pairer.gyro(10, g1, &mut out);
        pairer.accel(10, [0.0; 3], &mut out);
        assert!(out.is_empty());
        pairer.gyro(20, g2, &mut out);
        pairer.gyro(30, g2, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn sanitized_never_zero_gyro_or_identity() {
        let zero = SixAxisFrame {
            delta_time_ns: 0,
            accel: [0.0; 3],
            gyro: [-0.0, 0.0, 0.0],
            angle: [0.0; 3],
            direction: [[0.0; 3]; 3],
            at_rest: false,
        };
        let identity = SixAxisFrame { direction: IDENTITY_DIRECTION, ..zero };
        let broken = SixAxisFrame { delta_time_ns: 7_000_000, gyro: [f32::NAN, 0.0, 0.0], ..identity };
        for frame in [SixAxisFrame::REST, zero, identity, broken, SixAxisFrame::REST.hold()] {
            let clean = frame.sanitized();
            assert_ne!(clean.gyro, [0.0; 3]);
            assert_ne!(clean.accel, [0.0; 3]);
            assert_ne!(clean.direction, IDENTITY_DIRECTION);
            assert!(clean.gyro.iter().chain(&clean.accel).all(|value| value.is_finite()));
        }
        assert_eq!(broken.sanitized().delta_time_ns, 7_000_000);
        assert_eq!(identity.sanitized().direction, DIRECTION_NUDGE);
        assert_eq!(zero.sanitized().accel, [0.0, 0.0, -1.0]);
        assert_eq!(zero.sanitized().gyro, [-0.0, 0.0, REST_GYRO]);
        let moving = SixAxisFrame { gyro: [0.5, 0.0, 0.0], accel: [0.0, -1.0, 0.0], ..identity };
        assert_eq!(moving.sanitized().gyro, [0.5, 0.0, 0.0]);
        assert_eq!(moving.hold().gyro, [0.0, 0.0, REST_GYRO]);
        assert_eq!(moving.hold().accel, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn merge_preserves_rotation_and_time() {
        let base = SixAxisFrame::REST;
        let g1 = [0.1, 0.2, 0.3];
        let g2 = [0.5, -0.2, 0.0];
        let g3 = [1.0, 0.0, -0.4];
        let frames = [
            SixAxisFrame { delta_time_ns: 1_000_000, gyro: g1, angle: [0.1; 3], ..base },
            SixAxisFrame { delta_time_ns: 1_000_000, gyro: g2, angle: [0.2; 3], at_rest: false, ..base },
            SixAxisFrame {
                delta_time_ns: 2_000_000,
                gyro: g3,
                angle: [0.3; 3],
                direction: [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                ..base
            },
        ];
        let merged = SixAxisFrame::merge(&frames);
        assert_eq!(merged.delta_time_ns, 4_000_000);
        for axis in 0..3 {
            let expected = (g1[axis] + g2[axis] + 2.0 * g3[axis]) / 4.0;
            assert!(close(merged.gyro[axis], expected, 1e-6), "{:?}", merged.gyro);
        }
        assert_eq!(merged.angle, frames[2].angle);
        assert_eq!(merged.direction, frames[2].direction);
        assert!(!merged.at_rest);
        let zero_time = [
            SixAxisFrame { delta_time_ns: 0, ..frames[0] },
            SixAxisFrame { delta_time_ns: 0, ..frames[2] },
        ];
        assert_eq!(SixAxisFrame::merge(&zero_time), zero_time[1]);
    }

    #[test]
    fn hub_merges_high_rate_sources() {
        let mut hub = MotionHub::new();
        let now = Instant::now();
        assert!(hub.connect(SourceSet::Host, MotionSource::Primary, 1_000_000, BiasMode::Fixed, now));
        hub.push(SourceSet::Host, MotionSource::Primary, &rest_samples(1_000_000, 40, 1_000_000), now);
        let mut out: [Vec<SixAxisFrame>; 3] = Default::default();
        let snapshot = hub.drain(&mut out, now);
        assert_eq!(snapshot.connected, [true, false, false]);
        assert_eq!(out[0].len(), 10);
        assert!(out[0].iter().all(|frame| frame.delta_time_ns == 4_000_000));
        assert_eq!(out[0].iter().map(|frame| frame.delta_time_ns).sum::<u64>(), 40_000_000);
        assert!(out[1].is_empty() && out[2].is_empty());
    }

    #[test]
    fn hub_injection_is_isolated_from_host() {
        let mut hub = MotionHub::new();
        let now = Instant::now();
        let mut out: [Vec<SixAxisFrame>; 3] = Default::default();
        let mut generation = hub.generation;
        assert!(!hub.disconnect(SourceSet::Host, MotionSource::Left));
        assert!(!hub.set_injection(None));
        assert_eq!(hub.generation, generation);

        assert!(hub.connect(SourceSet::Host, MotionSource::Left, DEFAULT_PERIOD_NS, BiasMode::Learn, now));
        assert!(hub.generation > generation);
        generation = hub.generation;

        assert!(!hub.set_injection(Some([false, false, true])));
        assert!(hub.generation > generation);
        generation = hub.generation;
        assert!(!hub.set_injection(Some([false, false, true])));
        assert_eq!(hub.generation, generation);

        hub.push(SourceSet::Host, MotionSource::Left, &rest_samples(5_000_000, 8, 5_000_000), now);
        hub.push(SourceSet::Injected, MotionSource::Right, &rest_samples(5_000_000, 8, 5_000_000), now);
        let snapshot = hub.drain(&mut out, now);
        assert_eq!(snapshot.connected, [false, false, true]);
        assert!(snapshot.injecting);
        assert_eq!(snapshot.generation, generation);
        assert!(out[1].is_empty());
        assert_eq!(out[2].len(), 8);
        assert_eq!(hub.connected(), [false, false, true]);
        assert_eq!(hub.calibration(MotionSource::Left), None);
        assert_eq!(hub.calibration(MotionSource::Right), Some(CalibrationState::Calibrated));

        let host_calibration = hub.host[1].fusion.calibration();
        let host_last = hub.host[1].last;
        let host_ts = hub.host[1].fusion.last_ts;
        assert!(host_last.is_some());
        assert!(!hub.set_injection(None));
        assert!(hub.generation > generation);
        generation = hub.generation;
        assert_eq!(hub.connected(), [false, true, false]);
        assert_eq!(hub.host[1].last, host_last);
        assert_eq!(hub.host[1].fusion.last_ts, host_ts);
        assert_eq!(hub.calibration(MotionSource::Left), Some(host_calibration));
        assert!(hub.injected.iter().all(|state| !state.connected));

        hub.push(SourceSet::Host, MotionSource::Left, &rest_samples(50_000_000, 8, 5_000_000), now);
        assert!(!hub.set_injection(Some([true, false, false])));
        assert!(hub.generation > generation);
        generation = hub.generation;
        hub.push(SourceSet::Injected, MotionSource::Primary, &rest_samples(5_000_000, 8, 5_000_000), now);
        hub.push(SourceSet::Host, MotionSource::Left, &rest_samples(90_000_000, 8, 5_000_000), now);
        assert!(hub.host[1].queue.len() + hub.injected[0].queue.len() > 0);
        hub.recenter_all();
        assert!(hub.host.iter().chain(hub.injected.iter()).all(|state| state.queue.is_empty()));
        assert!(hub.host.iter().chain(hub.injected.iter()).all(|state| state.merge_buf.is_empty()));
        assert_eq!(hub.generation, generation);
        let snapshot = hub.drain(&mut out, now);
        assert!(out.iter().all(|frames| frames.is_empty()));
        assert_eq!(snapshot.connected, [true, false, false]);

        assert!(!hub.set_injection(None));
        assert!(hub.generation > generation);
        generation = hub.generation;
        assert!(hub.disconnect(SourceSet::Host, MotionSource::Left));
        assert!(hub.generation > generation);
        assert!(!hub.any_connected());
        assert!(hub.at_rest(MotionSource::Left));
    }

    #[test]
    fn hub_stale_after_250_ms() {
        let mut hub = MotionHub::new();
        let t0 = Instant::now();
        let mut out: [Vec<SixAxisFrame>; 3] = Default::default();
        hub.connect(SourceSet::Host, MotionSource::Right, DEFAULT_PERIOD_NS, BiasMode::Fixed, t0);
        assert!(!hub.stale(MotionSource::Right, t0));
        assert!(!hub.stale(MotionSource::Right, t0 + Duration::from_millis(250)));
        assert!(!hub.stale(MotionSource::Left, t0));
        assert!(hub.drain(&mut out, t0).stale[2]);
        hub.push(SourceSet::Host, MotionSource::Right, &rest_samples(5_000_000, 4, 5_000_000), t0);
        assert!(!hub.stale(MotionSource::Right, t0 + Duration::from_millis(100)));
        let fresh = hub.drain(&mut out, t0 + Duration::from_millis(100));
        assert!(!fresh.stale[2]);
        assert!(fresh.last[2].is_some());
        assert_eq!(out[2].len(), 4);
        let stale = hub.drain(&mut out, t0 + Duration::from_millis(300));
        assert!(stale.stale[2]);
        assert!(out[2].is_empty());
        assert!(!stale.stale[0] && !stale.stale[1]);
        assert!(hub.stale(MotionSource::Right, t0 + Duration::from_millis(300)));
        let t1 = t0 + Duration::from_millis(300);
        hub.connect(SourceSet::Host, MotionSource::Left, DEFAULT_PERIOD_NS, BiasMode::Learn, t1);
        assert!(!hub.stale(MotionSource::Left, t1 + Duration::from_millis(100)));
        assert!(hub.drain(&mut out, t1 + Duration::from_millis(100)).stale[1]);
        assert!(hub.stale(MotionSource::Left, t1 + Duration::from_millis(300)));
        hub.disconnect(SourceSet::Host, MotionSource::Right);
        assert!(!hub.stale(MotionSource::Right, t0 + Duration::from_millis(300)));
    }
}

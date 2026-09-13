use std::collections::VecDeque;
use std::time::{Duration, Instant};

const SAMPLE_SPACING: Duration = Duration::from_millis(50);
const WINDOW: Duration = Duration::from_secs(1);

pub struct PerformanceMonitor {
    samples: VecDeque<(Instant, u64)>,
    frames: u64,
    svc_count: u32,
    cycle_count: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            frames: 0,
            svc_count: 0,
            cycle_count: 0,
        }
    }

    pub fn record_frame(&mut self) {
        self.record_frames(1);
    }

    pub fn record_frames(&mut self, count: u64) {
        self.record_frames_at(Instant::now(), count);
    }

    pub fn tick(&mut self) {
        self.record_frames_at(Instant::now(), 0);
    }

    fn record_frames_at(&mut self, now: Instant, count: u64) {
        self.frames += count;
        if let Some(&(last, _)) = self.samples.back() {
            if now.saturating_duration_since(last) < SAMPLE_SPACING {
                return;
            }
        }
        self.samples.push_back((now, self.frames));
        while self.samples.len() > 1 && now.saturating_duration_since(self.samples[1].0) >= WINDOW
        {
            self.samples.pop_front();
        }
    }

    fn fps_at(&self, now: Instant) -> f32 {
        let Some(&(start, frames)) = self.samples.front() else {
            return 0.0;
        };
        let span = now.saturating_duration_since(start).as_secs_f32();
        if span <= 0.0 || self.frames <= frames {
            return 0.0;
        }
        (self.frames - frames) as f32 / span
    }

    fn frame_time_at(&self, now: Instant) -> f32 {
        let fps = self.fps_at(now);
        if fps > 0.0 {
            1000.0 / fps
        } else {
            0.0
        }
    }

    pub fn get_fps(&self) -> f32 {
        self.fps_at(Instant::now())
    }

    pub fn get_frame_time(&self) -> f32 {
        self.frame_time_at(Instant::now())
    }

    pub fn record_svc(&mut self) {
        self.svc_count += 1;
    }

    pub fn record_cycles(&mut self, count: u64) {
        self.cycle_count += count;
    }

    pub fn get_svc_count(&self) -> u32 {
        self.svc_count
    }

    pub fn get_cycle_count(&self) -> u64 {
        self.cycle_count
    }

    pub fn reset(&mut self) {
        self.svc_count = 0;
        self.cycle_count = 0;
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(t0: Instant, interval: Duration, frames: u64) -> (PerformanceMonitor, Instant) {
        let mut monitor = PerformanceMonitor::new();
        let mut now = t0;
        monitor.record_frames_at(now, 0);
        for _ in 0..frames {
            now += interval;
            monitor.record_frames_at(now, 1);
        }
        (monitor, now)
    }

    #[test]
    fn steady_frames_report_their_rate() {
        let (monitor, now) = steady(Instant::now(), Duration::from_micros(16_667), 180);
        let fps = monitor.fps_at(now);
        assert!((fps - 60.0).abs() < 0.6, "{fps}");
        let (monitor, now) = steady(Instant::now(), Duration::from_millis(25), 80);
        assert!((monitor.fps_at(now) - 40.0).abs() < 0.5);
        assert!((monitor.frame_time_at(now) - 25.0).abs() < 0.5);
    }

    #[test]
    fn rate_is_independent_of_observation_cadence() {
        let mut monitor = PerformanceMonitor::new();
        let mut now = Instant::now();
        monitor.record_frames_at(now, 0);
        for _ in 0..12 {
            now += Duration::from_millis(250);
            monitor.record_frames_at(now, 15);
        }
        let fps = monitor.fps_at(now);
        assert!((fps - 60.0).abs() < 0.6, "{fps}");
    }

    #[test]
    fn stalled_observer_averages_over_the_stall_then_recovers() {
        let (mut monitor, now) = steady(Instant::now(), Duration::from_millis(25), 40);
        let later = now + Duration::from_secs(30);
        monitor.record_frames_at(later, 1000);
        let fps = monitor.fps_at(later);
        assert!(fps > 32.0 && fps < 35.0, "{fps}");
        let mut t = later;
        for _ in 0..66 {
            t += Duration::from_micros(16_667);
            monitor.record_frames_at(t, 1);
        }
        let fps = monitor.fps_at(t);
        assert!((fps - 60.0).abs() < 1.0, "{fps}");
    }

    #[test]
    fn rate_decays_to_zero_when_frames_stop() {
        let (mut monitor, now) = steady(Instant::now(), Duration::from_micros(16_667), 120);
        assert!(monitor.fps_at(now) > 59.0);
        let mut t = now;
        for _ in 0..30 {
            t += Duration::from_millis(50);
            monitor.record_frames_at(t, 0);
        }
        assert_eq!(monitor.fps_at(t), 0.0);
        assert_eq!(monitor.frame_time_at(t), 0.0);
    }

    #[test]
    fn samples_stay_bounded_under_rapid_observation() {
        let mut monitor = PerformanceMonitor::new();
        let mut now = Instant::now();
        for _ in 0..10_000 {
            now += Duration::from_millis(1);
            monitor.record_frames_at(now, 1);
        }
        assert!(monitor.samples.len() <= 22, "{}", monitor.samples.len());
        assert!((monitor.fps_at(now) - 1000.0).abs() < 5.0);
    }

    #[test]
    fn empty_monitor_reports_zero() {
        let monitor = PerformanceMonitor::new();
        assert_eq!(monitor.get_fps(), 0.0);
        assert_eq!(monitor.get_frame_time(), 0.0);
    }
}

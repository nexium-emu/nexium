use std::time::{Duration, Instant};

pub struct PerformanceMonitor {
    frame_times: [Instant; 60],
    frame_index: usize,
    svc_count: u32,
    cycle_count: u64,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            frame_times: [Instant::now(); 60],
            frame_index: 0,
            svc_count: 0,
            cycle_count: 0,
        }
    }

    pub fn record_frame(&mut self) {
        self.frame_times[self.frame_index] = Instant::now();
        self.frame_index = (self.frame_index + 1) % 60;
    }

    pub fn record_svc(&mut self) {
        self.svc_count += 1;
    }

    pub fn record_cycles(&mut self, count: u64) {
        self.cycle_count += count;
    }

    pub fn get_fps(&self) -> f32 {
        let now = Instant::now();
        let oldest = self.frame_times[(self.frame_index + 1) % 60];

        if oldest == Instant::now() {
            return 0.0;
        }

        let elapsed = now.duration_since(oldest);
        if elapsed.as_secs_f32() > 0.0 {
            59.0 / elapsed.as_secs_f32()
        } else {
            0.0
        }
    }

    pub fn get_frame_time(&self) -> f32 {
        let now = Instant::now();
        let oldest = self.frame_times[(self.frame_index + 1) % 60];

        if oldest == Instant::now() {
            return 0.0;
        }

        let elapsed = now.duration_since(oldest);
        (elapsed.as_secs_f32() * 1000.0) / 59.0
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

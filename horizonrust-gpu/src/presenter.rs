use std::time::{Duration, Instant};

const VSYNC_PERIOD_NS: u128 = 16_666_667;

pub struct FramePresenter {
    last_swap_time: Option<Instant>,
}

impl FramePresenter {
    pub fn new() -> Self {
        Self {
            last_swap_time: None,
        }
    }

    pub fn present_frame(&mut self) -> Result<(), String> {
        let now = Instant::now();

        if let Some(last_time) = self.last_swap_time {
            let elapsed_ns = now.duration_since(last_time).as_nanos();
            if elapsed_ns < VSYNC_PERIOD_NS {
                let remaining = Duration::from_nanos((VSYNC_PERIOD_NS - elapsed_ns) as u64);
                std::thread::sleep(remaining);
            }
        }

        self.last_swap_time = Some(Instant::now());
        Ok(())
    }

    pub fn readback_frame(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, String> {
        let frame_size = (width * height * 4) as usize;
        Ok(vec![0u8; frame_size])
    }
}

use once_cell::sync::OnceCell;
use std::sync::Arc;

pub trait HostPcmSink: Send + Sync {
    fn push_stereo_f32(&self, samples: &[f32]) -> usize;

    fn samples_consumed(&self) -> u64;

    fn open_audio_out_stream(&self, _stream_id: u64) -> bool {
        false
    }

    fn close_audio_out_stream(&self, _stream_id: u64) {}

    fn start_audio_out_stream(&self, _stream_id: u64) {}

    fn stop_audio_out_stream(&self, _stream_id: u64) {}

    fn push_audio_out_stereo_f32(&self, _stream_id: u64, _samples: &[f32]) -> usize {
        0
    }

    fn audio_out_samples_consumed(&self, _stream_id: u64) -> u64 {
        0
    }

    fn set_audio_out_volume(&self, _stream_id: u64, _volume: f32) {}

    fn sample_rate(&self) -> u32 {
        48_000
    }

    fn drain_pending_events(&self) -> u64 {
        0
    }

    fn repost_pending_events(&self, _n: u64) {}

    fn queued_frames(&self) -> usize {
        0
    }

    fn vacant_frames(&self) -> usize {
        0
    }

    fn drain_underrun_frames(&self) -> u64 {
        0
    }

    fn drain_dropped_frames(&self) -> u64 {
        0
    }
}

static HOST_AUDIO_SINK: OnceCell<Arc<dyn HostPcmSink>> = OnceCell::new();

pub fn set_host_audio_sink(sink: Arc<dyn HostPcmSink>) {
    let _ = HOST_AUDIO_SINK.set(sink);
}

pub fn host_audio_sink() -> Option<&'static Arc<dyn HostPcmSink>> {
    HOST_AUDIO_SINK.get()
}

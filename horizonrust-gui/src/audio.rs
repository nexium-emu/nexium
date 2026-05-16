pub struct AudioSink;

impl AudioSink {
    pub fn new() -> Result<Self, String> {
        log::info!("Audio sink initialized (stub - CPAL integration deferred)");
        Ok(Self)
    }

    pub fn is_available(&self) -> bool {
        true
    }
}

impl Default for AudioSink {
    fn default() -> Self {
        Self
    }
}

use std::collections::HashMap;

struct DecoderState {
    decoder: opus::Decoder,
    channels: u32,
}

pub struct HwOpusService {
    decoders: HashMap<u32, DecoderState>,
}

impl HwOpusService {
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
        }
    }

    pub fn work_buffer_size(channels: u32) -> u32 {
        let channels = channels.clamp(1, 2);
        0x1000 + 0x8000 * channels
    }

    pub fn open(&mut self, key: u32, sample_rate: u32, channels: u32) -> bool {
        let rate = match sample_rate {
            8000 | 12000 | 16000 | 24000 | 48000 => sample_rate,
            _ => 48000,
        };
        let ch = if channels >= 2 {
            opus::Channels::Stereo
        } else {
            opus::Channels::Mono
        };
        match opus::Decoder::new(rate, ch) {
            Ok(decoder) => {
                self.decoders.insert(
                    key,
                    DecoderState {
                        decoder,
                        channels: channels.clamp(1, 2),
                    },
                );
                true
            }
            Err(e) => {
                log::warn!("hwopus: failed to open decoder rate={rate} ch={channels}: {e}");
                false
            }
        }
    }

    pub fn close(&mut self, key: u32) {
        self.decoders.remove(&key);
    }

    pub fn decode(&mut self, key: u32, input: &[u8], reset: bool) -> (u32, u32, Vec<u8>) {
        let Some(state) = self.decoders.get_mut(&key) else {
            return (0, 0, Vec::new());
        };
        if reset {
            let _ = state.decoder.reset_state();
        }
        if input.len() < 8 {
            return (0, 0, Vec::new());
        }
        let packet_len = u32::from_be_bytes([input[0], input[1], input[2], input[3]]) as usize;
        let end = (8 + packet_len).min(input.len());
        let packet = &input[8..end];
        let channels = state.channels as usize;
        let mut pcm = vec![0i16; 5760 * channels];
        match state.decoder.decode(packet, &mut pcm, false) {
            Ok(samples) => {
                let total = samples * channels;
                let mut bytes = Vec::with_capacity(total * 2);
                for s in &pcm[..total] {
                    bytes.extend_from_slice(&s.to_le_bytes());
                }
                (end as u32, samples as u32, bytes)
            }
            Err(e) => {
                log::warn!("hwopus: decode failed ({} bytes): {e}", packet.len());
                (end as u32, 0, Vec::new())
            }
        }
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::trace!("hwopus cmd: {}", cmd_id);
        0
    }
}

impl Default for HwOpusService {
    fn default() -> Self {
        Self::new()
    }
}

use std::collections::HashMap;
use std::ptr::NonNull;

struct ServiceDecoder {
    decoder: opus::Decoder,
    channels: u32,
}

pub struct HwOpusService {
    decoders: HashMap<u32, ServiceDecoder>,
}

const HWOPUS_MODULE: u32 = 111;
const OPUS_DECODER_BASE_MONO: u32 = 0x4844;
const OPUS_DECODER_BASE_STEREO: u32 = 0x6A84;
const OPUS_INPUT_BUFFER_SIZE: u32 = 0x600;

const fn hwopus_result(description: u32) -> u32 {
    HWOPUS_MODULE | (description << 9)
}

pub const RESULT_LIB_OPUS_BAD_ARG: u32 = hwopus_result(2);
pub const RESULT_BUFFER_TOO_SMALL: u32 = hwopus_result(3);
pub const RESULT_LIB_OPUS_INTERNAL_ERROR: u32 = hwopus_result(4);
pub const RESULT_LIB_OPUS_UNIMPLEMENTED: u32 = hwopus_result(5);
pub const RESULT_LIB_OPUS_INVALID_STATE: u32 = hwopus_result(6);
pub const RESULT_LIB_OPUS_ALLOC_FAIL: u32 = hwopus_result(7);
pub const RESULT_INPUT_DATA_TOO_SMALL: u32 = hwopus_result(8);
pub const RESULT_LIB_OPUS_INVALID_PACKET: u32 = hwopus_result(17);
pub const RESULT_INVALID_OPUS_DSP_RETURN_CODE: u32 = hwopus_result(259);
pub const RESULT_INVALID_OPUS_SAMPLE_RATE: u32 = hwopus_result(1001);
pub const RESULT_INVALID_OPUS_CHANNEL_COUNT: u32 = hwopus_result(1002);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderParameters {
    pub sample_rate: u32,
    pub channel_count: u32,
    pub use_large_frame_size: bool,
}

#[derive(Debug, Eq, PartialEq)]
pub struct DecodedPacket {
    pub consumed: u32,
    pub sample_count: u32,
    pub elapsed_us: u64,
    pub pcm: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PacketHeader {
    pub payload_size: u32,
    pub final_range: u32,
}

pub struct DecoderState {
    decoder: NonNull<opusic_sys::OpusDecoder>,
    parameters: DecoderParameters,
    max_frame_samples: i32,
}

unsafe impl Send for DecoderState {}

impl Drop for DecoderState {
    fn drop(&mut self) {
        unsafe {
            opusic_sys::opus_decoder_destroy(self.decoder.as_ptr());
        }
    }
}

impl DecoderState {
    pub fn new(parameters: DecoderParameters, transfer_memory_size: u32) -> Result<Self, u32> {
        let required = work_buffer_size_ex(
            parameters.sample_rate,
            parameters.channel_count,
            parameters.use_large_frame_size,
        )?;
        if transfer_memory_size < required {
            return Err(RESULT_BUFFER_TOO_SMALL);
        }

        let rate_divisor = sample_rate_divisor(parameters.sample_rate)?;
        let frame_size = if parameters.use_large_frame_size {
            5_760
        } else {
            1_920
        };
        let max_frame_samples = (frame_size / rate_divisor / 2) as i32;
        let mut error = opusic_sys::OPUS_OK;
        let raw = unsafe {
            opusic_sys::opus_decoder_create(
                parameters.sample_rate as i32,
                parameters.channel_count as i32,
                &mut error,
            )
        };
        let Some(decoder) = NonNull::new(raw) else {
            return Err(if error == opusic_sys::OPUS_OK {
                RESULT_LIB_OPUS_ALLOC_FAIL
            } else {
                opus_error_result(error)
            });
        };
        if error != opusic_sys::OPUS_OK {
            unsafe {
                opusic_sys::opus_decoder_destroy(decoder.as_ptr());
            }
            return Err(opus_error_result(error));
        }

        Ok(Self {
            decoder,
            parameters,
            max_frame_samples,
        })
    }

    pub fn parameters(&self) -> DecoderParameters {
        self.parameters
    }

    pub fn decode(
        &mut self,
        input: &[u8],
        output_capacity: usize,
        reset: bool,
    ) -> Result<DecodedPacket, u32> {
        let (header, payload) = parse_packet(input)?;
        if reset {
            let result = unsafe {
                opusic_sys::opus_decoder_ctl(self.decoder.as_ptr(), opusic_sys::OPUS_RESET_STATE)
            };
            if result != opusic_sys::OPUS_OK {
                return Err(opus_error_result(result));
            }
        }
        let packet_len = i32::try_from(payload.len()).map_err(|_| RESULT_BUFFER_TOO_SMALL)?;
        let expected_samples = if payload.is_empty() {
            self.max_frame_samples
        } else {
            unsafe {
                opusic_sys::opus_packet_get_nb_samples(
                    payload.as_ptr(),
                    packet_len,
                    self.parameters.sample_rate as i32,
                )
            }
        };
        if expected_samples < 0 {
            return Err(opus_error_result(expected_samples));
        }
        if expected_samples > self.max_frame_samples {
            return Err(RESULT_BUFFER_TOO_SMALL);
        }

        let expected_bytes = (expected_samples as usize)
            .saturating_mul(self.parameters.channel_count as usize)
            .saturating_mul(std::mem::size_of::<i16>());
        if expected_bytes > output_capacity {
            return Err(RESULT_BUFFER_TOO_SMALL);
        }

        let mut pcm = vec![
            0i16;
            (self.max_frame_samples as usize)
                .saturating_mul(self.parameters.channel_count as usize)
        ];
        let started = std::time::Instant::now();
        let packet_ptr = if payload.is_empty() {
            std::ptr::null()
        } else {
            payload.as_ptr()
        };
        let decoded_samples = unsafe {
            opusic_sys::opus_decode(
                self.decoder.as_ptr(),
                packet_ptr,
                packet_len,
                pcm.as_mut_ptr(),
                self.max_frame_samples,
                0,
            )
        };
        let elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        if decoded_samples < 0 {
            return Err(opus_error_result(decoded_samples));
        }

        let sample_values =
            (decoded_samples as usize).saturating_mul(self.parameters.channel_count as usize);
        let actual_bytes = sample_values.saturating_mul(std::mem::size_of::<i16>());
        if actual_bytes > output_capacity {
            return Err(RESULT_BUFFER_TOO_SMALL);
        }
        pcm.truncate(sample_values);
        let mut pcm_bytes = Vec::with_capacity(actual_bytes);
        for sample in pcm {
            pcm_bytes.extend_from_slice(&sample.to_le_bytes());
        }

        Ok(DecodedPacket {
            consumed: header.payload_size.saturating_add(8),
            sample_count: decoded_samples as u32,
            elapsed_us,
            pcm: pcm_bytes,
        })
    }
}

pub fn parse_open_request(input: &[u8], extended: bool) -> Result<(DecoderParameters, u32), u32> {
    let parameter_size = if extended { 0x10 } else { 0x8 };
    if input.len() < parameter_size + 4 {
        return Err(RESULT_INPUT_DATA_TOO_SMALL);
    }
    let sample_rate = read_u32_le(input, 0);
    let channel_count = read_u32_le(input, 4);
    let use_large_frame_size = extended && input[8] != 0;
    let transfer_memory_size = read_u32_le(input, parameter_size);
    work_buffer_size_ex(sample_rate, channel_count, use_large_frame_size)?;
    Ok((
        DecoderParameters {
            sample_rate,
            channel_count,
            use_large_frame_size,
        },
        transfer_memory_size,
    ))
}

pub fn parse_packet(input: &[u8]) -> Result<(PacketHeader, &[u8]), u32> {
    if input.len() <= 8 {
        return Err(RESULT_INPUT_DATA_TOO_SMALL);
    }
    let header = PacketHeader {
        payload_size: u32::from_be_bytes(input[0..4].try_into().unwrap()),
        final_range: u32::from_be_bytes(input[4..8].try_into().unwrap()),
    };
    let payload_size = header.payload_size as usize;
    if payload_size > OPUS_INPUT_BUFFER_SIZE as usize
        || payload_size.saturating_add(8) > input.len()
    {
        return Err(RESULT_BUFFER_TOO_SMALL);
    }
    Ok((header, &input[8..8 + payload_size]))
}

pub fn opus_error_result(error: i32) -> u32 {
    match error {
        opusic_sys::OPUS_OK => 0,
        opusic_sys::OPUS_BAD_ARG => RESULT_LIB_OPUS_BAD_ARG,
        opusic_sys::OPUS_BUFFER_TOO_SMALL => RESULT_BUFFER_TOO_SMALL,
        opusic_sys::OPUS_INTERNAL_ERROR => RESULT_LIB_OPUS_INTERNAL_ERROR,
        opusic_sys::OPUS_INVALID_PACKET => RESULT_LIB_OPUS_INVALID_PACKET,
        opusic_sys::OPUS_UNIMPLEMENTED => RESULT_LIB_OPUS_UNIMPLEMENTED,
        opusic_sys::OPUS_INVALID_STATE => RESULT_LIB_OPUS_INVALID_STATE,
        opusic_sys::OPUS_ALLOC_FAIL => RESULT_LIB_OPUS_ALLOC_FAIL,
        _ => RESULT_INVALID_OPUS_DSP_RETURN_CODE,
    }
}

pub fn get_work_buffer_size_ex(input: &[u8]) -> Result<(u32, u32, bool, u32), u32> {
    if input.len() < 0x10 {
        return Err(RESULT_INPUT_DATA_TOO_SMALL);
    }
    let sample_rate = read_u32_le(input, 0);
    let channel_count = read_u32_le(input, 4);
    let use_large_frame_size = input.get(8).copied().unwrap_or(0) != 0;
    let size = work_buffer_size_ex(sample_rate, channel_count, use_large_frame_size)?;
    Ok((sample_rate, channel_count, use_large_frame_size, size))
}

pub fn get_work_buffer_size(input: &[u8]) -> Result<(u32, u32, u32), u32> {
    if input.len() < 8 {
        return Err(RESULT_INPUT_DATA_TOO_SMALL);
    }
    let sample_rate = read_u32_le(input, 0);
    let channel_count = read_u32_le(input, 4);
    let size = work_buffer_size_ex(sample_rate, channel_count, false)?;
    Ok((sample_rate, channel_count, size))
}

pub fn work_buffer_size_ex(
    sample_rate: u32,
    channel_count: u32,
    use_large_frame_size: bool,
) -> Result<u32, u32> {
    let decoder_base = match channel_count {
        1 => OPUS_DECODER_BASE_MONO,
        2 => OPUS_DECODER_BASE_STEREO,
        _ => return Err(RESULT_INVALID_OPUS_CHANNEL_COUNT),
    };
    let rate_divisor = sample_rate_divisor(sample_rate)?;
    let frame_size = if use_large_frame_size { 5_760 } else { 1_920 };
    let output_scratch = (frame_size * channel_count) / rate_divisor;
    let output_scratch = (output_scratch + 63) & !63;
    Ok(decoder_base + output_scratch + OPUS_INPUT_BUFFER_SIZE)
}

fn sample_rate_divisor(sample_rate: u32) -> Result<u32, u32> {
    match sample_rate {
        8_000 => Ok(6),
        12_000 => Ok(4),
        16_000 => Ok(3),
        24_000 => Ok(2),
        48_000 => Ok(1),
        _ => Err(RESULT_INVALID_OPUS_SAMPLE_RATE),
    }
}

fn read_u32_le(input: &[u8], offset: usize) -> u32 {
    input
        .get(offset..offset + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .unwrap_or(0)
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
                    ServiceDecoder {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_princess_peach_parameters() {
        let mut input = [0u8; 0x10];
        input[0..4].copy_from_slice(&8_000u32.to_le_bytes());
        input[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            get_work_buffer_size_ex(&input),
            Ok((8_000, 1, false, 0x4F84))
        );
    }

    #[test]
    fn decoder_frame_limits_match_horizon_20_and_60_ms() {
        for (sample_rate, expected_normal, expected_large) in [
            (8_000, 160, 480),
            (12_000, 240, 720),
            (16_000, 320, 960),
            (24_000, 480, 1_440),
            (48_000, 960, 2_880),
        ] {
            for (use_large_frame_size, expected) in
                [(false, expected_normal), (true, expected_large)]
            {
                let parameters = DecoderParameters {
                    sample_rate,
                    channel_count: 2,
                    use_large_frame_size,
                };
                let size = work_buffer_size_ex(sample_rate, 2, use_large_frame_size).unwrap();
                let decoder = DecoderState::new(parameters, size).unwrap();
                assert_eq!(decoder.max_frame_samples, expected);
            }
        }
    }

    #[test]
    fn parses_old_and_extended_open_requests() {
        let mut old = [0u8; 0xC];
        old[0..4].copy_from_slice(&48_000u32.to_le_bytes());
        old[4..8].copy_from_slice(&2u32.to_le_bytes());
        old[8..12].copy_from_slice(&0x7F84u32.to_le_bytes());
        assert_eq!(
            parse_open_request(&old, false),
            Ok((
                DecoderParameters {
                    sample_rate: 48_000,
                    channel_count: 2,
                    use_large_frame_size: false,
                },
                0x7F84
            ))
        );
        assert_eq!(get_work_buffer_size(&old[..8]), Ok((48_000, 2, 0x7F84)));

        let mut extended = [0u8; 0x14];
        extended[0..4].copy_from_slice(&8_000u32.to_le_bytes());
        extended[4..8].copy_from_slice(&1u32.to_le_bytes());
        extended[8] = 1;
        extended[0x10..0x14].copy_from_slice(&0x5204u32.to_le_bytes());
        assert_eq!(
            parse_open_request(&extended, true),
            Ok((
                DecoderParameters {
                    sample_rate: 8_000,
                    channel_count: 1,
                    use_large_frame_size: true,
                },
                0x5204
            ))
        );
    }

    #[test]
    fn parses_big_endian_packet_header_and_bounds_payload() {
        let packet = [0, 0, 0, 3, 0x12, 0x34, 0x56, 0x78, 0xF8, 0xAA, 0xBB];
        let (header, payload) = parse_packet(&packet).unwrap();
        assert_eq!(
            header,
            PacketHeader {
                payload_size: 3,
                final_range: 0x1234_5678,
            }
        );
        assert_eq!(payload, &[0xF8, 0xAA, 0xBB]);

        assert_eq!(parse_packet(&packet[..8]), Err(RESULT_INPUT_DATA_TOO_SMALL));
        let truncated = [0, 0, 0, 4, 0, 0, 0, 0, 1];
        assert_eq!(parse_packet(&truncated), Err(RESULT_BUFFER_TOO_SMALL));
    }

    #[test]
    fn maps_libopus_errors_to_horizon_results() {
        let expected = [
            (opusic_sys::OPUS_OK, 0),
            (opusic_sys::OPUS_BAD_ARG, RESULT_LIB_OPUS_BAD_ARG),
            (opusic_sys::OPUS_BUFFER_TOO_SMALL, RESULT_BUFFER_TOO_SMALL),
            (
                opusic_sys::OPUS_INTERNAL_ERROR,
                RESULT_LIB_OPUS_INTERNAL_ERROR,
            ),
            (
                opusic_sys::OPUS_INVALID_PACKET,
                RESULT_LIB_OPUS_INVALID_PACKET,
            ),
            (
                opusic_sys::OPUS_UNIMPLEMENTED,
                RESULT_LIB_OPUS_UNIMPLEMENTED,
            ),
            (
                opusic_sys::OPUS_INVALID_STATE,
                RESULT_LIB_OPUS_INVALID_STATE,
            ),
            (opusic_sys::OPUS_ALLOC_FAIL, RESULT_LIB_OPUS_ALLOC_FAIL),
            (-999, RESULT_INVALID_OPUS_DSP_RETURN_CODE),
        ];
        for (error, result) in expected {
            assert_eq!(opus_error_result(error), result);
        }
    }

    #[test]
    fn normal_frame_rate_channel_table() {
        let expected = [
            ((8_000, 1), 0x4F84),
            ((8_000, 2), 0x7304),
            ((12_000, 1), 0x5044),
            ((12_000, 2), 0x7444),
            ((16_000, 1), 0x50C4),
            ((16_000, 2), 0x7584),
            ((24_000, 1), 0x5204),
            ((24_000, 2), 0x7804),
            ((48_000, 1), 0x55C4),
            ((48_000, 2), 0x7F84),
        ];
        for ((sample_rate, channel_count), size) in expected {
            assert_eq!(
                work_buffer_size_ex(sample_rate, channel_count, false),
                Ok(size)
            );
        }
    }

    #[test]
    fn large_frame_rate_channel_table() {
        let expected = [
            ((8_000, 1), 0x5204),
            ((8_000, 2), 0x7804),
            ((12_000, 1), 0x5404),
            ((12_000, 2), 0x7BC4),
            ((16_000, 1), 0x55C4),
            ((16_000, 2), 0x7F84),
            ((24_000, 1), 0x5984),
            ((24_000, 2), 0x8704),
            ((48_000, 1), 0x64C4),
            ((48_000, 2), 0x9D84),
        ];
        for ((sample_rate, channel_count), size) in expected {
            assert_eq!(
                work_buffer_size_ex(sample_rate, channel_count, true),
                Ok(size)
            );
        }
    }

    #[test]
    fn validates_channel_before_sample_rate() {
        assert_eq!(
            work_buffer_size_ex(44_100, 0, false),
            Err(RESULT_INVALID_OPUS_CHANNEL_COUNT)
        );
        assert_eq!(
            work_buffer_size_ex(44_100, 1, false),
            Err(RESULT_INVALID_OPUS_SAMPLE_RATE)
        );
    }
}

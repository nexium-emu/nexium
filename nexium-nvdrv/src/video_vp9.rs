use crate::video_decode::VideoDecodeError;

pub const VP9_PICTURE_INFO_SIZE: usize = 0x100;
pub const VP9_ENTROPY_PROBS_SIZE: usize = 0xEA0;

const DIFF_UPDATE_PROBABILITY: i32 = 252;
const FRAME_SYNC_CODE: u32 = 0x498342;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vp9EntropyProbs {
    pub y_mode_prob: [u8; 36],
    pub partition_prob: [u8; 64],
    pub coef_probs: [u8; 1728],
    pub switchable_interp_prob: [u8; 8],
    pub inter_mode_prob: [u8; 28],
    pub intra_inter_prob: [u8; 4],
    pub comp_inter_prob: [u8; 5],
    pub single_ref_prob: [u8; 10],
    pub comp_ref_prob: [u8; 5],
    pub tx_32x32_prob: [u8; 6],
    pub tx_16x16_prob: [u8; 4],
    pub tx_8x8_prob: [u8; 2],
    pub skip_probs: [u8; 3],
    pub joints: [u8; 3],
    pub sign: [u8; 2],
    pub classes: [u8; 20],
    pub class_0: [u8; 2],
    pub prob_bits: [u8; 20],
    pub class_0_fr: [u8; 12],
    pub fr: [u8; 6],
    pub class_0_hp: [u8; 2],
    pub high_precision: [u8; 2],
}

impl Vp9EntropyProbs {
    pub const fn zeroed() -> Self {
        Self {
            y_mode_prob: [0; 36],
            partition_prob: [0; 64],
            coef_probs: [0; 1728],
            switchable_interp_prob: [0; 8],
            inter_mode_prob: [0; 28],
            intra_inter_prob: [0; 4],
            comp_inter_prob: [0; 5],
            single_ref_prob: [0; 10],
            comp_ref_prob: [0; 5],
            tx_32x32_prob: [0; 6],
            tx_16x16_prob: [0; 4],
            tx_8x8_prob: [0; 2],
            skip_probs: [0; 3],
            joints: [0; 3],
            sign: [0; 2],
            classes: [0; 20],
            class_0: [0; 2],
            prob_bits: [0; 20],
            class_0_fr: [0; 12],
            fr: [0; 6],
            class_0_hp: [0; 2],
            high_precision: [0; 2],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vp9Segmentation {
    pub enabled: u8,
    pub update_map: u8,
    pub temporal_update: u8,
    pub abs_delta: u8,
    pub feature_enabled: [[u8; 4]; 8],
    pub feature_data: [[i16; 4]; 8],
}

#[derive(Clone, Debug)]
pub struct Vp9PictureInfo {
    pub bitstream_size: u32,
    pub frame_offsets: [u64; 4],
    pub ref_frame_sign_bias: [u8; 4],
    pub base_q_index: i32,
    pub y_dc_delta_q: i32,
    pub uv_dc_delta_q: i32,
    pub uv_ac_delta_q: i32,
    pub transform_mode: i32,
    pub interp_filter: i32,
    pub reference_mode: i32,
    pub log2_tile_cols: i32,
    pub log2_tile_rows: i32,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
    pub entropy: Vp9EntropyProbs,
    pub frame_width: i32,
    pub frame_height: i32,
    pub first_level: u8,
    pub sharpness_level: u8,
    pub is_key_frame: bool,
    pub intra_only: bool,
    pub last_frame_was_key: bool,
    pub error_resilient_mode: bool,
    pub last_frame_shown: bool,
    pub show_frame: bool,
    pub lossless: bool,
    pub allow_high_precision_mv: bool,
    pub segment_enabled: bool,
    pub mode_ref_delta_enabled: bool,
    pub segmentation: Vp9Segmentation,
}

pub struct Vp9SegmentProbs {
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
}

const FLAG_IS_KEY_FRAME: u32 = 1 << 0;
const FLAG_LAST_FRAME_IS_KEY: u32 = 1 << 1;
const FLAG_ERROR_RESILIENT: u32 = 1 << 3;
const FLAG_LAST_SHOW_FRAME: u32 = 1 << 4;
const FLAG_INTRA_ONLY: u32 = 1 << 5;

pub fn parse_picture_info(bytes: &[u8]) -> Result<Vp9PictureInfo, VideoDecodeError> {
    if bytes.len() < VP9_PICTURE_INFO_SIZE {
        return Err(VideoDecodeError::ContextTooShort {
            actual: bytes.len(),
        });
    }
    let flags = read_u32(bytes, 0x68);
    let mut segmentation = Vp9Segmentation {
        enabled: bytes[0x80],
        update_map: bytes[0x81],
        temporal_update: bytes[0x82],
        abs_delta: bytes[0x83],
        feature_enabled: [[0; 4]; 8],
        feature_data: [[0; 4]; 8],
    };
    for i in 0..8 {
        for j in 0..4 {
            segmentation.feature_enabled[i][j] = bytes[0x84 + i * 4 + j];
            segmentation.feature_data[i][j] = read_i16(bytes, 0xA4 + (i * 4 + j) * 2);
        }
    }
    let width = read_i16(bytes, 0x60) as i32;
    let height = read_i16(bytes, 0x62) as i32;
    if width <= 0 || height <= 0 {
        return Err(VideoDecodeError::InvalidContextField {
            field: "vp9_frame_size",
            value: ((width as i64) << 16) | (height as i64 & 0xFFFF),
        });
    }
    Ok(Vp9PictureInfo {
        bitstream_size: read_u32(bytes, 0x30),
        frame_offsets: [0; 4],
        ref_frame_sign_bias: [bytes[0x6C], bytes[0x6D], bytes[0x6E], bytes[0x6F]],
        base_q_index: bytes[0x72] as i32,
        y_dc_delta_q: bytes[0x73] as i32,
        uv_ac_delta_q: bytes[0x74] as i32,
        uv_dc_delta_q: bytes[0x75] as i32,
        transform_mode: bytes[0x77] as i32,
        interp_filter: bytes[0x79] as i32,
        reference_mode: bytes[0x7A] as i32,
        log2_tile_cols: bytes[0x7E] as i32,
        log2_tile_rows: bytes[0x7F] as i32,
        ref_deltas: [
            bytes[0xE5] as i8,
            bytes[0xE6] as i8,
            bytes[0xE7] as i8,
            bytes[0xE8] as i8,
        ],
        mode_deltas: [bytes[0xE9] as i8, bytes[0xEA] as i8],
        entropy: Vp9EntropyProbs::zeroed(),
        frame_width: width,
        frame_height: height,
        first_level: bytes[0x70],
        sharpness_level: bytes[0x71],
        is_key_frame: flags & FLAG_IS_KEY_FRAME != 0,
        intra_only: flags & FLAG_INTRA_ONLY != 0,
        last_frame_was_key: flags & FLAG_LAST_FRAME_IS_KEY != 0,
        error_resilient_mode: flags & FLAG_ERROR_RESILIENT != 0,
        last_frame_shown: flags & FLAG_LAST_SHOW_FRAME != 0,
        show_frame: true,
        lossless: bytes[0x76] != 0,
        allow_high_precision_mv: bytes[0x78] != 0,
        segment_enabled: bytes[0x80] != 0,
        mode_ref_delta_enabled: bytes[0xE4] != 0,
        segmentation,
    })
}

pub fn parse_entropy_probs(
    bytes: &[u8],
) -> Result<(Vp9EntropyProbs, Vp9SegmentProbs), VideoDecodeError> {
    if bytes.len() < VP9_ENTROPY_PROBS_SIZE {
        return Err(VideoDecodeError::ContextTooShort {
            actual: bytes.len(),
        });
    }
    let mut probs = Vp9EntropyProbs::zeroed();
    probs.inter_mode_prob.copy_from_slice(&bytes[0x400..0x41C]);
    probs.intra_inter_prob.copy_from_slice(&bytes[0x41C..0x420]);
    probs.tx_8x8_prob.copy_from_slice(&bytes[0x470..0x472]);
    probs.tx_16x16_prob.copy_from_slice(&bytes[0x472..0x476]);
    probs.tx_32x32_prob.copy_from_slice(&bytes[0x476..0x47C]);
    for i in 0..4 {
        for j in 0..9 {
            probs.y_mode_prob[j + 9 * i] = if j < 8 {
                bytes[0x480 + i * 8 + j]
            } else {
                bytes[0x47C + i]
            };
        }
    }
    probs.partition_prob.copy_from_slice(&bytes[0x4E0..0x520]);
    probs
        .switchable_interp_prob
        .copy_from_slice(&bytes[0x52A..0x532]);
    probs.comp_inter_prob.copy_from_slice(&bytes[0x532..0x537]);
    probs.skip_probs.copy_from_slice(&bytes[0x537..0x53A]);
    probs.joints.copy_from_slice(&bytes[0x53B..0x53E]);
    probs.sign.copy_from_slice(&bytes[0x53E..0x540]);
    probs.class_0.copy_from_slice(&bytes[0x540..0x542]);
    probs.fr.copy_from_slice(&bytes[0x542..0x548]);
    probs.class_0_hp.copy_from_slice(&bytes[0x548..0x54A]);
    probs.high_precision.copy_from_slice(&bytes[0x54A..0x54C]);
    probs.classes.copy_from_slice(&bytes[0x54C..0x560]);
    probs.class_0_fr.copy_from_slice(&bytes[0x560..0x56C]);
    probs.prob_bits.copy_from_slice(&bytes[0x56C..0x580]);
    probs.single_ref_prob.copy_from_slice(&bytes[0x580..0x58A]);
    probs.comp_ref_prob.copy_from_slice(&bytes[0x58A..0x58F]);
    for i in (0..2304).step_by(4) {
        let j = i - i / 4;
        probs.coef_probs[j] = bytes[0x5A0 + i];
        probs.coef_probs[j + 1] = bytes[0x5A0 + i + 1];
        probs.coef_probs[j + 2] = bytes[0x5A0 + i + 2];
    }
    let mut tree_probs = [0u8; 7];
    tree_probs.copy_from_slice(&bytes[0x387..0x38E]);
    let mut pred_probs = [0u8; 3];
    pred_probs.copy_from_slice(&bytes[0x38E..0x391]);
    Ok((
        probs,
        Vp9SegmentProbs {
            tree_probs,
            pred_probs,
        },
    ))
}

fn calc_min_log2_tile_cols(frame_width: i32) -> i32 {
    let sb64_cols = (frame_width + 63) / 64;
    let mut min_log2 = 0;
    while (64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    min_log2
}

fn calc_max_log2_tile_cols(frame_width: i32) -> i32 {
    let sb64_cols = (frame_width + 63) / 64;
    let mut max_log2 = 1;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 - 1
}

fn recenter_non_neg(new_prob: i32, old_prob: i32) -> i32 {
    if new_prob > old_prob * 2 {
        new_prob
    } else if new_prob >= old_prob {
        (new_prob - old_prob) * 2
    } else {
        (old_prob - new_prob) * 2 - 1
    }
}

fn remap_probability(new_prob: i32, old_prob: i32) -> i32 {
    let new_prob = new_prob - 1;
    let old_prob = old_prob - 1;
    let index = if old_prob * 2 <= 0xff {
        recenter_non_neg(new_prob, old_prob).max(1) - 1
    } else {
        recenter_non_neg(0xff - 1 - new_prob, 0xff - 1 - old_prob).max(1) - 1
    };
    MAP_LUT[index as usize] as i32
}

pub struct VpxRangeEncoder {
    buffer: Vec<u8>,
    low_value: u32,
    range: u32,
    count: i32,
}

impl Default for VpxRangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VpxRangeEncoder {
    pub fn new() -> Self {
        let mut encoder = Self {
            buffer: Vec::new(),
            low_value: 0,
            range: 0xff,
            count: -24,
        };
        encoder.write_bool(false);
        encoder
    }

    pub fn write_bits(&mut self, value: i32, value_size: i32) {
        for bit in (0..value_size).rev() {
            self.write_bool((value >> bit) & 1 != 0);
        }
    }

    pub fn write_bool(&mut self, bit: bool) {
        self.write_prob(bit, 128);
    }

    pub fn write_prob(&mut self, bit: bool, probability: i32) {
        let mut local_range = self.range;
        let split = 1 + (((local_range - 1) * probability as u32) >> 8);
        local_range = split;
        if bit {
            self.low_value = self.low_value.wrapping_add(split);
            local_range = self.range - split;
        }
        let mut shift = NORM_LUT[local_range as usize] as i32;
        local_range <<= shift;
        self.count += shift;
        if self.count >= 0 {
            let offset = shift - self.count;
            if (self.low_value.wrapping_shl((offset - 1) as u32) >> 31) != 0 {
                let mut index = self.buffer.len();
                while index > 0 && self.buffer[index - 1] == 0xff {
                    self.buffer[index - 1] = 0;
                    index -= 1;
                }
                if index > 0 {
                    self.buffer[index - 1] = self.buffer[index - 1].wrapping_add(1);
                }
            }
            self.buffer.push((self.low_value >> (24 - offset)) as u8);
            self.low_value = self.low_value.wrapping_shl(offset as u32);
            shift = self.count;
            self.low_value &= 0xffffff;
            self.count -= 8;
        }
        self.low_value = self.low_value.wrapping_shl(shift as u32);
        self.range = local_range;
    }

    pub fn end(&mut self) {
        for _ in 0..32 {
            self.write_bool(false);
        }
    }

    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }
}

pub struct VpxBitStreamWriter {
    byte_array: Vec<u8>,
    buffer: i32,
    buffer_pos: i32,
}

impl Default for VpxBitStreamWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl VpxBitStreamWriter {
    const BUFFER_SIZE: i32 = 8;

    pub fn new() -> Self {
        Self {
            byte_array: Vec::new(),
            buffer: 0,
            buffer_pos: 0,
        }
    }

    pub fn write_u(&mut self, value: u32, value_size: u32) {
        self.write_bits(value, value_size);
    }

    pub fn write_s(&mut self, value: i32, value_size: u32) {
        let sign = value < 0;
        let magnitude = if sign { -value } else { value };
        self.write_bits(((magnitude as u32) << 1) | u32::from(sign), value_size + 1);
    }

    pub fn write_delta_q(&mut self, value: u32) {
        let delta_coded = value != 0;
        self.write_bit(delta_coded);
        if delta_coded {
            self.write_bits(value, 4);
        }
    }

    pub fn write_bit(&mut self, state: bool) {
        self.write_bits(u32::from(state), 1);
    }

    fn write_bits(&mut self, value: u32, bit_count: u32) {
        let mut value_pos = 0i32;
        let mut remaining = bit_count as i32;
        while remaining > 0 {
            let mut copy_size = remaining;
            let free = self.free_buffer_bits();
            if copy_size > free {
                copy_size = free;
            }
            let mask = (1i32 << copy_size) - 1;
            let src_shift = (bit_count as i32 - value_pos) - copy_size;
            let dst_shift = (Self::BUFFER_SIZE - self.buffer_pos) - copy_size;
            self.buffer |= (((value >> src_shift) as i32) & mask) << dst_shift;
            value_pos += copy_size;
            self.buffer_pos += copy_size;
            remaining -= copy_size;
        }
    }

    fn free_buffer_bits(&mut self) -> i32 {
        if self.buffer_pos == Self::BUFFER_SIZE {
            self.flush();
        }
        Self::BUFFER_SIZE - self.buffer_pos
    }

    pub fn flush(&mut self) {
        if self.buffer_pos == 0 {
            return;
        }
        self.byte_array.push(self.buffer as u8);
        self.buffer = 0;
        self.buffer_pos = 0;
    }

    pub fn into_byte_array(self) -> Vec<u8> {
        self.byte_array
    }
}

struct Vp9PendingFrame {
    info: Vp9PictureInfo,
    bitstream: Vec<u8>,
}

pub struct Vp9FrameComposer {
    next_frame: Option<Vp9PendingFrame>,
    frame_ctxs: [Vp9EntropyProbs; 4],
    prev_frame_probs: Vp9EntropyProbs,
    loop_filter_ref_deltas: [i8; 4],
    loop_filter_mode_deltas: [i8; 2],
    last_segmentation: Vp9Segmentation,
    swap_ref_indices: bool,
}

impl Default for Vp9FrameComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp9FrameComposer {
    pub fn new() -> Self {
        Self {
            next_frame: None,
            frame_ctxs: [
                Vp9EntropyProbs::zeroed(),
                Vp9EntropyProbs::zeroed(),
                Vp9EntropyProbs::zeroed(),
                Vp9EntropyProbs::zeroed(),
            ],
            prev_frame_probs: Vp9EntropyProbs::zeroed(),
            loop_filter_ref_deltas: [0; 4],
            loop_filter_mode_deltas: [0; 2],
            last_segmentation: Vp9Segmentation::default(),
            swap_ref_indices: false,
        }
    }

    pub fn compose(
        &mut self,
        info: Vp9PictureInfo,
        bitstream: Vec<u8>,
        seg_probs: &Vp9SegmentProbs,
    ) -> (Vec<u8>, bool) {
        let current_segmentation = info.segmentation.clone();
        let (mut frame_info, frame_bitstream) = match self.next_frame.take() {
            Some(mut pending) => {
                pending.info.show_frame = info.last_frame_shown;
                let out = (pending.info, pending.bitstream);
                self.next_frame = Some(Vp9PendingFrame {
                    info: info.clone(),
                    bitstream,
                });
                out
            }
            None => {
                self.next_frame = Some(Vp9PendingFrame {
                    info: info.clone(),
                    bitstream: bitstream.clone(),
                });
                (info.clone(), bitstream)
            }
        };
        let next_offsets = self
            .next_frame
            .as_ref()
            .map(|frame| frame.info.frame_offsets)
            .unwrap_or_default();
        let mut uncomp_writer = self.compose_uncompressed_header(
            &mut frame_info,
            next_offsets,
            &current_segmentation,
            seg_probs,
        );
        let compressed_header = self.compose_compressed_header(&frame_info);
        uncomp_writer.write_u(compressed_header.len() as u32, 16);
        uncomp_writer.flush();
        let uncompressed_header = uncomp_writer.into_byte_array();

        let mut frame = Vec::with_capacity(
            uncompressed_header.len() + compressed_header.len() + frame_bitstream.len(),
        );
        frame.extend_from_slice(&uncompressed_header);
        frame.extend_from_slice(&compressed_header);
        frame.extend_from_slice(&frame_bitstream);
        (frame, frame_info.show_frame)
    }

    fn compose_uncompressed_header(
        &mut self,
        info: &mut Vp9PictureInfo,
        next_offsets: [u64; 4],
        current_segmentation: &Vp9Segmentation,
        seg_probs: &Vp9SegmentProbs,
    ) -> VpxBitStreamWriter {
        let mut writer = VpxBitStreamWriter::new();
        writer.write_u(2, 2);
        writer.write_u(0, 2);
        writer.write_bit(false);
        writer.write_bit(!info.is_key_frame);
        writer.write_bit(info.show_frame);
        writer.write_bit(info.error_resilient_mode);

        if info.is_key_frame {
            writer.write_u(FRAME_SYNC_CODE, 24);
            writer.write_u(0, 3);
            writer.write_u(0, 1);
            writer.write_u((info.frame_width - 1) as u32, 16);
            writer.write_u((info.frame_height - 1) as u32, 16);
            writer.write_bit(false);
            self.prev_frame_probs = DEFAULT_PROBS.clone();
            self.swap_ref_indices = false;
            self.loop_filter_ref_deltas = [0; 4];
            self.loop_filter_mode_deltas = [0; 2];
            for ctx in &mut self.frame_ctxs {
                *ctx = DEFAULT_PROBS.clone();
            }
            info.intra_only = true;
        } else {
            if !info.show_frame {
                writer.write_bit(info.intra_only);
            } else {
                info.intra_only = false;
            }
            if !info.error_resilient_mode {
                writer.write_u(0, 2);
            }
            let curr_offsets = info.frame_offsets;
            let ref_frames_different = curr_offsets[1] != curr_offsets[2];
            let next_references_swap =
                next_offsets[1] == curr_offsets[2] || next_offsets[2] == curr_offsets[1];
            let needs_ref_swap = ref_frames_different && next_references_swap;
            if needs_ref_swap {
                self.swap_ref_indices = !self.swap_ref_indices;
            }
            let mut refresh_frame_flags = 0u32;
            for index in 0..3 {
                if curr_offsets[3] == next_offsets[index] {
                    refresh_frame_flags |= 1 << index;
                }
            }
            if self.swap_ref_indices {
                let golden = (refresh_frame_flags >> 1) & 1;
                let alt = (refresh_frame_flags >> 2) & 1;
                refresh_frame_flags = (refresh_frame_flags & 1) | (alt << 1) | (golden << 2);
            }
            if info.intra_only {
                writer.write_u(FRAME_SYNC_CODE, 24);
                writer.write_u(refresh_frame_flags, 8);
                writer.write_u((info.frame_width - 1) as u32, 16);
                writer.write_u((info.frame_height - 1) as u32, 16);
                writer.write_bit(false);
            } else {
                let swap_indices = needs_ref_swap ^ self.swap_ref_indices;
                let ref_frame_index: [u32; 3] = if swap_indices { [0, 2, 1] } else { [0, 1, 2] };
                writer.write_u(refresh_frame_flags, 8);
                for index in 1..4 {
                    writer.write_u(ref_frame_index[index - 1], 3);
                    writer.write_u(u32::from(info.ref_frame_sign_bias[index]), 1);
                }
                writer.write_bit(true);
                writer.write_bit(false);
                writer.write_bit(info.allow_high_precision_mv);
                writer.write_bit(info.interp_filter == 4);
                if info.interp_filter != 4 {
                    writer.write_u(info.interp_filter as u32, 2);
                }
            }
        }

        if !info.error_resilient_mode {
            writer.write_bit(true);
            writer.write_bit(true);
        }

        let frame_ctx_idx = usize::from(!info.show_frame);
        writer.write_u(frame_ctx_idx as u32, 2);
        self.prev_frame_probs = self.frame_ctxs[frame_ctx_idx].clone();
        self.frame_ctxs[frame_ctx_idx] = info.entropy.clone();

        writer.write_u(u32::from(info.first_level), 6);
        writer.write_u(u32::from(info.sharpness_level), 3);
        writer.write_bit(info.mode_ref_delta_enabled);

        if info.mode_ref_delta_enabled {
            let mut update_ref_deltas = [false; 4];
            let mut update_mode_deltas = [false; 2];
            let mut loop_filter_delta_update = false;
            for index in 0..4 {
                let differing = self.loop_filter_ref_deltas[index] != info.ref_deltas[index];
                update_ref_deltas[index] = differing;
                loop_filter_delta_update |= differing;
            }
            for index in 0..2 {
                let differing = self.loop_filter_mode_deltas[index] != info.mode_deltas[index];
                update_mode_deltas[index] = differing;
                loop_filter_delta_update |= differing;
            }
            writer.write_bit(loop_filter_delta_update);
            if loop_filter_delta_update {
                for index in 0..4 {
                    writer.write_bit(update_ref_deltas[index]);
                    if update_ref_deltas[index] {
                        writer.write_s(i32::from(info.ref_deltas[index]), 6);
                    }
                }
                for index in 0..2 {
                    writer.write_bit(update_mode_deltas[index]);
                    if update_mode_deltas[index] {
                        writer.write_s(i32::from(info.mode_deltas[index]), 6);
                    }
                }
                self.loop_filter_ref_deltas = info.ref_deltas;
                self.loop_filter_mode_deltas = info.mode_deltas;
            }
        }

        writer.write_u(info.base_q_index as u32, 8);
        writer.write_delta_q(info.y_dc_delta_q as u32);
        writer.write_delta_q(info.uv_dc_delta_q as u32);
        writer.write_delta_q(info.uv_ac_delta_q as u32);

        self.write_segmentation(&mut writer, current_segmentation, seg_probs);

        let min_tile_cols_log2 = calc_min_log2_tile_cols(info.frame_width);
        let max_tile_cols_log2 = calc_max_log2_tile_cols(info.frame_width);
        let tile_cols_log2_diff = info.log2_tile_cols - min_tile_cols_log2;
        let tile_cols_log2_inc_mask = (1 << tile_cols_log2_diff) - 1;
        if info.log2_tile_cols < max_tile_cols_log2 {
            writer.write_u(
                (tile_cols_log2_inc_mask << 1) as u32,
                (tile_cols_log2_diff + 1) as u32,
            );
        } else {
            writer.write_u(tile_cols_log2_inc_mask as u32, tile_cols_log2_diff as u32);
        }
        let tile_rows_log2_is_nonzero = info.log2_tile_rows != 0;
        writer.write_bit(tile_rows_log2_is_nonzero);
        if tile_rows_log2_is_nonzero {
            writer.write_bit(info.log2_tile_rows > 1);
        }
        writer
    }

    fn write_segmentation(
        &mut self,
        writer: &mut VpxBitStreamWriter,
        segmentation: &Vp9Segmentation,
        seg_probs: &Vp9SegmentProbs,
    ) {
        let enabled = segmentation.enabled != 0;
        writer.write_bit(enabled);
        if !enabled {
            return;
        }
        let update_map = segmentation.update_map != 0;
        writer.write_bit(update_map);
        if update_map {
            let write_prob = |writer: &mut VpxBitStreamWriter, prob: u8| {
                let coded = prob != 255;
                writer.write_bit(coded);
                if coded {
                    writer.write_u(u32::from(prob), 8);
                }
            };
            for prob in seg_probs.tree_probs {
                write_prob(writer, prob);
            }
            let temporal_update = segmentation.temporal_update != 0;
            writer.write_bit(temporal_update);
            if temporal_update {
                for prob in seg_probs.pred_probs {
                    write_prob(writer, prob);
                }
            }
        }
        if self.last_segmentation == *segmentation {
            writer.write_bit(false);
            return;
        }
        self.last_segmentation = segmentation.clone();
        writer.write_bit(true);
        writer.write_bit(segmentation.abs_delta != 0);
        const FEATURE_BITS: [u32; 4] = [8, 6, 2, 0];
        for i in 0..8 {
            let q_enabled = segmentation.feature_enabled[i][0] != 0;
            writer.write_bit(q_enabled);
            if q_enabled {
                writer.write_s(i32::from(segmentation.feature_data[i][0]), FEATURE_BITS[0]);
            }
            let lf_enabled = segmentation.feature_enabled[i][1] != 0;
            writer.write_bit(lf_enabled);
            if lf_enabled {
                writer.write_s(i32::from(segmentation.feature_data[i][1]), FEATURE_BITS[1]);
            }
            let ref_enabled = segmentation.feature_enabled[i][2] != 0;
            writer.write_bit(ref_enabled);
            if ref_enabled {
                writer.write_u(segmentation.feature_data[i][2] as u32, FEATURE_BITS[2]);
            }
            let skip_enabled = segmentation.feature_enabled[i][3] != 0;
            writer.write_bit(skip_enabled);
        }
    }

    fn compose_compressed_header(&mut self, info: &Vp9PictureInfo) -> Vec<u8> {
        let mut writer = VpxRangeEncoder::new();
        let update_probs = !info.is_key_frame && info.show_frame;
        if !info.lossless {
            if info.transform_mode >= 3 {
                writer.write_bits(3, 2);
                writer.write_bool(info.transform_mode == 4);
            } else {
                writer.write_bits(info.transform_mode, 2);
            }
        }
        if info.transform_mode == 4 {
            write_prob_update_array(
                &mut writer,
                &info.entropy.tx_8x8_prob,
                &self.prev_frame_probs.tx_8x8_prob,
            );
            write_prob_update_array(
                &mut writer,
                &info.entropy.tx_16x16_prob,
                &self.prev_frame_probs.tx_16x16_prob,
            );
            write_prob_update_array(
                &mut writer,
                &info.entropy.tx_32x32_prob,
                &self.prev_frame_probs.tx_32x32_prob,
            );
            if update_probs {
                self.prev_frame_probs.tx_8x8_prob = info.entropy.tx_8x8_prob;
                self.prev_frame_probs.tx_16x16_prob = info.entropy.tx_16x16_prob;
                self.prev_frame_probs.tx_32x32_prob = info.entropy.tx_32x32_prob;
            }
        }
        write_coef_prob_update(
            &mut writer,
            info.transform_mode,
            &info.entropy.coef_probs,
            &self.prev_frame_probs.coef_probs,
        );
        write_prob_update_array(
            &mut writer,
            &info.entropy.skip_probs,
            &self.prev_frame_probs.skip_probs,
        );
        if update_probs {
            self.prev_frame_probs.coef_probs = info.entropy.coef_probs;
            self.prev_frame_probs.skip_probs = info.entropy.skip_probs;
        }

        if !info.intra_only {
            write_prob_update_aligned4(
                &mut writer,
                &info.entropy.inter_mode_prob,
                &self.prev_frame_probs.inter_mode_prob,
            );
            if info.interp_filter == 4 {
                write_prob_update_array(
                    &mut writer,
                    &info.entropy.switchable_interp_prob,
                    &self.prev_frame_probs.switchable_interp_prob,
                );
                if update_probs {
                    self.prev_frame_probs.switchable_interp_prob =
                        info.entropy.switchable_interp_prob;
                }
            }
            write_prob_update_array(
                &mut writer,
                &info.entropy.intra_inter_prob,
                &self.prev_frame_probs.intra_inter_prob,
            );
            if (info.ref_frame_sign_bias[1] & 1) != (info.ref_frame_sign_bias[2] & 1)
                || (info.ref_frame_sign_bias[1] & 1) != (info.ref_frame_sign_bias[3] & 1)
            {
                if info.reference_mode >= 1 {
                    writer.write_bits(1, 1);
                    writer.write_bool(info.reference_mode == 2);
                } else {
                    writer.write_bits(0, 1);
                }
            }
            if info.reference_mode == 2 {
                write_prob_update_array(
                    &mut writer,
                    &info.entropy.comp_inter_prob,
                    &self.prev_frame_probs.comp_inter_prob,
                );
                if update_probs {
                    self.prev_frame_probs.comp_inter_prob = info.entropy.comp_inter_prob;
                }
            }
            if info.reference_mode != 1 {
                write_prob_update_array(
                    &mut writer,
                    &info.entropy.single_ref_prob,
                    &self.prev_frame_probs.single_ref_prob,
                );
                if update_probs {
                    self.prev_frame_probs.single_ref_prob = info.entropy.single_ref_prob;
                }
            }
            if info.reference_mode != 0 {
                write_prob_update_array(
                    &mut writer,
                    &info.entropy.comp_ref_prob,
                    &self.prev_frame_probs.comp_ref_prob,
                );
                if update_probs {
                    self.prev_frame_probs.comp_ref_prob = info.entropy.comp_ref_prob;
                }
            }
            write_prob_update_array(
                &mut writer,
                &info.entropy.y_mode_prob,
                &self.prev_frame_probs.y_mode_prob,
            );
            write_prob_update_aligned4(
                &mut writer,
                &info.entropy.partition_prob,
                &self.prev_frame_probs.partition_prob,
            );
            for i in 0..3 {
                write_mv_prob_update(
                    &mut writer,
                    info.entropy.joints[i],
                    self.prev_frame_probs.joints[i],
                );
            }
            if update_probs {
                self.prev_frame_probs.inter_mode_prob = info.entropy.inter_mode_prob;
                self.prev_frame_probs.intra_inter_prob = info.entropy.intra_inter_prob;
                self.prev_frame_probs.y_mode_prob = info.entropy.y_mode_prob;
                self.prev_frame_probs.partition_prob = info.entropy.partition_prob;
                self.prev_frame_probs.joints = info.entropy.joints;
            }
            for i in 0..2 {
                write_mv_prob_update(
                    &mut writer,
                    info.entropy.sign[i],
                    self.prev_frame_probs.sign[i],
                );
                for j in 0..10 {
                    let index = i * 10 + j;
                    write_mv_prob_update(
                        &mut writer,
                        info.entropy.classes[index],
                        self.prev_frame_probs.classes[index],
                    );
                }
                write_mv_prob_update(
                    &mut writer,
                    info.entropy.class_0[i],
                    self.prev_frame_probs.class_0[i],
                );
                for j in 0..10 {
                    let index = i * 10 + j;
                    write_mv_prob_update(
                        &mut writer,
                        info.entropy.prob_bits[index],
                        self.prev_frame_probs.prob_bits[index],
                    );
                }
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..3 {
                        let index = i * 2 * 3 + j * 3 + k;
                        write_mv_prob_update(
                            &mut writer,
                            info.entropy.class_0_fr[index],
                            self.prev_frame_probs.class_0_fr[index],
                        );
                    }
                }
                for j in 0..3 {
                    let index = i * 3 + j;
                    write_mv_prob_update(
                        &mut writer,
                        info.entropy.fr[index],
                        self.prev_frame_probs.fr[index],
                    );
                }
            }
            if info.allow_high_precision_mv {
                for index in 0..2 {
                    write_mv_prob_update(
                        &mut writer,
                        info.entropy.class_0_hp[index],
                        self.prev_frame_probs.class_0_hp[index],
                    );
                    write_mv_prob_update(
                        &mut writer,
                        info.entropy.high_precision[index],
                        self.prev_frame_probs.high_precision[index],
                    );
                }
            }
            if update_probs {
                self.prev_frame_probs.sign = info.entropy.sign;
                self.prev_frame_probs.classes = info.entropy.classes;
                self.prev_frame_probs.class_0 = info.entropy.class_0;
                self.prev_frame_probs.prob_bits = info.entropy.prob_bits;
                self.prev_frame_probs.class_0_fr = info.entropy.class_0_fr;
                self.prev_frame_probs.fr = info.entropy.fr;
                self.prev_frame_probs.class_0_hp = info.entropy.class_0_hp;
                self.prev_frame_probs.high_precision = info.entropy.high_precision;
            }
        }
        writer.end();
        writer.into_buffer()
    }
}

fn write_prob_update(writer: &mut VpxRangeEncoder, new_prob: u8, old_prob: u8) {
    let update = new_prob != old_prob;
    writer.write_prob(update, DIFF_UPDATE_PROBABILITY);
    if update {
        let delta = remap_probability(i32::from(new_prob), i32::from(old_prob));
        encode_term_subexp(writer, delta);
    }
}

fn write_prob_update_array(writer: &mut VpxRangeEncoder, new_probs: &[u8], old_probs: &[u8]) {
    for (new_prob, old_prob) in new_probs.iter().zip(old_probs.iter()) {
        write_prob_update(writer, *new_prob, *old_prob);
    }
}

fn write_prob_update_aligned4(writer: &mut VpxRangeEncoder, new_probs: &[u8], old_probs: &[u8]) {
    for offset in (0..new_probs.len()).step_by(4) {
        write_prob_update(writer, new_probs[offset], old_probs[offset]);
        write_prob_update(writer, new_probs[offset + 1], old_probs[offset + 1]);
        write_prob_update(writer, new_probs[offset + 2], old_probs[offset + 2]);
    }
}

fn write_coef_prob_update(
    writer: &mut VpxRangeEncoder,
    tx_mode: i32,
    new_probs: &[u8; 1728],
    old_probs: &[u8; 1728],
) {
    const BLOCK_BYTES: usize = 2 * 2 * 6 * 6 * 3;
    for block_index in 0..4usize {
        let base_index = block_index * BLOCK_BYTES;
        let update = new_probs[base_index..base_index + BLOCK_BYTES]
            != old_probs[base_index..base_index + BLOCK_BYTES];
        writer.write_bool(update);
        if update {
            let mut index = base_index;
            for _ in 0..2 {
                for _ in 0..2 {
                    for k in 0..6 {
                        for l in 0..6 {
                            if k != 0 || l < 3 {
                                write_prob_update(writer, new_probs[index], old_probs[index]);
                                write_prob_update(
                                    writer,
                                    new_probs[index + 1],
                                    old_probs[index + 1],
                                );
                                write_prob_update(
                                    writer,
                                    new_probs[index + 2],
                                    old_probs[index + 2],
                                );
                            }
                            index += 3;
                        }
                    }
                }
            }
        }
        if block_index == tx_mode as usize {
            break;
        }
    }
}

fn write_mv_prob_update(writer: &mut VpxRangeEncoder, new_prob: u8, old_prob: u8) {
    let update = new_prob != old_prob;
    writer.write_prob(update, DIFF_UPDATE_PROBABILITY);
    if update {
        writer.write_bits(i32::from(new_prob >> 1), 7);
    }
}

fn encode_term_subexp(writer: &mut VpxRangeEncoder, mut value: i32) {
    if write_less_than(writer, value, 16) {
        writer.write_bits(value, 4);
    } else if write_less_than(writer, value, 32) {
        writer.write_bits(value - 16, 4);
    } else if write_less_than(writer, value, 64) {
        writer.write_bits(value - 32, 5);
    } else {
        value -= 64;
        const SIZE: i32 = 8;
        let mask = (1 << SIZE) - 191;
        let delta = value - mask;
        if delta < 0 {
            writer.write_bits(value, SIZE - 1);
        } else {
            writer.write_bits(delta / 2 + mask, SIZE - 1);
            writer.write_bits(delta & 1, 1);
        }
    }
}

fn write_less_than(writer: &mut VpxRangeEncoder, value: i32, test: i32) -> bool {
    let is_lt = value < test;
    writer.write_bool(!is_lt);
    is_lt
}

pub fn ivf_file_header(width: u16, height: u16) -> [u8; 32] {
    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(b"DKIF");
    header[4..6].copy_from_slice(&0u16.to_le_bytes());
    header[6..8].copy_from_slice(&32u16.to_le_bytes());
    header[8..12].copy_from_slice(b"VP90");
    header[12..14].copy_from_slice(&width.to_le_bytes());
    header[14..16].copy_from_slice(&height.to_le_bytes());
    header[16..20].copy_from_slice(&30u32.to_le_bytes());
    header[20..24].copy_from_slice(&1u32.to_le_bytes());
    header
}

pub fn ivf_frame_header(size: u32, pts: u64) -> [u8; 12] {
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(&size.to_le_bytes());
    header[4..12].copy_from_slice(&pts.to_le_bytes());
    header
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

pub const NORM_LUT: [u8; 256] = [
    0, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub const MAP_LUT: [u8; 254] = [
    20, 21, 22, 23, 24, 25, 0, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 1, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 2, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 3, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 4, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 5,
    86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 6, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 7, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 8, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 132, 133, 9, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    143, 144, 145, 10, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 11, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 12, 170, 171, 172, 173, 174, 175, 176, 177,
    178, 179, 180, 181, 13, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 14, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 15, 206, 207, 208, 209, 210, 211, 212,
    213, 214, 215, 216, 217, 16, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 17,
    230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 18, 242, 243, 244, 245, 246, 247,
    248, 249, 250, 251, 252, 253, 19,
];

pub const DEFAULT_PROBS: Vp9EntropyProbs = Vp9EntropyProbs {
    y_mode_prob: [
        65, 32, 18, 144, 162, 194, 41, 51, 98, 132, 68, 18, 165, 217, 196, 45, 40, 78, 173, 80, 19,
        176, 240, 193, 64, 35, 46, 221, 135, 38, 194, 248, 121, 96, 85, 29,
    ],
    partition_prob: [
        199, 122, 141, 0, 147, 63, 159, 0, 148, 133, 118, 0, 121, 104, 114, 0, 174, 73, 87, 0, 92,
        41, 83, 0, 82, 99, 50, 0, 53, 39, 39, 0, 177, 58, 59, 0, 68, 26, 63, 0, 52, 79, 25, 0, 17,
        14, 12, 0, 222, 34, 30, 0, 72, 16, 44, 0, 58, 32, 12, 0, 10, 7, 6, 0,
    ],
    coef_probs: [
        195, 29, 183, 84, 49, 136, 8, 42, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 107, 169, 35, 99, 159,
        17, 82, 140, 8, 66, 114, 2, 44, 76, 1, 19, 32, 40, 132, 201, 29, 114, 187, 13, 91, 157, 7,
        75, 127, 3, 58, 95, 1, 28, 47, 69, 142, 221, 42, 122, 201, 15, 91, 159, 6, 67, 121, 1, 42,
        77, 1, 17, 31, 102, 148, 228, 67, 117, 204, 17, 82, 154, 6, 59, 114, 2, 39, 75, 1, 15, 29,
        156, 57, 233, 119, 57, 212, 58, 48, 163, 29, 40, 124, 12, 30, 81, 3, 12, 31, 191, 107, 226,
        124, 117, 204, 25, 99, 155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 148, 210, 37, 126, 194, 8, 93,
        157, 2, 68, 118, 1, 39, 69, 1, 17, 33, 41, 151, 213, 27, 123, 193, 3, 82, 144, 1, 58, 105,
        1, 32, 60, 1, 13, 26, 59, 159, 220, 23, 126, 198, 4, 88, 151, 1, 66, 114, 1, 38, 71, 1, 18,
        34, 114, 136, 232, 51, 114, 207, 11, 83, 155, 3, 56, 105, 1, 33, 65, 1, 17, 34, 149, 65,
        234, 121, 57, 215, 61, 49, 166, 28, 36, 114, 12, 25, 76, 3, 16, 42, 214, 49, 220, 132, 63,
        188, 42, 65, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 137, 221, 104, 131, 216, 49, 111, 192, 21,
        87, 155, 2, 49, 87, 1, 16, 28, 89, 163, 230, 90, 137, 220, 29, 100, 183, 10, 70, 135, 2,
        42, 81, 1, 17, 33, 108, 167, 237, 55, 133, 222, 15, 97, 179, 4, 72, 135, 1, 45, 85, 1, 19,
        38, 124, 146, 240, 66, 124, 224, 17, 88, 175, 4, 58, 122, 1, 36, 75, 1, 18, 37, 141, 79,
        241, 126, 70, 227, 66, 58, 182, 30, 44, 136, 12, 34, 96, 2, 20, 47, 229, 99, 249, 143, 111,
        235, 46, 109, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 82, 158, 236, 94, 146, 224, 25, 117, 191, 9,
        87, 149, 3, 56, 99, 1, 33, 57, 83, 167, 237, 68, 145, 222, 10, 103, 177, 2, 72, 131, 1, 41,
        79, 1, 20, 39, 99, 167, 239, 47, 141, 224, 10, 104, 178, 2, 73, 133, 1, 44, 85, 1, 22, 47,
        127, 145, 243, 71, 129, 228, 17, 93, 177, 3, 61, 124, 1, 41, 84, 1, 21, 52, 157, 78, 244,
        140, 72, 231, 69, 58, 184, 31, 44, 137, 14, 38, 105, 8, 23, 61, 125, 34, 187, 52, 41, 133,
        6, 31, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 109, 153, 51, 102, 147, 23, 87, 128, 8, 67, 101,
        1, 41, 63, 1, 19, 29, 31, 154, 185, 17, 127, 175, 6, 96, 145, 2, 73, 114, 1, 51, 82, 1, 28,
        45, 23, 163, 200, 10, 131, 185, 2, 93, 148, 1, 67, 111, 1, 41, 69, 1, 14, 24, 29, 176, 217,
        12, 145, 201, 3, 101, 156, 1, 69, 111, 1, 39, 63, 1, 14, 23, 57, 192, 233, 25, 154, 215, 6,
        109, 167, 3, 78, 118, 1, 48, 69, 1, 21, 29, 202, 105, 245, 108, 106, 216, 18, 90, 144, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 33, 172, 219, 64, 149, 206, 14, 117, 177, 5, 90, 141, 2, 61, 95, 1,
        37, 57, 33, 179, 220, 11, 140, 198, 1, 89, 148, 1, 60, 104, 1, 33, 57, 1, 12, 21, 30, 181,
        221, 8, 141, 198, 1, 87, 145, 1, 58, 100, 1, 31, 55, 1, 12, 20, 32, 186, 224, 7, 142, 198,
        1, 86, 143, 1, 58, 100, 1, 31, 55, 1, 12, 22, 57, 192, 227, 20, 143, 204, 3, 96, 154, 1,
        68, 112, 1, 42, 69, 1, 19, 32, 212, 35, 215, 113, 47, 169, 29, 48, 105, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 74, 129, 203, 106, 120, 203, 49, 107, 178, 19, 84, 144, 4, 50, 84, 1, 15, 25, 71,
        172, 217, 44, 141, 209, 15, 102, 173, 6, 76, 133, 2, 51, 89, 1, 24, 42, 64, 185, 231, 31,
        148, 216, 8, 103, 175, 3, 74, 131, 1, 46, 81, 1, 18, 30, 65, 196, 235, 25, 157, 221, 5,
        105, 174, 1, 67, 120, 1, 38, 69, 1, 15, 30, 65, 204, 238, 30, 156, 224, 7, 107, 177, 2, 70,
        124, 1, 42, 73, 1, 18, 34, 225, 86, 251, 144, 104, 235, 42, 99, 181, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 85, 175, 239, 112, 165, 229, 29, 136, 200, 12, 103, 162, 6, 77, 123, 2, 53, 84, 75,
        183, 239, 30, 155, 221, 3, 106, 171, 1, 74, 128, 1, 44, 76, 1, 17, 28, 73, 185, 240, 27,
        159, 222, 2, 107, 172, 1, 75, 127, 1, 42, 73, 1, 17, 29, 62, 190, 238, 21, 159, 222, 2,
        107, 172, 1, 72, 122, 1, 40, 71, 1, 18, 32, 61, 199, 240, 27, 161, 226, 4, 113, 180, 1, 76,
        129, 1, 46, 80, 1, 23, 41, 7, 27, 153, 5, 30, 95, 1, 16, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50,
        75, 127, 57, 75, 124, 27, 67, 108, 10, 54, 86, 1, 33, 52, 1, 12, 18, 43, 125, 151, 26, 108,
        148, 7, 83, 122, 2, 59, 89, 1, 38, 60, 1, 17, 27, 23, 144, 163, 13, 112, 154, 2, 75, 117,
        1, 50, 81, 1, 31, 51, 1, 14, 23, 18, 162, 185, 6, 123, 171, 1, 78, 125, 1, 51, 86, 1, 31,
        54, 1, 14, 23, 15, 199, 227, 3, 150, 204, 1, 91, 146, 1, 55, 95, 1, 30, 53, 1, 11, 20, 19,
        55, 240, 19, 59, 196, 3, 52, 105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 166, 207, 104, 153, 199,
        31, 123, 181, 14, 101, 152, 5, 72, 106, 1, 36, 52, 35, 176, 211, 12, 131, 190, 2, 88, 144,
        1, 60, 101, 1, 36, 60, 1, 16, 28, 28, 183, 213, 8, 134, 191, 1, 86, 142, 1, 56, 96, 1, 30,
        53, 1, 12, 20, 20, 190, 215, 4, 135, 192, 1, 84, 139, 1, 53, 91, 1, 28, 49, 1, 11, 20, 13,
        196, 216, 2, 137, 192, 1, 86, 143, 1, 57, 99, 1, 32, 56, 1, 13, 24, 211, 29, 217, 96, 47,
        156, 22, 43, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 120, 193, 111, 116, 186, 46, 102, 164, 15,
        80, 128, 2, 49, 76, 1, 18, 28, 71, 161, 203, 42, 132, 192, 10, 98, 150, 3, 69, 109, 1, 44,
        70, 1, 18, 29, 57, 186, 211, 30, 140, 196, 4, 93, 146, 1, 62, 102, 1, 38, 65, 1, 16, 27,
        47, 199, 217, 14, 145, 196, 1, 88, 142, 1, 57, 98, 1, 36, 62, 1, 15, 26, 26, 219, 229, 5,
        155, 207, 1, 94, 151, 1, 60, 104, 1, 36, 62, 1, 16, 28, 233, 29, 248, 146, 47, 220, 43, 52,
        140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 163, 232, 179, 161, 222, 63, 142, 204, 37, 113, 174,
        26, 89, 137, 18, 68, 97, 85, 181, 230, 32, 146, 209, 7, 100, 164, 3, 71, 121, 1, 45, 77, 1,
        18, 30, 65, 187, 230, 20, 148, 207, 2, 97, 159, 1, 68, 116, 1, 40, 70, 1, 14, 29, 40, 194,
        227, 8, 147, 204, 1, 94, 155, 1, 65, 112, 1, 39, 66, 1, 14, 26, 16, 208, 228, 3, 151, 207,
        1, 98, 160, 1, 67, 117, 1, 41, 74, 1, 17, 31, 17, 38, 140, 7, 34, 80, 1, 17, 29, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 37, 75, 128, 41, 76, 128, 26, 66, 116, 12, 52, 94, 2, 32, 55, 1, 10, 16,
        50, 127, 154, 37, 109, 152, 16, 82, 121, 5, 59, 85, 1, 35, 54, 1, 13, 20, 40, 142, 167, 17,
        110, 157, 2, 71, 112, 1, 44, 72, 1, 27, 45, 1, 11, 17, 30, 175, 188, 9, 124, 169, 1, 74,
        116, 1, 48, 78, 1, 30, 49, 1, 11, 18, 10, 222, 223, 2, 150, 194, 1, 83, 128, 1, 48, 79, 1,
        27, 45, 1, 11, 17, 36, 41, 235, 29, 36, 193, 10, 27, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85,
        165, 222, 177, 162, 215, 110, 135, 195, 57, 113, 168, 23, 83, 120, 10, 49, 61, 85, 190,
        223, 36, 139, 200, 5, 90, 146, 1, 60, 103, 1, 38, 65, 1, 18, 30, 72, 202, 223, 23, 141,
        199, 2, 86, 140, 1, 56, 97, 1, 36, 61, 1, 16, 27, 55, 218, 225, 13, 145, 200, 1, 86, 141,
        1, 57, 99, 1, 35, 61, 1, 13, 22, 15, 235, 212, 1, 132, 184, 1, 84, 139, 1, 57, 97, 1, 34,
        56, 1, 14, 23, 181, 21, 201, 61, 37, 123, 10, 38, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 47, 106,
        172, 95, 104, 173, 42, 93, 159, 18, 77, 131, 4, 50, 81, 1, 17, 23, 62, 147, 199, 44, 130,
        189, 28, 102, 154, 18, 75, 115, 2, 44, 65, 1, 12, 19, 55, 153, 210, 24, 130, 194, 3, 93,
        146, 1, 61, 97, 1, 31, 50, 1, 10, 16, 49, 186, 223, 17, 148, 204, 1, 96, 142, 1, 53, 83, 1,
        26, 44, 1, 11, 17, 13, 217, 212, 2, 136, 180, 1, 78, 124, 1, 50, 83, 1, 29, 49, 1, 14, 23,
        197, 13, 247, 82, 17, 222, 25, 17, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 186, 247, 234, 191,
        243, 176, 177, 234, 104, 158, 220, 66, 128, 186, 55, 90, 137, 111, 197, 242, 46, 158, 219,
        9, 104, 171, 2, 65, 125, 1, 44, 80, 1, 17, 91, 104, 208, 245, 39, 168, 224, 3, 109, 162, 1,
        79, 124, 1, 50, 102, 1, 43, 102, 84, 220, 246, 31, 177, 231, 2, 115, 180, 1, 79, 134, 1,
        55, 77, 1, 60, 79, 43, 243, 240, 8, 180, 217, 1, 115, 166, 1, 84, 121, 1, 51, 67, 1, 16, 6,
    ],
    switchable_interp_prob: [235, 162, 36, 255, 34, 3, 149, 144],
    inter_mode_prob: [
        2, 173, 34, 0, 7, 145, 85, 0, 7, 166, 63, 0, 7, 94, 66, 0, 8, 64, 46, 0, 17, 81, 31, 0, 25,
        29, 30, 0,
    ],
    intra_inter_prob: [9, 102, 187, 225],
    comp_inter_prob: [9, 102, 187, 225, 0],
    single_ref_prob: [33, 16, 77, 74, 142, 142, 172, 170, 238, 247],
    comp_ref_prob: [50, 126, 123, 221, 226],
    tx_32x32_prob: [3, 136, 37, 5, 52, 13],
    tx_16x16_prob: [20, 152, 15, 101],
    tx_8x8_prob: [100, 66],
    skip_probs: [192, 128, 64],
    joints: [32, 64, 96],
    sign: [128, 128],
    classes: [
        224, 144, 192, 168, 192, 176, 192, 198, 198, 245, 216, 128, 176, 160, 176, 176, 192, 198,
        198, 208,
    ],
    class_0: [216, 208],
    prob_bits: [
        136, 140, 148, 160, 176, 192, 224, 234, 234, 240, 136, 140, 148, 160, 176, 192, 224, 234,
        234, 240,
    ],
    class_0_fr: [128, 128, 64, 96, 112, 64, 128, 128, 64, 96, 112, 64],
    fr: [64, 96, 64, 64, 96, 64],
    class_0_hp: [160, 160],
    high_precision: [128, 128],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn picture_info_bytes(width: i16, height: i16, flags: u32, last_shown: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; VP9_PICTURE_INFO_SIZE];
        bytes[0x30..0x34].copy_from_slice(&64u32.to_le_bytes());
        bytes[0x60..0x62].copy_from_slice(&width.to_le_bytes());
        bytes[0x62..0x64].copy_from_slice(&height.to_le_bytes());
        let mut all_flags = flags;
        if last_shown {
            all_flags |= FLAG_LAST_SHOW_FRAME;
        }
        bytes[0x68..0x6C].copy_from_slice(&all_flags.to_le_bytes());
        bytes
    }

    fn entropy_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; VP9_ENTROPY_PROBS_SIZE];
        for i in 0..2304usize {
            bytes[0x5A0 + i] = (i % 251) as u8;
        }
        for i in 0..4 {
            bytes[0x47C + i] = 200 + i as u8;
        }
        for i in 0..32 {
            bytes[0x480 + i] = i as u8;
        }
        bytes[0x387..0x38E].copy_from_slice(&[11, 12, 13, 14, 15, 16, 17]);
        bytes[0x38E..0x391].copy_from_slice(&[21, 22, 23]);
        bytes
    }

    #[test]
    fn picture_info_parses_nvdec_layout() {
        let mut bytes =
            picture_info_bytes(1600, 900, FLAG_IS_KEY_FRAME | FLAG_ERROR_RESILIENT, true);
        bytes[0x6C..0x70].copy_from_slice(&[0, 1, 0, 1]);
        bytes[0x70] = 32;
        bytes[0x71] = 5;
        bytes[0x72] = 100;
        bytes[0x73] = 1;
        bytes[0x74] = 2;
        bytes[0x75] = 3;
        bytes[0x76] = 0;
        bytes[0x77] = 4;
        bytes[0x78] = 1;
        bytes[0x79] = 4;
        bytes[0x7A] = 2;
        bytes[0x7E] = 1;
        bytes[0x7F] = 0;
        bytes[0xE4] = 1;
        bytes[0xE5] = 0xFF;
        bytes[0xE9] = 0xFE;
        let info = parse_picture_info(&bytes).unwrap();
        assert_eq!(info.bitstream_size, 64);
        assert_eq!((info.frame_width, info.frame_height), (1600, 900));
        assert!(info.is_key_frame);
        assert!(info.error_resilient_mode);
        assert!(info.last_frame_shown);
        assert!(info.show_frame);
        assert_eq!(info.ref_frame_sign_bias, [0, 1, 0, 1]);
        assert_eq!(info.first_level, 32);
        assert_eq!(info.sharpness_level, 5);
        assert_eq!(info.base_q_index, 100);
        assert_eq!(info.y_dc_delta_q, 1);
        assert_eq!(info.uv_ac_delta_q, 2);
        assert_eq!(info.uv_dc_delta_q, 3);
        assert_eq!(info.transform_mode, 4);
        assert!(info.allow_high_precision_mv);
        assert_eq!(info.interp_filter, 4);
        assert_eq!(info.reference_mode, 2);
        assert_eq!(info.log2_tile_cols, 1);
        assert!(info.mode_ref_delta_enabled);
        assert_eq!(info.ref_deltas[0], -1);
        assert_eq!(info.mode_deltas[0], -2);
    }

    #[test]
    fn entropy_probs_convert_matches_reference_mapping() {
        let bytes = entropy_bytes();
        let (probs, seg) = parse_entropy_probs(&bytes).unwrap();
        for i in (0..2304usize).step_by(4) {
            let j = i - i / 4;
            assert_eq!(probs.coef_probs[j], ((i) % 251) as u8);
            assert_eq!(probs.coef_probs[j + 1], ((i + 1) % 251) as u8);
            assert_eq!(probs.coef_probs[j + 2], ((i + 2) % 251) as u8);
        }
        for i in 0..4 {
            for j in 0..9 {
                let expected = if j < 8 {
                    (i * 8 + j) as u8
                } else {
                    200 + i as u8
                };
                assert_eq!(probs.y_mode_prob[j + 9 * i], expected);
            }
        }
        assert_eq!(seg.tree_probs, [11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(seg.pred_probs, [21, 22, 23]);
    }

    #[test]
    fn keyframe_uncompressed_header_bit_layout() {
        let bytes = picture_info_bytes(320, 180, FLAG_IS_KEY_FRAME, true);
        let mut info = parse_picture_info(&bytes).unwrap();
        info.entropy = DEFAULT_PROBS.clone();
        let seg = Vp9SegmentProbs {
            tree_probs: [255; 7],
            pred_probs: [255; 3],
        };
        let mut composer = Vp9FrameComposer::new();
        let (frame, show) = composer.compose(info, vec![0xAB; 64], &seg);
        assert!(show);
        assert_eq!(
            &frame[..12],
            &[0x82, 0x49, 0x83, 0x42, 0x00, 0x13, 0xF0, 0x0B, 0x36, 0x00, 0x00, 0x00]
        );
        let compressed_len = u16::from_be_bytes([frame[12], frame[13]]) as usize;
        assert!(compressed_len > 0);
        assert_eq!(frame.len(), 14 + compressed_len + 64);
        assert_eq!(&frame[frame.len() - 64..], &[0xAB; 64][..]);
    }

    #[test]
    fn composer_delays_frames_and_patches_show_flag() {
        let seg = Vp9SegmentProbs {
            tree_probs: [255; 7],
            pred_probs: [255; 3],
        };
        let mut composer = Vp9FrameComposer::new();

        let key_bytes = picture_info_bytes(320, 180, FLAG_IS_KEY_FRAME, true);
        let mut key_info = parse_picture_info(&key_bytes).unwrap();
        key_info.entropy = DEFAULT_PROBS.clone();
        let (first, first_show) = composer.compose(key_info, vec![0x11; 8], &seg);
        assert!(first_show);
        assert_eq!(&first[first.len() - 8..], &[0x11; 8][..]);

        let inter_bytes = picture_info_bytes(320, 180, 0, false);
        let mut inter_info = parse_picture_info(&inter_bytes).unwrap();
        inter_info.entropy = DEFAULT_PROBS.clone();
        let (second, second_show) = composer.compose(inter_info, vec![0x22; 8], &seg);
        assert!(!second_show);
        assert_eq!(&second[second.len() - 8..], &[0x11; 8][..]);

        let inter2_bytes = picture_info_bytes(320, 180, 0, true);
        let mut inter2_info = parse_picture_info(&inter2_bytes).unwrap();
        inter2_info.entropy = DEFAULT_PROBS.clone();
        let (third, third_show) = composer.compose(inter2_info, vec![0x33; 8], &seg);
        assert!(third_show);
        assert_eq!(&third[third.len() - 8..], &[0x22; 8][..]);
    }

    #[test]
    fn ivf_headers_have_reference_layout() {
        let file = ivf_file_header(1600, 900);
        assert_eq!(&file[0..4], b"DKIF");
        assert_eq!(u16::from_le_bytes([file[4], file[5]]), 0);
        assert_eq!(u16::from_le_bytes([file[6], file[7]]), 32);
        assert_eq!(&file[8..12], b"VP90");
        assert_eq!(u16::from_le_bytes([file[12], file[13]]), 1600);
        assert_eq!(u16::from_le_bytes([file[14], file[15]]), 900);
        assert_eq!(u32::from_le_bytes(file[16..20].try_into().unwrap()), 30);
        assert_eq!(u32::from_le_bytes(file[20..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(file[24..28].try_into().unwrap()), 0);

        let frame = ivf_frame_header(0x1234, 7);
        assert_eq!(u32::from_le_bytes(frame[0..4].try_into().unwrap()), 0x1234);
        assert_eq!(u64::from_le_bytes(frame[4..12].try_into().unwrap()), 7);
    }

    struct SpecBoolDecoder {
        bytes: Vec<u8>,
        bit_pos: usize,
        value: u32,
        range: u32,
    }

    impl SpecBoolDecoder {
        fn new(bytes: Vec<u8>) -> Self {
            let mut decoder = Self {
                bytes,
                bit_pos: 0,
                value: 0,
                range: 255,
            };
            for _ in 0..8 {
                decoder.value = (decoder.value << 1) | u32::from(decoder.next_bit());
            }
            decoder
        }

        fn next_bit(&mut self) -> u8 {
            let byte_index = self.bit_pos / 8;
            let bit = if byte_index < self.bytes.len() {
                (self.bytes[byte_index] >> (7 - self.bit_pos % 8)) & 1
            } else {
                0
            };
            self.bit_pos += 1;
            bit
        }

        fn read_bool(&mut self, probability: u32) -> bool {
            let split = 1 + (((self.range - 1) * probability) >> 8);
            let bit = if self.value < split {
                self.range = split;
                false
            } else {
                self.value -= split;
                self.range -= split;
                true
            };
            while self.range < 128 {
                self.value = (self.value << 1) | u32::from(self.next_bit());
                self.range <<= 1;
            }
            bit
        }
    }

    #[test]
    fn range_encoder_roundtrips_against_spec_bool_decoder() {
        let mut state = 0x12345678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut inputs = Vec::new();
        for _ in 0..20000 {
            let bit = next() & 1 != 0;
            let probability = 1 + next() % 255;
            inputs.push((bit, probability));
        }
        let mut encoder = VpxRangeEncoder::new();
        for (bit, probability) in &inputs {
            encoder.write_prob(*bit, *probability as i32);
        }
        encoder.end();
        let buffer = encoder.into_buffer();
        let mut decoder = SpecBoolDecoder::new(buffer);
        assert!(!decoder.read_bool(128));
        for (index, (bit, probability)) in inputs.iter().enumerate() {
            assert_eq!(decoder.read_bool(*probability), *bit, "bit {}", index);
        }
    }

    fn inv_recenter_non_neg(recentered: i32, old: i32) -> i32 {
        if recentered > old * 2 {
            recentered
        } else if recentered % 2 == 0 {
            old + recentered / 2
        } else {
            old - (recentered + 1) / 2
        }
    }

    #[test]
    fn probability_remap_is_invertible_for_all_pairs() {
        let mut inverse_map = [0usize; 256];
        for (index, value) in MAP_LUT.iter().enumerate() {
            inverse_map[*value as usize] = index;
        }
        for old in 1..=255i32 {
            for new in 1..=255i32 {
                if new == old {
                    continue;
                }
                let delta = remap_probability(new, old);
                let recentered = inverse_map[delta as usize] as i32 + 1;
                let reconstructed = if (old - 1) * 2 <= 0xff {
                    1 + inv_recenter_non_neg(recentered, old - 1)
                } else {
                    255 - inv_recenter_non_neg(recentered, 255 - old)
                };
                assert_eq!(reconstructed, new, "new={} old={}", new, old);
            }
        }
    }

    #[test]
    fn range_encoder_matches_reference_carry_behavior() {
        let mut encoder = VpxRangeEncoder::new();
        encoder.write_bits(0x5A, 8);
        encoder.write_prob(true, 200);
        encoder.write_prob(false, 10);
        encoder.write_prob(true, 252);
        encoder.end();
        let first = encoder.into_buffer();
        assert!(!first.is_empty());

        let mut repeat = VpxRangeEncoder::new();
        repeat.write_bits(0x5A, 8);
        repeat.write_prob(true, 200);
        repeat.write_prob(false, 10);
        repeat.write_prob(true, 252);
        repeat.end();
        assert_eq!(first, repeat.into_buffer());
    }
}

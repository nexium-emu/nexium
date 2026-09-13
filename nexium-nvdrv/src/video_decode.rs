use std::fmt;

pub const H264_DECODER_CONTEXT_SIZE: usize = 0x2fc;

const STREAM_LEN_OFFSET: usize = 0x48;
const PARAMETER_SET_OFFSET: usize = 0x58;
const PARAMETER_FLAGS_OFFSET: usize = PARAMETER_SET_OFFSET + 0x58;
const WEIGHT_SCALE_4X4_OFFSET: usize = 0x1c0;
const WEIGHT_SCALE_8X8_OFFSET: usize = 0x220;
const DECODER_FLAGS_OFFSET: usize = 0x2d0;

const SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const SCAN_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VideoDecodeError {
    ContextTooShort {
        actual: usize,
    },
    BitstreamTooShort {
        declared: usize,
        actual: usize,
    },
    InvalidContextField {
        field: &'static str,
        value: i64,
    },
    InvalidI420Dimensions {
        width: usize,
        height: usize,
    },
    InvalidPlaneStride {
        plane: &'static str,
        stride: usize,
        width: usize,
    },
    PlaneTooShort {
        plane: &'static str,
        required: usize,
        actual: usize,
    },
    SizeOverflow,
}

impl fmt::Display for VideoDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextTooShort { actual } => write!(
                f,
                "H.264 decoder context is {actual:#x} bytes, expected at least {H264_DECODER_CONTEXT_SIZE:#x}"
            ),
            Self::BitstreamTooShort { declared, actual } => write!(
                f,
                "H.264 context declares {declared} stream bytes, but only {actual} are available"
            ),
            Self::InvalidContextField { field, value } => {
                write!(f, "invalid H.264 context field {field}={value}")
            }
            Self::InvalidI420Dimensions { width, height } => {
                write!(f, "invalid I420 dimensions {width}x{height}")
            }
            Self::InvalidPlaneStride {
                plane,
                stride,
                width,
            } => write!(
                f,
                "I420 {plane} stride {stride} is smaller than its row width {width}"
            ),
            Self::PlaneTooShort {
                plane,
                required,
                actual,
            } => write!(
                f,
                "I420 {plane} plane has {actual} bytes, expected at least {required}"
            ),
            Self::SizeOverflow => f.write_str("video buffer size overflow"),
        }
    }
}

impl std::error::Error for VideoDecodeError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264ParameterSet {
    pub log2_max_pic_order_cnt_lsb_minus4: i32,
    pub delta_pic_order_always_zero: bool,
    pub frame_mbs_only: bool,
    pub pic_width_in_mbs: u32,
    pub frame_height_in_mbs: u32,
    pub entropy_coding_mode: bool,
    pub pic_order_present: bool,
    pub num_refidx_l0_default_active: i32,
    pub num_refidx_l1_default_active: i32,
    pub deblocking_filter_control_present: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub pitch_luma: u32,
    pub pitch_chroma: u32,
    pub luma_top_offset: u32,
    pub luma_bottom_offset: u32,
    pub luma_frame_offset: u32,
    pub chroma_top_offset: u32,
    pub chroma_bottom_offset: u32,
    pub chroma_frame_offset: u32,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    pub weighted_pred: bool,
    pub constrained_intra_pred: bool,
    pub field_picture: bool,
    pub bottom_field: bool,
    pub log2_max_frame_num_minus4: u32,
    pub chroma_format_idc: u32,
    pub pic_order_cnt_type: u32,
    pub pic_init_qp_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub second_chroma_qp_index_offset: i32,
    pub weighted_bipred_idc: u32,
    pub current_picture_index: u32,
    pub frame_number: u32,
    pub output_memory_layout: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H264DecoderContext {
    pub stream_len: u32,
    pub field_order_count: [i32; 2],
    pub parameter_set: H264ParameterSet,
    pub weight_scale_4x4: [u8; 0x60],
    pub weight_scale_8x8: [u8; 0x80],
    pub qpprime_y_zero_transform_bypass: bool,
}

impl H264DecoderContext {
    pub fn parse(bytes: &[u8]) -> Result<Self, VideoDecodeError> {
        if bytes.len() < H264_DECODER_CONTEXT_SIZE {
            return Err(VideoDecodeError::ContextTooShort {
                actual: bytes.len(),
            });
        }

        let flags = read_u64(bytes, PARAMETER_FLAGS_OFFSET);
        let parameter_set = H264ParameterSet {
            log2_max_pic_order_cnt_lsb_minus4: read_i32(bytes, PARAMETER_SET_OFFSET),
            delta_pic_order_always_zero: read_i32(bytes, PARAMETER_SET_OFFSET + 0x04) != 0,
            frame_mbs_only: read_i32(bytes, PARAMETER_SET_OFFSET + 0x08) != 0,
            pic_width_in_mbs: read_u32(bytes, PARAMETER_SET_OFFSET + 0x0c),
            frame_height_in_mbs: read_u32(bytes, PARAMETER_SET_OFFSET + 0x10),
            entropy_coding_mode: read_u32(bytes, PARAMETER_SET_OFFSET + 0x18) != 0,
            pic_order_present: read_i32(bytes, PARAMETER_SET_OFFSET + 0x1c) != 0,
            num_refidx_l0_default_active: read_i32(bytes, PARAMETER_SET_OFFSET + 0x20),
            num_refidx_l1_default_active: read_i32(bytes, PARAMETER_SET_OFFSET + 0x24),
            deblocking_filter_control_present: read_i32(bytes, PARAMETER_SET_OFFSET + 0x28) != 0,
            redundant_pic_cnt_present: read_i32(bytes, PARAMETER_SET_OFFSET + 0x2c) != 0,
            transform_8x8_mode: read_u32(bytes, PARAMETER_SET_OFFSET + 0x30) != 0,
            pitch_luma: read_u32(bytes, PARAMETER_SET_OFFSET + 0x34),
            pitch_chroma: read_u32(bytes, PARAMETER_SET_OFFSET + 0x38),
            luma_top_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x3c) << 8,
            luma_bottom_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x40) << 8,
            luma_frame_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x44) << 8,
            chroma_top_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x48) << 8,
            chroma_bottom_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x4c) << 8,
            chroma_frame_offset: read_u32(bytes, PARAMETER_SET_OFFSET + 0x50) << 8,
            mb_adaptive_frame_field: bit(flags, 0),
            direct_8x8_inference: bit(flags, 1),
            weighted_pred: bit(flags, 2),
            constrained_intra_pred: bit(flags, 3),
            field_picture: bit(flags, 5),
            bottom_field: bit(flags, 6),
            log2_max_frame_num_minus4: extract(flags, 8, 4) as u32,
            chroma_format_idc: extract(flags, 12, 2) as u32,
            pic_order_cnt_type: extract(flags, 14, 2) as u32,
            pic_init_qp_minus26: sign_extend(extract(flags, 16, 6), 6),
            chroma_qp_index_offset: sign_extend(extract(flags, 22, 5), 5),
            second_chroma_qp_index_offset: sign_extend(extract(flags, 27, 5), 5),
            weighted_bipred_idc: extract(flags, 32, 2) as u32,
            current_picture_index: extract(flags, 34, 7) as u32,
            frame_number: extract(flags, 46, 16) as u32,
            output_memory_layout: bit(flags, 63),
        };

        let mut weight_scale_4x4 = [0; 0x60];
        weight_scale_4x4
            .copy_from_slice(&bytes[WEIGHT_SCALE_4X4_OFFSET..WEIGHT_SCALE_4X4_OFFSET + 0x60]);
        let mut weight_scale_8x8 = [0; 0x80];
        weight_scale_8x8
            .copy_from_slice(&bytes[WEIGHT_SCALE_8X8_OFFSET..WEIGHT_SCALE_8X8_OFFSET + 0x80]);

        Ok(Self {
            stream_len: read_u32(bytes, STREAM_LEN_OFFSET),
            field_order_count: [read_i32(bytes, 0xb8), read_i32(bytes, 0xbc)],
            parameter_set,
            weight_scale_4x4,
            weight_scale_8x8,
            qpprime_y_zero_transform_bypass: bit(read_u32(bytes, DECODER_FLAGS_OFFSET) as u64, 1),
        })
    }

    pub fn frame_width(&self) -> Result<u32, VideoDecodeError> {
        self.parameter_set
            .pic_width_in_mbs
            .checked_mul(16)
            .ok_or(VideoDecodeError::SizeOverflow)
    }

    pub fn frame_height(&self) -> Result<u32, VideoDecodeError> {
        let divisor = if self.parameter_set.frame_mbs_only {
            1
        } else {
            2
        };
        self.parameter_set
            .frame_height_in_mbs
            .checked_div(divisor)
            .and_then(|height| height.checked_mul(16))
            .ok_or(VideoDecodeError::SizeOverflow)
    }
}

#[derive(Debug, Default)]
pub struct H264AnnexBComposer {
    has_composed_frame: bool,
}

impl H264AnnexBComposer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.has_composed_frame = false;
    }

    pub fn compose(
        &mut self,
        context_bytes: &[u8],
        bitstream_bytes: &[u8],
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let context = H264DecoderContext::parse(context_bytes)?;
        let include_parameter_sets =
            !self.has_composed_frame || context.parameter_set.frame_number == 0;
        let output =
            compose_parsed_h264_annex_b(&context, bitstream_bytes, include_parameter_sets)?;
        self.has_composed_frame = true;
        Ok(output)
    }
}

pub fn compose_h264_annex_b(
    context_bytes: &[u8],
    bitstream_bytes: &[u8],
    include_parameter_sets: bool,
) -> Result<Vec<u8>, VideoDecodeError> {
    let context = H264DecoderContext::parse(context_bytes)?;
    compose_parsed_h264_annex_b(&context, bitstream_bytes, include_parameter_sets)
}

fn compose_parsed_h264_annex_b(
    context: &H264DecoderContext,
    bitstream_bytes: &[u8],
    include_parameter_sets: bool,
) -> Result<Vec<u8>, VideoDecodeError> {
    let stream_len = context.stream_len as usize;
    let payload = bitstream_bytes
        .get(..stream_len)
        .ok_or(VideoDecodeError::BitstreamTooShort {
            declared: stream_len,
            actual: bitstream_bytes.len(),
        })?;
    if !include_parameter_sets {
        return Ok(payload.to_vec());
    }

    let mut output = synthesize_parameter_sets(context)?;
    output.reserve(payload.len());
    output.extend_from_slice(payload);
    Ok(output)
}

fn synthesize_parameter_sets(context: &H264DecoderContext) -> Result<Vec<u8>, VideoDecodeError> {
    let parameters = &context.parameter_set;
    validate_parameter_set(parameters)?;

    let mut output = Vec::with_capacity(256);
    output.extend_from_slice(&[0, 0, 1, 0x67]);
    let mut sps = BitWriter::default();
    sps.write_bits(100, 8);
    sps.write_bits(0, 8);
    sps.write_bits(31, 8);
    sps.write_ue(0);
    sps.write_ue(parameters.chroma_format_idc);
    if parameters.chroma_format_idc == 3 {
        sps.write_bit(false);
    }
    sps.write_ue(0);
    sps.write_ue(0);
    sps.write_bit(context.qpprime_y_zero_transform_bypass);
    sps.write_bit(false);
    sps.write_ue(parameters.log2_max_frame_num_minus4);
    sps.write_ue(parameters.pic_order_cnt_type);
    if parameters.pic_order_cnt_type == 0 {
        sps.write_ue(parameters.log2_max_pic_order_cnt_lsb_minus4 as u32);
    } else if parameters.pic_order_cnt_type == 1 {
        sps.write_bit(parameters.delta_pic_order_always_zero);
        sps.write_se(0);
        sps.write_se(0);
        sps.write_ue(0);
    }
    let max_num_ref_frames = parameters
        .num_refidx_l0_default_active
        .max(parameters.num_refidx_l1_default_active)
        .checked_add(1)
        .ok_or(VideoDecodeError::SizeOverflow)?
        .max(4) as u32;
    let picture_height =
        parameters.frame_height_in_mbs / if parameters.frame_mbs_only { 1 } else { 2 };
    sps.write_ue(max_num_ref_frames);
    sps.write_bit(false);
    sps.write_ue(parameters.pic_width_in_mbs - 1);
    sps.write_ue(picture_height - 1);
    sps.write_bit(parameters.frame_mbs_only);
    if !parameters.frame_mbs_only {
        sps.write_bit(parameters.mb_adaptive_frame_field);
    }
    sps.write_bit(parameters.direct_8x8_inference);
    sps.write_bit(false);
    sps.write_bit(false);
    output.extend_from_slice(&sps.finish_rbsp());

    output.extend_from_slice(&[0, 0, 1, 0x68]);
    let mut pps = BitWriter::default();
    pps.write_ue(0);
    pps.write_ue(0);
    pps.write_bit(parameters.entropy_coding_mode);
    pps.write_bit(parameters.pic_order_present);
    pps.write_ue(0);
    pps.write_ue(parameters.num_refidx_l0_default_active as u32);
    pps.write_ue(parameters.num_refidx_l1_default_active as u32);
    pps.write_bit(parameters.weighted_pred);
    pps.write_bits(parameters.weighted_bipred_idc as u64, 2);
    pps.write_se(parameters.pic_init_qp_minus26);
    pps.write_se(0);
    pps.write_se(parameters.chroma_qp_index_offset);
    pps.write_bit(parameters.deblocking_filter_control_present);
    pps.write_bit(parameters.constrained_intra_pred);
    pps.write_bit(parameters.redundant_pic_cnt_present);
    pps.write_bit(parameters.transform_8x8_mode);
    pps.write_bit(true);
    for list in context.weight_scale_4x4.chunks_exact(16) {
        pps.write_bit(true);
        write_scaling_list(&mut pps, list, &SCAN_4X4);
    }
    if parameters.transform_8x8_mode {
        for list in context.weight_scale_8x8.chunks_exact(64) {
            pps.write_bit(true);
            write_scaling_list(&mut pps, list, &SCAN_8X8);
        }
    }
    pps.write_se(parameters.second_chroma_qp_index_offset);
    output.extend_from_slice(&pps.finish_rbsp());
    Ok(output)
}

fn validate_parameter_set(parameters: &H264ParameterSet) -> Result<(), VideoDecodeError> {
    let checks = [
        (
            "log2_max_pic_order_cnt_lsb_minus4",
            parameters.log2_max_pic_order_cnt_lsb_minus4,
        ),
        (
            "num_refidx_l0_default_active",
            parameters.num_refidx_l0_default_active,
        ),
        (
            "num_refidx_l1_default_active",
            parameters.num_refidx_l1_default_active,
        ),
    ];
    for (field, value) in checks {
        if value < 0 {
            return Err(VideoDecodeError::InvalidContextField {
                field,
                value: value as i64,
            });
        }
    }
    for (field, value, maximum) in [
        ("chroma_format_idc", parameters.chroma_format_idc, 3),
        ("pic_order_cnt_type", parameters.pic_order_cnt_type, 2),
        ("weighted_bipred_idc", parameters.weighted_bipred_idc, 3),
    ] {
        if value > maximum {
            return Err(VideoDecodeError::InvalidContextField {
                field,
                value: value as i64,
            });
        }
    }
    if parameters.pic_width_in_mbs == 0 {
        return Err(VideoDecodeError::InvalidContextField {
            field: "pic_width_in_mbs",
            value: 0,
        });
    }
    let height_divisor = if parameters.frame_mbs_only { 1 } else { 2 };
    if parameters.frame_height_in_mbs / height_divisor == 0 {
        return Err(VideoDecodeError::InvalidContextField {
            field: "frame_height_in_mbs",
            value: parameters.frame_height_in_mbs as i64,
        });
    }
    Ok(())
}

fn write_scaling_list(writer: &mut BitWriter, list: &[u8], scan: &[usize]) {
    let mut last_scale = 8i32;
    for &index in scan {
        let value = list[index] as i32;
        writer.write_se(value - last_scale);
        last_scale = value;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedI420Frame {
    width: usize,
    height: usize,
    strides: (usize, usize, usize),
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl OwnedI420Frame {
    pub fn from_strided_planes(
        width: usize,
        height: usize,
        strides: (usize, usize, usize),
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<Self, VideoDecodeError> {
        if width == 0 || height == 0 || width & 1 != 0 || height & 1 != 0 {
            return Err(VideoDecodeError::InvalidI420Dimensions { width, height });
        }
        let chroma_width = width / 2;
        let chroma_height = height / 2;
        validate_stride("Y", strides.0, width)?;
        validate_stride("U", strides.1, chroma_width)?;
        validate_stride("V", strides.2, chroma_width)?;
        let y_len = plane_len(strides.0, height)?;
        let u_len = plane_len(strides.1, chroma_height)?;
        let v_len = plane_len(strides.2, chroma_height)?;
        validate_plane("Y", y, y_len)?;
        validate_plane("U", u, u_len)?;
        validate_plane("V", v, v_len)?;
        Ok(Self {
            width,
            height,
            strides,
            y: y[..y_len].to_vec(),
            u: u[..u_len].to_vec(),
            v: v[..v_len].to_vec(),
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn strides(&self) -> (usize, usize, usize) {
        self.strides
    }

    pub fn y(&self) -> &[u8] {
        &self.y
    }

    pub fn u(&self) -> &[u8] {
        &self.u
    }

    pub fn v(&self) -> &[u8] {
        &self.v
    }
}

fn validate_stride(
    plane: &'static str,
    stride: usize,
    width: usize,
) -> Result<(), VideoDecodeError> {
    if stride < width {
        return Err(VideoDecodeError::InvalidPlaneStride {
            plane,
            stride,
            width,
        });
    }
    Ok(())
}

fn plane_len(stride: usize, height: usize) -> Result<usize, VideoDecodeError> {
    stride
        .checked_mul(height)
        .ok_or(VideoDecodeError::SizeOverflow)
}

fn validate_plane(
    plane: &'static str,
    bytes: &[u8],
    required: usize,
) -> Result<(), VideoDecodeError> {
    if bytes.len() < required {
        return Err(VideoDecodeError::PlaneTooShort {
            plane,
            required,
            actual: bytes.len(),
        });
    }
    Ok(())
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    fn write_bit(&mut self, value: bool) {
        self.current |= u8::from(value) << (7 - self.used);
        self.used += 1;
        if self.used == 8 {
            self.flush();
        }
    }

    fn write_bits(&mut self, value: u64, count: u8) {
        debug_assert!(count <= 64);
        for shift in (0..count).rev() {
            self.write_bit(((value >> shift) & 1) != 0);
        }
    }

    fn write_ue(&mut self, value: u32) {
        let code_num = value as u64 + 1;
        let significant_bits = 64 - code_num.leading_zeros() as u8;
        for _ in 1..significant_bits {
            self.write_bit(false);
        }
        self.write_bits(code_num, significant_bits);
    }

    fn write_se(&mut self, value: i32) {
        let value = value as i64;
        let code_num = if value <= 0 {
            (-value * 2) as u32
        } else {
            (value * 2 - 1) as u32
        };
        self.write_ue(code_num);
    }

    fn finish_rbsp(mut self) -> Vec<u8> {
        self.write_bit(true);
        if self.used != 0 {
            self.flush();
        }
        self.bytes
    }

    fn flush(&mut self) {
        self.bytes.push(self.current);
        self.current = 0;
        self.used = 0;
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn bit(value: u64, index: u32) -> bool {
    value & (1u64 << index) != 0
}

fn extract(value: u64, offset: u32, width: u32) -> u64 {
    (value >> offset) & ((1u64 << width) - 1)
}

fn sign_extend(value: u64, width: u32) -> i32 {
    let shift = 64 - width;
    ((value << shift) as i64 >> shift) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_i32(bytes: &mut [u8], offset: usize, value: i32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn context(frame_number: u32, stream_len: usize) -> Vec<u8> {
        let mut bytes = vec![0; H264_DECODER_CONTEXT_SIZE];
        put_u32(&mut bytes, STREAM_LEN_OFFSET, stream_len as u32);
        put_i32(&mut bytes, PARAMETER_SET_OFFSET, 2);
        put_i32(&mut bytes, PARAMETER_SET_OFFSET + 0x08, 1);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x0c, 80);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x10, 45);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x18, 1);
        put_i32(&mut bytes, PARAMETER_SET_OFFSET + 0x28, 1);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x30, 1);
        let flags = (1u64 << 1)
            | (1u64 << 12)
            | (2u64 << 16)
            | (1u64 << 22)
            | (31u64 << 34)
            | ((frame_number as u64 & 0xffff) << 46)
            | (1u64 << 63);
        put_u64(&mut bytes, PARAMETER_FLAGS_OFFSET, flags);
        bytes[WEIGHT_SCALE_4X4_OFFSET..WEIGHT_SCALE_4X4_OFFSET + 0x60].fill(16);
        bytes[WEIGHT_SCALE_8X8_OFFSET..WEIGHT_SCALE_8X8_OFFSET + 0x80].fill(16);
        put_u32(&mut bytes, DECODER_FLAGS_OFFSET, 2);
        bytes
    }

    #[test]
    fn bit_writer_encodes_unsigned_and_signed_exp_golomb() {
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_ue(2);
        writer.write_se(0);
        writer.write_se(1);
        writer.write_se(-1);
        assert_eq!(writer.finish_rbsp(), [0xa7, 0x4e]);
    }

    #[test]
    fn parser_uses_nvdec_context_offsets_and_signed_flags() {
        let mut bytes = context(0x1234, 0x5678);
        put_i32(&mut bytes, PARAMETER_SET_OFFSET, 7);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x34, 0x1020_3040);
        put_u32(&mut bytes, PARAMETER_SET_OFFSET + 0x44, 0x1234);
        let mut flags = read_u64(&bytes, PARAMETER_FLAGS_OFFSET);
        flags &= !((0x3fu64 << 16) | (0x1fu64 << 22) | (0x1fu64 << 27));
        flags |= (0x3bu64 << 16) | (0x1du64 << 22) | (0x1cu64 << 27);
        put_u64(&mut bytes, PARAMETER_FLAGS_OFFSET, flags);

        put_i32(&mut bytes, 0xb8, -2);
        put_i32(&mut bytes, 0xbc, 3);
        let parsed = H264DecoderContext::parse(&bytes).unwrap();
        assert_eq!(parsed.field_order_count, [-2, 3]);
        assert_eq!(parsed.stream_len, 0x5678);
        assert_eq!(parsed.parameter_set.log2_max_pic_order_cnt_lsb_minus4, 7);
        assert_eq!(parsed.parameter_set.pitch_luma, 0x1020_3040);
        assert_eq!(parsed.parameter_set.luma_frame_offset, 0x123400);
        assert_eq!(parsed.parameter_set.pic_init_qp_minus26, -5);
        assert_eq!(parsed.parameter_set.chroma_qp_index_offset, -3);
        assert_eq!(parsed.parameter_set.second_chroma_qp_index_offset, -4);
        assert_eq!(parsed.parameter_set.current_picture_index, 31);
        assert_eq!(parsed.parameter_set.frame_number, 0x1234);
        assert!(parsed.parameter_set.output_memory_layout);
        assert!(parsed.qpprime_y_zero_transform_bypass);
        assert_eq!(parsed.frame_width().unwrap(), 1280);
        assert_eq!(parsed.frame_height().unwrap(), 720);
    }

    #[test]
    fn first_frame_contains_sps_pps_and_original_slice() {
        let payload = [0, 0, 1, 0x65, 0x88, 0x84];
        let output = compose_h264_annex_b(&context(1, payload.len()), &payload, true).unwrap();
        let nal_types: Vec<u8> = output
            .windows(4)
            .filter(|window| window[..3] == [0, 0, 1])
            .map(|window| window[3] & 0x1f)
            .collect();
        assert_eq!(nal_types, [7, 8, 5]);
        assert_eq!(&output[output.len() - payload.len()..], payload);
    }

    #[test]
    fn composer_only_repeats_headers_for_frame_zero() {
        let payload = [0, 0, 1, 0x41, 0x9a];
        let mut composer = H264AnnexBComposer::new();
        let first = composer
            .compose(&context(1, payload.len()), &payload)
            .unwrap();
        let second = composer
            .compose(&context(2, payload.len()), &payload)
            .unwrap();
        let reset_frame = composer
            .compose(&context(0, payload.len()), &payload)
            .unwrap();
        assert!(first.len() > payload.len());
        assert_eq!(second, payload);
        assert!(reset_frame.len() > payload.len());
    }

    #[test]
    fn owned_i420_frame_copies_openh264_strided_planes() {
        let y = [1, 2, 3, 4, 90, 91, 5, 6, 7, 8, 92, 93, 0xff];
        let u = [9, 10, 94, 0xff];
        let v = [11, 12, 95, 96, 0xff];
        let frame = OwnedI420Frame::from_strided_planes(4, 2, (6, 3, 4), &y, &u, &v).unwrap();
        assert_eq!((frame.width(), frame.height()), (4, 2));
        assert_eq!(frame.strides(), (6, 3, 4));
        assert_eq!(frame.y(), &y[..12]);
        assert_eq!(frame.u(), &u[..3]);
        assert_eq!(frame.v(), &v[..4]);
    }
}

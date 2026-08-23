pub const CURRENT_REVISION: u32 = 15;
pub const MAX_ERRORS: usize = 10;

const fn make_magic(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

const REVISION_MAGIC_BASE: u32 = make_magic(b'R', b'E', b'V', b'0');

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportTags {
    AudioRendererProcessingTimeLimit70Percent,
    Splitter,
    AdpcmLoopContextBugFix,
    LongSizePreDelay,
    AudioUsbDeviceOutput,
    AudioRendererProcessingTimeLimit75Percent,
    VoicePlayedSampleCountResetAtLoopPoint,
    VoicePitchAndSrcSkipped,
    SplitterBugFix,
    FlushVoiceWaveBuffers,
    ElapsedFrameCount,
    AudioRendererProcessingTimeLimit80Percent,
    AudioRendererVariadicCommandBufferSize,
    PerformanceMetricsDataFormatVersion2,
    CommandProcessingTimeEstimatorVersion2,
    DecodingBehaviourFlag,
    BiquadFilterEffectStateClearBugFix,
    BiquadFilterFloatProcessing,
    VolumeMixParameterPrecisionQ23,
    MixInParameterDirtyOnlyUpdate,
    WaveBufferVer2,
    CommandProcessingTimeEstimatorVersion3,
    EffectInfoVer2,
    CommandProcessingTimeEstimatorVersion4,
    MultiTapBiquadFilterProcessing,
    CommandProcessingTimeEstimatorVersion5,
    DelayChannelMappingChange,
    ReverbChannelMappingChange,
    I3dl2ReverbChannelMappingChange,
    DeviceApiVersion2,
    SplitterBiquadFilterParameter,
    SplitterPrevVolumeReset,
    SplitterDestinationV2b,
    VoiceInParameterV2,
}

const FEATURES: &[(SupportTags, u32)] = &[
    (SupportTags::AudioRendererProcessingTimeLimit70Percent, 1),
    (SupportTags::Splitter, 2),
    (SupportTags::AdpcmLoopContextBugFix, 2),
    (SupportTags::LongSizePreDelay, 3),
    (SupportTags::AudioUsbDeviceOutput, 4),
    (SupportTags::AudioRendererProcessingTimeLimit75Percent, 4),
    (SupportTags::VoicePlayedSampleCountResetAtLoopPoint, 5),
    (SupportTags::VoicePitchAndSrcSkipped, 5),
    (SupportTags::SplitterBugFix, 5),
    (SupportTags::FlushVoiceWaveBuffers, 5),
    (SupportTags::ElapsedFrameCount, 5),
    (SupportTags::AudioRendererProcessingTimeLimit80Percent, 5),
    (SupportTags::AudioRendererVariadicCommandBufferSize, 5),
    (SupportTags::PerformanceMetricsDataFormatVersion2, 5),
    (SupportTags::CommandProcessingTimeEstimatorVersion2, 5),
    (SupportTags::DecodingBehaviourFlag, 5),
    (SupportTags::BiquadFilterEffectStateClearBugFix, 6),
    (SupportTags::BiquadFilterFloatProcessing, 7),
    (SupportTags::VolumeMixParameterPrecisionQ23, 7),
    (SupportTags::MixInParameterDirtyOnlyUpdate, 7),
    (SupportTags::WaveBufferVer2, 8),
    (SupportTags::CommandProcessingTimeEstimatorVersion3, 8),
    (SupportTags::EffectInfoVer2, 9),
    (SupportTags::CommandProcessingTimeEstimatorVersion4, 10),
    (SupportTags::MultiTapBiquadFilterProcessing, 10),
    (SupportTags::CommandProcessingTimeEstimatorVersion5, 11),
    (SupportTags::DelayChannelMappingChange, 11),
    (SupportTags::ReverbChannelMappingChange, 11),
    (SupportTags::I3dl2ReverbChannelMappingChange, 11),
    (SupportTags::DeviceApiVersion2, 11),
    (SupportTags::SplitterBiquadFilterParameter, 12),
    (SupportTags::SplitterPrevVolumeReset, 13),
    (SupportTags::SplitterDestinationV2b, 15),
    (SupportTags::VoiceInParameterV2, 15),
];

pub const fn get_revision_num(user_revision: u32) -> u32 {
    if user_revision >= 0x100 {
        user_revision.wrapping_sub(REVISION_MAGIC_BASE) >> 24
    } else {
        user_revision
    }
}

pub const fn encode_revision(revision_num: u32) -> u32 {
    make_magic(b'R', b'E', b'V', b'0'.wrapping_add(revision_num as u8))
}

pub const fn check_valid_revision(user_revision: u32) -> bool {
    get_revision_num(user_revision) <= CURRENT_REVISION
}

pub fn check_feature_supported(tag: SupportTags, user_revision: u32) -> bool {
    let mut revision = get_revision_num(user_revision);
    if revision > CURRENT_REVISION {
        revision = 1;
    }
    FEATURES
        .iter()
        .find(|(feature, _)| *feature == tag)
        .map(|(_, minimum)| *minimum <= revision)
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct ErrorInfo {
    pub error_code: u32,
    pub unk_04: u32,
    pub address: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct Flags {
    pub is_memory_force_mapping_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct InParameter {
    pub revision: u32,
    pub padding: u32,
    pub flags: Flags,
    pub padding2: [u8; 7],
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct OutStatus {
    pub errors: [ErrorInfo; MAX_ERRORS],
    pub error_count: u32,
    pub unk_a4: [u8; 0xC],
}

pub const ERROR_INFO_SIZE: usize = std::mem::size_of::<ErrorInfo>();
pub const IN_PARAMETER_SIZE: usize = std::mem::size_of::<InParameter>();
pub const OUT_STATUS_SIZE: usize = std::mem::size_of::<OutStatus>();

const _: () = assert!(ERROR_INFO_SIZE == 0x10);
const _: () = assert!(IN_PARAMETER_SIZE == 0x10);
const _: () = assert!(OUT_STATUS_SIZE == 0xB0);

impl Default for OutStatus {
    fn default() -> Self {
        Self {
            errors: [ErrorInfo::default(); MAX_ERRORS],
            error_count: 0,
            unk_a4: [0; 0xC],
        }
    }
}

impl InParameter {
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < IN_PARAMETER_SIZE {
            return None;
        }
        Some(Self {
            revision: u32::from_le_bytes(bytes[0x00..0x04].try_into().ok()?),
            padding: 0,
            flags: Flags {
                is_memory_force_mapping_enabled: bytes[0x08] & 1 != 0,
            },
            padding2: [0; 7],
        })
    }
}

impl OutStatus {
    pub fn write_to(&self, dst: &mut [u8]) -> bool {
        if dst.len() < OUT_STATUS_SIZE {
            return false;
        }
        for (index, error) in self.errors.iter().enumerate() {
            let base = index * ERROR_INFO_SIZE;
            dst[base..base + 4].copy_from_slice(&error.error_code.to_le_bytes());
            dst[base + 4..base + 8].copy_from_slice(&error.unk_04.to_le_bytes());
            dst[base + 8..base + 16].copy_from_slice(&error.address.to_le_bytes());
        }
        let count_off = MAX_ERRORS * ERROR_INFO_SIZE;
        dst[count_off..count_off + 4].copy_from_slice(&self.error_count.to_le_bytes());
        dst[count_off + 4..OUT_STATUS_SIZE].fill(0);
        true
    }
}

pub struct BehaviorInfo {
    process_revision: u32,
    user_revision: u32,
    error_count: u32,
    errors: [ErrorInfo; MAX_ERRORS],
    flags: Flags,
}

impl BehaviorInfo {
    pub fn new() -> Self {
        Self {
            process_revision: CURRENT_REVISION,
            user_revision: 0,
            error_count: 0,
            errors: [ErrorInfo::default(); MAX_ERRORS],
            flags: Flags::default(),
        }
    }

    pub fn from_user_revision(user_revision: u32) -> Self {
        let mut info = Self::new();
        info.set_user_lib_revision(user_revision);
        info
    }

    pub fn get_process_revision_num(&self) -> u32 {
        self.process_revision
    }

    pub fn get_process_revision(&self) -> u32 {
        encode_revision(self.process_revision)
    }

    pub fn get_user_revision_num(&self) -> u32 {
        self.user_revision
    }

    pub fn get_user_revision(&self) -> u32 {
        encode_revision(self.user_revision)
    }

    pub fn set_user_lib_revision(&mut self, user_revision: u32) {
        self.user_revision = get_revision_num(user_revision);
    }

    pub fn update_flags(&mut self, flags: Flags) {
        self.flags = flags;
    }

    pub fn clear_error(&mut self) {
        self.error_count = 0;
        self.errors = [ErrorInfo::default(); MAX_ERRORS];
    }

    pub fn append_error(&mut self, error: ErrorInfo) {
        log::error!(
            "audren RequestUpdate error code={:#010x} address={:#x}",
            error.error_code,
            error.address
        );
        if (self.error_count as usize) < MAX_ERRORS {
            self.errors[self.error_count as usize] = error;
            self.error_count += 1;
        }
    }

    pub fn error_count(&self) -> u32 {
        self.error_count
    }

    pub fn out_status(&self) -> OutStatus {
        let mut status = OutStatus::default();
        let count = self.error_count.min(MAX_ERRORS as u32);
        status.error_count = count;
        for (index, slot) in status.errors.iter_mut().enumerate() {
            *slot = if index < count as usize {
                self.errors[index]
            } else {
                ErrorInfo::default()
            };
        }
        status
    }

    fn supported(&self, tag: SupportTags) -> bool {
        check_feature_supported(tag, self.user_revision)
    }

    pub fn is_memory_force_mapping_enabled(&self) -> bool {
        self.flags.is_memory_force_mapping_enabled
    }

    pub fn is_adpcm_loop_context_bug_fixed(&self) -> bool {
        self.supported(SupportTags::AdpcmLoopContextBugFix)
    }

    pub fn is_splitter_supported(&self) -> bool {
        self.supported(SupportTags::Splitter)
    }

    pub fn is_splitter_bug_fixed(&self) -> bool {
        self.supported(SupportTags::SplitterBugFix)
    }

    pub fn is_long_size_pre_delay_supported(&self) -> bool {
        self.supported(SupportTags::LongSizePreDelay)
    }

    pub fn is_audio_usb_device_output_supported(&self) -> bool {
        self.supported(SupportTags::AudioUsbDeviceOutput)
    }

    pub fn is_flush_voice_wave_buffers_supported(&self) -> bool {
        self.supported(SupportTags::FlushVoiceWaveBuffers)
    }

    pub fn is_elapsed_frame_count_supported(&self) -> bool {
        self.supported(SupportTags::ElapsedFrameCount)
    }

    pub fn is_variadic_command_buffer_size_supported(&self) -> bool {
        self.supported(SupportTags::AudioRendererVariadicCommandBufferSize)
    }

    pub fn is_decoding_behaviour_flag_supported(&self) -> bool {
        self.supported(SupportTags::DecodingBehaviourFlag)
    }

    pub fn is_voice_played_sample_count_reset_at_loop_point_supported(&self) -> bool {
        self.supported(SupportTags::VoicePlayedSampleCountResetAtLoopPoint)
    }

    pub fn is_voice_pitch_and_src_skipped_supported(&self) -> bool {
        self.supported(SupportTags::VoicePitchAndSrcSkipped)
    }

    pub fn is_biquad_filter_effect_state_clear_bug_fixed(&self) -> bool {
        self.supported(SupportTags::BiquadFilterEffectStateClearBugFix)
    }

    pub fn use_biquad_filter_float_processing(&self) -> bool {
        self.supported(SupportTags::BiquadFilterFloatProcessing)
    }

    pub fn is_volume_mix_parameter_precision_q23_supported(&self) -> bool {
        self.supported(SupportTags::VolumeMixParameterPrecisionQ23)
    }

    pub fn volume_mix_parameter_precision(&self) -> u8 {
        if self.is_volume_mix_parameter_precision_q23_supported() {
            23
        } else {
            15
        }
    }

    pub fn is_mix_in_parameter_dirty_only_update_supported(&self) -> bool {
        self.supported(SupportTags::MixInParameterDirtyOnlyUpdate)
    }

    pub fn is_wave_buffer_ver2_supported(&self) -> bool {
        self.supported(SupportTags::WaveBufferVer2)
    }

    pub fn is_effect_info_version2_supported(&self) -> bool {
        self.supported(SupportTags::EffectInfoVer2)
    }

    pub fn use_multi_tap_biquad_filter_processing(&self) -> bool {
        self.supported(SupportTags::MultiTapBiquadFilterProcessing)
    }

    pub fn is_device_api_version2_supported(&self) -> bool {
        self.supported(SupportTags::DeviceApiVersion2)
    }

    pub fn is_delay_channel_mapping_changed(&self) -> bool {
        self.supported(SupportTags::DelayChannelMappingChange)
    }

    pub fn is_reverb_channel_mapping_changed(&self) -> bool {
        self.supported(SupportTags::ReverbChannelMappingChange)
    }

    pub fn is_i3dl2_reverb_channel_mapping_changed(&self) -> bool {
        self.supported(SupportTags::I3dl2ReverbChannelMappingChange)
    }

    pub fn is_biquad_filter_parameter_for_splitter_enabled(&self) -> bool {
        self.supported(SupportTags::SplitterBiquadFilterParameter)
    }

    pub fn is_splitter_prev_volume_reset_supported(&self) -> bool {
        self.supported(SupportTags::SplitterPrevVolumeReset)
    }

    pub fn is_splitter_destination_v2b_supported(&self) -> bool {
        self.supported(SupportTags::SplitterDestinationV2b)
    }

    pub fn is_voice_in_parameter_v2_supported(&self) -> bool {
        self.supported(SupportTags::VoiceInParameterV2)
    }

    pub fn is_performance_metrics_data_format_version2_supported(&self) -> bool {
        self.supported(SupportTags::PerformanceMetricsDataFormatVersion2)
    }

    pub fn performance_metrics_data_format(&self) -> u32 {
        if self.is_performance_metrics_data_format_version2_supported() {
            2
        } else {
            1
        }
    }

    pub fn audio_renderer_processing_time_limit_percent(&self) -> u32 {
        if self.supported(SupportTags::AudioRendererProcessingTimeLimit80Percent) {
            80
        } else if self.supported(SupportTags::AudioRendererProcessingTimeLimit75Percent) {
            75
        } else {
            70
        }
    }

    pub fn command_processing_time_estimator_version(&self) -> u32 {
        if self.supported(SupportTags::CommandProcessingTimeEstimatorVersion5) {
            5
        } else if self.supported(SupportTags::CommandProcessingTimeEstimatorVersion4) {
            4
        } else if self.supported(SupportTags::CommandProcessingTimeEstimatorVersion3) {
            3
        } else if self.supported(SupportTags::CommandProcessingTimeEstimatorVersion2) {
            2
        } else {
            1
        }
    }
}

impl Default for BehaviorInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_every_revision_magic_including_those_above_rev9() {
        for revision in 0..=15u32 {
            let magic = encode_revision(revision);
            assert_eq!(get_revision_num(magic), revision, "revision {}", revision);
        }
        assert_eq!(get_revision_num(0x3856_4552), 8);
        assert_eq!(get_revision_num(encode_revision(10)), 10);
        assert_eq!(get_revision_num(encode_revision(15)), 15);
    }

    #[test]
    fn revision_magics_use_ascii_above_nine_past_rev9() {
        assert_eq!(encode_revision(9), u32::from_le_bytes(*b"REV9"));
        assert_eq!(encode_revision(10), u32::from_le_bytes(*b"REV:"));
        assert_eq!(encode_revision(11), u32::from_le_bytes(*b"REV;"));
        assert_eq!(encode_revision(15), u32::from_le_bytes(*b"REV?"));
    }

    #[test]
    fn bare_revision_numbers_pass_through_undecoded() {
        for revision in 0..0x100u32 {
            assert_eq!(get_revision_num(revision), revision);
        }
    }

    #[test]
    fn validity_follows_the_current_revision() {
        assert!(check_valid_revision(0));
        assert!(check_valid_revision(15));
        assert!(check_valid_revision(encode_revision(15)));
        assert!(!check_valid_revision(16));
        assert!(!check_valid_revision(encode_revision(16)));
    }

    #[test]
    fn every_feature_switches_on_at_its_own_revision() {
        for (tag, minimum) in FEATURES {
            if *minimum > 1 {
                assert!(
                    !check_feature_supported(*tag, minimum - 1),
                    "{:?} must be off at revision {}",
                    tag,
                    minimum - 1
                );
            }
            for revision in *minimum..=CURRENT_REVISION {
                assert!(
                    check_feature_supported(*tag, revision),
                    "{:?} must be on at revision {}",
                    tag,
                    revision
                );
            }
        }
    }

    #[test]
    fn an_out_of_range_revision_degrades_to_revision_one() {
        assert!(check_feature_supported(
            SupportTags::AudioRendererProcessingTimeLimit70Percent,
            99
        ));
        assert!(!check_feature_supported(SupportTags::Splitter, 99));
        assert!(!check_feature_supported(SupportTags::WaveBufferVer2, 99));
        assert!(!check_feature_supported(SupportTags::VoiceInParameterV2, 99));
    }

    #[test]
    fn metroid_dread_reports_the_rev8_feature_set() {
        let behavior = BehaviorInfo::from_user_revision(0x3856_4552);
        assert_eq!(behavior.get_user_revision_num(), 8);
        assert!(behavior.is_wave_buffer_ver2_supported());
        assert!(behavior.is_elapsed_frame_count_supported());
        assert!(behavior.is_mix_in_parameter_dirty_only_update_supported());
        assert!(!behavior.is_effect_info_version2_supported());
        assert!(!behavior.use_multi_tap_biquad_filter_processing());
        assert_eq!(behavior.command_processing_time_estimator_version(), 3);
        assert_eq!(behavior.audio_renderer_processing_time_limit_percent(), 80);
        assert_eq!(behavior.performance_metrics_data_format(), 2);
        assert_eq!(behavior.volume_mix_parameter_precision(), 23);
    }

    #[test]
    fn estimator_version_five_requires_revision_eleven() {
        assert_eq!(
            BehaviorInfo::from_user_revision(10).command_processing_time_estimator_version(),
            4
        );
        assert_eq!(
            BehaviorInfo::from_user_revision(11).command_processing_time_estimator_version(),
            5
        );
    }

    #[test]
    fn processing_time_limit_climbs_with_the_revision() {
        assert_eq!(
            BehaviorInfo::from_user_revision(1).audio_renderer_processing_time_limit_percent(),
            70
        );
        assert_eq!(
            BehaviorInfo::from_user_revision(4).audio_renderer_processing_time_limit_percent(),
            75
        );
        assert_eq!(
            BehaviorInfo::from_user_revision(5).audio_renderer_processing_time_limit_percent(),
            80
        );
    }

    #[test]
    fn errors_saturate_at_ten_and_the_tail_is_zero_filled() {
        let mut behavior = BehaviorInfo::new();
        for index in 0..(MAX_ERRORS as u64 + 5) {
            behavior.append_error(ErrorInfo {
                error_code: index as u32 + 1,
                unk_04: 0,
                address: 0x1000 + index,
            });
        }
        assert_eq!(behavior.error_count(), MAX_ERRORS as u32);

        let status = behavior.out_status();
        assert_eq!(status.error_count, MAX_ERRORS as u32);
        assert_eq!(status.errors[0].error_code, 1);
        assert_eq!(status.errors[MAX_ERRORS - 1].error_code, MAX_ERRORS as u32);

        behavior.clear_error();
        let cleared = behavior.out_status();
        assert_eq!(cleared.error_count, 0);
        assert!(cleared.errors.iter().all(|e| *e == ErrorInfo::default()));
    }

    #[test]
    fn out_status_serializes_to_the_guest_block_layout() {
        let mut behavior = BehaviorInfo::new();
        behavior.append_error(ErrorInfo {
            error_code: 0xDEAD_BEEF,
            unk_04: 0,
            address: 0x1234_5678_9ABC,
        });
        let mut block = vec![0xAAu8; OUT_STATUS_SIZE];
        assert!(behavior.out_status().write_to(&mut block));

        assert_eq!(&block[0x00..0x04], &0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(&block[0x08..0x10], &0x1234_5678_9ABCu64.to_le_bytes());
        assert_eq!(&block[0x10..0x20], &[0u8; 0x10]);
        assert_eq!(&block[0xA0..0xA4], &1u32.to_le_bytes());
        assert!(block[0xA4..0xB0].iter().all(|byte| *byte == 0));

        assert!(!behavior.out_status().write_to(&mut [0u8; OUT_STATUS_SIZE - 1]));
    }

    #[test]
    fn in_parameter_parses_revision_and_force_mapping_flag() {
        let mut bytes = [0u8; IN_PARAMETER_SIZE];
        bytes[0x00..0x04].copy_from_slice(&encode_revision(11).to_le_bytes());
        bytes[0x08] = 1;
        let parsed = InParameter::parse(&bytes).expect("parses");
        assert_eq!(get_revision_num(parsed.revision), 11);
        assert!(parsed.flags.is_memory_force_mapping_enabled);

        bytes[0x08] = 0;
        let cleared = InParameter::parse(&bytes).expect("parses");
        assert!(!cleared.flags.is_memory_force_mapping_enabled);

        assert!(InParameter::parse(&bytes[..IN_PARAMETER_SIZE - 1]).is_none());
    }

    #[test]
    fn revision_round_trips_through_the_encoder() {
        let behavior = BehaviorInfo::from_user_revision(encode_revision(13));
        assert_eq!(behavior.get_user_revision_num(), 13);
        assert_eq!(behavior.get_user_revision(), encode_revision(13));
        assert_eq!(behavior.get_process_revision_num(), CURRENT_REVISION);
        assert_eq!(
            behavior.get_process_revision(),
            encode_revision(CURRENT_REVISION)
        );
    }
}

#[derive(Clone, Copy)]
enum PatchData<'a> {
    Literal(&'a [u8]),
    Repeated { value: u8, len: usize },
}

impl PatchData<'_> {
    fn len(self) -> usize {
        match self {
            Self::Literal(bytes) => bytes.len(),
            Self::Repeated { len, .. } => len,
        }
    }

    fn skip(self, count: usize) -> Self {
        match self {
            Self::Literal(bytes) => Self::Literal(&bytes[count..]),
            Self::Repeated { value, len } => Self::Repeated {
                value,
                len: len - count,
            },
        }
    }
}

fn take<'a>(remaining: &mut &'a [u8], len: usize, field: &str) -> Result<&'a [u8], String> {
    if remaining.len() < len {
        return Err(format!(
            "truncated IPS {field}: need {len} bytes, found {}",
            remaining.len()
        ));
    }
    let (value, rest) = remaining.split_at(len);
    *remaining = rest;
    Ok(value)
}

fn walk_ips<'a>(
    patch: &'a [u8],
    image_len: usize,
    offset_bias: usize,
    mut apply: impl FnMut(usize, PatchData<'a>),
) -> Result<usize, String> {
    let mut remaining = patch;
    let (offset_len, terminator): (usize, &[u8]) = match take(&mut remaining, 5, "header")? {
        b"PATCH" => (3, b"EOF"),
        b"IPS32" => (4, b"EEOF"),
        _ => return Err("invalid IPS header: expected PATCH or IPS32".to_string()),
    };
    let mut written = 0usize;
    loop {
        let encoded_offset = take(&mut remaining, offset_len, "record offset or terminator")?;
        if encoded_offset == terminator {
            if !remaining.is_empty() {
                return Err(format!(
                    "unexpected {} bytes after IPS terminator",
                    remaining.len()
                ));
            }
            return Ok(written);
        }
        let offset = encoded_offset
            .iter()
            .fold(0u32, |value, byte| (value << 8) | u32::from(*byte));
        let offset = usize::try_from(offset).map_err(|_| "IPS offset exceeds address space")?;
        let size = take(&mut remaining, 2, "record length")?;
        let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
        let data = if size == 0 {
            let run = take(&mut remaining, 3, "RLE record")?;
            PatchData::Repeated {
                value: run[2],
                len: usize::from(u16::from_be_bytes([run[0], run[1]])),
            }
        } else {
            PatchData::Literal(take(&mut remaining, size, "literal payload")?)
        };
        let end = offset
            .checked_add(data.len())
            .ok_or("IPS record end overflow")?;
        if end <= offset_bias {
            continue;
        }
        let destination_end = end - offset_bias;
        if destination_end > image_len {
            return Err(format!(
                "IPS record {offset:#x}..{end:#x} exceeds image length {image_len:#x} with bias {offset_bias:#x}"
            ));
        }
        let skipped = offset_bias.saturating_sub(offset);
        let data = data.skip(skipped);
        let destination = offset.max(offset_bias) - offset_bias;
        written = written
            .checked_add(data.len())
            .ok_or("IPS written byte count overflow")?;
        apply(destination, data);
    }
}

pub fn apply_ips(image: &mut [u8], patch: &[u8], offset_bias: usize) -> Result<usize, String> {
    walk_ips(patch, image.len(), offset_bias, |_, _| {})?;
    walk_ips(patch, image.len(), offset_bias, |offset, data| {
        let destination = &mut image[offset..offset + data.len()];
        match data {
            PatchData::Literal(bytes) => destination.copy_from_slice(bytes),
            PatchData::Repeated { value, .. } => destination.fill(value),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::apply_ips;

    fn literal(patch: &mut Vec<u8>, wide: bool, offset: u32, bytes: &[u8]) {
        let encoded = offset.to_be_bytes();
        patch.extend_from_slice(&encoded[usize::from(!wide)..]);
        patch.extend_from_slice(&u16::try_from(bytes.len()).unwrap().to_be_bytes());
        patch.extend_from_slice(bytes);
    }

    fn repeated(patch: &mut Vec<u8>, wide: bool, offset: u32, len: u16, value: u8) {
        let encoded = offset.to_be_bytes();
        patch.extend_from_slice(&encoded[usize::from(!wide)..]);
        patch.extend_from_slice(&[0, 0]);
        patch.extend_from_slice(&len.to_be_bytes());
        patch.push(value);
    }

    #[test]
    fn ips_literal_and_rle_records_use_big_endian_offsets_and_lengths() {
        let patch = b"PATCH\x00\x01\x00\x00\x03\x12\x34\x56\x00\x01\x04\x00\x00\x01\x02\xA5EOF";
        let mut image = vec![0xCC; 0x110];
        assert_eq!(apply_ips(&mut image, patch, 0x100).unwrap(), 3 + 0x102);
        assert_eq!(&image[..4], &[0x12, 0x34, 0x56, 0xCC]);
        assert!(image[4..0x106].iter().all(|byte| *byte == 0xA5));
        assert!(image[0x106..].iter().all(|byte| *byte == 0xCC));
    }

    #[test]
    fn ips32_preserves_offsets_above_sixteen_megabytes() {
        let patch =
            b"IPS32\x01\x00\x01\x00\x00\x02\x12\x34\x01\x00\x01\x03\x00\x00\x00\x03\xA5EEOF";
        let mut image = [0xCC; 8];
        assert_eq!(apply_ips(&mut image, patch, 0x01000100).unwrap(), 5);
        assert_eq!(image, [0x12, 0x34, 0xCC, 0xA5, 0xA5, 0xA5, 0xCC, 0xCC]);
    }

    #[test]
    fn protected_header_is_excluded_and_crossing_records_are_clipped() {
        for wide in [false, true] {
            let mut patch = if wide {
                b"IPS32".to_vec()
            } else {
                b"PATCH".to_vec()
            };
            literal(&mut patch, wide, 0x10, &[0xEE; 8]);
            repeated(&mut patch, wide, 0xF0, 16, 0xDD);
            literal(&mut patch, wide, 0xFC, &[1, 2, 3, 4, 5, 6, 7, 8]);
            repeated(&mut patch, wide, 0xFE, 4, 0xAA);
            patch.extend_from_slice(if wide { b"EEOF" } else { b"EOF" });
            let mut image = [0xCC; 8];
            assert_eq!(apply_ips(&mut image, &patch, 0x100).unwrap(), 6);
            assert_eq!(image, [0xAA, 0xAA, 7, 8, 0xCC, 0xCC, 0xCC, 0xCC]);
        }
    }

    #[test]
    fn record_order_is_preserved_and_written_count_includes_overlaps() {
        let mut patch = b"PATCH".to_vec();
        literal(&mut patch, false, 0, &[1, 2, 3]);
        repeated(&mut patch, false, 1, 3, 9);
        patch.extend_from_slice(b"EOF");
        let mut image = [0; 4];
        assert_eq!(apply_ips(&mut image, &patch, 0).unwrap(), 6);
        assert_eq!(image, [1, 9, 9, 9]);
    }

    #[test]
    fn invalid_records_and_trailing_data_leave_the_whole_image_unchanged() {
        for wide in [false, true] {
            let mut prefix = if wide {
                b"IPS32".to_vec()
            } else {
                b"PATCH".to_vec()
            };
            literal(&mut prefix, wide, 0x100, &[1, 2]);
            let mut cases = Vec::new();
            for rle in [false, true] {
                let mut patch = prefix.clone();
                if rle {
                    repeated(&mut patch, wide, 0x103, 2, 0xEE);
                } else {
                    literal(&mut patch, wide, 0x103, &[0xDD, 0xEE]);
                }
                patch.extend_from_slice(if wide { b"EEOF" } else { b"EOF" });
                cases.push(patch);
            }
            cases.push(prefix.clone());
            let mut trailing = prefix;
            trailing.extend_from_slice(if wide { b"EEOF" } else { b"EOF" });
            for suffix in [&[1][..], &[0, 0, 4][..], &[0, 0, 0, 4][..]] {
                let mut patch = trailing.clone();
                patch.extend_from_slice(suffix);
                cases.push(patch);
            }
            for patch in cases {
                let mut image = [0xCC; 4];
                assert!(apply_ips(&mut image, &patch, 0x100).is_err());
                assert_eq!(image, [0xCC; 4]);
            }
        }
    }

    #[test]
    fn every_truncated_literal_rle_and_footer_prefix_is_rejected_atomically() {
        for wide in [false, true] {
            let mut patch = if wide {
                b"IPS32".to_vec()
            } else {
                b"PATCH".to_vec()
            };
            literal(&mut patch, wide, 0, &[1, 2]);
            repeated(&mut patch, wide, 2, 3, 9);
            patch.extend_from_slice(if wide { b"EEOF" } else { b"EOF" });
            for length in 0..patch.len() {
                let mut image = [0xCC; 8];
                assert!(
                    apply_ips(&mut image, &patch[..length], 0).is_err(),
                    "length={length}, wide={wide}"
                );
                assert_eq!(image, [0xCC; 8]);
            }
            let mut image = [0; 8];
            assert_eq!(apply_ips(&mut image, &patch, 0).unwrap(), 5);
        }
    }

    #[test]
    fn ignored_header_records_still_require_complete_payloads() {
        let mut image = [0xCC; 4];
        assert!(apply_ips(&mut image, b"PATCH\x00\x00\x00\x00\x05\xAAEOF", 0x100).is_err());
        assert_eq!(image, [0xCC; 4]);
        assert!(apply_ips(&mut image, b"PATCH\x00\x00\x00\x00\x00\x00", 0x100).is_err());
        assert_eq!(image, [0xCC; 4]);
    }

    #[test]
    fn empty_patches_zero_length_runs_and_address_boundaries_are_safe() {
        assert_eq!(apply_ips(&mut [], b"PATCHEOF", 0).unwrap(), 0);
        assert_eq!(apply_ips(&mut [], b"IPS32EEOF", usize::MAX).unwrap(), 0);
        let mut patch = b"IPS32".to_vec();
        repeated(&mut patch, true, 4, 0, 0xAA);
        patch.extend_from_slice(b"EEOF");
        let mut image = [0xCC; 4];
        assert_eq!(apply_ips(&mut image, &patch, 0).unwrap(), 0);
        assert_eq!(image, [0xCC; 4]);
        let mut patch = b"IPS32".to_vec();
        literal(&mut patch, true, u32::MAX, &[1, 2]);
        patch.extend_from_slice(b"EEOF");
        assert!(apply_ips(&mut image, &patch, 0).is_err());
        assert_eq!(image, [0xCC; 4]);
        if usize::BITS > 32 {
            assert_eq!(apply_ips(&mut image, &patch, usize::MAX).unwrap(), 0);
        } else {
            assert!(apply_ips(&mut image, &patch, usize::MAX).is_err());
        }
        assert_eq!(image, [0xCC; 4]);
    }

    #[test]
    fn wrong_format_and_mismatched_terminators_are_rejected() {
        for patch in [
            &b"patchEOF"[..],
            &b"IPS32EOF"[..],
            &b"PATCHEEOF"[..],
            &b"BPS1EOF"[..],
        ] {
            let mut image = [0xCC; 4];
            assert!(apply_ips(&mut image, patch, 0).is_err());
            assert_eq!(image, [0xCC; 4]);
        }
    }
}

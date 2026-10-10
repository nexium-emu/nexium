use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AstcDecodeMode {
    #[default]
    Gpu,
    Cpu,
    CpuAsynchronous,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AstcRecompression {
    #[default]
    Uncompressed,
    Bc1,
    Bc3,
}

static DECODE_MODE: AtomicU8 = AtomicU8::new(0);
static RECOMPRESSION: AtomicU8 = AtomicU8::new(0);

impl AstcDecodeMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gpu" => Some(Self::Gpu),
            "cpu" => Some(Self::Cpu),
            "async" | "cpu_async" | "cpu-async" | "cpuasync" | "cpu_asynchronous" => {
                Some(Self::CpuAsynchronous)
            }
            _ => None,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Cpu,
            2 => Self::CpuAsynchronous,
            _ => Self::Gpu,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Gpu => 0,
            Self::Cpu => 1,
            Self::CpuAsynchronous => 2,
        }
    }
}

impl AstcRecompression {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "uncompressed" | "none" | "off" | "rgba8" => Some(Self::Uncompressed),
            "bc1" => Some(Self::Bc1),
            "bc3" => Some(Self::Bc3),
            _ => None,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Bc1,
            2 => Self::Bc3,
            _ => Self::Uncompressed,
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Uncompressed => 0,
            Self::Bc1 => 1,
            Self::Bc3 => 2,
        }
    }
}

pub fn set_decode_mode(mode: AstcDecodeMode) {
    DECODE_MODE.store(mode.to_u8(), Ordering::Relaxed);
}

pub fn decode_mode() -> AstcDecodeMode {
    AstcDecodeMode::from_u8(DECODE_MODE.load(Ordering::Relaxed))
}

pub fn set_recompression(recompression: AstcRecompression) {
    RECOMPRESSION.store(recompression.to_u8(), Ordering::Relaxed);
}

pub fn recompression() -> AstcRecompression {
    AstcRecompression::from_u8(RECOMPRESSION.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astc_settings_parse_and_round_trip() {
        assert_eq!(AstcDecodeMode::parse(" GPU "), Some(AstcDecodeMode::Gpu));
        assert_eq!(AstcDecodeMode::parse("cpu"), Some(AstcDecodeMode::Cpu));
        assert_eq!(AstcDecodeMode::parse("cpu_async"), Some(AstcDecodeMode::CpuAsynchronous));
        assert_eq!(AstcDecodeMode::parse("fast"), None);
        assert_eq!(AstcRecompression::parse("none"), Some(AstcRecompression::Uncompressed));
        assert_eq!(AstcRecompression::parse("BC1"), Some(AstcRecompression::Bc1));
        assert_eq!(AstcRecompression::parse("bc3"), Some(AstcRecompression::Bc3));
        assert_eq!(AstcRecompression::parse("bc7"), None);
        for mode in [AstcDecodeMode::Gpu, AstcDecodeMode::Cpu, AstcDecodeMode::CpuAsynchronous] {
            assert_eq!(AstcDecodeMode::from_u8(mode.to_u8()), mode);
        }
        for value in [AstcRecompression::Uncompressed, AstcRecompression::Bc1, AstcRecompression::Bc3] {
            assert_eq!(AstcRecompression::from_u8(value.to_u8()), value);
        }
    }
}

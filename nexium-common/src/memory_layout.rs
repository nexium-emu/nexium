use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryLayout {
    #[default]
    Gb4,
    Gb6,
    Gb8,
    Gb10,
    Gb12,
}

static LAYOUT: AtomicU8 = AtomicU8::new(0);

impl MemoryLayout {
    pub const ALL: [Self; 5] = [Self::Gb4, Self::Gb6, Self::Gb8, Self::Gb10, Self::Gb12];

    pub fn application_pool_bytes(self) -> u64 {
        match self {
            Self::Gb4 => 3285 << 20,
            Self::Gb6 => 4916 << 20,
            Self::Gb8 => 6547 << 20,
            Self::Gb10 => 8178 << 20,
            Self::Gb12 => 9809 << 20,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let digits: String = value.chars().filter(char::is_ascii_digit).collect();
        match digits.as_str() {
            "4" => Some(Self::Gb4),
            "6" => Some(Self::Gb6),
            "8" => Some(Self::Gb8),
            "10" => Some(Self::Gb10),
            "12" => Some(Self::Gb12),
            _ => None,
        }
    }
}

pub fn set_memory_layout(layout: MemoryLayout) {
    let index = MemoryLayout::ALL.iter().position(|candidate| *candidate == layout).unwrap_or(0);
    LAYOUT.store(index as u8, Ordering::Relaxed);
}

pub fn memory_layout() -> MemoryLayout {
    static OVERRIDE: OnceLock<Option<MemoryLayout>> = OnceLock::new();
    OVERRIDE
        .get_or_init(|| std::env::var("NEXIUM_MEMORY_LAYOUT").ok().and_then(|value| MemoryLayout::parse(&value)))
        .unwrap_or_else(|| MemoryLayout::ALL[usize::from(LAYOUT.load(Ordering::Relaxed)).min(MemoryLayout::ALL.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_parse_and_size_the_application_pool() {
        assert_eq!(MemoryLayout::parse("8"), Some(MemoryLayout::Gb8));
        assert_eq!(MemoryLayout::parse("12 GB"), Some(MemoryLayout::Gb12));
        assert_eq!(MemoryLayout::parse("6gb"), Some(MemoryLayout::Gb6));
        assert_eq!(MemoryLayout::parse("5"), None);
        assert_eq!(MemoryLayout::Gb4.application_pool_bytes(), 0xCD50_0000);
        assert_eq!(MemoryLayout::Gb8.application_pool_bytes(), 6547 << 20);
        assert_eq!(MemoryLayout::Gb12.application_pool_bytes(), 9809 << 20);
    }
}

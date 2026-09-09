use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const MULTIPLIER: u64 = 0xf135_7aea_2e62_a9c5;

#[derive(Clone, Copy, Default)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word) ^ ((rest.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_u128(&mut self, value: u128) {
        self.add(value as u64);
        self.add((value >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn write_i8(&mut self, value: i8) {
        self.add(value as u8 as u64);
    }

    #[inline]
    fn write_i16(&mut self, value: i16) {
        self.add(value as u16 as u64);
    }

    #[inline]
    fn write_i32(&mut self, value: i32) {
        self.add(value as u32 as u64);
    }

    #[inline]
    fn write_i64(&mut self, value: i64) {
        self.add(value as u64);
    }

    #[inline]
    fn write_isize(&mut self, value: isize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut hash = self.hash;
        hash ^= hash >> 32;
        hash = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        hash ^= hash >> 29;
        hash
    }
}

pub type FastState = BuildHasherDefault<FastHasher>;
pub type FastMap<K, V> = HashMap<K, V, FastState>;
pub type FastSet<K> = HashSet<K, FastState>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn distinct_keys_hash_apart_and_maps_round_trip() {
        let state = FastState::default();
        let a = state.hash_one(0x1000u64);
        let b = state.hash_one(0x1001u64);
        let c = state.hash_one((0x1000u64, 7u32));
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, state.hash_one(0x1000u64));

        let mut map: FastMap<(u64, u32), &str> = FastMap::default();
        map.insert((1, 2), "one");
        map.insert((1, 3), "two");
        map.insert((u64::MAX, u32::MAX), "max");
        assert_eq!(map.get(&(1, 2)), Some(&"one"));
        assert_eq!(map.get(&(1, 3)), Some(&"two"));
        assert_eq!(map.get(&(u64::MAX, u32::MAX)), Some(&"max"));
        assert_eq!(map.get(&(2, 2)), None);

        let mut low_bits = std::collections::HashSet::new();
        for page in 0u64..4096 {
            low_bits.insert(state.hash_one(page << 16) & 0xfff);
        }
        assert!(
            low_bits.len() > 2400,
            "aligned keys cluster: {}",
            low_bits.len()
        );

        let mut set: FastSet<String> = FastSet::default();
        assert!(set.insert("abcdefghij".to_string()));
        assert!(!set.insert("abcdefghij".to_string()));
        assert!(set.insert("abcdefghik".to_string()));
    }
}

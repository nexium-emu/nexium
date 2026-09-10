use std::hint::black_box;
use std::time::{Duration, Instant};

fn previous_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325 ^ bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        hash ^= u64::from_le_bytes(chunk.try_into().unwrap());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut tail = 0u64;
    for (index, &byte) in chunks.remainder().iter().enumerate() {
        tail |= (byte as u64) << (index * 8);
    }
    (hash ^ tail).wrapping_mul(0x100000001b3)
}

fn measure(hash: fn(&[u8]) -> u64, data: &[u8], size: usize) -> f64 {
    let mut bytes = 0u64;
    let started = Instant::now();
    let mut slices = data.chunks_exact(size).cycle();
    loop {
        for _ in 0..16 {
            black_box(hash(black_box(slices.next().unwrap())));
            bytes += size as u64;
        }
        if started.elapsed() >= Duration::from_millis(200) {
            return bytes as f64 / started.elapsed().as_secs_f64() / 1_073_741_824.0;
        }
    }
}

fn main() {
    let data: Vec<u8> = (0..256 * 1024 * 1024usize)
        .map(|index| index.wrapping_mul(73).wrapping_add(index >> 8) as u8)
        .collect();
    println!("working_set,snapshot_bytes,previous_gib_s,xxh3_gib_s,speedup");
    for streaming in [false, true] {
        for size in [4096, 65536, 1 << 20, 16 << 20, 64 << 20] {
            let working_set = if streaming { &data[..] } else { &data[..size] };
            let mut previous = [0.0; 3];
            let mut current = [0.0; 3];
            for iteration in 0..3 {
                if iteration % 2 == 0 {
                    previous[iteration] = measure(previous_hash, working_set, size);
                    current[iteration] = measure(xxhash_rust::xxh3::xxh3_64, working_set, size);
                } else {
                    current[iteration] = measure(xxhash_rust::xxh3::xxh3_64, working_set, size);
                    previous[iteration] = measure(previous_hash, working_set, size);
                }
            }
            previous.sort_by(f64::total_cmp);
            current.sort_by(f64::total_cmp);
            println!(
                "{},{size},{:.3},{:.3},{:.3}",
                if streaming { "streaming" } else { "cached" },
                previous[1],
                current[1],
                current[1] / previous[1],
            );
        }
    }
}

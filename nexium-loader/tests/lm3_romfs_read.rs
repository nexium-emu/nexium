//! Replays representative RomFS reads against the user's Luigi's Mansion 3
//! dump, opened exactly as boot opens it (base + installed update + DLC).
//!
//! The profile is `#[ignore]`d: run it with
//! `cargo test --release -p nexium-loader --test lm3_romfs_read -- --ignored --nocapture`.
//! Every test prints a SKIP line and passes when the dump is not present.

use nexium_loader::{Application, LazyRomfs};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Default `NEXIUM_FS_OFFLOCK_READ_KIB` threshold in the kernel.
const OFF_LOCK_MIN: usize = 1 << 20;

const LM3: &str = "/home/sylv/Documents/NSP/LM3/Luigi\u{2019}s Mansion 3[0100DCA0064A6000][US][v0].dnsp";

fn content_root() -> PathBuf {
    std::env::var_os("NEXIUM_CONTENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config/NeXium/content")
        })
}

fn open() -> Option<LazyRomfs> {
    let path = std::env::var("NEXIUM_LM3_PATH").unwrap_or_else(|_| LM3.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("SKIP: {path} is not present");
        return None;
    }
    let started = Instant::now();
    let app = Application::load_with_content(&path, &content_root()).expect("load LM3");
    let romfs = app.romfs.expect("LM3 has a RomFS");
    eprintln!(
        "opened LM3 in {:?}: romfs len={:#x} compressed_or_patched={}",
        started.elapsed(),
        romfs.len(),
        romfs.is_compressed()
    );
    Some(romfs)
}

fn u32le(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64le(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// (absolute data offset, size) of every file in the RomFS.
fn files(romfs: &LazyRomfs) -> Vec<(u64, u64)> {
    let header = romfs.read(0, 0x50).unwrap();
    let table = romfs
        .read(u64le(&header, 0x38), u64le(&header, 0x40) as usize)
        .unwrap();
    let data = u64le(&header, 0x48);
    let mut out = Vec::new();
    let mut at = 0;
    while at + 0x20 <= table.len() {
        let name_len = u32le(&table, at + 0x1c) as usize;
        out.push((data + u64le(&table, at + 8), u64le(&table, at + 0x10)));
        at += 0x20 + name_len.div_ceil(4) * 4;
    }
    out
}

fn faults() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest.split_whitespace().collect())
        .unwrap_or_default();
    let field = |index: usize| fields.get(index).and_then(|v| v.parse().ok()).unwrap_or(0);
    (field(7), field(9))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[derive(Default)]
struct Tally {
    reads: u64,
    bytes: u64,
    read: Duration,
    copy: Duration,
    worst: Duration,
    /// Worst time the kernel lock would be held for one read: the whole
    /// read plus copy below the off-lock threshold, only the copy above it.
    worst_held: Duration,
    minor: u64,
    major: u64,
}

impl Tally {
    fn report(&self, label: &str) {
        let mb = self.bytes as f64 / (1 << 20) as f64;
        eprintln!(
            "{label:<44} reads={:<5} MiB={mb:<8.1} read={:>8.2}ms ({:>7.0} MiB/s, worst {:>6.2}ms) guest_copy={:>7.2}ms held_now<={:>6.2}ms faults minor={} major={}",
            self.reads,
            self.read.as_secs_f64() * 1e3,
            mb / self.read.as_secs_f64().max(1e-9),
            self.worst.as_secs_f64() * 1e3,
            self.copy.as_secs_f64() * 1e3,
            self.worst_held.as_secs_f64() * 1e3,
            self.minor,
            self.major
        );
    }
}

/// One guest-sized read: storage read plus the copy the kernel performs into
/// guest memory (`destination` stands in for the guest buffer).
/// Reads at or above `OFF_LOCK_MIN` reuse a pooled buffer the way the
/// kernel's off-lock worker does (`NEXIUM_BENCH_FRESH=1` allocates each time,
/// as the kernel did before).
fn replay(romfs: &LazyRomfs, offset: u64, size: usize, destination: &mut [u8], tally: &mut Tally) {
    thread_local! {
        static POOL: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let pooled = size >= OFF_LOCK_MIN && std::env::var_os("NEXIUM_BENCH_FRESH").is_none();
    let before = faults();
    let started = Instant::now();
    let bytes = if pooled {
        let mut buffer = POOL.with(|pool| std::mem::take(&mut *pool.borrow_mut()));
        buffer.resize(buffer.len().max(size), 0);
        assert_eq!(romfs.read_into(offset, &mut buffer[..size]).unwrap(), size);
        buffer
    } else {
        romfs.read(offset, size).unwrap()
    };
    let read = started.elapsed();
    let started = Instant::now();
    destination[..size.min(bytes.len())].copy_from_slice(&bytes[..size.min(bytes.len())]);
    let copy = started.elapsed();
    if pooled {
        POOL.with(|pool| *pool.borrow_mut() = bytes);
    } else {
        drop(bytes);
    }
    let after = faults();
    tally.reads += 1;
    tally.bytes += size as u64;
    tally.read += read;
    tally.copy += copy;
    tally.worst = tally.worst.max(read);
    let held = if size >= OFF_LOCK_MIN { copy } else { read + copy };
    tally.worst_held = tally.worst_held.max(held);
    tally.minor += after.0 - before.0;
    tally.major += after.1 - before.1;
}

#[test]
#[ignore]
fn lm3_romfs_read_profile() {
    let Some(romfs) = open() else { return };
    let files = files(&romfs);
    let total: u64 = files.iter().map(|file| file.1).sum();
    eprintln!("{} files, {:.1} MiB of file data", files.len(), total as f64 / (1 << 20) as f64);
    let mut destination = vec![0u8; 64 << 20];
    destination.fill(1);

    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for pass in ["first", "repeat"] {
        let mut rng_pass = Rng(rng.0);
        let mut tally = Tally::default();
        for _ in 0..2000 {
            let offset = rng_pass.next() % (romfs.len() - 0x4000);
            replay(&romfs, offset, 0x4000, &mut destination, &mut tally);
        }
        tally.report(&format!("random 16 KiB reads ({pass})"));
    }
    rng.next();

    let mut large: Vec<_> = files.iter().copied().filter(|file| file.1 >= 1 << 20).collect();
    large.sort_by_key(|file| std::cmp::Reverse(file.1));
    eprintln!(
        "{} files >= 1 MiB; largest {:.1} MiB",
        large.len(),
        large.first().map_or(0.0, |file| file.1 as f64 / (1 << 20) as f64)
    );
    let picks: Vec<_> = (0..24).map(|i| large[(i * 7919) % large.len()]).collect();
    for pass in ["first", "repeat"] {
        let mut tally = Tally::default();
        for &(offset, size) in &picks {
            let size = (size as usize).min(destination.len());
            replay(&romfs, offset, size, &mut destination, &mut tally);
        }
        tally.report(&format!("whole-file reads 1-64 MiB ({pass})"));
    }
    for chunk in [0x4_0000usize, 0x10_0000, 0x40_0000] {
        for pass in ["first", "repeat"] {
            let mut tally = Tally::default();
            let (offset, size) = large[(chunk / 0x4_0000 * 31) % large.len()];
            let mut done = 0u64;
            while done < size {
                let take = (size - done).min(chunk as u64) as usize;
                replay(&romfs, offset + done, take, &mut destination, &mut tally);
                done += take as u64;
            }
            tally.report(&format!("stream {:.1} MiB file in {} KiB chunks ({pass})", size as f64 / (1 << 20) as f64, chunk >> 10));
        }
    }
    for size in [0x10_0000usize, 0x40_0000, 0x100_0000, 0x200_0000] {
        let mut tally = Tally::default();
        for _ in 0..8 {
            let offset = rng.next() % (romfs.len() - size as u64);
            replay(&romfs, offset, size, &mut destination, &mut tally);
        }
        tally.report(&format!("random {} KiB reads (first)", size >> 10));
        let mut warm = Tally::default();
        let offset = rng.next() % (romfs.len() - size as u64);
        replay(&romfs, offset, size, &mut destination, &mut Tally::default());
        for _ in 0..8 {
            replay(&romfs, offset, size, &mut destination, &mut warm);
        }
        warm.report(&format!("same {} KiB read repeated (warm)", size >> 10));
    }
    let mut fresh = Tally::default();
    for _ in 0..8 {
        let size = 0x200_0000;
        let started = Instant::now();
        let mut buffer = Vec::<u8>::with_capacity(size);
        buffer.resize(size, 0);
        std::hint::black_box(&buffer);
        fresh.read += started.elapsed();
        fresh.reads += 1;
        fresh.bytes += size as u64;
    }
    fresh.report("baseline: allocate+zero 32 MiB Vec");
}

/// Real-dump consistency: one large read, the same range split at random
/// points, and `read_into` over a dirty buffer must agree byte for byte.
#[test]
fn lm3_split_and_direct_reads_agree() {
    let Some(romfs) = open() else { return };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut dirty = vec![0u8; 0x60_0000];
    for _ in 0..48 {
        let size = (rng.next() % 0x50_0000) as usize + 1;
        let offset = rng.next() % (romfs.len() - size as u64 + 0x100);
        let whole = romfs.read(offset, size).unwrap();
        let mut pieces = Vec::with_capacity(size);
        while pieces.len() < whole.len() {
            let take = ((rng.next() % 0x3_0000) as usize + 1).min(whole.len() - pieces.len());
            pieces.extend(romfs.read(offset + pieces.len() as u64, take).unwrap());
        }
        assert_eq!(pieces, whole, "{offset:#x}+{size:#x}");
        dirty.fill(0xa5);
        let got = romfs.read_into(offset, &mut dirty[..size + 0x10]).unwrap();
        assert_eq!(got as u64, romfs.len().saturating_sub(offset).min(size as u64 + 0x10));
        assert_eq!(&dirty[..whole.len()], &whole[..]);
        assert!(dirty[got..size + 0x10].iter().all(|byte| *byte == 0xa5));
    }
}

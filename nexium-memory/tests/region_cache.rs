//! The per-thread region cache in `AddressSpace` must never serve a stale
//! region: every test here first warms a cache (on this thread and/or on a
//! long-lived helper thread with its own thread-local cache), then mutates the
//! space, then checks that `read` / `write` / `atomic_load_u32` /
//! `atomic_cas_u32` on every thread observe the mutation exactly as the
//! uncached (locked) path does.

use nexium_memory::{AddressSpace, AddressSpaceError, Perm};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

const PAGE: u64 = 0x1000;

/// A persistent helper thread, so its thread-local cache survives between
/// the commands we send it.
struct Helper {
    tx: mpsc::Sender<Box<dyn FnOnce(&AddressSpace) + Send>>,
    done: mpsc::Receiver<()>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Helper {
    fn new(space: Arc<AddressSpace>) -> Self {
        let (tx, rx) = mpsc::channel::<Box<dyn FnOnce(&AddressSpace) + Send>>();
        let (done_tx, done) = mpsc::channel();
        let handle = thread::spawn(move || {
            for job in rx {
                job(&space);
                done_tx.send(()).unwrap();
            }
        });
        Helper { tx, done, handle: Some(handle) }
    }

    /// Runs `f` on the helper thread and waits for it (a full happens-before
    /// edge in both directions, like a guest SVC boundary).
    fn run<R: Send + 'static>(&self, f: impl FnOnce(&AddressSpace) -> R + Send + 'static) -> R {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(Box::new(move |space| {
                rtx.send(f(space)).unwrap();
            }))
            .unwrap();
        self.done.recv().unwrap();
        rrx.recv().unwrap()
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        let (tx, _) = mpsc::channel();
        drop(std::mem::replace(&mut self.tx, tx));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn unmapped<T: std::fmt::Debug>(result: Result<T, AddressSpaceError>) -> bool {
    matches!(result, Err(AddressSpaceError::Unmapped { .. }))
}

fn load(space: &AddressSpace, va: u64) -> Result<u32, AddressSpaceError> {
    space.atomic_load_u32(va)
}

fn read_u32(space: &AddressSpace, va: u64) -> Result<u32, AddressSpaceError> {
    let mut b = [0u8; 4];
    space.read(va, &mut b).map(|()| u32::from_le_bytes(b))
}

/// Every access kind on an address that must now be unmapped.
fn assert_all_unmapped(space: &AddressSpace, va: u64) {
    assert!(unmapped(read_u32(space, va)), "read {va:#x}");
    assert!(unmapped(space.write(va, &[1, 2, 3, 4])), "write {va:#x}");
    assert!(unmapped(load(space, va)), "atomic_load {va:#x}");
    assert!(unmapped(space.atomic_cas_u32(va, 0, 1)), "atomic_cas {va:#x}");
}

#[test]
fn decommit_on_one_thread_invalidates_cached_region_on_another() {
    let space = Arc::new(AddressSpace::new());
    let base = 0x5100_0000u64;
    space.map_reserved(base, PAGE * 4, Perm::RW, "reserved").unwrap();
    space.resize_committed(base, PAGE * 4).unwrap();
    let high = base + PAGE * 3 + 0x20;
    let low = base + 0x20;

    let helper = Helper::new(Arc::clone(&space));
    // Warm the helper's cache with every access kind on the region.
    helper.run(move |s| {
        s.write(high, &7u32.to_le_bytes()).unwrap();
        assert_eq!(load(s, high).unwrap(), 7);
        assert!(s.atomic_cas_u32(high, 7, 8).unwrap());
        assert_eq!(read_u32(s, high).unwrap(), 8);
    });
    assert_eq!(load(&space, high).unwrap(), 8);

    // Decommit the top pages on this thread. resize_committed does not bump
    // the generation, so the cached entry must be rejected by the live
    // committed length.
    let generation = space.generation();
    space.resize_committed(base, PAGE).unwrap();
    assert_eq!(space.generation(), generation);
    assert_all_unmapped(&space, high);
    helper.run(move |s| assert_all_unmapped(s, high));
    // The committed part is still served (from the same cached region).
    helper.run(move |s| {
        assert!(s.atomic_cas_u32(low, 0, 0x11).unwrap());
        assert_eq!(read_u32(s, low).unwrap(), 0x11);
    });
    assert_eq!(load(&space, low).unwrap(), 0x11);

    // Recommit: the old contents are gone, and both threads agree.
    space.resize_committed(base, PAGE * 4).unwrap();
    helper.run(move |s| {
        assert_eq!(load(s, high).unwrap(), 0);
        assert!(s.atomic_cas_u32(high, 0, 0x22).unwrap());
    });
    assert_eq!(read_u32(&space, high).unwrap(), 0x22);
    assert_eq!(load(&space, high).unwrap(), 0x22);
}

#[test]
fn alias_remap_at_same_address_is_seen_by_every_thread() {
    let space = Arc::new(AddressSpace::new());
    let src_a = 0x5200_0000u64;
    let src_b = 0x5210_0000u64;
    let dst = 0x5220_0000u64;
    for (va, name) in [(src_a, "a"), (src_b, "b")] {
        space.map_reserved(va, PAGE * 2, Perm::RW, name).unwrap();
        space.resize_committed(va, PAGE * 2).unwrap();
    }
    space.write(src_a + 0x40, &0xAAAA_0001u32.to_le_bytes()).unwrap();
    space.write(src_b + 0x40, &0xBBBB_0001u32.to_le_bytes()).unwrap();
    space.map_alias(dst, src_a, PAGE * 2, Perm::RW, "alias").unwrap();

    let helper = Helper::new(Arc::clone(&space));
    for s in [&*space] {
        assert_eq!(load(s, dst + 0x40).unwrap(), 0xAAAA_0001);
    }
    helper.run(move |s| {
        // Warm both the source and the alias on the helper.
        assert_eq!(load(s, src_a + 0x40).unwrap(), 0xAAAA_0001);
        assert_eq!(load(s, dst + 0x40).unwrap(), 0xAAAA_0001);
        assert!(s.atomic_cas_u32(dst + 0x40, 0xAAAA_0001, 0xAAAA_0002).unwrap());
        // A CAS through the alias hits the source's memory, cached or not.
        assert_eq!(load(s, src_a + 0x40).unwrap(), 0xAAAA_0002);
    });

    // Remap the same destination onto a different backing.
    space.unmap_alias(dst, src_a, PAGE * 2).unwrap();
    helper.run(move |s| assert_all_unmapped(s, dst + 0x40));
    assert_all_unmapped(&space, dst + 0x40);
    space.map_alias(dst, src_b, PAGE * 2, Perm::RW, "alias").unwrap();

    helper.run(move |s| {
        assert_eq!(read_u32(s, dst + 0x40).unwrap(), 0xBBBB_0001);
        assert_eq!(load(s, dst + 0x40).unwrap(), 0xBBBB_0001);
        assert!(s.atomic_cas_u32(dst + 0x40, 0xBBBB_0001, 0xBBBB_0002).unwrap());
        s.write(dst + 0x44, &[9, 9, 9, 9]).unwrap();
    });
    assert_eq!(load(&space, src_b + 0x40).unwrap(), 0xBBBB_0002);
    assert_eq!(read_u32(&space, src_b + 0x44).unwrap(), 0x0909_0909);
    // The old source is untouched by the writes through the new alias.
    assert_eq!(load(&space, src_a + 0x40).unwrap(), 0xAAAA_0002);
    assert_eq!(read_u32(&space, src_a + 0x44).unwrap(), 0);

    // With the alias gone the source can shrink again: nothing cached keeps
    // it pinned (on either thread).
    space.unmap_alias(dst, src_b, PAGE * 2).unwrap();
    space.resize_committed(src_b, PAGE).unwrap();
    helper.run(move |s| assert_all_unmapped(s, src_b + PAGE));
}

#[test]
fn permission_change_and_new_mappings_do_not_disturb_cached_accesses() {
    let space = Arc::new(AddressSpace::new());
    let base = 0x5300_0000u64;
    space.map(base, PAGE, Perm::RW, "rw").unwrap();
    let helper = Helper::new(Arc::clone(&space));
    helper.run(move |s| {
        assert!(s.atomic_cas_u32(base, 0, 1).unwrap());
    });
    let generation = space.generation();
    // Unchecked accessors ignore permissions, cached or not (as before the
    // cache); a perm change must still invalidate other threads' entries.
    space.protect(base, PAGE, Perm::R).unwrap();
    assert!(space.generation() > generation);
    helper.run(move |s| {
        assert!(s.atomic_cas_u32(base, 1, 2).unwrap());
        assert_eq!(read_u32(s, base).unwrap(), 2);
        // A checked write sees the new permission.
        assert!(matches!(
            s.write_checked(base, &[0]),
            Err(AddressSpaceError::PermissionDenied { .. })
        ));
    });
    // Mapping right next to the cached region: the cached entry must not be
    // stretched over it, and the new region must be reachable at once.
    space.map(base + PAGE, PAGE, Perm::RW, "next").unwrap();
    helper.run(move |s| {
        assert_eq!(load(s, base + PAGE).unwrap(), 0);
        assert!(s.atomic_cas_u32(base + PAGE, 0, 5).unwrap());
        // An access spanning both regions takes the multi-region path.
        let mut both = [0u8; 8];
        s.read(base + PAGE - 4, &mut both).unwrap();
        assert_eq!(both, [0, 0, 0, 0, 5, 0, 0, 0]);
    });
    assert_eq!(read_u32(&space, base + PAGE).unwrap(), 5);
}

#[test]
fn cached_and_uncached_cas_hit_the_same_word_under_contention() {
    // Two threads hammer one word with CAS increments. One keeps its cache
    // warm (many regions are touched in between on the other, which keeps
    // evicting its entries and going through the slow path). Every increment
    // must land exactly once.
    const PER_THREAD: u32 = 20_000;
    let space = Arc::new(AddressSpace::new());
    let base = 0x5400_0000u64;
    for i in 0..8u64 {
        space.map(base + i * 0x10_0000, PAGE, Perm::RW, format!("r{i}")).unwrap();
    }
    let word = base + 0x80;
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let workers: Vec<_> = (0..2)
        .map(|worker| {
            let space = Arc::clone(&space);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut done = 0;
                let mut spin = 0u64;
                while done < PER_THREAD {
                    if worker == 1 {
                        // Evict: touch more regions than the cache holds.
                        for i in 1..8u64 {
                            let _ = load(&space, base + i * 0x10_0000 + 4 * (spin % 4));
                        }
                    }
                    spin += 1;
                    let cur = load(&space, word).unwrap();
                    if space.atomic_cas_u32(word, cur, cur + 1).unwrap() {
                        done += 1;
                    }
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(load(&space, word).unwrap(), 2 * PER_THREAD);
    assert_eq!(read_u32(&space, word).unwrap(), 2 * PER_THREAD);
}

#[test]
fn concurrent_decommit_never_lets_a_cached_access_succeed_afterwards() {
    // A mutator thread flips a region's committed length while a reader on
    // another thread keeps a warm cache. After every flip (synchronised by a
    // channel round-trip) the reader must observe exactly the new state.
    let space = Arc::new(AddressSpace::new());
    let base = 0x5500_0000u64;
    space.map_reserved(base, PAGE * 2, Perm::RW, "flip").unwrap();
    space.resize_committed(base, PAGE * 2).unwrap();
    let high = base + PAGE + 8;
    let helper = Helper::new(Arc::clone(&space));
    for round in 0..200u32 {
        let committed = round % 2 == 0;
        space
            .resize_committed(base, if committed { PAGE * 2 } else { PAGE })
            .unwrap();
        helper.run(move |s| {
            // Keep the region in the cache through the low page.
            s.write(base, &round.to_le_bytes()).unwrap();
            if committed {
                assert_eq!(load(s, high).unwrap(), 0, "round {round}");
                assert!(s.atomic_cas_u32(high, 0, round + 1).unwrap());
                assert_eq!(read_u32(s, high).unwrap(), round + 1);
            } else {
                assert_all_unmapped(s, high);
            }
            assert_eq!(load(s, base).unwrap(), round);
        });
        if committed {
            assert_eq!(load(&space, high).unwrap(), round + 1);
        } else {
            assert_all_unmapped(&space, high);
        }
    }
}

#[test]
fn separate_spaces_never_share_cache_entries() {
    // Same guest address in two spaces with different contents: the cache is
    // keyed by space, so alternating accesses never cross over. (Outside the
    // fastmem arena window: inside it, both spaces would share host pages.)
    let a = AddressSpace::new();
    let b = AddressSpace::new();
    let va = 1u64 << 43;
    a.map(va, PAGE, Perm::RW, "a").unwrap();
    b.map(va, PAGE, Perm::RW, "b").unwrap();
    a.write(va, &1u32.to_le_bytes()).unwrap();
    b.write(va, &2u32.to_le_bytes()).unwrap();
    for _ in 0..4 {
        assert_eq!(load(&a, va).unwrap(), 1);
        assert_eq!(load(&b, va).unwrap(), 2);
        assert!(a.atomic_cas_u32(va, 1, 1).unwrap());
        assert!(!b.atomic_cas_u32(va, 1, 3).unwrap());
    }
    drop(a);
    let c = AddressSpace::new();
    assert!(unmapped(load(&c, va)));
    c.map(va, PAGE, Perm::RW, "c").unwrap();
    assert_eq!(load(&c, va).unwrap(), 0);
    assert_eq!(load(&b, va).unwrap(), 2);
}

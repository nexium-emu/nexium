//! Linux write-watch backend: userfaultfd asynchronous write-protect plus the
//! `PAGEMAP_SCAN` ioctl (Linux 6.7+).
//!
//! A range is registered once with `UFFDIO_REGISTER_MODE_WP` on a userfaultfd
//! opened with `UFFD_FEATURE_WP_ASYNC`. From then on the kernel keeps one
//! "written since last reset" bit per 4K page and resolves the write-protect
//! faults itself: no fault-handler thread and no SIGSEGV, so it cannot collide
//! with the JIT's fastmem signal handler. `PAGEMAP_SCAN` with
//! `PM_SCAN_WP_MATCHING` reads the written pages of a range and re-protects
//! exactly those pages in one step under the page-table lock, which is the
//! query-and-reset contract of `GetWriteWatch(WRITE_WATCH_FLAG_RESET)`.
//!
//! Pages the kernel has no write-protect marker for (never scanned, freshly
//! mapped, or discarded) report as written, so every unknown state reads as
//! dirty rather than clean.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::OnceLock;

const PAGE: usize = 4096;

const UFFD_USER_MODE_ONLY: libc::c_int = 1;
const UFFD_API: u64 = 0xAA;
const UFFD_FEATURE_WP_UNPOPULATED: u64 = 1 << 13;
const UFFD_FEATURE_WP_ASYNC: u64 = 1 << 15;
const REQUIRED_FEATURES: u64 = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED;
const UFFDIO_API: u64 = 0xc018_aa3f;
const UFFDIO_REGISTER: u64 = 0xc020_aa00;
const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;

const PAGEMAP_SCAN: u64 = 0xc060_6610;
const PM_SCAN_WP_MATCHING: u64 = 1 << 0;
const PM_SCAN_CHECK_WPASYNC: u64 = 1 << 1;
const PAGE_IS_WPALLOWED: u64 = 1 << 0;
const PAGE_IS_WRITTEN: u64 = 1 << 1;
const PAGE_IS_PRESENT: u64 = 1 << 3;
const PAGE_IS_SWAPPED: u64 = 1 << 4;

const SCAN_REGIONS: usize = 128;

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
struct UffdioRegister {
    start: u64,
    len: u64,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PageRegion {
    start: u64,
    end: u64,
    categories: u64,
}

#[repr(C)]
struct PmScanArg {
    size: u64,
    flags: u64,
    start: u64,
    end: u64,
    walk_end: u64,
    vec: u64,
    vec_len: u64,
    max_pages: u64,
    category_inverted: u64,
    category_mask: u64,
    category_anyof_mask: u64,
    return_mask: u64,
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::EIO)
}

pub struct Watch {
    uffd: OwnedFd,
    pagemap: OwnedFd,
}

impl Watch {
    /// Opens the backend and proves on a scratch mapping that it really
    /// tracks writes. Any failure means "write-watch unavailable".
    pub fn open() -> Result<Self, String> {
        let watch = Self::open_with_features(REQUIRED_FEATURES)?;
        watch.self_test()?;
        Ok(watch)
    }

    fn open_with_features(features: u64) -> Result<Self, String> {
        if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } != PAGE as libc::c_long {
            return Err("host page size is not 4096".into());
        }
        // UFFD_USER_MODE_ONLY is what lets an unprivileged process open a
        // userfaultfd when vm.unprivileged_userfaultfd=0. Async write-protect
        // never delivers faults to userspace, so kernel-mode writes (syscalls
        // writing into the range) are still tracked.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_userfaultfd,
                libc::O_CLOEXEC | libc::O_NONBLOCK | UFFD_USER_MODE_ONLY,
            )
        };
        if fd < 0 {
            return Err(format!("userfaultfd(): errno {}", errno()));
        }
        let uffd = unsafe { OwnedFd::from_raw_fd(fd as i32) };
        let mut api = UffdioApi { api: UFFD_API, features, ioctls: 0 };
        if unsafe { libc::ioctl(uffd.as_raw_fd(), UFFDIO_API as _, &mut api) } != 0 {
            return Err(format!("UFFDIO_API(features={features:#x}): errno {}", errno()));
        }
        if api.features & features != features {
            return Err(format!("kernel lacks uffd features (have {:#x})", api.features));
        }
        let fd = unsafe {
            libc::open(c"/proc/self/pagemap".as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC)
        };
        if fd < 0 {
            return Err(format!("open /proc/self/pagemap: errno {}", errno()));
        }
        let pagemap = unsafe { OwnedFd::from_raw_fd(fd) };
        Ok(Self { uffd, pagemap })
    }

    fn self_test(&self) -> Result<(), String> {
        const LEN: usize = 4 * PAGE;
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(format!("self-test mmap: errno {}", errno()));
        }
        let base = map as usize;
        let result = (|| {
            if !self.register(base, LEN) {
                return Err(format!("self-test UFFDIO_REGISTER: errno {}", errno()));
            }
            let fail = |step: &str, e: i32| format!("self-test {step}: errno {e}");
            let mut fresh = Vec::new();
            self.take(base, LEN, |lo, hi| fresh.push((lo, hi)))
                .map_err(|e| fail("first scan", e))?;
            if fresh != [(base, base + LEN)] {
                return Err(format!("self-test: fresh pages not all dirty {fresh:x?} (base {base:#x})"));
            }
            if self.take(base, LEN, |_, _| {}).map_err(|e| fail("second scan", e))? {
                return Err("self-test: pages still written after reset".into());
            }
            unsafe { (map as *mut u8).add(2 * PAGE + 5).write_volatile(1) };
            let _ = unsafe { (map as *const u8).add(PAGE).read_volatile() };
            let mut seen = Vec::new();
            self.take(base, LEN, |lo, hi| seen.push((lo, hi)))
                .map_err(|e| fail("third scan", e))?;
            if seen != [(base + 2 * PAGE, base + 3 * PAGE)] {
                return Err(format!("self-test: wrong dirty set {seen:x?} (base {base:#x})"));
            }
            Ok(())
        })();
        unsafe { libc::munmap(map, LEN) };
        result
    }

    /// Enables write tracking on a page-aligned range. Must be repeated after
    /// the range is replaced by a new mapping (mmap MAP_FIXED drops it).
    pub fn register(&self, start: usize, len: usize) -> bool {
        let mut register = UffdioRegister {
            start: start as u64,
            len: len as u64,
            mode: UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        unsafe { libc::ioctl(self.uffd.as_raw_fd(), UFFDIO_REGISTER as _, &mut register) == 0 }
    }

    /// Reports every run of pages in the page-aligned range written since the
    /// previous take (ascending, coalesced), and resets them. `Ok(true)` if
    /// anything was reported.
    ///
    /// Two scans, both of which write-protect exactly what they report:
    ///
    /// 1. Written pages: present or swapped entries without the uffd-wp bit,
    ///    plus empty PTE slots inside an existing page table (the kernel's
    ///    exact-`PAGE_IS_WRITTEN` fast path counts those as written and
    ///    installs a write-protect marker on them).
    /// 2. Holes: ranges with no page table at all (never touched, or freshly
    ///    mapped). The written-page scan skips those without protecting
    ///    them, so on their own they would read clean on every query even
    ///    though nothing is tracking them yet. They are reported as dirty and
    ///    marked (`UFFD_FEATURE_WP_UNPOPULATED`), so from then on they are
    ///    tracked like any other page.
    ///
    /// A write racing with a scan is never lost: either it lands before the
    /// page is protected, and the page is reported by this take, or after,
    /// and it clears the protection again for the next take.
    ///
    /// On `Err` the tracking state of the range is unknown (pages may have
    /// been reset without being reported), so the caller must treat the
    /// whole range as written.
    pub fn take(
        &self,
        start: usize,
        len: usize,
        mut dirty: impl FnMut(usize, usize),
    ) -> Result<bool, i32> {
        thread_local! {
            static RUNS: std::cell::Cell<Vec<(usize, usize)>> = const {
                std::cell::Cell::new(Vec::new())
            };
        }
        let end = start.checked_add(len).ok_or(libc::EINVAL)?;
        let mut runs = RUNS.take();
        runs.clear();
        // Holes first: once that pass has marked them, every slot in the range
        // is tracked, so a write landing between the two passes shows up as a
        // present page without the uffd-wp bit and the written pass reports
        // it. The other order can skip an untracked range in the written pass
        // and then find it populated (no longer a hole) in the hole pass.
        let scanned = self
            .scan(start, end, ScanFilter::Hole, &mut runs)
            .and_then(|()| {
                let holes = runs.len();
                self.scan(start, end, ScanFilter::Written, &mut runs)?;
                if runs.len() > holes && holes != 0 {
                    runs.sort_unstable();
                }
                Ok(())
            });
        let result = scanned.map(|()| {
            let mut pending: Option<(usize, usize)> = None;
            for &(lo, hi) in runs.iter() {
                match &mut pending {
                    Some((_, last_hi)) if lo <= *last_hi => *last_hi = (*last_hi).max(hi),
                    _ => {
                        if let Some((a, b)) = pending.replace((lo, hi)) {
                            dirty(a, b);
                        }
                    }
                }
            }
            if let Some((a, b)) = pending {
                dirty(a, b);
                true
            } else {
                false
            }
        });
        RUNS.set(runs);
        result
    }

    fn scan(
        &self,
        start: usize,
        end: usize,
        filter: ScanFilter,
        out: &mut Vec<(usize, usize)>,
    ) -> Result<(), i32> {
        let (inverted, mask, return_mask) = match filter {
            // Exactly this mask selects the kernel's write-tracking fast path.
            ScanFilter::Written => (0, PAGE_IS_WRITTEN, PAGE_IS_WRITTEN),
            // Neither present nor swapped (a uffd-wp marker counts as a swap
            // entry): an empty slot. Every page of a registered VMA carries
            // PAGE_IS_WPALLOWED, so that is what holes are returned with.
            ScanFilter::Hole => (
                PAGE_IS_PRESENT | PAGE_IS_SWAPPED,
                PAGE_IS_PRESENT | PAGE_IS_SWAPPED,
                PAGE_IS_WPALLOWED,
            ),
        };
        let mut regions = [PageRegion::default(); SCAN_REGIONS];
        let mut at = start;
        while at < end {
            let mut arg = PmScanArg {
                size: std::mem::size_of::<PmScanArg>() as u64,
                flags: PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC,
                start: at as u64,
                end: end as u64,
                walk_end: 0,
                vec: regions.as_mut_ptr() as u64,
                vec_len: SCAN_REGIONS as u64,
                max_pages: 0,
                category_inverted: inverted,
                category_mask: mask,
                category_anyof_mask: 0,
                return_mask,
            };
            let count = unsafe { libc::ioctl(self.pagemap.as_raw_fd(), PAGEMAP_SCAN as _, &mut arg) };
            if count < 0 {
                return Err(errno());
            }
            let count = count as usize;
            if count > SCAN_REGIONS {
                return Err(libc::EIO);
            }
            for region in &regions[..count] {
                let lo = (region.start as usize).max(at);
                let hi = (region.end as usize).min(end);
                if lo < hi {
                    match out.last_mut() {
                        Some((_, last_hi)) if *last_hi == lo => *last_hi = hi,
                        _ => out.push((lo, hi)),
                    }
                }
            }
            let walk_end = arg.walk_end as usize;
            if walk_end <= at || walk_end > end {
                return Err(libc::EIO);
            }
            at = walk_end;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ScanFilter {
    Written,
    Hole,
}

pub fn disabled_by_env() -> bool {
    std::env::var("NEXIUM_NO_WRITE_WATCH")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// The process-wide backend, or `None` when the kernel, sysctl policy, a
/// seccomp filter or `NEXIUM_NO_WRITE_WATCH` rules it out.
pub fn global() -> Option<&'static Watch> {
    static WATCH: OnceLock<Option<Watch>> = OnceLock::new();
    WATCH
        .get_or_init(|| {
            if disabled_by_env() {
                log::info!("write-watch: disabled by NEXIUM_NO_WRITE_WATCH");
                return None;
            }
            match Watch::open() {
                Ok(watch) => Some(watch),
                Err(reason) => {
                    log::warn!(
                        "write-watch: userfaultfd async write-protect unavailable ({reason}); \
                         GPU input caches will revalidate by copying"
                    );
                    None
                }
            }
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Map(usize, usize);

    impl Map {
        fn new(pages: usize) -> Self {
            let len = pages * PAGE;
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert_ne!(p, libc::MAP_FAILED);
            Self(p as usize, len)
        }

        fn write(&self, offset: usize) {
            unsafe { ((self.0 + offset) as *mut u8).write_volatile(0x5a) };
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.0 as *mut libc::c_void, self.1) };
        }
    }

    /// The backend under test. A machine that cannot run it fails loudly
    /// unless the skip is requested explicitly.
    fn watch() -> Option<Watch> {
        match Watch::open() {
            Ok(watch) => Some(watch),
            Err(reason) => {
                if std::env::var_os("NEXIUM_TEST_ALLOW_NO_WRITE_WATCH").is_some() {
                    eprintln!("SKIPPED: Linux write-watch backend unavailable: {reason}");
                    None
                } else {
                    panic!(
                        "Linux write-watch backend unavailable: {reason}. Set \
                         NEXIUM_TEST_ALLOW_NO_WRITE_WATCH=1 to skip the tests that need it."
                    );
                }
            }
        }
    }

    fn runs(watch: &Watch, map: &Map) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        watch
            .take(map.0, map.1, |lo, hi| out.push(((lo - map.0) / PAGE, (hi - map.0) / PAGE)))
            .expect("scan");
        out
    }

    #[test]
    fn fresh_pages_read_dirty_then_clean_and_writes_are_page_exact() {
        let Some(watch) = watch() else { return };
        let map = Map::new(16);
        assert!(watch.register(map.0, map.1));
        assert_eq!(runs(&watch, &map), [(0, 16)]);
        assert_eq!(runs(&watch, &map), []);
        map.write(3 * PAGE);
        map.write(4 * PAGE + 4095);
        map.write(9 * PAGE + 1);
        let _ = unsafe { ((map.0 + 12 * PAGE) as *const u8).read_volatile() };
        assert_eq!(runs(&watch, &map), [(3, 5), (9, 10)]);
        assert_eq!(runs(&watch, &map), []);
        map.write(3 * PAGE + 1);
        assert_eq!(runs(&watch, &map), [(3, 4)]);
    }

    /// Regression: the written-page scan leaves ranges without any page table
    /// untouched and unreported. Those must read dirty once, and then be
    /// tracked page-exactly, whether they sit next to populated tables or not.
    #[test]
    fn pure_holes_without_page_tables_read_dirty_once_then_track_exactly() {
        const PMD: usize = 2 << 20;
        let Some(watch) = watch() else { return };
        let map = Map::new(4 * PMD / PAGE);
        let pmd0 = map.0.next_multiple_of(PMD);
        let pmd1 = pmd0 + PMD;
        let pages = |lo: usize, hi: usize| ((lo - map.0) / PAGE, (hi - map.0) / PAGE);
        assert!(watch.register(map.0, map.1));
        // Populate one page of the first table only; the second stays a hole.
        map.write(pmd0 - map.0 + 7 * PAGE);
        let mut seen = Vec::new();
        assert_eq!(watch.take(pmd0, 2 * PMD, |lo, hi| seen.push(pages(lo, hi))), Ok(true));
        let first = (pmd0 - map.0) / PAGE;
        assert_eq!(seen, [(first, first + 2 * PMD / PAGE)]);
        seen.clear();
        assert_eq!(watch.take(pmd0, 2 * PMD, |lo, hi| seen.push(pages(lo, hi))), Ok(false));
        map.write(pmd1 - map.0 + 3 * PAGE + 1);
        map.write(pmd0 - map.0 + 9 * PAGE);
        let _ = unsafe { ((pmd1 + 100 * PAGE) as *const u8).read_volatile() };
        assert_eq!(watch.take(pmd0, 2 * PMD, |lo, hi| seen.push(pages(lo, hi))), Ok(true));
        let second = (pmd1 - map.0) / PAGE;
        assert_eq!(seen, [(first + 9, first + 10), (second + 3, second + 4)]);
        // A hole query that only partly overlaps a populated table.
        seen.clear();
        let tail = pmd1 + PMD;
        assert_eq!(watch.take(pmd1 + PMD - 2 * PAGE, 4 * PAGE, |lo, hi| seen.push(pages(lo, hi))), Ok(true));
        let tail_page = (tail - map.0) / PAGE;
        assert_eq!(seen, [(tail_page, tail_page + 2)]);
    }

    #[test]
    fn sub_range_take_leaves_the_rest_pending() {
        let Some(watch) = watch() else { return };
        let map = Map::new(8);
        assert!(watch.register(map.0, map.1));
        runs(&watch, &map);
        map.write(PAGE);
        map.write(6 * PAGE);
        let mut seen = Vec::new();
        assert_eq!(watch.take(map.0, 4 * PAGE, |lo, hi| seen.push((lo, hi))), Ok(true));
        assert_eq!(seen, [(map.0 + PAGE, map.0 + 2 * PAGE)]);
        assert_eq!(runs(&watch, &map), [(6, 7)]);
    }

    #[test]
    fn more_dirty_runs_than_one_scan_buffer_are_all_reported() {
        let Some(watch) = watch() else { return };
        let pages = SCAN_REGIONS * 6 + 2;
        let map = Map::new(pages);
        assert!(watch.register(map.0, map.1));
        runs(&watch, &map);
        let expected: Vec<_> = (0..pages).step_by(2).map(|p| (p, p + 1)).collect();
        for &(page, _) in &expected {
            map.write(page * PAGE);
        }
        assert_eq!(runs(&watch, &map), expected);
        assert_eq!(runs(&watch, &map), []);
    }

    #[test]
    fn kernel_mode_writes_are_tracked() {
        let Some(watch) = watch() else { return };
        let map = Map::new(4);
        assert!(watch.register(map.0, map.1));
        runs(&watch, &map);
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        assert_eq!(unsafe { libc::write(fds[1], b"abcd".as_ptr().cast(), 4) }, 4);
        let dst = (map.0 + 2 * PAGE + 8) as *mut libc::c_void;
        assert_eq!(unsafe { libc::read(fds[0], dst, 4) }, 4);
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
        assert_eq!(runs(&watch, &map), [(2, 3)]);
        assert_eq!(unsafe { ((map.0 + 2 * PAGE + 8) as *const u8).read() }, b'a');
    }

    #[test]
    fn protection_changes_and_discard_never_hide_writes() {
        let Some(watch) = watch() else { return };
        let map = Map::new(4);
        assert!(watch.register(map.0, map.1));
        runs(&watch, &map);
        let ptr = map.0 as *mut libc::c_void;
        assert_eq!(unsafe { libc::mprotect(ptr, map.1, libc::PROT_READ) }, 0);
        assert_eq!(unsafe { libc::mprotect(ptr, map.1, libc::PROT_READ | libc::PROT_WRITE) }, 0);
        assert_eq!(runs(&watch, &map), []);
        map.write(PAGE);
        assert_eq!(runs(&watch, &map), [(1, 2)]);
        map.write(0);
        assert_eq!(unsafe { libc::madvise(ptr, map.1, libc::MADV_DONTNEED) }, 0);
        // Discarded pages may read dirty (they do: the markers go with the
        // pages) but a later write must never be lost.
        let after_discard = runs(&watch, &map);
        assert!(after_discard.iter().any(|&(lo, hi)| lo == 0 && hi >= 1), "{after_discard:?}");
        map.write(3 * PAGE);
        assert!(runs(&watch, &map).iter().any(|&(lo, hi)| lo <= 3 && 3 < hi));
    }

    #[test]
    fn unregistered_or_replaced_ranges_are_an_error_not_clean() {
        let Some(watch) = watch() else { return };
        let plain = Map::new(2);
        assert_eq!(watch.take(plain.0, plain.1, |_, _| {}), Err(libc::EPERM));

        let map = Map::new(4);
        assert!(watch.register(map.0, map.1));
        runs(&watch, &map);
        let replaced = unsafe {
            libc::mmap(
                (map.0 + PAGE) as *mut libc::c_void,
                PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        assert_eq!(replaced as usize, map.0 + PAGE);
        assert_eq!(watch.take(map.0, map.1, |_, _| {}), Err(libc::EPERM));
        assert!(watch.register(map.0 + PAGE, PAGE));
        assert_eq!(runs(&watch, &map), [(1, 2)]);
    }

    #[test]
    fn probe_rejects_a_kernel_without_the_feature() {
        // Bit 62 is not a userfaultfd feature on any kernel: UFFDIO_API must
        // refuse it, which is exactly how a pre-6.7 kernel refuses WP_ASYNC.
        let Err(reason) = Watch::open_with_features(REQUIRED_FEATURES | 1 << 62) else {
            panic!("probe accepted an unknown userfaultfd feature");
        };
        eprintln!("probe failure path: {reason}");
        if let Some(watch) = watch() {
            assert!(!watch.register(0x1000, PAGE), "registering an unmapped range must fail");
        }
    }
}

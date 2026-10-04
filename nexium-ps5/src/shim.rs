use std::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    fn pthread_set_name_np(thread: usize, name: *const c_char);
}

#[no_mangle]
pub unsafe extern "C" fn pthread_setname_np(thread: usize, name: *const c_char) -> c_int {
    unsafe { pthread_set_name_np(thread, name) };
    #[cfg(feature = "title")]
    crate::sampler::note_name(thread, name);
    #[cfg(feature = "title")]
    crate::affinity::pin(thread, name);
    0
}

#[cfg(not(feature = "title"))]
#[repr(C)]
pub struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

#[cfg(not(feature = "title"))]
unsafe extern "C" {
    fn nanosleep(request: *const Timespec, remaining: *mut Timespec) -> c_int;
    fn clock_gettime(clock: c_int, now: *mut Timespec) -> c_int;
    fn __error() -> *mut c_int;
}

#[cfg(not(feature = "title"))]
#[no_mangle]
pub unsafe extern "C" fn clock_nanosleep(
    clock: c_int,
    flags: c_int,
    request: *const Timespec,
    remaining: *mut Timespec,
) -> c_int {
    const TIMER_ABSTIME: c_int = 1;
    if request.is_null() {
        return 14;
    }
    let mut wait = Timespec { tv_sec: unsafe { (*request).tv_sec }, tv_nsec: unsafe { (*request).tv_nsec } };
    let mut remaining = remaining;
    if flags & TIMER_ABSTIME != 0 {
        let mut now = Timespec { tv_sec: 0, tv_nsec: 0 };
        if unsafe { clock_gettime(clock, &mut now) } != 0 {
            return unsafe { *__error() };
        }
        let left = (wait.tv_sec - now.tv_sec) * 1_000_000_000 + (wait.tv_nsec - now.tv_nsec);
        if left <= 0 {
            return 0;
        }
        wait = Timespec { tv_sec: left / 1_000_000_000, tv_nsec: left % 1_000_000_000 };
        remaining = std::ptr::null_mut();
    }
    if unsafe { nanosleep(&wait, remaining) } == 0 { 0 } else { unsafe { *__error() } }
}

unsafe extern "C" {
    fn arc4random_buf(buf: *mut c_void, len: usize);
    fn accept(fd: c_int, addr: *mut c_void, len: *mut u32) -> c_int;
    fn pipe(fds: *mut c_int) -> c_int;
    fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    #[link_name = "__error"]
    fn errno_location() -> *mut c_int;
}

const F_GETFL: c_int = 3;
const F_SETFL: c_int = 4;
const F_SETFD: c_int = 2;
const FD_CLOEXEC: c_int = 1;
const O_NONBLOCK: c_int = 4;
const O_CLOEXEC: c_int = 0x0010_0000;
const SOCK_CLOEXEC: c_int = 0x1000_0000;
const SOCK_NONBLOCK: c_int = 0x2000_0000;
const EPERM: c_int = 1;

unsafe fn apply_fd_flags(fd: c_int, cloexec: bool, nonblock: bool) {
    unsafe {
        if cloexec {
            fcntl(fd, F_SETFD, FD_CLOEXEC);
        }
        if nonblock {
            let flags = fcntl(fd, F_GETFL);
            if flags >= 0 {
                fcntl(fd, F_SETFL, flags | O_NONBLOCK);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn getrandom(buf: *mut c_void, len: usize, _flags: u32) -> isize {
    unsafe { arc4random_buf(buf, len) };
    len as isize
}

#[no_mangle]
pub unsafe extern "C" fn accept4(fd: c_int, addr: *mut c_void, len: *mut u32, flags: c_int) -> c_int {
    unsafe {
        let new = accept(fd, addr, len);
        if new >= 0 {
            apply_fd_flags(new, flags & SOCK_CLOEXEC != 0, flags & SOCK_NONBLOCK != 0);
        }
        new
    }
}

#[no_mangle]
pub unsafe extern "C" fn pipe2(fds: *mut c_int, flags: c_int) -> c_int {
    unsafe {
        let rc = pipe(fds);
        if rc == 0 {
            for i in 0..2 {
                apply_fd_flags(*fds.add(i), flags & O_CLOEXEC != 0, flags & O_NONBLOCK != 0);
            }
        }
        rc
    }
}

#[no_mangle]
pub unsafe extern "C" fn posix_spawn_file_actions_addchdir_np(_actions: *mut c_void, _path: *const c_char) -> c_int {
    78
}

#[no_mangle]
pub unsafe extern "C" fn setgid(_gid: u32) -> c_int {
    unsafe { *errno_location() = EPERM };
    -1
}

#[no_mangle]
pub unsafe extern "C" fn dl_iterate_phdr(
    _callback: Option<unsafe extern "C" fn(*mut c_void, usize, *mut c_void) -> c_int>,
    _data: *mut c_void,
) -> c_int {
    0
}

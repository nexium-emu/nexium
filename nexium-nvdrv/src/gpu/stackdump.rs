#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::sync::Mutex;

    type Handle = *mut c_void;

    #[repr(C)]
    struct ThreadEntry32 {
        size: u32,
        usage: u32,
        thread_id: u32,
        owner_process_id: u32,
        base_pri: i32,
        delta_pri: i32,
        flags: u32,
    }

    #[repr(C, align(16))]
    struct Context {
        bytes: [u8; 1232],
    }

    #[link(name = "user32")]
    extern "system" {
        fn EnumWindows(callback: extern "system" fn(Handle, isize) -> i32, lparam: isize) -> i32;
        fn GetWindowThreadProcessId(window: Handle, process_id: *mut u32) -> u32;
        fn IsWindowVisible(window: Handle) -> i32;
        fn IsIconic(window: Handle) -> i32;
        fn IsZoomed(window: Handle) -> i32;
        fn GetForegroundWindow() -> Handle;
        fn GetWindowRect(window: Handle, rect: *mut [i32; 4]) -> i32;
        fn GetWindowLongPtrW(window: Handle, index: i32) -> isize;
        fn GetWindow(window: Handle, cmd: u32) -> Handle;
    }

    extern "system" fn collect_window(window: Handle, lparam: isize) -> i32 {
        let out = unsafe { &mut *(lparam as *mut Vec<(Handle, u32)>) };
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(window, &mut pid) };
        out.push((window, pid));
        1
    }

    pub fn dump_process_windows() {
        let mut windows: Vec<(Handle, u32)> = Vec::new();
        unsafe {
            EnumWindows(
                collect_window,
                &mut windows as *mut Vec<(Handle, u32)> as isize,
            )
        };
        let foreground = unsafe { GetForegroundWindow() };
        let own_pid = unsafe { GetCurrentProcessId() };
        log::error!(
            "[window-dump] top_level_windows={} own_pid={} foreground={:#x}",
            windows.len(),
            own_pid,
            foreground as usize
        );
        for (window, pid) in windows {
            if pid != own_pid {
                continue;
            }
            let mut rect = [0i32; 4];
            let (visible, iconic, zoomed, style, ex_style, owner) = unsafe {
                GetWindowRect(window, &mut rect);
                (
                    IsWindowVisible(window),
                    IsIconic(window),
                    IsZoomed(window),
                    GetWindowLongPtrW(window, -16),
                    GetWindowLongPtrW(window, -20),
                    GetWindow(window, 4),
                )
            };
            log::error!(
                "[window-dump] pid={} hwnd={:#x} visible={} iconic={} zoomed={} foreground={} owner={:#x} rect=[{},{},{},{}] style={:#x} ex_style={:#x}",
                pid,
                window as usize,
                visible,
                iconic,
                zoomed,
                window == foreground,
                owner as usize,
                rect[0],
                rect[1],
                rect[2],
                rect[3],
                style as usize,
                ex_style as usize,
            );
        }
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> Handle;
        fn Thread32First(snapshot: Handle, entry: *mut ThreadEntry32) -> i32;
        fn Thread32Next(snapshot: Handle, entry: *mut ThreadEntry32) -> i32;
        fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> Handle;
        fn SuspendThread(thread: Handle) -> u32;
        fn ResumeThread(thread: Handle) -> u32;
        fn GetThreadContext(thread: Handle, context: *mut Context) -> i32;
        fn GetThreadDescription(thread: Handle, description: *mut *mut u16) -> i32;
        fn LocalFree(memory: *mut c_void) -> *mut c_void;
        fn CloseHandle(handle: Handle) -> i32;
        fn GetCurrentProcess() -> Handle;
        fn GetCurrentProcessId() -> u32;
        fn GetCurrentThreadId() -> u32;
    }

    #[link(name = "dbghelp")]
    extern "system" {
        fn SymInitialize(process: Handle, search_path: *const u8, invade: i32) -> i32;
        fn SymSetOptions(options: u32) -> u32;
        fn SymFunctionTableAccess64(process: Handle, addr_base: u64) -> *mut c_void;
        fn SymGetModuleBase64(process: Handle, address: u64) -> u64;
        fn SymFromAddr(
            process: Handle,
            address: u64,
            displacement: *mut u64,
            symbol: *mut u8,
        ) -> i32;
        fn StackWalk64(
            machine_type: u32,
            process: Handle,
            thread: Handle,
            stack_frame: *mut u64,
            context: *mut Context,
            read_memory: *const c_void,
            function_table_access: *const c_void,
            get_module_base: *const c_void,
            translate_address: *const c_void,
        ) -> i32;
    }

    const TH32CS_SNAPTHREAD: u32 = 0x4;
    const THREAD_ACCESS: u32 = 0x2 | 0x8 | 0x40 | 0x800;
    const CONTEXT_FULL: u32 = 0x0010_000B;
    const IMAGE_FILE_MACHINE_AMD64: u32 = 0x8664;
    const ADDR_MODE_FLAT: u32 = 3;
    const SYMOPT_UNDNAME: u32 = 0x2;
    const SYMOPT_DEFERRED_LOADS: u32 = 0x4;
    const MAX_FRAMES: usize = 48;
    const SYMBOL_HEADER: usize = 88;
    const SYMBOL_NAME_CAPACITY: usize = 1024;

    static DUMP_LOCK: Mutex<bool> = Mutex::new(false);

    fn ensure_symbols(process: Handle) -> bool {
        let mut initialized = DUMP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if *initialized {
            return true;
        }
        unsafe {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
            if SymInitialize(process, std::ptr::null(), 1) == 0 {
                return false;
            }
        }
        *initialized = true;
        true
    }

    fn symbolize(process: Handle, address: u64) -> String {
        let mut buffer = vec![0u8; SYMBOL_HEADER + SYMBOL_NAME_CAPACITY];
        buffer[0..4].copy_from_slice(&(SYMBOL_HEADER as u32).to_le_bytes());
        buffer[80..84].copy_from_slice(&(SYMBOL_NAME_CAPACITY as u32).to_le_bytes());
        let mut displacement = 0u64;
        let ok = unsafe { SymFromAddr(process, address, &mut displacement, buffer.as_mut_ptr()) };
        if ok == 0 {
            let base = unsafe { SymGetModuleBase64(process, address) };
            return if base != 0 {
                format!("{:#x} (module+{:#x})", address, address - base)
            } else {
                format!("{:#x}", address)
            };
        }
        let name_len = u32::from_le_bytes(buffer[76..80].try_into().unwrap()) as usize;
        let name = &buffer[84..84 + name_len.min(SYMBOL_NAME_CAPACITY - 1)];
        let name = String::from_utf8_lossy(name);
        format!("{}+{:#x}", name, displacement)
    }

    fn thread_name(thread: Handle) -> String {
        let mut description: *mut u16 = std::ptr::null_mut();
        let hr = unsafe { GetThreadDescription(thread, &mut description) };
        if hr < 0 || description.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        unsafe {
            while *description.add(len) != 0 {
                len += 1;
            }
            let name = String::from_utf16_lossy(std::slice::from_raw_parts(description, len));
            LocalFree(description as *mut c_void);
            name
        }
    }

    fn walk(process: Handle, thread: Handle) -> Vec<u64> {
        let mut context = Context { bytes: [0u8; 1232] };
        context.bytes[0x30..0x34].copy_from_slice(&CONTEXT_FULL.to_le_bytes());
        if unsafe { GetThreadContext(thread, &mut context) } == 0 {
            return Vec::new();
        }
        let rip = u64::from_le_bytes(context.bytes[0xF8..0x100].try_into().unwrap());
        let rsp = u64::from_le_bytes(context.bytes[0x98..0xA0].try_into().unwrap());
        let rbp = u64::from_le_bytes(context.bytes[0xA0..0xA8].try_into().unwrap());
        let mut frame = [0u64; 33];
        let bytes = unsafe { std::slice::from_raw_parts_mut(frame.as_mut_ptr() as *mut u8, 264) };
        bytes[0..8].copy_from_slice(&rip.to_le_bytes());
        bytes[12..16].copy_from_slice(&ADDR_MODE_FLAT.to_le_bytes());
        bytes[32..40].copy_from_slice(&rbp.to_le_bytes());
        bytes[44..48].copy_from_slice(&ADDR_MODE_FLAT.to_le_bytes());
        bytes[48..56].copy_from_slice(&rsp.to_le_bytes());
        bytes[60..64].copy_from_slice(&ADDR_MODE_FLAT.to_le_bytes());
        let mut pcs = Vec::with_capacity(MAX_FRAMES);
        for _ in 0..MAX_FRAMES {
            let ok = unsafe {
                StackWalk64(
                    IMAGE_FILE_MACHINE_AMD64,
                    process,
                    thread,
                    frame.as_mut_ptr(),
                    &mut context,
                    std::ptr::null(),
                    SymFunctionTableAccess64 as *const c_void,
                    SymGetModuleBase64 as *const c_void,
                    std::ptr::null(),
                )
            };
            if ok == 0 {
                break;
            }
            let pc = u64::from_le_bytes(
                unsafe { std::slice::from_raw_parts(frame.as_ptr() as *const u8, 8) }
                    .try_into()
                    .unwrap(),
            );
            if pc == 0 {
                break;
            }
            pcs.push(pc);
        }
        pcs
    }

    pub fn exe_module_base() -> u64 {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetModuleHandleW(name: *const u16) -> Handle;
        }
        unsafe { GetModuleHandleW(std::ptr::null()) as usize as u64 }
    }

    pub fn sample_thread(tid: u32, samples: usize, interval: std::time::Duration) -> Vec<Vec<u64>> {
        let process = unsafe { GetCurrentProcess() };
        if !ensure_symbols(process) {
            return Vec::new();
        }
        let handle = unsafe { OpenThread(THREAD_ACCESS, 0, tid) };
        if handle.is_null() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(samples);
        for _ in 0..samples {
            std::thread::sleep(interval);
            if unsafe { SuspendThread(handle) } == u32::MAX {
                break;
            }
            let pcs = walk(process, handle);
            unsafe { ResumeThread(handle) };
            if !pcs.is_empty() {
                out.push(pcs);
            }
        }
        unsafe { CloseHandle(handle) };
        out
    }

    pub fn dump_all_threads(reason: &str) {
        dump_process_windows();
        let process = unsafe { GetCurrentProcess() };
        if !ensure_symbols(process) {
            log::error!("[stack-dump] SymInitialize failed");
            return;
        }
        let pid = unsafe { GetCurrentProcessId() };
        let me = unsafe { GetCurrentThreadId() };
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot.is_null() || snapshot as isize == -1 {
            log::error!("[stack-dump] thread snapshot failed");
            return;
        }
        let mut entry = ThreadEntry32 {
            size: std::mem::size_of::<ThreadEntry32>() as u32,
            usage: 0,
            thread_id: 0,
            owner_process_id: 0,
            base_pri: 0,
            delta_pri: 0,
            flags: 0,
        };
        let mut threads = Vec::new();
        if unsafe { Thread32First(snapshot, &mut entry) } != 0 {
            loop {
                if entry.owner_process_id == pid && entry.thread_id != me {
                    threads.push(entry.thread_id);
                }
                if unsafe { Thread32Next(snapshot, &mut entry) } == 0 {
                    break;
                }
            }
        }
        unsafe { CloseHandle(snapshot) };
        let mut report = Vec::new();
        for tid in threads {
            let handle = unsafe { OpenThread(THREAD_ACCESS, 0, tid) };
            if handle.is_null() {
                continue;
            }
            let name = thread_name(handle);
            let pcs = if unsafe { SuspendThread(handle) } == u32::MAX {
                Vec::new()
            } else {
                let pcs = walk(process, handle);
                unsafe { ResumeThread(handle) };
                pcs
            };
            unsafe { CloseHandle(handle) };
            report.push((tid, name, pcs));
        }
        log::error!("[stack-dump] reason={} threads={}", reason, report.len());
        for (tid, name, pcs) in report {
            let frames: Vec<String> = pcs.iter().map(|&pc| symbolize(process, pc)).collect();
            log::error!(
                "[stack-dump] tid={} name={:?} frames={} | {}",
                tid,
                name,
                frames.len(),
                frames.join(" <- ")
            );
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::mpsc;
        use std::time::Duration;

        #[link(name = "user32")]
        extern "system" {
            fn CreateWindowExW(
                ex_style: u32,
                class: *const u16,
                title: *const u16,
                style: u32,
                x: i32,
                y: i32,
                width: i32,
                height: i32,
                parent: Handle,
                menu: Handle,
                instance: Handle,
                parameter: Handle,
            ) -> Handle;
            fn DestroyWindow(window: Handle) -> i32;
        }

        #[test]
        fn window_dump_finishes_without_pumping_owned_window_messages() {
            let class: Vec<u16> = "STATIC\0".encode_utf16().collect();
            let title: Vec<u16> = "NeXium watchdog test\0".encode_utf16().collect();
            let window = unsafe {
                CreateWindowExW(
                    0,
                    class.as_ptr(),
                    title.as_ptr(),
                    0,
                    0,
                    0,
                    1,
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            assert!(!window.is_null());
            let (finished_tx, finished_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || {
                dump_process_windows();
                let _ = finished_tx.send(());
            });
            let finished = finished_rx.recv_timeout(Duration::from_secs(2));
            unsafe { DestroyWindow(window) };
            assert!(
                finished.is_ok(),
                "window diagnostic waited for the window owner's message loop"
            );
            worker.join().unwrap();
        }
    }
}

#[cfg(windows)]
pub fn dump_all_threads(reason: &str) {
    imp::dump_all_threads(reason);
}

#[cfg(windows)]
pub fn dump_process_windows() {
    imp::dump_process_windows();
}

#[cfg(windows)]
pub fn sample_thread(tid: u32, samples: usize, interval: std::time::Duration) -> Vec<Vec<u64>> {
    imp::sample_thread(tid, samples, interval)
}

#[cfg(windows)]
pub fn exe_module_base() -> u64 {
    imp::exe_module_base()
}

#[cfg(not(windows))]
pub fn exe_module_base() -> u64 {
    0
}

#[cfg(not(windows))]
pub fn sample_thread(_tid: u32, _samples: usize, _interval: std::time::Duration) -> Vec<Vec<u64>> {
    Vec::new()
}

#[cfg(not(windows))]
pub fn dump_process_windows() {}

#[cfg(not(windows))]
pub fn dump_all_threads(reason: &str) {
    log::error!("[stack-dump] unavailable on this platform ({reason})");
}

use std::ffi::{c_char, c_int, CStr};

use crate::{klog, probe, sys};

pub const DATA_ROOT: &str = "/app0/data";

pub fn share(path: &str, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode));
}

pub fn ensure_shared_dir(path: &str) -> std::io::Result<()> {
    let mut current = std::path::PathBuf::new();
    for component in std::path::Path::new(path).components() {
        current.push(component);
        if !current.exists() {
            std::fs::create_dir(&current)?;
            share(&current.to_string_lossy(), 0o777);
        }
    }
    Ok(())
}

fn args(argc: c_int, argv: *const *const c_char) -> Vec<String> {
    if argv.is_null() {
        return Vec::new();
    }
    (0..argc.max(0) as usize)
        .filter_map(|i| {
            let p = unsafe { *argv.add(i) };
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn main(argc: c_int, argv: *const *const c_char, _envp: *const *const c_char) -> c_int {
    let start_mxcsr = sys::mxcsr();
    sys::set_mxcsr(0x1f80);
    klog::install(log::LevelFilter::Info);
    #[cfg(feature = "title")]
    {
        klog::set_stdout(false);
        unsafe {
            sys::sceSystemServiceHideSplashScreen();
            sys::ps5_klog_capture_stderr(c"[nexium-ps5:stderr]".as_ptr());
        }
    }
    if let Err(e) = ensure_shared_dir(DATA_ROOT) {
        crate::klog!("data root {DATA_ROOT}: {e}");
    }
    let log_path = format!("{DATA_ROOT}/nexium-ps5.log");
    match klog::open_file(&log_path) {
        Ok(()) => share(&log_path, 0o666),
        Err(e) => crate::klog!("log file {log_path}: {e}"),
    }
    let args = args(argc, argv);
    crate::klog!(
        "start build={} mode={} mxcsr_at_entry={start_mxcsr:#06x} args={args:?}",
        env!("CARGO_PKG_VERSION"),
        if cfg!(feature = "title") { "title" } else { "payload" }
    );
    let mode_file = format!("{DATA_ROOT}/probe-mode.txt");
    let mode = std::fs::read_to_string(&mode_file).unwrap_or_default();
    let _ = std::fs::remove_file(&mode_file);
    let mut words = mode.split_whitespace();
    let failures = match words.next() {
        #[cfg(feature = "title")]
        Some("interactive") => {
            let seconds = words.next().and_then(|w| w.parse().ok()).unwrap_or(60);
            match crate::probe_io::interactive(seconds) {
                Ok(detail) => {
                    crate::klog!("PASS interactive-io: {detail}");
                    0
                }
                Err(detail) => {
                    crate::klog!("FAIL interactive-io: {detail}");
                    1
                }
            }
        }
        Some("probe") => probe::run(),
        #[cfg(feature = "title")]
        _ if std::path::Path::new(&format!("{DATA_ROOT}/launch.txt")).exists() => {
            crate::frontend::setup_environment();
            let cfg = crate::frontend::LaunchConfig::load();
            log::set_max_level(cfg.log_level);
            crate::frontend::run(&cfg)
        }
        #[cfg(feature = "title")]
        _ => crate::menu::run(),
        #[cfg(not(feature = "title"))]
        _ => probe::run(),
    };
    crate::klog!("exit failures={failures}");
    failures.min(255) as c_int
}

#[cfg(feature = "title")]
#[no_mangle]
pub extern "C" fn catchReturnFromMain(status: c_int) {
    crate::klog!("catchReturnFromMain status={status}; asking the shell to close the title");
    unsafe {
        sys::sceSystemServiceLoadExec(c"exit".as_ptr(), std::ptr::null());
        loop {
            sys::sceKernelUsleep(100_000);
        }
    }
}

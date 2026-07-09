fn main() {
    #[cfg(target_os = "windows")]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../branding/logo.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=icon embed failed: {}", e);
        }
    }
    println!("cargo:rerun-if-changed=../branding/logo.ico");

    let git = |args: &[&str]| -> Option<String> {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let hash = git(&["rev-parse", "--short=10", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).map_or(false, |s| !s.is_empty());
    let hash = if dirty { format!("{}-dirty", hash) } else { hash };
    println!("cargo:rustc-env=NEXIUM_GIT_HASH={}", hash);

    for p in ["../.git/HEAD", "../.git/index"] {
        if std::path::Path::new(p).exists() {
            println!("cargo:rerun-if-changed={}", p);
        }
    }
}
